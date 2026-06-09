"""
unet_segmentor.py
=================
U-Net leve para segmentação binária de fissuras.

Substitui o pipeline Canny-Frangi clássico por uma rede neural que aprende
a separar fissuras de textura, oferecendo robustez superior em superfícies
rugosas, concreto colorido e variações de iluminação.

Arquitetura
-----------
  Encoder: 4 blocos DoubleConv com MaxPool  (32 → 64 → 128 → 256 → 512 ch)
  Bottleneck: DoubleConv 512 ch
  Decoder: 4 blocos de UpConv + skip connection + DoubleConv
  Saída: Conv 1×1 → sigmoid → máscara binária [0, 1]

Interface compatível com CrackSegmentor
---------------------------------------
  segmentor = UNetSegmentor(weights_path="src/model_weights/unet_crack.pth")
  binary_mask = segmentor.segment(enhanced_img)   # igual ao CrackSegmentor

Treinamento
-----------
  Use train_segmentor.py com as máscaras geradas por main.py --export_masks
"""

import os
import copy
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image


# ─────────────────────────────────────────────────────────────────────────────
# Bloco base: dois Conv2d + BN + ReLU
# ─────────────────────────────────────────────────────────────────────────────

class _DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


# ─────────────────────────────────────────────────────────────────────────────
# U-Net
# ─────────────────────────────────────────────────────────────────────────────

class _UNet(nn.Module):
    """
    U-Net leve com ~7 M parâmetros — roda em CPU em tempo razoável.
    features=[32,64,128,256] mantém inferência rápida sem sacrificar precisão.
    """

    def __init__(self, in_channels: int = 1, features: list = None):
        super().__init__()
        if features is None:
            features = [32, 64, 128, 256]

        # Encoder
        self.encoders = nn.ModuleList()
        self.pools    = nn.ModuleList()
        prev = in_channels
        for f in features:
            self.encoders.append(_DoubleConv(prev, f))
            self.pools.append(nn.MaxPool2d(kernel_size=2, stride=2))
            prev = f

        # Bottleneck
        self.bottleneck = _DoubleConv(features[-1], features[-1] * 2)

        # Decoder
        self.upconvs  = nn.ModuleList()
        self.decoders = nn.ModuleList()
        rev = list(reversed(features))
        prev = features[-1] * 2
        for f in rev:
            self.upconvs.append(
                nn.ConvTranspose2d(prev, f, kernel_size=2, stride=2)
            )
            self.decoders.append(_DoubleConv(f * 2, f))
            prev = f

        # Saída
        self.final_conv = nn.Conv2d(features[0], 1, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        # Encoder pass
        for enc, pool in zip(self.encoders, self.pools):
            x = enc(x)
            skip_connections.append(x)
            x = pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        # Decoder pass
        for i, (up, dec) in enumerate(zip(self.upconvs, self.decoders)):
            x = up(x)
            skip = skip_connections[i]

            # Garante dimensões iguais (necessário quando tamanho da entrada não é múltiplo de 16)
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)

            x = torch.cat([skip, x], dim=1)
            x = dec(x)

        return torch.sigmoid(self.final_conv(x))


# ─────────────────────────────────────────────────────────────────────────────
# Funções de perda (BCE + Dice) — balanceiam o desbalanceamento fissura/fundo
# ─────────────────────────────────────────────────────────────────────────────

def dice_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    pred_flat   = pred.view(-1)
    target_flat = target.view(-1)
    intersection = (pred_flat * target_flat).sum()
    return 1.0 - (2.0 * intersection + eps) / (pred_flat.sum() + target_flat.sum() + eps)


def bce_dice_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    bce  = F.binary_cross_entropy(pred, target)
    dice = dice_loss(pred, target)
    return bce + dice


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

from torch.utils.data import Dataset


class CrackSegDataset(Dataset):
    """
    Dataset para treino da U-Net.

    Estrutura esperada:
        images_dir/   *.jpg / *.png  (imagens originais)
        masks_dir/    *.png          (máscaras binárias — mesmo nome)

    As máscaras devem ser imagens de 1 canal com valores 0 (fundo) ou 255 (fissura).
    Geradas automaticamente por: python main.py --export_masks
    """

    IMG_SIZE = 256  # tamanho interno de processamento

    _AUGMENT = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
    ])

    _TO_TENSOR = transforms.ToTensor()

    def __init__(self, images_dir: str, masks_dir: str, augment: bool = True):
        self.images_dir = images_dir
        self.masks_dir  = masks_dir
        self.augment    = augment

        exts = {".jpg", ".jpeg", ".png", ".bmp"}
        self.image_files = sorted(
            p for p in os.listdir(images_dir)
            if os.path.splitext(p)[1].lower() in exts
        )

        # Mantém apenas amostras que possuem máscara correspondente
        self.image_files = [
            p for p in self.image_files
            if os.path.exists(os.path.join(masks_dir, os.path.splitext(p)[0] + ".png"))
        ]

        if not self.image_files:
            raise RuntimeError(
                f"Nenhuma amostra com par imagem+máscara encontrada.\n"
                f"  Imagens: {images_dir}\n  Máscaras: {masks_dir}"
            )

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        name  = self.image_files[idx]
        stem  = os.path.splitext(name)[0]

        img_path  = os.path.join(self.images_dir, name)
        mask_path = os.path.join(self.masks_dir, stem + ".png")

        # Carrega e redimensiona
        img  = Image.open(img_path).convert("L").resize(
            (self.IMG_SIZE, self.IMG_SIZE), Image.BILINEAR
        )
        mask = Image.open(mask_path).convert("L").resize(
            (self.IMG_SIZE, self.IMG_SIZE), Image.NEAREST
        )

        # Augmentation sincronizada (mesma semente para img e mask)
        if self.augment:
            seed = torch.randint(0, 2**32, (1,)).item()
            torch.manual_seed(seed)
            img  = self._AUGMENT(img)
            torch.manual_seed(seed)
            mask = self._AUGMENT(mask)

        img_t  = self._TO_TENSOR(img)                              # [1, H, W] ∈ [0,1]
        mask_t = (self._TO_TENSOR(mask) > 0.5).float()            # [1, H, W] ∈ {0,1}

        return img_t, mask_t


# ─────────────────────────────────────────────────────────────────────────────
# Wrapper principal
# ─────────────────────────────────────────────────────────────────────────────

class UNetSegmentor:
    """
    Segmentador de fissuras baseado em U-Net.

    Uso (inferência):
        seg = UNetSegmentor(weights_path="src/model_weights/unet_crack.pth")
        binary_mask = seg.segment(enhanced_gray_img)   # igual ao CrackSegmentor

    Uso (treino):
        seg = UNetSegmentor()
        seg.train(
            images_dir="dataset_rotulado/images/CRITICA",  # ou todas as classes juntas
            masks_dir="dataset_rotulado/masks/CRITICA",
            epochs=30,
        )
        seg.save("src/model_weights/unet_crack.pth")
    """

    # Limiar de probabilidade para binarizar a saída da rede
    THRESHOLD = 0.40

    def __init__(self, weights_path: str = None, device: str = None,
                 threshold: float = None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.threshold = threshold if threshold is not None else self.THRESHOLD
        self.model = _UNet(in_channels=1, features=[32, 64, 128, 256])
        self.model.to(self.device)

        if weights_path is not None:
            self.load(weights_path)

        self.model.eval()

    # ── Inferência ────────────────────────────────────────────────────────────

    def segment(self, enhanced_img: np.ndarray) -> np.ndarray:
        """
        Segmenta a fissura.  Interface idêntica ao CrackSegmentor.segment().

        Parâmetros
        ----------
        enhanced_img : np.ndarray
            Imagem grayscale pré-processada (uint8, HxW).

        Retorna
        -------
        np.ndarray
            Máscara binária uint8 (0 ou 255), mesma resolução da entrada.
        """
        orig_h, orig_w = enhanced_img.shape[:2]

        # Redimensiona para 256×256 (tamanho interno)
        img_256 = cv2.resize(enhanced_img, (256, 256), interpolation=cv2.INTER_AREA)

        # Normaliza e cria tensor
        img_pil = Image.fromarray(img_256)
        tensor  = transforms.ToTensor()(img_pil).unsqueeze(0).to(self.device)

        with torch.no_grad():
            prob_map = self.model(tensor)            # [1, 1, 256, 256] ∈ [0,1]

        prob_np = prob_map.squeeze().cpu().numpy()   # [256, 256]

        # Binariza com threshold
        binary_256 = (prob_np >= self.threshold).astype(np.uint8) * 255

        # Volta para resolução original
        if (orig_h, orig_w) != (256, 256):
            binary_mask = cv2.resize(binary_256, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        else:
            binary_mask = binary_256

        # Pós-processamento leve: remove pontos isolados minúsculos
        binary_mask = self._clean_mask(binary_mask)

        return binary_mask

    @staticmethod
    def _clean_mask(mask: np.ndarray, min_area: int = 30) -> np.ndarray:
        """Remove componentes menores que min_area pixels."""
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if num_labels <= 1:
            return mask
        lut = np.zeros(num_labels, dtype=np.uint8)
        lut[stats[:, cv2.CC_STAT_AREA] >= min_area] = 255
        lut[0] = 0
        return lut[labels]

    # ── Treinamento ───────────────────────────────────────────────────────────

    def train(
        self,
        images_dir: str,
        masks_dir: str,
        epochs: int = 30,
        lr: float = 1e-3,
        batch_size: int = 8,
        val_split: float = 0.2,
        save_best_path: str = None,
        patience: int = 8,
    ):
        """
        Treina a U-Net com pares (imagem, máscara).

        Parâmetros
        ----------
        images_dir : str
            Pasta com imagens originais (qualquer classe — pode misturar todas).
        masks_dir : str
            Pasta com máscaras binárias correspondentes (geradas por main.py --export_masks).
        epochs : int
            Número máximo de épocas.
        lr : float
            Learning rate do AdamW.
        batch_size : int
            Tamanho do batch.
        val_split : float
            Fração dos dados para validação.
        save_best_path : str, opcional
            Caminho para salvar o melhor modelo (por IoU de validação).
        patience : int
            Épocas sem melhoria antes de early stopping.
        """
        from torch.utils.data import DataLoader, random_split
        from tqdm import tqdm

        print(f"[UNetSegmentor] Treinando em: {images_dir}")
        print(f"[UNetSegmentor] Máscaras em : {masks_dir}")
        print(f"[UNetSegmentor] Device      : {self.device}")

        full_ds = CrackSegDataset(images_dir, masks_dir, augment=True)
        print(f"[UNetSegmentor] Amostras     : {len(full_ds)}")

        val_size   = max(1, int(len(full_ds) * val_split))
        train_size = len(full_ds) - val_size
        train_ds, val_ds = random_split(full_ds, [train_size, val_size])

        # Validação sem augmentation
        val_ds.dataset = copy.copy(full_ds)
        val_ds.dataset.augment = False

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
        val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        best_iou   = 0.0
        best_state = None
        no_improve = 0

        for epoch in range(1, epochs + 1):
            # ── Treino ──
            self.model.train()
            train_loss = 0.0

            pbar = tqdm(train_loader, desc=f"Epoch {epoch:02d}/{epochs} [Train]", leave=False)
            for imgs, masks in pbar:
                imgs, masks = imgs.to(self.device), masks.to(self.device)
                optimizer.zero_grad()
                preds = self.model(imgs)
                loss  = bce_dice_loss(preds, masks)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                pbar.set_postfix(loss=f"{loss.item():.4f}")

            scheduler.step()

            # ── Validação ──
            self.model.eval()
            iou_scores = []
            val_loss   = 0.0

            with torch.no_grad():
                for imgs, masks in val_loader:
                    imgs, masks = imgs.to(self.device), masks.to(self.device)
                    preds = self.model(imgs)
                    val_loss += bce_dice_loss(preds, masks).item()

                    # IoU (Intersection over Union)
                    pred_bin = (preds >= self.threshold).float()
                    inter = (pred_bin * masks).sum(dim=(1, 2, 3))
                    union = (pred_bin + masks).clamp(0, 1).sum(dim=(1, 2, 3))
                    iou   = (inter / (union + 1e-6)).mean().item()
                    iou_scores.append(iou)

            mean_iou    = float(np.mean(iou_scores))
            avg_tloss   = train_loss / len(train_loader)
            avg_vloss   = val_loss   / len(val_loader)
            current_lr  = optimizer.param_groups[0]["lr"]

            print(
                f"Epoch [{epoch:02d}/{epochs}] "
                f"Train Loss: {avg_tloss:.4f}  |  "
                f"Val Loss: {avg_vloss:.4f}  |  "
                f"IoU: {mean_iou:.4f}  |  LR: {current_lr:.6f}"
            )

            if mean_iou > best_iou:
                best_iou   = mean_iou
                best_state = copy.deepcopy(self.model.state_dict())
                no_improve = 0
                if save_best_path:
                    self.save(save_best_path)
                    print(f"  ✓ Melhor modelo salvo (IoU={mean_iou:.4f}): {save_best_path}")
            else:
                no_improve += 1

            if no_improve >= patience:
                print(f"  ⚠ Early stopping (patience={patience}) após {epoch} épocas.")
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)

        self.model.eval()
        print(f"[UNetSegmentor] Treinamento concluído. Melhor IoU: {best_iou:.4f}")

    # ── Serialização ──────────────────────────────────────────────────────────

    def save(self, path: str):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save(self.model.state_dict(), path)

    def load(self, path: str):
        state = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()
        print(f"[UNetSegmentor] Pesos carregados de: {path}")

    # ── Utilitários ───────────────────────────────────────────────────────────

    @property
    def is_on_gpu(self) -> bool:
        return self.device.type == "cuda"

    def __repr__(self):
        return (
            f"UNetSegmentor("
            f"device={self.device}, "
            f"threshold={self.threshold})"
        )
