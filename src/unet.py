"""
unet.py
=======
U-Net e Attention U-Net para segmentação binária de fissuras em concreto.

Arquiteturas disponíveis:
  UNet            — U-Net padrão (Ronneberger et al., 2015)
  AttentionUNet   — U-Net com Attention Gates no decoder (Oktay et al., 2018)
                    Os gates suprimem regiões de fundo antes de cada skip
                    connection, forçando o decoder a focar na fissura.

Entrada esperada:
  - Tensor [B, 1, H, W] ou [B, 3, H, W] — imagem normalizada [0, 1]
  - Resolução recomendada: 256×256 (múltiplos de 16 também funcionam)

Saída:
  - 1 canal (logit) → sigmoid → máscara binária [0, 1]

Referências:
  Ronneberger et al., 2015 — "U-Net: Convolutional Networks for Biomedical
    Image Segmentation"
  Oktay et al., 2018 — "Attention U-Net: Learning Where to Look for the
    Pancreas"
"""

import os
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image


# --------------------------------------------------------------------------- #
# Blocos fundamentais
# --------------------------------------------------------------------------- #

class _DoubleConv(nn.Module):
    """Conv3x3 → BN → ReLU → Conv3x3 → BN → ReLU"""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class _Down(nn.Module):
    """MaxPool2×2 → DoubleConv"""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.pool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            _DoubleConv(in_ch, out_ch),
        )

    def forward(self, x):
        return self.pool_conv(x)


class _Up(nn.Module):
    """UpConv bilinear 2× → concat skip → DoubleConv"""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.up   = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = _DoubleConv(in_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        # Ajusta dimensões se houver diferença por arredondamento
        dy = skip.size(2) - x.size(2)
        dx = skip.size(3) - x.size(3)
        x  = F.pad(x, [dx // 2, dx - dx // 2, dy // 2, dy - dy // 2])
        x  = torch.cat([skip, x], dim=1)
        return self.conv(x)


# --------------------------------------------------------------------------- #
# Blocos de Atenção (Attention U-Net)
# --------------------------------------------------------------------------- #

class _AttentionGate(nn.Module):
    """
    Attention Gate — Oktay et al., 2018.

    Recebe o sinal de gating `g` (vindo do decoder, resolução mais baixa)
    e o skip connection `x` (vindo do encoder, mesma resolução do alvo).
    Produz coeficientes de atenção α ∈ [0, 1] por pixel que amplificam
    regiões de fissura e suprimem fundo antes da concatenação no decoder.

    Parâmetros
    ----------
    F_g  : canais do sinal de gating (decoder)
    F_l  : canais do skip connection (encoder)
    F_int: canais intermediários do gate (tipicamente F_l // 2)
    """

    def __init__(self, F_g: int, F_l: int, F_int: int):
        super().__init__()
        # Projeta gating signal para F_int (sem bias — BN absorve)
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, bias=False),
            nn.BatchNorm2d(F_int),
        )
        # Projeta skip connection para F_int
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, bias=False),
            nn.BatchNorm2d(F_int),
        )
        # Colapsa para 1 canal → mapa de atenção α
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        g : sinal de gating  [B, F_g, H, W]  (pode ter resolução menor que x)
        x : skip connection  [B, F_l, H, W]
        """
        # Upsample g para a resolução de x, se necessário
        if g.shape[2:] != x.shape[2:]:
            g = F.interpolate(g, size=x.shape[2:], mode="bilinear", align_corners=True)

        att = self.relu(self.W_g(g) + self.W_x(x))  # soma no espaço F_int
        att = self.psi(att)                           # α ∈ [0, 1], shape [B,1,H,W]
        return x * att                                # rescale do skip connection


class _AttentionUp(nn.Module):
    """UpConv bilinear 2× → AttentionGate no skip → concat → DoubleConv"""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up   = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.attn = _AttentionGate(F_g=in_ch, F_l=skip_ch, F_int=skip_ch // 2)
        self.conv = _DoubleConv(in_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # Ajusta dimensões por arredondamento (igual ao _Up original)
        dy = skip.size(2) - x.size(2)
        dx = skip.size(3) - x.size(3)
        x  = F.pad(x, [dx // 2, dx - dx // 2, dy // 2, dy - dy // 2])
        # Aplica atenção no skip antes de concatenar
        skip = self.attn(g=x, x=skip)
        x    = torch.cat([skip, x], dim=1)
        return self.conv(x)


# --------------------------------------------------------------------------- #
# Arquitetura U-Net
# --------------------------------------------------------------------------- #

class UNet(nn.Module):
    """
    U-Net leve para segmentação binária de fissuras.

    Parâmetros
    ----------
    in_channels : int
        Canais de entrada (1 = grayscale, 3 = RGB).
    base_filters : int
        Número de filtros no primeiro nível (default 32).
        Use 64 para maior capacidade (mais memória GPU).
    """

    def __init__(self, in_channels: int = 1, base_filters: int = 32):
        super().__init__()
        f = base_filters           # 32, 64, 128, 256, 512

        # Encoder
        self.enc1 = _DoubleConv(in_channels, f)
        self.enc2 = _Down(f,     f * 2)
        self.enc3 = _Down(f * 2, f * 4)
        self.enc4 = _Down(f * 4, f * 8)

        # Bottleneck
        self.bottleneck = _Down(f * 8, f * 16)

        # Decoder
        self.dec4 = _Up(f * 16 + f * 8, f * 8)
        self.dec3 = _Up(f * 8  + f * 4, f * 4)
        self.dec2 = _Up(f * 4  + f * 2, f * 2)
        self.dec1 = _Up(f * 2  + f,     f)

        # Cabeça de saída: 1 logit por pixel
        self.head = nn.Conv2d(f, 1, kernel_size=1)

    def forward(self, x):
        s1 = self.enc1(x)
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)
        s4 = self.enc4(s3)

        bn = self.bottleneck(s4)

        x  = self.dec4(bn, s4)
        x  = self.dec3(x,  s3)
        x  = self.dec2(x,  s2)
        x  = self.dec1(x,  s1)

        return self.head(x)   # logits — aplicar sigmoid externamente


# --------------------------------------------------------------------------- #
# Arquitetura Attention U-Net
# --------------------------------------------------------------------------- #

class AttentionUNet(nn.Module):
    """
    Attention U-Net para segmentação cirúrgica de fissuras.

    Idêntico ao U-Net padrão no encoder e bottleneck. A diferença está no
    decoder: cada nível usa _AttentionUp em vez de _Up, inserindo um
    Attention Gate que aprende a suprimir pixels de fundo (reboco, sombras,
    juntas) antes de concatenar o skip connection do encoder.

    Resultado prático: a máscara final contém quase exclusivamente pixels
    que pertencem à fissura, reduzindo falsos positivos sem precisar aumentar
    min_area_user no pós-processamento.

    Parâmetros
    ----------
    in_channels  : int  — canais de entrada (1 = grayscale, 3 = RGB)
    base_filters : int  — filtros no primeiro nível (default 32)

    Referência: Oktay et al., 2018 — "Attention U-Net: Learning Where to
    Look for the Pancreas"
    """

    def __init__(self, in_channels: int = 1, base_filters: int = 32):
        super().__init__()
        f = base_filters  # 32, 64, 128, 256, 512

        # Encoder (igual ao UNet padrão)
        self.enc1 = _DoubleConv(in_channels, f)
        self.enc2 = _Down(f,     f * 2)
        self.enc3 = _Down(f * 2, f * 4)
        self.enc4 = _Down(f * 4, f * 8)

        # Bottleneck
        self.bottleneck = _Down(f * 8, f * 16)

        # Decoder com Attention Gates
        # _AttentionUp(in_ch=canais_do_decoder, skip_ch=canais_do_encoder, out_ch)
        self.dec4 = _AttentionUp(f * 16, f * 8,  f * 8)
        self.dec3 = _AttentionUp(f * 8,  f * 4,  f * 4)
        self.dec2 = _AttentionUp(f * 4,  f * 2,  f * 2)
        self.dec1 = _AttentionUp(f * 2,  f,       f)

        # Cabeça de saída: 1 logit por pixel
        self.head = nn.Conv2d(f, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s1 = self.enc1(x)
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)
        s4 = self.enc4(s3)

        bn = self.bottleneck(s4)

        x = self.dec4(bn, s4)
        x = self.dec3(x,  s3)
        x = self.dec2(x,  s2)
        x = self.dec1(x,  s1)

        return self.head(x)   # logits — aplicar sigmoid externamente


# --------------------------------------------------------------------------- #
# Wrapper de alto nível
# --------------------------------------------------------------------------- #

# Transformação padrão para inferência
_UNET_INFER_TRANSFORM = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),           # [0, 255] → [0.0, 1.0]
])

# Normalização adicional (média/std de imagens de concreto em escala de cinza)
_UNET_NORMALIZE = transforms.Normalize(mean=[0.5], std=[0.5])


class UNetSegmentor:
    """
    Wrapper de alto nível em torno do U-Net / Attention U-Net para segmentação
    de fissuras.

    Uso básico (inferência):
        seg = UNetSegmentor(weights_path="src/model_weights/unet_crack.pth")
        binary_mask = seg.segment(image_bgr)   # np.ndarray uint8 (0/255)

    Uso básico (treino com Attention U-Net — padrão):
        seg = UNetSegmentor(use_attention=True)
        seg.train(images_dir="masks_dataset/images",
                  masks_dir="masks_dataset/masks",
                  epochs=5)
        seg.save("src/model_weights/unet_crack.pth")

    Parâmetro use_attention
    -----------------------
    True  (padrão) — usa AttentionUNet: gates suprimem fundo antes de cada
                     skip connection, focando a máscara na fissura.
    False           — usa UNet padrão (compatível com pesos antigos).
    O flag é salvo no checkpoint e restaurado automaticamente no load().
    """

    # Limiar de probabilidade para binarização da máscara
    THRESHOLD = 0.45

    def __init__(self, weights_path: str = None, device: str = None,
                 in_channels: int = 1, base_filters: int = 32,
                 threshold: float = None, use_attention: bool = True):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.threshold    = threshold or self.THRESHOLD
        self.in_channels  = in_channels
        self.use_attention = use_attention

        self.model = self._build_model(in_channels, base_filters, use_attention)
        self.model.to(self.device)

        if weights_path is not None:
            self.load(weights_path)

        self.model.eval()

    @staticmethod
    def _build_model(in_channels: int, base_filters: int,
                     use_attention: bool) -> nn.Module:
        if use_attention:
            return AttentionUNet(in_channels=in_channels, base_filters=base_filters)
        return UNet(in_channels=in_channels, base_filters=base_filters)

    # ----------------------------------------------------------------------- #
    # Inferência
    # ----------------------------------------------------------------------- #

    def segment(self, image_bgr: np.ndarray) -> np.ndarray:
        """
        Segmenta fissuras em uma imagem BGR (OpenCV).

        Retorna
        -------
        np.ndarray uint8 (0/255)
            Máscara binária com a mesma resolução da imagem de entrada.
        """
        import cv2
        orig_h, orig_w = image_bgr.shape[:2]

        # Pré-processamento: BGR → grayscale ou RGB → PIL
        if self.in_channels == 1:
            gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
            pil  = Image.fromarray(gray).convert("L")
        else:
            rgb  = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            pil  = Image.fromarray(rgb).convert("RGB")

        # Transforma e normaliza
        tensor = _UNET_INFER_TRANSFORM(pil)          # [C, 256, 256]
        if self.in_channels == 1:
            tensor = _UNET_NORMALIZE(tensor)
        tensor = tensor.unsqueeze(0).to(self.device)  # [1, C, 256, 256]

        with torch.no_grad():
            logits = self.model(tensor)               # [1, 1, 256, 256]
            prob   = torch.sigmoid(logits).squeeze().cpu().numpy()  # [256, 256]

        # Binariza
        binary_256 = (prob >= self.threshold).astype(np.uint8) * 255

        # Redimensiona de volta para a resolução original
        mask = cv2.resize(binary_256, (orig_w, orig_h),
                          interpolation=cv2.INTER_NEAREST)
        return mask

    # ----------------------------------------------------------------------- #
    # Treinamento
    # ----------------------------------------------------------------------- #

    def train(
        self,
        images_dir: str,
        masks_dir: str,
        epochs: int = 5,
        lr: float = 1e-3,
        batch_size: int = 8,
        val_split: float = 0.2,
        save_best_path: str = None,
        patience: int = 8,
    ):
        """
        Treina o U-Net em pares (imagem, máscara binária).

        Estrutura esperada:
            images_dir/   *.jpg / *.png  (imagens originais)
            masks_dir/    *.png          (máscaras binárias 0/255 com MESMO nome)

        As máscaras binárias podem ser geradas pelo pipeline principal:
            python main.py --input_dir ./SDNET2018 --output_dir ./dataset \\
                           --scale_factor 0.234 --export_masks

        Depois, concatene todas as subpastas de masks em masks_dir:
            dataset/masks/CRITICA/     *.png
            dataset/masks/SEMI_CRITICA/*.png
            ...
        """
        from torch.utils.data import DataLoader, Dataset, random_split
        import glob, tqdm as tqdm_mod

        print(f"[UNetSegmentor] Treinando em: {images_dir}")
        print(f"[UNetSegmentor] Máscaras em:  {masks_dir}")
        print(f"[UNetSegmentor] Device: {self.device}")

        # Dataset interno
        class _CrackSegDataset(Dataset):
            EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

            def __init__(self, img_dir, msk_dir, in_channels):
                self.pairs       = []
                self.in_channels = in_channels

                for ext in _CrackSegDataset.EXTS:
                    for img_p in glob.glob(os.path.join(img_dir, "**", f"*{ext}"), recursive=True):
                        stem   = os.path.splitext(os.path.basename(img_p))[0]
                        msk_p  = os.path.join(msk_dir, stem + ".png")
                        # Tenta subpastas também
                        if not os.path.exists(msk_p):
                            hits = glob.glob(os.path.join(msk_dir, "**", stem + ".png"), recursive=True)
                            if hits:
                                msk_p = hits[0]
                        if os.path.exists(msk_p):
                            self.pairs.append((img_p, msk_p))

                if len(self.pairs) == 0:
                    raise RuntimeError(
                        f"Nenhum par (imagem, máscara) encontrado.\n"
                        f"  images_dir: {img_dir}\n"
                        f"  masks_dir:  {msk_dir}\n"
                        "Verifique se as máscaras têm o mesmo nome das imagens (sem extensão)."
                    )
                print(f"[UNetSegmentor] Pares encontrados: {len(self.pairs)}")

            def __len__(self):
                return len(self.pairs)

            def __getitem__(self, idx):
                img_p, msk_p = self.pairs[idx]
                import cv2 as _cv2

                img_bgr = _cv2.imread(img_p)
                if self.in_channels == 1:
                    gray = _cv2.cvtColor(img_bgr, _cv2.COLOR_BGR2GRAY)
                    pil  = Image.fromarray(gray).convert("L")
                else:
                    rgb  = _cv2.cvtColor(img_bgr, _cv2.COLOR_BGR2RGB)
                    pil  = Image.fromarray(rgb).convert("RGB")

                # Augmentation leve: flip horizontal aleatório
                if np.random.rand() > 0.5:
                    pil = pil.transpose(Image.FLIP_LEFT_RIGHT)

                img_t = _UNET_INFER_TRANSFORM(pil)
                if self.in_channels == 1:
                    img_t = _UNET_NORMALIZE(img_t)

                # Máscara
                msk_pil = Image.open(msk_p).convert("L").resize((256, 256), Image.NEAREST)
                msk_np  = np.array(msk_pil, dtype=np.float32) / 255.0  # [0, 1]
                msk_t   = torch.from_numpy(msk_np).unsqueeze(0)         # [1, 256, 256]

                return img_t, msk_t

        full_ds  = _CrackSegDataset(images_dir, masks_dir, self.in_channels)
        val_size = max(1, int(len(full_ds) * val_split))
        trn_size = len(full_ds) - val_size
        trn_ds, val_ds = random_split(full_ds, [trn_size, val_size])

        trn_loader = DataLoader(trn_ds, batch_size=batch_size, shuffle=True,
                                num_workers=0, pin_memory=(self.device.type == "cuda"))
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                                num_workers=0, pin_memory=(self.device.type == "cuda"))

        # Função de perda: BCE + Dice (melhor convergência em dados desbalanceados)
        def dice_bce_loss(pred, target, smooth=1.0):
            bce  = F.binary_cross_entropy_with_logits(pred, target)
            pred_s = torch.sigmoid(pred)
            inter  = (pred_s * target).sum(dim=(2, 3))
            dice   = 1 - (2 * inter + smooth) / (pred_s.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) + smooth)
            return bce + dice.mean()

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        best_val_loss      = float("inf")
        best_state         = None
        epochs_no_improve  = 0

        for epoch in range(1, epochs + 1):
            # --- Treino ---
            self.model.train()
            trn_loss = 0.0
            for imgs, masks in trn_loader:
                imgs, masks = imgs.to(self.device), masks.to(self.device)
                optimizer.zero_grad()
                loss = dice_bce_loss(self.model(imgs), masks)
                loss.backward()
                optimizer.step()
                trn_loss += loss.item() * imgs.size(0)

            scheduler.step()

            # --- Validação ---
            self.model.eval()
            val_loss = 0.0
            iou_sum  = 0.0
            n_val    = 0

            with torch.no_grad():
                for imgs, masks in val_loader:
                    imgs, masks = imgs.to(self.device), masks.to(self.device)
                    logits = self.model(imgs)
                    val_loss += dice_bce_loss(logits, masks).item() * imgs.size(0)

                    # IoU (Jaccard)
                    pred_bin = (torch.sigmoid(logits) >= self.threshold).float()
                    inter    = (pred_bin * masks).sum().item()
                    union    = (pred_bin + masks).clamp(0, 1).sum().item()
                    if union > 0:
                        iou_sum += inter / union
                    n_val += 1

            avg_trn = trn_loss / trn_size
            avg_val = val_loss / val_size
            iou     = iou_sum / max(n_val, 1)
            lr_cur  = optimizer.param_groups[0]["lr"]

            print(
                f"Epoch [{epoch:03d}/{epochs}] "
                f"Train Loss: {avg_trn:.4f}  |  "
                f"Val Loss: {avg_val:.4f}  |  "
                f"IoU: {iou:.4f}  |  LR: {lr_cur:.6f}"
            )

            if save_best_path and avg_val < best_val_loss:
                best_val_loss = avg_val
                best_state    = copy.deepcopy(self.model.state_dict())
                self.save(save_best_path)
                epochs_no_improve = 0
                print(f"  ✓ Melhor modelo salvo (val_loss={avg_val:.4f}) em: {save_best_path}")
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= patience:
                print(f"  ⚠ Early stopping após {epoch} épocas (patience={patience})")
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)
            print(f"[UNetSegmentor] Melhor modelo restaurado (val_loss={best_val_loss:.4f})")

        self.model.eval()
        print("[UNetSegmentor] Treinamento concluído.")

    # ----------------------------------------------------------------------- #
    # Serialização
    # ----------------------------------------------------------------------- #

    def save(self, path: str):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save({
            "model_state":   self.model.state_dict(),
            "in_channels":   self.in_channels,
            "threshold":     self.threshold,
            "use_attention": self.use_attention,
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        if isinstance(ckpt, dict) and "model_state" in ckpt:
            self.threshold     = ckpt.get("threshold",     self.threshold)
            self.in_channels   = ckpt.get("in_channels",   self.in_channels)
            saved_attention    = ckpt.get("use_attention",  None)

            # Se o checkpoint veio de um modelo sem attention (legado),
            # reconstrói com UNet padrão para compatibilidade de pesos.
            if saved_attention is not None and saved_attention != self.use_attention:
                print(
                    f"[UNetSegmentor] Checkpoint usa use_attention={saved_attention}. "
                    f"Reconstruindo modelo para corresponder."
                )
                self.use_attention = saved_attention
                f = 32  # base_filters padrão
                self.model = self._build_model(self.in_channels, f, self.use_attention)
                self.model.to(self.device)

            self.model.load_state_dict(ckpt["model_state"])
        else:
            # Compatibilidade com salvamento direto de state_dict (legado)
            self.model.load_state_dict(ckpt)

        self.model.to(self.device)
        self.model.eval()
        arch = "AttentionUNet" if self.use_attention else "UNet"
        print(f"[UNetSegmentor] {arch} — pesos carregados de: {path}")

    # ----------------------------------------------------------------------- #
    # Utilitários
    # ----------------------------------------------------------------------- #

    @property
    def is_on_gpu(self) -> bool:
        return self.device.type == "cuda"

    def __repr__(self):
        arch = "AttentionUNet" if self.use_attention else "UNet"
        return (
            f"UNetSegmentor(arch={arch}, "
            f"in_channels={self.in_channels}, "
            f"device={self.device}, "
            f"threshold={self.threshold})"
        )
