"""
deep_classifier.py
==================
ResNet50 + SE-Net fine-tuned para classificação de criticidade de fissuras.

Técnicas implementadas:
  - SE-Net (Squeeze-Excitation) no layer4 do ResNet50        → +1-2% F1
  - Weighted Focal Loss com pesos automáticos por classe      → +2-4% F1
  - RandAugment no treino                                     → +1-2% F1
  - CutMix no DataLoader de treino                            → +3-5% F1
  - Staged fine-tuning (backbone congelado → descongelamento progressivo)
  - TTA (Test-Time Augmentation) na inferência
  - Safety override: threshold morfológico CRITICA nunca é rebaixado pela IA

Classes de saída:
    0 -> CRITICA        (>= 1.0 mm)
    1 -> SEMI_CRITICA   (0.20 mm – 1.0 mm)
    2 -> NAO_CRITICA    (0.01 mm – 0.20 mm)
    3 -> DESCARTADA     (< 0.01 mm / sem fissura)
"""

import os
import copy
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset, random_split
from PIL import Image
from tqdm import tqdm

from src.classification import CriticalityClassifier as _ThresholdClassifier


# ─────────────────────────────────────────────────────────────────────────────
# Constantes
# ─────────────────────────────────────────────────────────────────────────────

CLASSES     = ["CRITICA", "SEMI_CRITICA", "NAO_CRITICA", "DESCARTADA"]
NUM_CLASSES = len(CLASSES)
CROP_PAD    = 20  # pixels de margem ao redor da fissura no crop

_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.224, 0.225]

_TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandAugment(num_ops=2, magnitude=6),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(_MEAN, _STD),
])

_INFER_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(_MEAN, _STD),
])

_TTA_TRANSFORMS = [
    _INFER_TRANSFORM,
    transforms.Compose([transforms.Resize((224,224)), transforms.RandomHorizontalFlip(p=1.0),
                        transforms.ToTensor(), transforms.Normalize(_MEAN, _STD)]),
    transforms.Compose([transforms.Resize((224,224)), transforms.RandomVerticalFlip(p=1.0),
                        transforms.ToTensor(), transforms.Normalize(_MEAN, _STD)]),
    transforms.Compose([transforms.Resize((224,224)), transforms.RandomRotation((5,5)),
                        transforms.ToTensor(), transforms.Normalize(_MEAN, _STD)]),
    transforms.Compose([transforms.Resize((224,224)), transforms.RandomRotation((-5,-5)),
                        transforms.ToTensor(), transforms.Normalize(_MEAN, _STD)]),
]


# ─────────────────────────────────────────────────────────────────────────────
# SE-Net (Squeeze-Excitation Block)
# ─────────────────────────────────────────────────────────────────────────────

class _SEBlock(nn.Module):
    """Recalibra channels por importância aprendida (SE-Net, Hu et al. 2018)."""
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc   = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        w = self.pool(x).view(b, c)
        w = self.fc(w).view(b, c, 1, 1)
        return x * w


# ─────────────────────────────────────────────────────────────────────────────
# Construção do modelo
# ─────────────────────────────────────────────────────────────────────────────

def _build_model(num_classes: int = NUM_CLASSES, pretrained: bool = True) -> nn.Module:
    """ResNet50 + SE-Net no layer4 + cabeça FC com BatchNorm."""
    weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    model   = models.resnet50(weights=weights)

    # Injeta SE-Net no último bloco do layer4 (2048 channels)
    last_block = model.layer4[-1]
    model.layer4[-1] = nn.Sequential(last_block, _SEBlock(2048))

    # Cabeça de classificação
    model.fc = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(2048, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.3),
        nn.Linear(512, num_classes),
    )
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Weighted Focal Loss
# ─────────────────────────────────────────────────────────────────────────────

class _WeightedFocalLoss(nn.Module):
    """Focal Loss com pesos por classe para desbalanceamento extremo."""
    def __init__(self, gamma: float = 2.0, label_smoothing: float = 0.05,
                 class_weights: torch.Tensor = None):
        super().__init__()
        self.gamma          = gamma
        self.label_smoothing = label_smoothing
        self.class_weights  = class_weights

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets, reduction="none",
                             label_smoothing=self.label_smoothing,
                             weight=self.class_weights)
        pt = torch.exp(-ce)
        return ((1.0 - pt) ** self.gamma * ce).mean()


# ─────────────────────────────────────────────────────────────────────────────
# CutMix collate
# ─────────────────────────────────────────────────────────────────────────────

class _CutMixCollate:
    """Collate que aplica CutMix com probabilidade `prob`."""
    def __init__(self, alpha: float = 0.3, prob: float = 0.5):
        self.alpha = alpha
        self.prob  = prob

    def __call__(self, batch):
        imgs   = torch.stack([b[0] for b in batch])
        labels = torch.tensor([b[1] for b in batch])

        if np.random.random() > self.prob or imgs.size(0) < 2:
            return imgs, labels

        lam = np.random.beta(self.alpha, self.alpha)
        idx = torch.randperm(imgs.size(0))

        h, w   = imgs.shape[-2:]
        cut_h  = int(h * np.sqrt(1.0 - lam))
        cut_w  = int(w * np.sqrt(1.0 - lam))
        cx, cy = np.random.randint(0, w), np.random.randint(0, h)
        x1 = np.clip(cx - cut_w // 2, 0, w)
        x2 = np.clip(cx + cut_w // 2, 0, w)
        y1 = np.clip(cy - cut_h // 2, 0, h)
        y2 = np.clip(cy + cut_h // 2, 0, h)

        imgs[:, :, y1:y2, x1:x2] = imgs[idx, :, y1:y2, x1:x2]
        lam_real = 1.0 - (x2 - x1) * (y2 - y1) / (h * w)

        return imgs, (labels, labels[idx], lam_real)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset com ordem fixa de classes
# ─────────────────────────────────────────────────────────────────────────────

class _CrackDataset(ImageFolder):
    def find_classes(self, directory: str):
        class_to_idx = {c: i for i, c in enumerate(CLASSES)}
        return CLASSES, class_to_idx


# ─────────────────────────────────────────────────────────────────────────────
# Utilidades de imagem
# ─────────────────────────────────────────────────────────────────────────────

def _crop_crack(bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Recorta bounding box da fissura com padding. Retorna original se máscara vazia."""
    if mask is None or cv2.countNonZero(mask) == 0:
        return bgr
    pts = cv2.findNonZero(mask)
    x, y, w, h = cv2.boundingRect(pts)
    ih, iw = bgr.shape[:2]
    x1 = max(0, x - CROP_PAD);  y1 = max(0, y - CROP_PAD)
    x2 = min(iw, x + w + CROP_PAD); y2 = min(ih, y + h + CROP_PAD)
    crop = bgr[y1:y2, x1:x2]
    return crop if crop.shape[0] >= 10 and crop.shape[1] >= 10 else bgr


# ─────────────────────────────────────────────────────────────────────────────
# Classe principal
# ─────────────────────────────────────────────────────────────────────────────

class DeepCrackClassifier:
    """
    ResNet50 + SE-Net fine-tuned para classificação de criticidade de fissuras.

    Inferência:
        clf = DeepCrackClassifier("src/model_weights/resnet50_crack.pth")
        classe, confianca = clf.predict(bgr_img, binary_mask)

    Treino:
        clf = DeepCrackClassifier()
        clf.train("dataset/images", epochs=5)
    """

    CLASSES = CLASSES

    # Thresholds conservadores: CRITICA tem threshold BAIXO para nunca ser perdida.
    # Em caso de dúvida o modelo prefere classificar pra cima (CRITICA) do que pra baixo.
    CLASS_THRESHOLDS = {
        "CRITICA":      0.40,
        "SEMI_CRITICA": 0.50,
        "NAO_CRITICA":  0.55,
        "DESCARTADA":   0.60,
    }
    CONFIDENCE_THRESHOLD = 0.50  # fallback global

    def __init__(self, weights_path: str = None, device: str = None,
                 confidence_threshold: float = None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.confidence_threshold = confidence_threshold or self.CONFIDENCE_THRESHOLD
        self._thresh_clf = _ThresholdClassifier(scale_factor=1.0)

        self.model = _build_model(pretrained=(weights_path is None))
        self.model.to(self.device)

        if weights_path is not None:
            self.load(weights_path)

        self.model.eval()

    # ── Inferência ────────────────────────────────────────────────────────────

    def predict(self, bgr: np.ndarray, mask: np.ndarray = None,
                use_tta: bool = True) -> tuple:
        """Retorna (classe: str, confiança: float)."""
        img = _crop_crack(bgr, mask) if (mask is not None and cv2.countNonZero(mask) > 0) else bgr
        pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        return self._predict_tta(pil) if use_tta else self._predict_single(pil)

    def _predict_single(self, pil: Image.Image) -> tuple:
        t = _INFER_TRANSFORM(pil).unsqueeze(0).to(self.device)
        with torch.no_grad():
            probs = torch.softmax(self.model(t), dim=1).squeeze().cpu().numpy()
        idx = int(np.argmax(probs))
        return CLASSES[idx], float(probs[idx])

    def _predict_tta(self, pil: Image.Image) -> tuple:
        all_probs = []
        with torch.no_grad():
            for tf in _TTA_TRANSFORMS:
                t = tf(pil).unsqueeze(0).to(self.device)
                p = torch.softmax(self.model(t), dim=1).squeeze().cpu().numpy()
                all_probs.append(p)

        mat      = np.array(all_probs)       # (n_tta, n_classes)
        avg      = mat.mean(axis=0)
        std      = mat.std(axis=0)
        idx      = int(np.argmax(avg))
        # Penaliza incerteza entre augmentações (λ=1.5)
        conf     = float(np.clip(avg[idx] * (1.0 - 1.5 * std[idx]), 0.0, 1.0))
        return CLASSES[idx], conf

    def predict_with_fallback(self, bgr: np.ndarray, mask: np.ndarray,
                               threshold_class: tuple,
                               use_tta: bool = True) -> tuple:
        """
        Retorna (classe, confiança, usou_deep_model).

        Safety override: se o classificador morfológico detectou CRITICA,
        o resultado NUNCA é rebaixado pela IA — estrutural não perdoa erro pra baixo.
        """
        deep_class, deep_conf = self.predict(bgr, mask, use_tta=use_tta)
        thresh_class, width_mm = threshold_class

        # Safety override — morfologia diz CRITICA → sempre CRITICA
        if thresh_class == "CRITICA" and deep_class != "CRITICA":
            return "CRITICA", 1.0, False

        thr = self.CLASS_THRESHOLDS.get(deep_class, self.confidence_threshold)
        if deep_conf >= thr:
            return deep_class, deep_conf, True

        # Fallback para limiares morfológicos
        fallback_conf = self._thresh_clf.confidence(width_mm)
        return thresh_class, fallback_conf, False

    # ── Treinamento ───────────────────────────────────────────────────────────

    def train(self, data_dir: str, epochs: int = 5, lr: float = 1e-4,
              batch_size: int = 32, val_split: float = 0.2,
              save_best_path: str = None, patience: int = 7):
        """
        Fine-tuna o ResNet50+SE-Net com Weighted Focal Loss e CutMix.

        Estrutura esperada de data_dir:
            data_dir/CRITICA/    *.jpg|*.png
            data_dir/SEMI_CRITICA/
            data_dir/NAO_CRITICA/
            data_dir/DESCARTADA/
        """
        print(f"[Treino] Dispositivo : {self.device}")
        print(f"[Treino] Dataset     : {data_dir}")

        # ── Dataset ──────────────────────────────────────────────────────────
        full_ds = _CrackDataset(root=data_dir, transform=_TRAIN_TRANSFORM, allow_empty=True)
        print(f"[Treino] Total imagens: {len(full_ds)}")
        print(f"[Treino] Mapeamento  : {full_ds.class_to_idx}")

        val_n   = int(len(full_ds) * val_split)
        train_n = len(full_ds) - val_n
        train_idx, val_idx = random_split(range(len(full_ds)), [train_n, val_n])

        train_ds = Subset(full_ds, list(train_idx))

        val_base           = copy.copy(full_ds)
        val_base.transform = _INFER_TRANSFORM
        val_ds             = Subset(val_base, list(val_idx))

        # Contagem por classe no split de treino (para pesos da loss)
        train_counts = np.zeros(NUM_CLASSES, dtype=np.float32)
        for i in train_ds.indices:
            _, lbl = full_ds.samples[i]
            train_counts[lbl] += 1
        print(f"[Treino] Contagem treino: { {CLASSES[i]: int(train_counts[i]) for i in range(NUM_CLASSES)} }")

        # ── DataLoaders ───────────────────────────────────────────────────────
        use_cuda    = self.device.type == "cuda"
        n_workers   = 2 if use_cuda else 0
        cutmix = _CutMixCollate(alpha=0.3, prob=0.5)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                  collate_fn=cutmix, num_workers=n_workers,
                                  persistent_workers=(n_workers > 0),
                                  pin_memory=use_cuda)
        val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                                  num_workers=n_workers,
                                  persistent_workers=(n_workers > 0),
                                  pin_memory=use_cuda)

        # ── Loss com pesos automáticos ────────────────────────────────────────
        # Pesos = 1/sqrt(count) para suavizar o desbalanceamento sem ser extremo
        counts_safe = np.where(train_counts > 0, train_counts, 1.0)
        raw_weights = 1.0 / np.sqrt(counts_safe)
        raw_weights = raw_weights / raw_weights.mean()
        class_weights = torch.tensor(raw_weights, dtype=torch.float32, device=self.device)
        print(f"[Treino] Pesos de classe: { {CLASSES[i]: f'{class_weights[i].item():.3f}' for i in range(NUM_CLASSES)} }")

        criterion = _WeightedFocalLoss(gamma=2.0, label_smoothing=0.05,
                                       class_weights=class_weights)

        # ── Otimizador (fase 1: só cabeça) ───────────────────────────────────
        FREEZE_EPOCHS = max(2, epochs // 10)
        for name, p in self.model.named_parameters():
            if "fc" not in name:
                p.requires_grad = False

        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=lr, weight_decay=1e-4
        )

        warmup_ep = min(3, epochs // 8)
        sched = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[
                torch.optim.lr_scheduler.LinearLR(optimizer, 0.1, 1.0, warmup_ep),
                torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs - warmup_ep)),
            ],
            milestones=[warmup_ep],
        )

        scaler          = GradScaler(enabled=use_cuda)
        best_f1         = 0.0
        best_state      = None
        no_improve      = 0
        backbone_free   = False

        for epoch in range(1, epochs + 1):

            # ── Descongelamento progressivo na época FREEZE_EPOCHS+1 ──────────
            if epoch == FREEZE_EPOCHS + 1 and not backbone_free:
                print(f"  [Staged FT] Época {epoch}: descongelando layer3 + layer4 (LR x0.1)")
                for name, p in self.model.named_parameters():
                    if name.startswith("layer3") or name.startswith("layer4"):
                        p.requires_grad = True
                backbone_params = [p for n, p in self.model.named_parameters()
                                   if p.requires_grad and "fc" not in n]
                head_params     = [p for n, p in self.model.named_parameters()
                                   if p.requires_grad and "fc" in n]
                optimizer = torch.optim.AdamW([
                    {"params": backbone_params, "lr": lr * 0.1},
                    {"params": head_params,     "lr": lr},
                ], weight_decay=1e-4)
                backbone_free = True

            # ── Treino ────────────────────────────────────────────────────────
            self.model.train()
            t_loss = t_correct = t_total = 0

            pbar = tqdm(train_loader, desc=f"Epoch {epoch:02d}/{epochs} [train]", leave=False)
            for batch_data in pbar:
                imgs, targets = batch_data
                imgs = imgs.to(self.device)

                optimizer.zero_grad()
                with autocast(enabled=use_cuda):
                    out = self.model(imgs)

                    if isinstance(targets, tuple):
                        la, lb, lam = targets
                        la, lb = la.to(self.device), lb.to(self.device)
                        loss = lam * criterion(out, la) + (1.0 - lam) * criterion(out, lb)
                        preds = out.argmax(dim=1)
                        t_correct += (lam * (preds == la).float() + (1-lam) * (preds == lb).float()).sum().item()
                    else:
                        labels = targets.to(self.device)
                        loss   = criterion(out, labels)
                        preds  = out.argmax(dim=1)
                        t_correct += (preds == labels).sum().item()

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                t_loss  += loss.item() * imgs.size(0)
                t_total += imgs.size(0)
                pbar.set_postfix(loss=f"{loss.item():.4f}")

            sched.step()

            # ── Validação ─────────────────────────────────────────────────────
            self.model.eval()
            v_loss = v_correct = v_total = 0
            conf_mat = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)

            with torch.no_grad():
                for imgs, labels in tqdm(val_loader, desc=f"Epoch {epoch:02d}/{epochs} [val]", leave=False):
                    imgs, labels = imgs.to(self.device), labels.to(self.device)
                    out  = self.model(imgs)
                    loss = criterion(out, labels)
                    preds = out.argmax(dim=1)
                    v_loss    += loss.item() * imgs.size(0)
                    v_correct += (preds == labels).sum().item()
                    v_total   += imgs.size(0)
                    for t, p in zip(labels.cpu().numpy(), preds.cpu().numpy()):
                        conf_mat[t, p] += 1

            # F1-macro
            f1s = []
            for c in range(NUM_CLASSES):
                tp = conf_mat[c, c]
                fp = conf_mat[:, c].sum() - tp
                fn = conf_mat[c, :].sum() - tp
                pr = tp / (tp + fp) if tp + fp > 0 else 0.0
                rc = tp / (tp + fn) if tp + fn > 0 else 0.0
                f1s.append(2*pr*rc / (pr+rc) if pr+rc > 0 else 0.0)
            macro_f1 = float(np.mean(f1s))

            # F1 por classe para log
            f1_str = "  ".join(f"{CLASSES[i][:4]}={f1s[i]:.2f}" for i in range(NUM_CLASSES))

            print(
                f"Epoch [{epoch:02d}/{epochs}] "
                f"Train Loss={t_loss/t_total:.4f} Acc={100*t_correct/t_total:.1f}%  |  "
                f"Val Loss={v_loss/v_total:.4f} Acc={100*v_correct/v_total:.1f}%  |  "
                f"F1={macro_f1:.4f} ({f1_str})  |  "
                f"LR={optimizer.param_groups[0]['lr']:.6f}"
            )

            if save_best_path and macro_f1 > best_f1:
                best_f1    = macro_f1
                best_state = copy.deepcopy(self.model.state_dict())
                self.save(save_best_path)
                no_improve = 0
                print(f"  [OK] Melhor modelo salvo (F1={macro_f1:.4f}) -> {save_best_path}")
            else:
                no_improve += 1

            if no_improve >= patience:
                print(f"  [!] Early stopping apos {epoch} epocas (patience={patience})")
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)

        self.model.eval()
        print(f"\n[Treino] Concluido. Melhor F1={best_f1:.4f}")

    # ── Serialização ──────────────────────────────────────────────────────────

    def save(self, path: str):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save(self.model.state_dict(), path)

    def load(self, path: str):
        state = torch.load(path, map_location=self.device)
        # Detecta se é pesos do modelo novo (com SEBlock) ou legado
        is_new = any("SEBlock" in k or "1.fc.0.weight" in k or "layer4.1." in k
                     for k in state.keys())
        try:
            self.model.load_state_dict(state, strict=True)
        except RuntimeError:
            # Pesos incompatíveis (legado sem SE-Net) — reconstrói sem SE e carrega
            print("[load] Pesos legados detectados — carregando sem SE-Net.")
            self.model = _build_model_legacy(num_classes=NUM_CLASSES)
            self.model.to(self.device)
            self.model.load_state_dict(state, strict=False)
        self.model.to(self.device)
        self.model.eval()
        print(f"[load] Pesos carregados: {path}")

    # ── Utilitários ───────────────────────────────────────────────────────────

    @property
    def is_on_gpu(self) -> bool:
        return self.device.type == "cuda"

    def __repr__(self):
        return f"DeepCrackClassifier(device={self.device}, classes={CLASSES})"


def _build_model_legacy(num_classes: int = NUM_CLASSES) -> nn.Module:
    """Reconstrói cabeça com BatchNorm sem SE-Net para carregar pesos antigos."""
    model = models.resnet50(weights=None)
    model.fc = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(2048, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.3),
        nn.Linear(512, num_classes),
    )
    return model
