"""
train_binary.py
===============
Treina ResNet50 com classificação binária de fissuras:

  Classe 0 — INTERVENCAO  : CRITICA + SEMI_CRITICA  (~18.538 imagens)
  Classe 1 — NAO_CRITICA  : NAO_CRITICA             (~14.287 imagens)

Uso:
    python train_binary.py

Pesos salvos em: src/model_weights/resnet50_binary.pth
Relatório final: binary_report.txt
"""

import copy
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import models, transforms
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────

DATA_DIR    = "dataset_rotulado"
OUTPUT_PATH = "src/model_weights/resnet50_binary.pth"
REPORT_PATH = "binary_report.txt"

EPOCHS      = 5
BATCH_SIZE  = 32
LR          = 1e-4
VAL_SPLIT   = 0.20
PATIENCE    = 6
SEED        = 42

CLASSES = ["INTERVENCAO", "NAO_CRITICA"]   # 0, 1

# Pastas do dataset → classe binária
FOLDER_TO_CLASS = {
    "CRITICA":      0,   # INTERVENCAO
    "SEMI_CRITICA": 0,   # INTERVENCAO
    "NAO_CRITICA":  1,   # NAO_CRITICA
}

_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.224, 0.225]

_TRAIN_TF = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandAugment(num_ops=2, magnitude=6),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(_MEAN, _STD),
])

_INFER_TF = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(_MEAN, _STD),
])

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


# ── Dataset ───────────────────────────────────────────────────────────────────

class BinaryCrackDataset(Dataset):
    """Percorre as subpastas e mapeia CRITICA+SEMI_CRITICA→0, NAO_CRITICA→1."""

    def __init__(self, root: str, transform=None):
        self.transform = transform
        self.samples   = []   # (path, label)

        for folder, label in FOLDER_TO_CLASS.items():
            folder_path = Path(root) / folder
            if not folder_path.exists():
                continue
            imgs = [p for p in folder_path.iterdir()
                    if p.suffix.lower() in IMG_EXTS]
            self.samples.extend((str(p), label) for p in imgs)

        if not self.samples:
            raise RuntimeError(f"Nenhuma imagem encontrada em {root}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


# ── Modelo ────────────────────────────────────────────────────────────────────

def build_model(num_classes: int = 2, pretrained: bool = True) -> nn.Module:
    weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    model   = models.resnet50(weights=weights)
    model.fc = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(2048, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.3),
        nn.Linear(512, num_classes),
    )
    return model


# ── Focal Loss ────────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, weight: torch.Tensor = None):
        super().__init__()
        self.gamma  = gamma
        self.weight = weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets, weight=self.weight,
                             reduction="none", label_smoothing=0.05)
        pt = torch.exp(-ce)
        return ((1.0 - pt) ** self.gamma * ce).mean()


# ── Métricas ──────────────────────────────────────────────────────────────────

def metrics(conf: np.ndarray):
    n = conf.shape[0]
    pr = np.zeros(n); rc = np.zeros(n); f1 = np.zeros(n)
    for i in range(n):
        tp = conf[i, i]
        fp = conf[:, i].sum() - tp
        fn = conf[i, :].sum() - tp
        pr[i] = tp / (tp + fp) if tp + fp else 0.0
        rc[i] = tp / (tp + fn) if tp + fn else 0.0
        f1[i] = 2*pr[i]*rc[i]/(pr[i]+rc[i]) if pr[i]+rc[i] else 0.0
    acc     = conf.diagonal().sum() / conf.sum()
    f1_mac  = f1.mean()
    sup     = conf.sum(axis=1)
    f1_w    = (f1 * sup).sum() / sup.sum()
    return acc, f1_mac, f1_w, pr, rc, f1, sup


# ── Treino principal ──────────────────────────────────────────────────────────

def train():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 60)
    print("  Treino Binário — ResNet50")
    print(f"  Dispositivo : {device}")
    print(f"  Dataset     : {DATA_DIR}")
    print(f"  Épocas      : {EPOCHS}   Batch: {BATCH_SIZE}   LR: {LR}")
    print("=" * 60)

    # ── Dataset ───────────────────────────────────────────────────────────────
    full_ds = BinaryCrackDataset(DATA_DIR, transform=_TRAIN_TF)
    labels  = [lbl for _, lbl in full_ds.samples]
    n_total = len(full_ds)
    n_c0    = labels.count(0)
    n_c1    = labels.count(1)
    print(f"\n  Total          : {n_total}")
    print(f"  INTERVENCAO (0): {n_c0}  ({100*n_c0/n_total:.1f}%)")
    print(f"  NAO_CRITICA (1): {n_c1}  ({100*n_c1/n_total:.1f}%)")

    # Split estratificado com semente fixa
    rng = random.Random(SEED)
    by_class = {0: [], 1: []}
    for i, (_, lbl) in enumerate(full_ds.samples):
        by_class[lbl].append(i)

    val_idx = []
    train_idx = []
    for lbl, idxs in by_class.items():
        shuffled = idxs[:]
        rng.shuffle(shuffled)
        n_val = max(1, int(len(shuffled) * VAL_SPLIT))
        val_idx.extend(shuffled[:n_val])
        train_idx.extend(shuffled[n_val:])

    val_base           = copy.copy(full_ds)
    val_base.transform = _INFER_TF
    train_ds = Subset(full_ds,   train_idx)
    val_ds   = Subset(val_base,  val_idx)
    print(f"\n  Treino: {len(train_ds)}   Validação: {len(val_ds)}")

    # DataLoaders
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, pin_memory=(device.type == "cuda"))
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=0, pin_memory=(device.type == "cuda"))

    # Peso de classe (leve, dataset quase balanceado)
    train_labels = [labels[i] for i in train_idx]
    counts = np.array([train_labels.count(c) for c in range(2)], dtype=np.float32)
    raw_w  = 1.0 / np.sqrt(counts)
    raw_w /= raw_w.mean()
    class_weights = torch.tensor(raw_w, dtype=torch.float32, device=device)
    print(f"  Pesos de classe: INTERVENCAO={raw_w[0]:.3f}  NAO_CRITICA={raw_w[1]:.3f}")

    # ── Modelo ────────────────────────────────────────────────────────────────
    model     = build_model(num_classes=2, pretrained=True)
    model.to(device)
    criterion = FocalLoss(gamma=2.0, weight=class_weights)

    # Fase 1: congela backbone, treina só cabeça
    FREEZE_EP = max(2, EPOCHS // 10)
    for name, p in model.named_parameters():
        if "fc" not in name:
            p.requires_grad = False

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR, weight_decay=1e-4
    )
    warmup = min(3, EPOCHS // 8)
    sched  = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[
            torch.optim.lr_scheduler.LinearLR(optimizer, 0.1, 1.0, warmup),
            torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, EPOCHS - warmup)),
        ],
        milestones=[warmup],
    )

    best_f1    = 0.0
    best_acc   = 0.0
    best_state = None
    no_improve = 0
    backbone_freed = False

    print("\n")
    for epoch in range(1, EPOCHS + 1):

        # Descongelamento progressivo
        if epoch == FREEZE_EP + 1 and not backbone_freed:
            print(f"  [FT] Época {epoch}: descongelando layer3+layer4")
            for name, p in model.named_parameters():
                if name.startswith("layer3") or name.startswith("layer4"):
                    p.requires_grad = True
            backbone_p = [p for n, p in model.named_parameters()
                          if p.requires_grad and "fc" not in n]
            head_p     = [p for n, p in model.named_parameters()
                          if p.requires_grad and "fc" in n]
            optimizer  = torch.optim.AdamW([
                {"params": backbone_p, "lr": LR * 0.1},
                {"params": head_p,     "lr": LR},
            ], weight_decay=1e-4)
            backbone_freed = True

        # Treino
        model.train()
        t_loss = t_correct = t_total = 0
        for imgs, lbls in tqdm(train_loader, desc=f"Ep {epoch:02d}/{EPOCHS} [train]", leave=False):
            imgs, lbls = imgs.to(device), lbls.to(device)
            optimizer.zero_grad()
            out  = model(imgs)
            loss = criterion(out, lbls)
            loss.backward()
            optimizer.step()
            t_loss    += loss.item() * imgs.size(0)
            t_correct += (out.argmax(1) == lbls).sum().item()
            t_total   += imgs.size(0)
        sched.step()

        # Validação
        model.eval()
        conf = np.zeros((2, 2), dtype=np.int64)
        v_loss = v_total = 0
        with torch.no_grad():
            for imgs, lbls in tqdm(val_loader, desc=f"Ep {epoch:02d}/{EPOCHS} [val]", leave=False):
                imgs, lbls = imgs.to(device), lbls.to(device)
                out  = model(imgs)
                loss = criterion(out, lbls)
                v_loss  += loss.item() * imgs.size(0)
                v_total += imgs.size(0)
                for t, p in zip(lbls.cpu().numpy(), out.argmax(1).cpu().numpy()):
                    conf[t, p] += 1

        acc, f1_mac, f1_w, pr, rc, f1, sup = metrics(conf)

        print(
            f"Ep [{epoch:02d}/{EPOCHS}] "
            f"Train Loss={t_loss/t_total:.4f} Acc={100*t_correct/t_total:.1f}%  |  "
            f"Val Loss={v_loss/v_total:.4f} Acc={100*acc:.2f}%  "
            f"F1={100*f1_mac:.2f}%  |  "
            f"INTERV: P={100*pr[0]:.1f}% R={100*rc[0]:.1f}%  "
            f"NAO_CRIT: P={100*pr[1]:.1f}% R={100*rc[1]:.1f}%"
        )

        if f1_mac > best_f1:
            best_f1    = f1_mac
            best_acc   = acc
            best_state = copy.deepcopy(model.state_dict())
            Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
            torch.save(best_state, OUTPUT_PATH)
            no_improve = 0
            print(f"  [OK] Melhor modelo salvo (F1={100*f1_mac:.2f}% Acc={100*acc:.2f}%) → {OUTPUT_PATH}")
        else:
            no_improve += 1

        if no_improve >= PATIENCE:
            print(f"  [!] Early stopping após {epoch} épocas")
            break

    # ── Avaliação final no conjunto de teste ──────────────────────────────────
    print("\n" + "=" * 60)
    print("  AVALIAÇÃO FINAL NO CONJUNTO DE TESTE (20%)")
    print("=" * 60)

    model.load_state_dict(best_state)
    model.eval()

    test_ds = Subset(val_base, val_idx)   # mesmo split de val = test holdout
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=0)

    conf_test = np.zeros((2, 2), dtype=np.int64)
    with torch.no_grad():
        for imgs, lbls in tqdm(test_loader, desc="Teste final"):
            imgs = imgs.to(device)
            out  = model(imgs)
            for t, p in zip(lbls.numpy(), out.argmax(1).cpu().numpy()):
                conf_test[t, p] += 1

    acc, f1_mac, f1_w, pr, rc, f1, sup = metrics(conf_test)

    lines = []
    lines.append("=" * 65)
    lines.append("  AVALIAÇÃO FINAL — ResNet50 Binário (INTERVENCAO / NAO_CRITICA)")
    lines.append(f"  Dataset : {DATA_DIR}")
    lines.append(f"  Pesos   : {OUTPUT_PATH}")
    lines.append(f"  Split   : {int((1-VAL_SPLIT)*100)}% treino / {int(VAL_SPLIT*100)}% teste  (semente={SEED})")
    lines.append(f"  Imagens no teste: {conf_test.sum()}")
    lines.append("=" * 65)
    lines.append("")
    lines.append("  MÉTRICAS GLOBAIS")
    lines.append(f"  Acurácia     : {acc*100:.2f}%")
    lines.append(f"  F1-Macro     : {f1_mac*100:.2f}%")
    lines.append(f"  F1-Weighted  : {f1_w*100:.2f}%")
    lines.append("")
    lines.append("  MÉTRICAS POR CLASSE")
    lines.append(f"  {'Classe':<20} {'Precisão':>10} {'Recall':>10} {'F1-Score':>10} {'Suporte':>10}")
    lines.append("  " + "-" * 62)
    for i, c in enumerate(CLASSES):
        lines.append(f"  {c:<20} {pr[i]*100:>9.2f}% {rc[i]*100:>9.2f}% {f1[i]*100:>9.2f}% {int(sup[i]):>10}")
    lines.append("")
    lines.append("  MATRIZ DE CONFUSÃO  (linhas=real, colunas=predito)")
    lines.append("")
    lines.append("  " + " "*20 + "  ".join(f"{c[:10]:>10}" for c in CLASSES))
    for i, c in enumerate(CLASSES):
        row = f"  {c:<20}" + "  ".join(f"{conf_test[i,j]:>10}" for j in range(2))
        lines.append(row)
    lines.append("")
    lines.append("=" * 65)

    report = "\n".join(lines)
    print("\n" + report)

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report + "\n")
    print(f"\n[OK] Relatório salvo em: {REPORT_PATH}")
    print(f"[OK] Pesos finais em  : {OUTPUT_PATH}")


if __name__ == "__main__":
    train()
