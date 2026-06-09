"""
evaluate.py
===========
Avaliação quantitativa do ResNet50+SE-Net treinado para classificação de fissuras.

Gera:
  - Acurácia global, F1-macro, F1-weighted
  - Precisão, Recall, F1 por classe
  - Matriz de Confusão
  - Arquivo evaluate_report.txt com todos os números para o TCC

Uso:
    python evaluate.py
    python evaluate.py --data_dir dataset_rotulado --weights src/model_weights/resnet50_crack.pth
"""

import argparse
import sys
import os
import copy
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset
from PIL import Image
from tqdm import tqdm

# ── Constantes (devem coincidir com deep_classifier.py) ──────────────────────

CLASSES = ["CRITICA", "SEMI_CRITICA", "NAO_CRITICA", "DESCARTADA"]
NUM_CLASSES = len(CLASSES)

_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.224, 0.225]

_INFER_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(_MEAN, _STD),
])


# ── Modelo (mesmo que deep_classifier.py) ────────────────────────────────────

class _SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        w = self.pool(x).view(b, c)
        w = self.fc(w).view(b, c, 1, 1)
        return x * w


def _build_model(num_classes=NUM_CLASSES):
    model = models.resnet50(weights=None)
    last_block = model.layer4[-1]
    model.layer4[-1] = nn.Sequential(last_block, _SEBlock(2048))
    model.fc = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(2048, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.3),
        nn.Linear(512, num_classes),
    )
    return model


def _build_model_legacy(num_classes=NUM_CLASSES):
    """ResNet50 sem SE-Net — para pesos treinados antes do SE-Net."""
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


# ── Dataset com ordem fixa de classes ────────────────────────────────────────

class _CrackDataset(ImageFolder):
    def find_classes(self, directory):
        # Filtra só as classes que têm imagens
        existing = [c for c in CLASSES
                    if (Path(directory) / c).exists()
                    and len(list((Path(directory) / c).glob("*.*"))) > 0]
        class_to_idx = {c: CLASSES.index(c) for c in existing}
        return existing, class_to_idx


# ── Métricas ─────────────────────────────────────────────────────────────────

def compute_metrics(conf_mat: np.ndarray, present_classes: list):
    """Calcula precisão, recall, F1 por classe e métricas globais."""
    n = conf_mat.shape[0]
    precision = np.zeros(n)
    recall    = np.zeros(n)
    f1        = np.zeros(n)

    for i in range(n):
        tp = conf_mat[i, i]
        fp = conf_mat[:, i].sum() - tp
        fn = conf_mat[i, :].sum() - tp
        precision[i] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall[i]    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1[i]        = (2 * precision[i] * recall[i] / (precision[i] + recall[i])
                        if (precision[i] + recall[i]) > 0 else 0.0)

    total   = conf_mat.sum()
    correct = conf_mat.diagonal().sum()
    accuracy     = correct / total if total > 0 else 0.0
    f1_macro     = f1.mean()
    # F1-weighted: peso = suporte de cada classe
    support      = conf_mat.sum(axis=1)
    f1_weighted  = (f1 * support).sum() / support.sum() if support.sum() > 0 else 0.0

    return {
        "accuracy":    accuracy,
        "f1_macro":    f1_macro,
        "f1_weighted": f1_weighted,
        "per_class": {
            CLASSES[i]: {
                "precision": precision[i],
                "recall":    recall[i],
                "f1":        f1[i],
                "support":   int(support[i]),
            }
            for i in range(n)
        },
    }


# ── Avaliação principal ───────────────────────────────────────────────────────

def evaluate(data_dir: str, weights_path: str, test_split: float = 0.2,
             batch_size: int = 32, seed: int = 42, output_file: str = "evaluate_report.txt"):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Eval] Dispositivo : {device}")
    print(f"[Eval] Dataset     : {data_dir}")
    print(f"[Eval] Pesos       : {weights_path}")

    # ── Carrega dataset ───────────────────────────────────────────────────────
    dataset = _CrackDataset(root=data_dir, transform=_INFER_TRANSFORM)
    present = dataset.classes
    print(f"[Eval] Classes presentes : {present}")
    print(f"[Eval] Total de imagens  : {len(dataset)}")

    # Contagem por classe
    labels_all = [s[1] for s in dataset.samples]
    for c in present:
        idx_c = CLASSES.index(c)
        count = labels_all.count(idx_c)
        print(f"       {c}: {count}")

    # ── Split estratificado (semente fixa para reprodutibilidade) ─────────────
    rng = random.Random(seed)
    by_class = {}
    for i, (_, lbl) in enumerate(dataset.samples):
        by_class.setdefault(lbl, []).append(i)

    test_indices = []
    train_indices = []
    for lbl, idxs in by_class.items():
        shuffled = idxs[:]
        rng.shuffle(shuffled)
        n_test = max(1, int(len(shuffled) * test_split))
        test_indices.extend(shuffled[:n_test])
        train_indices.extend(shuffled[n_test:])

    test_ds = Subset(dataset, test_indices)
    print(f"\n[Eval] Split de teste ({test_split*100:.0f}%): {len(test_ds)} imagens")

    # Contagem por classe no teste
    test_labels = [labels_all[i] for i in test_indices]
    for c in present:
        idx_c = CLASSES.index(c)
        count = test_labels.count(idx_c)
        print(f"       {c}: {count}")

    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=0, pin_memory=(device.type == "cuda"))

    # ── Carrega modelo (detecta arquitetura automaticamente) ──────────────────
    state = torch.load(weights_path, map_location=device)

    # SE-Net wraps layer4[-1] in Sequential → chave "layer4.2.0.conv1.weight"
    # Legacy (sem SE-Net) usa chave direta "layer4.2.conv1.weight"
    is_senet = any(k.startswith("layer4.2.0.") for k in state.keys())
    arch_name = "ResNet50+SE-Net" if is_senet else "ResNet50 (legacy, sem SE-Net)"
    print(f"\n[Eval] Arquitetura detectada: {arch_name}")

    model = _build_model(NUM_CLASSES) if is_senet else _build_model_legacy(NUM_CLASSES)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[Eval] Chaves ausentes ({len(missing)}): {missing[:3]}{'...' if len(missing)>3 else ''}")
    if unexpected:
        print(f"[Eval] Chaves inesperadas ({len(unexpected)}): {unexpected[:3]}{'...' if len(unexpected)>3 else ''}")
    if not missing and not unexpected:
        print("[Eval] Pesos carregados com sucesso (100% match).")
    model.to(device)
    model.eval()

    # ── Inferência ────────────────────────────────────────────────────────────
    # Dimensão da matriz = número de índices CLASSES realmente usados
    used_idx = sorted({CLASSES.index(c) for c in present})
    idx_remap = {orig: new for new, orig in enumerate(used_idx)}
    n_used = len(used_idx)
    conf_mat = np.zeros((n_used, n_used), dtype=np.int64)

    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for imgs, labels in tqdm(test_loader, desc="Avaliando"):
            imgs = imgs.to(device)
            logits = model(imgs)
            # Pega apenas as colunas correspondentes às classes presentes
            logits_used = logits[:, used_idx]
            preds = logits_used.argmax(dim=1).cpu().numpy()
            # Remap labels para índice local
            labels_np = np.array([idx_remap[int(l)] for l in labels.numpy()])
            all_preds.extend(preds.tolist())
            all_labels.extend(labels_np.tolist())
            for t, p in zip(labels_np, preds):
                conf_mat[t, p] += 1

    # ── Métricas ──────────────────────────────────────────────────────────────
    # Recalcula com classes presentes apenas
    present_classes_used = [CLASSES[i] for i in used_idx]
    n = n_used
    precision = np.zeros(n)
    recall    = np.zeros(n)
    f1        = np.zeros(n)
    support   = conf_mat.sum(axis=1)

    for i in range(n):
        tp = conf_mat[i, i]
        fp = conf_mat[:, i].sum() - tp
        fn = conf_mat[i, :].sum() - tp
        precision[i] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall[i]    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1[i]        = (2 * precision[i] * recall[i] / (precision[i] + recall[i])
                        if (precision[i] + recall[i]) > 0 else 0.0)

    total    = conf_mat.sum()
    correct  = conf_mat.diagonal().sum()
    accuracy = correct / total if total > 0 else 0.0
    f1_macro = f1.mean()
    f1_weighted = (f1 * support).sum() / support.sum() if support.sum() > 0 else 0.0

    # ── Relatório ─────────────────────────────────────────────────────────────
    lines = []
    lines.append("=" * 65)
    lines.append("  AVALIAÇÃO DO MODELO — ResNet50 + SE-Net")
    lines.append(f"  Dataset : {data_dir}")
    lines.append(f"  Pesos   : {weights_path}")
    lines.append(f"  Split   : {(1-test_split)*100:.0f}% treino / {test_split*100:.0f}% teste (semente={seed})")
    lines.append(f"  Total de imagens no teste: {len(test_ds)}")
    lines.append("=" * 65)
    lines.append("")
    lines.append("  MÉTRICAS GLOBAIS")
    lines.append(f"  Acurácia     : {accuracy*100:.2f}%")
    lines.append(f"  F1-Macro     : {f1_macro*100:.2f}%")
    lines.append(f"  F1-Weighted  : {f1_weighted*100:.2f}%")
    lines.append("")
    lines.append("  MÉTRICAS POR CLASSE")
    lines.append(f"  {'Classe':<15} {'Precisão':>10} {'Recall':>10} {'F1-Score':>10} {'Suporte':>10}")
    lines.append("  " + "-" * 57)
    for i, c in enumerate(present_classes_used):
        lines.append(f"  {c:<15} {precision[i]*100:>9.2f}% {recall[i]*100:>9.2f}% {f1[i]*100:>9.2f}% {int(support[i]):>10}")
    lines.append("")
    lines.append("  MATRIZ DE CONFUSÃO")
    lines.append("  (linhas = real, colunas = predito)")
    lines.append("")
    header = "  " + " " * 15
    for c in present_classes_used:
        header += f"  {c[:8]:>8}"
    lines.append(header)
    for i, c in enumerate(present_classes_used):
        row = f"  {c:<15}"
        for j in range(n_used):
            row += f"  {conf_mat[i,j]:>8}"
        lines.append(row)
    lines.append("")
    lines.append("=" * 65)

    report = "\n".join(lines)
    print("\n" + report)

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(report + "\n")
    print(f"\n[OK] Relatório salvo em: {output_file}")

    return {
        "accuracy": accuracy,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "conf_mat": conf_mat,
        "classes": present_classes_used,
    }


# ── Ponto de entrada ──────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Avaliação do ResNet50+SE-Net para fissuras")
    p.add_argument("--data_dir",   default="dataset_rotulado",
                   help="Diretório com subpastas por classe")
    p.add_argument("--weights",    default="src/model_weights/resnet50_crack.pth",
                   help="Caminho para os pesos do modelo")
    p.add_argument("--test_split", default=0.2, type=float,
                   help="Fração reservada para teste (default: 0.2 = 20%%)")
    p.add_argument("--batch_size", default=32, type=int)
    p.add_argument("--seed",       default=42, type=int,
                   help="Semente para reprodutibilidade do split")
    p.add_argument("--output",     default="evaluate_report.txt",
                   help="Arquivo de saída com o relatório")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(
        data_dir=args.data_dir,
        weights_path=args.weights,
        test_split=args.test_split,
        batch_size=args.batch_size,
        seed=args.seed,
        output_file=args.output,
    )
