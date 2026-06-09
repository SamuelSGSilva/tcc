"""
build_dataset_v2.py — Reconstrução limpa do dataset de treinamento.

Critérios:
  DESCARTADA  : imagens U* (sem fissura confirmada pelo SDNET2018)
  NAO_CRITICA : imagens C* onde segmentador detectou >= MIN_PIXELS E mediana 0.01-0.20 mm
  SEMI_CRITICA: imagens C* onde segmentador detectou >= MIN_PIXELS E mediana 0.20-1.00 mm
  CRITICA     : imagens C* onde segmentador detectou >= MIN_PIXELS E (mediana >= 1.00 mm OU max >= 1.00 mm)

Imagens C* onde o segmentador detectou < MIN_PIXELS são REJEITADAS (não entram em nenhuma classe).
Cada classe é limitada a MAX_PER_CLASS imagens (subsample aleatório).
CRITICA recebe augmentation se não atingir MAX_PER_CLASS naturalmente.
"""

import os
import sys
import random
import shutil
import cv2
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))

from src.preprocessing import CrackPreprocessor
from src.segmentation import CrackSegmentor
from src.estimation import WidthEstimator

# ── Configurações ────────────────────────────────────────────────────────────
SDNET_ROOT    = os.path.join(os.path.dirname(__file__), "SDNET2018")
OUT_ROOT      = os.path.join(os.path.dirname(__file__), "dataset", "images")
SCALE_MM_PX   = 0.05          # mm por pixel (consistente com o app)
MIN_PIXELS    = 100            # mínimo de pixels segmentados para aceitar imagem C*
MAX_PER_CLASS = 2000           # cap por classe
SEED          = 42

# Limiares ABNT NBR 6118 (mm)
THR_DESCARTADA  = 0.01
THR_NAO_CRITICA = 0.20
THR_SEMI        = 1.00

CRACKED_FOLDERS   = ["D/CD", "P/CP", "W/CW"]
UNCRACKED_FOLDERS = ["D/UD", "P/UP", "W/UW"]
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def collect_images(folders: list[str], root: str) -> list[str]:
    paths = []
    for rel in folders:
        folder = os.path.join(root, rel)
        if not os.path.isdir(folder):
            print(f"  [!] Pasta nao encontrada: {folder}")
            continue
        for f in os.listdir(folder):
            if os.path.splitext(f)[1].lower() in IMG_EXTS:
                paths.append(os.path.join(folder, f))
    return paths


def augment_image(bgr: np.ndarray, idx: int) -> np.ndarray:
    """Augmentation simples: flip H/V + variacao de brilho."""
    ops = [
        lambda x: cv2.flip(x, 1),
        lambda x: cv2.flip(x, 0),
        lambda x: cv2.convertScaleAbs(x, alpha=1.15, beta=10),
        lambda x: cv2.convertScaleAbs(x, alpha=0.85, beta=-10),
        lambda x: cv2.rotate(x, cv2.ROTATE_90_CLOCKWISE),
        lambda x: cv2.rotate(x, cv2.ROTATE_90_COUNTERCLOCKWISE),
    ]
    return ops[idx % len(ops)](bgr)


def safe_copy(src: str, dst_folder: str, prefix: str = "") -> str:
    os.makedirs(dst_folder, exist_ok=True)
    fname = prefix + os.path.basename(src)
    dst = os.path.join(dst_folder, fname)
    # Garante nome único se houver colisão
    counter = 1
    base, ext = os.path.splitext(dst)
    while os.path.exists(dst):
        dst = f"{base}_{counter}{ext}"
        counter += 1
    shutil.copy2(src, dst)
    return dst


def main():
    random.seed(SEED)
    print("=" * 65)
    print("  FissuRA — Build Dataset v2 (criterios limpos)")
    print("=" * 65)

    # ── 1. Limpar dataset existente ──────────────────────────────────────────
    if os.path.isdir(OUT_ROOT):
        print(f"\n  Apagando {OUT_ROOT} ...")
        shutil.rmtree(OUT_ROOT)
    for cls in ["DESCARTADA", "NAO_CRITICA", "SEMI_CRITICA", "CRITICA"]:
        os.makedirs(os.path.join(OUT_ROOT, cls), exist_ok=True)
    print("  Pastas recriadas.\n")

    # ── 2. DESCARTADA — imagens sem fissura (U*) ─────────────────────────────
    print("  [1/2] Coletando DESCARTADA (imagens U*)...")
    uncracked = collect_images(UNCRACKED_FOLDERS, SDNET_ROOT)
    random.shuffle(uncracked)
    uncracked = uncracked[:MAX_PER_CLASS]
    prefix_map = {"UD": "UD_", "UP": "UP_", "UW": "UW_"}
    for src in tqdm(uncracked, desc="  DESCARTADA"):
        parent = os.path.basename(os.path.dirname(src))
        prefix = prefix_map.get(parent, "")
        safe_copy(src, os.path.join(OUT_ROOT, "DESCARTADA"), prefix)
    print(f"  -> {len(uncracked)} imagens DESCARTADA\n")

    # ── 3. Rotular imagens crackeadas ────────────────────────────────────────
    print("  [2/2] Rotulando imagens C* pelo pipeline morfologico...")
    preprocessor = CrackPreprocessor()
    segmentor    = CrackSegmentor(merge_components=True, min_area_user=100,
                                   canny_thresh=13, max_brightness=180,
                                   filter_stains=False)  # sem filtro HSV no label
    estimator    = WidthEstimator()

    cracked_all = collect_images(CRACKED_FOLDERS, SDNET_ROOT)
    random.shuffle(cracked_all)

    buckets = {"NAO_CRITICA": [], "SEMI_CRITICA": [], "CRITICA": []}
    rejected = 0

    prefix_map_c = {"CD": "CD_", "CP": "CP_", "CW": "CW_"}

    for img_path in tqdm(cracked_all, desc="  Rotulando C*"):
        bgr = cv2.imread(img_path)
        if bgr is None:
            continue

        # Pré-processa
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as _t:
            tmp = _t.name
        cv2.imwrite(tmp, bgr)
        try:
            _, enhanced = preprocessor.process(tmp)
        except Exception:
            continue
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)

        # Segmenta
        try:
            mask = segmentor.segment(enhanced, orig_bgr=bgr)
        except Exception:
            continue

        num_pixels = int(np.sum(mask > 0))

        # Rejeita imagens onde quase nada foi detectado
        if num_pixels < MIN_PIXELS:
            rejected += 1
            continue

        # Mede largura
        try:
            median_px, _, max_px, _, _, _ = estimator.estimate(mask)
        except Exception:
            continue

        median_mm = median_px * SCALE_MM_PX
        max_mm    = max_px    * SCALE_MM_PX

        # Classifica
        if median_mm >= THR_SEMI or max_mm >= THR_SEMI:
            label = "CRITICA"
        elif median_mm >= THR_NAO_CRITICA:
            label = "SEMI_CRITICA"
        elif median_mm >= THR_DESCARTADA:
            label = "NAO_CRITICA"
        else:
            # Fissura detectada mas mediana < 0.01 mm — provavelmente ruído
            rejected += 1
            continue

        buckets[label].append(img_path)

    print(f"\n  Rejeitadas (< {MIN_PIXELS} px ou mediana < 0.01 mm): {rejected}")
    for lbl, imgs in buckets.items():
        print(f"  {lbl}: {len(imgs)} imagens detectadas")

    # ── 4. Copiar com cap ────────────────────────────────────────────────────
    print("\n  Copiando com cap de", MAX_PER_CLASS, "por classe...")
    for label, imgs in buckets.items():
        random.shuffle(imgs)
        selected = imgs[:MAX_PER_CLASS]
        dst_folder = os.path.join(OUT_ROOT, label)
        for src in tqdm(selected, desc=f"  {label}"):
            parent = os.path.basename(os.path.dirname(src))
            prefix = prefix_map_c.get(parent, "")
            safe_copy(src, dst_folder, prefix)
        print(f"  -> {len(selected)} copiadas para {label}")

    # ── 5. Augmentation para classes com poucas imagens ─────────────────────
    MIN_PER_CLASS = 1000  # mínimo aceitável para treino
    for cls_name in ["CRITICA", "NAO_CRITICA"]:
        cls_folder = os.path.join(OUT_ROOT, cls_name)
        cls_files  = [f for f in os.listdir(cls_folder) if os.path.splitext(f)[1].lower() in IMG_EXTS]
        current_count = len(cls_files)
        target = MAX_PER_CLASS if cls_name == "CRITICA" else MIN_PER_CLASS

        if current_count < target:
            needed = target - current_count
            print(f"\n  {cls_name} tem {current_count} imagens — gerando {needed} por augmentation...")
            aug_idx = 0
            while needed > 0:
                src_name = cls_files[aug_idx % current_count]
                src_path = os.path.join(cls_folder, src_name)
                bgr = cv2.imread(src_path)
                if bgr is None:
                    aug_idx += 1
                    continue
                aug = augment_image(bgr, aug_idx)
                base, ext = os.path.splitext(src_name)
                out_name = f"{base}_aug{aug_idx}{ext}"
                cv2.imwrite(os.path.join(cls_folder, out_name), aug)
                needed  -= 1
                aug_idx += 1
            print(f"  -> {cls_name} agora com {len(os.listdir(cls_folder))} imagens")

    # ── 6. Resumo final ──────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  Dataset v2 concluido!")
    print("=" * 65)
    total = 0
    for cls in ["DESCARTADA", "NAO_CRITICA", "SEMI_CRITICA", "CRITICA"]:
        n = len(os.listdir(os.path.join(OUT_ROOT, cls)))
        print(f"  {cls:<15}: {n} imagens")
        total += n
    print(f"  {'TOTAL':<15}: {total} imagens")
    print("=" * 65)


if __name__ == "__main__":
    main()
