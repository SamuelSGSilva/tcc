"""
Script de teste automático: roda o pipeline em imagens locais do dataset
e compara a predição com o label real (nome da pasta).
NÃO modifica nenhum arquivo existente do projeto.
"""

import os
import sys
import random
import tempfile

import cv2

sys.path.insert(0, os.path.dirname(__file__))

from src.preprocessing import CrackPreprocessor
from src.segmentation import CrackSegmentor
from src.estimation import WidthEstimator
from src.classification import CriticalityClassifier

try:
    from src.deep_classifier import DeepCrackClassifier
    DEEP_AVAILABLE = True
except ImportError:
    DEEP_AVAILABLE = False

WEIGHTS_PATH  = os.path.join(os.path.dirname(__file__), "src", "model_weights", "resnet50_crack.pth")
DATASET_ROOT  = os.path.join(os.path.dirname(__file__), "dataset", "images")
FALLBACK_SCALE = 0.05
N_SAMPLES = 5   # imagens por classe


def collect_samples(root: str, n: int) -> list[tuple[str, str]]:
    """Retorna lista de (caminho, label_real) amostrada aleatoriamente."""
    samples = []
    for label in os.listdir(root):
        folder = os.path.join(root, label)
        if not os.path.isdir(folder):
            continue
        files = [f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))]
        chosen = random.sample(files, min(n, len(files)))
        for f in chosen:
            samples.append((os.path.join(folder, f), label))
    random.shuffle(samples)
    return samples


def run_pipeline(bgr_img, preprocessor, segmentor, estimator, classifier, deep_clf):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        cv2.imwrite(tmp.name, bgr_img)
        tmp_path = tmp.name

    try:
        _, enhanced = preprocessor.process(tmp_path)
    finally:
        os.unlink(tmp_path)

    binary_mask = segmentor.segment(enhanced, orig_bgr=bgr_img)
    median_px, _, max_px, _, num_pixels, _ = estimator.estimate(binary_mask)

    thresh_class, median_mm = classifier.classify(median_px)
    max_mm = max_px * FALLBACK_SCALE

    if max_mm >= 1.0:
        thresh_class = "CRITICA"

    deep_confidence = None
    final_class = thresh_class

    if deep_clf is not None:
        final_class, deep_confidence, _ = deep_clf.predict_with_fallback(
            bgr_img, binary_mask,
            threshold_class=(thresh_class, median_mm),
            use_tta=False,
        )

    return {
        "class":      final_class,
        "median_mm":  median_mm,
        "max_mm":     max_mm,
        "pixels":     num_pixels,
        "confidence": deep_confidence,
    }


def main():
    print("=" * 65)
    print("  FissuRA — Teste com Dataset Local (com rótulos reais)")
    print("=" * 65)

    preprocessor = CrackPreprocessor()
    segmentor    = CrackSegmentor(merge_components=True, min_area_user=200, canny_thresh=20)
    estimator    = WidthEstimator()
    classifier   = CriticalityClassifier(scale_factor=FALLBACK_SCALE)

    deep_clf = None
    if DEEP_AVAILABLE and os.path.exists(WEIGHTS_PATH):
        try:
            deep_clf = DeepCrackClassifier(weights_path=WEIGHTS_PATH)
            print("  Motor: ResNet50 carregado\n")
        except Exception as e:
            print(f"  ResNet50 indisponível: {e}\n")
    else:
        print("  Motor: Limiares morfológicos\n")

    if not os.path.isdir(DATASET_ROOT):
        print(f"ERRO: dataset não encontrado em {DATASET_ROOT}")
        return

    samples = collect_samples(DATASET_ROOT, N_SAMPLES)
    print(f"  Testando {len(samples)} imagens ({N_SAMPLES} por classe)\n")

    correct = 0
    results = []

    for img_path, label_real in samples:
        nome = os.path.basename(img_path)
        print(f"  [{label_real}] {nome}")

        bgr = cv2.imread(img_path)
        if bgr is None:
            print("    ERRO: não foi possível ler a imagem\n")
            continue

        try:
            r = run_pipeline(bgr, preprocessor, segmentor, estimator, classifier, deep_clf)
        except Exception as e:
            print(f"    ERRO no pipeline: {e}\n")
            continue

        acerto = r["class"] == label_real
        if acerto:
            correct += 1

        conf_str = f"{r['confidence']*100:.1f}%" if r["confidence"] is not None else "-"
        status   = "[OK]    ACERTO" if acerto else f"[ERRO]  ERROU (esperado: {label_real})"

        print(f"    Predicao:   {r['class']}")
        print(f"    Med: {r['median_mm']:.3f} mm  |  Max: {r['max_mm']:.3f} mm  |  Confianca IA: {conf_str}")
        print(f"    {status}\n")

        results.append((nome, label_real, r["class"], acerto))

    total = len(results)
    acc   = correct / total * 100 if total else 0

    print("=" * 65)
    print(f"  Acurácia: {correct}/{total} ({acc:.1f}%)")
    print("=" * 65)
    print(f"  {'Imagem':<30} {'Real':<15} {'Predito':<15} OK?")
    print(f"  {'-'*30} {'-'*15} {'-'*15} ---")
    for nome, real, pred, ok in results:
        print(f"  {nome:<30} {real:<15} {pred:<15} {'OK' if ok else 'X'}")


if __name__ == "__main__":
    main()
