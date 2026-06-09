"""
train.py
========
Script standalone para fine-tuning do ResNet50 no dataset de fissuras.

Uso:
    python train.py --data_dir ./dataset_rotulado/images --epochs 25 --batch_size 32

Estrutura esperada de --data_dir:
    data_dir/
        CRITICA/      *.jpg / *.png ...
        SEMI_CRITICA/
        NAO_CRITICA/
        DESCARTADA/

O melhor modelo (maior F1-score macro na validação) é salvo automaticamente em:
    src/model_weights/resnet50_crack.pth
"""

import argparse
import sys
import os
from pathlib import Path

# Garante que o pacote src está no path
sys.path.insert(0, str(Path(__file__).parent))

from src.deep_classifier import DeepCrackClassifier


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tuning do ResNet50 para classificação de fissuras estruturais"
    )
    parser.add_argument(
        "--data_dir",
        required=True,
        type=str,
        help="Pasta com subdiretórios por classe (CRITICA/, SEMI_CRITICA/, NAO_CRITICA/, DESCARTADA/)",
    )
    parser.add_argument(
        "--epochs",
        default=5,
        type=int,
        help="Número de épocas de treinamento (default: 5)",
    )
    parser.add_argument(
        "--batch_size",
        default=32,
        type=int,
        help="Tamanho do batch (default: 32). Reduza para 16 se ficar sem memória.",
    )
    parser.add_argument(
        "--lr",
        default=1e-4,
        type=float,
        help="Taxa de aprendizado AdamW (default: 1e-4)",
    )
    parser.add_argument(
        "--val_split",
        default=0.2,
        type=float,
        help="Fração dos dados para validação (default: 0.2 = 20%%)",
    )
    parser.add_argument(
        "--output",
        default="src/model_weights/resnet50_crack.pth",
        type=str,
        help="Caminho de saída para os pesos do melhor modelo",
    )
    parser.add_argument(
        "--device",
        default=None,
        type=str,
        choices=["cuda", "cpu"],
        help="Dispositivo de treinamento. Padrão: detecta automaticamente (CUDA se disponível).",
    )
    parser.add_argument(
        "--patience",
        default=7,
        type=int,
        help="Épocas sem melhoria antes de early stopping (default: 7)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"[ERRO] Diretório não encontrado: {data_dir}")
        sys.exit(1)

    # Verifica se as subpastas de classe existem
    expected_classes = {"CRITICA", "SEMI_CRITICA", "NAO_CRITICA", "DESCARTADA"}
    found_classes = {p.name for p in data_dir.iterdir() if p.is_dir()}
    missing = expected_classes - found_classes
    if missing:
        print(f"[AVISO] Subpastas ausentes em {data_dir}: {missing}")
        print("         O treinamento continuará com as classes disponíveis.")

    print("=" * 60)
    print("  Fine-tuning ResNet50 — Classificação de Fissuras")
    print("=" * 60)
    print(f"  Data Dir   : {data_dir}")
    print(f"  Epochs     : {args.epochs}")
    print(f"  Batch Size : {args.batch_size}")
    print(f"  LR         : {args.lr}")
    print(f"  Val Split  : {args.val_split * 100:.0f}%")
    print(f"  Patience   : {args.patience}")
    print(f"  Output     : {args.output}")
    print("=" * 60)

    # Garante que o diretório de saída existe
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Cria o modelo com pesos ImageNet como ponto de partida
    clf = DeepCrackClassifier(weights_path=None, device=args.device)

    # Inicia o treinamento
    clf.train(
        data_dir=str(data_dir),
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        val_split=args.val_split,
        save_best_path=str(output_path),
        patience=args.patience,
    )

    print(f"\n[OK] Modelo final disponível em: {output_path.resolve()}")
    print("     Para usar no app: streamlit run app.py")


if __name__ == "__main__":
    main()
