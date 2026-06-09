"""
train_unet.py
=============
Script standalone para treinar o U-Net de segmentação de fissuras.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PASSO 1 — Gerar as máscaras do SDNET2018
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    python main.py \\
        --input_dir  ./SDNET2018 \\
        --output_dir ./dataset_rotulado \\
        --scale_factor 0.234 \\
        --export_masks \\
        --workers 4

Isso cria:
    dataset_rotulado/
        images/CRITICA/     *.jpg
        images/SEMI_CRITICA/*.jpg
        images/NAO_CRITICA/ *.jpg
        images/DESCARTADA/  *.jpg
        masks/CRITICA/      *.png   ← máscaras binárias
        masks/SEMI_CRITICA/ *.png
        ...

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PASSO 2 — Treinar o U-Net
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    python train_unet.py \\
        --images_dir ./dataset_rotulado/images \\
        --masks_dir  ./dataset_rotulado/masks  \\
        --epochs 40 --batch_size 8

O melhor modelo é salvo em:
    src/model_weights/unet_crack.pth

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PASSO 3 — Usar no app / pipeline
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # App Streamlit:
    streamlit run app.py
    → O app detecta automaticamente os pesos e ativa o U-Net

    # Pipeline de rotulagem:
    python main.py --input_dir ./SDNET2018 --output_dir ./saida \\
                   --scale_factor 0.234 --use_unet --export_masks
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.unet import UNetSegmentor


def parse_args():
    parser = argparse.ArgumentParser(
        description="Treino do U-Net para segmentação de fissuras estruturais",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--images_dir",
        required=True,
        type=str,
        help="Pasta com as imagens originais (pode ter subpastas por classe).",
    )
    parser.add_argument(
        "--masks_dir",
        required=True,
        type=str,
        help="Pasta com as máscaras binárias PNG (mesmo nome das imagens).",
    )
    parser.add_argument(
        "--epochs",
        default=40,
        type=int,
        help="Número de épocas de treinamento.",
    )
    parser.add_argument(
        "--batch_size",
        default=8,
        type=int,
        help="Tamanho do batch. Reduza para 4 se ficar sem memória.",
    )
    parser.add_argument(
        "--lr",
        default=1e-3,
        type=float,
        help="Taxa de aprendizado AdamW.",
    )
    parser.add_argument(
        "--val_split",
        default=0.2,
        type=float,
        help="Fração dos dados para validação (0.0 – 0.5).",
    )
    parser.add_argument(
        "--output",
        default="src/model_weights/unet_crack.pth",
        type=str,
        help="Caminho de saída para os pesos do melhor modelo.",
    )
    parser.add_argument(
        "--device",
        default=None,
        type=str,
        choices=["cuda", "cpu"],
        help="Dispositivo de treino. Padrão: detecta automaticamente.",
    )
    parser.add_argument(
        "--patience",
        default=8,
        type=int,
        help="Épocas sem melhoria antes do early stopping.",
    )
    parser.add_argument(
        "--in_channels",
        default=1,
        type=int,
        choices=[1, 3],
        help="Canais de entrada: 1 (grayscale) ou 3 (RGB).",
    )
    parser.add_argument(
        "--base_filters",
        default=32,
        type=int,
        help="Filtros no primeiro nível do U-Net (32 = leve, 64 = mais preciso).",
    )
    parser.add_argument(
        "--threshold",
        default=0.45,
        type=float,
        help="Limiar de probabilidade para binarização da máscara (0.0 – 1.0).",
    )
    parser.add_argument(
        "--no_attention",
        action="store_true",
        help="Usa U-Net padrão em vez do Attention U-Net (compatível com pesos antigos).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    images_dir = Path(args.images_dir)
    masks_dir  = Path(args.masks_dir)
    output     = Path(args.output)

    if not images_dir.exists():
        print(f"[ERRO] Pasta de imagens não encontrada: {images_dir}")
        sys.exit(1)
    if not masks_dir.exists():
        print(f"[ERRO] Pasta de máscaras não encontrada: {masks_dir}")
        sys.exit(1)

    output.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  Treinamento U-Net — Segmentação de Fissuras")
    print("=" * 60)
    print(f"  Images Dir  : {images_dir}")
    print(f"  Masks Dir   : {masks_dir}")
    print(f"  Epochs      : {args.epochs}")
    print(f"  Batch Size  : {args.batch_size}")
    print(f"  LR          : {args.lr}")
    print(f"  Val Split   : {args.val_split * 100:.0f}%")
    print(f"  In Channels : {args.in_channels}")
    print(f"  Base Filters: {args.base_filters}")
    print(f"  Threshold   : {args.threshold}")
    print(f"  Patience    : {args.patience}")
    print(f"  Output      : {output}")
    print("=" * 60)

    use_attention = not args.no_attention
    arch_name     = "AttentionUNet" if use_attention else "UNet"
    print(f"  Arquitetura : {arch_name}")
    print("=" * 60)

    seg = UNetSegmentor(
        weights_path   = None,
        device         = args.device,
        in_channels    = args.in_channels,
        base_filters   = args.base_filters,
        threshold      = args.threshold,
        use_attention  = use_attention,
    )

    seg.train(
        images_dir    = str(images_dir),
        masks_dir     = str(masks_dir),
        epochs        = args.epochs,
        lr            = args.lr,
        batch_size    = args.batch_size,
        val_split     = args.val_split,
        save_best_path= str(output),
        patience      = args.patience,
    )

    print(f"\n[OK] Modelo U-Net disponível em: {output.resolve()}")
    print("     Para usar no app:   streamlit run app.py")
    print("     Para usar no pipeline: adicione --use_unet ao main.py")


if __name__ == "__main__":
    main()
