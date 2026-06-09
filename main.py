import argparse
import os
from pathlib import Path
from src.labeler import DatasetLabeler

def main():
    parser = argparse.ArgumentParser(description="Rotulador Automático de Fissuras do Dataset SDNET2018")
    parser.add_argument("--input_dir", required=True, type=str, help="Pasta com imagens originais do SDNET2018")
    parser.add_argument("--output_dir", default="./output", type=str, help="Pasta de saída para imagens e CSV")
    parser.add_argument("--scale_factor", default=0.1, type=float, help="Fator de conversão px -> mm manual (default: 0.1)")
    parser.add_argument("--enable_aruco", action="store_true", help="Ativar detecção automática de escala via marcador ArUco")
    parser.add_argument("--aruco_size_mm", default=50.0, type=float, help="Tamanho do lado do marcador ArUco em mm (default: 50.0)")
    parser.add_argument("--export_masks", action="store_true", help="Salvar máscaras binárias na pasta masks/ (para Segmentação Semântica)")
    parser.add_argument("--export_yolo", action="store_true", help="Salvar bounding boxes no formato YOLOv8 na pasta labels/")
    parser.add_argument("--debug", action="store_true", help="Salvar versão com overlay de validação")
    parser.add_argument("--workers", default=4, type=int, help="Número de threads/processos paralelos (default: 4)")
    
    # --- ResNet50 ---
    parser.add_argument(
        "--use_deep_model",
        action="store_true",
        help="Usar ResNet50 fine-tuned para classificação de criticidade.",
    )
    parser.add_argument(
        "--model_weights",
        default="src/model_weights/resnet50_crack.pth",
        type=str,
        help="Caminho para os pesos do ResNet50 (default: src/model_weights/resnet50_crack.pth)",
    )

    # --- U-Net ---
    parser.add_argument(
        "--use_unet",
        action="store_true",
        help="Usar U-Net para segmentação (substitui o pipeline Canny-Frangi).",
    )
    parser.add_argument(
        "--unet_weights",
        default="src/model_weights/unet_crack.pth",
        type=str,
        help="Caminho para os pesos do U-Net (default: src/model_weights/unet_crack.pth)",
    )

    args = parser.parse_args()

    # --- Carrega o ResNet50 se solicitado ---
    deep_classifier = None
    if args.use_deep_model:
        try:
            from src.deep_classifier import DeepCrackClassifier
            weights_path = Path(args.model_weights)
            if not weights_path.exists():
                print(f"[AVISO] Pesos ResNet50 não encontrados em '{weights_path}'. Execute python train.py primeiro.")
                print("        Usando classificador por limiares como fallback.")
            else:
                deep_classifier = DeepCrackClassifier(weights_path=str(weights_path))
                print(f"[ResNet50] Modelo carregado de: {weights_path}")
        except ImportError:
            print("[AVISO] PyTorch não instalado. Usando classificador por limiares como fallback.")

    # --- Carrega o U-Net se solicitado ---
    unet_segmentor = None
    if args.use_unet:
        try:
            from src.unet import UNetSegmentor
            unet_path = Path(args.unet_weights)
            if not unet_path.exists():
                print(f"[AVISO] Pesos U-Net não encontrados em '{unet_path}'. Execute python train_unet.py primeiro.")
                print("        Usando segmentação Canny-Frangi como fallback.")
            else:
                unet_segmentor = UNetSegmentor(weights_path=str(unet_path))
                print(f"[U-Net] Modelo carregado de: {unet_path}")
        except ImportError:
            print("[AVISO] PyTorch não instalado. Usando segmentação Canny-Frangi como fallback.")
    
    print(f"Inicializando Labeler em {args.input_dir} com {args.workers} workers...")
    seg_label = "U-Net (Deep Learning)" if unet_segmentor else "Canny-Frangi (Visão Computacional)"
    cls_label = "ResNet50 (Deep Learning)" if deep_classifier else "Limiares de Largura"
    print(f"[OK] Segmentação:    {seg_label}")
    print(f"[OK] Classificação:  {cls_label}")

    labeler = DatasetLabeler(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        fallback_scale_factor=args.scale_factor,
        enable_aruco=args.enable_aruco,
        aruco_size_mm=args.aruco_size_mm,
        export_masks=args.export_masks,
        export_yolo=args.export_yolo,
        debug=args.debug,
        workers=args.workers,
        deep_classifier=deep_classifier,
        unet_segmentor=unet_segmentor,
    )
    
    labeler.run()

if __name__ == "__main__":
    main()
