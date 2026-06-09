import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import shutil
import time

from src.preprocessing import CrackPreprocessor
from src.segmentation import CrackSegmentor
from src.estimation import WidthEstimator
from src.classification import CriticalityClassifier
from src.calibration import ScaleCalibrator
from src.export import create_yolo_yaml, generate_yolo_annotation

# Import opcional — só falha em runtime se o usuário tentar usar sem PyTorch instalado
try:
    from src.deep_classifier import DeepCrackClassifier
except ImportError:
    DeepCrackClassifier = None

class DatasetLabeler:
    def __init__(self, input_dir, output_dir, fallback_scale_factor, enable_aruco,
                 aruco_size_mm, export_masks, export_yolo, debug, workers,
                 deep_classifier=None, unet_segmentor=None):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.debug = debug
        self.workers = workers
        self.fallback_scale_factor = fallback_scale_factor
        self.enable_aruco = enable_aruco
        self.export_masks = export_masks
        self.export_yolo = export_yolo

        self.preprocessor = CrackPreprocessor()
        self.segmentor = CrackSegmentor(merge_components=True, unet_segmentor=unet_segmentor)
        self.estimator = WidthEstimator()
        self.deep_classifier = deep_classifier  # None = usa classificador por limiares
        self.unet_segmentor  = unet_segmentor   # None = usa Canny-Frangi
        
        self.aruco_size_mm = aruco_size_mm
        
        # Setup logging
        log_path = self.output_dir / "processing.log"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Subdirectories for classes
        self.classes = ["CRITICA", "SEMI_CRITICA", "NAO_CRITICA", "DESCARTADA"]
        # In YOLO format, classes are integers. We map them by index.
        # CRITICA=0, SEMI_CRITICA=1, NAO_CRITICA=2. DESCARTADA is omitted (background).
        self.yolo_classes = ["CRITICA", "SEMI_CRITICA", "NAO_CRITICA"]
        
        for cls in self.classes:
            (self.output_dir / "images" / cls).mkdir(parents=True, exist_ok=True)
            if self.debug:
                (self.output_dir / "debug" / cls).mkdir(parents=True, exist_ok=True)
                
        if self.export_masks:
            for cls in self.classes:
                (self.output_dir / "masks" / cls).mkdir(parents=True, exist_ok=True)
                
        if self.export_yolo:
            (self.output_dir / "labels").mkdir(parents=True, exist_ok=True)
            # Create dataset.yaml
            create_yolo_yaml(self.output_dir, self.yolo_classes)
                
        logging.basicConfig(
            filename=str(log_path),
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

    def process_image(self, img_path):
        """
        Process a single image pipeline.
        Returns a dict of results or raises an Exception.
        """
        try:
            # Etapa 0: Calibração de escala
            current_scale_factor = self.fallback_scale_factor
            aruco_corners = None
            
            # Etapa 1: Pré-processamento
            orig_img, enhanced_img = self.preprocessor.process(img_path)
            
            if self.enable_aruco:
                calibrator = ScaleCalibrator(marker_size_mm=self.aruco_size_mm)
                dynamic_scale, aruco_corners = calibrator.calibrate(orig_img)
                if dynamic_scale is not None:
                    current_scale_factor = dynamic_scale
            
            # Classifier needs to be instantiated per image because scale factor might change
            classifier = CriticalityClassifier(scale_factor=current_scale_factor)
            
            # Etapa 2: Segmentação (Canny-Frangi ou U-Net)
            binary_mask = self.segmentor.segment(enhanced_img, orig_bgr=orig_img)
            
            # Etapa 3: Estimativa de largura
            median_px, min_px, max_px, std_px, num_pixels, skel_mask = self.estimator.estimate(binary_mask)
            
            # Etapa 4 — Classificação com fallback automático
            deep_confidence = None
            used_deep_model = False

            if self.deep_classifier is not None:
                # Classificador por limiares (referência)
                threshold_class, median_mm = classifier.classify(median_px)

                # ResNet50 com crop da máscara + fallback por confiança
                crack_class, deep_confidence, used_deep_model = \
                    self.deep_classifier.predict_with_fallback(
                        orig_img,
                        binary_mask,
                        threshold_class=(threshold_class, median_mm),
                        use_tta=True
                    )

                # Recalcula mm para min/max (sempre usa limiares para medição morfológica)
                _, min_mm = classifier.classify(min_px)
                _, max_mm = classifier.classify(max_px)
            else:
                # Apenas classificador por limiares
                crack_class, median_mm = classifier.classify(median_px)
                _, min_mm = classifier.classify(min_px)
                _, max_mm = classifier.classify(max_px)

            std_mm = std_px * classifier.scale_factor

            # Confiança final
            confidence = 0.0
            if median_mm > 0.0:
                cv_pct = std_mm / median_mm
                confidence = max(0.0, 1.0 - cv_pct)
            if crack_class == "DESCARTADA":
                confidence = 1.0
            # Se modelo profundo foi usado e forneceu confiança, ela prevalece
            if deep_confidence is not None and used_deep_model:
                confidence = deep_confidence
                
            # Etapa 5: Exportação
            # Save copy to target dir inside images/
            out_img_path = self.output_dir / "images" / crack_class / img_path.name
            shutil.copy2(str(img_path), str(out_img_path))
            
            # Save Mask
            if self.export_masks:
                out_mask_path = self.output_dir / "masks" / crack_class / (img_path.stem + ".png")
                cv2.imwrite(str(out_mask_path), binary_mask)
                
            # Export YOLO
            if self.export_yolo and crack_class != "DESCARTADA":
                class_id = self.yolo_classes.index(crack_class)
                yolo_annotation = generate_yolo_annotation(binary_mask, class_id)
                if yolo_annotation:
                    out_txt_path = self.output_dir / "labels" / (img_path.stem + ".txt")
                    with open(out_txt_path, "w") as f:
                        f.write(yolo_annotation + "\n")
            
            # Save debug if requested
            if self.debug:
                debug_img = orig_img.copy()
                # Overlay mask in blue
                debug_img[binary_mask > 0] = [255, 0, 0] # BGR
                # Overlay skeleton in red
                debug_img[skel_mask > 0] = [0, 0, 255]
                # Annotate text
                method = "ResNet50" if used_deep_model else "Limiares"
                cv2.putText(debug_img, f"Classe: {crack_class} ({method})", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.putText(debug_img, f"Largura Mediana: {median_mm:.3f} mm", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.putText(debug_img, f"Confianca: {confidence:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.putText(debug_img, f"Fator de Escala: {current_scale_factor:.5f} mm/px", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0) if aruco_corners is not None else (0, 165, 255), 1)
                
                if aruco_corners is not None:
                    # Draw polygon around aruco marker
                    corners_int = np.int32(aruco_corners).reshape(-1, 2)
                    cv2.polylines(debug_img, [corners_int], True, (0, 255, 255), 2)
                    cv2.putText(debug_img, "ArUco Detectado", tuple(corners_int[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                    
                debug_out_path = self.output_dir / "debug" / crack_class / img_path.name
                cv2.imwrite(str(debug_out_path), debug_img)
                
            result = {
                "arquivo": img_path.name,
                "caminho_original": str(img_path),
                "classe": crack_class,
                "metodo_classificacao": "ResNet50" if used_deep_model else "Limiares",
                "largura_mediana_mm": round(median_mm, 4),
                "largura_min_mm": round(min_mm, 4),
                "largura_max_mm": round(max_mm, 4),
                "std_mm": round(std_mm, 4),
                "num_pixels_fissura": num_pixels,
                "confianca": round(confidence, 4),
                "fator_escala": current_scale_factor,
                "calibracao_auto": aruco_corners is not None,
                "error": None
            }
            return result
            
        except Exception as e:
            logging.error(f"Error processing {img_path}: {str(e)}")
            return {
                "arquivo": img_path.name,
                "caminho_original": str(img_path),
                "classe": "ERRO",
                "largura_mediana_mm": 0,
                "largura_min_mm": 0,
                "largura_max_mm": 0,
                "std_mm": 0,
                "num_pixels_fissura": 0,
                "confianca": 0.0,
                "fator_escala": 0.0,
                "calibracao_auto": False,
                "error": str(e)
            }

    def run(self):
        start_time = time.time()

        # Valid image extensions
        exts = ['.jpg', '.jpeg', '.png', '.bmp']
        image_paths = [p for p in self.input_dir.rglob('*') if p.suffix.lower() in exts]

        filtered_paths = []
        for p in image_paths:
            filtered_paths.append(p)

        total_images = len(filtered_paths)
        if total_images == 0:
            print(f"Nenhuma imagem encontrada em {self.input_dir}")
            return

        print(f"Iniciando processamento de {total_images} imagens...")

        results = []
        errors = 0

        # Quando o DeepCrackClassifier (PyTorch) está ativo, modelos não são
        # serializáveis (pickle) pelo ProcessPoolExecutor.  Nesse caso usamos
        # processamento sequencial para evitar erros de serialização.
        use_parallel = self.workers > 1 and self.deep_classifier is None

        if use_parallel:
            with ProcessPoolExecutor(max_workers=self.workers) as executor:
                futures = {executor.submit(self.process_image, p): p for p in filtered_paths}
                for future in tqdm(as_completed(futures), total=total_images, desc="Processando img..."):
                    res = future.result()
                    results.append(res)
                    if res["error"] is not None:
                        errors += 1
        else:
            if self.deep_classifier is not None and self.workers > 1:
                print("[AVISO] ResNet50 ativo: processamento serial (PyTorch não é picklable). Use --workers 1 para suprimir este aviso.")
            for p in tqdm(filtered_paths, total=total_images, desc="Processando img..."):
                res = self.process_image(p)
                results.append(res)
                if res["error"] is not None:
                    errors += 1
                    
        # Filter successful results for stats
        valid_results = [r for r in results if r["error"] is None]
        df = pd.DataFrame(valid_results)
        
        # Export CSV
        csv_path = self.output_dir / "results.csv"
        # Drop the error column for final CSV
        df_csv = df.drop(columns=["error"])
        df_csv.to_csv(csv_path, index=False)
        
        # Calculations for report
        end_time = time.time()
        total_time = end_time - start_time
        
        class_counts = df["classe"].value_counts()
        print("\n" + "="*50)
        print(" RELATORIO DE PROCESSAMENTO DA ROTULAGEM ")
        print("="*50)
        print(f"Tempo Total:         {total_time:.2f} segundos")
        print(f"Total Processadas:   {total_images}")
        print(f"Erros de Leitura/Proc: {errors}")
        print("-" * 50)
        
        print("Contagem por Classe:")
        for cls in ["CRITICA", "SEMI_CRITICA", "NAO_CRITICA", "DESCARTADA"]:
            count = class_counts.get(cls, 0)
            pct = (count / total_images) * 100 if total_images > 0 else 0
            print(f"  - {cls:<15}: {count:<6} ({pct:.1f}%)")
            
        # Método de classificação usado
        if "metodo_classificacao" in df.columns:
            method_counts = df["metodo_classificacao"].value_counts()
            print("-" * 50)
            print("Metodo de Classificacao:")
            for method, count in method_counts.items():
                pct = (count / total_images) * 100 if total_images > 0 else 0
                print(f"  - {method:<15}: {count:<6} ({pct:.1f}%)")
            
        # Distribuição de larguras
        # Apenas para imagens com fissura identificada (não descartadas)
        mask = ~df["classe"].isin(["DESCARTADA"])
        df_cracks = df[mask]
        
        print("-" * 50)
        print("Estatisticas de Largura (Fissuras detectadas):")
        if len(df_cracks) > 0:
            avg_width = df_cracks["largura_mediana_mm"].mean()
            med_width = df_cracks["largura_mediana_mm"].median()
            p25 = df_cracks["largura_mediana_mm"].quantile(0.25)
            p75 = df_cracks["largura_mediana_mm"].quantile(0.75)
            
            print(f"  Media:             {avg_width:.4f} mm")
            print(f"  Mediana:           {med_width:.4f} mm")
            print(f"  Percentil 25:      {p25:.4f} mm")
            print(f"  Percentil 75:      {p75:.4f} mm")
        else:
            print("  Nenhuma fissura valida detectada.")
            
        print("="*50)
        print(f"Resultados detalhados salvos em: {csv_path}")
        print(f"Log de processamento salvo em: {self.output_dir / 'processing.log'}")
