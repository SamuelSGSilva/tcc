import os
import cv2
import numpy as np
from skimage.filters import frangi
from src.preprocessing import CrackPreprocessor
from src.segmentation import CrackSegmentor

# Configurações Iniciais
os.makedirs("tcc_figuras_pipeline", exist_ok=True)
image_path = "dataset_rotulado/CRITICA/semi_critica_0011.jpg"

if not os.path.exists(image_path):
    print(f"Erro: Imagem {image_path} não encontrada! Usando fallback.")
    # Tenta puxar qualquer uma do SEMI_CRITICA
    images = os.listdir("dataset_rotulado/SEMI_CRITICA")
    if images:
        image_path = os.path.join("dataset_rotulado/SEMI_CRITICA", images[0])
    else:
        print("Nenhuma imagem para gerar o exemplo.")
        exit(1)

print(f"Gerando Pipeline Visual a partir de: {image_path}")

# Lendo Imagem e Redimensionando (para ficar bonito no artigo)
img_bgr = cv2.imread(image_path)
h, w = img_bgr.shape[:2]
if max(h, w) > 800:
    scale = 800 / max(h, w)
    img_bgr = cv2.resize(img_bgr, (int(w*scale), int(h*scale)))

cv2.imwrite("tcc_figuras_pipeline/0_Imagem_Original.jpg", img_bgr)

# 1. Pŕe-Processamento (CLAHE/Cinza)
pre = CrackPreprocessor()
_, enhanced_img = pre.process(image_path)
enhanced_img = cv2.resize(enhanced_img, (img_bgr.shape[1], img_bgr.shape[0]))
cv2.imwrite("tcc_figuras_pipeline/1_Pre_Processada.jpg", enhanced_img)

# Iniciando as entranhas da nossa velha Segmentação (Passo a Passo)
seg = CrackSegmentor(merge_components=True, min_area_user=200, canny_thresh=20)
sigmas = seg._compute_adaptive_sigmas(enhanced_img.shape[0], enhanced_img.shape[1])

# A. Frangi (Assinatura Tubular)
img_float = enhanced_img.astype(np.float32) / 255.0
vesselness = frangi(img_float, sigmas=sigmas, black_ridges=True)
vesselness_norm = cv2.normalize(vesselness, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
_, thresh_frangi = cv2.threshold(vesselness_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# B. Canny (Bordas)
edges = cv2.Canny(enhanced_img, seg.canny_thresh, seg.canny_thresh * 3)
kernel_small = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
dilated_edges = cv2.dilate(edges, kernel_small, iterations=1)

# C. Intersecção dos Domínios (Frangi AND Canny)
intersection = cv2.bitwise_and(thresh_frangi, dilated_edges)

# E. Pós Processamento Morfológico (O Famoso Filtro LUT Refatorado)
intersection_clean = seg._remove_small_components(intersection, min_area=8)
combined = cv2.morphologyEx(intersection_clean, cv2.MORPH_CLOSE, kernel_small, iterations=2)
final_mask = seg._remove_small_components(combined, min_area=seg.min_area_user)

# Salvando a Arte Gráfica
cv2.imwrite("tcc_figuras_pipeline/2_Dominio_Tubular_Frangi.jpg", thresh_frangi)
cv2.imwrite("tcc_figuras_pipeline/3_Dominio_Bordas_Canny.jpg", dilated_edges)
cv2.imwrite("tcc_figuras_pipeline/4_Interseccao_Bruta.jpg", intersection)
cv2.imwrite("tcc_figuras_pipeline/5_Mascara_Limpa_Final.jpg", final_mask)

# Criando um Overlay Bonito ("Imagem Final da Rachadura em Azul")
overlay = img_bgr.copy()
overlay[final_mask > 0] = [255, 0, 0] # Fissura em Azul Puro
cv2.imwrite("tcc_figuras_pipeline/6_Overlay_Interface.jpg", overlay)

print("Imagens salvas com SUCESSO na pasta 'tcc_figuras_pipeline'!")
