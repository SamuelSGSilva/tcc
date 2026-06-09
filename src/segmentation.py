import cv2
import numpy as np
from skimage.filters import frangi

# Import opcional do U-Net — só falha em runtime se PyTorch não estiver instalado
try:
    from src.unet import UNetSegmentor
except ImportError:
    UNetSegmentor = None


class CrackSegmentor:
    """
    Segmentador de fissuras com parâmetros adaptativos.

    Melhorias:
      - Sigmas do Frangi adaptativos baseados na resolução da imagem
      - Suporte a múltiplos componentes conectados (não apenas o maior)
      - Thresholding adaptativo com parâmetros ajustados
      - Morfologia refinada para preservar fissuras finas
    """

    # Área mínima em pixels para considerar um componente como fissura (não ruído) (Legado)
    MIN_COMPONENT_AREA = 30

    def __init__(self, merge_components: bool = True, min_area_user: int = 200,
                 canny_thresh: int = 20, ignore_grid: bool = False,
                 max_brightness: int = 255, unet_segmentor=None,
                 filter_stains: bool = True):
        """
        Parâmetros
        ----------
        merge_components : bool
            Se True, combina todos os componentes detectados pelo Canny-Frangi.
        min_area_user : int
            Área mínima em pixels (usado pelo Canny-Frangi e pós-processamento U-Net).
        canny_thresh : int
            Limiar de detecção Canny (ignorado quando U-Net está ativo).
        ignore_grid : bool
            Se True, aplica morfologia direcional para apagar rejuntes (Canny-Frangi).
        max_brightness : int
            Valor máximo de brilho (0-255) para um pixel ser considerado fenda (Canny-Frangi).
        unet_segmentor : UNetSegmentor | None
            Instância do UNetSegmentor. Quando fornecida, o U-Net substitui o
            Canny-Frangi inteiramente. O pós-processamento (remoção de ruído por
            área) ainda é aplicado sobre a máscara U-Net.
        filter_stains : bool
            Se True, aplica filtro HSV para excluir pixels de ferrugem/enferrujamento
            (laranja, H=5-20) e algas/musgo (verde, H=35-85) antes da segmentação.
            Requer orig_bgr no método segment(). Default: True.
        """
        self.merge_components  = merge_components
        self.min_area_user     = min_area_user
        self.canny_thresh      = canny_thresh
        self.ignore_grid       = ignore_grid
        self.max_brightness    = max_brightness
        self.unet_segmentor    = unet_segmentor  # None = usa Canny-Frangi
        self.filter_stains     = filter_stains

    def _remove_small_components(self, binary_img, min_area):
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_img, connectivity=8)

        # Otimização: se só houver o background, retorna imagem preta
        if num_labels <= 1:
            return np.zeros_like(binary_img)

        # Otimização Vetorial Extrema (Lookup Table - LUT):
        # Em vez de iterar milhões de pixels num laço lento do Python (O(K*N)),
        # criamos um array 1D onde o indíce é o ID da mancha, e o valor é 0 ou 255.
        lut = np.zeros(num_labels, dtype=np.uint8)

        # As manchas que passarem na checagem de tamanho viram BRANCO (255)
        lut[stats[:, cv2.CC_STAT_AREA] >= min_area] = 255

        # O Background (ID=0) NUNCA é uma fissura, independente se a área dele passar ou não.
        lut[0] = 0

        # O Numpy substitui as ID's da imagem inteira pelo LUT instantaneamente em C++ (O(N)):
        mask = lut[labels]

        return mask

    def _filter_by_shape(self, binary_img: np.ndarray,
                          min_aspect_ratio: float = 2.5,
                          large_area_bypass: int = 4000) -> np.ndarray:
        """
        Remove componentes que não têm geometria de fissura.

        Fissuras são estruturas alongadas — o lado maior do bounding box é
        consideravelmente maior que o lado menor (aspect ratio alto).
        Ruídos típicos (manchas de reboco, poros, reflexos) são aproximadamente
        circulares ou quadrados (aspect ratio próximo de 1).

        Parâmetros
        ----------
        min_aspect_ratio : float
            Razão mínima entre o lado maior e o lado menor do bounding box
            para que o componente seja considerado uma fissura. Default: 2.5.
        large_area_bypass : int
            Componentes com área acima deste valor são sempre mantidos,
            independentemente do aspect ratio — fissuras grandes e ramificadas
            podem preencher o bounding box em múltiplas direções.
        """
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            binary_img, connectivity=8
        )
        if num_labels <= 1:
            return np.zeros_like(binary_img)

        lut = np.zeros(num_labels, dtype=np.uint8)
        for i in range(1, num_labels):
            w    = stats[i, cv2.CC_STAT_WIDTH]
            h    = stats[i, cv2.CC_STAT_HEIGHT]
            area = stats[i, cv2.CC_STAT_AREA]

            short_side = min(w, h)
            long_side  = max(w, h)
            aspect     = long_side / short_side if short_side > 0 else long_side

            if aspect >= min_aspect_ratio or area >= large_area_bypass:
                lut[i] = 255

        return lut[labels].astype(np.uint8)

    def segment(self, enhanced_img, orig_bgr: np.ndarray = None):
        """
        Segmenta a fissura.

        Quando `unet_segmentor` está configurado, usa U-Net (deep learning).
        Caso contrário, usa o pipeline clássico Frangi + Canny.

        Parâmetros
        ----------
        enhanced_img : np.ndarray
            Imagem pré-processada em escala de cinza (uint8).
        orig_bgr : np.ndarray, opcional
            Imagem BGR original. Necessária para inferência U-Net quando
            in_channels=3. Se None, o U-Net converte enhanced_img para BGR.
        """
        # ── Backend U-Net ──────────────────────────────────────────────────
        if self.unet_segmentor is not None:
            if orig_bgr is not None:
                input_img = orig_bgr
            else:
                # Converte grayscale → BGR para manter interface uniforme
                input_img = cv2.cvtColor(enhanced_img, cv2.COLOR_GRAY2BGR)

            raw_mask = self.unet_segmentor.segment(input_img)

            # Pós-processamento:
            # 1. MORPH_OPEN (erosão → dilatação) desconecta pequenas pontes de ruído
            #    que ligam manchas de reboco à fissura antes da filtragem por área.
            kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            raw_mask = cv2.morphologyEx(raw_mask, cv2.MORPH_OPEN, kernel_open, iterations=1)

            # 2. Remove fragmentos menores que min_area_user
            clean = self._remove_small_components(raw_mask, min_area=self.min_area_user)

            # 3. Filtra componentes que não têm geometria de fissura (blobs circulares)
            clean = self._filter_by_shape(clean)
            return clean

        # ── Backend Canny-Frangi (padrão) ──────────────────────────────────
        img_h, img_w = enhanced_img.shape[:2]
        img_area = img_h * img_w

        # 1. Sigmas adaptativos baseados na resolução
        sigmas = self._compute_adaptive_sigmas(img_h, img_w)

        # 2. Frangi Filter
        img_float = enhanced_img.astype(np.float32) / 255.0
        vesselness = frangi(img_float, sigmas=sigmas, black_ridges=True)

        # Normaliza vesselness
        vesselness_norm = cv2.normalize(vesselness, None, alpha=0, beta=255,
                                        norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        # 3. Thresholding Simples (Otsu) -> Separa o "tubular" do "não tubular"
        _, thresh_frangi = cv2.threshold(vesselness_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # 4. Canny Edges (Alta sensibilidade para fissuras sutis controlada pelo UI)
        edges = cv2.Canny(enhanced_img, self.canny_thresh, self.canny_thresh * 3)
        
        # Dilatamos as bordas discretamente
        kernel_small = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dilated_edges = cv2.dilate(edges, kernel_small, iterations=1)

        # 5. Intersecção de Domínios
        intersection = cv2.bitwise_and(thresh_frangi, dilated_edges)

        # 5.5 Filtro de Profundidade Fotométrica (Isolamento de Rachadura Escura)
        # Se a max_brightness for 255, este filtro é ignorado.
        # Se for menor (ex: 80), ele só aprova pixels extretamente escuros (fendas profundas).
        if self.max_brightness < 255:
            _, dark_mask = cv2.threshold(enhanced_img, self.max_brightness, 255, cv2.THRESH_BINARY_INV)
            intersection = cv2.bitwise_and(intersection, dark_mask)

        # 5.6 Filtro HSV — Exclusão de Manchas Coloridas (Ferrugem / Algas)
        # Pixels com alta saturação e tonalidade de ferrugem (laranja) ou algas (verde)
        # não são fissuras: são artefatos de degradação da superfície.
        # Só é aplicado se orig_bgr for fornecido e filter_stains=True.
        if self.filter_stains and orig_bgr is not None:
            hsv = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2HSV)
            # Ferrugem / enferrujamento: H=5-20, S>70, V>40
            rust_lo = np.array([5,  70,  40], dtype=np.uint8)
            rust_hi = np.array([20, 255, 255], dtype=np.uint8)
            rust_mask = cv2.inRange(hsv, rust_lo, rust_hi)
            # Algas / musgo: H=35-85, S>50, V>30
            algae_lo = np.array([35, 50,  30], dtype=np.uint8)
            algae_hi = np.array([85, 255, 255], dtype=np.uint8)
            algae_mask = cv2.inRange(hsv, algae_lo, algae_hi)
            # Une ambas as regiões a excluir e dilata levemente para cobrir bordas
            stain_mask = cv2.bitwise_or(rust_mask, algae_mask)
            kernel_stain = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            stain_mask = cv2.dilate(stain_mask, kernel_stain, iterations=1)
            # Remove pixels de mancha da intersecção
            intersection = cv2.bitwise_and(intersection, cv2.bitwise_not(stain_mask))

        # 6. FAXINA NÍVEL 1: Apagar Poeira do Reboco
        # Paredes com textura grossa formam milhares de pontinhos 1x1 soltos na intersecção.
        # Se fizermos o MORPH_CLOSE com eles ali, eles se fundem numa teia de aranha inquebrável.
        # Deleter qualquer sujeira menor que 10 pixels instantaneamente extingue a "estática".
        intersection_clean = self._remove_small_components(intersection, min_area=8)

        # 7. Conectividade da Fenda
        # Esticamos as pequenas interrupções geradas por imperfeições da própria fenda real
        # O `dilate` estúpido que ficava aqui foi removido para PARAR de engordar a fenda em +2 pixels (o que viciava a régua)
        combined = cv2.morphologyEx(intersection_clean, cv2.MORPH_CLOSE, kernel_small, iterations=2)

        # 7.5. Filtro Direcional Ortagnal para Paredes de Alvenaria (Rejuntes/Grid)
        if self.ignore_grid:
            # Detecta linhas perfeitamente horizontais longas (como rejuntes de tijolos)
            kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
            mask_h = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel_h)
            
            # Detecta linhas perfeitamente verticais longas
            kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
            mask_v = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel_v)
            
            # Subtrai o grid (rejuntes) da máscara da fenda
            grid_mask = cv2.bitwise_or(mask_h, mask_v)
            # Dilata o grid um pouco pra garantir um corte limpo
            grid_mask = cv2.dilate(grid_mask, kernel_small, iterations=1)
            combined = cv2.bitwise_and(combined, cv2.bitwise_not(grid_mask))

        # 8. FAXINA NÍVEL 2: Remoção de Ruído Avançada (Controlada pelo Usuário)
        final_mask = self._remove_small_components(combined, min_area=self.min_area_user)

        # 9. Filtra componentes com geometria de ruído (blobs circulares/quadrados)
        final_mask = self._filter_by_shape(final_mask)

        if not self.merge_components:
            # Comportamento antigo: reter apenas a maior fissura
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(final_mask, connectivity=8)
            if num_labels <= 1:
                return np.zeros_like(final_mask)
            largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            strict_mask = np.zeros_like(final_mask)
            strict_mask[labels == largest_label] = 255
            return strict_mask

        return final_mask

    def _compute_adaptive_sigmas(self, img_h: int, img_w: int) -> list:
        """
        Calcula sigmas adaptativos baseados na resolução da imagem.
        Imagens maiores podem ter fissuras mais largas, então usamos sigmas maiores.
        """
        min_dim = min(img_h, img_w)

        if min_dim <= 256:
            return [0.3, 0.5, 0.8, 1.0, 1.5]
        elif min_dim <= 512:
            return [0.5, 1.0, 1.5, 2.0, 2.5]
        elif min_dim <= 1024:
            return [0.5, 1.0, 1.5, 2.0, 3.0, 4.0]
        else:
            return [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]
