import cv2
import numpy as np
from skimage.morphology import skeletonize


class WidthEstimator:
    """
    Estimador de largura de fissura com distância transform refinada.

    Melhorias:
      - Usa mediana do distance transform ao invés de apenas o esqueleto
      - Filtra outliers na estimativa de largura
      - Retorna percentis além de mediana/min/max/std
    """

    def __init__(self):
        pass

    def estimate(self, binary_mask):
        """
        Estima a largura da fissura em pixels ao longo do esqueleto.

        Retorna
        -------
        tuple
            (mediana, min, max, std, num_pixels, esqueleto_mask)
        """
        if cv2.countNonZero(binary_mask) == 0:
            return 0.0, 0.0, 0.0, 0.0, 0, np.zeros_like(binary_mask)

        # 1. Distance Transform (L2, mask=5 para precisão)
        dist_transform = cv2.distanceTransform(binary_mask, cv2.DIST_L2, 5)

        # 2. Esqueleto (skimage requer boolean)
        skel = skeletonize(binary_mask > 0)

        # 3. Coleta larguras ao longo do esqueleto
        # Largura = 2 * distância do centro até a borda
        widths_px = dist_transform[skel] * 2.0

        num_pixels = np.count_nonzero(binary_mask)

        if len(widths_px) == 0:
            return 0.0, 0.0, 0.0, 0.0, num_pixels, skel.astype(np.uint8) * 255

        # 4. Filtra outliers (remove valores abaixo de 0.5px que são artefatos)
        widths_filtered = widths_px[widths_px >= 0.5]

        if len(widths_filtered) == 0:
            widths_filtered = widths_px

        median_w = float(np.median(widths_filtered))
        min_w = float(np.min(widths_filtered))
        max_w = float(np.percentile(widths_filtered, 95))  # P95 para ignorar outliers máximos
        std_w = float(np.std(widths_filtered))

        return median_w, min_w, max_w, std_w, num_pixels, skel.astype(np.uint8) * 255
