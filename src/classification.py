class CriticalityClassifier:
    """
    Classificador de criticidade de fissuras baseado em limiares físicos (largura em mm).

    Limiares alinhados com a ABNT NBR 6118:2014 (Tabela 13.3):
        DESCARTADA   : largura < 0,01 mm  (ruído / sem fissura relevante)
        NAO_CRITICA  : 0,01 mm <= largura < 0,20 mm
        SEMI_CRITICA : 0,20 mm <= largura < 1,00 mm
        CRITICA      : largura >= 1,00 mm

    A confiança retornada por confidence() reflete a distância relativa do valor
    medido ao limiar de classe mais próximo, normalizada ao intervalo [0,30, 0,90].
    Essa métrica é usada como sinal de qualidade quando o ResNet50 opera em modo
    fallback (confiança abaixo do CONFIDENCE_THRESHOLD).
    """

    def __init__(self, scale_factor=0.1):
        # 1 pixel = scale_factor mm (padrão de fallback sem calibração ArUco)
        self.scale_factor = scale_factor

        # Limiares em mm
        self.THRESH_CRITICA          = 1.00
        self.THRESH_SEMI_CRITICA_MIN = 0.20
        self.THRESH_NAO_CRITICA_MIN  = 0.01

    def classify(self, width_px):
        """
        Converte largura em pixels para mm e retorna a classe de criticidade.

        Parâmetros
        ----------
        width_px : float
            Largura mediana da fissura em pixels (medida pela transformada de distância).

        Retorna
        -------
        tuple[str, float]
            (classe: str, largura_mm: float)
        """
        width_mm = round(width_px * self.scale_factor, 4)

        if width_mm < self.THRESH_NAO_CRITICA_MIN:
            return "DESCARTADA", width_mm
        elif width_mm < self.THRESH_SEMI_CRITICA_MIN:   # 0,01 <= w < 0,20 → NAO_CRITICA
            return "NAO_CRITICA", width_mm
        elif width_mm < self.THRESH_CRITICA:             # 0,20 <= w < 1,00 → SEMI_CRITICA
            return "SEMI_CRITICA", width_mm
        else:                                            # w >= 1,00 → CRITICA
            return "CRITICA", width_mm

    def confidence(self, width_mm: float) -> float:
        """
        Estima a confiança da classificação por limiares com base na distância
        relativa ao limiar de classe mais próximo.

        Recebe diretamente a largura em mm (não em pixels).  Quanto mais próximo
        o valor estiver de um limiar de fronteira, maior a ambiguidade e menor a
        confiança. Valores no interior da faixa de classe recebem confiança mais alta.

        Mapeamento por faixa:
            DESCARTADA  (< 0,01 mm) : confiança fixa 0,90 — ausência inequívoca
            NAO_CRITICA (0,01–0,20 mm): escala linear 0,30–0,70 pela distância ao limiar
            SEMI_CRITICA(0,20–1,00 mm): escala linear 0,45–0,75 pelo centro da faixa (0,60 mm)
            CRITICA     (>= 1,00 mm) : escala linear 0,70–0,85 pela distância acima do limiar

        Retorna
        -------
        float
            Confiança no intervalo [0,30, 0,90].
        """
        import numpy as np

        if width_mm < self.THRESH_NAO_CRITICA_MIN:
            # DESCARTADA: ausência de fissura é inequívoca
            conf = 0.90
        elif width_mm < self.THRESH_SEMI_CRITICA_MIN:
            # NAO_CRITICA: confiança cresce conforme afasta do limiar inferior (0,01 mm)
            conf = 0.30 + 0.40 * min(1.0, (width_mm - 0.01) / 0.19)
        elif width_mm < self.THRESH_CRITICA:
            # SEMI_CRITICA: máxima confiança no centro da faixa (0,60 mm); decai nas bordas
            dist_center = abs(width_mm - 0.60) / 0.40  # 0 no centro, 1 nas bordas
            conf = 0.75 - 0.30 * dist_center
        else:
            # CRITICA: confiança cresce conforme afasta do limiar (1,00 mm)
            conf = 0.70 + 0.15 * min(1.0, (width_mm - 1.00) / 0.50)

        return float(np.clip(conf, 0.30, 0.90))
