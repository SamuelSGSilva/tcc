import os
import tempfile
import cv2
import numpy as np
import streamlit as st
from PIL import Image

from src.preprocessing import CrackPreprocessor
from src.segmentation import CrackSegmentor
from src.estimation import WidthEstimator
from src.classification import CriticalityClassifier
from src.calibration import ScaleCalibrator

try:
    from src.deep_classifier import DeepCrackClassifier
    DEEP_LEARNING_AVAILABLE = True
except ImportError:
    DeepCrackClassifier = None
    DEEP_LEARNING_AVAILABLE = False

try:
    from src.unet import UNetSegmentor
    UNET_AVAILABLE = True
except ImportError:
    UNetSegmentor = None
    UNET_AVAILABLE = False

DEFAULT_WEIGHTS_PATH  = os.path.join(os.path.dirname(__file__), "src", "model_weights", "resnet50_crack.pth")
DEFAULT_UNET_PATH     = os.path.join(os.path.dirname(__file__), "src", "model_weights", "unet_crack.pth")

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="FissuRA — Análise Estrutural",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  GLOBAL CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
/* ── Base ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: #0b0f19;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #10141f;
    border-right: 1px solid rgba(255,255,255,0.06);
}
[data-testid="stSidebar"] .stMarkdown p {
    color: #8b95a8;
    font-size: 0.82rem;
}

/* ── Hide default streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0b0f19; }
::-webkit-scrollbar-thumb { background: #2a3147; border-radius: 3px; }

/* ── Typography ── */
h1, h2, h3 { color: #f0f4ff !important; }

/* ── Divider ── */
hr { border-color: rgba(255,255,255,0.07) !important; margin: 1.2rem 0 !important; }

/* ── Buttons ── */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #3b82f6 0%, #6366f1 100%);
    border: none;
    border-radius: 10px;
    font-weight: 600;
    font-size: 0.95rem;
    letter-spacing: 0.3px;
    padding: 0.65rem 1.5rem;
    transition: all 0.2s ease;
    box-shadow: 0 4px 20px rgba(99, 102, 241, 0.35);
}
.stButton > button[kind="primary"]:hover {
    transform: translateY(-1px);
    box-shadow: 0 6px 28px rgba(99, 102, 241, 0.5);
}

/* ── Metric cards ── */
div[data-testid="stMetric"] {
    background: #151929;
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 12px;
    padding: 1rem 1.2rem;
}
div[data-testid="stMetricValue"] {
    font-size: 1.7rem !important;
    font-weight: 700 !important;
    color: #e2e8f0 !important;
}
div[data-testid="stMetricLabel"] {
    font-size: 0.75rem !important;
    color: #64748b !important;
    text-transform: uppercase;
    letter-spacing: 0.8px;
}
div[data-testid="stMetricDelta"] {
    font-size: 0.78rem !important;
}

/* ── Alerts ── */
.stAlert {
    border-radius: 10px !important;
    border: none !important;
}
div[data-testid="stAlert"] {
    border-radius: 10px;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: #151929;
    border: 1px dashed rgba(99, 102, 241, 0.4);
    border-radius: 12px;
    padding: 0.5rem;
    transition: border-color 0.2s;
}
[data-testid="stFileUploader"]:hover {
    border-color: rgba(99, 102, 241, 0.7);
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: #151929;
    border-radius: 10px;
    padding: 3px;
    gap: 2px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 7px;
    font-size: 0.82rem;
    font-weight: 500;
    color: #64748b;
    padding: 0.4rem 0.8rem;
}
.stTabs [aria-selected="true"] {
    background: #1e2847 !important;
    color: #818cf8 !important;
}

/* ── Expander ── */
.stExpander {
    background: #151929 !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
    border-radius: 10px !important;
}

/* ── Sliders ── */
[data-testid="stSlider"] [data-testid="stSliderThumb"] {
    background: #6366f1 !important;
}

/* ── Radio ── */
.stRadio [data-testid="stWidgetLabel"] { color: #94a3b8 !important; }
.stRadio label span { color: #cbd5e1 !important; font-size: 0.9rem; }

/* ── Select ── */
.stSelectbox label, .stNumberInput label, .stCheckbox label span {
    color: #94a3b8 !important;
    font-size: 0.85rem !important;
}

/* ── Image captions ── */
.stImage caption, .stImage figcaption {
    color: #64748b;
    font-size: 0.8rem;
}

/* ── Custom cards ── */
.card {
    background: #151929;
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 14px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 0.8rem;
}
.card-header {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 1px;
    text-transform: uppercase;
    color: #475569;
    margin-bottom: 0.5rem;
}

/* ── Status badge ── */
.badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 600;
}
.badge-green  { background: rgba(34,197,94,0.12);  color: #4ade80; border: 1px solid rgba(34,197,94,0.25); }
.badge-yellow { background: rgba(234,179,8,0.12);  color: #facc15; border: 1px solid rgba(234,179,8,0.25); }
.badge-red    { background: rgba(239,68,68,0.12);  color: #f87171; border: 1px solid rgba(239,68,68,0.25); }
.badge-blue   { background: rgba(99,102,241,0.12); color: #818cf8; border: 1px solid rgba(99,102,241,0.25); }
.badge-gray   { background: rgba(100,116,139,0.12); color: #94a3b8; border: 1px solid rgba(100,116,139,0.25); }

/* ── Diagnosis banner ── */
.diag-banner {
    border-radius: 14px;
    padding: 1.5rem 2rem;
    display: flex;
    align-items: center;
    gap: 1rem;
    margin: 1.2rem 0;
}
.diag-critica    { background: rgba(239,68,68,0.10); border: 1.5px solid rgba(239,68,68,0.35); }
.diag-semi       { background: rgba(234,179,8,0.10); border: 1.5px solid rgba(234,179,8,0.35); }
.diag-nao        { background: rgba(34,197,94,0.10); border: 1.5px solid rgba(34,197,94,0.35); }
.diag-descartada { background: rgba(100,116,139,0.10); border: 1.5px solid rgba(100,116,139,0.25); }
.diag-icon { font-size: 2.4rem; line-height: 1; }
.diag-title { font-size: 1.25rem; font-weight: 700; margin-bottom: 0.2rem; }
.diag-critica    .diag-title { color: #f87171; }
.diag-semi       .diag-title { color: #fbbf24; }
.diag-nao        .diag-title { color: #4ade80; }
.diag-descartada .diag-title { color: #94a3b8; }
.diag-desc { font-size: 0.9rem; color: #94a3b8; }

/* ── Step indicator ── */
.step-row { display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.3rem; }
.step-dot  { width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; }
.step-ok   { background: #4ade80; }
.step-run  { background: #6366f1; animation: pulse 1s infinite; }
.step-wait { background: #334155; }
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.4} }
.step-label { font-size: 0.82rem; color: #94a3b8; }

/* ── Section header ── */
.sec-header {
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: #475569;
    margin: 1.2rem 0 0.6rem;
}

/* ── Image label ── */
.img-label {
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.8px;
    text-transform: uppercase;
    color: #475569;
    margin-bottom: 0.4rem;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  CACHE
# ─────────────────────────────────────────────
@st.cache_resource
def load_processors():
    return CrackPreprocessor(), CrackSegmentor(), WidthEstimator()

@st.cache_resource
def load_deep_classifier():
    if not DEEP_LEARNING_AVAILABLE:
        return None
    if os.path.exists(DEFAULT_WEIGHTS_PATH):
        try:
            return DeepCrackClassifier(weights_path=DEFAULT_WEIGHTS_PATH)
        except Exception:
            return None
    return None

@st.cache_resource
def load_unet():
    if not UNET_AVAILABLE:
        return None
    if os.path.exists(DEFAULT_UNET_PATH):
        try:
            return UNetSegmentor(weights_path=DEFAULT_UNET_PATH)
        except Exception:
            return None
    return None

preprocessor, segmentor, estimator = load_processors()
deep_classifier = load_deep_classifier()
unet_segmentor  = load_unet()


# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    # Logo / título
    st.markdown("""
    <div style="padding: 0.6rem 0 1rem;">
        <div style="font-size:1.35rem; font-weight:800; color:#e2e8f0; letter-spacing:-0.5px;">
            🔬 FissuRA
        </div>
        <div style="font-size:0.78rem; color:#475569; margin-top:2px;">
            Análise Estrutural de Fissuras
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Status do motor de IA
    st.markdown('<div class="sec-header">Motor de Análise</div>', unsafe_allow_html=True)

    # Segmentação
    if unet_segmentor is not None:
        st.markdown('<span class="badge badge-green">● U-Net Ativo</span>', unsafe_allow_html=True)
        st.caption("Segmentação por rede neural profunda")
    elif UNET_AVAILABLE and not os.path.exists(DEFAULT_UNET_PATH):
        st.markdown('<span class="badge badge-yellow">● Canny-Frangi</span>', unsafe_allow_html=True)
        st.caption("Pesos U-Net não encontrados. Execute `python train_unet.py`.")
    else:
        st.markdown('<span class="badge badge-gray">● Canny-Frangi</span>', unsafe_allow_html=True)
        st.caption("Segmentação clássica por visão computacional")

    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

    # Classificação
    if deep_classifier is not None:
        st.markdown('<span class="badge badge-green">● ResNet50 Ativo</span>', unsafe_allow_html=True)
        st.caption("Classificação por rede neural fine-tuned")
    elif DEEP_LEARNING_AVAILABLE and not os.path.exists(DEFAULT_WEIGHTS_PATH):
        st.markdown('<span class="badge badge-yellow">● Régua Clássica</span>', unsafe_allow_html=True)
        st.caption("Pesos ResNet50 não encontrados. Execute `python train.py`.")
    else:
        st.markdown('<span class="badge badge-gray">● Régua Clássica</span>', unsafe_allow_html=True)
        st.caption("Classificação por limiares morfológicos")

    st.divider()

    # Abas de configuração
    st.markdown('<div class="sec-header">Configurações</div>', unsafe_allow_html=True)
    tab_escala, tab_cv, tab_seg = st.tabs(["📐 Escala", "👁️ Visão", "🛡️ Segurança"])

    with tab_escala:
        st.markdown("")
        enable_aruco = st.checkbox(
            "Auto-calibração ArUco",
            value=True,
            help="Detecta automaticamente o fator de escala usando um marcador ArUco na imagem.",
        )
        aruco_size_mm = st.number_input(
            "Lado do marcador (mm)",
            value=50.0, step=1.0, min_value=1.0,
            help="Tamanho físico do marcador ArUco impresso.",
        )
        st.divider()
        fallback_scale = st.number_input(
            "Escala manual (mm/px)",
            value=0.050, step=0.005, format="%.3f",
            help="Usado quando o ArUco não está presente na imagem.",
        )

    with tab_cv:
        st.markdown("")
        ignore_grid_ui = st.checkbox(
            "Ignorar rejuntes",
            value=False,
            help="Filtra linhas de rejunte em pisos e paredes.",
        )
        filter_stains_ui = st.checkbox(
            "Filtrar ferrugem / algas",
            value=True,
            help="Remove manchas laranja (ferrugem) e verdes (algas/musgo) antes da segmentação.",
        )
        max_brightness_ui = st.slider(
            "Brilho máximo (filtro 3D)",
            min_value=10, max_value=255, value=180, step=5,
            help="Remove regiões muito claras causadas por reflexos.",
        )
        min_area_ui = st.slider(
            "Área mínima (px²)",
            min_value=10, max_value=2000, value=600, step=10,
            help="Remove ruídos e partículas pequenas.",
        )
        canny_thresh_ui = st.slider(
            "Sensibilidade Canny",
            min_value=5, max_value=80, value=13, step=1,
            help="Quanto menor, mais bordas são detectadas.",
        )

    with tab_seg:
        st.markdown("")
        use_max_width_override = st.checkbox(
            "Safety override (pico máximo)",
            value=True,
            help="Se a largura máxima ≥ 1.0 mm, decreta CRÍTICA independentemente da mediana.",
        )

    st.divider()
    st.markdown("""
    <div style="font-size:0.72rem; color:#334155; line-height:1.6;">
        Baseado no dataset SDNET2018.<br>
        Pipeline: Canny-Frangi + ResNet50.
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  MAIN — HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div style="padding: 2rem 0 1rem;">
    <h1 style="font-size:2rem; font-weight:800; margin:0; background: linear-gradient(90deg,#818cf8,#38bdf8); -webkit-background-clip:text; -webkit-text-fill-color:transparent;">
        Diagnóstico de Fissuras Estruturais
    </h1>
    <p style="color:#475569; font-size:0.92rem; margin:0.4rem 0 0;">
        Visão computacional morfológica acoplada com ResNet50 fine-tuned (SDNET2018)
    </p>
</div>
""", unsafe_allow_html=True)

st.divider()

# ─────────────────────────────────────────────
#  MAIN — INPUT
# ─────────────────────────────────────────────
st.markdown('<div class="sec-header">1 · Enviar Imagem</div>', unsafe_allow_html=True)

input_col, info_col = st.columns([2, 1], gap="large")

with input_col:
    input_method = st.radio(
        "Método de captura",
        ["📁  Upload de arquivo", "📷  Câmera ao vivo"],
        horizontal=True,
        label_visibility="collapsed",
    )
    st.markdown("")

    uploaded_file = None
    if "Upload" in input_method:
        uploaded_file = st.file_uploader(
            "Arraste ou selecione uma imagem",
            type=["jpg", "jpeg", "png", "bmp"],
            label_visibility="collapsed",
        )
    else:
        uploaded_file = st.camera_input(
            "Tire uma foto da fissura",
            label_visibility="collapsed",
        )

with info_col:
    st.markdown("""
    <div class="card" style="margin-top:1.8rem;">
        <div class="card-header">Formatos aceitos</div>
        <div style="color:#cbd5e1; font-size:0.85rem; line-height:1.8;">
            JPG / JPEG / PNG / BMP<br>
            Resolução máxima processada: <strong style="color:#818cf8;">1 200 px</strong><br>
            Imagem colorida ou grayscale
        </div>
    </div>
    <div class="card">
        <div class="card-header">Dica de captura</div>
        <div style="color:#64748b; font-size:0.82rem; line-height:1.7;">
            • Iluminação uniforme e frontal<br>
            • Inclua o marcador ArUco para escala precisa<br>
            • Evite sombras fortes sobre a fissura
        </div>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  PROCESSAMENTO
# ─────────────────────────────────────────────
if uploaded_file is not None:

    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    orig_bgr_raw = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # Redimensiona se necessário
    MAX_DIM = 1200
    h_orig, w_orig = orig_bgr_raw.shape[:2]
    if max(h_orig, w_orig) > MAX_DIM:
        scale_down = MAX_DIM / max(h_orig, w_orig)
        orig_bgr_raw = cv2.resize(
            orig_bgr_raw,
            (int(w_orig * scale_down), int(h_orig * scale_down)),
            interpolation=cv2.INTER_AREA,
        )
        fallback_scale = fallback_scale / scale_down

    st.divider()

    # ── Recorte opcional ──
    with st.expander("✂️  Recorte de ROI  (opcional)", expanded=False):
        st.caption("Elimine bordas, janelas ou regiões que possam interferir na detecção.")
        rc1, rc2, rc3, rc4 = st.columns(4)
        crop_left   = rc1.slider("Esquerda (%)",  0, 45, 0, key="cl")
        crop_right  = rc2.slider("Direita (%)",   0, 45, 0, key="cr")
        crop_top    = rc3.slider("Topo (%)",       0, 45, 0, key="ct")
        crop_bottom = rc4.slider("Base (%)",       0, 45, 0, key="cb")

    h_raw, w_raw = orig_bgr_raw.shape[:2]
    orig_bgr = orig_bgr_raw[
        int(crop_top / 100.0 * h_raw): int((1 - crop_bottom / 100.0) * h_raw),
        int(crop_left / 100.0 * w_raw): int((1 - crop_right / 100.0) * w_raw),
    ]
    orig_rgb = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB)

    # ── Prévia + botão ──
    st.markdown('<div class="sec-header">2 · Pré-visualização</div>', unsafe_allow_html=True)

    prev_col, btn_col = st.columns([3, 1], gap="large")

    with prev_col:
        st.markdown('<div class="img-label">Imagem carregada</div>', unsafe_allow_html=True)
        st.image(orig_rgb, use_container_width=True)

    with btn_col:
        st.markdown("<div style='height:3rem'></div>", unsafe_allow_html=True)
        analyze_pressed = st.button(
            "🔬  Analisar",
            type="primary",
            use_container_width=True,
            help="Inicia o pipeline completo de diagnóstico.",
        )
        st.markdown("""
        <div style="font-size:0.78rem; color:#334155; margin-top:0.8rem; line-height:1.6; text-align:center;">
            Pré-processamento → Segmentação → Estimativa de largura → Classificação
        </div>
        """, unsafe_allow_html=True)

    # ─────────────────────────────────────────
    #  PIPELINE
    # ─────────────────────────────────────────
    if analyze_pressed:
        with st.spinner("Analisando fissura…"):
            try:
                # ── 1. Calibração ──
                current_scale = fallback_scale
                aruco_corners = None

                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                    cv2.imwrite(tmp.name, orig_bgr)
                    tmp_path = tmp.name

                _, enhanced_img = preprocessor.process(tmp_path)

                if enable_aruco:
                    calibrator = ScaleCalibrator(marker_size_mm=aruco_size_mm)
                    dynamic_scale, aruco_corners = calibrator.calibrate(orig_bgr)
                    if dynamic_scale is not None:
                        current_scale = dynamic_scale

                classifier = CriticalityClassifier(scale_factor=current_scale)

                # ── 2. Segmentação (U-Net ou Canny-Frangi) ──
                dynamic_segmentor = CrackSegmentor(
                    merge_components=True,
                    min_area_user=min_area_ui,
                    canny_thresh=canny_thresh_ui,
                    ignore_grid=ignore_grid_ui,
                    max_brightness=max_brightness_ui,
                    unet_segmentor=unet_segmentor,
                    filter_stains=filter_stains_ui,
                )
                binary_mask = dynamic_segmentor.segment(enhanced_img, orig_bgr=orig_bgr)

                # ── 3. Estimativa ──
                median_px, min_px, max_px, std_px, num_pixels, skel_mask = estimator.estimate(binary_mask)

                # ── 4. Classificação híbrida ──
                deep_confidence = None
                usou_ia = False
                deep_class_original = None

                thresh_class_median, median_mm = classifier.classify(median_px)
                max_width_mm = max_px * current_scale
                _, min_mm  = classifier.classify(min_px)
                _, max_mm  = classifier.classify(max_px)
                std_mm = std_px * current_scale

                thresh_class_base = "CRITICA" if (use_max_width_override and max_width_mm >= 1.0) else thresh_class_median

                if deep_classifier is not None:
                    crack_class, deep_confidence, usou_ia = deep_classifier.predict_with_fallback(
                        orig_bgr,
                        binary_mask,
                        threshold_class=(thresh_class_base, median_mm),
                        use_tta=True,
                    )
                    deep_class_original, _ = deep_classifier.predict(orig_bgr, binary_mask, use_tta=False)
                else:
                    crack_class = thresh_class_base

                # ── Overlay ──
                overlay_rgb = orig_rgb.copy()
                overlay_rgb[binary_mask > 0] = [30, 120, 255]
                overlay_rgb[skel_mask > 0]   = [255, 80, 60]
                if aruco_corners is not None:
                    corners_int = np.int32(aruco_corners).reshape(-1, 2)
                    cv2.polylines(overlay_rgb, [corners_int], True, (250, 230, 50), 2)

            except Exception as e:
                st.error(f"Erro durante o processamento: {e}")
                st.stop()

        # ─────────────────────────────────────
        #  RESULTADOS
        # ─────────────────────────────────────
        st.divider()
        st.markdown('<div class="sec-header">3 · Resultado</div>', unsafe_allow_html=True)

        # Imagens lado a lado
        img_a, img_b = st.columns(2, gap="medium")
        with img_a:
            st.markdown('<div class="img-label">Original</div>', unsafe_allow_html=True)
            st.image(orig_rgb, use_container_width=True)
        with img_b:
            st.markdown('<div class="img-label">Máscara de Segmentação</div>', unsafe_allow_html=True)
            st.image(overlay_rgb, use_container_width=True)
            if aruco_corners is not None:
                st.caption("🟡 ArUco detectado — escala calibrada automaticamente")

        st.markdown("")

        # ── Banner de diagnóstico ──
        _cls_map = {
            "CRITICA":     ("diag-critica",    "⚠️",  "CRÍTICA",      "Estrutura gravemente comprometida. Intervenção imediata recomendada."),
            "SEMI_CRITICA":("diag-semi",       "🚧",  "SEMI-CRÍTICA", "Fissuração moderada. Exige monitoramento e acompanhamento técnico."),
            "NAO_CRITICA": ("diag-nao",        "✅",  "NÃO-CRÍTICA",  "Fissura estética ou superficial. Sem risco estrutural imediato."),
            "DESCARTADA":  ("diag-descartada", "ℹ️",  "DESCARTADA",   "Nenhuma anomalia estrutural significativa detectada."),
        }
        cls_style, cls_icon, cls_label, cls_desc = _cls_map.get(
            crack_class, ("diag-descartada", "❓", crack_class, "")
        )
        st.markdown(f"""
        <div class="diag-banner {cls_style}">
            <div class="diag-icon">{cls_icon}</div>
            <div>
                <div style="font-size:0.7rem;font-weight:700;letter-spacing:1.2px;text-transform:uppercase;color:#475569;margin-bottom:3px;">Diagnóstico Final</div>
                <div class="diag-title">{cls_label}</div>
                <div class="diag-desc">{cls_desc}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Métricas ──
        st.markdown("")
        m1, m2, m3, m4, m5 = st.columns(5, gap="small")

        m1.metric("Largura Mediana", f"{median_mm:.3f} mm")
        m2.metric("Largura Máxima", f"{max_width_mm:.3f} mm")
        m3.metric(
            "Escala",
            f"{current_scale:.4f}",
            delta="ArUco" if aruco_corners is not None else "Manual",
        )
        if deep_classifier is not None and deep_confidence is not None:
            m4.metric("Confiança (IA)", f"{deep_confidence * 100:.1f}%")
            m5.metric(
                "Sugestão ResNet50",
                deep_class_original.replace("_", " ") if deep_class_original else "—",
            )
        else:
            m4.metric("Pixels Detectados", f"{num_pixels:,}")
            m5.metric("Desvio Padrão", f"{std_mm:.3f} mm")

        # ── Detalhamento técnico ──
        with st.expander("📋  Detalhamento técnico", expanded=False):
            det1, det2, det3 = st.columns(3, gap="medium")

            with det1:
                st.markdown("**Medições morfológicas**")
                st.markdown(f"""
                | Parâmetro | Valor |
                |-----------|-------|
                | Largura mínima | `{min_mm:.4f} mm` |
                | Largura mediana | `{median_mm:.4f} mm` |
                | Largura máxima | `{max_mm:.4f} mm` |
                | Desvio padrão | `{std_mm:.4f} mm` |
                """)

            with det2:
                st.markdown("**Segmentação**")
                st.markdown(f"""
                | Parâmetro | Valor |
                |-----------|-------|
                | Pixels de fissura | `{num_pixels:,}` |
                | Fator de escala | `{current_scale:.5f} mm/px` |
                | Calibração ArUco | `{"Sim" if aruco_corners is not None else "Não"}` |
                """)

            with det3:
                st.markdown("**Classificação**")
                metodo = "ResNet50" if usou_ia else "Limiares"
                st.markdown(f"""
                | Parâmetro | Valor |
                |-----------|-------|
                | Classe final | `{crack_class}` |
                | Método utilizado | `{metodo}` |
                | Classe (limiares) | `{thresh_class_base}` |
                | Classe (ResNet50) | `{deep_class_original or "N/A"}` |
                | Confiança (IA) | `{f"{deep_confidence*100:.1f}%" if deep_confidence is not None else "N/A"}` |
                """)

        # ── Nota sobre metodologia ──
        st.markdown("")
        if deep_classifier is not None:
            if usou_ia:
                method_note = "🤖 A **IA (ResNet50)** determinou a classe final com confiança acima do limiar configurado."
            elif thresh_class_base == "CRITICA" and thresh_class_median != "CRITICA":
                method_note = "🛡️ **Safety Override ativo:** pico morfológico acima de 1,0 mm forçou classificação como CRÍTICA."
            else:
                method_note = "📐 **Régua física:** IA abaixo do limiar de confiança; a medição morfológica em mm determinou a classe."
            st.caption(method_note)
        else:
            st.caption("📐 Classificação baseada exclusivamente em medição morfológica (visão computacional).")
