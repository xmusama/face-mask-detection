import sys
from pathlib import Path

import streamlit as st


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from app_data import load_metrics
from app_models import list_model_options
from app_views import render_artifacts, render_figures, render_overview, render_predict, render_scenarios


st.set_page_config(
    page_title="Face Mask Detection",
    page_icon="",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container { padding-top: 1.6rem; padding-bottom: 2rem; }
    div[data-testid="stMetric"] {
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 0.75rem 0.85rem;
        background: #ffffff;
        color: #111827;
    }
    div[data-testid="stMetric"] * { color: #111827 !important; }
    div[data-testid="stMetricLabel"] { font-size: 0.82rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


metrics = load_metrics()

st.title("Face Mask Detection")
st.caption("Dashboard eksperimen HOG-SVM dan CNN from scratch untuk klasifikasi penggunaan masker.")

if not metrics:
    st.warning("File metrics belum tersedia. Jalankan pipeline sampai menghasilkan results/metrics/metrics.json.")
    st.stop()

summary_rows = metrics.get("summary_rows", [])
model_options = list_model_options(metrics)

with st.sidebar:
    st.header("Run Info")
    st.write(f"TensorFlow: `{metrics.get('tensorflow_version', '-')}`")
    st.write(f"Max epochs: `{metrics.get('max_epochs', '-')}`")
    st.write(f"Early stopping patience: `{metrics.get('early_stop_patience', '-')}`")
    st.write(f"Scenarios: `{len(summary_rows)}`")

overview_tab, scenarios_tab, figures_tab, predict_tab, artifacts_tab = st.tabs(
    ["Overview", "Scenarios", "Figures", "Predict", "Artifacts"]
)

with overview_tab:
    render_overview(metrics)

with scenarios_tab:
    render_scenarios(metrics)

with figures_tab:
    render_figures()

with predict_tab:
    render_predict(metrics, model_options)

with artifacts_tab:
    render_artifacts(metrics)
