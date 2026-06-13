import json

import numpy as np
import streamlit as st

from app_paths import METRICS_PATH


@st.cache_data(show_spinner=False)
def load_metrics() -> dict:
    if not METRICS_PATH.exists():
        return {}
    return json.loads(METRICS_PATH.read_text(encoding="utf-8"))


def format_number(value, precision: int = 4) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        if np.isnan(value):
            return "nan"
        return f"{value:.{precision}f}"
    return str(value)


def metric_card(label: str, value, precision: int = 4) -> None:
    st.metric(label, format_number(value, precision=precision))
