from pathlib import Path
import importlib.util

import numpy as np
import streamlit as st

from app_image import extract_hog_features, preprocess_image_for_scenario
from app_paths import CLASS_NAMES, MODELS_DIR


def tensorflow_available() -> bool:
    return importlib.util.find_spec("tensorflow") is not None


def classical_available() -> bool:
    return importlib.util.find_spec("joblib") is not None and importlib.util.find_spec("sklearn") is not None


def scenario_from_model_name(path: Path, best_scenario: str) -> str:
    if path.name == "face_mask_custom_cnn_from_scratch_best.keras":
        return best_scenario
    if path.name.startswith("best_") and path.suffix == ".keras":
        return path.stem.removeprefix("best_")
    if path.name.startswith("model_") and path.suffix == ".joblib":
        return path.stem.removeprefix("model_")
    return path.stem


def list_model_options(metrics: dict) -> list[dict]:
    best_scenario = metrics.get("best_cnn_scenario", "")
    has_tensorflow = tensorflow_available()
    has_classical_runtime = classical_available()
    options = []

    for path in sorted(MODELS_DIR.glob("*.keras")):
        scenario = scenario_from_model_name(path, best_scenario)
        label = f"{scenario} (.keras)"
        if path.name == "face_mask_custom_cnn_from_scratch_best.keras":
            label = f"{scenario} (.keras, selected checkpoint)"
        if not has_tensorflow:
            label = f"{label} - TensorFlow belum tersedia"
        options.append(
            {
                "label": label,
                "path": path,
                "kind": "cnn",
                "scenario": scenario,
                "available": has_tensorflow,
            }
        )

    for path in sorted(MODELS_DIR.glob("*.joblib")):
        scenario = scenario_from_model_name(path, best_scenario)
        label = f"{scenario} (.joblib)"
        if not has_classical_runtime:
            label = f"{label} - scikit-learn belum tersedia"
        options.append(
            {
                "label": label,
                "path": path,
                "kind": "classical",
                "scenario": scenario,
                "available": has_classical_runtime,
            }
        )

    return options


@st.cache_resource(show_spinner=False)
def load_keras_model(model_path: str):
    import tensorflow as tf

    return tf.keras.models.load_model(model_path)


@st.cache_resource(show_spinner=False)
def load_joblib_model(model_path: str):
    import joblib

    return joblib.load(model_path)


def predict_with_cnn(model_path: Path, image_rgb: np.ndarray, scenario_id: str) -> dict:
    processed, face_detected = preprocess_image_for_scenario(image_rgb, scenario_id)
    model_input = np.expand_dims(processed.astype(np.float32) / 255.0, axis=0)
    model = load_keras_model(str(model_path))
    score = float(model.predict(model_input, verbose=0).reshape(-1)[0])
    label_idx = int(score >= 0.5)
    confidence = score if label_idx == 1 else 1.0 - score
    return {
        "label": CLASS_NAMES[label_idx],
        "score_without_mask": score,
        "confidence": confidence,
        "processed": processed,
        "face_detected": face_detected,
    }


def predict_cnn_processed(model_path: Path, processed_rgb: np.ndarray) -> dict:
    model_input = np.expand_dims(processed_rgb.astype(np.float32) / 255.0, axis=0)
    model = load_keras_model(str(model_path))
    score = float(model.predict(model_input, verbose=0).reshape(-1)[0])
    label_idx = int(score >= 0.5)
    confidence = score if label_idx == 1 else 1.0 - score
    return {
        "label": CLASS_NAMES[label_idx],
        "score_without_mask": score,
        "confidence": confidence,
    }


def predict_with_classical(model_path: Path, image_rgb: np.ndarray, scenario_id: str) -> dict:
    processed, face_detected = preprocess_image_for_scenario(image_rgb, scenario_id)
    features = extract_hog_features(np.expand_dims(processed, axis=0))
    model = load_joblib_model(str(model_path))
    score = float(model.decision_function(features).reshape(-1)[0])
    probability = float(1.0 / (1.0 + np.exp(-np.clip(score, -30, 30))))
    label_idx = int(score >= 0.0)
    confidence = probability if label_idx == 1 else 1.0 - probability
    return {
        "label": CLASS_NAMES[label_idx],
        "score_without_mask": probability,
        "confidence": confidence,
        "processed": processed,
        "face_detected": face_detected,
    }


def predict_classical_processed(model_path: Path, processed_rgb: np.ndarray) -> dict:
    features = extract_hog_features(np.expand_dims(processed_rgb, axis=0))
    model = load_joblib_model(str(model_path))
    score = float(model.decision_function(features).reshape(-1)[0])
    probability = float(1.0 / (1.0 + np.exp(-np.clip(score, -30, 30))))
    label_idx = int(score >= 0.0)
    confidence = probability if label_idx == 1 else 1.0 - probability
    return {
        "label": CLASS_NAMES[label_idx],
        "score_without_mask": probability,
        "confidence": confidence,
    }
