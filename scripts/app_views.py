import streamlit as st
import cv2

from app_data import metric_card
from app_image import (
    apply_clahe_rgb,
    apply_gaussian_denoise,
    crop_box_with_padding,
    detect_faces,
    read_uploaded_image,
    scenario_uses_crop,
    scenario_uses_enhancement,
)
from app_paths import IMAGE_SIZE
from app_models import (
    predict_classical_processed,
    predict_cnn_processed,
    predict_with_classical,
    predict_with_cnn,
)
from app_paths import FIGURES_DIR, METRICS_DIR, MODELS_DIR, RESULTS_DIR, figure_path


def format_cell(value, digits: int = 4):
    if value is None:
        return "-"
    if isinstance(value, float):
        if value != value:
            return "nan"
        return f"{value:.{digits}f}"
    if isinstance(value, bool):
        return "Ya" if value else "Tidak"
    return str(value)


def build_scenario_table(summary_rows: list[dict], scenario_results: list[dict]) -> list[dict]:
    result_by_scenario = {row.get("scenario"): row for row in scenario_results}
    purpose_by_scenario = {
        "A_hog_svm_plain_full": "Baseline HOG-SVM tanpa enhancement",
        "B_hog_svm_enhanced_full": "HOG-SVM dengan enhancement CLAHE dan denoise",
    }
    table_rows = []

    for row in summary_rows:
        scenario = row.get("scenario", "-")
        config = row.get("config", {}) or {}
        result = result_by_scenario.get(scenario, {})
        table_rows.append(
            {
                "scenario": scenario,
                "purpose": row.get("purpose") or result.get("purpose") or purpose_by_scenario.get(scenario, "-"),
                "model_type": row.get("model_type", "-"),
                "accuracy": format_cell(row.get("accuracy")),
                "precision": format_cell(row.get("precision")),
                "recall": format_cell(row.get("recall")),
                "f1_score": format_cell(row.get("f1_score")),
                "auc": format_cell(row.get("auc")),
                "epochs": format_cell(row.get("epochs"), digits=0),
                "val_accuracy": format_cell(row.get("best_val_accuracy") or result.get("validation_accuracy")),
                "val_loss": format_cell(row.get("best_val_loss")),
                "enhancement": format_cell(config.get("enhancement", "enhanced" in scenario)),
                "augmentation": format_cell(config.get("augmentation", "-")),
                "use_se": format_cell(config.get("use_se", "-")),
                "optimizer": format_cell(config.get("optimizer", "-")),
                "learning_rate": format_cell(config.get("learning_rate", "-")),
                "batch_size": format_cell(config.get("batch_size", "-")),
            }
        )

    return table_rows


def render_overview(metrics: dict) -> None:
    class_counts = metrics.get("class_counts", {})
    split_sizes = metrics.get("split_sizes", {})

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        metric_card("Total Images", sum(class_counts.values()) if class_counts else "-")
    with col2:
        metric_card("Train", split_sizes.get("train", "-"))
    with col3:
        metric_card("Validation", split_sizes.get("validation", "-"))
    with col4:
        metric_card("Test", split_sizes.get("test", "-"))
    with col5:
        metric_card("Max Epochs", metrics.get("max_epochs", "-"), precision=0)

    col1, col2 = st.columns(2)
    col1.info(f"Best CNN: **{metrics.get('best_cnn_scenario', '-')}**")
    col2.info(f"Best test F1: **{metrics.get('best_test_scenario_for_reporting', '-')}**")

    st.subheader("Class Distribution")
    class_fig = figure_path("class_distribution.png")
    if class_fig.exists():
        st.image(str(class_fig), width="stretch")
    else:
        st.dataframe([{"class": key, "count": value} for key, value in class_counts.items()], hide_index=True)

    comparison_fig = figure_path("scenario_comparison.png")
    if comparison_fig.exists():
        st.subheader("Scenario Comparison")
        st.image(str(comparison_fig), width="stretch")


def render_scenarios(metrics: dict) -> None:
    summary_rows = metrics.get("summary_rows", [])
    scenario_results = metrics.get("scenario_results", [])
    st.subheader("Scenario Summary")
    scenario_table = build_scenario_table(summary_rows, scenario_results)
    st.dataframe(scenario_table, width="stretch", height=460, hide_index=True)

    scenario_names = [row["scenario"] for row in summary_rows]
    selected = st.selectbox("Scenario", scenario_names, index=0 if scenario_names else None)
    selected_row = next((row for row in summary_rows if row["scenario"] == selected), {})
    selected_result = next((row for row in scenario_results if row["scenario"] == selected), {})

    cols = st.columns(5)
    for col, key in zip(cols, ["accuracy", "precision", "recall", "f1_score", "auc"]):
        with col:
            metric_card(key.replace("_", " ").title(), selected_row.get(key))

    col1, col2 = st.columns(2)
    cm = figure_path(f"confusion_matrix_{selected}.png")
    roc = figure_path(f"roc_curve_{selected}.png")
    history = figure_path(f"training_history_{selected}.png")
    if cm.exists():
        col1.image(str(cm), caption="Confusion Matrix", width="stretch")
    if roc.exists():
        col2.image(str(roc), caption="ROC Curve", width="stretch")
    if history.exists():
        st.image(str(history), caption="Training History Accuracy dan Loss", width="stretch")

    with st.expander("Config dan classification report"):
        st.json(
            {
                "purpose": selected_row.get("purpose", ""),
                "config": selected_row.get("config", {}),
                "classification_report": selected_result.get("classification_report", {}),
            }
        )


def render_figures() -> None:
    figure_groups = {
        "EDA": [
            "eda_class_distribution.png",
            "eda_original_samples.png",
            "eda_resolution_aspect_ratio.png",
            "eda_brightness_contrast.png",
            "eda_noise_blur.png",
            "eda_edge_hog_examples.png",
            "eda_pose_background_variation.png",
            "eda_face_detection_coverage.png",
            "eda_face_detection_examples.png",
            "eda_class_split_distribution.png",
        ],
        "Preprocessing": ["preprocessing_examples.png"],
        "Evaluation": ["scenario_comparison.png", "sample_predictions_by_model.png"],
    }
    group = st.segmented_control("Figure group", list(figure_groups), default="EDA")
    available = [name for name in figure_groups[group] if figure_path(name).exists()]
    if not available:
        st.info("Belum ada figure untuk group ini.")
    else:
        for name in available:
            st.image(str(figure_path(name)), caption=name, width="stretch")


def render_predict(metrics: dict, model_options: list[dict]) -> None:
    st.subheader("Image Prediction")
    if not model_options:
        st.info("Belum ada model di results/models/.")
        return

    labels = [option["label"] for option in model_options]
    default_idx = next(
        (
            idx
            for idx, option in enumerate(model_options)
            if option["available"] and option["path"].name == "face_mask_custom_cnn_from_scratch_best.keras"
        ),
        next((idx for idx, option in enumerate(model_options) if option["available"]), 0),
    )
    selected_label = st.selectbox("Model", labels, index=default_idx)
    selected_model = model_options[labels.index(selected_label)]

    st.caption(f"Artifact: {selected_model['path'].name}")
    if not selected_model["available"]:
        st.warning(
            "Model CNN `.keras` membutuhkan TensorFlow di environment lokal. "
            "Pilih model HOG-SVM `.joblib`, atau jalankan Streamlit dengan environment Python yang sudah punya TensorFlow."
        )
        return

    uploaded_file = st.file_uploader("Upload gambar wajah", type=["jpg", "jpeg", "png", "bmp", "webp"])

    if uploaded_file is None:
        return

    try:
        image_rgb = read_uploaded_image(uploaded_file)
        faces = detect_faces(image_rgb)
        if faces:
            render_single_face_prediction(image_rgb, selected_model, faces=faces)
            return

        if selected_model["kind"] == "cnn":
            prediction = predict_with_cnn(selected_model["path"], image_rgb, selected_model["scenario"])
        else:
            prediction = predict_with_classical(selected_model["path"], image_rgb, selected_model["scenario"])

        col1, col2, col3 = st.columns([1, 1, 1])
        col1.image(image_rgb, caption="Original", width="stretch")
        col2.image(prediction["processed"], caption="Input Model", width="stretch")
        with col3:
            st.metric("Prediction", prediction["label"])
            st.metric("Confidence", f"{prediction['confidence']:.2%}")
            st.progress(prediction["score_without_mask"], text=f"Probability without_mask: {prediction['score_without_mask']:.2%}")
    except Exception as error:
        st.error(f"Prediksi gagal: {error}")


def prepare_face_crop(face_rgb, scenario_id: str):
    processed = face_rgb.copy()
    if scenario_uses_enhancement(scenario_id):
        processed = apply_clahe_rgb(processed)
        processed = apply_gaussian_denoise(processed)
    return cv2.resize(processed, IMAGE_SIZE, interpolation=cv2.INTER_AREA)


def render_single_face_prediction(image_rgb, selected_model: dict, faces=None) -> None:
    faces = faces if faces is not None else detect_faces(image_rgb)
    face_box = max(faces, key=lambda box: box[2] * box[3])
    face_rgb = crop_box_with_padding(image_rgb, face_box)
    processed = prepare_face_crop(face_rgb, selected_model["scenario"])

    if selected_model["kind"] == "cnn":
        prediction = predict_cnn_processed(selected_model["path"], processed)
    else:
        prediction = predict_classical_processed(selected_model["path"], processed)

    annotated = image_rgb.copy()
    x, y, w, h = face_box
    color = (39, 174, 96) if prediction["label"] == "with_mask" else (235, 87, 87)
    cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 3)
    label = f"{prediction['label']} {prediction['confidence']:.0%}"
    cv2.putText(annotated, label, (x, max(20, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)

    col1, col2, col3 = st.columns([1, 1, 1])
    col1.image(annotated, caption="Detected Face", width="stretch")
    col2.image(processed, caption="Input Model", width="stretch")
    with col3:
        st.metric("Prediction", prediction["label"])
        st.metric("Confidence", f"{prediction['confidence']:.2%}")
        st.progress(prediction["score_without_mask"], text=f"Probability without_mask: {prediction['score_without_mask']:.2%}")


def artifact_scenario_name(path_name: str) -> str | None:
    for prefix in ["confusion_matrix_", "roc_curve_", "training_history_", "best_", "model_"]:
        if path_name.startswith(prefix):
            stem = path_name.removeprefix(prefix).rsplit(".", 1)[0]
            return stem
    return None


def render_artifacts(metrics: dict) -> None:
    st.subheader("Generated Artifacts")
    allowed_scenarios = {row.get("scenario") for row in metrics.get("summary_rows", [])}
    rows = []
    for folder in [FIGURES_DIR, MODELS_DIR, METRICS_DIR]:
        if not folder.exists():
            continue
        for path in sorted(folder.glob("*")):
            if path.is_file() and path.name != ".gitkeep":
                scenario = artifact_scenario_name(path.name)
                if scenario and allowed_scenarios and scenario not in allowed_scenarios:
                    continue
                rows.append(
                    {
                        "folder": path.parent.name,
                        "file": path.name,
                        "size_kb": round(path.stat().st_size / 1024, 1),
                    }
                )
    st.dataframe(rows, width="stretch", hide_index=True)
