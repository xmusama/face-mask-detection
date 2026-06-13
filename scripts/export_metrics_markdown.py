import json
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
METRICS_JSON = ROOT_DIR / "results" / "metrics" / "metrics.json"
METRICS_MD = ROOT_DIR / "results" / "metrics" / "metrics.md"


def fmt(value, digits: int = 4) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend("| " + " | ".join(str(item) for item in row) + " |" for row in rows)
    return "\n".join(lines)


def main() -> None:
    metrics = json.loads(METRICS_JSON.read_text(encoding="utf-8"))
    purpose_by_scenario = {
        "A_hog_svm_plain_full": "Baseline HOG-SVM tanpa enhancement",
        "B_hog_svm_enhanced_full": "HOG-SVM dengan enhancement CLAHE dan denoise",
    }

    class_rows = [
        [class_name, count]
        for class_name, count in metrics.get("class_counts", {}).items()
    ]
    split_rows = [
        [split_name, size]
        for split_name, size in metrics.get("split_sizes", {}).items()
    ]
    scenario_rows = [
        [
            row.get("scenario", "-"),
            row.get("purpose") or purpose_by_scenario.get(row.get("scenario", ""), "-"),
            row.get("model_type", "-"),
            fmt(row.get("accuracy")),
            fmt(row.get("precision")),
            fmt(row.get("recall")),
            fmt(row.get("f1_score")),
            fmt(row.get("auc")),
            fmt(row.get("epochs"), 0),
        ]
        for row in metrics.get("summary_rows", [])
    ]

    output_lines = [
        "# Metrics Report",
        "",
        "## Ringkasan Run",
        "",
        table(
            ["Item", "Value"],
            [
                ["Dataset", metrics.get("dataset_handle", "-")],
                ["TensorFlow", metrics.get("tensorflow_version", "-")],
                ["Max epochs", metrics.get("max_epochs", "-")],
                ["Early stopping patience", metrics.get("early_stop_patience", "-")],
                ["Best CNN scenario", metrics.get("best_cnn_scenario", "-")],
                ["Best test F1 scenario", metrics.get("best_test_scenario_for_reporting", "-")],
            ],
        ),
        "",
        "## Distribusi Kelas",
        "",
        table(["Class", "Count"], class_rows),
        "",
        "## Split Data",
        "",
        table(["Split", "Size"], split_rows),
        "",
        "## Hasil Skenario",
        "",
        table(
            ["Scenario", "Purpose", "Model Type", "Accuracy", "Precision", "Recall", "F1-score", "AUC", "Epochs"],
            scenario_rows,
        ),
        "",
        "## Artefak Utama",
        "",
        table(
            ["Artifact", "Path"],
            [[key, value] for key, value in metrics.get("outputs", {}).items() if isinstance(value, str)],
        ),
        "",
    ]

    METRICS_MD.write_text("\n".join(output_lines), encoding="utf-8")
    print(METRICS_MD)


if __name__ == "__main__":
    main()
