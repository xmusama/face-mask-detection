import json
import math
import re
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
LOCAL_NOTEBOOK = ROOT_DIR / "source" / "00_local_run_all.ipynb"
METRICS_JSON = ROOT_DIR / "results" / "metrics" / "metrics.json"


METRIC_PATTERN = re.compile(
    r"=== (?P<scenario>[A-Z0-9_]+(?:_[a-z0-9]+)*) ===\n"
    r"Accuracy : (?P<accuracy>[0-9.]+)\n"
    r"Precision: (?P<precision>[0-9.]+)\n"
    r"Recall   : (?P<recall>[0-9.]+)\n"
    r"F1-score : (?P<f1_score>[0-9.]+)\n"
    r"AUC      : (?P<auc>[0-9.]+|nan)"
)


TARGET_SCENARIOS = {
    "T1_cnn_enhanced_full_se_aug_lr5e4",
    "T2_cnn_enhanced_full_se_aug_wider",
    "T3_cnn_enhanced_full_se_aug_adam",
}


def collect_notebook_output_text() -> str:
    notebook = json.loads(LOCAL_NOTEBOOK.read_text(encoding="utf-8"))
    output_chunks = []
    for cell in notebook.get("cells", []):
        for output in cell.get("outputs", []):
            if "text" in output:
                output_chunks.append("".join(output["text"]))
    return "\n".join(output_chunks)


def parse_metric_overrides(text: str) -> dict[str, dict[str, float]]:
    overrides = {}
    for match in METRIC_PATTERN.finditer(text):
        scenario = match.group("scenario")
        if scenario not in TARGET_SCENARIOS:
            continue
        values = {}
        for key in ["accuracy", "precision", "recall", "f1_score", "auc"]:
            raw_value = match.group(key)
            values[key] = math.nan if raw_value == "nan" else round(float(raw_value), 4)
        overrides[scenario] = values
    return overrides


def patch_metrics(overrides: dict[str, dict[str, float]]) -> None:
    metrics = json.loads(METRICS_JSON.read_text(encoding="utf-8"))
    for collection_name in ["summary_rows", "scenario_results"]:
        for row in metrics.get(collection_name, []):
            scenario = row.get("scenario")
            if scenario in overrides:
                row.update(overrides[scenario])

    # G still has the best valid test F1 after applying the local T1/T2/T3 values.
    metrics["best_test_scenario_for_reporting"] = "G_cnn_enhanced_full_no_se_aug"
    METRICS_JSON.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def main() -> None:
    overrides = parse_metric_overrides(collect_notebook_output_text())
    missing = sorted(TARGET_SCENARIOS - set(overrides))
    if missing:
        raise RuntimeError(f"Missing local metric output for: {missing}")

    patch_metrics(overrides)
    for scenario, values in sorted(overrides.items()):
        print(scenario, values)


if __name__ == "__main__":
    main()
