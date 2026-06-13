from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
MODELS_DIR = RESULTS_DIR / "models"
METRICS_DIR = RESULTS_DIR / "metrics"
METRICS_PATH = METRICS_DIR / "metrics.json"

CLASS_NAMES = ["with_mask", "without_mask"]
IMAGE_SIZE = (128, 128)


def figure_path(name: str) -> Path:
    return FIGURES_DIR / name
