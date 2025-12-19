from pathlib import Path
from src.models.factory import ModelArch

PROJECT_ROOT = Path(__file__).resolve().parent

PATCH_SIZE = 224
STRIDE = 224

DOWNSCALE_FACTOR = 8
MIN_TISSUE_COVERAGE = 0.05

VIS_DIR = PROJECT_ROOT / "visualizations"
CACHE_DIR = PROJECT_ROOT / "cache"
VSI_RAW_DIR = Path("/home/niklas/ZStack_HE/raws")
VSI_ZIP_DIR = Path("/home/niklas/ZStack_HE/zips")
GENERATE_VISUALIZATIONS = True

SPLIT_FILE = PROJECT_ROOT / "splits.json"
SPLIT_RATIO = 0.1
VAL_RATIO = 0.1
SEARCH_SEED = 42

INDEX_PREFIX = "dataset_index"

BATCH_SIZE = 128
NUM_WORKERS = 16
EPOCHS = 50
LEARNING_RATE = 1e-4
PATIENCE = 5
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
MODEL_ARCH = ModelArch.VIT_B_16


def get_index_path(mode: str) -> Path:
    """Path to the index file for a given split (train/val/test)."""
    return CACHE_DIR / f"{INDEX_PREFIX}_{mode}.pkl"
