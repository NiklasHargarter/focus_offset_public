import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Data Root: Defaults to Home, but can be overridden via environment variable
DATA_ROOT = Path(os.environ.get("DATA_ROOT", Path.home()))

# ---------------------------------------------------------------------------
# Directory Constants
# ---------------------------------------------------------------------------
CACHE_DIR = PROJECT_ROOT / "cache"
DATASET_NAME = "ZStack_HE"  # Default fallback

# ---------------------------------------------------------------------------
# Preprocessing Defaults
# ---------------------------------------------------------------------------
PATCH_SIZE = 224  # Fixed for ViT — not configurable
STRIDE = 448
DOWNSAMPLE_FACTOR = 1
MIN_TISSUE_COVERAGE = 0.05
MASK_DOWNSCALE = 8
EXCLUDE_PATTERN = "_all_"  # Skip slides whose name contains this substring

# ---------------------------------------------------------------------------
# Training Defaults
# ---------------------------------------------------------------------------
BATCH_SIZE = 512
NUM_WORKERS = 24
PREFETCH_FACTOR = 1
MAX_EPOCHS = 50
PATIENCE = 5
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.05
LOG_EVERY_N_STEPS = 100


def get_vsi_raw_dir(dataset_name: str = DATASET_NAME) -> Path:
    return DATA_ROOT / dataset_name / "raws"


def get_vsi_zip_dir(dataset_name: str = DATASET_NAME) -> Path:
    return DATA_ROOT / dataset_name / "zips"


def get_split_path(dataset_name: str = DATASET_NAME) -> Path:
    path = PROJECT_ROOT / "splits" / f"splits_{dataset_name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _run_dir(
    dataset_name: str,
    stride: int,
    downsample_factor: int,
    min_tissue_coverage: float,
) -> Path:
    """Cache subdirectory that encodes all variable preprocessing settings.

    Layout: cache/{dataset}/s{stride}_ds{downsample}_cov{coverage}/
    Patch size is always 224 (ViT) and not encoded in the path.
    """
    cov_str = f"{min_tissue_coverage:.2f}".replace(".", "")
    folder = f"s{stride}_ds{downsample_factor}_cov{cov_str}"
    return CACHE_DIR / dataset_name / folder


def get_master_index_path(
    dataset_name: str,
    stride: int,
    downsample_factor: int,
    min_tissue_coverage: float,
) -> Path:
    """Path to the master manifest file for a dataset run."""
    path = _run_dir(dataset_name, stride, downsample_factor, min_tissue_coverage) / "manifest.pkl"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def get_slide_index_dir(
    dataset_name: str,
    stride: int,
    downsample_factor: int,
    min_tissue_coverage: float,
) -> Path:
    """Directory for individual slide metadata pickles."""
    path = _run_dir(dataset_name, stride, downsample_factor, min_tissue_coverage) / "indices"
    path.mkdir(parents=True, exist_ok=True)
    return path
