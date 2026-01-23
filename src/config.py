import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Data Root: Defaults to Home, but can be overridden via environment variable
DATA_ROOT = Path(os.environ.get("DATA_ROOT", Path.home()))

# Directory Constants
VIS_DIR = PROJECT_ROOT / "visualizations"
CACHE_DIR = PROJECT_ROOT / "cache"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
DATASET_NAME = "ZStack_HE"  # Default fallback
PATCH_SIZE = 224


def get_vsi_raw_dir(dataset_name: str = DATASET_NAME) -> Path:
    return DATA_ROOT / dataset_name / "raws"


def get_vis_dir(dataset_name: str = DATASET_NAME, patch_size: int = PATCH_SIZE) -> Path:
    """Path to the visualization directory for a given dataset and patch size."""
    path = VIS_DIR / f"p{patch_size}" / dataset_name
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_vsi_zip_dir(dataset_name: str = DATASET_NAME) -> Path:
    return DATA_ROOT / dataset_name / "zips"


def get_split_path(dataset_name: str = DATASET_NAME) -> Path:
    return PROJECT_ROOT / f"splits_{dataset_name}.json"


def get_index_path(
    mode: str, dataset_name: str = DATASET_NAME, patch_size: int = PATCH_SIZE
) -> Path:
    """Path to the index file for a given split (train/val/test) and dataset."""
    path = CACHE_DIR / f"p{patch_size}" / f"dataset_index_{dataset_name}_{mode}.pkl"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def get_master_index_path(
    dataset_name: str = DATASET_NAME, patch_size: int = PATCH_SIZE
) -> Path:
    """Path to the master manifest file for a dataset."""
    path = CACHE_DIR / f"p{patch_size}" / dataset_name / "manifest.pkl"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def get_slide_index_dir(
    dataset_name: str = DATASET_NAME, patch_size: int = PATCH_SIZE
) -> Path:
    """Directory for individual slide metadata pickles."""
    path = CACHE_DIR / f"p{patch_size}" / dataset_name / "indices"
    path.mkdir(parents=True, exist_ok=True)
    return path
