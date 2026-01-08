from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

# Directory Constants
VIS_DIR = PROJECT_ROOT / "visualizations"
CACHE_DIR = PROJECT_ROOT / "cache"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
DATASET_NAME = "ZStack_HE"  # Default fallback


def get_vsi_raw_dir(dataset_name: str = DATASET_NAME) -> Path:
    return Path(f"/home/niklas/{dataset_name}/raws")


def get_vsi_zip_dir(dataset_name: str = DATASET_NAME) -> Path:
    return Path(f"/home/niklas/{dataset_name}/zips")


def get_split_path(dataset_name: str = DATASET_NAME) -> Path:
    return PROJECT_ROOT / f"splits_{dataset_name}.json"


def get_index_path(mode: str, dataset_name: str = DATASET_NAME) -> Path:
    """Path to the index file for a given split (train/val/test) and dataset."""
    return CACHE_DIR / f"dataset_index_{dataset_name}_{mode}.pkl"


# Visualization Flags
GENERATE_VISUALIZATIONS = True
