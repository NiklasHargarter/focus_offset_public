from pathlib import Path
from src.models.factory import ModelArch

# Global configuration for VSI processing
# Changing these values typically requires re-running precompute (src/preprocess.py)

PROJECT_ROOT = Path(__file__).resolve().parent

# Patch extraction settings
PATCH_SIZE = 512
STRIDE = 512

# Preprocessing settings
DOWNSCALE_FACTOR = 8  # Reduced to preserve detail
MIN_TISSUE_COVERAGE = 0.05  # Minimum tissue ratio to keep a patch

# IO Settings
VIS_DIR = PROJECT_ROOT / "visualizations"
CACHE_DIR = PROJECT_ROOT / "cache"
VSI_RAW_DIR = Path("/home/niklas/ZStack_HE/raws")  # Raw downloaded VSI files
VSI_ZIP_DIR = Path("/home/niklas/ZStack_HE/zips")  # Intermediate ZIP downloads

# Split Configuration
SPLIT_FILE = PROJECT_ROOT / "splits.json"

# Splitting Settings
SPLIT_RATIO = 0.2  # Fraction of data for testing
SEARCH_SEED = 42

# Default output pattern
INDEX_PREFIX = "dataset_index"

# Training Settings
BATCH_SIZE = 16
EPOCHS = 20
LEARNING_RATE = 1e-4
PATIENCE = 5
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
MODEL_ARCH = ModelArch.EFFICIENTNET_B0


def get_index_path(mode: str) -> Path:
    """Returns the absolute path to the index file for a given split mode (train/test)."""
    return CACHE_DIR / f"{INDEX_PREFIX}_{mode}.pkl"
