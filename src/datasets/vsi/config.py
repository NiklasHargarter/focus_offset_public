import os
from pathlib import Path

# Layout Constants
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = Path(os.environ.get("DATA_ROOT", "/data/niklas"))
CACHE_ROOT = PROJECT_ROOT / "cache"
SPLITS_ROOT = PROJECT_ROOT / "splits"
