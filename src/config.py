import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = Path(os.environ.get("DATA_ROOT", "/data/niklas"))
CACHE_DIR = PROJECT_ROOT / "cache"


@dataclass
class TrainConfig:
    """Hyperparameters for model training."""

    batch_size: int = 512
    num_workers: int = os.cpu_count() or 4
    prefetch_factor: int = 1
    max_epochs: int = 5
    patience: int = 5
    learning_rate: float = 1e-4
    weight_decay: float = 0.05
    log_every_n_steps: int = 100
