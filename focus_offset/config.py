from dataclasses import dataclass
from pathlib import Path

# NOTE: SyntheticConfig moved to synthetic_data.config for centralization.
# Keeping it here for backward compatibility if needed, but unused in this file.

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class TrainConfig:
    """Hyperparameters for model training."""

    batch_size: int = 512
    num_workers: int = 16
    prefetch_factor: int = 1
    max_epochs: int = 50
    patience: int = 10
    learning_rate: float = 5e-5
    weight_decay: float = 0.001
    log_every_n_steps: int = 100
