import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = Path(os.environ.get("DATA_ROOT", Path.home()))
CACHE_DIR = PROJECT_ROOT / "cache"


@dataclass
class PrepConfig:
    """Settings used for extracting tiles from raw whole-slide images."""

    stride: int = 224
    downsample_factor: int = 1
    min_tissue_coverage: float = 0.80
    mask_downscale: int = 8
    patch_size: int = 224  # Fixed for network architectures


@dataclass
class TrainConfig:
    """Hyperparameters for model training."""

    batch_size: int = 512
    num_workers: int = 24
    prefetch_factor: int = 1
    max_epochs: int = 50
    patience: int = 5
    learning_rate: float = 1e-4
    weight_decay: float = 0.05
    log_every_n_steps: int = 100


@dataclass
class DatasetConfig:
    """Dataset-specific metadata and path resolution."""

    name: str = "ZStack_HE"
    exclude_pattern: str = "_all_"

    # Nested prep config
    prep: PrepConfig = field(default_factory=PrepConfig)

    @property
    def raw_dir(self) -> Path:
        """Generic replacement for get_vsi_raw_dir()"""
        return DATA_ROOT / self.name / "raws"

    @property
    def zip_dir(self) -> Path:
        """Generic replacement for get_vsi_zip_dir()"""
        return DATA_ROOT / self.name / "zips"

    @property
    def split_path(self) -> Path:
        path = PROJECT_ROOT / "splits" / f"splits_{self.name}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def get_run_dir(self) -> Path:
        cov_str = f"{self.prep.min_tissue_coverage:.2f}".replace(".", "")
        folder = f"s{self.prep.stride}_ds{self.prep.downsample_factor}_cov{cov_str}"
        return CACHE_DIR / self.name / folder

    def get_master_index_path(self) -> Path:
        path = self.get_run_dir() / "manifest.pkl"
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def get_slide_index_dir(self) -> Path:
        path = self.get_run_dir() / "indices"
        path.mkdir(parents=True, exist_ok=True)
        return path
