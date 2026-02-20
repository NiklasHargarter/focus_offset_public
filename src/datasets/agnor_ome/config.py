import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DATA_ROOT = Path(os.environ.get("DATA_ROOT", Path.home()))
CACHE_DIR = PROJECT_ROOT / "cache"


@dataclass
class PrepConfig:
    stride: int = 224
    downsample_factor: int = 1
    min_tissue_coverage: float = 0.80
    patch_size: int = 224


@dataclass
class AgNorOMEConfig:
    name: str = "AgNor_OME"
    exclude_pattern: str = "_all_"
    prep: PrepConfig = field(default_factory=PrepConfig)

    @property
    def raw_dir(self) -> Path:
        return DATA_ROOT / self.name / "raws"

    @property
    def zip_dir(self) -> Path:
        return DATA_ROOT / self.name / "zips"

    @property
    def split_path(self) -> Path:
        path = PROJECT_ROOT / "splits" / f"splits_{self.name}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        return path
    def get_train_index_path(self) -> Path:
        path = self.get_run_dir() / "train.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def get_test_index_path(self) -> Path:
        path = self.get_run_dir() / "test.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def get_run_dir(self) -> Path:
        cov_str = f"{self.prep.min_tissue_coverage:.2f}".replace(".", "")
        folder = f"s{self.prep.stride}_ds{self.prep.downsample_factor}_cov{cov_str}"
        return CACHE_DIR / self.name / folder

    def get_index_path(self) -> Path:
        path = self.get_run_dir() / "index.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        return path
