import os
from pathlib import Path

# Layout Constants
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = Path(os.environ.get("DATA_ROOT", "/data/niklas"))
CACHE_ROOT = PROJECT_ROOT / "cache"
SPLITS_ROOT = PROJECT_ROOT / "splits"


class AgNorOMEConfig:
    name: str = "AgNor"
    exclude_pattern: str = "_all_"

    class prep:
        patch_size: int = 224
        stride: int = 224
        downsample_factor: int = 1
        min_tissue_coverage: float = 0.2

    @property
    def raw_dir(self) -> Path:
        return DATA_ROOT / "AgNor_OME" / "raws"

    def get_run_dir(self) -> Path:
        _cov_str = f"{self.prep.min_tissue_coverage:.2f}".replace(".", "")
        return (
            CACHE_ROOT
            / self.name
            / f"s{self.prep.patch_size}_ds{self.prep.downsample_factor}_cov{_cov_str}"
        )

    @property
    def split_path(self) -> Path:
        return SPLITS_ROOT / f"splits_{self.name}.json"
