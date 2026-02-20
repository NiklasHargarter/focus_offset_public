import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DATA_ROOT = Path(os.environ.get("DATA_ROOT", Path.home()))
CACHE_DIR = PROJECT_ROOT / "cache"


@dataclass
class Jiang2018Config:
    name: str = "Jiang2018"

    @property
    def raw_dir(self) -> Path:
        return DATA_ROOT / self.name / "raws"

    @property
    def zip_dir(self) -> Path:
        return DATA_ROOT / self.name / "zips"
