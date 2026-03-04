"""AgNor OME dataset configuration and public API."""

import pandas as pd
from .agnor_ome import get_ome_dataloaders, get_ome_test_loader, get_ome_dataset
from .agnor_ome.config import AgNorOMEConfig

_cfg = AgNorOMEConfig()

# Configuration mirrored from zstack_he/zstack_ihc style
NAME = _cfg.name
SLIDE_DIR = _cfg.raw_dir
INDEX_DIR = _cfg.get_run_dir()
DOWNSAMPLE = _cfg.prep.downsample_factor


def get_agnor_dataset(index_df: pd.DataFrame | None = None):
    if index_df is None:
        index_df = pd.read_parquet(INDEX_DIR / "train.parquet")
    return get_ome_dataset(
        index_df=index_df,
        slide_dir=SLIDE_DIR,
        downsample=DOWNSAMPLE,
    )


def get_dataloaders(train_cfg):
    return get_ome_dataloaders(
        train_parquet=INDEX_DIR / "train.parquet",
        slide_dir=SLIDE_DIR,
        downsample=DOWNSAMPLE,
        train_cfg=train_cfg,
    )


def get_test_loader(train_cfg):
    return get_ome_test_loader(
        test_parquet=INDEX_DIR / "test.parquet",
        slide_dir=SLIDE_DIR,
        downsample=DOWNSAMPLE,
        train_cfg=train_cfg,
    )
