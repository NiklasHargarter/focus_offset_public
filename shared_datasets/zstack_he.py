"""ZStack HE dataset configuration and public API."""

import argparse
import pandas as pd
from shared_datasets.vsi.config import CACHE_ROOT, DATA_ROOT, SPLITS_ROOT
from shared_datasets.vsi.dataset import get_vsi_dataset
from shared_datasets.vsi.loader import get_vsi_dataloaders, get_vsi_test_loader
from shared_datasets.vsi.prep.download import download_vsi_dataset
from shared_datasets.vsi.prep.indexing import index_vsi_dataset


# Configuration
NAME = "ZStack_HE"
FULL_NAME = "ZStack_HE"
SLIDE_DIR = DATA_ROOT / NAME
DOWNSAMPLE = 2
COV = 0.70

_cov_str = f"{COV:.2f}".replace(".", "")
INDEX_DIR = CACHE_ROOT / NAME / f"ds{DOWNSAMPLE}_cov{_cov_str}"
SPLIT_PATH = SPLITS_ROOT / f"{NAME}.json"
PATCH_SIZE = 224
STRIDE = PATCH_SIZE * DOWNSAMPLE
EXCLUDE_PATTERN = "_all_"


# Dataset and DataLoaders API
def get_he_dataset(index_df: pd.DataFrame | None = None):
    if index_df is None:
        index_df = pd.read_parquet(INDEX_DIR / "train.parquet")
    return get_vsi_dataset(
        index_df=index_df,
        slide_dir=SLIDE_DIR,
        downsample=DOWNSAMPLE,
        patch_size=PATCH_SIZE,
    )


def get_dataloaders(train_cfg):
    return get_vsi_dataloaders(
        train_parquet=INDEX_DIR / "train.parquet",
        slide_dir=SLIDE_DIR,
        downsample=DOWNSAMPLE,
        train_cfg=train_cfg,
    )


def get_test_loader(train_cfg):
    return get_vsi_test_loader(
        test_parquet=INDEX_DIR / "test.parquet",
        slide_dir=SLIDE_DIR,
        downsample=DOWNSAMPLE,
        train_cfg=train_cfg,
    )


# Prep Commands API
def download_zstack_he(workers: int = 4):
    download_vsi_dataset(
        full_name=FULL_NAME,
        slide_dir=SLIDE_DIR,
        workers=workers,
        exclude_pattern=EXCLUDE_PATTERN,
    )


def preprocess_zstack_he(workers: int | None = None, dry_run: bool = False):
    params = {
        "downsample": DOWNSAMPLE,
        "cov": COV,
        "patch_size": PATCH_SIZE,
        "stride": STRIDE,
        "exclude_pattern": EXCLUDE_PATTERN,
    }
    index_vsi_dataset(
        slide_dir=SLIDE_DIR,
        index_dir=INDEX_DIR,
        split_path=SPLIT_PATH,
        params=params,
        workers=workers,
        dry_run=dry_run,
    )


# CLI for preparation commands
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=f"Prepare the {NAME} dataset.")
    parser.add_argument(
        "--download", action="store_true", help="Download missing slides from EXACT"
    )
    parser.add_argument(
        "--preprocess", action="store_true", help="Preprocess slides to build indices"
    )
    parser.add_argument("--workers", type=int, default=4, help="Number of workers")
    parser.add_argument("--dry_run", action="store_true", help="Run in dry run mode")
    args = parser.parse_args()

    if args.download:
        download_zstack_he(workers=args.workers)
    if args.preprocess:
        preprocess_zstack_he(workers=args.workers, dry_run=args.dry_run)
    if not (args.download or args.preprocess):
        parser.print_help()
