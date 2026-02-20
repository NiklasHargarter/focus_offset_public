import argparse

from src.config import DatasetConfig
from src.dataset.vsi_prep.create_split import create_split
from src.dataset.vsi_prep.download import download_dataset
from src.dataset.vsi_prep.fix_zip import fix_zip_structure
from src.dataset.vsi_prep.preprocess import preprocess_dataset


def sync(
    dataset_name: str,
    stride: int,
    downsample_factor: int,
    min_tissue_coverage: float,
    force: bool = False,
    skip_preprocess: bool = False,
    skip_split: bool = False,
    limit: int | None = None,
    exclude: str = "_all_",
):
    """
    Orchestrates the full dataset preparation pipeline:
    1. Download missing ZIPs from EXACT.
    2. Extract and verify ZIP structures (converts to VSI).
    3. Incremental Preprocessing (generates MasterIndex).
    4. Split Generation (train_pool/test).
    """
    print(f"=== Syncing Dataset: {dataset_name} ===")

    print("\n[Step 1/2] Downloading missing files...")
    download_dataset(
        dataset_name=dataset_name, force=force, limit=limit, exclude=exclude
    )

    print("\n[Step 2/2] Extracting and verifying ZIP structures...")
    fix_zip_structure(dataset_name=dataset_name)

    if not skip_preprocess:
        print("\n[Step 3] Preprocessing (Incremental)...")
        preprocess_dataset(
            dataset_name=dataset_name,
            stride=stride,
            downsample_factor=downsample_factor,
            min_tissue_coverage=min_tissue_coverage,
            force=force,
            exclude=exclude,
        )
    else:
        print("\n[Step 3] Skipping Preprocessing.")

    if not skip_split:
        print("\n[Step 4] Updating splits...")
        create_split(
            dataset_name=dataset_name,
            stride=stride,
            downsample_factor=downsample_factor,
            min_tissue_coverage=min_tissue_coverage,
            force=force,
        )
    else:
        print("\n[Step 4] Skipping Split Generation.")

    print(f"\n=== Sync for {dataset_name} complete ===")


if __name__ == "__main__":
    dataset_cfg = DatasetConfig()
    parser = argparse.ArgumentParser(
        description="Synchronize and preprocess the dataset."
    )
    parser.add_argument("--dataset", type=str, default=dataset_cfg.name)
    parser.add_argument("--stride", type=int, default=dataset_cfg.prep.stride)
    parser.add_argument(
        "--downsample_factor", type=int, default=dataset_cfg.prep.downsample_factor
    )
    parser.add_argument(
        "--min_tissue_coverage",
        type=float,
        default=dataset_cfg.prep.min_tissue_coverage,
    )
    parser.add_argument(
        "--force", action="store_true", help="Force re-preprocessing of ALL files"
    )
    parser.add_argument(
        "--skip-preprocess", action="store_true", help="Only download and unzip"
    )
    parser.add_argument(
        "--skip-split", action="store_true", help="Skip split generation"
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Limit number of slides to download"
    )
    parser.add_argument(
        "--exclude",
        type=str,
        default=dataset_cfg.exclude_pattern,
        help="Exclude slides containing this string in their name",
    )

    args = parser.parse_args()
    sync(
        args.dataset,
        args.stride,
        args.downsample_factor,
        args.min_tissue_coverage,
        args.force,
        args.skip_preprocess,
        args.skip_split,
        args.limit,
        args.exclude,
    )
