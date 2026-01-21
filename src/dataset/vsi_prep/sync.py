import argparse
from src import config
from src.dataset.vsi_prep.download import download_dataset
from src.dataset.vsi_prep.fix_zip import fix_zip_structure
from src.dataset.vsi_prep.preprocess import preprocess_dataset
from src.dataset.vsi_prep.create_split import create_split


def sync(
    dataset_name: str,
    patch_size: int,
    stride: int,
    min_tissue_coverage: float,
    force: bool = False,
    skip_preprocess: bool = False,
    skip_split: bool = False,
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
    download_dataset(dataset_name=dataset_name, force=force)

    print("\n[Step 2/2] Extracting and verifying ZIP structures...")
    fix_zip_structure(dataset_name=dataset_name)

    if not skip_preprocess:
        print("\n[Step 3] Preprocessing (Incremental)...")
        preprocess_dataset(
            dataset_name=dataset_name,
            patch_size=patch_size,
            stride=stride,
            min_tissue_coverage=min_tissue_coverage,
            force=force,
        )
    else:
        print("\n[Step 3] Skipping Preprocessing.")

    if not skip_split:
        print("\n[Step 4] Updating splits...")
        create_split(dataset_name=dataset_name, force=force)
    else:
        print("\n[Step 4] Skipping Split Generation.")

    print(f"\n=== Sync for {dataset_name} complete ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Synchronize and preprocess the dataset."
    )
    parser.add_argument("--dataset", type=str, default=config.DATASET_NAME)
    parser.add_argument("--patch_size", type=int, default=224)
    parser.add_argument("--stride", type=int, default=224)
    parser.add_argument("--min_tissue_coverage", type=float, default=0.05)
    parser.add_argument(
        "--force", action="store_true", help="Force re-preprocessing of ALL files"
    )
    parser.add_argument(
        "--skip-preprocess", action="store_true", help="Only download and unzip"
    )
    parser.add_argument(
        "--skip-split", action="store_true", help="Skip split generation"
    )

    args = parser.parse_args()
    sync(
        args.dataset,
        args.patch_size,
        args.stride,
        args.min_tissue_coverage,
        args.force,
        args.skip_preprocess,
        args.skip_split,
    )
