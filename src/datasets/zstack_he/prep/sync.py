import argparse

from .create_split import create_split
from .download import download_zstack_he
from .fix_zip import fix_zip_structure
from .preprocess import preprocess_dataset


def sync(
    dataset_name: str = "ZStack_HE",
    skip_preprocess: bool = False,
    skip_split: bool = False,
    skip_download: bool = False,
    dry_run: bool = False,
):
    """
    Orchestrates the full dataset preparation pipeline:
    1. Download missing ZIPs from EXACT.
    2. Extract and verify ZIP structures (converts to VSI).
    3. Split Generation (train_pool/test).
    4. Preprocessing (generates Index Parquets).
    """
    print(f"=== Syncing Dataset: {dataset_name} ===")

    if not skip_download:
        print("\n[Step 1/2] Downloading missing files...")
        download_zstack_he(
            dataset_name=dataset_name
        )

    print("\n[Step 2/2] Extracting and verifying ZIP structures...")
    fix_zip_structure(dataset_name=dataset_name)

    if not skip_split:
        print("\n[Step 3] Splitting raw slides...")
        create_split(
            dataset_name=dataset_name,
        )
    else:
        print("\n[Step 3] Skipping Split Generation.")

    if not skip_preprocess:
        print("\n[Step 4] Preprocessing (Extracting patches to Parquet)...")
        preprocess_dataset(
            dataset_name=dataset_name,
            dry_run=dry_run,
        )
    else:
        print("\n[Step 4] Skipping Preprocessing.")

    print(f"\n=== Sync for {dataset_name} complete ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Synchronize and preprocess the dataset."
    )
    parser.add_argument("--dataset", type=str, default="ZStack_HE")
    parser.add_argument(
        "--skip-preprocess", action="store_true", help="Only download and unzip"
    )
    parser.add_argument(
        "--skip-download", action="store_true", help="Skip the downloading step"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run preprocessing on a small subset for testing",
    )
    parser.add_argument(
        "--skip-split", action="store_true", help="Skip split generation"
    )
    args = parser.parse_args()
    sync(
        args.dataset,
        args.skip_preprocess,
        args.skip_split,
        args.skip_download,
        args.dry_run,
    )
