import argparse

from src.datasets.agnor_ome.config import AgNorOMEConfig
from .preprocess import preprocess_dataset

def sync(
    dataset_name: str = "AgNor_OME",
    skip_preprocess: bool = False,
    dry_run: bool = False,
):
    """
    Orchestrates the standalone dataset preparation pipeline for AgNor OME evaluations.
    1. Preprocessing (generates Index Parquet table).
    """
    print(f"=== Syncing Dataset: {dataset_name} ===")
    dataset_cfg = AgNorOMEConfig(name=dataset_name)

    if not skip_preprocess:
        index_path = dataset_cfg.get_index_path()
        if index_path.exists():
            print(f"\n[Step 1] Index {index_path} exists. Skipping.")
        else:
            print("\n[Step 1] Preprocessing (Extracting patches to flat Parquet)...")
            preprocess_dataset(
                dataset_name=dataset_name,
                dry_run=dry_run,
            )
    else:
        print("\n[Step 1] Skipping Preprocessing.")

    print(f"\n=== Sync for {dataset_name} complete ===")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Synchronize and preprocess the dataset."
    )
    parser.add_argument("--dataset", type=str, default="AgNor_OME")
    parser.add_argument(
        "--skip-preprocess", action="store_true", help="Only skip the prep phase"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run preprocessing on a small subset for testing",
    )
    args = parser.parse_args()
    sync(
        args.dataset,
        args.skip_preprocess,
        args.dry_run,
    )
