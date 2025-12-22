import sys
import config
from src.processing.download import download_dataset
from src.processing.fix_zip import fix_zip_structure
from src.processing.create_split import create_split
from src.processing.preprocess import preprocess_dataset
from src.dataset.vsi_dataset import VSIDataset


import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a VSI dataset.")
    parser.add_argument(
        "--dataset",
        type=str,
        default=config.DATASET_NAME,
        help="Name of the dataset to prepare.",
    )
    args = parser.parse_args()

    dataset_name = args.dataset
    print(f"Starting Dataset Preparation Pipeline for: {dataset_name}...")

    print("\n--- Step 1: Download ---")
    try:
        download_dataset(dataset_name=dataset_name)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

    print("\n--- Step 2: Extract & Fix ---")
    try:
        fix_zip_structure(dataset_name=dataset_name)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

    print("\n--- Step 3: Create Splits ---")
    try:
        create_split(dataset_name=dataset_name)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

    print("\n--- Step 4: Preprocess ---")
    try:
        preprocess_dataset(dataset_name=dataset_name)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

    print("\n--- Step 5: Verification ---")
    try:
        for mode in ["train", "val", "test"]:
            try:
                ds = VSIDataset(mode=mode, dataset_name=dataset_name)
                print(f"[OK] {mode.capitalize()} Dataset loadable: {len(ds)} samples.")
            except FileNotFoundError:
                print(
                    f"[INFO] {mode.capitalize()} Dataset index not found (expected for some OOD sets)."
                )

        print(f"\nPipeline Complete! {dataset_name} is ready.")
    except Exception as e:
        print(f"[FAIL] Verification failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
