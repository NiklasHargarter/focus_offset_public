import sys
from src.processing.download import download_dataset
from src.processing.fix_zip import fix_zip_structure
from src.processing.create_split import create_split
from src.processing.preprocess import preprocess_dataset
from src.dataset.vsi_dataset import VSIDataset


def main() -> None:
    print("Starting Dataset Preparation Pipeline...")

    print("\n--- Step 1: Download ---")
    try:
        download_dataset()
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

    print("\n--- Step 2: Extract & Fix ---")
    try:
        fix_zip_structure()
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

    print("\n--- Step 3: Create Splits ---")
    try:
        create_split()
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

    print("\n--- Step 4: Preprocess ---")
    try:
        preprocess_dataset()
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

    print("\n--- Step 5: Verification ---")
    try:
        ds_train = VSIDataset(mode="train")
        print(f"[OK] Train Dataset loadable: {len(ds_train)} samples.")

        ds_val = VSIDataset(mode="val")
        print(f"[OK] Val Dataset loadable: {len(ds_val)} samples.")

        ds_test = VSIDataset(mode="test")
        print(f"[OK] Test Dataset loadable: {len(ds_test)} samples.")
        print("\nPipeline Complete! Dataset is ready for training.")
    except Exception as e:
        print(f"[FAIL] Verification failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
