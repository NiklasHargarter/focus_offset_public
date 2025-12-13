import sys
from src.processing.download import download_dataset
from src.processing.fix_zip import fix_zip_structure
from src.processing.create_split import create_split
from src.processing.preprocess import preprocess_dataset
from src.dataset.vsi_dataset import VSIDataset


def main() -> None:
    print("Starting Dataset Preparation Pipeline (Module-based)...")

    # 1. Download
    print("\n--- Run Step 1: Download from EXACT ---")
    try:
        download_dataset()
    except Exception as e:
        print(f"Error executing download_dataset: {e}")
        sys.exit(1)

    # 2. Fix Zip
    print("\n--- Run Step 2: Extract & Fix Structure ---")
    try:
        fix_zip_structure()
    except Exception as e:
        print(f"Error executing fix_zip_structure: {e}")
        sys.exit(1)

    # 3. Create Split
    print("\n--- Run Step 3: Create Train/Test Split ---")
    try:
        create_split()
    except Exception as e:
        print(f"Error executing create_split: {e}")
        sys.exit(1)

    # 4. Preprocess
    print("\n--- Run Step 4: Preprocess VSI Files ---")
    try:
        preprocess_dataset()
    except Exception as e:
        print(f"Error executing preprocess_dataset: {e}")
        sys.exit(1)

    # 5. Verify
    print("\n--- Run Step 5: Verification ---")
    try:
        ds_train = VSIDataset(mode="train")
        print(f"[OK] Train Dataset loadable: {len(ds_train)} samples.")

        ds_test = VSIDataset(mode="test")
        print(f"[OK] Test Dataset loadable: {len(ds_test)} samples.")
        print("\nPipeline Complete! Dataset is ready for training.")
    except Exception as e:
        print(f"[FAIL] Verification failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
