import argparse

from src.config import DatasetConfig
from src.dataset.vsi_prep.fix_zip import verify_vsi


def cleanup_zips(dataset_name: str = "ZStack_HE"):
    """
    Check raws folder for vsi files, verify them, and delete the corresponding zip file.
    """
    dataset_cfg = DatasetConfig(name=dataset_name)
    zip_dir = dataset_cfg.zip_dir
    raw_dir = dataset_cfg.raw_dir

    if not raw_dir.exists():
        print(f"Error: Raw directory {raw_dir} does not exist.")
        return

    if not zip_dir.exists():
        print(f"Error: Zip directory {zip_dir} does not exist.")
        return

    vsi_files = list(raw_dir.glob("*.vsi"))
    print(f"Found {len(vsi_files)} VSI files in {raw_dir}")

    total_deleted = 0
    total_space_saved = 0

    for vsi_path in vsi_files:
        # 1. Quick check like in fix_zip
        if verify_vsi(vsi_path):
            # 2. Find corresponding zip
            zip_filename = vsi_path.name.replace(".vsi", ".zip")
            zip_path = zip_dir / zip_filename

            if zip_path.exists():
                file_size = zip_path.stat().st_size
                print(
                    f"[OK] {vsi_path.name} is valid. Deleting {zip_filename} ({file_size / 1e9:.2f} GB)..."
                )
                try:
                    zip_path.unlink()
                    total_deleted += 1
                    total_space_saved += file_size
                except Exception as e:
                    print(f"      Error deleting {zip_filename}: {e}")
            else:
                # Zip might already be deleted or name doesn't match
                pass
        else:
            print(f"[SKIP] {vsi_path.name} failed verification. Keeping ZIP.")

    print(f"\nCleanup complete for {dataset_name}!")
    print(f"Deleted {total_deleted} zip files.")
    print(f"Total space saved: {total_space_saved / 1e9:.2f} GB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Cleanup ZIP files for VSIs that are already extracted and verified."
    )
    dataset_cfg = DatasetConfig()
    parser.add_argument(
        "--dataset",
        type=str,
        default=dataset_cfg.name,
        help="Dataset name to clean up",
    )
    args = parser.parse_args()

    cleanup_zips(dataset_name=args.dataset)
