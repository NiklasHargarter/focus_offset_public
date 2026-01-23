import os
import argparse
from src import config
from exact_sync.v1.configuration import Configuration
from exact_sync.v1.api_client import ApiClient
from exact_sync.v1.api.images_api import ImagesApi
from src.utils.exact_utils import get_exact_image_list
from src.dataset.vsi_prep.fix_zip import (
    extract_zip,
    verify_vsi,
    organize_vsi_files,
    cleanup_corrupt_vsi,
)


def download_dataset(
    dataset_name: str = config.DATASET_NAME, force: bool = False, keep_zip: bool = False
) -> None:
    """Download, extract, and verify VSI files one-by-one to save space."""
    print(f"Fetching full image list for {dataset_name} from EXACT...")
    all_images = get_exact_image_list(dataset_name=dataset_name, force=force)

    zip_dir = config.get_vsi_zip_dir(dataset_name)
    raw_dir = config.get_vsi_raw_dir(dataset_name)

    zip_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    print(f"Total images in dataset: {len(all_images)}")

    configuration = Configuration()
    configuration.username = os.environ.get("EXACT_USERNAME", "niklas.hargarter")
    configuration.password = os.environ.get("EXACT_PASSWORD")
    if configuration.password is None:
        raise ValueError("Environment variable 'EXACT_PASSWORD' is not set.")
    configuration.host = "https://exact.hs-flensburg.de"

    client = ApiClient(configuration)
    images_api = ImagesApi(client)

    print(f"Starting processing for {len(all_images)} images in {dataset_name}...")

    for img_info in all_images:
        image_id = img_info["id"]
        vsi_name = img_info["name"]
        zip_name = vsi_name.replace(".vsi", ".zip")

        vsi_path = raw_dir / vsi_name
        zip_path = zip_dir / zip_name

        print(f"\n--- Processing {vsi_name} ---")

        # 1. Step: Check if valid VSI already exists
        if vsi_path.exists():
            print("VSI already exists locally. Verifying...")
            if verify_vsi(vsi_path):
                print(f"[OK] {vsi_name} is valid.")
                # If valid VSI exists, we don't need the zip
                if zip_path.exists() and not keep_zip:
                    print(f"Cleaning up redundant ZIP: {zip_name}")
                    zip_path.unlink()
                continue
            else:
                print(f"[FAIL] {vsi_name} is corrupt. Re-processing...")
                cleanup_corrupt_vsi(vsi_path, zip_dir, raw_dir)

        # 2. Step: Check for ZIP if VSI is missing/corrupt
        if not zip_path.exists():
            print(f"ZIP missing. Downloading {zip_name}...")
            try:
                images_api.download_image(id=image_id, target_path=str(zip_path))
            except Exception as e:
                print(f"[ERROR] Failed to download {vsi_name}: {e}")
                continue
        else:
            print(f"ZIP already exists: {zip_name}")

        # 3. Step: Unzip and Verify
        print(f"Extracting {zip_name}...")
        try:
            extract_zip(zip_path, raw_dir)
            organize_vsi_files(raw_dir)  # Handle any nested structures

            if verify_vsi(vsi_path):
                print(f"[OK] {vsi_name} extracted and verified.")
                if not keep_zip:
                    print(f"Deleting ZIP to save space: {zip_name}")
                    zip_path.unlink()
            else:
                print(f"[FAIL] Verification failed after extraction for {vsi_name}.")
                cleanup_corrupt_vsi(vsi_path, zip_dir, raw_dir)
        except Exception as e:
            print(f"[ERROR] Unzipping/Verification failed for {zip_name}: {e}")

    print(f"\nProcessing for {dataset_name} complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=config.DATASET_NAME)
    parser.add_argument(
        "--force", action="store_true", help="Force refresh image list from EXACT"
    )
    parser.add_argument(
        "--keep_zip",
        action="store_true",
        help="Keep the ZIP file after successful extraction",
    )
    args = parser.parse_args()
    download_dataset(
        dataset_name=args.dataset, force=args.force, keep_zip=args.keep_zip
    )
