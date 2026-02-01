import os
import argparse
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
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


def process_image(img_info, dataset_name, raw_dir, zip_dir, keep_zip):
    """Worker function to process a single image."""
    image_id = img_info["id"]
    vsi_name = img_info["name"]
    zip_name = vsi_name.replace(".vsi", ".zip")

    vsi_path = raw_dir / vsi_name
    zip_path = zip_dir / zip_name

    # Create a fresh client for thread safety
    configuration = Configuration()
    configuration.username = os.environ.get("EXACT_USERNAME", "niklas.hargarter")
    configuration.password = os.environ.get("EXACT_PASSWORD")
    configuration.host = "https://exact.hs-flensburg.de"

    # Suppress huge logs from urllib3/swagger if needed, or just let them be
    client = ApiClient(configuration)
    images_api = ImagesApi(client)

    log_prefix = f"[{vsi_name}]"

    # 1. Step: Check if valid VSI already exists
    if vsi_path.exists():
        if verify_vsi(vsi_path):
            print(f"{log_prefix} VSI valid. Skipping.")
            if zip_path.exists() and not keep_zip:
                zip_path.unlink()
            return
        else:
            print(f"{log_prefix} VSI corrupt. Cleaning up...")
            cleanup_corrupt_vsi(vsi_path, zip_dir, raw_dir)

    # 2. Step: Check for ZIP if VSI is missing/corrupt
    if not zip_path.exists():
        print(f"{log_prefix} Downloading ZIP...")
        try:
            images_api.download_image(id=image_id, target_path=str(zip_path))
        except Exception as e:
            print(f"{log_prefix} [ERROR] Download failed: {e}")
            return
    else:
        print(f"{log_prefix} ZIP found.")

    # 3. Step: Unzip and Verify
    print(f"{log_prefix} Extracting...")
    try:
        extract_zip(zip_path, raw_dir)
        organize_vsi_files(raw_dir)

        if verify_vsi(vsi_path):
            print(f"{log_prefix} [OK] Extracted and verified.")
            if not keep_zip:
                zip_path.unlink()
        else:
            print(f"{log_prefix} [FAIL] Verification failed after extraction.")
            cleanup_corrupt_vsi(vsi_path, zip_dir, raw_dir)
    except Exception as e:
        print(f"{log_prefix} [ERROR] Extraction failed: {e}")


def download_dataset(
    dataset_name: str = config.DATASET_NAME,
    force: bool = False,
    keep_zip: bool = False,
    limit: int = None,
    exclude: str = None,
    workers: int = 4,
) -> None:
    """Download, extract, and verify VSI files in parallel."""
    print(f"Fetching full image list for {dataset_name} from EXACT...")
    all_images = get_exact_image_list(dataset_name=dataset_name, force=force)

    zip_dir = config.get_vsi_zip_dir(dataset_name)
    raw_dir = config.get_vsi_raw_dir(dataset_name)

    zip_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    print(f"Total images in dataset: {len(all_images)}")

    if exclude:
        print(f"Excluding images containing: '{exclude}' (case-insensitive)")
        all_images = [
            img for img in all_images if exclude.lower() not in img["name"].lower()
        ]
        print(f"Images remaining after exclusion: {len(all_images)}")

    if limit is not None:
        print(f"Limiting download to the first {limit} slides.")
        all_images = all_images[:limit]

    if os.environ.get("EXACT_PASSWORD") is None:
        raise ValueError("Environment variable 'EXACT_PASSWORD' is not set.")

    print(
        f"Starting parallel processing with {workers} workers for {len(all_images)} images..."
    )

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(
                process_image, img_info, dataset_name, raw_dir, zip_dir, keep_zip
            )
            for img_info in all_images
        ]

        # Wait for all to complete
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"Worker exception: {e}")

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
    parser.add_argument(
        "--limit", type=int, default=None, help="Limit number of slides to download"
    )
    parser.add_argument(
        "--exclude",
        type=str,
        default=None,
        help="Exclude slides containing this string in their name",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel download workers",
    )
    args = parser.parse_args()
    download_dataset(
        dataset_name=args.dataset,
        force=args.force,
        keep_zip=args.keep_zip,
        limit=args.limit,
        exclude=args.exclude,
        workers=args.workers,
    )
