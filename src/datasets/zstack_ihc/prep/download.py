import concurrent.futures
import os
from concurrent.futures import ThreadPoolExecutor

from exact_sync.v1.api.images_api import ImagesApi
from exact_sync.v1.api_client import ApiClient
from exact_sync.v1.configuration import Configuration

from src.datasets.zstack_ihc.config import ZStackIHCConfig

from src.utils.exact_utils import get_exact_image_list


def process_image(img_info, dataset_name, raw_dir, zip_dir):
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

    # 1. Check if we already have the target slide
    if vsi_path.exists():
        print(f"{log_prefix} VSI already exists. Skipping download.")
        return

    # 2. Check if the zip is already downloaded
    if zip_path.exists():
        print(f"{log_prefix} ZIP already exists. Skipping download.")
        return

    # 3. Download the ZIP file from EXACT
    print(f"{log_prefix} Downloading ZIP...")
    try:
        images_api.download_image(id=image_id, target_path=str(zip_path))
        print(f"{log_prefix} [OK] Download complete.")
    except Exception as e:
        print(f"{log_prefix} [ERROR] Download failed: {e}")


def download_zstack_ihc(
    dataset_name: str = "ZStack_IHC",
    workers: int = 4,
) -> None:
    """Download, extract, and verify VSI files in parallel."""
    print(f"Fetching full image list for {dataset_name} from EXACT...")
    all_images = get_exact_image_list(dataset_name=dataset_name, force=False)

    dataset_cfg = ZStackIHCConfig(name=dataset_name)
    zip_dir = dataset_cfg.zip_dir
    raw_dir = dataset_cfg.raw_dir

    zip_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    exclude = "_all_"
    if exclude:
        print(f"Excluding images containing: '{exclude}' (case-insensitive)")
        all_images = [
            img for img in all_images if exclude.lower() not in img["name"].lower()
        ]
        print(f"Images remaining after exclusion: {len(all_images)}")

    if os.environ.get("EXACT_PASSWORD") is None:
        raise ValueError("Environment variable 'EXACT_PASSWORD' is not set.")

    print(
        f"Starting parallel processing with {workers} workers for {len(all_images)} images..."
    )

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(
                process_image, img_info, dataset_name, raw_dir, zip_dir
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


