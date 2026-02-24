import concurrent.futures
import os
from concurrent.futures import ThreadPoolExecutor

from src.datasets.zstack_ihc.config import ZStackIHCConfig
from src.datasets.zstack_he.prep.download import process_image
from src.utils.exact_utils import get_exact_image_list


def download_zstack_ihc(
    dataset_name: str = "ZStack_IHC",
    workers: int = 4,
) -> None:
    """Download, extract, and verify VSI files in parallel."""
    print(f"Fetching full image list for {dataset_name} from EXACT...")
    all_images = get_exact_image_list(dataset_name=dataset_name, force=False)

    dataset_cfg = ZStackIHCConfig()
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
            executor.submit(process_image, img_info, dataset_name, raw_dir, zip_dir)
            for img_info in all_images
        ]

        # Wait for all to complete
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"Worker exception: {e}")

    print(f"\nProcessing for {dataset_name} complete.")
