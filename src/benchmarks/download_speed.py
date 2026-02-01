import time
import os
import argparse
from src import config
from exact_sync.v1.configuration import Configuration
from exact_sync.v1.api_client import ApiClient
from exact_sync.v1.api.images_api import ImagesApi
from src.utils.exact_utils import get_exact_image_list


def benchmark_download(dataset_name: str = config.DATASET_NAME):
    print(f"=== Download Speed Benchmark for {dataset_name} ===")

    # 1. Fetch image list
    all_images = get_exact_image_list(dataset_name=dataset_name)
    total_count = len(all_images)
    print(f"Total images in dataset: {total_count}")

    # 2. Setup EXACT client
    configuration = Configuration()
    configuration.username = os.environ.get("EXACT_USERNAME", "niklas.hargarter")
    configuration.password = os.environ.get("EXACT_PASSWORD")
    if configuration.password is None:
        raise ValueError("Environment variable 'EXACT_PASSWORD' is not set.")
    configuration.host = "https://exact.hs-flensburg.de"

    client = ApiClient(configuration)
    images_api = ImagesApi(client)

    # 3. Pick a slide to download (first one)
    img_info = all_images[0]
    image_id = img_info["id"]
    vsi_name = img_info["name"]
    zip_name = vsi_name.replace(".vsi", ".zip")

    temp_dir = config.DATA_ROOT / "benchmark_temp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    target_path = temp_dir / zip_name

    print(f"\nBenchmarking download of: {zip_name}")
    print("This might take a while depending on your connection...")

    start_time = time.time()
    try:
        images_api.download_image(id=image_id, target_path=str(target_path))
        end_time = time.time()
    except Exception as e:
        print(f"[ERROR] Download failed: {e}")
        return

    duration = end_time - start_time
    file_size_mb = target_path.stat().st_size / (1024 * 1024)
    speed_mbps = file_size_mb / duration

    print("\n--- Results ---")
    print(f"File size: {file_size_mb:.2f} MB")
    print(f"Download time: {duration:.2f} seconds")
    print(f"Average speed: {speed_mbps:.2f} MB/s")

    # 4. Estimation
    avg_total_size_gb = (file_size_mb * total_count) / 1024
    total_time_hours = (duration * total_count) / 3600

    print(f"\n--- Estimation for {total_count} slides ---")
    print(f"Estimated total size: ~{avg_total_size_gb:.2f} GB")
    print(f"Estimated total time: ~{total_time_hours:.2f} hours")

    # Cleanup
    print(f"\nCleaning up benchmark file: {target_path}")
    target_path.unlink()
    if not any(temp_dir.iterdir()):
        temp_dir.rmdir()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark download speed from EXACT.")
    parser.add_argument("--dataset", type=str, default=config.DATASET_NAME)
    args = parser.parse_args()

    benchmark_download(dataset_name=args.dataset)
