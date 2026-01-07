import os
import argparse
import config
from exact_sync.v1.configuration import Configuration
from exact_sync.v1.api_client import ApiClient
from exact_sync.v1.api.images_api import ImagesApi
from src.utils.exact_utils import get_exact_image_list


def download_dataset(dataset_name: str = config.DATASET_NAME) -> None:
    """Download missing VSI zips from EXACT."""
    print(f"Fetching full image list for {dataset_name} from EXACT...")
    all_images = get_exact_image_list(dataset_name=dataset_name)

    download_target = config.get_vsi_zip_dir(dataset_name)
    download_target.mkdir(parents=True, exist_ok=True)

    print(f"Total images in dataset: {len(all_images)}")
    print(f"Checking for missing files in: {download_target}")

    configuration = Configuration()
    configuration.username = "niklas.hargarter"
    configuration.password = os.environ.get("EXACT_PASSWORD")
    if configuration.password is None:
        raise ValueError("Environment variable 'EXACT_PASSWORD' is not set.")
    configuration.host = "https://exact.hs-flensburg.de"

    client = ApiClient(configuration)
    images_api = ImagesApi(client)

    print(f"Starting download check for {len(all_images)} images in {dataset_name}...")

    for img_info in all_images:
        image_id = img_info["id"]
        original_name = img_info["name"]

        target_name = original_name.replace(".vsi", ".zip")
        file_path = download_target / target_name

        if not file_path.exists():
            print(f"Downloading {original_name} -> {target_name}...")
            try:
                images_api.download_image(id=image_id, target_path=str(file_path))
            except Exception as e:
                print(f"Failed to download {original_name}: {e}")
        else:
            print(f"Skipping {target_name} (already exists).")

    print(f"Download process for {dataset_name} complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=config.DATASET_NAME)
    args = parser.parse_args()
    download_dataset(dataset_name=args.dataset)
