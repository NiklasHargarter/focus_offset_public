import os
import config
from exact_sync.v1.configuration import Configuration
from exact_sync.v1.api_client import ApiClient
from exact_sync.v1.api.images_api import ImagesApi
from src.utils.exact_utils import get_exact_image_list


def download_dataset() -> None:
    """Downloads all missing images from the EXACT dataset."""
    print("Fetching full image list from EXACT...")
    all_images = get_exact_image_list()

    download_target = config.VSI_ZIP_DIR
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

    print(f"Starting download check for {len(all_images)} images...")

    for img_info in all_images:
        image_id = img_info["id"]
        original_name = img_info["name"]

        target_name = original_name.replace(".vsi", ".zip")
        file_path = download_target / target_name

        if not file_path.exists():
            print(f"Downloading {original_name} -> {target_name}...")
            try:
                images_api.download_image(id=image_id, target_path=file_path)
            except Exception as e:
                print(f"Failed to download {original_name}: {e}")
        else:
            print(f"Skipping {target_name} (already exists).")

    print("Download process complete.")


if __name__ == "__main__":
    download_dataset()
