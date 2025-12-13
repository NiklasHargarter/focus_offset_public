import os
import config
from exact_sync.v1.configuration import Configuration
from exact_sync.v1.api_client import ApiClient
from exact_sync.v1.api.images_api import ImagesApi
from exact_sync.v1.api.image_sets_api import ImageSetsApi
import json


def download_dataset() -> None:
    # 1. Setup API
    configuration = Configuration()
    configuration.username = "niklas.hargarter"
    configuration.password = os.environ.get("EXACT_PASSWORD")
    if configuration.password is None:
        raise ValueError("Environment variable 'EXACT_PASSWORD' is not set.")
    configuration.host = "https://exact.hs-flensburg.de"

    client = ApiClient(configuration)
    image_sets_api = ImageSetsApi(client)
    images_api = ImagesApi(client)

    # 2. Setup Directories from Config
    download_target = config.VSI_ZIP_DIR
    download_target.mkdir(parents=True, exist_ok=True)

    print(f"Downloading to: {download_target}")

    dataset_name = "ZStack_HE"

    # 3. Check Local Cache

    cache_file = config.CACHE_DIR / "exact_images.json"
    target_images = []

    if cache_file.exists():
        print(f"Loading image list from cache: {cache_file}")
        with open(cache_file, "r") as f:
            target_images = json.load(f)
    else:
        # 3b. List Datasets from API
        print("Querying EXACT API for image list...")
        image_sets = image_sets_api.list_image_sets(name__contains=dataset_name)
        print(f"Found {len(image_sets.results)} datasets matching '{dataset_name}'")

        if len(image_sets.results) > 1:
            raise ValueError("Error: More than one dataset with the same name found!")
        if len(image_sets.results) == 0:
            raise ValueError("Error: No dataset found.")

        target_set = image_sets.results[0]
        print(f"Dataset: {target_set.name}, Images: {len(target_set.images)}")

        # Resolve Image Objects to get names
        print("Resolving image details...")
        for image_id in target_set.images:
            # We need the name to check existence, so we must fetch the image object
            # This part might still be slow initially, but we cache the result
            img_obj = images_api.retrieve_image(id=image_id)
            target_images.append({"id": image_id, "name": img_obj.name})

        # Save to cache
        config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with open(cache_file, "w") as f:
            json.dump(target_images, f, indent=4)
        print(f"Saved image list to cache: {cache_file}")

    # 4. Download Loop
    for img_info in target_images:
        image_id = img_info["id"]
        original_name = img_info["name"]

        # EXACT often lists .vsi files but downloads as .zip if it's a multi-file format
        # We rename to .zip locally so we know it needs extraction
        target_name = original_name.replace(".vsi", ".zip")

        file_path = download_target / target_name

        if not file_path.exists():
            print(f"Downloading {original_name} -> {target_name}...")
            images_api.download_image(id=image_id, target_path=file_path)
        else:
            print(f"Skipping {target_name} (already exists).")

    print("Download complete.")


if __name__ == "__main__":
    download_dataset()
