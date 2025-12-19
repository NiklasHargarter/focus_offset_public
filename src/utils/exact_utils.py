import os
import json
from exact_sync.v1.configuration import Configuration
from exact_sync.v1.api_client import ApiClient
from exact_sync.v1.api.images_api import ImagesApi
from exact_sync.v1.api.image_sets_api import ImageSetsApi
import config


def get_exact_image_list(dataset_name: str = "ZStack_HE") -> list[dict]:
    """Queries EXACT for the list of images and returns them. Results are cached."""
    cache_file = config.CACHE_DIR / "exact_images.json"

    if cache_file.exists():
        with open(cache_file, "r") as f:
            return json.load(f)

    configuration = Configuration()
    configuration.username = "niklas.hargarter"
    configuration.password = os.environ.get("EXACT_PASSWORD")
    if configuration.password is None:
        raise ValueError("Environment variable 'EXACT_PASSWORD' is not set.")
    configuration.host = "https://exact.hs-flensburg.de"

    client = ApiClient(configuration)
    image_sets_api = ImageSetsApi(client)
    images_api = ImagesApi(client)

    print("Querying EXACT API for image list...")
    image_sets = image_sets_api.list_image_sets(name__contains=dataset_name)

    if len(image_sets.results) == 0:
        raise ValueError(f"No dataset found matching '{dataset_name}'")

    target_set = image_sets.results[0]
    print(f"Dataset found: {target_set.name}, Images: {len(target_set.images)}")

    target_images = []
    print("Resolving image details...")
    for image_id in target_set.images:
        img_obj = images_api.retrieve_image(id=image_id)
        target_images.append({"id": image_id, "name": img_obj.name})

    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(cache_file, "w") as f:
        json.dump(target_images, f, indent=4)

    return target_images
