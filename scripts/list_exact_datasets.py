import os
from exact_sync.v1.configuration import Configuration
from exact_sync.v1.api_client import ApiClient
from exact_sync.v1.api.image_sets_api import ImageSetsApi

configuration = Configuration()
configuration.username = os.environ.get("EXACT_USERNAME", "niklas.hargarter")
configuration.password = os.environ.get("EXACT_PASSWORD")
configuration.host = "https://exact.hs-flensburg.de"

client = ApiClient(configuration)
image_sets_api = ImageSetsApi(client)

print("Listing image sets...")
image_sets = image_sets_api.list_image_sets()
for res in image_sets.results:
    print(f"- {res.name}")
