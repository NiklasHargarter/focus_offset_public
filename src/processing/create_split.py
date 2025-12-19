import random
import json
import config
from src.utils.exact_utils import get_exact_image_list


def create_split(force: bool = False) -> None:
    """Generates a 3-way split (train/val/test) based on EXACT image list."""
    if config.SPLIT_FILE.exists() and not force:
        print(f"Split file {config.SPLIT_FILE} already exists. Skipping generation.")
        return

    print("Fetching image list from EXACT...")
    images = get_exact_image_list()
    filenames = sorted([img["name"] for img in images])

    total_files = len(filenames)
    print(f"Found {total_files} total VSI files in EXACT.")

    if total_files == 0:
        print("No files found. Aborting.")
        return

    random.seed(config.SEARCH_SEED)
    random.shuffle(filenames)

    num_test = int(total_files * config.SPLIT_RATIO)
    test_files = filenames[:num_test]
    remaining = filenames[num_test:]

    num_val = int(len(remaining) * config.VAL_RATIO)
    val_files = remaining[:num_val]
    train_files = remaining[num_val:]

    split_data = {
        "train": sorted(train_files),
        "val": sorted(val_files),
        "test": sorted(test_files),
        "seed": config.SEARCH_SEED,
        "total": total_files,
    }

    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.SPLIT_FILE, "w") as f:
        json.dump(split_data, f, indent=4)

    print("\n--- 3-Way Split Created ---")
    print(f"Train: {len(train_files)}")
    print(f"Val:   {len(val_files)}")
    print(f"Test:  {len(test_files)}")
    print(f"Saved to: {config.SPLIT_FILE}")


if __name__ == "__main__":
    create_split()
