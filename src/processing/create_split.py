import random
import argparse
import json
import config
from src.utils.exact_utils import get_exact_image_list


def create_split(dataset_name: str = config.DATASET_NAME, force: bool = False) -> None:
    """Generate image splits from EXACT image list."""
    split_file = config.get_split_path(dataset_name)

    if split_file.exists() and not force:
        print(f"Split file {split_file} already exists. Skipping generation.")
        return

    print(f"Fetching image list for {dataset_name} from EXACT...")
    images = get_exact_image_list(dataset_name=dataset_name)
    filenames = sorted([img["name"] for img in images])

    total_files = len(filenames)
    print(f"Found {total_files} total VSI files for {dataset_name} in EXACT.")

    if total_files == 0:
        print("No files found. Aborting.")
        return

    if dataset_name == "ZStack_HE":
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
    else:
        split_data = {
            "train": [],
            "val": [],
            "test": sorted(filenames),
            "seed": config.SEARCH_SEED,
            "total": total_files,
        }

    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(split_file, "w") as f:
        json.dump(split_data, f, indent=4)

    print(f"\n--- Split Created for {dataset_name} ---")
    if dataset_name == "ZStack_HE":
        print(f"Train: {len(train_files)}")
        print(f"Val:   {len(val_files)}")
    print(f"Test:  {len(split_data['test'])}")
    print(f"Saved to: {split_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=config.DATASET_NAME)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    create_split(dataset_name=args.dataset, force=args.force)
