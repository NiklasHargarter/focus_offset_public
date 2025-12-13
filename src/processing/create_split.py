import random
import json
import config


def create_split(force: bool = False) -> None:
    """
    Scans the raw VSI directory and generates a train/test split JSON file.
    Does NOT require the files to be pre-processed first.
    """
    if config.SPLIT_FILE.exists() and not force:
        print(f"Split file {config.SPLIT_FILE} already exists. Skipping generation.")
        return

    print(f"Scanning {config.VSI_RAW_DIR} for VSI files...")

    # Use glob to match files, but ensure we work with Path objects if possible or string consistency
    # config.VSI_RAW_DIR is a Path
    files = sorted(list(config.VSI_RAW_DIR.glob("*.vsi")))

    total_files = len(files)
    print(f"Found {total_files} VSI files.")

    if total_files == 0:
        print("No files found. Aborting.")
        return

    # Shuffle and Split
    random.seed(config.SEARCH_SEED)
    random.shuffle(files)

    num_test = int(total_files * config.SPLIT_RATIO)
    test_files = [f.name for f in files[:num_test]]
    train_files = [f.name for f in files[num_test:]]

    split_data = {
        "train": train_files,
        "test": test_files,
        "seed": config.SEARCH_SEED,
        "total": total_files,
    }

    # Save
    with open(config.SPLIT_FILE, "w") as f:
        json.dump(split_data, f, indent=4)

    print("Split created!")
    print(f"Train: {len(train_files)}")
    print(f"Test:  {len(test_files)}")
    print(f"Saved to: {config.SPLIT_FILE}")


if __name__ == "__main__":
    create_split()
