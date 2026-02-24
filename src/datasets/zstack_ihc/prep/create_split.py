import json
import random

from src.datasets.zstack_ihc.config import ZStackIHCConfig

def create_split(
    dataset_name: str = "ZStack_IHC",
    split_ratio: float = 0.3,
    seed: int = 42,
) -> None:
    """Generate image splits (test and train_pool) from master index."""
    dataset_cfg = ZStackIHCConfig()
    split_file = dataset_cfg.split_path
    raw_dir = dataset_cfg.raw_dir

    all_files = sorted([f.name for f in raw_dir.glob("*.vsi")])
    if not all_files:
        print(f"Error: No .vsi files found in {raw_dir}.")
        return

    total_slides = len(all_files)
    target_test_slides = max(1, int(total_slides * split_ratio))

    print(f"Total slides found: {total_slides}")
    print(f"Target test slides: {target_test_slides} ({split_ratio:.1%})")

    random.seed(seed)
    random.shuffle(all_files)

    test_files = all_files[:target_test_slides]
    train_pool_files = all_files[target_test_slides:]

    print(f"Test split: {len(test_files)} slides")
    print(f"Train pool: {len(train_pool_files)} slides")

    split_data = {
        "test": sorted(test_files),
        "train_pool": sorted(train_pool_files),
        "seed": seed,
        "total_slides": total_slides,
    }

    with open(split_file, "w") as f:
        json.dump(split_data, f, indent=4)

    print(f"Saved JSON splits to {split_file}")

