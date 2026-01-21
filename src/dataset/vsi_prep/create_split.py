import random
import argparse
import json
from src import config
import pickle

from src.dataset.vsi_types import MasterIndex


def create_split(
    dataset_name: str,
    force: bool = False,
    split_ratio: float = 0.3,
    seed: int = 42,
) -> None:
    """Generate image splits (test and train_pool) from master index."""
    split_file = config.get_split_path(dataset_name)
    master_index_path = config.get_master_index_path(dataset_name)

    if split_file.exists() and not force:
        print(f"Split file {split_file} already exists. Skipping generation.")
        return

    if not master_index_path.exists():
        print(
            f"Error: Master index {master_index_path} not found. Run preprocessing first."
        )
        return

    with open(master_index_path, "rb") as f:
        master_index: MasterIndex = pickle.load(f)

    file_registry = master_index.file_registry
    slides = []
    for entry in file_registry:
        slides.append({"name": entry.name, "count": entry.total_samples})

    total_patches = sum(s["count"] for s in slides)
    target_test_count = total_patches * split_ratio

    print(f"Total slides: {len(slides)}, Total patches: {total_patches}")
    print(f"Target test patches: {target_test_count:.0f} ({split_ratio:.1%})")

    random.seed(seed)
    random.shuffle(slides)

    test_files = []
    train_pool_files = []
    current_test_count = 0

    for slide in slides:
        if current_test_count < target_test_count:
            test_files.append(slide["name"])
            current_test_count += slide["count"]
        else:
            train_pool_files.append(slide["name"])

    print(
        f"Test split: {len(test_files)} slides, {current_test_count} patches ({current_test_count / total_patches:.2%})"
    )
    print(
        f"Train pool: {len(train_pool_files)} slides, {total_patches - current_test_count} patches"
    )

    split_data = {
        "test": sorted(test_files),
        "train_pool": sorted(train_pool_files),
        "seed": seed,
        "total_slides": len(file_registry),
        "total_patches": total_patches,
    }

    with open(split_file, "w") as f:
        json.dump(split_data, f, indent=4)

    print(f"Saved splits to {split_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="ZStack_HE")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split_ratio", type=float, default=0.3)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    create_split(
        dataset_name=args.dataset,
        force=args.force,
        seed=args.seed,
        split_ratio=args.split_ratio,
    )
