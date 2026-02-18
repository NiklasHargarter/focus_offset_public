import argparse
import json
import random

from src import config
from src.dataset.vsi_prep.preprocess import load_master_index


def create_split(
    dataset_name: str,
    stride: int = config.STRIDE,
    downsample_factor: int = config.DOWNSAMPLE_FACTOR,
    min_tissue_coverage: float = config.MIN_TISSUE_COVERAGE,
    force: bool = False,
    split_ratio: float = 0.3,
    seed: int = 42,
) -> None:
    """Generate image splits (test and train_pool) from master index."""
    split_file = config.get_split_path(dataset_name)

    if split_file.exists() and not force:
        print(f"Split file {split_file} already exists. Skipping generation.")
        return

    master_index = load_master_index(
        dataset_name, stride, downsample_factor, min_tissue_coverage
    )

    if master_index is None:
        print(
            f"Error: Master index/manifest for {dataset_name} not found. Run preprocessing first."
        )
        return

    file_registry = master_index.file_registry
    slides = []
    for entry in file_registry:
        slides.append(
            {
                "name": entry.name,
                "samples": entry.total_samples,
                "patches": entry.patch_count,
            }
        )

    total_samples = sum(s["samples"] for s in slides)
    total_patches = sum(s["patches"] for s in slides)
    target_test_samples = total_samples * split_ratio

    print(f"Total slides: {len(slides)}")
    print(
        f"Total patches: {total_patches} | Total samples (patches * Z): {total_samples}"
    )
    print(
        f"Target test samples: {target_test_samples:.0f} ({split_ratio:.1%} of samples)"
    )

    random.seed(seed)
    random.shuffle(slides)

    test_files = []
    train_pool_files = []
    current_test_samples = 0

    for slide in slides:
        if current_test_samples < target_test_samples:
            test_files.append(slide["name"])
            current_test_samples += slide["samples"]
        else:
            train_pool_files.append(slide["name"])

    print(
        f"Test split: {len(test_files)} slides, {current_test_samples} samples ({current_test_samples / total_samples:.2%})"
    )
    print(
        f"Train pool: {len(train_pool_files)} slides, {total_samples - current_test_samples} samples"
    )

    split_data = {
        "test": sorted(test_files),
        "train_pool": sorted(train_pool_files),
        "seed": seed,
        "total_slides": len(file_registry),
        "total_patches": total_patches,
        "total_samples": total_samples,
    }

    with open(split_file, "w") as f:
        json.dump(split_data, f, indent=4)

    print(f"Saved splits to {split_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=config.DATASET_NAME)
    parser.add_argument("--stride", type=int, default=config.STRIDE)
    parser.add_argument(
        "--downsample_factor", type=int, default=config.DOWNSAMPLE_FACTOR
    )
    parser.add_argument(
        "--min_tissue_coverage", type=float, default=config.MIN_TISSUE_COVERAGE
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split_ratio", type=float, default=0.3)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    create_split(
        dataset_name=args.dataset,
        stride=args.stride,
        downsample_factor=args.downsample_factor,
        min_tissue_coverage=args.min_tissue_coverage,
        force=args.force,
        seed=args.seed,
        split_ratio=args.split_ratio,
    )
