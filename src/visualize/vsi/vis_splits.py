import argparse
import pickle
import json
import matplotlib.pyplot as plt
import numpy as np
import config
from src.dataset.vsi_types import MasterIndex


def analyze_splits(dataset_name, patch_size):
    master_index_path = config.get_master_index_path(
        dataset_name, patch_size=patch_size
    )
    split_path = config.get_split_path(dataset_name)

    if not master_index_path.exists() or not split_path.exists():
        print(
            f"Missing data for {dataset_name} (p{patch_size}). Run preprocessing and split creation first."
        )
        return

    with open(master_index_path, "rb") as f:
        master_index: MasterIndex = pickle.load(f)
    with open(split_path, "r") as f:
        splits = json.load(f)

    # Map filename to metadata
    name_to_metadata = {m.name: m for m in master_index.file_registry}

    # Analyze each split
    split_stats = {}

    # Identify keys that look like splits (lists of filenames)
    split_keys = [
        k
        for k, v in splits.items()
        if isinstance(v, list) and all(isinstance(x, str) for x in v)
    ]

    for key in split_keys:
        filenames = splits[key]
        slide_metadata = [
            name_to_metadata[f] for f in filenames if f in name_to_metadata
        ]

        if not slide_metadata:
            continue

        sample_counts = [m.total_samples for m in slide_metadata]
        patch_counts = [len(m.patches) for m in slide_metadata]

        split_stats[key] = {
            "num_slides": len(slide_metadata),
            "total_samples": sum(sample_counts),
            "total_patches": sum(patch_counts),
            "sample_counts": sample_counts,
            "patch_counts": patch_counts,
            "slide_names": [m.name for m in slide_metadata],
        }

    # Print summary
    print(f"\nAnalysis for dataset: {dataset_name} (patch_size={patch_size})")
    print("-" * 50)
    for key, stats in split_stats.items():
        print(f"Split: {key}")
        print(f"  Slides:  {stats['num_slides']}")
        print(
            f"  Samples: {stats['total_samples']:,} (Total Z-stacks across all patches)"
        )
        print(f"  Patches: {stats['total_patches']:,} (Unique spatial locations)")
        print(
            f"  Average samples per slide: {np.mean(stats['sample_counts']):.1f} ± {np.std(stats['sample_counts']):.1f}"
        )
        print("-" * 20)

    # Visualization
    output_dir = config.VIS_DIR / f"p{patch_size}" / "splits" / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Slide and Sample Counts
    labels = list(split_stats.keys())
    slide_counts = [s["num_slides"] for s in split_stats.values()]
    sample_counts = [s["total_samples"] for s in split_stats.values()]

    fig, ax1 = plt.subplots(figsize=(10, 6))

    x = np.arange(len(labels))
    width = 0.35

    ax1.bar(x - width / 2, slide_counts, width, label="Slides", color="skyblue")
    ax1.set_ylabel("Number of Slides")
    ax1.set_title(f"Split Distribution: {dataset_name} (p{patch_size})")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.legend(loc="upper left")

    ax2 = ax1.twinx()
    ax2.bar(x + width / 2, sample_counts, width, label="Samples", color="salmon")
    ax2.set_ylabel("Total Samples")
    ax2.legend(loc="upper right")

    fig.tight_layout()
    plt.savefig(output_dir / "split_counts.png")
    plt.close()

    # 2. Sample Distribution Boxplot
    plt.figure(figsize=(10, 6))
    plt.boxplot([s["sample_counts"] for s in split_stats.values()], labels=labels)
    plt.ylabel("Samples per Slide")
    plt.title(f"Slide Size Distribution across Splits: {dataset_name} (p{patch_size})")
    plt.savefig(output_dir / "slide_size_dist.png")
    plt.close()

    # 3. Cumulative Sample Count
    plt.figure(figsize=(12, 6))
    for key, stats in split_stats.items():
        sorted_counts = sorted(stats["sample_counts"], reverse=True)
        cumulative = np.cumsum(sorted_counts)
        plt.plot(
            range(1, len(cumulative) + 1),
            cumulative,
            marker="o",
            label=f"{key} (Total: {stats['total_samples']:,})",
        )

    plt.xlabel("Number of Slides")
    plt.ylabel("Cumulative Samples")
    plt.title(f"Cumulative Sample Coverage: {dataset_name} (p{patch_size})")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.savefig(output_dir / "cumulative_samples.png")
    plt.close()

    print(f"\nVisualizations saved to: {output_dir}")

    # 4. Export Markdown Table
    md_content = f"# Split Analysis: {dataset_name} (p{patch_size})\n\n"
    md_content += "| Split | Slides | Samples (Total Z-stacks) | Patches (Spatial Locations) | Avg Samples/Slide |\n"
    md_content += "| :--- | :---: | :---: | :---: | :---: |\n"

    for key, stats in split_stats.items():
        avg = np.mean(stats["sample_counts"])
        std = np.std(stats["sample_counts"])
        md_content += f"| **{key.capitalize()}** | {stats['num_slides']} | {stats['total_samples']:,} | {stats['total_patches']:,} | {avg:,.0f} ± {std:,.0f} |\n"

    docs_dir = config.PROJECT_ROOT / "docs" / f"p{patch_size}"
    docs_dir.mkdir(parents=True, exist_ok=True)
    md_path = docs_dir / f"split_analysis_{dataset_name}.md"
    with open(md_path, "w") as f:
        f.write(md_content)
    print(f"Markdown table saved to: {md_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="ZStack_HE")
    parser.add_argument("--patch_size", type=int, default=config.PATCH_SIZE)
    args = parser.parse_args()

    analyze_splits(args.dataset, args.patch_size)
