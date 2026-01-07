import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import re
import sys
import random

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent))

import config
from src.dataset.jiang2018 import Jiang2018Dataset


def visualize_segment(samples: list, output_path: Path, stack_title: str):
    """
    Visualize a single segment's stack.
    samples: List of {"path": ..., "offset": ...}
    """
    # Sort samples by offset
    samples = sorted(samples, key=lambda x: x["offset"])

    # Prepare data for plotting
    plot_data = []
    for s in samples:
        path = s["path"]
        offset = s["offset"]
        abs_nm = s["abs_nm"]
        score = s["score"]

        img = cv2.imread(str(path))
        if img is None:
            continue

        plot_data.append(
            {
                "image": img,
                "rel_offset": offset,
                "abs_nm": abs_nm,
                "score": score,
                "is_best": (abs(offset) < 1e-7),
            }
        )

    num_imgs = len(plot_data)
    if num_imgs == 0:
        return

    cols = 5
    rows = (num_imgs + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    axes = axes.flatten()

    for i, d in enumerate(plot_data):
        ax = axes[i]
        img_rgb = cv2.cvtColor(d["image"], cv2.COLOR_BGR2RGB)

        # Place image in the top part of the axes area
        # Extent: [left, right, bottom, top]
        ax.imshow(img_rgb, extent=[0, 10, 5, 15])

        # Labels in the bottom part of the SAME axes area
        label = f"Off: {d['rel_offset']:.2f} um\nPos: {d['abs_nm']:.0f} nm\nScore: {d['score']:.0e}"
        ax.text(5, 2.5, label, ha="center", va="center", fontsize=11, linespacing=1.5)

        # Set limits and hide ticks to create the card look
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 15)
        ax.set_xticks([])
        ax.set_yticks([])

        # Style the single shared border
        for spine in ax.spines.values():
            spine.set_visible(True)
            if d["is_best"]:
                spine.set_edgecolor("green")
                spine.set_linewidth(8)
            else:
                spine.set_edgecolor("black")
                spine.set_linewidth(1.5)

    for i in range(num_imgs, len(axes)):
        axes[i].axis("off")

    plt.suptitle(stack_title, fontsize=20, y=0.98)
    # Adjust layout to make room for suptitle and labels
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig(output_path, dpi=100)
    plt.close()
    print(f"Visualization saved to {output_path}")


if __name__ == "__main__":
    # 1. Initialize Dataset (this loads the cache)
    raw_dir = config.get_vsi_raw_dir("Jiang2018")
    dataset = Jiang2018Dataset(raw_dir)

    if not dataset.samples:
        print("No samples found in dataset index. Check data/cache.")
        sys.exit(1)

    # 2. Group samples by Stack and Segment
    # Structure: { (stack_dir_name, seg_id): [sample1, sample2, ...] }
    groups = {}
    for s in dataset.samples:
        path = s["path"]
        parent_name = path.parent.name
        seg_match = re.match(r"(Seg\d+)_", path.name)
        if seg_match:
            seg_id = seg_match.group(1)
            key = (parent_name, seg_id)
            if key not in groups:
                groups[key] = []
            groups[key].append(s)

    # 3. Pick 5 random groups to visualize
    random.seed(42)
    sample_keys = random.sample(list(groups.keys()), min(5, len(groups)))

    output_dir = Path("visualizations/jiang_checks")
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, key in enumerate(sample_keys):
        stack_name, seg_id = key
        samples_subset = groups[key]
        output_file = output_dir / f"cache_check_{i}_{stack_name}_{seg_id}.png"
        visualize_segment(
            samples_subset, output_file, f"Stack: {stack_name} | {seg_id}"
        )
