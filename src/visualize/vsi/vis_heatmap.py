import argparse
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from src import config
from src.dataset.vsi_prep.preprocess import load_master_index


def save_heatmap(slide, output_dir, patch_size):
    ds = 8
    h, w = slide.height // ds, slide.width // ds
    heatmap = np.full((h, w), np.nan)

    for p in slide.patches:
        x, y, z = p
        dx, dy = int(x / ds), int(y / ds)
        dx_end, dy_end = (
            int((x + patch_size) / ds),
            int((y + patch_size) / ds),
        )
        heatmap[dy:dy_end, dx:dx_end] = z

    plt.figure(figsize=(10, 8))
    plt.imshow(heatmap, cmap="viridis")
    plt.colorbar(label="Z-Slice")
    plt.title(f"Focus Heatmap: {slide.name} (p{patch_size})")
    plt.axis("off")

    out_path = output_dir / f"{Path(slide.name).stem}_heatmap.png"
    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="ZStack_HE")
    parser.add_argument("--patch_size", type=int, default=config.PATCH_SIZE)
    parser.add_argument(
        "--limit", type=int, default=5, help="Number of slides to process (default: 5)"
    )
    args = parser.parse_args()

    master = load_master_index(args.dataset, patch_size=args.patch_size)
    if master is None:
        print(f"Error: Master index not found for {args.dataset}")
        exit(1)

    vis_root = config.get_vis_dir("heatmaps", args.dataset, patch_size=args.patch_size)

    for slide in master.file_registry[: args.limit]:
        save_heatmap(slide, vis_root, master.patch_size)
