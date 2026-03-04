import argparse
import os
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import slideio

from shared_datasets.vsi.prep.preprocess import get_tissue_patches
from focus_offset.utils.io_utils import suppress_stderr


def plot_patches(
    slide_path: Path, downsample: int, patch_size: int, stride_dist: int, cov: float
):
    print(f"Loading {slide_path.name}...")
    with suppress_stderr():
        slide = slideio.open_slide(str(slide_path), "VSI")
    scene = slide.get_scene(0)
    width, height = scene.size
    mid_z = scene.num_z_slices // 2

    print("Generating patches...")
    candidates, dropped = get_tissue_patches(
        scene=scene,
        patch_size_raw=patch_size * downsample,
        stride=stride_dist,
        min_coverage=cov,
        return_dropped=True,
    )
    print(
        f"Found {len(candidates)} valid patches and {len(dropped)} discarded patches."
    )

    print("Extracting thumbnail for plotting...")
    # Get a very fast, fairly clean thumbnail
    thumb_scale = 32
    d_w = width // thumb_scale
    d_h = height // thumb_scale

    with suppress_stderr():
        bg_raw = scene.read_block(
            rect=(0, 0, width, height), size=(d_w, d_h), slices=(mid_z, mid_z + 1)
        )
    background = cv2.cvtColor(bg_raw, cv2.COLOR_BGR2RGB)

    print("Plotting overlay...")
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(background)

    # Calculate bounding boxes mapped down to the thumbnail layer
    extent_raw = patch_size * downsample
    for x_raw, y_raw, _cov in candidates:
        tx = x_raw / thumb_scale
        ty = y_raw / thumb_scale
        te = extent_raw / thumb_scale

        # Soft green for valid candidates
        rect = plt.Rectangle(
            (tx, ty),
            te,
            te,
            fill=True,
            facecolor="#4CAF50",
            edgecolor="none",
            alpha=0.15,
        )
        ax.add_patch(rect)

    for x_raw, y_raw, _cov in dropped:
        tx = x_raw / thumb_scale
        ty = y_raw / thumb_scale
        te = extent_raw / thumb_scale

        # Soft red for dropped candidates
        rect = plt.Rectangle(
            (tx, ty),
            te,
            te,
            fill=True,
            facecolor="#F44336",
            edgecolor="none",
            alpha=0.15,
        )
        ax.add_patch(rect)

    ax.set_title(
        f"{slide_path.name} | coverage >= {cov} | {len(candidates)} valid / {len(dropped)} dropped"
    )
    ax.axis("off")

    out_file = f"patch_overlay_{slide_path.stem}.png"
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    print(f"Saved visualization to {out_file}")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--slide", type=Path, required=True, help="Path to a single .vsi slide"
    )
    parser.add_argument("--downsample", type=int, default=2)
    parser.add_argument("--patch_size", type=int, default=224)
    parser.add_argument("--stride", type=int, default=448)  # 224 * 2
    parser.add_argument("--cov", type=float, default=0.7)

    args = parser.parse_args()
    plot_patches(args.slide, args.downsample, args.patch_size, args.stride, args.cov)
