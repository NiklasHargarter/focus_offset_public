import argparse
import random
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import slideio
from matplotlib.patches import Rectangle

from shared_datasets.vsi.prep.preprocess import get_tissue_patches, DRY_RUN_MAX_PATCHES
from focus_offset.utils.io_utils import suppress_stderr

# Plotting constants
THUMBNAIL_DOWNSCALE = 32
CANDIDATE_COLOR = "#4CAF50"
DROPPED_COLOR = "#F44336"
RECT_ALPHA = 0.15


def plot_patches(slide_path: Path, downsample: int, patch_size: int, cov: float):
    print(f"Loading {slide_path.name}...")
    with suppress_stderr():
        slide = slideio.open_slide(str(slide_path), "VSI")
    scene = slide.get_scene(0)
    width, height = scene.size
    mid_z = scene.num_z_slices // 2

    print("Generating patches...")
    # Modified to separate candidates and dropped, and then limit candidates
    all_candidates, dropped = get_tissue_patches(
        scene=scene,
        patch_size=patch_size,
        downsample=downsample,
        min_coverage=cov,
        target_downscale=8,
        return_dropped=True,
    )

    # Apply dry run limit to candidates for plotting
    random.shuffle(all_candidates)
    candidates = all_candidates[:DRY_RUN_MAX_PATCHES]

    print(
        f"Found {len(all_candidates)} total valid patches. Plotting {len(candidates)} "
        f"valid patches and {len(dropped)} discarded patches."
    )

    print("Extracting thumbnail for plotting...")
    # Map raw coordinates to thumbnail coordinates
    d_w, d_h = width // THUMBNAIL_DOWNSCALE, height // THUMBNAIL_DOWNSCALE

    with suppress_stderr():
        bg_raw = scene.read_block(
            rect=(0, 0, width, height), size=(d_w, d_h), slices=(mid_z, mid_z + 1)
        )
    background = cv2.cvtColor(bg_raw, cv2.COLOR_BGR2RGB)

    print("Plotting overlay...")
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(background)

    for x_raw, y_raw, _cov in candidates:
        # Soft green for valid candidates
        rect = Rectangle(
            xy=(x_raw / THUMBNAIL_DOWNSCALE, y_raw / THUMBNAIL_DOWNSCALE),
            width=(patch_size * downsample) / THUMBNAIL_DOWNSCALE,
            height=(patch_size * downsample) / THUMBNAIL_DOWNSCALE,
            linewidth=1,
            edgecolor=CANDIDATE_COLOR,
            facecolor=CANDIDATE_COLOR,
            alpha=RECT_ALPHA,
        )
        ax.add_patch(rect)

    for x_raw, y_raw, _cov in dropped:
        # Soft red for dropped candidates
        rect = Rectangle(
            xy=(x_raw / THUMBNAIL_DOWNSCALE, y_raw / THUMBNAIL_DOWNSCALE),
            width=(patch_size * downsample) / THUMBNAIL_DOWNSCALE,
            height=(patch_size * downsample) / THUMBNAIL_DOWNSCALE,
            linewidth=1,
            edgecolor=DROPPED_COLOR,
            facecolor=DROPPED_COLOR,
            alpha=RECT_ALPHA,
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
    parser.add_argument("--cov", type=float, default=0.7)

    args = parser.parse_args()
    plot_patches(args.slide, args.downsample, args.patch_size, args.cov)
