import argparse
import numpy as np
import cv2
import slideio
from pathlib import Path
import matplotlib.pyplot as plt
from src import config
from src.dataset.vsi_prep.preprocess import (
    SlidePreprocessor,
    PreprocessConfig,
    fit_plane_robust,
)


def visualize_outliers(
    slide_name, dataset_name="ZStack_HE", patch_size=224, threshold=4.0
):
    raw_dir = config.get_vsi_raw_dir(dataset_name)
    vsi_path = raw_dir / slide_name

    if not vsi_path.exists():
        print(f"Slide not found: {vsi_path}")
        return

    # 1. Detect focus again (Dry Run)
    cfg = PreprocessConfig(
        patch_size=patch_size,
        stride=448,
        downsample_factor=2,
        min_tissue_coverage=0.05,
        dataset_name=dataset_name,
    )

    print(f"Processing {slide_name} for outlier detection...")
    preprocessor = SlidePreprocessor(cfg)
    meta = preprocessor.process(vsi_path)

    patches = meta.patches
    X, Y, Z = patches[:, 0], patches[:, 1], patches[:, 2]

    _, inlier_mask = fit_plane_robust(X, Y, Z, threshold=threshold)
    outlier_mask = ~inlier_mask
    outlier_indices = np.where(outlier_mask)[0]
    inlier_indices = np.where(inlier_mask)[0]

    # 2. Extract slide preview
    slide = slideio.open_slide(str(vsi_path), "VSI")
    scene = slide.get_scene(0)
    w_raw, h_raw = scene.size
    median_z = int(np.median(Z))

    preview_w = 1200
    preview_h = int(h_raw * preview_w / w_raw)
    slide_img = scene.read_block(
        rect=(0, 0, w_raw, h_raw),
        size=(preview_w, preview_h),
        slices=(median_z, median_z + 1),
    )
    slide_img = cv2.cvtColor(slide_img, cv2.COLOR_BGR2RGB)

    # 3. Create Visualization
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(3, 4)

    # Main Slide Plot
    ax_main = fig.add_subplot(gs[:, 0:2])
    ax_main.imshow(slide_img)
    scale = preview_w / w_raw
    ax_main.scatter(
        X[inlier_mask] * scale,
        Y[inlier_mask] * scale,
        c="green",
        s=2,
        label="Inliers",
        alpha=0.3,
    )
    ax_main.scatter(
        X[outlier_mask] * scale,
        Y[outlier_mask] * scale,
        c="red",
        s=15,
        label="Outliers",
        marker="x",
    )
    ax_main.set_title(f"Outlier Distribution: {slide_name}")
    ax_main.legend()
    ax_main.axis("off")

    # Example Patches
    def plot_patches(indices, title_prefix, start_col):
        for i in range(3):
            if i >= len(indices):
                break
            idx = indices[np.random.choice(len(indices))]
            px, py, pz = patches[idx]
            # Read patch at its detected "best" Z
            patch = scene.read_block(
                rect=(int(px), int(py), int(patch_size * 2), int(patch_size * 2)),
                size=(patch_size, patch_size),
                slices=(int(pz), int(pz + 1)),
            )
            ax = fig.add_subplot(gs[i, start_col])
            ax.imshow(cv2.cvtColor(patch, cv2.COLOR_BGR2RGB))
            ax.set_title(f"{title_prefix} (Z={pz})")
            ax.axis("off")

    if len(outlier_indices) > 0:
        print("Extracting sample outliers...")
        plot_patches(outlier_indices, "OUTLIER", 2)

    if len(inlier_indices) > 0:
        print("Extracting sample inliers...")
        plot_patches(inlier_indices, "INLIER", 3)

    plt.tight_layout()
    out_dir = Path("visualizations/outliers")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{vsi_path.stem}_inspection.png"
    plt.savefig(out_path, dpi=120)
    print(f"\nDone! Inspection report saved to: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--slide", type=str, required=True)
    parser.add_argument("--threshold", type=float, default=4.0)
    args = parser.parse_args()

    visualize_outliers(args.slide, threshold=args.threshold)
