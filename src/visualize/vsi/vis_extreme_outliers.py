import argparse
import numpy as np
import cv2
import slideio
from pathlib import Path
import matplotlib.pyplot as plt
from src import config
from src.utils.io_utils import suppress_stderr
from src.dataset.vsi_prep.preprocess import load_master_index
from src.utils.focus_metrics import compute_brenner_gradient


def visualize_extreme_outliers(
    dataset_name, patch_size_raw, threshold=5, limit_slides=3
):
    master = load_master_index(dataset_name, patch_size=patch_size_raw)
    if master is None:
        print(f"Error: Master index not found for {dataset_name}")
        return

    vis_root = config.get_vis_dir("outliers", dataset_name, patch_size=patch_size_raw)

    for i, slide_meta in enumerate(master.file_registry):
        if i >= limit_slides:
            break

        print(f"[{i + 1}/{limit_slides}] Analyzing {slide_meta.name}...")

        patches = slide_meta.patches  # [x, y, z]
        num_z = slide_meta.num_z
        z_levels = patches[:, 2]
        median_z = np.median(z_levels)

        # Identify extreme outliers
        diffs = np.abs(z_levels - median_z)
        outlier_indices = np.where(diffs > threshold)[0]

        if len(outlier_indices) == 0:
            print(f"  No outliers found for {slide_meta.name} (threshold={threshold})")
            continue

        # Sort outliers by deviance
        outlier_indices = outlier_indices[np.argsort(diffs[outlier_indices])[::-1]]
        # Take top 5
        top_outliers = outlier_indices[:5]

        # Open slide for visualization
        vsi_path = config.get_vsi_raw_dir(dataset_name) / Path(slide_meta.name)
        with suppress_stderr():
            slide = slideio.open_slide(str(vsi_path), "VSI")
            scene = slide.get_scene(0)

        # 1. Slide Overview with Outliers
        raw_w, raw_h = scene.size
        vis_downsample = max(1, raw_w // 1200)
        vis_w, vis_h = raw_w // vis_downsample, raw_h // vis_downsample

        with suppress_stderr():
            thumb = scene.read_block(
                rect=(0, 0, raw_w, raw_h),
                size=(vis_w, vis_h),
                slices=(int(median_z), int(median_z + 1)),
            )
        thumb = cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB)

        fig = plt.figure(figsize=(24, 12))
        gs = fig.add_gridspec(len(top_outliers), 10)

        # Overview axis
        ax_ov = fig.add_subplot(gs[:, :3])
        ax_ov.imshow(thumb)
        # Plot all patches as small green dots
        ax_ov.scatter(
            patches[:, 0] / vis_downsample,
            patches[:, 1] / vis_downsample,
            c="green",
            s=1,
            alpha=0.3,
        )
        # Plot outliers as larger red X
        ax_ov.scatter(
            patches[top_outliers, 0] / vis_downsample,
            patches[top_outliers, 1] / vis_downsample,
            c="red",
            s=40,
            marker="x",
            label="Extreme Outliers",
        )
        ax_ov.set_title(f"Slide Overview: {slide_meta.name}\n(Median Z={median_z:.1f})")
        ax_ov.axis("off")
        ax_ov.legend()

        # 2. Z-Stack Strips for top outliers
        for row_idx, out_idx in enumerate(top_outliers):
            px, py, pz = patches[out_idx]

            # Request 7-slice window around pz
            z_start = max(0, pz - 3)
            z_end = min(num_z, pz + 4)

            with suppress_stderr():
                # Read raw patch size (or slightly larger for context)
                ds = master.config_state.downsample_factor
                raw_p = patch_size_raw * ds
                z_stack = scene.read_block(
                    rect=(int(px), int(py), int(raw_p), int(raw_p)),
                    size=(patch_size_raw, patch_size_raw),
                    slices=(int(z_start), int(z_end)),
                )

            # Ensure z_stack is 4D
            if z_stack.ndim == 3:
                z_stack = np.expand_dims(z_stack, axis=0)

            # Display slices
            num_slices = z_stack.shape[0]
            for s_idx in range(num_slices):
                ax = fig.add_subplot(gs[row_idx, 3 + s_idx])
                img = cv2.cvtColor(z_stack[s_idx], cv2.COLOR_BGR2RGB)
                ax.imshow(img)
                current_z = int(z_start + s_idx)

                title = f"Z={current_z}"
                if current_z == pz:
                    title += " (PEAK)"
                    for spine in ax.spines.values():
                        spine.set_edgecolor("red")
                        spine.set_linewidth(3)

                # Compute Brenner for label
                score = compute_brenner_gradient(z_stack[s_idx])
                ax.set_title(f"{title}\nSc:{score:.0f}", fontsize=8)
                ax.axis("off")

        plt.tight_layout()
        out_path = vis_root / f"{vsi_path.stem}_extreme_outliers.png"
        plt.savefig(out_path, dpi=120)
        plt.close()
        print(f"  Saved outlier report to: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="ZStack_HE")
    parser.add_argument("--patch_size", type=int, default=224)
    parser.add_argument("--threshold", type=int, default=5)
    parser.add_argument("--limit", type=int, default=3)
    args = parser.parse_args()

    visualize_extreme_outliers(
        args.dataset, args.patch_size, args.threshold, args.limit
    )
