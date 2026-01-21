import argparse
import multiprocessing
from pathlib import Path
from functools import partial

import numpy as np
import matplotlib.pyplot as plt
import slideio
import cv2
from tqdm import tqdm

from src.utils.focus_metrics import compute_brenner_gradient
from src.utils.io_utils import suppress_stderr
from src.dataset.vsi_prep.preprocess import detect_tissue
from src import config


def get_best_z_legacy_worker(coords, vsi_path, native_patch_size, proc_scale):
    """
    Worker function to find the best Z for a fixed physical area,
    but downscaling the patch internally before computing Brenner.
    """
    cx, cy = coords
    with suppress_stderr():
        try:
            slide = slideio.open_slide(str(vsi_path), "VSI")
            scene = slide.get_scene(0)
            num_z = scene.num_z_slices

            best_s, best_z = -1.0, 0
            proc_size = native_patch_size // proc_scale

            for z in range(num_z):
                # Read at native resolution
                img = scene.read_block(
                    rect=(
                        cx - native_patch_size // 2,
                        cy - native_patch_size // 2,
                        native_patch_size,
                        native_patch_size,
                    ),
                    size=(native_patch_size, native_patch_size),
                    slices=(z, z + 1),
                )

                # Downscale for "fast" processing
                if proc_scale > 1:
                    img = cv2.resize(
                        img, (proc_size, proc_size), interpolation=cv2.INTER_AREA
                    )

                s = compute_brenner_gradient(img)
                if s > best_s:
                    best_s, best_z = s, z
            return best_z
        except Exception:
            return None


def run_legacy_precision_experiment(vsi_path: Path, output_dir: Path):
    print(" Experiment: Legacy Internal Downscaling Precision Loss")
    print(f"Slide: {vsi_path.name}")

    with suppress_stderr():
        slide = slideio.open_slide(str(vsi_path), "VSI")
    scene = slide.get_scene(0)
    width, height = scene.size

    # 1. Select ROI with tissue
    print("  -> Scanning for tissue...")
    _, mask = detect_tissue(scene, downscale_factor=16)
    y_idxs, x_idxs = np.where(mask > 0)

    if x_idxs.size == 0:
        print(" No tissue found.")
        return

    avg_x, avg_y = int(np.mean(x_idxs) * 16), int(np.mean(y_idxs) * 16)
    roi_size = 10752  # Same ROI as previous experiment for comparability
    start_x = max(0, min(avg_x - roi_size // 2, width - roi_size))
    start_y = max(0, min(avg_y - roi_size // 2, height - roi_size))

    native_patch_size = 224
    grid_dim = 24
    stride = roi_size // grid_dim

    locs = []
    for gy in range(grid_dim):
        for gx in range(grid_dim):
            locs.append(
                (
                    start_x + gx * stride + stride // 2,
                    start_y + gy * stride + stride // 2,
                )
            )

    # Internal processing scales
    # 1x = 224px, 2x = 112px, 4x = 56px, 8x = 28px
    scales = [1, 2, 4, 8]
    maps = {}

    num_workers = min(multiprocessing.cpu_count(), 16)

    # Reference Truth (1x scale)
    print(f"  -> Generating Reference (1x, {native_patch_size}px)...")
    with multiprocessing.Pool(num_workers) as pool:
        ref_results = list(
            tqdm(
                pool.imap(
                    partial(
                        get_best_z_legacy_worker,
                        vsi_path=vsi_path,
                        native_patch_size=native_patch_size,
                        proc_scale=1,
                    ),
                    locs,
                ),
                total=len(locs),
                leave=False,
            )
        )
    ref_map = np.array([r if r is not None else 0 for r in ref_results]).reshape(
        (grid_dim, grid_dim)
    )
    maps[1] = ref_map

    for sc in scales[1:]:
        proc_size = native_patch_size // sc
        print(
            f"  -> Testing Legacy Scale: {sc}x Downscale ({proc_size}px processing)..."
        )

        with multiprocessing.Pool(num_workers) as pool:
            results = list(
                tqdm(
                    pool.imap(
                        partial(
                            get_best_z_legacy_worker,
                            vsi_path=vsi_path,
                            native_patch_size=native_patch_size,
                            proc_scale=sc,
                        ),
                        locs,
                    ),
                    total=len(locs),
                    leave=False,
                )
            )

        z_map = np.array([r if r is not None else 0 for r in results]).reshape(
            (grid_dim, grid_dim)
        )
        maps[sc] = z_map

    # 2. Plotting
    print("  -> Generating Comparison Visualizations...")
    fig, axes = plt.subplots(1, 4, figsize=(24, 7), constrained_layout=True)
    fig.suptitle(
        f"Legacy Internal Downscaling Impact on Focus Estimates\nFixed 224px Native Area | Slide: {vsi_path.name}",
        fontsize=22,
        fontweight="bold",
        y=1.12,
    )

    for i, sc in enumerate(scales):
        ax = axes[i]
        z_map = maps[sc]
        proc_size = native_patch_size // sc
        mae = np.mean(np.abs(z_map - ref_map))

        im = ax.imshow(
            z_map,
            cmap="viridis",
            interpolation="nearest",
            vmin=0,
            vmax=max(scene.num_z_slices, 1),
        )

        res_label = "Native Res (1x)" if sc == 1 else f"{sc}x Internal Downscale"
        ax.set_title(
            f"{res_label}\nProc Size: {proc_size}px\nMAE vs 1x: {mae:.2f}",
            fontsize=14,
            pad=10,
        )
        ax.axis("off")

    cbar = fig.colorbar(im, ax=axes, shrink=0.7, pad=0.02)
    cbar.set_label("Optimal Z-slice Index", rotation=270, labelpad=15)
    out_path = output_dir / f"exp02_legacy_downscale_{vsi_path.stem}.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()

    print("\n" + "=" * 50)
    print("LEGACY DOWNSCALE PRECISION SUMMARY")
    print("=" * 50)
    print(f"{'Downscale':<10} | {'Proc Size':<10} | {'MAE (Z-slices)':<15}")
    print("-" * 50)
    for sc in scales:
        z_map = maps[sc]
        mae = np.mean(np.abs(z_map - ref_map))
        proc_size = native_patch_size // sc
        print(f"{sc:>8}x | {proc_size:>8}px | {mae:14.2f}")
    print("=" * 50)

    # Generate Markdown Report
    report_path = output_dir / f"exp02_legacy_downscale_{vsi_path.stem}.md"
    with open(report_path, "w") as f:
        f.write("# Experiment Report: Legacy Downscale Precision Loss\n\n")
        f.write(f"- **Slide**: {vsi_path.name}\n")
        f.write(f"- **Native Patch Size**: {native_patch_size}px (1x Area)\n\n")
        f.write("## Precision Results (MAE vs 1x Reference)\n\n")
        f.write("| Downscale | Processing Size | MAE (Z-slices) |\n")
        f.write("| :--- | :--- | :--- |\n")
        for sc in scales:
            z_map = maps[sc]
            mae = np.mean(np.abs(z_map - ref_map))
            proc_size = native_patch_size // sc
            f.write(f"| {sc}x | {proc_size}px | {mae:.2f} |\n")
        f.write(
            f"\n![Comparison Heatmap](exp02_legacy_downscale_{vsi_path.stem}.png)\n"
        )

    print(
        f" Experiment complete. Results saved to:\n   - {out_path}\n   - {report_path}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    default_vsi = config.get_vsi_raw_dir("ZStack_HE") / "001_1_HE_stack.vsi"
    parser.add_argument("--vsi_path", type=str, default=str(default_vsi))
    args = parser.parse_args()

    out_dir = Path("src/experiments/focus_mechanics_v2/results")
    out_dir.mkdir(parents=True, exist_ok=True)

    run_legacy_precision_experiment(Path(args.vsi_path), out_dir)
