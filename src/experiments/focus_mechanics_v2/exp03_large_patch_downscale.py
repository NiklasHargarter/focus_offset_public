import argparse
import time
from pathlib import Path
from functools import partial
import multiprocessing

import numpy as np
import slideio
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.utils.focus_metrics import compute_brenner_gradient
from src.utils.io_utils import suppress_stderr
from src.dataset.vsi_prep.preprocess import detect_tissue
from src import config


def focus_worker(loc, vsi_path, native_size, downscale_factor, num_z):
    """Worker to compute focus score for a specific location and scale."""
    vx, vy = loc
    with suppress_stderr():
        try:
            slide = slideio.open_slide(str(vsi_path), "VSI")
            scene = slide.get_scene(0)

            # The processing size is the native_size // downscale
            proc_size = native_size // downscale_factor

            scores = []
            for z in range(num_z):
                # Read the Z-slice at the requested downscaled size
                img = scene.read_block(
                    rect=(vx, vy, native_size, native_size),
                    size=(proc_size, proc_size),
                    slices=(z, z + 1),
                )
                scores.append(compute_brenner_gradient(img))

            return np.argmax(scores)
        except Exception:
            return None


def run_experiment(vsi_path: Path, native_size: int = 896):
    print(
        f" Experiment: Large Patch Downscaling Precision Loss (Native Area: {native_size}px)"
    )
    print(f"Slide: {vsi_path.name}")

    # 1. Setup
    with suppress_stderr():
        slide = slideio.open_slide(str(vsi_path), "VSI")
    scene = slide.get_scene(0)
    width, height = scene.size
    num_z = scene.num_z_slices

    output_dir = Path("src/experiments/focus_mechanics_v2/results")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 2. Scanning for Region of Interest
    print("  -> Scanning for tissue...")
    _, mask = detect_tissue(scene, downscale_factor=16)

    # Find a middle region with tissue
    grid_dim = 15
    stride = native_size
    valid_locs = []

    # We'll look for a dense tissue area
    for y in range(height // 4, 3 * height // 4, stride):
        for x in range(width // 4, 3 * width // 4, stride):
            mx, my = x // (native_size * 2), y // (native_size * 2)  # check mask
            if my < mask.shape[0] and mx < mask.shape[1] and mask[my, mx]:
                valid_locs.append((x, y))
            if len(valid_locs) >= grid_dim**2:
                break
        if len(valid_locs) >= grid_dim**2:
            break

    if len(valid_locs) < grid_dim**2:
        print("Warning: Could not find enough tissue. Using fallback grid.")
        valid_locs = []
        for y in range(height // 4, height // 4 + grid_dim * stride, stride):
            for x in range(width // 4, width // 4 + grid_dim * stride, stride):
                valid_locs.append((x, y))

    scales = [1, 2, 4, 8, 16]
    maps = {}
    num_workers = min(multiprocessing.cpu_count(), 16)

    for sc in scales:
        proc_size = native_size // sc
        print(f"  -> Testing Scale: {sc}x Downscale ({proc_size}px processing)...")

        t0 = time.time()
        with multiprocessing.Pool(num_workers) as pool:
            results = list(
                tqdm(
                    pool.imap(
                        partial(
                            focus_worker,
                            vsi_path=vsi_path,
                            native_size=native_size,
                            downscale_factor=sc,
                            num_z=num_z,
                        ),
                        valid_locs,
                    ),
                    total=len(valid_locs),
                    leave=False,
                )
            )

        duration = time.time() - t0
        z_map = np.array([r if r is not None else 0 for r in results]).reshape(
            (grid_dim, grid_dim)
        )
        maps[sc] = z_map
        print(f"     Completed in {duration:.1f}s")

    # 3. Visualization
    ref_map = maps[1]
    fig, axes = plt.subplots(1, len(scales), figsize=(18, 5), constrained_layout=True)
    fig.suptitle(
        f"Z-map Precision Study: Effects of Downscaling on {native_size}px Native Area",
        fontsize=16,
        y=1.05,
    )

    for i, i_sc in enumerate(scales):
        z_map = maps[i_sc]
        mae = np.mean(np.abs(z_map - ref_map))
        im = axes[i].imshow(z_map, cmap="viridis", vmin=0, vmax=num_z)
        eff_size = native_size // i_sc
        scale_label = f"Scale: {i_sc}x" if i_sc > 1 else "Native Res (1x)"
        axes[i].set_title(
            f"{scale_label}\n{eff_size}x{eff_size} px\nMAE: {mae:.2f}",
            fontsize=10,
            pad=10,
        )
        axes[i].axis("off")

    cbar = fig.colorbar(im, ax=axes, shrink=0.6, pad=0.02)
    cbar.set_label("Optimal Z-slice Index", rotation=270, labelpad=15)

    out_img = output_dir / f"exp03_large_patch_downscale_{vsi_path.stem}.png"
    plt.savefig(out_img, dpi=200, bbox_inches="tight")
    plt.close()

    # 4. Global Markdown Report
    report_path = output_dir / f"exp03_large_patch_downscale_{vsi_path.stem}.md"
    with open(report_path, "w") as f:
        f.write("# Experiment Report: Large Patch Downscale Study\n\n")
        f.write(f"- **Slide**: {vsi_path.name}\n")
        f.write(f"- **Fixed Native Area**: {native_size} x {native_size} px\n\n")
        f.write("## Precision Results\n\n")
        f.write(
            "| Downscale | Effective Patch Size | MAE (vs 1x) | Stability (Std) |\n"
        )
        f.write("| :--- | :--- | :--- | :--- |\n")
        for sc in scales:
            z_map = maps[sc]
            mae = np.mean(np.abs(z_map - ref_map))
            eff_size = native_size // sc
            f.write(
                f"| {sc}x | {eff_size} x {eff_size} px | {mae:.2f} | {np.std(z_map):.2f} |\n"
            )

        f.write(
            f"\n![Comparison Heatmap](exp03_large_patch_downscale_{vsi_path.stem}.png)\n"
        )
        f.write("\n## Analysis\n")
        f.write(
            f"This experiment tests if utilizing a larger context area ({native_size}px) "
            f"can mitigate the precision loss of downscaled focus estimation. "
            "Compare these results to the 224px study to see if the error scales linearly."
        )

    print(
        f"\n Experiment complete. Results saved to:\n   - {out_img}\n   - {report_path}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    default_vsi = config.get_vsi_raw_dir("ZStack_HE") / "001_1_HE_stack.vsi"
    parser.add_argument("--vsi_path", type=str, default=str(default_vsi))
    parser.add_argument("--native_size", type=int, default=896)  # 4 * 224
    args = parser.parse_args()
    run_experiment(Path(args.vsi_path), args.native_size)
