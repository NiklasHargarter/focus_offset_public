import argparse
import numpy as np
import matplotlib.pyplot as plt
import slideio
import multiprocessing
import os
from pathlib import Path
from tqdm import tqdm
from functools import partial

from src.utils.focus_metrics import compute_brenner_gradient
from src.utils.io_utils import suppress_stderr
from src.dataset.vsi_prep.preprocess import detect_tissue

def get_z_profile_worker(coords, vsi_path, patch_size, downscale=1):
    """Worker function to get Z-profile for a single patch. Initializes slide internally."""
    cx, cy = coords
    with suppress_stderr():
        try:
            slide = slideio.open_slide(str(vsi_path), "VSI")
            scene = slide.get_scene(0)
            num_z = scene.num_z_slices
            read_size = patch_size // downscale

            scores = []
            for z in range(num_z):
                img = scene.read_block(
                    rect=(cx - patch_size // 2, cy - patch_size // 2, patch_size, patch_size),
                    size=(read_size, read_size),
                    slices=(z, z + 1)
                )
                scores.append(compute_brenner_gradient(img))
            return np.array(scores)
        except Exception:
            return None

def run_showcase(vsi_path: Path, output_dir: Path):
    print(f"🚀 Running Optimized Reasoning Showcase: {vsi_path.name}")

    with suppress_stderr():
        slide = slideio.open_slide(str(vsi_path), "VSI")
    scene = slide.get_scene(0)
    width, height = scene.size
    num_z = scene.num_z_slices

    _, mask = detect_tissue(scene, downscale=16)
    y_idxs, x_idxs = np.where(mask > 0)
    if x_idxs.size == 0:
        print("❌ No tissue found.")
        return

    avg_x, avg_y = int(np.mean(x_idxs) * 16), int(np.mean(y_idxs) * 16)
    cut_x = max(0, min(avg_x - 5000, width - 10000))
    cut_y = max(0, min(avg_y - 5000, height - 10000))
    center_x, center_y = cut_x + 5000, cut_y + 5000

    print("📈 Generating Focus Curves (1x resolution)...")
    sizes = [224, 896, 1792]
    curves = {}
    for s in sizes:

        curves[s] = get_z_profile_worker((center_x, center_y), vsi_path, s, downscale=1)

    print("🗺️ Generating Spatial Maps (Sparse 10x10 Grid)...")
    grid_size = 10
    step = 10000 // grid_size
    locations = []
    for gy in range(grid_size):
        for gx in range(grid_size):
            locations.append((cut_x + gx * step + step // 2, cut_y + gy * step + step // 2))

    num_workers = min(multiprocessing.cpu_count(), 16)

    print(f"  -> Scanning Local Z (224px) using {num_workers} workers...")
    with multiprocessing.Pool(num_workers) as pool:
        local_profiles = list(tqdm(pool.imap(partial(get_z_profile_worker, vsi_path=vsi_path, patch_size=224, downscale=4), locations), total=len(locations), leave=False))

    print(f"  -> Scanning Super Z (1792px) using {num_workers} workers...")
    with multiprocessing.Pool(num_workers) as pool:
        super_profiles = list(tqdm(pool.imap(partial(get_z_profile_worker, vsi_path=vsi_path, patch_size=1792, downscale=4), locations), total=len(locations), leave=False))

    local_profiles = [p if p is not None else np.zeros(num_z) for p in local_profiles]
    super_profiles = [p if p is not None else np.zeros(num_z) for p in super_profiles]

    z_noisy = np.argmax(local_profiles, axis=1).reshape((grid_size, grid_size))
    z_smooth = np.argmax(super_profiles, axis=1).reshape((grid_size, grid_size))

    print("📉 Generating Downscale Jitter Benchmark...")
    downscales = [1, 4, 16]
    ds_results = []

    sample_indices = np.random.choice(len(locations), min(3, len(locations)), replace=False)
    for ds in downscales:
        errors = []
        for idx in sample_indices:

            truth_z = np.argmax(get_z_profile_worker(locations[idx], vsi_path, 1792, downscale=1))

            test_z = np.argmax(get_z_profile_worker(locations[idx], vsi_path, 1792, downscale=ds))
            errors.append(abs(test_z - truth_z))
        ds_results.append(np.mean(errors))

    print("🎨 Stitching Showcase Infographic...")
    fig = plt.figure(figsize=(20, 14))
    plt.suptitle(f"Reasoning Showcase: {vsi_path.name}\nEstablishing Robust Ground Truth for Focus Offset Training", fontsize=22, fontweight='bold', y=0.98)

    ax1 = plt.subplot(2, 3, 1)
    colors = ['#E63946', '#457B9D', '#1D3557']
    for i, s in enumerate(sizes):
        v = curves[s]
        v_norm = (v - v.min()) / (v.max() - v.min() + 1e-9)
        ax1.plot(v_norm, label=f"{s}px Context", color=colors[i], linewidth=2.5 if s==1792 else 1.5)
    ax1.set_title("1. Metric Signal Sharpness\n(Aperture Effect)", fontweight='bold', fontsize=14)
    ax1.set_xlabel("Z-Slice Index", fontsize=12)
    ax1.set_ylabel("Normalized Focus Score", fontsize=12)
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    ax2 = plt.subplot(2, 3, 2)
    im1 = ax2.imshow(z_noisy, cmap='viridis', interpolation='nearest')
    ax2.set_title("2a. Local Focus Mapping (224px)\n'Noisy and Inconsistent'", fontweight='bold', fontsize=14)
    plt.colorbar(im1, ax=ax2, label="Best Z-Slice")

    ax3 = plt.subplot(2, 3, 3)
    im2 = ax3.imshow(z_smooth, cmap='viridis', interpolation='nearest')
    ax3.set_title("2b. Super-Patch Mapping (1792px)\n'Spatially Stable Truth'", fontweight='bold', fontsize=14)
    plt.colorbar(im2, ax=ax3, label="Best Z-Slice")

    ax4 = plt.subplot(2, 1, 2)
    bars = ax4.bar([f"{d}x Downscale" for d in downscales], ds_results, color=['#1D3557', '#457B9D', '#A8DADC'], width=0.6)
    ax4.set_title("3. Reliability across Spatial Resolutions (Context: 1792px)", fontweight='bold', fontsize=14)
    ax4.set_ylabel("Mean Z-Error (Deviation from 1x Truth)", fontsize=12)
    for bar in bars:
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height()+0.02, f"{bar.get_height():.2f} Slices", ha='center', fontweight='bold', fontsize=12)
    ax4.set_ylim(0, max(ds_results) + 0.5)
    ax4.grid(axis='y', alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out_path = output_dir / f"showcase_{vsi_path.stem}.png"
    plt.savefig(out_path, dpi=300)
    print(f"✅ Reasoning Showcase saved to: {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("vsi_path", type=str)
    args = parser.parse_args()

    out_dir = Path("experiments/focus_strategy/results/showcase")
    out_dir.mkdir(parents=True, exist_ok=True)

    run_showcase(Path(args.vsi_path), out_dir)
