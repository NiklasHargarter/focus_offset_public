import argparse
import numpy as np
import matplotlib.pyplot as plt
import slideio
import multiprocessing
import os
import time
from pathlib import Path
from tqdm import tqdm
from functools import partial
from src.utils.focus_metrics import compute_brenner_gradient
from src.utils.io_utils import suppress_stderr

def get_best_z_worker(coords, vsi_path, patch_size):
    cx, cy = coords
    with suppress_stderr():
        try:
            slide = slideio.open_slide(str(vsi_path), "VSI")
            scene = slide.get_scene(0)
            num_z = scene.num_z_slices
            best_s, best_z = -1.0, 0
            for z in range(num_z):
                read_size = patch_size if patch_size <= 224 else 224
                img = scene.read_block(rect=(cx - patch_size // 2, cy - patch_size // 2, patch_size, patch_size),
                                       size=(read_size, read_size), slices=(z, z + 1))
                s = compute_brenner_gradient(img)
                if s > best_s:
                    best_s, best_z = s, z
            return best_z
        except Exception:
            return None

def run_local_topography_exp(vsi_path: Path, output_dir: Path):
    print(f"Running Multi-Strategy Focus Analysis: {vsi_path.name}")
    with suppress_stderr():
        slide = slideio.open_slide(str(vsi_path), "VSI")
    scene = slide.get_scene(0)
    width, height = scene.size

    ps_4x = 896
    ps_8x = 1792
    roi_size = ps_8x * 6 # 10752px
    stride_224 = 224
    grid_dim = roi_size // stride_224
    
    start_x = (width - roi_size) // 2
    start_y = (height - roi_size) // 2
    
    locs_224 = []
    for gy in range(grid_dim):
        for gx in range(grid_dim):
            locs_224.append((start_x + gx * stride_224 + stride_224 // 2,
                             start_y + gy * stride_224 + stride_224 // 2))

    num_blocks_4x = roi_size // ps_4x
    locs_block_4x = []
    for gy in range(num_blocks_4x):
        for gx in range(num_blocks_4x):
            locs_block_4x.append((start_x + gx * ps_4x + ps_4x // 2, 
                                  start_y + gy * ps_4x + ps_4x // 2))

    num_blocks_8x = roi_size // ps_8x
    locs_block_8x = []
    for gy in range(num_blocks_8x):
        for gx in range(num_blocks_8x):
            locs_block_8x.append((start_x + gx * ps_8x + ps_8x // 2, 
                                  start_y + gy * ps_8x + ps_8x // 2))

    print(f"ROI Size: {roi_size}px squared ({grid_dim}x{grid_dim} grid)")
    
    print("  -> Reading Reference ROI Image...")
    with suppress_stderr():
        roi_img = scene.read_block(rect=(start_x, start_y, roi_size, roi_size),
                                   size=(roi_size // 8, roi_size // 8))
        roi_img_rgb = roi_img[..., ::-1]

    num_workers = min(multiprocessing.cpu_count(), 16)
    runtimes = {}

    print("  -> Strategy 1: Local Z (224px)...")
    t0 = time.time()
    with multiprocessing.Pool(num_workers) as pool:
        z_224 = list(tqdm(pool.imap(partial(get_best_z_worker, vsi_path=vsi_path, patch_size=224), locs_224), total=len(locs_224), leave=False))
    runtimes['Local'] = time.time() - t0
    z_224_map = np.array(z_224).reshape((grid_dim, grid_dim))

    print("  -> Strategy 2: Sliding 4x (896px)...")
    t0 = time.time()
    with multiprocessing.Pool(num_workers) as pool:
        z_s4 = list(tqdm(pool.imap(partial(get_best_z_worker, vsi_path=vsi_path, patch_size=ps_4x), locs_224), total=len(locs_224), leave=False))
    runtimes['S4'] = time.time() - t0
    z_s4_map = np.array(z_s4).reshape((grid_dim, grid_dim))

    print("  -> Strategy 3: Sliding 8x (1792px)...")
    t0 = time.time()
    with multiprocessing.Pool(num_workers) as pool:
        z_s8 = list(tqdm(pool.imap(partial(get_best_z_worker, vsi_path=vsi_path, patch_size=ps_8x), locs_224), total=len(locs_224), leave=False))
    runtimes['S8'] = time.time() - t0
    z_s8_map = np.array(z_s8).reshape((grid_dim, grid_dim))

    print("  -> Strategy 4: Tiled 4x (896px)...")
    t0 = time.time()
    with multiprocessing.Pool(num_workers) as pool:
        z_t4_raw = list(tqdm(pool.imap(partial(get_best_z_worker, vsi_path=vsi_path, patch_size=ps_4x), locs_block_4x), total=len(locs_block_4x), leave=False))
    runtimes['T4'] = time.time() - t0
    z_t4_map = np.repeat(np.repeat(np.array(z_t4_raw).reshape((num_blocks_4x, num_blocks_4x)), 4, axis=0), 4, axis=1)

    print("  -> Strategy 5: Tiled 8x (1792px)...")
    t0 = time.time()
    with multiprocessing.Pool(num_workers) as pool:
        z_t8_raw = list(tqdm(pool.imap(partial(get_best_z_worker, vsi_path=vsi_path, patch_size=ps_8x), locs_block_8x), total=len(locs_block_8x), leave=False))
    runtimes['T8'] = time.time() - t0
    z_t8_map = np.repeat(np.repeat(np.array(z_t8_raw).reshape((num_blocks_8x, num_blocks_8x)), 8, axis=0), 8, axis=1)

    fig, axes = plt.subplots(4, 2, figsize=(22, 40))
    
    axes[0, 0].imshow(roi_img_rgb)
    axes[0, 0].set_title("0. Physical ROI Image", fontsize=18, pad=20)
    
    im1 = axes[0, 1].imshow(z_224_map, cmap='viridis', interpolation='nearest')
    axes[0, 1].set_title(f"1. Local Tiled (224px)\nTime: {runtimes['Local']:.1f}s", fontsize=18, pad=20)
    plt.colorbar(im1, ax=axes[0, 1], shrink=0.8)
    
    im2 = axes[1, 0].imshow(z_t4_map, cmap='viridis', interpolation='nearest')
    axes[1, 0].set_title(f"2. Coarse Tiled 4x ({ps_4x}px)\nMAE vs S8: {np.mean(np.abs(z_t4_map - z_s8_map)):.2f} | Time: {runtimes['T4']:.1f}s", fontsize=18, pad=20)
    plt.colorbar(im2, ax=axes[1, 0], shrink=0.8)

    im3 = axes[1, 1].imshow(z_s4_map, cmap='viridis', interpolation='nearest')
    axes[1, 1].set_title(f"3. Sliding Super 4x ({ps_4x}px)\nMAE vs S8: {np.mean(np.abs(z_s4_map - z_s8_map)):.2f} | Time: {runtimes['S4']:.1f}s", fontsize=18, pad=20)
    plt.colorbar(im3, ax=axes[1, 1], shrink=0.8)

    im4 = axes[2, 0].imshow(z_t8_map, cmap='viridis', interpolation='nearest')
    axes[2, 0].set_title(f"4. Coarse Tiled 8x ({ps_8x}px)\nMAE vs S8: {np.mean(np.abs(z_t8_map - z_s8_map)):.2f} | Time: {runtimes['T8']:.1f}s", fontsize=18, pad=20)
    plt.colorbar(im4, ax=axes[2, 0], shrink=0.8)

    im5 = axes[2, 1].imshow(z_s8_map, cmap='viridis', interpolation='nearest')
    axes[2, 1].set_title(f"5. Sliding Super 8x ({ps_8x}px)\nContinuous Surface | Time: {runtimes['S8']:.1f}s", fontsize=18, fontweight='bold', pad=20)
    plt.colorbar(im5, ax=axes[2, 1], shrink=0.8)

    residual = z_224_map - z_s8_map
    limit = max(abs(residual.min()), abs(residual.max()), 2)
    im6 = axes[3, 0].imshow(residual, cmap='RdBu_r', vmin=-limit, vmax=limit, interpolation='nearest')
    axes[3, 0].set_title("6. Residual (Local - Sliding 8x)\nRed/Blue Clusters = Topography Signal", fontsize=18, fontweight='bold', pad=20)
    plt.colorbar(im6, ax=axes[3, 0], shrink=0.8)

    axes[3, 1].axis('off')
    text = (
        "Strategy Legend\n\n"
        "1. LOCAL TILED (224px):\n"
        "   - Standard noisy baseline.\n\n"
        "2 & 4. COARSE TILED (4x / 8x):\n"
        "   - Blocks are non-overlapping.\n"
        "   - Fast but creates 'staircase' steps.\n\n"
        "3 & 5. SLIDING SUPER (4x / 8x):\n"
        "   - Overlapping context per patch.\n"
        "   - Provides smooth, continuous surfaces.\n"
        "   - 8x (1792px) is the physics-neutral Truth.\n\n"
        "6. RESIDUAL MAP:\n"
        "   - Highlights spatial patterns in focus drift."
    )
    axes[3, 1].text(0.05, 0.5, text, fontsize=18, verticalalignment='center',
                    bbox=dict(facecolor='ivory', alpha=0.5, edgecolor='gray'))

    for r in range(4):
        for c in range(2):
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])

    plt.suptitle(f"Multi-Strategy Context Analysis\nSlide: {vsi_path.name}", fontsize=28, fontweight='bold', y=0.98)
    plt.subplots_adjust(top=0.94, bottom=0.03, hspace=0.25, wspace=0.15)
    
    out_path = output_dir / f"exp1_multi_strategy_{vsi_path.stem}.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Full Multi-Strategy Analysis saved to: {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("vsi_path", type=str)
    args = parser.parse_args()
    out_dir = Path("experiments/focus_strategy/results/showcase")
    out_dir.mkdir(parents=True, exist_ok=True)
    run_local_topography_exp(Path(args.vsi_path), out_dir)
