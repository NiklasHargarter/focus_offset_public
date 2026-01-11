import argparse
import time
import numpy as np
from pathlib import Path
import slideio
import cv2
import pandas as pd
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.dataset.vsi_prep.preprocess import (
    compute_focus_score,
    generate_tissue_mask,
    find_valid_patches,
    calculate_stability_metrics,
)
from src.utils.io_utils import suppress_stderr

def benchmark_global_read(scene, valid_patches, width, height, num_z, downscale, super_patch_size):
    """
    Strategy 1: Read the entire slide (or bounding box) at a downscaled resolution once per Z-level.
    """
    print(f"  [Global Read] factor={downscale}...")
    
    down_w = width // downscale
    down_h = height // downscale
    
    # Safety check for memory (approx check assuming uint8/3 channels even though we convert to gray)
    # 40GB limit just to be safe-ish?
    estimated_bytes = down_w * down_h * 3 
    if estimated_bytes > 20 * 1024**3: 
        print(f"    Skipping Global Read 1/{downscale} (Estimated {estimated_bytes/1024**3:.1f} GB is too large)")
        return None, None
        
    start_time = time.time()
    
    super_patches = []
    for y in range(0, height, super_patch_size):
        for x in range(0, width, super_patch_size):
            super_patches.append((x, y))
            
    num_super = len(super_patches)
    super_scores = np.zeros((num_super, num_z), dtype=np.float32)
    
    for z in range(num_z):
        try:
            # Read global thumbnail for this Z
            img_z = scene.read_block(
                rect=(0, 0, width, height), size=(down_w, down_h), slices=(z, z + 1)
            )
            
            for i, (sx, sy) in enumerate(super_patches):
                # Map coordinates to downscaled image
                dx = int(sx / downscale)
                dy = int(sy / downscale)
                dx_end = int((sx + super_patch_size) / downscale)
                dy_end = int((sy + super_patch_size) / downscale)
                
                dx_end = min(dx_end, down_w)
                dy_end = min(dy_end, down_h)
                
                if dx >= down_w or dy >= down_h:
                    continue
                    
                roi = img_z[dy:dy_end, dx:dx_end]
                if roi.size > 0:
                     # We use default metric='brenner' and binning=1 (since we already downscaled globally)
                    super_scores[i, z] = compute_focus_score(roi)
        except Exception as e:
            print(f"    Error in Global Read z={z}: {e}")
            
    # Resolve best Z
    best_super_z = np.argmax(super_scores, axis=1)
    
    # Map back to patches
    final_patches = _map_super_to_patches(valid_patches, best_super_z, width, height, super_patch_size)
    
    duration = time.time() - start_time
    return final_patches, duration


def benchmark_individual_read(scene, valid_patches, width, height, num_z, downscale, super_patch_size):
    """
    Strategy 2: Read each super-patch individually from the file at the target resolution.
    """
    print(f"  [Individual Read] factor={downscale}...")

    start_time = time.time()

    # Identify active super-patches (those containing valid tissue patches)
    # to avoid reading empty background if possible (though global read reads everything).
    # For fair comparison with Global Read (which reads whole slide), we could read ALL super-patches.
    # But usually "Individual Read" is an optimization to ONLY read relevant areas.
    # However, to be apples-to-apples on "IO Speed vs Resolution", let's read the same grid locations 
    # that global read would process.
    
    super_patches = []
    for y in range(0, height, super_patch_size):
        for x in range(0, width, super_patch_size):
            super_patches.append((x, y))
            
    num_super = len(super_patches)
    super_scores = np.zeros((num_super, num_z), dtype=np.float32)

    # Pre-calculate read sizes
    # We want the read_block `size` param to reflect the downscale
    
    for i, (sx, sy) in enumerate(super_patches):
        sw = min(super_patch_size, width - sx)
        sh = min(super_patch_size, height - sy)
        
        tw = max(1, sw // downscale)
        th = max(1, sh // downscale)
        
        for z in range(num_z):
            try:
                roi = scene.read_block(
                    rect=(sx, sy, sw, sh), size=(tw, th), slices=(z, z + 1)
                )
                if roi.size > 0:
                    super_scores[i, z] = compute_focus_score(roi)
            except Exception as e:
                # print(f"Error reading chunk {i} z={z}: {e}")
                pass

    best_super_z = np.argmax(super_scores, axis=1)
    final_patches = _map_super_to_patches(valid_patches, best_super_z, width, height, super_patch_size)
    
    duration = time.time() - start_time
    return final_patches, duration


def _map_super_to_patches(valid_patches, best_super_z, width, height, focus_patch_size):
    from src.dataset.vsi_types import Patch
    final_patches_list = []
    
    num_super = len(best_super_z)
    
    for px, py in valid_patches:
        sx_idx = px // focus_patch_size
        sy_idx = py // focus_patch_size
        super_cols = (width + focus_patch_size - 1) // focus_patch_size
        super_idx = sy_idx * super_cols + sx_idx

        if super_idx < num_super:
            best_z = int(best_super_z[super_idx])
        else:
            best_z = 0

        final_patches_list.append(Patch(x=px, y=py, z=best_z))
    return final_patches_list

def run_io_benchmark(vsi_path: Path, output_dir: Path):
    with suppress_stderr():
        slide = slideio.open_slide(str(vsi_path), "VSI")
    scene = slide.get_scene(0)
    width, height = scene.size
    num_z = scene.num_z_slices

    print(f"Benchmarking IO for slide: {vsi_path.name}")
    print(f"Dimensions: {width}x{height}, Z-slices: {num_z}")

    # Generate Mask (Standardized)
    ds_mask = 16
    mask_dw, mask_dh = width // ds_mask, height // ds_mask
    mask_img = scene.read_block(rect=(0, 0, width, height), size=(mask_dw, mask_dh))
    gray = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)
    mask = generate_tissue_mask(gray)
    
    patch_size = 224
    valid_patches = find_valid_patches(
        mask, width, height, patch_size, ds_mask, mask_dw, mask_dh
    )
    print(f"Found {len(valid_patches)} valid tissue patches (p{patch_size}).")

    factors = [2, 4, 8, 16, 32]
    # For Individual Read, we might want to skip 1x if it's too slow, but let's try.
    
    results = []
    super_patch_size = 2048 # Fixed optimal size
    
    for f in factors:
        # 1. Global Read
        # User constraint: Never read full slide at full size. 
        # We skip global read for factors < 2 (i.e. 1x) to allow safe benchmarking.
        if f >= 2:
            patches, dur = benchmark_global_read(scene, valid_patches, width, height, num_z, f, super_patch_size)
            if patches:
                tv, outliers = calculate_stability_metrics(patches, width, height, patch_size)
                results.append({
                    "strategy": "global_read",
                    "downscale": f,
                    "duration": dur,
                    "tv": tv,
                    "outliers": outliers,
                    "throughput": len(patches) / dur if dur > 0 else 0
                })
                print(f"    -> Duration: {dur:.2f}s | Outliers: {outliers:.2%}")
        else:
            print(f"  [Global Read] factor={f} skipped (too large)")

        # 2. Individual Read
        # We assume checking EVERY super-patch for apples-to-apples comparison
        # (Though practically we'd only check valid tissue ones, but Global Read pays cost for full slide)
        # Actually, let's optimize Individual Read to ONLY read valid super-patches?
        # That would be the "advantage" of Individual Read.
        # But wait, Global Read reads the whole thumbnail.
        # Let's run Individual Read and see.
        
        patches_ind, dur_ind = benchmark_individual_read(scene, valid_patches, width, height, num_z, f, super_patch_size)
        if patches_ind:
            tv_ind, out_ind = calculate_stability_metrics(patches_ind, width, height, patch_size)
            results.append({
                "strategy": "individual_read",
                "downscale": f,
                "duration": dur_ind,
                "tv": tv_ind,
                "outliers": out_ind,
                "throughput": len(patches_ind) / dur_ind if dur_ind > 0 else 0
            })
            print(f"    -> Duration: {dur_ind:.2f}s | Outliers: {out_ind:.2%}")

    # Save Results
    df = pd.DataFrame(results)
    csv_path = output_dir / "io_benchmark.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved benchmark results to {csv_path}")
    print(df.to_string())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vsi", type=str, required=True)
    args = parser.parse_args()
    
    output_dir = Path("experiments/focus_strategy/results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    run_io_benchmark(Path(args.vsi), output_dir)
