import argparse
import pandas as pd
import numpy as np
import slideio
import sys
from pathlib import Path
from tqdm import tqdm
import random

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.utils.io_utils import suppress_stderr
from src.utils.focus_metrics import compute_brenner_gradient
from src.dataset.vsi_prep.preprocess import detect_tissue

def run_stability_benchmark(vsi_path: Path, output_dir: Path):
    """
    Benchmark to demonstrate how focus stability (Z-jitter) improves
    with larger context (Super-Patches).
    """
    print(f"Benchmarking Focus Stability: {vsi_path.name}")

    with suppress_stderr():
        slide = slideio.open_slide(str(vsi_path), "VSI")
    scene = slide.get_scene(0)
    width, height = scene.size
    num_z = scene.num_z_slices

    print("Finding tissue (4x)...")
    _, mask = detect_tissue(scene, downscale=4)

    y_idxs, x_idxs = np.where(mask > 0)
    if x_idxs.size < 50:
        print("Not enough tissue found.")
        return

    indices = np.random.choice(range(x_idxs.size), size=min(30, x_idxs.size), replace=False)
    sample_coords = [(int(x_idxs[i] * 4), int(y_idxs[i] * 4)) for i in indices]

    context_sizes = [224, 448, 896, 1792]

    downscale = 4

    print("Establishing Ground Truth (1792px @ 1x)...")
    ground_truth = []
    for cx, cy in tqdm(sample_coords, desc="GT Sync", leave=False):
        best_gt_score = -1.0
        gt_z = 0
        read_size = 1792 // 1
        for z in range(num_z):
            img = scene.read_block(rect=(cx-896, cy-896, 1792, 1792), size=(read_size, read_size), slices=(z, z+1))
            score = compute_brenner_gradient(img)
            if score > best_gt_score:
                best_gt_score, gt_z = score, z
        ground_truth.append(gt_z)

    results = []
    for i, (cx, cy) in enumerate(tqdm(sample_coords, desc="Benchmarking")):
        gt_z = ground_truth[i]

        for size in context_sizes:

            read_px = size // downscale
            best_score = -1.0
            best_z = 0

            for z in range(num_z):
                img = scene.read_block(rect=(cx-size//2, cy-size//2, size, size), size=(read_px, read_px), slices=(z, z+1))
                score = compute_brenner_gradient(img)
                if score > best_score:
                    best_score, best_z = score, z

            results.append({
                "Context_Size": size,
                "Z_Error": abs(best_z - gt_z),
                "Exact_Hit": 1 if best_z == gt_z else 0
            })

    df = pd.DataFrame(results)
    stats = df.groupby("Context_Size").agg({
        "Z_Error": ["mean", "max"],
        "Exact_Hit": "mean"
    }).reset_index()
    stats.columns = ["Context_Size", "MAE", "Max_Error", "Accuracy"]

    print("\n--- Stability Benchmark Results ---")
    print("Reference: 1792px Super-Patch @ 1x Resolution")
    print(stats.to_string(index=False))

    out_file = output_dir / f"bench_stability_{vsi_path.stem}.csv"
    stats.to_csv(out_file, index=False)
    print(f"\nSaved summary to {out_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("vsi_path", type=str)
    args = parser.parse_args()

    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    run_stability_benchmark(Path(args.vsi_path), out_dir)
