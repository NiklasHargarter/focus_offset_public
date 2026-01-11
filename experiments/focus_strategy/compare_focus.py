import argparse
import time
import numpy as np
from pathlib import Path
import slideio
import cv2
import pandas as pd
import matplotlib.pyplot as plt

from src.dataset.vsi_prep.preprocess import (
    SlidePreprocessor,
    calculate_stability_metrics,
)
from src.dataset.vsi_types import PreprocessConfig
from src.utils.io_utils import suppress_stderr


def save_benchmark_vis(
    final_patches, height, width, ps, name, tv, outliers, output_dir
):
    """Save a small heatmap for comparison."""
    plt.figure(figsize=(8, 6))
    ds_vis = 16
    h_vis, w_vis = height // ds_vis, width // ds_vis
    heatmap = np.full((h_vis, w_vis), np.nan)
    for p in final_patches:
        dx, dy = p.x // ds_vis, p.y // ds_vis
        dx_e, dy_e = (p.x + ps) // ds_vis, (p.y + ps) // ds_vis
        heatmap[dy:dy_e, dx:dx_e] = p.z
    plt.imshow(heatmap, cmap="viridis")
    plt.title(f"{name}\nTV: {tv:.3f}, Outliers: {outliers:.2%}")
    plt.axis("off")
    plt.savefig(output_dir / f"heatmap_{name}.png")
    plt.close()


def run_benchmark(vsi_path: Path, output_dir: Path):
    metrics = ["brenner"]
    patch_sizes = [128, 512, 1024]
    super_patch_sizes = [512, 1024, 2048]

    results = []

    # 1. Test Per-Patch (Baseline for each size)
    # 1. Test Per-Patch (Baseline for each size)
    for ps in patch_sizes:
        metric = "brenner"
        name = f"{metric}_ps{ps}_patch"
        print(f"Benchmarking: {name}")

        start_time = time.time()
        
        cfg = PreprocessConfig(
            patch_size=ps, 
            focus_patch_size=ps, # Baseline: focus_patch_size = patch_size
            downscale_factor=16, 
            min_tissue_coverage=0.05, 
            dataset_name="benchmark"
        )
        processor = SlidePreprocessor(cfg)
        metadata = processor.process(vsi_path)
        final_patches = metadata.patches
        width, height = metadata.width, metadata.height

        duration = time.time() - start_time
        tv, outliers = calculate_stability_metrics(
            final_patches, width, height, ps
        )
        throughput = len(final_patches) / duration if duration > 0 else 0

        results.append(
            {
                "strategy": name,
                "metric": metric,
                "patch_size": ps,
                "super_ps": ps,
                "approach": "patch",
                "duration_sec": duration,
                "total_variation": tv,
                "outlier_ratio": outliers,
                "patch_count": len(final_patches),
                "throughput": throughput,
            }
        )
        save_benchmark_vis(
            final_patches, height, width, ps, name, tv, outliers, output_dir
        )

    # 2. Test Super-Patch combinations
    for ps in patch_sizes:
        for fps in super_patch_sizes:
            if fps < ps:
                continue
            
            metric = "brenner"
            name = f"{metric}_ps{ps}_super{fps}"
            print(f"Benchmarking: {name}")

            start_time = time.time()
            
            cfg = PreprocessConfig(
                patch_size=ps, 
                focus_patch_size=fps, 
                downscale_factor=16, 
                min_tissue_coverage=0.05, 
                dataset_name="benchmark"
            )
            processor = SlidePreprocessor(cfg)
            metadata = processor.process(vsi_path)
            final_patches = metadata.patches
            width, height = metadata.width, metadata.height

            duration = time.time() - start_time
            tv, outliers = calculate_stability_metrics(
                final_patches, width, height, ps
            )
            throughput = len(final_patches) / duration if duration > 0 else 0

            results.append(
                {
                    "strategy": name,
                    "metric": metric,
                    "patch_size": ps,
                    "super_ps": fps,
                    "approach": "super",
                    "duration_sec": duration,
                    "total_variation": tv,
                    "outlier_ratio": outliers,
                    "patch_count": len(final_patches),
                    "throughput": throughput,
                }
            )
            save_benchmark_vis(
                final_patches, height, width, ps, name, tv, outliers, output_dir
            )

    df = pd.DataFrame(results)
    df.to_csv(output_dir / "focus_benchmark_results.csv", index=False)
    print(f"Benchmark complete. Results saved to {output_dir}")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vsi", type=str, required=True, help="Path to a VSI slide")
    args = parser.parse_args()

    output_dir = Path("experiments/focus_strategy/results")
    output_dir.mkdir(exist_ok=True)

    run_benchmark(Path(args.vsi), output_dir)
