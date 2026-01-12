import sys
from pathlib import Path
import time
import json
import numpy as np
import matplotlib.pyplot as plt

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.dataset.vsi_prep.preprocess import (  # noqa: E402
    SlidePreprocessor,
    calculate_stability_metrics,
)
from src.dataset.vsi_types import PreprocessConfig  # noqa: E402


def save_3d_heatmap(final_patches, height, width, ps, title, output_path):
    """Save a 3D surface-like plot of the focal plane."""
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")

    xs = [p.x for p in final_patches]
    ys = [p.y for p in final_patches]
    zs = [p.z for p in final_patches]

    # Use scatter for 3D vis as it's easier for sparse/patchy data
    sc = ax.scatter(xs, ys, zs, c=zs, cmap="viridis", s=2)

    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z (Slice)")
    ax.invert_yaxis()

    plt.colorbar(sc, ax=ax, label="Z Slice")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def save_2d_heatmap(final_patches, height, width, ps, title, output_path):
    """Save a high-quality 2D heatmap."""
    plt.figure(figsize=(10, 8))

    ds_vis = 32
    h_vis, w_vis = height // ds_vis, width // ds_vis
    heatmap = np.full((h_vis, w_vis), np.nan)

    for p in final_patches:
        dx, dy = p.x // ds_vis, p.y // ds_vis
        dx_e, dy_e = (p.x + ps) // ds_vis, (p.y + ps) // ds_vis
        heatmap[dy:dy_e, dx:dx_e] = p.z

    plt.imshow(heatmap, cmap="viridis")
    plt.title(title)
    plt.colorbar(label="Z Slice")
    plt.axis("off")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def run_demo(vsi_path: Path, output_dir: Path):
    print(f"Running stabilization demo for: {vsi_path.name}")


    ps = 224
    configs = [
        {
            "name": "baseline_p224_perpatch",
            "fps": ps,
            "label": "Baseline (Per-Patch Focus)",
        },
        {
            "name": "stabilized_p224_super2048",
            "fps": 2048,
            "label": "Stabilized (Super-Patch 2048)",
        },
    ]

    stats = {}

    for cfg in configs:
        name = cfg["name"]
        print(f"  Processing {name}...")

        start_time = time.time()
        
        # Instantiate Processor with correct config
        preprocess_config = PreprocessConfig(
             patch_size=ps,
             focus_patch_size=cfg["fps"],
             downscale_factor=16,
             min_tissue_coverage=0.05,
             dataset_name="ZStack_HE"
        )
        processor = SlidePreprocessor(preprocess_config)
        
        # Run processing
        metadata = processor.process(vsi_path)
        final_patches = metadata.patches
        width, height = metadata.width, metadata.height # Capture dims from metadata
        duration = time.time() - start_time
        tv, outliers = calculate_stability_metrics(final_patches, width, height, ps)

        # 3. Save Visualizations
        save_2d_heatmap(
            final_patches,
            height,
            width,
            ps,
            f"{cfg['label']}\nTV: {tv:.3f}, Outliers: {outliers:.2%}",
            output_dir / f"{name}_2d.png",
        )

        save_3d_heatmap(
            final_patches,
            height,
            width,
            ps,
            f"{cfg['label']} (3D View)",
            output_dir / f"{name}_3d.png",
        )

        stats[name] = {
            "duration_sec": duration,
            "total_variation": tv,
            "outlier_ratio": outliers,
            "patch_count": len(final_patches),
        }

    # Save Report
    with open(output_dir / "stabilization_report.json", "w") as f:
        json.dump(stats, f, indent=4)

    print(f"Demo complete. Results saved to {output_dir}")


if __name__ == "__main__":
    vsi_path = Path("/home/niklas/ZStack_HE/raws/001_1_HE_stack.vsi")
    output_dir = Path(__file__).resolve().parent / "results"
    run_demo(vsi_path, output_dir)
