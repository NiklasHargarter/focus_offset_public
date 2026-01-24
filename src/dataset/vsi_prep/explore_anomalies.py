import argparse
import numpy as np
import pickle
from typing import List, Dict
import pandas as pd

from src import config
from src.dataset.vsi_types import SlideMetadata
from src.dataset.vsi_prep.preprocess import fit_plane_robust, get_spatial_clusters


def load_all_slides(dataset_name: str, patch_size: int) -> List[SlideMetadata]:
    """Manually load all slide pickles since MasterIndex might be huge/slow."""
    indices_dir = config.get_slide_index_dir(dataset_name, patch_size)
    slides = []
    print(f"Loading slides from {indices_dir}...")
    files = list(indices_dir.glob("*.pkl"))
    for pkl_path in files:
        try:
            with open(pkl_path, "rb") as f:
                slides.append(pickle.load(f))
        except Exception as e:
            print(f"Error loading {pkl_path}: {e}")
    print(f"Loaded {len(slides)} slides.")
    return slides


def analyze_slide(slide: SlideMetadata, spatial_threshold: float = 4.0) -> Dict:
    """Analyze a single slide for focus anomalies."""
    if len(slide.patches) == 0:
        return {
            "name": slide.name,
            "mean_residual": 0.0,
            "max_residual": 0.0,
            "outlier_count": 0,
            "outlier_ratio": 0.0,
            "patch_count": 0,
        }

    X = slide.patches[:, 0]
    Y = slide.patches[:, 1]
    Z = slide.patches[:, 2]

    # Find dominant cluster to fit the "main" plane
    # If there are multiple tissues, we might get multiple planes, but let's
    # assume the largest one represents the main focal plane for anomaly detection logic
    # or just fit planes per cluster. Best to fit per cluster.

    clusters, num_clusters = get_spatial_clusters(X, Y, threshold=5000)

    total_outliers = 0
    all_residuals = []

    for cid in range(num_clusters):
        c_mask = clusters == cid
        if np.sum(c_mask) < 4:
            continue

        cX = X[c_mask]
        cY = Y[c_mask]
        cZ = Z[c_mask]

        # Robust fit
        params, inliers = fit_plane_robust(cX, cY, cZ, threshold=spatial_threshold)

        if params is None:
            continue

        # Calculate residuals for ALL points in this cluster against the robust fit
        Z_pred = np.column_stack([cX, cY, np.ones_like(cX)]) @ params
        residuals = np.abs(cZ - Z_pred)  # Absolute residual

        # Outliers are those exceeding threshold
        outliers = residuals > spatial_threshold
        total_outliers += np.sum(outliers)
        all_residuals.extend(residuals)

    if not all_residuals:
        return {
            "name": slide.name,
            "mean_residual": 0.0,
            "max_residual": 0.0,
            "outlier_count": 0,
            "outlier_ratio": 0.0,
            "patch_count": len(slide.patches),
        }

    all_residuals = np.array(all_residuals)

    return {
        "name": slide.name,
        "mean_residual": np.mean(all_residuals),
        "max_residual": np.max(all_residuals),
        "outlier_count": total_outliers,
        "outlier_ratio": total_outliers / len(slide.patches),
        "patch_count": len(slide.patches),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="ZStack_HE")
    parser.add_argument("--patch_size", type=int, default=224)
    parser.add_argument("--top_n", type=int, default=20)
    args = parser.parse_args()

    slides = load_all_slides(args.dataset, args.patch_size)

    results = []
    print("Analyzing slides...")
    for slide in slides:
        res = analyze_slide(slide)
        results.append(res)

    df = pd.DataFrame(results)

    # Sort by Outlier Ratio (descending)
    df_sorted = df.sort_values(by="outlier_ratio", ascending=False)

    print("\n=== TOP ANOMALOUS SLIDES (By Outlier Ratio) ===")
    print(df_sorted.head(args.top_n).to_string())

    # Check for extreme residuals
    df_resid = df.sort_values(by="max_residual", ascending=False)
    print("\n=== TOP ANOMALOUS SLIDES (By Max Residual) ===")
    print(df_resid.head(args.top_n).to_string())

    # Save to CSV
    out_csv = config.CACHE_DIR / f"anomalies_{args.dataset}.csv"
    df_sorted.to_csv(out_csv, index=False)
    print(f"\nFull report saved to {out_csv}")


if __name__ == "__main__":
    main()
