# VSI Dataset Preprocessing

This document explains the preprocessing pipeline for VSI datasets and the rationale behind the parameter decisions for focal plane estimation and data tiling.

## Overview
The goal of the preprocessing pipeline is to convert raw Whole Slide Images (WSI) in VSI format into a structured set of training patches with reliable "ground truth" focus designations.

## Pipeline Architecture

The preprocessing is split into shared, format-agnostic modules and format-specific scripts:

```
src/dataset/prep/                  # Shared modules
    tissue_detection.py            # Otsu-based tissue masking (histolab)
    grid.py                        # Coverage-filtered patch grid (integral image)

src/dataset/vsi_prep/preprocess.py # VSI-specific pipeline (slideio)
src/dataset/ome_prep/preprocess.py # OME-TIFF pipeline (tifffile)
```

### Single-slide processing steps

Each slide goes through five named functions, called in order by `process_slide`:

1. **`read_thumbnail_rgb`** — read a 16× downsampled RGB thumbnail from z-slice 0
2. **`detect_tissue_mask`** — generate a binary tissue mask via histolab's pipeline (grayscale conversion → Otsu → dilation → remove small holes/objects)
3. **`generate_patch_candidates`** — lay a grid over the slide and keep only patches with sufficient tissue coverage, using an integral image for O(1) coverage lookup
4. **`find_best_z_per_patch`** — for each accepted patch, read the full z-stack and select the sharpest slice via Brenner gradient
5. **`build_patch_index`** — combine (x, y) positions with best-z into the final `(N, 3)` int32 array

## Focal Plane Estimation Strategy
A critical challenge in Z-stack datasets is the instability of per-patch focus estimation. Small regions (e.g., 128x128) often lack sufficient texture or contain artifacts that lead to erratic focus scoring.

### The "Super-Patch" Grid
To solve this, we implemented a **Super-Patch strategy**:
1.  **Grid Tiling**: The slide is divided into large, non-overlapping regions (e.g., 2048x2048 pixels), termed "Super-Patches".
2.  **Representative Focus**: We estimate the optimal Z-slice for the *entire* Super-Patch using the selected focus metric.
3.  **Inheritance**: All smaller data patches (e.g., 224x224) falling within a Super-Patch inherit its optimal Z-level.

This strategy ensures a smooth, continuous focal plane that is physically realistic and robust to local image noise.

## Parameter Rationale
The following parameters were determined through extensive benchmarking (see `experiments/focus_strategy` and `experiments/stabilization_demo`).

### 1. Focus Metric: Brenner Gradient
*   **Decision**: `brenner`
*   **Rationale**: 
    *   **Scientific Alignment**: Alignment with the [Jiang2018 paper](https://example.com) methodology, which defines in-focus ground truth by maximizing the Brenner gradient.
    *   **Performance**: Brenner is ~25% faster than Tenengrad in our environment while matching its stability at the Super-Patch scale.
    *   **Stability**: At the 2048p scale, Brenner achieves a near-perfect outlier ratio (0.04%).

### 2. Focus Patch Size (Grid Scale)
*   **Decision**: `2048`
*   **Rationale**: 
    *   **Stability Scaling**: Stability improves dramatically as context increases. A 2048p grid reduces the "Outlier Ratio" by over 100x compared to per-patch focus.
    *   **Efficiency**: Larger grid sizes reduce the total number of focus calculations required per slide, significantly speeding up the preprocessing phase.

### 3. Focus Binning (`binning_factor`)
*   **Decision**: `1` (No downsampling)
*   **Rationale**: 
    *   While higher binning (e.g., 2x or 4x) can provide a further ~20% speedup, `binning_factor=1` provides the cleanest and most accurate "ground truth" focal plane. 
    *   Since the bottleneck is often VSI reading rather than calculation, we prioritize focus precision.


## Evaluation Metrics
We use two primary quantitative metrics to evaluate the quality of a focal plane:

1.  **Total Variation (TV)**: Measures the smoothness of the focal plane. Lower values indicate a more continuous surface without sharp jumps.
2.  **Outlier Ratio**: The percentage of patches that differ by more than 2 Z-slices from their neighbors. This identifies "focus failures" where the algorithm was fooled by noise or artifacts.

## Comparison Summary
| Metric | Per-Patch (Baseline) | Super-Patch 2048 (Optimized) |
| :--- | :--- | :--- |
| **Outlier Ratio** | ~17.8% | **~0.04%** |
| **Total Variation** | ~2.13 | **~0.03** |
| **Preprocessing Speed**| Moderate | **High** |

---
*For more details on the experiments, refer to the `experiments/` directory or the `walkthrough.md` artifact.*
