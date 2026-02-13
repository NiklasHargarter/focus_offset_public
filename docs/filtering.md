# Dataset Filtering & Cleaning Strategy

This document describes the automated filtering pipeline used to ensure the `ZStack_HE` dataset contains only high-quality, physically consistent focus labels.

## 1. Tissue Masking (Pre-scan)
Before running focus detection, we identify "foreground" tissue to skip empty background regions.

### The Problem
Classic thresholding often picks up:
*   Dust specs and debris on the glass.
*   "Vignetting" (darker corners) caused by uneven illumination.
*   Faint smudges or marker pen ink.

### The Solution
Tissue masking is handled by a shared module (`src/dataset/prep/tissue_detection.py`) using **histolab**'s recommended composition pipeline:

1.  **RGB → Grayscale** conversion
2.  **Otsu Threshold** — separates tissue from background
3.  **Binary Dilation** (disk size 5) — fills small gaps within tissue
4.  **Remove Small Holes** (area < 3000) — closes internal voids
5.  **Remove Small Objects** (size < 3000) — removes dust and debris

This pipeline is shared across both VSI and OME-TIFF preprocessing. The caller reads a 16× downsampled RGB thumbnail from z-slice 0 and passes it in; the function returns a binary uint8 mask.

Grid candidate generation (`src/dataset/prep/grid.py`) then uses an integral image to compute tissue coverage per patch in O(1), accepting only patches above `min_tissue_coverage`.

---

## 2. Integrated Quality Filters
During the main preprocessing loop, every potential patch undergoes two rigorous checks. Patches failing *either* check are discarded immediately.

### A. Peak Ambiguity Filter (The "Brenner" Check)
We analyze the Brenner focus curve (Focus Score vs. Z-slice) for every patch.

*   **Logic**: A healthy patch should have a single, dominant peak. Double peaks indicate failure modes like:
    *   Dust on coverslip + Tissue below.
    *   Transparent folds.
    *   Chromatic aberration.
*   **Metric**: Peak Prominence.
    *   We identify the Top 2 peaks ($P_1, P_2$).
    *   Condition: If $P_2 > (0.85 \times P_1)$, the patch is **REJECTED**.
    *   **Why?** If the runner-up is nearly as good as the winner, the label is unstable.

### B. Spatial Consistency Filter (The "Plane" Check)
We assume tissue generally lies flat on the glass (or follows a smooth curve).

*   **Logic**:
    1.  **Clustering**: We use `KDTree` to group patches into spatial clusters (separate tissue sections on the same slide).
    2.  **Robust Plane Fit**: For each cluster, we fit a 3D plane ($Z = aX + bY + c$) using iterative outlier rejection.
    3.  **Outlier Detection**: Any patch whose "best focus Z" deviates from this plane by more than **4.0 slices** is **REJECTED**.
    *   **Why?** Physical tilt is linear. Random deviations >4 slices are almost always artifacts (folds, thick dirt) where the focus metric failed.

## Parameters Reference

| Parameter | Value | Description |
| :--- | :--- | :--- |
| `MASK_DOWNSCALE` | 16 | Resolution factor for tissue masking (speed optimization) |
| `Otsu` | histolab default | Automatic threshold via `OtsuThreshold` filter |
| `Dilation disk` | 5 | Binary dilation disk size (fills small gaps) |
| `Small holes` | 3000 | Area threshold for `RemoveSmallHoles` |
| `Small objects` | 3000 | Min size for `RemoveSmallObjects` |
| `Spatial Threshold` | 4.0 | Max allowed Z-distance from the local tissue plane (slices) |
| `Min Prominence` | 0.85 | Rejection ratio for double peaks (Lower = Stricter/More Rejections) |

## Running the Filter
The filtering is now **fully integrated** into `preprocess.py`.

```bash
# Clean entire dataset (WILL OVERWRITE CACHE)
uv run python src/dataset/vsi_prep/preprocess.py --dataset ZStack_HE --force
```
