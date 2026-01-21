# Focus Estimation Strategy

This document outlines the design decisions and methodology used for generating high-quality focus ground-truth for deep learning.

## 1. The Core Challenge: Noise vs. Physics
At 80x magnification, a standard **224x224 patch** is physically too small to reliably define "sharpness."
*   **Measurement Noise**: Small areas lack enough contrast features, causing focus scores to fluctuate.
*   **Biological Deviations**: Local topography (tissue folds) can differ from the physical glass plane.

## 2. The Solution: Dense Grid + Neighborhood Context
To maximize both data density and label quality, we use a **Dense Neighborhood Sync** strategy. We recommend a **4x context (896px)** for 80x magnification data.

### Methodology
*   **Sampling Grid**: Patches are extracted on a dense **224px stride**. No tissue data is skipped.
*   **Neighborhood window**: For every 224px grid point, we load a centered **896px "Super-Patch"** (4x neighborhood).
*   **Magnification Trade-off**: The 896px area is read directly from the VSI pyramid into a **224px buffer**. This effectively treats a **20x magnification** equivalent (80x / 4) as our gold standard, trading raw resolution for signal stability.
*   **Verified Labels**: The focus truth (Z-label) is calculated from this same 224px (20x equivalent) buffer using the Brenner focus metric.

## 3. Findings from Experiments

### Stability via Neighborhoods (Magnification Trade-off)
At 80x, sensor noise is high and tissue content in a 224px area is low. By using the VSI pyramid to read at an **effective 20x magnification** (896px area downsampled to 224px), we capture a physically stable "Consensus Focus" for the entire neighborhood.
*   **Finding**: Local 80x patches drift significantly compared to the 20x neighborhood consensus.
*   **Decision**: 20x Effective Magnification (4x context) is the optimal balance between physical stability and information density.

### Efficiency & Scalability
Using parallel processing (16 workers), the entire 69-gigapixel HE dataset can be processed in **~11 hours**.

## 4. Validating Local Topography
By subtracting the **Neighborhood Truth** from the **Local Choice**, we can visualize "Residual Topography."
*   If deviations are clustered, they represent valid biological height changes (tissue wrinkles).
*   If deviations are random, they represent measurement noise that should be ignored during training.

## 5. Experiment Toolkit
The following scripts in `src/experiments/focus_strategy/` support these findings:

| Script | Purpose |
| :--- | :--- |
| `compare_context_strategies.py` | Visualizes Local vs. Tiled vs. Sliding strategies side-by-side with raw tissue imagery. |
| `visualize_preprocessing_steps.py` | Walkthrough of masking, grid filtering, and neighborhood sync. |
| `estimate_throughput.py` | Benchmarks hardware to estimate the time required to process entire datasets. |
| `measure_surface_tilt.py` | Fits a 3D plane to calculate the physical tilt of the slide. |

## 6. Comparison with Legacy Approach
Previously, focus estimation used a **Whole-Slide 8x Downscale** approach.

### The Problem
1.  **Tiny Patches**: Reading the whole slide at 8x forced the Brenner metric to run on **28px x 28px** patches in RAM.
2.  **Result**: Labels were extremely noisy because the metric would peak on sensor noise rather than tissue features.

### The New Sync Approach (4x Dense)
By switching to **Neighborhood Sync (4x)**, we preserve the smooth focal plane while providing 4x more pixel information per patch than the old 8x whole-slide method.
