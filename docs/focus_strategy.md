# Focus Estimation Strategy

This document outlines the design decisions and methodology used for focus estimation in this project, specifically the "Super-Patch" mapping and binning strategies.

## 1. The Core Challenge: Noise at 80x Magnification
At 80x magnification, a standard 224x224 patch is physically very small. Finding the "optimal focus" for such a small area using focus metrics (like the Brenner Gradient) is highly susceptible to noise due to:
*   **Sparse Tissue**: Some patches contain very few cells or features.
*   **Sensor Noise**: Small areas don't provide enough data points to average out pixel-level noise.
*   **Local Artifacts**: A single piece of dust or a tissue fold can dominate the focus score of a small patch.

## 2. The Solution: Super-Patch Focal Mapping
Instead of calculating focus independently for every training patch, we use a **Two-Pass** approach:

### Stage 1: Global Focal Mapping (Physics)
*   **Patch Size**: 1792 x 1792 raw pixels.
*   **Downscale**: 4x (reads 448x448 pixels in RAM).
*   **Goal**: Find the stable, physical plane of the glass/tissue.
*   **Why 1792px?**: This size is the "stability sweet spot." It is large enough to average out biological and sensor noise, finding the true focal plane, while remaining local enough to track the slide's natural tilt.

### Stage 2: Training Patch Extraction (Data)
*   **Model Input**: 224 x 224 pixels.
*   **Binning**: 4:1 (reads 896x896 raw pixels, squeezed to 224x224).
*   **Z-Level**: Each patch inherits its "Truth" Z-level from the stable Focal Map computed in Stage 1.

## 3. Binning vs. Downscaling
| term | Context | Purpose |
| :--- | :--- | :--- |
| **Downscaling** | Preprocessing (Offline) | Speeds up the mapping of the slide's focal plane. |
| **Binning** | Training (On-the-fly) | Provides the model with broader morphological context while keeping input size constant (224px). |

In the current recommended configuration, both are set to **4x**, resulting in a consistent resolution across the pipeline.

## 4. Key Metrics: RMSE and Stability
Local variations in focus that cannot be explained by the global slide tilt are measured as **Residual RMSE**.
*   **224px Patch**: RMSE ≈ 2.5 (High jitter)
*   **1792px Patch**: RMSE ≈ 1.9 (Stable focus)

By using the 1792px Super-Patch truth, we ensure the model is trained to reach the **physically optimal** focal plane, rather than chasing local noise.

## 5. Directory Structure for Experiments
The following scripts in `src/benchmarks/focus_experiments/` demonstrate these principles:
*   `analyze_focus_surface.py`: Fits a 3D plane to focus data to calculate tilt and residual noise.
*   `focus_precision_benchmark.py`: Compares focus accuracy across different resolutions and context sizes.
*   `focus_size_stability_sweep.py`: Sweeps patch sizes to find the minimum stable aperture for a given slide.
