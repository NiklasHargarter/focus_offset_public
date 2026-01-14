# Focus Estimation Strategy

This document outlines the design decisions and methodology used for generating high-quality focus ground-truth for deep learning.

## 1. The Core Challenge: Noise vs. Physics
At 80x magnification, a standard **224x224 patch** is physically too small to reliably define "sharpness."
*   **Measurement Noise**: Small areas lack enough contrast features, causing focus scores to fluctuate.
*   **Biological Deviations**: Local topography (tissue folds) can differ from the physical glass plane.
*   **Staircase Effect**: Tiling large patches without overlapping creates artificial jumps in focus labels.

## 2. The Solution: Sliding Super-Patch Mapping
To maximize label quality and spatial consistency, we use a **Sliding Super-Patch** strategy. We recommend an **8x context** (1792px) for 80x magnification data.

### Methodology
*   **Local Patch**: 224 x 224 pixels (The model's training input).
*   **Super-Patch**: 1792 x 1792 pixels (Provides 8x neighborhood context).
*   **Sliding Window**: We calculate the Super-Patch's focal point at **every 224px grid point**.
*   **Efficiency**: Super-patches are downsampled to **224px (8x)** during calculation. This leveraging of the VSI pyramid makes the neighborhood search as fast as (or faster than) raw 1x local searches due to optimized I/O.

## 3. Findings from Experiments

### Context Matters (Local vs. Global)
Experiments show that **224px patches** often have "split peaks" or noisy focus scores. Scaling up to an **8x context (1792px)** provides a highly smooth and reliable focal plane.
*   **Finding**: Local patches drift significantly compared to the neighborhood consensus.
*   **Trade-off**: While a 4x context (896px) is already stable, an 8x context (1792px) provides the most physically robust ground truth for 80x scans with virtually no additional runtime cost.

### Efficiency & Scalability
Using parallel processing (16 workers), the entire 69-gigapixel HE dataset can be processed in **~11 hours**.
*   **Discovery**: The **8x sliding context** is throughput-neutral compared to the **4x context**. Both are roughly the same speed as the local 1x search because the final image block read (224px) is the same size; only the pyramid source level changes.

## 4. Validating Local Topography
By subtracting the **Neighborhood Truth** from the **Local Choice**, we can visualize "Residual Topography."
*   If deviations are clustered, they represent valid biological height changes (tissue wrinkles).
*   If deviations are random, they represent measurement noise that should be ignored during training.

## 5. Experiment Toolkit
The following scripts in `src/experiments/focus_strategy/` support these findings:

| Script | Purpose |
| :--- | :--- |
| `compare_context_strategies.py` | Visualizes Local vs. Tiled vs. Sliding strategies side-by-side with raw tissue imagery. |
| `estimate_throughput.py` | Benchmarks hardware to estimate the time required to process entire datasets (comparing 4x and 8x). |
| `measure_surface_tilt.py` | Fits a 3D plane to calculate the physical tilt of the slide. |
| `benchmark_z_accuracy.py` | Quantifies Mean Absolute Error (MAE) when reducing context or resolution. |
| `generate_full_infographic.py` | Creates a comprehensive "Reasoning Showcase" for use in papers or presentations. |
