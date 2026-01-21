# VSI Data Extraction & Focus Estimation Strategies

This document summarizes the findings from a series of benchmarks conducted in `src/experiments/focus_mechanics_v2/` regarding the optimization of dataset preparation for multi-scale focus estimation.

## 1. Precision vs. Speed: The Downscaling Cost
Our experiments compared finding the optimal Z-slice using raw pixels (1x) versus downscaled representations (the "Legacy Approach").

| Internal Downscale | Processing Size | **MAE (Z-slice Error)** |
| :--- | :--- | :--- |
| **1x (Raw)** | **224 px** | **0.00** (Reference) |
| 2x | 112 px | 0.22 |
| 4x | 56 px | 0.47 |
| **8x (Legacy)** | **28 px** | **0.79** |

**Conclusion**: Using downscaled representations (8x) for focus estimation induces a jitter of nearly **0.8 Z-slices**. For high-precision 60x magnification models, ground truth labels must be generated at **1x (Raw Resolution)** to avoid training on blurry/noisy labels.

## 2. Overcoming the I/O Bottleneck
VSI files suffer from significant seek-time overhead when performing many small individual reads. We benchmarked different extraction strategies at Raw Resolution (60x).

| Strategy | Speed (patches/s) | Note |
| :--- | :--- | :--- |
| Naive Individual | 120.8 | 1,600 disjoint jump-reads. |
| **8x8 Block-Based** | **152.9** | **The Sweet Spot**. Balance of I/O batching and RAM. |
| 12x12 Block-Based | 157.4 | Slight gain, but higher RAM risk. |
| 40x40 Block-Based | 37.9 | **Performance Collapse**. Single reads too large for `slideio`. |

**Conclusion**: The optimal "Unit of Work" for Raw Resolution extraction is an **8x8 or 12x12 Master Block**. This minimizes disk seeks while keeping memory buffers manageable (~600MB - 1GB).

## 3. Utilizing the Native Pyramid
VSI files often contain pre-computed pyramid levels. Inspecting the hardware levels reveals:
- **Level 0 (1x)**: 60.0x Mag
- **Level 1 (2x)**: 30.0x Mag
- **Level 2 (4x)**: 15.0x Mag
- **Level 3 (8x)**: 7.5x Mag

**Benchmark Result**: Reading global Z-slices from native levels is extremely fast (e.g., **0.05s** for a full 8x slice). For a balanced dataset, we can reuse these global reads to extract thousands of patches per second.

## 4. Final Balanced Multi-Scale Strategy
To create a dataset with equal sample counts for 1x, 2x, 4x, and 8x magnifications:

1.  **High-Density Sampling**: For a single global Z-stack read at 2x, using overlapping patches (e.g., 112px stride) increases throughput from **108 to 319 patches/s**.
2.  **Aligned Extraction**: For every 1x center, we extract 4 concentric patches (resized from the 1x raw buffer):
    - **1x**: 224x224 raw.
    - **2x**: 448x448 resized.
    - **4x**: 896x896 resized.
    - **8x**: 1792x1792 resized.
3.  **Result**: 100% balanced samples, zero inter-scale jitter, and maximum hardware utilization.

---
*For more details, see the benchmark scripts in `src/experiments/focus_mechanics_v2/`.*
