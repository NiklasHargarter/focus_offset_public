# Dataset Preparation Runtime Estimate (ZStack_HE)

This document provides a realistic runtime estimation for preparing the `ZStack_HE` dataset with balanced multi-scale samples (1x, 2x, 4x, 8x).

## Dataset Statistics
- **Slides**: 43
- **Average Dimensions**: 35,883 x 33,751 pixels
- **Sample Density**: 1x non-overlapping (224px stride)
- **Total Samples per Scale**: ~1,032,000

## Estimated Throughput & Runtime

| Phase | Magnification | Extraction Strategy | Estimated Throughput | Total Time |
| :--- | :--- | :--- | :--- | :--- |
| **Phase 1** | 60x (1x) | 8x8 Parallel Master-Blocks | 153.0 patches/s | **1.87 hours** |
| **Phase 2** | 30x (2x) | Global Z-Stack Read + MP Slide | ~2,500 patches/s | ~2.5 min |
| **Phase 3** | 15x (4x) | Global Z-Stack Read + MP Slide | ~6,000 patches/s | ~1.5 min |
| **Phase 4** | 7.5x (8x) | Global Z-Stack Read + MP Slide | ~10,000 patches/s | ~1 min |
| **TOTAL** | **Combined** | **Hybrid Pipeline** | **-** | **~2 hours** |

## Summary of Strategy
1.  **High-Resolution Precision**: Phase 1 uses raw-resolution 1x reads to ensure 100% accurate ground truth focus labels.
2.  **Pyramid Optimization**: Phases 2-4 utilize the native VSI pyramid levels. This avoids expensive downscaling in RAM and allows for extremely fast global slide reads.
3.  **Balanced Dataset**: The pipeline ensures an identical number of samples for all magnification levels by using aligned center-sampling.

---
*Generated based on benchmarks in `src/experiments/focus_mechanics_v2/`.*
