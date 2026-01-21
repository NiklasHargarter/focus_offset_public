# Experiment Report: Large Patch Downscale Study

- **Slide**: 001_1_HE_stack.vsi
- **Fixed Native Area**: 896 x 896 px

## Precision Results

| Downscale | Effective Patch Size | MAE (vs 1x) | Stability (Std) |
| :--- | :--- | :--- | :--- |
| 1x | 896 x 896 px | 0.00 | 1.32 |
| 2x | 448 x 448 px | 0.14 | 1.32 |
| 4x | 224 x 224 px | 0.42 | 1.13 |
| 8x | 112 x 112 px | 0.72 | 1.01 |
| 16x | 56 x 56 px | 1.14 | 0.92 |

![Comparison Heatmap](exp03_large_patch_downscale_001_1_HE_stack.png)

## Analysis
This experiment tests if utilizing a larger context area (896px) can mitigate the precision loss of downscaled focus estimation. Compare these results to the 224px study to see if the error scales linearly.