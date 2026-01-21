# Experiment Report: Multi-Scale Focus Stability

## Overview
This experiment analyzes how the **focus topology** (the map of optimal Z-slices) changes as we increase the context area and decrease the magnification. All patches are resized to a fixed processing size of **224px**, but their source area on the slide grows with the downscale factor.

## Methodology
1. A grid of 24x24 locations was selected from a tissue-rich ROI.
2. For each location, a Z-stack was extracted at 4 different scales.
3. The 'Best Z' was determined using the Brenner Gradient score on the 224px resized patch.
4. **Stability** is measured as the Standard Deviation (Std) of the Z-map. A higher Std usually indicates a more 'noisy' or 'jittery' focus map, whereas a lower Std indicates a smoother, more consistent focus surface.

## Statistics

- **Slide**: 001_1_HE_stack.vsi
- **Native Resolution**: 60x
- **Patch Processing Size**: 224px

### Results Table

| Downscale | Effective Area on Slide | Stability (Std) | MAE (vs Native 1x) |
| :--- | :--- | :--- | :--- |
| 1x | 224 x 224 px | 2.09 | 0.00 |
| 2x | 448 x 448 px | 1.47 | 1.06 |
| 4x | 896 x 896 px | 0.95 | 1.39 |
| 8x | 1792 x 1792 px | 0.62 | 1.57 |

## Visual Comparison

![Focus Topology Map](exp01_mag_impact_001_1_HE_stack.png)

## Discussion
1. **Stabilization**: As the downscale factor increases (larger source area), the focus map becomes significantly more stable (Std drops from 2.09 to 0.62). This confirms that larger context windows help filter out local pixel noise.
2. **Drift (MAE)**: The MAE shows how much the 'Global Focus' estimate disagrees with the 'Local Focus' estimate. A low MAE suggests that the context window is a faithful representative of the local spot, while a high MAE might indicate that the context is 'averaging out' real local features like tissue tilt or curvature.
3. **Conclusion**: Using a 4x (896px) or 8x (1792px) window as a 'stable ground truth' is viable if the MAE stays low, as it removes the high-frequency jitter of 1x reads.