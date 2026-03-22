# Experiment Report: Learning Synthetic Kernels on GPU

## Objective
Train a supervised image-to-image model (`SyntheticConvModel`) to learn a fixed synthetic blur kernel (Teacher) from sharp Whole Slide Image (WSI) patches.

## Experiment Configuration
- **Kernel Size**: 63x63
- **Input Size**: 256x256
- **Target Size**: 194x194 (Validated `F.conv2d` without padding)
- **Teacher Kernel**: Centered Morphological Disk (`skimage.morphology.disk`) normalized to 1.0.
- **Simulation Mode**: GPU-based `F.conv2d` dynamic target generation.
- **Training Set**: 20% most-focused crops (Top-Laplacian).

## Primary Pain Points: Kernel Convergence
The most significant challenge in this experiment was getting the large `63x63` learnable kernel to converge into a structured morphological disk rather than looking like random noise.

### 1. The Initialization Gap
Default `Kaiming Uniform` initialization was the primary hurdle. With 3,969 parameters per channel, starting from random "static" noise made it mathematically difficult for the optimizer to find a sparse, structured disk shape. 
- **Solution**: Implementing **Constant Initialization** (`1.0/area`) provided a stable, uniform "blob" that the optimizer could efficiently sculpt into the target disk.

### 2. Architecture Constraints
The model initially lacked the `groups=3` depthwise constraint. 
- **Impact**: The student was attempting to learn cross-channel artifacts, further diffusing the learned structure into high-frequency noise.
- **Solution**: Enforcing **Strict Depthwise Convolution** isolated the R, G, and B learning paths, leading to immediate structural coherence.

### 3. Optimization Speed
Standard learning rates (`1e-4`) were too timid for shaping such a large parameter space in early epochs.
- **Solution**: Bumping the **Learning Rate to 1e-3** allowed the model to rapidly find the macro-structure of the disk before fine-tuning.

## Secondary Observations
During the setup, minor technical hurdles like `slideio` forking behavior and an accidental dataset filtering override were identified and resolved to ensure robust, high-speed iteration (~1.8m per epoch).

## Implementation Details
### Model Initialization
```python
# model.py
self.conv = nn.Conv2d(3, 3, kernel_size=63, groups=3, bias=False)
nn.init.constant_(self.conv.weight, 1.0 / (63 * 63))
```

### Learning Rate Tuning
Increased from `1e-4` to `1e-3` to accelerate the spatial shaping of the large 63x63 weights.

## Final Results: Comparative Study (g3 vs. g1)
| Setting | Epoch Time | Epoch 0 Val Loss | Observations |
| :--- | :--- | :--- | :--- |
| **Depthwise (groups=3)** | **~24s** | 0.023 | Clean spatial disk learning. Extremely efficient. |
| **Full (groups=1)** | **~108s** | 0.020 | Slightly lower loss but ~4.5x slower. Risk of color cross-talk artifacts. |

### Conclusion
While Full Convolution (`groups=1`) provides a marginally lower Mean Squared Error, the **Depthwise (`groups=3`)** configuration is significantly superior for this task. It matches the physical optics (where channels blur independently) and minimizes computational overhead by 75% while ensuring the learned kernel is structurally pure.

## Summary
The system is now a lean, high-performance synthetic data pipeline. Moving simulation to the GPU eliminated OpenCV dependencies and forking deadlocks while significantly increasing iteration speed. Constant initialization and depthwise constraints were the keys to successful kernel structural learning.
