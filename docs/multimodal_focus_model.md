# Multimodal Focus Offset Model (ConvNeXt + FFT + DWT)

## Overview
This document describes the updated model architecture for **Focus Offset Regression**. The model has been enhanced to accept **frequency domain** (FFT) and **time-frequency** (Wavelet) inputs in addition to standard spatial RGB data. This helps the model explicitly learn sharpness features (high-frequency components) which are critical for autofocus tasks.

## Model Architecture
-   **Backbone**: `ConvNeXt V2 Tiny` (Impl. by `timm` with FCMAE weights).
-   **Key Feature**: Includes **Global Response Normalization (GRN)** layer.
-   **Class**: `src.models.architectures.ConvNeXtV2FocusRegressor`
-   **Input Channels**: **7** (up from 3).
-   **Input Strategy**: On-the-fly computation (GPU).

### Input Configuration (7 Channels)
The input tensor passed to the backbone is a concatenation of three domains. Transforms are computed on the **Grayscale** version of the input image to focus on structural/luminance information and reduce noise.

| Type | Channels | Source | Description |
| :--- | :--- | :--- | :--- |
| **Spatial** | 3 | RGB | Original raw image patches. Captures color and texture. |
| **Spectral (FFT)** | 1 | Grayscale | **2D Fast Fourier Transform**. Represented as Log-Magnitude, shifted so zero-frequency is in the center. Captures global frequency distribution (sharpness). |
| **Wavelet (DWT)** | 3 | Grayscale | **Haar Discrete Wavelet Transform**. Only the **Detail Coefficients** (LH, HL, HH) are used. Upsampled to match input resolution. Captures high-frequency edges at different orientations. |

### Implementation Details
-   **Transform Layer**: Executed inside the model `forward` pass (`InputTransformLayer`).
-   **Efficiency**: 
    -   GPU On-the-fly computation cost: **~2.4 ms/batch**.
    -   No valid need for offline preprocessing or caching.
-   **Weights**: The first 3 channels (RGB) reuse the pretrained ImageNet weights. The 4 new channels (FFT+DWT) are initialized with Kaiming Normal initialization.

## Training Configuration
Optimized parameters found via `src/benchmarks/auto_tune.py` on the current hardware:

-   **Precision**: `bf16-mixed` (Recommended for RTX 30/40 series).
-   **Batch Size**: `128`
-   **Num Workers**: `4`
-   **Throughput**: ~557 img/s

### Recommended Command (Using Config)
```bash
python -m src.train fit --config configs/convnext_v2_multimodal.yaml
```

The config overrides the defaults with optimal settings (Batch 64, Workers 4, BF16, 7-channel input).
**Note**: Batch size is set to 64 to ensure stability on 12GB GPUs. 128 requires 24GB+ VRAM.
```

### Baseline Training (RGB Only)
To train a standard ConvNeXt V2 (RGB 3-channels) as a baseline for comparison, simply set `use_transforms` to `false`. This disables the FFT/DWT branch.

```bash
python -m src.train fit --config configs/convnext_v2_baseline.yaml
```

This ensures an "apples-to-apples" comparison where the only difference is the absence of the FFT/DWT branch. All other augmentations (elastic, noise, crop) will still apply.
Autofocus is fundamentally about maximizing high-frequency energy (sharp edges).
1.  **Spatial (RGB)**: Provides context and standard features.
2.  **FFT**: Provides a global "sharpness score" view. A blurry image has energy concentrated at the center (low freq), while a sharp image has energy spread out.
3.  **DWT**: Provides localized edge detection. The LH, HL, and HH bands strictly contain edge information (horizontal, vertical, diagonal), acting as explicit "edge maps" for the network.
