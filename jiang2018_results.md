# Evaluation Results: Jiang2018 Dataset

**Dataset Size:** 128,699 samples
**Date:** 2026-02-17

| Model | MAE (µm) | Median AE (µm) | Max Error (µm) |
| :--- | :--- | :--- | :--- |
| **RGB** | 5.1207 | 4.9672 | 15.1375 |
| **FFT** | 5.1218 | 4.5805 | 16.4187 |
| **Multimodal** | 5.2625 | 4.9563 | 14.5000 |
| **DWT** | 5.3787 | 5.4000 | 15.6375 |

## Observations
-   **Best MAE**: RGB (5.1207 µm)
-   **Best Median AE**: FFT (4.5805 µm)
-   **Lowest Max Error**: Multimodal (14.5000 µm)
-   **Worst MAE**: DWT (5.3787 µm)

## CSV Locations
-   `logs/rgb/version_0/eval_rgb_Jiang2018.csv`
-   `logs/fft/version_0/eval_fft_Jiang2018.csv`
-   `logs/multimodal/version_0/eval_multimodal_Jiang2018.csv`
-   `logs/dwt/version_0/eval_dwt_Jiang2018.csv`
