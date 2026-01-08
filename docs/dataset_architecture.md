# VSI Dataset Architecture

This document explains the high-performance pipeline designed to process, filter, and serve VSI whole-slide images for deep learning.

## 1. Directory Structure
The system uses a metadata-based split. All raw VSI files reside in one directory (`VSI_RAW_DIR`).
A `splits.json` file defines which files belong to train/test sets. 
The preprocessor generates a single master index.

## 1. Directory Structure

**Run the Pipeline**: `python prepare_dataset.py`

This single command handles splitting, preprocessing, and verification. It is idempotent (safe to run multiple times).

```
project_root/
├── splits.json               # Lockfile for the split
├── cache/                    # Isolated Index Files
│   ├── dataset_index_train.pkl
│   └── dataset_index_test.pkl
├── visualizations/           # Debug outputs (masks)
```

## 2. Efficient Preprocessing & Filtering
**Script**: [`src/processing/preprocess.py`](./src/processing/preprocess.py)

To avoid reading terabytes of empty background glass, we pre-calculate which patches contain valid tissue. This is done once per dataset.

### The Algorithm
1.  **Downsampled Overview**: Instead of reading high-res tiles (slow), we read an 8x downsampled version of the entire slide (configurable via `DOWNSCALE_FACTOR`).
2.  **Adaptive Focus**: We scan the downsampled overview at all Z-levels and select the slice with the highest *Brenner Gradient* score. This ensures the mask is generated from the sharpest image.
3.  **Tissue Segmentation (Otsu)**:
    -   We apply **Otsu's thresholding** to the best-focus slice.
    -   **Result**: A clean binary mask where `1`=Tissue, `0`=Glass.
4.  **Patch Selection**:
    -   We slide a window across the mask.
    -   If the tissue ratio in that window > `MIN_TISSUE_COVERAGE`, the top-left coordinate `(x, y)` is saved.
    -   **Focus Scoring**: For each valid patch, we also compute the *Brenner Gradient* across all Z-levels to find the `best_z` index for that specific patch.

### Visualization
The process creates two debug images in `visualizations/`:
*   `*_mask.png`: The raw binary tissue mask.
*   `*_patch_mask.png`: A blocky map showing exactly which patches were kept (white) vs discarded (black).

---

## 3. The Flat Index (`cache/*.pkl`)
The result of preprocessing is a lightweight index file containing metadata only.

```python
dataset_index = {
    "file_registry": [
        {
            "path": "data_train/slide_001.vsi",
            "num_z": 15,
            "patches": [(x, y, best_z), ...] # Coordinates + Best Focus Z
        },
        ...
    ],
    "cumulative_indices": [...], 
    "total_samples": 12000
}
```

---

## 4. Sample Serving (`VSIDataset`)
**Class**: `VSIDataset` in [`src/dataset/vsi_dataset.py`](./src/dataset/vsi_dataset.py)

The dataset serves normalized tensors and regression targets efficiently using **O(1) mapping** and **Per-Worker Caching**.

### Mapping Logic (`__getitem__`)
The dataset uses `bisect_right` to map a single global index to specific files efficiently (**O(log N)**).

**Example**:
Imagine 3 files with different sample counts:
*   File A: 100 samples
*   File B: 50 samples
*   File C: 200 samples

We store `cumulative_indices = [100, 150, 350]`.

When the DataLoader requests **Item 120**:
1.  **Find File**: `bisect.bisect_right([100, 150, 350], 120)` returns index `1` (File B).
2.  **Localize**: Subtract the previous total to get the local index: `120 - 100 = 20`.
3.  **Result**: We retrieve **Sample 20** from **File B**.

Once the file and local sample are identified:
1.  **Unpack Dimensions**:
    -   `patch_index = local_index // num_z` (Which spatial crop?)
    -   `z_level = local_index % num_z` (Which Z-slice?)
2.  **Retrieve Metadata**: Get `(x, y, best_z)` from the index.

### Transforms & Normalization
The dataset automatically handles image standardization:
-   **Input**: Raw `uint8` numpy array `[H, W, C]` (RGB).
-   **Output**: Float Tensor `[C, H, W]` normalized to range `[0, 1]`.
-   **Custom Transforms**: You can pass `transform=...` to `VSIDataset` (e.g., standard augmentations). If present, it overrides the default normalization.

### Regression Target (Focus Offset)
The model needs to predict how far out of focus the current slice is.
-   **Calculation**: `offset = (best_z - current_z) * z_resolution`
-   **Unit**: Micrometers (um).
-   **Target Tensor**: Returns a float tensor of shape `(1,)`.

### Per-Worker Caching
To achieve high throughput (~2000 items/sec):
*   Each DataLoader worker maintains its own `slideio` file handles in `self.slide_cache`.
*   Files are opened lazily and kept open, avoiding expensive file-open overhead on every patch.
