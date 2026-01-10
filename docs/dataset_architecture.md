# VSI Dataset Architecture

This document explains the high-performance pipeline designed to process, filter, and serve VSI whole-slide images for deep learning.

## 1. Directory Structure
The system uses a metadata-driven approach where data preparation is decoupled from dataset consumption.

1.  **Raw Data**: All VSI files reside in dataset-specific directories (e.g., `/home/niklas/ZStack_HE/raws`).
2.  **Master Index**: Preprocessing converts all raw slides into a single holistic `master_index_{dataset}.pkl`.
3.  **Primary Split**: A `splits_{dataset}.json` defines the balanced **test** set and the **train_pool** (everyone else).
4.  **Specialized DataModules**: DataModules filter the master index at runtime to create specific folds or hold-out splits.

```
project_root/
├── splits_ZStack_HE.json      # Balanced test vs train_pool
├── cache/
│   ├── master_index_ZStack_HE.pkl  # Processed metadata for ALL slides
│   └── ...
├── src/
│   ├── dataset/
│   │   └── vsi_datamodule.py  # Specialized DataModules (HoldOut, Fold, IHC)
│   └── processing/
│       ├── preprocess.py      # Holistic Preprocessor
│       └── create_split.py    # Primary Splitting (Relative Deficit)
```

---

## 2. Preparation Pipeline

The preparation is divided into two decoupled stages: **Holistic Preprocessing** and **Primary Splitting**.

### Stage A: Holistic Preprocessing
**Command**: `python -m src.dataset.vsi_prep.preprocess --dataset ZStack_HE`

To avoid redundant compute, the entire dataset is processed into a single archive.
1.  **Tissue Segmentation (Otsu)**: For every slide, we read an 8x downsampled overview and apply Otsu's thresholding to generate a binary tissue mask.
2.  **Patch Extraction**: We slide a window across the mask. If tissue coverage > `MIN_TISSUE_COVERAGE`, the spatial coordinate `(x, y)` is kept.
3.  **Optimal Focus**: For every valid patch, we scan all Z-levels to find the `best_z` (highest Brenner Gradient).
4.  **Save Master Index**: The metadata (paths, patches, dimensions) is saved to `master_index_ZStack_HE.pkl`.

### Stage B: Primary Splitting (Balanced)
**Command**: `python -m src.dataset.vsi_prep.create_split --dataset ZStack_HE --split_ratio 0.1`

We use the **Relative Deficit** algorithm to ensure large slides are distributed fairly.
1.  **Sort by Size**: Slides are sorted by total patch count (descending).
2.  **Relative Deficit**: For each slide, we calculate our "distance" to the target test capacity. Slides are assigned to the `test` split until the target ratio is reached.
3.  **train_pool**: All slides not in `test` are saved in the `train_pool`.

---

## 3. Data Structures

The pipeline relies on three main data structures to manage high-throughput access and flexible splitting.

### A. Master Index (`master_index_{dataset}.pkl`)
This is the holistic archive containing metadata for **all** slides. It is a pickled dataclass (`MasterIndex`).

```python
from dataclasses import dataclass
from typing import Any
from pathlib import Path

class Patch(NamedTuple):
    x: int
    y: int
    z: int

@dataclass
class SlideMetadata:
    name: str
    path: Path
    num_z: int
    patches: list[Patch]

@dataclass
class MasterIndex:
    file_registry: list[SlideMetadata]
    patch_size: int
    config_state: dict[str, Any]

# Detailed Content:
{
    "file_registry": [
        {
            "name": "slide_001.vsi",         # SlideMetadata
            "path": Path("/data/slide_001.vsi"), # Absolute path
            "num_z": 15,                     # Total Z-slices
            "patches": [                     # List of valid patches
                (1024, 2048, 7),             # (x_px, y_px, optimal_best_z)
                (1248, 2048, 8),
                ...
            ]
        },
        ...
    ],
    "patch_size": 224,                       # Spatial dimensions
    "config_state": {                        # Traceability for cache invalidation
        "PATCH_SIZE": 224,
        "STRIDE": 224,
        "DOWNSCALE_FACTOR": 8,
        "MIN_TISSUE_COVERAGE": 0.05,
        "DATASET_NAME": "ZStack_HE"
    }
}
```

### B. Split Lockfile (`splits_{dataset}.json`)
A human-readable JSON that defines the primary distribution of slides.

```json
{
    "test": ["slide_001.vsi", "slide_005.vsi"],
    "train_pool": ["slide_002.vsi", "slide_003.vsi", "slide_004.vsi"],
    "seed": 42,
    "total_slides": 5,
    "total_patches": 125430
}
```

### C. Runtime Filtered Index
Generated inside the DataModule's `setup()` method. This is what the `VSIDatasetLightning` consumes. Note the addition of `cumulative_indices` for O(log N) lookup.

```python
{
    "file_registry": [...],     # Subset of the master registry
    "cumulative_indices": [     # Running sum of (patches * num_z)
        15000,                  # Samples in File 0
        22500,                  # Samples in File 0 + File 1
        57500                   # Total Samples
    ],
    "total_samples": 57500,
    "patch_size": 224
}
```

---

## 4. Runtime Data Selection (Retrieval)

The final dataset composition happens inside the **Lightning DataModules**.

### Filtering Logic
During `setup()`, the DataModule loads the `master_index` and filters it based on the filenames defined in `splits.json`.

**Example: K-Fold Selection (`HEFoldDataModule`)**
1.  Loads `train_pool` filenames from `splits.json`.
2.  Sorts them by slide size (patch count).
3.  Assigns slides to $K$ folds using **Round-Robin** (0, 1, 2, 3, 4, 0, 1...).
4.  If `fold_idx=0`, it uses Fold 0 for validation and Folds [1, 2, 3, 4] for training.
5.  Calculates **Cumulative Indices** for the filtered subset of slides.

---

## 4. Sample Serving (`VSIDatasetLightning`)

Samples are retrieved via a global index, mapped to files using **O(log N)** binary search.

### Mapping Example
Assume a filtered index with 3 slides:
-   **Slide A**: 1,000 patches
-   **Slide B**: 500 patches
-   **Slide C**: 2,000 patches
-   **$\rightarrow$ Cumulative Indices**: `[1000, 1500, 3500]`

**Retrieving Global Index 1200**:
1.  **Find File**: `bisect.bisect_right([1000, 1500, 3500], 1200)` returns `1` (Slide B).
2.  **Localize**: `1200 - 1000 = 200`. We need Sample 200 from Slide B.
3.  **Unpack Z-Levels**:
    -   `patch_idx = 200 // num_z` (e.g., if num_z=15, patch 13).
    -   `z_level = 200 % num_z` (e.g., z=5).
4.  **Read Image**: The worker retrieves `(x, y)` and `best_z` from the metadata and reads the specific block using `slideio`.

### Performance Optimizations
-   **Lazy Handles**: File handles are opened only when needed and cached per-worker process to avoid sharing issues in multiprocessing.
-   **Z-Level Regression**: `__getitem__` returns the image and the focus offset in micrometers:
    `target = (best_z - current_z) * z_resolution_um`.
