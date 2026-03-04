# VSI Preparation Pipeline

This module provides a unified 2-step pipeline for the acquisition and indexing of VSI datasets.

## 1. Acquisition (`download.py`)
The download utility handles the entire synchronization process from EXACT:
- **Sync**: Fetches missing ZIP files based on the dataset name.
- **Structural Fix**: Automatically unzips and flattens the file structure (moves nested `.vsi` to the slide root).
- **Verification**: Opens the resulting VSI via `slideio` to confirm readability.
- **Cleanup**: Deletes the source ZIP after successful verification.

## 2. Indexing (`preprocess.py`)
The indexing orchestrator handles metadata extraction and patch generation:
- **Split Handling**: Automatically creates a `splits.json` (train/test) if it doesn't exist.
- **Patch Generation**: Detects tissue regions and generates patch metadata.
- **Focus Scoring**: Computes sharpness across the Z-stack to determine regression targets.
- **Output**: Writes highly efficient `train.parquet` and `test.parquet` files to the cache.

## Configuration
The pipeline strictly uses the `DATASET_REGISTRY` in `src.datasets.vsi.config`. 

| Dataset | Slide Dir | Min Tissue Coverage | Downsample |
|---|---|---|---|
| ZStack_HE | `DATA_ROOT/ZStack_HE` | 0.80 | 2x |
| ZStack_IHC | `DATA_ROOT/ZStack_IHC` | 0.30 | 1x |

---

Each dataset (HE, IHC) provides thin wrapper scripts in its `prep/` directory that delegate to these shared utilities.
