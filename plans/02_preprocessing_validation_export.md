# Phase 2: Formalized Preprocessing Pipeline & Index Export

**Status:** Not Started  
**Dependencies:** Phase 1 (Minimum Viable Data Foundation)  
**Goal:** Stop relying on magical constants for tissue filtering and patch downscaling. Make preprocessing an explicit pipeline that exports a plain-text version-controllable index.

## Requirements Addressed
- Preprocessing & Tissue Filtering Validation
- Dataset Sharing & Archival Strategy

## Vertical Slice Execution Steps
1. **Focus Metric & Peak Detection Module:** Build a robust focus-scoring module that computes focus curves across the Z-stack using configurable metrics. Implement peak detection to explicitly identify multi-focal patches (patches with multiple distinct Z-peaks).
2. **Exhaustive Pyramid Indexing:** Hook this module into the Shared Indexer to target specific native pyramid downscales (e.g., 1x, 2x, 4x) for consistent 224x224 window extraction. The indexer computes the exhaustive grid (including all empty background padding) and exports the raw atomic flat CSVs directly (e.g., `he_index_1x.csv`, `he_index_4x.csv`).
3. **Decoupled Tissue Filtering (Pure Core Module):** Build the background masking/tissue filtering function inside the library. Implement this mathematically pure utility to calculate an **adaptive, slide-wide threshold** (e.g., via Otsu's method on a low-res thumbnail). It returns a raw boolean boolean mask or a filtered DataFrame, but must contain strictly zero plotting code (no `matplotlib`/`seaborn`) to remain lightning-fast in PyTorch or batch scripts.
4. **Visual Proof Experiment (`exp_tissue_filtering.py`):** Create an explicit, isolated experiment script. This wrapper imports the pure core module from Step 3, runs it over sample HE and IHC slides, and utilizes `seaborn` to render downsampled slide-images with exact patch grid overlays. This safely generates the formal visual proof of correctness without polluting the core codebase with slow IO operations.
5. **Modify `OffsetPredictionDataset`:** Update the dataset interface so it accepts this flat CSV database. It must utilize PyTorch or Pandas to dynamically group by `(slide, x, y)` to dynamically define the experimental `optimal_z` and resulting offset targets on the fly based on the chosen metric parameters.
6. **Visualization Script:** Write a short offline script (`plot_focus_distributions.py`) utilizing `seaborn` that loads the index and generates high-quality statistical curve plots (e.g., Brenner vs Laplacian) for arbitrary slides, capitalizing on the atomic flat data to deliver rapid visual metric evaluation.
7. **Verify:** Run the preprocessing script to generate the raw metric database CSV. Then run `train_focus.py` successfully utilizing it.

## Completion Criteria
- [ ] Preprocessing module outputs a valid `.csv` mapping spatial coordinates and z-offsets.
- [ ] `OffsetPredictionDataset` seamlessly dynamically loads blocks using only the `.csv` and raw `.vsi` files.
- [ ] `train_focus.py` successfully runs using the newly generated index.
