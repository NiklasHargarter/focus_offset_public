# Phase 2: Formalized Preprocessing Pipeline & Index Export

**Status:** Not Started  
**Dependencies:** Phase 1 (Minimum Viable Data Foundation)  
**Goal:** Stop relying on magical constants for tissue filtering and patch downscaling. Make preprocessing an explicit pipeline that exports a plain-text version-controllable index.

## Requirements Addressed
- Preprocessing & Tissue Filtering Validation
- Dataset Sharing & Archival Strategy

## Vertical Slice Execution Steps
1. **Build Tissue Filtering Module:** Create a deep module dedicated to background thresholding and tissue masking. It must accept parameters (not rely on hard-coded constants).
2. **Export Index:** Hook this module into the Phase 1 Shared Indexer. When run, it must calculate valid patches and export the result to `he_index.csv` (containing `slide_name, x, y, z_level, optimal_z, z_offset_microns`).
3. **Modify `OffsetPredictionDataset`:** Update the dataset interface so it accepts a path to this precomputed CSV file instead of computing splits on the fly.
4. **Verify:** Run the preprocessing script to generate the CSV. Then run `train_focus.py` exclusively utilizing the exported CSV as its data source.

## Completion Criteria
- [ ] Preprocessing module outputs a valid `.csv` mapping spatial coordinates and z-offsets.
- [ ] `OffsetPredictionDataset` seamlessly dynamically loads blocks using only the `.csv` and raw `.vsi` files.
- [ ] `train_focus.py` successfully runs using the newly generated index.
