# Phase 0: Data Ingestion & Sanitization

**Status:** Not Started  
**Dependencies:** None  
**Goal:** Build an isolated utility to download, sanitize, and strictly filter the raw datasets. This must remain structurally separate from the PyTorch indexers and dataloaders.

## Requirements Addressed
- Phase 0: Ingestion & Sanitization Separation (Strict Separation of Concerns)

## Vertical Slice Execution Steps
1. **Download Orchestration:** Create a distinct ingestion script (e.g., `src/data/download_datasets.py`) that utilizes the internal `exact_sync` python library to download the HE and IHC datasets from the internal EXACT server (as well as handling AgNor/Jiang).
2. **Hard Metadata Filtering & Tiers (HE/IHC Only):** Implement strict multi-tiered ingestion logic based on a configuration or CLI flag (e.g., `--subset {full, normal, small}`) specifically for the massive `ZStack HE` and `ZStack IHC` datasets.
   - **`full`:** Downloads everything, no filtering. Output targets `data/he_full/`.
   - **`normal`:** Explicitly drops anomalous macro-scans (e.g., slides containing `_all_` in their filename). Output targets `data/he_normal/`.
   - **`small`:** A highly restricted but cleanly stratified subset containing enough slides to form minimal Train/Val/Test splits. Output targets `data/he_small/`.
   - *Architecture Rule:* Subsets absolutely must exist in independent, physically isolated directories. Duplication > Shared Logic.
3. **VSI Zip Sanitization:** The raw VSI downloads arrive with a corrupted or nested ZIP structure from the internal EXACT server. Write an automated sanitization utility immediately following the download step that unzips, restructures, and properly formats the `.vsi` and accompanying `_vsi` directories so `slideio` can correctly parse them.
4. **Jiang2018 Scale Correction:** The public Jiang2018 test datasets are natively sized at 2x the resolution of the training dataset. The ingestion script must perform a one-time, offline spatial correction: loading all `test_same` and `test_diff` split images, downscaling them by 50% (`fx=0.5, fy=0.5` via interpolation), cropping them into discrete 224x224 tiles, and saving the normalized tiles to disk. This purges resizing operations from the downstream PyTorch `__getitem__`.
5. **Verify:** Run the ingestion script. It should cleanly download a subset of the data, actively reject an `"all"` slide, fix the VSI structure, correctly downscale the Jiang2018 test images, and terminate. The resulting local directories must be perfectly uniform and ready for Phase 1.

## Completion Criteria
- [ ] Ingestion script successfully downloads and authenticates against internal servers.
- [ ] Slides containing `"all"` in the name are correctly ignored.
- [ ] VSI directory structures are repaired natively after download.
- [ ] Downstream components (Phase 1 indexers) have strictly zero knowledge of download logic and only expect local paths.
