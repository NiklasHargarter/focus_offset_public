# Phase 7: Multi-Dataset Scale Up & Synthesis

**Status:** Not Started  
**Dependencies:** Phase 2, Phase 6  
**Goal:** Expand parsing logic beyond the initial HE slice to fully encompass the IHC, AgNor, and Jiang2018 datasets using the unified indexer.

## Requirements Addressed
- Data Foundation & Serving Architecture (All 4 Datasets)

## Vertical Slice Execution Steps
1. **Expand Shared Indexer:** Add parsing capabilities for `.ometiff` (AgNor) and the raw `.jpeg` patches (Jiang2018) while maintaining the exact same spatial grid and target schema.
2. **Export Multiple Indices:** Run the preprocessing validation for IHC, AgNor, and Jiang, saving them as their respective `_index.csv` files.
3. **Orchestrator Routing:** Ensure both `train_focus.py` and the synthetic orchestrators can dynamically target any index CSV via CLI arguments.
4. **Verify:** Run a focus prediction dry-run crossing datasets (e.g., train on `he_index.csv`, validate on `jiang_index.csv`).

## Completion Criteria
- [ ] All 4 datasets can be indexed and exported to standard plain-text CSV format.
- [ ] Dynamic loading occurs from VSI, OME-TIFF, and JPEG effortlessly by the unified PyTorch Datasets.
- [ ] Legacy `shared_datasets` and `synthetic_data` folders are completely fully deprecated and safely deleted.
