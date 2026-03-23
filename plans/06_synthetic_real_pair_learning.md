# Phase 6: Synthetic Kernel Generation (Real Data Pairing)

**Status:** Not Started  
**Dependencies:** Phase 5 (Synthetic Kernel Simulation)  
**Goal:** Transition from mathematical simulation to real biological focus offsets. Utilize the Shared Indexer to yield perfect `(infocus_patch, target_offset_patch)` pairs and learn the true physical point-spread function of the microscope's defocus.

## Requirements Addressed
- Synthetic Kernel Learning Validation (Phase 2: True Kernel Extraction)

## Vertical Slice Execution Steps
1. **Implement `SyntheticPairDataset`:** Create the PyTorch dataset that takes the `.csv` index and enforces the filtering strategy to yield strict `(infocus_patch, target_offset_patch)` spatial pairs.
2. **Real Data Orchestrator:** Create `train_synthetic_real.py`. 
3. **Verify:** Run the orchestrator over the HE dataset for a specific physical offset (e.g., +10µm). Ensure the model successfully converges and saves a kernel heatmap representing the true physical defocus blur.

## Completion Criteria
- [ ] `SyntheticPairDataset` correctly enforces the "at most one offset per patch" yielding rule based on the shared indexer CSV.
- [ ] The model can successfully learn kernels directly from paired VSI patches instead of mathematically simulated targets.
