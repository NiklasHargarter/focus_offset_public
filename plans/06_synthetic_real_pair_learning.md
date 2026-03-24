# Phase 6: Synthetic Kernel Generation (Real Data Pairing)

**Status:** Not Started  
**Dependencies:** Phase 5 (Synthetic Kernel Simulation)  
**Goal:** Transition from mathematical simulation to real biological focus offsets. Utilize the Shared Indexer to yield perfect `(infocus_patch, target_offset_patch)` pairs and learn the true physical point-spread function of the microscope's defocus.

## Requirements Addressed
- Synthetic Kernel Learning Validation (Phase 2: True Kernel Extraction)

## Vertical Slice Execution Steps
1. **Implement `SyntheticPairDataset`:** Create the PyTorch dataset that takes the `.csv` index and enforces the filtering strategy to yield strict `(infocus_patch, target_offset_patch)` spatial pairs.
2. **Real Data Orchestrator:** Create the automated CLI training script to learn the real microscope point-spread function from the dataset.
3. **Visual Proof Experiment:** Run the trained real-data model inside an isolated experiment script and utilize **seaborn** to save a beautiful kernel heatmap representing the true physical defocus blur, safely adhering to the "Core vs Experiment" IO separation rule.

## Completion Criteria
- [ ] `SyntheticPairDataset` correctly enforces the "at most one offset per patch" yielding rule based on the shared indexer CSV.
- [ ] The model can successfully learn kernels directly from paired VSI patches instead of mathematically simulated targets.
