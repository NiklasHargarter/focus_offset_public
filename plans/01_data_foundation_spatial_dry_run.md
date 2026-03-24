# Phase 1: Minimum Viable Data Foundation & Spatial Dry-Run

**Status:** Not Started  
**Dependencies:** None  
**Goal:** Prove the foundational architecture. Extract the core `Shared Indexer` logic and connect it to a simple dummy spatial model and a top-level orchestrator. 

## Requirements Addressed
- Shared Index & Split Datasets Pattern
- Deep Modules & Slim Interfaces
- Testing & Reproducibility (Initial)

## Vertical Slice Execution Steps
1. **Create the Shared Indexer:** Build a new core indexing module (e.g., in a `src/` hierarchy) that scans the `ZStack HE` dataset across defined magnifications to generate the **Exhaustive Pyramid Index (Flat CSV)** containing the spatial patch grid coordinates `(x,y)` and absolute z-levels.
2. **Create `OffsetPredictionDataset`:** Implement the PyTorch Dataset interface that wraps the indexer and yields `(any_patch, focus_offset)`.
3. **Build Dummy Model:** Implement or scavenge a very simple spatial model (e.g., RGB ResNet) from legacy code, placing it in an appropriate new module.
4. **Create Orchestrator:** Write a primary execution script—a slim CLI interface that takes basic config parameters, instances the dataset and model, and runs a dummy PyTorch training loop (logging fake/real losses to stdout).
5. **Verify:** Run the orchestrator script utilizing the new clean-slate modules. Ensure it parses HE data, feeds it to the model, and completes an epoch.

## Completion Criteria
- [ ] Pytest passes for the generic dataset index generator.
- [ ] `train_focus.py` runs end-to-end without crashing.
- [ ] Legacy HE loading code is marked for deprecation.
