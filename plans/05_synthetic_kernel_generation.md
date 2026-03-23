# Phase 5: Synthetic Kernel Learning (Simulation Phase)

**Status:** Not Started  
**Dependencies:** Phase 2 (Preprocessing Export)  
**Goal:** Prove mathematically that the model architecture can successfully recover a known convolutional operator. We will train the model against dynamically generated targets created by a "Teacher Kernel" using purely in-focus patches.

## Requirements Addressed
- Synthetic Kernel Learning Validation (Phase 1: Proof of Concept)

## Vertical Slice Execution Steps
1. **Teacher Kernel Generator:** Extract and upgrade the `get_teacher_kernel` logic to support both the existing morphological `disk` (circle) as well as a new `square` shape.
2. **Explicit `SimulationDataset`:** Rip out the unsafe `simulation_mode` boolean config flag. Create a strictly isolated `SimulationDataset` Python class that *only* serves mathematically convolved targets and explicitly documents its simulation nature. This absolutely prevents any accidental confusion with real data results.
3. **Simulation Orchestrator:** Create `train_synthetic_sim.py`. This distinct script instantiates the `SimulationDataset`, applies the chosen Teacher Kernel on the GPU, and trains the model strictly to minimize `K-MAE`.
4. **Verify:** Run the orchestrator twice: once with the `circle` teacher and once with the `square` teacher. Confirm the model successfully learns both shapes and correctly plots the recovered kernel heatmaps.

## Completion Criteria
- [ ] Teacher kernel supports dynamically switching between circle and square.
- [ ] Model weights reliably converge to match the exact mathematical shape of the Teacher Kernel.
- [ ] Legacy simulation logic in `synthetic_data/scripts/train_synthetic.py` is safely replaced.
