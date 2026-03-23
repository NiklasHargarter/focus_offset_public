# Phase 4: Configurable Stain Augmentation Injection

**Status:** Not Started  
**Dependencies:** Phase 3 (Feature Domain Ablation)  
**Goal:** Rip out hardcoded `albumentations` logic heavily coupled to older dataloaders. Replace it with a robust `AugmentationProvider` interface.

## Requirements Addressed
- Stain Invariance & Augmentations

## Vertical Slice Execution Steps
1. **Augmentation Module:** Create an `AugmentationProvider` that generates transformation compositions (e.g., "Baseline", "Stain-Normalized", "Aggressive Jitter").
2. **Dataset Integration:** Inject the configured augmentation pipeline into `OffsetPredictionDataset` cleanly.
3. **CLI Orchestration Update:** Expose augmentation strategies to `train_focus.py` via an argument (e.g., `--aug-preset stain-norm`).
4. **Delete Legacy Hardcoding:** Locate and delete `shared_datasets/vsi/loader.py` and any other orphaned hardcoded augment pipelines.
5. **Verify:** Run a dry-run training loop over the HE dataset specifying an aggressive augmentation preset and observe successful execution.

## Completion Criteria
- [ ] Albumentations config is completely decoupled from the dataset class via an injected provider.
- [ ] `train_focus.py` handles the new `--aug-preset` CLI flag.
- [ ] Legacy loader configurations are safely deleted.
