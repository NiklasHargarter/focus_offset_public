# Phase 05: Stain Augmentation Injection

**Status:** Not Started
**Dependencies:** Phase 03
**Goal:** Replace any hardcoded augmentation logic with a configurable `AugmentationProvider` interface. Enables controlled experiments on the effect of stain normalization and augmentation strategies.

## Requirements Addressed
- Stain Invariance & Augmentations: configurable augmentation pipeline, not hardcoded
- `--aug-preset` CLI flag

## Interface Decisions

> To be populated by `/design-an-interface` during Phase 00 and refined after Phase 03.

Expected decisions:
- `AugmentationProvider` base class — `__call__(image: Tensor) -> Tensor`
- `NullAugmentation` — identity, no-op (default preset)
- `StainNormAugmentation(method: str)` — Macenko, Vahadane, or Reinhard normalisation via albumentations
- `ColorJitterAugmentation` — random HSV jitter
- `get_augmentation(preset: str) -> AugmentationProvider` — registry lookup
- CLI: `--aug-preset none | stain-norm | color-jitter`
- Where augmentation is applied: training only (not val/test)

## Vertical Slice Execution Steps

1. **AugmentationProvider base class** — define interface. Write pytest: `NullAugmentation` is identity (input == output).
2. **StainNormAugmentation** — albumentations-based stain normalisation. Write pytest: output same shape as input, values in valid range.
3. **ColorJitterAugmentation** — random HSV. Write pytest: deterministic when seeded.
4. **Augmentation registry** — `get_augmentation(preset)` factory. Write pytest for all preset strings.
5. **DataModule integration** — pass augmentation provider to `OffsetPredictionDataset`. Applied only on training split.
6. **CLI update** — add `--aug-preset` flag to `train_focus.py`.
7. **Delete legacy hardcoding** — scan for any hardcoded augmentation logic in legacy or new modules and remove/replace.
8. **Visual proof experiment** — show 5 sample patches under each augmentation preset side-by-side.

## Completion Criteria

- [ ] Pytest passes for all augmentation implementations
- [ ] Pytest passes for augmentation registry
- [ ] `train_focus.py --aug-preset none` and `--aug-preset stain-norm` both run without crashing
- [ ] No hardcoded augmentation logic remains outside the `AugmentationProvider` system
- [ ] **Visual proof:** `artifacts/phase_05/` contains a grid of 5 patches × 3 presets showing augmentation effects

## Notes

- Augmentation is applied **after** the `TransformStrategy` in the DataLoader pipeline, or before — decide in Phase 00 and lock it. The order matters for stain normalisation.
- The `NullAugmentation` preset is the default for all previous phases — confirm no behaviour change when `--aug-preset none`.
- Stain normalisation methods require a reference image — document how this is configured in `config.py`.
