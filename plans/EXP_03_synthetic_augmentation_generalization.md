# EXP 03: Synthetic Augmentation for Generalization

**Status:** Not Started
**Dependencies:** Phase 05, Phase 06, Phase 07
**Goal:** Test whether using the learned PSF kernels from Phase 07 to synthesize out-of-focus HE patches — and mixing them into training — closes the generalization gap to IHC and AgNor measured in Phase 06.

## Motivation

Phase 06 measures the raw generalization gap: a model trained on HE underperforms on IHC and AgNor. Phase 05 tests stain augmentation as one intervention. This experiment tests a complementary approach: synthesize realistic defocus blur using the real physical PSF, making the model more robust to scanner-specific blur characteristics.

This is the closing experiment of Goal 3 — connecting synthetic kernel learning back to Goal 2 (generalization).

## Experiment Design

### Conditions

| Condition | Training data | Expected |
|-----------|--------------|----------|
| Baseline | HE only (Phase 03) | — |
| + Stain aug | HE + color-jitter/stain-norm (Phase 05) | Closes stain gap |
| + Synthetic blur | HE + synthetic OOF patches via HE PSF | Closes scanner gap |
| + Both | HE + stain aug + synthetic blur | Best generalization? |

### Steps

1. **Load Phase 07 kernels** — load `artifacts/phase_07/kernels/he_psf.pt` (and optionally `agnor_psf.pt`).
2. **Synthetic augmentation module** — implement `PSFAugmentation(kernel_path)` as a new `AugmentationProvider` preset (extends Phase 05 system). Applies learned PSF convolution at random strength during training.
3. **Write pytest** — `PSFAugmentation` output is more blurred than input (verify with focus metric).
4. **Training runs** — run all 4 conditions above on `small`. Use ZStack HE for training, evaluate on IHC + AgNor test splits.
5. **Compare** — per-dataset MAE across conditions. Does synthetic augmentation close the gap?
6. **Visual proof** — 4-condition × 3-dataset MAE heatmap. Sample synthetic patches showing PSF-augmented examples.

### Implementation
- `PSFAugmentation` extends `AugmentationProvider` from Phase 05 — adds one new preset `"psf-blur"`
- `train_focus.py --aug-preset psf-blur --tier small`
- Evaluation via `evaluate.py --ckpt <path> --datasets ZStack_IHC,AgNor`

## Completion Criteria

- [ ] Pytest passes for `PSFAugmentation`
- [ ] All 4 training conditions complete on `small`
- [ ] Per-dataset MAE matrix (4 conditions × 3 datasets) saved
- [ ] **Visual proof:** `artifacts/exp_03/` — MAE heatmap + sample PSF-augmented patches

## Decision Gate

After reviewing:
- Does synthetic blur augmentation outperform stain augmentation on scanner generalization (HE → AgNor)?
- Does combining both work better than either alone?
- Promote findings to `research.md` and consider whether `psf-blur` becomes a default preset.

## Notes

- If Phase 07 produced distinct HE and AgNor PSFs, test both as augmentation kernels — an AgNor PSF applied to HE training may generalise better to AgNor.
- This experiment closes the full arc: Goal 0 (dataset) → Goal 1 (predict) → Goal 3 (learn PSF) → Goal 2 (generalise).
