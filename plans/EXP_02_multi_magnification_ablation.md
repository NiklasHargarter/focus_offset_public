# EXP 02: Multi-Magnification Model Ablation

**Status:** Not Started
**Dependencies:** Phase 02, Phase 03
**Goal:** Determine the optimal magnification level for focus offset prediction. Does a wider field-of-view (lower magnification) give the model more context and improve accuracy, or does native resolution detail matter most?

## Motivation

Phase 02 produces three CSV indices at native, 2×, and 4× downscale (same patch size in pixels, increasing field-of-view). This experiment trains equivalent models on each and compares prediction accuracy. The result is a Goal 0 dataset design recommendation: which magnification should future work use?

## Experiment Design

All three conditions use identical hyperparameters, architecture, and `--domain spatial`. The only variable is which magnification index is loaded.

### Steps

1. **Verify indices exist** — confirm `ZStack_HE_native_small.csv`, `ZStack_HE_2x_small.csv`, `ZStack_HE_4x_small.csv` from Phase 02.
2. **Train native** — `train_focus.py --index ZStack_HE_native_small --tier small --domain spatial`. Save checkpoint.
3. **Train 2×** — same, with `ZStack_HE_2x_small`.
4. **Train 4×** — same, with `ZStack_HE_4x_small`.
5. **Compare** — val MAE per condition. Also compare training loss curves for convergence speed.
6. **Visual proof** — 3-panel loss curve comparison + MAE bar chart.

### Implementation
- No new ML modules — reuses Phase 03 training pipeline
- Only change: `--index` flag to select which CSV to load (or handle via DataModule config)
- Script: `train_focus.py` with appropriate index flag

## Completion Criteria

- [ ] All 3 training runs complete without crashing on `small`
- [ ] Val MAE recorded for each magnification
- [ ] **Visual proof:** `artifacts/exp_02/` — loss curves + MAE comparison bar chart

## Decision Gate

After reviewing results, promote the magnification recommendation to `research.md`:
- Which magnification gives best MAE?
- Is there a speed/accuracy trade-off (4× trains faster, does it pay off)?
- Does the winning magnification match the physical intuition for defocus blur size?

## Notes

- This experiment can run in parallel with Phase 04 and Phase 05.
- The winning magnification informs Phase 06 and Phase 07 multi-dataset indices — if 2× wins, AgNor and Jiang2018 should also be indexed at 2×.
- No new modules needed — entirely driven by CSV index selection.
