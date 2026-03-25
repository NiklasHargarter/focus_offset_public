# EXP 01: Multi-Focal Detection & Discard Ablation

**Status:** Not Started
**Dependencies:** Phase 02, Phase 03
**Goal:** Two-part experiment. Part A: determine whether multi-focal slides are a real, measurable problem and quantify the argmax labelling error in µm. Part B: test whether discarding multi-focal patches from training actually improves model accuracy.

## Motivation

Phase 02 uses global argmax as the in-focus reference. This is correct for uniformly focused slides but introduces label noise on slides with spatial depth variation. This experiment first quantifies the problem, then measures its practical impact on the trained model.

---

## Part A: Spatial Focus Map & Error Quantification

### Steps

1. **Spatial focus map** — for each tissue patch in the Phase 02 CSV, find per-location argmax z-level `(x, y) → z_local`. Produce a spatial heatmap per slide.
2. **Variability quantification** — compute std of `z_local` across all tissue patches per slide. Distribution across all slides in `small`.
3. **Error estimation** — `error_um = (z_local - z_global_argmax) * z_step_um` per patch. Show distribution per slide and dataset-wide.
4. **Visual summary** — spatial in-focus map for 3 representative slides (low / medium / high variability), with error scale bar in µm.

### Implementation
- Pure pandas + numpy + seaborn
- Input: `data/indices/ZStack_HE_small.csv`
- Script: `focus_offset/experiments/multifocal_analysis.py --tier small`

### Decision Gate (Part A)
- **Negligible** (std < 1 z-step in >80% of slides): document in `research.md` that argmax is valid. Part B can be skipped.
- **Significant** (std ≥ 1 z-step in >20% of slides): proceed to Part B.

---

## Part B: Model Performance Discard Ablation

Requires Part A to confirm multi-focal is significant, and requires Phase 03 baseline model.

### Steps

1. **Multi-focal patch labelling** — add `is_multifocal: bool` column to the HE CSV. A patch is multi-focal if its local argmax z deviates from the slide global argmax by ≥1 z-step.
2. **Filtered training set** — create a filtered version of the index with `is_multifocal=False` only. This is a CSV filter, not a code change.
3. **Retrain on filtered set** — run `train_focus.py --dataset ZStack_HE --tier small --domain spatial` using the filtered index. 2 epochs, same hyperparameters as Phase 03 baseline.
4. **Compare** — val MAE of filtered model vs Phase 03 baseline model. Is the gap meaningful?
5. **Visual proof** — loss curves side-by-side (baseline vs filtered). Report val MAE table.

### Implementation
- No new ML modules — reuse Phase 03 training pipeline
- Script: extend `focus_offset/experiments/multifocal_analysis.py --ablation`

---

## Completion Criteria

### Part A
- [ ] Spatial focus maps produced for ≥3 slides
- [ ] Variability distribution computed across dataset
- [ ] Error distribution in µm computed
- [ ] **Visual proof:** `artifacts/exp_01/part_a/` — spatial maps + error distribution

### Part B (only if Part A gate triggers)
- [ ] `is_multifocal` column added to HE index
- [ ] Filtered training run completes on `small`
- [ ] Val MAE comparison saved
- [ ] **Visual proof:** `artifacts/exp_01/part_b/` — loss curves + MAE comparison table

## Notes

- Part B reuses Phase 03's training pipeline unchanged — the only difference is the filtered CSV input.
- Results from Part A and B should be promoted to `research.md` by the user after review.
- Do **not** modify `SharedIndexer` or `OffsetPredictionDataset` — all filtering is done at the CSV level.
- Can be run in parallel with Phase 04 and Phase 05.
