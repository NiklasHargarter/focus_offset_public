# EXP 01: Multi-Focal Peak Detection

**Status:** Not Started
**Dependencies:** Phase 02
**Goal:** Determine whether multi-focal slides (where the in-focus z-level varies spatially across the slide) are a real, measurable problem in this dataset. If yes, quantify the extent and show that argmax-based focus reference introduces systematic error. Does NOT integrate with the training pipeline.

## Motivation

Phase 02 uses global argmax (single best z-level per slide) as the in-focus reference. This is correct for uniformly focused slides but wrong for slides with depth variation — thick tissue sections, coverslip tilt, etc. This experiment proves whether that assumption is violated.

## Experiment Design

### Step 1 — Spatial Focus Map
For a sample of slides from ZStack HE (and optionally AgNor), compute `focus_metric` per patch per z-level (already in the Phase 02 CSV). For each `(x, y)` location, find the argmax z-level. Produce a spatial heatmap: `in_focus_z(x, y)` across the slide.

### Step 2 — Variability Quantification
Compute the standard deviation of `in_focus_z` across all tissue patches per slide. Slides with high std are multi-focal. Produce a distribution across the dataset.

### Step 3 — Error Estimation
For multi-focal slides, estimate the systematic error introduced by global argmax: `error_um = (local_infocus_z - global_infocus_z) * z_step_um`. Show error distribution per slide.

### Step 4 — Visual Summary
Produce a report-quality figure: spatial in-focus map for 3 representative slides (low/medium/high multi-focal variability), with error scale bar in microns.

## Implementation

- All analysis is pure pandas + numpy + matplotlib/seaborn — no new ML modules
- Input: `data/indices/ZStack_HE_small.csv` (Phase 02 output)
- Output script: `focus_offset/experiments/multifocal_analysis.py`
- Run: `uv run python -m focus_offset.experiments.multifocal_analysis --tier small`

## Completion Criteria

- [ ] Script runs without crashing on `small` tier
- [ ] Spatial in-focus map produced for ≥3 slides
- [ ] Variability distribution across dataset produced
- [ ] Error estimation in microns computed
- [ ] **Visual proof:** `artifacts/exp_01/` contains spatial focus maps + error distribution figure

## Decision Gate

After reviewing the visual proof:
- If multi-focal variability is **negligible** (std < 1 z-step across slides): document in `research.md` that argmax is valid. No further action.
- If multi-focal variability is **significant** (std ≥ 1 z-step in >20% of slides): document magnitude, consider whether a future phase implements local argmax or full peak detection. This remains an experiment — not automatically integrated.

## Notes

- This experiment requires Phase 02 to be complete but has no dependency on Phase 03 or later.
- Can be run in parallel with Phase 03/04/05 if desired.
- Results should be promoted to `research.md` by the user after reviewing the artifact.
- Do **not** modify `SharedIndexer` or `OffsetPredictionDataset` in this experiment.
