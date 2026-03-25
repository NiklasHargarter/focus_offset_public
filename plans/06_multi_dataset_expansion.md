# Phase 06: Multi-Dataset Expansion & Cross-Dataset Evaluation

**Status:** Not Started
**Dependencies:** Phase 02, Phase 03, Phase 04
**Goal:** Extend the SharedIndexer to all 4 datasets, then run the cross-dataset evaluations required by Goals 1 and 2. This phase both expands the infrastructure and produces the generalization benchmark results.

## Requirements Addressed
- Expand SharedIndexer for OME-TIFF (AgNor) and JPEG (Jiang2018) (Goal 0)
- Dynamic dataset routing in DataLoader (Goal 0)
- Jiang2018 as external benchmark for Goal 1 (Goal 1)
- Stain & scanner generalization baseline (Goal 2)
- Feature domain × generalization evaluation (Goal 2)

## Interface Decisions

> To be populated by `/design-an-interface` during Phase 00 and refined after Phase 03.

Expected decisions:
- `SlideReader` protocol — `read_patch(slide_path, x, y, z, size) -> np.ndarray`
- `OmeTiffReader` — implements `SlideReader` using `tifffile`
- `JiangAdapter` — reads pre-tiled JPEGs, maps filename metadata to canonical CSV schema
- `SharedIndexer(root, tier, dataset: str, magnification: str)` — dataset-aware
- How `OffsetPredictionDataset` routes to correct reader at runtime (CSV `dataset` column)
- Multi-dataset DataModule: `--dataset ZStack_HE,ZStack_IHC,AgNor,Jiang2018`
- Evaluation script interface: `evaluate.py --ckpt <path> --datasets <list> --domain <domain>`

## Datasets Added This Phase

### ZStack IHC
Identical pipeline to HE — same VSI reader, same `SharedIndexer`, different root path. Produces `ZStack_IHC_small.csv`. No new code if Phase 02 is clean.

### AgNor (OME-TIFF)
Read z-stacks via `tifffile`. Same focus metric + argmax pipeline as Phase 02. z_step_um from OME-TIFF metadata, fall back to `config.Z_STEP_UM_AGNOR`. Produces `AgNor_small.csv`.

### Jiang2018 (JPEG patches — unified schema adapter)
Pre-tiled patches with focus offset in filename/metadata. `JiangAdapter` maps to canonical CSV schema. Additional Jiang2018 metadata fields used where available. Produces `Jiang2018_small.csv`.

## Vertical Slice Execution Steps

### Infrastructure

1. **`SlideReader` protocol** — define common interface. Write pytest: `VsiReader` and `OmeTiffReader` both satisfy protocol.
2. **`OmeTiffReader`** — implement via tifffile. Write pytest against synthetic OME-TIFF. Log AgNor z_step_um as `research.md` candidate.
3. **AgNor index** — run `SharedIndexer(dataset="AgNor")`. Produce `AgNor_small.csv`.
4. **IHC index** — run existing `SharedIndexer` on IHC root. Confirm zero code changes needed. Produce `ZStack_IHC_small.csv`.
5. **`JiangAdapter`** — parse Jiang2018 filenames/metadata, write conforming CSV. Write pytest: schema matches canonical exactly.
6. **Multi-dataset DataModule** — `--dataset` accepts comma-separated list, concatenates indices, routes patch loading via `dataset` column.

### Goal 1: External Benchmark Evaluation

7. **Jiang2018 external benchmark** — load the Phase 03 best checkpoint. Run inference on Jiang2018 test split. Report MAE in µm. This is the first fully independent external validation of the focus offset prediction model.

### Goal 2: Generalization Evaluation

8. **Baseline generalization gap** — train on ZStack HE only (Phase 03 model), evaluate on IHC, AgNor, Jiang2018 test splits. Report per-dataset MAE. This establishes the generalization baseline before any augmentation intervention.
9. **Feature domain × generalization** — for each domain from Phase 04 (`spatial`, `fft`, `dwt`): train on HE, evaluate on IHC + AgNor. Produce a domain × dataset MAE matrix. Answers: does frequency domain generalise better than RGB?

### Visual Proof

10. **Visual summary** — 4-panel `focus_offset_um` distribution across all datasets + domain × dataset MAE heatmap.

## Completion Criteria

- [ ] All 4 CSV indices exist with identical schema
- [ ] Pytest passes for `OmeTiffReader` and `JiangAdapter`
- [ ] `train_focus.py --dataset ZStack_HE,AgNor --tier small` runs without crashing
- [ ] Jiang2018 MAE evaluation result saved
- [ ] Baseline generalization gap (per-dataset MAE) saved
- [ ] Domain × dataset MAE matrix saved
- [ ] **Visual proof:** `artifacts/phase_06/` contains dataset distribution plot + generalization MAE heatmap

## Notes

- Phase 04 is a dependency because the domain × generalization evaluation requires trained domain variants.
- AgNor z_step_um discovered here — flag for `research.md` and check consistency with HE/IHC.
- Jiang2018 metadata: investigate available fields before Phase 00 interface design.
- The generalization gap measured here is the baseline that Phase 05 (stain augmentation) and EXP 03 (synthetic augmentation) aim to close.
- `SlideReader` protocol makes the DataLoader dataset-agnostic — `dataset` column in CSV drives reader selection.
