# Phase 06: Multi-Dataset Expansion

**Status:** Not Started
**Dependencies:** Phase 02, Phase 03
**Goal:** Extend the SharedIndexer to cover all 4 datasets. The Phase 02 CSV schema is the canonical contract — each new dataset adapter must conform to it. After this phase, `train_focus.py` can train on any combination of datasets.

## Requirements Addressed
- Expand SharedIndexer for OME-TIFF (AgNor) and JPEG (Jiang2018)
- Dynamic dataset routing in DataLoader
- Cross-dataset benchmarking foundation

## Interface Decisions

> To be populated by `/design-an-interface` during Phase 00 and refined after Phase 03.

Expected decisions:
- `SlideReader` protocol — `read_patch(slide_path, x, y, z, size) -> np.ndarray` — common interface over VSI/OME-TIFF
- `OmeTiffReader` — implements `SlideReader` using `tifffile`
- `JiangAdapter` — reads pre-tiled JPEGs, maps filename metadata to CSV schema columns
- `SharedIndexer(root, tier, dataset: str)` — dataset-aware constructor variant, or factory `IndexerFactory.get(dataset)`
- How `OffsetPredictionDataset` selects the correct reader at runtime (by dataset column or config)
- Multi-dataset DataModule: `--dataset ZStack_HE,ZStack_IHC,AgNor,Jiang2018`

## Datasets Added This Phase

### ZStack IHC
Identical pipeline to HE — same VSI reader, same `SharedIndexer`, different `root` path. No code changes needed; just run the indexer on the IHC root.

### AgNor (OME-TIFF)
- Read z-stacks using `tifffile` / OME-TIFF metadata
- Same focus metric + argmax pipeline as Phase 02
- z_step_um from OME-TIFF metadata; fall back to `config.Z_STEP_UM_AGNOR`
- Output: `data/indices/AgNor_small.csv` with same schema

### Jiang2018 (JPEG patches — unified schema adapter)
- Pre-tiled patches: focus offset encoded in filename/metadata
- `JiangAdapter` maps to CSV schema: `slide_name` from parent folder, `focus_offset_um` from metadata, `x/y/z_level` filled with available data
- Additional Jiang2018 metadata (if available) used to populate genuine schema fields
- Output: `data/indices/Jiang2018_small.csv` with same schema

## Vertical Slice Execution Steps

1. **`SlideReader` protocol** — define common interface. Write pytest: both `VsiReader` and `OmeTiffReader` satisfy the protocol.
2. **`OmeTiffReader`** — implement using tifffile. Write pytest against a synthetic OME-TIFF.
3. **AgNor indexer run** — extend `SharedIndexer` to accept `dataset="AgNor"` and use `OmeTiffReader`. Produce `AgNor_small.csv`.
4. **IHC indexer run** — run existing `SharedIndexer` on IHC root. Produce `ZStack_IHC_small.csv`. Confirm no code changes needed.
5. **`JiangAdapter`** — parse Jiang2018 filenames/metadata, write conforming CSV. Write pytest: output schema matches canonical schema exactly.
6. **Multi-dataset DataModule** — accept comma-separated `--dataset` list, load and concatenate indices, route patch loading to correct reader per row.
7. **Visual proof experiment** — for each of the 4 datasets: plot `focus_offset_um` distribution. Show on a single 4-panel figure to visualise cross-dataset range differences.

## Completion Criteria

- [ ] `data/indices/ZStack_HE_small.csv`, `ZStack_IHC_small.csv`, `AgNor_small.csv`, `Jiang2018_small.csv` all exist with identical schema
- [ ] Pytest passes for `OmeTiffReader` and `JiangAdapter`
- [ ] `train_focus.py --dataset ZStack_HE,AgNor --tier small --domain spatial` runs without crashing
- [ ] `JiangAdapter` output schema passes dtype/column assertions
- [ ] **Visual proof:** `artifacts/phase_06/` contains 4-panel `focus_offset_um` distribution plot across all datasets

## Notes

- IHC is truly free here if Phase 02 code is clean — confirm this before building anything new.
- AgNor's z_step_um may differ from HE/IHC — log the discovered value as a `research.md` candidate.
- Jiang2018 metadata: investigate what fields are genuinely available before Phase 00 interface design to inform `JiangAdapter` constructor.
- The `SlideReader` protocol enables the DataLoader to be dataset-agnostic — the CSV `dataset` column drives reader selection.
