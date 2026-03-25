# Phase 02: Preprocessing & CSV Index (ZStack HE)

**Status:** Not Started
**Dependencies:** Phase 00, Phase 01
**Goal:** Build the full preprocessing pipeline for ZStack HE and produce a stable, self-contained CSV index. This index is the foundation for all downstream DataLoader phases. Also establish methodology decisions that constitute Goal 0 contributions: focus metric selection, magnification strategy, tissue filtering, and loading performance.

## Requirements Addressed
- Shared Index & Split Datasets Pattern (Goal 0)
- Atomic Raw Metric Index — flat CSV in physical units (Goal 0)
- Focus metric evaluation: Brenner vs Laplacian vs others (Goal 0, Goal 1)
- Multi-magnification index creation: native / 2× / 4× (Goal 0, Goal 1)
- Tissue filtering validation: adaptive Otsu per-slide (Goal 0)
- Loading performance benchmark: on-the-fly vs pre-extracted (Goal 0)
- Physical offset labelling in µm (Goal 0)

## Interface Decisions

> To be populated by `/design-an-interface` during Phase 00.

Expected decisions:
- `SharedIndexer(root: Path, tier: str, magnification: str = "native", max_slides: int = 3)` — magnification is `"native"`, `"2x"`, or `"4x"`
- `SharedIndexer.build() -> pd.DataFrame` — full scan, writes CSV
- `SharedIndexer.load() -> pd.DataFrame` — reads existing CSV
- CSV schema (canonical — all phases depend on this):
  - `slide_name: str`
  - `x: int` (pixel coordinate, top-left of patch at native resolution)
  - `y: int`
  - `z_level: int` (raw integer z-index)
  - `focus_offset_um: float` (signed µm from in-focus z)
  - `focus_metric: float` (winning metric score — decided in this phase)
  - `is_tissue: bool`
  - `split: str` (`train` / `val` / `test`)
  - `magnification: str`
- `compute_focus_metric(patch: np.ndarray, method: str) -> float` — supports `"brenner"`, `"laplacian"`, `"variance"`
- `detect_infocus_z(metrics: np.ndarray) -> int` — argmax (peak detection deferred to EXP 01)

## Vertical Slice Execution Steps

### Core Pipeline

1. **Focus metric module** — implement `compute_focus_metric` for Brenner gradient, Laplacian variance, and pixel variance. Write pytest: each method returns higher values for sharper synthetic images.
2. **Focus metric comparison experiment** — on 3 HE slides, compute all three metrics per patch per z-level. Plot z-profiles for each metric side-by-side. Identify which produces the cleanest, most peaked profile. Log winner as `research.md` candidate. Use winner for all downstream steps.
3. **SharedIndexer** — scan ZStack HE, extract patch grid, compute winning focus metric, detect in-focus z via argmax, compute `focus_offset_um`. Hard limit: `max_slides=3`.
4. **Tissue filtering** — Otsu thresholding on downsampled thumbnail, per-slide adaptive threshold. Store `is_tissue`. Write pytest with synthetic slide.
5. **z_step_um resolution** — read from VSI metadata; fall back to `config.Z_STEP_UM_HE`. Log actual value as `research.md` candidate.
6. **Train/val/test split** — deterministic by `slide_name` hash. Write to `split` column.
7. **CSV export** — write `data/indices/ZStack_HE_native_small.csv`. Validate schema with pandas dtype assertions in pytest.

### Multi-Magnification Indices (Goal 0)

8. **Multi-mag index creation** — run `SharedIndexer` at `magnification="2x"` and `magnification="4x"`. Produce `ZStack_HE_2x_small.csv` and `ZStack_HE_4x_small.csv` with identical schema. Patch size stays constant in pixels; field-of-view increases with downscale.

### Loading Performance Benchmark (Goal 0)

9. **Benchmark: on-the-fly VSI loading vs pre-extracted patches** — time 1000 random patch loads from VSI directly vs loading pre-saved PNG patches. Report throughput (patches/sec) at `batch_size=32` with `num_workers=4`. Save as `artifacts/phase_02/loading_benchmark.csv` and bar chart. This informs the final dataset design recommendation.

### Visual Proof

10. **Visual summary** — for 3 slides: patch grid coloured by `focus_metric`, `is_tissue` overlay, `focus_offset_um` histogram. One panel per slide.

## Completion Criteria

- [ ] Pytest passes for focus metric module (all 3 methods, synthetic sharpness test)
- [ ] Pytest passes for tissue filter (synthetic slide)
- [ ] Pytest passes for SharedIndexer CSV schema (dtype/column assertions)
- [ ] `ZStack_HE_native_small.csv`, `ZStack_HE_2x_small.csv`, `ZStack_HE_4x_small.csv` all exist
- [ ] `focus_offset_um` values are physically plausible (range ±20µm for HE)
- [ ] Loading benchmark results saved
- [ ] **Visual proof:** `artifacts/phase_02/` contains focus metric comparison plot + patch grid overlays + loading benchmark chart

## Notes

- Focus metric winner is a scientific decision — flag it as `research.md` candidate, not a code constant.
- Multi-mag indices enable EXP 02 (magnification ablation) to run without any new indexer code.
- Loading benchmark result informs the dataset design recommendation (Goal 0 contribution).
- Do **not** implement multi-focal peak detection. Argmax only. Peak detection is EXP 01.
- IHC uses identical code — different root path only. No changes needed for Phase 06.
