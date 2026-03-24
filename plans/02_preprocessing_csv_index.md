# Phase 02: Preprocessing & CSV Index (ZStack HE)

**Status:** Not Started
**Dependencies:** Phase 00, Phase 01
**Goal:** Build the full preprocessing pipeline for ZStack HE and produce a stable, self-contained CSV index. This index is the foundation for all downstream DataLoader phases. Get it right here — all other datasets extend this schema in Phase 06.

## Requirements Addressed
- Shared Index & Split Datasets Pattern
- Atomic Raw Metric Index (flat CSV)
- Focus Metric & Multi-Focal Resolution (argmax only — peak detection is a separate experiment)
- Preprocessing & Tissue Filtering Validation
- Adaptive tissue filtering (Otsu's method)

## Interface Decisions

> To be populated by `/design-an-interface` during Phase 00.

Expected decisions:
- `SharedIndexer(root: Path, tier: str, max_slides: int = 3)` — exact constructor
- `SharedIndexer.build() -> pd.DataFrame` — triggers full scan and writes CSV
- `SharedIndexer.load() -> pd.DataFrame` — reads existing CSV
- CSV schema (canonical, all phases depend on this):
  - `slide_name: str`
  - `x: int` (pixel coordinate, top-left of patch)
  - `y: int`
  - `z_level: int` (raw integer z-index within the slide)
  - `focus_offset_um: float` (signed physical offset from in-focus z, in microns)
  - `focus_metric: float` (Brenner gradient score or similar)
  - `is_tissue: bool` (Otsu tissue mask result)
  - `split: str` (`train` / `val` / `test`)
- `compute_focus_metric(patch: np.ndarray) -> float` — metric function signature
- `detect_infocus_z(metrics: np.ndarray) -> int` — returns argmax z-index (peak detection deferred)

## Vertical Slice Execution Steps

1. **Focus metric module** — implement `compute_focus_metric` (Brenner gradient). Write pytest proving it returns higher values for sharper images using synthetic data.
2. **SharedIndexer** — scan ZStack HE slides, extract patch grid at defined magnification, compute focus metric per patch per z-level, detect in-focus z via argmax, compute `focus_offset_um = (z - z_infocus) * z_step_um`. Hard limit: `max_slides=3` default.
3. **Tissue filtering** — Otsu's method on downsampled slide thumbnail. Store `is_tissue` per patch in CSV. Write pytest with a synthetic slide.
4. **z_step_um resolution** — read from VSI slide metadata if available; fall back to `config.Z_STEP_UM_HE` constant. Log which was used.
5. **Train/val/test split assignment** — deterministic by `slide_name` hash. Write to `split` column.
6. **CSV export** — write to `data/indices/ZStack_HE_<tier>.csv`. Verify schema with pandas dtype assertions in pytest.
7. **Visual proof experiment** — for 3 sample slides: plot patch grid coloured by `focus_metric`, overlay `is_tissue` mask, show focus offset distribution histogram.

## Completion Criteria

- [ ] Pytest passes for focus metric module (synthetic sharpness test)
- [ ] Pytest passes for tissue filter module (synthetic slide test)
- [ ] Pytest passes for SharedIndexer CSV schema (dtype/column assertions)
- [ ] `data/indices/ZStack_HE_small.csv` exists and has correct schema
- [ ] `focus_offset_um` values are physically plausible (e.g. range ±20µm)
- [ ] **Visual proof:** `artifacts/phase_02/` contains patch grid plot + focus metric overlay + offset histogram for 3 slides

## Notes

- `z_step_um` is slide-level in principle but fixed per dataset in practice. Log the actual value used so it can be promoted to `research.md`.
- IHC uses identical code — just a different root path. No changes needed for Phase 06 to add IHC.
- Do **not** implement multi-focal peak detection here. Argmax only. Peak detection is `EXP_01`.
- Patch size and magnification level are fixed constants from `config.py` — do not expose as CLI args.
