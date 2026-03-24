# Phase 01: Raw Data Acquisition & Sanitization

**Status:** Not Started
**Dependencies:** Phase 00
**Goal:** Download, sanitize, and physically subset all four raw datasets into clean local directories. Produces nothing except data on disk — no PyTorch, no indexing, no ML dependencies.

## Requirements Addressed
- Phase 0: Ingestion & Sanitization Separation (strict isolation from PyTorch ecosystem)
- Multi-tiered ingestion: `full`, `normal`, `small` physical subsets
- VSI ZIP structure repair on disk
- Jiang2018 scale correction

## Interface Decisions

> To be populated by `/design-an-interface` during Phase 00.

Expected decisions:
- `sync_dataset(dataset: str, tier: str) -> None` — CLI entry point signature
- Directory layout conventions: `data/<dataset>/<tier>/`
- Tier size definitions: how many slides per tier for HE/IHC/AgNor, how many patches for Jiang2018
- Sanitization report format (what gets logged when a VSI ZIP is repaired)

## Datasets

| Dataset | Format | Source | Notes |
|---------|--------|--------|-------|
| ZStack HE | Olympus VSI | Own team (EXACT server) | Primary training data |
| ZStack IHC | Olympus VSI | Own team (EXACT server) | Same pipeline as HE, different dir |
| AgNor | OME-TIFF | Own team | Different scanner, has z-stacks |
| Jiang2018 | JPEG patches (pre-tiled) | Public | Scale correction required |

## Vertical Slice Execution Steps

1. **Implement sync utility** — isolated from PyTorch; plain Python + requests/paramiko only. CLI: `uv run python -m focus_offset.data.sync --dataset ZStack_HE --tier small`
2. **Multi-tier physical subsetting** — for VSI datasets, copy a fixed number of slides per tier into `data/<dataset>/<tier>/`. Never filter logically — copy physical files.
3. **VSI ZIP sanitization** — detect and repair malformed VSI ZIP structures on disk. Log repairs to `data/<dataset>/sanitization_report.json`.
4. **AgNor download** — OME-TIFF files via same sync mechanism. Verify z-stack structure is intact (multiple z-planes per file).
5. **Jiang2018 download + scale correction** — download patch archives, apply scale correction, organize into `data/Jiang2018/<tier>/`.
6. **Verification dry-run** — open one file from each dataset/tier using its respective reader and assert the expected structure (number of z-planes, image shape, etc.).

## Completion Criteria

- [ ] `data/ZStack_HE/small/`, `data/ZStack_IHC/small/`, `data/AgNor/small/`, `data/Jiang2018/small/` all exist and are non-empty
- [ ] Sanitization report exists for VSI datasets
- [ ] Verification dry-run passes for all 4 datasets
- [ ] Sync utility has no PyTorch imports (enforced by a lint check or import test)
- [ ] **Visual proof:** log output showing file counts and one sample file path per dataset/tier

## Notes

- This phase has **zero ML dependencies** by design — it must be runnable without the full venv if needed
- `small` tier: ~3 slides for HE/IHC/AgNor; ~500 patches for Jiang2018 (exact numbers to be fixed in Phase 00)
- IHC is identical pipeline to HE — reuse the same sync code with different source paths
