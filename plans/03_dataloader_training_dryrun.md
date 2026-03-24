# Phase 03: DataLoader & Training Dry-Run (ZStack HE)

**Status:** Not Started
**Dependencies:** Phase 02
**Goal:** Prove the full vertical slice end-to-end: CSV index → DataLoader → model → training loop. Uses ZStack HE only. After this phase, the foundational architecture is locked.

## Requirements Addressed
- OffsetPredictionDataset interface
- Deep Modules & Slim Interfaces
- Command-Line First orchestrator
- Tests First at operational boundaries

## Interface Decisions

> To be populated by `/design-an-interface` during Phase 00 and refined after Phase 02.

Expected decisions:
- `OffsetPredictionDataset(index_path: Path, root: Path, transform: TransformStrategy)` — constructor
- `OffsetPredictionDataset.__getitem__` → `(Tensor, Tensor)` — image patch + 0-dim float offset tensor
- `OffsetPredictionDataset.__len__` — filters on `is_tissue=True` and `split` column
- `DataModule(dataset: str, tier: str, domain: str)` — PyTorch Lightning DataModule
- `train_focus.py --dataset ZStack_HE --tier small --domain spatial` — CLI shape
- Model registry: `get_model(name: str) -> nn.Module` — lookup by string key

## Vertical Slice Execution Steps

1. **OffsetPredictionDataset** — reads Phase 02 CSV, filters `is_tissue=True`, loads VSI patches on-the-fly via `slideio`. Yields `(patch_tensor, focus_offset_um_tensor)`. Write pytest using a synthetic CSV + mock patch loader.
2. **TransformStrategy (spatial stub)** — identity-style spatial transform (resize, normalize). Implement the base class interface only; FFT/DWT variants come in Phase 04.
3. **Model registry** — `get_model("resnet18")` returns a simple ResNet with a scalar regression head. Scavenge architecture from legacy if applicable.
4. **DataModule** — wraps `OffsetPredictionDataset` with train/val splits from the CSV `split` column.
5. **Orchestrator `train_focus.py`** — slim CLI: `--dataset`, `--tier`, `--domain`, `--model`. Instantiates DataModule + model + PyTorch Lightning Trainer. Runs for 2 epochs on `small`.
6. **Visual proof experiment** — run `train_focus.py --dataset ZStack_HE --tier small --domain spatial --model resnet18` for 2 epochs. Save loss curve plot.

## Completion Criteria

- [ ] Pytest passes for `OffsetPredictionDataset` (synthetic CSV test)
- [ ] Pytest passes for model registry (`get_model` returns correct output shape)
- [ ] `train_focus.py --dataset ZStack_HE --tier small --domain spatial` runs 2 epochs without crashing
- [ ] Loss decreases (or at minimum does not NaN) over 2 epochs
- [ ] **Visual proof:** `artifacts/phase_03/` contains training loss curve (train + val) for the 2-epoch run

## Notes

- `TransformStrategy` base class is defined here but only the spatial variant is implemented. Phase 04 adds FFT/DWT.
- The dataset filters on `is_tissue=True` and the `split` column — do not add additional filtering logic here.
- If loss is NaN from epoch 1, it is a data issue (check `focus_offset_um` scale), not a model issue. Log this to `research.md` as a candidate.
- Legacy `focus_offset/models/architectures.py` may be scavenged — read it, copy what's useful, do not import from it.
