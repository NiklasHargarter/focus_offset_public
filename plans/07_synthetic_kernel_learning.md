# Phase 07: Synthetic Kernel Learning

**Status:** Not Started
**Dependencies:** Phase 03
**Goal:** Learn the physical point-spread function (PSF) of microscope defocus. Two sub-phases: first prove the architecture recovers known synthetic kernels; then apply it to real paired data.

## Requirements Addressed
- Synthetic Data Generation & Kernel Learning
- Simulation Validation (separate from real data)
- True PSF Prediction from real data pairs

## Interface Decisions

> To be populated by `/design-an-interface` during Phase 00 and refined after Phase 03.

Expected decisions:
- `SimulationDataset(kernel_fn, n_samples, patch_size)` — yields `(sharp_patch, blurred_patch, kernel)` from synthetic data
- `SyntheticPairDataset(index_path, root, z_offset_target)` — yields `(in_focus_patch, target_offset_patch)` from real data, using Phase 02 CSV to find paired patches
- `KernelLearningModel` — architecture stub (small CNN or deconvolution net)
- `train_kernel.py --mode simulation | real --tier small` — orchestrator CLI
- Success criterion for simulation phase: how close must learned kernel be to teacher kernel?

## Sub-Phase A: Simulation Validation

Prove the model can recover a known convolutional operator before touching real data.

### Steps

1. **Teacher kernel generator** — generates known 2D Gaussian and disk blur kernels at parameterised sizes. Write pytest: kernels sum to 1, shapes are correct.
2. **`SimulationDataset`** — synthesises `(sharp_patch, blurred_patch)` pairs by convolving sharp patches with teacher kernel. Sharp patches can be random noise or real HE patches. Write pytest: blurred patch is detectable as more blurred than sharp (via focus metric).
3. **`KernelLearningModel`** — simple architecture. Write pytest: output shape matches kernel shape.
4. **Simulation orchestrator** — `train_kernel.py --mode simulation --tier small`. Trains model to predict teacher kernel from `(sharp, blurred)` pairs.
5. **Visual proof experiment:** overlay predicted kernel vs teacher kernel shape for 3 kernel types. Convergence plot.

### Simulation Completion Gate

Before proceeding to Sub-Phase B: model must recover teacher kernel shape within acceptable tolerance (e.g. >0.9 cosine similarity). If not, Sub-Phase B does not begin.

## Sub-Phase B: Real Data Pairing

Apply the validated architecture to real microscope data.

### Steps

1. **`SyntheticPairDataset`** — uses Phase 02 CSV to find pairs: for each in-focus patch (argmax z), find the same `(slide, x, y)` at a target `z_offset`. Uses `focus_offset_um` column. Write pytest with a synthetic CSV.
2. **Real data orchestrator** — `train_kernel.py --mode real --dataset ZStack_HE --tier small`.
3. **Visual proof experiment:** visualise predicted PSF for 3 slides. Show predicted blur kernel shape. Compare to expected Gaussian defocus model.

## Completion Criteria

### Sub-Phase A
- [ ] Pytest passes for teacher kernel generator
- [ ] Pytest passes for `SimulationDataset`
- [ ] Model converges on teacher kernel (>0.9 cosine similarity)
- [ ] **Visual proof:** `artifacts/phase_07/simulation/` — kernel overlay + convergence curve

### Sub-Phase B
- [ ] Pytest passes for `SyntheticPairDataset`
- [ ] `train_kernel.py --mode real` runs without crashing on `small`
- [ ] **Visual proof:** `artifacts/phase_07/real/` — predicted PSF visualisations for 3 slides

## Notes

- `SyntheticPairDataset` uses global argmax in-focus z from Phase 02 — not blocked on peak detection. If multi-focal is a real problem, it will be visible in the predicted PSF (inconsistent or multi-lobed kernels). Flag as `research.md` candidate.
- Real-data PSF quality depends heavily on tissue type and slide quality. AgNor may produce cleaner PSF estimates (different scanner, better z-spacing). Log findings as candidates.
- Sub-Phase B **only begins** after Sub-Phase A passes the convergence gate. Document the gate result in the completion summary.
