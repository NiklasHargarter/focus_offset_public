# Phase 07: Synthetic Kernel Learning

**Status:** Not Started
**Dependencies:** Phase 03, Phase 06
**Goal:** Learn the physical point-spread function (PSF) of microscope defocus. Two sub-phases: prove the architecture recovers known synthetic kernels; then learn per-dataset PSFs from real paired data. Produces kernels used by EXP 03 (synthetic augmentation for generalization).

## Requirements Addressed
- Synthetic Data Generation & Kernel Learning (Goal 3)
- Simulation Validation separate from real data (Goal 3)
- True PSF Prediction from real data pairs (Goal 3)
- Per-dataset PSF: do HE, IHC, AgNor require distinct kernels? (Goal 3)

## Interface Decisions

> To be populated by `/design-an-interface` during Phase 00 and refined after Phase 03.

Expected decisions:
- `SimulationDataset(kernel_fn, n_samples, patch_size)` — yields `(sharp_patch, blurred_patch, kernel)`
- `SyntheticPairDataset(index_path, root, z_offset_target)` — yields `(in_focus_patch, target_offset_patch)` using Phase 02 CSV
- `KernelLearningModel` — architecture stub (small CNN or deconvolution net)
- `train_kernel.py --mode simulation | real --dataset <name> --tier small` — CLI
- Kernel export format — how learned kernels are saved for use in EXP 03
- Success criterion for simulation gate: cosine similarity threshold (e.g. >0.9)

## Sub-Phase A: Simulation Validation

Prove the architecture can recover a known convolutional operator before touching real data.

### Steps

1. **Teacher kernel generator** — known 2D Gaussian and disk blur kernels at parameterised sizes. Write pytest: kernels sum to 1, correct shapes.
2. **`SimulationDataset`** — synthesises `(sharp_patch, blurred_patch)` pairs by convolving with teacher kernel. Write pytest: blurred patch scores lower on focus metric than sharp.
3. **`KernelLearningModel`** — small architecture. Write pytest: output shape matches kernel shape.
4. **Simulation orchestrator** — `train_kernel.py --mode simulation --tier small`.
5. **Visual proof:** overlay predicted vs teacher kernel for 3 kernel types. Convergence curve.

### Simulation Completion Gate
Model must recover teacher kernel at >0.9 cosine similarity. Sub-Phase B does not begin until gate passes. Document gate result in completion summary.

## Sub-Phase B: Real Data PSF Learning

### Steps

1. **`SyntheticPairDataset`** — uses Phase 02 CSV to pair in-focus patch (argmax z) with same location at target z_offset. Write pytest with synthetic CSV.
2. **HE PSF** — `train_kernel.py --mode real --dataset ZStack_HE --tier small`. Save learned kernel to `artifacts/phase_07/kernels/he_psf.pt`.
3. **AgNor PSF** — `train_kernel.py --mode real --dataset AgNor --tier small`. Save to `artifacts/phase_07/kernels/agnor_psf.pt`.
4. **Per-dataset PSF comparison** — overlay HE and AgNor learned kernels. Do they differ in shape, width, isotropy? This answers: is PSF scanner-specific or universal? Flag findings as `research.md` candidates.
5. **Visual proof:** PSF visualisations for HE vs AgNor side-by-side + per-dataset convergence curves.

## Completion Criteria

### Sub-Phase A
- [ ] Pytest passes for teacher kernel generator and `SimulationDataset`
- [ ] Simulation gate passes (>0.9 cosine similarity)
- [ ] **Visual proof:** `artifacts/phase_07/simulation/` — kernel overlays + convergence

### Sub-Phase B
- [ ] Pytest passes for `SyntheticPairDataset`
- [ ] HE and AgNor PSF kernels saved to `artifacts/phase_07/kernels/`
- [ ] Per-dataset PSF comparison figure saved
- [ ] **Visual proof:** `artifacts/phase_07/real/` — PSF side-by-side + convergence curves

## Notes

- Phase 06 is a dependency because AgNor requires the OME-TIFF reader built in Phase 06.
- Saved kernel files in `artifacts/phase_07/kernels/` are the direct input to EXP 03.
- If multi-focal slides cause PSF inconsistency, it will appear as a multi-lobed predicted kernel. Flag as `research.md` candidate.
- Sub-Phase B only begins after Sub-Phase A gate passes. Document gate result in completion summary.
