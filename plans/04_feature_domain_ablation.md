# Phase 04: Feature Domain Ablation (Spatial vs Frequency)

**Status:** Not Started
**Dependencies:** Phase 03
**Goal:** Add dynamic input transform switching so the same training pipeline can run with spatial (RGB/grayscale), FFT, or DWT input representations. Enables the core feature domain ablation experiment.

## Requirements Addressed
- Feature Domain Ablation: Dynamic toggling of spatial vs frequency transforms
- Model registry expansion for domain-appropriate architectures

## Interface Decisions

> To be populated by `/design-an-interface` during Phase 00 and refined after Phase 03.

Expected decisions:
- `TransformStrategy` base class — `__call__(patch: Tensor) -> Tensor`, `output_channels: int` property
- `SpatialTransform(grayscale: bool)` — RGB or grayscale
- `FFTTransform(magnitude_only: bool)` — magnitude spectrum or full complex representation
- `DWTTransform(wavelet: str, level: int)` — wavelet family and decomposition level
- `get_transform(domain: str) -> TransformStrategy` — registry lookup
- CLI: `--domain spatial | fft | dwt | spatial-gray`

## Vertical Slice Execution Steps

1. **FFTTransform** — 2D FFT magnitude spectrum, normalized. Write pytest: output shape, output range, determinism.
2. **DWTTransform** — PyWavelets/ptwt discrete wavelet transform. Write pytest: output shape, invertibility check.
3. **SpatialTransform** — formalise the Phase 03 stub into a full implementation with grayscale option.
4. **Transform registry** — `get_transform(domain)` factory function. Write pytest for all 4 domain strings.
5. **Model registry update** — ensure `get_model` is domain-aware: FFT/DWT inputs have different channel counts. Model head is always a scalar regression output.
6. **CLI update** — add `--domain` flag to `train_focus.py`. Validated at startup.
7. **Visual proof experiment** — run the same 3-slide `small` training run under all 4 domains. Save a 2x2 plot of loss curves, one per domain.

## Completion Criteria

- [ ] Pytest passes for all 3 transform implementations
- [ ] Pytest passes for transform registry (all domain strings)
- [ ] `train_focus.py --domain spatial`, `--domain fft`, `--domain dwt` all run without crashing
- [ ] Model handles different input channel counts correctly for each domain
- [ ] **Visual proof:** `artifacts/phase_04/` contains 4-panel loss curve comparison across domains

## Notes

- Do not hardcode any transform logic inside `OffsetPredictionDataset` — it receives a `TransformStrategy` and calls it.
- DWT output shape depends on decomposition level — `output_channels` property must be accurate so the model head can be constructed correctly.
- The ablation result (which domain performs best) is a scientific finding — flag it as a `research.md` candidate, not a code decision.
