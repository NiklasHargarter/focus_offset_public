# Phase 3: Feature Domain Ablation (Spatial vs. Frequency) Injection

**Status:** Not Started  
**Dependencies:** Phase 1 (Minimum Viable Data Foundation)  
**Goal:** Introduce the dynamic input transform mechanism to easily ablate between spatial and frequency features without hardcoding.

## Requirements Addressed
- Feature Domain Ablation (Spatial vs. Frequency)
- Configuration Philosophy (Dynamic Parameters)

## Vertical Slice Execution Steps
1. **Transform Strategy Interface:** Create a deep module defining an Input Transform pipeline. Salvage the FFT and DWT mathematical operations from legacy `architectures.py`.
2. **Model Registry Extraction:** Build a clean model factory/registry capable of providing models matching the chosen input dimensionality (e.g., 3-channel RGB vs 1-channel Grayscale FFT).
3. **CLI Orchestration Update:** Modify `train_focus.py` to accept arguments like `--domain spatial` or `--domain fft`. 
4. **Verify:** Run `train_focus.py --domain spatial` followed by `train_focus.py --domain fft`. Both must dynamically adjust transformations, initialize the correct model sizes, and complete a training epoch.

## Completion Criteria
- [ ] CLI correctly parses and acts on `--domain` flags.
- [ ] Data loaders apply spatial vs frequency transforms dynamically based on configuration.
- [ ] Models adapt to input channel sizes without crashing.
