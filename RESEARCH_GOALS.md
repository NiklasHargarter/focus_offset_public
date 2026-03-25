# Research Goals

High-level scientific objectives and the experiments planned under each one.

---

## Goal 0: WSI Focus Offset Dataset Design & Methodology

**Question:** What is the right methodology for constructing a reproducible, efficient, and scientifically valid focus offset dataset from whole slide imaging z-stacks? This is a standalone contribution — the methodology and dataset could serve as a public benchmark for others.

### Contributions

| Contribution | What it addresses | Phase |
|--------------|-------------------|-------|
| Patch tiling strategy | Optimal patch size, stride, and grid layout for WSI focus offset tasks | 02 |
| Downscale & magnification ablation | Does native vs 2×/4× downscale change the task difficulty and model performance? | 02 |
| Tissue filtering methodology | Adaptive per-slide Otsu thresholding vs global threshold; which patches are valid training samples? | 02 |
| Focus metric evaluation | Brenner gradient vs Laplacian variance vs others — which produces the cleanest z-profile? | 02 |
| Multi-focal patch detection | Quantifying how often z-stacks have multiple focal planes and what this means for label quality | EXP 01 |
| Dataset structure & reproducibility | Plain-text CSV index + raw WSI files: no patch extraction, version-controllable, portable, archivable | 02 |
| Loading performance | Benchmarking on-the-fly VSI patch loading vs pre-extracted patches at training scale | 02/03 |
| Train/val/test split design | Slide-level splits preventing cross-contamination; split strategy for multi-dataset training | 02 |
| Physical offset labelling | Deriving signed `focus_offset_um` in physical units (µm) rather than z-index integers, enabling cross-scanner comparison | 02 |

---

## Goal 1: Single-Shot Focus Offset Prediction

**Question:** Can a model predict the signed z-distance to optimal focus from a single out-of-focus patch?

### Experiments

| Experiment | What it tests | Phase |
|------------|---------------|-------|
| Baseline spatial model (HE) | Does the task work at all? | 03 |
| Feature domain: spatial vs FFT vs DWT | Which input representation gives the best signal? | 04 |
| Multi-magnification: native vs 2x vs 4x | Does field-of-view change the prediction quality? | EXP 02 |
| Tissue filtering threshold sweep | Which patches are clean enough to train on? | 02 |
| Multi-focal patch impact | Does argmax labelling introduce systematic noise? | EXP 01 Part A |
| Multi-focal discard ablation | Does removing multi-focal patches from training actually improve model accuracy? | EXP 01 Part B |
| External benchmark: Jiang2018 | Does the model trained on HE generalise to a fully independent public dataset? | 06 |

---

## Goal 2: Stain & Scanner Generalization

**Question:** Does a model trained on HE generalize to IHC and other scanners (AgNor, Jiang2018)? What makes it fail or succeed?

### Experiments

| Experiment | What it tests | Phase |
|------------|---------------|-------|
| Stain augmentation ablation: none / color-jitter / stain-norm | Does augmentation close the HE→IHC gap? | 05 |
| Cross-dataset evaluation: train HE, test IHC / AgNor / Jiang2018 | Baseline generalization gap without any intervention | 06 |
| Feature domain × generalization | Does frequency domain (FFT/DWT) generalize better than RGB? | 04 + 06 |

---

## Goal 3: Synthetic Z-Stack Generation

**Question:** Can we learn the physical point-spread function of defocus from data and use it to generate synthetic out-of-focus images?

### Experiments

| Experiment | What it tests | Phase |
|------------|---------------|-------|
| Simulation validation: recover known kernels | Does the architecture work at all before touching real data? | 07A |
| Real PSF learning from paired HE patches | Can we learn the true physical defocus blur from real z-stacks? | 07B |
| Per-dataset PSF: HE vs AgNor vs IHC | Do different scanners/stains require distinct kernels, or does one generalise? | 07B + 06 |
| Synthetic augmentation for generalization | Do synthetic defocus patches generated from Goal 3 kernels close the HE→IHC/AgNor gap in Goal 2? | EXP 03 |

---

## Open Questions (not yet assigned to a phase)

- Does multi-focal peak detection (vs argmax) meaningfully improve model performance on Goal 1?
- Does the learned PSF from Goal 3 generalize across scanners, or is it scanner-specific?
- Is Jiang2018 a fair benchmark given it is pre-tiled and may have different defocus characteristics?
