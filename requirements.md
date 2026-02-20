# Focus Offset Prediction — Master Plan

## Project Context

This is a master's research project investigating whether deep learning models can predict the **focus offset** of a Whole Slide Image (WSI) patch — the signed distance in micrometres between the current Z-plane and the optimal focal plane — without any motorised hardware sweep.

**Constraints:**
- Limited GPU budget (RTX 5090 available, but compute time must be treated as scarce).
- Research-grade code: results must be credible and reproducible, not optimised for production.
- All data is non-human or fully public; no patient-data obligations apply.

---

## Objective

Given a **224×224 RGB image patch** from a WSI, predict the signed focus offset in **micrometres (µm)** in a single forward pass — without any additional inputs such as Z-level index or scanner metadata. The model must infer defocus purely from visual appearance.

The Z-level index is used only during training to compute the supervision target (`(best_z − current_z) × z_resolution_µm`). At inference time the model receives only the image.

---

## Datasets

The project uses four datasets with distinct roles. Each must be fully self-contained — its own download, preprocessing, and `Dataset` implementation — so that changes to one never affect another.

| Dataset | Format | Domain | Role |
|---|---|---|---|
| **ZStack HE** | Olympus VSI | In-domain | Primary training set |
| **ZStack IHC** | Olympus VSI | In-domain | Stain-generalisation control (same scanner, same slides, different stain) |
| **AgNor OME** | OME-TIFF | Out-of-domain | Scanner-generalisation probe (different equipment and tissue) |
| **Jiang2018** | JPEG tiles (public) | Out-of-domain | External benchmark; used by other published works for direct comparison |


**Label unit consistency:** every dataset must convert its native label encoding (e.g. nm in filenames, Z-index × scanner resolution) to µm inside its `Dataset` class before anything else sees the target value.

**Sign convention:** all datasets must use the same physical sign convention. The reference is Jiang2018, whose labels are signed physical offsets from the in-focus plane (range −10 µm to +10 µm, step 0.5 µm). For VSI datasets the label is computed as `(best_z − current_z) × z_resolution_µm`, which is positive when the current patch is below the focal plane and negative when above it.

> **⚠ Verification required:** the physical direction of increasing Z index in the Olympus VSI scanner must be confirmed to align with Jiang's positive/negative convention. If the scanner's Z axis runs in the opposite direction, all VSI targets must be negated (`(current_z − best_z) × z_resolution_µm`) to remain compatible. This must be verified empirically by visual inspection — load a slide at a known below-focus and above-focus Z level and confirm the sign of the resulting target matches the Jiang convention before training.


---

## Pipeline

### 1. Data Preparation

- Download and extraction must be automatable and resumable; re-running must be idempotent.
- Preprocessing results must be cached to disk so they survive session restarts and are never recomputed unnecessarily.
- Corrupt or missing source files must be logged and skipped, not treated as fatal errors.
- Train and test data must be split at the **slide level** to prevent patches from the same slide appearing in both sets.

### 2. Training

- Models must be **lightweight** — small enough to converge within a single GPU session, enabling multiple experiment iterations.
- A **dry-run / smoke-test mode** must exist to validate the full pipeline cheaply before committing real GPU time.
- Training must be **reproducible**: fixed seeds, versioned split files, and all hyperparameters logged alongside results.
- The pipeline must be **model-agnostic**: swapping or adding a model variant requires minimal changes.

### 3. Evaluation

- Metrics must be reported in physical units (µm): **MAE** and **Median Absolute Error** at minimum.
- For Jiang2018, the published evaluation protocol must be followed exactly to allow direct comparison with other works.
- Every result file must be **self-describing**: model name, dataset, checkpoint, and config must be recorded so results can be reconstructed without relying on memory or commit history.

---

## Principles

### Isolation Over Premature Abstraction
Duplication between datasets and model variants is explicitly preferred over shared abstractions. Each component owns its full lifecycle. A change to one must never require touching another. Shared code is only acceptable for truly generic, stable helpers.

### Prefer External Libraries
Custom implementations introduce bugs and maintenance burden. Established libraries must be used wherever they exist (e.g. `slideio`/`tifffile` for slide reading, `timm` for model backbones, `albumentations` for augmentation, Accelerate for training infrastructure). Custom code should be limited to what is genuinely novel to this project.

### Open, Non-Proprietary Formats
All outputs must be readable without this codebase:
- Predictions and metrics → **CSV**
- Dataset splits → **JSON**
- Model weights → **PyTorch state dict (`.pt`)**

### Hypothesis-Driven Experiments
Every model variant or ablation must test a specific, stated hypothesis. The hypothesis must be documented alongside the experiment. Results without a documented hypothesis are hard to interpret and build on.

---

## Engineering Standards

- **Type safety**: `mypy` must pass; all public functions and classes must be annotated.
- **Linting**: `ruff` enforced.
- **Tests**: `pytest` smoke tests covering data loading, model forward pass, and loss computation — runnable without real data or GPU.
- **Documentation**: CLI commands for all major steps must be documented. Design decisions should be captured so context is not lost over time.
- **Dependency management**: a single `uv sync` must be sufficient to reproduce the environment.
