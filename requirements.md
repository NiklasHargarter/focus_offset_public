# Focus Offset Prediction — Project Blueprint & Requirements

## 1. Project Context & Objectives
This is a research project investigating focus offset prediction for z-stacks from whole slide imaging (WSI) in digital pathology. The overarching goal is to maximize knowledge gained, investigate domain generalization, and validate theories, rather than attempting to build a production-ready software application.

The project revolves around two main pillars:
1. **Focus Offset Prediction:** Predicting the signed distance the microscope z-axis needs to be adjusted to reach optimal focus, using purely a single out-of-focus image patch. A major focus is discovering methods to achieve stain and sample generalization (e.g., train on HE and test on IHC).
2. **Synthetic Data Generation & Kernel Learning:** Using a known operator approach to learn dataset-specific offset blur kernels. This allows inspection of the physical form of an offset blur, and ultimately, utilization of these kernels to generate synthetic out-of-focus data that improves model stability and domain generalization.

---

## 2. Codebase Architecture: AI-Optimized "Grey Box" Modules
A strictly enforced project requirement is that the codebase must be optimized for navigation and modification by AI agents. This is achieved through the architectural pattern of **Deep Modules** with **Slim Interfaces**.

- **Grey Box Pattern:** The developer owns the top-level interfaces and architectural seams; the AI owns the internal implementations. 
- **Progressive Disclosure:** Each module must live in its own folder exposing a clean, simple public interface. The AI must be able to comprehend the system's capabilities by reading the interfaces without digging into the implementation loops.
- **Test-Driven Lockdown:** Every module must have matching tests (mocked or real data) that lock down its expected behavior. If tests pass, the module's internal logic is trusted without manual inspection.
- **Reduced Cognitive Load:** High-level execution scripts (e.g., for training focus offset or learning synthetic kernels) should remain as slim orchestrators that instantiate and connect the deep modules via declarative configurations.

---

## 3. Data Foundation & Serving Architecture
The project utilizes four distinct dataset sources with differing metadata, file types, and purposes:
1. **ZStack HE** (Olympus VSI, own team) - Primary in-domain training data.
2. **ZStack IHC** (Olympus VSI, own team) - Stain-generalization control (same slides/scanner, different stain).
3. **AgNor** (OME-TIFF, own team) - Scanner-generalization probe.
4. **Jiang2018** (JPEG patches, public) - External benchmark.

### Shared Index & Split Datasets Pattern
Data loading must NOT duplicate underlying pre-work. The architecture requires a **Shared Data Foundation/Indexer** that handles:
- Patch splitting computations.
- Resolving optimal z-levels.
- Maintaining strict train/val/test splits at the **slide level** to prevent cross-contamination.

### Dataset Sharing & Archival Strategy
Once preprocessing and filtering routines are validated, the parsed dataset must be securely shareable and portable.
- To prevent massive storage inflation, extracted patches must **not** be saved to disk as individual image files.
- The final shareable dataset artifact must consist strictly of the **original raw WSI files (VSI/OME-TIFF)** paired with **plain-text, machine-readable index files (e.g., CSV or JSON Lines)**. These index files must be easily inspectable and suitable for checking into version control (Git). They must simply contain the **spatial patch origin coordinates (x, y)**, the **z-plane indices**, and the computed focus offsets, allowing the data loader to dynamically extract the patches from the raw WSI at runtime (e.g., via `slideio`) without creating any intermediate image files on disk.

On top of this unified foundation, distinct PyTorch `Dataset` interfaces wrap the indexer to yield task-specific pairs with explicitly distinct sample definitions:
- **`OffsetPredictionDataset`**: For the primary prediction task, a sample is defined as a specific Z-level image from a specific patch. The total dataset size is `sum(slides × patches × available_z_levels)`. It yields `(any_patch, signed_focus_offset)`.
- **`SyntheticPairDataset`**: For the kernel learning experiment, a sample requires a matching offset pair. For any given offset (e.g. +10µm), a sample can exist *at most once* per patch, and may not exist at all if the individual stack's z-range does not extend that far. It filters the index strictly to yield `(infocus_patch, target_offset_patch)`.

---

## 4. Experimental Requirements & Ablations

### A. Preprocessing & Tissue Filtering Validation
Tissue filtering, masking, and optimal downscaling factors are complex operations currently governed by untracked config values lost in Git history.
- These preprocessing steps must be elevated to formalized, trackable experiments rather than hand-tuned constants.
- The pipeline must systematically ablate and validate filtering thresholds (e.g., background thresholding, tissue boundaries) and downsample scales.
- The recorded outcomes of these studies will explicitly prove the robustness of the chosen preprocessing configuration and highlight the complexity of the data preparation phase.

### B. Feature Domain Ablation (Spatial vs. Frequency)
To evaluate pathways for model agnosticism, the pipeline must support dynamic toggling and combinations of input transforms.
- Readily inject spatial (RGB, Grayscale) or frequency (FFT, DWT) transforms into the model pipeline.
- Ensure the codebase can rapidly deploy ablation studies comparing these domains without requiring hardcoded structural changes in core model components.

### C. Stain Invariance & Augmentations
To prevent overfitting on stain-specific information, the augmentation pipeline must be treated as a primary experimental variable.
- Augmentations must *not* be deeply hardcoded into data loaders.
- The system must provide a clean, configurable mechanism to swap, define, and execute ablation studies against specific augmentations (e.g., color jitter, stain normalization).

### D. Synthetic Kernel Learning
- The project will learn specific target blur kernels corresponding to physical focus offsets per dataset (or grouped constraints).
- **Phase 1 (Simulation Validations):** Learn and visually inspect the kernels against mathematically generated targets to prove their correctness. To protect empirical integrity, the codebase must **strictly separate simulation logic from real-data logic** via completely distinct `Dataset` classes and orchestrator scripts, heavily forbidding "unsafe" boolean config flags that could cross-contaminate result logs.
- **Phase 2 (True Point-Spread Function):** Utilize the proven mathematical models to safely predict the true physical defocus blur from genuine, real-world `(infocus, defocus)` pairs.

---

## 5. Engineering & Reproducibility Standards

- **Models:** Built intentionally simple. Architecture complexity should only be introduced when attempting to prove a specific hypothesis.
- **Configuration Philosophy:** The current configuration structure must be completely rethought to avoid overly bloated config objects:
  - **Fixed Constants:** Values that do not change (e.g. fixed domain constraints, standard kernel block sizes) must be hard-coded near where they are used to keep modules self-contained.
  - **Dynamic Parameters:** Configurations should strictly be limited to structural environment variables (e.g., dataset paths, hardware toggles) and explicitly active experimental hyperparameters (e.g., learning rates, ablation toggles).
- **Living Context File (`research.md`):** Agents must be continuously pointed to `research.md` for fast, hyper-relevant, and short-lived technical context (e.g., active bugs, environment tooling like `uv`, or helper functions like `suppress_stderr` for VSI reads). This file must be kept up-to-date and aggressively pruned as work phases conclude.
- **Outputs & Logging:** All experiment artifacts must be meticulously structured. Metrics, hyperparameter configs, and model weights must be continuously saved enabling seamless tracking, replication, and publication of results.
- **Tests First:** All distinct operational boundaries must have functional tests to uphold the "AI owns the implementation, tests keep it honest" contract.
