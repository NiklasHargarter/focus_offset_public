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

### Phase 0: Ingestion & Sanitization Separation
Downloading and repairing the original WSI files requires team-specific backend libraries (e.g., the `exact_sync` download lib) and dataset-specific structural hacks. This introduces a heavy dependency that must not pollute the PyTorch ML ecosystem.
- The pipeline enforces **Strict Separation of Concerns**: the dataset download and file sanitization utilities must be built as an entirely isolated entrypoint.
- Downstream tasks (Focus Metric Calculation, Indexing, PyTorch DataLoaders) must strictly expect clean, unified local file paths to already-downloaded files, with zero knowledge of HTTP requests or server logic.
- **Multi-Tiered Ingestion (HE & IHC Only):** The ingestion script must natively support downloading three defined variants of the massive `ZStack HE` and `IHC` datasets to accommodate different compute and experimental bounds (AgNor and Jiang are treated as single, fixed targets):
  - **`full`:** Pulls every slide indiscriminately.
  - **`normal`:** Applies hard file-name heuristics *before* downloading (e.g., strictly filtering out anomalous `_all_` background macro-scans).
  - **`small`:** A manually restricted minimum subset explicitly hand-picked to ensure it gracefully covers Train/Val/Test splits, enabling fast continuous integration and architectural iterating.
- **Physical Subset Isolation (Strong Guarantee of Correctness):** Downstream code *must* inherently know which subset variant it is processing. Instead of relying on a shared pool of data with abstract logical filters, subsets must be exported into completely isolated physical directories (e.g., `data/he_full/` vs `data/he_small/`). The architecture explicitly prefers physical file duplication to guarantee absolute correctness and transparency. If an orchestrator requests the `he_full` preprocessing pipeline but only `data/he_small/` exists on disk, it must hard-crash.
- **Structural Repair:** The internal EXACT server often yields messy VSI downloads (e.g., corrupted ZIP structures). The ingestion utility must automatically unzip and legally repair the VSI folder structures natively on disk before the next phase begins.

### Shared Index & Split Datasets Pattern
Data loading must NOT duplicate underlying pre-work. The architecture requires a **Shared Data Foundation/Indexer** that handles:
- Patch splitting computations.
- Resolving optimal z-levels.
- Maintaining strict train/val/test splits at the **slide level** to prevent cross-contamination.

### Dataset Sharing & Archival Strategy
Once preprocessing and filtering routines are validated, the parsed dataset must be securely shareable and portable.
- To prevent massive storage inflation, extracted patches must **not** be saved to disk as individual image files.
- The final shareable dataset artifact must consist strictly of the **original raw WSI files (VSI/OME-TIFF)** paired with **plain-text, machine-readable index files (e.g., CSV or Parquet)**. 
- These index files must contain the **spatial patch origin coordinates (x, y)** and the **z-plane indices**. Because exhaustive index files will become very large (millions of rows), they should be managed via cloud storage or DVC, not raw Git.
- **Atomic Raw Metric Index (Flat CSV):** The plain-text dataset index must serve as a foundational, flat database rather than an array-packed object store. Each individual row is uniquely keyed by `(slide_name, x, y, z_level)` and contains strictly the computed scalar focus metrics for that specific patch (e.g., `brenner_score`, `laplacian_score`). This perfectly rigid relational structure allows downstream scripts to trivially load the CSV in Pandas and perform global groupby operations (e.g., `idxmax()`) to dynamically redefine what `optimal_z` means based on any experimental metric, creating entirely new focus-offset datasets instantaneously.

On top of this unified foundation, distinct PyTorch `Dataset` interfaces wrap the indexer to yield task-specific pairs with explicitly distinct sample definitions:
- **`OffsetPredictionDataset`**: For the primary prediction task, a sample is defined as a specific Z-level image from a specific patch. The total dataset size is `sum(slides × patches × available_z_levels)`. It yields `(any_patch, signed_focus_offset)`.
- **`SyntheticPairDataset`**: For the kernel learning experiment, a sample requires a matching offset pair. For any given offset (e.g. +10µm), a sample can exist *at most once* per patch, and may not exist at all if the individual stack's z-range does not extend that far. It filters the index strictly to yield `(infocus_patch, target_offset_patch)`.

---

## 4. Experimental Requirements & Ablations

### A. Preprocessing & Tissue Filtering Validation
Tissue filtering, masking, and optimal downscaling factors are complex operations currently governed by untracked config values lost in Git history.
- These preprocessing steps must be elevated to formalized, trackable experiments rather than hand-tuned constants.
- **Dynamic Slide-Wide Thresholding:** A hardcoded global RGB threshold (e.g., `< 240`) fails completely across different stains (like IHC, which has massive swathes of bright white background). The tissue filtering logic must enforce **adaptive, slide-wide thresholds** (e.g., Otsu's method) calculated precisely for each individual slide to guarantee robust separation of tissue from background regardless of the dataset's baseline brightness.
- **Decoupled Tissue Filtering & Pyramid Scales:** A native VSI pyramid dramatically alters the contextual field-of-view for a fixed 224x224 patch (e.g., native vs 4x downscale). To evaluate changing image context systematically, the Focus Index precomputation must be strictly decoupled from tissue filtering. The indexer must compute exhaustive focus metrics for the entire generic spatial grid (including useless empty background) uniformly across these key native magnification levels (creating, for instance, `he_index_native.csv`, `he_index_2x.csv`, `he_index_4x.csv`).
- Tissue masking is then implemented as a rapid post-processing step dynamically filtering these exhaustive text databases, empowering researchers to instantly ablate and sweep tissue density thresholds without ever having to re-run the extraordinarily expensive image-read computations.
- The recorded outcomes of these studies will explicitly prove the robustness of the chosen preprocessing configuration and highlight the complexity of the data preparation phase.

### B. Focus Metric & Multi-Focal Resolution Validation
Assigning an `optimal_z` typically involves calculating a focus metric (e.g., Brenner Gradient) and taking the argmax. However, thick biological tissue often contains multiple focal planes, resulting in multi-peak focus curves. A naive argmax in multi-focal patches creates mathematically contradictory training targets, as some in-focus features are erroneously labeled as "defocused."
- The project must systematically evaluate focus metric formulas to determine which yields the cleanest signal for WSI patches.
- The indexer must implement mathematical peak-detection rather than simple argmaxing.
- The codebase must support an ablation study comparing datasets where multi-focal patches are safely discarded versus naively preserved, officially solving the multi-focal noise problem.

### C. Feature Domain Ablation (Spatial vs. Frequency)
To evaluate pathways for model agnosticism, the pipeline must support dynamic toggling and combinations of input transforms.
- Readily inject spatial (RGB, Grayscale) or frequency (FFT, DWT) transforms into the model pipeline.
- Ensure the codebase can rapidly deploy ablation studies comparing these domains without requiring hardcoded structural changes in core model components.

### D. Stain Invariance & Augmentations
To prevent overfitting on stain-specific information, the augmentation pipeline must be treated as a primary experimental variable.
- Augmentations must *not* be deeply hardcoded into data loaders.
- The system must provide a clean, configurable mechanism to swap, define, and execute ablation studies against specific augmentations (e.g., color jitter, stain normalization).

### E. Synthetic Kernel Learning
- The project will learn specific target blur kernels corresponding to physical focus offsets per dataset (or grouped constraints).
- **Phase 1 (Simulation Validations):** Learn and visually inspect the kernels against mathematically generated targets to prove their correctness. To protect empirical integrity, the codebase must **strictly separate simulation logic from real-data logic** via completely distinct `Dataset` classes and orchestrator scripts, heavily forbidding "unsafe" boolean config flags that could cross-contaminate result logs.
- **Phase 2 (True Point-Spread Function):** Utilize the proven mathematical models to safely predict the true physical defocus blur from genuine, real-world `(infocus, defocus)` pairs.

---

## 5. Engineering & Reproducibility Standards

- **Structural Freedom & Clean Slate Architecture:** The legacy directory structure (e.g., highly specific `scripts/` or `synthetic_data/` splintered folders) is strictly disposable. The existing codebase serves solely as a technical reference for specific domain math (like `slideio` loading quirks or FFT algorithms). Nothing else is sacred. AI agents are empowered and expected to implement a complete, clean-slate design using optimal, modern Python and PyTorch design patterns (e.g., a unified `src/` layout) rather than adhering to legacy constraints. You have complete autonomy over the exact file names, module boundaries, and class structures—the `plans/` simply define the logical execution steps, not the physical file tree.
- **Optimal Library Philosophy (No Hero Coding):** Never hand-write nested loops or custom algorithms if an established, highly optimized library exists for that exact use case. Lean heavily on Pandas, PyTorch, and specific domain libraries.
  - *Data Visualization:* The project officially adopts **Seaborn** as the primary graphing library. Do not raw-dog complex plots in basic `matplotlib` if Seaborn provides a cleaner, more statistically rigorous API for the visual proofs.
- **Core vs Experiment Separation (Side-Effect Free Libraries):** To satisfy the Visual Proof Mandate without polluting production code or tanking execution speed across thousands of files, the architecture enforces a strict separation of concerns:
  - **Core Modules (`src/core/`)** must be pure mathematical black boxes. They take inputs (e.g., an image array) and return structured outputs (e.g., a binary tissue mask). They must strictly contain **zero** side-effects like initializing `matplotlib` figures or saving `.png` files to disk.
  - **Experiment Scripts (`src/experiments/`)** exist solely to generate visual and empirical proofs. They import the fast core modules, run them on a sample subset of data, and construct the slow, IO-heavy visual overlays as tangible evidence.
  - **Production Pipelines (`src/pipeline/` or orchestrator entrypoints)** import those same core modules but wrap them in rapid `multiprocessing` or batch loops, completely skipping the visualization overhead.
- **Strict Sub-Sampling & The Small Subset Mandate:** The raw HE and IHC datasets consist of hundreds of gigabytes of dense biological imaging. A naive full `for`-loop across the `full` or `normal` datasets will consume immense compute capital and hang the execution flow for hours. Therefore, during all architectural development, local execution, and script verification, developers and AI Agents are strictly mandated to exclusively work with the `small` dataset variant. Only ever process the heavier tiers if absolutely required to prove a specific, non-scalable hypothesis. Additionally, even when utilizing the `small` variant, agents must explicitly hardcode temporary sub-sampling bounds (e.g., `limit_slides=2`, `limit_patches=10`) directly into their loops or utilize "dry-run" configuration arguments to guarantee that active script development iterations finish aggressively fast (in seconds, not minutes).
- **Command-Line First, Interactive Second (No `.ipynb`):** The project strictly forbids the use of raw Jupyter Notebook `.ipynb` JSON files. The optimal choice for all exploratory, experimental, and visual proof generation code is a **solid, executable Python script** designed to be run from the command line (e.g., `python exp_tissue.py`). The percent format (`# %%` cell markers) should only be used as an *optional* overlay for interactive editing in an IDE—the script itself must remain structurally sound and capable of executing end-to-end via CLI.
- **Proof-by-Visual-Experiment Mandate:** If an internal threshold or configurable value is decided by the research team (e.g., tissue density threshold, peak detection sensitivity), it **MUST** be backed by a formal experiment generating visual or mathematical proof of its correctness. Conversely, externally mandated constants standard to the field (e.g., a $224 \times 224$ px patch size derived from common Computer Vision models) are strictly fixed and do not require empirical proof. *Example:* Finding the optimal tissue filter threshold for HE versus IHC cannot be a silent log—an **Experiment Script** must output a slide-level image rendering the patch grid overlay, visually proving exactly which background patches were discarded and which tissue patches were kept.
- **Models:** Built intentionally simple. Architecture complexity should only be introduced when attempting to prove a specific hypothesis.
- **Configuration Philosophy:** The current configuration structure must be completely rethought to avoid overly bloated config objects:
  - **Fixed Constants:** Values that do not change (e.g. fixed domain constraints, standard kernel block sizes) must be hard-coded near where they are used to keep modules self-contained.
  - **Dynamic Parameters:** Configurations should strictly be limited to structural environment variables (e.g., dataset paths, hardware toggles) and explicitly active experimental hyperparameters (e.g., learning rates, ablation toggles).
- **Living Context File (`research.md`):** Agents must be continuously pointed to `research.md` for fast, hyper-relevant, and short-lived technical context (e.g., active bugs, environment tooling like `uv`, or helper functions like `suppress_stderr` for VSI reads). This file must be kept up-to-date and aggressively pruned as work phases conclude.
- **Outputs & Logging:** All experiment artifacts must be meticulously structured. Metrics, hyperparameter configs, and model weights must be continuously saved enabling seamless tracking, replication, and publication of results.
- **Tests First:** All distinct operational boundaries must have functional tests to uphold the "AI owns the implementation, tests keep it honest" contract.
