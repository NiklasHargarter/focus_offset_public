# Dataset Preparation Pipeline

This directory contains the decoupled scripts responsible for transitioning raw datasets (such as `.vsi` slides from EXACT) into a highly optimized, flat Parquet cache ready for PyTorch dataloading.

The entire pipeline is orchestrator-driven. The individual scripts act as "dumb" unconditional workers that perform their specific sub-domain task, while the orchestrator (`sync.py`) manages the state logic and early-exits.

## `sync.py`
**The Orchestrator.**
Coordinates the full dataset preparation pipeline. It sequentially invokes the granular worker scripts based on whether the required target artifacts already exist. It completely manages state and prevents redundant work.

- Downloads missing files (delegates to `download.py`)
- Inflates and verifies source data structures (delegates to `fix_zip.py`)
- Creates train/test sets (delegates to `create_split.py`)
- Extracts index metadata (delegates to `preprocess.py`)

## `download.py`
**Pure Network Client.**
Responsible *only* for crawling EXACT to fetch the necessary missing ZIP files exactly as they exist remotely. It strictly avoids meddling with extraction or file verification—its sole job is to populate the `zips` directory securely. It also contains hardcoded business logic to exclude arbitrary metadata images (e.g. `_all_`).

## `fix_zip.py`
**Inflator & Verifier.**
Responsible for inflating the downloaded raw ZIP files into standard `.vsi` slides. It actively scans the VSI files using SlideIO to ensure structural integrity and correct nested folder mapping.
*If the verification passes natively, it automatically deletes the source ZIP to avoid storage bloat. If it fails, both the ZIP and broken VSI components are aggressively purged.*

## `create_split.py`
**Partitioning Engine.**
A fully deterministic script that reads the available valid `.vsi` slides and segregates them into `train_pool` and `test` clusters using a fixed ratio and seed, dumping the output directly to a central `splits.json` configuration file.

## `preprocess.py`
**The Metadata Compiler.**

Scans the valid slides dictated by the JSON splits, detects valid tissue regions using classical filtering, determines optimal Z-slices dynamically via Variance of Laplacian, and outputs discrete rows into highly efficient Pandas Parquet tables (`train.parquet` / `test.parquet`). All labels and quality metrics are computed at preprocessing time — training never reopens VSI files for anything other than pixel data.

### Configuration (`PrepConfig`)

| Parameter | Default | Description |
|---|---|---|
| `patch_size` | `224` | Model input size in pixels |
| `downsample_factor` | `2` | Read pyramid level (2 = 30× magnification; reads 448 px, downscales to 224 px) |
| `stride` | `patch_size × downsample_factor` | Step between patch origins — derived, always non-overlapping |
| `min_tissue_coverage` | `0.80` | Minimum tissue fraction to include a patch |
| `mask_downscale` | `8` | Downscale factor used when building the tissue mask thumbnail |

The output cache directory is named `ds{downsample_factor}_cov{min_tissue_coverage}/` so any parameter change produces a fresh, non-conflicting cache automatically.

### Per-Slide Pipeline

1. **Open slide** — load via `slideio` (scene 0); `read_block` handles on-the-fly downscaling from the full-resolution pyramid.
2. **Z-metadata** — read `z_resolution` once per slide; raise early if missing (required for physical label computation).
3. **Tissue mask** — read a small thumbnail at the middle Z-level (`width / mask_downscale`), apply Otsu thresholding via `histolab` to produce a binary tissue mask.
4. **Patch grid** — slide a `(patch_size × downsample_factor)²` window across the slide at `stride` intervals; retain only patches where the tissue mask covers ≥ `min_tissue_coverage`. The exact coverage float is recorded.
5. **Focus scoring** — for each candidate patch, read the full Z-stack (native `448×448`, downscaled to `224×224` by slideio) and compute a Variance of Laplacian focus score for every Z-slice.
6. **Label computation** — find the sharpest Z (`optimal_z` = argmax of focus scores), then for every Z-slice compute the regression target: `z_offset_microns = (optimal_z − z_level) × z_res_microns`.
7. **Write parquet** — emit one row per `(patch, z_level)` pair with all metadata, labels, and quality metrics pre-baked.

### Parquet Schema

Each row represents one `(slide, patch position, z-slice)` triplet:

| Column | Type | Description |
|---|---|---|
| `slide_name` | str | VSI filename |
| `x`, `y` | int | Patch top-left in full-resolution pixel coordinates |
| `z_level` | int | Z-slice index this row represents |
| `optimal_z` | int | Sharpest Z-slice index for this patch |
| `num_z` | int | Total Z-slices in the slide |
| `z_res_microns` | float | Physical Z step size in microns |
| `z_offset_microns` | float | **Regression target**: `(optimal_z − z_level) × z_res_microns` |
| `focus_score` | float | Variance of Laplacian sharpness score at this `z_level` |
| `max_focus_score` | float | Peak sharpness score across all Z for this patch |
| `focus_score_range` | float | `max − min` focus score — low = flat z-stack, uninformative for training |
| `tissue_coverage` | float | Actual tissue fraction in the patch mask region |

