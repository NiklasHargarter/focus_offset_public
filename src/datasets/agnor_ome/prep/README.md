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
Scans the valid slides dictated by the JSON splits, detects valid tissue regions using classical filtering, determines optimal Z-slices dynamically via Brenner Gradients, and outputs discrete rows containing purely integers mapping global paths to coordinates. Finally, these row mappings are dumped directly into highly efficient Pandas Parquet tables (`train.parquet` / `test.parquet`).
