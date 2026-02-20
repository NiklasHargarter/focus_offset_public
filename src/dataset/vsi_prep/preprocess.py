import argparse
import json
import multiprocessing
import os
import pickle
from dataclasses import asdict
from functools import partial
from pathlib import Path

import cv2
import numpy as np
import slideio

from src.config import DatasetConfig
from src.dataset.prep.grid import filter_by_tissue_coverage, generate_grid
from src.dataset.prep.tissue_detection import detect_tissue_mask
from src.dataset.index_types import MasterIndex, PreprocessConfig, SlideMetadata
from src.utils.focus_metrics import compute_brenner_gradient
from src.utils.io_utils import suppress_stderr


# ---------------------------------------------------------------------------
# Single-slide processing
# ---------------------------------------------------------------------------


def read_thumbnail_rgb(
    scene, width: int, height: int, mask_downscale: int
) -> np.ndarray:
    """Read a small RGB thumbnail from the first z-slice."""
    d_w = width // mask_downscale
    d_h = height // mask_downscale
    img = scene.read_block(rect=(0, 0, width, height), size=(d_w, d_h), slices=(0, 1))
    if img.ndim == 4:
        img = img[0]
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def find_best_z_per_patch(scene, candidates, raw_patch_size, patch_size, num_z):
    """For each candidate patch, read the full z-stack and pick the sharpest slice."""
    best_zs = np.zeros(len(candidates), dtype=np.int32)

    for i, (x, y) in enumerate(candidates):
        if i % 100 == 0:
            print(f"    patch {i}/{len(candidates)}")

        with suppress_stderr():
            z_stack = scene.read_block(
                rect=(x, y, raw_patch_size, raw_patch_size),
                size=(patch_size, patch_size),
                slices=(0, num_z),
            )
        if z_stack.ndim == 3:
            z_stack = z_stack[np.newaxis]

        scores = [compute_brenner_gradient(z_stack[z]) for z in range(num_z)]
        best_zs[i] = int(np.argmax(scores))
        del z_stack

    return best_zs


def build_patch_index(candidates, best_zs) -> np.ndarray:
    """Combine (x, y) candidates with best-z into an (N, 3) int32 array."""
    coords = np.array(candidates, dtype=np.int32)
    return np.column_stack([coords, best_zs])


def process_slide(
    slide_path: Path, cfg: PreprocessConfig, mask_downscale: int
) -> SlideMetadata:
    """Full single-slide pipeline: thumbnail → mask → grid → focus → index."""
    raw_extent = cfg.patch_size * cfg.downsample_factor

    with suppress_stderr():
        slide = slideio.open_slide(str(slide_path), "VSI")
    scene = slide.get_scene(0)
    width, height = scene.size
    num_z = scene.num_z_slices

    print(f"[{slide_path.name}] tissue mask")
    thumbnail = read_thumbnail_rgb(scene, width, height, mask_downscale)
    mask = detect_tissue_mask(thumbnail_rgb=thumbnail)

    print(f"[{slide_path.name}] grid + tissue filter")
    grid = generate_grid(width, height, raw_extent, cfg.stride)
    candidates = filter_by_tissue_coverage(
        grid,
        mask,
        raw_extent,
        cfg.min_tissue_coverage,
        mask_downscale=mask_downscale,
    )

    if not candidates:
        return SlideMetadata(
            name=slide_path.name,
            num_z=num_z,
            patches=np.empty((0, 3), dtype=np.int32),
        )

    print(f"[{slide_path.name}] focus search ({len(candidates)} patches)")
    best_zs = find_best_z_per_patch(
        scene, candidates, raw_extent, cfg.patch_size, num_z
    )

    return SlideMetadata(
        name=slide_path.name,
        num_z=num_z,
        patches=build_patch_index(candidates, best_zs),
    )


def load_master_index(
    dataset_name: str,
    stride: int,
    downsample_factor: int,
    min_tissue_coverage: float,
) -> MasterIndex | None:
    dataset_cfg = DatasetConfig(name=dataset_name)
    dataset_cfg.prep.stride = stride
    dataset_cfg.prep.downsample_factor = downsample_factor
    dataset_cfg.prep.min_tissue_coverage = min_tissue_coverage

    manifest_path = dataset_cfg.get_master_index_path()
    indices_dir = dataset_cfg.get_slide_index_dir()

    if not manifest_path.exists():
        print(f"DEBUG: Manifest path {manifest_path} does not exist.")
        return None

    try:
        print(f"DEBUG: Loading manifest from {manifest_path}...")
        with open(manifest_path, "rb") as f:
            manifest_data = pickle.load(f)
        print("DEBUG: Manifest loaded.")

        if isinstance(manifest_data, MasterIndex):
            return manifest_data

        print(f"DEBUG: Loading individual slide indices from {indices_dir}...")
        slide_metadatas = []
        pkl_files = sorted(indices_dir.glob("*.pkl"))
        print(f"DEBUG: Found {len(pkl_files)} pkl files.")
        for i, pkl_path in enumerate(pkl_files):
            if i % 100 == 0:
                print(f"DEBUG: Loading {i}/{len(pkl_files)}")
            try:
                with open(pkl_path, "rb") as f:
                    slide_metadatas.append(pickle.load(f))
            except Exception:
                continue
        print("DEBUG: All slide indices loaded.")
        return MasterIndex(
            file_registry=slide_metadatas,
            config_state=manifest_data["config_state"],
        )
    except Exception as e:
        print(f"Error loading indices: {e}")
        return None


def save_slide_json(result: SlideMetadata, pkl_path: Path):
    json_path = pkl_path.with_suffix(".json")
    data = {
        "name": result.name,
        "num_z": result.num_z,
        "patches": result.patches.tolist(),
    }
    with open(json_path, "w") as f:
        json.dump(data, f)


def save_master_index_json(
    master_index: MasterIndex, dataset_name: str, dataset_cfg: DatasetConfig
) -> None:
    cfg = master_index.config_state
    json_path = dataset_cfg.get_master_index_path().with_suffix(".json")
    data = {
        "dataset_name": dataset_name,
        "config": asdict(cfg),
        "slides": [],
    }

    for slide in master_index.file_registry:
        data["slides"].append(  # type: ignore
            {
                "name": slide.name,
                "num_z": slide.num_z,
                "patches": slide.patches.tolist(),
            }
        )

    with open(json_path, "w") as f:
        json.dump(data, f)
    print(f"  [EXPORT] Consolidated JSON index saved to {json_path}")


def preprocess_dataset(
    dataset_name: str,
    stride: int,
    downsample_factor: int,
    min_tissue_coverage: float,
    limit: int | None = None,
    workers: int | None = None,
    force: bool = False,
    exclude: str = "_all_",
) -> None:
    dataset_cfg = DatasetConfig(name=dataset_name)
    dataset_cfg.prep.stride = stride
    dataset_cfg.prep.downsample_factor = downsample_factor
    dataset_cfg.prep.min_tissue_coverage = min_tissue_coverage

    raw_dir = dataset_cfg.raw_dir
    manifest_path = dataset_cfg.get_master_index_path()
    indices_dir = dataset_cfg.get_slide_index_dir()

    workers = workers or os.cpu_count() or 1

    current_config = PreprocessConfig(
        patch_size=dataset_cfg.prep.patch_size,
        stride=stride,
        downsample_factor=downsample_factor,
        min_tissue_coverage=min_tissue_coverage,
        dataset_name=dataset_name,
    )

    existing_results = []
    processed_names = set()

    if manifest_path.exists() and not force:
        with open(manifest_path, "rb") as f:
            manifest_data = pickle.load(f)

        if manifest_data["config_state"] != current_config:
            print("Config mismatch! Use --force to reprocess.")
            return

        for pkl_path in sorted(indices_dir.glob("*.pkl")):
            try:
                with open(pkl_path, "rb") as f:
                    res = pickle.load(f)
                existing_results.append(res)
                processed_names.add(res.name)
            except Exception:
                continue

    all_files = sorted(list(raw_dir.glob("*.vsi")))
    if exclude:
        all_files = [f for f in all_files if exclude.lower() not in f.name.lower()]
    if limit:
        all_files = all_files[:limit]

    files_to_process = [f for f in all_files if f.name not in processed_names]

    print(
        f"Preprocessing {dataset_name} (Stride={stride}, Patch={dataset_cfg.prep.patch_size})"
    )
    print(f"Total: {len(all_files)} | Remaining: {len(files_to_process)}")

    if not manifest_path.exists() or force:
        with open(manifest_path, "wb") as f_out:
            pickle.dump({"config_state": current_config}, f_out)

    if files_to_process:
        process_func = partial(
            _process_slide_worker,
            cfg=current_config,
            mask_downscale=dataset_cfg.prep.mask_downscale,
        )

        for f in files_to_process:
            result = process_func(f)
            if result is not None:
                slide_pkl_path = indices_dir / f"{result.name}.pkl"
                with open(slide_pkl_path, "wb") as f_slide:
                    pickle.dump(result, f_slide)
                save_slide_json(result, slide_pkl_path)
                existing_results.append(result)
                print(f"  [SAVE] {result.name} saved")

    if existing_results:
        master_index = MasterIndex(
            file_registry=sorted(existing_results, key=lambda x: x.name),
            config_state=current_config,
        )
        with open(manifest_path, "wb") as f_final:
            pickle.dump(master_index, f_final)
        save_master_index_json(master_index, dataset_name, dataset_cfg)


def _process_slide_worker(slide_path: Path, cfg: PreprocessConfig, mask_downscale: int):
    return process_slide(slide_path, cfg, mask_downscale)


if __name__ == "__main__":
    dataset_cfg = DatasetConfig()
    parser = argparse.ArgumentParser(description="Baseline VSI Preprocessor.")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--stride", type=int, default=dataset_cfg.prep.stride)
    parser.add_argument(
        "--downsample_factor", type=int, default=dataset_cfg.prep.downsample_factor
    )
    parser.add_argument(
        "--min_tissue_coverage",
        type=float,
        default=dataset_cfg.prep.min_tissue_coverage,
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    preprocess_dataset(
        args.dataset,
        args.stride,
        args.downsample_factor,
        args.min_tissue_coverage,
        limit=args.limit,
        workers=args.workers,
        force=args.force,
    )
