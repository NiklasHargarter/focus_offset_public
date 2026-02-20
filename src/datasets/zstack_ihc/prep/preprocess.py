import json
import os
from functools import partial
from pathlib import Path

import cv2
import numpy as np
import slideio

from src.datasets.zstack_ihc.config import ZStackIHCConfig
from src.utils.grid import filter_by_tissue_coverage, generate_grid
from src.utils.tissue_mask import detect_tissue_mask
import pandas as pd
from src.datasets.zstack_ihc.config import PrepConfig
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
    slide_path: Path, cfg: PrepConfig, mask_downscale: int, dry_run: bool = False
) -> list[dict]:
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

    if dry_run and candidates:
        import random
        random.shuffle(candidates)
        candidates = candidates[:20]

    if not candidates:
        return []

    print(f"[{slide_path.name}] focus search ({len(candidates)} patches)")
    best_zs = find_best_z_per_patch(
        scene, candidates, raw_extent, cfg.patch_size, num_z
    )

    rows = []
    for (x, y), best_z in zip(candidates, best_zs):
        for z in range(num_z):
            rows.append({
                "slide_name": slide_path.name,
                "x": int(x),
                "y": int(y),
                "z_level": int(z),
                "optimal_z": int(best_z),
                "num_z": int(num_z),
            })
    
    return rows





def preprocess_dataset(
    dataset_name: str = "ZStack_IHC",
    workers: int | None = None,
    exclude: str = "_all_",
    dry_run: bool = False,
) -> None:
    dataset_cfg = ZStackIHCConfig(name=dataset_name)

    raw_dir = dataset_cfg.raw_dir
    split_file = dataset_cfg.split_path
    train_path = dataset_cfg.get_train_index_path()
    test_path = dataset_cfg.get_test_index_path()

    workers = workers or os.cpu_count() or 1

    if not split_file.exists():
        print(f"Error: Split file {split_file} not found. Run create_split first.")
        return

    with open(split_file, "r") as f:
        splits = json.load(f)

    train_files_names = set(splits.get("train_pool", []))
    test_files_names = set(splits.get("test", []))

    all_files = sorted(list(raw_dir.glob("*.vsi")))
    if exclude:
        all_files = [f for f in all_files if exclude.lower() not in f.name.lower()]
    if dry_run:
        all_files = all_files[:10]

    train_files = [f for f in all_files if f.name in train_files_names]
    test_files = [f for f in all_files if f.name in test_files_names]
    
    print(
        f"Preprocessing {dataset_name} (Stride={dataset_cfg.prep.stride}, Patch={dataset_cfg.prep.patch_size})"
    )
    print(f"Train files to process: {len(train_files)} | Test files to process: {len(test_files)}")

    from multiprocessing import Pool
    
    process_func = partial(
        _process_slide_worker,
        cfg=dataset_cfg.prep,
        mask_downscale=dataset_cfg.prep.mask_downscale,
        dry_run=dry_run,
    )

    with Pool(workers) as pool:
        if train_files:
            print(">> Processing Train Split...")
            train_rows = []
            for result_rows in pool.map(process_func, train_files):
                if result_rows:
                    train_rows.extend(result_rows)
            if train_rows:
                pd.DataFrame(train_rows).to_parquet(train_path)
                print(f"  [EXPORT] Train parquet saved to {train_path}")

        if test_files:
            print(">> Processing Test Split...")
            test_rows = []
            for result_rows in pool.map(process_func, test_files):
                if result_rows:
                    test_rows.extend(result_rows)
            if test_rows:
                pd.DataFrame(test_rows).to_parquet(test_path)
                print(f"  [EXPORT] Test parquet saved to {test_path}")


def _process_slide_worker(slide_path: Path, cfg: PrepConfig, mask_downscale: int, dry_run: bool):
    return process_slide(slide_path, cfg, mask_downscale, dry_run)

