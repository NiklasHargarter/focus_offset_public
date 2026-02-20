import json
import os
import random
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import slideio
from histolab.filters.image_filters import Compose, OtsuThreshold, RgbToGrayscale
from PIL import Image

from src.datasets.zstack_he.config import PrepConfig, ZStackHEConfig
from src.utils.focus_metrics import compute_focus_score
from src.utils.io_utils import suppress_stderr


# ---------------------------------------------------------------------------
# Tissue masking
# ---------------------------------------------------------------------------


def detect_tissue_mask(thumbnail_rgb: np.ndarray) -> np.ndarray:
    """Generate a binary tissue mask via Otsu thresholding on a downscaled thumbnail."""
    pipeline = Compose([RgbToGrayscale(), OtsuThreshold()])
    bool_mask = pipeline(Image.fromarray(thumbnail_rgb))
    return (bool_mask * 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Patch grid generation
# ---------------------------------------------------------------------------


def generate_tissue_patches(
    width: int,
    height: int,
    cfg: PrepConfig,
    mask: np.ndarray,
) -> list[tuple[int, int]]:
    """Return all (x, y) patch positions that meet the tissue coverage threshold."""
    patch_size = cfg.patch_size * cfg.downsample_factor
    xs = range(0, width - patch_size + 1, cfg.stride)
    ys = range(0, height - patch_size + 1, cfg.stride)

    mw = max(1, patch_size // cfg.mask_downscale)
    mh = max(1, patch_size // cfg.mask_downscale)
    binary = mask > 0

    return [
        (x, y)
        for y in ys
        for x in xs
        if (patch := binary[y // cfg.mask_downscale : y // cfg.mask_downscale + mh,
                            x // cfg.mask_downscale : x // cfg.mask_downscale + mw]).size > 0
        and patch.mean() >= cfg.min_tissue_coverage
    ]


# ---------------------------------------------------------------------------
# Per-slide pipeline
# ---------------------------------------------------------------------------


def process_slide(slide_path: Path, cfg: PrepConfig, dry_run: bool = False) -> list[dict]:
    """Full single-slide pipeline: open → thumbnail → mask → patches → focus → rows."""
    raw_extent = cfg.patch_size * cfg.downsample_factor

    with suppress_stderr():
        slide = slideio.open_slide(str(slide_path), "VSI")
    scene = slide.get_scene(0)
    width, height = scene.size
    num_z = scene.num_z_slices

    # Tissue mask from middle Z-level (most likely to be in focus)
    mid_z = num_z // 2
    d_w, d_h = width // cfg.mask_downscale, height // cfg.mask_downscale
    with suppress_stderr():
        thumb_raw = scene.read_block(
            rect=(0, 0, width, height), size=(d_w, d_h), slices=(mid_z, mid_z + 1)
        )
    thumbnail = cv2.cvtColor(thumb_raw, cv2.COLOR_BGR2RGB)
    mask = detect_tissue_mask(thumbnail)

    # Candidate patches that overlap enough tissue
    print(f"[{slide_path.name}] tiling ({width}x{height})")
    candidates = generate_tissue_patches(width, height, cfg, mask)
    print(f"[{slide_path.name}] {len(candidates)} tissue patches")

    if not candidates:
        return []

    if dry_run:
        random.shuffle(candidates)
        candidates = candidates[:20]

    # Find sharpest Z-level per patch
    best_zs = np.zeros(len(candidates), dtype=np.int32)
    for i, (x, y) in enumerate(candidates):
        if i % 100 == 0:
            print(f"[{slide_path.name}] focus {i}/{len(candidates)}")
        with suppress_stderr():
            z_stack = scene.read_block(
                rect=(x, y, raw_extent, raw_extent),
                size=(cfg.patch_size, cfg.patch_size),
                slices=(0, num_z),
            )
        if z_stack.ndim == 3:
            z_stack = z_stack[np.newaxis]
        best_zs[i] = int(np.argmax([compute_focus_score(z_stack[z]) for z in range(num_z)]))

    # Build output rows
    return [
        {"slide_name": slide_path.name, "x": int(x), "y": int(y),
         "z_level": int(z), "optimal_z": int(best_z), "num_z": int(num_z)}
        for (x, y), best_z in zip(candidates, best_zs)
        for z in range(num_z)
    ]


# ---------------------------------------------------------------------------
# Dataset-level runner
# ---------------------------------------------------------------------------


def _process_slide_worker(slide_path: Path, cfg: PrepConfig, dry_run: bool):
    return process_slide(slide_path, cfg, dry_run)


def _run_split(pool: Pool, files: list[Path], process_func, out_path: Path, label: str) -> None:
    """Process a list of slides in parallel and save the result as parquet."""
    if not files:
        return
    print(f">> Processing {label} split ({len(files)} slides)...")
    rows = [row for result in pool.map(process_func, files) for row in result]
    if rows:
        pd.DataFrame(rows).to_parquet(out_path)
        print(f"  Saved {len(rows)} rows -> {out_path}")


def preprocess_dataset(
    dataset_name: str = "ZStack_HE",
    workers: int | None = None,
    exclude: str = "_all_",
    dry_run: bool = False,
) -> None:
    cfg = ZStackHEConfig(name=dataset_name)
    workers = workers or os.cpu_count() or 1

    if not cfg.split_path.exists():
        print(f"Error: split file {cfg.split_path} not found. Run create_split first.")
        return

    splits = json.loads(cfg.split_path.read_text())
    train_names = set(splits.get("train_pool", []))
    test_names = set(splits.get("test", []))

    all_files = sorted(cfg.raw_dir.glob("*.vsi"))
    if exclude:
        all_files = [f for f in all_files if exclude.lower() not in f.name.lower()]
    if dry_run:
        all_files = all_files[:10]

    train_files = [f for f in all_files if f.name in train_names]
    test_files = [f for f in all_files if f.name in test_names]

    print(f"Preprocessing {dataset_name} | stride={cfg.prep.stride} patch={cfg.prep.patch_size}")
    print(f"Train: {len(train_files)} slides  Test: {len(test_files)} slides")

    process_func = partial(_process_slide_worker, cfg=cfg.prep, dry_run=dry_run)

    with Pool(workers) as pool:
        _run_split(pool, train_files, process_func, cfg.get_train_index_path(), "train")
        _run_split(pool, test_files, process_func, cfg.get_test_index_path(), "test")
