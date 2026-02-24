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
) -> list[tuple[int, int, float]]:
    """Return (x, y, tissue_coverage) for patches meeting the tissue coverage threshold."""
    patch_size = cfg.patch_size * cfg.downsample_factor
    xs = range(0, width - patch_size + 1, cfg.stride)
    ys = range(0, height - patch_size + 1, cfg.stride)

    mw = max(1, patch_size // cfg.mask_downscale)
    mh = max(1, patch_size // cfg.mask_downscale)
    binary = mask > 0

    results = []
    for y in ys:
        for x in xs:
            patch = binary[
                y // cfg.mask_downscale : y // cfg.mask_downscale + mh,
                x // cfg.mask_downscale : x // cfg.mask_downscale + mw,
            ]
            if patch.size > 0:
                coverage = float(patch.mean())
                if coverage >= cfg.min_tissue_coverage:
                    results.append((x, y, coverage))
    return results


# ---------------------------------------------------------------------------
# Per-slide pipeline
# ---------------------------------------------------------------------------


def process_slide(
    slide_path: Path, cfg: PrepConfig, dry_run: bool = False
) -> list[dict]:
    """Full single-slide pipeline: open → thumbnail → mask → patches → focus → rows."""
    raw_extent = cfg.patch_size * cfg.downsample_factor

    with suppress_stderr():
        slide = slideio.open_slide(str(slide_path), "VSI")
    scene = slide.get_scene(0)
    width, height = scene.size
    num_z: int = scene.num_z_slices

    # Z-resolution in microns (slide-level constant)
    z_res = getattr(scene, "z_resolution", 0)
    if not z_res:
        raise RuntimeError(
            f"Slide {slide_path.name} has no Z-resolution metadata. "
            "Cannot compute physical focus offsets."
        )
    z_res_microns = z_res * 1e6

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

    # Compute focus scores and find sharpest Z-level per patch
    all_scores: list[list[float]] = []
    best_zs = np.zeros(len(candidates), dtype=np.int32)
    for i, (x, y, _cov) in enumerate(candidates):
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
        scores = [float(compute_focus_score(z_stack[z])) for z in range(num_z)]
        all_scores.append(scores)
        best_zs[i] = int(np.argmax(scores))

    best_z_list: list[int] = best_zs.tolist()

    # Build output rows — all labels precomputed
    return [
        {
            "slide_name": slide_path.name,
            "x": x,
            "y": y,
            "z_level": z,
            "optimal_z": best_z,
            "num_z": num_z,
            "z_res_microns": z_res_microns,
            "z_offset_microns": (best_z - z) * z_res_microns,
            "focus_score": scores[z],
            "max_focus_score": max(scores),
            "focus_score_range": max(scores) - min(scores),
            "tissue_coverage": cov,
        }
        for (x, y, cov), best_z, scores in zip(candidates, best_z_list, all_scores)
        for z in range(num_z)
    ]


# ---------------------------------------------------------------------------
# Dataset-level runner
# ---------------------------------------------------------------------------


def _process_slide_worker(slide_path: Path, cfg: PrepConfig, dry_run: bool):
    return process_slide(slide_path, cfg, dry_run)


def _run_split(
    pool: Pool, files: list[Path], process_func, out_path: Path, label: str
) -> None:
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
    cfg = ZStackHEConfig()
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

    print(
        f"Preprocessing {dataset_name} | stride={cfg.prep.stride} patch={cfg.prep.patch_size}"
    )
    print(f"Train: {len(train_files)} slides  Test: {len(test_files)} slides")

    process_func = partial(_process_slide_worker, cfg=cfg.prep, dry_run=dry_run)

    with Pool(workers) as pool:
        _run_split(pool, train_files, process_func, cfg.get_train_index_path(), "train")
        _run_split(pool, test_files, process_func, cfg.get_test_index_path(), "test")
