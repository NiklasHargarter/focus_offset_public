import cv2
import os
import time
from functools import partial
import multiprocessing as mp
from pathlib import Path

import slideio
from tqdm import tqdm

from src.datasets.zstack_he import (
    SLIDE_DIR,
    DOWNSAMPLE,
    PATCH_SIZE,
    EXCLUDE_PATTERN,
)
from src.datasets.vsi.prep.preprocess import (
    detect_tissue_mask,
    generate_tissue_patches,
)
from src.utils.io_utils import suppress_stderr


def worker_estimate_slide(slide_path: Path, params: dict) -> tuple[int, int]:
    """Return the number of valid patches for a single slide and the num_z."""

    try:
        with suppress_stderr():
            slide = slideio.open_slide(str(slide_path), "VSI")
        scene = slide.get_scene(0)
        width, height = scene.size
        num_z = scene.num_z_slices

        mid_z = num_z // 2
        d_w = width // 8
        d_h = height // 8
        with suppress_stderr():
            thumb_raw = scene.read_block(
                rect=(0, 0, width, height), size=(d_w, d_h), slices=(mid_z, mid_z + 1)
            )
        thumbnail = cv2.cvtColor(thumb_raw, cv2.COLOR_BGR2RGB)
        mask = detect_tissue_mask(thumbnail)

        candidates = generate_tissue_patches(width, height, params, mask)
        return len(candidates), num_z
    except Exception as e:
        print(f"Error processing {slide_path.name}: {e}")
        return 0, 1


def benchmark_dataset_size(dataset_name: str = "ZStack_HE", workers: int | None = None):
    raw_dir = SLIDE_DIR

    workers = workers or os.cpu_count() or 1

    all_files = sorted(list(raw_dir.glob("*.vsi")))

    print(
        f"Benchmarking dataset patches & estimating size for {dataset_name} ({len(all_files)} slides)"
    )
    print(f"Using {workers} workers.")

    params = {
        "downsample": DOWNSAMPLE,
        "cov": 0.8,
        "patch_size": PATCH_SIZE,
        "stride": PATCH_SIZE * DOWNSAMPLE,
        "exclude_pattern": EXCLUDE_PATTERN,
    }

    process_func = partial(
        worker_estimate_slide,
        params=params,
    )

    total_patches = 0
    total_images = 0

    start_time = time.time()
    ctx = mp.get_context("spawn")
    with ctx.Pool(workers) as pool:
        # use tqdm for progress bar
        results = list(tqdm(pool.imap(process_func, all_files), total=len(all_files)))

    for slide_path, (count, num_z) in zip(all_files, results):
        # We don't print every slide if it's too many, but for debugging dataset size it's fine
        # print(f"{slide_path.name}: {count} patches")
        total_patches += count
        total_images += count * num_z

    elapsed = time.time() - start_time

    print("\n--- Benchmark Results ---")
    print(f"Execution Time: {elapsed:.2f} seconds")
    print(
        f"Total valid patches (XY positions) across {len(all_files)} slides: {total_patches}"
    )
    print(f"Total individual crop images (Patches x Z-levels): {total_images}")


if __name__ == "__main__":
    benchmark_dataset_size()
