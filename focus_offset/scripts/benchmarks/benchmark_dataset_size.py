import argparse
import os
import time
from functools import partial
import multiprocessing as mp
from pathlib import Path

import slideio
from tqdm import tqdm

from shared_datasets.vsi.prep.preprocess import get_tissue_patches
from focus_offset.utils.io_utils import suppress_stderr


def worker_estimate_slide(
    slide_path: Path, patch_size: int, downsample: int, cov: float
) -> tuple[int, int]:
    """Return the number of valid patches for a single slide and the num_z."""

    try:
        with suppress_stderr():
            slide = slideio.open_slide(str(slide_path), "VSI")
        scene = slide.get_scene(0)
        width, height = scene.size
        num_z = scene.num_z_slices

        candidates = get_tissue_patches(
            scene=scene,
            patch_size=patch_size,
            downsample=downsample,
            min_coverage=cov,
            target_downscale=8,
            return_dropped=False,
        )
        return len(candidates), num_z
    except Exception as e:
        print(f"Error processing {slide_path.name}: {e}")
        return 0, 1


def benchmark_dataset_size(dataset_path: Path, cov: float, workers: int | None = None):
    raw_dir = dataset_path

    workers = workers or os.cpu_count() or 1

    all_files = sorted(list(raw_dir.glob("*.vsi")))

    print(
        f"Benchmarking dataset patches & estimating size for {dataset_path} ({len(all_files)} slides)"
    )
    print(f"Using {workers} workers.")

    process_func = partial(
        worker_estimate_slide,
        patch_size=224,
        downsample=2,
        cov=cov,
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=Path, required=True, help="Path to slide dir")
    parser.add_argument(
        "--cov", type=float, default=0.7, help="Minimum tissue coverage"
    )
    args = parser.parse_args()
    benchmark_dataset_size(args.path, args.cov)
