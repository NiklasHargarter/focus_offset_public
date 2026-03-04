"""
Benchmark 2: Optimal worker count for preprocessing.

Dispatches all slides (dry_run=True, 20 patches each) across different
worker counts to find the optimal parallelism for this machine.

dry_run limits each slide to 20 random patches so the full dataset of
slides can be used — giving every worker count a fair, comparable load —
while still exercising real slideio I/O and spawn overhead.

Usage:
    uv run python -m src.benchmarks.benchmark_workers [--max-workers N]
"""

import time
import os
from functools import partial
import multiprocessing as mp
from pathlib import Path

from tqdm import tqdm

from shared_datasets.zstack_he import (
    SLIDE_DIR,
    DOWNSAMPLE,
    PATCH_SIZE,
    EXCLUDE_PATTERN,
    COV,
    STRIDE,
)
from shared_datasets.vsi.prep.preprocess import process_vsi_slide


def _worker(slide_path: Path, params: dict) -> None:
    process_vsi_slide(slide_path, params, dry_run=True)


def benchmark_workers(
    dataset_name: str = "ZStack_HE", max_workers: int | None = None
) -> None:
    all_files = sorted(SLIDE_DIR.glob("*.vsi"))
    if not all_files:
        print(f"No .vsi files found in {SLIDE_DIR}")
        return

    n_cpu = os.cpu_count() or 1
    max_workers = max_workers or n_cpu
    candidates = sorted(
        {4, *[2**i for i in range(3, 8) if 2**i <= max_workers], max_workers}
    )

    print(
        f"Dataset  : {dataset_name} ({len(all_files)} slides, dry_run=20 patches each)"
    )
    print(f"CPUs     : {n_cpu}  |  sweeping workers: {candidates}")
    print()

    params = {
        "downsample": DOWNSAMPLE,
        "cov": COV,
        "patch_size": PATCH_SIZE,
        "stride": STRIDE,
        "exclude_pattern": EXCLUDE_PATTERN,
    }

    ctx = mp.get_context("spawn")
    func = partial(_worker, params=params)
    results: dict[int, float] = {}

    for n_workers in candidates:
        t0 = time.perf_counter()
        with ctx.Pool(n_workers) as pool:
            list(
                tqdm(
                    pool.imap(func, all_files),
                    total=len(all_files),
                    desc=f"workers={n_workers:2d}",
                    leave=False,
                )
            )
        elapsed = time.perf_counter() - t0
        throughput = len(all_files) / elapsed
        results[n_workers] = throughput
        print(f"  workers={n_workers:2d}  {elapsed:.1f}s  ({throughput:.2f} slides/s)")

    best = max(results, key=results.get)
    print()
    print(f"Optimal worker count: {best}  ({results[best]:.2f} slides/s)")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="ZStack_HE")
    parser.add_argument("--max-workers", type=int, default=None)
    args = parser.parse_args()
    benchmark_workers(args.dataset, args.max_workers)
