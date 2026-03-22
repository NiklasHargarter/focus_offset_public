"""
Benchmark 1: Single-slide full timing for extrapolation.

Runs process_slide fully (with focus search) on a small sample of slides,
reports per-slide timing stats, and extrapolates the expected total runtime
for the full dataset.

Usage:
    uv run python -m src.benchmarks.benchmark_slide_timing [--sample-size N]
"""

import time
from tqdm import tqdm

from shared_datasets.zstack_he import (
    SLIDE_DIR,
    DOWNSAMPLE,
    PATCH_SIZE,
    COV,
)
from shared_datasets.vsi.prep.preprocess import process_vsi_slide


def benchmark_slide_timing(
    dataset_name: str = "ZStack_HE", sample_size: int = 4
) -> None:
    all_files = sorted(SLIDE_DIR.glob("*.vsi"))
    if not all_files:
        print(f"No .vsi files found in {SLIDE_DIR}")
        return

    sample = all_files[:sample_size]
    print(f"Dataset   : {dataset_name} ({len(all_files)} slides total)")
    print(f"Sample    : {len(sample)} slides, full focus search")
    print()

    slide_times: list[float] = []
    for s in tqdm(sample):
        t0 = time.perf_counter()
        process_vsi_slide(
            s,
            patch_size=PATCH_SIZE,
            downsample=DOWNSAMPLE,
            min_coverage=COV,
            dry_run=False,
        )
        slide_times.append(time.perf_counter() - t0)

    avg = sum(slide_times) / len(slide_times)
    total_est = avg * len(all_files)

    print()
    print(
        f"Per-slide : avg {avg:.1f}s  min {min(slide_times):.1f}s  max {max(slide_times):.1f}s"
    )
    print(
        f"Estimated total runtime: {total_est / 60:.1f} min  "
        f"(range {total_est * 0.8 / 60:.1f}–{total_est * 1.5 / 60:.1f} min)"
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="ZStack_HE")
    parser.add_argument("--sample-size", type=int, default=4)
    args = parser.parse_args()
    benchmark_slide_timing(args.dataset, args.sample_size)
