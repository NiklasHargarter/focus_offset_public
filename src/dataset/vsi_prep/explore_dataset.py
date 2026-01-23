import argparse
import multiprocessing
from pathlib import Path
from typing import List, Tuple
import slideio
from src import config
from src.dataset.vsi_prep.preprocess import detect_tissue, generate_patch_candidates
from src.utils.io_utils import suppress_stderr


def estimate_slide_samples(
    vsi_path: Path, patch_size: int, strides: List[int], min_cov: float
) -> Tuple[str, int, List[int]]:
    """Estimate number of patches for different strides for a single slide."""
    try:
        with suppress_stderr():
            slide = slideio.open_slide(str(vsi_path), "VSI")
        scene = slide.get_scene(0)
        width, height = scene.size
        num_z = scene.num_z_slices

        # Get tissue mask once
        _, mask = detect_tissue(scene)

        counts = []
        for stride in strides:
            candidates = generate_patch_candidates(
                mask, width, height, patch_size, stride, min_cov
            )
            counts.append(len(candidates) * num_z)

        return vsi_path.name, num_z, counts
    except Exception as e:
        print(f"Error processing {vsi_path.name}: {e}")
        return vsi_path.name, 0, [0] * len(strides)


def explore_dataset(
    dataset_name: str, strides: List[int], patch_size: int, min_cov: float, workers: int
):
    raw_dir = config.get_vsi_raw_dir(dataset_name)
    all_files = sorted(list(raw_dir.glob("*.vsi")))

    if not all_files:
        print(f"No VSI files found in {raw_dir}")
        return

    print(f"Exploring {dataset_name} with {len(all_files)} slides...")
    print(f"Patch Size: {patch_size}, Min Coverage: {min_cov}")
    print(f"Testing Strides: {strides}")
    print(f"Using {workers} workers...\n")

    with multiprocessing.Pool(workers) as pool:
        process_func = estimate_slide_samples
        # Need to wrap arguments
        args = [(f, patch_size, strides, min_cov) for f in all_files]
        results = pool.starmap(process_func, args)

    # Totals
    total_z_slices = 0
    total_samples_per_stride = [0] * len(strides)

    print(
        f"{'Slide Name':<40} | {'Z':<4} | " + " | ".join([f"S={s:<4}" for s in strides])
    )
    print("-" * (40 + 8 + len(strides) * 10))

    for name, num_z, counts in results:
        total_z_slices += num_z
        status_line = f"{name:<40} | {num_z:<4} | "
        status_line += " | ".join([f"{c:<8}" for c in counts])
        print(status_line)

        for i, c in enumerate(counts):
            total_samples_per_stride[i] += c

    print("-" * (40 + 8 + len(strides) * 10))
    summary_line = f"{'TOTAL':<40} | {total_z_slices:<4} | "
    summary_line += " | ".join([f"{ts:<8}" for ts in total_samples_per_stride])
    print(summary_line)

    print("\nSummary (Millions of Samples):")
    for i, s in enumerate(strides):
        print(f"  Stride {s:4}: {total_samples_per_stride[i] / 1e6:6.2f}M samples")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Estimate dataset size for different strides."
    )
    parser.add_argument("--dataset", type=str, default="ZStack_HE")
    parser.add_argument("--patch_size", type=int, default=224)
    parser.add_argument("--min_tissue_coverage", type=float, default=0.05)
    parser.add_argument("--strides", type=int, nargs="+", default=[224, 448, 896])
    parser.add_argument("--workers", type=int, default=multiprocessing.cpu_count())

    args = parser.parse_args()
    explore_dataset(
        args.dataset,
        args.strides,
        args.patch_size,
        args.min_tissue_coverage,
        args.workers,
    )
