import argparse
import multiprocessing
import concurrent.futures
import time
from pathlib import Path
from functools import partial

import numpy as np
import slideio

from src.utils.io_utils import suppress_stderr
from src.utils.focus_metrics import compute_brenner_gradient


def individual_read_worker(
    coords_list, vsi_path, patch_size, downscale, use_stack_read=True
):
    """
    Strategy A: Each worker processes a list of patches.
    For each patch, it reads the Z-stack.
    """
    with suppress_stderr():
        try:
            slide = slideio.open_slide(str(vsi_path), "VSI")
            scene = slide.get_scene(0)
            num_z = scene.num_z_slices
            read_size = patch_size * downscale

            for cx, cy in coords_list:
                if use_stack_read:
                    scene.read_block(
                        rect=(
                            cx - read_size // 2,
                            cy - read_size // 2,
                            read_size,
                            read_size,
                        ),
                        size=(patch_size, patch_size),
                        slices=(0, num_z),
                    )
                    # Result is (H, W, C * Z) or (H, W, Z) depending on channels
                    # We just simulate the read here
                else:
                    # Naive Individual: Loop through Z
                    for z in range(num_z):
                        scene.read_block(
                            rect=(
                                cx - read_size // 2,
                                cy - read_size // 2,
                                read_size,
                                read_size,
                            ),
                            size=(patch_size, patch_size),
                            slices=(z, z + 1),
                        )
            return True
        except Exception:
            return False


def block_read_worker(block_info, vsi_path, patch_size, downscale):
    """
    Strategy B: Read a large Z-stack block and then 'slice' it in Python.
    """
    start_x, start_y, grid_w, grid_h = block_info
    with suppress_stderr():
        try:
            slide = slideio.open_slide(str(vsi_path), "VSI")
            scene = slide.get_scene(0)
            num_z = scene.num_z_slices

            # Size of the large block to read
            native_stride = patch_size * downscale
            block_w = grid_w * native_stride
            block_h = grid_h * native_stride

            # Read the whole multi-patch block for all Z
            block_stack = scene.read_block(
                rect=(start_x, start_y, block_w, block_h),
                size=(grid_w * patch_size, grid_h * patch_size),
                slices=(0, num_z),
            )

            # Simulate 'chopping' into patches in RAM
            # In VSI files, stacked blocks usually come as [H, W, Z*C]
            for gy in range(grid_h):
                for gx in range(grid_w):
                    px = gx * patch_size
                    py = gy * patch_size
                    block_stack[py : py + patch_size, px : px + patch_size, :]

            return True
        except Exception:
            return False


def brenner_wrapper(patch):
    """Small wrapper for thread pool."""
    return compute_brenner_gradient(patch)


def parts_master_worker(roi_info, vsi_path, patch_size, downscale, num_threads=8):
    """
    Strategy D: Read a large part of the slide at raw res,
    then use internal multi-threading to process patches in RAM.
    """
    start_x, start_y, grid_w, grid_h = roi_info
    with suppress_stderr():
        try:
            slide = slideio.open_slide(str(vsi_path), "VSI")
            scene = slide.get_scene(0)
            num_z = scene.num_z_slices

            native_stride = patch_size * downscale
            block_w = grid_w * native_stride
            block_h = grid_h * native_stride

            # 1. READ LARGE BLOCK (The "Part")
            block_stack = scene.read_block(
                rect=(start_x, start_y, block_w, block_h),
                size=(grid_w * patch_size, grid_h * patch_size),
                slices=(0, num_z),
            )

            # 2. Extract patch views (no copying)
            patches = []
            for gy in range(grid_h):
                for gx in range(grid_w):
                    py, px = gy * patch_size, gx * patch_size
                    patches.append(
                        block_stack[py : py + patch_size, px : px + patch_size, :]
                    )

            # 3. Process patches in parallel using threads
            # (NumPy releases GIL for these operations)
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=num_threads
            ) as executor:
                list(executor.map(brenner_wrapper, patches))

            return True
        except Exception:
            return False


def run_benchmark(vsi_path: Path, downscale: int, num_patches: int, block_size: int):
    print(f" Benchmarking Extraction Throughput: {vsi_path.name}")
    print(f"   Config: {num_patches} patches | {downscale}x downscale")

    with suppress_stderr():
        slide = slideio.open_slide(str(vsi_path), "VSI")
    scene = slide.get_scene(0)
    width, height = scene.size

    patch_size = 224
    stride = patch_size * downscale
    grid_dim = int(np.ceil(np.sqrt(num_patches)))
    actual_num_patches = grid_dim**2
    start_x, start_y = 5000, 5000

    locs = []
    for gy in range(grid_dim):
        for gx in range(grid_dim):
            locs.append(
                (
                    start_x + gx * stride + stride // 2,
                    start_y + gy * stride + stride // 2,
                )
            )

    num_workers = 16
    chunk_size = len(locs) // num_workers
    loc_chunks = [locs[i : i + chunk_size] for i in range(0, len(locs), chunk_size)]

    # --- STRATEGY A1: Individual ---
    print("\n Strategy A: Naive Individual (1x Loop Z)...")
    t0 = time.time()
    with multiprocessing.Pool(num_workers) as pool:
        pool.map(
            partial(
                individual_read_worker,
                vsi_path=vsi_path,
                patch_size=patch_size,
                downscale=1,
                use_stack_read=False,
            ),
            loc_chunks,
        )
    dt_a = time.time() - t0
    print(f"   Speed: {actual_num_patches / dt_a:.1f} patches/sec")

    # --- STRATEGY B: Block-Based 12x12 ---
    block_bs = 12
    num_blocks = (grid_dim // block_bs) ** 2 if (grid_dim // block_bs) > 0 else 1
    blocks = []
    grid_w_blocks = grid_dim // block_bs
    if grid_w_blocks == 0:
        grid_w_blocks = 1

    for bi in range(num_blocks):
        bx = start_x + (bi % grid_w_blocks) * (block_bs * patch_size)
        by = start_y + (bi // grid_w_blocks) * (block_bs * patch_size)
        blocks.append((bx, by, block_bs, block_bs))

    print("\n Strategy B: Block-Based (12x12 at 1x Resolution)...")
    t0 = time.time()
    with multiprocessing.Pool(num_workers) as pool:
        pool.map(
            partial(
                block_read_worker, vsi_path=vsi_path, patch_size=patch_size, downscale=1
            ),
            blocks,
        )
    dt_b = time.time() - t0
    print(f"   Speed: {actual_num_patches / dt_b:.1f} patches/sec")

    # --- STRATEGY D: Parts Master (40x40 Blocks + Threaded Patches) ---
    part_size = 40
    num_parts = (grid_dim // part_size) ** 2 if (grid_dim // part_size) > 0 else 1
    parts = []
    grid_w_parts = grid_dim // part_size
    if grid_w_parts == 0:
        grid_w_parts = 1

    for pi in range(num_parts):
        px = start_x + (pi % grid_w_parts) * (part_size * patch_size)
        py = start_y + (pi // grid_w_parts) * (part_size * patch_size)
        parts.append((px, py, part_size, part_size))

    print(
        f"\n Strategy D: Parts Master ({part_size}x{part_size} + 16 internal threads)..."
    )
    t0 = time.time()
    # We use fewer workers for Strategy D because it's many-threaded internally
    # Let's use 2 workers each with 8 threads = 16 cores total
    with multiprocessing.Pool(2) as pool:
        pool.map(
            partial(
                parts_master_worker,
                vsi_path=vsi_path,
                patch_size=patch_size,
                downscale=1,
                num_threads=8,
            ),
            parts,
        )
    dt_d = time.time() - t0
    print(f"   Speed: {actual_num_patches / dt_d:.1f} patches/sec")

    print("\n" + "=" * 50)
    print("HYBRID PERFORMANCE COMPARISON")
    print("=" * 50)
    print(f"A. Individual (1x):         {actual_num_patches / dt_a:8.1f} patches/s")
    print(f"B. Block 12x12 (1x):        {actual_num_patches / dt_b:8.1f} patches/s")
    print(f"D. Parts Master (40x40):    {actual_num_patches / dt_d:8.1f} patches/s")
    print("=" * 50)


def run_sweep(vsi_path: Path, num_patches: int):
    print(f" Total Performance Sweep: {vsi_path.name}")
    print("   Searching for optimal block size across magnifications...")

    downscales = [1, 2, 4, 8]
    block_sizes = [1, 4, 8, 12, 16, 24, 32]

    matrix_results = {}  # (ds, bs) -> speed
    baselines = {}  # ds -> speed

    for ds in downscales:
        mag = 60 / ds
        print(f"\n--- Testing {mag:.1f}x Magnification ({ds}x downscale) ---")

        # Baseline: Individual
        with suppress_stderr():
            slide = slideio.open_slide(str(vsi_path), "VSI")
            slide.get_scene(0)
            test_patches = 400  # Smaller sample for the multi-sweep
            grid_dim = 20
            locs = []
            for gy in range(grid_dim):
                for gx in range(grid_dim):
                    locs.append((5000 + gx * 224 * ds, 5000 + gy * 224 * ds))
            num_workers = 16
            chunk_size = len(locs) // num_workers
            loc_chunks = [
                locs[i : i + chunk_size] for i in range(0, len(locs), chunk_size)
            ]

            t0 = time.time()
            with multiprocessing.Pool(num_workers) as pool:
                pool.map(
                    partial(
                        individual_read_worker,
                        vsi_path=vsi_path,
                        patch_size=224,
                        downscale=ds,
                        use_stack_read=False,
                    ),
                    loc_chunks,
                )
            dt = time.time() - t0
            baselines[ds] = test_patches / dt
            print(f"   Baseline (Individual): {baselines[ds]:.1f} patches/s")

        for bs in block_sizes:
            # Round up grid to multiples of bs
            grid_dim_bs = int(np.ceil(np.sqrt(num_patches) / bs) * bs)
            actual_patches = grid_dim_bs**2
            num_blocks = actual_patches // (bs**2)

            blocks = []
            for bi in range(num_blocks):
                cols = grid_dim_bs // bs
                bx = 5000 + (bi % cols) * (bs * 224 * ds)
                by = 5000 + (bi // cols) * (bs * 224 * ds)
                blocks.append((bx, by, bs, bs))

            t0 = time.time()
            # Safety: use fewer workers for huge memory blocks at low downscale
            workers = 16 if (bs * 224 / ds) < 4000 else 8
            with multiprocessing.Pool(workers) as pool:
                pool.map(
                    partial(
                        block_read_worker,
                        vsi_path=vsi_path,
                        patch_size=224,
                        downscale=ds,
                    ),
                    blocks,
                )
            dt = time.time() - t0
            speed = actual_patches / dt
            matrix_results[(ds, bs)] = speed
            print(f"   Block {bs:2}x{bs:<2}: {speed:7.1f} patches/s")

    # Final Summary
    output_dir = Path("src/experiments/focus_mechanics_v2/results")
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / f"bench_throughput_matrix_{vsi_path.stem}.md"

    with open(report_path, "w") as f:
        f.write("# Throughput Optimization Matrix\n\n")
        f.write(f"Slide: {vsi_path.name}\n\n")
        f.write("## Best Block Size per Magnification\n\n")
        f.write("| Magnification | Best Block Size | Peak Speed | Gain vs Baseline |\n")
        f.write("| :--- | :--- | :--- | :--- |\n")

        for ds in downscales:
            mag = 60 / ds
            best_bs = max(block_sizes, key=lambda b: matrix_results[(ds, b)])
            peak_speed = matrix_results[(ds, best_bs)]
            gain = (peak_speed / baselines[ds] - 1) * 100
            f.write(
                f"| {mag:.1f}x | {best_bs}x{best_bs} | {peak_speed:.1f} | {gain:>+6.1f}% |\n"
            )

        f.write("\n## Full Matrix (patches/s)\n\n")
        f.write(
            "| Block Size | "
            + " | ".join([f"{60 / ds:.1f}x" for ds in downscales])
            + " |\n"
        )
        f.write("| :--- | " + " | ".join([":---" for _ in downscales]) + " |\n")
        for bs in block_sizes:
            line = f"| **{bs}x{bs}** | "
            line += " | ".join([f"{matrix_results[(ds, bs)]:.1f}" for ds in downscales])
            f.write(line + " |\n")

    print(f"\n Multi-factor sweep complete. Results saved to: {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vsi_path", type=str, default="001_1_HE_stack.vsi")
    parser.add_argument("--downscale", type=int, default=1)
    parser.add_argument("--num_patches", type=int, default=256)
    parser.add_argument("--block_size", type=int, default=4)
    parser.add_argument("--sweep", action="store_true")
    args = parser.parse_args()

    if args.sweep:
        run_sweep(Path(args.vsi_path), 400)  # Use 400 patches per point for speed
    else:
        run_benchmark(
            Path(args.vsi_path), args.downscale, args.num_patches, args.block_size
        )
