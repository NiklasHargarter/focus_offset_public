import argparse
import multiprocessing
import time
from pathlib import Path
from functools import partial

import slideio

from src.utils.focus_metrics import compute_brenner_gradient
from src.utils.io_utils import suppress_stderr

# Global reference for the RAM-based processing
BLOCK_RAM_BUFFER = None


def brenner_math_worker(coord_patch):
    """Worker that only does Brenner math on a view of the global RAM buffer."""
    global BLOCK_RAM_BUFFER
    if BLOCK_RAM_BUFFER is None:
        return 0

    px, py, ps = coord_patch
    # Extract the Z-stack for this patch from the global buffer
    # Assumption: Buffer is (H, W, Z*C)
    patch_stack = BLOCK_RAM_BUFFER[py : py + ps, px : px + ps, :]

    # We simulate find_best_z here
    best_s = -1.0
    # The buffer contains all Z slices
    # Let's assume it's stored packed: [..., z0c0, z0c1, z0c2, z1c0, ...]
    # compute_brenner_gradient handles Gray/BGR

    # Simple simulation: just compute brenner for one slice to measure math speed
    # or loop through simulated slices
    for z in range(27):  # Fixed slice count for benchmark consistency
        s = compute_brenner_gradient(patch_stack[..., z * 3 : (z + 1) * 3])
        if s > best_s:
            best_s = s
    return 1


def full_op_worker(loc_list, vsi_path, stride, patch_size, num_z):
    with suppress_stderr():
        slide = slideio.open_slide(str(vsi_path), "VSI")
        scene = slide.get_scene(0)
        for cx, cy in loc_list:
            img = scene.read_block(
                rect=(cx - stride // 2, cy - stride // 2, stride, stride),
                size=(patch_size, patch_size),
                slices=(0, num_z),
            )
            # Math loop
            for z in range(num_z):
                compute_brenner_gradient(img[..., z * 3 : (z + 1) * 3])


def run_downscale_benchmark(vsi_path: Path, downscale: int):
    global BLOCK_RAM_BUFFER
    print(f" Benchmarking Downscale Extraction Strategies: {vsi_path.name}")
    print(f"Target Magnification: {60 / downscale:.1f}x ({downscale}x downscale)")

    with suppress_stderr():
        slide = slideio.open_slide(str(vsi_path), "VSI")
    scene = slide.get_scene(0)
    width, height = scene.size
    num_z = scene.num_z_slices  # 27 for this slide

    patch_size = 224
    grid_dim = 40  # 1600 patches
    num_patches = grid_dim**2
    stride = patch_size * downscale

    start_x, start_y = 5000, 5000

    # Strategy A: Individual MP (I/O + Math in each worker)
    # This is our previous standard
    print("\n Strategy A: Individual MP (Parallel I/O + Math)...")
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

    t0 = time.time()
    with multiprocessing.Pool(num_workers) as pool:
        pool.map(
            partial(
                full_op_worker,
                vsi_path=vsi_path,
                stride=stride,
                patch_size=patch_size,
                num_z=num_z,
            ),
            loc_chunks,
        )
    dt_a = time.time() - t0
    print(f"   Total Time: {dt_a:.2f}s | Speed: {num_patches / dt_a:.1f} patches/s")

    # Strategy B: Single Sync Read + MP Math
    print("\n Strategy B: Single Sync Read + MP Math...")
    t_start = time.time()

    # 1. READ LARGE BLOCK (The "Part")
    print(
        f"   -> Reading {grid_dim * patch_size}x{grid_dim * patch_size} Z-stack ROI..."
    )
    with suppress_stderr():
        BLOCK_RAM_BUFFER = scene.read_block(
            rect=(start_x, start_y, grid_dim * stride, grid_dim * stride),
            size=(grid_dim * patch_size, grid_dim * patch_size),
            slices=(0, num_z),
        )
    t_io = time.time() - t_start
    print(f"      I/O completed in {t_io:.2f}s")

    # 2. Parallel Math
    patch_coords = []
    for gy in range(grid_dim):
        for gx in range(grid_dim):
            patch_coords.append((gx * patch_size, gy * patch_size, patch_size))

    t_math_start = time.time()
    with multiprocessing.Pool(num_workers) as pool:
        pool.map(brenner_math_worker, patch_coords)
    t_math = time.time() - t_math_start

    dt_b = time.time() - t_start
    print(f"      Math completed in {t_math:.2f}s")
    print(f"   Total Time: {dt_b:.2f}s | Speed: {num_patches / dt_b:.1f} patches/s")

    # Clean up global
    BLOCK_RAM_BUFFER = None

    # Final Report
    output_dir = Path("src/experiments/focus_mechanics_v2/results")
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / f"bench_downscale_strat_{vsi_path.stem}.md"

    with open(report_path, "w") as f:
        f.write("# Downscale Strategy Comparison\n\n")
        f.write(f"- **Slide**: {vsi_path.name}\n")
        f.write(f"- **Magnification**: {60 / downscale:.1f}x\n")
        f.write(f"- **Patches**: {num_patches}\n\n")
        f.write("## Results\n\n")
        f.write("| Strategy | I/O Time | Math Time | Total Time | Speed |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- |\n")
        f.write(
            f"| Individual MP | - | - | {dt_a:.2f}s | {num_patches / dt_a:.1f} p/s |\n"
        )
        f.write(
            f"| Sync Read + MP Math | {t_io:.2f}s | {t_math:.2f}s | {dt_b:.2f}s | {num_patches / dt_b:.1f} p/s |\n"
        )

    print(f"\n Benchmark complete. Results saved to: {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vsi_path", type=str, default="001_1_HE_stack.vsi")
    parser.add_argument("--downscale", type=int, default=8)
    args = parser.parse_args()
    run_downscale_benchmark(Path(args.vsi_path), args.downscale)
