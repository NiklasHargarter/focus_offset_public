import argparse
import time
from pathlib import Path
import slideio

from src.utils.focus_metrics import compute_brenner_gradient
from src.utils.io_utils import suppress_stderr


def run_2x_full_bench(vsi_path: Path):
    print(
        f" Benchmarking Full Slide Processing at 2x (Sequential/No MP): {vsi_path.name}"
    )

    with suppress_stderr():
        slide = slideio.open_slide(str(vsi_path), "VSI")
    scene = slide.get_scene(0)
    width, height = scene.size
    num_z = scene.num_z_slices  # 27

    downscale = 2
    patch_size = 224

    # 1. Calculate Grid
    w_ds = width // downscale
    h_ds = height // downscale

    cols = w_ds // patch_size
    rows = h_ds // patch_size
    num_patches = cols * rows

    print(f"Slide: {width}x{height} native | {w_ds}x{h_ds} at 2x")
    print(f"Grid: {cols}x{rows} = {num_patches} patches (non-overlapping)")
    print(f"Z-slices: {num_z}")
    print("-" * 50)

    # We will simulate the "Legacy Style" Global Read which is fastest for single-threaded
    # because it minimizes Disk Seeks.

    print(" Simulating Global Read Strategy (1 Z-slice at a time)...")

    # Measure one slice read
    t0 = time.time()
    with suppress_stderr():
        global_slice = scene.read_block(
            rect=(0, 0, width, height), size=(w_ds, h_ds), slices=(0, 1)
        )
    t_io_single = time.time() - t0

    # Measure math for 100 patches to estimate total math time
    print(f"  -> I/O for 1 slice: {t_io_single:.3f}s")

    t_math_start = time.time()
    test_count = min(100, num_patches)
    for i in range(test_count):
        # Extract patch from global slice in RAM
        px = (i % cols) * patch_size
        py = (i // cols) * patch_size
        patch = global_slice[py : py + patch_size, px : px + patch_size]
        compute_brenner_gradient(patch)
    t_math_avg = (time.time() - t_math_start) / test_count

    print(f"  -> Math for 1 patch/1 slice: {t_math_avg * 1000:.3f}ms")

    # Estimated Totals
    total_io_time = t_io_single * num_z
    total_math_time = t_math_avg * num_patches * num_z
    total_estimated = total_io_time + total_math_time

    print("-" * 50)
    print("ESTIMATED TOTALS (Single Slide, Single Thread, 2x)")
    print("-" * 50)
    print(f"Total I/O Time:   {total_io_time:7.2f}s")
    print(f"Total Math Time:  {total_math_time:7.2f}s")
    print(
        f"TOTAL TIME:       {total_estimated:7.2f}s (~{total_estimated / 60:.1f} minutes)"
    )
    print(f"Throughput:       {num_patches / total_estimated:7.1f} patches/s")
    print("-" * 50)

    # Store results in MD
    output_dir = Path("src/experiments/focus_mechanics_v2/results")
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / f"bench_2x_full_slide_{vsi_path.stem}.md"
    with open(report_path, "w") as f:
        f.write("# Benchmark: Full Slide Processing (2x, Sequential)\n\n")
        f.write(f"- **Slide**: {vsi_path.name}\n")
        f.write(
            "- **Configuration**: No Multiprocessing, 2x Downscale, 224px Non-overlapping\n\n"
        )
        f.write("## Stats\n")
        f.write("| Metric | Value |\n")
        f.write("| :--- | :--- |\n")
        f.write(f"| Patches | {num_patches} |\n")
        f.write(f"| Total I/O Time | {total_io_time:.2f}s |\n")
        f.write(f"| Total Math Time | {total_math_time:.2f}s |\n")
        f.write(
            f"| **TOTAL TIME** | **{total_estimated:.2f}s** ({total_estimated / 60:.2f} min) |\n"
        )
        f.write(f"| Efficiency | {num_patches / total_estimated:.1f} patches/s |\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vsi_path", type=str, default="001_1_HE_stack.vsi")
    args = parser.parse_args()
    run_2x_full_bench(Path(args.vsi_path))
