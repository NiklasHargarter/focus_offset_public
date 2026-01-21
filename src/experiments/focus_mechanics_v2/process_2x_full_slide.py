import argparse
import time
from pathlib import Path
import numpy as np
import slideio
from tqdm import tqdm

from src.utils.focus_metrics import compute_brenner_gradient
from src.utils.io_utils import suppress_stderr


def process_full_slide_2x(vsi_path: Path):
    print(f" Processing Full Slide at 2x (30x Mag): {vsi_path.name}")

    with suppress_stderr():
        slide = slideio.open_slide(str(vsi_path), "VSI")
    scene = slide.get_scene(0)
    width, height = scene.size
    num_z = scene.num_z_slices  # 27

    downscale = 2
    patch_size = 224

    # Calculate 2x dimensions
    w_ds = width // downscale
    h_ds = height // downscale

    # To match 1x sample count (224px stride at 1x), we need 112px stride at 2x
    stride_2x = patch_size // downscale

    cols = (w_ds - patch_size) // stride_2x + 1
    rows = (h_ds - patch_size) // stride_2x + 1
    num_num_patches = cols * rows

    print(f"  -> Slide Area: {w_ds}x{h_ds} pixels")
    print(
        f"  -> Patch Grid: {cols}x{rows} = {num_num_patches} patches (Overlapping, 112px stride)"
    )
    print(f"  -> Total Operations: {num_num_patches * num_z} Brenner calculations")
    print("-" * 50)

    # State tracking: (num_patches,)
    best_scores = np.full(num_num_patches, -1.0, dtype=np.float32)
    best_z = np.zeros(num_num_patches, dtype=np.int32)

    total_start = time.time()
    io_time = 0
    math_time = 0

    # Process one Z-slice at a time across the whole slide
    for z in tqdm(range(num_z), desc="Processing Z-slices"):
        # 1. READ GLOBAL SLICE (Native Pyramid Level 1)
        io_start = time.time()
        with suppress_stderr():
            # Reading at 2x size from the VSI
            global_slice = scene.read_block(
                rect=(0, 0, width, height), size=(w_ds, h_ds), slices=(z, z + 1)
            )
        io_time += time.time() - io_start

        # 2. Iterate through all patches in this slice
        math_slice_start = time.time()
        for idx in range(num_num_patches):
            gx = idx % cols
            gy = idx // cols

            px = gx * stride_2x
            py = gy * stride_2x

            # Extract zero-copy view
            patch_view = global_slice[py : py + patch_size, px : px + patch_size]

            # Compute score
            score = compute_brenner_gradient(patch_view)

            # Update best Z
            if score > best_scores[idx]:
                best_scores[idx] = score
                best_z[idx] = z
        math_time += time.time() - math_slice_start

    total_duration = time.time() - total_start

    print("\n" + "=" * 50)
    print("2X FULL-SLIDE PROCESSING COMPLETE")
    print("=" * 50)
    print(f"Total Duration:   {total_duration:7.2f}s ({total_duration / 60:.2f} min)")
    print(f"I/O Time:         {io_time:7.2f}s ({io_time / total_duration * 100:4.1f}%)")
    print(
        f"Math Time:        {math_time:7.2f}s ({math_time / total_duration * 100:4.1f}%)"
    )
    print(f"Throughput:       {num_num_patches / total_duration:7.1f} patches/s")
    print("=" * 50)

    # Save Results
    output_dir = Path("src/experiments/focus_mechanics_v2/results")
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / f"run_2x_full_slide_{vsi_path.stem}.md"

    # Reshape best_z for a quick check
    z_map = best_z.reshape((rows, cols))

    with open(report_path, "w") as f:
        f.write("# 2x Full-Slide Processing Results\n\n")
        f.write("Processed the entire slide area at 2x natively.\n\n")
        f.write(f"- **Slide**: {vsi_path.name}\n")
        f.write(f"- **Patches**: {num_num_patches}\n")
        f.write(f"- **Total Time**: {total_duration:.2f}s\n\n")
        f.write("## Focus Map Snapshot (Top-Left 10x10)\n\n")
        small_grid = z_map[:10, :10]
        f.write("```\n")
        f.write(str(small_grid))
        f.write("\n```\n")

    print(f" Report saved to: {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vsi_path", type=str, default="001_1_HE_stack.vsi")
    args = parser.parse_args()
    process_full_slide_2x(Path(args.vsi_path))
