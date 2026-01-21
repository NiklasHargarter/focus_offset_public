import numpy as np
import slideio
import json

from src.utils.io_utils import suppress_stderr

from src import config


def get_slide_stats():
    split_file = config.get_split_path("ZStack_HE")
    with open(split_file) as f:
        splits = json.load(f)
    all_files = []
    for s in ["test", "train_pool"]:
        if s in splits:
            all_files.extend(splits[s])

    raw_dir = config.get_vsi_raw_dir("ZStack_HE")
    counts = 0
    total_w = 0
    total_h = 0
    for f in all_files:
        p = raw_dir / f
        if p.exists():
            try:
                with suppress_stderr():
                    slide = slideio.open_slide(str(p), "VSI")
                scene = slide.get_scene(0)
                total_w += scene.size[0]
                total_h += scene.size[1]
                counts += 1
            except Exception:
                pass
    return counts, total_w / counts, total_h / counts


def estimate_total_time():
    print("⏳ Estimating Total Preprocessing Time for HE Dataset...")

    num_slides, avg_w, avg_h = get_slide_stats()
    patch_size = 224
    num_z = 27

    # Calculate patches per slide (1x non-overlapping)
    patches_per_slide = (avg_w // patch_size) * (avg_h // patch_size)
    total_patches = num_slides * patches_per_slide

    print("\n Dataset Stats:")
    print(f"   Slides: {num_slides}")
    print(f"   Avg Size: {avg_w:.0f} x {avg_h:.0f}")
    print(f"   Samples/Slide (at 1x): {patches_per_slide:.0f}")
    print(f"   Total Samples (Total): {total_patches:.0f}")

    print("\n Performance Baselines:")

    # 1. 1x Strategy: 8x8 Parallel Blocks
    # From bench_extraction_throughput.py
    speed_1x = 153.0  # patches/s (aggregate across 16 workers)
    time_1x = total_patches / speed_1x

    # 2. 2x, 4x, 8x Strategy: Global Read + MP Slide Level
    # We estimate based on 2x benchmark: 35.9s for 11,466 patches (Full Slide)
    # Scaled to average slide (24,000 patches)
    # Time = (IO_time_for_2x * num_z) + (Math_per_patch * num_patches * num_z)
    io_2x = 0.8  # s per slice
    io_4x = 0.17
    io_8x = 0.05
    math_per_patch_per_z = 0.00005  # s (from 3.86s / 78057 calcs)

    def time_per_slide(io_per_z):
        io_total = io_per_z * num_z
        math_total = math_per_patch_per_z * (patches_per_slide * num_z)
        return io_total + math_total

    t_slide_2x = time_per_slide(io_2x)
    t_slide_4x = time_per_slide(io_4x)
    t_slide_8x = time_per_slide(io_8x)

    total_per_slide_ds = t_slide_2x + t_slide_4x + t_slide_8x

    # Since we process slides in parallel (16 workers)
    num_batches = np.ceil(num_slides / 16)
    time_ds = num_batches * total_per_slide_ds

    print(f"   1x Master-Block Speed: {speed_1x:.1f} patches/s")
    print(f"   2x Full-Slide Time:    {t_slide_2x:.1f}s")
    print(f"   4x Full-Slide Time:    {t_slide_4x:.1f}s")
    print(f"   8x Full-Slide Time:    {t_slide_8x:.1f}s")

    print("\n FINAL ESTIMATION:")
    print("-" * 50)
    print(f"1x Dataset Prep:    {time_1x / 3600:7.2f} hours")
    print(f"2x-8x Dataset Prep: {time_ds / 60:7.2f} minutes (Parallel across slides)")
    print("-" * 50)
    print(f"TOTAL TIME:         {(time_1x + time_ds) / 3600:7.2f} hours")
    print("-" * 50)


if __name__ == "__main__":
    estimate_total_time()
