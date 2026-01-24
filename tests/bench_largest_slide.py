import slideio
import numpy as np
import time
from src import config
from src.dataset.vsi_prep.preprocess import (
    detect_tissue,
    generate_patch_candidates,
)


def benchmark_largest_slide_load():
    # 1. Find the largest slide in the HE dataset
    raw_dir = config.get_vsi_raw_dir("ZStack_HE")
    all_files = list(raw_dir.glob("*.vsi"))

    print(f"Scanning {len(all_files)} files to find the largest...")

    largest_file = None
    max_pixels = 0
    max_scene = None

    for f in all_files:
        try:
            slide = slideio.open_slide(str(f), "VSI")
            scene = slide.get_scene(0)
            pixels = scene.size[0] * scene.size[1]
            if pixels > max_pixels:
                max_pixels = pixels
                largest_file = f
                max_scene = scene
        except Exception:
            continue

    if not largest_file:
        print("No files found!")
        return

    width, height = max_scene.size
    print(f"\nTarget Slide: {largest_file.name}")
    print(f"Dimensions: {width} x {height} ({max_pixels / 1e6:.1f} MP)")

    # 2. Simulate Stage 1 & 2 to find the crop area
    print("Simulating Tissue Detect & ROI Calculation...")
    # NOTE: We use the actual functions from preprocess.py to be accurate
    best_z, mask = detect_tissue(max_scene)

    patch_size = 224
    stride = 448  # Using your settings
    ds = 2
    raw_patch = patch_size * ds
    raw_stride = stride * ds

    candidates = generate_patch_candidates(
        mask, width, height, raw_patch, raw_stride, 0.05
    )

    num_patches = len(candidates)
    print(f"Candidates found: {num_patches}")

    if num_patches == 0:
        print("No tissue found, cannot benchmark read.")
        return

    # Calculate ROI
    c_arr = np.array(candidates)
    min_x_raw, min_y_raw = np.min(c_arr, axis=0)
    max_x_raw, max_y_raw = np.max(c_arr, axis=0)

    max_x_raw += raw_patch
    max_y_raw += raw_patch

    # Clip
    min_x_raw = max(0, min_x_raw)
    min_y_raw = max(0, min_y_raw)
    max_x_raw = min(width, max_x_raw)
    max_y_raw = min(height, max_y_raw)

    roi_w_raw = int(max_x_raw - min_x_raw)
    roi_h_raw = int(max_y_raw - min_y_raw)

    roi_w = roi_w_raw // ds
    roi_h = roi_h_raw // ds

    print("\n--- Read Statistics ---")
    print(f"Full dimensions: {width} x {height}")
    print(f"ROI coordinates: x={min_x_raw}, y={min_y_raw}")
    print(
        f"ROI Raw Scale:   {roi_w_raw} x {roi_h_raw} ({roi_w_raw * roi_h_raw / 1e6:.1f} MP)"
    )
    print(
        f"ROI Target (2x): {roi_w} x {roi_h} -> {roi_w * roi_h * 3 / 1024**2:.2f} MB (BGR uncompressed)"
    )

    pct_area = (roi_w_raw * roi_h_raw) / (width * height) * 100
    print(f"Area Saving: Reading {pct_area:.1f}% of the slide surface.")

    # 3. Perform the actual read benchmark
    print("\nExecuting Read Benchmark (10 iterations)...")
    times = []

    # Warmup
    _ = max_scene.read_block(
        rect=(min_x_raw, min_y_raw, roi_w_raw, roi_h_raw),
        size=(roi_w, roi_h),
        slices=(0, 1),
    )

    for i in range(5):
        t0 = time.time()
        # This is the exact line used in preprocess.py
        block = max_scene.read_block(
            rect=(min_x_raw, min_y_raw, roi_w_raw, roi_h_raw),
            size=(roi_w, roi_h),
            slices=(0, 1),  # Reading 1 Z-slice
        )
        dt = time.time() - t0
        times.append(dt)
        print(f"Iter {i + 1}: {dt:.4f}s")
        del block

    avg = np.mean(times)
    print(f"\nAverage Read Time per Z-slice: {avg:.4f}s")
    print(f"Projected time for 27 Z-slices (I/O only): {avg * 27:.2f}s")


if __name__ == "__main__":
    benchmark_largest_slide_load()
