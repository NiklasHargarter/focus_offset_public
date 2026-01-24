import time
import numpy as np
import cv2

MASK_DOWNSCALE = 16


def generate_patch_candidates_old(mask, width, height, patch_size, stride, min_cov):
    m_h, m_w = mask.shape
    candidates = []
    # Original iterative logic
    for y in range(0, height - patch_size + 1, stride):
        my, mye = int(y / MASK_DOWNSCALE), int((y + patch_size) / MASK_DOWNSCALE)
        for x in range(0, width - patch_size + 1, stride):
            mx, mxe = int(x / MASK_DOWNSCALE), int((x + patch_size) / MASK_DOWNSCALE)

            # Slicing is relatively slow in a loop
            mask_patch = mask[my : min(mye, m_h), mx : min(mxe, m_w)]
            if (
                mask_patch.size > 0
                and (np.count_nonzero(mask_patch) / mask_patch.size) >= min_cov
            ):
                candidates.append((x, y))
    return candidates


def generate_patch_candidates_vec(mask, width, height, patch_size, stride, min_cov):
    m_h, m_w = mask.shape

    x_range = np.arange(0, width - patch_size + 1, stride)
    y_range = np.arange(0, height - patch_size + 1, stride)

    X, Y = np.meshgrid(x_range, y_range, indexing="xy")

    mx = (X / MASK_DOWNSCALE).astype(np.int32)
    my = (Y / MASK_DOWNSCALE).astype(np.int32)
    mxe = ((X + patch_size) / MASK_DOWNSCALE).astype(np.int32)
    mye = ((Y + patch_size) / MASK_DOWNSCALE).astype(np.int32)

    mx = np.clip(mx, 0, m_w)
    my = np.clip(my, 0, m_h)
    mxe = np.clip(mxe, 0, m_w)
    mye = np.clip(mye, 0, m_h)

    # Integral image
    mask_bin = (mask > 0).astype(np.uint8)
    sat = cv2.integral(mask_bin)

    tissue_area = sat[mye, mxe] - sat[my, mxe] - sat[mye, mx] + sat[my, mx]

    total_area = (mye - my) * (mxe - mx)
    total_area[total_area == 0] = 1

    coverage = tissue_area / total_area
    valid_mask = (coverage >= min_cov) & (total_area > 0)

    valid_y_idx, valid_x_idx = np.where(valid_mask)
    valid_X = X[valid_y_idx, valid_x_idx]
    valid_Y = Y[valid_y_idx, valid_x_idx]

    return list(zip(valid_X, valid_Y))


def run_benchmark():
    # Setup - simulate a large whole slide image
    W, H = 60000, 40000
    mask_w, mask_h = W // MASK_DOWNSCALE, H // MASK_DOWNSCALE

    print(f"Creating random mask ({mask_w}x{mask_h})...")
    mask = (np.random.rand(mask_h, mask_w) > 0.9).astype(np.uint8) * 255

    patch_size = 224
    stride = 224
    min_cov = 0.05

    print(f"Benchmarking Candidate Generation (Grid: {W}x{H}, Stride: {stride})...")

    # Run Old
    start = time.time()
    res_old = generate_patch_candidates_old(mask, W, H, patch_size, stride, min_cov)
    end = time.time()
    time_old = end - start
    print(f"Old Implementation: {time_old:.4f}s | Found {len(res_old)} patches")

    # Run New
    start = time.time()
    res_vec = generate_patch_candidates_vec(mask, W, H, patch_size, stride, min_cov)
    end = time.time()
    time_vec = end - start
    print(f"New Implementation: {time_vec:.4f}s | Found {len(res_vec)} patches")

    if time_vec > 0:
        print(f"Speedup: {time_old / time_vec:.2f}x")

    # Verification
    # Sort to ensure order doesn't matter (though both should be raster order)
    set_old = set(res_old)
    set_new = set(res_vec)

    if set_old == set_new:
        print("SUCCESS: Results match exactly.")
    else:
        print(f"FAILURE: Mismatch! Old: {len(set_old)}, New: {len(set_new)}")
        missing = set_old - set_new
        extra = set_new - set_old
        if len(missing) < 5:
            print("Missing in New:", missing)
        if len(extra) < 5:
            print("Extra in New:", extra)


if __name__ == "__main__":
    run_benchmark()
