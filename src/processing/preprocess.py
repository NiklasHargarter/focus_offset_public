import os
import json

import pickle
import numpy as np
import cv2
import slideio
import multiprocessing
from functools import partial
from skimage.filters import threshold_otsu

import config
from src.utils.io_utils import suppress_stderr


def compute_brenner_gradient(image: np.ndarray) -> int:
    """Compute Brenner Gradient focus measure (sum of squared differences)."""
    # Image is BGR; convert to gray
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Cast to int32 to prevent overflow
    gray = gray.astype(np.int32)
    shifted = np.roll(gray, -2, axis=1)
    return int(np.sum((gray - shifted) ** 2))


def select_best_focus_slice(scene, width, height, num_z, down_w, down_h):
    """Iterate all Z-slices to find the global best focus slice (grayscale)."""
    best_focus_score = -1
    best_img = None

    for z in range(num_z):
        img = scene.read_block(
            rect=(0, 0, width, height), size=(down_w, down_h), slices=(z, z + 1)
        )
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        score = compute_brenner_gradient(img)

        if score > best_focus_score:
            best_focus_score = score
            best_img = gray

    return best_img


def generate_tissue_mask(image_gray):
    """Generate binary tissue mask via Otsu's method."""
    if image_gray is None:
        return None
    thresh = threshold_otsu(image_gray)
    # Tissue is darker than background in Brightfield
    return ((image_gray <= thresh) * 255).astype(np.uint8)


def find_valid_patches(
    mask, width, height, patch_size, stride, downscale_factor, down_w, down_h
):
    """Scan mask/slide area to find valid patches based on tissue coverage."""
    valid_patches = []
    min_tissue_coverage = config.MIN_TISSUE_COVERAGE

    # Iterate in Full Resolution Coordinates
    for y in range(0, height - patch_size + 1, stride):
        ds_y = int(y / downscale_factor)
        ds_y_end = int((y + patch_size) / downscale_factor)
        ds_y_end = min(ds_y_end, down_h)
        if ds_y >= down_h:
            continue

        for x in range(0, width - patch_size + 1, stride):
            ds_x = int(x / downscale_factor)
            ds_x_end = int((x + patch_size) / downscale_factor)
            ds_x_end = min(ds_x_end, down_w)
            if ds_x >= down_w:
                continue

            mask_patch = mask[ds_y:ds_y_end, ds_x:ds_x_end]
            if mask_patch.size == 0:
                continue

            tissue_ratio = np.count_nonzero(mask_patch) / mask_patch.size

            if tissue_ratio >= min_tissue_coverage:
                valid_patches.append((x, y))

    return valid_patches


def resolve_patch_z_levels(
    scene,
    valid_patches,
    width,
    height,
    num_z,
    down_w,
    down_h,
    patch_size,
    downscale_factor,
):
    """Determine the best Z-level for each identified patch."""
    num_patches = len(valid_patches)
    if num_patches == 0:
        return []

    # Shape: [num_patches, num_z]
    patch_scores = np.zeros((num_patches, num_z), dtype=np.float32)

    for z in range(num_z):
        # Read full downsampled frame for this Z
        img_z = scene.read_block(
            rect=(0, 0, width, height), size=(down_w, down_h), slices=(z, z + 1)
        )

        for i, (vx, vy) in enumerate(valid_patches):
            # Map valid patch coords to downsampled coords
            dx = int(vx / downscale_factor)
            dy = int(vy / downscale_factor)
            dx_end = int((vx + patch_size) / downscale_factor)
            dy_end = int((vy + patch_size) / downscale_factor)

            dx_end = min(dx_end, down_w)
            dy_end = min(dy_end, down_h)

            if dx >= down_w or dy >= down_h:
                continue

            patch_roi = img_z[dy:dy_end, dx:dx_end]
            if patch_roi.size > 0:
                patch_scores[i, z] = compute_brenner_gradient(patch_roi)

    # Select best Z
    best_z_indices = np.argmax(patch_scores, axis=1)

    final_patches = []
    for i in range(num_patches):
        vx, vy = valid_patches[i]
        final_patches.append((vx, vy, int(best_z_indices[i])))

    return final_patches


def save_visualization(
    path,
    image,
    suffix="_mask.png",
    is_patch_vis=False,
    patches=None,
    patch_size=None,
    downscale_factor=None,
):
    """Helper to save mask or patch visualizations."""
    os.makedirs(config.VIS_DIR, exist_ok=True)
    basename = os.path.splitext(os.path.basename(path))[0]
    out_path = os.path.join(config.VIS_DIR, f"{basename}{suffix}")

    if is_patch_vis and patches is not None:
        vis_img = (
            np.zeros_like(image)
            if image is not None
            else np.zeros((100, 100), np.uint8)
        )  # Fallback
        # Recalculate if image provided is mask or new canvas
        h, w = image.shape[:2]
        vis_img = np.zeros((h, w), dtype=np.uint8)

        for vx, vy in patches:
            dx = int(vx / downscale_factor)
            dy = int(vy / downscale_factor)
            dx_end = int((vx + patch_size) / downscale_factor)
            dy_end = int((vy + patch_size) / downscale_factor)
            cv2.rectangle(vis_img, (dx, dy), (dx_end, dy_end), 255, -1)

        cv2.imwrite(out_path, vis_img)
    else:
        cv2.imwrite(out_path, image)

    print(f"Saved visualization to {out_path}")


def process_slide(
    vsi_path,
    patch_size=config.PATCH_SIZE,
    stride=config.STRIDE,
    downscale_factor=config.DOWNSCALE_FACTOR,
    otsu_threshold_rel=None,
):
    with suppress_stderr():
        slide = slideio.open_slide(vsi_path, "VSI")
    scene = slide.get_scene(0)
    width, height = scene.size
    num_z = scene.num_z_slices

    # Dimensions for downsampled operations
    down_w = width // downscale_factor
    down_h = height // downscale_factor

    # 1. Select Best Focus Slice
    best_gray_img = select_best_focus_slice(scene, width, height, num_z, down_w, down_h)

    if best_gray_img is None:
        print(f"Skipping {vsi_path}: Unable to find focus slice")
        return None

    # 2. Generate Mask
    mask = generate_tissue_mask(best_gray_img)
    save_visualization(vsi_path, mask, suffix="_mask.png")

    # 3. Find Valid Patches
    valid_patches_spatial = find_valid_patches(
        mask, width, height, patch_size, stride, downscale_factor, down_w, down_h
    )

    # Visualise Patches
    save_visualization(
        vsi_path,
        mask,
        suffix="_patch_mask.png",
        is_patch_vis=True,
        patches=valid_patches_spatial,
        patch_size=patch_size,
        downscale_factor=downscale_factor,
    )

    # 4. Resolve Z-levels
    final_patches = resolve_patch_z_levels(
        scene,
        valid_patches_spatial,
        width,
        height,
        num_z,
        down_w,
        down_h,
        patch_size,
        downscale_factor,
    )  # Phase 2

    print(f"Processed {os.path.basename(vsi_path)}: {len(final_patches)} valid patches")
    return {
        "path": vsi_path,
        "width": width,
        "height": height,
        "num_z": num_z,
        "patches": final_patches,
    }


def resolve_file_list(split_files):
    """
    Validates existence of files listed in the split.
    Assumes split_files contains valid absolute paths.
    """
    valid_files = []
    missing_files = []

    for path in split_files:
        if os.path.exists(path):
            valid_files.append(path)
        else:
            missing_files.append(path)

    if missing_files:
        print(f"Warning: {len(missing_files)} files from split not found on disk.")
        for m in missing_files[:5]:
            print(f"  Missing: {m}")

    return sorted(valid_files)


def preprocess_dataset(
    patch_size=config.PATCH_SIZE,
    stride=config.STRIDE,
    workers=os.cpu_count(),
    force=False,
):
    # Ensure cache directory exists
    os.makedirs(config.CACHE_DIR, exist_ok=True)

    if not os.path.exists(config.SPLIT_FILE):
        print(
            f"Error: Split file {config.SPLIT_FILE} not found. Run create_split.py first."
        )
        return

    print(f"Loading splits from {config.SPLIT_FILE}...")
    with open(config.SPLIT_FILE, "r") as f:
        splits = json.load(f)

    # Process each split independently
    for split_name, split_files in splits.items():
        if split_name in ["seed", "total"]:
            continue  # Skip metadata keys

        output_path = os.path.join(
            config.CACHE_DIR, f"{config.INDEX_PREFIX}_{split_name}.pkl"
        )

        if os.path.exists(output_path) and not force:
            print(f"Index for {split_name} already exists at {output_path}. Skipping.")
            continue

        print(f"\n=== Generating Index for Split: {split_name.upper()} ===")

        # 1. Resolve actual files for this split
        files = resolve_file_list(split_files)

        if not files:
            print(f"No valid files found for split {split_name}. Skipping.")
            continue

        output_path = os.path.join(
            config.CACHE_DIR, f"{config.INDEX_PREFIX}_{split_name}.pkl"
        )

        print(f"Processing {len(files)} files for {split_name}...")

        # 2. Process Files (Reuse process_slide logic but feed specific file list)
        process_func = partial(process_slide, patch_size=patch_size, stride=stride)

        with multiprocessing.Pool(workers) as pool:
            results = pool.map(process_func, files)

        # 3. Aggregate
        results = [r for r in results if r is not None]

        file_registry = []
        cumulative_count = 0
        cumulative_indices = []
        total_valid_patches = 0
        total_z_slices = 0

        for res in results:
            path = res["path"]
            patches = res["patches"]
            num_z = res["num_z"]
            samples_in_file = len(patches) * num_z

            cumulative_count += samples_in_file
            cumulative_indices.append(cumulative_count)
            file_registry.append({"path": path, "patches": patches, "num_z": num_z})

            total_valid_patches += len(patches)
            total_z_slices += samples_in_file

        final_index = {
            "file_registry": file_registry,
            "cumulative_indices": np.array(cumulative_indices, dtype=np.int64),
            "total_samples": cumulative_count,
            "patch_size": patch_size,
        }

        with open(output_path, "wb") as f:
            pickle.dump(final_index, f)

        print(f"Saved {split_name} index to {output_path}")
        print(f"  - Slides: {len(file_registry)}")
        print(f"  - Patches: {total_valid_patches}")
        print(f"  - Total Samples: {cumulative_count}")


def main():
    preprocess_dataset()


if __name__ == "__main__":
    main()
