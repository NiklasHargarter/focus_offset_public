import os
import argparse
import multiprocessing
import pickle
from functools import partial
from pathlib import Path
from skimage.filters import threshold_otsu
from typing import Any, Tuple, List
import cv2
import numpy as np
import slideio
from scipy.signal import find_peaks
from scipy.spatial import KDTree

from src.utils.io_utils import suppress_stderr

from src import config
from src.dataset.vsi_types import SlideMetadata, PreprocessConfig, MasterIndex
import json
from dataclasses import asdict
from src.utils.focus_metrics import compute_brenner_gradient

MASK_DOWNSCALE = 16


def detect_tissue(scene: Any) -> Tuple[int, np.ndarray]:
    """Find the sharpest slice and generate a tissue mask for sparse filtering."""
    width, height = scene.size
    num_z = scene.num_z_slices
    d_w, d_h = width // MASK_DOWNSCALE, height // MASK_DOWNSCALE

    best_score, best_img, best_z = -1.0, None, 0
    # Sample every 3rd slice for speed - mask just needs a reasonably sharp image
    for z in range(0, num_z, 3):
        img = scene.read_block(
            rect=(0, 0, width, height), size=(d_w, d_h), slices=(z, z + 1)
        )
        score = compute_brenner_gradient(img)
        if score > best_score:
            best_score, best_img, best_z = score, img, z

    if best_img is None:
        gray = np.zeros((d_h, d_w), dtype=np.uint8)
    else:
        gray = cv2.cvtColor(best_img, cv2.COLOR_BGR2GRAY)

    thresh = threshold_otsu(gray)
    # Aggressive Masking:
    # 1. Be stricter about what counts as tissue (darker pixels only).
    #    Since tissue is dark (low val) and background is white (high val),
    #    lowering the threshold excludes light-gray background noise.
    thresh = thresh * 0.90

    mask = ((gray <= thresh) * 255).astype(np.uint8)

    # 2. Morphological Opening to remove small noise specks (dust)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    return best_z, mask


def generate_patch_candidates(
    mask: np.ndarray,
    width: int,
    height: int,
    patch_size: int,
    stride: int,
    min_cov: float,
) -> List[Tuple[int, int]]:
    """Return top-left coordinates for grid patches based on masking (Vectorized)."""
    m_h, m_w = mask.shape

    # Grid generation (TOP-LEFT coordinates in RAW scale)
    x_range = np.arange(0, width - patch_size + 1, stride)
    y_range = np.arange(0, height - patch_size + 1, stride)

    X, Y = np.meshgrid(x_range, y_range, indexing="xy")

    # Map to mask coordinates (downscaled)
    # Note: mask indices are int(raw / MASK_DOWNSCALE)
    mx = (X / MASK_DOWNSCALE).astype(np.int32)
    my = (Y / MASK_DOWNSCALE).astype(np.int32)
    mxe = ((X + patch_size) / MASK_DOWNSCALE).astype(np.int32)
    mye = ((Y + patch_size) / MASK_DOWNSCALE).astype(np.int32)

    # Clip to legal mask boundaries
    mx = np.clip(mx, 0, m_w)
    my = np.clip(my, 0, m_h)
    mxe = np.clip(mxe, 0, m_w)
    mye = np.clip(mye, 0, m_h)

    # Compute Tissue Area using Integral Image (Summed Area Table)
    # mask is uint8 (0 or 255). converting to binary 0/1 for area count.
    mask_bin = (mask > 0).astype(np.uint8)
    sat = cv2.integral(mask_bin)  # Shape (H+1, W+1)

    # Sum of rect defined by [y1, y2) and [x1, x2) in 0-indexed terms is:
    # SAT[y2, x2] - SAT[y1, x2] - SAT[y2, x1] + SAT[y1, x1]
    # Here, my/mye are already consistent with 0-indexed slicing range [my:mye].
    # So y1=my, y2=mye.
    tissue_area = sat[mye, mxe] - sat[my, mxe] - sat[mye, mx] + sat[my, mx]

    # Total area of the patch in the mask domain
    total_area = (mye - my) * (mxe - mx)

    # Avoid division by zero
    total_area[total_area == 0] = 1

    coverage = tissue_area / total_area
    valid_mask = (coverage >= min_cov) & (total_area > 0)

    # Extract valid coordinates
    valid_y_idx, valid_x_idx = np.where(valid_mask)
    valid_X = X[valid_y_idx, valid_x_idx]
    valid_Y = Y[valid_y_idx, valid_x_idx]

    return list(zip(valid_X, valid_Y))


def fit_plane_robust(X, Y, Z, iterations=5, threshold=4.0):
    mask = np.ones_like(Z, dtype=bool)
    params = None
    for _ in range(iterations):
        if np.sum(mask) < 4:
            break
        A = np.column_stack([X[mask], Y[mask], np.ones_like(X[mask])])
        params, _, _, _ = np.linalg.lstsq(A, Z[mask], rcond=None)

        Z_pred = np.column_stack([X, Y, np.ones_like(X)]) @ params
        errors = np.abs(Z - Z_pred)
        mask = errors < threshold

    return params, (errors < threshold) if params is not None else mask


def get_spatial_clusters(X, Y, threshold=2000):
    """Simple distance-based clustering to find separate tissue sections."""
    n = len(X)
    clusters = -1 * np.ones(n, dtype=int)
    cluster_id = 0

    if n == 0:
        return clusters, 0

    tree = KDTree(np.column_stack([X, Y]))

    for i in range(n):
        if clusters[i] == -1:
            # New cluster
            to_process = [i]
            clusters[i] = cluster_id
            while to_process:
                curr = to_process.pop()
                neighbors = tree.query_ball_point([X[curr], Y[curr]], threshold)
                for next_node in neighbors:
                    if clusters[next_node] == -1:
                        clusters[next_node] = cluster_id
                        to_process.append(next_node)
            cluster_id += 1
    return clusters, cluster_id


class SlidePreprocessor:
    def __init__(self, config: PreprocessConfig):
        self.cfg = config
        self.patch_px = self.cfg.patch_size
        self.ds = self.cfg.downsample_factor
        self.spatial_threshold = 4.0  # Slices
        self.min_prominence = 0.85  # Peak ambiguity

    def process(self, vsi_path: Path) -> SlideMetadata:
        with suppress_stderr():
            slide = slideio.open_slide(str(vsi_path), "VSI")
        scene = slide.get_scene(0)
        width_raw, height_raw = scene.size
        num_z = scene.num_z_slices

        print(f"[{vsi_path.name}] Stage 1: Tissue Masking...")
        _, mask = detect_tissue(scene)

        print(f"[{vsi_path.name}] Stage 2: Grid Generation ({self.ds}x)...")
        raw_patch_size = self.cfg.patch_size * self.ds
        raw_stride = self.cfg.stride * self.ds

        candidates = generate_patch_candidates(
            mask,
            width_raw,
            height_raw,
            raw_patch_size,
            raw_stride,
            self.cfg.min_tissue_coverage,
        )

        total_patches = len(candidates)
        print(
            f"[{vsi_path.name}] Stage 3: Focus Search + AMBIGUITY CHECK ({total_patches} patches)..."
        )

        if total_patches == 0:
            print(f"[{vsi_path.name}] No patches found.")
            return SlideMetadata(
                name=vsi_path.name,
                width=width_raw,
                height=height_raw,
                num_z=num_z,
                patches=np.array([], dtype=np.int32),
            )

        # Optimization: Find bounding box of all candidates to read minimized area
        c_arr = np.array(candidates)
        min_x_raw, min_y_raw = np.min(c_arr, axis=0)
        max_x_raw, max_y_raw = np.max(c_arr, axis=0)

        # Add width/height of one patch
        max_x_raw += raw_patch_size
        max_y_raw += raw_patch_size

        # Clamp
        min_x_raw = max(0, min_x_raw)
        min_y_raw = max(0, min_y_raw)
        max_x_raw = min(width_raw, max_x_raw)
        max_y_raw = min(height_raw, max_y_raw)

        roi_w_raw = max_x_raw - min_x_raw
        roi_h_raw = max_y_raw - min_y_raw

        # Downsampled ROI
        roi_w = roi_w_raw // self.ds
        roi_h = roi_h_raw // self.ds

        # Offsets for coordinate mapping
        offset_x_raw = min_x_raw
        offset_y_raw = min_y_raw

        scores = np.zeros((total_patches, num_z), dtype=np.float32)

        # Pre-compute Vectorized Arrays for Scoring
        # This replaces the inner loop over candidates
        c_arr = np.array(candidates)  # Shape (N, 2)
        oxs = c_arr[:, 0]
        oys = c_arr[:, 1]

        # Map global coordinates to ROI-local coordinates
        local_oxs = oxs - offset_x_raw
        local_oys = oys - offset_y_raw

        # Calculate patch bounds in the focus metric arrays (SAT)
        dxs = local_oxs // self.ds
        dys = local_oys // self.ds

        y1s = dys
        y2s = dys + self.cfg.patch_size
        x1s = dxs
        x2s = dxs + self.cfg.patch_size - 2  # -2 accounts for gradient image width redn

        # We assume ROI is large enough, but we perform a validity check just in case.
        # The SAT shape will be (roi_h + 1, roi_w - 2 + 1)
        # Note: roi_w and roi_h are calculated in lines 202-203
        sat_max_w = (roi_w_raw // self.ds) - 2 + 1
        sat_max_h = (roi_h_raw // self.ds) + 1

        valid_patch_mask = (
            (x2s > x1s)
            & (y2s > y1s)
            & (x2s < sat_max_w)
            & (y2s < sat_max_h)
            & (x1s >= 0)
            & (y1s >= 0)
        )

        valid_indices_vec = np.where(valid_patch_mask)
        vp_y1 = y1s[valid_indices_vec]
        vp_y2 = y2s[valid_indices_vec]
        vp_x1 = x1s[valid_indices_vec]
        vp_x2 = x2s[valid_indices_vec]

        for z in range(num_z):
            with suppress_stderr():
                # Read ONLY the ROI
                full_slice = scene.read_block(
                    rect=(min_x_raw, min_y_raw, roi_w_raw, roi_h_raw),
                    size=(roi_w, roi_h),
                    slices=(z, z + 1),
                )

            gray = cv2.cvtColor(full_slice, cv2.COLOR_BGR2GRAY)
            del full_slice

            # Calculate Gradient Image
            diff_sq = (
                gray[:, 2:].astype(np.float32) - gray[:, :-2].astype(np.float32)
            ) ** 2
            sat = cv2.integral(diff_sq)

            # Vectorized Score Extraction
            # SAT sum: sat[y2, x2] - sat[y1, x2] - sat[y2, x1] + sat[y1, x1]
            patch_sums = (
                sat[vp_y2, vp_x2]
                - sat[vp_y1, vp_x2]
                - sat[vp_y2, vp_x1]
                + sat[vp_y1, vp_x1]
            )

            scores[valid_indices_vec, z] = patch_sums

            del gray
            del diff_sq
            del sat

        # --- Integrated Filtering Logic ---
        final_patches = []
        valid_indices = []

        # 1. Peak Ambiguity Filter (In-Memory!)
        ambiguous_count = 0
        for i, (ox, oy) in enumerate(candidates):
            patch_scores = scores[i]
            if patch_scores.max() == 0:
                continue

            # Normalize and find peaks
            norm_scores = patch_scores / patch_scores.max()
            peaks, properties = find_peaks(norm_scores, height=0.5, distance=3)

            is_ambiguous = False
            if len(peaks) > 1:
                peak_heights = properties["peak_heights"]
                sorted_indices = np.argsort(peak_heights)[::-1]
                p1 = peak_heights[sorted_indices[0]]
                p2 = peak_heights[sorted_indices[1]]
                if p2 > (self.min_prominence * p1):
                    is_ambiguous = True

            if not is_ambiguous:
                best_z = int(np.argmax(patch_scores))
                # Store candidate for spatial check
                final_patches.append([ox, oy, best_z])
                valid_indices.append(i)
            else:
                ambiguous_count += 1

        print(f"[{vsi_path.name}] Ambiguity Filter: Removed {ambiguous_count} patches.")

        # 2. Spatial Filter
        final_patches = np.array(final_patches, dtype=np.int32)
        if len(final_patches) > 10:
            X, Y, Z = final_patches[:, 0], final_patches[:, 1], final_patches[:, 2]

            # Clustering (5000px threshold ~ 4-5 patches at stride 448)
            clusters, num_clusters = get_spatial_clusters(X, Y, threshold=5000)

            keep_mask = np.zeros(len(final_patches), dtype=bool)

            for cid in range(num_clusters):
                c_mask = clusters == cid
                if np.sum(c_mask) < 4:
                    continue
                _, inliers = fit_plane_robust(
                    X[c_mask], Y[c_mask], Z[c_mask], threshold=self.spatial_threshold
                )
                keep_mask[c_mask] = inliers

            spatial_removed = len(final_patches) - np.sum(keep_mask)
            final_patches = final_patches[keep_mask]

            print(
                f"[{vsi_path.name}] Spatial Filter: Removed {spatial_removed} outliers."
            )

        return SlideMetadata(
            name=vsi_path.name,
            width=width_raw,
            height=height_raw,
            num_z=num_z,
            patches=final_patches,
        )


def load_master_index(dataset_name: str, patch_size: int) -> MasterIndex | None:
    """Helper to load all individual slide indices and the manifest configuration."""
    manifest_path = config.get_master_index_path(dataset_name, patch_size)
    indices_dir = config.get_slide_index_dir(dataset_name, patch_size)

    if not manifest_path.exists():
        return None

    try:
        with open(manifest_path, "rb") as f:
            manifest_data = pickle.load(f)

        # Load all slide pickles from the indices directory
        slide_metadatas = []
        for pkl_path in sorted(indices_dir.glob("*.pkl")):
            try:
                with open(pkl_path, "rb") as f:
                    slide_metadatas.append(pickle.load(f))
            except Exception as slide_err:
                print(
                    f"Warning: Could not load {pkl_path}, might be corrupted correctly. Skipping. Error: {slide_err}"
                )
                continue

        return MasterIndex(
            file_registry=slide_metadatas,
            patch_size=patch_size,
            config_state=manifest_data["config_state"],
        )
    except Exception as e:
        print(f"Error loading indices: {e}")
        return None


def save_slide_json(result: SlideMetadata, pkl_path: Path):
    """Save a compact JSON sidecar for a slide index."""
    json_path = pkl_path.with_suffix(".json")
    data = {
        "name": result.name,
        "width": result.width,
        "height": result.height,
        "num_z": result.num_z,
        "patch_count": result.patch_count,
        "patches": result.patches.tolist(),
    }
    with open(json_path, "w") as f:
        json.dump(data, f)


def save_master_index_json(
    master_index: MasterIndex, dataset_name: str, patch_size: int
) -> None:
    """Export the MasterIndex to a JSON file for shareability."""
    json_path = config.get_master_index_path(dataset_name, patch_size).with_suffix(
        ".json"
    )

    data = {
        "dataset_name": dataset_name,
        "patch_size": patch_size,
        "config": asdict(master_index.config_state),
        "slides": [],
    }

    for slide in master_index.file_registry:
        data["slides"].append(
            {
                "name": slide.name,
                "width": slide.width,
                "height": slide.height,
                "num_z": slide.num_z,
                "patch_count": slide.patch_count,
                "patches": slide.patches.tolist(),
            }
        )

    with open(json_path, "w") as f:
        json.dump(data, f)
    print(f"  [EXPORT] Consolidated JSON index saved to {json_path}")


def preprocess_dataset(
    dataset_name: str,
    patch_size: int,
    stride: int,
    downsample_factor: int,
    min_tissue_coverage: float,
    limit: int | None = None,
    workers: int | None = None,
    force: bool = False,
) -> None:
    """Preprocess all slides in a dataset into a MasterIndex."""
    raw_dir = config.get_vsi_raw_dir(dataset_name)
    manifest_path = config.get_master_index_path(dataset_name, patch_size)
    indices_dir = config.get_slide_index_dir(dataset_name, patch_size)
    # Limit workers: Each worker can use up to 15GB RAM (Full Slide BGR + Gray + SAT)
    # 128GB / 15GB = ~8 workers max.
    # To be extremely safe and avoid system slowdown, we default to 2.
    workers = workers or min(os.cpu_count() or 1, 2)

    current_config = PreprocessConfig(
        patch_size=patch_size,
        stride=stride,
        downsample_factor=downsample_factor,
        min_tissue_coverage=min_tissue_coverage,
        dataset_name=dataset_name,
    )

    # 1. Check for existing indices
    existing_results = []
    processed_names = set()

    # Load manifest config
    if manifest_path.exists() and not force:
        with open(manifest_path, "rb") as f:
            manifest_data = pickle.load(f)

        if manifest_data["config_state"] != current_config:
            print(f"Config mismatch! Existing config: {manifest_data['config_state']}")
            print("Delete the index directory or use --force to reprocess.")
            return

        # Load all individual slide pickles
        for pkl_path in sorted(indices_dir.glob("*.pkl")):
            try:
                with open(pkl_path, "rb") as f:
                    res = pickle.load(f)

                # Structural Validation: Ensure it uses the new Numpy format
                if not isinstance(res.patches, np.ndarray):
                    print(
                        f"  [REPAIR] {pkl_path.name} uses obsolete format. Queuing for re-processing."
                    )
                    continue

                existing_results.append(res)
                processed_names.add(res.name)
            except Exception as e:
                print(
                    f"  [REPAIR] {pkl_path.name} is corrupted. Queuing for re-processing. Error: {e}"
                )
                continue

        print(f"Found {len(existing_results)} already processed slides.")

    # 2. Identify remaining work
    all_files = sorted(list(raw_dir.glob("*.vsi")))
    if limit:
        all_files = all_files[:limit]

    files_to_process = [f for f in all_files if f.name not in processed_names]

    print(f"Preprocessing {dataset_name} (Stride={stride}, Patch={patch_size})")
    print(
        f"Total: {len(all_files)} | Skipping: {len(processed_names)} | Remaining: {len(files_to_process)}"
    )

    # Save initial manifest to mark the start/config
    with open(manifest_path, "wb") as f:
        pickle.dump({"config_state": current_config}, f)

    if files_to_process:
        # Use a Pool to process slides
        with multiprocessing.Pool(workers) as pool:
            process_func = partial(process_slide_wrapper, config=current_config)

            # Use imap to save results as they come in (atomic saving)
            for result in pool.imap_unordered(process_func, files_to_process):
                if result is not None:
                    # Atomic Save: Write to .tmp then rename to .pkl
                    slide_pkl_path = indices_dir / f"{result.name}.pkl"
                    tmp_path = slide_pkl_path.with_suffix(".tmp")
                    with open(tmp_path, "wb") as f:
                        pickle.dump(result, f)
                    tmp_path.rename(slide_pkl_path)

                    save_slide_json(result, slide_pkl_path)

                    existing_results.append(result)
                    print(f"  [SAVE] {result.name} saved (.pkl + .json)")

    # 3. Final Consolidation & Master JSON Export
    if existing_results:
        master_index = MasterIndex(
            file_registry=sorted(existing_results, key=lambda x: x.name),
            patch_size=patch_size,
            config_state=current_config,
        )
        total_patches = sum(s.patch_count for s in master_index.file_registry)
        print(f"Completed! Total Unique Patches: {total_patches:,}")
        save_master_index_json(master_index, dataset_name, patch_size)
    else:
        print("No slides were processed or found.")


def process_slide_wrapper(vsi_path: Path, config: PreprocessConfig):
    return SlidePreprocessor(config).process(vsi_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="High-Throughput VSI Preprocessor.")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--patch_size", type=int, default=224)
    parser.add_argument("--stride", type=int, default=448)
    parser.add_argument("--downsample_factor", type=int, default=2)
    parser.add_argument("--min_tissue_coverage", type=float, default=0.05)
    parser.add_argument(
        "--limit", type=int, default=None, help="Limit number of slides to process"
    )
    parser.add_argument(
        "--workers", type=int, default=None, help="Number of parallel workers"
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    preprocess_dataset(
        args.dataset,
        args.patch_size,
        args.stride,
        args.downsample_factor,
        args.min_tissue_coverage,
        limit=args.limit,
        workers=args.workers,
        force=args.force,
    )
