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
    mask = ((gray <= thresh) * 255).astype(np.uint8)
    return best_z, mask


def generate_patch_candidates(
    mask: np.ndarray,
    width: int,
    height: int,
    patch_size: int,
    stride: int,
    min_cov: float,
) -> List[Tuple[int, int]]:
    """Return top-left coordinates for grid patches based on masking."""
    m_h, m_w = mask.shape
    candidates = []

    for y in range(0, height - patch_size + 1, stride):
        my, mye = int(y / MASK_DOWNSCALE), int((y + patch_size) / MASK_DOWNSCALE)
        for x in range(0, width - patch_size + 1, stride):
            mx, mxe = int(x / MASK_DOWNSCALE), int((x + patch_size) / MASK_DOWNSCALE)

            mask_patch = mask[my : min(mye, m_h), mx : min(mxe, m_w)]
            if (
                mask_patch.size > 0
                and (np.count_nonzero(mask_patch) / mask_patch.size) >= min_cov
            ):
                # Returns top-left (ox, oy)
                candidates.append((x, y))
    return candidates


class SlidePreprocessor:
    def __init__(self, config: PreprocessConfig):
        self.cfg = config
        self.patch_px = self.cfg.patch_size
        self.ds = self.cfg.downsample_factor

    def process(self, vsi_path: Path) -> SlideMetadata:
        with suppress_stderr():
            slide = slideio.open_slide(str(vsi_path), "VSI")
        scene = slide.get_scene(0)
        width_raw, height_raw = scene.size
        num_z = scene.num_z_slices

        # Metadata stores downsampled dimensions
        width = width_raw // self.ds
        height = height_raw // self.ds

        print(f"[{vsi_path.name}] Stage 1: Tissue Masking...")
        _, mask = detect_tissue(scene)

        print(
            f"[{vsi_path.name}] Stage 2: Grid Generation (Stride={self.cfg.stride}, Downsample={self.ds}x)..."
        )
        # Coordinates in the RAW slide for grid generation
        # A 224px output patch at 2x downsample covers a 448px raw area.
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
            f"[{vsi_path.name}] Stage 3: Full Slide Focus Search ({total_patches} patches)..."
        )

        # scores matrix: (patch_idx, z_slice)
        scores = np.zeros((total_patches, num_z), dtype=np.float32)

        for z in range(num_z):
            print(f"  [{vsi_path.name}] Processing Z-slice {z + 1}/{num_z}...")
            with suppress_stderr():
                # Read the full slide at current Z-level, downsampled
                full_slice = scene.read_block(
                    rect=(0, 0, width_raw, height_raw),
                    size=(width, height),
                    slices=(z, z + 1),
                )

            # Convert to gray and delete BGR buffer immediately to save memory
            gray = cv2.cvtColor(full_slice, cv2.COLOR_BGR2GRAY)
            del full_slice

            # Vectorized Brenner: (I(x+2) - I(x))^2
            # We use float32 for the difference image to prevent overflow
            # We slice gray to get pixels offset by 2
            diff_sq = (
                gray[:, 2:].astype(np.float32) - gray[:, :-2].astype(np.float32)
            ) ** 2

            # Integral image allows O(1) sum retrieval for any patch
            # sat: Summed Area Table
            sat = cv2.integral(diff_sq)

            for i, (ox, oy) in enumerate(candidates):
                # Map global raw coordinates to local downsampled coordinates
                dx, dy = ox // self.ds, oy // self.ds

                # Retrieve sum from SAT: sum = sat(y2, x2) - sat(y1, x2) - sat(y2, x1) + sat(y1, x1)
                # SAT has size (H+1, W+1). diff_sq has width W-2.
                y1, y2 = dy, dy + self.cfg.patch_size
                x1, x2 = dx, dx + self.cfg.patch_size - 2

                if x2 > x1 and y2 > y1 and x2 < sat.shape[1] and y2 < sat.shape[0]:
                    patch_sum = sat[y2, x2] - sat[y1, x2] - sat[y2, x1] + sat[y1, x1]
                    scores[i, z] = patch_sum

            del gray
            del diff_sq
            del sat

        final_patches = np.zeros((total_patches, 3), dtype=np.int32)
        for i, (ox, oy) in enumerate(candidates):
            best_z = int(np.argmax(scores[i]))
            # Store patches in RAW coordinates: [x, y, z]
            final_patches[i] = [ox, oy, best_z]

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
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    preprocess_dataset(
        args.dataset,
        args.patch_size,
        args.stride,
        args.downsample_factor,
        args.min_tissue_coverage,
        limit=args.limit,
        force=args.force,
    )
