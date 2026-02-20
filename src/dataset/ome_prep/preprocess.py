import argparse
import multiprocessing
import os
import pickle
from functools import partial
from pathlib import Path

import cv2
import numpy as np
import tifffile

from src.config import DatasetConfig
from src.dataset.prep.grid import filter_by_tissue_coverage, generate_grid
from src.dataset.prep.tissue_detection import detect_tissue_mask
from src.dataset.index_types import MasterIndex, PreprocessConfig, SlideMetadata
from src.utils.focus_metrics import compute_brenner_gradient


class OMEImageReader:
    """Uses tifffile to read multi-series OME-TIFFs as Z-stacks."""

    def __init__(self, path: Path):
        self.path = path
        self._tif = tifffile.TiffFile(path)
        self.num_z = len(self._tif.series)
        first_series = self._tif.series[0]
        self.height, self.width = first_series.shape[:2]
        self.size = (self.width, self.height)

    def read_block(
        self, rect: tuple[int, int, int, int], slices: tuple[int, int] | None = None
    ) -> np.ndarray:
        x, y, w, h = rect
        z_start, z_end = slices if slices else (0, self.num_z)

        stack = []
        for z in range(z_start, z_end):
            data = self._tif.series[z].asarray()
            if data.ndim == 2:
                data = np.expand_dims(data, axis=-1)
            stack.append(data[y : y + h, x : x + w])

        return np.stack(stack)

    def close(self):
        self._tif.close()


# ---------------------------------------------------------------------------
# Single-image processing
# ---------------------------------------------------------------------------


def read_thumbnail_rgb(
    reader: OMEImageReader, width: int, height: int, mask_downscale: int
) -> np.ndarray:
    """Read a small RGB thumbnail from the first z-slice."""
    d_w = width // mask_downscale
    d_h = height // mask_downscale
    full_img = reader.read_block((0, 0, width, height), (0, 1))[0]
    small = cv2.resize(full_img, (d_w, d_h))
    return cv2.cvtColor(small, cv2.COLOR_BGR2RGB)


def find_best_z_per_patch(reader, candidates, raw_patch_size, patch_size, num_z):
    """For each candidate patch, read the full z-stack and pick the sharpest slice."""
    best_zs = np.zeros(len(candidates), dtype=np.int32)

    for i, (x, y) in enumerate(candidates):
        z_stack = reader.read_block(
            rect=(x, y, raw_patch_size, raw_patch_size), slices=(0, num_z)
        )
        scores = [
            compute_brenner_gradient(cv2.resize(z_stack[z], (patch_size, patch_size)))
            for z in range(num_z)
        ]
        best_zs[i] = int(np.argmax(scores))

    return best_zs


def build_patch_index(candidates, best_zs) -> np.ndarray:
    """Combine (x, y) candidates with best-z into an (N, 3) int32 array."""
    coords = np.array(candidates, dtype=np.int32)
    return np.column_stack([coords, best_zs])


def process_image(
    img_path: Path, cfg: PreprocessConfig, mask_downscale: int
) -> SlideMetadata:
    """Full single-image pipeline: thumbnail → mask → grid → focus → index."""
    raw_extent = cfg.patch_size * cfg.downsample_factor

    reader = OMEImageReader(img_path)
    width, height = reader.size
    num_z = reader.num_z

    print(f"[{img_path.name}] tissue mask")
    thumbnail = read_thumbnail_rgb(reader, width, height, mask_downscale)
    mask = detect_tissue_mask(thumbnail_rgb=thumbnail)

    print(f"[{img_path.name}] grid + tissue filter")
    grid = generate_grid(width, height, raw_extent, cfg.stride)
    candidates = filter_by_tissue_coverage(
        grid,
        mask,
        raw_extent,
        cfg.min_tissue_coverage,
        mask_downscale=mask_downscale,
    )

    if not candidates:
        reader.close()
        return SlideMetadata(
            name=img_path.name,
            num_z=num_z,
            patches=np.empty((0, 3), dtype=np.int32),
        )

    print(f"[{img_path.name}] focus search ({len(candidates)} patches)")
    best_zs = find_best_z_per_patch(
        reader, candidates, raw_extent, cfg.patch_size, num_z
    )
    reader.close()

    return SlideMetadata(
        name=img_path.name,
        num_z=num_z,
        patches=build_patch_index(candidates, best_zs),
    )


def _process_image_worker(img_path: Path, cfg: PreprocessConfig, mask_downscale: int):
    try:
        return process_image(img_path, cfg, mask_downscale)
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        return None


def preprocess_dataset(
    dataset_name: str,
    stride: int,
    downsample_factor: int,
    min_tissue_coverage: float,
    limit: int | None = None,
    workers: int | None = None,
    force: bool = False,
) -> None:
    dataset_cfg = DatasetConfig(name=dataset_name)
    dataset_cfg.prep.stride = stride
    dataset_cfg.prep.downsample_factor = downsample_factor
    dataset_cfg.prep.min_tissue_coverage = min_tissue_coverage

    raw_dir = dataset_cfg.raw_dir
    manifest_path = dataset_cfg.get_master_index_path()
    indices_dir = dataset_cfg.get_slide_index_dir()
    indices_dir.mkdir(parents=True, exist_ok=True)

    workers = workers or os.cpu_count() or 1
    current_config = PreprocessConfig(
        patch_size=dataset_cfg.prep.patch_size,
        stride=stride,
        downsample_factor=downsample_factor,
        min_tissue_coverage=min_tissue_coverage,
        dataset_name=dataset_name,
    )

    existing_results = []
    processed_names = set()

    if manifest_path.exists() and not force:
        with open(manifest_path, "rb") as f:
            manifest_data = pickle.load(f)
        if manifest_data.config_state == current_config:
            for pkl_path in sorted(indices_dir.glob("*.pkl")):
                with open(pkl_path, "rb") as f:
                    res = pickle.load(f)
                existing_results.append(res)
                processed_names.add(res.name)

    all_files = sorted(
        list(raw_dir.glob("*.ome.tiff")) + list(raw_dir.glob("*.ome.tif"))
    )
    if limit:
        all_files = all_files[:limit]

    files_to_process = [f for f in all_files if f.name not in processed_names]
    print(
        f"Preprocessing OME-TIFF {dataset_name} (Total: {len(all_files)}, Remaining: {len(files_to_process)})"
    )

    if files_to_process:
        with multiprocessing.Pool(workers) as pool:
            process_func = partial(
                _process_image_worker,
                cfg=current_config,
                mask_downscale=dataset_cfg.prep.mask_downscale,
            )
            for result in pool.imap_unordered(process_func, files_to_process):
                if result is not None:
                    slide_pkl_path = indices_dir / f"{result.name}.pkl"
                    with open(slide_pkl_path, "wb") as f_slide:
                        pickle.dump(result, f_slide)
                    existing_results.append(result)

    if existing_results:
        master_index = MasterIndex(
            file_registry=sorted(existing_results, key=lambda x: x.name),
            config_state=current_config,
        )
        with open(manifest_path, "wb") as f_final:
            pickle.dump(master_index, f_final)
        print(f"Master index saved to {manifest_path}")


if __name__ == "__main__":
    dataset_cfg = DatasetConfig()
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--stride", type=int, default=dataset_cfg.prep.stride)
    parser.add_argument("--downsample_factor", type=int, default=1)
    parser.add_argument("--min_tissue_coverage", type=float, default=0.01)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    preprocess_dataset(
        args.dataset,
        args.stride,
        args.downsample_factor,
        args.min_tissue_coverage,
        limit=args.limit,
        workers=args.workers,
        force=args.force,
    )
