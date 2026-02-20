import bisect
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypedDict

import slideio
import torch
from torch.utils.data import Dataset

from src.config import DatasetConfig
from src.dataset.index_types import ProcessedIndex
from src.utils.io_utils import suppress_stderr


class SampleMetadata(TypedDict, total=False):
    filename: str
    x: int
    y: int
    z_level: int
    optimal_z: int
    num_z: int


METADATA_DEFAULTS: dict[str, Any] = {
    "filename": "unknown",
    "x": 0,
    "y": 0,
    "z_level": 0,
    "optimal_z": 0,
    "num_z": 1,
}


class VSIDataset(Dataset):
    """
    Dataset for VSI patches.
    Uses worker-safe lazy initialization with PID tracking to ensure slide handles
    are never shared across processes.
    """

    def __init__(
        self,
        index_data: ProcessedIndex,
        transform: Any | None = None,
    ):
        self.index = index_data
        self.transform = transform

        self.patch_size = self.index.patch_size
        self.downsample_factor = self.index.downsample_factor
        self.dataset_name = self.index.dataset_name

        self.cumulative_indices = self.index.cumulative_indices
        self.total_samples = self.index.total_samples
        self.file_registry = self.index.file_registry

        self._slides: dict[str, Any] | None = None
        self._owner_pid: int | None = None

    def __len__(self) -> int:
        return self.total_samples

    def _get_grid_info(self, idx: int):
        """
        Resolve a global index to specific slide and patch coordinates.
        """
        # Find which slide this index belongs to
        file_idx = bisect.bisect_right(self.cumulative_indices, idx)

        if file_idx == 0:
            local_idx = idx
        else:
            local_idx = idx - self.cumulative_indices[file_idx - 1]

        file_meta = self.file_registry[file_idx]
        num_z = file_meta.num_z

        # Within the slide, find the patch and the specific Z-slice
        patch_idx = local_idx // num_z
        z_level = local_idx % num_z

        # patches shape: (N, 3) -> [x, y, best_z]
        patch_info = file_meta.patches[patch_idx]

        return file_meta, file_idx, patch_info, z_level

    def _get_scene(self, vsi_path: str):
        """
        Lazy initialization of slide handles, unique to each worker process.
        """
        current_pid = os.getpid()

        if self._slides is None or self._owner_pid != current_pid:
            self._slides = {}
            self._owner_pid = current_pid

        if vsi_path not in self._slides:
            # Robust path handling: Check if the stored path exists,
            # if not, try to find it in the current dataset raws directory.
            actual_path = Path(vsi_path)
            if not actual_path.exists() and self.dataset_name:
                raw_dir = DatasetConfig(name=self.dataset_name).raw_dir
                alt_path = raw_dir / actual_path.name
                if alt_path.exists():
                    actual_path = alt_path

            try:
                with suppress_stderr():
                    slide = slideio.open_slide(str(actual_path), "VSI")
                    scene = slide.get_scene(0)
                self._slides[vsi_path] = (slide, scene)
            except Exception as e:
                raise RuntimeError(
                    f"Error opening slide in process {current_pid}: {actual_path}. Error: {e}"
                )

        return self._slides[vsi_path][1]

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """
        Maps a global linear index to a specific (Slide, Patch, Z-slice) triplet.
        """
        try:
            file_meta, _, patch_info, z_level = self._get_grid_info(idx)

            vsi_name = file_meta.name
            num_z = file_meta.num_z
            x, y, best_z = patch_info

            scene = self._get_scene(vsi_name)

            # Calculate focus offset in microns (Target for the regression model)
            z_res = getattr(scene, "z_resolution", 0)
            if not z_res:
                raise RuntimeError(
                    f"Slide {vsi_name} from {self.dataset_name} has no Z-resolution metadata. "
                    "Physical micron-based focus offset cannot be calculated."
                )
            z_res_microns = z_res * 1e6
            z_offset = (best_z - z_level) * z_res_microns

            raw_extent = self.patch_size * self.downsample_factor
            rect = (
                int(x),
                int(y),
                raw_extent,
                raw_extent,
            )

            # Efficient single-slice read from VSI, with downscaling
            block = scene.read_block(
                rect=rect,
                size=(self.patch_size, self.patch_size),
                slices=(z_level, z_level + 1),
            )

            if self.transform:
                # Albumentations standard call
                image = self.transform(image=block)["image"]
            else:
                image = torch.from_numpy(block).permute(2, 0, 1).float() / 255.0

            metadata = {
                "filename": vsi_name,
                "x": int(x),
                "y": int(y),
                "z_level": int(z_level),
                "optimal_z": int(best_z),
                "num_z": int(num_z),
            }

            # Fill defaults if missing (though here we explicitly set them all)
            for key, default in METADATA_DEFAULTS.items():
                if key not in metadata:
                    metadata[key] = default

            return {
                "image": image,
                "target": torch.tensor(z_offset, dtype=torch.float32),
                "metadata": metadata,
            }

        except Exception as e:
            # We don't have vsi_name/rect/z_level if _get_grid_info fails
            # But usually it fails inside the try block after resolving
            print(f"Error reading index {idx}: {e}")
            raise
