import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import slideio
import torch

from src import config
from src.dataset.base_dataset import BasePatchDataset
from src.dataset.vsi_types import ProcessedIndex
from src.utils.io_utils import suppress_stderr


from src.dataset.base_dataset import BaseGridDataset


class VSIDatasetLightning(BaseGridDataset):
    """
    Dataset for VSI patches.
    Uses worker-safe lazy initialization with PID tracking to ensure slide handles
    are never shared across processes.
    """

    def __init__(
        self,
        index_data: ProcessedIndex,
        transform: Callable[[Any], torch.Tensor] | None = None,
    ):
        super().__init__(index_data)
        self.transform = transform

        self.patch_size = self.index.patch_size
        self.downsample_factor = self.index.downsample_factor
        self.dataset_name = self.index.dataset_name

        self._slides = None
        self._owner_pid = None

    def _get_scene(self, vsi_path: str):
        """
        Lazy initialization of slide handles, unique to each worker process.
        VSI handles (via slideio) are not multi-processing safe. We track the PID
        to ensure that if a worker process is forked/spawned, it creates its own
        slide handles rather than inheriting (and corrupting) parent handles.
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
                raw_dir = config.get_vsi_raw_dir(self.dataset_name)
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

    def _get_sample(self, idx: int) -> dict[str, Any]:
        """
        Maps a global linear index to a specific (Slide, Patch, Z-slice) triplet.
        Calculates the focus offset in microns relative to the precomputed 'sharpest' Z.
        """
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
        try:
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

            metadata = BasePatchDataset.create_metadata(
                filename=vsi_name,
                x=int(x),
                y=int(y),
                z_level=z_level,
                optimal_z=int(best_z),
                num_z=num_z,
            )

            return {
                "image": image,
                "target": torch.tensor(z_offset, dtype=torch.float32),
                "metadata": metadata,
            }

        except Exception as e:
            raw_dir = config.get_vsi_raw_dir(self.dataset_name)
            full_path = raw_dir / vsi_name
            print(f"Error reading {full_path} at {rect} z={z_level}: {e}")
            raise
