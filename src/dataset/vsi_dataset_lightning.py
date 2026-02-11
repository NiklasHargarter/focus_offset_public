import torch
from torch.utils.data import Dataset
import bisect
import slideio
import os
from typing import Optional, Callable, Any
from pathlib import Path
from src import config
from src.utils.io_utils import suppress_stderr

from src.dataset.vsi_types import ProcessedIndex, SlideMetadata


class VSIDatasetLightning(Dataset):
    """
    State-of-the-art Dataset for VSI patches.
    Uses worker-safe lazy initialization with PID tracking to ensure slide handles
    are never shared across processes.
    """

    def __init__(
        self,
        index_data: ProcessedIndex,
        transform: Optional[Callable[[Any], torch.Tensor]] = None,
    ):
        self.index = index_data
        self.transform = transform

        self.file_registry = self.index.file_registry
        self.cumulative_indices = self.index.cumulative_indices
        self.total_samples = self.index.total_samples
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

    def __len__(self) -> int:
        return self.total_samples

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """
        Maps a global linear index to a specific (Slide, Patch, Z-slice) triplet.
        Calculates the focus offset in microns relative to the precomputed 'sharpest' Z.
        """
        # Find which slide this index belongs to using binary search on cumulative counts
        file_idx = bisect.bisect_right(self.cumulative_indices, idx)

        if file_idx == 0:
            local_idx = idx
        else:
            local_idx = idx - self.cumulative_indices[file_idx - 1]

        file_meta: SlideMetadata = self.file_registry[file_idx]
        vsi_name = file_meta.name
        num_z = file_meta.num_z

        # Within the slide, find the patch and the specific Z-slice
        patch_idx = local_idx // num_z
        z_level = local_idx % num_z

        # patches shape: (N, 3) -> [x, y, best_z]
        x, y, best_z = file_meta.patches[patch_idx]

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

        rect = (
            int(x),
            int(y),
            self.patch_size * self.downsample_factor,
            self.patch_size * self.downsample_factor,
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

            metadata = {
                "filename": vsi_name,
                "x": int(x),
                "y": int(y),
                "z_level": z_level,
                "optimal_z": int(best_z),
            }

            return {
                "image": image,
                "target": torch.tensor(z_offset, dtype=torch.float32),
                "metadata": metadata,
            }

        except Exception as e:
            raw_dir = config.get_vsi_raw_dir(self.dataset_name)
            full_path = raw_dir / vsi_name
            print(f"Error reading {full_path} at {rect} z={z_level}: {e}")
            raise e
