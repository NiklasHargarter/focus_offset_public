import torch
from torch.utils.data import Dataset
import bisect
import slideio
import os
from typing import Optional, Callable, Any
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
            try:
                with suppress_stderr():
                    slide = slideio.open_slide(str(vsi_path), "VSI")
                    scene = slide.get_scene(0)
                self._slides[vsi_path] = (slide, scene)
            except Exception as e:
                raise RuntimeError(
                    f"Error opening slide in process {current_pid}: {vsi_path}. Error: {e}"
                )

        return self._slides[vsi_path][1]

    def __len__(self) -> int:
        return self.total_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
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
        vsi_path = file_meta.path
        patches = file_meta.patches
        num_z = file_meta.num_z

        # Within the slide, find the patch and the specific Z-slice
        patch_idx = local_idx // num_z
        z_level = local_idx % num_z
        patch = patches[patch_idx]
        x, y, best_z = patch.x, patch.y, patch.z

        scene = self._get_scene(str(vsi_path))

        # Calculate focus offset in microns (Target for the regression model)
        z_res_microns = scene.z_resolution * 1e6
        z_offset = float(best_z - z_level) * z_res_microns

        rect = (
            x,
            y,
            self.patch_size,
            self.patch_size,
        )
        try:
            # Efficient single-slice read from VSI
            block = scene.read_block(
                rect=rect,
                size=(self.patch_size, self.patch_size),
                slices=(z_level, z_level + 1),
            )

            if self.transform:
                image = self.transform(block)
            else:
                image = torch.from_numpy(block).permute(2, 0, 1).float() / 255.0

            return image, torch.tensor(z_offset, dtype=torch.float32)

        except Exception as e:
            print(f"Error reading {vsi_path} at {rect} z={z_level}: {e}")
            raise e
