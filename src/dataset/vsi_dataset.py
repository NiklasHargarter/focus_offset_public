import torch
from torch.utils.data import Dataset
import pickle
import bisect
import slideio
from typing import Optional, Callable, Tuple, Any
import config
from src.utils.io_utils import suppress_stderr


class VSIDataset(Dataset):
    """
    Efficient Dataset for flattened Z-level VSI patches.
    Consumes the master index and filters based on split configuration.
    """

    def __init__(
        self,
        mode: str,
        dataset_name: str = config.DATASET_NAME,
        transform: Optional[Callable[[Any], torch.Tensor]] = None,
    ):
        self.mode = mode
        self.dataset_name = dataset_name
        self.transform = transform
        self.index_path = config.get_index_path(mode, dataset_name=dataset_name)

        if not self.index_path.exists():
            raise FileNotFoundError(
                f"Index not found at {self.index_path}. Run preprocess.py first."
            )

        with open(self.index_path, "rb") as f:
            self.index = pickle.load(f)

        self.file_registry = self.index["file_registry"]
        self.cumulative_indices = self.index["cumulative_indices"]
        self.total_samples = self.index["total_samples"]
        self.patch_size = self.index["patch_size"]

    def __len__(self) -> int:
        return self.total_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        file_idx = bisect.bisect_right(self.cumulative_indices, idx)

        if file_idx == 0:
            local_idx = idx
        else:
            local_idx = idx - self.cumulative_indices[file_idx - 1]

        file_meta = self.file_registry[file_idx]
        vsi_path = file_meta["path"]
        patches = file_meta["patches"]
        num_z = file_meta["num_z"]

        patch_idx = local_idx // num_z
        z_level = local_idx % num_z

        x, y, best_z = patches[patch_idx]

        if not hasattr(self, "slide_cache"):
            self.slide_cache = {}

        if vsi_path not in self.slide_cache:
            try:
                with suppress_stderr():
                    slide = slideio.open_slide(str(vsi_path), "VSI")
                scene = slide.get_scene(0)
                self.slide_cache[vsi_path] = (slide, scene)
            except Exception as e:
                raise RuntimeError(f"Failed to open {vsi_path}: {e}")

        _, scene = self.slide_cache[vsi_path]

        z_res_microns = scene.z_resolution * 1e6
        z_offset = float(best_z - z_level) * z_res_microns

        rect = (x, y, self.patch_size, self.patch_size)

        try:
            block = scene.read_block(rect=rect, slices=(z_level, z_level + 1))

            if self.transform:
                image = self.transform(block)
            else:
                image = torch.from_numpy(block).permute(2, 0, 1).float() / 255.0

            return image, torch.tensor(z_offset, dtype=torch.float32)

        except Exception as e:
            print(f"Error reading {vsi_path} at {rect} z={z_level}: {e}")
            raise e
