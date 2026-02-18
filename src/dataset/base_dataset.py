"""
Base dataset class that defines the canonical sample schema.

Every dataset in this project returns samples as::

    {
        "image":    Tensor (C, H, W),
        "target":   Tensor (scalar) — focus offset in µm,
        "metadata": SampleMetadata,
    }

Subclasses implement ``_get_sample(idx)`` and return whatever they have.
The base ``__getitem__`` fills in any missing metadata keys with safe
defaults so downstream code (eval CSVs, analysis scripts) can always
rely on a fixed column set.

To add a new metadata field:
    1. Add the key + type to ``SampleMetadata``
    2. Add the default to ``METADATA_DEFAULTS``
    That's it — datasets, eval CSVs, and analysis scripts all pick it up.
"""

from typing import Any, TypedDict

from torch.utils.data import Dataset


class SampleMetadata(TypedDict, total=False):
    """Canonical per-sample metadata schema.

    ``total=False`` means all keys are optional at construction time —
    the base class fills in anything a subclass doesn't provide.
    """

    filename: str  # slide / image filename
    x: int  # patch x coordinate (raw pixels)
    y: int  # patch y coordinate (raw pixels)
    z_level: int  # current z-slice index
    optimal_z: int  # sharpest z-slice index
    num_z: int  # total z-slices in this slide


# Default values — must have exactly the same keys as SampleMetadata.
METADATA_DEFAULTS: dict[str, Any] = {
    "filename": "unknown",
    "x": 0,
    "y": 0,
    "z_level": 0,
    "optimal_z": 0,
    "num_z": 1,
}


class BasePatchDataset(Dataset):
    """Abstract base for all patch-level datasets in this project.

    Subclasses must implement:
        ``_get_sample(idx) -> dict``  with at least ``image`` and ``target``.
            ``metadata`` is optional; missing keys get safe defaults.

        ``__len__() -> int``
    """

    def _get_sample(self, idx: int) -> dict[str, Any]:
        """Return a raw sample dict. Must contain 'image' and 'target'."""
        raise NotImplementedError

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Subclasses should implement _get_sample(idx). This base ensures metadata completeness."""
        sample = self._get_sample(idx)

        # Guarantee metadata exists with every canonical key
        if "metadata" not in sample:
            sample["metadata"] = {}

        meta = sample["metadata"]
        for key, default in METADATA_DEFAULTS.items():
            if key not in meta:
                meta[key] = default

        return sample

    @staticmethod
    def create_metadata(
        filename: str,
        x: int,
        y: int,
        z_level: int,
        optimal_z: int,
        num_z: int,
    ) -> SampleMetadata:
        """Helper to create a type-safe metadata dict."""
        return {
            "filename": filename,
            "x": x,
            "y": y,
            "z_level": z_level,
            "optimal_z": optimal_z,
            "num_z": num_z,
        }


class BaseGridDataset(BasePatchDataset):
    """
    Abstract base for datasets that map a flat index to a (slide, patch, z) grid.

    Handlers logic for:
      - Mapping global `idx` -> `(slide_idx, patch_idx, z_level)`
      - Retrieving slide metadata from a `ProcessedIndex`
      - Implementing `__len__`
    """

    def __init__(self, index_data: Any):
        """
        Args:
            index_data: Must have .cumulative_indices, .total_samples, .file_registry
        """
        self.index = index_data
        self.cumulative_indices = self.index.cumulative_indices
        self.total_samples = self.index.total_samples
        self.file_registry = self.index.file_registry

    def __len__(self) -> int:
        return self.total_samples

    def _get_grid_info(self, idx: int):
        """
        Resolve a global index to specific slide and patch coordinates.

        Returns:
            (file_meta, slide_idx, (x, y, best_z), z_level)
        """
        import bisect

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
