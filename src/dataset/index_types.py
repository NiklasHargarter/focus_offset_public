from dataclasses import dataclass

import numpy as np


@dataclass
class SlideMetadata:
    """Metadata for a single VSI slide (Coordinates are in RAW 60x resolution)."""

    name: str
    num_z: int
    patches: np.ndarray  # Shape (N, 3) -> [raw_x, raw_y, best_z]

    @property
    def patch_count(self) -> int:
        return self.patches.shape[0]

    @property
    def total_samples(self) -> int:
        return self.patch_count * self.num_z


@dataclass
class PreprocessConfig:
    """Configuration used during preprocessing for traceability."""

    patch_size: int  # ViT output size, always 224
    stride: int  # Step size in raw image coordinates (e.g. 448)
    min_tissue_coverage: float
    dataset_name: str
    downsample_factor: int = 1  # Raw extent per patch = patch_size × ds


@dataclass
class MasterIndex:
    """Consolidated index for all slides in a dataset."""

    file_registry: list[SlideMetadata]
    config_state: PreprocessConfig

    @property
    def total_patches(self) -> int:
        return sum(slide.patch_count for slide in self.file_registry)

    @property
    def total_samples(self) -> int:
        return sum(slide.total_samples for slide in self.file_registry)


@dataclass
class ProcessedIndex:
    """Runtime index used by the dataset, filtered and with cumulative indices."""

    file_registry: list[SlideMetadata]
    cumulative_indices: np.ndarray  # Shape (N,)
    patch_size: int
    downsample_factor: int = 1
    dataset_name: str = ""

    @property
    def total_patches(self) -> int:
        return sum(slide.patch_count for slide in self.file_registry)

    @property
    def total_samples(self) -> int:
        return self.cumulative_indices[-1] if len(self.cumulative_indices) > 0 else 0
