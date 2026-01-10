from typing import Any, NamedTuple
from dataclasses import dataclass
from pathlib import Path
from functools import cached_property


class Patch(NamedTuple):
    """Spatial coordinate and optimal Z-level for a single patch."""

    x: int
    y: int
    z: int


@dataclass
class SlideMetadata:
    """Metadata for a single VSI slide."""

    name: str
    path: Path
    width: int
    height: int
    num_z: int
    patches: list[Patch]

    @cached_property
    def patch_count(self) -> int:
        return len(self.patches)

    @cached_property
    def total_samples(self) -> int:
        return self.patch_count * self.num_z


@dataclass
class PreprocessConfig:
    """Configuration used during preprocessing for traceability."""

    patch_size: int
    stride: int
    downscale_factor: int
    min_tissue_coverage: float
    dataset_name: str


@dataclass
class MasterIndex:
    """Consolidated index for all slides in a dataset."""

    file_registry: list[SlideMetadata]
    patch_size: int
    config_state: PreprocessConfig

    @cached_property
    def total_samples(self) -> int:
        return sum(slide.total_samples for slide in self.file_registry)


@dataclass
class ProcessedIndex:
    """Runtime index used by the dataset, filtered and with cumulative indices."""

    file_registry: list[SlideMetadata]
    cumulative_indices: list[int]
    patch_size: int

    @cached_property
    def total_samples(self) -> int:
        return sum(slide.total_samples for slide in self.file_registry)
