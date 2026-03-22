from dataclasses import dataclass
from pathlib import Path


@dataclass
class SyntheticConfig:
    """Configuration for synthetic data generation and relations."""

    # Path to the external VSI slide directory
    slide_dir: str = "/data/niklas/ZStack_HE"

    # Synthetic Generation
    patch_size_input: int = 256
    kernel_size: int = 63
    z_offset_steps: int = -10
    downsample: int = 2
    min_coverage: float = 0.5
    top_percent_laplacian: float = 0.2

    # Simulation Mode (Teacher Kernel Simulation)
    simulation_mode: bool = False
    simulation_radius: float = (
        10.0  # Controls the width (sigma/radius) of the teacher kernel
    )

    # Training Hyperparameters
    batch_size: int = 128
    max_epochs: int = 30
    lr: float = 1e-3
    groups: int = 3  # Depthwise=3, Full=1
    use_amp: bool = False

    # Execution Flags
    workers: int = 4
    dry_run: bool = False
    weight_init: str = "random"  # "uniform", "ident", "random"
    vis_interval: int = 5  # Save checkpoint every N epochs
    base_log_dir: str = "logs"

    @property
    def dataset_name(self) -> str:
        return Path(self.slide_dir).name

    @property
    def experiment_name(self) -> str:
        """Dynamic experiment name based on kernel size and offset."""
        mode_prefix = (
            f"sim_r{self.simulation_radius}"
            if self.simulation_mode
            else f"off{self.z_offset_steps}"
        )
        name = (
            f"conv_k{self.kernel_size}_{mode_prefix}_g{self.groups}_{self.weight_init}"
        )
        if self.dry_run:
            name += "_dry_run"
        return name

    @property
    def log_dir(self) -> Path:
        """Absolute path to the log directory for current experiment."""
        return Path(self.base_log_dir) / self.experiment_name

    @property
    def patch_size_target(self) -> int:
        """Relation: PatchIn - PatchOut = k - 1 => PatchOut = PatchIn - k + 1"""
        return self.patch_size_input - self.kernel_size + 1

    @property
    def split_path(self) -> Path:
        return Path("splits") / f"splits_{self.dataset_name}.json"

    @property
    def index_dir(self) -> Path:
        return Path("cache/synthetic_indices") / self.dataset_name
