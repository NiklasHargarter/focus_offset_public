"""
Train focus-offset kernels for comparison across datasets and offset directions.

Trains 4 independent models:
  HE +10, HE -10, IHC +10, IHC -10

All runs are saved under a single timestamped suite directory for easy comparison.
"""

import argparse
import time
from pathlib import Path

import torch

from synthetic_data.config import SyntheticConfig
from synthetic_data.dataset import get_synthetic_dataloaders
from synthetic_data.model import SyntheticConvModel
from synthetic_data.scripts.train_synthetic import train_synthetic
from focus_offset.utils.env import setup_environment


DATASETS = {
    "ZStack_HE": "/data/niklas/ZStack_HE",
    "ZStack_IHC": "/data/niklas/ZStack_IHC",
}

OFFSETS = [+10, -10]


def _ensure_indices(config: SyntheticConfig) -> None:
    """Build synthetic indices if they don't exist yet."""
    parquet = config.index_dir / "train_synthetic.parquet"
    if parquet.exists():
        return

    print(f"Synthetic indices missing for {config.dataset_name}. Building...")
    from synthetic_data.scripts.prep_vsi_synthetic import index_vsi_synthetic_dataset

    config.index_dir.mkdir(parents=True, exist_ok=True)
    index_vsi_synthetic_dataset(
        slide_dir=Path(config.slide_dir),
        index_dir=config.index_dir,
        split_path=config.split_path,
        patch_size=config.patch_size_input,
        downsample=config.downsample,
        min_coverage=config.min_coverage,
        workers=config.workers,
        dry_run=config.dry_run,
    )


def run_kernel_comparison(dry_run: bool = False) -> None:
    setup_environment()

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    suite_name = f"comparison_{timestamp}"
    if dry_run:
        suite_name += "_dry_run"

    suite_dir = Path("logs") / suite_name
    suite_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"=== Kernel Comparison Suite: {suite_name} ===")
    print(f"Datasets: {list(DATASETS.keys())}")
    print(f"Offsets:  {OFFSETS}")
    print(f"Device:   {device}")
    print(f"Total runs: {len(DATASETS) * len(OFFSETS)}\n")

    for dataset_name, slide_dir in DATASETS.items():
        for offset in OFFSETS:
            config = SyntheticConfig(
                slide_dir=slide_dir,
                z_offset_steps=offset,
                dry_run=dry_run,
            )
            config.base_log_dir = str(suite_dir)

            if dry_run:
                config.max_epochs = 2

            _ensure_indices(config)

            print(f"\n>>> {dataset_name} offset={offset:+d}")
            print(f"    experiment: {config.experiment_name}")
            print(f"    log_dir:    {config.log_dir}")

            train_loader, val_loader = get_synthetic_dataloaders(
                config=config, num_workers=config.workers
            )

            model = SyntheticConvModel(
                kernel_size=config.kernel_size,
                groups=config.groups,
                weight_init=config.weight_init,
            )

            train_synthetic(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                log_name=config.experiment_name,
                max_epochs=config.max_epochs,
                learning_rate=config.lr,
                device=device,
                dry_run=config.dry_run,
                config=config,
            )

    print("\n=== Comparison Suite Finished ===")
    print(f"Results: {suite_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Train focus-offset kernels for HE/IHC comparison."
    )
    parser.add_argument(
        "--dry-run",
        dest="dry_run",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Short run for testing (2 epochs, limited batches).",
    )
    args = parser.parse_args()
    run_kernel_comparison(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
