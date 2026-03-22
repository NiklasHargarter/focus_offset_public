import time
from pathlib import Path
from synthetic_data.config import SyntheticConfig
from synthetic_data.dataset import get_synthetic_dataloaders
import torch
from synthetic_data.scripts.visualize_results_sim import (
    create_kernel_progression_for_experiment,
    create_suite_overview,
)


def run_experiment_suite(dry_run: bool = False):
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    suite_name = f"suite_{timestamp}"
    if dry_run:
        suite_name += "_dry_run"

    suite_dir = Path("logs") / suite_name
    suite_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    group_options = [3]  # [3, 1]
    # `ident` is unstable for this setup and `uniform`/`random` converge similarly.
    # Keep one representative init for faster iteration.
    init_options = ["random"]

    config = SyntheticConfig()
    config.base_log_dir = str(suite_dir)
    config.dry_run = dry_run
    # We want epoch-by-epoch kernel progression for initialization comparison.
    config.vis_interval = 1
    if dry_run:
        config.max_epochs = 2
        config.vis_interval = 1

    print(f"Starting Experiment Suite (Sequential): {suite_name}")
    print(f"Mode: {'DRY RUN' if config.dry_run else 'FULL RUN'}")
    print(f"Combinations: {len(group_options) * len(init_options)}")
    print(f"Using {config.workers} workers for data loading.")

    # In simulation mode, validation is kernel distance — no val data needed
    if config.simulation_mode:
        train_loader, _ = get_synthetic_dataloaders(
            config=config, num_workers=config.workers
        )
        val_loader = None
    else:
        train_loader, val_loader = get_synthetic_dataloaders(
            config=config, num_workers=config.workers
        )

    print(f"Train samples: {len(train_loader.dataset)}")
    if val_loader is not None:
        print(f"Val samples: {len(val_loader.dataset)}")

    for groups in group_options:
        for weight_init in init_options:
            # Update config for current run
            config.groups = groups
            config.weight_init = weight_init

            print(f"\n>>> Running Config: groups={groups}, init={weight_init}")

            from synthetic_data.model import SyntheticConvModel
            from synthetic_data.scripts.train_synthetic import train_synthetic

            model = SyntheticConvModel(
                kernel_size=config.kernel_size,
                groups=config.groups,
                weight_init=config.weight_init,
            ).to(device)

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

            create_kernel_progression_for_experiment(config.log_dir)

    print("\nExperiment Suite Finished.")
    print(f"Results saved in: {suite_dir}")
    create_suite_overview(suite_dir)
    print(f"Run: python -m synthetic_data.scripts.visualize_results_sim {suite_dir}")


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dry-run",
        dest="dry_run",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable dry run mode (also supports --no-dry-run).",
    )
    args = parser.parse_args()

    run_experiment_suite(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
