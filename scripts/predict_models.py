"""
Run prediction for all ablation variants.
Fully hardcoded, no CLI, no magic.
"""

import argparse
from pathlib import Path

import torch

from src import config
from src.datasets.agnor_ome.loader import get_ome_test_loader
from src.datasets.jiang2018 import (
    get_jiang2018_dataloaders,
    get_jiang2018_test_loaders,
)
from src.datasets.zstack_he import get_test_loader as get_he_test_loader
from src.datasets.zstack_ihc import get_test_loader as get_ihc_test_loader
from src.models.architectures import MODEL_REGISTRY
from src.prediction import evaluate
from src.utils.env import setup_environment

# Hardcoded datasets to evaluate
DATASETS = ["jiang2018_same", "jiang2018_diff", "zstack_he", "zstack_ihc", "agnor_ome"]

# Datasets the models were trained on
TRAIN_DATASETS = ["jiang2018", "zstack_he"]

# Models to evaluate
MODELS = [
    "rgb",
    "dwt",
    "rgb_fft",
    "grayscale_fft",
    "fourier_domain",
    "two_domain",
]


def get_loader(dataset_name, train_cfg):
    """Explicit loader setup."""
    if dataset_name == "jiang2018":
        _, loader = get_jiang2018_dataloaders(
            batch_size=train_cfg.batch_size, num_workers=train_cfg.num_workers
        )
        return loader
    elif dataset_name == "jiang2018_same":
        loaders = get_jiang2018_test_loaders(
            batch_size=train_cfg.batch_size,
            num_workers=train_cfg.num_workers,
            protocol="same",
        )
        return loaders["same"]
    elif dataset_name == "jiang2018_diff":
        loaders = get_jiang2018_test_loaders(
            batch_size=train_cfg.batch_size,
            num_workers=train_cfg.num_workers,
            protocol="diff",
        )
        return loaders["diff"]
    elif dataset_name == "zstack_he":
        return get_he_test_loader(train_cfg=train_cfg)
    elif dataset_name == "zstack_ihc":
        return get_ihc_test_loader(train_cfg=train_cfg)
    elif dataset_name == "agnor_ome":
        return get_ome_test_loader(
            test_parquet=Path("cache/AgNor/s224_ds1_cov020/test.parquet"),
            slide_dir=Path("/data/niklas/AgNor_OME/raws"),
            downsample=1,
            train_cfg=train_cfg,
        )
    else:
        raise ValueError(f"Add explicit loader for {dataset_name} if needed.")


def main():
    setup_environment()

    parser = argparse.ArgumentParser(description="Evaluate models")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    dry_run = args.dry_run
    if dry_run:
        print("!!! DRY RUN ENABLED !!!")

    # Explicitly set low worker count for stability but enough for performance
    train_cfg = config.TrainConfig()
    train_cfg.num_workers = 16

    for train_dataset in TRAIN_DATASETS:
        print("\n=======================================================")
        print(f"=== Processing models trained on {train_dataset} ===")
        print("=======================================================")

        for model_key in MODELS:
            display_name = model_key
            log_dir = Path(f"logs/{train_dataset}/{model_key}")

            # Find the single best checkpoint file avoiding redundant names
            ckpt_files = list(log_dir.glob("best_e*vloss*.pt"))
            if not ckpt_files:
                print(f"Skipping {display_name}: No checkpoint found in {log_dir}")
                continue

            ckpt_path = ckpt_files[0]
            print(f"\nEvaluating: {display_name} trained on {train_dataset}")

            # Load raw state_dict (strict contract with training)
            model = MODEL_REGISTRY[model_key]()
            model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))

            for dataset_name in DATASETS:
                print(f"--- Dataset: {dataset_name} ---")
                loader = get_loader(dataset_name, train_cfg)

                save_path = ckpt_path.parent / f"eval_{display_name}_{dataset_name}.csv"

                evaluate(
                    model=model,
                    dataloader=loader,
                    save_path=save_path,
                    metadata={
                        "model_name": display_name,
                        "dataset": dataset_name,
                        "checkpoint": ckpt_path.name,
                    },
                    dry_run=dry_run,
                )

    print("\nDONE.")


if __name__ == "__main__":
    main()
