"""
Run prediction for all ablation variants.
Fully hardcoded, no CLI, no magic.
"""

import sys
from pathlib import Path

import torch
from src import config
from src.prediction import evaluate
from src.utils.env import setup_environment
from src.models.architectures import MODEL_REGISTRY

# Hardcoded evaluation items: (model_key_in_registry, checkpoint_path, display_name)
MODEL_CHECKPOINTS = [
    ("rgb", "logs/rgb/best_weights.pt", "rgb"),
    ("dwt", "logs/dwt/best_weights.pt", "dwt"),
    ("rgb_fft", "logs/rgb_fft/best_weights.pt", "rgb_fft"),
    ("grayscale_fft", "logs/grayscale_fft/best_weights.pt", "grayscale_fft"),
    ("hematoxylin_fft", "logs/hematoxylin_fft/best_weights.pt", "hematoxylin_fft"),
    ("radial_profile", "logs/radial_profile/best_weights.pt", "radial_profile"),
]

# Hardcoded datasets to evaluate
DATASETS = ["jiang2018", "zstack_he", "zstack_ihc", "agnor_ome"]


def get_loader(dataset_name, train_cfg):
    """Explicit loader setup."""
    if dataset_name == "jiang2018":
        from src.datasets.jiang2018 import get_jiang2018_dataloaders

        _, loader = get_jiang2018_dataloaders(
            batch_size=train_cfg.batch_size, num_workers=train_cfg.num_workers
        )
        return loader
    elif dataset_name == "zstack_he":
        from src.datasets.zstack_he import get_test_loader

        return get_test_loader(train_cfg=train_cfg)
    elif dataset_name == "zstack_ihc":
        from src.datasets.zstack_ihc import get_test_loader

        return get_test_loader(train_cfg=train_cfg)
    elif dataset_name == "agnor_ome":
        from src.datasets.agnor_ome.loader import get_ome_test_loader

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
    dry_run = "--dry-run" in sys.argv
    if dry_run:
        print("!!! DRY RUN ENABLED !!!")

    # Explicitly set low worker count for stability but enough for performance
    train_cfg = config.TrainConfig()
    train_cfg.num_workers = 4

    for model_key, ckpt_path_str, display_name in MODEL_CHECKPOINTS:
        ckpt_path = Path(ckpt_path_str)
        if not ckpt_path.exists():
            print(f"Skipping {display_name}: Checkpoint not found at {ckpt_path}")
            continue

        print(f"\nEvaluating: {display_name}")

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
