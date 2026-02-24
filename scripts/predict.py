"""CLI for running prediction on a dataset using a checkpoint."""

import argparse
import importlib

from src import config
from src.models.architectures import MODEL_REGISTRY
from src.prediction import evaluate
from src.utils.env import setup_environment


def main():
    parser = argparse.ArgumentParser(
        description="Run prediction on a dataset using a checkpoint.",
    )
    parser.add_argument(
        "checkpoint", help="Path to .pt file (created by this pipeline)."
    )
    parser.add_argument("--output_dir", default=None, help="Override output dir")
    train_cfg = config.TrainConfig()
    parser.add_argument("--batch_size", type=int, default=train_cfg.batch_size)
    parser.add_argument(
        "--dataset",
        default="zstack_he",
        choices=["zstack_he", "zstack_ihc", "agnor_ome", "jiang2018"],
        help="Dataset to evaluate on.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        choices=list(MODEL_REGISTRY.keys()),
        help="Manually specify model architecture if not in checkpoint.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    setup_environment()

    # Dynamically select DataLoader based on dataset
    train_cfg = config.TrainConfig(batch_size=args.batch_size)

    if args.dataset in ["zstack_he", "zstack_ihc"]:
        dataset_module = importlib.import_module(f"src.datasets.{args.dataset}")
        loader = dataset_module.get_test_loader(train_cfg=train_cfg)
    elif args.dataset == "agnor_ome":
        dataset_module = importlib.import_module(f"src.datasets.{args.dataset}")
        loader = dataset_module.get_dataloader(train_cfg=train_cfg)
    elif args.dataset == "jiang2018":
        dataset_module = importlib.import_module(f"src.datasets.{args.dataset}")
        # Jiang2018 returns (train_loader, test_loader)
        _, loader = dataset_module.get_jiang2018_dataloaders(
            batch_size=args.batch_size, num_workers=train_cfg.num_workers
        )
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    evaluate(
        dataloader=loader,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        dry_run=args.dry_run,
        dataset_name=args.dataset,
        model_name_override=args.model,
    )


if __name__ == "__main__":
    main()
