"""CLI for training a single focus offset model."""

import argparse

from focus_offset import config
from focus_offset.models.architectures import MODEL_REGISTRY
from focus_offset.training import train_one
from focus_offset.utils.env import setup_environment


def main():
    parser = argparse.ArgumentParser(description="Train a single focus offset model")
    parser.add_argument(
        "--model",
        type=str,
        default="rgb",
        choices=list(MODEL_REGISTRY.keys()),
        help="Model variant to train",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="zstack_he",
        choices=["zstack_he", "zstack_ihc", "agnor_ome", "jiang2018"],
        help="Dataset module to load data from",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Quick smoke test with 2 epochs and limited batches",
    )
    train_cfg = config.TrainConfig()
    parser.add_argument("--batch_size", type=int, default=train_cfg.batch_size)
    parser.add_argument("--max_epochs", type=int, default=train_cfg.max_epochs)
    parser.add_argument("--patience", type=int, default=train_cfg.patience)
    parser.add_argument("--learning_rate", type=float, default=train_cfg.learning_rate)
    args = parser.parse_args()

    setup_environment()

    # Dynamically import the requested dataset module
    import importlib

    dataset_module = importlib.import_module(f"src.datasets.{args.dataset}")

    train_cfg = config.TrainConfig(batch_size=args.batch_size)

    # Some datasets have distinct loading logic (like AgNor which uses get_dataloader instead of get_dataloaders)
    if args.dataset in ["zstack_he", "zstack_ihc"]:
        train_loader, val_loader = dataset_module.get_dataloaders(train_cfg=train_cfg)
    elif args.dataset == "agnor_ome":
        # Usually AgNor is just test evaluation, but provided here for matching interface
        train_loader = dataset_module.get_dataloader(train_cfg=train_cfg)
        val_loader = train_loader
    else:
        train_loader, val_loader = dataset_module.get_jiang2018_dataloaders(
            batch_size=args.batch_size,
            num_workers=train_cfg.num_workers,
        )

    model = MODEL_REGISTRY[args.model]()

    train_one(
        model,
        train_loader,
        val_loader,
        log_name=f"{args.dataset}/{args.model}",
        max_epochs=args.max_epochs,
        patience=args.patience,
        dry_run=args.dry_run,
        learning_rate=args.learning_rate,
        weight_decay=train_cfg.weight_decay,
    )


if __name__ == "__main__":
    main()
