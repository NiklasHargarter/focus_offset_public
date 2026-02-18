"""CLI for training a single focus offset model."""

import argparse

from src import config
from src.dataset.vsi_datamodule import VSIDataModule
from src.models.architectures import MODEL_REGISTRY
from src.models.lightning_module import FocusOffsetRegressor
from src.training import train_one
from src.utils.env import setup_environment


def main():
    parser = argparse.ArgumentParser(description="Train a single focus offset model")
    parser.add_argument(
        "--model",
        type=str,
        default="multimodal",
        choices=list(MODEL_REGISTRY.keys()),
        help="Model variant to train",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Quick smoke test with 2 epochs and limited batches",
    )
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--max_epochs", type=int, default=config.MAX_EPOCHS)
    parser.add_argument("--patience", type=int, default=config.PATIENCE)
    parser.add_argument("--learning_rate", type=float, default=config.LEARNING_RATE)
    args = parser.parse_args()

    setup_environment()

    datamodule = VSIDataModule(
        batch_size=args.batch_size,
    )

    model = FocusOffsetRegressor(
        model_name=args.model,
        learning_rate=args.learning_rate,
        weight_decay=config.WEIGHT_DECAY,
    )

    train_one(
        datamodule,
        model,
        log_name=args.model,
        max_epochs=args.max_epochs,
        patience=args.patience,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
