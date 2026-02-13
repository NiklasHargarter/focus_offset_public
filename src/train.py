"""
Core training module. Provides train_one() for single-model training
and a CLI for standalone use.
"""

import argparse
import os
import warnings

import lightning as L
import torch
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import CSVLogger

from src import config
from src.dataset.vsi_datamodule import VSIDataModule
from src.models.architectures import MODEL_REGISTRY
from src.models.lightning_module import FocusOffsetRegressor

# Suppress harmless TorchInductor warning for FFT (Complex Operators)
warnings.filterwarnings(
    "ignore",
    message=".*Torchinductor does not support code generation for complex operators.*",
)


def setup_environment():
    """Set environment variables and torch defaults for optimal training."""
    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    torch.set_float32_matmul_precision("medium")


def train_one(
    datamodule: L.LightningDataModule,
    model: L.LightningModule,
    log_name: str,
    *,
    max_epochs: int = 50,
    patience: int = 5,
    dry_run: bool = False,
    seed: int = 42,
) -> L.Trainer:
    """
    Train a single model. Returns the trainer for post-training inspection.

    Args:
        datamodule: Lightning DataModule providing train/val/test splits.
        model: Lightning Module to train.
        log_name: Name for the logger directory (under logs/).
        max_epochs: Maximum training epochs.
        patience: Early stopping patience (epochs without val_loss improvement).
        dry_run: If True, run 2 epochs with limited batches for smoke testing.
        seed: Random seed for reproducibility.
    """
    L.seed_everything(seed, workers=True)

    # Logger
    log_dir = "logs_smoke_test" if dry_run else "logs"
    logger = CSVLogger(log_dir, name=log_name)

    # Callbacks
    callbacks = [
        ModelCheckpoint(
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            filename="best-e{epoch:02d}-vl{val_loss:.4f}",
            auto_insert_metric_name=False,
        ),
        EarlyStopping(monitor="val_loss", patience=patience, mode="min"),
        LearningRateMonitor(logging_interval="step"),
    ]

    # Trainer
    trainer_kwargs = {
        "max_epochs": 2 if dry_run else max_epochs,
        "precision": "bf16-mixed",
        "accelerator": "auto",
        "devices": 1,
        "logger": logger,
        "callbacks": callbacks,
        "log_every_n_steps": 5 if dry_run else config.LOG_EVERY_N_STEPS,
        "gradient_clip_val": 1.0,
        "deterministic": True,
    }

    if dry_run:
        trainer_kwargs["limit_train_batches"] = 20
        trainer_kwargs["limit_val_batches"] = 10

    trainer = L.Trainer(**trainer_kwargs)
    trainer.fit(model, datamodule=datamodule)

    return trainer


def main():
    """CLI for training a single model."""
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
