"""
Core training logic.

Provides ``train_one()`` — the single entry-point for fitting a Lightning model.
Imported by the CLI scripts in ``scripts/``.
"""

import warnings

import lightning as L
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import CSVLogger

from src import config


warnings.filterwarnings(
    "ignore",
    message=".*Torchinductor does not support code generation for complex operators.*",
)


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

    log_dir = "logs_smoke_test" if dry_run else "logs"
    logger = CSVLogger(log_dir, name=log_name)

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
