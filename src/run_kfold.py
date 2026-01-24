import argparse
import os
import torch
import lightning as L
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
)
from lightning.pytorch.loggers import TensorBoardLogger
import warnings

# Import your codebase modules
from src.models.lightning_module import FocusOffsetRegressor
from src.models.architectures import ConvNeXtV2FocusRegressor
from src.dataset.vsi_datamodule import HEFoldDataModule


def main():
    parser = argparse.ArgumentParser(description="Run K-Fold Cross Validation")
    parser.add_argument("--folds", type=int, default=5, help="Number of folds")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Perform a quick check with 1 epoch and limited batches",
    )
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=os.cpu_count())
    parser.add_argument(
        "--patience", type=int, default=5, help="Early stopping patience"
    )
    parser.add_argument(
        "--log_interval", type=int, default=1000, help="Log every N steps"
    )
    parser.add_argument(
        "--prefetch_factor", type=int, default=4, help="Batches to prefetch per worker"
    )
    args = parser.parse_args()

    # Best practices for performance & stability
    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    torch.set_float32_matmul_precision("medium")

    print(
        f"Starting {args.folds}-Fold Cross Validation ("
        + ("DRY RUN" if args.dry_run else "FULL RUN")
        + ")..."
    )

    for i in range(args.folds):
        print("\n" + "=" * 40)
        print(f"RUNNING FOLD {i + 1}/{args.folds} (Index {i})")
        print("=" * 40 + "\n")

        # 1. Initialize DataModule
        datamodule = HEFoldDataModule(
            fold_idx=i,
            num_folds=args.folds,
            dataset_name="ZStack_HE",
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            patch_size=224,
            stride=448,
            downsample_factor=2,
            min_tissue_coverage=0.05,
            prefetch_factor=args.prefetch_factor,
        )

        # 2. Initialize Model Components
        backbone = ConvNeXtV2FocusRegressor(version="tiny", pretrained=True)

        model = FocusOffsetRegressor(
            backbone=backbone,
            learning_rate=1e-4,
            weight_decay=0.05,
            save_predictions=False,  # Disable heavy CSV logging for CV speed
        )

        # Optimization for RTX 5090: Kernel fusion and graph optimization
        # Note: The first few batches will be slow due to compilation overhead
        if not args.dry_run:
            print("Compiling model for maximum performance...")
            model = torch.compile(model)

        # 3. Setup Callbacks & Logger
        log_name = f"focus_convnextv2_kfold_fold_{i}"
        if args.dry_run:
            log_name += "_dryrun"

        logger = TensorBoardLogger("logs", name=log_name)

        callbacks = [
            ModelCheckpoint(
                monitor="val_loss",
                mode="min",
                save_top_k=1,
                filename="best_model-{epoch:02d}-{val_loss:.4f}",
            ),
            EarlyStopping(monitor="val_loss", patience=args.patience, mode="min"),
            LearningRateMonitor(logging_interval="step"),
        ]

        # 4. Initialize Trainer
        trainer_kwargs = {
            "max_epochs": 1 if args.dry_run else 100,
            "precision": "bf16-mixed",  # Highly recommended for RTX 5090 / Blackwell
            "accelerator": "auto",
            "devices": 1,
            "logger": logger,
            "callbacks": callbacks,
            "log_every_n_steps": 10 if args.dry_run else args.log_interval,
        }

        if args.dry_run:
            trainer_kwargs["limit_train_batches"] = 100
            trainer_kwargs["limit_val_batches"] = 50

        trainer = L.Trainer(**trainer_kwargs)

        # 5. Run Training
        trainer.fit(model, datamodule=datamodule)

    print("\n" + "=" * 40)
    print("5-FOLD CROSS VALIDATION COMPLETED")
    print("=" * 40)


if __name__ == "__main__":
    main()
