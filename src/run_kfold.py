import argparse
import os
import torch
import lightning as L
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
)
import warnings
from lightning.pytorch.loggers import TensorBoardLogger

# Import your codebase modules
from src.models.lightning_module import FocusOffsetRegressor
from src.dataset.vsi_datamodule import HEFoldDataModule

# Suppress harmless TorchInductor warning for FFT (Complex Operators)
# This is expected since Inductor falls back to eager mode for the first layer (FFT) on Blackwell/RTX 5090
warnings.filterwarnings(
    "ignore",
    message=".*Torchinductor does not support code generation for complex operators.*",
)


def main():
    parser = argparse.ArgumentParser(description="Run K-Fold Cross Validation")
    parser.add_argument("--folds", type=int, default=5, help="Number of folds")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Perform a quick check with 1 epoch and limited batches",
    )
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=24)
    parser.add_argument(
        "--patience", type=int, default=10, help="Early stopping patience"
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

        model = FocusOffsetRegressor(
            pretrained=True,
            use_transforms=True,  # Assuming multimodal is default for kfold
            learning_rate=5e-5,
            weight_decay=0.05,
            save_predictions=False,
        )

        # Optimization for RTX 5090: Kernel fusion and graph optimization
        # Note: The first few batches will be slow due to compilation overhead
        if not args.dry_run:
            print("Compiling backbone for maximum performance...")
            model.backbone = torch.compile(model.backbone)
        else:
            # High-fidelity dry-run: compile with a faster mode to catch errors
            print("Smoke Test: Compiling backbone in fast mode...")
            model.backbone = torch.compile(model.backbone, mode="reduce-overhead")

        # 3. Setup Callbacks & Logger
        log_name = f"focus_convnextv2_kfold_fold_{i}"
        if args.dry_run:
            log_name += "_smoke_test"

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
            "max_epochs": 2
            if args.dry_run
            else 60,  # At least 2 epochs to test transitions
            "precision": "bf16-mixed",  # Highly recommended for RTX 5090 / Blackwell
            "accelerator": "auto",
            "devices": 1,
            "logger": logger,
            "callbacks": callbacks,
            "log_every_n_steps": 5 if args.dry_run else args.log_interval,
            "gradient_clip_val": 1.0,  # Stabilize "jumpy" validation
        }

        if args.dry_run:
            # We only limit the batches, nothing else!
            trainer_kwargs["limit_train_batches"] = 20
            trainer_kwargs["limit_val_batches"] = 10

        trainer = L.Trainer(**trainer_kwargs)

        # 5. Run Training
        trainer.fit(model, datamodule=datamodule)

    print("\n" + "=" * 40)
    print("5-FOLD CROSS VALIDATION COMPLETED")
    print("=" * 40)


if __name__ == "__main__":
    main()
