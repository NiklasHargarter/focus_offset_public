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
from src.dataset.vsi_datamodule import VSIDataModule

# Suppress harmless TorchInductor warning for FFT (Complex Operators)
# This is expected since Inductor falls back to eager mode for the first layer (FFT) on Blackwell/RTX 5090
warnings.filterwarnings(
    "ignore",
    message=".*Torchinductor does not support code generation for complex operators.*",
)


def run_training(mode_name: str, transform_mode: str, args):
    print("\n" + "=" * 40)
    print(f"RUNNING MODEL: {mode_name} (Mode: {transform_mode})")
    print("=" * 40 + "\n")

    # 1. Initialize DataModule
    datamodule = VSIDataModule(
        dataset_name="ZStack_HE",
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        patch_size=224,
        stride=448,
        downsample_factor=2,
        min_tissue_coverage=0.05,
        prefetch_factor=args.prefetch_factor,
        split_ratio=0.3,  # Standard split ratio
    )

    # 2. Initialize Model Components

    model = FocusOffsetRegressor(
        pretrained=False,
        transform_mode=transform_mode,
        learning_rate=2e-4,  # Increased slightly for scratch training
        weight_decay=0.05,
        save_predictions=True,
    )

    # Optimization for RTX 5090
    if not args.dry_run:
        print(f"Compiling {mode_name} backbone for maximum performance...")
        model.backbone = torch.compile(model.backbone)
    else:
        print(f"Smoke Test: Compiling {mode_name} backbone in fast mode...")
        model.backbone = torch.compile(model.backbone, mode="reduce-overhead")

    # 3. Setup Callbacks & Logger
    log_name = f"focus_convnextv2_{mode_name.lower().replace(' ', '_')}"
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
        "max_epochs": 2 if args.dry_run else args.max_epochs,
        "precision": "bf16-mixed",
        "accelerator": "auto",
        "devices": 1,
        "logger": logger,
        "callbacks": callbacks,
        "log_every_n_steps": 5 if args.dry_run else args.log_interval,
        "gradient_clip_val": 1.0,
    }

    if args.dry_run:
        trainer_kwargs["limit_train_batches"] = 20
        trainer_kwargs["limit_val_batches"] = 10

    trainer = L.Trainer(**trainer_kwargs)

    # 5. Run Training
    trainer.fit(model, datamodule=datamodule)


def main():
    parser = argparse.ArgumentParser(description="Run Multimodal and Baseline Models")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Perform a quick check with 1 epoch and limited batches",
    )
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=24)
    parser.add_argument("--max_epochs", type=int, default=20)
    parser.add_argument(
        "--patience", type=int, default=3, help="Early stopping patience"
    )
    parser.add_argument(
        "--log_interval", type=int, default=100, help="Log every N steps"
    )
    parser.add_argument(
        "--prefetch_factor", type=int, default=4, help="Batches to prefetch per worker"
    )
    args = parser.parse_args()

    # Best practices for performance & stability
    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    torch.set_float32_matmul_precision("medium")

    # Run all 4 ablation variants
    variants = [
        ("Baseline RGB", "none"),
        ("FFT Only", "fft"),
        ("Wavelets Only", "dwt"),
        ("Multimodal All", "all"),
    ]

    for name, mode in variants:
        run_training(name, mode, args)

    print("\n" + "=" * 40)
    print("ALL MODEL RUNS COMPLETED")
    print("=" * 40)


if __name__ == "__main__":
    main()
