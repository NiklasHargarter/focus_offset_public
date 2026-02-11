"""
Train all ablation variants sequentially.
Each variant uses a different input domain (RGB, FFT, DWT, Multimodal).
"""

import argparse

from src.models.lightning_module import FocusOffsetRegressor
from src.dataset.vsi_datamodule import VSIDataModule
from src.models.architectures import MODEL_REGISTRY
from src.train import (
    train_one,
    setup_environment,
    NUM_WORKERS,
    PREFETCH_FACTOR,
)


def main():
    parser = argparse.ArgumentParser(description="Train all ablation variants")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--max_epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=5)
    args = parser.parse_args()

    setup_environment()

    for variant_name, _ in MODEL_REGISTRY.items():
        print("\n" + "=" * 40)
        print(f"TRAINING: {variant_name}")
        print("=" * 40 + "\n")

        datamodule = VSIDataModule(
            dataset_name="ZStack_HE",
            batch_size=args.batch_size,
            num_workers=NUM_WORKERS,
            prefetch_factor=PREFETCH_FACTOR,
        )

        model = FocusOffsetRegressor(
            model_name=variant_name,
            learning_rate=2e-4,
            weight_decay=0.05,
        )

        train_one(
            datamodule,
            model,
            log_name=variant_name,
            max_epochs=args.max_epochs,
            patience=args.patience,
            dry_run=args.dry_run,
        )

    print("\n" + "=" * 40)
    print("ALL VARIANTS COMPLETED")
    print("=" * 40)


if __name__ == "__main__":
    main()
