"""
Train all ablation variants sequentially (ResNet-18 from scratch).
Each variant uses a different input domain (RGB, FFT, DWT, Multimodal).
The production ConvNeXt model is trained separately via train_ensemble.py.
"""

import argparse

from src import config
from src.dataset.vsi_datamodule import VSIDataModule
from src.models.lightning_module import FocusOffsetRegressor
from src.train import setup_environment, train_one

# Only the fair-comparison ablation models (all ResNet-18, from scratch)
ABLATION_VARIANTS = ["rgb", "fft", "dwt", "multimodal"]


def main():
    parser = argparse.ArgumentParser(description="Train all ablation variants")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--max_epochs", type=int, default=config.MAX_EPOCHS)
    parser.add_argument("--patience", type=int, default=config.PATIENCE)
    parser.add_argument("--learning_rate", type=float, default=config.LEARNING_RATE)
    parser.add_argument("--weight_decay", type=float, default=config.WEIGHT_DECAY)
    args = parser.parse_args()

    setup_environment()

    for variant_name in ABLATION_VARIANTS:
        print("\n" + "=" * 40)
        print(f"TRAINING ABLATION: {variant_name}")
        print("=" * 40 + "\n")

        datamodule = VSIDataModule(
            batch_size=args.batch_size,
        )

        model = FocusOffsetRegressor(
            model_name=variant_name,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            scheduler_patience=max(1, args.patience // 2),
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
    print("ALL ABLATION VARIANTS COMPLETED")
    print("=" * 40)


if __name__ == "__main__":
    main()
