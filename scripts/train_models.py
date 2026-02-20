"""
Train all ablation variants sequentially (ResNet-18 from scratch).
Each variant uses a different input domain (RGB, FFT, DWT, Multimodal).
"""

import argparse

from src.config import DatasetConfig, TrainConfig
from src.dataset.loader import get_holdout_dataloaders
from src.models.architectures import MODEL_REGISTRY
from src.training import train_one
from src.utils.env import setup_environment

# Only the fair-comparison ablation models (all ResNet-18, from scratch)
ABLATION_VARIANTS = ["rgb", "fft", "dwt", "multimodal"]


def main():
    parser = argparse.ArgumentParser(description="Train all ablation variants")
    parser.add_argument("--dry-run", action="store_true")
    train_cfg = TrainConfig()
    parser.add_argument("--batch_size", type=int, default=train_cfg.batch_size)
    parser.add_argument("--max_epochs", type=int, default=train_cfg.max_epochs)
    parser.add_argument("--patience", type=int, default=train_cfg.patience)
    parser.add_argument("--learning_rate", type=float, default=train_cfg.learning_rate)
    parser.add_argument("--weight_decay", type=float, default=train_cfg.weight_decay)
    args = parser.parse_args()

    setup_environment()

    for variant_name in ABLATION_VARIANTS:
        print("\n" + "=" * 40)
        print(f"TRAINING ABLATION: {variant_name}")
        print("=" * 40 + "\n")

        dataset_cfg = DatasetConfig()

        # Override default train config based on user arguments
        train_cfg.batch_size = args.batch_size
        train_cfg.max_epochs = args.max_epochs
        train_cfg.patience = args.patience
        train_cfg.learning_rate = args.learning_rate
        train_cfg.weight_decay = args.weight_decay

        train_loader, val_loader = get_holdout_dataloaders(
            dataset_cfg=dataset_cfg,
            train_cfg=train_cfg,
        )

        model = MODEL_REGISTRY[variant_name]()

        train_one(
            model,
            train_loader,
            val_loader,
            log_name=variant_name,
            max_epochs=args.max_epochs,
            patience=args.patience,
            dry_run=args.dry_run,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            scheduler_patience=max(1, args.patience // 2),
        )

    print("\n" + "=" * 40)
    print("ALL ABLATION VARIANTS COMPLETED")
    print("=" * 40)


if __name__ == "__main__":
    main()
