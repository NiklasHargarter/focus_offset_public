"""
Train all ablation variants sequentially (ResNet-18 from scratch).
Each variant uses a different input domain (RGB, FFT, DWT, Multimodal).
"""

import argparse

from src.config import TrainConfig
from src.models.architectures import MODEL_REGISTRY
from src.training import train_one
from src.utils.env import setup_environment

# Only the fair-comparison ablation models (all ResNet-18, from scratch)
ABLATION_VARIANTS = [
    # "rgb",
    # "dwt",
    "rgb_fft",
    "grayscale_fft",
    # "hematoxylin_fft",
    # "radial_profile",
]


def main():
    parser = argparse.ArgumentParser(description="Train all ablation variants")
    parser.add_argument("--dry-run", action="store_true")
    train_cfg = TrainConfig()
    parser.add_argument("--batch_size", type=int, default=train_cfg.batch_size)
    parser.add_argument("--max_epochs", type=int, default=train_cfg.max_epochs)
    parser.add_argument("--patience", type=int, default=train_cfg.patience)
    parser.add_argument("--learning_rate", type=float, default=train_cfg.learning_rate)
    parser.add_argument("--weight_decay", type=float, default=train_cfg.weight_decay)
    parser.add_argument(
        "--dataset",
        type=str,
        default="zstack_he",
        choices=["zstack_he", "zstack_ihc", "agnor_ome", "jiang2018"],
        help="Dataset module to load data from",
    )
    args = parser.parse_args()

    setup_environment()

    if args.dry_run:
        train_cfg.num_workers = 4
        print(f"Dry run active: limiting workers to {train_cfg.num_workers}")

    for variant_name in ABLATION_VARIANTS:
        print(f"\nTraining ablation: {variant_name}")

        import importlib

        dataset_module = importlib.import_module(f"src.datasets.{args.dataset}")

        if args.dataset in ["zstack_he", "zstack_ihc"]:
            train_loader, val_loader = dataset_module.get_dataloaders(
                train_cfg=train_cfg
            )
        elif args.dataset == "agnor_ome":
            # Usually AgNor is just test evaluation, but provided here for matching interface
            train_loader = dataset_module.get_dataloader(train_cfg=train_cfg)
            val_loader = train_loader
        else:
            train_loader, val_loader = dataset_module.get_jiang2018_dataloaders(
                batch_size=args.batch_size, num_workers=train_cfg.num_workers
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

    print("\nAll ablation variants completed.")


if __name__ == "__main__":
    main()
