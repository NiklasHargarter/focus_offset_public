"""
Train a model using K-Fold cross validation.
Each fold gets its own model, datamodule, and checkpoint.
"""

import argparse

from src.models.lightning_module import FocusOffsetRegressor
from src.dataset.vsi_datamodule import HEFoldDataModule
from src.train import train_one, setup_environment, NUM_WORKERS, PREFETCH_FACTOR


def main():
    parser = argparse.ArgumentParser(description="K-Fold Cross Validation Training")
    parser.add_argument("--folds", type=int, default=5, help="Number of folds")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--max_epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=10)
    args = parser.parse_args()

    setup_environment()

    print(
        f"Starting {args.folds}-Fold Cross Validation ("
        + ("DRY RUN" if args.dry_run else "FULL RUN")
        + ")..."
    )

    for i in range(args.folds):
        print("\n" + "=" * 40)
        print(f"FOLD {i + 1}/{args.folds}")
        print("=" * 40 + "\n")

        datamodule = HEFoldDataModule(
            fold_idx=i,
            num_folds=args.folds,
            dataset_name="ZStack_HE",
            batch_size=args.batch_size,
            num_workers=NUM_WORKERS,
            prefetch_factor=PREFETCH_FACTOR,
        )

        model = FocusOffsetRegressor(
            model_name="multimodal",
            learning_rate=5e-5,
            weight_decay=0.05,
        )

        train_one(
            datamodule,
            model,
            log_name=f"kfold/fold_{i}",
            max_epochs=args.max_epochs,
            patience=args.patience,
            dry_run=args.dry_run,
        )

    print("\n" + "=" * 40)
    print(f"{args.folds}-FOLD CROSS VALIDATION COMPLETED")
    print("=" * 40)


if __name__ == "__main__":
    main()
