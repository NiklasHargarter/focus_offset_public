"""CLI for running prediction on a dataset using a checkpoint."""

import argparse

from src.config import TrainConfig
from src.dataset.jiang2018 import get_jiang2018_dataloaders
from src.dataset.loader import get_agnor_dataloader, get_holdout_dataloaders
from src.prediction import evaluate
from src.utils.env import setup_environment


def main():
    parser = argparse.ArgumentParser(
        description="Run prediction on a dataset using a checkpoint.",
    )
    parser.add_argument(
        "checkpoint", help="Path to .ckpt file or .pt (if created by this pipeline)."
    )
    parser.add_argument("--output_dir", default=None, help="Override output dir")
    parser.add_argument("--batch_size", type=int, default=TrainConfig().batch_size)
    parser.add_argument(
        "--dataset",
        default="he",
        choices=["he", "agnor", "jiang2018"],
        help="Dataset to evaluate on.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    setup_environment()

    # Select DataLoader
    if args.dataset == "he":
        # For HE we use the validation set of the holdout split
        # We discard the train loader
        _, loader = get_holdout_dataloaders(
            batch_size=args.batch_size,
        )
    elif args.dataset == "agnor":
        loader = get_agnor_dataloader(
            batch_size=args.batch_size,
        )
    elif args.dataset == "jiang2018":
        # For Jiang2018 we usually want the test set (combined same/diff protocol)
        _, loader = get_jiang2018_dataloaders(
            batch_size=args.batch_size,
        )
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    evaluate(
        dataloader=loader,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        dry_run=args.dry_run,
        dataset_name=args.dataset,
    )


if __name__ == "__main__":
    main()
