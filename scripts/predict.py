"""CLI for running prediction on a dataset using a checkpoint."""

import argparse

from src import config
from src.dataset import DATAMODULE_REGISTRY
from src.prediction import evaluate
from src.utils.env import setup_environment


def main():
    parser = argparse.ArgumentParser(
        description="Run prediction on a dataset using a checkpoint.",
    )
    parser.add_argument("checkpoint", help="Path to .ckpt file.")
    parser.add_argument("--output_dir", default=None, help="Override output dir")
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE)
    parser.add_argument(
        "--dataset",
        default="he",
        choices=list(DATAMODULE_REGISTRY.keys()),
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    setup_environment()

    dm_cls = DATAMODULE_REGISTRY[args.dataset]
    datamodule = dm_cls(batch_size=args.batch_size)

    evaluate(
        datamodule=datamodule,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
