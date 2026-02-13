"""
Ensemble evaluation module. Averages predictions from multiple checkpoints.
"""

import argparse
from pathlib import Path

import lightning as L
import pandas as pd
import torch

from src import config
from src.dataset import DATAMODULE_REGISTRY
from src.eval import _build_results_df
from src.models.lightning_module import FocusOffsetRegressor
from src.train import setup_environment


def eval_ensemble(
    datamodule: L.LightningDataModule,
    checkpoint_paths: list[str | Path],
    label: str = "ensemble",
    output_dir: str | Path | None = None,
    dry_run: bool = False,
) -> pd.DataFrame:
    """
    Evaluate an ensemble by averaging predictions from multiple checkpoints.

    Args:
        datamodule: Lightning DataModule providing test data.
        checkpoint_paths: Paths to model checkpoints.
        label: Human-readable label for this evaluation.
        output_dir: If provided, save predictions CSV here.
        dry_run: If True, limit to 2 batches for smoke testing.

    Returns:
        DataFrame with columns: pred, target, error (plus metadata if available).
    """
    if not checkpoint_paths:
        raise ValueError("At least one checkpoint path is required")

    checkpoint_paths = [Path(p) for p in checkpoint_paths]
    for p in checkpoint_paths:
        if not p.exists():
            raise FileNotFoundError(f"Checkpoint not found: {p}")

    trainer = L.Trainer(
        accelerator="auto",
        devices=1,
        precision="bf16-mixed",
        logger=False,
        limit_predict_batches=2 if dry_run else None,
    )

    # Collect predictions from each model
    all_model_preds = []
    num_batches = None
    for i, ckpt in enumerate(checkpoint_paths):
        print(f"  Running model {i + 1}/{len(checkpoint_paths)}: {ckpt.name}")
        model = FocusOffsetRegressor.load_from_checkpoint(ckpt)
        model.eval()
        predictions = trainer.predict(model, datamodule=datamodule)
        num_batches = len(predictions)
        all_model_preds.append(torch.cat(predictions).squeeze())

    # Average across models
    all_preds = torch.stack(all_model_preds).mean(dim=0)

    return _build_results_df(datamodule, all_preds, num_batches, label, output_dir)


def main():
    """CLI for ensemble evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate an ensemble of checkpoints")
    parser.add_argument(
        "checkpoints",
        type=str,
        nargs="+",
        help="Paths to model checkpoints (.ckpt)",
    )
    parser.add_argument(
        "--label",
        type=str,
        default="ensemble",
        help="Label for this evaluation",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="eval_results",
        help="Directory to save predictions CSV",
    )
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE)
    parser.add_argument(
        "--dataset",
        type=str,
        default="he",
        choices=list(DATAMODULE_REGISTRY.keys()),
        help="Dataset to evaluate on (default: he)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Quick smoke test")
    args = parser.parse_args()

    setup_environment()

    dm_factory = DATAMODULE_REGISTRY[args.dataset]
    datamodule = dm_factory(batch_size=args.batch_size)

    eval_ensemble(
        datamodule=datamodule,
        checkpoint_paths=args.checkpoints,
        label=args.label,
        output_dir=args.output_dir,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
