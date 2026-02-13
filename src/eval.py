"""
Core evaluation module. Provides eval_one() for single-checkpoint evaluation
and a CLI for standalone use.
"""

import argparse
from pathlib import Path

import lightning as L
import pandas as pd
import torch

from src import config
from src.dataset import DATAMODULE_REGISTRY
from src.dataset.vsi_datamodule import VSIDataModule
from src.models.lightning_module import FocusOffsetRegressor
from src.train import setup_environment


def eval_one(
    datamodule: L.LightningDataModule,
    checkpoint_path: str | Path,
    label: str = "model",
    output_dir: str | Path | None = None,
    dry_run: bool = False,
) -> pd.DataFrame:
    """
    Evaluate a single checkpoint. Returns a DataFrame of predictions.

    Args:
        datamodule: Lightning DataModule providing test data.
        checkpoint_path: Path to the model checkpoint.
        label: Human-readable label for this evaluation (used in output naming).
        output_dir: If provided, save predictions CSV here.
        dry_run: If True, limit to 2 batches for smoke testing.

    Returns:
        DataFrame with columns: pred, target, error (plus metadata if available).
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Load model from checkpoint
    model = FocusOffsetRegressor.load_from_checkpoint(checkpoint_path)
    model.eval()

    # Run predictions
    trainer = L.Trainer(
        accelerator="auto",
        devices=1,
        precision="bf16-mixed",
        logger=False,
        limit_predict_batches=2 if dry_run else None,
    )

    predictions = trainer.predict(model, datamodule=datamodule)
    all_preds = torch.cat(predictions).squeeze()

    return _build_results_df(datamodule, all_preds, len(predictions), label, output_dir)


def _build_results_df(
    datamodule: L.LightningDataModule,
    all_preds: torch.Tensor,
    num_batches: int,
    label: str,
    output_dir: str | Path | None,
) -> pd.DataFrame:
    """Build a results DataFrame from predictions, targets, and metadata."""
    # Collect targets and metadata from the test dataloader
    all_targets = []
    all_metadata = []

    dataloader = datamodule.test_dataloader()

    for i, batch in enumerate(dataloader):
        if i >= num_batches:
            break

        all_targets.append(batch["target"])
        if "metadata" in batch:
            meta = batch["metadata"]
            for k in range(len(batch["target"])):
                all_metadata.append(
                    {
                        key: v[k].item() if isinstance(v[k], torch.Tensor) else v[k]
                        for key, v in meta.items()
                    }
                )

    all_targets = torch.cat(all_targets)

    # Build results
    results = {
        "pred": all_preds.float().numpy(),
        "target": all_targets.float().numpy(),
        "error": (all_preds - all_targets).abs().float().numpy(),
    }
    df = pd.DataFrame(results)

    # Append metadata columns if available
    if all_metadata:
        meta_df = pd.DataFrame(all_metadata)
        df = pd.concat([df, meta_df], axis=1)

    # Print summary
    mae = df["error"].mean()
    print(f"\n{'=' * 40}")
    print(f"  {label}")
    print(f"  Samples:  {len(df)}")
    print(f"  MAE:      {mae:.4f}")
    print(f"{'=' * 40}\n")

    # Save if requested
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        csv_path = output_dir / f"{label}_predictions.csv"
        df.to_csv(csv_path, index=False)
        print(f"Predictions saved to {csv_path}")

    return df


def main():
    """CLI for evaluating a single checkpoint."""
    parser = argparse.ArgumentParser(description="Evaluate a trained model checkpoint")
    parser.add_argument(
        "checkpoint",
        type=str,
        help="Path to model checkpoint (.ckpt)",
    )
    parser.add_argument(
        "--label",
        type=str,
        default=None,
        help="Label for this evaluation (default: inferred from checkpoint path)",
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

    # Infer label from checkpoint path if not provided
    # e.g. logs/rgb/version_0/checkpoints/best.ckpt -> "rgb"
    if args.label is None:
        ckpt = Path(args.checkpoint)
        args.label = f"{ckpt.parents[2].name}_{args.dataset}"

    dm_factory = DATAMODULE_REGISTRY[args.dataset]
    datamodule = dm_factory(batch_size=args.batch_size)

    eval_one(
        datamodule=datamodule,
        checkpoint_path=args.checkpoint,
        label=args.label,
        output_dir=args.output_dir,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
