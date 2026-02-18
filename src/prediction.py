"""
Core prediction / evaluation logic.

Provides ``evaluate()`` — run inference for a single checkpoint on a datamodule
and persist a flat CSV of predictions.

Imported by ``scripts/predict.py`` (CLI) and any other script that needs
programmatic evaluation.
"""

from pathlib import Path

import lightning as L
import pandas as pd
import torch


def evaluate(
    datamodule: L.LightningDataModule,
    checkpoint_path: str | Path,
    output_dir: str | Path | None = None,
    dry_run: bool = False,
) -> pd.DataFrame:
    """Run inference on the test set and save a predictions CSV.

    Columns
    -------
    model_name, dataset, checkpoint,          ← run context
    pred, target,                             ← core results
    filename, x, y, z_level, optimal_z, …     ← dataset metadata (variable)

    Parameters
    ----------
    datamodule : L.LightningDataModule
        Must expose a ``predict_dataloader`` (usually == test_dataloader).
    checkpoint_path : path
        Path to the single model checkpoint.
    output_dir : path, optional
        Where to write the CSV.  Defaults to the log folder inferred from
        the checkpoint path.
    dry_run : bool
        Limit to 2 batches for smoke testing.

    Returns
    -------
    pd.DataFrame
        The full predictions table (same content that was written to disk).
    """
    from src.models.lightning_module import FocusOffsetRegressor

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    trainer = L.Trainer(
        accelerator="auto",
        devices=1,
        precision="bf16-mixed",
        logger=False,
        limit_predict_batches=2 if dry_run else None,
    )

    print(f"Loading model from {checkpoint_path.name}...")
    model = FocusOffsetRegressor.load_from_checkpoint(checkpoint_path)
    model.eval()

    model_name = model.hparams.get("model_name", "unknown")
    outputs = trainer.predict(model, datamodule=datamodule)
    df = _outputs_to_df(outputs)

    dataset_name = getattr(datamodule, "dataset_name", "unknown")
    df.insert(0, "model_name", model_name)
    df.insert(1, "dataset", dataset_name)
    df.insert(2, "checkpoint", checkpoint_path.name)

    if output_dir is None:
        output_dir = _infer_log_dir(checkpoint_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_name = f"eval_{model_name}_{dataset_name}.csv"
    csv_path = output_dir / csv_name
    df.to_csv(csv_path, index=False)
    print(f"  → {csv_path}  ({len(df)} rows × {len(df.columns)} cols)\n")

    return df


def _outputs_to_df(predict_outputs: list[dict]) -> pd.DataFrame:
    """Flatten predict_step batch dicts into a per-sample DataFrame."""
    rows: list[dict] = []
    for batch in predict_outputs:
        preds = batch["pred"].squeeze(-1).float().cpu()
        targets = batch["target"].float().cpu()
        meta = batch.get("metadata", {})
        meta_keys = list(meta.keys())

        for i in range(len(preds)):
            row: dict = {"pred": preds[i].item(), "target": targets[i].item()}
            for key in meta_keys:
                vals = meta[key]
                row[key] = (
                    vals[i].item() if isinstance(vals[i], torch.Tensor) else vals[i]
                )
            rows.append(row)

    return pd.DataFrame(rows)


def _infer_log_dir(checkpoint_path: Path) -> Path:
    """``logs/rgb/version_0/checkpoints/best.ckpt`` → ``logs/rgb/version_0/``"""
    if checkpoint_path.parent.name == "checkpoints":
        return checkpoint_path.parent.parent
    return checkpoint_path.parent
