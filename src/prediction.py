"""
Core prediction / evaluation logic.

Provides ``evaluate()`` — run inference for a single model on a dataloader
and persist a flat CSV of predictions. Fully explicit, no "smart" logic.
"""

from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn


def evaluate(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    save_path: str | Path,
    metadata: dict,
    dry_run: bool = False,
) -> pd.DataFrame:
    """Run inference and save results.

    Parameters
    ----------
    model : nn.Module
        Weighted model in eval mode.
    dataloader : DataLoader
        Data to predict on.
    save_path : path
        Where to save the CSV.
    metadata : dict
        Contextual columns to add to EVERY row (e.g. model_name, dataset, checkpoint).
    dry_run : bool, optional
        If True, only process 2 batches for verification.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    batch_dfs = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if dry_run and batch_idx >= 2:
                break
            images = batch["image"].to(device, non_blocking=True)
            targets = batch["target"].to(device, non_blocking=True)

            with torch.autocast(
                device_type="cuda" if device.type == "cuda" else "cpu",
                dtype=torch.bfloat16,
            ):
                preds = model(images)

            # Standardize for numpy/pandas
            preds_np = preds.detach().float().cpu().numpy().flatten()
            targets_np = targets.detach().float().cpu().numpy().flatten()

            # Construct batch dataframe
            batch_data = {**metadata}
            batch_data["pred"] = preds_np
            batch_data["target"] = targets_np

            # Add batch-specific metadata
            batch_meta = batch["metadata"]
            for key, val in batch_meta.items():
                batch_data[key] = val

            batch_dfs.append(pd.DataFrame(batch_data))

    df = pd.concat(batch_dfs, ignore_index=True)
    df.to_csv(save_path, index=False)
    return df
