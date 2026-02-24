"""
Summarize evaluation metrics from a prediction CSV.

Usage:
    python scripts/summarize.py logs/model/version_0/eval_model_dataset.csv
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def aggregate_jiang_predictions(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-tile predictions into per-image predictions using Jiang protocol.

    Protocol: Discard 10 outliers (5 highest, 5 lowest) and average the rest.
    Assumes ~20 tiles per image.
    """
    if "filename" not in df.columns:
        return df

    agg_rows = []
    # Group by filename to aggregate tiles belonging to the same image
    for filename, group in df.groupby("filename"):
        preds = group["pred"].values
        if len(preds) > 10:
            # Calculate median and distance from median
            median = np.median(preds)
            diffs = np.abs(preds - median)

            # Discard 10 predictions with the largest distance from median
            # argsort sorts ascending, so [-10:] gives indices of largest diffs
            outlier_indices = np.argsort(diffs)[-10:]

            # Keep only non-outliers
            mask = np.ones(len(preds), dtype=bool)
            mask[outlier_indices] = False
            final_pred = preds[mask].mean()
        else:
            final_pred = preds.mean()

        # Build aggregated row
        row = {
            "filename": filename,
            "pred": final_pred,
            "target": group["target"].iloc[0],
        }
        # Carry over other metadata if present
        for col in ["model_name", "dataset", "checkpoint", "split"]:
            if col in group.columns:
                row[col] = group[col].iloc[0]
        agg_rows.append(row)

    return pd.DataFrame(agg_rows)


def print_summary(df: pd.DataFrame, title: str) -> None:
    """Print standard evaluation metrics."""
    if "pred" not in df.columns or "target" not in df.columns:
        print(
            f"Error: CSV must contain 'pred' and 'target' columns. Found: {df.columns.tolist()}"
        )
        return

    # Jiang Protocol Detection
    is_jiang = False
    if "dataset" in df.columns:
        is_jiang = df["dataset"].str.contains("Jiang2018", case=False).any()

    if is_jiang or ("x" in df.columns and "segment" in df.columns):
        print(f"  [Protocol] Jiang2018 detected. Aggregating {len(df)} tiles...")
        df = aggregate_jiang_predictions(df)

    # Calculate errors on the fly
    error = df["pred"] - df["target"]
    abs_error = error.abs()

    mae = abs_error.mean()
    median = abs_error.median()
    max_error = abs_error.max()

    print(f"\n{'=' * 50}")
    print(f"  {title}")
    print(f"  Samples:   {len(df):,}")
    print(f"  MAE:       {mae:.4f} µm")
    print(f"  Median AE: {median:.4f} µm")
    print(f"  Max |err|: {max_error:.4f} µm")

    if "filename" in df.columns:
        print(f"  Images:    {df['filename'].nunique()}")
    print(f"{'=' * 50}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Calculate metrics from prediction CSV."
    )
    parser.add_argument("csv_path", type=Path, help="Path to the prediction CSV file.")
    args = parser.parse_args()

    if not args.csv_path.exists():
        print(f"Error: File not found: {args.csv_path}")
        sys.exit(1)

    print(f"Reading {args.csv_path}...")
    df = pd.read_csv(args.csv_path)

    # Try to infer a nice title from filename
    # e.g. eval_rgb_Jiang2018.csv -> rgb on Jiang2018
    name = args.csv_path.stem
    if name.startswith("eval_"):
        parts = name.split("_")
        if len(parts) >= 3:
            model = parts[1]
            dataset = "_".join(parts[2:])
            title = f"{model} on {dataset}"
        else:
            title = name
    else:
        title = name

    print_summary(df, title)


if __name__ == "__main__":
    main()
