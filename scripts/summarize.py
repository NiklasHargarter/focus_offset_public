"""
Summarize evaluation metrics from a prediction CSV.

Usage:
    python scripts/summarize.py logs/model/version_0/eval_model_dataset.csv
"""

import argparse
import sys
from pathlib import Path

import pandas as pd


def print_summary(df: pd.DataFrame, title: str) -> None:
    """Print standard evaluation metrics."""
    if "pred" not in df.columns or "target" not in df.columns:
        print(
            f"Error: CSV must contain 'pred' and 'target' columns. Found: {df.columns.tolist()}"
        )
        return

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
        print(f"  Slides:    {df['filename'].nunique()}")
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
