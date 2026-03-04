"""
Data loading for training & eval CSVs.

Training CSVs: ``logs/{model}/version_N/metrics.csv``  (Lightning CSVLogger)
Eval CSVs:     ``logs/{model}/version_N/eval_*.csv``   (from focus_offset.eval)
"""

from pathlib import Path

import pandas as pd


def load_training_curves(
    log_dir: str | Path = "logs",
) -> pd.DataFrame:
    """Read all ``metrics.csv`` files under *log_dir* and return epoch-level
    training curves with a ``model`` column.

    Expects structure: logs/{dataset}/{model}/metrics.csv

    Lightning's CSVLogger writes train and val epoch metrics on separate rows.
    This function groups by epoch and collapses them into one row per epoch.
    """
    frames: list[pd.DataFrame] = []
    # Search for exactly metrics.csv in expected depth, or just rglob and parse strictly
    for csv in sorted(Path(log_dir).rglob("metrics.csv")):
        rel_path = csv.relative_to(log_dir)
        parts = rel_path.parts

        # We strictly expect (dataset, model, metrics.csv)
        if len(parts) != 3:
            continue

        dataset_name = parts[0]
        model_name = parts[1]

        raw = pd.read_csv(csv)
        if "epoch" not in raw.columns:
            continue

        epoch_df = raw.groupby("epoch").first().reset_index()
        epoch_df["model"] = model_name
        epoch_df["dataset"] = dataset_name

        frames.append(epoch_df)

    if not frames:
        raise FileNotFoundError(f"No metrics.csv found under {log_dir}")

    df = pd.concat(frames, ignore_index=True)
    df["epoch"] = df["epoch"].astype(int)
    print(f"Training curves: {df['model'].nunique()} models, {len(df)} epoch rows")
    return df


def load_eval_results(*patterns: str | Path) -> pd.DataFrame:
    """Read one or more eval CSVs (paths or globs) into one DataFrame."""
    frames: list[pd.DataFrame] = []
    for pattern in patterns:
        pattern_str = str(pattern)
        matches = (
            sorted(Path(".").glob(pattern_str))
            if ("*" in pattern_str or "?" in pattern_str)
            else [Path(pattern_str)]
        )
        for p in matches:
            if p.is_file() and p.suffix == ".csv":
                frames.append(pd.read_csv(p))

    if not frames:
        raise FileNotFoundError(f"No CSVs matched: {patterns}")

    df = pd.concat(frames, ignore_index=True)
    print(f"Eval results: {df['model_name'].nunique()} models, {len(df):,} rows")
    return df
