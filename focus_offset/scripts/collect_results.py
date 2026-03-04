import pandas as pd
from pathlib import Path
import numpy as np

# Use exact same constants as predict_models.py
TRAIN_DATASETS = ["jiang2018", "zstack_he"]
MODELS = ["rgb", "dwt", "rgb_fft", "grayscale_fft", "fourier_domain", "two_domain"]
DATASETS = ["jiang2018_same", "jiang2018_diff", "zstack_he", "zstack_ihc", "agnor_ome"]


def aggregate_jiang_predictions(df):
    """Aggregate per-tile predictions for Jiang2018 dataset."""
    if "filename" not in df.columns:
        return df
    agg_rows = []
    for filename, group in df.groupby("filename"):
        preds = group["pred"].values
        if len(preds) > 10:
            median = np.median(preds)
            diffs = np.abs(preds - median)
            outlier_indices = np.argsort(diffs)[-10:]
            mask = np.ones(len(preds), dtype=bool)
            mask[outlier_indices] = False
            final_pred = preds[mask].mean()
        else:
            final_pred = preds.mean()
        agg_rows.append({"pred": final_pred, "target": group["target"].iloc[0]})
    return pd.DataFrame(agg_rows)


def main():
    print("Collecting Results...")

    # We will aggregate to display:
    # Rows: Models (e.g. RGB(Jiang2018), RGB(ZSTACK_HE))
    # Columns: Datasets (Jiang2018 SAME, Jiang2018 DIFF, HE, IHC, AGNOR OME)

    results = []

    for train_domain in TRAIN_DATASETS:
        for m in MODELS:
            for d in DATASETS:
                csv_path = Path(f"logs/{train_domain}/{m}/eval_{m}_{d}.csv")
                if not csv_path.exists():
                    continue

                try:
                    df = pd.read_csv(csv_path)
                except pd.errors.EmptyDataError:
                    continue

                if "jiang2018" in d:
                    df = aggregate_jiang_predictions(df)

                error = df["pred"] - df["target"]
                abs_error = error.abs()
                mae = abs_error.mean()
                median = abs_error.median()

                results.append(
                    {
                        "Model": f"{m.upper()} ({train_domain.upper()})",
                        "Dataset": d.replace("zstack_", "").upper(),
                        "MAE": f"{mae:.4f}",
                        "Median AE": f"{median:.4f}",
                    }
                )

    if not results:
        print("\nNo results found.\n")
        return

    res_df = pd.DataFrame(results)

    # Build pivot table
    pivot = res_df.pivot(index="Model", columns="Dataset", values=["MAE", "Median AE"])

    # Reorder columns to group by dataset, then metric
    dataset_order = [d.replace("zstack_", "").upper() for d in DATASETS]

    # Only include datasets that actually have results
    available_datasets = [d for d in dataset_order if d in pivot.columns.levels[1]]

    metric_order = ["MAE", "Median AE"]
    pivot = pivot.reindex(
        columns=pd.MultiIndex.from_product([metric_order, available_datasets])
    )
    pivot.columns = [f"{ds} {metric}" for metric, ds in pivot.columns]
    pivot.index.name = "Model"

    print("\n=======================================================")
    print("=== Model Performance Across Train Domains & Datasets ===")
    print("=======================================================\n")
    print(pivot.to_markdown())
    print("\n")


if __name__ == "__main__":
    main()
