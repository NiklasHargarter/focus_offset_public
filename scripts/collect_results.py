import pandas as pd
from pathlib import Path
import numpy as np


def aggregate_jiang_predictions(df):
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


models = ["dwt", "fft", "rgb", "multimodal"]
datasets = ["zstack_he", "zstack_ihc", "jiang2018"]

results = []

for m in models:
    for d in datasets:
        csv_path = Path(f"logs/{m}/eval_{m}_{d}.csv")
        if not csv_path.exists():
            continue

        df = pd.read_csv(csv_path)
        is_jiang = d == "jiang2018"
        if is_jiang:
            df = aggregate_jiang_predictions(df)

        error = df["pred"] - df["target"]
        abs_error = error.abs()
        mae = abs_error.mean()
        median = abs_error.median()

        results.append(
            {
                "Model": m.upper(),
                "Dataset": d.replace("zstack_", "").upper(),
                "MAE": f"{mae:.4f}",
                "Median AE": f"{median:.4f}",
            }
        )

res_df = pd.DataFrame(results)
pivot = res_df.pivot(index="Model", columns="Dataset", values=["MAE", "Median AE"])
# Reorder columns: group by dataset, then metric
dataset_order = [d.replace("zstack_", "").upper() for d in datasets]
metric_order = ["MAE", "Median AE"]
pivot = pivot.reindex(columns=pd.MultiIndex.from_product([metric_order, dataset_order]))
pivot.columns = [f"{ds} {metric}" for metric, ds in pivot.columns]
pivot.index.name = "Model"
print(pivot.to_markdown())
