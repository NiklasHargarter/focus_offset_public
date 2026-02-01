import json
import argparse
from pathlib import Path
import torch
import yaml
import lightning as L
import numpy as np
from lightning.pytorch.cli import LightningArgumentParser

import sys
import os

# Add project root to sys.path
sys.path.append(os.getcwd())

from src import config
from src.dataset.vsi_datamodule import VSIDataModule
from src.dataset.jiang2018 import Jiang2018DataModule
from src.models.lightning_module import FocusOffsetRegressor, FocusEnsemble

RESULTS_FILE = config.PROJECT_ROOT / "evaluation_results.json"
torch.set_float32_matmul_precision("medium")


def load_model(model_config: str, ckpt_path: str):
    """Load a single FocusOffsetRegressor from config and checkpoint."""
    with open(model_config, "r") as f:
        full_cfg = yaml.safe_load(f)

    parser = LightningArgumentParser()
    parser.add_lightning_class_args(FocusOffsetRegressor, "model")

    # Instantiate only the model
    model_cfg = {"model": full_cfg["model"]}
    cfg_init = parser.instantiate_classes(model_cfg)
    model = cfg_init.model

    print(f"  Loading weights from {ckpt_path}...")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt["state_dict"]

    # Handle torch.compile prefixes if present
    new_state_dict = {}
    for k, v in state_dict.items():
        if "_orig_mod." in k:
            new_state_dict[k.replace("_orig_mod.", "")] = v
        else:
            new_state_dict[k] = v

    model.load_state_dict(new_state_dict)
    return model


def evaluate_fold(trainer, model, datamodule, name):
    print(f"\n--- Testing {name} ---")
    test_results = trainer.test(model, datamodule=datamodule, verbose=False)
    if test_results:
        res = test_results[0]
        mae = res.get("test_mae_final", res.get("test_mae", 0.0))
        std = res.get("test_std", 0.0)
        return {"mae": mae, "std": std}
    return None


def main():
    parser = argparse.ArgumentParser(
        description="K-Fold and Ensemble evaluation script."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/models/convnext_v2_tiny.yaml",
        help="Model config",
    )
    parser.add_argument(
        "--log_dir", type=str, default="logs", help="Root log directory"
    )
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument(
        "--num_workers", type=int, default=os.cpu_count(), help="Number of workers"
    )
    parser.add_argument("--datasets", nargs="+", default=["ZStack_HE", "Jiang2018"])
    parser.add_argument(
        "--limit_batches", type=int, default=None, help="Limit batches for quick check"
    )
    args = parser.parse_args()

    log_root = Path(args.log_dir)
    # Discover checkpoints: logs/focus_convnextv2_kfold_fold_X/version_0/checkpoints/best_model-*.ckpt
    ckpt_paths = sorted(
        list(
            log_root.glob(
                "focus_convnextv2_kfold_fold_*/version_0/checkpoints/best_model-*.ckpt"
            )
        )
    )

    if not ckpt_paths:
        print(f"No kfold checkpoints found in {log_root}")
        return

    print(f"Found {len(ckpt_paths)} checkpoints.")

    trainer = L.Trainer(
        accelerator="auto",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        limit_test_batches=args.limit_batches,
        enable_progress_bar=True,
    )

    models = []
    for ckpt in ckpt_paths:
        model = load_model(args.config, str(ckpt))
        models.append(model)

    ensemble = FocusEnsemble(models)

    all_summaries = []

    for ds_name in args.datasets:
        print(f"\n\n{'#' * 60}")
        print(f"EVALUATING ON DATASET: {ds_name}")
        print(f"{'#' * 60}")

        if ds_name == "Jiang2018":
            datamodule = Jiang2018DataModule(
                batch_size=args.batch_size, num_workers=args.num_workers
            )
        else:
            datamodule = VSIDataModule(
                dataset_name=ds_name,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                patch_size=224,
                stride=448,
                min_tissue_coverage=0.05,
            )

        fold_results = []
        for i, model in enumerate(models):
            res = evaluate_fold(trainer, model, datamodule, f"Fold {i}")
            if res:
                res["ckpt"] = ckpt_paths[i].name
                fold_results.append(res)

        # 1. Individual Folds Aggregation
        maes = [r["mae"] for r in fold_results]

        summary = {
            "dataset": ds_name,
            "type": "kfold_individual",
            "mean_mae": float(np.mean(maes)),
            "std_mae": float(np.std(maes)),
            "folds": fold_results,
        }

        print("\n" + "=" * 40)
        print(f"K-FOLD INDIVIDUAL SUMMARY ({ds_name})")
        print(f"MEAN MAE: {summary['mean_mae']:.4f} ± {summary['std_mae']:.4f}")
        print("=" * 40)

        # 2. Ensemble Evaluation
        ensemble_res = evaluate_fold(trainer, ensemble, datamodule, "Ensemble")

        if ensemble_res:
            print("\n" + "=" * 40)
            print(f"ENSEMBLE SUMMARY ({ds_name})")
            print(f"MAE: {ensemble_res['mae']:.4f}")
            print("=" * 40)
            summary["ensemble"] = ensemble_res

        all_summaries.append(summary)

    # Save to history
    all_history = []
    if RESULTS_FILE.exists():
        with open(RESULTS_FILE, "r") as f:
            try:
                all_history = json.load(f)
            except Exception:
                pass

    all_history.extend(all_summaries)
    with open(RESULTS_FILE, "w") as f:
        json.dump(all_history, f, indent=4)


if __name__ == "__main__":
    main()
