import json
import datetime
import argparse
from pathlib import Path
import torch
import yaml
import lightning as L
from lightning.pytorch.cli import LightningArgumentParser

import config
from src.dataset.vsi_datamodule import VSIDataModule
from src.dataset.jiang2018 import Jiang2018DataModule
from src.models.lightning_module import FocusOffsetRegressor

RESULTS_FILE = config.PROJECT_ROOT / "evaluation_results.json"
torch.set_float32_matmul_precision("medium")


def evaluate(
    model_config: str,
    ckpt_path: str,
    dataset_names: list[str],
    limit_batches: int = None,
):
    ckpt_path = Path(ckpt_path)

    # 1. Instantiate the model using the same logic as train.py
    # We manually load the YAML and then use the parser only for the "model" part
    # to avoid validation errors on optimizer/scheduler sections.
    with open(model_config, "r") as f:
        full_cfg = yaml.safe_load(f)

    parser = LightningArgumentParser()
    parser.add_lightning_class_args(FocusOffsetRegressor, "model")

    # Instantiate only the model
    model_cfg = {"model": full_cfg["model"]}
    cfg_init = parser.instantiate_classes(model_cfg)
    model = cfg_init.model

    print(f"🚀 Loading weights into {model.backbone.__class__.__name__}...")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["state_dict"])

    trainer = L.Trainer(
        accelerator="auto",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        limit_test_batches=limit_batches,
        enable_progress_bar=True,
    )

    all_results = []
    if RESULTS_FILE.exists():
        with open(RESULTS_FILE, "r") as f:
            try:
                all_results = json.load(f)
            except json.JSONDecodeError:
                all_results = []

    for ds_name in dataset_names:
        print(f"\n--- Testing on: {ds_name} ---")
        datamodule = (
            Jiang2018DataModule()
            if ds_name == "Jiang2018"
            else VSIDataModule(dataset_name=ds_name)
        )

        test_results = trainer.test(model, datamodule=datamodule, verbose=False)

        if test_results:
            res = test_results[0]
            mae = res.get("test_mae_final", res.get("test_mae", 0.0))
            std = res.get("test_std", 0.0)
            print(f"✅ MAE: {mae:.4f}")

            all_results.append(
                {
                    "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "architecture": Path(
                        model_config
                    ).stem,  # Log the config name as the arch
                    "dataset": ds_name,
                    "mae": mae,
                    "std": std,
                    "checkpoint": ckpt_path.name,
                }
            )

    with open(RESULTS_FILE, "w") as f:
        json.dump(all_results, f, indent=4)
    print(f"\n✨ Results saved to {RESULTS_FILE}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Consistent evaluation script.")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to the model YAML config used for training.",
    )
    parser.add_argument(
        "--ckpt", type=str, required=True, help="Path to the checkpoint file."
    )
    parser.add_argument(
        "--datasets", nargs="+", default=["ZStack_HE", "ZStack_IHC", "Jiang2018"]
    )
    parser.add_argument(
        "--limit_batches",
        type=int,
        default=None,
        help="Limit number of test batches for a quick run.",
    )

    args = parser.parse_args()
    evaluate(args.model, args.ckpt, args.datasets, args.limit_batches)
