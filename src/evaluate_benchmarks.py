import json
import argparse
from pathlib import Path
import torch
import lightning as L
import os
import sys

# Add project root to sys.path
sys.path.append(os.getcwd())

from src import config
from src.dataset.vsi_datamodule import VSIDataModule, IHCDataModule
from src.dataset.jiang2018 import Jiang2018DataModule
from src.models.lightning_module import FocusOffsetRegressor

RESULTS_FILE = config.PROJECT_ROOT / "benchmark_results.json"
torch.set_float32_matmul_precision("medium")


MODEL_CHECKPOINTS = {
    "Multimodal": {
        "path": "logs/focus_convnextv2_multimodal/version_1/checkpoints/best_model-epoch=07-val_loss=0.1566.ckpt",
        "transform_mode": "all",
    },
    "Baseline RGB": {
        "path": "logs/focus_convnextv2_baseline_rgb/version_0/checkpoints/best_model-epoch=05-val_loss=0.1478.ckpt",
        "transform_mode": "none",
    },
}


def load_model_from_path(model_name: str, ckpt_path: str, transform_mode: str):
    """Load a model from a specific checkpoint path."""
    if not os.path.exists(ckpt_path):
        print(f"Checkpoint not found for {model_name}: {ckpt_path}")
        return None

    print(f"Loading {model_name} from {ckpt_path}")

    model = FocusOffsetRegressor(pretrained=False, transform_mode=transform_mode)

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


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark evaluation for single models."
    )
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument(
        "--num_workers", type=int, default=os.cpu_count(), help="Number of workers"
    )
    parser.add_argument(
        "--limit_batches", type=int, default=None, help="Limit batches for quick check"
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["HE", "IHC", "Jiang2018"],
        help="Datasets to evaluate on",
    )
    args = parser.parse_args()

    models_to_test = {}
    for name, cfg in MODEL_CHECKPOINTS.items():
        m = load_model_from_path(name, cfg["path"], cfg["transform_mode"])
        if m is not None:
            models_to_test[name] = m

    # Filter out models that couldn't be loaded
    models_to_test = {k: v for k, v in models_to_test.items() if v is not None}

    if not models_to_test:
        print("No models loaded. Exiting.")
        return

    trainer = L.Trainer(
        accelerator="auto",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        limit_test_batches=args.limit_batches,
        enable_progress_bar=True,
    )

    all_results = []

    for ds_name in args.datasets:
        print(f"\n\n{'#' * 60}")
        print(f"EVALUATING ON DATASET: {ds_name}")
        print(f"{'#' * 60}")

        try:
            if ds_name == "Jiang2018":
                datamodule = Jiang2018DataModule(
                    batch_size=args.batch_size, num_workers=args.num_workers
                )
            elif ds_name == "HE":
                datamodule = VSIDataModule(
                    dataset_name="ZStack_HE",
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                    patch_size=224,
                    stride=448,
                    downsample_factor=2,
                    min_tissue_coverage=0.05,
                )
            elif ds_name == "IHC":
                # Check if IHC data exists before attempting to load
                ihc_master_path = config.get_master_index_path(
                    "ZStack_IHC", patch_size=224
                )
                if not ihc_master_path.exists():
                    print(f"Skipping IHC: Master index not found at {ihc_master_path}")
                    continue

                datamodule = IHCDataModule(
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                    patch_size=224,
                    stride=448,
                    downsample_factor=2,
                    min_tissue_coverage=0.05,
                )
            else:
                print(f"Unknown dataset: {ds_name}")
                continue

            # We need to call prepare_data and setup manually if not using trainer.fit
            try:
                datamodule.prepare_data()
                datamodule.setup(stage="test")
            except Exception as e:
                print(f"Failed to setup datamodule for {ds_name}: {e}")
                continue

            for model_name, model in models_to_test.items():
                print(f"\n--- Testing {model_name} on {ds_name} ---")
                test_results = trainer.test(model, datamodule=datamodule, verbose=False)

                if test_results:
                    res = test_results[0]
                    mae = res.get("test_mae_final", res.get("test_mae", 0.0))
                    std = res.get("test_std", 0.0)

                    result_entry = {
                        "model": model_name,
                        "dataset": ds_name,
                        "mae": float(mae),
                        "std": float(std),
                    }
                    all_results.append(result_entry)
                    print(f"RESULT: MAE={mae:.4f} STD={std:.4f}")

        except Exception as e:
            print(f"Error evaluating on {ds_name}: {e}")

    # Save results
    if all_results:
        with open(RESULTS_FILE, "w") as f:
            json.dump(all_results, f, indent=4)
        print(f"\nFinal results saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
