import json
import hashlib
import datetime
from pathlib import Path
import argparse
from typing import Any
import torch
import numpy as np
import lightning as L

import config
from src.dataset.vsi_datamodule import VSIDataModule
from src.models.lightning_module import FocusOffsetRegressor
from src.models.factory import ModelArch

RESULTS_FILE = config.PROJECT_ROOT / "evaluation_results.json"
torch.set_float32_matmul_precision("medium")


def load_persistent_results() -> list[dict[str, Any]]:
    if RESULTS_FILE.exists():
        with open(RESULTS_FILE, "r") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return []
    return []


def save_persistent_result(entry: dict[str, Any]) -> None:
    results = load_persistent_results()
    results.append(entry)
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=4)


def get_file_hash(filepath: str | Path) -> str:
    with open(filepath, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def get_existing_result(
    arch_name: str, checkpoint_hash: str, dataset_name: str
) -> dict[str, Any] | None:
    results = load_persistent_results()
    matches = [
        r
        for r in results
        if r.get("architecture") == arch_name
        and r.get("checkpoint_sha256") == checkpoint_hash
        and r.get("dataset") == dataset_name
    ]
    if matches:
        return matches[-1]
    return None


def run_lightning_inference(
    model: L.LightningModule, datamodule: L.LightningDataModule
) -> tuple[float, float]:
    """Calculate MAE and std dev for a given DataModule."""
    trainer = L.Trainer(devices=1, logger=False, enable_progress_bar=True)

    print(f"Starting evaluation for {datamodule.dataset_name}...")
    predictions = trainer.predict(model, datamodule=datamodule)

    all_errors = []
    for batch_out, batch_targets in predictions:
        batch_out = batch_out.cpu()
        batch_targets = batch_targets.cpu()
        abs_err = torch.abs(batch_out - batch_targets)
        all_errors.extend(abs_err.view(-1).tolist())

    if not all_errors:
        print(f"Warning: No predictions for {datamodule.dataset_name}.")
        return 0.0, 0.0

    errors = np.array(all_errors)
    mae = np.mean(errors)
    std = np.std(errors)
    return float(mae), float(std)


def evaluate(dataset_names: list[str]) -> None:
    L.seed_everything(42)

    ckpt_dir = config.CHECKPOINT_DIR / config.MODEL_ARCH
    ckpt_path = ckpt_dir / "best_model.ckpt"
    if not ckpt_path.exists():
        print(f"Checkpoint not found at {ckpt_path}!")
        return

    checkpoint_hash = get_file_hash(ckpt_path)
    current_arch = config.MODEL_ARCH.value

    for ds_name in dataset_names:
        existing_result = get_existing_result(current_arch, checkpoint_hash, ds_name)
        if existing_result:
            print(f"\n--- Loaded Existing Evaluation Results for {ds_name} ---")
            print(f"Architecture: {existing_result['architecture']}")
            print(
                f"Mean Absolute Error (MAE): {existing_result['mae']:.4f} micrometers"
            )
            print(f"Error Std Dev: {existing_result['std']:.4f} micrometers")
            print(f"Timestamp: {existing_result['timestamp']}")
            print("------------------------------------------")
            continue

        print(f"\n=== Evaluating Dataset: {ds_name} ===")
        datamodule = VSIDataModule(dataset_name=ds_name)

        print(f"Loading Model: {config.MODEL_ARCH} for evaluation on {ds_name}...")
        with torch.serialization.safe_globals([ModelArch]):
            model = FocusOffsetRegressor.load_from_checkpoint(ckpt_path)

        mae, std = run_lightning_inference(model, datamodule)

        print(f"\n--- Test Evaluation Results ({ds_name}) ---")
        print(f"Architecture: {config.MODEL_ARCH}")
        print(f"Mean Absolute Error (MAE): {mae:.4f} micrometers")
        print(f"Error Std Dev: {std:.4f} micrometers")
        print("-------------------------------")

        result_entry = {
            "architecture": current_arch,
            "dataset": ds_name,
            "mae": mae,
            "std": std,
            "timestamp": datetime.datetime.now().isoformat(),
            "checkpoint_path": str(ckpt_path.relative_to(config.PROJECT_ROOT)),
            "checkpoint_sha256": checkpoint_hash,
        }
        save_persistent_result(result_entry)
        print(f"Results for {ds_name} saved to {RESULTS_FILE}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=[config.DATASET_NAME])
    args = parser.parse_args()
    evaluate(dataset_names=args.datasets)
