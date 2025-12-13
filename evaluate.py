import os
import json
import hashlib
import datetime
from pathlib import Path
from typing import Any
import torch
from torch.utils.data import DataLoader
import numpy as np
import lightning as L

import config
from src.dataset.vsi_dataset import VSIDataset
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


def get_existing_result(arch_name: str, checkpoint_hash: str) -> dict[str, Any] | None:
    results = load_persistent_results()
    matches = [
        r
        for r in results
        if r.get("architecture") == arch_name
        and r.get("checkpoint_sha256") == checkpoint_hash
    ]
    if matches:
        return matches[-1]
    return None


def run_lightning_inference(
    model: L.LightningModule, loader: DataLoader
) -> tuple[float, float]:
    """Runs inference using Lightning Trainer and manual calculation for std dev (since Trainer just gives avg)."""
    # We use trainer.predict to get all raw outputs so we can calculate std dev and mae manually
    trainer = L.Trainer(devices=1, logger=False, enable_progress_bar=True)

    print("Starting evaluation via Lightning...")
    predictions = trainer.predict(model, loader)

    # predictions is a list of (outputs, targets) tuples from predict_step
    all_errors = []

    for batch_out, batch_targets in predictions:
        # Move to CPU for numpy calc
        batch_out = batch_out.cpu()
        batch_targets = batch_targets.cpu()

        abs_err = torch.abs(batch_out - batch_targets)
        all_errors.extend(abs_err.view(-1).tolist())

    errors = np.array(all_errors)
    mae = np.mean(errors)
    std = np.std(errors)

    return float(mae), float(std)


def evaluate() -> None:
    L.seed_everything(42)

    # Checkpoint path handling
    # We only support Lightning .ckpt format now.

    ckpt_dir = config.CHECKPOINT_DIR / config.MODEL_ARCH
    ckpt_path = ckpt_dir / "best_model.ckpt"
    if not ckpt_path.exists():
        print(f"Checkpoint not found at {ckpt_path}!")
        return

    # Calculate hash to ensure we match the exact file
    checkpoint_hash = get_file_hash(ckpt_path)

    # 0. Check for existing results
    current_arch = config.MODEL_ARCH.value
    existing_result = get_existing_result(current_arch, checkpoint_hash)

    if existing_result:
        print("\n--- Loaded Existing Evaluation Results ---")
        print(f"Architecture: {existing_result['architecture']}")
        print(f"Checkpoint SHA256: {existing_result.get('checkpoint_sha256', 'N/A')}")
        print(f"Mean Absolute Error (MAE): {existing_result['mae']:.4f} micrometers")
        print(f"Error Std Dev: {existing_result['std']:.4f} micrometers")
        print(f"Timestamp: {existing_result['timestamp']}")
        print("------------------------------------------")
        print("Skipping re-evaluation.")
        return

    # 1. Prepare Data
    index_path = config.get_index_path("test")
    if not index_path.exists():
        print(f"Test index not found at {index_path}. Run preprocess first.")
        return

    dataset = VSIDataset(mode="test")
    print(f"Test samples: {len(dataset)}")

    loader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=os.cpu_count(),
        persistent_workers=True,
    )

    # 2. Prepare Model
    print(f"Loading Model: {config.MODEL_ARCH} from {ckpt_path}...")

    # Load from Lightning Checkpoint
    # We need to allow ModelArch in safe globals because it was saved as a hyperparameter
    with torch.serialization.safe_globals([ModelArch]):
        model = FocusOffsetRegressor.load_from_checkpoint(ckpt_path)

    # 3. Evaluate
    mae, std = run_lightning_inference(model, loader)

    print("\n--- Test Evaluation Results ---")
    print(f"Architecture: {config.MODEL_ARCH}")
    print(f"Mean Absolute Error (MAE): {mae:.4f} micrometers")
    print(f"Error Std Dev: {std:.4f} micrometers")
    print("-------------------------------")

    # Save Results
    result_entry = {
        "architecture": current_arch,
        "mae": mae,
        "std": std,
        "timestamp": datetime.datetime.now().isoformat(),
        # Convert path to string for JSON serialization
        "checkpoint_path": str(ckpt_path.relative_to(config.PROJECT_ROOT)),
        "checkpoint_sha256": checkpoint_hash,
    }
    save_persistent_result(result_entry)
    print(f"Results saved to {RESULTS_FILE}")


if __name__ == "__main__":
    evaluate()
