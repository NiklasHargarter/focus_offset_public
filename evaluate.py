import os
from typing import Any
import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import json
import datetime
import hashlib

import config
from src.dataset.vsi_dataset import VSIDataset
from src.models.factory import get_model


RESULTS_FILE = config.PROJECT_ROOT / "evaluation_results.json"


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


def get_file_hash(filepath: str) -> str:
    with open(filepath, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def get_existing_result(
    arch_name: str, checkpoint_hash: str
) -> dict[str, Any] | None:
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


def evaluate() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Checkpoint path matches train.py structure
    checkpoint_path = config.CHECKPOINT_DIR / config.MODEL_ARCH / "best_model.pth"

    if not checkpoint_path.exists():
        print(f"Checkpoint not found at {checkpoint_path}!")
        return

    # Calculate hash to ensure we match the exact file
    checkpoint_hash = get_file_hash(checkpoint_path)

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
    print(f"Loading Model: {config.MODEL_ARCH}...")
    model = get_model(config.MODEL_ARCH, device)

    # We already checked existence above
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    print(f"Loaded checkpoint from {checkpoint_path}")

    # 3. Evaluate
    mae, std = run_inference(model, loader, device)

    print("\n--- Test Evaluation Results ---")
    print(f"Architecture: {config.MODEL_ARCH}")
    print(f"Mean Absolute Error (MAE): {mae:.4f} micrometers")
    print(f"Error Std Dev: {std:.4f} micrometers")
    print("-------------------------------")

    # Save Results
    result_entry = {
        "architecture": current_arch,
        "mae": float(mae),
        "std": float(std),
        "timestamp": datetime.datetime.now().isoformat(),
        # Convert path to string for JSON serialization
        "checkpoint_path": str(checkpoint_path.relative_to(config.PROJECT_ROOT)),
        "checkpoint_sha256": checkpoint_hash,
    }
    save_persistent_result(result_entry)
    print(f"Results saved to {RESULTS_FILE}")


def run_inference(
    model: torch.nn.Module, loader: DataLoader, device: torch.device
) -> tuple[float, float]:
    """Runs inference on the dataset and returns MAE and Std Dev."""
    model.eval()
    errors = []

    print("Starting evaluation...")
    with torch.no_grad():
        for images, targets in tqdm(loader, desc="Evaluating"):
            images = images.to(device)
            # targets are raw floats (micrometers)
            targets = targets.to(device).unsqueeze(1)

            outputs = model(images)

            # Calculate Absolute Error
            batch_errors = torch.abs(outputs - targets)
            errors.extend(batch_errors.cpu().tolist())

    errors = np.array(errors)
    mae = np.mean(errors)
    std = np.std(errors)

    return mae, std


if __name__ == "__main__":
    evaluate()
