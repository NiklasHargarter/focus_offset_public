import os
import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

import config
from src.dataset.vsi_dataset import VSIDataset
from src.models.factory import get_model


def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Checkpoint path matches train.py structure
    checkpoint_path = os.path.join(
        config.CHECKPOINT_DIR, config.MODEL_ARCH, "best_model.pth"
    )

    # 1. Prepare Data
    index_path = config.get_index_path("test")
    if not os.path.exists(index_path):
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

    if os.path.exists(checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)
        print(f"Loaded checkpoint from {checkpoint_path}")
    else:
        print(f"Checkpoint not found at {checkpoint_path}!")
        return

    # 3. Evaluate
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

    print("\n--- Test Evaluation Results ---")
    print(f"Architecture: {config.MODEL_ARCH}")
    print(f"Mean Absolute Error (MAE): {mae:.4f} micrometers")
    print(f"Error Std Dev: {std:.4f} micrometers")
    print("-------------------------------")


if __name__ == "__main__":
    evaluate()
