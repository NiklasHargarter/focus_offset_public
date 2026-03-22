import torch
import torch.nn as nn
from synthetic_data.config import SyntheticConfig
from synthetic_data.dataset import get_synthetic_dataloaders
from synthetic_data.model import SyntheticConvModel


def verify_dimensions():
    config = SyntheticConfig()
    config.batch_size = 4  # Small batch for testing
    config.dry_run = True  # Use small subset

    print("--- Synthetic Configuration ---")
    print(f"Kernel Size: {config.kernel_size}")
    print(f"Input Patch Size: {config.patch_size_input}")
    print(f"Target Patch Size: {config.patch_size_target}")

    # 1. Check DataLoader
    train_loader, _ = get_synthetic_dataloaders(config, num_workers=0)
    batch = next(iter(train_loader))

    inputs = batch["input"]
    targets = batch["target"]

    print("\n--- Dataset Output ---")
    print(f"Input Shape: {inputs.shape}")  # Expected: [B, 3, 256, 256]
    print(f"Target Shape: {targets.shape}")  # Expected: [B, 3, 226, 226]

    assert inputs.shape[-1] == config.patch_size_input, (
        f"Input width mismatch: {inputs.shape[-1]} != {config.patch_size_input}"
    )
    assert targets.shape[-1] == config.patch_size_target, (
        f"Target width mismatch: {targets.shape[-1]} != {config.patch_size_target}"
    )

    # 2. Check Model Output
    model = SyntheticConvModel(kernel_size=config.kernel_size)
    model.eval()

    with torch.no_grad():
        preds = model(inputs)

    print("\n--- Model Output ---")
    print(f"Preds Shape: {preds.shape}")  # Expected: [B, 3, 226, 226]

    assert preds.shape == targets.shape, (
        f"Prediction shape {preds.shape} does not match target shape {targets.shape}"
    )

    # 3. Check Loss Calculation
    criterion = nn.MSELoss()
    loss = criterion(preds, targets)

    print("\n--- Loss Calculation ---")
    print(f"MSE Loss: {loss.item():.6f}")

    # Check if a manual MSE matches
    manual_mse = ((preds - targets) ** 2).mean()
    print(f"Manual MSE: {manual_mse.item():.6f}")

    assert torch.allclose(loss, manual_mse), "Loss calculation mismatch!"

    print("\nVerification Successful: All dimensions and loss calculations align!")


if __name__ == "__main__":
    verify_dimensions()
