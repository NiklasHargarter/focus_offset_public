import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

import config
from src.dataset.vsi_dataset import VSIDataset
from src.models.factory import get_model


def train_one_epoch(model, loader, criterion, optimizer, device, epoch):
    model.train()
    running_loss = 0.0

    pbar = tqdm(loader, desc=f"Epoch {epoch} [Train]", leave=False)
    for images, targets in pbar:
        images = images.to(device)
        targets = targets.to(device).unsqueeze(1)  # [Batch, 1]

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    avg_loss = running_loss / len(loader)
    return avg_loss


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for images, targets in tqdm(loader, desc="Validation", leave=False):
            images = images.to(device)
            targets = targets.to(device).unsqueeze(1)

            outputs = model(images)
            loss = criterion(outputs, targets)
            running_loss += loss.item()

    return running_loss / len(loader)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Prepare Data
    index_path = config.get_index_path("train")
    if not os.path.exists(index_path):
        print(f"Training index not found at {index_path}. Run preprocess first.")
        return

    full_dataset = VSIDataset(mode="train")
    total_size = len(full_dataset)
    val_size = int(0.1 * total_size)
    train_size = total_size - val_size

    train_subset, val_subset = random_split(full_dataset, [train_size, val_size])

    print(f"Train samples: {train_size}, Val samples: {val_size}")

    train_loader = DataLoader(
        train_subset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=os.cpu_count(),
        persistent_workers=True,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=os.cpu_count(),
        persistent_workers=True,
    )

    # 2. Prepare Model
    model = get_model(config.MODEL_ARCH, device)

    # 3. Optimization
    criterion = nn.MSELoss()  # Regression loss
    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)

    # 4. Training Loop
    # Use model-specific checkpoint dir to avoid overwriting
    checkpoint_dir = os.path.join(config.CHECKPOINT_DIR, config.MODEL_ARCH)
    os.makedirs(checkpoint_dir, exist_ok=True)

    best_loss = float("inf")
    epochs_no_improve = 0

    print(f"Starting training for architecture: {config.MODEL_ARCH}")

    for epoch in range(config.EPOCHS):
        print(f"\n--- Epoch {epoch + 1}/{config.EPOCHS} ---")

        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch + 1
        )
        val_loss = validate(model, val_loader, criterion, device)

        print(f"Train Loss (MSE): {train_loss:.4f} | Val Loss (MSE): {val_loss:.4f}")

        # Checkpointing
        if val_loss < best_loss:
            best_loss = val_loss
            epochs_no_improve = 0
            save_path = os.path.join(checkpoint_dir, "best_model.pth")
            torch.save(model.state_dict(), save_path)
            print(f"Saved new best model to {save_path}")
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve} epochs.")

        # Early Stopping
        if epochs_no_improve >= config.PATIENCE:
            print("Early stopping triggered.")
            break

    print("Training Complete.")


if __name__ == "__main__":
    main()
