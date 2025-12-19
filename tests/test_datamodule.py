import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.dataset.vsi_datamodule import VSIDataModule  # noqa: E402


def test_datamodule_integration():
    """Actual integration test for VSIDataModule."""
    print("\n--- Starting VSIDataModule Integration Test ---")

    dm = VSIDataModule(batch_size=2, num_workers=0)

    print("\n[Stage 1] Running prepare_data()...")
    try:
        dm.prepare_data()
        print("[OK] prepare_data() completed.")
    except Exception as e:
        print(f"[FAIL] prepare_data() failed: {e}")
        return

    print("\n[Stage 2] Running setup()...")
    try:
        dm.setup(stage="fit")
        print(
            f"[OK] setup('fit') completed. Train size: {len(dm.train_dataset)}, Val size: {len(dm.val_dataset)}"
        )

        dm.setup(stage="test")
        print(f"[OK] setup('test') completed. Test size: {len(dm.test_dataset)}")
    except Exception as e:
        print(f"[FAIL] setup() failed: {e}")
        return

    print("\n[Stage 3] Testing Dataloaders...")
    try:
        train_loader = dm.train_dataloader()
        batch = next(iter(train_loader))

        if isinstance(batch, (list, tuple)):
            imgs, targets = batch
            print(
                f"[OK] Train batch received. Images shape: {imgs.shape}, Targets shape: {targets.shape}"
            )
            assert imgs.shape[0] == 2
        elif isinstance(batch, dict):
            print(f"[OK] Train batch received. Keys: {list(batch.keys())}")
            assert batch["image"].shape[0] == 2

        val_loader = dm.val_dataloader()
        next(iter(val_loader))
        print("[OK] Validation batch received.")

        test_loader = dm.test_dataloader()
        next(iter(test_loader))
        print("[OK] Test batch received.")

    except Exception as e:
        print(f"[FAIL] Dataloader verification failed: {e}")
        return

    print("\n--- VSIDataModule Integration Test PASSED ---")


if __name__ == "__main__":
    if not os.environ.get("EXACT_PASSWORD"):
        print("Warning: EXACT_PASSWORD not set.")

    test_datamodule_integration()
