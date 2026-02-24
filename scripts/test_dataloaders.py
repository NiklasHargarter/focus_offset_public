import torch

from src.config import TrainConfig
from src.datasets.zstack_he import (
    get_dataloaders as get_he_loaders,
    get_test_loader as get_he_test_loader,
)
from src.datasets.zstack_ihc import (
    get_dataloaders as get_ihc_loaders,
    get_test_loader as get_ihc_test_loader,
)
from src.datasets.jiang2018 import get_jiang2018_dataloaders


def test_loader(name, loader):
    print(f"\n>>>> Testing {name} loader...")
    if len(loader.dataset) == 0:
        print(f"  [SKIP] Dataset is empty ({name}).")
        return True  # Not a failure of logic, just no data in this split

    try:
        batch = next(iter(loader))

        # Check keys
        expected_keys = {"image", "target", "metadata"}
        actual_keys = set(batch.keys())
        if not expected_keys.issubset(actual_keys):
            print(
                "[FAIL] Batch keys missing. Expected "
                + str(expected_keys)
                + ", got "
                + str(actual_keys)
            )
            return False

        images = batch["image"]
        targets = batch["target"]

        print("  [OK] Batch retrieved successfully.")
        print(f"  [OK] Image shape: {images.shape} (Expected [B, 3, 224, 224])")
        print(f"  [OK] Target shape: {targets.shape}")
        print(f"  [OK] Sample targets: {targets[:5].tolist()}")

        # Validation checks
        if images.shape[1:] != (3, 224, 224):
            print(f"  [WARN] Unexpected image shape: {images.shape}")

        if not isinstance(targets, torch.Tensor):
            print("  [FAIL] Target is not a tensor")
            return False

        return True
    except Exception as e:
        print(f"  [FAIL] Error during iteration: {e}")
        return False


def main():
    cfg = TrainConfig(batch_size=8, num_workers=0)  # Use 0 workers for easier debugging

    results = {}

    # Test HE
    try:
        he_train, he_val = get_he_loaders(cfg)
        results["ZStack HE [Train]"] = test_loader("ZStack HE [Train]", he_train)
        results["ZStack HE [Val]"] = test_loader("ZStack HE [Val]", he_val)

        he_test = get_he_test_loader(cfg)
        results["ZStack HE [Test]"] = test_loader("ZStack HE [Test]", he_test)
    except Exception as e:
        print("[SKIP] ZStack HE failed to initialize: " + str(e))
        results["ZStack HE"] = False

    # Test IHC
    try:
        ihc_train, ihc_val = get_ihc_loaders(cfg)
        results["ZStack IHC [Train]"] = test_loader("ZStack IHC [Train]", ihc_train)
        results["ZStack IHC [Val]"] = test_loader("ZStack IHC [Val]", ihc_val)

        ihc_test = get_ihc_test_loader(cfg)
        results["ZStack IHC [Test]"] = test_loader("ZStack IHC [Test]", ihc_test)
    except Exception as e:
        print("[SKIP] ZStack IHC failed to initialize: " + str(e))
        results["ZStack IHC"] = False

    # Test Jiang2018
    try:
        j_train, j_test = get_jiang2018_dataloaders(batch_size=8, num_workers=0)
        results["Jiang2018 [Train]"] = test_loader("Jiang2018 [Train]", j_train)
        results["Jiang2018 [Test]"] = test_loader("Jiang2018 [Test]", j_test)
    except Exception as e:
        print("[SKIP] Jiang2018 failed to initialize: " + str(e))
        results["Jiang2018"] = False

    print("\n" + "=" * 30)
    print("      TEST SUMMARY")
    print("=" * 30)
    for name, success in results.items():
        status = "PASSED" if success else "FAILED/SKIPPED"
        print(f"{name:20}: {status}")


if __name__ == "__main__":
    main()
