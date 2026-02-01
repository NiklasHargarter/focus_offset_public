import torch
from src.dataset.vsi_datamodule import VSIDataModule
from src.dataset.jiang2018 import Jiang2018DataModule
import numpy as np


def check_targets():
    print("Checking Target Distributions (Optimized)...")

    # HE
    dm_he = VSIDataModule(dataset_name="ZStack_HE", batch_size=256)
    dm_he.setup()
    he_targets = []
    for i, (_, target, _) in enumerate(dm_he.test_dataloader()):
        he_targets.extend(target.tolist())
        if i >= 9:
            break
    he_targets = np.array(he_targets)
    print(
        f"HE Targets: mean={he_targets.mean():8.4f}, std={he_targets.std():8.4f}, range=[{he_targets.min():8.4f}, {he_targets.max():8.4f}]"
    )

    # Jiang
    dm_jiang = Jiang2018DataModule(batch_size=256)
    dm_jiang.setup()
    jiang_targets = []
    for i, (_, target) in enumerate(dm_jiang.test_dataloader()):
        jiang_targets.extend(target.tolist())
        if i >= 9:
            break
    jiang_targets = np.array(jiang_targets)
    print(
        f"Jiang Targets: mean={jiang_targets.mean():8.4f}, std={jiang_targets.std():8.4f}, range=[{jiang_targets.min():8.4f}, {jiang_targets.max():8.4f}]"
    )


if __name__ == "__main__":
    check_targets()
