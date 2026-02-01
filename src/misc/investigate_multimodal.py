from src.models.architectures import InputTransformLayer
from src.dataset.vsi_datamodule import VSIDataModule
from src.dataset.jiang2018 import Jiang2018DataModule


def investigate_real_data():
    print("Investigating Real Data Features...")

    transform = InputTransformLayer()

    # HE
    dm_he = VSIDataModule(dataset_name="ZStack_HE", batch_size=4)
    dm_he.setup()
    x_he, _, _ = next(iter(dm_he.test_dataloader()))

    # Jiang
    dm_jiang = Jiang2018DataModule(batch_size=4)
    dm_jiang.setup()
    x_jiang, _ = next(iter(dm_jiang.test_dataloader()))

    datasets = [("HE", x_he), ("Jiang", x_jiang)]
    names = ["R", "G", "B", "FFT", "DWT_LH", "DWT_HL", "DWT_HH"]

    for ds_name, x in datasets:
        print(f"\n--- Dataset: {ds_name} ---")
        features = transform(x)
        for i in range(7):
            channel = features[:, i]
            mean = channel.mean().item()
            std = channel.std().item()
            print(f"Channel {i} ({names[i]}): mean={mean:8.4f}, std={std:8.4f}")


if __name__ == "__main__":
    investigate_real_data()
