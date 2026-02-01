import cv2
from src.dataset.vsi_datamodule import VSIDataModule
from src.dataset.jiang2018 import Jiang2018DataModule
import os


def save_samples():
    print("Saving samples for visual inspection...")
    out_dir = "/home/niklas/.gemini/antigravity/brain/efb5014d-6bdf-4cf1-82cf-207f6b8d32fc/samples"
    os.makedirs(out_dir, exist_ok=True)

    # HE
    dm_he = VSIDataModule(dataset_name="ZStack_HE", batch_size=1)
    dm_he.setup()
    x_he, target_he, _ = next(iter(dm_he.test_dataloader()))
    img_he = (
        x_he[0].permute(1, 2, 0).numpy() * 0.224 + 0.456
    ) * 255  # Approx de-norm for visualization
    img_he = cv2.cvtColor(img_he.astype("uint8"), cv2.COLOR_RGB2BGR)
    cv2.imwrite(
        os.path.join(out_dir, f"he_sample_offset_{target_he.item():.2f}.jpg"), img_he
    )

    # Jiang
    dm_jiang = Jiang2018DataModule(batch_size=1)
    dm_jiang.setup()
    x_jiang, target_jiang = next(iter(dm_jiang.test_dataloader()))
    img_jiang = (x_jiang[0].permute(1, 2, 0).numpy() * 0.224 + 0.456) * 255
    img_jiang = cv2.cvtColor(img_jiang.astype("uint8"), cv2.COLOR_RGB2BGR)
    cv2.imwrite(
        os.path.join(out_dir, f"jiang_sample_offset_{target_jiang.item():.2f}.jpg"),
        img_jiang,
    )

    print(f"Samples saved to {out_dir}")


if __name__ == "__main__":
    save_samples()
