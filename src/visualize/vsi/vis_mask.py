import argparse
from pathlib import Path
import cv2
import slideio
from src import config
from src.utils.io_utils import suppress_stderr
from src.dataset.vsi_prep.preprocess import (
    detect_tissue,
)


def save_mask(vsi_path: Path):
    output_dir = config.VIS_DIR / "vsi" / vsi_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{vsi_path.stem}_mask.png"

    print(f"Generating mask for {vsi_path.name}...")
    with suppress_stderr():
        slide = slideio.open_slide(str(vsi_path), "VSI")
    scene = slide.get_scene(0)
    width, height = scene.size
    num_z = scene.num_z_slices

    ds = 8
    dw, dh = width // ds, height // ds

    best_z, mask = detect_tissue(scene)
    cv2.imwrite(str(out_path), mask)
    print(f"Saved: {out_path} (Best Z: {best_z})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="ZStack_HE")
    args = parser.parse_args()

    raw_dir = config.get_vsi_raw_dir(args.dataset)
    for vsi_path in sorted(raw_dir.glob("*.vsi")):
        save_mask(vsi_path)
