import argparse
import pickle
import cv2
import numpy as np
import config
from pathlib import Path


def save_layout(slide, output_dir, patch_size):
    ds = 8  # Consistent visualization downscale
    h, w = slide.height // ds, slide.width // ds
    vis_img = np.zeros((h, w), dtype=np.uint8)

    for p in slide.patches:
        dx, dy = int(p.x / ds), int(p.y / ds)
        dx_end = int((p.x + patch_size) / ds)
        dy_end = int((p.y + patch_size) / ds)
        cv2.rectangle(vis_img, (dx, dy), (dx_end, dy_end), 255, -1)

    out_path = output_dir / f"{slide.name}_patch_layout.png"
    cv2.imwrite(str(out_path), vis_img)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="ZStack_HE")
    parser.add_argument("--patch_size", type=int, default=config.PATCH_SIZE)
    args = parser.parse_args()

    index_path = config.get_master_index_path(args.dataset, patch_size=args.patch_size)
    if not index_path.exists():
        print(f"Index not found: {index_path}")
        exit(1)

    with open(index_path, "rb") as f:
        master = pickle.load(f)

    vis_root = config.get_vis_dir(args.dataset, patch_size=args.patch_size)

    for slide in master.file_registry:
        output_dir = vis_root / Path(slide.name).stem
        output_dir.mkdir(parents=True, exist_ok=True)
        save_layout(slide, output_dir, master.patch_size)
