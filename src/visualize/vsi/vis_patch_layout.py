import argparse
import pickle
import cv2
import numpy as np
import config
from src.dataset.vsi_types import MasterIndex

def save_layout(slide, output_dir):
    ds = 8 # Consistent visualization downscale
    h, w = slide.height // ds, slide.width // ds
    vis_img = np.zeros((h, w), dtype=np.uint8)
    
    patch_size = 224
    for p in slide.patches:
        dx, dy = int(p.x / ds), int(p.y / ds)
        dx_end, dy_end = int((p.x + patch_size) / ds), int((p.y + patch_size) / ds)
        cv2.rectangle(vis_img, (dx, dy), (dx_end, dy_end), 255, -1)
        
    out_path = output_dir / f"{slide.name}_patch_layout.png"
    cv2.imwrite(str(out_path), vis_img)
    print(f"Saved: {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="ZStack_HE")
    args = parser.parse_args()
    
    index_path = config.get_master_index_path(args.dataset)
    if not index_path.exists():
        print(f"Index not found: {index_path}")
        exit(1)
        
    with open(index_path, "rb") as f:
        master = pickle.load(f)
        
    for slide in master.file_registry:
        output_dir = config.VIS_DIR / "vsi" / Path(slide.name).stem
        output_dir.mkdir(parents=True, exist_ok=True)
        save_layout(slide, output_dir)
from pathlib import Path
