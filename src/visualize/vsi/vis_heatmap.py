import argparse
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import config
from src.dataset.vsi_types import MasterIndex

def save_heatmap(slide, output_dir):
    ds = 8
    h, w = slide.height // ds, slide.width // ds
    heatmap = np.full((h, w), np.nan)
    
    patch_size = 224
    for p in slide.patches:
        dx, dy = int(p.x / ds), int(p.y / ds)
        dx_end, dy_end = int((p.x + patch_size) / ds), int((p.y + patch_size) / ds)
        heatmap[dy:dy_end, dx:dx_end] = p.z
        
    plt.figure(figsize=(10, 8))
    plt.imshow(heatmap, cmap="viridis")
    plt.colorbar(label="Z-Slice")
    plt.title(f"Focus Heatmap: {slide.name}")
    plt.axis("off")
    
    out_path = output_dir / f"{Path(slide.name).stem}_heatmap.png"
    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Saved: {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="ZStack_HE")
    args = parser.parse_args()
    
    index_path = config.get_master_index_path(args.dataset)
    with open(index_path, "rb") as f:
        master = pickle.load(f)
        
    for slide in master.file_registry:
        output_dir = config.VIS_DIR / "vsi" / Path(slide.name).stem
        output_dir.mkdir(parents=True, exist_ok=True)
        save_heatmap(slide, output_dir)
