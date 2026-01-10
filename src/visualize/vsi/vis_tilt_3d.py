import argparse
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import config
from src.dataset.vsi_types import MasterIndex

def save_3d(slide, output_dir):
    X = np.array([p.x for p in slide.patches])
    Y = np.array([p.y for p in slide.patches])
    Z = np.array([p.z for p in slide.patches])
    
    if len(X) < 3: return
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_trisurf(X, Y, Z, cmap="viridis")
    ax.invert_yaxis()
    ax.set_title(f"Focus Tilt 3D: {slide.name}")
    
    out_path = output_dir / f"{Path(slide.name).stem}_tilt_3d.png"
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
        save_3d(slide, output_dir)
