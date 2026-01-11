import argparse
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import config


def save_3d(slide, output_dir):
    X = np.array([p.x for p in slide.patches])
    Y = np.array([p.y for p in slide.patches])
    Z = np.array([p.z for p in slide.patches])

    if len(X) < 3:
        return

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
    parser.add_argument("--patch_size", type=int, default=config.PATCH_SIZE)
    args = parser.parse_args()

    index_path = config.get_master_index_path(args.dataset, patch_size=args.patch_size)
    with open(index_path, "rb") as f:
        master = pickle.load(f)

    vis_root = config.get_vis_dir(args.dataset, patch_size=args.patch_size)
    for slide in master.file_registry:
        output_dir = vis_root / Path(slide.name).stem
        output_dir.mkdir(parents=True, exist_ok=True)
        save_3d(slide, output_dir)
