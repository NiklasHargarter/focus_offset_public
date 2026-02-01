import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from src import config
from src.dataset.vsi_prep.preprocess import load_master_index


def save_3d(slide, output_dir):
    X = slide.patches[:, 0]
    Y = slide.patches[:, 1]
    Z = slide.patches[:, 2]

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
    parser.add_argument(
        "--limit", type=int, default=5, help="Number of slides to process (default: 5)"
    )
    args = parser.parse_args()

    master = load_master_index(args.dataset, patch_size=args.patch_size)
    if master is None:
        print(f"Error: Master index not found for {args.dataset}")
        exit(1)

    vis_root = config.get_vis_dir("tilts_3d", args.dataset, patch_size=args.patch_size)
    for slide in master.file_registry[: args.limit]:
        save_3d(slide, vis_root)
