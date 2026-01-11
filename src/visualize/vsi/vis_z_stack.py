import argparse
from pathlib import Path
import cv2
import slideio
import matplotlib.pyplot as plt
import config
from src.utils.io_utils import suppress_stderr


def save_z_stack(vsi_path: Path, output_dir: Path, patch_size: int):
    output_dir.mkdir(parents=True, exist_ok=True)

    with suppress_stderr():
        slide = slideio.open_slide(str(vsi_path), "VSI")
    scene = slide.get_scene(0)
    w, h = scene.size
    num_z = scene.num_z_slices

    # Sample at center
    cx, cy = w // 2, h // 2

    slices = []
    for z in range(num_z):
        img = scene.read_block(
            rect=(cx, cy, patch_size, patch_size),
            size=(patch_size, patch_size),
            slices=(z, z + 1),
        )
        slices.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    cols = min(num_z, 5)
    rows = (num_z + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    axes = axes.flatten()

    for i, img in enumerate(slices):
        axes[i].imshow(img)
        axes[i].set_title(f"Z={i}")
        axes[i].axis("off")
    for i in range(num_z, len(axes)):
        axes[i].axis("off")

    plt.suptitle(f"Z-Stack: {vsi_path.name} (p{patch_size})")
    out_path = output_dir / f"{vsi_path.stem}_z_stack.png"
    plt.savefig(out_path, dpi=100)
    plt.close()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="ZStack_HE")
    parser.add_argument("--patch_size", type=int, default=config.PATCH_SIZE)
    args = parser.parse_args()

    raw_dir = config.get_vsi_raw_dir(args.dataset)
    vis_root = config.get_vis_dir(args.dataset, patch_size=args.patch_size)
    for vsi_path in sorted(raw_dir.glob("*.vsi")):
        output_dir = vis_root / vsi_path.stem
        save_z_stack(vsi_path, output_dir, args.patch_size)
