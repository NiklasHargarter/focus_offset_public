import argparse
import pickle
import cv2
import slideio
from pathlib import Path
from src import config
from src.utils.io_utils import suppress_stderr
from src.dataset.vsi_prep.preprocess import load_master_index


def visualize_patch_layout(dataset_name, patch_size, limit_slides=1):
    master = load_master_index(dataset_name, patch_size=patch_size)
    if master is None:
        print(f"Error: Master index not found for {dataset_name}")
        return

    ds_factor = master.config_state.downsample_factor
    stride = master.config_state.stride
    print(f"Visualizing layout for {dataset_name}:")
    print(f"  Downsample Factor: {ds_factor}x")
    print(f"  Patch Size: {patch_size}px (Output) -> {patch_size * ds_factor}px (Raw)")
    print(f"  Stride: {stride}px (Output) -> {stride * ds_factor}px (Raw)")

    vis_root = config.get_vis_dir("layouts", dataset_name, patch_size=patch_size)

    for i, slide_meta in enumerate(master.file_registry[:limit_slides]):
        vsi_path = config.get_vsi_raw_dir(dataset_name) / Path(slide_meta.name)

        print(f"[{i + 1}/{limit_slides}] Processing {vsi_path.name}...")

        with suppress_stderr():
            slide = slideio.open_slide(str(vsi_path), "VSI")
            scene = slide.get_scene(0)

        # Original Raw Dimensions
        raw_w, raw_h = scene.size

        # We want a manageable visualization image (e.g., ~2000px wide)
        vis_downsample = max(1, raw_w // 2000)
        vis_w, vis_h = raw_w // vis_downsample, raw_h // vis_downsample

        print(f"  Generating thumbnail ({vis_w}x{vis_h})...")
        with suppress_stderr():
            thumb = scene.read_block(rect=(0, 0, raw_w, raw_h), size=(vis_w, vis_h))

        # Draw patches
        # patches are in DOWNSAMPLED coordinates
        # To get raw: p.x * ds_factor
        # To get vis: (p.x * ds_factor) / vis_downsample

        overlay = thumb.copy()

        # Use a subset of patches if there are too many, or just draw all with thin lines
        print(f"  Drawing {len(slide_meta.patches)} patches...")
        for p in slide_meta.patches:
            x, y, _ = p
            # Calculate raw coordinates (x, y are already in RAW scale)
            rx, ry = x, y
            rw, rh = patch_size * ds_factor, patch_size * ds_factor

            # Scale to visualization coordinates
            vx, vy = int(rx / vis_downsample), int(ry / vis_downsample)
            vw, vh = int(rw / vis_downsample), int(rh / vis_downsample)

            # Draw semi-transparent rectangle
            cv2.rectangle(overlay, (vx, vy), (vx + vw, vy + vh), (0, 255, 0), -1)
            # Draw border
            cv2.rectangle(thumb, (vx, vy), (vx + vw, vy + vh), (0, 0, 255), 1)

        # Blend
        alpha = 0.3
        result = cv2.addWeighted(overlay, alpha, thumb, 1 - alpha, 0)

        # Add legend/info text
        text = f"Zoom: {ds_factor}x | Stride: {stride}px | Patch: {patch_size}px"
        cv2.putText(
            result, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3
        )
        cv2.putText(result, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 1)

        out_path = vis_root / f"{vsi_path.stem}_layout_v2.jpg"
        cv2.imwrite(str(out_path), result)
        print(f"  Success! Visualization saved to: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="ZStack_HE")
    parser.add_argument("--patch_size", type=int, default=224)
    parser.add_argument("--limit", type=int, default=1)
    args = parser.parse_args()

    visualize_patch_layout(args.dataset, args.patch_size, args.limit)
