import argparse
import cv2
import slideio
from src import config
from src.utils.io_utils import suppress_stderr
from src.dataset.vsi_prep.preprocess import detect_tissue, generate_patch_candidates


def generate_preview(vsi_path, patch_size, stride, downsample_factor):
    print(f"Generating Layout Preview for: {vsi_path.name}")
    print(f"  Zoom: {60 / downsample_factor}x (Downsample: {downsample_factor}x)")
    print(
        f"  Patch Size: {patch_size}px (Output) -> {patch_size * downsample_factor}px (Raw)"
    )
    print(f"  Stride: {stride}px (Output) -> {stride * downsample_factor}px (Raw)")

    with suppress_stderr():
        slide = slideio.open_slide(str(vsi_path), "VSI")
        scene = slide.get_scene(0)

    raw_w, raw_h = scene.size

    # 1. Detect tissue and generate grid (using the same logic as the preprocessor)
    print("  Detecting tissue mask...")
    _, mask = detect_tissue(scene)

    print("  Generating patch candidates...")
    raw_patch_size = patch_size * downsample_factor
    raw_stride = stride * downsample_factor

    candidates = generate_patch_candidates(
        mask, raw_w, raw_h, raw_patch_size, raw_stride, min_cov=0.05
    )

    print(f"  Total patches found: {len(candidates)}")

    # 2. Create Visualization
    vis_downsample = max(1, raw_w // 2000)
    vis_w, vis_h = raw_w // vis_downsample, raw_h // vis_downsample

    print(f"  Reading thumbnail ({vis_w}x{vis_h})...")
    with suppress_stderr():
        thumb = scene.read_block(rect=(0, 0, raw_w, raw_h), size=(vis_w, vis_h))

    overlay = thumb.copy()

    # Draw patches
    for rx, ry in candidates:
        rw, rh = raw_patch_size, raw_patch_size

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
    text = f"Zoom: {60 / downsample_factor:.0f}x | Stride; {stride}px | Patch: {patch_size}px"
    cv2.putText(
        result, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3
    )
    cv2.putText(result, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 1)

    # Highlight the GAP
    if stride > patch_size:
        cv2.putText(
            result,
            "GAPS VISIBLE (Non-adjacent)",
            (20, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            2,
        )
    else:
        cv2.putText(
            result,
            "ADJACENT (Full Coverage)",
            (20, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2,
        )

    vis_dir = config.VIS_DIR / "previews"
    vis_dir.mkdir(parents=True, exist_ok=True)
    out_path = vis_dir / f"{vsi_path.stem}_s{stride}_d{downsample_factor}.jpg"
    cv2.imwrite(str(out_path), result)

    print("\n  SUCCESS!")
    print(f"  Preview image saved to: {out_path}")
    print(
        f"  The green boxes represent the 224px model inputs (covering {raw_patch_size}px raw)."
    )
    print(f"  The gap between them is due to the {stride}px stride.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--vsi",
        type=str,
        help="Filename of the VSI to preview",
        default="001_1_HE_stack.vsi",
    )
    parser.add_argument("--dataset", type=str, default="ZStack_HE")
    parser.add_argument("--patch_size", type=int, default=224)
    parser.add_argument("--stride", type=int, default=448)
    parser.add_argument("--downsample", type=int, default=2)
    args = parser.parse_args()

    vsi_dir = config.get_vsi_raw_dir(args.dataset)
    vsi_path = vsi_dir / args.vsi

    if not vsi_path.exists():
        # Try to find any VSI if default doesn't exist
        vsi_files = list(vsi_dir.glob("*.vsi"))
        if vsi_files:
            vsi_path = vsi_files[0]
        else:
            print(f"No VSI files found in {vsi_dir}")
            exit(1)

    generate_preview(vsi_path, args.patch_size, args.stride, args.downsample)
