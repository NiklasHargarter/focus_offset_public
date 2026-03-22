import argparse
import sys
from pathlib import Path

import slideio
from tabulate import tabulate


def inspect_slide(slide_path: str):
    """Prints comprehensive metadata for a VSI slide."""
    if not Path(slide_path).exists():
        print(f"Error: File not found: {slide_path}")
        sys.exit(1)

    try:
        slide = slideio.open_slide(str(slide_path), "VSI")
    except Exception as e:
        print(f"Error opening slide: {e}")
        sys.exit(1)

    num_scenes = slide.num_scenes
    print(f"Slide: {slide_path}")
    print(f"Number of scenes: {num_scenes}")
    print("-" * 40)

    for i in range(num_scenes):
        scene = slide.get_scene(i)
        name = scene.name
        width, height = scene.size
        num_z = scene.num_z_slices
        num_t = scene.num_t_frames
        num_channels = scene.num_channels

        # Pixels to Microns conversion (usually slideio returns resolution in meters/pixel)
        res_x, res_y = scene.resolution
        microns_per_px_x = res_x * 1e6 if res_x else 0
        microns_per_px_y = res_y * 1e6 if res_y else 0

        z_res = getattr(scene, "z_resolution", 0)
        z_res_microns = z_res * 1e6 if z_res else 0

        magnification = scene.magnification

        metadata = [
            ["Scene Name", name],
            ["Dimensions (WxH)", f"{width} x {height}"],
            ["Z Levels", num_z],
            ["T Frames", num_t],
            ["Channels", num_channels],
            ["Magnification", f"{magnification}x" if magnification else "N/A"],
            ["Pixel Size (X)", f"{microns_per_px_x:.4f} µm/px"],
            ["Pixel Size (Y)", f"{microns_per_px_y:.4f} µm/px"],
            ["Z Step Size", f"{z_res_microns:.4f} µm" if z_res_microns else "N/A"],
        ]

        print(f"Scene {i}:")
        print(tabulate(metadata, tablefmt="simple"))

        # Pyramid Levels
        num_levels = scene.num_zoom_levels
        if num_levels > 0:
            print("\nPyramid Levels:")
            level_data = []
            for lvl in range(num_levels):
                lvl_info = scene.get_zoom_level_info(lvl)
                l_w, l_h = lvl_info.size.width, lvl_info.size.height
                l_mag = lvl_info.magnification
                scale = width / l_w
                level_data.append(
                    [lvl, f"{l_w} x {l_h}", f"{l_mag:.1f}x", f"{scale:.1f}x"]
                )

            print(
                tabulate(
                    level_data,
                    headers=["Level", "Dimensions", "Magnification", "Downscale"],
                    tablefmt="simple",
                )
            )

        # Additional raw properties if needed
        # raw_props = scene.get_raw_metadata()
        # if raw_props:
        #     # Optional: print raw metadata if requested via a flag
        #     pass

        print("-" * 40)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect a VSI slide's metadata.")
    parser.add_argument("slide_path", type=str, help="Path to the VSI file.")
    args = parser.parse_args()

    inspect_slide(args.slide_path)
