import slideio
from src import config


def inspect_pyramid(vsi_path):
    print(f" Inspecting VSI Pyramid: {vsi_path}\n")

    # Open the slide
    slide = slideio.open_slide(vsi_path, "VSI")
    scene = slide.get_scene(0)

    # Basic Scene Info
    print(f"  Scene Name: {scene.name}")
    print(f"  Base Resolution: {scene.size} (Width x Height)")
    print(f"  Native Magnification: {scene.magnification}x")
    print(f"  Pixel Size (x, y): {scene.resolution} meters")
    print(f"  Z-slices: {scene.num_z_slices}")
    print(f"\n  Pyramid Levels: {scene.num_zoom_levels}")
    print("-" * 60)
    print(
        f"  {'Lvl':<4} | {'Width':<10} | {'Height':<10} | {'Scale factor':<12} | {'Eff. Mag':<10} | {'Tile Size':<10}"
    )
    print("-" * 60)

    base_width = scene.size[0]

    # Iterate through all zoom levels
    for i in range(scene.num_zoom_levels):
        info = scene.get_zoom_level_info(i)

        # Calculate scale factor relative to level 0
        scale = base_width / info.size.width

        # Estimate effective magnification (assuming base is native)
        # Note: scene.magnification might be 0.0 if not stored in metadata,
        # so we'll handle that gracefully
        native_mag = scene.magnification if scene.magnification > 0 else 80.0
        eff_mag = native_mag / scale

        print(
            f"  {i:<4} | {info.size.width:<10} | {info.size.height:<10} | {scale:<12.2f} | {eff_mag:<10.1f} | {info.tile_size.width}x{info.tile_size.height}"
        )

    print("-" * 60)


if __name__ == "__main__":
    # Default to first file in the HE raws directory
    raw_dir = config.get_vsi_raw_dir("ZStack_HE")
    vsi_files = list(raw_dir.glob("*.vsi"))

    if vsi_files:
        vsi_file = vsi_files[0]
        inspect_pyramid(str(vsi_file))
    else:
        print(f"No VSI files found in {raw_dir}")
