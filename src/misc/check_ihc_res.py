import slideio
import os
from src.utils.io_utils import suppress_stderr


def check_ihc_resolution():
    ihc_file = "/home/niklas/ZStack_IHC/raws/001_1_IHC_stack.vsi"
    if not os.path.exists(ihc_file):
        print(f"File {ihc_file} not found.")
        return

    print(f"Checking {ihc_file}...")
    with suppress_stderr():
        slide = slideio.open_slide(ihc_file, "VSI")
        scene = slide.get_scene(0)

    z_res = getattr(scene, "z_resolution", 0)
    print(f"Z-Resolution (raw): {z_res}")
    print(f"Z-Resolution (microns): {z_res * 1e6}")


if __name__ == "__main__":
    check_ihc_resolution()
