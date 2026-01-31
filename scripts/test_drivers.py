import slideio
import os
from pathlib import Path

f = next((Path.home() / "AgNor" / "raws").glob("*.ome.tiff"))
drivers = ['AFI', 'CZI', 'DCM', 'GDAL', 'NDPI', 'QPTIFF', 'SCN', 'SVS', 'VSI', 'ZVI']

print(f"File: {f.name}")
for d in drivers:
    try:
        s = slideio.open_slide(str(f), d)
        print(f"{d}: {s.num_scenes} scenes")
        for i in range(s.num_scenes):
            scene = s.get_scene(i)
            print(f"  Scene {i}: {scene.size}, Z: {scene.num_z_slices}, T: {scene.num_t_frames}")
    except Exception:
        print(f"{d}: Driver Error or Not Supported")
