import tifffile
from pathlib import Path

f = next((Path.home() / 'AgNor' / 'raws').glob('*.ome.tiff'))
with tifffile.TiffFile(f) as tif:
    print(f'File: {f.name}')
    print(f'Num series: {len(tif.series)}')
    for i, s in enumerate(tif.series):
        # Read a small part of each series to make sure it's readable
        data = s.asarray() # This might be slow if large, but these are small (2k x 2k)
        print(f'Series {i}: shape={s.shape}, dtype={data.dtype}, mean={data.mean():.2f}')
