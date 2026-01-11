
import os
import glob
from PIL import Image

results_dir = "/home/niklas/focus_offset_public/experiments/focus_strategy/results/"
png_files = glob.glob(os.path.join(results_dir, "*.png"))

print(f"Found {len(png_files)} png files.")
for f in png_files:
    try:
        with Image.open(f) as img:
            img.verify()
        # Re-open to check if we can actually load data (verify only checks headers)
        with Image.open(f) as img:
            img.load()
    except Exception as e:
        print(f"ERROR: File {f} is corrupt. Reason: {e}")
print("Verification complete.")
