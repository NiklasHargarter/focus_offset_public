import slideio
import cv2
import numpy as np
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from skimage.filters import threshold_otsu  # noqa: E402
import config  # noqa: E402
from src.utils.io_utils import suppress_stderr  # noqa: E402
from src.dataset.vsi_prep.preprocess import compute_brenner_gradient  # noqa: E402
from src.dataset.vsi_dataset import VSIDataset  # noqa: E402


def analyze_otsu_stability(vsi_path, downscale=config.DOWNSCALE_FACTOR):
    try:
        with suppress_stderr():
            slide = slideio.open_slide(str(vsi_path), "VSI")
        scene = slide.get_scene(0)
        width, height = scene.size
        num_z = scene.num_z_slices

        down_w = width // downscale
        down_h = height // downscale

        print(f"\nAnalyzing {Path(vsi_path).name} (Z={num_z})...")

        results = []

        for z in range(num_z):
            img = scene.read_block(
                rect=(0, 0, width, height), size=(down_w, down_h), slices=(z, z + 1)
            )
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            thresh_val = threshold_otsu(gray)
            focus = compute_brenner_gradient(img)
            results.append((z, thresh_val, focus))

        # Stats
        thresholds = [r[1] for r in results]
        focus_scores = [r[2] for r in results]

        avg_thresh = np.mean(thresholds)
        std_thresh = np.std(thresholds)
        min_thresh = np.min(thresholds)
        max_thresh = np.max(thresholds)

        best_focus_idx = np.argmax(focus_scores)
        best_z = results[best_focus_idx][0]
        best_z_thresh = results[best_focus_idx][1]

        z0_thresh = results[0][1]

        print(
            f"  Threshold Stats: Mean={avg_thresh:.2f}, Std={std_thresh:.2f}, Range=[{min_thresh}, {max_thresh}]"
        )
        print(f"  Z=0 Threshold: {z0_thresh}")
        print(
            f"  Best Focus Z={best_z} (Score={focus_scores[best_focus_idx]:.1f}), Threshold: {best_z_thresh}"
        )
        print(f"  Diff (Best Focus - Z0): {best_z_thresh - z0_thresh}")

        # Check specific values
        print("  Z-Level | Threshold | Focus Score")
        # Print first few, best, and last
        indices_to_show = sorted(list(set([0, 1, 2, best_z, num_z - 1])))
        for idx in indices_to_show:
            if idx < num_z:
                r = results[idx]
                print(f"  {r[0]:<7} | {r[1]:<9.1f} | {r[2]:.1f}")

        return std_thresh, abs(best_z_thresh - z0_thresh)

    except Exception as e:
        print(f"Error analyzing {vsi_path}: {e}")
        return None, None


def main():
    # Ensure index exists
    index_path = config.get_index_path("train")

    if not index_path.exists():
        print(f"Index {index_path} not found. Run preprocess.py first.")
        return

    dataset = VSIDataset(mode="train")

    # Extract unique files from dataset
    unique_files = sorted(list(set([f["path"] for f in dataset.file_registry])))

    for f in unique_files:
        analyze_otsu_stability(f)


if __name__ == "__main__":
    main()
