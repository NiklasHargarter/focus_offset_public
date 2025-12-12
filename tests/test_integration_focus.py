import unittest
import os
import sys
import slideio

# Ensure src is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src import config
from src.vsi_dataset import VSIDataset
from src.io_utils import suppress_stderr
from src.preprocess import compute_brenner_gradient


class TestIntegrationFocus(unittest.TestCase):
    def test_dataset_focus_consistency(self):
        """Integration: Verify Z-offsets and focus checks on actual dataset."""
        index_path = os.path.join(config.CACHE_DIR, "dataset_index_data_train.pkl")

        if not os.path.exists(index_path):
            self.skipTest(
                f"Dataset index not found at {index_path}. Run preprocess first."
            )

        dataset = VSIDataset(index_path)
        registry = dataset.index["file_registry"]

        if not registry:
            self.skipTest("Registry is empty.")

        # Pick first file and patch
        fmeta = registry[0]
        fname = fmeta["path"]
        patches = fmeta["patches"]
        num_z = fmeta["num_z"]

        if not patches:
            self.skipTest("No patches in first file.")

        px, py, best_z = patches[0]

        # Open slide for validation details
        if not os.path.exists(fname):
            self.skipTest(f"Slide file {fname} not found.")

        with suppress_stderr():
            slide = slideio.open_slide(fname, "VSI")
        scene = slide.get_scene(0)
        z_res_microns = scene.z_resolution * 1e6

        # Verify Z-offsets for start, best, and end slices
        z_indices = sorted(list(set([0, best_z, num_z - 1])))

        for z in z_indices:
            # Flattened dataset: index 0..num_z-1 corresponds to first patch's Z-stack
            idx = z

            img, offset_tensor = dataset[idx]
            offset = offset_tensor.item()
            expected = float(best_z - z) * z_res_microns

            # Allow small float error
            self.assertAlmostEqual(
                offset,
                expected,
                delta=0.001,
                msg=f"Offset mismatch at Z={z}. Got {offset}, Expected {expected}",
            )

        # Verify 'Best Z' has better focus than Z=0 (if widely separated)
        rect = (px, py, config.PATCH_SIZE, config.PATCH_SIZE)
        score_best = compute_brenner_gradient(
            scene.read_block(rect=rect, slices=(best_z, best_z + 1))
        )
        score_far = compute_brenner_gradient(scene.read_block(rect=rect, slices=(0, 1)))

        if best_z > 5:
            self.assertGreater(
                score_best,
                score_far,
                msg=f"Best Z={best_z} (score={score_best}) should be sharper than Z=0 (score={score_far})",
            )


if __name__ == "__main__":
    unittest.main()
