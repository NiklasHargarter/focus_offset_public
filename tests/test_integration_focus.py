import unittest
import cv2
import numpy as np
import slideio

import config
from src.dataset.vsi_datamodule import HEHoldOutDataModule
from src.utils.io_utils import suppress_stderr

def compute_brenner_gradient(image: np.ndarray) -> float:
    """Compute focus score using Brenner Gradient (local helper)."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = gray.astype(np.int32)
    shifted = np.roll(gray, -2, axis=1)
    return float(np.sum((gray - shifted) ** 2))


class TestIntegrationFocus(unittest.TestCase):
    def test_dataset_focus_consistency(self):
        """Integration: Verify Z-offsets and focus checks on actual dataset."""
        # Instantiate DataModule
        dm = HEHoldOutDataModule(
            dataset_name="ZStack_HE",
            batch_size=1,
            num_workers=0,
            patch_size=224,
            downscale_factor=16,
            min_tissue_coverage=0.05
        )

        try:
             dm.setup(stage="fit")
        except Exception as e:
             self.skipTest(f"DataModule setup failed (likely missing data): {e}")

        dataset = dm.train_dataset
        if dataset is None:
             self.skipTest("Train dataset is None after setup.")

        # Access file registry
        registry = dataset.file_registry
        if not registry:
            self.skipTest("Registry is empty.")

        # Pick first file and patch
        fmeta = registry[0]
        # Access dataclass fields directly
        fname = fmeta.path
        patches = fmeta.patches
        num_z = fmeta.num_z

        if not patches:
            self.skipTest("No patches in first file.")

        px, py, best_z = patches[0].x, patches[0].y, patches[0].z

        # Open slide for validation details
        if not fname.exists():
            self.skipTest(f"Slide file {fname} not found.")

        with suppress_stderr():
            slide = slideio.open_slide(str(fname), "VSI")
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
