import unittest
from unittest.mock import MagicMock, patch
import numpy as np
from pathlib import Path

from src.dataset.vsi_prep.preprocess import (
    detect_tissue,
    generate_patch_candidates,
    find_best_z,
    SlidePreprocessor,
)
from src.dataset.vsi_types import PreprocessConfig


class TestPreprocessingFunctions(unittest.TestCase):
    def test_detect_tissue(self):
        mock_scene = MagicMock()
        mock_scene.size = (10, 10)
        mock_scene.num_z_slices = 1

        img = np.zeros((10, 10, 3), dtype=np.uint8)
        img[:, 5:] = 255
        mock_scene.read_block.return_value = img

        best_z, mask = detect_tissue(mock_scene, downscale=1)

        self.assertEqual(best_z, 0)
        self.assertEqual(mask.shape, (10, 10))
        self.assertTrue(np.any(mask == 0))
        self.assertTrue(np.any(mask == 255))

    def test_generate_patch_candidates(self):
        mask = np.zeros((20, 20), dtype=np.uint8)
        mask[0:10, 0:6] = 255

        candidates = generate_patch_candidates(
            mask,
            width=20,
            height=10,
            patch_size=10,
            binning=1,
            downscale=1,
            min_cov=0.5,
        )

        self.assertEqual(len(candidates), 1)
        cx, cy, ox, oy = candidates[0]
        self.assertEqual(ox, 0)
        self.assertEqual(cx, 5)

    def test_find_best_z(self):
        mock_scene = MagicMock()
        mock_scene.num_z_slices = 2

        def side_effect(rect, size, slices):
            z = slices[0]
            img = np.zeros((10, 10, 3), dtype=np.uint8)
            if z == 1:
                img[:, :2] = 255
            return img

        mock_scene.read_block.side_effect = side_effect

        best_z = find_best_z(mock_scene, cx=5, cy=5, focus_size=10, calc_size=10)
        self.assertEqual(best_z, 1)


class TestSlidePreprocessor(unittest.TestCase):
    def test_orchestration(self):
        config = PreprocessConfig(
            patch_size=10,
            downscale_factor=1,
            min_tissue_coverage=0.5,
            dataset_name="test",
            focus_patch_size=10,
        )
        processor = SlidePreprocessor(config)

        mock_slide = MagicMock()
        mock_scene = MagicMock()
        mock_slide.get_scene.return_value = mock_scene
        mock_scene.size = (10, 10)
        mock_scene.num_z_slices = 1

        img = np.zeros((10, 10, 3), dtype=np.uint8)
        img[:, 5:] = 255
        mock_scene.read_block.return_value = img

        with patch("slideio.open_slide", return_value=mock_slide):
            metadata = processor.process(Path("dummy.vsi"))
            self.assertEqual(len(metadata.patches), 1)


if __name__ == "__main__":
    unittest.main()
