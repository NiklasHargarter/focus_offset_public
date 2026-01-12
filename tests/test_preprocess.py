import unittest
from unittest.mock import MagicMock
import numpy as np

# Update imports to match the current class-based structure
from src.dataset.vsi_prep.preprocess import SlidePreprocessor
from src.dataset.vsi_types import PreprocessConfig


class TestImageMetrics(unittest.TestCase):
    def setUp(self):
        # Create a mock config to instantiate SlidePreprocessor
        self.config = PreprocessConfig(
            patch_size=100,
            downscale_factor=10,
            min_tissue_coverage=0.05,
            dataset_name="test_dataset",
            focus_patch_size=1000
        )
        self.processor = SlidePreprocessor(self.config)

    def test_compute_focus_score_simple_edge(self):
        """Test Brenner Gradient on a simple known gradient."""
        # Create a small 1x6 BGR image (using 6 columns due to shift-2 logic)
        img = np.zeros((1, 6, 3), dtype=np.uint8)

        # Set a pattern to produce known edges
        img[0, 0:2] = 10
        img[0, 2:4] = 20  # Edge UP
        img[0, 4:6] = 10  # Edge DOWN

        # Expected Sum of Squared Diff (Shift 2):
        # Gray conversion usually maintains values if channels are equal, but here BGR=(0,0,0) -> 0, etc.
        # Let's use grayscale directly to avoid OpenCV conversion ambiguity in test setup if possible,
        # but the method takes BGR. 
        # cv2.cvtColor(BGR) where B=G=R sends the same value to Y.
        # So 10 -> 10, 20 -> 20.
        
        # Row: [10, 10, 20, 20, 10, 10]
        # Shifted (-2): [20, 20, 10, 10, 10, 10] (wrapped)
        # Diff: [-10, -10, 10, 10, 0, 0]
        # Squared: [100, 100, 100, 100, 0, 0]
        # Sum: 400
        
        score = self.processor._compute_focus_score(img)
        self.assertEqual(score, 400)

    def test_compute_focus_score_flat(self):
        """Test that a flat image has 0 gradient."""
        img = np.full((10, 10, 3), 50, dtype=np.uint8)
        score = self.processor._compute_focus_score(img)
        self.assertEqual(score, 0)


class TestMaskProcessing(unittest.TestCase):
    def setUp(self):
        self.width = 1000
        self.height = 1000
        self.patch_size = 100
        self.downscale = 10
        self.config = PreprocessConfig(
            patch_size=self.patch_size,
            downscale_factor=self.downscale,
            min_tissue_coverage=0.05,
            dataset_name="test_dataset",
            focus_patch_size=1000
        )
        self.processor = SlidePreprocessor(self.config)
        # Mock dimensions on the processor since it normally sets them from slide metadata
        self.processor.width = self.width
        self.processor.height = self.height

    def test_get_valid_indices(self):
        """Test correct identification of patches based on tissue coverage."""
        d_w, d_h = 100, 100
        mask = np.zeros((d_h, d_w), dtype=np.uint8)

        # 100% coverage at (0,0) -> ds (0,0) to (10,10)
        # y: 0-100 -> ds 0-10
        # x: 0-100 -> ds 0-10
        mask[0:10, 0:10] = 255
        
        # 10% coverage at (200, 0) -> ds (20,0) to (30,10)
        # This zone has 10x10=100 pixels. 10% is 10 pixels.
        # Let's fill 10 pixels in this block.
        mask[0, 20:30] = 255
        
        # 0% coverage at (400,0) -> ds (40,0) to (50,10)
        # Leave 0

        valid_patches = self.processor._get_valid_indices(mask, d_w, d_h)

        self.assertIn((0, 0), valid_patches)  # 100%
        self.assertIn((200, 0), valid_patches)  # 10% (>= 5%)
        self.assertNotIn((400, 0), valid_patches)  # 0%


class TestFocusSelection(unittest.TestCase):
    def setUp(self):
        self.config = PreprocessConfig(
            patch_size=100,
            downscale_factor=10,
            min_tissue_coverage=0.05,
            dataset_name="test_dataset",
            focus_patch_size=1000
        )
        self.processor = SlidePreprocessor(self.config)

    def test_find_best_global_z(self):
        """
        Test that the function picks the slice with the highest Score (Laplacian for global z).
        Note: The implementation uses Laplacian for global Z selection, not Brenner.
        """
        # Mock the slideio scene
        mock_scene = MagicMock()
        mock_scene.size = (100, 100)
        
        d_w, d_h = 10, 10
        num_z = 3

        # Create dummy images (BGR)
        # Gray: 0 -> Variance 0 -> Score 0
        img_blurry = np.zeros((10, 10, 3), dtype=np.uint8) 

        # High variance image
        img_sharp = np.zeros((10, 10, 3), dtype=np.uint8)
        # Random noise or checkerboard used to create high Laplacian variance
        for i in range(10):
            for j in range(10):
                if (i+j) % 2 == 0:
                    img_sharp[i, j] = 255

        # Define side effect to return distinct images based on slices
        def side_effect(rect, size, slices):
            z_start = slices[0]
            if z_start == 1:
                return img_sharp
            return img_blurry

        mock_scene.read_block.side_effect = side_effect

        best_z, best_img, mask = self.processor._find_best_global_z(
            mock_scene, d_w, d_h, num_z
        )

        self.assertEqual(best_z, 1)
        # Verify mask generation as well (Otsu on sharp image)
        # Since sharp image is half 0 half 255, it should produce a mask.
        self.assertTrue(np.any(mask > 0))


if __name__ == "__main__":
    unittest.main()
