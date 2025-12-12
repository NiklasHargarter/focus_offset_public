import unittest
import numpy as np
import cv2
import sys
import os

# Add src to path to import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from preprocess import (
    compute_brenner_gradient,
    find_valid_patches,
    select_best_focus_slice,
    generate_tissue_mask,
)
from unittest.mock import MagicMock, patch


class TestImageMetrics(unittest.TestCase):
    def test_brenner_gradient_simple_edge(self):
        """Test Brenner Gradient on a simple known gradient."""
        # Create a small 1x4 BGR image
        img = np.zeros((1, 6, 3), dtype=np.uint8)

        # Set a pattern to produce known edges
        img[0, 0:2] = 10
        img[0, 2:4] = 20  # Edge UP
        img[0, 4:6] = 10  # Edge DOWN

        # Expected Sum of Squared Diff (Shift 2): 400
        score = compute_brenner_gradient(img)
        self.assertEqual(score, 400)

    def test_brenner_gradient_flat(self):
        """Test that a flat image has 0 gradient."""
        img = np.full((10, 10, 3), 50, dtype=np.uint8)
        score = compute_brenner_gradient(img)
        self.assertEqual(score, 0)

    def test_brenner_gradient_overflow_safety(self):
        """Test that uint8 overflow is handled correctly (int32 casting)."""
        img = np.zeros((1, 3, 3), dtype=np.uint8)
        img[0, 0] = 10
        img[0, 1] = 10
        img[0, 2] = 200  # Large difference (190)

        # Should be 2 * (10-200)^2 = 72200
        score = compute_brenner_gradient(img)
        self.assertEqual(score, 72200)


class TestMaskProcessing(unittest.TestCase):
    def test_find_valid_patches(self):
        """Test correct identification of patches based on tissue coverage."""
        width, height = 1000, 1000
        patch_size, stride = 100, 100
        downscale_factor = 10
        down_w, down_h = 100, 100

        mask = np.zeros((down_h, down_w), dtype=np.uint8)

        # 100% coverage
        mask[0:10, 0:10] = 255
        # 10% coverage (valid)
        mask[0, 20:30] = 255
        # 1% coverage (invalid)
        mask[0, 30] = 255

        with patch("preprocess.config") as mock_config:
            mock_config.MIN_TISSUE_COVERAGE = 0.05

            valid_patches = find_valid_patches(
                mask,
                width,
                height,
                patch_size,
                stride,
                downscale_factor,
                down_w,
                down_h,
            )

        self.assertIn((0, 0), valid_patches)  # 100%
        self.assertNotIn((100, 0), valid_patches)  # 0%
        self.assertIn((200, 0), valid_patches)  # 10%
        self.assertNotIn((300, 0), valid_patches)  # 1%


class TestFocusSelection(unittest.TestCase):
    def test_select_best_focus_slice(self):
        """
        Test that the function picks the slice with the highest Brenner score.
        """
        # Mock the slideio scene
        mock_scene = MagicMock()

        # Setup dims
        width, height = 100, 100
        down_w, down_h = 10, 10
        num_z = 3

        # We need to control what read_block returns for different Z
        # z=0: blurry image (low score)
        # z=1: sharp image (high score)
        # z=2: blurry image (low score)

        # Create dummy images (BGR)
        img_blurry = np.zeros((10, 10, 3), dtype=np.uint8)  # Flat -> Score 0

        img_sharp = np.zeros((10, 10, 3), dtype=np.uint8)
        # Use simple vertical bar (avoid period-2 aliasing with Brenner)
        img_sharp[:, 4:6] = 255

        # Define side effect to return distinct images based on 'slices' arg
        def side_effect(rect, size, slices):
            z_start = slices[0]
            if z_start == 1:
                return img_sharp
            return img_blurry

        mock_scene.read_block.side_effect = side_effect

        best_img = select_best_focus_slice(
            mock_scene, width, height, num_z, down_w, down_h
        )

        # The best image should be the grayscale version of img_sharp
        expected_gray = cv2.cvtColor(img_sharp, cv2.COLOR_BGR2GRAY)

        np.testing.assert_array_equal(best_img, expected_gray)


class TestTissueMasking(unittest.TestCase):
    def test_identifies_dark_tissue(self):
        """Test that dark pixels are correctly identified as tissue (255)."""
        # Create synthetic image: 10x10
        # Left half = 0 (Dark/Tissue)
        # Right half = 255 (Bright/Background)
        img = np.zeros((10, 10), dtype=np.uint8)
        img[:, 0:5] = 0
        img[:, 5:10] = 255

        # Expected Mask: Left half 255, Right half 0
        expected_mask = np.zeros((10, 10), dtype=np.uint8)
        expected_mask[:, 0:5] = 255

        mask = generate_tissue_mask(img)

        np.testing.assert_array_equal(mask, expected_mask)

    def test_none_input(self):
        """Test graceful handling of None input."""
        res = generate_tissue_mask(None)
        self.assertIsNone(res)


if __name__ == "__main__":
    unittest.main()
