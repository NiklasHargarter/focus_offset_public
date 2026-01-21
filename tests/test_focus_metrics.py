import unittest
import numpy as np
from src.utils.focus_metrics import compute_brenner_gradient


class TestFocusMetrics(unittest.TestCase):
    def test_brenner_overflow_handling(self):
        """
        Verify that compute_brenner_gradient correctly handles large images without overflow.
        An int32 overflow occurs around 2.1 billion.
        """
        # Create a large image: 2048 x 2048
        # We'll use a striped pattern: [0, 0, 255, 255, 0, 0, 255, 255, ...]
        # This ensures every (x) vs (x+2) comparison is 255 - 0 = 255
        h, w = 2048, 2048
        image = np.zeros((h, w), dtype=np.uint8)

        # Fill alternating columns so that G(x+2) - G(x) is always 255
        # Column 2, 3, 6, 7, ... = 255
        # Column 0, 1, 4, 5, ... = 0
        for x in range(0, w):
            if (x // 2) % 2 == 1:
                image[:, x] = 255

        # Calculation:
        # Each row has (w-2) valid comparisons.
        # w-2 = 2046 comparisons per row.
        # Each comparison is (255 - 0)^2 = 65025.
        # Total rows = 2048.
        expected_total = float(h * (w - 2) * (255**2))

        # This number should be ~272 Billion (2.7e11)
        # int32 overflow limit is ~2.1 Billion (2.1e9)
        self.assertGreater(
            expected_total, 2**31, "The test case itself should exceed int32 limits."
        )

        # Execute
        actual_score = compute_brenner_gradient(image)

        print(f"\nExpected: {expected_total}")
        print(f"Actual:   {actual_score}")

        self.assertEqual(
            actual_score,
            expected_total,
            f"Score mismatch! Likely overflow. Expected {expected_total}, got {actual_score}",
        )

    def test_brenner_consistency(self):
        """Test basic functionality on a small known patch."""
        patch = np.array([[10, 10, 20, 20], [10, 10, 20, 20]], dtype=np.uint8)

        # (20-10)^2 + (20-10)^2 = 100 + 100 per row
        # 2 rows * 200 = 400
        # Valid pairs in each row:
        # p[0, 2] - p[0, 0] = 20 - 10 = 10
        # p[0, 3] - p[0, 1] = 20 - 10 = 10

        expected = float(2 * (10**2 + 10**2))
        self.assertEqual(compute_brenner_gradient(patch), expected)


if __name__ == "__main__":
    unittest.main()
