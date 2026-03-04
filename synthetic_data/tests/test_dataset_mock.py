import pandas as pd
from pathlib import Path
import torch
import numpy as np
from synthetic_data.dataset import SyntheticVSIDataset


# We will mock the slideio scene since the VSI loading gets stuck/takes too long
class MockScene:
    def __init__(self, width, height, num_z, z_res_microns):
        self.size = (width, height)
        self.num_z_slices = num_z
        self.z_resolution = z_res_microns / 1e6

    def read_block(self, rect, size, slices):
        x, y, w, h = rect
        sw, sh = size
        z_start, z_end = slices

        # Return a dummy numpy array of shape (H, W, Channels)
        # Note: slideio typically returns HWC images (RGB)
        return np.ones((sh, sw, 3), dtype=np.uint8) * 128


class MockSyntheticVSIDataset(SyntheticVSIDataset):
    def _get_scene(self, vsi_filename: str):
        # Override to return our dummy scene
        if self._slides is None:
            self._slides = {}

        if vsi_filename not in self._slides:
            print(f"Mocking scene for {vsi_filename}...")
            # create a big dummy slide
            scene = MockScene(100000, 100000, 10, 1.0)
            self._slides[vsi_filename] = (None, scene)

        return self._slides[vsi_filename][1]


def test_dataset_mock():
    # Make a dummy df
    mock_data = [
        {
            "slide_name": "dummy_slide.vsi",
            "x": 10000,
            "y": 10000,
            "optimal_z": 3,
            "num_z": 10,
            "z_res_microns": 1.0,
            "max_focus_score": 100.0,
            "tissue_coverage": 1.0,
        }
    ]
    df = pd.DataFrame(mock_data)

    N = 1024
    K = 256
    offset = 2

    print(f"Testing Mock SyntheticVSIDataset with N={N}, K={K}, offset={offset}")

    try:
        # Pass any dummy path for slide_dir as it won't be hit with our mock
        dataset = MockSyntheticVSIDataset(
            index_df=df,
            slide_dir=Path("/dummy"),
            patch_size_n=N,
            patch_size_k=K,
            z_offset_steps=offset,
            downsample=2,
        )

        sample = dataset[0]
        opt_patch = sample["optimal_patch"]
        off_patch = sample["offset_patch"]
        meta = sample["metadata"]
        z_microns = sample["z_offset_microns"]
        z_actual = sample["z_offset_actual"]

        print("\n--- Success! ---")
        print(f"Optimal Patch Shape: {opt_patch.shape}")
        print(f"Offset Patch Shape: {off_patch.shape}")
        print(f"Z Offset Actual Output: {z_actual} steps")
        print(f"Z Offset Microns Output: {z_microns} um")
        print(f"Metadata Dict: {meta}")

        assert opt_patch.shape == (3, N, N), (
            f"Expected {(3, N, N)}, got {opt_patch.shape}"
        )
        assert off_patch.shape == (3, K, K), (
            f"Expected {(3, K, K)}, got {off_patch.shape}"
        )
        assert meta["optimal_z"] == 3
        assert meta["offset_z"] == 5
        assert z_microns == 2.0

        print("\nAll assertions passed natively.")

    except Exception as e:
        print(f"\n--- Error ---")
        print(e)
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    test_dataset_mock()
