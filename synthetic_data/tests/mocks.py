"""Shared mock objects for synthetic_data test suite."""

import numpy as np
from synthetic_data.dataset import SyntheticVSIDataset


class MockScene:
    def __init__(self, width: int, height: int, num_z: int, z_res_microns: float):
        self.size = (width, height)
        self.num_z_slices = num_z
        self.z_resolution = z_res_microns / 1e6

    def read_block(self, rect, size, slices):
        sw, sh = size
        z_start, _ = slices
        arr = np.ones((sh, sw, 3), dtype=np.uint8) * 100
        if z_start == 3:  # "Optimal" Z plane
            arr[sh // 4 : 3 * sh // 4, sw // 4 : 3 * sw // 4, :] = 200
        else:
            arr[sh // 4 : 3 * sh // 4, sw // 4 : 3 * sw // 4, :] = 150
        return arr


class MockSyntheticVSIDataset(SyntheticVSIDataset):
    def _get_scene(self, vsi_filename: str):
        if self._slides is None:
            self._slides = {}
        if vsi_filename not in self._slides:
            scene = MockScene(1000, 1000, 10, 1.0)
            self._slides[vsi_filename] = (None, scene)
        return self._slides[vsi_filename][1]
