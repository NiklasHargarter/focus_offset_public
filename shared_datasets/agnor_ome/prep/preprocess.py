import cv2
import numpy as np
from PIL import Image
from histolab.filters.image_filters import Compose, OtsuThreshold, RgbToGrayscale
import tifffile
from pathlib import Path

from focus_offset.utils.focus_metrics import compute_focus_score


def detect_tissue_mask(thumbnail_rgb: np.ndarray) -> np.ndarray:
    pipeline = Compose([RgbToGrayscale(), OtsuThreshold()])
    bool_mask = pipeline(Image.fromarray(thumbnail_rgb))
    return (bool_mask * 255).astype(np.uint8)


def generate_tissue_patches(
    width: int, height: int, params: dict, mask: np.ndarray
) -> list[tuple[int, int, float]]:
    patch_size_raw = params["patch_size"] * params.get("downsample", 1)
    stride = params["stride"]
    mask_downscale = 8  # hardcoded as per user request for simplicity

    xs = range(0, width - patch_size_raw + 1, stride)
    ys = range(0, height - patch_size_raw + 1, stride)

    mw, mh = patch_size_raw // mask_downscale, patch_size_raw // mask_downscale
    binary = mask > 0

    results = []
    for y in ys:
        for x in xs:
            patch = binary[
                y // mask_downscale : y // mask_downscale + mh,
                x // mask_downscale : x // mask_downscale + mw,
            ]
            if patch.size > 0:
                coverage = float(patch.mean())
                if coverage >= params.get("cov", 0.2):
                    results.append((x, y, coverage))
    return results


class OMEImageReader:
    """Uses tifffile to read multi-series OME-TIFFs as Z-stacks."""

    def __init__(self, path: Path):
        self.path = path
        self._tif = tifffile.TiffFile(path)
        self.num_z = len(self._tif.series)
        first_series = self._tif.series[0]
        self.height, self.width = first_series.shape[:2]
        self.size = (self.width, self.height)

    def read_block(
        self, rect: tuple[int, int, int, int], slices: tuple[int, int] | None = None
    ) -> np.ndarray:
        x, y, w, h = rect
        z_start, z_end = slices if slices else (0, self.num_z)

        stack = []
        for z in range(z_start, z_end):
            data = self._tif.series[z].asarray()
            if data.ndim == 2:
                data = np.expand_dims(data, axis=-1)
            stack.append(data[y : y + h, x : x + w])

        return np.stack(stack)

    def close(self):
        self._tif.close()


def read_thumbnail_rgb(
    reader: OMEImageReader, width: int, height: int, mask_downscale: int
) -> np.ndarray:
    """Read a small RGB thumbnail from the first z-slice."""
    d_w = width // mask_downscale
    d_h = height // mask_downscale
    full_img = reader.read_block((0, 0, width, height), (0, 1))[0]
    small = cv2.resize(full_img, (d_w, d_h))
    return cv2.cvtColor(small, cv2.COLOR_BGR2RGB)


def process_ome_slide(
    slide_path: str, params: dict, dry_run: bool = False
) -> list[dict]:
    img_path = Path(slide_path)
    patch_size = params["patch_size"]
    downsample = params.get("downsample", 1)
    raw_extent = patch_size * downsample
    mask_downscale = 8

    reader = OMEImageReader(img_path)
    width, height = reader.size
    num_z = reader.num_z

    # Calculate focus offset dynamically from OME metadata
    try:
        ome_metadata = reader._tif.ome_metadata
        import xml.etree.ElementTree as ET

        root = ET.fromstring(ome_metadata)
        ns = {"ome": "http://www.openmicroscopy.org/Schemas/OME/2016-06"}
        pixels = root.find(".//ome:Pixels", ns)
        if pixels is not None and "PhysicalSizeZ" in pixels.attrib:
            z_res_microns = float(pixels.attrib["PhysicalSizeZ"])
        else:
            z_res_microns = 1.0  # Fallback
    except Exception:
        z_res_microns = 1.0

    thumbnail = read_thumbnail_rgb(reader, width, height, mask_downscale)
    mask = detect_tissue_mask(thumbnail_rgb=thumbnail)
    candidates = generate_tissue_patches(width, height, params, mask)

    if not candidates:
        reader.close()
        return []

    if dry_run:
        import random

        random.shuffle(candidates)
        candidates = candidates[:20]

    all_scores = []
    best_zs = np.zeros(len(candidates), dtype=np.int32)

    for i, (x, y, _cov) in enumerate(candidates):
        if i % 100 == 0:
            print(f"[{img_path.name}] focus {i}/{len(candidates)}")

        z_stack = reader.read_block(
            rect=(int(x), int(y), raw_extent, raw_extent), slices=(0, num_z)
        )

        if z_stack.ndim == 4 and z_stack.shape[-1] == 1:
            z_stack = z_stack.squeeze(-1)

        scores = [
            float(compute_focus_score(cv2.resize(z_stack[z], (patch_size, patch_size))))
            for z in range(num_z)
        ]
        all_scores.append(scores)
        best_zs[i] = int(np.argmax(scores))

    reader.close()

    rows = []
    slide_name = img_path.name
    for (x, y, cov), best_z, scores in zip(candidates, best_zs, all_scores):
        for z in range(num_z):
            rows.append(
                {
                    "slide_name": slide_name,
                    "x": int(x),
                    "y": int(y),
                    "z_level": int(z),
                    "optimal_z": int(best_z),
                    "num_z": int(num_z),
                    "z_res_microns": z_res_microns,
                    "z_offset_microns": (int(best_z) - z) * z_res_microns,
                    "focus_score": scores[z],
                    "max_focus_score": max(scores),
                    "tissue_coverage": cov,
                }
            )
    return rows
