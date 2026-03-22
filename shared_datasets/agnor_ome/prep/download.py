import concurrent.futures
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import tifffile
from exact_sync.v1.api.images_api import ImagesApi
from exact_sync.v1.api_client import ApiClient
from exact_sync.v1.configuration import Configuration

from focus_offset.utils.exact_utils import get_exact_image_list


def verify_ome(ome_path: Path) -> bool:
    try:
        tif = tifffile.TiffFile(ome_path)
        if len(tif.series) == 0:
            return False
        # Read a small block from the first series and slice
        first_series = tif.series[0]
        w, h = (
            first_series.shape[1],
            first_series.shape[0] if first_series.ndim > 2 else first_series.shape[:2],
        )
        test_rect = (w // 2, h // 2, min(256, w), min(256, h))
        block = first_series.asarray()[
            test_rect[1] : test_rect[1] + test_rect[3],
            test_rect[0] : test_rect[0] + test_rect[2],
        ]
        tif.close()
        return block is not None and block.size > 0
    except Exception as e:
        print(f"       [FAIL] Integrity check failed for {ome_path.name}: {e}")
        return False


def fix_and_verify(ome_path: Path, temp_path: Path, slide_dir: Path) -> str:
    ome_name = ome_path.name

    # EXACT downloads single TIFF files directly
    if temp_path.exists() and not ome_path.exists():
        try:
            shutil.move(str(temp_path), str(ome_path))
        except Exception as e:
            return f"{ome_name}: [FAIL] Move failed: {e}"

    if ome_path.exists():
        if verify_ome(ome_path):
            if temp_path.exists():
                temp_path.unlink()
            return f"{ome_name}: [OK] Verified"
        else:
            ome_path.unlink(missing_ok=True)
            return f"{ome_name}: [FAIL] Corrupt - Deleted"
    return f"{ome_name}: [MISSING]"


def process_ome_image(img_info, slide_dir: Path):
    ome_name = img_info["name"]
    ome_path, temp_path = (
        slide_dir / ome_name,
        slide_dir / (ome_name + ".tmp"),
    )
    if not ome_path.exists() and not temp_path.exists():
        cfg = Configuration()
        cfg.username = os.environ.get("EXACT_USERNAME", "niklas.hargarter")
        cfg.password = os.environ.get("EXACT_PASSWORD")
        cfg.host = "https://exact.hs-flensburg.de"
        api = ImagesApi(ApiClient(cfg))
        try:
            api.download_image(id=img_info["id"], target_path=str(temp_path))
        except Exception as e:
            return f"{ome_name}: [FAIL] Download failed: {e}"
    return fix_and_verify(ome_path, temp_path, slide_dir)


def download_ome_dataset(
    full_name: str,
    slide_dir: Path,
    workers: int = 4,
    limit: Optional[int] = None,
    exclude_pattern: str = "_all_",
) -> None:
    slide_dir.mkdir(parents=True, exist_ok=True)
    if os.environ.get("EXACT_PASSWORD") is None:
        raise ValueError("Environment variable 'EXACT_PASSWORD' is not set.")
    print(f"Syncing {full_name} images from EXACT to {slide_dir}...")
    all_images = get_exact_image_list(dataset_name=full_name, force=False)
    if exclude_pattern:
        all_images = [
            img
            for img in all_images
            if exclude_pattern.lower() not in img["name"].lower()
        ]
    if limit:
        all_images = all_images[:limit]
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(process_ome_image, img, slide_dir) for img in all_images
        ]
        for future in concurrent.futures.as_completed(futures):
            try:
                print(f"  {future.result()}")
            except Exception as e:
                print(f"  Worker exception: {e}")
    print(f"\nAcquisition for {full_name} complete.")
