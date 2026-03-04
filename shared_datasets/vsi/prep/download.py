import concurrent.futures
import os
import shutil
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import slideio
from exact_sync.v1.api.images_api import ImagesApi
from exact_sync.v1.api_client import ApiClient
from exact_sync.v1.configuration import Configuration

from focus_offset.utils.exact_utils import get_exact_image_list
from focus_offset.utils.io_utils import suppress_stderr


def verify_vsi(vsi_path: Path) -> bool:
    try:
        with suppress_stderr():
            slide = slideio.open_slide(str(vsi_path), "VSI")
            scene = slide.get_scene(0)
            w, h = scene.size
            test_rect = (w // 2, h // 2, min(256, w), min(256, h))
            block = scene.read_block(
                rect=test_rect, size=(test_rect[2], test_rect[3]), slices=(0, 1)
            )
            return block is not None and block.size > 0
    except Exception as e:
        print(f"       [FAIL] Integrity check failed for {vsi_path.name}: {e}")
        return False


def fix_and_verify(vsi_path: Path, zip_path: Path, slide_dir: Path) -> str:
    vsi_name = vsi_path.name
    if zip_path.exists() and not vsi_path.exists():
        try:
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(slide_dir)
            for nested in slide_dir.rglob("*.vsi"):
                if nested.parent != slide_dir:
                    dest = slide_dir / nested.name
                    if not dest.exists():
                        shutil.move(str(nested), str(dest))
                    try:
                        nested.parent.rmdir()
                    except OSError:
                        pass
        except Exception as e:
            return f"{vsi_name}: [FAIL] Extraction/Fix failed: {e}"

    if vsi_path.exists():
        if verify_vsi(vsi_path):
            if zip_path.exists():
                zip_path.unlink()
            return f"{vsi_name}: [OK] Verified"
        else:
            vsi_path.unlink(missing_ok=True)
            return f"{vsi_name}: [FAIL] Corrupt - Deleted"
    return f"{vsi_name}: [MISSING]"


def process_vsi_image(img_info, slide_dir: Path):
    vsi_name = img_info["name"]
    vsi_path, zip_path = (
        slide_dir / vsi_name,
        slide_dir / vsi_name.replace(".vsi", ".zip"),
    )
    if not vsi_path.exists() and not zip_path.exists():
        cfg = Configuration()
        cfg.username = os.environ.get("EXACT_USERNAME", "niklas.hargarter")
        cfg.password = os.environ.get("EXACT_PASSWORD")
        cfg.host = "https://exact.hs-flensburg.de"
        api = ImagesApi(ApiClient(cfg))
        try:
            api.download_image(id=img_info["id"], target_path=str(zip_path))
        except Exception as e:
            return f"{vsi_name}: [FAIL] Download failed: {e}"
    return fix_and_verify(vsi_path, zip_path, slide_dir)


def download_vsi_dataset(
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
            executor.submit(process_vsi_image, img, slide_dir) for img in all_images
        ]
        for future in concurrent.futures.as_completed(futures):
            try:
                print(f"  {future.result()}")
            except Exception as e:
                print(f"  Worker exception: {e}")
    print(f"\nAcquisition for {full_name} complete.")
