import argparse
import shutil
import zipfile
from pathlib import Path

import slideio

from src.config import DatasetConfig
from src.utils.io_utils import suppress_stderr


def extract_zip(zip_path: Path, extract_target: Path) -> bool:
    """Extract zip file if the target VSI is missing."""
    expected_vsi_name = zip_path.name.replace(".zip", ".vsi")
    expected_vsi_path = extract_target / expected_vsi_name

    if expected_vsi_path.exists():
        return False

    try:
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_target)
        return True
    except Exception as e:
        print(f"[FAIL] Could not extract {zip_path.name}: {e}")
        return False


def organize_vsi_files(extract_target: Path) -> None:
    """Move nested VSI files to root directory."""
    vsi_files = list(extract_target.rglob("*.vsi"))
    for vsi_file in vsi_files:
        if vsi_file.parent != extract_target:
            dest = extract_target / vsi_file.name
            if not dest.exists():
                print(f"Moving nested VSI: {vsi_file} -> {dest}")
                shutil.move(str(vsi_file), str(dest))


def verify_vsi(vsi_path: Path) -> bool:
    """Verify VSI file readability."""
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
        print(f"[FAIL] Integrity check failed for {vsi_path.name}: {e}")
        return False


def cleanup_corrupt_vsi(vsi_path: Path, zip_source: Path, extract_target: Path) -> None:
    """Delete corrupt VSI files and source ZIPs."""
    vsi_path.unlink(missing_ok=True)
    print(f"       Deleted corrupt VSI: {vsi_path.name}")

    aux_folder_path = extract_target / f"_{vsi_path.stem}_"
    if aux_folder_path.exists() and aux_folder_path.is_dir():
        shutil.rmtree(aux_folder_path)
        print(f"       Deleted aux folder: {aux_folder_path.name}")

    zip_name = vsi_path.name.replace(".vsi", ".zip")
    zip_path = zip_source / zip_name
    if zip_path.exists():
        zip_path.unlink()
        print(f"       Deleted source zip: {zip_path.name}")


def fix_zip_structure(dataset_name: str = "ZStack_HE") -> None:
    """Extract and verify all VSI zips."""
    dataset_cfg = DatasetConfig(name=dataset_name)
    zip_source = dataset_cfg.zip_dir
    extract_target = dataset_cfg.raw_dir

    if not zip_source.exists():
        print(f"Error: Zip directory {zip_source} does not exist for {dataset_name}.")
        return

    extract_target.mkdir(parents=True, exist_ok=True)

    zips = list(zip_source.glob("*.zip"))
    for zip_path in zips:
        extract_zip(zip_path, extract_target)

    print(f"\n--- Organizing and Verifying VSI Files for {dataset_name} ---")
    organize_vsi_files(extract_target)

    all_vsis = list(extract_target.glob("*.vsi"))
    for vsi_file in all_vsis:
        if not verify_vsi(vsi_file):
            print(f"[FAIL] Corrupt or unreadable VSI detected: {vsi_file.name}")
            cleanup_corrupt_vsi(vsi_file, zip_source, extract_target)

    print(f"Fix/Extraction structure for {dataset_name} complete.")


if __name__ == "__main__":
    dataset_cfg = DatasetConfig()
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=dataset_cfg.name)
    args = parser.parse_args()
    fix_zip_structure(dataset_name=args.dataset)
