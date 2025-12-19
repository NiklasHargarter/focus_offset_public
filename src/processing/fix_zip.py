import zipfile
import shutil
from pathlib import Path
import config
from src.utils.io_utils import suppress_stderr
import slideio


def extract_zip(zip_path: Path, extract_target: Path) -> bool:
    """Extracts a zip file if the VSI is not already present."""
    expected_vsi_name = zip_path.name.replace(".zip", ".vsi")
    expected_vsi_path = extract_target / expected_vsi_name

    if expected_vsi_path.exists():
        return False

    print(f"Extracting {zip_path.name}...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_target)
    return True


def organize_vsi_files(extract_target: Path) -> None:
    """Moves nested VSIs to the base directory."""
    all_vsis = list(extract_target.rglob("*.vsi"))
    for vsi_file in all_vsis:
        if vsi_file.parent != extract_target:
            dest = extract_target / vsi_file.name
            if not dest.exists():
                print(f"Moving nested VSI: {vsi_file} -> {dest}")
                shutil.move(str(vsi_file), str(dest))


def verify_vsi(vsi_path: Path) -> bool:
    """Verifies that a VSI file is readable."""
    try:
        with suppress_stderr():
            slide = slideio.open_slide(str(vsi_path), "VSI")
        scene = slide.get_scene(0)

        w, h = scene.size
        test_rect = (w // 2, h // 2, 256, 256)
        block = scene.read_block(rect=test_rect, size=(256, 256), slices=(0, 1))

        return block is not None and block.size > 0
    except Exception as e:
        print(f"[FAIL] Integrity check failed for {vsi_path.name}: {e}")
        return False


def cleanup_corrupt_vsi(vsi_path: Path, zip_source: Path, extract_target: Path) -> None:
    """Cleans up corrupted VSI files and their source ZIPs."""
    vsi_path.unlink(missing_ok=True)
    print(f"       Deleted corrupt VSI: {vsi_path.name}")

    aux_folder_path = extract_target / f"_{vsi_path.stem}_"
    if aux_folder_path.exists() and aux_folder_path.is_dir():
        shutil.rmtree(aux_folder_path)
        print(f"       Deleted aux folder: {aux_folder_path.name}")

    zip_path = zip_source / vsi_path.name.replace(".vsi", ".zip")
    if zip_path.exists():
        zip_path.unlink()
        print(f"       Deleted source zip: {zip_path.name}")


def fix_zip_structure() -> None:
    """Orchestrates extraction and verification of dataset files."""
    zip_source = config.VSI_ZIP_DIR
    extract_target = config.VSI_RAW_DIR

    if not zip_source.exists():
        print(f"Error: Zip directory {zip_source} does not exist.")
        return

    extract_target.mkdir(parents=True, exist_ok=True)

    zips = list(zip_source.glob("*.zip"))
    for zip_path in zips:
        extract_zip(zip_path, extract_target)

    print("\n--- Organizing and Verifying VSI Files ---")
    organize_vsi_files(extract_target)

    all_vsis = list(extract_target.glob("*.vsi"))
    for vsi_file in all_vsis:
        if not verify_vsi(vsi_file):
            print(f"[FAIL] Corrupt VSI detected: {vsi_file.name}")
            cleanup_corrupt_vsi(vsi_file, zip_source, extract_target)

    print("Fix/Extraction structure complete.")


if __name__ == "__main__":
    fix_zip_structure()
