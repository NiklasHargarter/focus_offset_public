import zipfile
import shutil
import config
from src.utils.io_utils import suppress_stderr
import slideio


def fix_zip_structure() -> None:
    # 1. Setup Directories from Config
    # Input: VSI_ZIP_DIR (Zips)
    # Output: VSI_RAW_DIR (Extracted VSI + folders)

    zip_source = config.VSI_ZIP_DIR
    extract_target = config.VSI_RAW_DIR

    extract_target.mkdir(parents=True, exist_ok=True)

    print(f"Source Zips: {zip_source}")
    print(f"Extract Destination: {extract_target}")

    if not zip_source.exists():
        print(f"Error: Zip directory {zip_source} does not exist.")
        return

    # 2. Iterate and Extract
    zips = list(zip_source.glob("*.zip"))
    if not zips:
        print("No .zip files found to extract.")
        return

    for zip_path in zips:
        # Check if extracted file already exists
        expected_vsi_name = zip_path.name.replace(".zip", ".vsi")
        expected_vsi_path = extract_target / expected_vsi_name

        if expected_vsi_path.exists():
            print(f"Skipping {zip_path.name} (VSI already exists).")
            continue

        print(f"Processing {zip_path.name}...")

        # We extract into the target dir directly
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_target)

    # 3. Post-Extraction Cleanup / Structure Fix
    print("\n--- Organizing and Verifying VSI Files ---")

    all_vsis = list(extract_target.rglob("*.vsi"))

    for vsi_file in all_vsis:
        # Move if nested
        final_path = vsi_file
        if vsi_file.parent != extract_target:
            dest = extract_target / vsi_file.name
            if not dest.exists():
                print(f"Moving nested VSI: {vsi_file} -> {dest}")
                shutil.move(str(vsi_file), str(dest))
            final_path = dest

        # Verify Readability
        try:
            with suppress_stderr():
                slide = slideio.open_slide(str(final_path), "VSI")
            scene = slide.get_scene(0)

            # Read a small test block from the middle to ensure data integrity
            w, h = scene.size
            cx, cy = w // 2, h // 2
            test_rect = (cx, cy, 256, 256)

            # Read slice 0 (or first available)
            block = scene.read_block(rect=test_rect, size=(256, 256), slices=(0, 1))

            if block is None or block.size == 0:
                raise ValueError("Read empty block")

            # print(f"[OK] Verified {final_path.name}")

        except Exception as e:
            print(f"[FAIL] Corrupt VSI detected: {final_path.name}")
            print(f"       Error: {e}")

            # 1. Delete VSI File
            try:
                final_path.unlink()
                print(f"       Deleted corrupt file: {final_path.name}")
            except OSError as del_err:
                print(f"       Failed to delete VSI: {del_err}")

            # 2. Delete Aux Folder (_filename_)
            # Convention: 001.vsi -> _001_
            aux_folder_name = "_" + final_path.stem + "_"
            aux_folder_path = extract_target / aux_folder_name

            if aux_folder_path.exists() and aux_folder_path.is_dir():
                try:
                    shutil.rmtree(str(aux_folder_path))
                    print(f"       Deleted aux folder: {aux_folder_name}")
                except OSError as del_err:
                    print(f"       Failed to delete aux folder: {del_err}")

            # 3. Delete Source Zip (to force re-download)
            # We need to find the zip that corresponds to this file.
            # Usually: filename.vsi -> filename.zip
            zip_name = final_path.name.replace(".vsi", ".zip")
            zip_path = zip_source / zip_name

            if zip_path.exists():
                try:
                    zip_path.unlink()
                    print(
                        f"       Deleted source zip: {zip_name} (forcing re-download)"
                    )
                except OSError as del_err:
                    print(f"       Failed to delete source zip: {del_err}")

    print("Fix/Extraction structure complete.")


if __name__ == "__main__":
    fix_zip_structure()
