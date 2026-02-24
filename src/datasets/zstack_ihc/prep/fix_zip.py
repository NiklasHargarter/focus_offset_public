from src.datasets.zstack_ihc.config import ZStackIHCConfig
from src.datasets.zstack_he.prep.fix_zip import (
    extract_zip,
    organize_vsi_files,
    verify_vsi,
    cleanup_corrupt_vsi,
)


def fix_zip_structure(dataset_name: str = "ZStack_IHC") -> None:
    """Extract and verify all VSI zips."""
    dataset_cfg = ZStackIHCConfig()
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
        else:
            # VSI is valid, delete the original zip to save space
            zip_name = vsi_file.name.replace(".vsi", ".zip")
            zip_path = zip_source / zip_name
            if zip_path.exists():
                print(f"[OK] Verified {vsi_file.name}. Deleting source ZIP.")
                zip_path.unlink(missing_ok=True)

    print(f"Fix/Extraction structure for {dataset_name} complete.")
