from src.dataset.vsi_datamodule import IHCDataModule
from src.dataset.ome_datamodule import OMEDataModule


def test_ihc_fit_rejection():
    print("Testing IHCDataModule fit rejection...")
    dm = IHCDataModule(
        batch_size=1, num_workers=0, patch_size=224, stride=112, min_tissue_coverage=0.1
    )
    try:
        dm.setup(stage="fit")
        print("FAIL: IHCDataModule should have raised RuntimeError for fit stage")
    except RuntimeError as e:
        print(f"SUCCESS: Caught expected error: {e}")
    except Exception as e:
        print(f"FAIL: IHCDataModule raised unexpected error: {type(e).__name__}: {e}")


def test_ome_fit_rejection():
    print("Testing OMEDataModule fit rejection...")
    # Mocking minimum required args for OMEDataModule
    dm = OMEDataModule(
        dataset_name="AgNor",
        batch_size=1,
        num_workers=0,
        patch_size=224,
        stride=112,
        min_tissue_coverage=0.1,
    )
    try:
        dm.setup(stage="fit")
        print("FAIL: OMEDataModule should have raised RuntimeError for fit stage")
    except RuntimeError as e:
        print(f"SUCCESS: Caught expected error: {e}")
    except Exception as e:
        print(f"FAIL: OMEDataModule raised unexpected error: {type(e).__name__}: {e}")


if __name__ == "__main__":
    test_ihc_fit_rejection()
    test_ome_fit_rejection()
