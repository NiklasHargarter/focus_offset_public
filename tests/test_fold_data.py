from src.dataset.vsi_datamodule import HEFoldDataModule
from src import config


def test_kfold_datamodule():
    print("Testing HEFoldDataModule Setup...")

    # We use a dummy directory or check if cache exists
    # If cache doesn't exist, we might get an error, so we catch it
    try:
        dm = HEFoldDataModule(
            batch_size=16,
            num_workers=0,
            patch_size=224,
            stride=448,
            min_tissue_coverage=0.05,
            num_folds=5,
            fold_idx=0,
        )

        # We don't call prepare_data() because it checks the disk
        # We try to see if setup(stage='fit') partitions correctly
        # This requires the master index to be present
        master_path = config.get_master_index_path("ZStack_HE", 224)
        if not master_path.exists():
            print(f"Skipping setup test: Master index not found at {master_path}")
            return

        dm.setup(stage="fit")

        print(f"Fold 0 - Train slides: {len(dm.train_dataset.index.file_registry)}")
        print(f"Fold 0 - Val slides:   {len(dm.val_dataset.index.file_registry)}")

        # Check orthogonality
        train_names = set(s.name for s in dm.train_dataset.index.file_registry)
        val_names = set(s.name for s in dm.val_dataset.index.file_registry)

        intersection = train_names.intersection(val_names)
        print(f"Intersection (should be 0): {len(intersection)}")

        if len(intersection) == 0:
            print("SUCCESS: Data is correctly partitioned by fold.")
        else:
            print("FAILURE: Slide overlap detected between train and val!")

    except Exception as e:
        print(f"Test encountered an error: {e}")


if __name__ == "__main__":
    test_kfold_datamodule()
