import json

from src import config
from src.dataset.ome_dataset import OMEDataset
from src.dataset.vsi_datamodule import BaseVSIDataModule
from src.dataset.vsi_prep.preprocess import load_master_index


class AgNorDataModule(BaseVSIDataModule):
    """AgNor OME-TIFF DataModule (evaluation only).

    Inherits DataLoader, transform, and _filter_index logic from BaseVSIDataModule.
    Only overrides prepare_data/setup for OME-TIFF loading.
    """

    def __init__(self, **kwargs):
        kwargs.setdefault("dataset_name", "AgNor")
        kwargs.setdefault("downsample_factor", 1)
        kwargs.setdefault("min_tissue_coverage", 0.01)
        super().__init__(**kwargs)

    def prepare_data(self):
        master_index_path = config.get_master_index_path(
            self.dataset_name,
            stride=self.stride,
            downsample_factor=self.downsample_factor,
            min_tissue_coverage=self.min_tissue_coverage,
        )
        if not master_index_path.exists():
            print(
                f"\n[ERROR] Required master index for {self.dataset_name} is missing."
            )
            print(
                f"Please run: python src/dataset/ome_prep/preprocess.py --dataset {self.dataset_name}"
            )
            raise RuntimeError("Data preparation check failed.")

    def setup(self, stage: str | None = None):
        master_index = load_master_index(
            self.dataset_name, self.stride, self.downsample_factor, self.min_tissue_coverage
        )
        split_path = config.get_split_path(self.dataset_name)

        if not split_path.exists():
            print(
                f"Warning: Split file {split_path} missing. Using all data for test."
            )
            splits = {
                "train_pool": [],
                "test": [s.name for s in master_index.file_registry],
            }
        else:
            with open(split_path, "r") as f:
                splits = json.load(f)

        if stage == "fit":
            raise RuntimeError(
                f"{self.__class__.__name__} is for evaluation only. Training is disabled."
            )

        if stage == "test" or stage == "predict" or stage is None:
            test_files = splits.get("test", [])
            self.test_dataset = OMEDataset(
                index_data=self._filter_index(master_index, test_files),
                transform=self.val_transform,
            )
