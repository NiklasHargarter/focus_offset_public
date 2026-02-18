"""Dataset registry — maps CLI names to DataModule classes."""

from src.dataset.jiang2018 import Jiang2018DataModule
from src.dataset.ome_datamodule import AgNorDataModule
from src.dataset.vsi_datamodule import HEHoldOutDataModule, IHCDataModule

# All available evaluation datasets.
# Training always uses HE (VSIDataModule / HEHoldOutDataModule).
DATAMODULE_REGISTRY: dict[str, type] = {
    "he": HEHoldOutDataModule,  # ZStack_HE (default, also used for training)
    "ihc": IHCDataModule,  # ZStack_IHC
    "agnor": AgNorDataModule,  # AgNor OME-TIFF
    "jiang2018": Jiang2018DataModule,  # External benchmark
}
