#!/usr/bin/env python
from pathlib import Path
from shared_datasets.agnor_ome.prep.indexing import index_ome_dataset


def main():
    slide_dir = Path("/data/niklas/AgNor/raws")
    if not slide_dir.exists():
        slide_dir = Path("/data/niklas/AgNor_OME/raws")

    index_dir = Path("cache/AgNor/s224_ds1_cov020")
    split_path = Path("splits/splits_AgNor.json")

    # 0.2 is the required coverage for AgNor due to sparse silver stain characteristics
    params = {"patch_size": 224, "stride": 224, "downsample": 1, "cov": 0.2}

    index_ome_dataset(
        slide_dir=slide_dir,
        index_dir=index_dir,
        split_path=split_path,
        params=params,
        workers=8,
    )


if __name__ == "__main__":
    main()
