import json
import os
from pathlib import Path
from multiprocessing import Pool
from functools import partial

from shared_datasets.vsi.prep.indexing import create_vsi_split, _run_split
from synthetic_data.prep.preprocess_synthetic import process_vsi_slide_synthetic


def index_vsi_synthetic_dataset(
    slide_dir: Path,
    index_dir: Path,
    split_path: Path,
    patch_size: int,
    downsample: int,
    min_coverage: float,
    exclude_pattern: str = "_all_",
    workers: int | None = None,
    dry_run: bool = False,
):
    splits = (
        create_vsi_split(slide_dir, split_path, exclude_pattern)
        if not split_path.exists()
        else json.loads(split_path.read_text())
    )

    train_names = set(splits.get("train_pool", []))
    test_names = set(splits.get("test", []))

    all_files = sorted(slide_dir.glob("*.vsi"))
    all_files = [f for f in all_files if exclude_pattern.lower() not in f.name.lower()]

    train_files = (
        [f for f in all_files if f.name in train_names] if train_names else all_files
    )
    test_files = [f for f in all_files if f.name in test_names]

    if dry_run:
        train_files = train_files[:2]
        test_files = test_files[:2]

    process_func = partial(
        process_vsi_slide_synthetic,
        patch_size=patch_size,
        downsample=downsample,
        min_coverage=min_coverage,
        dry_run=dry_run,
    )

    with Pool(workers or os.cpu_count() or 1) as pool:
        _run_split(
            pool,
            train_files,
            process_func,
            index_dir / "train_synthetic.parquet",
            "train",
        )
        _run_split(
            pool, test_files, process_func, index_dir / "test_synthetic.parquet", "test"
        )


def main():
    from synthetic_data.config import SyntheticConfig

    config = SyntheticConfig()

    config.index_dir.mkdir(parents=True, exist_ok=True)

    index_vsi_synthetic_dataset(
        slide_dir=Path(config.slide_dir),
        index_dir=config.index_dir,
        split_path=config.split_path,
        patch_size=config.patch_size_input,
        downsample=config.downsample,
        min_coverage=config.min_coverage,
        workers=config.workers,
        dry_run=config.dry_run,
    )


if __name__ == "__main__":
    main()
