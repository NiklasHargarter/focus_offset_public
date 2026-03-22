import pandas as pd
from pathlib import Path
from synthetic_data.config import SyntheticConfig
from synthetic_data.tests.mocks import MockSyntheticVSIDataset


def test_dataset_mock():
    mock_data = [
        {
            "slide_name": "dummy_slide.vsi",
            "x": 10000,
            "y": 10000,
            "optimal_z": 3,
            "num_z": 10,
            "max_focus_score": 100.0,
            "tissue_coverage": 1.0,
        }
    ]
    df = pd.DataFrame(mock_data)
    config = SyntheticConfig(top_percent_laplacian=1.0)

    dataset = MockSyntheticVSIDataset(
        index_df=df,
        slide_dir=Path("/dummy"),
        config=config,
    )

    sample = dataset[0]
    off_patch = sample["input"]
    opt_patch = sample["target"]
    meta = sample["metadata"]

    assert off_patch.shape == (3, config.patch_size_input, config.patch_size_input), (
        f"Expected input shape {(3, config.patch_size_input, config.patch_size_input)}, got {off_patch.shape}"
    )
    assert opt_patch.shape == (3, config.patch_size_target, config.patch_size_target), (
        f"Expected target shape {(3, config.patch_size_target, config.patch_size_target)}, got {opt_patch.shape}"
    )

    # PatchIn - PatchOut = k - 1
    assert off_patch.shape[2] - opt_patch.shape[2] == config.kernel_size - 1

    assert meta["focal_plane_index"] == 3
    assert meta["relative_offset_steps"] == config.z_offset_steps
    assert meta["z_step_size_microns"] == 1.0


def test_laplacian_filtering():
    # Use optimal_z=3 so that 3+5=8 < 10 (valid)
    mock_data = [
        {
            "slide_name": "s1.vsi",
            "x": 0,
            "y": 0,
            "optimal_z": 3,
            "num_z": 10,
            "max_focus_score": 10.0,
        },
        {
            "slide_name": "s1.vsi",
            "x": 0,
            "y": 0,
            "optimal_z": 3,
            "num_z": 10,
            "max_focus_score": 50.0,
        },
        {
            "slide_name": "s1.vsi",
            "x": 0,
            "y": 0,
            "optimal_z": 3,
            "num_z": 10,
            "max_focus_score": 30.0,
        },
        {
            "slide_name": "s1.vsi",
            "x": 0,
            "y": 0,
            "optimal_z": 3,
            "num_z": 10,
            "max_focus_score": 5.0,
        },
        {
            "slide_name": "s1.vsi",
            "x": 0,
            "y": 0,
            "optimal_z": 3,
            "num_z": 10,
            "max_focus_score": 40.0,
        },
    ]
    df = pd.DataFrame(mock_data)
    config = SyntheticConfig(top_percent_laplacian=0.4)  # Keep top 40% (2 patches)

    dataset = MockSyntheticVSIDataset(
        index_df=df,
        slide_dir=Path("/dummy"),
        config=config,
    )

    assert len(dataset) == 2
    # Should be the ones with scores 50.0 and 40.0
    scores = [dataset.df.iloc[i]["max_focus_score"] for i in range(len(dataset))]
    assert 50.0 in scores
    assert 40.0 in scores
    assert 10.0 not in scores

    print("Laplacian filtering test passed!")


if __name__ == "__main__":
    test_dataset_mock()
    test_laplacian_filtering()
