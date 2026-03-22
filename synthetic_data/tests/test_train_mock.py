"""Mock test for Synthetic Training Loop."""

import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from synthetic_data.config import SyntheticConfig
from synthetic_data.model import SyntheticConvModel
from synthetic_data.tests.mocks import MockSyntheticVSIDataset


def test_train_mock():
    mock_data = [
        {
            "slide_name": "s1.vsi",
            "x": 100,
            "y": 100,
            "optimal_z": 3,
            "num_z": 10,
            "max_focus_score": 100.0,
        },
        {
            "slide_name": "s1.vsi",
            "x": 200,
            "y": 200,
            "optimal_z": 3,
            "num_z": 10,
            "max_focus_score": 100.0,
        },
    ]
    df = pd.DataFrame(mock_data)
    config = SyntheticConfig(top_percent_laplacian=1.0)

    dataset = MockSyntheticVSIDataset(df, Path("/tmp"), config)
    loader = torch.utils.data.DataLoader(dataset, batch_size=2)

    model = SyntheticConvModel(kernel_size=config.kernel_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    model.train()
    batch = next(iter(loader))
    inputs = batch["input"]
    targets = batch["target"]

    initial_loss = None
    for i in range(100):
        optimizer.zero_grad()
        preds = model(inputs)
        loss = criterion(preds, targets)
        loss.backward()
        optimizer.step()
        if initial_loss is None:
            initial_loss = loss.item()

    assert loss.item() < initial_loss, (
        f"Loss did not decrease (Initial: {initial_loss}, Final: {loss.item()})"
    )


if __name__ == "__main__":
    test_train_mock()
