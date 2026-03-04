"""Environment setup utilities shared across training scripts."""

import os

import torch


def setup_environment() -> None:
    """Set environment variables and torch defaults for optimal training."""
    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    torch.set_float32_matmul_precision("medium")
