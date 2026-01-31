# Focus Offset Prediction for Whole Slide Imaging (WSI)

This repository provides a complete pipeline for training and evaluating deep learning models for focus offset prediction in Whole Slide Imaging, specifically tailored for Olympus VSI formats.

##  Overview

Focusing is a critical challenge in high-resolution digital pathology. This project implements:
- **Scalable Preprocessing**: High-throughput 30x downsampled strategy with vectorized focus estimation.
- **Flexible Architectures**: Support for ResNet, Vision Transformers (ViT), ConvNeXt, and EfficientNet.
- **Incremental Data Management**: Automated download, extraction, and indexing of >200GB datasets.
- **Robust Evaluation**: Cross-dataset benchmarking (ZStack HE/IHC, Jiang2018).

##  Installation

This project uses `uv` for lightning-fast dependency management.

```bash
# Clone the repository
git clone https://github.com/NiklasHargarter/focus_offset_public.git
cd focus_offset_public

# Create environment and install dependencies
uv sync
```

##  Dataset Synchronization

For large datasets (HE/IHC), use the standalone synchronization script. This handles downloading missing files from the EXACT server and performing incremental preprocessing.

```bash
# Synchronize the HE dataset
uv run python -m src.dataset.vsi_prep.sync --dataset ZStack_HE
```

**Key Flags:**
- `--force`: Refresh image list from server (use if dataset was extended).
- `--skip-preprocess`: Only download and unzip.
- `--skip-split`: Keep existing training splits.

Training is powered by PyTorch Lightning. We provide a default configuration for stable training.

```bash
# Training with default configuration (Early Stopping, Checkpointing)
uv run python -m src.train fit -c configs/trainer.yaml

# Multimodal Version (Custom overrides)
uv run python -m src.train fit -c configs/trainer.yaml --model.use_transforms=True

# RGB Only Version
uv run python -m src.train fit -c configs/trainer.yaml --model.use_transforms=False
```

**Note on Memory:** The default batch size has been set to **32** to ensure stability on consumer GPUs.


**Common CLI Overrides:**
- `--data.batch_size 32`: Change batch size if encountering memory issues.
- `--data.num_workers 4`: Adjust number of data loading workers.
- `--trainer.max_epochs 50`: Set training duration.

## Evaluation

Run evaluation on multiple datasets using a trained checkpoint. The `--model` config is now optional and only needed for custom architectures.

```bash
# Standard evaluation
uv run python -m src.evaluate --ckpt checkpoints/convnextv2/best_model.ckpt --datasets ZStack_HE ZStack_IHC Jiang2018
```

##  Project Structure

```text
.
├── configs/            # YAML configurations (Models, Data, Trainer)
├── src/
│   ├── dataset/        # Custom Dataset and DataModule logic
│   ├── models/         # Architecture definitions and LightningModule
│   ├── processing/     # Dataset preparation (Download, Preprocess, Split)
│   ├── utils/          # Focus metrics (Brenner) and EXACT API helpers
│   ├── visualize/      # Heatmaps, Z-stacks, and mask visualizations
│   ├── train.py        # Main training entry point
│   ├── evaluate.py     # Standalone evaluation script
│   └── config.py       # Global path and directory constants
└── README.md
```

##  Documentation

Detailed documentation is available in the `docs/` directory:
- [Dataset Architecture](docs/dataset_architecture.md)
- [Focus Estimation Strategy](docs/focus_strategy.md)
- [VSI Preprocessing Benchmarks](docs/vsi_extraction_benchmarks.md)
- [Dataloader Optimization](docs/dataloader.md)
- [Fast Brenner Vectorization](docs/fast_brenner_vectorization.md)

