# Focus Offset Prediction for Whole Slide Imaging (WSI)

This repository provides a complete pipeline for training and evaluating deep learning models for focus offset prediction in Whole Slide Imaging, specifically tailored for Olympus VSI formats.

##  Overview

Focusing is a critical challenge in high-resolution digital pathology. This project implements:
- **Scalable Preprocessing**: High-throughput "Master Block" strategy for focus estimation (Olympus VSI).
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

##  Training

Training is powered by PyTorch Lightning and `LightningCLI` for configuration management.

```bash
uv run python -m src.train fit --config configs/trainer.yaml --config configs/models/resnet50.yaml --config configs/data/zstack_he.yaml
```

##  Evaluation

Run evaluation on multiple datasets using a trained checkpoint:

```bash
uv run python -m src.evaluate --model configs/models/resnet50.yaml --ckpt checkpoints/resnet50/best_model.ckpt --datasets ZStack_HE ZStack_IHC Jiang2018
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

