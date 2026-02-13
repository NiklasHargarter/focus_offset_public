# Preprocessing Pipeline Optimization & Scaling

This document details the optimizations made to the preprocessing pipeline to handle datasets with millions of patches efficiently and resiliently.

The pipeline is structured as shared modules (`src/dataset/prep/`) consumed by format-specific scripts (`vsi_prep/`, `ome_prep/`). See `docs/vsi_preprocessing.md` for the step-by-step flow.

## 1. Atomic Per-Slide Indexing (Resiliency & Efficiency)
We benchmarked the scalability of the metadata structure. For the current `ZStack_HE` dataset (~2 Million patches), the final master index is only **~50 MB**. While this size is easily handled by modern systems, the "monolithic" approach had significant workflow drawbacks:

- **Resiliency (Primary Benefit)**: In a monolithic system, if a 16-hour preprocessing run crashes at hour 15, the entire index is lost because it hasn't been "committed" to disk. Atomic indexing saves each slide (~0.5 MB) immediately.
- **Write Frequency Overhead**: Saving a growing monolithic pickle after every slide means re-writing the entire dataset history to disk repeatedly (Total bytes written = $N_{slides} \times Size_{total} / 2$). Atomic indexing writes each patch exactly once.
- **Maintenance**: Individual slide pickles allow for re-processing a single slide without touching the rest of the dataset.

**Architecture:**
- **`manifest.pkl`**: Global configuration (stride, patch size).
- **`indices/*.pkl`**: One metadata file per slide, saved immediately upon completion.
- **Aggregation**: The `MasterIndex` is assembled dynamically from these atomic files.

## 2. Storage & Download Optimization
To prevent disk space exhaustion (the "Out of Space" error), the download script was refactored:
- **Individual Processing**: Slides are downloaded, extracted, and verified one-by-one.
- **Immediate Cleanup**: ZIP files are deleted *immediately* after a VSI is successfully verified, maintaining a much lower storage floor.
- **Integrity First**: Every extraction is followed by an integrity check using `slideio` to ensure no corrupt data enters the pipeline.

## 3. Multi-Processing Strategy Benchmark
We conducted a head-to-head comparison between two multi-processing approaches on 16 actual slides from the `ZStack_HE` dataset (Stride 448).

| Metric | **Strategy A: Slide-Parallel** | **Strategy B: Block-Parallel** |
| :--- | :--- | :--- |
| **Logic** | 1 Slide per Process | 1 Block per Task |
| **Handle Management** | Opens VSI **once** per slide | Opens VSI **per task (~15-50x)** |
| **Throughput** | **33.1 patches/s** | 28.7 patches/s |
| **Sample Rate** | **894.1 samples/s** | 775.1 samples/s |

**Conclusion**: **Strategy A (Slide-Level)** is ~15% faster. The cost of "opening" a VSI handle (parsing XML/metadata) is significant. Reusing the handle across many blocks provides higher throughput.

## 4. Performance Projections (ZStack_HE)
Based on real-world throughput of **~33 patches/s** (with 16 workers):

| Stride | Approx. Patches | Approx. Samples (x27 Z) | **Estimated Time** |
| :--- | :--- | :--- | :--- |
| **Stride 224** | **1.94M** | **52.5M** | **~16.3 Hours** |
| Stride 448 | 0.48M | 13.1M | ~4.1 Hours |
| Stride 896 | 0.12M | 3.3M | ~1.1 Hours |

## 5. Usage Recommendations
When running the pipeline on shared machines:
1. **Worker Capping**: Use `--workers 12` or `--workers 16` to allow others to use the CPU/IO.
2. **Atomic Recovery**: If the run is interrupted, simply run the same command again. The system will detect existing indices in `cache/` and skip those slides automatically.
3. **Stride Selection**: Stride 448 provides a 4x speedup for initial training and model architecture search, while Stride 224 is recommended for final high-density model runs.
