# ZStack HE Cache: Zarr Design

## Why Zarr for image storage

Training reads patches in random order (shuffled DataLoader). The bottleneck is per-sample I/O:

| Storage | Random access cost | Fork-safe | Notes |
|---|---|---|---|
| VSI via slideio | High — open slide, seek, decompress | ❌ | Current approach |
| PNG files | Medium — one file open per patch | ✅ | Millions of tiny files |
| **Zarr DirectoryStore** | Low — one chunk = one file = one seek + decompress | ✅ | **Chosen** |

With chunk shape `(1, H, W, C)`, `images[i]` reads exactly one file from disk. No shared handles, no locking — works with any DataLoader worker count and context (fork or spawn).

## Schema

All fields are stored as flat per-sample arrays. Fields like `num_z` and `z_res_um` repeat across z-levels of the same patch, but the overhead (~100 MB total at 7.7M samples) is negligible. A normalized two-level structure would add complexity for no real gain.

### Per-sample arrays

| Field | dtype | Notes |
|---|---|---|
| `images` | uint8 `(N, 224, 224, 3)` | Pixel data, chunk=(1,224,224,3) |
| `z_offset` | float32 `(N,)` | **Training target** — `(optimal_z − z_level) × z_res_um` |
| `z_level` | int16 `(N,)` | Z-slice index for this sample |
| `optimal_z` | int16 `(N,)` | Sharpest Z-level for this patch position |
| `num_z` | int16 `(N,)` | Total Z-slices in source slide |
| `x` | int32 `(N,)` | Patch top-left x on slide |
| `y` | int32 `(N,)` | Patch top-left y on slide |
| `slide_idx` | int16 `(N,)` | Index into `store.attrs["slides"]` |
| `z_res_um` | float32 `(N,)` | Physical Z-resolution in microns |
| `focus_score` | float32 `(N,)` | Laplacian variance of the chosen z-level |
| `focus_score_max` | float32 `(N,)` | Max Laplacian variance across the full z-stack |

> `focus_score` and `focus_score_max` are already computed during the focus search loop (currently discarded after `argmax`). Retaining them costs nothing and enables patch quality filtering and loss weighting.

### Store-level attributes

```
store.attrs:
  slides:       ["slide_001.vsi", ...]   ordered list — slide_idx[i] → filename
  prep_config:  {"patch_size": 224, "stride": 224, "ds": 1, "cov": 0.80}
  zarr_version: 2
  created_at:   "2026-02-20"
```

### Full structure

```
cache/ZStack_HE/s224_ds1_cov080/
  train.zarr/
    images           # uint8   (N, 224, 224, 3)   chunk=(1,224,224,3)   blosc/zstd cl=3
    z_offset         # float32 (N,)
    z_level          # int16   (N,)
    optimal_z        # int16   (N,)
    num_z            # int16   (N,)
    x                # int32   (N,)
    y                # int32   (N,)
    slide_idx        # int16   (N,)
    z_res_um         # float32 (N,)
    focus_score      # float32 (N,)
    focus_score_max  # float32 (N,)
  test.zarr/         ← identical layout
```

All 1D metadata arrays share a single chunk (65536,) and blosc/zstd cl=1.

## Preprocessing plan

Two phases, parallel across slides:

**Phase 1 — Count** (fast, metadata only)
> open slide → mask → grid → count valid patches, read `num_z` and `z_resolution`

Builds a per-slide `(n_patches, num_z, z_res_um)` table. Computes prefix sums → assigns each slide a non-overlapping zarr index range. Persists a **slide manifest JSON** with each slide's assigned `zarr_start`/`zarr_end`. Pre-allocates the zarr arrays.

```json
[{"slide": "slide_001.vsi", "n_patches_per_z": 1234, "num_z": 15,
  "z_res_um": 0.5, "zarr_start": 0, "zarr_end": 18510}, ...]
```

**Phase 2 — Extract** (slow, full focus search + pixel reads)
> Each worker: mask → grid → focus search → read all z-level crops → write to zarr at assigned offsets

Workers write to non-overlapping chunk ranges on disk — safe with no locking on `DirectoryStore`. On completion each worker writes a `{slide_name}.done` sentinel. On re-run, slides with a `.done` marker are skipped, enabling crash recovery.

### Codec config

```python
import numcodecs
image_codec = numcodecs.Blosc(cname='zstd', clevel=3, shuffle=numcodecs.Blosc.BITSHUFFLE)
meta_codec  = numcodecs.Blosc(cname='zstd', clevel=1, shuffle=numcodecs.Blosc.SHUFFLE)
```

## Val split

Val split is done at **slide level**, not sample level. Using individual samples would cause data leakage — different z-levels of the same `(slide, x, y)` position are near-identical images, and patches from the same slide share scanner optics and tissue characteristics.

Since `slide_idx` is fully loaded into RAM at init, the split is a simple numpy mask:

```python
val_slide_ids  = {3, 7, 12, ...}                        # slides held out for val
val_mask       = np.isin(slide_idx, list(val_slide_ids))
train_indices  = np.where(~val_mask)[0]
val_indices    = np.where(val_mask)[0]
```

The dataset receives an `indices` array and maps `__getitem__` through it. This replaces the current parquet `groupby`-on-slide-name approach with a single numpy operation.

## Dataset at training time

```python
store    = zarr.open("train.zarr", "r")
images   = store["images"]           # lazy proxy, not loaded
z_offset = store["z_offset"][:]      # fully loaded into RAM at init
# ... all other 1D arrays loaded into RAM

def __getitem__(self, i):
    image  = images[i]               # one chunk read from disk
    target = z_offset[i]             # RAM lookup
    return {"image": image, "target": target}
```

No slideio. No shared handles. No spawn workaround needed.
