# Accelerated Focus Metric: Fast Whole-Slide Brenner Calculation

This document describes the high-throughput implementation of the Brenner focus metric used during the dataset preprocessing stage. To handle datasets with millions of patches across hundreds of high-resolution VSI slides, we transitioned from a patch-based approach to a vectorized **Integral Image (Summed Area Table)** strategy.

## 1. The Standard Brenner Metric
The Brenner focus score for a patch $\mathcal{P}$ is defined as the sum of squared horizontal differences between pixels offset by 2:

$$Score = \sum_{(x,y) \in \mathcal{P}} (I(x+2, y) - \mathbf{I}(x, y))^2$$

In a naive implementation, this requires:
1. Extracting the patch from the slide.
2. Calculating the squared differences for that patch.
3. Summing the results.

For $N$ patches of size $K \times K$, the complexity is $O(N \cdot K^2)$. With 12 million patches, this creates a massive bottleneck.

## 2. Vectorized Implementation Strategy

The fast implementation decouples the image analysis from the patch coordinates by processing the entire Z-slice at once.

### Step 1: Whole-Slide Difference Image
We calculate the squared difference for the entire downsampled slide in a single operation using NumPy:

```python
# gray: (H, W) grayscale image of the full slide
diff_sq = (gray[:, 2:].astype(np.float32) - gray[:, :-2].astype(np.float32)) ** 2
```
This produces a `diff_sq` array of size $(H, W-2)$.

### Step 2: Summed Area Table (Integral Image)
We compute the Integral Image of `diff_sq`. In this table, the value at any point $(x, y)$ is the sum of all pixels to the left and above it:

```python
# sat: (H+1, W-1) Summed Area Table
sat = cv2.integral(diff_sq)
```
This is a one-time operation per Z-slice with $O(H \cdot W)$ complexity.

### Step 3: $O(1)$ Patch Retrieval
With the SAT computed, the Brenner score for any patch starting at $(dx, dy)$ can be calculated using exactly four array lookups, regardless of its size:

```python
# Retrieve sum for patch of size patch_size
y1, y2 = dy, dy + patch_size
x1, x2 = dx, dx + patch_size - 2

patch_sum = sat[y2, x2] - sat[y1, x2] - sat[y2, x1] + sat[y1, x1]
```

## 3. Benefits

### Performance
The complexity is reduced from $O(N \cdot K^2)$ to $O(S + N)$, where $S$ is the total pixels in the slide and $N$ is the number of patches. For high patch density, the cost of processing the patches becomes nearly zero once the slide-wide SAT is built.

### Mathematical Equivalence
This method is **mathematically identical** to the patch-based approach. Because the summation operator is associative and commutative, summing values in a pre-computed difference image yields the exact same floating-point result as computing differences within a cropped patch.

### Memory Optimization
To handle very large slides (e.g., 70k x 75k pixels), the implementation explicitly manages memory buffers:
1. Load BGR block.
2. Convert to Grayscale.
3. **Delete BGR buffer.**
4. Compute Difference-Squared.
5. **Delete Grayscale buffer.**
6. Compute SAT.
7. **Delete Difference-Squared buffer.**
8. Extract Patch Scores.
9. **Delete SAT buffer.**

This ensures the peak memory usage never exceeds the minimum requirements for the current operation.
