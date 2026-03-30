# Position Detection Optimizations

Three composable optimizations reduce the per-call cost of position detection from O(n³) to O(m·k) where m ≪ n is the spatial window size and k ≪ n is the number of surviving eigenvalues.

## Benchmark Summary (69×69 = 4,761 node grid)

| Component | Before | After | Speedup |
|-----------|:------:|:-----:|:-------:|
| `_find_power` | 1,258ms | 34ms | 37× |
| T^p computation | 60,000ms | 48ms | 1,250× |
| NNLS solve | 1,280ms | 0.3ms | 4,267× |
| **Total per call** | **62,538ms** | **381ms** | **164×** |

## 1. Spatial Windowing

**Idea**: The histogram `x` has nonzero values only in a small region (the walker's distribution is localized). Most of the n×n system is zero.

**Implementation**: Find the bounding box of nonzero bins, add padding (default 5 bins), restrict all operations to this window of m bins.

```
Full system:  NNLS on n×n matrix (4761×4761)
Windowed:     NNLS on m×m matrix (121×121)
```

Speedup: (n/m)^α where α ≈ 2-3 for NNLS. For n=4761, m=121: ~1,500× for NNLS alone.

**Correctness**: Exact. Any NNLS solution component outside the window would have zero weight (the corresponding x entries are zero), so removing them doesn't change the optimal solution.

## 2. Truncated Spectral Decomposition

**Idea**: T^p = V · diag(λ^p) · V⁻¹. For large p, most |λ_k|^p → 0. These eigenvalues contribute nothing to T^p.

**Implementation**: Keep only eigenvalues with |λ_k|^p > threshold (default 1e-3). Reduces the sum from n terms to k terms.

```
Full:      V[m×n] @ diag[n×n] @ V⁻¹[n×m]    n = 4761
Truncated: V[m×k] @ diag[k×k] @ V⁻¹[k×m]    k = 79 (at p=200, threshold=1e-2)
```

At p=200 on a 69×69 grid:
| Threshold | k (surviving) | V_win time |
|:---------:|:-------------:|:----------:|
| 1e-8 | 3,478 | 767ms |
| 1e-3 | 1,273 | ~200ms |
| 1e-2 | 79 | 8ms |

**Correctness**: Approximate, but the discarded terms contribute < threshold to each entry of T^p. At threshold=1e-3, the maximum error in any T^p entry is bounded by k_discarded × 1e-3 / n, which is negligible.

**Interaction with power**: At low p (e.g., p=17), fewer eigenvalues decay, so k is larger. The truncation is most effective for large p. The threshold should be tuned to the power range.

## 3. Precomputed Power Search

**Idea**: `_find_power` calls Brent's method ~20 times. Each evaluation computes `mean_column_spread(p) = (1/n) Σ λ_k^{2p} · ||row_k(V⁻¹)||²`. The row norms `||row_k(V⁻¹)||²` are O(n²) to compute but don't depend on x.

**Implementation**: `_precompute_power_search(eigvals, eigvecs_inv)` computes `row_norms_sq` once per graph. Each subsequent `_find_power` call receives the precomputed data and only does the O(n) Brent evaluation.

```
Before: O(n² + 20 × n)  per call    (n² dominates)
After:  O(n²) once + O(20 × n) per call
```

For n=4761: 1,258ms → 34ms per call (the O(n) evaluations plus one O(n) projection `eigvecs_inv @ x`).

## Composition

All three optimizations compose cleanly in `_compute_position_cell_fast`:

```python
def _compute_position_cell_fast(x_norm, eigvals, eigvecs, eigvecs_inv, ...):
    # 1. Power search with precomputed row norms [34ms]
    power = _find_power(..., precomputed=precomputed)

    # 2. Spatial window from nonzero support + padding [0.1ms]
    window = compute_bounding_box(x_norm, padding=5)

    # 3. Truncated spectral T^p within window [48ms]
    top_k = eigvals where |λ|^p > threshold
    V_win = eigvecs[window, top_k] @ diag(λ^p[top_k]) @ eigvecs_inv[top_k, window]

    # 4. Windowed NNLS [0.3ms]
    w, residual = nnls(V_win, x_win)

    # 5. Cell identification + GBC [0.1ms]
    ...
```

## OpenBLAS Fork Deadlock

**Problem**: `multiprocessing.Pool` uses fork(). OpenBLAS initializes pthreads internally. After fork(), the parent's BLAS thread pool state is corrupted, causing `np.linalg.eig()` and `scipy.linalg.inv()` to deadlock.

**Current workaround**: Set `OPENBLAS_NUM_THREADS=1` before importing numpy. This forces single-threaded BLAS, avoiding the deadlock but sacrificing parallelism in the calculation phase.

**Proper fix**: After simulation completes (no more Pool usage), re-enable BLAS threading:
```python
os.environ.pop("OPENBLAS_NUM_THREADS", None)
os.environ.pop("OMP_NUM_THREADS", None)
```

This allows eigendecomposition and matrix operations to use all CPU cores during the calculation phase.

## Matching Pursuit (Approximate)

**Idea**: Since the NNLS solution is sparse (1-4 components), use greedy column selection instead of solving the full optimization.

**Algorithm**:
1. Find column of V_win most positively correlated with x
2. Add to selected set, solve small NNLS on selected columns only
3. Repeat until residual is small or max components reached

**Speed**: O(m × max_components) vs O(m² to m³) for full NNLS.

**Caveat**: Greedy selection may miss weak distant components in non-local distributions. Use NNLS for studies involving non-locality.
