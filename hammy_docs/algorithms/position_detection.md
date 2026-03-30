# Position Detection Algorithm

## Problem Statement

Given a histogram of random walk positions on a graph (observed at some checkpoint, conditional on the walker ending at a specific target), determine the walker's continuous position on the graph.

The histogram is a vector `x` of counts across graph nodes (bins). The position detection algorithm finds which node (or which cell between nodes) the walker was most likely at, producing a continuous 2D position `(row, col)` with sub-node precision.

## Pipeline Overview

```
x (histogram)
  │
  ├─ 1. Power Search (_find_power)
  │     Find walk power p matching spectral spread of x
  │
  ├─ 2. Spectral Matrix Power (T^p)
  │     Compute transition matrix raised to power p
  │
  ├─ 3. NNLS Decomposition
  │     Solve: min ||T^p · w - x||₂  subject to w ≥ 0
  │     Find which source nodes explain the observed distribution
  │
  ├─ 4. Cell/Simplex Identification
  │     Map NNLS components to a graph cell (triangle, square, etc.)
  │
  └─ 5. Generalized Barycentric Coordinates
        Compute continuous position within the cell
        → (position_row, position_col, fit_quality, cell_dim)
```

## Step 1: Power Search

**Function**: `_find_power(eigvals, eigvecs_inv, x)`

The observed distribution `x` results from a random walk of unknown duration. The "power" `p` represents the effective number of transition steps that best explains the spread of `x`.

**Method**: Spectral spread matching via Brent's root-finding.

- **Spectral spread of x**: Project x onto non-stationary eigenmodes of the transition matrix. `spread(x) = Σ_{|λ_k|<1} (V⁻¹x)_k²`
- **Mean spread of T^p columns**: `mean_spread(p) = (1/n) Σ_{|λ_k|<1} λ_k^{2p} ||row_k(V⁻¹)||²`
- **Root finding**: `mean_spread(p) = spread(x)` has a unique root because `mean_spread(p)` is monotonically decreasing in p.

The power p is typically 100-300 for T=1000 step walks with //2 binning.

## Step 2: Spectral Matrix Power

**T^p = V · diag(λ^p) · V⁻¹**

where V is the eigenvector matrix and λ are eigenvalues. Each column of T^p is the distribution that would result from a point source at that node after p steps.

## Step 3: NNLS Decomposition

**Solve**: `min ||T^p · w - x||₂` subject to `w ≥ 0`

This finds non-negative weights `w` such that the observed distribution `x` is best explained as a mixture of point-source distributions (columns of T^p).

**Key property**: The solution `w` is sparse — typically 1-4 nonzero components. These are the graph nodes that "explain" where the walker was.

**Thresholding**: Components with weight < 1% of the peak are discarded.

## Step 4a: Simplex Identification (Original Method)

**Function**: `_identify_simplex(indices, weights, adjacency, max_dim)`

Greedy algorithm:
1. Start with highest-weight node
2. Add next node only if adjacent to ALL existing simplex nodes
3. Stop when `max_dim` is reached

**Output**: Simplex nodes + barycentric coordinates.

**Limitation**: Requires graph cliques. On a square lattice (bipartite, no triangles), the maximum simplex is an edge (2 nodes). Diagonal positions inside square cells cannot be captured — fit_quality drops to ~0.59.

## Step 4b: Cell Identification (SOTA Method)

**Function**: `_identify_cell(indices, weights, cells, node_to_cells)`

Instead of finding cliques, find the pre-defined cell (triangle, square, hexagon) that captures the most NNLS weight.

1. For each NNLS node, look up which cells contain it
2. Sum NNLS weight for each candidate cell's corners
3. Pick the cell with maximum total weight
4. fit_quality = captured weight / total weight

**Works on all lattice types** because cells are defined by the graph topology, not by adjacency cliques.

## Step 5: Generalized Barycentric Coordinates

Given the best cell's corner weights, compute position within the cell:

**Triangle (3 nodes)**: Standard barycentric coordinates.
- `s = w₁ / (w₀ + w₁ + w₂)`, `t = w₂ / (w₀ + w₁ + w₂)`
- Position: `(1-s-t)·p₀ + s·p₁ + t·p₂`

**Quadrilateral (4 nodes)**: Inverse bilinear interpolation.
- Cell corners: `(n₀₀, n₁₀, n₀₁, n₁₁)` at positions `(r, c), (r, c+1), (r+1, c), (r+1, c+1)`
- `s = (w₁₀ + w₁₁) / w_total` (fraction on column+1 side)
- `t = (w₀₁ + w₁₁) / w_total` (fraction on row+1 side)
- Position: bilinear interpolation of corner coordinates

**General**: Weighted centroid as fallback.

## Output

| Field | Type | Description |
|-------|------|-------------|
| position_row | float | Continuous row coordinate (graph space) |
| position_col | float | Continuous column coordinate (graph space) |
| fit_quality | float | Fraction of NNLS weight captured by the cell (0-1) |
| cell_dim | int | Cell dimension: -1 (none), 0 (vertex), 1 (edge), 2 (face) |
| nonzero_count | int | Number of NNLS components after thresholding |
| power | float | Fitted walk power p |
| residual | float | Relative NNLS residual ||Vw-x|| / ||x|| |
| bilinear_s, bilinear_t | float | Cell-local coordinates |

## Algorithm Methods

| Method | Function | Exact? | Speed | Use case |
|--------|----------|:------:|:-----:|----------|
| `nnls` | `_compute_position_cell` | Yes | Slow | Small grids, reference |
| `nnls_fast` | `_compute_position_cell_fast` | Yes | Fast | **Default**. Large grids, production |
| `matching_pursuit` | `_compute_position_cell_mp` | No | Fastest | Screening. Caveat: may miss non-local components |

## Graph Support

| Graph | Cell type | Nodes/cell | GBC method |
|-------|-----------|:----------:|------------|
| LinearGraph (1D) | Edge | 2 | Linear interpolation |
| LatticeGraph2D (square) | Square | 4 | Bilinear |
| TriangularGraph2D | Triangle | 3 | Barycentric |
| HexagonalGraph2D | Square (approx) | 4 | Bilinear |
| BrickGraph2D | Offset rectangle | 4 | Bilinear |
