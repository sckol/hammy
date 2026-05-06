# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**hammy** is a Python library for exact random walk analysis on graphs via matrix multiplication.
Instead of Monte Carlo sampling, it computes exact probability distributions via repeated `T @ v`.
Supports numpy (CPU) and cupy (GPU) backends through a drop-in switcher (`hampy`).

## Build & Run

### Python environment
```bash
python3 -m venv .venv
.venv/bin/pip install -e .
```

### Tests
```bash
.venv/bin/pytest
```

### Results storage
- S3 bucket: `hammy`. Use `--profile hammy` for all `yc` CLI commands.
- Downloaded results go to `~/hammy/results/`.
- `HammyObject.RESULTS_DIR` controls where files are stored (default `Path("results")`).

## Architecture

### Core pipeline

```
Graph → Task → run() → xr.DataArray → Calculation → Visualization
```

- `Graph` — defines the topology and transition matrix; cached via `HammyObject`.
- `Task` — wraps a graph + initial condition; `run()` returns exact distributions.
- `Calculation` — post-processes results (iterates over xarray dimension combinations).
- `Dispatcher` — routes tasks to CPU/GPU workers by graph size.

### hampy (`hammy_lib/hampy.py`) — backend switcher

Drop-in numpy-compatible module. Call once at worker startup:

```python
import hammy_lib.hampy as hp
hp.set_backend(numpy)   # CPU worker
hp.set_backend(cupy)    # GPU worker
```

All task code uses `hp.array()`, `hp.matmul()`, `hp.to_numpy()` — no platform awareness needed.

### Task hierarchy (`hammy_lib/task.py`)

```python
@dataclass
class Task:
    graph: Graph
    def run(self) -> xr.DataArray: ...

@dataclass
class WalkTask(Task):
    initial: np.ndarray   # initial probability vector
    t_steps: list[int]    # step counts to record
```

`WalkTask.run()` applies `T.T @ v` at each step (T is row-stochastic: `T[i,j]` = P(go to j | at i);
distribution propagation uses the transpose).

### Dispatcher (`hammy_lib/dispatcher.py`)

```python
d = Dispatcher(tasks)
d.next_for_cpu()   # smallest graph (lowest overhead)
d.next_for_gpu()   # largest graph (max throughput)
```

### HammyObject hierarchy (`hammy_lib/hammy_object.py`)

All persistent objects inherit from `HammyObject`:
- **Content-addressed IDs** — derived from metadata, enables caching.
- **Load/save with validation** — metadata checked on load to detect stale cache.
- **S3 sync** via `YandexCloudStorage`.

Two concrete base classes:
- `DictHammyObject` — JSON (configs).
- `ArrayHammyObject` — xarray DataArray/Dataset as NetCDF via h5netcdf (graphs, results).

`_not_checked_fields` (class variable, list) — fields excluded from metadata conflict checks.

### Graph types (`hammy_lib/graph.py`)

`Graph(ArrayHammyObject)` — stores `transition_matrix` and `node_positions` in `_results` Dataset;
`faces` as private `_faces` list (not persisted). Provides spectral decomposition in `calculate()`.

Constructor: `Graph(transition_matrix, node_positions, faces, id=None)`.

Key properties and methods:
- `size` — number of nodes.
- `dim` — spatial dimensionality (from `node_positions.shape[1]`).
- `faces` — list of face node-index lists.
- `triangulation` — `@cached_property`, Scipy Delaunay on `node_positions`.
- `euclidean_to_barycentric(point)` → `(simplex_idx, weights)`.
- `barycentric_to_euclidean(simplex_idx, weights)` → Euclidean point.
- `get_cells()` — returns `faces` as list of tuples (used by PositionCalculation).
- `get_node_to_cells()` — reverse index: node → cell indices.
- `node_to_coords(node_index)` → coordinate tuple (overridden in subclasses for integer row/col).

**Transition matrix convention**: row-stochastic. `T[i,j]` = P(step to j | at i). Rows sum to 1.
Off-diagonal probability is fixed at `0.5/n_directions` for all edges; blocked steps (boundary)
accumulate in the self-loop. This guarantees symmetry whenever the adjacency graph is undirected.

#### Subclasses

| Class | Directions | node_positions | faces |
|---|---|---|---|
| `LinearGraph(length)` | 2 | (N, 2), y=0 | edges |
| `LatticeGraph2D(rows, cols)` | 4 | (r, c) Euclidean | squares |
| `TriangularGraph2D(rows, cols)` | 6 | axial: (r·√3/2, c+r·0.5) | triangles |
| `HexagonalGraph2D(rows, cols)` | 3 | (r, c+0.1·sublattice) | approx. squares |
| `BrickGraph2D(rows, cols)` | 6 | (r, c+0.5·(r%2)) | approx. squares |

`TriangularGraph2D` and `BrickGraph2D` use the same 6 step directions in index space; they differ
only in `node_positions` (hence Euclidean/spatial computations differ).

### Calculations framework (`hammy_lib/calculation.py`)

`Calculation(main_input)` iterates over all coordinate combinations of `independent_dimensions`,
calls `calculate_unit()` for each slice, combines via `xr.combine_by_coords`.

`FlexDimensionCalculation(main_input, dimensions)` — `independent_dimensions` is all dims except
the listed `dimensions`.

`extend_simulation_results()` — adds cumulative sums over `level` and `TOTAL` platform; call
explicitly when working with multi-level simulation data.

### PositionCalculation (`hammy_lib/calculations/position.py`)

Decomposes an observed distribution as a sparse NNLS mixture of `T^p` columns:
1. Normalize to probability vector.
2. Find walk power `p` by matching spectral spread (monotone binary search via Brent).
3. Compute `T^p` via spectral decomposition (`eigvecs @ diag(λ^p) @ eigvecs_inv`).
4. Solve NNLS: `T^p @ w ≈ x_norm`. Threshold at 1% of peak weight.
5. Map to graph cell (simplex or GBC quad/triangle).

Two variants:
- `PositionCalculation` — 1D simplex output (continuous_position scalar).
- `CellPositionCalculation` — 2D cell output (position_row, position_col via GBC).

Fast path: `_compute_position_cell_fast` uses spatial windowing + truncated spectrum (~400× speedup).
Matching pursuit: `_compute_position_cell_mp` — greedy alternative, faster for large windows.

Algorithm registry: `POSITION_METHODS = {"nnls", "nnls_fast", "matching_pursuit"}`.

### Visualization

- `hammy_lib/vizualization.py` — 1D/xarray line plots (distributions over time).
- `hammy_lib/visualization_spatial.py` — Polyscope spatial rendering:

```python
from hammy_lib.visualization_spatial import show_distribution
show_distribution(graph, distribution, quantity_name="p")
```

Polyscope accepts `faces` as `list[list[int]]` — quads stay quads, no pre-triangulation.

## Key conventions

- Results stored in `results/<experiment_string>/` as `.json` (metadata) and `.nc` (data).
- `resolve()` handles full lifecycle: resolve dependencies → check cache → compute if needed.
- xarray string coordinates: use `xr.DataArray([...], dims='dim_name')` — NOT `pd.Index`.
  Pandas 2.x converts strings to `StringDtype` which breaks h5netcdf/np.result_type().
- `faces` lives only as `self._faces` (private, excluded from metadata). Do NOT include in
  `_results` Dataset — jagged lists can't go into NetCDF.
- `T.T @ v` for distribution propagation (T is row-stochastic, NOT column-stochastic).

## Error handling philosophy

Never hide errors via clamping, silent returns, or default values. Anomalous results are the first
signal of real bugs. Guard clauses should `raise ValueError` with context.
