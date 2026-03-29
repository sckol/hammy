import numpy as np
import xarray as xr
import scipy.optimize
from ..calculation import Calculation
from ..graph import Graph


def _find_power(eigvals, eigvecs_inv, x, p_min=1.0, p_max=10000.0):
    """Find p by matching spectral spread of f to mean spectral spread of T^p columns.

    Spectral spread = energy in non-stationary eigenmodes.
    For observed f: Σ_{k≥1} (V⁻¹f)_k²
    For mean T^p column: (1/n) Σ_{k≥1} λ_k^{2p} ||row_k(V⁻¹)||²
    The latter is monotonically decreasing in p → unique root.
    """
    n = len(x)

    # Identify non-stationary modes (|λ| < 1)
    nontrivial = np.abs(eigvals) < 1.0 - 1e-10
    lam_sq = eigvals[nontrivial].astype(float) ** 2
    rows_inv = eigvecs_inv[nontrivial]

    # Spectral spread of observed distribution
    c = eigvecs_inv @ x
    spread_f = np.sum(c[nontrivial] ** 2)

    # Row norms: ||row_k(V⁻¹)||²
    row_norms_sq = np.sum(rows_inv ** 2, axis=1)

    def mean_column_spread(p):
        return np.sum(lam_sq ** p * row_norms_sq) / n

    spread_at_min = mean_column_spread(p_min)
    if spread_f >= spread_at_min:
        return p_min

    spread_at_max = mean_column_spread(p_max)
    if spread_f <= spread_at_max:
        return p_max

    return scipy.optimize.brentq(
        lambda p: mean_column_spread(p) - spread_f,
        p_min, p_max,
    )


def _identify_simplex(indices, weights, adjacency, max_dim):
    """Map weighted graph nodes to the best-fitting simplex.

    Greedy: start with highest-weight node, add next only if adjacent to all
    existing simplex nodes and simplex dimension <= max_dim.

    Returns (simplex_nodes, barycentric_coords, fit_quality).
    fit_quality = fraction of total weight captured by the simplex.
    """
    order = np.argsort(weights)[::-1]
    sorted_indices = indices[order]
    sorted_weights = weights[order]

    simplex = [sorted_indices[0]]
    simplex_weights = [sorted_weights[0]]

    for idx, w in zip(sorted_indices[1:], sorted_weights[1:]):
        if len(simplex) > max_dim:
            break
        if all(adjacency[idx, s] for s in simplex):
            simplex.append(idx)
            simplex_weights.append(w)

    simplex_nodes = np.array(simplex)
    bary = np.array(simplex_weights)
    bary = bary / bary.sum()
    fit_quality = float(np.sum(simplex_weights) / weights.sum())

    return simplex_nodes, bary, fit_quality


def _compute_position(x_norm, eigvals, eigvecs, eigvecs_inv, adjacency,
                      max_simplex_dim=1, p_min=1.0, p_max=10000.0):
    """Core position computation reusable by PositionCalculation and bootstrap.

    Args:
        x_norm: normalized probability vector (sum=1).
        eigvals, eigvecs, eigvecs_inv: graph spectral decomposition.
        adjacency: boolean adjacency matrix (T > 0, no self-loops).
        max_simplex_dim: max simplex dimension (1=edges, 2=faces).
        p_min, p_max: power search bounds.

    Returns dict with:
        nonzero_indices, norm_values, power, n_components,
        residual, continuous_position, simplex_dim, fit_quality.
    """
    power = _find_power(eigvals, eigvecs_inv, x_norm, p_min, p_max)

    # T^p via spectral decomposition
    eigvals_powered = eigvals.astype(np.complex128) ** power
    V = np.real(eigvecs @ np.diag(eigvals_powered) @ eigvecs_inv)

    # NNLS
    weights, nnls_residual = scipy.optimize.nnls(V, x_norm)

    # Threshold and normalize
    threshold = 0.01 * weights.max() if weights.max() > 0 else 0.0
    nonzero_indices = np.flatnonzero(weights >= threshold)
    nonzero_values = weights[nonzero_indices]
    weight_sum = nonzero_values.sum()
    norm_values = nonzero_values / weight_sum if weight_sum > 0 else nonzero_values
    n_components = len(nonzero_indices)

    # Simplex identification
    if n_components > 0:
        simplex_nodes, bary, fit_quality = _identify_simplex(
            nonzero_indices, norm_values, adjacency, max_simplex_dim
        )
        continuous_position = float(np.sum(simplex_nodes * bary))
        simplex_dim = len(simplex_nodes) - 1
    else:
        continuous_position = -1.0
        simplex_dim = -1
        fit_quality = 0.0

    x_norm_norm = np.linalg.norm(x_norm)
    relative_residual = float(nnls_residual / x_norm_norm) if x_norm_norm > 0 else 0.0

    return {
        'nonzero_indices': nonzero_indices,
        'norm_values': norm_values,
        'power': power,
        'n_components': n_components,
        'residual': relative_residual,
        'continuous_position': continuous_position,
        'simplex_dim': simplex_dim,
        'fit_quality': fit_quality,
    }


def _identify_cell(indices, weights, cells, node_to_cells):
    """Map weighted graph nodes to the best-fitting cell using GBCs.

    Dispatches based on cell size:
    - 3-node (triangle): barycentric coordinates → (s, t) where position =
      v0 + s*(v1-v0) + t*(v2-v0), i.e. s = w1, t = w2.
    - 4-node (quad): inverse bilinear → s = w10+w11, t = w01+w11.
    - Other: centroid (weighted average of all cell node weights).

    Returns (cell_nodes, (s, t), cell_dim, fit_quality).
    """
    index_to_weight = dict(zip(indices, weights))

    # Find the cell capturing the most weight
    best_cell_idx = -1
    best_weight = 0.0
    seen_cells = set()
    for idx in indices:
        for ci in node_to_cells.get(int(idx), []):
            if ci in seen_cells:
                continue
            seen_cells.add(ci)
            cell = cells[ci]
            cell_weight = sum(index_to_weight.get(n, 0.0) for n in cell)
            if cell_weight > best_weight:
                best_weight = cell_weight
                best_cell_idx = ci

    if best_cell_idx < 0:
        # No cell found — vertex fallback
        top = np.argmax(weights)
        return np.array([indices[top]]), (0.0, 0.0), 0, float(weights[top])

    cell = cells[best_cell_idx]
    cell_weights = [index_to_weight.get(n, 0.0) for n in cell]
    w_total = sum(cell_weights)
    fit_quality = float(w_total / weights.sum()) if weights.sum() > 0 else 0.0
    nonzero_corners = sum(1 for w in cell_weights if w > 0)
    cell_dim = max(0, nonzero_corners - 1)
    cell_nodes = np.array(cell)

    if w_total <= 0:
        return cell_nodes, (0.0, 0.0), 0, fit_quality

    if len(cell) == 3:
        # Triangle: barycentric coordinates
        # cell = (n0, n1, n2). s = w1/total, t = w2/total.
        # Position = n0 + s*(n1-n0) + t*(n2-n0) = (1-s-t)*n0 + s*n1 + t*n2
        s = cell_weights[1] / w_total
        t = cell_weights[2] / w_total
    elif len(cell) == 4:
        # Quad: inverse bilinear
        # cell = (n00, n10, n01, n11)
        s = (cell_weights[1] + cell_weights[3]) / w_total  # right side
        t = (cell_weights[2] + cell_weights[3]) / w_total  # bottom side
    else:
        # General: centroid-based (s, t as fraction along cell span)
        norm_w = [w / w_total for w in cell_weights]
        s = sum(i * w for i, w in enumerate(norm_w))  # weighted index
        t = 0.0  # 1D for general cells

    return cell_nodes, (s, t), cell_dim, fit_quality


def _compute_position_cell(x_norm, eigvals, eigvecs, eigvecs_inv, adjacency,
                           cells, node_to_cells, node_to_coords_fn,
                           p_min=1.0, p_max=10000.0):
    """Position computation using cell identification + GBCs.

    Same as _compute_position steps 1-3, then _identify_cell instead of _identify_simplex.
    Returns 2D continuous position (row, col).
    """
    power = _find_power(eigvals, eigvecs_inv, x_norm, p_min, p_max)

    eigvals_powered = eigvals.astype(np.complex128) ** power
    V = np.real(eigvecs @ np.diag(eigvals_powered) @ eigvecs_inv)

    weights, nnls_residual = scipy.optimize.nnls(V, x_norm)

    threshold = 0.01 * weights.max() if weights.max() > 0 else 0.0
    nonzero_indices = np.flatnonzero(weights >= threshold)
    nonzero_values = weights[nonzero_indices]
    weight_sum = nonzero_values.sum()
    norm_values = nonzero_values / weight_sum if weight_sum > 0 else nonzero_values
    n_components = len(nonzero_indices)

    if n_components > 0:
        cell_nodes, (s, t), cell_dim, fit_quality = _identify_cell(
            nonzero_indices, norm_values, cells, node_to_cells
        )
        # Compute position using GBC weights applied to node coordinates.
        # For triangle: w0=1-s-t, w1=s, w2=t → pos = w0*p0 + w1*p1 + w2*p2
        # For quad: bilinear → pos = (1-t)*((1-s)*p00 + s*p10) + t*((1-s)*p01 + s*p11)
        coords = [node_to_coords_fn(int(n)) for n in cell_nodes]
        if len(cell_nodes) == 3:
            w0, w1, w2 = 1 - s - t, s, t
            position_row = float(w0 * coords[0][0] + w1 * coords[1][0] + w2 * coords[2][0])
            position_col = float(w0 * coords[0][1] + w1 * coords[1][1] + w2 * coords[2][1])
        elif len(cell_nodes) == 4:
            position_row = float((1 - t) * ((1 - s) * coords[0][0] + s * coords[1][0])
                                 + t * ((1 - s) * coords[2][0] + s * coords[3][0]))
            position_col = float((1 - t) * ((1 - s) * coords[0][1] + s * coords[1][1])
                                 + t * ((1 - s) * coords[2][1] + s * coords[3][1]))
        else:
            # Fallback: weighted centroid
            norm_w = norm_values[:len(cell_nodes)] / norm_values[:len(cell_nodes)].sum()
            position_row = float(sum(w * node_to_coords_fn(int(n))[0] for w, n in zip(norm_w, cell_nodes)))
            position_col = float(sum(w * node_to_coords_fn(int(n))[1] for w, n in zip(norm_w, cell_nodes)))
    else:
        cell_nodes = np.array([])
        s, t = 0.0, 0.0
        cell_dim = -1
        fit_quality = 0.0
        position_row = -1.0
        position_col = -1.0

    x_norm_norm = np.linalg.norm(x_norm)
    relative_residual = float(nnls_residual / x_norm_norm) if x_norm_norm > 0 else 0.0

    return {
        'nonzero_indices': nonzero_indices,
        'norm_values': norm_values,
        'power': power,
        'n_components': n_components,
        'residual': relative_residual,
        'position_row': position_row,
        'position_col': position_col,
        'cell_dim': cell_dim,
        'fit_quality': fit_quality,
        'bilinear_s': s,
        'bilinear_t': t,
    }


class CellPositionCalculation(Calculation):
    """Position on graph using cell identification + Generalized Barycentric Coordinates.

    For 2D lattice graphs, cells are squares and GBCs reduce to bilinear
    interpolation.  Returns 2D continuous position (row, col) instead of
    a single continuous_position scalar.
    """

    DIMENSIONALITY = 4  # max cell corners
    P_MIN = 1.0
    P_MAX = 10000.0

    def __init__(self, main_input, graph, spatial_dims=("position_index",), id=None):
        super().__init__(main_input, id)
        self.graph = graph
        self.spatial_dims = tuple(spatial_dims)

    @property
    def independent_dimensions(self) -> list[str]:
        return [str(d) for d in self.main_input.results.dims if d not in self.spatial_dims]

    @property
    def simple_type_return(self):
        return False

    def calculate_unit(self, input_array: xr.DataArray, coords: dict) -> xr.DataArray:
        eigvals = self.graph.results["eigenvalues"].values
        eigvecs = self.graph.results["eigenvectors"].values
        eigvecs_inv = self.graph.results["eigenvectors_inv"].values

        tm = self.graph.results["transition_matrix"].values
        n = tm.shape[0]
        adjacency = (tm > 0) & ~np.eye(n, dtype=bool)

        cells = self.graph.get_cells()
        node_to_cells = self.graph.get_node_to_cells()

        if len(self.spatial_dims) > 1:
            x = input_array.stack(position_index=self.spatial_dims).values.astype(float)
        else:
            x = input_array.values.astype(float)

        total = x.sum()
        if total == 0:
            # No data for this combination — return NaN-filled result
            D = self.DIMENSIONALITY
            output_coords = (
                [f"index{i}" for i in range(D)]
                + [f"value{i}" for i in range(D)]
                + ["power", "nonzero_count",
                   "position_row", "position_col",
                   "residual", "cell_dim", "fit_quality",
                   "bilinear_s", "bilinear_t"]
            )
            data = np.full(len(output_coords), np.nan)
            return xr.DataArray(data, dims=["position_data"], coords={"position_data": output_coords})
        x_norm = x / total

        result = _compute_position_cell(
            x_norm, eigvals, eigvecs, eigvecs_inv, adjacency,
            cells, node_to_cells, self.graph.node_to_coords,
            self.P_MIN, self.P_MAX,
        )

        nonzero_indices = result['nonzero_indices']
        norm_values = result['norm_values']
        n_components = result['n_components']

        if n_components > self.DIMENSIONALITY:
            top = np.argsort(norm_values)[-self.DIMENSIONALITY:][::-1]
            top_indices = nonzero_indices[top]
            top_values = norm_values[top]
        else:
            top_indices = nonzero_indices
            top_values = norm_values

        D = self.DIMENSIONALITY
        output_coords = (
            [f"index{i}" for i in range(D)]
            + [f"value{i}" for i in range(D)]
            + ["power", "nonzero_count",
               "position_row", "position_col",
               "residual", "cell_dim", "fit_quality",
               "bilinear_s", "bilinear_t"]
        )
        data = np.zeros(len(output_coords), dtype=float)
        for i in range(D):
            data[i] = float(top_indices[i]) if i < len(top_indices) else -1.0
            data[D + i] = float(top_values[i]) if i < len(top_values) else 0.0
        data[2 * D] = float(result['power'])
        data[2 * D + 1] = float(n_components)
        data[2 * D + 2] = result['position_row']
        data[2 * D + 3] = result['position_col']
        data[2 * D + 4] = result['residual']
        data[2 * D + 5] = float(result['cell_dim'])
        data[2 * D + 6] = result['fit_quality']
        data[2 * D + 7] = result['bilinear_s']
        data[2 * D + 8] = result['bilinear_t']

        return xr.DataArray(data, dims=["position_data"], coords={"position_data": output_coords})


class PositionCalculation(Calculation):
    """Describes the state of a non-standard random walk as a position on the graph.

    Algorithm:
      1. Normalize the observed distribution to a probability vector (sum=1).
      2. Find walk power p by matching spectral spread (diffusion variance).
      3. At optimal p, compute T^p via spectral decomposition.
      4. Solve NNLS: find non-negative weights minimizing ||T^p w - f||₂.
      5. Threshold: discard components with weight < 1% of the peak weight.
      6. Normalize remaining weights to sum to 1.
      7. Identify best-fitting graph simplex (vertex/edge/face) from components.

    Returns a flat DataArray:
      [index0, ..., value0, ..., power, nonzero_count,
       continuous_position, residual, simplex_dim, fit_quality].
    """

    P_MIN = 1.0
    P_MAX = 10000.0

    def __init__(self, main_input, graph: Graph, dimensionality: int,
                 max_simplex_dim: int = 1,
                 spatial_dims=("position_index",), id: str = None):
        super().__init__(main_input, id)
        self.graph = graph
        self.dimensionality = dimensionality
        self.max_simplex_dim = max_simplex_dim
        self.spatial_dims = tuple(spatial_dims)

    @property
    def independent_dimensions(self) -> list[str]:
        return [str(d) for d in self.main_input.results.dims if d not in self.spatial_dims]

    @property
    def simple_type_return(self):
        return False

    def calculate_unit(self, input_array: xr.DataArray, coords: dict) -> xr.DataArray:
        eigvals = self.graph.results["eigenvalues"].values
        eigvecs = self.graph.results["eigenvectors"].values
        eigvecs_inv = self.graph.results["eigenvectors_inv"].values

        tm = self.graph.results["transition_matrix"].values
        n = tm.shape[0]
        adjacency = (tm > 0) & ~np.eye(n, dtype=bool)

        if len(self.spatial_dims) > 1:
            x = input_array.stack(position_index=self.spatial_dims).values.astype(float)
        else:
            x = input_array.values.astype(float)

        total = x.sum()
        if total == 0:
            raise ValueError(f"Zero distribution at coords {coords} — cannot compute position")
        x_norm = x / total

        result = _compute_position(
            x_norm, eigvals, eigvecs, eigvecs_inv, adjacency,
            self.max_simplex_dim, self.P_MIN, self.P_MAX,
        )

        nonzero_indices = result['nonzero_indices']
        norm_values = result['norm_values']
        n_components = result['n_components']

        # Keep top dimensionality components by weight for backward-compatible output
        if n_components > self.dimensionality:
            top = np.argsort(norm_values)[-self.dimensionality:][::-1]
            top_indices = nonzero_indices[top]
            top_values = norm_values[top]
        else:
            top_indices = nonzero_indices
            top_values = norm_values

        output_coords = (
            [f"index{i}" for i in range(self.dimensionality)]
            + [f"value{i}" for i in range(self.dimensionality)]
            + ["power", "nonzero_count",
               "continuous_position", "residual", "simplex_dim", "fit_quality"]
        )
        data = np.zeros(len(output_coords), dtype=float)
        for i in range(self.dimensionality):
            data[i] = float(top_indices[i]) if i < len(top_indices) else -1.0
            data[self.dimensionality + i] = float(top_values[i]) if i < len(top_values) else 0.0
        data[2 * self.dimensionality] = float(result['power'])
        data[2 * self.dimensionality + 1] = float(n_components)
        data[2 * self.dimensionality + 2] = result['continuous_position']
        data[2 * self.dimensionality + 3] = result['residual']
        data[2 * self.dimensionality + 4] = float(result['simplex_dim'])
        data[2 * self.dimensionality + 5] = result['fit_quality']

        return xr.DataArray(data, dims=["position_data"], coords={"position_data": output_coords})
