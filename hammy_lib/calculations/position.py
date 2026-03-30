import numpy as np
import xarray as xr
import scipy.optimize
from ..calculation import Calculation
from ..graph import Graph


def _precompute_power_search(eigvals, eigvecs_inv):
    """Precompute data needed for power search. Call once per graph.

    Returns a dict to pass to _find_power_fast.
    """
    nontrivial = np.abs(eigvals) < 1.0 - 1e-10
    lam_sq = np.abs(eigvals[nontrivial]) ** 2  # use abs() to avoid complex issues
    rows_inv = eigvecs_inv[nontrivial]
    row_norms_sq = np.sum(rows_inv ** 2, axis=1).real  # precompute O(n²) once
    return {
        'nontrivial': nontrivial,
        'lam_sq': lam_sq,
        'row_norms_sq': row_norms_sq,
        'eigvecs_inv': eigvecs_inv,
        'n': eigvecs_inv.shape[1],
    }


def _find_power(eigvals, eigvecs_inv, x, p_min=1.0, p_max=10000.0, precomputed=None):
    """Find p by matching spectral spread of f to mean spectral spread of T^p columns.

    Spectral spread = energy in non-stationary eigenmodes.
    For observed f: Σ_{k≥1} (V⁻¹f)_k²
    For mean T^p column: (1/n) Σ_{k≥1} λ_k^{2p} ||row_k(V⁻¹)||²
    The latter is monotonically decreasing in p → unique root.

    If precomputed is provided (from _precompute_power_search), skips O(n²)
    row_norms_sq computation.
    """
    if precomputed is not None:
        nontrivial = precomputed['nontrivial']
        lam_sq = precomputed['lam_sq']
        row_norms_sq = precomputed['row_norms_sq']
        n = precomputed['n']
        eigvecs_inv = precomputed['eigvecs_inv']
    else:
        n = len(x)
        nontrivial = np.abs(eigvals) < 1.0 - 1e-10
        lam_sq = np.abs(eigvals[nontrivial]) ** 2
        rows_inv = eigvecs_inv[nontrivial]
        row_norms_sq = np.sum(rows_inv ** 2, axis=1).real

    # Spectral spread of observed distribution
    c = eigvecs_inv @ x
    spread_f = np.sum(np.abs(c[nontrivial]) ** 2)

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


def _compute_position_cell_fast(x_norm, eigvals, eigvecs, eigvecs_inv,
                                cells, node_to_cells, node_to_coords_fn,
                                window_padding=5, eigval_threshold=1e-3,
                                p_min=1.0, p_max=10000.0,
                                power_precomputed=None):
    """Optimized position computation: windowed NNLS + truncated spectral.

    Three optimizations over _compute_position_cell:
    1. Spatial windowing: restrict NNLS to nonzero bins ± padding.
    2. Truncated spectral: keep only eigenvalues with |λ|^p > threshold.
    3. Compute T^p only within the window: (m×k) @ (k×k) @ (k×m).

    Pass power_precomputed (from _precompute_power_search) to avoid O(n²)
    row_norms computation on every call.

    Typical speedup: 400-500x on large grids (5000→200 nodes).
    """
    power = _find_power(eigvals, eigvecs_inv, x_norm, p_min, p_max,
                        precomputed=power_precomputed)

    # --- Spatial windowing: restrict to support of x + padding ---
    support = np.flatnonzero(x_norm > 0)
    if len(support) == 0:
        return _empty_cell_result(power)

    n = len(x_norm)
    # Assume 2D grid — compute bounding box with padding
    # (works for any flat index if we just pad the index range)
    all_indices = set(support.tolist())
    # Expand: for each support node, add neighbors within padding distance
    # Simple approach: use min/max of support indices ± padding * stride
    # For row-major 2D: infer grid width from graph
    s_min, s_max = support.min(), support.max()
    # Pad by window_padding in the flat index space (conservative)
    # Better: if node_to_coords_fn is available, pad in 2D
    coords_s = np.array([node_to_coords_fn(int(i)) for i in support])
    r_min = max(0, coords_s[:, 0].min() - window_padding)
    r_max = coords_s[:, 0].max() + window_padding
    c_min = max(0, coords_s[:, 1].min() - window_padding)
    c_max = coords_s[:, 1].max() + window_padding
    # Build window index set (need to know grid dims)
    # Infer cols from node_to_coords_fn
    test_r, test_c = node_to_coords_fn(1)
    if test_r == 0:
        cols = n  # 1D graph
    else:
        cols = 1  # node 1 is in row 1 → cols = 1? No.
    # More robust: cols = max(c for r,c in all node coords) + 1
    last_r, last_c = node_to_coords_fn(n - 1)
    cols = last_c + 1
    rows = last_r + 1
    r_max = min(rows - 1, r_max)
    c_max = min(cols - 1, c_max)

    window = []
    for r in range(r_min, r_max + 1):
        for c in range(c_min, c_max + 1):
            idx = r * cols + c
            if idx < n:
                window.append(idx)
    window = np.array(window, dtype=int)
    m = len(window)

    # --- Truncated spectral: keep eigenvalues with |λ|^p > threshold ---
    lam_p_abs = np.abs(eigvals) ** power
    keep = lam_p_abs > eigval_threshold
    k = int(keep.sum())
    if k == 0:
        return _empty_cell_result(power)
    top_k = np.flatnonzero(keep)

    # --- Compute T^p within window using truncated spectrum ---
    # V_win = eigvecs[window, top_k] @ diag(λ^p[top_k]) @ eigvecs_inv[top_k, window]
    eigvals_powered = eigvals[top_k].astype(np.complex128) ** power
    V_win = np.real(
        eigvecs[np.ix_(window, top_k)]
        @ np.diag(eigvals_powered)
        @ eigvecs_inv[np.ix_(top_k, window)]
    )

    x_win = x_norm[window]

    # --- NNLS on reduced system ---
    weights_win, nnls_residual = scipy.optimize.nnls(V_win, x_win)

    threshold = 0.01 * weights_win.max() if weights_win.max() > 0 else 0.0
    nonzero_win = np.flatnonzero(weights_win >= threshold)
    # Map back to full indices
    nonzero_indices = window[nonzero_win]
    nonzero_values = weights_win[nonzero_win]
    weight_sum = nonzero_values.sum()
    norm_values = nonzero_values / weight_sum if weight_sum > 0 else nonzero_values
    n_components = len(nonzero_indices)

    # --- Cell identification + GBC (same as unoptimized path) ---
    return _assemble_cell_result(
        nonzero_indices, norm_values, n_components, power,
        nnls_residual, x_norm, cells, node_to_cells, node_to_coords_fn,
    )


def _empty_cell_result(power):
    return {
        'nonzero_indices': np.array([], dtype=int),
        'norm_values': np.array([]),
        'power': power,
        'n_components': 0,
        'residual': 0.0,
        'position_row': -1.0,
        'position_col': -1.0,
        'cell_dim': -1,
        'fit_quality': 0.0,
        'bilinear_s': 0.0,
        'bilinear_t': 0.0,
    }


def _compute_position_cell_mp(x_norm, eigvals, eigvecs, eigvecs_inv,
                              cells, node_to_cells, node_to_coords_fn,
                              window_padding=5, eigval_threshold=1e-3,
                              max_components=6,
                              p_min=1.0, p_max=10000.0,
                              power_precomputed=None):
    """Position via matching pursuit (greedy sparse NNLS).

    Same pipeline as _fast, but replaces scipy.nnls with greedy column
    selection.  Faster for large windows, but approximate — may miss
    weak distant components in non-local distributions.

    Use NNLS methods for studies involving non-locality.
    """
    power = _find_power(eigvals, eigvecs_inv, x_norm, p_min, p_max,
                        precomputed=power_precomputed)

    support = np.flatnonzero(x_norm > 0)
    if len(support) == 0:
        return _empty_cell_result(power)

    n = len(x_norm)
    coords_s = np.array([node_to_coords_fn(int(i)) for i in support])
    last_r, last_c = node_to_coords_fn(n - 1)
    graph_cols = last_c + 1
    graph_rows = last_r + 1
    r_min = max(0, int(coords_s[:, 0].min()) - window_padding)
    r_max = min(graph_rows - 1, int(coords_s[:, 0].max()) + window_padding)
    c_min = max(0, int(coords_s[:, 1].min()) - window_padding)
    c_max = min(graph_cols - 1, int(coords_s[:, 1].max()) + window_padding)
    window = np.array([r * graph_cols + c
                       for r in range(r_min, r_max + 1)
                       for c in range(c_min, c_max + 1)
                       if r * graph_cols + c < n], dtype=int)
    m = len(window)

    lam_p_abs = np.abs(eigvals) ** power
    top_k = np.flatnonzero(lam_p_abs > eigval_threshold)
    if len(top_k) == 0:
        return _empty_cell_result(power)
    eigvals_powered = eigvals[top_k].astype(np.complex128) ** power
    V_win = np.real(
        eigvecs[np.ix_(window, top_k)]
        @ np.diag(eigvals_powered)
        @ eigvecs_inv[np.ix_(top_k, window)]
    )
    x_win = x_norm[window]

    # --- Greedy matching pursuit ---
    residual = x_win.copy()
    selected = []
    for _ in range(max_components):
        correlations = V_win.T @ residual
        correlations[correlations < 0] = 0
        for s in selected:
            correlations[s] = 0
        best = int(np.argmax(correlations))
        if correlations[best] < 0.001 * np.linalg.norm(x_win):
            break
        selected.append(best)
        V_sel = V_win[:, selected]
        w_sel, _ = scipy.optimize.nnls(V_sel, x_win)
        keep = w_sel > 0.001 * w_sel.max()
        selected = [selected[i] for i in range(len(selected)) if keep[i]]
        w_sel = w_sel[keep]
        residual = x_win - V_win[:, selected] @ w_sel

    if not selected:
        return _empty_cell_result(power)

    nnls_residual = float(np.linalg.norm(residual))
    nonzero_indices = window[np.array(selected)]
    nonzero_values = w_sel
    weight_sum = nonzero_values.sum()
    norm_values = nonzero_values / weight_sum if weight_sum > 0 else nonzero_values
    n_components = len(nonzero_indices)

    return _assemble_cell_result(
        nonzero_indices, norm_values, n_components, power,
        nnls_residual, x_norm, cells, node_to_cells, node_to_coords_fn,
    )


# --- Algorithm registry ---
POSITION_METHODS = {
    "nnls": "_compute_position_cell",
    "nnls_fast": "_compute_position_cell_fast",
    "matching_pursuit": "_compute_position_cell_mp",
}


def _assemble_cell_result(nonzero_indices, norm_values, n_components, power,
                          nnls_residual, x_norm, cells, node_to_cells,
                          node_to_coords_fn):
    """Shared post-NNLS logic: cell identification + GBC + position."""
    if n_components > 0:
        cell_nodes, (s, t), cell_dim, fit_quality = _identify_cell(
            nonzero_indices, norm_values, cells, node_to_cells
        )
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
            nw = norm_values[:len(cell_nodes)] / norm_values[:len(cell_nodes)].sum()
            position_row = float(sum(w * node_to_coords_fn(int(n))[0] for w, n in zip(nw, cell_nodes)))
            position_col = float(sum(w * node_to_coords_fn(int(n))[1] for w, n in zip(nw, cell_nodes)))
    else:
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


def _compute_position_cell(x_norm, eigvals, eigvecs, eigvecs_inv, adjacency,
                           cells, node_to_cells, node_to_coords_fn,
                           p_min=1.0, p_max=10000.0):
    """Position computation using cell identification + GBCs (unoptimized).

    Same as _compute_position steps 1-3, then _identify_cell instead of _identify_simplex.
    Returns 2D continuous position (row, col).

    For large grids, use _compute_position_cell_fast instead.
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

    return _assemble_cell_result(
        nonzero_indices, norm_values, n_components, power,
        nnls_residual, x_norm, cells, node_to_cells, node_to_coords_fn,
    )


class CellPositionCalculation(Calculation):
    """Position on graph using cell identification + Generalized Barycentric Coordinates.

    For 2D lattice graphs, cells are squares and GBCs reduce to bilinear
    interpolation.  Returns 2D continuous position (row, col) instead of
    a single continuous_position scalar.
    """

    DIMENSIONALITY = 4  # max cell corners
    P_MIN = 1.0
    P_MAX = 10000.0

    def __init__(self, main_input, graph, spatial_dims=("position_index",),
                 method="nnls_fast", id=None):
        super().__init__(main_input, id)
        self.graph = graph
        self.spatial_dims = tuple(spatial_dims)
        self.method = method

    @property
    def independent_dimensions(self) -> list[str]:
        return [str(d) for d in self.main_input.results.dims if d not in self.spatial_dims]

    @property
    def simple_type_return(self):
        return False

    def _ensure_precomputed(self):
        """Precompute expensive data once (called on first calculate_unit)."""
        if hasattr(self, '_power_precomputed'):
            return
        eigvals = self.graph.results["eigenvalues"].values
        eigvecs_inv = self.graph.results["eigenvectors_inv"].values
        self._power_precomputed = _precompute_power_search(eigvals, eigvecs_inv)
        self._cells = self.graph.get_cells()
        self._node_to_cells = self.graph.get_node_to_cells()

    def calculate_unit(self, input_array: xr.DataArray, coords: dict) -> xr.DataArray:
        self._ensure_precomputed()
        eigvals = self.graph.results["eigenvalues"].values
        eigvecs = self.graph.results["eigenvectors"].values
        eigvecs_inv = self.graph.results["eigenvectors_inv"].values

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

        if self.method == "nnls":
            tm = self.graph.results["transition_matrix"].values
            adjacency = (tm > 0) & ~np.eye(tm.shape[0], dtype=bool)
            result = _compute_position_cell(
                x_norm, eigvals, eigvecs, eigvecs_inv, adjacency,
                self._cells, self._node_to_cells, self.graph.node_to_coords,
                self.P_MIN, self.P_MAX,
            )
        elif self.method == "matching_pursuit":
            result = _compute_position_cell_mp(
                x_norm, eigvals, eigvecs, eigvecs_inv,
                self._cells, self._node_to_cells, self.graph.node_to_coords,
                p_min=self.P_MIN, p_max=self.P_MAX,
                power_precomputed=self._power_precomputed,
            )
        else:  # "nnls_fast" (default)
            result = _compute_position_cell_fast(
                x_norm, eigvals, eigvecs, eigvecs_inv,
                self._cells, self._node_to_cells, self.graph.node_to_coords,
                p_min=self.P_MIN, p_max=self.P_MAX,
                power_precomputed=self._power_precomputed,
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
