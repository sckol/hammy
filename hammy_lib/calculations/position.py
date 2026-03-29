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
