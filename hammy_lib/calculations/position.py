import numpy as np
import xarray as xr
import scipy.optimize
from ..calculation import Calculation
from ..graph import Graph


class PositionCalculation(Calculation):
    """Describes the state of a non-standard random walk as a position on the graph.

    Algorithm:
      1. Normalize the observed distribution to a probability vector (sum=1).
      2. Find walk power p by matching spectral spread (diffusion variance).
         Spectral spread of f = Σ_{k≥1} c_k² where c = V⁻¹f (energy in
         non-stationary eigenmodes). Match to mean column spread at power p:
         (1/n) Σ_{k≥1} λ_k^{2p} ||row_k(V⁻¹)||². This is monotonically
         decreasing in p, so solved by root-finding. Topology-agnostic.
      3. At optimal p, compute T^p via spectral decomposition.
      4. Solve NNLS: find non-negative weights minimizing ||T^p w - f||₂.
      5. Threshold: discard components with weight < 1% of the peak weight.
      6. Normalize remaining weights to sum to 1.

    Returns a flat DataArray: [index0, ..., value0, ..., power, nonzero_count].

    Note: the output position (weighted sum of node indices) assumes a lattice
    embedding. For non-lattice graphs, the NNLS decomposition is still valid
    but the position needs a different representation.
    """

    P_MIN = 1.0
    P_MAX = 10000.0

    def __init__(self, main_input, graph: Graph, dimensionality: int,
                 spatial_dims=("position_index",), id: str = None):
        super().__init__(main_input, id)
        self.graph = graph
        self.dimensionality = dimensionality
        self.spatial_dims = tuple(spatial_dims)

    @property
    def independent_dimensions(self) -> list[str]:
        return [str(d) for d in self.main_input.results.dims if d not in self.spatial_dims]

    @property
    def simple_type_return(self):
        return False

    def _find_power(self, eigvals, eigvecs_inv, x) -> float:
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

        spread_at_min = mean_column_spread(self.P_MIN)
        if spread_f >= spread_at_min:
            return self.P_MIN

        spread_at_max = mean_column_spread(self.P_MAX)
        if spread_f <= spread_at_max:
            return self.P_MAX

        return scipy.optimize.brentq(
            lambda p: mean_column_spread(p) - spread_f,
            self.P_MIN, self.P_MAX,
        )

    def calculate_unit(self, input_array: xr.DataArray, coords: dict) -> xr.DataArray:
        eigvals = self.graph.results["eigenvalues"].values
        eigvecs = self.graph.results["eigenvectors"].values
        eigvecs_inv = self.graph.results["eigenvectors_inv"].values

        if len(self.spatial_dims) > 1:
            x = input_array.stack(position_index=self.spatial_dims).values.astype(float)
        else:
            x = input_array.values.astype(float)

        total = x.sum()
        if total == 0:
            raise ValueError(f"Zero distribution at coords {coords} — cannot compute position")
        x = x / total

        # Find power by diffusion variance matching
        power = self._find_power(eigvals, eigvecs_inv, x)

        # T^p via spectral decomposition
        eigvals_powered = eigvals.astype(np.complex128) ** power
        V = np.real(eigvecs @ np.diag(eigvals_powered) @ eigvecs_inv)

        # NNLS (L2)
        weights, _ = scipy.optimize.nnls(V, x)

        # Threshold and normalize
        threshold = 0.01 * weights.max() if weights.max() > 0 else 0.0
        nonzero_indices = np.flatnonzero(weights >= threshold)
        nonzero_values = weights[nonzero_indices]
        weight_sum = nonzero_values.sum()
        norm_values = nonzero_values / weight_sum if weight_sum > 0 else nonzero_values
        n_components = len(nonzero_indices)

        # Keep top MAX_COMPONENTS by weight for position output
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
            + ["power", "nonzero_count"]
        )
        data = np.zeros(len(output_coords), dtype=float)
        for i in range(self.dimensionality):
            data[i] = float(top_indices[i]) if i < len(top_indices) else -1.0
            data[self.dimensionality + i] = float(top_values[i]) if i < len(top_values) else 0.0
        data[-2] = float(power)
        data[-1] = float(n_components)

        return xr.DataArray(data, dims=["position_data"], coords={"position_data": output_coords})
