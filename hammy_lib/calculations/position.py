import numpy as np
import xarray as xr
import scipy.optimize
from ..calculation import Calculation
from ..graph import Graph

def get_nnls_result(power, eigvals, eigvecs_inv, eigvecs, x):
    eigvals_powered = eigvals.astype(np.complex64) ** power
    V = eigvecs @ np.diag(eigvals_powered) @ eigvecs_inv
    nnls_result = scipy.optimize.nnls(np.real(V), np.real(x))
    nonzero_indices = np.where(nnls_result[0] < 1e-6, 0.0, nnls_result[0])
    nonzero_indices = np.flatnonzero(nonzero_indices)
    nonzero_values = nnls_result[0][nonzero_indices]
    score = len(nonzero_indices) * nnls_result[1]
    return nonzero_indices, nonzero_values, score

def score_for_power(power, eigvals, eigvecs_inv, eigvecs, x):
    _, _, score = get_nnls_result(power, eigvals, eigvecs_inv, eigvecs, x)
    return score

class PositionCalculation(Calculation):
    """Describes the state of a non-standard random walk as a position on the graph.

    The problem: we run a bridged random walk with sample refusion, which produces
    a distribution over graph nodes. We want to describe this state in a simpler
    space. The zero-order approximation is just the mode (which node has the most
    probability) -- but that is too coarse, giving only discrete positions.

    The idea: decompose the observed distribution as a mixture of standard random
    walk distributions originating from different nodes. Each column of T^p gives
    the distribution of a standard walk starting at node i after p steps. We find
    a sparse non-negative mixture of these columns that best matches the observed
    distribution. The mixture weights and node indices then define a continuous
    position -- e.g. a 0.5/0.5 mix of nodes 2 and 3 means position 2.5.

    Hypothesis: for regular graphs (line, ring) this recovers the underlying
    dimensionality -- a linear graph yields mixtures of nearby nodes, giving a
    continuous 1D coordinate along the edge. The hope is that this generalizes
    the notion of position to arbitrary graphs.

    Algorithm:
      1. Normalize the observed distribution to unit norm.
      2. Search for the optimal matrix power p in [50, max_power]. For each p,
         compute T^p = V diag(lambda^p) V^{-1} via spectral decomposition, then
         solve NNLS: find non-negative weights x0 minimizing ||T^p x0 - x||.
      3. Score = (number of nonzero components) * (NNLS residual), favoring
         solutions that are both sparse and accurate.
      4. At the optimal power, extract node indices and normalized weights.
         Raise if the number of components exceeds `dimensionality`.

    Returns a flat DataArray: [index0, ..., value0, ..., power, nonzero_count].
    """

    def __init__(self, main_input, graph: Graph, dimensionality: int, max_power,
                 spatial_dims=("position_index",), id: str = None):
        super().__init__(main_input, id)
        self.graph = graph
        self.dimensionality = dimensionality
        self.max_power = max_power
        self.spatial_dims = tuple(spatial_dims)

    @property
    def independent_dimensions(self):
        return [d for d in self.main_input.results.dims if d not in self.spatial_dims]

    @property
    def simple_type_return(self):
        return False

    def calculate_unit(self, input_array: xr.DataArray, coords: dict) -> xr.DataArray:
        eigvals = self.graph.results["eigenvalues"].values
        eigvecs = self.graph.results["eigenvectors"].values
        eigvecs_inv = self.graph.results["eigenvectors_inv"].values
        if len(self.spatial_dims) > 1:
            x = input_array.stack(position_index=self.spatial_dims).values
        else:
            x = input_array.values
        x = x / np.linalg.norm(x) if np.linalg.norm(x) != 0 else x
        min_power = 50.0
        max_power = self.max_power

        opt_result = scipy.optimize.minimize_scalar(
            score_for_power,
            args=(eigvals, eigvecs_inv, eigvecs, x),
            bounds=(min_power, max_power),
            method="bounded",
        )

        best_power = opt_result.x
        best_nonzero_indices, best_nonzero_values, _ = get_nnls_result(best_power, eigvals, eigvecs_inv, eigvecs, x)
        if len(best_nonzero_indices) > self.dimensionality:
            raise ValueError(
                f"NNLS found {len(best_nonzero_indices)} nonzero components (expected {self.dimensionality}). "
                f"Indices: {best_nonzero_indices}, Values: {best_nonzero_values}, Power: {best_power}"
            )
        norm_values = best_nonzero_values / np.sum(best_nonzero_values) if np.sum(best_nonzero_values) != 0 else best_nonzero_values
        # Prepare xarray result as a single DataArray
        coords = [f"index{i}" for i in range(self.dimensionality)] + [f"value{i}" for i in range(self.dimensionality)] + ["power"] + ["nonzero_count"]
        data = np.zeros(len(coords), dtype=float)
        for i in range(self.dimensionality):
            data[i] = int(best_nonzero_indices[i]) if i < len(best_nonzero_indices) else -1
            data[self.dimensionality + i] = float(norm_values[i]) if i < len(norm_values) else 0.0
        data[-2] = float(best_power)
        data[-1] = len(best_nonzero_indices)
        result = xr.DataArray(data, dims=["position_data"], coords={"position_data": coords})
        return result
