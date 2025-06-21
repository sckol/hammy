import numpy as np
import xarray as xr
import scipy.optimize
import scipy.linalg as la
from ..calculation import Calculation
from ..graph import Graph

def get_nnls_result(power, eigvals, eigvecs, x):
    eigvals_powered = eigvals.astype(np.complex64) ** power
    V = eigvecs @ np.diag(eigvals_powered) @ la.inv(eigvecs)
    nnls_result = scipy.optimize.nnls(np.real(V), np.real(x))                      
    nonzero_indices = np.where(nnls_result[0] < 1e-6, 0.0, nnls_result[0])
    nonzero_indices = np.flatnonzero(nonzero_indices)
    nonzero_values = nnls_result[0][nonzero_indices]
    score = len(nonzero_indices) * nnls_result[1]
    return nonzero_indices, nonzero_values, score

def score_for_power(power, eigvals, eigvecs, x):
    _, _, score = get_nnls_result(power, eigvals, eigvecs, x)
    return score  # negative for minimization

class PositionCalculation(Calculation):
    def __init__(self, main_input, graph: Graph, dimensionality: int, max_power, id: str = None):
        super().__init__(main_input, id)
        self.graph = graph
        self.dimensionality = dimensionality
        self.max_power = max_power

    @property
    def independent_dimensions(self):
        if "position_index" not in self.main_input.results.dims:
            raise ValueError("The input must contain 'position_index' dimension.")
        return [d for d in self.main_input.results.dims if d != "position_index"]

    @property
    def simple_type_return(self):
        return False

    def calculate_unit(self, input_array: xr.DataArray) -> xr.DataArray:
        eigvals = self.graph.results["eigenvalues"].values
        eigvecs = self.graph.results['eigenvectors'].values
        x = input_array.values        
        x = x / np.linalg.norm(x) if np.linalg.norm(x) != 0 else x
        min_power = 50.0
        max_power = self.max_power
        opt_result = scipy.optimize.differential_evolution(
            score_for_power,
            args=(eigvals, eigvecs, x),
            bounds=[(min_power, max_power)],
            #workers=-1,  # Use all available cores
            #updating='deferred'          
        )
        best_power = opt_result.x[0]
        best_nonzero_indices, best_nonzero_values, _ = get_nnls_result(best_power, eigvals, eigvecs, x)
        if len(best_nonzero_indices) > self.dimensionality:
            raise ValueError(f"Number of nonzero elements ({len(best_nonzero_indices)}) exceeds dimensionality ({self.dimensionality}). Indices: {best_nonzero_indices}, Values: {best_nonzero_values}, Power: {best_power}")
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