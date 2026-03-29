import numpy as np
import scipy.optimize
import xarray as xr
from ..calculation import Calculation
from ..graph import Graph
from .position import _find_power, _identify_simplex, _identify_cell


class BootstrapPositionCalculation(Calculation):
    """Bootstrap confidence intervals for continuous position.

    Resamples the observed histogram counts via multinomial, then for each
    resample runs only the NNLS step (with T^p precomputed from the original
    distribution) and simplex identification. This avoids recomputing the
    power search and matrix exponentiation per sample.

    Output DataArray coords (bootstrap_data dimension):
      bootstrap_mean, bootstrap_std, bootstrap_p05, bootstrap_p95
    """

    def __init__(self, main_input, graph: Graph, dimensionality: int,
                 max_simplex_dim: int = 1, n_bootstrap: int = 200,
                 spatial_dims=("position_index",), id: str = None):
        super().__init__(main_input, id)
        self.graph = graph
        self.dimensionality = dimensionality
        self.max_simplex_dim = max_simplex_dim
        self.n_bootstrap = n_bootstrap
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

        # Precompute T^p once from the original distribution
        power = _find_power(eigvals, eigvecs_inv, x_norm)
        eigvals_powered = eigvals.astype(np.complex128) ** power
        V = np.real(eigvecs @ np.diag(eigvals_powered) @ eigvecs_inv)

        # Bootstrap: resample histogram, re-run only NNLS + simplex ID
        rng = np.random.default_rng(42)
        p = x_norm

        positions = []
        for _ in range(self.n_bootstrap):
            x_boot = rng.multinomial(int(total), p).astype(float)
            boot_total = x_boot.sum()
            if boot_total == 0:
                continue
            x_boot_norm = x_boot / boot_total

            weights, _ = scipy.optimize.nnls(V, x_boot_norm)

            # Threshold and normalize
            threshold = 0.01 * weights.max() if weights.max() > 0 else 0.0
            nonzero_indices = np.flatnonzero(weights >= threshold)
            nonzero_values = weights[nonzero_indices]
            weight_sum = nonzero_values.sum()
            norm_values = nonzero_values / weight_sum if weight_sum > 0 else nonzero_values

            if len(nonzero_indices) > 0:
                simplex_nodes, bary, _ = _identify_simplex(
                    nonzero_indices, norm_values, adjacency, self.max_simplex_dim,
                )
                positions.append(float(np.sum(simplex_nodes * bary)))
            else:
                positions.append(-1.0)

        positions = np.array(positions)

        output_coords = ["bootstrap_mean", "bootstrap_std",
                         "bootstrap_p05", "bootstrap_p95"]
        data = np.array([
            float(np.mean(positions)),
            float(np.std(positions)),
            float(np.percentile(positions, 5)),
            float(np.percentile(positions, 95)),
        ])

        return xr.DataArray(data, dims=["bootstrap_data"],
                            coords={"bootstrap_data": output_coords})


class BootstrapCellPositionCalculation(Calculation):
    """Bootstrap confidence intervals for 2D cell-based position.

    Uses _identify_cell + GBCs instead of simplex identification.
    Returns bootstrap stats for both position_row and position_col.
    """

    def __init__(self, main_input, graph: Graph, n_bootstrap: int = 200,
                 spatial_dims=("position_index",), id: str = None):
        super().__init__(main_input, id)
        self.graph = graph
        self.n_bootstrap = n_bootstrap
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

        cells = self.graph.get_cells()
        node_to_cells = self.graph.get_node_to_cells()

        if len(self.spatial_dims) > 1:
            x = input_array.stack(position_index=self.spatial_dims).values.astype(float)
        else:
            x = input_array.values.astype(float)

        total = x.sum()
        if total == 0:
            raise ValueError(f"Zero distribution at coords {coords} — cannot compute position")

        x_norm = x / total

        power = _find_power(eigvals, eigvecs_inv, x_norm)
        eigvals_powered = eigvals.astype(np.complex128) ** power
        V = np.real(eigvecs @ np.diag(eigvals_powered) @ eigvecs_inv)

        rng = np.random.default_rng(42)
        rows_list = []
        cols_list = []

        for _ in range(self.n_bootstrap):
            x_boot = rng.multinomial(int(total), x_norm).astype(float)
            boot_total = x_boot.sum()
            if boot_total == 0:
                continue
            x_boot_norm = x_boot / boot_total

            weights, _ = scipy.optimize.nnls(V, x_boot_norm)

            threshold = 0.01 * weights.max() if weights.max() > 0 else 0.0
            nonzero_indices = np.flatnonzero(weights >= threshold)
            nonzero_values = weights[nonzero_indices]
            weight_sum = nonzero_values.sum()
            norm_values = nonzero_values / weight_sum if weight_sum > 0 else nonzero_values

            if len(nonzero_indices) > 0:
                cell_nodes, (s, t), _, _ = _identify_cell(
                    nonzero_indices, norm_values, cells, node_to_cells,
                )
                r0, c0 = self.graph.node_to_coords(int(cell_nodes[0]))
                rows_list.append(float(r0 + t))
                cols_list.append(float(c0 + s))
            else:
                rows_list.append(-1.0)
                cols_list.append(-1.0)

        rows_arr = np.array(rows_list)
        cols_arr = np.array(cols_list)

        output_coords = [
            "bootstrap_row_mean", "bootstrap_row_std",
            "bootstrap_col_mean", "bootstrap_col_std",
        ]
        data = np.array([
            float(np.mean(rows_arr)), float(np.std(rows_arr)),
            float(np.mean(cols_arr)), float(np.std(cols_arr)),
        ])

        return xr.DataArray(data, dims=["bootstrap_data"],
                            coords={"bootstrap_data": output_coords})
