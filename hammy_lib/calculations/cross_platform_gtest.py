"""Cross-platform G-test: compare PYTHON vs CFFI histograms for RNG quality.

If both platforms sample from the same distribution with independent RNGs,
their histograms should be statistically consistent. Large G-test values
indicate potential RNG correlation, seed duplication, or implementation bugs.

Output: chi2_pvalue per (target, checkpoint, level) combination.
High p-value (>0.05) = platforms agree = RNG quality is good.
"""
import numpy as np
import xarray as xr
from scipy import stats
from ..calculation import Calculation
from ..simulation import Simulation


class CrossPlatformGTest(Calculation):
    """Compare PYTHON and CFFI simulation histograms via chi-squared test.

    For each (target, checkpoint, level), computes a chi-squared goodness-of-fit
    test between the two platform histograms (both scaled to the same total).

    Output dimension: gtest_data = ["chi2_statistic", "df", "pvalue", "ratio"]
    where ratio = PYTHON_total / CFFI_total (should be consistent across checkpoints).
    """

    def __init__(self, main_input, spatial_dims=("x", "y"), id=None):
        super().__init__(main_input, id)
        self.spatial_dims = tuple(spatial_dims)

    @property
    def independent_dimensions(self) -> list[str]:
        base = [str(d) for d in self.main_input.results.dims
                if d not in self.spatial_dims and d != "platform"]
        return base

    @property
    def simple_type_return(self):
        return False

    def calculate_unit(self, input_array, coords):
        raise NotImplementedError("Use calculate() directly")

    def calculate(self) -> None:
        """Override to handle platform comparison directly."""
        if isinstance(self.main_input, Simulation):
            data = self.extend_simulation_results()
        else:
            data = self.main_input.results

        if "PYTHON" not in data.platform.values or "CFFI" not in data.platform.values:
            raise ValueError("Need both PYTHON and CFFI platforms for cross-platform test")

        python_data = data.sel(platform="PYTHON")
        cffi_data = data.sel(platform="CFFI")

        import itertools
        from tqdm import tqdm

        # Iterate over all non-spatial, non-platform dimensions
        dims = [d for d in data.dims if d not in self.spatial_dims and d != "platform"]
        all_coords = [data.coords[d].values for d in dims]
        all_combos = list(itertools.product(*all_coords))

        results = []
        for indices in tqdm(all_combos, desc="G-test", unit="combo"):
            coords_dict = {d: indices[i] for i, d in enumerate(dims)}

            py = python_data.sel(coords_dict)
            cf = cffi_data.sel(coords_dict)

            if len(self.spatial_dims) > 1:
                py_flat = py.stack(pos=self.spatial_dims).values.astype(float)
                cf_flat = cf.stack(pos=self.spatial_dims).values.astype(float)
            else:
                py_flat = py.values.astype(float)
                cf_flat = cf.values.astype(float)

            py_total = py_flat.sum()
            cf_total = cf_flat.sum()

            if py_total == 0 or cf_total == 0:
                result_data = np.array([np.nan, np.nan, np.nan, np.nan])
            else:
                # G-test of homogeneity: are two multinomial samples from same distribution?
                # E1_i = (k1_i + k2_i) * N1 / (N1 + N2), E2_i similarly
                N_total = py_total + cf_total
                pooled = py_flat + cf_flat
                mask = pooled > 0
                k1 = py_flat[mask]
                k2 = cf_flat[mask]
                E1 = pooled[mask] * (py_total / N_total)
                E2 = pooled[mask] * (cf_total / N_total)

                # G = 2 * Σ [k1*log(k1/E1) + k2*log(k2/E2)]
                G = 0.0
                nz1 = k1 > 0
                G += 2.0 * float(np.sum(k1[nz1] * np.log(k1[nz1] / E1[nz1])))
                nz2 = k2 > 0
                G += 2.0 * float(np.sum(k2[nz2] * np.log(k2[nz2] / E2[nz2])))

                df = int(mask.sum()) - 1
                if df <= 0:
                    result_data = np.array([np.nan, 0.0, np.nan, py_total / cf_total])
                else:
                    pvalue = float(stats.chi2.sf(G, df))
                    result_data = np.array([
                        float(G),
                        float(df),
                        float(pvalue),
                        float(py_total / cf_total),
                    ])

            output_coords = ["chi2_statistic", "df", "pvalue", "ratio"]
            result_arr = xr.DataArray(
                result_data,
                dims=["gtest_data"],
                coords={"gtest_data": output_coords},
            )
            for d in dims:
                result_arr = result_arr.expand_dims({d: [coords_dict[d]]})
            results.append(result_arr)

        self._results = xr.combine_by_coords(results)

    def generate_id(self) -> str:
        return f"{self.main_input.id}_gtest"

    @property
    def simple_name(self) -> str:
        return "GTest"
