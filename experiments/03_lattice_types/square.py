"""Square lattice: 4-direction walk (±x, ±y)."""
import numpy as np
import xarray as xr
from pathlib import Path
from hammy_lib.experiment import Experiment
from hammy_lib.ccode import CCode
from . import common as C


_BINS = C.get_bins("square")


class SquareLatticeExperiment(Experiment):
    experiment_number = 3
    experiment_name = "square"
    experiment_version = 1
    T = C.T
    CHECKPOINTS = C.CHECKPOINTS
    CHECKPOINTS_LEN = C.CHECKPOINTS_LEN
    TARGETS_X = C.TARGETS_X
    TARGETS_Y = C.TARGETS_Y
    TARGETS_LEN = C.TARGETS_LEN
    X_BINS_TUPLE = _BINS[0]
    Y_BINS_TUPLE = _BINS[1]
    X_BINS_LEN = _BINS[2]
    Y_BINS_LEN = _BINS[3]
    BINS_FLAT_LEN = _BINS[4]
    NUMPY_WIDTH = C.NUMPY_WIDTH

    # Reuse experiment 2's C code (same walk)
    try:
        _c_dir = Path(__file__).parent.parent / "02_lattice"
    except NameError:
        _c_dir = Path("experiments/02_lattice")
    c_code = CCode((_c_dir / "lattice.c").read_text(), C.make_c_definitions("square"))

    def create_empty_results(self) -> xr.DataArray:
        return C.create_empty_results("square")

    def simulate_using_python(self, loops: int, out: xr.DataArray, seed: int) -> None:
        rng = np.random.default_rng(seed)
        x_min, y_min = self.X_BINS_TUPLE[0], self.Y_BINS_TUPLE[0]
        x_max, y_max = self.X_BINS_TUPLE[1] - 1, self.Y_BINS_TUPLE[1] - 1
        out_vals = out.values
        for _ in range(loops):
            n = self.NUMPY_WIDTH
            dim_choices = rng.integers(0, 2, size=(n, self.T))
            dir_choices = rng.integers(0, 2, size=(n, self.T)) * 2 - 1
            dx = np.where(dim_choices == 0, dir_choices, 0)
            dy = np.where(dim_choices == 1, dir_choices, 0)
            cum_x = np.cumsum(dx, axis=1)
            cum_y = np.cumsum(dy, axis=1)
            ci = [c - 1 for c in self.CHECKPOINTS]
            raw_x, raw_y = cum_x[:, ci], cum_y[:, ci]
            cp_x = np.where(raw_x >= 0, raw_x // 2, -((-raw_x) // 2))
            cp_y = np.where(raw_y >= 0, raw_y // 2, -((-raw_y) // 2))
            for ti in range(self.TARGETS_LEN):
                mask = (cp_x[:, -1] == self.TARGETS_X[ti]) & (cp_y[:, -1] == self.TARGETS_Y[ti])
                if not mask.any():
                    continue
                for c in range(self.CHECKPOINTS_LEN):
                    bx, by = cp_x[mask, c], cp_y[mask, c]
                    v = (bx >= x_min) & (bx <= x_max) & (by >= y_min) & (by <= y_max)
                    np.add.at(out_vals[ti, c], (bx[v] - x_min, by[v] - y_min), 1)


EXPERIMENT_CLASS = SquareLatticeExperiment
