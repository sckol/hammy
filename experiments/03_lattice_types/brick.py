"""Brick (offset rectangular) lattice: 4-direction walk with row-parity offset."""
import numpy as np
import xarray as xr
from pathlib import Path
from hammy_lib.experiment import Experiment
from hammy_lib.ccode import CCode
from . import common as C


class BrickLatticeExperiment(Experiment):
    experiment_number = 3
    experiment_name = "brick"
    experiment_version = 1
    T = C.T
    CHECKPOINTS = C.CHECKPOINTS
    CHECKPOINTS_LEN = C.CHECKPOINTS_LEN
    TARGETS_X = C.TARGETS_X
    TARGETS_Y = C.TARGETS_Y
    TARGETS_LEN = C.TARGETS_LEN
    X_BINS_TUPLE = C.X_BINS_TUPLE
    Y_BINS_TUPLE = C.Y_BINS_TUPLE
    X_BINS_LEN = C.X_BINS_LEN
    Y_BINS_LEN = C.Y_BINS_LEN
    BINS_FLAT_LEN = C.BINS_FLAT_LEN
    NUMPY_WIDTH = C.NUMPY_WIDTH

    try:
        _c_dir = Path(__file__).parent
    except NameError:
        _c_dir = Path("experiments/03_lattice_types")
    c_code = CCode((_c_dir / "brick.c").read_text(), C.C_DEFINITIONS)

    def create_empty_results(self) -> xr.DataArray:
        return C.create_empty_results()

    def simulate_using_python(self, loops: int, out: xr.DataArray, seed: int) -> None:
        rng = np.random.default_rng(seed)
        x_min, y_min = self.X_BINS_TUPLE[0], self.Y_BINS_TUPLE[0]
        x_max, y_max = self.X_BINS_TUPLE[1] - 1, self.Y_BINS_TUPLE[1] - 1
        out_vals = out.values
        for _ in range(loops):
            n = self.NUMPY_WIDTH
            dirs = rng.integers(0, 4, size=(n, self.T))  # 0=right,1=left,2=up,3=down

            pos_x = np.zeros(n, dtype=np.int64)
            pos_y = np.zeros(n, dtype=np.int64)
            ci_set = set(c - 1 for c in self.CHECKPOINTS)
            cp_x = np.zeros((n, self.CHECKPOINTS_LEN), dtype=np.int64)
            cp_y = np.zeros((n, self.CHECKPOINTS_LEN), dtype=np.int64)
            cp_idx = 0

            for k in range(self.T):
                d = dirs[:, k]
                parity = pos_x & 1  # 0=even, 1=odd
                # Right/left: pure y movement
                dy = np.where(d == 0, 1, np.where(d == 1, -1, 0))
                # Up: x-=1, odd rows also shift y+=1
                dx_up = np.where(d == 2, -1, 0)
                dy_up = np.where((d == 2) & (parity == 1), 1, 0)
                # Down: x+=1, odd rows also shift y+=1
                dx_dn = np.where(d == 3, 1, 0)
                dy_dn = np.where((d == 3) & (parity == 1), 1, 0)
                pos_x += dx_up + dx_dn
                pos_y += dy + dy_up + dy_dn
                if k in ci_set:
                    raw_x, raw_y = pos_x.copy(), pos_y.copy()
                    cp_x[:, cp_idx] = np.where(raw_x >= 0, raw_x // 2, -((-raw_x) // 2))
                    cp_y[:, cp_idx] = np.where(raw_y >= 0, raw_y // 2, -((-raw_y) // 2))
                    cp_idx += 1

            for ti in range(self.TARGETS_LEN):
                mask = (cp_x[:, -1] == self.TARGETS_X[ti]) & (cp_y[:, -1] == self.TARGETS_Y[ti])
                if not mask.any():
                    continue
                for c in range(self.CHECKPOINTS_LEN):
                    bx, by = cp_x[mask, c], cp_y[mask, c]
                    v = (bx >= x_min) & (bx <= x_max) & (by >= y_min) & (by <= y_max)
                    np.add.at(out_vals[ti, c], (bx[v] - x_min, by[v] - y_min), 1)


EXPERIMENT_CLASS = BrickLatticeExperiment
