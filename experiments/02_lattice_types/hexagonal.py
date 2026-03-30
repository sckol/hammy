"""Hexagonal (honeycomb) lattice: 3-direction walk, sublattice-dependent."""
import numpy as np
import xarray as xr
from pathlib import Path
from hammy_lib.experiment import Experiment
from hammy_lib.ccode import CCode
from . import common as C


_BINS = C.get_bins("hexagonal")


class HexagonalLatticeExperiment(Experiment):
    experiment_number = 2
    experiment_name = "hexagonal"
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

    try:
        _c_dir = Path(__file__).parent
    except NameError:
        _c_dir = Path("experiments/02_lattice_types")
    c_code = CCode((_c_dir / "hexagonal.c").read_text(), C.make_c_definitions("hexagonal"))

    def create_empty_results(self) -> xr.DataArray:
        return C.create_empty_results("hexagonal")

    def simulate_using_python(self, loops: int, out: xr.DataArray, seed: int) -> None:
        rng = np.random.default_rng(seed)
        x_min, y_min = self.X_BINS_TUPLE[0], self.Y_BINS_TUPLE[0]
        x_max, y_max = self.X_BINS_TUPLE[1] - 1, self.Y_BINS_TUPLE[1] - 1
        out_vals = out.values
        for _ in range(loops):
            n = self.NUMPY_WIDTH
            # 3 directions per step, sublattice-dependent
            # Rejection: 2 bits, reject >= 3
            dirs = rng.integers(0, 4, size=(n, self.T))
            bad = dirs >= 3
            while bad.any():
                dirs[bad] = rng.integers(0, 4, size=bad.sum())
                bad = dirs >= 3

            # Step-by-step: need to track sublattice
            pos_x = np.zeros(n, dtype=np.int64)
            pos_y = np.zeros(n, dtype=np.int64)
            ci_set = set(c - 1 for c in self.CHECKPOINTS)
            ci_list = [c - 1 for c in self.CHECKPOINTS]
            cp_x = np.zeros((n, self.CHECKPOINTS_LEN), dtype=np.int64)
            cp_y = np.zeros((n, self.CHECKPOINTS_LEN), dtype=np.int64)
            cp_idx = 0

            for k in range(self.T):
                sublattice = (pos_x + pos_y) & 1  # 0=A, 1=B
                d = dirs[:, k]
                # dir 0: (+1, 0), dir 1: (-1, 0), dir 2: (0, +1 if A else -1)
                dx = np.where(d == 0, 1, np.where(d == 1, -1, 0))
                dy = np.where(d == 2, np.where(sublattice == 0, 1, -1), 0)
                pos_x += dx
                pos_y += dy
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


EXPERIMENT_CLASS = HexagonalLatticeExperiment
