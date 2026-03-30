# ! uv pip install git+https://github.com/sckol/hammy#egg=hammy_lib
# Prevent OpenBLAS deadlock after multiprocessing.Pool fork.
# Must be set BEFORE numpy/scipy are imported.
import os as _os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    _os.environ.setdefault(_v, "1")
del _v

import os
import sys
import argparse
import glob
import ctypes.util
import xarray as xr
import numpy as np
from pathlib import Path
from hammy_lib.hammy_object import HammyObject
from hammy_lib.machine_configuration import MachineConfiguration
from hammy_lib.experiment import Experiment
from hammy_lib.ccode import CCode
from hammy_lib.experiment_configuration import ExperimentConfiguration
from hammy_lib.sequential_calibration import SequentialCalibration
from hammy_lib.parallel_calibration import ParallelCalibration
from hammy_lib.simulation import Simulation
from hammy_lib.graph import LatticeGraph2D
from hammy_lib.calculations.position import CellPositionCalculation
from hammy_lib.calculations.bootstrap_position import BootstrapCellPositionCalculation
from hammy_lib.vizualization import Vizualization, line_renderer, line_with_errors_renderer


def _fix_openblas_fork():
    """Prevent OpenBLAS deadlock after multiprocessing.Pool fork.

    OpenBLAS uses pthreads internally. After Pool.map() returns, the parent
    process's BLAS thread pool state may be corrupted by the fork, causing
    np.linalg.eig() and scipy.linalg.inv() to hang.  Setting threads=1
    disables the internal thread pool, avoiding the deadlock.
    """
    import os
    for var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(var, "1")


def _try_enable_mkl():
    """Find libmkl_rt and restart the process with LD_PRELOAD if not already active."""
    if "libmkl_rt" in os.environ.get("LD_PRELOAD", ""):
        return
    candidates = set()
    for prefix in {sys.prefix, os.path.join(os.path.dirname(sys.executable), "..")}:
        candidates.update(glob.glob(os.path.join(prefix, "lib", "libmkl_rt*")))
    for sp in sys.path:
        if "site-packages" in sp or "dist-packages" in sp:
            candidates.update(glob.glob(os.path.join(os.path.dirname(sp), "libmkl_rt*")))
            candidates.update(glob.glob(os.path.join(sp, "*", "libmkl_rt*")))
    found = ctypes.util.find_library("mkl_rt")
    if found and os.path.isabs(found):
        candidates.add(found)
    for path in candidates:
        if os.path.isfile(path):
            print(f"[MKL] Using {os.path.basename(path)} from {os.path.dirname(path)}", flush=True)
            os.environ["LD_PRELOAD"] = path
            os.execv(sys.executable, [sys.executable, os.path.abspath(__file__)] + sys.argv[1:])

if __name__ == "__main__" and '__file__' in globals():
    _fix_openblas_fork()
    _try_enable_mkl()

try:
    HammyObject.RESULTS_DIR = Path(__file__).parent.parent.parent / "results"
except NameError:
    HammyObject.RESULTS_DIR = Path("results")


# Experiment definition
class LatticeExperiment(Experiment):
    experiment_number = 2
    experiment_name = "lattice"
    experiment_version = 1
    T = 1000  # Number of steps in a single random walk
    CHECKPOINTS = [100, 200, 300, 400, 500, 600, 700, 800, 900, T]
    CHECKPOINTS_LEN = len(CHECKPOINTS)
    # 2D targets as (x, y) pairs
    TARGETS_X = [0, 1, 0, 1, 3]
    TARGETS_Y = [0, 0, 1, 1, 2]
    TARGETS_LEN = len(TARGETS_X)
    # Per-dimension bin ranges (small grid for fast eigendecomp + NNLS)
    X_BINS_TUPLE = (-7, 8)  # 15 bins per dimension → 225 nodes
    Y_BINS_TUPLE = (-7, 8)
    X_BINS_LEN = X_BINS_TUPLE[1] - X_BINS_TUPLE[0]
    Y_BINS_LEN = Y_BINS_TUPLE[1] - Y_BINS_TUPLE[0]
    BINS_FLAT_LEN = X_BINS_LEN * Y_BINS_LEN
    NUMPY_WIDTH = 100
    C_DEFINITIONS = f"""
  #define T {T}
  int TARGETS_X[] = {{{",".join(str(x) for x in TARGETS_X)}}};
  int TARGETS_Y[] = {{{",".join(str(y) for y in TARGETS_Y)}}};
  #define TARGETS_LEN { TARGETS_LEN }
  int CHECKPOINTS[] = {{{",".join(str(x) for x in CHECKPOINTS)}}};
  #define CHECKPOINTS_LEN { CHECKPOINTS_LEN }
  #define X_BINS_MIN { X_BINS_TUPLE[0] }
  #define X_BINS_MAX { X_BINS_TUPLE[1] - 1 }
  #define X_BINS_LEN { X_BINS_LEN }
  #define Y_BINS_MIN { Y_BINS_TUPLE[0] }
  #define Y_BINS_MAX { Y_BINS_TUPLE[1] - 1 }
  #define Y_BINS_LEN { Y_BINS_LEN }
  #define BINS_FLAT_LEN { BINS_FLAT_LEN }
  """

    if 'CCODE' in globals():
        c_code = CCode(CCODE, C_DEFINITIONS)  # noqa: F821  # pyright: ignore[reportUndefinedVariable]
    else:
        try:
            _lattice_c_dir = Path(__file__).parent
        except NameError:
            _lattice_c_dir = Path("experiments/02_lattice")
        c_code = CCode((_lattice_c_dir / "lattice.c").read_text(), C_DEFINITIONS)

    def create_empty_results(self) -> xr.DataArray:
        dims = ["target", "checkpoint", "x", "y"]
        coords = {
            "target": list(range(self.TARGETS_LEN)),
            "checkpoint": self.CHECKPOINTS,
            "x": np.arange(*self.X_BINS_TUPLE),
            "y": np.arange(*self.Y_BINS_TUPLE),
        }
        return xr.DataArray(
            np.zeros(tuple(len(coords[i]) for i in dims), dtype=np.int64),
            coords=coords,
            dims=dims,
        )

    def simulate_using_python(self, loops: int, out: xr.DataArray, seed: int) -> None:
        rng = np.random.default_rng(seed)
        x_min, y_min = self.X_BINS_TUPLE[0], self.Y_BINS_TUPLE[0]
        x_max, y_max = self.X_BINS_TUPLE[1] - 1, self.Y_BINS_TUPLE[1] - 1
        out_vals = out.values  # direct numpy array for speed
        for _ in range(loops):
            n_walkers = self.NUMPY_WIDTH
            dim_choices = rng.integers(0, 2, size=(n_walkers, self.T))
            dir_choices = rng.integers(0, 2, size=(n_walkers, self.T)) * 2 - 1

            dx = np.where(dim_choices == 0, dir_choices, 0)
            dy = np.where(dim_choices == 1, dir_choices, 0)

            cum_x = np.cumsum(dx, axis=1)
            cum_y = np.cumsum(dy, axis=1)

            checkpoint_indices = [c - 1 for c in self.CHECKPOINTS]
            # C-style integer division (truncate towards zero), not Python floor division
            raw_x = cum_x[:, checkpoint_indices]
            raw_y = cum_y[:, checkpoint_indices]
            cp_x = np.where(raw_x >= 0, raw_x // 2, -((-raw_x) // 2))
            cp_y = np.where(raw_y >= 0, raw_y // 2, -((-raw_y) // 2))

            final_x = cp_x[:, -1]
            final_y = cp_y[:, -1]

            for target_idx in range(self.TARGETS_LEN):
                tx, ty = self.TARGETS_X[target_idx], self.TARGETS_Y[target_idx]
                mask = (final_x == tx) & (final_y == ty)
                if not np.any(mask):
                    continue
                sel_x = cp_x[mask]  # (n_hits, n_checkpoints)
                sel_y = cp_y[mask]
                for c in range(self.CHECKPOINTS_LEN):
                    bx = sel_x[:, c]
                    by = sel_y[:, c]
                    valid = (bx >= x_min) & (bx <= x_max) & (by >= y_min) & (by <= y_max)
                    bx_v = bx[valid] - x_min
                    by_v = by[valid] - y_min
                    np.add.at(out_vals[target_idx, c], (bx_v, by_v), 1)


# Configuration
def run(level=4, dry_run=False, no_calculations=False, no_viz=False, no_upload=False):
    # S3 Storage — set up FIRST so all objects auto-download from cache
    if not no_upload:
        try:
            from google.colab import userdata
            access_key = userdata.get('access_key')
            secret_key = userdata.get('secret_key')
        except ImportError:
            access_key = os.environ.get('S3_ACCESS_KEY')
            secret_key = os.environ.get('S3_SECRET_KEY')

        if access_key and secret_key:
            from hammy_lib.yandex_cloud_storage import YandexCloudStorage
            storage = YandexCloudStorage(access_key, secret_key, "hammy")
            HammyObject.STORAGE = storage
        else:
            no_upload = True

    experiment = LatticeExperiment()
    experiment.dump()
    experiment.compile()

    conf = MachineConfiguration()
    conf.dump()

    experiment_configuration = ExperimentConfiguration(experiment, conf, seed=1748065639484)
    experiment_configuration.dump()

    # Calibration
    sequential_calibration = SequentialCalibration(experiment_configuration, dry_run=dry_run)
    sequential_calibration.dump()

    calibration_tolerance = 100 if experiment_configuration.cores == 1 else 25
    parallel_calibration = ParallelCalibration(
        sequential_calibration, calibration_tolerance=calibration_tolerance
    )
    parallel_calibration.dump()

    # Simulation
    simulation = Simulation(parallel_calibration, simulation_level=level)
    simulation.dump()

    if not no_upload:
        storage.upload()

    # Re-enable multi-core BLAS for calculation phase (Pool is done)
    for var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.pop(var, None)

    # Calculation: Cell Position (2D lattice, GBC / inverse bilinear)
    if not no_calculations:
        graph = LatticeGraph2D(LatticeExperiment.X_BINS_LEN, LatticeExperiment.Y_BINS_LEN)
        graph.dump()

        position = CellPositionCalculation(
            simulation, graph=graph, spatial_dims=("x", "y"),
        )
        position.dump()

    # Viz helpers
    if not no_calculations and not no_viz:
        last_level = int(simulation.results.level.values[-1])

        def row_ref(data):
            ti = int(data.coords["target"].item())
            tx = LatticeExperiment.TARGETS_X[ti]
            checkpoints = data.coords["checkpoint"].values
            return tx * checkpoints / LatticeExperiment.T - LatticeExperiment.X_BINS_TUPLE[0]

        def col_ref(data):
            ti = int(data.coords["target"].item())
            ty = LatticeExperiment.TARGETS_Y[ti]
            checkpoints = data.coords["checkpoint"].values
            return ty * checkpoints / LatticeExperiment.T - LatticeExperiment.Y_BINS_TUPLE[0]

    # Viz: position_row by checkpoint
    if not no_calculations and not no_viz:
        Vizualization(
            results_object=position,  # pyright: ignore[reportPossiblyUnboundVariable]
            x="platform",
            y="target",
            cell_renderer=line_renderer(axis="checkpoint", reference=row_ref),  # pyright: ignore[reportPossiblyUnboundVariable]
            filter={"level": last_level, "position_data": "position_row"},
            title="Position row (x) by checkpoint",
            id="viz_position_row",
        ).dump()

    # Viz: position_col by checkpoint
    if not no_calculations and not no_viz:
        Vizualization(
            results_object=position,  # pyright: ignore[reportPossiblyUnboundVariable]
            x="platform",
            y="target",
            cell_renderer=line_renderer(axis="checkpoint", reference=col_ref),  # pyright: ignore[reportPossiblyUnboundVariable]
            filter={"level": last_level, "position_data": "position_col"},
            title="Position col (y) by checkpoint",
            id="viz_position_col",
        ).dump()

    # Viz: nonzero_count by checkpoint
    if not no_calculations and not no_viz:
        Vizualization(
            results_object=position,  # pyright: ignore[reportPossiblyUnboundVariable]
            x="platform",
            y="target",
            cell_renderer=line_renderer(
                axis="checkpoint",
                reference=lambda data: np.full(len(data.coords["checkpoint"].values), 2.0),
            ),
            filter={"level": last_level, "position_data": "nonzero_count"},
            title="NNLS component count by checkpoint (ref=2)",
            id="viz_nonzero_by_checkpoint",
        ).dump()

    # Viz: cell_dim by checkpoint
    if not no_calculations and not no_viz:
        Vizualization(
            results_object=position,  # pyright: ignore[reportPossiblyUnboundVariable]
            x="platform",
            y="target",
            cell_renderer=line_renderer(
                axis="checkpoint",
                reference=lambda data: np.full(len(data.coords["checkpoint"].values), 2.0),
            ),
            filter={"level": last_level, "position_data": "cell_dim"},
            title="Cell dimension by checkpoint (ref=2=face)",
            id="viz_cell_dim",
        ).dump()

    # Viz: fit_quality by checkpoint
    if not no_calculations and not no_viz:
        Vizualization(
            results_object=position,  # pyright: ignore[reportPossiblyUnboundVariable]
            x="platform",
            y="target",
            cell_renderer=line_renderer(
                axis="checkpoint",
                reference=lambda data: np.ones(len(data.coords["checkpoint"].values)),
            ),
            filter={"level": last_level, "position_data": "fit_quality"},
            title="Cell fit quality by checkpoint (ref=1.0)",
            id="viz_fit_quality_by_checkpoint",
        ).dump()

    # Viz: fit_quality convergence by level
    if not no_calculations and not no_viz:
        Vizualization(
            results_object=position,  # pyright: ignore[reportPossiblyUnboundVariable]
            x="platform",
            y="target",
            cell_renderer=line_renderer(
                axis="level",
                reference=lambda data: np.ones(len(data.coords["level"].values)),
            ),
            filter={"checkpoint": 500, "position_data": "fit_quality"},
            title="Cell fit quality by level (checkpoint=500)",
            id="viz_fit_quality_by_level",
        ).dump()

    # Viz: NNLS residual by level
    if not no_calculations and not no_viz:
        Vizualization(
            results_object=position,  # pyright: ignore[reportPossiblyUnboundVariable]
            x="platform",
            y="target",
            cell_renderer=line_renderer(
                axis="level",
                reference=lambda data: np.zeros(len(data.coords["level"].values)),
            ),
            filter={"checkpoint": 500, "position_data": "residual"},
            title="NNLS residual by level (checkpoint=500)",
            id="viz_residual_by_level",
        ).dump()

    # Viz: power by checkpoint
    if not no_calculations and not no_viz:
        Vizualization(
            results_object=position,  # pyright: ignore[reportPossiblyUnboundVariable]
            x="platform",
            y="target",
            cell_renderer=line_renderer(axis="checkpoint"),
            filter={"level": last_level, "position_data": "power"},
            title="Spectral power p by checkpoint",
            id="viz_power_by_checkpoint",
        ).dump()

    # Calculation: Bootstrap cell position
    if not no_calculations:
        bootstrap = BootstrapCellPositionCalculation(
            simulation, graph=graph,  # pyright: ignore[reportPossiblyUnboundVariable]
            spatial_dims=("x", "y"), n_bootstrap=20,
        )
        bootstrap.dump()

    # Viz: Bootstrap row std by level
    if not no_calculations and not no_viz:
        Vizualization(
            results_object=bootstrap,  # pyright: ignore[reportPossiblyUnboundVariable]
            x="platform",
            y="target",
            cell_renderer=line_renderer(
                axis="level",
                reference=lambda data: np.zeros(len(data.coords["level"].values)),
            ),
            filter={"checkpoint": 500, "bootstrap_data": "bootstrap_row_std"},
            title="Bootstrap row position std by level (checkpoint=500)",
            id="viz_bootstrap_row_std",
        ).dump()

    # Viz: Bootstrap col std by level
    if not no_calculations and not no_viz:
        Vizualization(
            results_object=bootstrap,  # pyright: ignore[reportPossiblyUnboundVariable]
            x="platform",
            y="target",
            cell_renderer=line_renderer(
                axis="level",
                reference=lambda data: np.zeros(len(data.coords["level"].values)),
            ),
            filter={"checkpoint": 500, "bootstrap_data": "bootstrap_col_std"},
            title="Bootstrap col position std by level (checkpoint=500)",
            id="viz_bootstrap_col_std",
        ).dump()

    # Final S3 upload (calculations + vizs)
    if not no_upload:
        storage.upload()

    return simulation


# Run
if __name__ == "__main__" and '__file__' in globals():
    # CLI mode (python lattice.py / python -m experiments.02_lattice)
    parser = argparse.ArgumentParser(description="Lattice experiment (2D)")
    parser.add_argument("--level", type=int, default=4, help="Simulation level (0-N)")
    parser.add_argument("--no-viz", action="store_true", help="Skip visualizations")
    parser.add_argument("--no-upload", action="store_true", help="Skip S3 upload")
    parser.add_argument("--no-calculations", action="store_true", help="Skip calculations and viz")
    parser.add_argument("--dry-run", action="store_true", help="Fast calibration (10loops)")
    args = parser.parse_args()
    run(
        level=args.level,
        dry_run=args.dry_run,
        no_calculations=args.no_calculations,
        no_viz=args.no_viz,
        no_upload=args.no_upload,
    )
elif __name__ == "__main__":
    # Notebook mode (Colab / Jupyter)
    run()
