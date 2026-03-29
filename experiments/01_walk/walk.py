# ! uv pip install git+https://github.com/sckol/hammy#egg=hammy_lib
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
from hammy_lib.calculations.popsize import PopulationSizeCalculation, bridged_random_walk_distribution
from hammy_lib.graph import LinearGraph
from hammy_lib.calculations.position import PositionCalculation
from hammy_lib.calculations.bootstrap_position import BootstrapPositionCalculation
from hammy_lib.vizualization import Vizualization, line_renderer, line_with_errors_renderer, bar_chart_renderer


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
    _try_enable_mkl()

try:
    HammyObject.RESULTS_DIR = Path(__file__).parent.parent.parent / "results"
except NameError:
    HammyObject.RESULTS_DIR = Path("results")

# Experiment definition
class WalkExperiment(Experiment):
    experiment_number = 1
    experiment_name = "walk"
    experiment_version = 1
    T = 1000  # Number of steps in a single random walk
    CHECKPOINTS = [100, 200, 300, 400, 500, 600, 700, 800, 900, T]
    CHECKPOINTS_LEN = len(CHECKPOINTS)
    TARGETS = [0, 1, 2, 5, 10]
    TARGETS_LEN = len(TARGETS)
    BINS_TUPLE = (-T // 20, T // 20 + 1)
    BINS_LEN = BINS_TUPLE[1] - BINS_TUPLE[0]
    BINS = np.hstack([np.arange(*BINS_TUPLE), BINS_TUPLE[1] + 0.9])
    NUMPY_WIDTH = 100
    C_DEFINITIONS = f"""
  #define T {T}
  int TARGETS[] = {{{",".join([str(x) for x in TARGETS])}}};
  #define TARGETS_LEN { TARGETS_LEN }
  int CHECKPOINTS[] = {{{",".join([str(x) for x in CHECKPOINTS])}}};
  #define CHECKPOINTS_LEN { CHECKPOINTS_LEN }
  #define BINS_MIN { BINS_TUPLE[0] }
  #define BINS_MAX { BINS_TUPLE[1] - 1 }
  #define BINS_LEN { BINS_LEN }
  """

    if 'CCODE' in globals():
        c_code = CCode(CCODE, C_DEFINITIONS)  # noqa: F821  # pyright: ignore[reportUndefinedVariable]
    else:
        try:
            _walk_c_dir = Path(__file__).parent
        except NameError:
            _walk_c_dir = Path("experiments/01_walk")
        c_code = CCode((_walk_c_dir / "walk.c").read_text(), C_DEFINITIONS)

    def create_empty_results(self) -> xr.DataArray:
        dims = ["target", "checkpoint", "x"]
        coords = {
            "x": np.arange(*self.BINS_TUPLE),
            "target": self.TARGETS,
            "checkpoint": self.CHECKPOINTS,
        }
        return xr.DataArray(
            np.zeros(tuple(len(coords[i]) for i in dims), dtype=np.int64),
            coords=coords,
            dims=dims,
        )

    def simulate_using_python(self, loops: int, out: xr.DataArray, seed: int) -> None:
        rng = np.random.default_rng(seed)
        for _ in range(loops):
            data = None
            diffs = np.diff(self.CHECKPOINTS, prepend=0).tolist()
            steps = np.tile(diffs, (self.NUMPY_WIDTH, 1))
            data = rng.binomial(steps, 0.5) - (steps / 2)
            data = np.cumsum(data, axis=-1)
            for target_idx, target in enumerate(self.TARGETS):
                if target != 0:
                    data[data[:, -1] == -target, :] *= -1
                for c in range(self.CHECKPOINTS_LEN):
                    out[target_idx, c, :] += np.histogram(
                        data[data[:, -1] == target, c], bins=self.BINS
                    )[0]

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

    experiment = WalkExperiment()
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

    # Calculation: PopulationSize
    if not no_calculations:
        popsize = PopulationSizeCalculation(simulation, bridged_random_walk_distribution)
        popsize.dump()

    # Calculation: Position
    if not no_calculations:
        graph = LinearGraph(WalkExperiment.BINS_LEN)
        graph.dump()

        position = PositionCalculation(simulation, graph=graph, dimensionality=2, spatial_dims=("x",))
        position.dump()

    # Viz: Simulation histogram
    if not no_calculations and not no_viz:
        last_level = int(simulation.results.level.values[-1])

        Vizualization(
            results_object=simulation,
            x="checkpoint",
            y="target",
            cell_renderer=bar_chart_renderer(axis="x"),
            filter={"level": last_level, "platform": "CFFI",
                    "checkpoint": WalkExperiment.CHECKPOINTS[:-1]},
            title="Count by x",
        ).dump()

    # Viz: Position (line)
    if not no_calculations and not no_viz:
        position_ref = lambda data: (  # noqa: E731
            data.coords["target"].item() * data.coords["checkpoint"].values / WalkExperiment.T
            - WalkExperiment.BINS_TUPLE[0]
        )

        Vizualization(
            results_object=position,  # pyright: ignore[reportPossiblyUnboundVariable]
            x="platform",
            y="target",
            cell_renderer=line_renderer(
                axis="checkpoint",
                reference=position_ref,
            ),
            filter={"level": last_level, "position_data": "index0"},  # pyright: ignore[reportPossiblyUnboundVariable]
            title="Position (bin index) by checkpoint",
        ).dump()

    # Viz: Position with error band
    if not no_calculations and not no_viz:
        Vizualization(
            results_object=position,  # pyright: ignore[reportPossiblyUnboundVariable]
            x="platform",
            y="target",
            cell_renderer=line_with_errors_renderer(
                axis="checkpoint",
                error=2.0,
                reference=position_ref,  # pyright: ignore[reportPossiblyUnboundVariable]
            ),
            filter={"level": last_level, "position_data": "index0"},  # pyright: ignore[reportPossiblyUnboundVariable]
            title="Position (bin index) with error band",
            sharey=False,
        ).dump()

    # Viz: Population size
    if not no_calculations and not no_viz:
        Vizualization(
            results_object=popsize,  # pyright: ignore[reportPossiblyUnboundVariable]
            x="platform",
            y="target",
            cell_renderer=line_renderer(
                axis="level",
                reference=lambda data: np.ones(len(data.coords["level"].values)),
            ),
            filter={"checkpoint": 500},
            title="1/φ by level",
        ).dump()

    # Viz: NNLS component count
    if not no_calculations and not no_viz:
        Vizualization(
            results_object=position,  # pyright: ignore[reportPossiblyUnboundVariable]
            x="platform",
            y="target",
            cell_renderer=line_renderer(
                axis="level",
                reference=lambda data: np.full(len(data.coords["level"].values), 2.0),
            ),
            filter={"checkpoint": 500, "position_data": "nonzero_count"},
            title="NNLS component count by level",
        ).dump()

    # Calculation: Bootstrap position
    if not no_calculations:
        bootstrap = BootstrapPositionCalculation(
            simulation, graph=graph, dimensionality=2,  # pyright: ignore[reportPossiblyUnboundVariable]
            spatial_dims=("x",), n_bootstrap=200,
        )
        bootstrap.dump()

    # Continuous position reference (node-index space: target * checkpoint/T + 50)
    if not no_calculations and not no_viz:
        continuous_ref = lambda data: (  # noqa: E731
            data.coords["target"].item() * data.coords["checkpoint"].values / WalkExperiment.T
            - WalkExperiment.BINS_TUPLE[0]
        )

        def _make_bootstrap_error(bootstrap_calc):
            std_data = bootstrap_calc.results.sel(bootstrap_data="bootstrap_std")
            def error_func(data):
                return std_data.sel(
                    target=data.coords["target"].item(),
                    platform=data.coords["platform"].item(),
                    level=data.coords["level"].item(),
                ).values
            return error_func

        # Viz: Continuous position with bootstrap CI
        Vizualization(
            results_object=position,  # pyright: ignore[reportPossiblyUnboundVariable]
            x="platform",
            y="target",
            cell_renderer=line_with_errors_renderer(
                axis="checkpoint",
                error=_make_bootstrap_error(bootstrap),  # pyright: ignore[reportPossiblyUnboundVariable]
                reference=continuous_ref,
            ),
            filter={"level": last_level, "position_data": "continuous_position"},  # pyright: ignore[reportPossiblyUnboundVariable]
            title="Continuous position with bootstrap CI",
            sharey=False,
        ).dump()

        # Viz: Continuous position convergence by level
        Vizualization(
            results_object=position,  # pyright: ignore[reportPossiblyUnboundVariable]
            x="platform",
            y="target",
            cell_renderer=line_renderer(
                axis="level",
                reference=lambda data: np.full(
                    len(data.coords["level"].values),
                    data.coords["target"].item() * 500 / WalkExperiment.T
                    - WalkExperiment.BINS_TUPLE[0],
                ),
            ),
            filter={"checkpoint": 500, "position_data": "continuous_position"},
            title="Continuous position by level (checkpoint=500)",
        ).dump()

        # Viz: NNLS residual by level
        Vizualization(
            results_object=position,  # pyright: ignore[reportPossiblyUnboundVariable]
            x="platform",
            y="target",
            cell_renderer=line_renderer(
                axis="level",
                reference=lambda data: np.zeros(len(data.coords["level"].values)),
            ),
            filter={"checkpoint": 500, "position_data": "residual"},
            title="NNLS residual by level",
        ).dump()

        # Viz: Simplex fit quality by level
        Vizualization(
            results_object=position,  # pyright: ignore[reportPossiblyUnboundVariable]
            x="platform",
            y="target",
            cell_renderer=line_renderer(
                axis="level",
                reference=lambda data: np.ones(len(data.coords["level"].values)),
            ),
            filter={"checkpoint": 500, "position_data": "fit_quality"},
            title="Simplex fit quality by level",
        ).dump()

        # Viz: Bootstrap position std by level
        Vizualization(
            results_object=bootstrap,  # pyright: ignore[reportPossiblyUnboundVariable]
            x="platform",
            y="target",
            cell_renderer=line_renderer(
                axis="level",
                reference=lambda data: np.zeros(len(data.coords["level"].values)),
            ),
            filter={"checkpoint": 500, "bootstrap_data": "bootstrap_std"},
            title="Bootstrap position std by level",
        ).dump()

    # Final S3 upload (calculations + vizs)
    if not no_upload:
        storage.upload()

    return simulation


# Run
if __name__ == "__main__" and '__file__' in globals():
    # CLI mode (python walk.py / python -m experiments.01_walk)
    parser = argparse.ArgumentParser(description="Walk experiment")
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
