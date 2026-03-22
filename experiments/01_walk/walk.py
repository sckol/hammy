def _try_enable_mkl():
    """Find libmkl_rt and restart the process with LD_PRELOAD if not already active."""
    import os, sys, glob, ctypes.util
    if "libmkl_rt" in os.environ.get("LD_PRELOAD", ""):
        return
    candidates = set()
    # {venv or conda prefix}/lib/
    for prefix in {sys.prefix, os.path.join(os.path.dirname(sys.executable), "..")}:
        candidates.update(glob.glob(os.path.join(prefix, "lib", "libmkl_rt*")))
    # one level above any site-packages / dist-packages on sys.path
    for sp in sys.path:
        if "site-packages" in sp or "dist-packages" in sp:
            candidates.update(glob.glob(os.path.join(os.path.dirname(sp), "libmkl_rt*")))
            candidates.update(glob.glob(os.path.join(sp, "*", "libmkl_rt*")))
    # system search via ld.so / LD_LIBRARY_PATH
    found = ctypes.util.find_library("mkl_rt")
    if found and os.path.isabs(found):
        candidates.add(found)
    for path in candidates:
        if os.path.isfile(path):
            print(f"[MKL] Using {os.path.basename(path)} from {os.path.dirname(path)}", flush=True)
            os.environ["LD_PRELOAD"] = path
            os.execv(sys.executable, [sys.executable, os.path.abspath(__file__)] + sys.argv[1:])

_try_enable_mkl()

import xarray as xr
import numpy as np
import json
import matplotlib.pyplot as plt
from pathlib import Path
from hammy_lib.hammy_object import HammyObject
from hammy_lib.machine_configuration import MachineConfiguration

# Use the shared results dir at hammy root (contains cached simulation data).
# Remove this line to use the local results/ dir (experiments/01_walk/results/).
HammyObject.RESULTS_DIR = Path(__file__).parent.parent.parent / "results"
from hammy_lib.experiment import Experiment
from hammy_lib.ccode import CCode
from hammy_lib.experiment_configuration import ExperimentConfiguration
from hammy_lib.sequential_calibration import SequentialCalibration
from hammy_lib.parallel_calibration import ParallelCalibration
from hammy_lib.simulation import Simulation
from hammy_lib.simulator_platforms import SimulatorPlatforms
from hammy_lib.calculations.popsize import PopulationSizeCalculation, bridged_random_walk_distribution
from hammy_lib.graph import LinearGraph
from hammy_lib.calculations.position import PositionCalculation
from hammy_lib.vizualization import Vizualization


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

    c_code = CCode((Path(__file__).parent / "walk.c").read_text(), C_DEFINITIONS)

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
            #steps = rng.binomial(np.tile(diffs, (self.NUMPY_WIDTH, 1)), 0.75)
            #data = rng.binomial(steps, 0.5) - steps / 2
            steps = np.tile(diffs, (self.NUMPY_WIDTH, 1))          # deterministic number of micro-steps
            data = rng.binomial(steps, 0.5) - (steps / 2)          # (R - L)/2, matches your C scaling
            data = np.cumsum(data, axis=-1)
            for target_idx, target in enumerate(self.TARGETS):
                if target != 0:
                    data[data[:, -1] == -target, :] *= -1
                for c in range(self.CHECKPOINTS_LEN):
                    out[target_idx, c, :] += np.histogram(
                        data[data[:, -1] == target, c], bins=self.BINS
                    )[0]


if __name__ == "__main__":
    experiment = WalkExperiment()
    experiment.dump()
    experiment.compile()
    conf = MachineConfiguration()
    conf.dump()
    experiment_configuration = ExperimentConfiguration(experiment, conf, seed=1748065639484)
    experiment_configuration.dump()
    sequential_calibration = SequentialCalibration(experiment_configuration, dry_run=False)
    sequential_calibration.dump()
    parallel_calibration = ParallelCalibration(sequential_calibration)
    parallel_calibration.dump()
    simulation = Simulation(parallel_calibration, simulation_level=4)
    simulation.dump()
    popsize = PopulationSizeCalculation(simulation, bridged_random_walk_distribution)
    popsize.dump()
    graph = LinearGraph(WalkExperiment.BINS_LEN)
    graph.dump()
    position = PositionCalculation(simulation, graph=graph, dimensionality=2, spatial_dims=("x",))
    position.dump()

    last_level = int(simulation.results.level.values[-1])

    Vizualization(
        results_object=simulation,
        x="checkpoint",
        y="target",
        axis="x",
        filter={"level": last_level, "platform": "CFFI"},
        y_axis_label="Count",
    ).dump()

    Vizualization(
        results_object=position,
        x="platform",
        y="target",
        axis="checkpoint",
        filter={"level": last_level, "position_data": "index0"},
        reference=lambda data: (
            data.coords["target"].item() * data.coords["checkpoint"].values / WalkExperiment.T
            - WalkExperiment.BINS_TUPLE[0]
        ),
        y_axis_label="Position (bin index)",
    ).dump()

    Vizualization(
        results_object=popsize,
        x="platform",
        y="target",
        axis="level",
        filter={"checkpoint": 500, },
        reference=lambda data: np.ones(len(data.coords["level"].values)),
        y_axis_label="1/φ",
    ).dump()

    Vizualization(
        results_object=position,
        x="platform",
        y="target",
        axis="level",
        filter={"checkpoint": 500, "position_data": "nonzero_count"},
        reference=lambda data: np.full(len(data.coords["level"].values), 2.0),
        y_axis_label="NNLS component count",
    ).dump()
