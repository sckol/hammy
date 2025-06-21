import xarray as xr
import numpy as np
import json 
from pathlib import Path
from hammy_lib import graph
from hammy_lib.machine_configuration import MachineConfiguration
from hammy_lib.experiment import Experiment
from hammy_lib.ccode import CCode
from hammy_lib.experiment_configuration import ExperimentConfiguration
from hammy_lib.sequential_calibration import SequentialCalibration
from hammy_lib.parallel_calibration import ParallelCalibration
from hammy_lib.simulation import Simulation
from hammy_lib.graph import CircularGraph, LinearGraph
from hammy_lib.calculations.position import PositionCalculation
from hammy_lib.calculations.argmax import ArgMaxCalculation
from hammy_lib.vizualization import Vizualization
from hammy_lib.yandex_cloud_storage import YandexCloudStorage


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
            steps = rng.binomial(np.tile(diffs, (self.NUMPY_WIDTH, 1)), 0.75)
            data = rng.binomial(steps, 0.5) - steps / 2
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
    #from hammy_lib.simulator_platforms import SimulatorPlatforms
    #experiment.run_single_simulation(1000, SimulatorPlatforms.PYTHON, 100, False)
    conf = MachineConfiguration("c31778_machine_configuration")
    conf.dump()
    experiment_configuration = ExperimentConfiguration(
        experiment, conf, seed=1748065639484
    )
    experiment_configuration.dump()
    sequential_calibration = SequentialCalibration(
        experiment_configuration, dry_run=False
    )
    sequential_calibration.dump()
    parallel_calibration = ParallelCalibration(sequential_calibration)
    parallel_calibration.dump()
    simulation = Simulation(parallel_calibration, simulation_level=4)
    simulation.dump()    
    simulation._results = simulation.results.rename({"x": "position_index"})
    simulation._results = simulation.results.assign_coords(
        position_index=np.arange(WalkExperiment.BINS_LEN)
    )    
    circular_graph = CircularGraph(WalkExperiment.BINS_LEN)
    position = PositionCalculation(simulation, graph=circular_graph, dimensionality=10, max_power=WalkExperiment.T)
    position.dump()

    # Add to position.results a new coordinate to dimension "position_data" with values position
    # It must be nan if [position_data=nonzero_count] is greater than 2, otherwise it must be weighted average of
    # [position_data=index0, position_data=index1] with weights [position_data=value0, position_data=value1]
    # Also if [position_data=nonzero_count] is exactly 2,  get nan if abs([position_data=index0] - [position_data=index1]) != 1
    # def weighted_average_two_positions(row):
    #   nonzero = np.flatnonzero(row.values)
    #   if len(nonzero) != 2:
    #     return np.nan
    #   idx0, idx1 = nonzero
    #   val0, val1 = row.values[idx0], row.values[idx1]
    #   if abs(idx0 - idx1) != 1:
    #     return np.nan
    #   return (idx0 * val0 + idx1 * val1) / (val0 + val1)
    
    # weighted_avg = position.results.reduce(
    #   weighted_average_two_positions, dim="position_index"
    # )

    # position.results = position.results.assign_coords(
    #   position_data=("position_data", ["weighted_average"])
    # )
    # position.results = position.results.expand_dims("position_data")
    # position.results.loc[{"position_data": "weighted_average"}] = weighted_avg
    # argmax = ArgMaxCalculation(simulation, ["x"])
    # argmax.dump()    
    viz = Vizualization(
        position,
        x="platform",
        y="target",        
        axis="checkpoint",       
        filter={"position_data": 'power', "level": 4},
        comparison={"position_data": "nonzero_count"},
        allow_aggregation=False,        
    )
    viz.dump() 
    # with open(".s3_credentials.json") as f:
    #   credentials = json.load(f)
    # storage = YandexCloudStorage(credentials['access_key'], credentials['secret_key'], credentials['bucket_name'])
    # storage.upload()


# Allow to switch down aggregation in vizualization (raise an error)
# Add/test reference lines
# Calculate ton-5 most frequent positions, make csc calculation for them
# Make "trim" calculation where we detect the number of positions > 1e-9 for a whole dataset (we must get two)
# Calculate "theorethical" positions and fit the number of steps based on theoretical positions
# Estimate the number of probes and if it correspond to the real one (i.e. they are independent)

# Negative probability
# Consider we have a function with negative values and we need to "normalize" it. Then we represent it as a difference 
# of two functions, one for particles, other for antiparticles. And instead of one series event we generate two. Then
# we "annihilate" particles and antiparticles with some probability.
