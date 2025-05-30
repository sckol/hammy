import json
import xarray as xr
import numpy as np

# from hammy_lib.simulator import Simulator
# from hammy_lib.util import SimulatorConstants, CCode, Experiment, generate_random_seed
# from hammy_lib.yandex_cloud_storage import YandexCloudStorage
# from hammy_lib.calculator import Calculator, ArgMaxCalculator
from pathlib import Path
from hammy_lib.machine_configuration import MachineConfiguration
from hammy_lib.experiment import Experiment
from hammy_lib.ccode import CCode
from hammy_lib.experiment_configuration import ExperimentConfiguration
from hammy_lib.sequential_calibration import SequentialCalibration
from hammy_lib.parallel_calibration import ParallelCalibration
from hammy_lib.simulation import Simulation
from hammy_lib.calculations.argmax import ArgMaxCalculation


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
            np.zeros(tuple(len(coords[i]) for i in dims)), coords=coords, dims=dims
        )

    def simulate_using_python(self, loops: int, out: xr.DataArray, seed: int) -> None:
        for _ in range(loops):
            data = None
            rng = np.random.default_rng(seed)
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
    conf = MachineConfiguration("c31778_machine_configuration")
    conf.dump()
    experiment_configuration = ExperimentConfiguration(
        experiment, conf, seed=1748065639484
    )
    experiment_configuration.dump()
    sequential_calibration = SequentialCalibration(
        experiment_configuration, dry_run=True
    )
    sequential_calibration.dump()
    parallel_calibration = ParallelCalibration(sequential_calibration)
    parallel_calibration.dump()
    simulation = Simulation(parallel_calibration, simulation_level=4)
    simulation.dump()
    argmax = ArgMaxCalculation(simulation, ["x"])
    argmax.dump()
    print(simulation.results)
    from hammy_lib.vizualization import visualize
    import matplotlib.pyplot as plt

    visualize(
        simulation.results,
        x="checkpoint",
        y="target",
        axis="x",
        filter={"checkpoint": [300, 500, 700], "level": 4, "x": list(range(-10, 10))},
    )
    plt.show()
    # Get access_key and secret_key from .s3_credentials.json file
    # with open(".s3_credentials.json") as f:
    #   credentials = json.load(f)
    # storage = YandexCloudStorage(credentials['access_key'], credentials['secret_key'], credentials['bucket_name'])
    #
    # simulator = Simulator(EXPERIMENT, N, simulate, C_CODE, use_cuda=False, seed=generate_random_seed(), digest="103fbf")
    # storage.download_simulator_results(simulator)
    # simulator.compile()
    # calibration_results = simulator.run_parallel_calibration()
    # simulator.dump_calibration_results(calibration_results)
    # simulation_results = simulator.run_level_simulation(2)
    # simulator.dump_simulation_results(simulation_results)
    # simulation_results = simulator.load_simulation_results()
    # extended_results = Calculator.extend_simulation_results(simulation_results)
    # argmax_calculator = ArgMaxCalculator(extended_results, ['target', 'checkpoint'])
    # print(argmax_calculator.calculate())

#  simulator.dump_simulation_results(simulation_results)
# Calculate extended dataset (all platforms and cumulative)
# Write dask computation
# Calculate mean position for each target and time
# Calculate chi suqared test for each target and time
# Run random quality test and fail it via bug in CFFI rng argument
# Define csc calculation
# Calculate csc position
# Visualize
