import json
import xarray as xr
import numpy as np
#from hammy_lib.simulator import Simulator
#from hammy_lib.util import SimulatorConstants, CCode, Experiment, generate_random_seed
#from hammy_lib.yandex_cloud_storage import YandexCloudStorage
#from hammy_lib.calculator import Calculator, ArgMaxCalculator
from pathlib import Path
from hammy_lib.machine_configuration import MachineConfiguration
from hammy_lib.experiment import Experiment
from hammy_lib.ccode import CCode

class WalkExperiment(Experiment):
  number = 1
  name = "walk"
  version = 1
  T = 1000 # Number of steps in a single random walk
  CHECKPOINTS = [100, 200, 300, 400, 500, 600, 700, 800, 900, T]
  CHECKPOINTS_LEN = len(CHECKPOINTS)
  TARGETS = [0, 1, 2, 5, 10]
  TARGETS_LEN = len(TARGETS)
  BINS_TUPLE = (-T // 20, T // 20 + 1)
  BINS_LEN = BINS_TUPLE[1] - BINS_TUPLE[0]
  BINS = np.hstack([np.arange(*BINS_TUPLE), BINS_TUPLE[1] + .9])
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
  c_code = CCode(Path(__file__).parent / "walk.c", C_DEFINITIONS)

  def create_empty_results(self) -> xr.DataArray:
    dims=["target", "checkpoint", "x"]
    coords = {
      "x": np.arange(*BINS_TUPLE),
      "target": TARGETS,
      "checkpoint": CHECKPOINTS
    }
    return xr.DataArray(  
      np.zeros(tuple(len(coords[i]) for i in dims)),
      coords=coords,
      dims=dims
    )

  def simulate_using_python(loops: int, out: xr.DataArray, seed: int) -> None:
    for _ in range(loops):
      data = None      
      rng = np.random.default_rng(seed)
      diffs = np.diff(CHECKPOINTS, prepend=0).tolist()
      steps = rng.binomial(np.tile(diffs, (NUMPY_WIDTH, 1)), .75)
      data = rng.binomial(steps, .5) - steps / 2
      data = np.cumsum(data, axis=-1)
      for target_idx, target in enumerate(TARGETS):
        if target != 0:
          data[data[:, -1] == -target, :] *= -1
        for c in range(CHECKPOINTS_LEN):
          out[target_idx, c, :] += np.histogram(data[data[:, -1] == target, c], bins=BINS)[0]

if __name__ == "__main__":
  experiment = WalkExperiment()
  conf = MachineConfiguration(digest='645b93')
  conf.resolve(no_load=False)
  conf.dump()
  
  # Get access_key and secret_key from .s3_credentials.json file
  # with open(".s3_credentials.json") as f:
  #   credentials = json.load(f)
  # storage = YandexCloudStorage(credentials['access_key'], credentials['secret_key'], credentials['bucket_name'])  
  # 
  # simulator = Simulator(EXPERIMENT, N, simulate, C_CODE, use_cuda=False, seed=generate_random_seed(), digest="103fbf")
  #storage.download_simulator_results(simulator)
  # simulator.compile()  
  #calibration_results = simulator.run_parallel_calibration()
  #simulator.dump_calibration_results(calibration_results)
  #simulation_results = simulator.run_level_simulation(2)
  #simulator.dump_simulation_results(simulation_results)
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
