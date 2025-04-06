import xarray as xr
import numpy as np
from hammy import Simulator, SimulatorPlatforms, CCode, Experiment

EXPERIMENT = Experiment(1, "walk")

class N(SimulatorConstants):
  TAG = "walk";
  EXPERIMENT_NUMBER = 1;
  IMPLEMENTATION_NUMBER = 1;
  T = 1000 # Number of steps in a single random walk
  CHECKPOINTS = [100, 200, 300, 400, 500, 600, 700, 800, 900, T]
  CHECKPOINTS_LEN = len(CHECKPOINTS)
  TARGETS = [0, 1, 2, 5, 10]
  TARGETS_LEN = len(TARGETS)
  BINS_TUPLE = (-T // 20, T // 20 + 1)
  BINS_LEN = BINS_TUPLE[1] - BINS_TUPLE[0]
  BINS = np.hstack([np.arange(*BINS_TUPLE), BINS_TUPLE[1] + .9])
  NUMPY_WIDTH = 100
  @staticmethod
  def create_empty_results() -> xr.DataArray:
    dims=["targets", "checkpoints", "x"]
    coords = {
      "x": np.arange(*N.BINS_TUPLE),
      "targets": N.TARGETS,
      "checkpoints": N.CHECKPOINTS
    }
    return xr.DataArray(  
      np.zeros(tuple(len(coords[i]) for i in dims)),
      coords=coords,
      dims=dims
    )
    

def simulate(loops: int, out: xr.DataArray, seed: int) -> None:
  for _ in range(loops):
      data = None
      modes = None
      rng = np.random.default_rng(seed)
      diffs = np.diff(N.CHECKPOINTS, prepend=0).tolist()
      steps = rng.binomial(np.tile(diffs, (N.NUMPY_WIDTH, 1)), .75)
      data = rng.binomial(steps, .5) - steps / 2
      data = np.cumsum(data, axis=-1)
      for target_idx, target in enumerate(N.TARGETS):
        if target != 0:
            data[data[:, -1] == -target, :] *= -1
        for c in range(N.CHECKPOINTS_LEN):
            out[target_idx, c, :] += np.histogram(data[data[:, -1] == target, c], bins=N.BINS)[0]

if __name__ == "__main__":
  C_DEFINITIONS = f"""
#define T {N.T}
int TARGETS[] = {{{",".join([str(x) for x in N.TARGETS])}}};
#define TARGETS_LEN { N.TARGETS_LEN }
int CHECKPOINTS[] = {{{",".join([str(x) for x in N.CHECKPOINTS])}}};
#define CHECKPOINTS_LEN { N.CHECKPOINTS_LEN }
#define BINS_MIN { N.BINS_TUPLE[0] }
#define BINS_MAX { N.BINS_TUPLE[1] - 1 }
#define BINS_LEN { N.BINS_LEN }
"""
  C_CODE = CCode(EXPERIMENT.get_path() / "walk.c", C_DEFINITIONS)
  simulator = Simulator(N, simulate, C_CODE, threads=1, seed=0)
  #print(simulator.run_calibration(SimulatorPlatforms.CFFI))
  simulator.run_parallel_simulations(1000, 1)