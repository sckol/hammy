import xarray as xr
from .hammy_object import ArrayHammyObject
from .calibration_results import CalibrationResults
from .parallel_calibration import ParallelCalibration


class Simulation(ArrayHammyObject):
    """Runs the experiment at exponentially increasing time levels and accumulates results.

    Each simulation level runs for 2^(level-1) minutes (level 0 = 1 min,
    level 1 = 1 min, level 2 = 2 min, level 3 = 4 min, level 4 = 8 min).
    The number of loops per platform is derived from calibration:
    loops = calibrated_loops_per_minute * minutes.

    Simulation(level=N) recursively resolves levels 0..N-1 first, then runs
    level N and concatenates all results along the "level" dimension. Each
    level's result has a "platform" dimension (PYTHON, CFFI, and optionally
    CUDA), so the final xarray has shape [..., level, platform].

    The dependency chain:
        Simulation → ParallelCalibration → SequentialCalibration
            → ExperimentConfiguration → (Experiment, MachineConfiguration)

    ExperimentConfiguration.run_parallel_simulations() handles the actual
    parallel execution: CUDA launches async in the main process, then a
    multiprocessing Pool runs PYTHON (1 thread) + CFFI (remaining threads)
    simultaneously. Results from all platforms are summed per-platform and
    concatenated into a single xr.DataArray.
    """

    _not_checked_fields = ['simulation_level', 'previous_level_simulation']

    def __init__(
        self,
        parallel_calibration: ParallelCalibration,
        simulation_level: int,
        id: str | None = None,
    ) -> None:
        super().__init__(id=id)
        self.parallel_calibration = parallel_calibration
        self.simulation_level = simulation_level
        self._previous_level_simulation: "Simulation | None" = None

    @property
    def experiment_configuration(self):
        return self.parallel_calibration.experiment_configuration

    @property
    def parallel_calibration_results(self) -> CalibrationResults:
        return self.parallel_calibration.parallel_calibration_results

    def calculate(self) -> None:
        if self.simulation_level > 0:
            self._previous_level_simulation = Simulation(
                self.parallel_calibration, simulation_level=self.simulation_level - 1
            )
            self._previous_level_simulation.resolve()
        minutes = 2 ** (self.simulation_level - 1) if self.simulation_level > 0 else 1
        loops_by_platform = {
            platform: int(self.parallel_calibration_results[platform] * minutes)
            for platform in self.parallel_calibration_results
        }
        loops_str = ", ".join(f"{p.name}={loops}" for p, loops in loops_by_platform.items())
        print(f"Simulation level {self.simulation_level} ({minutes} min): {loops_str}")
        current_results = self.experiment_configuration.run_parallel_simulations(
            loops_by_platform, calibration_mode=False
        )
        if (
            self._previous_level_simulation
            and self._previous_level_simulation._results is not None
        ):
            self._results = xr.concat(
                [self._previous_level_simulation._results, current_results],
                dim="level",
            )
        else:
            self._results = current_results

    def generate_id(self) -> str:
        return f"{self.experiment_configuration.experiment_configuration_string}_simulation_{self.simulation_level}"
    
    @property
    def simple_name(self) -> str:
        return "Simulation"
