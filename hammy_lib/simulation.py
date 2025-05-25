import xarray as xr
from .hammy_object import ArrayHammyObject
from .calibration_results import CalibrationResults
from .parallel_calibration import ParallelCalibration


class Simulation(ArrayHammyObject):
    def __init__(
        self,
        parallel_calibration: ParallelCalibration,
        simulation_level: int,
        id: str = None,
    ) -> None:
        super().__init__(id=id)
        self.parallel_calibration = parallel_calibration
        self.simulation_level = simulation_level
        self.previous_level_simulation: "Simulation" | None = None

    @property
    def experiment_configuration(self):
        return self.parallel_calibration.experiment_configuration

    @property
    def parallel_calibration_results(self) -> CalibrationResults:
        return self.parallel_calibration.parallel_calibration_results

    def calculate(self) -> None:
        if self.simulation_level > 0:
            self.previous_level_simulation = Simulation(
                self.parallel_calibration, simulation_level=self.simulation_level - 1
            )
            self.previous_level_simulation.resolve()
        minutes = 2 ** (self.simulation_level - 1) if self.simulation_level > 0 else 1
        print(f"Running level {self.simulation_level} simulation for {minutes} minutes...")
        loops_by_platform = {
            platform: int(self.parallel_calibration_results[platform] * minutes)
            for platform in self.parallel_calibration_results
        }
        current_results = self.experiment_configuration.run_parallel_simulations(
            loops_by_platform, calibration_mode=False
        )
        if (
            self.previous_level_simulation
            and self.previous_level_simulation._results is not None
        ):
            self._results = xr.concat(
                [self.previous_level_simulation._results, current_results],
                dim="level",
            )
        else:
            self._results = current_results

    def generate_id(self) -> str:
        return f"{self.experiment_configuration.experiment_configuration_string}_simulation_{self.simulation_level}"
