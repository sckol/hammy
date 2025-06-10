import json
from .hammy_object import DictHammyObject
from .calibration_results import (
    CalibrationResults,
    calibration_results_from_plain_dict,
    calibration_results_to_plain_dict,
)
from .experiment_configuration import ExperimentConfiguration
from .simulator_platforms import SimulatorPlatforms


class SequentialCalibration(DictHammyObject):
    _no_check_metadata = True

    def __init__(
        self,
        experiment_configuration: ExperimentConfiguration,
        id: str = None,
        dry_run: bool = False,
    ) -> None:
        super().__init__(id=id)
        self.experiment_configuration = experiment_configuration
        self.dry_run = dry_run

    def run_single_calibration(self, platform: SimulatorPlatforms) -> float:
        dry_run_multiplier = 0.1 if self.dry_run else 1.0
        loops = int(1000 * dry_run_multiplier)
        while True:
            print(f"Running calibration with {loops} loops...")
            elapsed_time = self.experiment_configuration.run_single_simulation(
                platform, loops, calibration_mode=True
            )
            print(f"Simulation took {elapsed_time:.2f} seconds")
            if elapsed_time > 15 * dry_run_multiplier:
                # Calculate loops needed for 1 minute
                one_min_loops = int(loops * 60 * dry_run_multiplier / elapsed_time)
                print(f"Estimated {one_min_loops} loops needed for 1 minute")
                # Run with calculated loops
                print(f"Running verification with {one_min_loops} loops...")
                elapsed_time = self.experiment_configuration.run_single_simulation(
                    platform, one_min_loops, calibration_mode=True
                )
                print(f"Verification took {elapsed_time:.2f} seconds")
                # Final adjustment
                final_loops = int(
                    one_min_loops * 60 * dry_run_multiplier / elapsed_time
                )
                print(f"Final calibration: {final_loops} loops per minute")
                return final_loops
            loops *= 2

    def calculate(self) -> None:
        results: CalibrationResults = {}
        platforms = [SimulatorPlatforms.PYTHON, SimulatorPlatforms.CFFI]
        platforms += (
            [SimulatorPlatforms.CUDA] if self.experiment_configuration.use_cuda else []
        )
        for platform in platforms:
            results[platform] = self.run_single_calibration(platform)
        self.metadata["sequential_calibration_results"] = json.dumps(
            calibration_results_to_plain_dict(results)
        )

    @property
    def sequential_calibration_results(self) -> CalibrationResults:
        data = json.loads(self.metadata["sequential_calibration_results"])
        return calibration_results_from_plain_dict(data)

    def generate_id(self) -> str:
        return f"{self.experiment_configuration.experiment_configuration_string}_sequential_calibration"
