import json
from .hammy_object import DictHammyObject
from .calibration_results import (
    CalibrationResults,
    calibration_results_from_plain_dict,
    calibration_results_to_plain_dict,
)
from .sequential_calibration import SequentialCalibration


class ParallelCalibration(DictHammyObject):
    def __init__(
        self,
        sequential_calibration: SequentialCalibration,
        calibration_tolerance=25,
        id: str = None,
    ) -> None:
        super().__init__(id=id)
        self.sequential_calibration = sequential_calibration
        self.calibration_tolerance = calibration_tolerance

    @property
    def experiment_configuration(self):
        return self.sequential_calibration.experiment_configuration

    @property
    def sequential_calibration_results(self) -> CalibrationResults:
        return self.sequential_calibration.sequential_calibration_results

    def calculate(self) -> None:
        parallel_results = self.experiment_configuration.run_parallel_simulations(
            self.sequential_calibration_results, calibration_mode=True
        )
        discrepancies = []
        for platform in parallel_results:
            diff_percent = (
                abs(
                    parallel_results[platform]
                    - self.sequential_calibration_results[platform]
                )
                / self.sequential_calibration_results[platform]
                * 100
            )
            if diff_percent > self.calibration_tolerance:
                discrepancies.append(
                    f"{platform.value}: {diff_percent:.1f}% difference ({parallel_results[platform] - self.sequential_calibration_results[platform]:+d} loops, parallel: {parallel_results[platform]}, sequential: {self.sequential_calibration_results[platform]})"
                )
                print(
                    f"WARNING: {platform.value} calibration differs by {diff_percent:.1f}% ({parallel_results[platform] - self.sequential_calibration_results[platform]:+d} loops, parallel value: {parallel_results[platform]})"
                )
            else:
                print(
                    f"OK: {platform.value} calibration matches within {diff_percent:.1f}% ({parallel_results[platform] - self.sequential_calibration_results[platform]:+d} loops, parallel value: {parallel_results[platform]})"
                )
        if discrepancies:
            raise ValueError(
                "Parallel calibration failed:\n" + "\n".join(discrepancies)
            )
        self.metadata["parallel_calibration_results"] = json.dumps(
            calibration_results_to_plain_dict(parallel_results)
        )

    @property
    def parallel_calibration_results(self) -> CalibrationResults:
        data = json.loads(self.metadata["parallel_calibration_results"])
        return calibration_results_from_plain_dict(data)

    def generate_id(self) -> str:
        return f"{self.experiment_configuration.experiment_configuration_string}_parallel_calibration"
