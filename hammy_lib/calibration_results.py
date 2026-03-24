from .simulator_platforms import SimulatorPlatforms

CalibrationResults = dict[SimulatorPlatforms, int]


def calibration_results_from_plain_dict(data: dict) -> CalibrationResults:
    return {SimulatorPlatforms[platform]: loops for platform, loops in data.items()}


def calibration_results_to_plain_dict(data: CalibrationResults) -> dict:
    return {platform.name: loops for platform, loops in data.items()}
