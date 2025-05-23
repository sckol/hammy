from .simulator_platforms import SimulatorPlatforms
from typing import Dict

CalibrationResults = Dict[SimulatorPlatforms, int]

def calibration_results_from_plain_dict(data: Dict) -> CalibrationResults:
    return {SimulatorPlatforms[platform]: loops for platform, loops in data.items()}

def calibration_results_to_plain_dict(data: CalibrationResults) -> Dict:
    return {platform.name: loops for platform, loops in data.items()}