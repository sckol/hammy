from multiprocessing import Pool
from .hammy_object import DictHammyObject
from .machine_configuration import MachineConfiguration
from .experiment import Experiment
from .simulator_platforms import SimulatorPlatforms
from .calibration_results import CalibrationResults
import xarray as xr
from typing import Optional
from functools import partial
import pandas as pd
from time import time
import psutil


class ExperimentConfiguration(DictHammyObject):
    def __init__(
        self,
        experiment: Experiment,
        machine_configuration: MachineConfiguration,
        seed: Optional[int] = None,
        threads: Optional[int] = None,
        digest: Optional[str] = None,
    ) -> None:
        super().__init__(digest=digest)
        self.experiment = experiment
        self.machine_configuration = machine_configuration
        self.seed = seed or int(time() * 1000)
        self.use_cuda = (machine_configuration.metadata.get("cuda_cores") or 0) > 0
        if threads is None:
            threads = psutil.cpu_count(logical=False) + int(self.use_cuda)
        self.threads = threads
        self._pool = Pool(threads)

    @property
    def id(self) -> str:
        return f"{self.experiment.experiment_string}_{self.machine_configuration.digest}_experiment_configuration"

    @property
    def file_extension(self) -> str:
        return "json"

    @property
    def experiment_string(self) -> str:
        return self.experiment.experiment_string

    @property
    def pool(self) -> Pool:
        return self._pool

    @property
    def filename(self) -> str:
        return super().filename

    def calculate(self) -> None:
        pass

    def run_single_simulation(
        self, platform: SimulatorPlatforms, loops: int, calibration_mode=False
    ) -> xr.DataArray | float:
        # Use self.seed and increment for each call
        result = self.experiment.run_single_simulation(
            self.seed, platform, loops, calibration_mode
        )
        self.seed += 1
        return result

    def run_simulation_thread(
        self,
        thread_id: int,
        loops_by_platform: CalibrationResults,
        calibration_mode=False,
    ) -> xr.DataArray | float:
        # Each thread gets a unique seed
        platform = (
            SimulatorPlatforms.CFFI if thread_id == 0 else SimulatorPlatforms.PYTHON
        )
        return self.experiment.run_single_simulation(
            self.seed + thread_id,
            platform,
            loops_by_platform[platform],
            calibration_mode,
        )

    def run_parallel_simulations(
        self, loops_by_platform: CalibrationResults, calibration_mode=False
    ) -> xr.DataArray | CalibrationResults:
        res = self.pool.map(
            partial(
                self.run_simulation_thread,
                loops_by_platform=loops_by_platform,
                calibration_mode=calibration_mode,
            ),
            list(range(self.threads)),
        )
        if calibration_mode:

            def time_to_loops(time: float, platform: SimulatorPlatforms) -> int:
                return int(loops_by_platform[platform] / time * 60)

            results = {
                SimulatorPlatforms.PYTHON: time_to_loops(
                    res[0], SimulatorPlatforms.PYTHON
                )
            }
            cffi_times = res[1:-1] if self.use_cuda else res[1:]
            min_cffi = min(cffi_times)
            max_cffi = max(cffi_times)
            if max_cffi > min_cffi * 1.25:
                raise ValueError("CFFI times vary too much between threads")
            results[SimulatorPlatforms.CFFI] = time_to_loops(
                sum(cffi_times) / len(cffi_times), SimulatorPlatforms.CFFI
            )
            if self.use_cuda:
                results[SimulatorPlatforms.CUDA] = time_to_loops(
                    res[-1], SimulatorPlatforms.CUDA
                )
            return results
        python_result = res[0]
        cffi_results = res[1:-1] if self.use_cuda else res[1:]
        cuda_result = res[-1] if self.use_cuda else None
        cffi_combined = sum(cffi_results[1:], cffi_results[0]) if cffi_results else None
        platform_results = [python_result, cffi_combined]
        if self.use_cuda:
            platform_results.append(cuda_result)
        return xr.concat(
            platform_results,
            dim=pd.Index(
                [
                    p.name
                    for p in [SimulatorPlatforms.PYTHON, SimulatorPlatforms.CFFI]
                    + ([SimulatorPlatforms.CUDA] if self.use_cuda else [])
                ],
                name="platform",
            ),
        )
