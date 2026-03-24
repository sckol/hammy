from multiprocessing import Pool
from .hammy_object import DictHammyObject
from .machine_configuration import MachineConfiguration
from .experiment import Experiment
from .simulator_platforms import SimulatorPlatforms
from .calibration_results import CalibrationResults
import xarray as xr
from typing import Optional
from functools import partial
from time import time
import psutil


class ExperimentConfiguration(DictHammyObject):
    def __init__(
        self,
        experiment: Experiment,
        machine_configuration: MachineConfiguration,
        seed: Optional[int] = None,
        threads: Optional[int] = None,
        id: Optional[str] = None,
    ) -> None:
        super().__init__(id=id)
        self.experiment = experiment
        self.machine_configuration = machine_configuration
        self.seed = seed or int(time() * 1000)
        self.use_cuda = (machine_configuration.metadata.get("cuda_cores") or 0) > 0
        self.cores = psutil.cpu_count(logical=False)
        if threads is None:
            # Ensure CFFI gets at least 1 thread even on single-core machines.
            # PYTHON=thread 0, CFFI=thread 1+, CUDA=main process (async).
            threads = max(3, self.cores + 1) if self.use_cuda else max(2, self.cores)
        self.threads = threads
        self._pool = Pool(threads)

    @property
    def experiment_configuration_string(self) -> str:
        return (
            f"{self.experiment.experiment_string}_{self.machine_configuration.digest}"
        )

    def generate_id(self) -> str:
        return f"{self.experiment_configuration_string}_experiment_configuration"

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

    @staticmethod
    def run_simulation_thread(
        thread_id: int,
        experiment: Experiment,
        seed: int,
        loops_by_platform: CalibrationResults,
        calibration_mode=False,
    ) -> xr.DataArray | float:
        # Each thread gets a unique seed
        platform = (
            SimulatorPlatforms.PYTHON if thread_id == 0 else SimulatorPlatforms.CFFI
        )
        return experiment.run_single_simulation(
            seed + thread_id,
            platform,
            loops_by_platform[platform],
            calibration_mode,
        )

    def run_parallel_simulations(
        self, loops_by_platform: CalibrationResults, calibration_mode=False
    ) -> xr.DataArray | CalibrationResults:
        # CUDA runs in main process (not in Pool — CUDA contexts are not fork-safe).
        # Launch CUDA first (async — returns immediately), then run CPU Pool
        # while GPU computes, then sync GPU before collecting results.
        cpu_threads = self.threads - (1 if self.use_cuda else 0)
        cuda_result = None
        gpu_out = None
        cuda_out = None
        mode = "calibration" if calibration_mode else "simulation"
        platforms_str = f"PYTHON + {cpu_threads - 1} CFFI" + (" + CUDA" if self.use_cuda else "")
        print(f"Running parallel {mode}: {platforms_str}")
        if self.use_cuda:
            cuda_out = self.experiment.create_empty_results()
            cuda_start = time()
            print(f"[CUDA] Launching {loops_by_platform[SimulatorPlatforms.CUDA]} loops (async)...")
            gpu_out = self.experiment.cuda_simulator_launch(
                loops_by_platform[SimulatorPlatforms.CUDA],
                cuda_out,
                self.seed + cpu_threads,
            )
        # CPU work via Pool (runs while GPU is still computing)
        res = self.pool.map(
            partial(
                self.run_simulation_thread,
                experiment=self.experiment,
                seed=self.seed,
                loops_by_platform=loops_by_platform,
                calibration_mode=calibration_mode,
            ),
            list(range(cpu_threads)),
        )
        if self.use_cuda:
            self.experiment.cuda_simulator_sync(gpu_out, cuda_out)
            cuda_elapsed = time() - cuda_start
            print(f"[CUDA] Done in {cuda_elapsed:.2f}s")
            cuda_result = cuda_elapsed if calibration_mode else cuda_out
        self.seed += self.threads
        if calibration_mode:

            def time_to_loops(time: float, platform: SimulatorPlatforms) -> int:
                return int(loops_by_platform[platform] / time * 60)

            results = {
                SimulatorPlatforms.PYTHON: time_to_loops(
                    res[0], SimulatorPlatforms.PYTHON
                )
            }
            cffi_times = res[1:]
            if cffi_times:
                min_cffi = min(cffi_times)
                max_cffi = max(cffi_times)
                if max_cffi > min_cffi * 1.25:
                    raise ValueError("CFFI times vary too much between threads")
                results[SimulatorPlatforms.CFFI] = time_to_loops(
                    sum(cffi_times) / len(cffi_times), SimulatorPlatforms.CFFI
                )
            if self.use_cuda:
                results[SimulatorPlatforms.CUDA] = time_to_loops(
                    cuda_result, SimulatorPlatforms.CUDA
                )
            return results
        python_result = res[0]
        cffi_results = res[1:]
        cffi_combined = sum(cffi_results[1:], cffi_results[0]) if cffi_results else None
        platform_results = [python_result]
        if cffi_combined is not None:
            platform_results.append(cffi_combined)
        if self.use_cuda:
            platform_results.append(cuda_result)
        platforms = [SimulatorPlatforms.PYTHON] + (
            [SimulatorPlatforms.CFFI] if cffi_results else []
        ) + (
            [SimulatorPlatforms.CUDA] if self.use_cuda else []
        )
        return xr.concat(
            platform_results,
            dim=xr.DataArray([p.name for p in platforms], dims="platform"),
        )
