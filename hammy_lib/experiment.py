from abc import abstractmethod
from pathlib import Path
import xarray as xr
from cffi import FFI
from time import time
from .simulator_platforms import SimulatorPlatforms
from .ccode import CCode
from .calibration_results import CalibrationResults
from .hammy_object import DictHammyObject


class Experiment(DictHammyObject):
    """Defines a Monte Carlo simulation: its parameters, result shape, and kernels.

    An Experiment provides three equivalent implementations of the same simulation:
    - simulate_using_python() — NumPy reference (subclass implements)
    - cffi_simulator() — C kernel compiled to .so, called via CFFI
    - cuda_simulator() — same C kernel compiled for GPU via CuPy RawKernel

    The C code is written once (as a CCode dataclass) and compiled differently
    per platform. See common.h for how the same source works on CPU and GPU.

    Experiment does NOT run simulations by itself — it only knows how to execute
    a single batch of `loops` iterations on a given platform. The orchestration
    (how many loops, which platforms, parallelism) is handled by
    ExperimentConfiguration, which calls run_single_simulation().

    Subclasses must define:
    - experiment_number, experiment_name, experiment_version — identity
    - c_code — CCode with the C kernel source and constants
    - create_empty_results() — xarray with the right shape/coords for output
    - simulate_using_python() — reference implementation mutating out in-place

    Lifecycle:
    1. compile() — builds CFFI .so (once, before any CFFI simulation)
    2. run_single_simulation(seed, platform, loops) — dispatches to the right
       simulator, returns xr.DataArray (or elapsed time in calibration mode)
    """

    experiment_number: int
    experiment_name: str
    experiment_version: int
    c_code: CCode

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        required_attributes = [
            "experiment_number",
            "experiment_name",
            "experiment_version",
            "c_code",
        ]
        for attr in required_attributes:
            if not hasattr(cls, attr):
                raise AttributeError(
                    f"Class {cls.__name__} must define attribute {attr}"
                )

    def calculate(self) -> None:
        pass

    def generate_id(self) -> str:
        return f"{self.experiment_string}_experiment"

    @property
    def experiment_string(self) -> str:
        return super().experiment_string

    @property
    def file_extension(self) -> str:
        return super().file_extension

    @property
    def filename(self) -> str:
        return super().filename

    @abstractmethod
    def create_empty_results(self) -> xr.DataArray:
        pass

    @abstractmethod
    def simulate_using_python(self, loops: int, out: xr.DataArray, seed: int) -> None:
        pass

    def compile(self) -> None:
        ffi = FFI()
        ffi_libs = ["pcg_basic"]
        ffi_sources = [
            str(Path(__file__).parent / "c_libs" / lib / f"{lib}.c") for lib in ffi_libs
        ]
        ffi.set_source("hammy_cpu_kernel", "#define FROM_PYTHON\n" + str(self.c_code), sources=ffi_sources)
        ffi.cdef(self.c_code.function_header)
        output_dir = Path().parent / "build_cffi"
        ffi.compile(
            verbose=True, tmpdir=str(output_dir), target=str("hammy_cpu_kernel_lib.*")
        )

    def cffi_simulator(self, loops: int, out: xr.DataArray, seed: int) -> None:
        ffi = FFI()
        lib = ffi.dlopen(
            str(list((Path.cwd() / "build_cffi").glob("hammy_cpu_kernel_lib.*"))[0])
        )
        ffi.cdef(self.c_code.function_header)
        buffer = ffi.cast("unsigned long long*", ffi.from_buffer(out.values))
        lib.run_simulation(loops, seed, buffer)
        ffi.dlclose(lib)
        return out

    _cuda_kernels: dict = {}

    def _get_cuda_kernel(self, blocks: int):
        if blocks not in self._cuda_kernels:
            import cupy as cp
            source = self.c_code.to_cuda_source(blocks)
            self._cuda_kernels[blocks] = cp.RawKernel(source, 'run_simulation')
        return self._cuda_kernels[blocks]

    def cuda_simulator(self, loops: int, out: xr.DataArray, seed: int, blocks: int = 1000) -> None:
        """Launch CUDA kernel and copy results back synchronously."""
        gpu_out = self.cuda_simulator_launch(loops, out, seed, blocks)
        self.cuda_simulator_sync(gpu_out, out)

    def cuda_simulator_launch(self, loops: int, out: xr.DataArray, seed: int, blocks: int = 1000):
        """Launch CUDA kernel (async — returns immediately while GPU computes).

        Returns the GPU output buffer. Call cuda_simulator_sync() to wait and
        copy results back to the xarray.
        """
        import cupy as cp
        kernel = self._get_cuda_kernel(blocks)
        gpu_out = cp.zeros(out.values.shape, dtype=cp.int64)
        kernel((blocks,), (32,), (loops, seed, gpu_out))
        return gpu_out

    @staticmethod
    def cuda_simulator_sync(gpu_out, out: xr.DataArray) -> None:
        """Wait for GPU kernel to finish and copy results to CPU."""
        import cupy as cp
        cp.cuda.get_current_stream().synchronize()
        out.values[:] = gpu_out.get()

    def run_single_simulation(
        self,
        seed: int,
        platform: SimulatorPlatforms,
        loops: int,
        calibration_mode=False,
    ) -> xr.DataArray | float:
        print(f"Running {platform.name} simulation ({loops} loops, seed {seed})...")
        out = self.create_empty_results()
        start_time = time()
        match platform:
            case SimulatorPlatforms.PYTHON:
                self.simulate_using_python(loops, out, seed)
            case SimulatorPlatforms.CFFI:
                self.cffi_simulator(loops, out, seed)
            case SimulatorPlatforms.CUDA:
                self.cuda_simulator(loops, out, seed)
            case _:
                raise ValueError(f"Unknown platform: {platform}")
        elapsed_time = time() - start_time
        return elapsed_time if calibration_mode else out
