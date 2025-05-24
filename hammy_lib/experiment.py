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

    @property
    def id(self) -> str:
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
        ffi.set_source("hammy_cpu_kernel", self.ffi_source, sources=ffi_sources)
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

    def run_single_simulation(
        self,
        seed: int,
        platform: SimulatorPlatforms,
        loops: int,
        calibration_mode=False,
    ) -> xr.DataArray | float:
        print(f"Running simulation with seed {seed}...")
        out = self.simulator_constants.create_empty_results()
        start_time = time.time()
        match platform:
            case SimulatorPlatforms.PYTHON:
                self.simulate_using_python(loops, out, seed)
            case SimulatorPlatforms.CFFI:
                self.cffi_simulator(loops, out, seed)
            case SimulatorPlatforms.CUDA:
                raise ValueError(f"CUDA platform not implemented yet")
            case _:
                raise ValueError(f"Unknown platform: {platform}")
        elapsed_time = time.time() - start_time
        return elapsed_time if calibration_mode else out
