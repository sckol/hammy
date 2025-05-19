from abc import ABC, abstractmethod
from pathlib import Path
import xarray as xr
from cffi import FFI
from time import time
from .machine_configuration import MachineConfiguration
from .simulator_platforms import SimulatorPlatforms
from .ccode import CCode

class Experiment(ABC):
  number: int    
  name: str
  version: int  
  c_code: CCode
  use_cuda: bool
  machine_configuration: MachineConfiguration

  def to_folder_name(self) -> str:
    return f"{self.number}_{self.version}"
  
  @abstractmethod
  def create_empty_results(self) -> xr.DataArray:
    pass
  
  @abstractmethod
  def simulate_using_python(loops: int, out: xr.DataArray, seed: int) -> None:
    pass

  def compile(self) -> None:          
    ffi = FFI()     
    ffi_libs = ["pcg_basic"]
    ffi_sources = [str(Path(__file__).parent / "c_libs" / lib / f"{lib}.c") 
      for lib in ffi_libs]
    ffi.set_source("hammy_cpu_kernel", self.ffi_source, sources=ffi_sources)
    ffi.cdef(self.c_code.function_header)
    output_dir = Path().parent / "build_cffi"
    ffi.compile(verbose=True, tmpdir=str(output_dir), target=str("hammy_cpu_kernel_lib.*"))

  def cffi_simulator(self, loops: int, out: xr.DataArray, seed: int) -> None:
    ffi = FFI()
    lib = ffi.dlopen(str(list((Path.cwd() / 'build_cffi').glob("hammy_cpu_kernel_lib.*"))[0]))
    ffi.cdef(self.c_code.function_header)
    buffer = ffi.cast("unsigned long long*", ffi.from_buffer(out.values))
    lib.run_simulation(loops, seed, buffer)
    ffi.dlclose(lib)
    return out

  def run_single_simulation(self, platform: SimulatorPlatforms, loops: int, calibration_mode=False) -> xr.DataArray | float:
    print(f"Running simulation with seed {self.seed}...")
    out = self.simulator_constants.create_empty_results()
    start_time = time.time()        
    match platform:
        case SimulatorPlatforms.PYTHON:
            self.simulate_using_python(loops, out, self.seed)            
        case SimulatorPlatforms.CFFI:          
            self.cffi_simulator(loops, out, self.seed)                     
        case SimulatorPlatforms.CUDA:
            raise ValueError(f"CUDA platform not implemented yet")
        case _:
            raise ValueError(f"Unknown platform: {platform}")
    elapsed_time = time.time() - start_time
    self.seed = self.seed + 1
    return elapsed_time if calibration_mode else out
  
  def run_simulation_thread(self, thread_id: int, loops_by_platform: CalibrationResults, calibration_mode=False) -> xr.DataArray | float:
    self.seed = self.seed + thread_id
    platform = SimulatorPlatforms.CFFI if thread_id == 0 else SimulatorPlatforms.PYTHON
    return self.run_single_simulation(platform, loops_by_platform[platform], calibration_mode)
      
  def run_parallel_simulations(self, loops_by_platform: CalibrationResults, calibration_mode=False) -> xr.DataArray | CalibrationResults:
    res = self.pool.map(partial(self.run_simulation_thread, loops_by_platform=loops_by_platform, calibration_mode=calibration_mode), 
            list(range(self.threads)))
    self.seed = self.seed + self.threads  
    if calibration_mode:
      def time_to_loops(time: float, platform: SimulatorPlatforms) -> int:
        return int(loops_by_platform[platform] / time * 60)
      results = {SimulatorPlatforms.PYTHON: time_to_loops(res[0], SimulatorPlatforms.PYTHON)}
      cffi_times = res[1:-1] if self.use_cuda else res[1:]
      min_cffi = min(cffi_times)
      max_cffi = max(cffi_times)
      if max_cffi > min_cffi * 1.25:
        raise ValueError("CFFI times vary too much between threads")
      results[SimulatorPlatforms.CFFI] = time_to_loops(sum(cffi_times) / len(cffi_times), SimulatorPlatforms.CFFI)
      if self.use_cuda:
        results[SimulatorPlatforms.CUDA] = time_to_loops(res[-1], SimulatorPlatforms.CUDA)
      return results    
    python_result = res[0]
    cffi_results = res[1:-1] if self.use_cuda else res[1:]
    cuda_result = res[-1] if self.use_cuda else None
    cffi_combined = sum(cffi_results[1:], cffi_results[0])
    platform_results = [python_result, cffi_combined]
    if self.use_cuda:
      platform_results.append(cuda_result)
    return xr.concat(platform_results, dim=pd.Index([p.name for p in [SimulatorPlatforms.PYTHON, SimulatorPlatforms.CFFI] + ([SimulatorPlatforms.CUDA] if self.use_cuda else [])], name='platform'))
