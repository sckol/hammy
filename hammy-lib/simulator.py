import time
import xarray as xr
import random
from cffi import FFI
from pathlib import Path
from multiprocessing import Pool
import psutil
from functools import partial
from typing import Dict, Callable
import ctypes
from .util import SimulatorPlatforms, CCode, SimulatorConstants, CalibrationResults

class Simulator:
  def __init__(self, simulator_constants: SimulatorConstants, 
      python_simulator_function: Callable[[int, xr.DataArray], None], 
      c_code: CCode, threads: int = psutil.cpu_count(logical=False), 
      seed : int = random.randint(0, 2**32 - 1)):
    try:
      ctypes.CDLL("nvcuda.dll" if os.name == "nt" else "libcuda.so")
      self.use_cuda = True
    except:
      self.use_cuda = False
    self.threads = threads
    self.simulator_constants = simulator_constants
    self.python_simulator = python_simulator_function
    self.c_code = c_code    
    self.pool = Pool(threads)
    self.seed = seed
  
  # To avoid cannot pickle error by the multiprocessing module
  def __getstate__(self):
    state = self.__dict__.copy()
    del state['pool'] 
    return state
    
  def compile(self) -> None:      
    includes = self.c_code.generate_include()
    if isinstance(c_code.code, Path):
      with open(c_code.code, "r") as f:
        code_text = f.read()
      with open(c_code.code.with_suffix(".h"), "w") as f:
        f.write(includes)
    else:
      code_text = c_code
    ffi = FFI()     
    ffi_source = "\n".join(["#define FROM_PYTHON", includes, code_text])
    ffi_libs = ["pcg_basic"]
    ffi_sources = [str(Path(__file__).parent / "c-libs" / lib / f"{lib}.c") 
      for lib in ffi_libs]
    ffi.set_source("hammy_cpu_kernel", ffi_source, sources=ffi_sources)
    ffi.cdef(c_code.function_header)
    output_dir = Path().parent / "build-cffi"
    ffi.compile(verbose=True, tmpdir=str(output_dir), target=str("hammy_cpu_kernel_lib.*"))

  def cffi_simulator(self, loops: int, out: xr.DataArray, seed: int) -> None:
    ffi = FFI()
    lib = ffi.dlopen(str(list((Path.cwd() / 'build-cffi').glob("hammy_cpu_kernel_lib.*"))[0]))
    ffi.cdef(self.c_code.function_header)
    buffer = ffi.cast("unsigned long long*", ffi.from_buffer(out.values))
    lib.run_simulation(loops, seed, buffer)
    ffi.dlclose(lib)
    return out

  def run_single_simulation(self, platform: SimulatorPlatforms, loops: int, calibration_mode=False) -> xr.DataArray | float:
    out = self.simulator_constants.create_empty_results()
    start_time = time.time()    
    match platform:
        case SimulatorPlatforms.PYTHON:
            self.python_simulator(loops, out, seed)            
        case SimulatorPlatforms.CFFI:          
            self.cffi_simulator(loops, out, seed)                     
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
          results = {SimulatorPlatforms.PYTHON: res[0]}
          cffi_times = res[1:-1] if self.use_cuda else res[1:]
          min_cffi = min(cffi_times)
          max_cffi = max(cffi_times)
          if max_cffi > min_cffi * 1.25:
            raise ValueError("CFFI times vary too much between threads")
          results[SimulatorPlatforms.CFFI] = sum(cffi_times) / len(cffi_times)
          if self.use_cuda:
            results[SimulatorPlatforms.CUDA] = res[-1]
          return results
    combined = xr.concat(res, dim='thread')
    return combined

  def run_single_calibration(self, platform: SimulatorPlatforms) -> float:
    loops = 1000
    while True:
        print(f"Running calibration with {loops} loops...")
        elapsed_time = self.run_single_simulation(platform, loops, calibration_mode=True)
        print(f"Simulation took {elapsed_time:.2f} seconds")            
        if elapsed_time > 15:
            # Calculate loops needed for 1 minute
            one_min_loops = int(loops * 60 / elapsed_time)
            print(f"Estimated {one_min_loops} loops needed for 1 minute")            
            # Run with calculated loops
            print(f"Running verification with {one_min_loops} loops...")
            elapsed_time = self.run_single_simulation(platform, one_min_loops, calibration_mode=True)
            print(f"Verification took {elapsed_time:.2f} seconds")                
            # Final adjustment
            final_loops = int(one_min_loops * 60 / elapsed_time)
            print(f"Final calibration: {final_loops} loops per minute")
            return final_loops            
        loops *= 2

  def run_sequential_calibration(self) -> CalibrationResults:
    results = {}
    platforms = [SimulatorPlatforms.PYTHON, SimulatorPlatforms.CFFI] 
    platforms += [SimulatorPlatforms.CUDA] if self.use_cuda else []
    for platform in platforms:    
        results[platform] = self.run_single_calibration(platform)
    return results
  
  def run_parallel_calibration(self, sequential_results: CalibrationResults) -> CalibrationResults:
    parallel_results = self.run_parallel_simulations(sequential_results, calibration_mode=True)
    for platform in parallel_results:
      if abs(parallel_results[platform] - sequential_results[platform]) > sequential_results[platform] * 0.25:
        raise ValueError(f"Parallel calibration for {platform} differs too much from sequential calibration")
    return parallel_results
    

