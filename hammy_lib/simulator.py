import json
import time
import xarray as xr
from cffi import FFI
from pathlib import Path
from multiprocessing import Pool
import psutil
from functools import partial
from typing import Dict, Callable
import ctypes
import inspect
from .util import SimulatorPlatforms, CCode, SimulatorConstants, CalibrationResults, Experiment
from .machine_configuration import MachineConfiguration

class Simulator:
  def __init__(self, experiment: Experiment, 
      simulator_constants: SimulatorConstants, 
      python_simulator_function: Callable[[int, xr.DataArray], None], 
      c_code: CCode,
      use_cuda: bool,
      seed : int,
      threads: int | None = None): 
    self.experiment = experiment 
    self.simulator_constants = simulator_constants
    self.python_simulator_function = python_simulator_function
    self.c_code = c_code    
    self.use_cuda = use_cuda
    self.seed = seed
    if threads is None:
      threads = psutil.cpu_count(logical=False) + int(use_cuda)
    self.pool = Pool(threads)
    try:
      ctypes.CDLL("nvcuda.dll" if os.name == "nt" else "libcuda.so")
      if not use_cuda:
         raise ValueError("CUDA is available but not enabled in arguments")
    except:
      if use_cuda:
          raise ValueError("CUDA is not available but is enabled in arguments")
    includes = self.c_code.generate_include()
    if isinstance(self.c_code.code, Path):
      with open(self.c_code.code, "r") as f:
        code_text = f.read()
      with open(self.c_code.code.with_suffix(".h"), "w") as f:
        f.write(includes)
    else:
      code_text = self.c_code
    self.ffi_source = "\n".join(["#define FROM_PYTHON", includes, code_text])
    self.cuda_source = "xxx" if use_cuda else None
    self.machine_configuration = MachineConfiguration.detect(experiment, threads)
    if not Path('.calibration_results_cache').exists():
       self.calibration_results_cache = {}
    else:
      with open(path, 'r') as f:
        self.calibration_results_cache = json.load(f)    
    self.calibration_results_cache_key = CalibrationResultsCacheKey(
      experiment=experiment,
      threads=threads,
      use_cuda=use_cuda,
      machine_configuration=self.machine_configuration,
      numpy_hash=hash(inspect.getsource(self.python_simulator_function)),
      cffi_hash=hash(self.ffi_source),
      cuda_hash=hash(self.cuda_source) if use_cuda else None
    )

  # To avoid cannot pickle error by the multiprocessing module
  def __getstate__(self):
    state = self.__dict__.copy()
    del state['pool'] 
    return state

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
            self.python_simulator_function(loops, out, self.seed)            
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

  def run_sequential_calibration(self, force=False) -> CalibrationResults:
    if not force and self.machine_configuration in self.calibration_results_cache:
      print(f"Using cached calibration results for {self.calibration_results_cache_key.to_id()}")
      return self.calibration_results_cache[self.calibration_results_cache_key.to_id()]
    results = {}
    platforms = [SimulatorPlatforms.PYTHON, SimulatorPlatforms.CFFI] 
    platforms += [SimulatorPlatforms.CUDA] if self.use_cuda else []
    for platform in platforms:    
        results[platform] = self.run_single_calibration(platform)    
    return results
  
  def run_parallel_calibration(self) -> CalibrationResults:
    sequential_results = self.run_sequential_calibration()
    parallel_results = self.run_parallel_simulations(sequential_results, calibration_mode=True)
    for platform in parallel_results:
      if abs(parallel_results[platform] - sequential_results[platform]) > sequential_results[platform] * 0.25:
        raise ValueError(f"Parallel calibration for {platform} differs too much from sequential calibration")
    return parallel_results
    

