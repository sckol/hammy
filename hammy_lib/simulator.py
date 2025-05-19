import json
import time
import xarray as xr
from cffi import FFI
from pathlib import Path
from multiprocessing import Pool
import psutil
from functools import partial
from typing import Callable
import ctypes
import inspect
from .util import SimulatorPlatforms, CCode, SimulatorConstants, CalibrationResults, Experiment, CalibrationResultsCacheKey, flatten_dict, calibration_results_from_plain_dict, calibration_results_to_plain_dict
from .hashes import to_int_hash, hash_to_digest
from .machine_configuration import MachineConfiguration
import pandas as pd

class Simulator:  
  def __init__(self, experiment: Experiment, 
      simulator_constants: SimulatorConstants, 
      python_simulator_function: Callable[[int, xr.DataArray], None], 
      c_code: CCode,
      use_cuda: bool,
      seed : int,
      threads: int | None = None,
      digest: str | None = None) -> None: 
    self.experiment = experiment 
    self.simulator_constants = simulator_constants
    self.python_simulator_function = python_simulator_function
    self.c_code = c_code    
    self.use_cuda = use_cuda
    self.seed = seed
    if threads is None:
      threads = psutil.cpu_count(logical=False) + int(use_cuda)
    self.threads = threads
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
    if digest:
      self.digest = digest      
      self.calibration_results_cache_key = None
    else:
      self.machine_configuration = MachineConfiguration.detect()    
      self.calibration_results_cache_key = CalibrationResultsCacheKey(
        experiment=experiment,
        threads=threads,
        use_cuda=use_cuda,
        machine_configuration=MachineConfiguration.detect(),
        numpy_hash=hash_to_digest(to_int_hash(inspect.getsource(self.python_simulator_function))),
        cffi_hash=hash_to_digest(to_int_hash(self.ffi_source)),
        cuda_hash=hash_to_digest(to_int_hash(self.cuda_source)) if use_cuda else None
      )
      self.digest = self.calibration_results_cache_key.digest()       
    self.calibration_results_cache_file = self.RESULTS_DIR / f"{self.experiment.to_folder_name()}_{self.digest}_calibration.json"            
    self.simulation_results_cache_file = self.RESULTS_DIR / str(self.experiment.to_folder_name()) / f"{self.digest}_simulation.nc"    
    
  def dump_calibration_results(self, calibration_results: CalibrationResults) -> None: 
    if self.calibration_results_cache_key is None:
      raise ValueError("The simulator was created with a ready digest. Cannot dump calibration results.")   
    self.calibration_results_cache_file.parent.mkdir(parents=True, exist_ok=True)
    with open(self.calibration_results_cache_file, 'w') as f:
      res = calibration_results_to_plain_dict(calibration_results)
      res['__key'] = self.calibration_results_cache_key.digest()
      res['__key_data'] = self.calibration_results_cache_key
      json.dump(flatten_dict(res), f, indent=2)      

  def load_calibration_results(self) -> CalibrationResults:
    if not Path(self.calibration_results_cache_file).exists():
      return None
    with open(self.calibration_results_cache_file, 'r') as f:
      raw_data = json.load(f)
      return calibration_results_from_plain_dict(raw_data)

  def dump_simulation_results(self, simulation_results: xr.DataArray) -> None:
    self.simulation_results_cache_file.parent.mkdir(parents=True, exist_ok=True)      
    # Save as netCDF with compression    
    simulation_results.to_netcdf(self.simulation_results_cache_file
                                 #, encoding={ 'SimulationResults': {'zlib': True, 'complevel': 5}} #TODO: FIXMEEEEEEEEEEE
    )

  def load_simulation_results(self) -> xr.DataArray:    
    if not Path(self.simulation_results_cache_file).exists():
      return None
    return xr.load_dataarray(self.simulation_results_cache_file)

  # To avoid cannot pickle error by the multiprocessing module
  def __getstate__(self):
    state = self.__dict__.copy()
    del state['pool'] 
    return state


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
    if not force:
      cached_calibration_results = self.load_calibration_results()
      if cached_calibration_results:
        print(f"Using cached calibration results from {self.calibration_results_cache_file}")
        return cached_calibration_results
    results : CalibrationResults = {}
    platforms = [SimulatorPlatforms.PYTHON, SimulatorPlatforms.CFFI] 
    platforms += [SimulatorPlatforms.CUDA] if self.use_cuda else []
    for platform in platforms:    
        results[platform] = self.run_single_calibration(platform)    
    return results
  
  def run_parallel_calibration(self, force_run_sequential=False, tolerance=25) -> CalibrationResults:
    sequential_results = self.run_sequential_calibration(force=force_run_sequential)
    parallel_results = self.run_parallel_simulations(sequential_results, calibration_mode=True)
    discrepancies = []
    for platform in parallel_results:
      diff_percent = abs(parallel_results[platform] - sequential_results[platform]) / sequential_results[platform] * 100
      if diff_percent > tolerance:
        discrepancies.append(f"{platform.value}: {diff_percent:.1f}% difference ({parallel_results[platform] - sequential_results[platform]:+d} loops, parallel: {parallel_results[platform]}, sequential: {sequential_results[platform]})")
        print(f"WARNING: {platform.value} calibration differs by {diff_percent:.1f}% ({parallel_results[platform] - sequential_results[platform]:+d} loops, parallel value: {parallel_results[platform]})")
      else:
        print(f"OK: {platform.value} calibration matches within {diff_percent:.1f}% ({parallel_results[platform] - sequential_results[platform]:+d} loops, parallel value: {parallel_results[platform]})")    
    if discrepancies:
      raise ValueError("Parallel calibration failed:\n" + "\n".join(discrepancies))  
    return parallel_results

  def run_level_simulation(self, max_level: int, calibration_results: CalibrationResults | None = None,
                            force_rebuild: bool = False) -> xr.DataArray:
    # TODO: Take the calibration_results from the simulation results 
    if max_level < 0:
      raise ValueError("max_level must be >= 0")
    if force_rebuild:
      cached_simulation_results = None
    else:
      cached_simulation_results = self.load_simulation_results()
    if cached_simulation_results is not None:
      dataset_calibration_results = calibration_results_from_plain_dict(
        json.loads(cached_simulation_results.attrs['CalibrationResults']))
      if calibration_results and calibration_results != dataset_calibration_results:
        raise ValueError("Calibration results do not match cached simulation results.")
      calibration_results = dataset_calibration_results
      if not calibration_results:
        raise ValueError("No calibration results found in cached simulation results.")
    elif not calibration_results:
      calibration_results = self.load_calibration_results()      
    if not calibration_results:
      raise ValueError("No calibration results found. Please run calibration first.")
    if cached_simulation_results is not None:    
      current_level = cached_simulation_results.level.size - 1
    else:             
      current_level = -1        
    if current_level >= max_level:
      print(f"Simulation already completed for level {max_level}")
      return cached_simulation_results    
    current_results = []
    for level in range(current_level + 1, max_level + 1):      
      minutes = 2 ** (level - 1) if level > 0 else 1
      print(f"Running level {level} simulation for {minutes} minutes...")
      loops_by_platform = {platform: int(calibration_results[platform] * minutes) for platform in calibration_results}
      current_results.append(self.run_parallel_simulations(loops_by_platform))      
    new_results = xr.concat(current_results, dim='level')            
    new_results = new_results.assign_coords(level=range(current_level + 1, max_level + 1))  
    if cached_simulation_results is not None:
        res = xr.concat([cached_simulation_results, new_results], dim='level')
    else:
        res = new_results
    res.attrs['CalibrationResults'] = json.dumps(calibration_results_to_plain_dict(calibration_results))
    res.name = 'SimulationResults'
    return res