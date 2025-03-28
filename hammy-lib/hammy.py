import abc
import time
import importlib
import xarray as xr
from enum import Enum
from typing import Callable
import sys
import os
import random
from cffi import FFI
from pathlib import Path
import numpy as np
from multiprocessing import Pool
from dataclasses import dataclass


class SimulatorPlatforms(Enum):
    PYTHON = "python"
    CFFI = "cffi" 
    CUDA = "cuda"

@dataclass
class CCode:  
  code: str
  constants: str

  def generate_include(self) -> str:
    return "\n".join([
      self.constants, 
      "#ifdef CFFI", 
      self.generate_include_for_platform(SimulatorPlatforms.CFFI),     
      "#else", 
      self.generate_include_for_platform(SimulatorPlatforms.CUDA), 
      "#endif"])

  def generate_include_for_platform(self, platform: SimulatorPlatforms) -> str:
    hammy_path = Path(__file__).parent 
    libs = [hammy_path / "common.h"]
    if platform == SimulatorPlatforms.CFFI:
      libs += [hammy_path / "cuda_cpu.h", hammy_path / "c-libs" / "pcg_basic" / "pcg_basic.h"]    
    res = [self.constants]
    for lib in libs:
      with open(lib, "r") as f:
        res.append(f.read())
    print(res)
    return "\n".join(res)

@dataclass
class Experiment:
  number: int
  name: str

  def get_path(self) -> Path:
    return Path(__file__).parent.parent / "experiments" / f"{self.number:02d}_{self.name}"

class SimulatorConstants(abc.ABC):
  @abc.abstractmethod
  def create_empty_results(self) -> xr.DataArray:
    pass

class Simulator:
  @staticmethod
  def generate_seed():
    return random.randint(0, 2**32 - 1)

  def __init__(self, experiment: Experiment,
      simulator_constants: SimulatorConstants, 
      python_simulator_function: Callable[[int, xr.DataArray], None], 
      c_code: str):
    self.experiment = experiment
    self.simulator_constants = simulator_constants
    self.python_simulator = python_simulator_function
    self.c_code = c_code
    if c_code:
      includes = self.c_code.generate_include()
      with open(self.experiment.get_path() / f"{self.experiment.name}.h", "w") as f:
        f.write(includes)
      ffi = FFI()     
      ffi_source = "\n".join(["#define CFFI", includes, self.c_code])
      ffi_libs = ["pcg_basic"]
      ffi_sources = [str(Path(__file__).parent / "c_libs" / f"{x}.c") 
        for x in ffi_libs]
      ffi.set_source("hammy_cpu_kernel", ffi_source, sources=ffi_sources)
      ffi.cdef("""void run(unsigned long long loops, unsigned long long* out, long long seed);""")
      ffi.compile(verbose=True)

  def run_single_simulation(self, platform: SimulatorPlatforms, loops: int, seed: int):
    out = self.simulator_constants.create_empty_results()
    match platform:
        case SimulatorPlatforms.PYTHON:
            self.python_simulator(loops, out, seed)            
        case SimulatorPlatforms.CFFI:
          import hammy_cpu
          hammy_cpu.run(loops, out, seed)                     
        case SimulatorPlatforms.CUDA:
            raise ValueError(f"CUDA platform not implemented yet")
        case _:
            raise ValueError(f"Unknown platform: {platform}")

  def run_calibration(self, platform: SimulatorPlatforms):
    loops = 1000
    while True:
        print(f"Running calibration with {loops} loops...")
        start_time = time.time()
        self.run_single_simulation(platform, loops, self.generate_seed())
        elapsed_time = time.time() - start_time
        print(f"Simulation took {elapsed_time:.2f} seconds")            
        if elapsed_time > 30:
            # Calculate loops needed for 1 minute
            one_min_loops = int(loops * 60 / elapsed_time)
            print(f"Estimated {one_min_loops} loops needed for 1 minute")            
            # Run with calculated loops
            print(f"Running verification with {one_min_loops} loops...")
            start_time = time.time()
            self.run_single_simulation(platform, one_min_loops, self.generate_seed())
            elapsed_time = time.time() - start_time
            print(f"Verification took {elapsed_time:.2f} seconds")                
            # Final adjustment
            final_loops = int(one_min_loops * 60 / elapsed_time)
            print(f"Final calibration: {final_loops} loops per minute")
            return final_loops            
        loops *= 2
