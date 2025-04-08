import abc
from enum import Enum
from dataclasses import dataclass
from pathlib import Path
import xarray as xr
from typing import Dict
from time import time
from .machine_configuration import MachineConfiguration
from typing import Optional

def generate_random_seed() -> int:
    return int(time() * 1000)

class SimulatorPlatforms(Enum):
    PYTHON = "python"
    CFFI = "cffi" 
    CUDA = "cuda"

@dataclass(frozen=True)
class Experiment:    
    name: str
    version: int    

    def to_id(self) -> str:
        return f"exp_name:{self.name}/exp_ver:{self.version}"  

CalibrationResults = Dict[SimulatorPlatforms, int]

@dataclass(frozen=True)
class CalibrationResultsCacheKey:
    experiment: Experiment
    threads: int
    use_cuda: bool
    machine_configuration: MachineConfiguration
    numpy_hash: int
    cffi_hash: int
    cuda_hash: Optional[int]
    def to_id(self) -> str:
        return f"{self.experiment.to_id()}/threads:{self.threads}/use_cuda:{self.use_cuda}/numpy_hash:{self.numpy_hash}/cffi_hash:{self.cffi_hash}/cuda_hash:{self.cuda_hash or None}"

CalibrationResultsCache = Dict[CalibrationResultsCacheKey, CalibrationResults]

@dataclass(frozen=True)
class CCode:  
    code: str
    constants: str | Path
    function_header : str = "void run_simulation(unsigned long long loops, const unsigned long long seed,  unsigned long long* out);"

    def generate_include(self) -> str:
        return "\n".join([      
            self.constants,
            self.read_common_header(),  
            "#ifdef USE_CUDA",       
            self.generate_include_for_platform(SimulatorPlatforms.CUDA),     
            "#else", 
            self.generate_include_for_platform(SimulatorPlatforms.CFFI), 
            "#endif"])

    @staticmethod
    def read_common_header() -> str:
        lib_path = Path(__file__).parent / "c_libs" / "common" / "common.h"
        with open(lib_path, "r") as f:
            return f.read()

    def generate_include_for_platform(self, platform: SimulatorPlatforms) -> str:
        lib_path = Path(__file__).parent / "c_libs"
        libs = []
        if platform == SimulatorPlatforms.CFFI:
            libs += ["pcg_basic", "cuda_cpu"]    
        res = []    
        for lib in libs:
            with open(lib_path /  lib / f"{lib}.h", "r") as f:        
                res.append(f.read())    
        return "\n".join(res)

class SimulatorConstants(abc.ABC):
    @abc.abstractmethod
    def create_empty_results(self) -> xr.DataArray:
        pass
