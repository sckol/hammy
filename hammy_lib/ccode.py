from pathlib import Path
from dataclasses import dataclass
from .simulator_platforms import SimulatorPlatforms

@dataclass(frozen=True)
class CCode:  
    code: str
    constants: str
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

