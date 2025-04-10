import platform
import subprocess
import psutil
from dataclasses import dataclass
from typing import Optional
from cffi import FFI
import numpy as np
from pathlib import Path
import re

@dataclass(frozen=True)
class MachineConfiguration:
    """System hardware and software configuration"""
    cpu_model: str
    physical_cores: int
    logical_cores: int
    total_ram_gb: float
    gpu_model: Optional[str]    
    cuda_cores: Optional[int]
    gpu_memory_gb: Optional[float]
    os_name: str
    os_version: str
    python_version: str
    numpy_version: str
    ccompiler: str
    cuda_version: Optional[str]

    @staticmethod
    def detect() -> "MachineConfiguration":
        # Detect CPU and memory info
        cpu_model = MachineConfiguration.clear_string(platform.processor())
        physical_cores = psutil.cpu_count(logical=False)
        logical_cores = psutil.cpu_count(logical=True)
        total_ram_gb = psutil.virtual_memory().total / (1024**3)

        # Detect OS information
        os_name = platform.system()
        if os_name == "Windows":
            os_version = platform.win32_ver()[1]
        elif os_name == "Linux":
            os_version = platform.release()
        else:
            raise ValueError("Unsupported OS")        
        
        # Detect GPU info
        gpu_model = None
        cuda_cores = None
        gpu_memory_gb = None
        try:
            gpu_info = subprocess.check_output(
                "nvidia-smi --query-gpu=name,memory.total,count --format=csv,noheader",
                shell=True, text=True
            ).strip().split(',')
            if gpu_info and len(gpu_info) >= 2:
                gpu_model = gpu_info[0].strip()
                gpu_memory_gb = float(gpu_info[1].strip().split()[0]) / 1024  # Convert MiB to GB
                # CUDA cores requires additional querying based on GPU model
                # This is a simplified example, actual implementation would need GPU architecture mapping
        except Exception:
            pass

        # Detect CUDA version
        cuda_version = None
        try:
            cuda_info = subprocess.check_output("nvcc --version", shell=True, text=True)
            cuda_version = cuda_info.split("release")[-1].split(",")[0].strip() \
                         if "release" in cuda_info else None
        except Exception:
            pass

        # Detect Python version
        python_version = platform.python_version()

        numpy_version = np.__version__
        ccompiler = MachineConfiguration.get_compiler(os_name)

        return MachineConfiguration(
            cpu_model=cpu_model,
            physical_cores=physical_cores,
            logical_cores=logical_cores,
            total_ram_gb=total_ram_gb,
            os_name=os_name,
            os_version=os_version,
            gpu_model=gpu_model,
            cuda_version=cuda_version,
            cuda_cores=cuda_cores,
            gpu_memory_gb=gpu_memory_gb,
            python_version=python_version,
            numpy_version=numpy_version,
            ccompiler=ccompiler
        )

    @staticmethod
    def get_compiler(os_name) -> str:
        ffibuilder = FFI()
        # Define your C declarations
        ffibuilder.cdef("""
            double dummy();
        """)
        # Specify source code and other parameters
        ffibuilder.set_source("_example",
        """ 
            double dummy() { 
                1 // intentional error           
            }     
        """,
        sources=[])
        # Compile with verbose output to see the command
        output_dir = Path().parent / "build_cffi"
        try:
            ffibuilder.compile(tmpdir=str(output_dir))   
        except Exception as e:                     
            compiler_path = str(e).split("command ")[1].split("failed")[0].strip().strip("'")               
            compiler_version = subprocess.run(
                [compiler_path] + ([] if os_name == "Windows" else ["--version"]), stderr=subprocess.STDOUT, stdout=subprocess.PIPE, text=True, check=True)                
            print(compiler_version.stdout)                             
            return MachineConfiguration.clear_string(compiler_version.stdout.split("\n")[0].strip())   
            
    @staticmethod
    def clear_string(s: str) -> str:        
        return re.sub(r'[^a-zA-Z0-9.]', '_', s.strip())

    def to_id(self) -> str:
        return "/".join([
            f"cpu_model:{self.cpu_model}",
            f"physical_cores:{self.physical_cores}",
            f"logical_cores:{self.logical_cores}",
            f"total_ram_gb:{self.total_ram_gb:.1f}",
            f"os_name:{self.os_name}",
            f"os_version:{self.os_version}",
            f"gpu_model:{self.gpu_model or 'None'}",
            f"gpu_memory_gb:{(self.gpu_memory_gb or 0):.1f}",
            f"cuda_cores:{self.cuda_cores or 0}",
            f"python_version:{self.python_version}",
            f"numpy_version:{self.numpy_version}",
            f"ccompiler:{self.ccompiler}",
            f"cuda_version:{self.cuda_version or 'None'}"
        ])

    def __str__(self) -> str:
        """Pretty print the configuration"""
        gpu_info = (f"GPU: {self.gpu_model}\n"
                   f"  CUDA: {self.cuda_version}\n"
                   f"  Memory: {self.gpu_memory_gb:.1f} GB\n"
                   f"  CUDA Cores: {self.cuda_cores}") \
                   if self.gpu_model else "GPU: Not available"
        
        return (f"System Configuration:\n"
                f"  OS: {self.os_name} {self.os_version}\n"
                f"  CPU: {self.cpu_model}\n"
                f"    Physical cores: {self.physical_cores}\n"
                f"    Logical cores: {self.logical_cores}\n"
                f"    RAM: {self.total_ram_gb:.1f} GB\n"
                f"  {gpu_info}\n"
                f"  Python: {self.python_version}\n"
                f"  NumPy: {self.numpy_version}\n"
                f"  Compiler: {self.ccompiler}\n")
    
