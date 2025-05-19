import hashlib
import platform
import subprocess
import psutil
from cffi import FFI
import numpy as np
from pathlib import Path
import re
from .hammy_object import DictHammyObject

class MachineConfiguration(DictHammyObject):
  def __init__(self, digest: str | None=None) -> None:
    super().__init__()    
    self.digest = digest
  
  def calculate(self) -> None:
    # Detect CPU and memory info    
    self.data['cpu_model'] = MachineConfiguration.clear_string(platform.processor())
    self.data['physical_cores'] = psutil.cpu_count(logical=False)
    self.data['logical_cores'] = psutil.cpu_count(logical=True)
    self.data['total_ram_gb'] = round(float(psutil.virtual_memory().total / (1024**3)), 1)
    # Detect OS information
    self.data['os_name'] = platform.system()
    if self.data['os_name'] == "Windows":
      self.data['os_version'] = platform.win32_ver()[1]
    elif self.data['os_name'] == "Linux":
      self.data['os_version'] = platform.release()
    else:
        raise ValueError("Unsupported OS")                
    # Detect GPU info
    self.data['gpu_model'] = None
    self.data['cuda_cores'] = None
    self.data['gpu_memory_gb'] = None
    try:
      gpu_info = subprocess.check_output(
        "nvidia-smi --query-gpu=name,memory.total,count --format=csv,noheader",
        shell=True, text=True
      ).strip().split(',')
      if gpu_info and len(gpu_info) >= 2:
        self.data['gpu_model'] = gpu_info[0].strip()
        self.data['gpu_memory_gb'] = round(float(gpu_info[1].strip().split()[0]) / 1024, 1)  # Convert MiB to GB
        self.data['cuda_cores'] = int(gpu_info[2].strip()) if len(gpu_info) > 2 else None
    except Exception:
      pass
    # Detect CUDA version
    self.data['cuda_version'] = None
    try:
      cuda_info = subprocess.check_output("nvcc --version", shell=True, text=True)
      self.data['cuda_version'] = cuda_info.split("release")[-1].split(",")[0].strip() \
                if "release" in cuda_info else None
    except Exception:
      pass
    # Detect Python version
    self.data['python_version'] = platform.python_version()
    self.data['numpy_version'] = np.__version__
    self.data['ccompiler'] = MachineConfiguration.get_compiler(self.data['os_name'])
    if self.digest is None:
      self.digest = self.get_digest()
    self.data['digest'] = self.digest        

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
      
  def get_digest(self) -> str:
    return hex(abs(int(hashlib.sha256("/".join([str(v) for k, v in self.data.items() if k != 'digest']).encode()).hexdigest(), 16)))[2:].zfill(6)[:6]
  
  def get_id(self) -> str:
    return f"{self.digest}_machine_configuration"

  def validate_loaded_object(self):
    new_digest = self.get_digest()
    if self.digest != new_digest:
      raise ValueError(f"Machine configuration digest mismatch: {self.digest} != {new_digest}")
      
  def get_foldername(self):
    return ""