import platform
import subprocess
from dataclasses import dataclass
from typing import Optional


@dataclass
class MachineConfiguration:
    """System hardware and software configuration"""
    cpu_model: str
    os_name: str
    os_version: str
    gpu_model: Optional[str]
    cuda_version: Optional[str]

    @staticmethod
    def detect() -> "MachineConfiguration":
        # Detect CPU model
        cpu_model = platform.processor()

        # Detect OS information
        os_name = platform.system()
        if os_name == "Windows":
            os_version = platform.win32_ver()[1]
        elif os_name == "Linux":
            os_version = platform.release()
        elif os_name == "Darwin":
            os_version = platform.mac_ver()[0]
        else:
            os_version = platform.version()

        # Detect GPU model
        try:
            gpu_info = subprocess.check_output("nvidia-smi --query-gpu=name --format=csv,noheader", 
                                            shell=True, text=True)
            gpu_model = gpu_info.strip().split("\n")[0] if gpu_info else None
        except Exception:
            gpu_model = None

        # Detect CUDA version
        try:
            cuda_version = subprocess.check_output("nvcc --version", 
                                                shell=True, text=True)
            cuda_version = cuda_version.split("release")[-1].split(",")[0].strip() \
                         if "release" in cuda_version else None
        except Exception:
            cuda_version = None

        return MachineConfiguration(
            cpu_model=cpu_model,
            os_name=os_name,
            os_version=os_version,
            gpu_model=gpu_model,
            cuda_version=cuda_version,
        )

    def __str__(self) -> str:
        """Pretty print the configuration"""
        gpu_info = f"GPU: {self.gpu_model} (CUDA {self.cuda_version})" \
                   if self.gpu_model and self.cuda_version else "GPU: Not available"
        
        return (f"System Configuration:\n"
                f"  OS: {self.os_name} {self.os_version}\n"
                f"  CPU: {self.cpu_model}\n"
                f"  {gpu_info}")