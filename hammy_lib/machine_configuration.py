import platform
import subprocess
import psutil
from cffi import FFI
import numpy as np
from pathlib import Path
from .hammy_object import DictHammyObject


class MachineConfiguration(DictHammyObject):
    def calculate(self) -> None:
        # Detect CPU and memory info
        self.metadata["cpu_model"] = platform.processor()
        self.metadata["physical_cores"] = psutil.cpu_count(logical=False)
        self.metadata["logical_cores"] = psutil.cpu_count(logical=True)
        self.metadata["total_ram_gb"] = round(
            float(psutil.virtual_memory().total / (1024**3)), 1
        )
        # Detect OS information
        self.metadata["os_name"] = platform.system()
        if self.metadata["os_name"] == "Windows":
            self.metadata["os_version"] = platform.win32_ver()[1]
        elif self.metadata["os_name"] == "Linux":
            self.metadata["os_version"] = platform.release()
        else:
            raise ValueError("Unsupported OS")
        # Detect GPU info
        self.metadata["gpu_model"] = None
        self.metadata["cuda_cores"] = None
        self.metadata["gpu_memory_gb"] = None
        try:
            gpu_info = (
                subprocess.check_output(
                    "nvidia-smi --query-gpu=name,memory.total,count --format=csv,noheader",
                    shell=True,
                    text=True,
                )
                .strip()
                .split(",")
            )
            if gpu_info and len(gpu_info) >= 2:
                self.metadata["gpu_model"] = gpu_info[0].strip()
                self.metadata["gpu_memory_gb"] = round(
                    float(gpu_info[1].strip().split()[0]) / 1024, 1
                )  # Convert MiB to GB
                self.metadata["cuda_cores"] = (
                    int(gpu_info[2].strip()) if len(gpu_info) > 2 else None
                )
        except Exception:
            pass
        # Detect CUDA version
        self.metadata["cuda_version"] = None
        try:
            cuda_info = subprocess.check_output("nvcc --version", shell=True, text=True)
            self.metadata["cuda_version"] = (
                cuda_info.split("release")[-1].split(",")[0].strip()
                if "release" in cuda_info
                else None
            )
        except Exception:
            pass
        # Detect Python version
        self.metadata["python_version"] = platform.python_version()
        self.metadata["numpy_version"] = np.__version__
        self.metadata["ccompiler"] = MachineConfiguration.get_compiler(
            self.metadata["os_name"]
        )

    @staticmethod
    def get_compiler(os_name) -> str:
        ffibuilder = FFI()
        # Define your C declarations
        ffibuilder.cdef("""
        double dummy();
    """)
        # Specify source code and other parameters
        ffibuilder.set_source(
            "_example",
            """ 
        double dummy() { 
            1 // intentional error           
        }     
    """,
            sources=[],
        )
        # Compile with verbose output to see the command
        output_dir = Path().parent / "build_cffi"
        try:
            ffibuilder.compile(tmpdir=str(output_dir))
        except Exception as e:
            compiler_path = (
                str(e).split("command ")[1].split("failed")[0].strip().strip("'")
            )
            compiler_version = subprocess.run(
                [compiler_path] + ([] if os_name == "Windows" else ["--version"]),
                stderr=subprocess.STDOUT,
                stdout=subprocess.PIPE,
                text=True,
                check=True,
            )
            print(compiler_version.stdout)
            return compiler_version.stdout.split("\n")[0].strip()

    @property
    def id(self) -> str:
        return f"{self.digest}_machine_configuration"
