# Technology Stack

**Analysis Date:** 2026-03-24

## Languages

**Primary:**
- Python 3.12 - Monte Carlo simulation orchestration, experiment control, post-processing, visualization
- C - CPU simulation kernels, compiled via CFFI or standalone CMake

**Secondary:**
- CUDA - GPU simulation kernels (optional, auto-detected)
- Bash - VM provisioning and Docker entrypoint scripts

## Runtime

**Environment:**
- Python 3.12 (standard CPython)
- CFFI for C interop - compiles C simulation code at runtime into shared libraries
- CMake 3.0.0+ - builds standalone experiment executables

**Package Manager:**
- pip (via Python venv)
- hatchling - build backend for `pyproject.toml`

## Frameworks

**Core Simulation:**
- cffi - Foreign Function Interface for Python↔C calling
- cupy - GPU computing (preinstalled in Docker base image via `cupy/cupy:v13.1.0`)

**Data Processing:**
- xarray - Multi-dimensional array manipulation (simulation results with dimensions: level, platform, target, checkpoint, position)
- h5netcdf - NetCDF4 storage engine for xarray (used for all `.nc` result files)
- numpy - Numerical arrays, CPU simulation reference implementation

**Analysis:**
- scipy - Scientific computing utilities
- scs - Convex optimization solver for NNLS (nonnegative least squares) in position decomposition

**Visualization:**
- matplotlib - Plotting library
- Pillow (PIL) - Image handling for PNG embedding

**Build/Dev:**
- hatchling - Python package building
- psutil - System hardware detection (CPU cores, RAM, GPU detection)
- cffi - Runtime compilation of C kernels

## Key Dependencies

**Critical:**
- xarray [version not pinned] - Central data structure for all simulation results and calculations; missing leads to serialization/deserialization failures
- cffi [version not pinned] - Runtime compilation of C simulation kernel; required for CPU path
- boto3 [version not pinned] - AWS S3 API for Yandex Cloud Object Storage uploads/downloads
- h5netcdf [version not pinned] - NetCDF storage layer; must support xarray's `to_netcdf(..., engine='h5netcdf')`

**Infrastructure:**
- psutil [version not pinned] - CPU/GPU hardware detection; informs thread pool sizing and CUDA availability
- matplotlib [version not pinned] - Required for all visualization outputs (PNG grids)
- scipy [version not pinned] - NNLS solver in position calculations
- scs [version not pinned] - Convex optimization for NNLS (alternative solver path)

**GPU Optional:**
- cupy [version not pinned] - GPU array operations when CUDA is available; base Docker image includes `cupy/cupy:v13.1.0`

## Configuration

**Environment:**
- Experiment configuration: class attributes on `Experiment` subclasses (e.g., `WalkExperiment.T`, `WalkExperiment.TARGETS`)
- Machine detection: runtime auto-detection via `MachineConfiguration` (CPU cores, GPU model, CUDA version)
- Results directory: configurable via `HammyObject.RESULTS_DIR` class variable (default: `Path("results")`)
- S3 credentials: Yandex Cloud KMS-encrypted `.cipher` files, decrypted at container runtime via `yc kms symmetric-crypto decrypt`

**Build:**
- CMakeLists.txt - Detects CUDA compiler; auto-discovers experiments matching `NN_name/` pattern
- pyproject.toml - Declares dependencies, build backend (hatchling), package entry point
- pyrightconfig.json - Type checking relaxed (most type rules disabled) for experiment notebook compatibility

**Python venv:**
- Location: `.venv/` at repo root
- Installation: `python3 -m venv .venv && .venv/bin/pip install -e .`
- Activation: `.venv/bin/python` or `.venv/bin/pip`

## Platform Requirements

**Development:**
- Linux or WSL2 Ubuntu (primary dev environment: WSL2)
- CMake 3.0.0+ for standalone C builds
- Python 3.12
- Optional: NVIDIA GPU + CUDA toolkit (auto-detected; not required for CPU simulation)
- Optional: GCC or Clang (auto-detected for C compilation)

**Production (Yandex Cloud):**
- Preemptible VM spec:
  - 8 vCPU (compute-optimized)
  - 1 NVIDIA GPU (gpu-standard-v2 platform)
  - 48 GB RAM
  - Zone: ru-central1-a
  - Service account with KMS decrypt permission for `hammy` key
- Docker runtime with container support
- Base image: `cupy/cupy:v13.1.0` (includes CUDA 12.x, cupy, Python 3.x)

**Google Colab:**
- Free tier GPU: Tesla T4 (16 GB VRAM)
- Python runtime 3.10+
- cupy pre-installed
- Inline execution via notebook cells (set `CCODE` variable for C source)

## Key Tools

**Local Development:**
- `.venv/bin/python` - Python interpreter
- `cmake` - Build system for standalone C executables
- `uv` - Optional offline pip installation (compatible with project's `--offline` flag)

**Docker/Cloud:**
- `yc` - Yandex Cloud CLI for VM provisioning, KMS decryption, logging, instance cleanup
- `docker` - Container runtime (via `docker compose`)
- `unified_agent` - Yandex Cloud Unified Agent (system metrics streaming to monitoring)

**RNG:**
- PCG32 (`c_libs/pcg_basic/pcg_basic.{h,c}`) - Pseudorandom number generator for CPU path
- CUDA curand (`curand_kernel.h`) - GPU RNG (CUDA path only)

---

*Stack analysis: 2026-03-24*
