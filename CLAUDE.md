# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**hammy** is a Python+C simulation library for running Monte Carlo random walk experiments across CPU (NumPy/CFFI) and GPU (CUDA) platforms. It manages the full pipeline: machine detection, calibration, parallel simulation execution, statistical calculations on results, and visualization.

## Build & Run

### Python package (hammy_lib)
```bash
pip install -e .                    # Install in editable mode (uses hatchling)
```

### C experiments (CMake)
```bash
cmake -B build && cmake --build build   # Builds experiment executables from experiments/*/
```

### CFFI compilation (per-experiment)
Experiments compile C code to shared libraries via CFFI at runtime:
```bash
python -c "from experiments.01_walk.walk import WalkExperiment; WalkExperiment().compile()"
# Output goes to build_cffi/
```

### Running an experiment
```bash
cd experiments/01_walk && python walk.py
```

### Cloud execution
```bash
./create_hammy_machine.sh <version> <minutes>   # Creates Yandex Cloud GPU VM with Docker
```

## Architecture

### Dual-language simulation kernel

Each experiment has **three equivalent implementations** of the same simulation:
1. **Python/NumPy** — reference implementation in the experiment's `simulate_using_python()`
2. **C via CFFI** — compiled at runtime, loaded as shared library via `cffi.dlopen()`
3. **CUDA** — same C code compiled for GPU (not yet fully implemented)

The C code uses a compatibility layer (`c_libs/cuda_cpu/cuda_cpu.h`) that maps CUDA constructs to CPU equivalents, allowing a single `.c` source to compile for both CPU and GPU.

### CPU/GPU compatibility macros (`c_libs/cuda_cpu/cuda_cpu.h`)

The macro layer emulates CUDA's threading model on CPU by simulating a single 32-thread warp as a sequential loop:

| CUDA concept | CPU macro/expansion | Purpose |
|---|---|---|
| `__shared__`, `__global__`, `__device__`, `__EXTERN` | empty (no-op) | Storage/linkage qualifiers |
| `_32` | `[32]` | Declares arrays of 32 (one per virtual thread) |
| `_` | `[threadIdx.x]` | Indexes per-thread arrays by current virtual thread |
| `__WARP_INIT` | `for(threadIdx.x = 0..31) {` | Starts a warp-simulation loop over 32 virtual threads |
| `__SYNCTHREADS` | `} __WARP_INIT` | Ends one warp pass, starts another (barrier emulation) |
| `__WARP_END` | `}` | Ends the warp loop |
| `TID` | `threadIdx.x + blockIdx.x * blockDim.x` | Global thread ID (CPU: just threadIdx.x since BLOCKS=1) |
| `TID_LOCAL` | `threadIdx.x` | Thread-local ID within block |
| `atomicAdd(addr, val)` | `*addr += val` | Safe on CPU (sequential within warp loop) |
| `ZERO(arr, type, aligned)` | Parallel zero-fill macro | Zeroes array using warp-parallel pattern |

**RNG mapping**: `curandStateXORWOW_t` → `pcg32_random_t` (PCG32 from `c_libs/pcg_basic/`). Initialized via `curand_init(seed, sequence, offset, &state)` which maps to `pcg32_srandom_r(state, offset, (seed_low32 << 32) | seq_low32)`. The sequence parameter (typically `TID`) selects an independent PCG stream.

### Seed and RNG architecture

Seeds flow: `ExperimentConfiguration.seed` → `run_parallel_simulations(seed + thread_id)` → C kernel `curand_init(seed, threadIdx.x, 0, &state)`.

Three levels of RNG independence:
1. **Between parallel Python threads**: `seed + thread_id` (0 = Python/NumPy, 1..N-1 = CFFI, N = CUDA)
2. **Between virtual threads in C kernel**: `threadIdx.x` (0..31) as PCG sequence parameter → different LCG increment
3. **Between simulation levels**: `ExperimentConfiguration.seed` increments by `self.threads` after each `run_parallel_simulations()` call

### HammyObject hierarchy (`hammy_lib/hammy_object.py`)

All persistent objects inherit from `HammyObject`, which provides:
- **Automatic metadata collection** from all fields (including nested HammyObjects)
- **Content-addressed IDs** — each object's ID is derived from its metadata, enabling caching
- **Load/save with validation** — metadata is checked on load to detect stale cache
- **S3 sync** via `YandexCloudStorage` — auto-downloads missing files from Yandex Object Storage

Two concrete base classes:
- `DictHammyObject` — stores metadata as JSON (for configs, calibrations)
- `ArrayHammyObject` — stores xarray DataArrays as NetCDF with h5netcdf engine (for simulation results, calculations)

### Experiment execution pipeline

The pipeline objects form a dependency chain, each resolving its dependencies recursively:

```
Experiment → ExperimentConfiguration → SequentialCalibration → ParallelCalibration → Simulation → Calculation → Visualization
```

1. **Experiment** — defines simulation parameters, C code, and numpy equivalent
2. **MachineConfiguration** — auto-detects CPU/GPU/compiler/memory
3. **ExperimentConfiguration** — binds experiment to machine, manages multiprocessing Pool
4. **SequentialCalibration** — measures loops/minute for each platform (doubling until ≥15s, then verifying)
5. **ParallelCalibration** — validates that parallel execution matches sequential timing (within tolerance)
6. **Simulation** — runs parallel simulations at exponential time levels (1, 1, 2, 4, 8... minutes); results accumulate as xarray with `level` and `platform` dimensions
7. **Calculation** — post-processing on simulation results (iterates over dimension combinations)
8. **Visualization** — grid plots with filtering, comparison overlays, and groupby

### Calculations framework (`hammy_lib/calculation.py`)

`Calculation` iterates over all coordinate combinations of `independent_dimensions` (always includes `level` and `platform`), calls `calculate_unit()` for each slice, then combines results via `xr.combine_by_coords`. The `extend_simulation_results()` method adds cumulative sums across levels and a `TOTAL` platform.

Concrete calculations: `ArgMaxCalculation`, `PositionCalculation` (uses NNLS + eigendecomposition on graph), `PopulationSizeCalculation` (G-test overdispersion estimate).

### Error handling philosophy

Never hide errors via clamping, silent returns, or default values. If a statistic can be > 1 or < 0, show the actual value — anomalous results are often the first signal of a real bug (seed duplication, model mismatch, data corruption). Guard clauses should `raise ValueError` with context, not `return 0.0` or `return float("nan")`.

### Adding a new experiment

1. Create `experiments/NN_name/` with `name.py`, `name.c`, and optionally `name.h`
2. Subclass `Experiment` and define: `experiment_number`, `experiment_name`, `experiment_version`, `c_code` (a `CCode` instance), `create_empty_results()`, `simulate_using_python()`
3. CMakeLists.txt auto-discovers experiment directories matching `NN_name` pattern

### Key conventions

- Simulation results are xarray DataArrays with dimensions like `target`, `checkpoint`, `position_index`, `level`, `platform`
- Results are stored in `results/<experiment_string>/` as `.json` (metadata) and `.nc` (data)
- The `resolve()` method handles the full lifecycle: resolve dependencies → check cache → compute if needed
- C simulation function signature: `void run_simulation(unsigned long long loops, const unsigned long long seed, unsigned long long* out)`
- **C position convention**: the C kernel stores positions as `raw_position // 2` (integer division). Bin `x` collects raw positions `2x` and `2x+1`. Any theoretical distribution must account for this (sum probabilities over both raw positions per bin).

### xarray string coordinates

Never use `pd.Index` for string dimension coordinates in `xr.concat`. Pandas 2.x converts strings to `StringDtype`, which becomes numpy `StringDType(na_object=nan)` — unsupported by `np.result_type()` and h5netcdf/NetCDF4.

Use `xr.DataArray([...], dims='dim_name')` instead — produces `<U` dtype that works everywhere.
