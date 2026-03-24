# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**hammy** is a Python+C simulation library for running Monte Carlo random walk experiments across CPU (NumPy/CFFI) and GPU (CUDA) platforms. It manages the full pipeline: machine detection, calibration, parallel simulation execution, statistical calculations on results, and visualization.

## Build & Run

### Python environment
```bash
# Venv is at repo root
python3 -m venv .venv
.venv/bin/pip install -e .
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

### Execution modes

Experiments support 6 execution modes:

| Mode | Command | Notes |
|---|---|---|
| **Local file** | `cd experiments/01_walk && python walk.py` | Full pipeline, results in project root |
| **Local module** | `python -m experiments.01_walk --level 2 --no-viz` | Selective steps via CLI flags |
| **Yandex Cloud** | `./create_hammy_machine.sh <version> <minutes>` | Docker container, same CLI flags |
| **Google Colab** | Open walk.py as notebook | Set `CCODE` variable for C source, one cell per viz |
| **Debug C** | `cmake -B build && cmake --build build && ./build/walk` | Standalone C executable via walk.h |
| **Debug CUDA** | Compile .cu on GPU server | Uses `to_cuda_source()` output |

**CLI flags** (modes 2-3):
- `--level N` — simulation level (default 4)
- `--no-viz` — skip visualizations
- `--no-upload` — skip S3 upload
- `--no-calculations` — skip calculations and viz (just simulate)
- `--dry-run` — fast calibration (10% loops, relaxed tolerance)

**Colab specifics**: set `CCODE = "..."` in the CCODE cell to provide C source inline (otherwise reads from walk.c file). Each visualization has its own cell for inline display. S3 credentials via Colab Secrets (`access_key`, `secret_key`).

**Single-core machines** (e.g. Colab): `ExperimentConfiguration` auto-detects cores and ensures CFFI gets at least 1 thread. PYTHON and CFFI time-share the CPU core while CUDA runs async on GPU. Use `calibration_tolerance=100` for `ParallelCalibration` (exposed via `experiment_configuration.cores`).

Docker files are in `docker/`: `hammy-base.Dockerfile` (base deps), `hammy.Dockerfile` (experiment runner), `hammy_entrypoint.sh` (entrypoint), `hammy_machine.compose` (Compose config).

## Architecture

### Dual-language simulation kernel

Each experiment has **three equivalent implementations** of the same simulation:
1. **Python/NumPy** — reference implementation in the experiment's `simulate_using_python()`
2. **C via CFFI** — compiled at runtime, loaded as shared library via `cffi.dlopen()`
3. **CUDA** — same C code compiled for GPU (not yet fully implemented)

The C code uses a compatibility layer (`c_libs/cuda_cpu/cuda_cpu.h`) that maps CUDA constructs to CPU equivalents, allowing a single `.c` source to compile for both CPU and GPU.

### CCode dataclass (`hammy_lib/ccode.py`)

`CCode(code, constants, function_header?)` is a frozen dataclass that bundles a C kernel:
- `code` — C source string (the experiment's `.c` file content)
- `constants` — C preprocessor `#define` and variable declarations (experiment-specific sizing)
- `function_header` — CFFI signature (default: `void run_simulation(unsigned long long loops, const unsigned long long seed, unsigned long long* out)`)

Two compilation paths:
- `str(c_code)` — CFFI source: constants → `common.h` (BLOCKS=1) → pcg_basic + cuda_cpu headers → experiment code
- `c_code.to_cuda_source(blocks)` — CUDA source for `cp.RawKernel`: constants → USE_CUDA + BLOCKS define → `common.h` (curand_kernel.h) → CUDA-native macros → experiment code

### How the dual CPU/GPU compilation works

CUDA's programming model uses **warps** — groups of 32 threads executing in lockstep on a GPU. Each warp runs inside a **block**, and a kernel launches a **grid** of many blocks. Threads within a warp share memory and synchronize at barriers (`__syncthreads()`).

The trick: `cuda_cpu.h` emulates one warp as a sequential `for` loop over 32 iterations on CPU. The experiment C code is written once using macros that expand differently per platform:

| Concept | CUDA (real GPU) | CPU emulation (`cuda_cpu.h`) |
|---|---|---|
| `__shared__ int x` | Per-block shared memory | Regular local variable |
| `__device__ int x` | Global GPU memory | Regular global variable |
| `int val _32` | `int val` (per-thread register) | `int val[32]` (array, one slot per virtual thread) |
| `val _` | `val` (thread's own value) | `val[threadIdx.x]` (indexed by loop counter) |
| `__WARP_INIT` | no-op (real hardware warp) | `for(threadIdx.x = 0; threadIdx.x < 32; threadIdx.x++) {` |
| `__SYNCTHREADS` | `__syncthreads();` (barrier) | `} for(threadIdx.x = 0; ...) {` (restart loop) |
| `__WARP_END` | no-op | `}` (close loop) |
| `curand_init` / `curand` | CUDA curand library | PCG32 (`pcg_basic.h`) |
| `atomicAdd(addr, val)` | Hardware atomic operation | `*addr += val` (safe — sequential in loop) |
| `BLOCKS` | Grid size (many, e.g. 1000) | Always 1 |

**On CPU:** one block, 32 virtual threads executed sequentially in a loop. `_32` creates arrays of 32 elements, `_` indexes them by the loop counter. Barriers (`__SYNCTHREADS`) end the loop and restart it, ensuring all 32 "threads" complete before the next phase.

**On GPU:** many blocks, each with 32 real threads (one warp). `_32` and `_` expand to nothing — each thread has its own registers. `__SYNCTHREADS` becomes a real barrier. `BLOCKS` controls how many blocks run in parallel.

### Memory: `__device__` vs `__shared__`

The `counts` array in walk.c uses `__device__` (global GPU memory), indexed by `blockIdx.x` so each block writes to its own slice. Block 0 reduces all slices at the end.

Why not `__shared__` (per-block, faster)? Shared memory is limited to 48-96 KB per SM. The per-block counts array for the walk experiment is `32 * TARGETS_LEN * CHECKPOINTS_LEN * BINS_LEN * 8 bytes` = **1.26 MB** — far exceeds the limit. `__device__` memory is the only option, with BLOCKS sized to fit GPU RAM (BLOCKS=1000 uses ~1.3 GB).

### RNG mapping

`curandStateXORWOW_t` → `pcg32_random_t` (PCG32 from `c_libs/pcg_basic/`). Initialized via `curand_init(seed, sequence, offset, &state)` which maps to `pcg32_srandom_r(state, offset, (seed_low32 << 32) | seq_low32)`. The sequence parameter (typically `TID`) selects an independent PCG stream.

On GPU, the real curand library provides `curandStateXORWOW_t` natively via `<curand_kernel.h>`.

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

**`HammyObject.RESULTS_DIR`** (class variable, default `Path("results")`) controls where files are stored. Override at the top of an experiment script to redirect to a shared results directory:
```python
HammyObject.RESULTS_DIR = Path(__file__).parent.parent.parent / "results"
```

**`_not_checked_fields`** (class variable, list of strings) lists field names excluded from metadata conflict checks in `fill_metadata()`. Use this in subclasses for fields that legitimately differ between objects sharing the same cache (e.g., `simulation_level`, `previous_level_simulation`).

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

`FlexDimensionCalculation(main_input, dimensions)` is a variant where `independent_dimensions` is automatically derived as all dimensions except the ones listed in `dimensions`.

Concrete calculations: `PositionCalculation` (see below), `PopulationSizeCalculation` (G-test overdispersion estimate).

### Graph types (`hammy_lib/graph.py`)

`Graph` is an `ArrayHammyObject` wrapping a transition matrix with precomputed eigenvalues/eigenvectors. Subclass `Graph` and build the transition matrix in `__init__`.

`LinearGraph(length)` — 1D lazy walk chain: P(stay)=0.5, P(±1)=0.25, boundary self-loop=0.75. Matches the effective bin dynamics of walk.c (which stores `raw_position // 2`).

### PositionCalculation (`hammy_lib/calculations/position.py`)

Decomposes observed distribution as a sparse NNLS mixture of walk columns `T^p[:,i]`:
1. Normalize distribution to probability vector
2. Binary search for walk power `p`: find the smallest `p` where NNLS gives `≤ MAX_COMPONENTS` components (the minimum diffusion time to explain the distribution with few sources)
3. Compute `T^p` via spectral decomposition, solve NNLS
4. Threshold at 1% of peak weight, normalize

Output: top `MAX_COMPONENTS=2` by weight (index + value pairs), plus `power` and `nonzero_count`. The binary search guarantees `nonzero_count ≤ MAX_COMPONENTS`. The algorithm is topology-agnostic — it uses only the transition matrix, no node coordinates.

**Scope**: validated on 1D lattice (linear graph). Higher-dimensional graphs expected to need larger `MAX_COMPONENTS` (≤2×dimensionality) but untested.

### Visualization (`hammy_lib/vizualization.py`)

`Vizualization(results_object, x, y, axis, filter, groupby, comparison, reference, y_axis_label)` produces a grid of line plots (one subplot per `(x_val, y_val)` pair), saved as PNG with JSON metadata embedded in the `hammy` PNG chunk. Key parameters:
- `x`, `y` — dimensions defining the subplot grid columns/rows
- `axis` — dimension plotted on the x-axis within each subplot
- `filter` — `{dim: value}` selections applied before plotting
- `groupby` — additional dimension to split into separate lines within each subplot
- `comparison` — additional filter for an overlay line (rendered in gray)
- `reference` — callable `(DataArray) -> array` producing a black reference line

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
