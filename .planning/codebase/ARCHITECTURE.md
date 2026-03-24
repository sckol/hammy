# Architecture

**Analysis Date:** 2026-03-24

## Pattern Overview

**Overall:** Dependency-chain orchestration of Monte Carlo simulation layers

**Key Characteristics:**
- Single dependency chain: Experiment → Configuration → Calibration → Simulation → Calculation → Visualization
- Each stage resolves dependencies recursively before computing
- Dual-language kernel: C code compiles to three platforms (Python/NumPy, CFFI/CPU, CUDA/GPU) from single source
- Content-addressed caching: all objects generate IDs from metadata, enabling automatic cache validation
- xarray-centric: all results are multidimensional labeled arrays (NetCDF on disk)

## Layers

**Experiment Layer:**
- Purpose: Define simulation parameters, C code, and reference NumPy implementation
- Location: `hammy_lib/experiment.py`, `experiments/01_walk/`
- Contains: Experiment subclasses with simulation logic
- Depends on: CCode (C kernel wrapper), SimulatorPlatforms (enum)
- Used by: ExperimentConfiguration

**Machine Configuration Layer:**
- Purpose: Auto-detect CPU cores, GPU memory, CUDA version, compiler, OS metadata
- Location: `hammy_lib/machine_configuration.py`
- Contains: MachineConfiguration (DictHammyObject), compiler detection via failed CFFI compile
- Depends on: psutil, subprocess, platform, FFI
- Used by: ExperimentConfiguration

**Orchestration Layer (ExperimentConfiguration):**
- Purpose: Bind experiment to machine, manage multiprocessing.Pool, dispatch simulations
- Location: `hammy_lib/experiment_configuration.py`
- Contains: ExperimentConfiguration (DictHammyObject), run_parallel_simulations() dispatching
- Depends on: Experiment, MachineConfiguration, SimulatorPlatforms, multiprocessing.Pool
- Used by: SequentialCalibration, ParallelCalibration, Simulation

**Calibration Layer:**
- Purpose: Measure performance (loops/minute) and validate parallelism
- Location: `hammy_lib/sequential_calibration.py`, `hammy_lib/parallel_calibration.py`
- Contains: SequentialCalibration (single-threaded perf per platform), ParallelCalibration (multi-threaded validation)
- Depends on: ExperimentConfiguration, CalibrationResults
- Used by: Simulation

**Simulation Layer:**
- Purpose: Run experiment at exponentially increasing time levels, accumulate results across levels
- Location: `hammy_lib/simulation.py`
- Contains: Simulation (ArrayHammyObject), recursive level resolution, xarray concatenation
- Depends on: ParallelCalibration, CalibrationResults
- Used by: Calculation, Visualization

**Calculation Layer:**
- Purpose: Post-process simulation results via iterable unit operations
- Location: `hammy_lib/calculation.py`, `hammy_lib/calculations/`
- Contains: Calculation (base, abstract), FlexDimensionCalculation (auto-derive iteration dims), concrete subclasses (PositionCalculation, PopulationSizeCalculation)
- Depends on: Simulation (or other ArrayHammyObject), xarray
- Used by: Visualization

**Visualization Layer:**
- Purpose: Render multidimensional data as grid of filtered/grouped line plots
- Location: `hammy_lib/vizualization.py`
- Contains: Vizualization (HammyObject), matplotlib rendering, PNG with embedded JSON metadata
- Depends on: Calculation (or other ArrayHammyObject), matplotlib, PIL
- Used by: end users (PNG output)

## Data Flow

**Full Pipeline Execution:**

1. Define Experiment subclass (`experiments/01_walk/walk.py` → WalkExperiment)
2. Create Experiment instance and dump to cache (JSON metadata, no compute yet)
3. Create MachineConfiguration, auto-detect system, dump to cache
4. Create ExperimentConfiguration binding experiment + machine (allocates multiprocessing.Pool)
5. Create SequentialCalibration, resolve() triggers calculate():
   - For each platform (PYTHON, CFFI, CUDA): run single simulation with increasing loop counts
   - Measure time, solve for loops/minute
   - Store in JSON metadata
6. Create ParallelCalibration, resolve() triggers calculate():
   - Run parallel simulation at sequential calibration's loop counts
   - Validate results within calibration_tolerance (25% default, 100% on single-core)
   - Store in JSON metadata
7. Create Simulation(level=N), resolve() triggers calculate():
   - Recursively resolve Simulation(level=N-1), Simulation(level=N-2), ..., Simulation(level=0)
   - Each level runs parallel simulations for 2^(level-1) minutes
   - Accumulate results via xr.concat(..., dim='level')
   - Store xarray as NetCDF on disk
8. Create Calculation (e.g., PositionCalculation), resolve() triggers calculate():
   - Extend simulation results (cumsum along level, add TOTAL platform)
   - Iterate over independent_dimensions, call calculate_unit() per slice
   - Combine results via xr.combine_by_coords()
   - Store xarray as NetCDF on disk
9. Create Visualization, resolve() triggers calculate():
   - Apply filters, groupby, comparison overlays
   - Render matplotlib grid, save PNG with embedded metadata JSON

**State Management:**
- ExperimentConfiguration.seed increments after each simulation batch to maintain RNG independence
- Parallel runs: PYTHON gets thread 0, CFFI gets threads 1..N-1, CUDA runs async in main process
- Results accumulate via xr.concat() along level and platform dimensions
- All floating-point data is compressed (zlib, complevel=5) when stored to NetCDF

## Key Abstractions

**HammyObject Hierarchy:**
- Purpose: Base class for all cacheable, validated, content-addressed objects
- Examples: `hammy_lib/hammy_object.py` (HammyObject, DictHammyObject, ArrayHammyObject)
- Pattern:
  - fill_metadata() collects all attributes into OrderedDict
  - generate_id() creates content hash from metadata
  - resolve() recursively resolves dependencies, loads from cache or computes via calculate()
  - load_from_filename() / dump_to_filename() handle persistence

**CCode Dataclass:**
- Purpose: Bundle C kernel with constants and platform-specific compilation paths
- Examples: `hammy_lib/ccode.py` → CCode(code, constants, function_header)
- Pattern:
  - str(c_code) returns CFFI source (constants + common.h + cuda_cpu.h + code)
  - c_code.to_cuda_source(blocks) returns CUDA source (constants + USE_CUDA + common.h + code)
  - Single .c file compiles to three platforms via conditional compilation

**Calculation Framework:**
- Purpose: Iterate over data dimensions, apply function per slice, combine results
- Examples: `hammy_lib/calculation.py` → Calculation, FlexDimensionCalculation
- Pattern:
  - independent_dimensions: dims to iterate (always includes level, platform)
  - simple_type_return: True if calculate_unit() returns scalar, False if DataArray
  - Iteration: all coordinate combinations enumerated via itertools.product()
  - Results combined via xr.combine_by_coords()

**SimulatorPlatforms Enum:**
- Purpose: Type-safe platform selection
- Examples: `hammy_lib/simulator_platforms.py` → PYTHON, CFFI, CUDA
- Pattern: Used in run_parallel_simulations() to dispatch CUDA async + CPU Pool

**Graph (Markov Chain):**
- Purpose: Store transition matrix + precomputed eigenvalues/eigenvectors for spectral methods
- Examples: `hammy_lib/graph.py` → Graph (base), LinearGraph (1D lazy walk)
- Pattern: ArrayHammyObject storing xr.Dataset with transition_matrix, eigenvalues, eigenvectors, eigenvectors_inv

## Entry Points

**Experiment Execution:**
- Location: `experiments/01_walk/walk.py` (module level), line 116 → run(level=4, ...)
- Triggers: Full pipeline from Experiment through Visualization
- Responsibilities:
  - Create all pipeline objects in order
  - Call dump() on each to resolve + cache
  - Upload results to Yandex Cloud Storage (optional)

**Compilation (CFFI to .so):**
- Location: `hammy_lib/experiment.py` → Experiment.compile(), line 85
- Triggers: Builds CFFI shared library before any CFFI simulations
- Responsibilities:
  - Create FFI instance, set_source() with C code
  - Compile to build_cffi/hammy_cpu_kernel_lib.* (auto-discovered by cffi_simulator)

**Parallel Simulation Launch:**
- Location: `hammy_lib/experiment_configuration.py` → run_parallel_simulations(), line 92
- Triggers: ExperimentConfiguration.run_parallel_simulations(loops_by_platform, calibration_mode)
- Responsibilities:
  - Launch CUDA kernel async (cuda_simulator_launch)
  - Dispatch PYTHON + CFFI via multiprocessing.Pool.map()
  - Sync GPU results (cuda_simulator_sync)
  - Concatenate platform results via xr.concat()

**Machine Detection:**
- Location: `hammy_lib/machine_configuration.py` → calculate(), line 21
- Triggers: MachineConfiguration.resolve() (called by dependency chain)
- Responsibilities:
  - CPU: psutil.cpu_count(), platform.processor()
  - GPU: subprocess nvidia-smi query, CUDA version detection
  - Compiler: intentional CFFI compile failure to extract command line

## Error Handling

**Strategy:** Fail fast on invalid/unexpected data; never return None or silently skip

**Patterns:**
- `hammy_lib/hammy_object.py`: fill_metadata() raises ValueError on ID mismatch or metadata conflict (line 48)
- `hammy_lib/calculation.py`: calculate_unit() raises ValueError if zero distribution (line 98 in position.py)
- `hammy_lib/parallel_calibration.py`: raise ValueError if calibration tolerance exceeded (line 60)
- `hammy_lib/experiment_configuration.py`: raise ValueError if CFFI times vary >25% (line 147)
- CLI: flags like --dry-run (10% calibration loops, relaxed tolerance) allow graceful testing

## Cross-Cutting Concerns

**Logging:**
- Approach: print() statements with context (platform name, loop count, elapsed time)
- Examples: "Running PYTHON simulation (loops, seed)", "[CUDA] Done in X.XXs"

**Validation:**
- Metadata conflict detection in fill_metadata() (recursively checks nested HammyObjects)
- Calibration tolerance checks (parallel vs sequential results)
- Platform-specific assertions: STATIC_ASSERT in walk.c checks BINS_LEN matches

**Authentication (Cloud):**
- S3 credentials via YandexCloudStorage class
- API key from environment (~/secrets.env)
- Auto-download missing results from Yandex Object Storage on resolve()

**Parallelism:**
- Python GIL doesn't block: PYTHON + CFFI time-share via multiprocessing.Pool (separate processes)
- CUDA runs async in main process (CUDA contexts not fork-safe)
- RNG independence: seed + thread_id for CPU threads, sequence parameter for GPU warps

---

*Architecture analysis: 2026-03-24*
