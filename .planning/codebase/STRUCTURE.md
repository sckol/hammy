# Codebase Structure

**Analysis Date:** 2026-03-24

## Directory Layout

```
hammy/
├── .planning/               # GSD planning documents (generated)
├── .venv/                   # Python virtual environment
├── build/                   # CMake-built C executables (auto-generated)
├── build_cffi/              # CFFI-compiled .so files (auto-generated)
├── docker/                  # Docker container definitions
├── experiments/             # Experiment implementations
│   └── 01_walk/             # Walk experiment (primary demo)
├── hammy_docs/              # Markdown documentation
├── hammy_lib/               # Core simulation library
│   ├── c_libs/              # C helper libraries
│   │   ├── common/          # Platform abstraction (common.h)
│   │   ├── cuda_cpu/        # CPU warp emulation (cuda_cpu.h)
│   │   └── pcg_basic/       # PCG32 RNG (pcg_basic.h, pcg_basic.c)
│   ├── calculations/        # Post-processing calculations
│   └── *.py                 # Core classes
├── results/                 # Cached results (experiment outputs)
│   └── 1_walk_1/            # Results for walk experiment v1
├── CMakeLists.txt           # C project build config
├── pyproject.toml           # Python package config
├── pyrightconfig.json       # Type checker config
└── README.md                # Project overview
```

## Directory Purposes

**experiments/:**
- Purpose: Self-contained experiment implementations
- Contains: Experiment subclasses, experiment-specific C code, CLI entry points
- Key files: `01_walk/walk.py` (WalkExperiment), `01_walk/walk.c` (C kernel), `01_walk/walk.h` (constants)

**hammy_lib/:**
- Purpose: Reusable simulation framework
- Contains: Base classes, orchestration, calibration, calculation, visualization
- Key files: `hammy_object.py` (base), `experiment.py` (Experiment base), `experiment_configuration.py` (orchestration)

**hammy_lib/c_libs/:**
- Purpose: C helper libraries for dual-platform compilation
- Contains: Platform abstraction macros, RNG, CPU warp emulation
- Key files:
  - `common/common.h` — conditional compilation (USE_CUDA, BLOCKS, macro definitions)
  - `cuda_cpu/cuda_cpu.h` — CPU emulation of CUDA (warp loop, atomicAdd, barriers)
  - `pcg_basic/pcg_basic.h/c` — PCG32 random number generator (seed-to-stream mapping)

**hammy_lib/calculations/:**
- Purpose: Post-processing calculations on simulation results
- Contains: Concrete Calculation subclasses
- Key files: `position.py` (PositionCalculation), `popsize.py` (PopulationSizeCalculation)

**results/:**
- Purpose: Cache of all computed objects
- Contains: JSON (metadata) + NetCDF (xarray data)
- Structure: `results/{experiment_number}_{experiment_name}_{experiment_version}/{object_id}.{json|nc}`
- Auto-downloaded: from Yandex Object Storage if missing locally

## Key File Locations

**Entry Points:**
- `experiments/01_walk/walk.py` (lines 44-150): WalkExperiment class + run() orchestration
- `experiments/01_walk/__main__.py`: CLI entry (parses --level, --no-viz, etc.)

**Configuration:**
- `pyproject.toml`: Package metadata, dependencies
- `pyrightconfig.json`: Type checker settings (venv path, python version)
- `CMakeLists.txt`: C build config (discovers experiments/NN_*/CMakeLists.txt)

**Core Classes:**
- `hammy_lib/hammy_object.py` (235 lines): HammyObject hierarchy, caching, validation
- `hammy_lib/experiment.py` (163 lines): Experiment base, simulator dispatch, compile()
- `hammy_lib/ccode.py` (65 lines): CCode dataclass, platform-specific source generation
- `hammy_lib/experiment_configuration.py` (173 lines): Orchestration, multiprocessing.Pool, run_parallel_simulations()

**Simulation Pipeline:**
- `hammy_lib/sequential_calibration.py` (67 lines): Measure loops/minute per platform
- `hammy_lib/parallel_calibration.py` (74 lines): Validate parallel results within tolerance
- `hammy_lib/simulation.py` (85 lines): Run experiment levels 0..N, concatenate results

**Post-Processing:**
- `hammy_lib/calculation.py` (170 lines): Base Calculation, iteration framework
- `hammy_lib/calculations/position.py` (150+ lines): Spectral decomposition of distribution
- `hammy_lib/calculations/popsize.py`: Population size statistics

**Graphics:**
- `hammy_lib/vizualization.py` (337 lines): matplotlib grid plots, PNG export with metadata

**Utilities:**
- `hammy_lib/machine_configuration.py` (125 lines): Auto-detect CPU/GPU/compiler/OS
- `hammy_lib/graph.py` (52 lines): Markov chain transition matrices + eigenvalues
- `hammy_lib/simulator_platforms.py` (8 lines): SimulatorPlatforms enum
- `hammy_lib/yandex_cloud_storage.py`: S3 client for Yandex Object Storage
- `hammy_lib/calibration_results.py`: CalibrationResults type alias + (de)serialization

## Naming Conventions

**Files:**
- Class definition: PascalCase, e.g., `experiment.py` contains Experiment class
- Calculation subclass: PascalCase, e.g., `position.py` contains PositionCalculation
- Entry point: snake_case matching experiment name, e.g., `walk.py` for WalkExperiment

**Directories:**
- Numbered experiment prefix: `NN_name`, e.g., `01_walk` (auto-discovered by CMake)
- Reusable library: `hammy_lib/` (installed as package)
- Specific subdomain: descriptive, e.g., `c_libs/`, `calculations/`

**C files:**
- Platform-specific: suffixed with platform, e.g., no suffix = CPU/CUDA unified, `.h` = header
- Macro conventions: UPPERCASE for constants, TID/TID_LOCAL for thread indexing, _32/_ for array suffixes

**Results objects:**
- ID format: `{experiment_configuration_string}_{stage}_{optional_suffix}`
  - Example: `1_walk_1_a1b2c3_simulation_4` (experiment, machine digest, level)
- File naming: `{id}.{json|nc}`

## Where to Add New Code

**New Experiment:**
1. Create `experiments/NN_name/` directory
2. Add `name.py` with Experiment subclass (define experiment_number, experiment_name, experiment_version, c_code, create_empty_results(), simulate_using_python())
3. Add `name.c` with C kernel (signature: void run_simulation(unsigned long long loops, const unsigned long long seed, unsigned long long* out))
4. Optional: add `name.h` with compile-time constants
5. CMake auto-discovers NN_* pattern

**New Calculation:**
1. Add file to `hammy_lib/calculations/` matching class name, e.g., `my_calc.py`
2. Subclass Calculation or FlexDimensionCalculation
3. Implement: independent_dimensions (property), simple_type_return (property), calculate_unit(input_array, coords)
4. Return scalar (if simple_type_return=True) or xarray (if False)

**New Core Utility:**
1. Add to `hammy_lib/` if reusable across experiments
2. Subclass HammyObject if it needs caching/validation
3. Implement abstract methods: generate_id(), calculate(), file_extension, etc.

**New C Library:**
1. Add to `hammy_lib/c_libs/{lib_name}/` with header `{lib_name}.h`
2. Optional: add `{lib_name}.c` for implementation
3. Update CCode.generate_include_for_platform() if needed to include in compilation

**Testing:**
- Add `test_*.py` or `*_test.py` in experiment or hammy_lib directories
- Run via pytest (not currently configured, but structure is ready)

## Special Directories

**build/ (CMake):**
- Purpose: Auto-generated C executables
- Generated: `cmake -B build && cmake --build build`
- Committed: No (in .gitignore)
- Contents: `walk` (standalone C executable for debugging)

**build_cffi/ (CFFI compilation):**
- Purpose: Auto-generated shared libraries (.so / .dll)
- Generated: Experiment.compile() → FFI.compile()
- Committed: No (in .gitignore)
- Contents: `hammy_cpu_kernel_lib.so` or similar

**results/ (Cache):**
- Purpose: Persistent cache of all computed objects
- Generated: HammyObject.dump() → creates results/{experiment_string}/{id}.{json|nc}
- Committed: No (in .gitignore)
- Overridable: HammyObject.RESULTS_DIR = Path(...) at experiment start

**.planning/codebase/ (GSD):**
- Purpose: Generated documentation for code execution
- Generated: By `/gsd:map-codebase` orchestrator
- Committed: Yes (tracks architecture decisions)
- Contents: ARCHITECTURE.md, STRUCTURE.md, CONVENTIONS.md, TESTING.md, STACK.md, INTEGRATIONS.md, CONCERNS.md

---

*Structure analysis: 2026-03-24*
