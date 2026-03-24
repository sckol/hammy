# Coding Conventions

**Analysis Date:** 2026-03-24

## Naming Patterns

**Files:**
- Module names use lowercase with underscores: `hammy_object.py`, `machine_configuration.py`, `sequential_calibration.py`
- Experiment experiment files follow `{number}_{name}/{name}.py` pattern: `experiments/01_walk/walk.py`
- Related calculation/utility modules organized in subdirectories: `hammy_lib/calculations/position.py`, `hammy_lib/calculations/popsize.py`

**Classes:**
- PascalCase for all classes: `HammyObject`, `DictHammyObject`, `ArrayHammyObject`, `Experiment`, `ExperimentConfiguration`, `SequentialCalibration`, `ParallelCalibration`, `Simulation`, `MachineConfiguration`, `Calculation`, `FlexDimensionCalculation`, `PositionCalculation`, `LinearGraph`, `Vizualization`, `YandexCloudStorage`
- Class attributes (constants) use UPPERCASE with underscores: `RESULTS_DIR`, `STORAGE`, `TARGETS`, `CHECKPOINTS`, `BINS_LEN`, `P_MIN`, `P_MAX`
- Private/internal class variables use leading underscore: `_no_check_metadata`, `_not_checked_fields`, `_id`, `_results`, `_pool`, `_cuda_kernels`, `_previous_level_simulation`, `_figure`, `_axes`

**Functions/Methods:**
- snake_case for all functions and methods: `generate_id()`, `fill_metadata()`, `value_to_clear_string()`, `get_all_variables()`, `resolve()`, `load()`, `dump_to_filename()`, `create_empty_results()`, `simulate_using_python()`, `run_single_simulation()`, `run_parallel_simulations()`
- Private methods use leading underscore: `_apply_filters()`, `_get_blue_palette()`, `_to_gray()`, `_plot_grouped_lines()`, `_find_power()`, `_get_hammy_objects_in_globals()`, `_get_s3_key_from_object()`
- Static methods marked with `@staticmethod`: `generate_digest()`, `value_to_clear_string()`, `is_regular_field()`, `get_compiler()`, `cuda_simulator_sync()`

**Variables:**
- Local and instance variables use snake_case: `calibration_tolerance`, `simulation_level`, `parallel_calibration`, `loops_by_platform`, `cpu_threads`, `gpu_out`, `cuda_out`, `cuda_start`, `platforms_str`
- Boolean flags include status context: `resolved`, `use_cuda`, `dry_run`, `calibration_mode`, `allow_aggregation`, `show_legend`, `is_comparison`
- Dimension names match xarray convention: lowercase with underscores: `platform`, `level`, `target`, `checkpoint`, `position_index`, `eigen_index`

**Types/Enums:**
- Enum members in UPPERCASE: `SimulatorPlatforms.PYTHON`, `SimulatorPlatforms.CFFI`, `SimulatorPlatforms.CUDA`

## Code Style

**Formatting:**
- No explicit formatter configured (Black/Ruff not detected in pyproject.toml)
- Line length appears to follow ~88 character convention (modern Python standard)
- Imports organized: standard library first, third-party packages second, relative imports last
- Blank lines: two lines between top-level class/function definitions, one line between methods within a class

**Linting:**
- Type hints used for function signatures: `def __init__(self, seed: int | None = None)`, `def resolve(self, no_load=False, no_check: bool = False) -> None`
- Union types use modern syntax: `id: str | None = None`, `self._results: xr.DataArray | xr.Dataset | None = None`
- Abstract method decorators: `@abstractmethod`, `@property`, `@staticmethod`, `@classmethod`
- Dataclasses with `@dataclass(frozen=True)` for immutable configuration: `CCode` in `hammy_lib/ccode.py`

**Import Organization:**

Order observed across codebase:
1. Standard library imports (re, json, hashlib, abc, pathlib, collections, multiprocessing, functools, time, inspect, subprocess, etc.)
2. Third-party framework imports (xarray, cffi, numpy, psutil, scipy, matplotlib, boto3, PIL, tqdm, enum)
3. Relative local imports (from . import or from .module import)

Example from `hammy_lib/experiment.py`:
```python
from abc import ABC, abstractmethod
from typing import assert_never
from pathlib import Path
import xarray as xr
from cffi import FFI
from time import time

from .simulator_platforms import SimulatorPlatforms
from .ccode import CCode
from .hammy_object import DictHammyObject
```

**Path Aliases:**
- Relative imports used throughout: `.simulator_platforms`, `.hammy_object`, `.ccode`, `..calculation`, `..graph`
- Paths resolved at runtime using `Path(__file__).parent`: `Path(__file__).parent / "c_libs" / lib / f"{lib}.c"`

## Error Handling

**Philosophy:** "Fail early" — never return `None` or silently skip on invalid/unexpected data. Raise descriptive exceptions immediately.

**Patterns:**

1. **Metadata validation** (`hammy_lib/hammy_object.py`):
```python
if attr_name in ["metadata", "id", "resolved", ...]:
    continue
if isinstance(attr_value, HammyObject):
    ...
    if k in self.metadata and self.metadata[k] != v and k not in self.get_not_checked_fields():
        raise ValueError(f"Metadata conflict for {k}: {self.metadata[k]} != {v}")
```

2. **Data integrity checks** (`hammy_lib/calculations/position.py`):
```python
total = x.sum()
if total == 0:
    raise ValueError(f"Zero distribution at coords {coords} — cannot compute position")
```

3. **Dimensional validation** (`hammy_lib/vizualization.py`):
```python
for dim in dims_to_sum:
    if data.sizes[dim] != 1:
        raise ValueError(
            f"Dimension '{dim}' has size {data.sizes[dim]}, but allow_aggregation is False. ..."
        )
```

4. **Parallel calibration validation** (`hammy_lib/experiment_configuration.py`):
```python
if max_cffi > min_cffi * 1.25:
    raise ValueError("CFFI times vary too much between threads")
```

5. **Optional error modes:** Some classes support `no_check` parameter to print warnings instead of raising:
```python
if no_check:
    print(f"[WARNING] Metadata conflict for {k}: {self.metadata[k]} != {v}")
else:
    raise ValueError(...)
```

6. **Graceful degradation for optional features** (`hammy_lib/machine_configuration.py`):
```python
try:
    gpu_info = subprocess.check_output("nvidia-smi ...", shell=True, text=True)
    # Process GPU info
except Exception:
    pass  # GPU detection is optional; set to None
```

**Exception Types Used:**
- `ValueError`: Invalid data states, metadata conflicts, dimension mismatches, zero distributions
- `AttributeError`: Missing required class attributes in subclass validation
- `RuntimeError`: (Implicit via `assert_never`) for exhaustive match failures on enums

## Logging

**Framework:** `print()` for user-facing messages; no logger module detected.

**Patterns:**

1. **Status messages with platform context** (`hammy_lib/sequential_calibration.py`):
```python
print(f"[{platform.name}] Calibrating with {loops} loops...")
print(f"[{platform.name}] Took {elapsed_time:.2f}s")
print(f"[{platform.name}] Calibrated: {final_loops} loops/min")
```

2. **S3 operation logs** (`hammy_lib/yandex_cloud_storage.py`):
```python
print(f"[S3] Uploaded {s3_key} from {local_path}")
print(f"[S3] {s3_key} already exists. Skipping upload.")
print(f"[LOCAL] {local_path} already exists. Skipping download.")
```

3. **Cache/load messages** (`hammy_lib/hammy_object.py`):
```python
print(f"Not found the file {self.id}")
print(f"Loaded cached object {self.id} from file")
```

4. **Detailed debug messages**:
```python
print(f"Running {platform.name} simulation ({loops} loops, seed {seed})...")
print(f"Running parallel {mode}: {platforms_str}")
print(f"[CUDA] Launching {loops_by_platform[SimulatorPlatforms.CUDA]} loops (async)...")
print(f"[CUDA] Done in {cuda_elapsed:.2f}s")
```

5. **Configuration discovery** (`hammy_lib/machine_configuration.py`):
```python
print(f"[MKL] Using {os.path.basename(path)} from {os.path.dirname(path)}", flush=True)
```

No custom log levels; all output goes to stdout via `print()` or stderr via `subprocess` output.

## Comments

**When to Comment:**
- Complex algorithms documented inline (e.g., spectral decomposition in `PositionCalculation`)
- State machine transitions explained (e.g., calibration loop in `SequentialCalibration`)
- Mathematical concepts with references (e.g., eigenvalue decomposition in `graph.py`)

**JSDoc/TSDoc:**
- Module and class docstrings in NumPy style:
```python
class Experiment(DictHammyObject, ABC):
    """Defines a Monte Carlo simulation: its parameters, result shape, and kernels.

    An Experiment provides three equivalent implementations of the same simulation:
    - simulate_using_python() — NumPy reference (subclass implements)
    - cffi_simulator() — C kernel compiled to .so, called via CFFI
    - cuda_simulator() — same C kernel compiled for GPU via CuPy RawKernel
    ...
    """
```

- Method docstrings for complex logic:
```python
def _find_power(self, eigvals, eigvecs_inv, x) -> float:
    """Find p by matching spectral spread of f to mean spectral spread of T^p columns.

    Spectral spread = energy in non-stationary eigenmodes.
    For observed f: Σ_{k≥1} (V⁻¹f)_k²
    ...
    """
```

- Parameter documentation in docstrings:
```python
def calculate_unit(self, input_array: xr.DataArray, coords: dict) -> xr.DataArray | int | float | bool:
    """Calculate a single unit given a selected DataArray slice and the
    current coordinates dictionary.

    Args:
        input: xarray DataArray selected for the current coordinates
        coords: dict mapping dimension names to their current coordinate values

    Returns:
        xr.DataArray or a simple type (int/float/bool)
    """
```

## Function Design

**Size:**
- Most methods 5-30 lines; complex workflows delegate to private helpers
- Recursive methods like `resolve()` handle dependency trees by iterating over attributes

**Parameters:**
- Type hints required for all public function signatures
- Optional parameters use modern `| None` syntax with defaults: `seed: int | None = None`
- Boolean flags grouped logically: `no_load=False, no_check: bool = False`
- Large parameter sets avoided; configuration passed via class constructors

**Return Values:**
- Methods return xarray objects, floats (for timing), or dictionaries (for calibration results)
- No silent returns; all code paths explicit
- Union return types documented: `xr.DataArray | float` for calibration mode

## Module Design

**Exports:**
- Public classes and functions at module level
- Internal utilities prefixed with underscore
- Submodules follow package structure: `hammy_lib.calculations.position`, `hammy_lib.calculations.popsize`

**Barrel Files:**
- `hammy_lib/__init__.py` present but minimal (checked)
- Direct imports used: `from hammy_lib.experiment import Experiment`
- No star imports observed

**Import Patterns:**
- Relative imports within package: `from .hammy_object import DictHammyObject`
- Relative imports from subpackages: `from ..calculation import Calculation` (from `calculations/position.py`)

## Xarray and Data Structure Conventions

**Dimension naming:** lowercase with underscores: `platform`, `level`, `target`, `checkpoint`, `position_index`, `eigen_index`, `position_data`

**Coordinate handling:**
- String coordinates use `xr.DataArray([...], dims='dim_name')` (avoids `pd.Index` compatibility issues with h5netcdf)
- Example from `hammy_lib/experiment_configuration.py`:
```python
return xr.concat(
    platform_results,
    dim=xr.DataArray([p.name for p in platforms], dims="platform"),
)
```

**Data types:**
- Simulations use `int64` for counts: `np.zeros(..., dtype=np.int64)`
- Results stored as `float32` when possible, downcast in `dump_to_filename()`
- Complex calculations (eigenvalues) explicitly upcast: `.astype(np.complex128)`

---

*Convention analysis: 2026-03-24*
