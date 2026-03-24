# Testing Patterns

**Analysis Date:** 2026-03-24

## Test Framework

**Status:** No formal test framework configured (pytest/unittest not detected).

**How tests are run in practice:**
- Experiments run end-to-end via CLI: `python -m experiments.01_walk --level 2 --no-viz`
- Manual validation during development: run experiment locally, check results in Google Colab, deploy to Yandex Cloud
- Validation embedded in pipeline: `ParallelCalibration` raises `ValueError` if platform timing drifts >25% from sequential

**Run Commands:**
```bash
# Full pipeline (local file)
cd experiments/01_walk && python walk.py

# Selective steps via CLI
python -m experiments.01_walk --level 2 --no-viz --no-calculations

# Dry run (10% loops, relaxed tolerance)
python -m experiments.01_walk --dry-run

# C compilation test
python -c "from experiments.01_walk.walk import WalkExperiment; WalkExperiment().compile()"

# Debug C executable
cmake -B build && cmake --build build && ./build/walk
```

## Test File Organization

**Location:** No dedicated test directory (`tests/` not found). Validation logic integrated into core modules.

**Validation pattern:**
- Input validation at method entry points (fail-fast principle)
- State validation during resolution: `fill_metadata()` checks for conflicts
- Output validation in calibration: `ParallelCalibration` compares parallel vs sequential timing

## Test Structure

**Integration tests embedded in pipeline:**

The `hammy_lib/experiment_configuration.py` implements validation during parallel calibration:

```python
def run_parallel_simulations(
    self, loops_by_platform: CalibrationResults, calibration_mode=False
) -> xr.DataArray | CalibrationResults:
    # ... CUDA and CPU execution ...
    if calibration_mode:
        # Calculate loops per minute for each platform
        results = {
            SimulatorPlatforms.PYTHON: time_to_loops(res[0], SimulatorPlatforms.PYTHON)
        }
        cffi_times = res[1:]
        if cffi_times:
            min_cffi = min(cffi_times)
            max_cffi = max(cffi_times)
            if max_cffi > min_cffi * 1.25:
                raise ValueError("CFFI times vary too much between threads")
        # ... more validation ...
```

**State machine validation:**

`HammyObject` uses metadata-based versioning to detect stale cache:

```python
def load(self, no_check: bool = False) -> bool:
    # ... download if needed ...
    if not self._no_check_metadata:
        old_metadata = self.metadata.copy()
        self.load_from_filename(self.filename)
        # ... compare metadata ...
        differences = {
            k: (old_metadata.get(k), self.metadata.get(k))
            for k in set(old_metadata) | set(self.metadata)
            if old_metadata.get(k) != self.metadata.get(k)
            and k not in not_checked
        }
        if differences:
            raise ValueError(f"Metadata mismatch for {self.id}: {differences}")
```

## Mocking

**Framework:** No mocking framework detected (unittest.mock or pytest-mock not used).

**Manual mocking patterns observed:**

1. **Optional feature detection** (`hammy_lib/machine_configuration.py`):
```python
try:
    gpu_info = subprocess.check_output("nvidia-smi ...", shell=True, text=True)
    # Process GPU info
except Exception:
    pass  # GPU optional; metadata["gpu_model"] = None
```

2. **Platform abstraction** (`hammy_lib/simulator_platforms.py`):
Uses enum to switch between implementations without runtime branching:
```python
match platform:
    case SimulatorPlatforms.PYTHON:
        self.simulate_using_python(loops, out, seed)
    case SimulatorPlatforms.CFFI:
        self.cffi_simulator(loops, out, seed)
    case SimulatorPlatforms.CUDA:
        self.cuda_simulator(loops, out, seed)
    case _:
        assert_never(platform)
```

3. **Dry-run mode** (used for lightweight testing):
```python
def __init__(self, ..., dry_run: bool = False):
    self.dry_run = dry_run

def run_single_calibration(self, platform: SimulatorPlatforms) -> float:
    dry_run_multiplier = 0.1 if self.dry_run else 1.0
    loops = int(1000 * dry_run_multiplier)
    # Reduces calibration time to 10% for quick validation
```

**What to Mock:**
- External services: S3 bucket access (boto3) could be mocked in isolation
- Hardware detection: GPU/CUDA detection could be disabled for CI
- Timing-sensitive operations: calibration could use fixed loop counts in test mode

**What NOT to Mock:**
- Core simulation logic (PYTHON, CFFI platforms) — validates algorithm correctness
- xarray operations — validates data structure consistency
- File I/O for metadata/results — validates serialization

## Fixtures and Factories

**Test data:**

No dedicated fixture framework, but pattern observed in `experiments/01_walk/walk.py`:

```python
class WalkExperiment(Experiment):
    experiment_number = 1
    experiment_name = "walk"
    experiment_version = 1
    T = 1000  # Steps in single walk
    CHECKPOINTS = [100, 200, 300, 400, 500, 600, 700, 800, 900, T]
    TARGETS = [0, 1, 2, 5, 10]
    BINS_TUPLE = (-T // 20, T // 20 + 1)

    def create_empty_results(self) -> xr.DataArray:
        dims = ["target", "checkpoint", "x"]
        coords = {
            "x": np.arange(*self.BINS_TUPLE),
            "target": self.TARGETS,
            "checkpoint": self.CHECKPOINTS,
        }
        return xr.DataArray(
            np.zeros(tuple(len(coords[i]) for i in dims), dtype=np.int64),
            coords=coords,
            dims=dims,
        )
```

**Manual factory pattern** (`hammy_lib/calculations/position.py`):
```python
class PositionCalculation(Calculation):
    def __init__(self, main_input, graph: Graph, dimensionality: int,
                 spatial_dims=("position_index",), id: str = None):
        super().__init__(main_input, id)
        self.graph = graph
        self.dimensionality = dimensionality
```

Graph objects can be created with different topologies:
```python
graph = LinearGraph(length=50)  # 1D chain
```

**Location:** Constants defined in experiment class itself; graphs defined in `hammy_lib/graph.py`

## Coverage

**Requirements:** Not enforced; no coverage tool configured.

**Implicit coverage through validation:**
- Every simulation platform (PYTHON, CFFI, CUDA) compared during calibration
- Calculation pipeline tested end-to-end in experiment runs
- Metadata versioning catches serialization bugs

## Test Types

**Unit Tests (implicit via integration):**

Platform-specific implementations validated independently:
- `Experiment.simulate_using_python()` — NumPy reference implementation
- `Experiment.cffi_simulator()` — compiled C via CFFI
- `Experiment.cuda_simulator()` — GPU kernel via CuPy

Each platform produces same output structure (xr.DataArray), enabling direct comparison:
```python
python_result = res[0]      # From PYTHON platform
cffi_results = res[1:]      # From CFFI threads
cuda_result = cuda_out      # From GPU
# Results concatenated:
xr.concat([python_result, cffi_combined, cuda_result], dim="platform")
```

**Calibration Tests:**

Built into `hammy_lib/parallel_calibration.py`:
```python
def calculate(self) -> None:
    parallel_results = self.experiment_configuration.run_parallel_simulations(
        self.sequential_calibration_results, calibration_mode=True
    )
    discrepancies = []
    for platform in parallel_results:
        diff_percent = (
            abs(parallel_results[platform] - self.sequential_calibration_results[platform])
            / self.sequential_calibration_results[platform]
            * 100
        )
        if diff_percent > self.calibration_tolerance:  # Default 25%
            discrepancies.append(...)
            raise ValueError("Parallel calibration failed:\n" + ...)
```

**Integration Tests (end-to-end):**

Full pipeline invoked via CLI:
```bash
python -m experiments.01_walk --level 2 --no-viz
```

Validates:
1. Experiment initialization (Experiment class loaded)
2. Machine detection (MachineConfiguration.calculate())
3. Sequential calibration (loops/minute per platform)
4. Parallel calibration (timing consistency check)
5. Simulation execution (xarray results generated)
6. Calculations (post-processing on results)
7. Visualization (PNG output with metadata)
8. S3 sync (upload/download if configured)

**E2E Tests (manual):**

Primary validation happens in Google Colab (browser, CUDA T4 GPU) and Yandex Cloud (production run with many iterations). These are manual but systematic:
1. Run locally to verify correctness
2. Run in Colab to verify GPU/T4 compat
3. Run on Yandex Cloud to generate final results

## Common Patterns

**Async GPU execution validated via sync:**

`hammy_lib/experiment.py`:
```python
def cuda_simulator(self, loops: int, out: xr.DataArray, seed: int, blocks: int = 1000) -> None:
    """Launch CUDA kernel and copy results back synchronously."""
    gpu_out = self.cuda_simulator_launch(loops, out, seed, blocks)
    self.cuda_simulator_sync(gpu_out, out)

@staticmethod
def cuda_simulator_sync(gpu_out, out: xr.DataArray) -> None:
    """Wait for GPU kernel to finish and copy results to CPU."""
    import cupy as cp
    cp.cuda.get_current_stream().synchronize()
    out.values[:] = gpu_out.get()
```

**Zero-distribution detection:**

`hammy_lib/calculations/position.py`:
```python
total = x.sum()
if total == 0:
    raise ValueError(f"Zero distribution at coords {coords} — cannot compute position")
```

**Optional dimensions validated at plot time:**

`hammy_lib/vizualization.py`:
```python
if self.allow_aggregation:
    plot_data = data.sum(dims_to_sum) if dims_to_sum else data
else:
    # Check that all dims_to_sum have only one element
    for dim in dims_to_sum:
        if data.sizes[dim] != 1:
            raise ValueError(
                f"Dimension '{dim}' has size {data.sizes[dim]}, but allow_aggregation is False. ..."
            )
```

## Validation Without Tests

The codebase relies on **runtime validation** instead of test suites:

1. **Type hints** — used throughout for early detection of type mismatches
2. **Fail-fast exceptions** — invalid data detected immediately with descriptive errors
3. **Metadata versioning** — cache invalidation detects stale results
4. **Platform consistency** — all three simulators produce compatible output
5. **Calibration gates** — parallel timing must match sequential within tolerance
6. **Multi-environment validation** — code tested on local machine, Colab (GPU), and Yandex Cloud

This pattern works because:
- Simulations are deterministic given a seed
- Results are large, making partial failures obvious
- Multiple execution platforms (PYTHON, CFFI, CUDA) provide cross-validation
- Full pipeline runs in ~minutes, enabling frequent end-to-end testing

---

*Testing analysis: 2026-03-24*
