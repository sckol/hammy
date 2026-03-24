# Codebase Concerns

**Analysis Date:** 2026-03-24

## Tech Debt

**Multiprocessing Pool not cleaned up:**
- Issue: `ExperimentConfiguration.__init__()` creates a `Pool` object in `self._pool` but never closes it. The Pool remains alive for the lifetime of the ExperimentConfiguration instance, holding system resources.
- Files: `hammy_lib/experiment_configuration.py` (line 33)
- Impact: Long-running experiments or notebook sessions that create multiple ExperimentConfigurations will accumulate zombie processes and consume file descriptors. On systems with process limits, this can cause `OSError: [Errno 24] Too many open files`.
- Fix approach: Implement `__del__()` or context manager (`__enter__`/`__exit__`) to call `self._pool.close()` and `self._pool.terminate()`. Alternatively, use `Pool` as a context manager with `with` statement in the experiment pipeline.

**Compiler detection via intentional error:**
- Issue: `MachineConfiguration.get_compiler()` deliberately triggers a CFFI compilation error to extract the compiler path (line 91: `1 // intentional error`). This is fragile and relies on parsing exception strings.
- Files: `hammy_lib/machine_configuration.py` (lines 81-112)
- Impact: If FFI changes its error message format, or if the compiler path appears elsewhere in the exception text, string parsing will fail or return incorrect path. Compiler detection will silently return malformed version strings or crash.
- Fix approach: Use `distutils.ccompiler.get_default_compiler()` or query the compiler directly (e.g., `cc --version`) instead of triggering an intentional failure.

**Recursive metadata conflict checking with no_check flag:**
- Issue: `HammyObject.fill_metadata()` can print warnings but continue (when `no_check=True`) instead of raising errors (line 46). This creates two silent failure modes: conflicts are logged but execution proceeds, or cached metadata mismatches are silently printed (line 145). Data corruption can occur undetected.
- Files: `hammy_lib/hammy_object.py` (lines 34-62, 127-151)
- Impact: Metadata integrity is a core contract of the caching system — if an object with ID `xyz` loads but its metadata differs from what was saved, the cache is stale and results are unreliable. Silent warnings allow invalid cache hits to propagate.
- Fix approach: Remove `no_check` parameter entirely, or make it only apply to validation of loaded data (not conflicts). Always raise on ID/metadata mismatch in `fill_metadata()`.

**Print statements instead of logging:**
- Issue: Debug and status output uses `print()` throughout the codebase instead of a logging framework.
- Files: `hammy_lib/` (all modules), `experiments/01_walk/walk.py`
- Impact: No log levels, no filtering, no timestamp, no file output. In long-running Yandex Cloud experiments (hours of execution), error context is lost. Difficult to retroactively debug failures from notebook output.
- Fix approach: Replace all `print()` calls with a configured logger (Python `logging` module). Add log file output via FileHandler.

**Magic numbers in calibration logic:**
- Issue: Hardcoded time thresholds (15 seconds in line 34), tolerance ratios (1.25 in line 146), and loop multipliers (doubling in line 46 of sequential_calibration.py) are scattered throughout calibration code with no documented rationale.
- Files: `hammy_lib/sequential_calibration.py` (lines 34, 45-46), `hammy_lib/experiment_configuration.py` (line 146)
- Impact: Changing calibration behavior requires finding and understanding multiple disconnected constants. `calibration_tolerance` varies widely (25 vs 100, determined in walk.py line 131) with no clear policy.
- Fix approach: Move all calibration constants to a `CALIBRATION_CONFIG` dict at module level. Document why each value was chosen (e.g., "15s chosen to be > 5 sigma of timing noise").

## Known Bugs

**Seed increment in run_single_simulation() creates asymmetric seeding:**
- Issue: `ExperimentConfiguration.run_single_simulation()` increments `self.seed` after each call (line 70), but this only affects sequential calls. When `run_parallel_simulations()` is called, it uses `self.seed` plus offsets (`seed + thread_id`, `seed + cpu_threads`). This asymmetry means seeds are not uniformly spaced across threads and sequential runs.
- Files: `hammy_lib/experiment_configuration.py` (lines 63-71, 113, 120, 131)
- Impact: If calibration and simulation both call `run_single_simulation()`, calibration seeds and simulation seeds overlap or repeat. Reduces RNG independence between calibration phases and simulation phases.
- Trigger: Sequential calibration → Parallel calibration → Simulation (walk.py lines 128-139) with continuous seed increments across all three stages.
- Workaround: Document that seed space is continuous across calibration and simulation; may be acceptable if the pipeline only runs once per ExperimentConfiguration.

**CFFI library lookup assumes first glob match is correct:**
- Issue: `Experiment.cffi_simulator()` uses `list(...glob(...))[0]` to find the compiled library (line 101), but glob order is not guaranteed to be deterministic. If multiple CFFI library versions exist in `build_cffi/`, the wrong one may be loaded.
- Files: `hammy_lib/experiment.py` (lines 100-102)
- Impact: Silently loads stale or incompatible CFFI library. Results will be incorrect without warning.
- Fix approach: Sort glob results explicitly (e.g., `sorted(glob(...))[0]`) or use the most recent file by mtime: `max(...glob(...), key=os.path.getmtime)`.

**CUDA kernel caching by block count only:**
- Issue: `Experiment._cuda_kernels` dict keys on `blocks` parameter only, but different block values can compile to identical kernels if BLOCKS define is the same. No cache invalidation if CCode changes.
- Files: `hammy_lib/experiment.py` (lines 109-116)
- Impact: If an experiment's C code is modified and a new Experiment instance is created, the old cached CUDA kernel persists. Results will silently use stale GPU code.
- Fix approach: Include C code hash in cache key: `cache_key = (blocks, hash(self.c_code.to_cuda_source(blocks)))`. Add method to flush cache on object recreation.

**Division by zero in PositionCalculation weight normalization:**
- Issue: `PositionCalculation.calculate_unit()` normalizes weights by `weight_sum`, but only checks if `weight_sum > 0` after thresholding (line 116). If all non-zero weights fall below 1% threshold, `weight_sum` is zero and returns `nonzero_values` unchanged (un-normalized).
- Files: `hammy_lib/calculations/position.py` (line 116)
- Impact: Returned weights sum to < 1, violating the invariant that output weights should sum to 1. Downstream aggregation will treat them as probabilities, producing incorrect totals.
- Trigger: Distribution with many small equal-weight components (e.g., all components near 1% threshold).
- Fix approach: Raise error if all components are thresholded out: `if weight_sum == 0: raise ValueError("All NNLS components fall below threshold")`

**YandexCloudStorage._get_hammy_objects_in_globals() frame leak:**
- Issue: The frame variable is deleted in the finally block (line 34), but intermediate frame references in the loop may still reference parent frames. This is generally safe in CPython but relies on reference counting and can cause circular reference memory leaks in other Python implementations (PyPy, etc.).
- Files: `hammy_lib/yandex_cloud_storage.py` (lines 23-35)
- Impact: Minor — CPython will clean up, but fragile to implementation changes. Remote environments with resource constraints may see memory growth.
- Fix approach: Use `sys._getframe()` depth-limited loop instead of walking f_back manually, or use `inspect.stack()` which handles cleanup correctly.

## Security Considerations

**S3 credentials passed as plain strings:**
- Risk: Access/secret keys are accepted as string parameters in `YandexCloudStorage.__init__()` (line 8). These keys are then stored in instance variables accessible to any code with a reference to the storage object.
- Files: `hammy_lib/yandex_cloud_storage.py` (lines 8-11), `experiments/01_walk/walk.py` (lines 210-217)
- Current mitigation: Credentials come from `google.colab.userdata` (Colab secrets, ephemeral) or environment variables (can be logged or leaked). walk.py does not validate/redact them before passing.
- Recommendations: (1) Never store keys in instance variables; (2) Accept credentials as `os.environ` lookups only, with validation that they exist; (3) Use boto3 session credentials (IAM roles) if on Yandex Cloud VMs; (4) Never print or log credentials (add redaction to logging module).

**Frame introspection in upload/download:**
- Risk: `YandexCloudStorage._get_hammy_objects_in_globals()` iterates through all frame locals and globals (lines 19-35). This grants access to all variables in the call stack — local variables, function parameters, etc. If this method is ever exposed to untrusted code (e.g., plugin system), credential leaks are possible.
- Files: `hammy_lib/yandex_cloud_storage.py` (lines 19-35)
- Current mitigation: Only used internally within trusted hammy library.
- Recommendations: (1) Document that this method should never be exposed publicly; (2) Add explicit allow-list of object types to search for (only HammyObject subclasses); (3) Use explicit object registration instead of frame introspection.

**Subprocess calls with shell=True:**
- Risk: `MachineConfiguration.calculate()` and `machine_configuration.py` use `subprocess.check_output(..., shell=True)` (lines 43-50, 64). Though the commands are hardcoded, shell=True is vulnerable if command strings are ever constructed from user input.
- Files: `hammy_lib/machine_configuration.py` (lines 43-50, 64)
- Current mitigation: Commands are literals, not user-supplied.
- Recommendations: (1) Use `shell=False` with `shlex.split()` or command list; (2) Add a comment explaining why shell=True is safe (or make it unsafe-to-prove and refactor).

## Performance Bottlenecks

**Graph eigenvalue computation has no caching:**
- Problem: `Graph.calculate()` recomputes eigenvalues and eigenvector inverse on every resolve (line 18-24). For LinearGraph with 100-200 nodes, `np.linalg.eig()` is fast, but `scipy.linalg.inv()` on eigenvectors is O(n³). If PositionCalculation is called multiple times per graph, recalculation is wasted.
- Files: `hammy_lib/graph.py` (lines 18-24)
- Cause: `Graph.results` is set in `__init__()` but eigenvalues only computed in `calculate()`. Multiple Graph instances with same size recompute identical results.
- Improvement path: Cache Graph results by size (e.g., `_graph_cache[size]`). Or compute eigenvalues in `__init__()` rather than `calculate()`.

**Calculation.calculate() does not parallelize independent_dimensions iteration:**
- Problem: `Calculation.calculate()` iterates over all coordinate combinations sequentially (line 106-135). For a 5-level × 3-platform × 5-target × 10-checkpoint simulation, this is 750 `calculate_unit()` calls, each reading xarray slices from disk. No parallelization.
- Files: `hammy_lib/calculation.py` (lines 106-135)
- Cause: Easy to iterate sequentially with tqdm, hard to parallelize and merge results via `xr.combine_by_coords`.
- Improvement path: (1) Use `multiprocessing.Pool` or `concurrent.futures.ThreadPoolExecutor` to parallelize over coordinate combinations; (2) Each worker writes results to temporary netCDF file; (3) Merge temporary files at the end. Expected 3-5x speedup on multi-core systems.

**Position calculation O(n²) NNLS on large graphs:**
- Problem: `PositionCalculation.calculate_unit()` solves NNLS on V matrix of size `(n_nodes, n_nodes)` for each coordinate combination (line 109). For 200-node LinearGraph with 750 combinations, this is 200² = 40k × 750 = 30M float operations. NNLS uses iterative descent (slow).
- Files: `hammy_lib/calculations/position.py` (lines 101-109)
- Cause: Graph size grows with walk length (T/2 bins). NNLS becomes bottleneck for large experiments.
- Improvement path: (1) Use sparse matrix NNLS if most columns are zero; (2) Warm-start NNLS with solution from adjacent level/target; (3) Cache T^p computation instead of recomputing for each coordinate.

**xarray string coordinates using StringDtype:**
- Problem: Comment in CLAUDE.md (hammy_object.py section) warns that `pd.Index` for string coordinates converts to StringDtype, which h5netcdf/NetCDF4 cannot write. However, `Vizualization` and other modules may still use xarray operations that introduce StringDtype.
- Files: `hammy_lib/vizualization.py`, `hammy_lib/experiment_configuration.py` (line 171)
- Cause: Easy to introduce via `xr.DataArray([...], dims='dim_name')` without explicit dtype control.
- Improvement path: Add a helper function `safe_string_array(values, dtype='<U')` to ensure U dtype is used everywhere string coordinates are created.

## Fragile Areas

**Visualizations with hardcoded filters and pyright ignores:**
- Files: `experiments/01_walk/walk.py` (lines 158-204)
- Why fragile: Multiple visualizations access `position` and `popsize` variables conditionally (inside `if not no_calculations`), then reference them unconditionally (lines 170, 185, 197). Pyright ignores are used to silence errors (line 170, 174, 185). If someone adds a code path where `no_calculations=True` but visualization runs, runtime NameError.
- Safe modification: (1) Move all Viz creation inside their conditional blocks; (2) Remove pyright ignores and move variables to outer scope if they must be used.
- Test coverage: No unit tests for walk.py CLI modes — cannot verify that all combinations of `--no-calculations`, `--no-viz`, `--no-upload` work correctly.

**C code macro system for CPU/GPU compatibility:**
- Files: `hammy_lib/c_libs/cuda_cpu/cuda_cpu.h`, `hammy_lib/c_libs/common/common.h`, `experiments/01_walk/walk.c`
- Why fragile: The dual-compilation system (same C source for CPU and GPU via macros) is powerful but error-prone. `_32` suffix means "32 virtual threads", `__WARP_INIT`/`__WARP_END` expand to for-loops on CPU but no-ops on GPU. Off-by-one errors in `threadIdx.x` indexing will be silent on GPU (wrong data) or fail on CPU (array index out of bounds).
- Safe modification: (1) Add comprehensive test cases for both CPU and GPU paths; (2) Add assertions in macro definitions to catch common mistakes; (3) Use a linter to verify macros are used correctly.
- Test coverage: walk.c has no unit tests. Only integration tests (full simulation runs). A bug in macro expansion can only be caught at runtime in full pipeline.

**ExperimentConfiguration thread count logic is implicit:**
- Files: `hammy_lib/experiment_configuration.py` (lines 27-31)
- Why fragile: `threads = max(3, self.cores + 1) if self.use_cuda else max(2, self.cores)` compresses complex thread allocation into one line. CUDA case reserves 1 thread for main process, CFFI gets cores-1 threads, Python gets 1 thread (commented at line 30). If someone changes line 31 without understanding the invariant, thread allocation breaks.
- Safe modification: Extract thread allocation to a named constant or method with comments explaining the reservation.

**SequentialCalibration._no_check_metadata = True silences validation:**
- Files: `hammy_lib/sequential_calibration.py` (line 13), `hammy_lib/parallel_calibration.py` (line 12)
- Why fragile: Both calibration classes disable metadata conflict checking. This is intended (calibration may be re-run with same ID), but if someone adds a field to calibration results that shouldn't vary (e.g., hardware info), the class flag will silently skip validation.
- Safe modification: Document why `_no_check_metadata` is set. Consider using `_not_checked_fields` instead to allow selective validation of non-varying fields.

## Scaling Limits

**CFFI library compilation directory (`build_cffi/`) is global per process:**
- Current capacity: System /tmp or current directory disk space (usually GiB)
- Limit: If 100+ experiments are run in one session, each generates a separate CFFI .so file in build_cffi/. Disk fills or glob() becomes slow.
- Scaling path: (1) Use `tempfile.TemporaryDirectory()` per experiment, compile to temp dir, dlopen before temp cleanup; (2) Use in-memory compilation via RAM disk; (3) Implement CFFI library versioning and cleanup of old .so files.

**Multiprocessing Pool size is fixed at experiment creation time:**
- Current capacity: `threads = max(3, cores + 1)` means on a 256-core machine, creates 257 processes. Each process spawns CFFI interpreter, consuming ~100-300 MB.
- Limit: On memory-constrained systems (Colab with 12 GB), Pool creation may fail or OOM during first calibration.
- Scaling path: (1) Make Pool size configurable; (2) Use `ProcessPoolExecutor` with dynamic worker scaling; (3) Implement lazy worker startup (start 1 worker, add more as needed).

**xarray Dataset memory for high-dimensional results:**
- Current capacity: Simulation results xarray with dims [target, checkpoint, x, level, platform] at level 4 (5 levels) with 2 platforms = ~10M elements × 8 bytes = 80 MB per experiment. Acceptable.
- Limit: At level 6 (7 levels) with 4 platforms, becomes 500+ MB. At level 8 (9 levels) with GPU results, approaches GiB. Further calculations (PositionCalculation) create derived arrays of similar size.
- Scaling path: (1) Use float32 instead of float64 for counts (with validation that precision is acceptable); (2) Stream results to NetCDF incrementally instead of holding in memory; (3) Implement dimension-wise slicing in Calculation.calculate() instead of loading full results.

**Graph eigenvalue computation complexity O(n³):**
- Current capacity: LinearGraph with 100 nodes (BINS_LEN), eigenvalue decomposition via `np.linalg.eig()` is ~1M operations, ~1 ms.
- Limit: At 1000 nodes, becomes 1B operations, ~1 s. At 10k nodes, becomes 1T operations, may fail on memory.
- Scaling path: Use iterative eigenvalue solvers (scipy.sparse) if graph is sparse; use randomized eigenvalue approximation for large dense graphs.

## Dependencies at Risk

**CFFI compilation fragility:**
- Risk: CFFI caches compiled libraries in `.egg-info/cffi_modules` and `build_cffi/`. Changes to C code or C_DEFINITIONS may not recompile if CFFI thinks cache is valid. Manual deletion of build_cffi/ is required to force recompile.
- Impact: Development cycle is slow; mistakes in C code can persist across test runs.
- Migration plan: (1) Hash C code + constants, store in build_cffi/ filename; (2) Detect hash mismatch and force recompile; (3) Or use `ffi.compile(tmpdir=...)` with unique tmpdir per C code hash.

**CuPy (GPU support) is optional but not enforced:**
- Risk: Import of CuPy happens lazily inside `Experiment._get_cuda_kernel()` (line 113) and `cuda_simulator_launch()` (line 129). If CUDA is detected in MachineConfiguration but CuPy is not installed, failure occurs at simulation time (hours into Yandex Cloud run), not at startup.
- Impact: Job failure with no early warning.
- Migration plan: Check CuPy availability in `MachineConfiguration.calculate()` when CUDA is detected. Raise error immediately with helpful message: "CUDA detected but CuPy not installed. Install with: pip install cupy".

**h5netcdf compression parameters are hardcoded:**
- Risk: `ArrayHammyObject.dump_to_filename()` uses compression level 5 (line 208, 213). If zlib version changes or h5netcdf changes default behavior, old files may decompress differently.
- Impact: Low — zlib is stable, but hardcoded compression reduces flexibility if storage limits change.
- Migration plan: Move compression params to a configurable class variable: `HammyObject.NETCDF_COMPRESSION = {"zlib": True, "complevel": 5}`.

## Missing Critical Features

**No built-in result validation:**
- Problem: Simulation results are never validated against expected bounds. A buggy C kernel could produce negative counts, NaN, or inf without detection. Results are saved to S3 and used for analysis, potentially publishing incorrect conclusions.
- Blocks: Cannot validate without running the experiment (data-dependent validation).
- Recommendation: Add post-simulation sanity checks (e.g., `assert results >= 0`, `assert results.sum() == expected_loop_count * threads`).

**No cancellation/checkpointing mechanism:**
- Problem: Simulation at level 8 takes hours. If it crashes at level 7, the entire pipeline must restart from level 0. No way to resume mid-run.
- Blocks: Would require significant refactoring to checkpoint level results and resume from a checkpoint.
- Recommendation: Implement checkpoint files for each level; `Simulation.calculate()` should check for level N-1 checkpoint and resume from there if it exists.

**No result comparison/aggregation across runs:**
- Problem: Multiple runs of the same experiment (e.g., with different hardware or seeds) produce separate result files. No built-in way to aggregate results or statistically compare convergence.
- Blocks: Would require new Calculation subclass and multi-simulation handling.
- Recommendation: Implement `AggregatedCalculation` that takes multiple Simulation objects and computes statistics (mean, std, etc.) across runs.

## Test Coverage Gaps

**No unit tests for hammy_lib modules:**
- What's not tested: Individual components (Calculation, Simulation, Graph, etc.) tested in isolation. All tests (if any) are integration tests of the full pipeline.
- Files: No `tests/` directory exists.
- Risk: Refactoring is high-risk. A broken change to `hammy_object.py` caching logic might only be caught when a full experiment fails after hours of computation.
- Priority: High — unit tests for core classes (`HammyObject`, `Calculation`, `Graph`, `PositionCalculation`) would catch ~70% of potential bugs.

**No tests for calibration accuracy:**
- What's not tested: Whether sequential and parallel calibration results match expected behavior. No tests for edge cases (single-core machines, no GPU, etc.).
- Files: `hammy_lib/sequential_calibration.py`, `hammy_lib/parallel_calibration.py`
- Risk: Calibration tolerance logic (line 131 in walk.py) is chosen empirically with no validation.
- Priority: Medium — calibration is critical path for result correctness.

**No C code correctness tests:**
- What's not tested: The C kernel (walk.c) is never compared against a reference implementation in isolation. Only tested via full pipeline (Python reference is in walk.py, but integration with C is via simulation results, not direct comparison).
- Files: `experiments/01_walk/walk.c`, `experiments/01_walk/walk.h`
- Risk: A bug in walk.c (e.g., array indexing off-by-one) could silently produce incorrect distributions. Hard to catch because Python reference is also complex.
- Priority: High — add unit test in C that compares walk.c output to hand-calculated expected values for small T (e.g., T=10, 10 loops).

**No visualization regression tests:**
- What's not tested: Walk.py generates four visualizations (lines 158-204). PNG output is never validated. No tests for filters, references, or groupby logic.
- Files: `experiments/01_walk/walk.py`, `hammy_lib/vizualization.py`
- Risk: Visualization changes silently (e.g., filter doesn't apply, reference curve is wrong). Not caught until human reviews output.
- Priority: Medium — add pytest fixtures to generate test data and compare PNG output (hash or pixel-level with tolerance).

**No error injection tests:**
- What's not tested: Behavior when Yandex Cloud storage is unreachable, when S3 upload fails, when CFFI compilation fails.
- Files: `hammy_lib/yandex_cloud_storage.py`, `hammy_lib/experiment.py`
- Risk: Runtime failure modes are untested. Graceful degradation (skip upload) is assumed but never verified.
- Priority: Low — nice-to-have for robustness, but not blocking.

---

*Concerns audit: 2026-03-24*
