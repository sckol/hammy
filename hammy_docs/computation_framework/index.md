# Computation Framework

## Architecture

**hammy** runs Monte Carlo random walk simulations across three platforms in parallel:

| Platform | Implementation | Typical throughput |
|---|---|---|
| PYTHON | NumPy vectorized | ~2,000 loops/min |
| CFFI | C via CFFI (CPU) | ~800,000 loops/min |
| CUDA | C via CuPy RawKernel (GPU) | ~450,000 loops/min (V100) |

Each platform produces an independent dataset. Results are compared to detect bugs, then aggregated for final analysis.

## Dual-language simulation

Each experiment is written once in C using a macro layer (`cuda_cpu.h`) that compiles to both CPU and GPU:

- **CPU mode**: CUDA's warp (32 threads) is emulated as a sequential loop. `__shared__` becomes a local variable, `__syncthreads()` restarts the loop.
- **GPU mode**: Real warps, real shared memory, real atomic operations. Same source file, different `#define`s.

The Python implementation serves as a reference. If all three platforms agree on results (checked via the population size metric $1/\phi$), the simulation is correct.

## Pipeline objects

Objects form a dependency chain, each caching its results as content-addressed files:

```
Experiment → MachineConfiguration → ExperimentConfiguration
    → SequentialCalibration → ParallelCalibration → Simulation
        → Calculation → Visualization
```

Each object has a deterministic ID derived from its metadata (including all upstream dependencies). If a cached file with that ID exists locally or on S3, it is loaded instead of recomputed. This enables:
- **Level resumption**: `--level 4` after a previous `--level 2` downloads levels 0-2 from S3
- **Cross-machine reuse**: calibration and experiment configs are cached per machine hash
- **Incremental development**: changing a calculation doesn't invalidate the simulation cache

## Simulation levels

Each level $\ell$ runs for $2^{\ell-1}$ minutes (level 0 = 1 min, level 4 = 8 min). Results accumulate: level $\ell$ contains the sum of all data from levels $0 \ldots \ell$. Total simulation time for level $N$: $\sum_{\ell=0}^{N} 2^{\max(\ell-1, 0)}$ minutes.

| Level | Minutes | Cumulative | Typical data points (V100) |
|---|---|---|---|
| 0 | 1 | 1 | 17B |
| 1 | 1 | 2 | 35B |
| 2 | 2 | 4 | 69B |
| 3 | 4 | 8 | 136B |
| 4 | 8 | 16 | 272B |

## Cloud execution

Experiments run on Yandex Cloud preemptible VMs via Docker containers. The VM self-destructs on success; on failure it stays alive for 5 minutes for SSH debugging.

### Usage

```bash
# GPU (V100, 8 cores, 48G RAM)
./create_hammy_machine.sh 5.1 "--level 4 --no-calculations"

# CPU only (2 cores, 4G RAM) — for testing
./create_hammy_machine.sh 5.1 "--level 0 --no-calculations" --cpu
```

S3 credentials are read from `~/secrets.env` and injected as container environment variables.

### Image build

```bash
pip download --dest docker/wheels --python-version 3.10 --only-binary=:all: \
    --platform manylinux2014_x86_64 --platform manylinux_2_17_x86_64 \
    xarray cffi psutil boto3 matplotlib h5netcdf h5py scs
docker build -f docker/hammy.Dockerfile -t sckol/hammy:<version> .
docker push sckol/hammy:<version>
rm -rf docker/wheels
```

Wheels are downloaded offline because Docker builds have unreliable PyPI connectivity.
