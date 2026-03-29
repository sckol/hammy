# Cost Analysis

## Throughput by platform

Data from Experiment 1 (1D walk, 101-node graph), level 0 (1 minute of simulation):

| Platform | Colab T4 | YC V100 | YC CPU 2-core |
|---|---|---|---|
| PYTHON | 359K dp/min | 388K dp/min | 372K dp/min |
| CFFI | 21M dp/min | 100M dp/min | 66M dp/min |
| CUDA | 2.5B dp/min | 17.3B dp/min | — |
| **Total** | **2.6B dp/min** | **17.4B dp/min** | **66M dp/min** |

CUDA dominates by 2-3 orders of magnitude. On the V100 machine, the GPU produces 177x more data points per minute than the same machine's 8 CPU cores running CFFI.

## Cost per data point

Yandex Cloud preemptible VM pricing (ru-central1, approximate):

| Config | Spec | RUB/hr | dp/min | RUB per 1B dp |
|---|---|---|---|---|
| GPU `gpu-standard-v2` | 8c, V100, 48G | ~180 | 17.4B | **0.17** |
| CPU `standard-v3` | 4c, 8G | ~5 | 132M | **0.63** |
| CPU `standard-v3` | 2c, 4G | ~2.5 | 66M | **0.63** |
| CPU `standard-v3` | 8c, 16G | ~10 | 84M | **1.98** |

**GPU is 3.7x cheaper per data point** than the best CPU config (4-core), despite being 36x more expensive per hour.

### Why 4-core CPU beats 8-core

CFFI throughput scales linearly with cores, but cost also scales linearly. The cost per data point is the same for 2c and 4c. However, the 8-core CPU is *more expensive per data point* because the V100 machine's 8-core CFFI benchmark (877K loops/min) reflects better hardware than a standard-v3 8-core VM. The actual `standard-v3` 8-core would likely match ~2x the 4-core throughput at 2x the cost — same efficiency.

The fixed overhead (boot, image pull, calibration) favors larger machines: a 4-core VM finishes the same work in half the wall-clock time as a 2-core, with the same total cost but less time exposed to preemption risk.

## Including overhead

For a level-4 run (16 minutes of simulation):

| Config | Overhead | Total time | Total cost | Data points | Effective RUB/1B dp |
|---|---|---|---|---|---|
| V100 GPU | 5 min | 21 min | 63 RUB | 283B | **0.22** |
| CPU 4c/8G | 15 min | 31 min | 2.6 RUB | 2.1B | **1.22** |
| CPU 2c/4G | 15 min | 31 min | 1.3 RUB | 1.1B | **1.22** |

GPU overhead is lower (5 min vs 15 min) because GPU VMs boot faster and calibration is cached from previous runs. CPU VMs have longer image pull times (7.6 GB image on limited bandwidth).

## When to use each

| Scenario | Recommendation |
|---|---|
| Quick validation | CPU 2c/4G `--cpu --level 0` (~1.3 RUB) |
| Development iteration | CPU 4c/8G `--cpu --level 1` (~2.6 RUB) |
| Production run | GPU V100 `--level 4` (~63 RUB for 283B dp) |
| Maximum data | GPU V100 `--level 6` (~250 RUB for ~1.1T dp) |

## Scaling to larger graphs

The simulation kernel throughput is **independent of graph size** — the walk kernel only needs the transition probabilities at the current node, not the full graph. However:
- **Position calculation** (NNLS on $n \times n$ matrix) scales as $O(n^3)$ — this should run locally, not on GPU VMs
- **Bootstrap** ($B \times$ NNLS) also scales with graph size — precompute $T^p$ and bootstrap only the NNLS step
- For a 2D lattice with $n = 10{,}000$ nodes, NNLS becomes the bottleneck. Consider sparse solvers for future experiments.
