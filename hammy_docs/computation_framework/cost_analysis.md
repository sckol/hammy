# Cost Analysis

## Throughput comparison

Data from Experiment 1 (walk), level 0 (1 minute of simulation):

| Platform | Colab T4 | YC V100 | YC CPU (2-core) |
|---|---|---|---|
| PYTHON | 359K dp/min | 388K dp/min | 388K dp/min |
| CFFI | 21M dp/min | 100M dp/min | 66M dp/min |
| CUDA | 2.5B dp/min | 17.3B dp/min | — |
| **Total** | **2.6B dp/min** | **17.4B dp/min** | **66M dp/min** |

CUDA dominates throughput by 2-3 orders of magnitude over CFFI.

## Cost per data point

Yandex Cloud preemptible VM pricing (ru-central1, approximate):

| Config | Spec | Price | Throughput | Cost per 1B dp |
|---|---|---|---|---|
| GPU `gpu-standard-v2` | 8 cores, V100, 48G | ~180 RUB/hr | 17.4B dp/min | **0.17 RUB** |
| CPU `standard-v3` | 2 cores, 4G | ~2.5 RUB/hr | 66M dp/min | **0.63 RUB** |

**GPU is ~3.7x cheaper per data point** despite being 72x more expensive per hour, because the V100 produces 264x more data points per minute.

## Overhead

Fixed costs per VM launch (not producing data points):

| Phase | GPU VM | CPU VM |
|---|---|---|
| Boot + image pull | ~5 min | ~10 min |
| Calibration (first run) | ~10 min | ~5 min |
| Calibration (cached) | 0 | 0 |

Calibration results are cached on S3 and reused across runs with the same machine type.

## Practical recommendations

For Experiment 1 (1D walk, 101-node graph):

- **Level 2** (4 min simulation, ~69B dp on V100): sufficient for sub-node position precision on non-zero targets. Cost: ~$\text{180} \times \frac{4 + 5}{60} \approx 27$ RUB including boot overhead.
- **Level 4** (16 min simulation, ~272B dp on V100): bootstrap std $< 0.001$ nodes. Cost: ~$\text{180} \times \frac{16 + 5}{60} \approx 63$ RUB.
- **CPU-only** testing: use `--cpu` for quick validation before committing to a GPU run. Level 0 costs ~$\text{2.5} \times \frac{1 + 15}{60} \approx 0.7$ RUB.

For future experiments on larger graphs (2D/3D lattices), the graph size does not affect simulation throughput (the walk kernel is graph-independent). However, the position calculation (NNLS on $n \times n$ matrix) scales as $O(n^3)$ — for a $100 \times 100$ 2D lattice ($n = 10{,}000$), NNLS becomes the bottleneck. Run NNLS locally, not on GPU VMs.
