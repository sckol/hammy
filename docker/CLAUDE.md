# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

The docker setup implements a **self-destructing GPU VM** pattern on Yandex Cloud: a preemptible VM is created, runs the experiment inside a container, then deletes itself and powers off when done. On failure, the VM stays alive for 5 minutes for SSH debugging, then self-destructs.

## Image (`hammy.Dockerfile`)

Single-layer image built on `cupy/cupy:v13.1.0` (CUDA + Python). Installs:
- `yc` CLI (for self-destruct only)
- PCG random C library (simulation kernel dependency)
- Python deps via offline wheels (avoids PyPI connectivity issues in Docker build)
- `hammy_lib/` and `experiments/` copied from repo

## VM lifecycle (`entrypoint.sh`)

1. Configures `yc` for instance metadata auth
2. Gets instance ID from metadata endpoint (`169.254.169.254`)
3. Runs `python3 -m experiments.01_walk "$@"` with `tee` to `/root/experiment.log`
4. On success (exit 0): self-destructs immediately
5. On failure (exit != 0): waits 5 minutes (SSH debugging window), then self-destructs

S3 credentials are passed as container environment variables (`S3_ACCESS_KEY`, `S3_SECRET_KEY`), injected by `create_hammy_machine.sh` from `~/secrets.env`.

## Compose templates

Two templates with placeholders (`XXX`=version, `YYY`=args, `AAA`/`SSS`=S3 keys):

- `hammy_machine.compose` — GPU mode (8 cores, 1 GPU, 48G RAM, `gpu-standard-v2`)
- `hammy_machine_cpu.compose` — CPU-only mode (2 cores, 4G RAM, `standard-v3`) for cheap testing

## Creating a VM (`create_hammy_machine.sh`)

```bash
./create_hammy_machine.sh <version> "<experiment-args>"          # GPU
./create_hammy_machine.sh <version> "<experiment-args>" --cpu    # CPU only
```

Examples:
```bash
./create_hammy_machine.sh 5.1 "--level 2 --no-calculations"          # GPU, levels 0-2
./create_hammy_machine.sh 5.1 "--level 0 --no-calculations" --cpu    # CPU test
./create_hammy_machine.sh 5.1 "--level 4 --no-calculations"          # GPU, resumes from S3 cache
```

## Level resumption

S3 storage is configured at the top of `run()`, before any objects are resolved. This enables auto-download of cached results: running `--level 4` after a previous `--level 2` run automatically downloads levels 0-2 from S3 and computes only levels 3-4.

## Building and pushing

```bash
# Download wheels (only needed when deps change)
pip download --dest docker/wheels --python-version 3.10 --only-binary=:all: \
    --platform manylinux2014_x86_64 --platform manylinux_2_17_x86_64 \
    xarray cffi psutil boto3 matplotlib h5netcdf scs

# Build and push
docker build -f docker/hammy.Dockerfile -t sckol/hammy:<version> .
docker push sckol/hammy:<version>

# Clean up wheels (don't commit them)
rm -rf docker/wheels
```
