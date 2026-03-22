# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

The docker setup implements a **self-destructing GPU VM** pattern on Yandex Cloud: a preemptible VM is created, runs the experiment inside a container, then deletes itself and powers off when done.

## Two-layer image hierarchy

- **`hammy-base` (`hammy-base.Dockerfile`)** — base image built on `cupy/cupy:v13.1.0` (CUDA + Python). Installs `yc` CLI, Yandex Unified Agent (monitoring), and copies the helper scripts. `countdown.sh` is installed as `/root/main.sh` (placeholder — overridden in the experiment image).
- **`hammy` (`hammy.Dockerfile`)** — experiment image built on top of base. Installs Python deps, downloads pcg-c-basic from pcg-random.org, copies the experiment script and encrypted S3 credentials. Replaces `main.sh` with `hammy.sh`.

## VM lifecycle (`hammy_entrypoint.sh`)

1. Starts Yandex Unified Agent (background, for metrics streaming)
2. Configures `yc` to use the VM's instance service account (no credentials needed)
3. Retrieves the instance ID via the GCP-compatible metadata endpoint (`169.254.169.254`) — Yandex Cloud exposes this same endpoint
4. Runs `main.sh` wrapped with `log_yc` (see below)
5. **Self-destructs**: deletes the VM (`yc compute instance delete --async`) and triggers kernel shutdown via sysrq (`echo o > /proc/sysrq-trigger`)

**To prevent self-destruction** (e.g. for debugging), set environment variables:
- `ND=1` — skip VM deletion
- `NS=1` — skip kernel shutdown (sysrq poweroff)

## Experiment entry point (`hammy.sh`)

Runs inside the container as `main.sh`. Steps:
1. Decrypts S3 credentials from KMS-encrypted `.cipher` files:
   ```
   yc kms symmetric-crypto decrypt --name hammy --ciphertext-file s3_access_key_id.cipher
   ```
   The ciphertext files are checked into the repo; decryption requires the VM's service account to have KMS decrypt permission on the `hammy` key.
2. Patches `.s3cfg` in-place (`sed -i`) with the decrypted key ID and secret key.
3. Runs the Python experiment via `run_nice` (see below).

## Helper scripts

**`run_nice.sh`** — runs a command with `OOM score 1000` (first killed under memory pressure) and `nice -n 10` (reduced CPU priority). Uses a subshell + `exec` trick to apply both adjustments to the same process:
```bash
(echo 1000 > /proc/self/oom_score_adj && exec nice -n 10 "$@")
```

**`log_yc.sh`** — wraps a command and streams its output to Yandex Cloud Logging (`yc logging write --group-name=hammy-compute`). stdout → INFO level, stderr → ERROR level. Features built-in rate limiting to avoid flooding the logging API:
- `LYC_T` — minimum seconds between log writes per level (default: 5)
- `LYC_ML` — max consecutive lines before throttling kicks in (default: 50); throttled messages get a `...~50` suffix
- `DISABLE_YC_LOG=true` — suppresses actual API calls (useful for local testing while keeping the counting/formatting)

Each log line gets a `(NNNNNN)` global counter prefix.

**`countdown.sh`** — placeholder `main.sh` in the base image. Counts down 5→1 and exits. Not used in production.

## Compose template (`hammy_machine.compose`)

`hammy_machine.compose` is **not a valid compose file** — it contains two placeholders:
- `XXX` → image version (e.g. `4.1`)
- `YYY` → command argument (minutes to run)

`create_hammy_machine.sh` performs `sed` substitution into a temp file before passing to `yc compute instance create-with-container`. The compose spec sets `privileged: true` and reserves 1 NVIDIA GPU.

## Creating a VM (`create_hammy_machine.sh`)

```bash
./create_hammy_machine.sh <version> <minutes>
```

Creates a preemptible Yandex Cloud VM:
- **Spec**: 8 vCPU, 1 GPU, 48 GB RAM, `gpu-standard-v2`, zone `ru-central1-a`
- **`--async`**: returns immediately without waiting for the VM to start
- Service account `compute` must have KMS decrypt permission for the `hammy` key

## Monitoring (`ymonitoring.yml`)

Yandex Unified Agent config, copied into the base image at `/etc/yandex/unified_agent/config.yml`. Streams Linux system metrics (CPU, memory, IO, network, kernel, storage) to Yandex Cloud Monitoring every 1s, buffered locally in `/var/lib/yandex/unified_agent/main` (100 MB max).
