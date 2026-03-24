# External Integrations

**Analysis Date:** 2026-03-24

## APIs & External Services

**Yandex Cloud:**
- VM provisioning and management
  - SDK/Client: `yc` CLI (binary at `/usr/local/bin/yc`)
  - Auth: Service account (instance metadata; no explicit credentials on VM)
- KMS encryption/decryption
  - Purpose: Decrypt S3 credentials stored in `docker/s3_access_key_id.cipher` and `docker/s3_secret_access_key.cipher`
  - Auth: Service account `compute` must have role `kms.keys.decrypter` on key named `hammy`
  - Command: `yc kms symmetric-crypto decrypt --name hammy --ciphertext-file <file>`
- Cloud Logging
  - Purpose: Stream experiment stdout/stderr and metrics to Yandex Cloud Logging
  - SDK/Client: `yc logging write` (via `log_yc.sh` script)
  - Command: `yc logging write --group-name=hammy-compute --message=<msg> --level=<INFO|ERROR>`
- Cloud Monitoring
  - Purpose: Collect Linux system metrics (CPU, memory, IO, network, storage)
  - SDK/Client: Yandex Unified Agent (daemon process)
  - Config: `/etc/yandex/unified_agent/config.yml` (copied from `docker/ymonitoring.yml`)
  - Metrics: streamed every 1 second, buffered locally in `/var/lib/yandex/unified_agent/main`

**Yandex Cloud Object Storage (S3-compatible):**
- Endpoint: `https://storage.yandexcloud.net`
- Region: `ru-central1`
- Bucket: `hammy`
- Purpose: Persist simulation results and intermediate calculations
  - SDK/Client: `boto3.session.Session(...).client('s3', endpoint_url='https://storage.yandexcloud.net')`
  - Auth: AWS access key ID + secret key (stored encrypted in `.cipher` files, decrypted at runtime via KMS)
  - Operations:
    - `head_object(Bucket='hammy', Key=...)` - check object existence
    - `upload_file(local_path, Bucket='hammy', Key=...)` - upload file
    - `download_file(Bucket='hammy', Key=..., local_path)` - download file
  - File types:
    - `.json` - metadata (experiment params, calibration results)
    - `.nc` - xarray results (simulation data, calculations) in NetCDF4 format

## Data Storage

**Databases:**
- None; all data stored as files

**File Storage:**
- Local filesystem (primary): `results/` directory (configurable via `HammyObject.RESULTS_DIR`)
  - Structure: `results/{experiment_number}_{experiment_name}_{experiment_version}/{id}.{json|nc}`
  - Examples:
    - `results/1_walk_1/config_abc123.json` - experiment configuration
    - `results/1_walk_1/simulation_def456.nc` - simulation results (xarray DataArray)
    - `results/1_walk_1/calculation_ghi789.nc` - post-processed calculation output
- Yandex Cloud Object Storage (S3): `hammy` bucket
  - Objects synced via `YandexCloudStorage.upload()` and `YandexCloudStorage.download()`
  - Uses boto3 client with Yandex-compatible endpoint

**Caching:**
- Content-addressed caching: object IDs derived from metadata SHA256 hash
- Mechanism: `HammyObject.generate_digest(metadata_string)` → 6-char hex ID
- On resolve: checks for existing file at `results/{id}.{json|nc}`; if missing and `STORAGE` is set, attempts S3 download
- If neither exists: `calculate()` runs and result is dumped to file

## Authentication & Identity

**Auth Provider:**
- Yandex Cloud service accounts (instance identity)
  - No explicit credentials on VM — uses instance metadata endpoint (169.254.169.254)
  - Service account name: `compute` (required permission: KMS decrypt)
  - For S3: credentials are KMS-encrypted in Docker image; decrypted at container startup via `yc kms symmetric-crypto decrypt`

**S3 Credentials:**
- Storage location: Encrypted `.cipher` files in Docker image (`docker/s3_access_key_id.cipher`, `docker/s3_secret_access_key.cipher`)
- Decryption: `yc kms symmetric-crypto decrypt --name hammy --ciphertext-file <file>` (requires KMS key access)
- Injection: `hammy.sh` script decrypts and patches `.s3cfg` via `sed -i`
- Library: `boto3.session.Session(aws_access_key_id=..., aws_secret_access_key=...)`

## Monitoring & Observability

**Error Tracking:**
- Not detected; errors are logged to stdout/stderr

**Logs:**
- Local: stdout/stderr captured during experiment runs
- Cloud:
  - Yandex Cloud Logging: `log_yc.sh` wrapper sends experiment output to `yc logging write`
    - Group name: `hammy-compute`
    - Levels: INFO (stdout) / ERROR (stderr)
    - Rate limiting: 5 sec minimum between writes per level (configurable `LYC_T`)
  - Yandex Cloud Monitoring: Unified Agent streams system metrics (CPU%, memory%, IO, network, storage)
    - Metrics interval: 1 second
    - Buffer: local `/var/lib/yandex/unified_agent/main` (100 MB max)

**Metrics:**
- Automatic collection via Yandex Unified Agent:
  - CPU usage, cores, context switches
  - Memory usage (free, used, cached, buffers)
  - Disk IO (read/write bytes, operations)
  - Network IO (bytes in/out, packets)
  - Storage filesystem metrics
  - Kernel messages

## CI/CD & Deployment

**Hosting:**
- Yandex Cloud Compute Engine (preemptible VMs)
- Container runtime via Docker Compose
- Auto-scaling: per-experiment VM creation via `create_hammy_machine.sh` script

**Provisioning:**
- `create_hammy_machine.sh` — Bash script that:
  1. Substitutes image version and experiment duration into `docker/hammy_machine.compose`
  2. Calls `yc compute instance create-with-container` with compose file
  3. Returns immediately (`--async` flag)
  4. Prints instance details (IP, instance ID)

**Deployment Pipeline:**
- **Local → Yandex Cloud:**
  1. Experiment code checked into git
  2. Docker image built and pushed to `sckol/hammy:version` (registry: Docker Hub or internal)
  3. `create_hammy_machine.sh <version> <minutes>` provisions VM with image
  4. VM auto-configures via `hammy_entrypoint.sh`
  5. Container runs experiment (`hammy.sh`)
  6. Results uploaded to S3 (`YandexCloudStorage.upload()`)
  7. VM self-destructs (`yc compute instance delete --async`)

**Image Versions:**
- Versioning: Dockerfile image tags like `sckol/hammy:4.0fix`, `sckol/hammy:4.1`
- Passed to `create_hammy_machine.sh` as first arg (e.g., `./create_hammy_machine.sh 4.1 60`)

## CI/CD Integration

**Not detected** — no GitHub Actions, GitLab CI, or Jenkins found. Experiments run manually via:
- Local: `python experiments/01_walk/walk.py [--flags]`
- Google Colab: `walk.py` notebook in browser
- Yandex Cloud: `./create_hammy_machine.sh <version> <minutes>`

## Environment Configuration

**Required env vars (local):**
- `CCODE` (optional) - Inline C source for Colab (if not set, reads from `walk.c`)
- `LD_PRELOAD` (optional, auto-set) - Intel MKL library path (for accelerated NumPy)

**Required env vars (Docker/Cloud):**
- Service account metadata (injected by Yandex Cloud):
  - Instance ID, project ID, service account ID (retrieved from 169.254.169.254)
- Optional:
  - `ND=1` - Skip VM deletion on completion
  - `NS=1` - Skip kernel shutdown via sysrq
  - `DISABLE_YC_LOG=true` - Skip actual `yc logging write` calls (for local testing)

**Secrets location:**
- Local: Yandex Cloud service account credentials (via `~/.ssh/id_rsa.pub` in `create_hammy_machine.sh`)
- Docker: KMS-encrypted S3 credentials (`docker/s3_access_key_id.cipher`, `docker/s3_secret_access_key.cipher`)

**S3 Configuration:**
- File: `.s3cfg` (s3cmd config format)
- Credentials patched at runtime by `hammy.sh` via `sed -i`
- Used by Python via boto3, not s3cmd directly

## Webhooks & Callbacks

**Incoming:**
- None detected

**Outgoing:**
- None detected
- Note: VM self-destruction (`yc compute instance delete --async`) is not a webhook but a programmatic cleanup call made by `hammy_entrypoint.sh`

## Network & Connectivity

**Yandex Cloud VM networking:**
- Public IP (assigned via `--public-ip` flag in `create_hammy_machine.sh`)
- Metadata endpoint: 169.254.169.254 (instance identity)
- Outbound: Object Storage (`storage.yandexcloud.net`), Logging API, Monitoring API, KMS API

---

*Integration audit: 2026-03-24*
