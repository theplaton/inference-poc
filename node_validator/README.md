# gpu-validator

Quick preflight check for a freshly provisioned **single-node** NVIDIA GPU machine,
before you hand it to a training or inference workload. One small container, three
tests, one PASS/FAIL verdict.

It answers only:

1. Did we get the GPUs we asked for?
2. Does each GPU deliver roughly the expected compute?
3. Does multi-GPU communication run at roughly the expected bandwidth?

## Requirements

- NVIDIA datacenter GPU(s) + host driver (`nvidia-smi` works on the host)
- Docker
- **NVIDIA Container Toolkit** — without it `--gpus all` fails with
  `failed to discover GPU vendor from CDI: no known GPU vendor found`. The image
  ships cuBLAS/NCCL and the two benchmark binaries, but the driver-side pieces they
  need (`libcuda.so.1`, `libnvidia-ml.so.1`, `nvidia-smi`, `/dev/nvidia*`) are
  version-locked to the host kernel driver and must be injected at runtime — that
  injection is what the toolkit does.

  ```bash
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
    | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
    | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
    > /etc/apt/sources.list.d/nvidia-container-toolkit.list
  apt-get update && apt-get install -y nvidia-container-toolkit
  nvidia-ctk runtime configure --runtime=docker && systemctl restart docker
  # no systemd (e.g. dockerd started by hand)? use CDI instead of the runtime hook:
  #   nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml
  ```

  Verify: `docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu24.04 nvidia-smi -L`
  (if that errors on vendor detection, see the `--device` note under [Run](#run))

`make build` needs none of the above — nvcc cross-compiles. Only `make run` does.

## Tests

```text
nvidia-smi        verifies requested GPU inventory/configuration
BF16 GEMM         verifies individual GPU Tensor Core compute performance
NCCL AllReduce    verifies multi-GPU communication performance
```

- **GEMM**: `cuda/gemm_bench.cu`, cuBLAS `cublasGemmEx`, BF16 in / FP32 accumulate,
  16384³ by default, 5 warmup + 15 timed iterations, run on one GPU at a time.
  TFLOPS = `2·M·N·K / avg_seconds / 1e12`.
- **NCCL**: pinned `nccl-tests` `all_reduce_perf -b 8M -e 1G -f 2 -g <gpu_count>`,
  built without MPI (single node). Reported `busbw` is the out-of-place value at the
  largest message size (1 GiB).

Compute is judged two ways: an optional absolute floor, and a **relative** floor
against the median of all GPUs — which catches one abnormal GPU even when no SKU
baseline is configured.

## Build

```bash
make build          # docker build -t gpu-validator .
```

Multi-stage: the builder compiles `gemm_bench` and `all_reduce_perf`; the final
image carries only those two binaries, `validate.sh`, and the three libraries they
link (cudart, cuBLAS/cuBLASLt, NCCL). No nvcc, no gcc, no Python, no PyTorch,
no MPI, no DCGM. Result: **2.57 GB on disk / 993 MB to pull.**

It starts from the `-base` image and adds `libcublas` + `libnccl2` explicitly,
rather than starting from `-runtime` and inheriting the `cuda-libraries`
metapackage — that would drag in cuFFT, cuSPARSE, cuSOLVER, cuRAND, NPP and nvRTC
for a 5.59 GB / 2.16 GB image, none of which this validator calls.

Build args:

| Arg | Default | Notes |
| --- | --- | --- |
| `CUDA_ARCHS` | `80 90` | Ampere + Hopper. Blackwell: `"90 100"` |
| `CUDA_VERSION` | `12.8.1` | Selects both base images; `libcublas-12-8` is derived from it |
| `NCCL_PKG_VERSION` | `2.25.1-1+cuda12.8` | Runtime NCCL. **Bump together with `CUDA_VERSION`** — it is pinned to match the NCCL the builder linked against |
| `NCCL_TESTS_TAG` | `v2.19.7` | `nccl-tests` source pin |

```bash
docker build --build-arg CUDA_ARCHS="90 100" -t gpu-validator .
```

If a future CUDA bump makes the pinned pair inconsistent, the fallback is a
one-liner: change the final stage's `FROM` to `-runtime-` and drop the `apt-get`
line — bigger image, no version bookkeeping.

## Run

**The GPUs must be idle.** Any other workload holding them (a served model, a
training job) time-slices against the benchmarks: compute roughly halves and NCCL
fails with `unhandled cuda error`, because the other process already owns the
NVLink communicators. Check with `nvidia-smi --query-compute-apps=pid,process_name
--format=csv` before running — this is a preflight tool for an *unloaded* node.

```bash
docker run --rm \
    --device nvidia.com/gpu=all \
    --ipc=host \
    -e EXPECTED_GPU_COUNT=8 \
    -e EXPECTED_GPU_NAME="H200" \
    -e MIN_GEMM_TFLOPS=750 \
    -e MIN_NCCL_BUSBW_GBPS=300 \
    -v "$PWD/results:/results" \
    gpu-validator
```

`make run` does the same with dev defaults (thresholds unset). No `--privileged`
needed. `--ipc=host` is required — NCCL uses host shared memory for intra-node
transport. Exit code: `0` = PASS, `1` = FAIL.

`--device nvidia.com/gpu=all` is the CDI form and works everywhere the NVIDIA
Container Toolkit is installed. The shorter `--gpus all` is equivalent on most
hosts, but on Docker 29 + CDI its vendor auto-detection can fail with
`AMD CDI spec not found` or `no known GPU vendor found` — hence the explicit form
above. With make:

```bash
make run GPU_FLAG="--device nvidia.com/gpu=all" \
  EXPECTED_GPU_COUNT=8 EXPECTED_GPU_NAME=H200 \
  MIN_GEMM_TFLOPS=750 MIN_NCCL_BUSBW_GBPS=300
```

### Environment variables

| Variable | Default | Effect |
| --- | --- | --- |
| `EXPECTED_GPU_COUNT` | unset | Visible GPU count must equal it; unset = report only |
| `EXPECTED_GPU_NAME` | unset | Every GPU name must contain it; unset = report only |
| `MIN_GEMM_TFLOPS` | unset | Absolute per-GPU floor; unset = relative check only |
| `MIN_NCCL_BUSBW_GBPS` | unset | Absolute busbw floor; unset = successful run is PASS |
| `RELATIVE_GPU_FLOOR` | `0.90` | Each GPU must reach this fraction of the median TFLOPS |

The thresholds above are **examples** — nothing SKU-specific is baked in.

## Results

```text
results/
├── report.txt         final human-readable report (also printed to stdout)
├── nvidia-smi.txt     nvidia-smi -L plus index,uuid,name,memory,bus_id,temp CSV
├── topology.txt       nvidia-smi topo -m
├── gemm.txt           raw per-GPU GEMM lines: GPU=0 TFLOPS=... AVG_MS=...
└── nccl.txt           raw all_reduce_perf output
```

## Example result

Real container run on an 8× H200 node (`results/example-report.txt`), thresholds
`MIN_GEMM_TFLOPS=700`, `MIN_NCCL_BUSBW_GBPS=300`:

```text
====================================================
 NVIDIA GPU NODE VALIDATION
====================================================

System
  GPU count:           8                          PASS
  GPU model:           NVIDIA H200                PASS

Per-GPU Compute
  GPU0:                813.1 TFLOPS (100% median) PASS
  GPU1:                812.6 TFLOPS (100% median) PASS
  GPU2:                799.1 TFLOPS (98% median)  PASS
  GPU3:                803.8 TFLOPS (99% median)  PASS
  GPU4:                817.1 TFLOPS (101% median) PASS
  GPU5:                811.7 TFLOPS (100% median) PASS
  GPU6:                809.6 TFLOPS (100% median) PASS
  GPU7:                811.5 TFLOPS (100% median) PASS

  Median:              811.6 TFLOPS
  Relative floor:      90%
  Absolute floor:      700 TFLOPS

Multi-GPU
  NCCL AllReduce
  busbw:               468.68 GB/s                PASS

----------------------------------------------------
 OVERALL: PASS
====================================================
```

## Limitations

This MVP does **not** validate:

```text
HBM bandwidth independently
historical XIDs
full GPU health (ECC, retired pages, link errors)
power/thermal behavior under extended load
P2P bandwidth matrix
multi-node networking
application/model performance
```

Those, plus DCGM, MIG validation, and structured JSON output, are Phase 2.

`make clean` removes the generated files in `results/` (keeps `example-report.txt`).
