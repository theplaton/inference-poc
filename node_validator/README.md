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
image carries only those two binaries, `validate.sh`, and the CUDA/cuBLAS/NCCL
runtimes. No nvcc, no gcc, no Python, no PyTorch, no MPI, no DCGM.

For non-Hopper/Ampere hardware, set the arch list:

```bash
docker build --build-arg CUDA_ARCHS="90 100" -t gpu-validator .
```

## Run

```bash
docker run --rm \
    --gpus all \
    --ipc=host \
    -e EXPECTED_GPU_COUNT=8 \
    -e EXPECTED_GPU_NAME="H200" \
    -e MIN_GEMM_TFLOPS=800 \
    -e MIN_NCCL_BUSBW_GBPS=300 \
    -v "$PWD/results:/results" \
    gpu-validator
```

`make run` does the same with dev defaults (thresholds unset). No `--privileged`
needed. `--ipc=host` is required — NCCL uses host shared memory for intra-node
transport. Exit code: `0` = PASS, `1` = FAIL.

If `--gpus all` fails with `AMD CDI spec not found` or `no known GPU vendor
found`, the `--gpus` shorthand is misdetecting the vendor (seen on Docker 29 with
CDI). Address the GPUs explicitly instead:

```bash
docker run --rm --device nvidia.com/gpu=all --ipc=host ... gpu-validator
make run GPU_FLAG="--device nvidia.com/gpu=all"
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
  GPU0:                806.0 TFLOPS (99% median)  PASS
  GPU1:                811.7 TFLOPS (100% median) PASS
  GPU2:                802.1 TFLOPS (99% median)  PASS
  GPU3:                817.0 TFLOPS (101% median) PASS
  GPU4:                801.6 TFLOPS (99% median)  PASS
  GPU5:                813.4 TFLOPS (100% median) PASS
  GPU6:                817.5 TFLOPS (101% median) PASS
  GPU7:                821.1 TFLOPS (101% median) PASS

  Median:              812.5 TFLOPS
  Relative floor:      90%
  Absolute floor:      700 TFLOPS

Multi-GPU
  NCCL AllReduce
  busbw:               468.69 GB/s                PASS

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
