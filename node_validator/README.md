# gpu-validator

Quick preflight check for a freshly provisioned **single-node** NVIDIA GPU machine,
before you hand it to a training or inference workload. One small container, three
tests, one PASS/FAIL verdict.

It answers only:

1. Did we get the GPUs we asked for?
2. Does each GPU deliver roughly the expected compute?
3. Does multi-GPU communication run at roughly the expected bandwidth?

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
needed. Exit code: `0` = PASS, `1` = FAIL.

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

Real run on an 8× H200 node (`results/example-report.txt`). Docker was not
installed on that host, so `validate.sh` was executed directly against natively
compiled copies of the same two binaries — identical code path, thresholds
`MIN_GEMM_TFLOPS=700`, `MIN_NCCL_BUSBW_GBPS=300`:

```text
====================================================
 NVIDIA GPU NODE VALIDATION
====================================================

System
  GPU count:           8                          PASS
  GPU model:           NVIDIA H200                PASS

Per-GPU Compute
  GPU0:                799.2 TFLOPS (99% median)  PASS
  GPU1:                807.0 TFLOPS (100% median) PASS
  GPU2:                809.6 TFLOPS (100% median) PASS
  GPU3:                807.8 TFLOPS (100% median) PASS
  GPU4:                802.0 TFLOPS (99% median)  PASS
  GPU5:                812.6 TFLOPS (100% median) PASS
  GPU6:                810.5 TFLOPS (100% median) PASS
  GPU7:                817.2 TFLOPS (101% median) PASS

  Median:              808.7 TFLOPS
  Relative floor:      90%
  Absolute floor:      700 TFLOPS

Multi-GPU
  NCCL AllReduce
  busbw:               468.16 GB/s                PASS

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
