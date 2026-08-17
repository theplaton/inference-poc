ARG CUDA_VERSION=12.8.1
ARG UBUNTU=ubuntu24.04

# ---------- builder ----------
FROM nvidia/cuda:${CUDA_VERSION}-devel-${UBUNTU} AS builder

# Pinned so image contents are reproducible.
ARG NCCL_TESTS_TAG=v2.19.7
# Ampere + Hopper by default; add e.g. "100" for Blackwell (needs CUDA >= 12.8).
ARG CUDA_ARCHS="80 90"

# The devel image already provides nvcc, g++, make, cuBLAS and NCCL headers.
RUN apt-get update && apt-get install -y --no-install-recommends \
        git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
COPY cuda/gemm_bench.cu .

RUN GENCODE=""; for a in ${CUDA_ARCHS}; do \
        GENCODE="${GENCODE} -gencode arch=compute_${a},code=sm_${a}"; done; \
    LAST=$(echo ${CUDA_ARCHS} | tr ' ' '\n' | tail -1); \
    GENCODE="${GENCODE} -gencode arch=compute_${LAST},code=compute_${LAST}"; \
    mkdir -p /out && \
    nvcc -O3 -std=c++17 ${GENCODE} -o /out/gemm_bench gemm_bench.cu -lcublas

RUN git clone --depth 1 --branch ${NCCL_TESTS_TAG} \
        https://github.com/NVIDIA/nccl-tests.git && \
    GENCODE=""; for a in ${CUDA_ARCHS}; do \
        GENCODE="${GENCODE} -gencode=arch=compute_${a},code=sm_${a}"; done; \
    make -C nccl-tests -j"$(nproc)" MPI=0 CUDA_HOME=/usr/local/cuda \
        NVCC_GENCODE="${GENCODE}" && \
    cp nccl-tests/build/all_reduce_perf /out/

# ---------- runtime ----------
# The base image (cudart only, ~411MB) plus just cuBLAS and NCCL, instead of the
# runtime image's full cuda-libraries metapackage (cuFFT/cuSPARSE/cuSOLVER/
# cuRAND/NPP/nvRTC — ~4GB we never call). Keep NCCL_PKG_VERSION matched to the
# NCCL the builder linked against; bump both it and CUDA_VERSION together.
FROM nvidia/cuda:${CUDA_VERSION}-base-${UBUNTU}
ARG CUDA_VERSION
ARG NCCL_PKG_VERSION=2.25.1-1+cuda12.8

RUN CUDA_PKG=$(echo "${CUDA_VERSION}" | cut -d. -f1,2 | tr . -) && \
    apt-get update && apt-get install -y --no-install-recommends \
        "libcublas-${CUDA_PKG}" "libnccl2=${NCCL_PKG_VERSION}" \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /out/gemm_bench /usr/local/bin/gemm_bench
COPY --from=builder /out/all_reduce_perf /usr/local/bin/all_reduce_perf
COPY validate.sh /usr/local/bin/validate.sh

ENV RELATIVE_GPU_FLOOR=0.90
ENTRYPOINT ["/usr/local/bin/validate.sh"]
