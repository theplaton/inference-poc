// Single-GPU BF16 GEMM benchmark (cuBLAS, FP32 accumulate, Tensor Cores).
// Prints one parseable line: GPU=<id> TFLOPS=<x> AVG_MS=<y>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err_ = (call);                                            \
        if (err_ != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,     \
                    cudaGetErrorString(err_));                                \
            return 1;                                                         \
        }                                                                     \
    } while (0)

#define CUBLAS_CHECK(call)                                                    \
    do {                                                                      \
        cublasStatus_t st_ = (call);                                          \
        if (st_ != CUBLAS_STATUS_SUCCESS) {                                   \
            fprintf(stderr, "cuBLAS error %s:%d: status %d\n", __FILE__,      \
                    __LINE__, (int)st_);                                      \
            return 1;                                                         \
        }                                                                     \
    } while (0)

__global__ void fill_bf16(__nv_bfloat16 *p, size_t n, float v) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (; i < n; i += stride) p[i] = __float2bfloat16(v);
}

static void usage(const char *argv0) {
    fprintf(stderr,
            "usage: %s [--device N] [--size N] [--iters N] [--warmup N]\n",
            argv0);
}

int main(int argc, char **argv) {
    int device = 0, size = 16384, iters = 15, warmup = 5;

    for (int i = 1; i < argc; ++i) {
        const char *a = argv[i];
        bool has_val = (i + 1 < argc);
        if (!strcmp(a, "--device") && has_val)      device = atoi(argv[++i]);
        else if (!strcmp(a, "--size") && has_val)   size   = atoi(argv[++i]);
        else if (!strcmp(a, "--iters") && has_val)  iters  = atoi(argv[++i]);
        else if (!strcmp(a, "--warmup") && has_val) warmup = atoi(argv[++i]);
        else { usage(argv[0]); return 2; }
    }
    if (device < 0 || size <= 0 || iters <= 0) { usage(argv[0]); return 2; }

    const int M = size, N = size, K = size;
    const size_t elems = (size_t)M * N;

    CUDA_CHECK(cudaSetDevice(device));

    __nv_bfloat16 *A = nullptr, *B = nullptr, *C = nullptr;
    CUDA_CHECK(cudaMalloc(&A, elems * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&B, elems * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&C, elems * sizeof(__nv_bfloat16)));

    fill_bf16<<<1024, 256>>>(A, elems, 1.0f / 128.0f);
    fill_bf16<<<1024, 256>>>(B, elems, 1.0f / 128.0f);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemset(C, 0, elems * sizeof(__nv_bfloat16)));

    cublasHandle_t h;
    CUBLAS_CHECK(cublasCreate(&h));

    const float alpha = 1.0f, beta = 0.0f;
    // Column-major C = A * B, all square, so leading dims are all = size.
    auto gemm = [&]() {
        return cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K,
                            &alpha, A, CUDA_R_16BF, M, B, CUDA_R_16BF, K,
                            &beta, C, CUDA_R_16BF, M,
                            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    };

    for (int i = 0; i < warmup; ++i) CUBLAS_CHECK(gemm());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iters; ++i) CUBLAS_CHECK(gemm());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms_total = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms_total, start, stop));

    double avg_ms = ms_total / iters;
    double flops = 2.0 * M * N * K;
    double tflops = flops / (avg_ms / 1e3) / 1e12;

    printf("GPU=%d TFLOPS=%.1f AVG_MS=%.2f\n", device, tflops, avg_ms);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUBLAS_CHECK(cublasDestroy(h));
    CUDA_CHECK(cudaFree(A));
    CUDA_CHECK(cudaFree(B));
    CUDA_CHECK(cudaFree(C));
    return 0;
}
