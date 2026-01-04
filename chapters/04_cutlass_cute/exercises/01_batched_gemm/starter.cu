// Starter code for batched GEMM exercise
// TODO: Complete the implementation

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>
#include <vector>
#include <random>
#include <cmath>

#include <cute/tensor.hpp>
#include <cute/layout.hpp>

using namespace cute;

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// TODO: Implement batched GEMM kernel
template <int BM, int BN, int BK, int TM, int TN>
__global__ void batched_gemm_kernel(
    const float* __restrict__ A,  // [batch, M, K]
    const float* __restrict__ B,  // [batch, K, N]
    float* __restrict__ C,        // [batch, M, N]
    int batch_size, int M, int N, int K,
    float alpha, float beta)
{
    // TODO: Get batch index
    int batch_idx = 0;  // HINT: Use blockIdx.z or compute from linear index

    // TODO: Calculate strides for this batch
    // int A_batch_offset = batch_idx * M * K;
    // int B_batch_offset = batch_idx * K * N;
    // int C_batch_offset = batch_idx * M * N;

    // TODO: Implement shared memory tiling GEMM for this batch
    // HINT: Reuse logic from gemm_tiled.cu, but operate on batch slice

    // __shared__ float smem_A[BM * BK];
    // __shared__ float smem_B[BK * BN];

    // Your implementation here
}

void init_batched_matrices(float* data, int batch, int rows, int cols) {
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);

    for (int b = 0; b < batch; ++b) {
        for (int i = 0; i < rows * cols; ++i) {
            data[b * rows * cols + i] = dis(gen);
        }
    }
}

float verify_batched_result(const float* C, const float* C_ref,
                             int batch, int M, int N) {
    float max_error = 0.0f;
    for (int b = 0; b < batch; ++b) {
        for (int i = 0; i < M * N; ++i) {
            int idx = b * M * N + i;
            float error = std::abs(C[idx] - C_ref[idx]);
            max_error = std::max(max_error, error);
        }
    }
    return max_error;
}

void test_batched_gemm(int batch_size, int M, int N, int K) {
    std::cout << "\nTesting Batched GEMM:" << std::endl;
    std::cout << "  Batch: " << batch_size << ", M: " << M
              << ", N: " << N << ", K: " << K << std::endl;

    // Allocate host memory
    size_t size_A = batch_size * M * K;
    size_t size_B = batch_size * K * N;
    size_t size_C = batch_size * M * N;

    std::vector<float> h_A(size_A);
    std::vector<float> h_B(size_B);
    std::vector<float> h_C(size_C, 0.0f);
    std::vector<float> h_C_ref(size_C, 0.0f);

    init_batched_matrices(h_A.data(), batch_size, M, K);
    init_batched_matrices(h_B.data(), batch_size, K, N);

    // Allocate device memory
    float *d_A, *d_B, *d_C, *d_C_ref;
    CUDA_CHECK(cudaMalloc(&d_A, size_A * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B, size_B * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C, size_C * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C_ref, size_C * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), size_A * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), size_B * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_C, 0, size_C * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_C_ref, 0, size_C * sizeof(float)));

    float alpha = 1.0f, beta = 0.0f;

    // TODO: Configure kernel launch parameters
    constexpr int BM = 64, BN = 64, BK = 8;
    constexpr int TM = 8, TN = 8;
    constexpr int THREADS = (BM / TM) * (BN / TN);

    // TODO: Setup grid dimensions to handle batch dimension
    dim3 block(THREADS);
    dim3 grid(
        (M + BM - 1) / BM,
        (N + BN - 1) / BN,
        batch_size  // HINT: Use Z dimension for batch
    );

    std::cout << "  Grid: (" << grid.x << ", " << grid.y << ", " << grid.z << ")" << std::endl;
    std::cout << "  Block: " << block.x << " threads" << std::endl;

    // Warmup
    batched_gemm_kernel<BM, BN, BK, TM, TN><<<grid, block>>>(
        d_A, d_B, d_C, batch_size, M, N, K, alpha, beta);
    CUDA_CHECK(cudaDeviceSynchronize());

    // TODO: Benchmark your implementation

    // cuBLAS reference
    cublasHandle_t handle;
    cublasCreate(&handle);

    cublasGemmStridedBatchedEx(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        N, M, K,
        &alpha,
        d_B, CUDA_R_32F, N, K * N,
        d_A, CUDA_R_32F, K, M * K,
        &beta,
        d_C_ref, CUDA_R_32F, N, M * N,
        batch_size,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT
    );

    // Verify
    CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, size_C * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_C_ref.data(), d_C_ref, size_C * sizeof(float), cudaMemcpyDeviceToHost));

    float max_error = verify_batched_result(h_C.data(), h_C_ref.data(), batch_size, M, N);
    std::cout << "  Max error: " << max_error << std::endl;

    if (max_error < 1e-3f) {
        std::cout << "  ✓ Verification passed!" << std::endl;
    } else {
        std::cout << "  ✗ Verification failed!" << std::endl;
    }

    // Cleanup
    cublasDestroy(handle);
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaFree(d_C_ref));
}

int main() {
    std::cout << "Batched GEMM Exercise - Starter Code" << std::endl;
    std::cout << "Complete the TODOs to implement batched matrix multiplication" << std::endl;

    // Test cases
    test_batched_gemm(10, 64, 64, 64);     // Small matrices
    test_batched_gemm(5, 256, 256, 256);   // Medium matrices
    test_batched_gemm(2, 512, 512, 512);   // Large matrices

    return 0;
}
