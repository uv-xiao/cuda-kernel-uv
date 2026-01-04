// Solution for batched GEMM exercise
// This is a reference implementation

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

template <int BM, int BN, int BK, int TM, int TN>
__global__ void batched_gemm_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int batch_size, int M, int N, int K,
    float alpha, float beta)
{
    // Get batch index from Z dimension
    int batch_idx = blockIdx.z;

    if (batch_idx >= batch_size) return;

    // Calculate batch offsets
    const float* A_batch = A + batch_idx * M * K;
    const float* B_batch = B + batch_idx * K * N;
    float* C_batch = C + batch_idx * M * N;

    // Shared memory for this batch's tile
    __shared__ float smem_A[BM * BK];
    __shared__ float smem_B[BK * BN];

    // Thread organization
    int tid = threadIdx.x;
    constexpr int THREADS_M = BM / TM;
    constexpr int THREADS_N = BN / TN;
    constexpr int NUM_THREADS = THREADS_M * THREADS_N;

    int thread_m = tid / THREADS_N;
    int thread_n = tid % THREADS_N;

    int row_base = blockIdx.x * BM + thread_m * TM;
    int col_base = blockIdx.y * BN + thread_n * TN;

    // Accumulator
    float acc[TM][TN];
    for (int i = 0; i < TM; ++i) {
        for (int j = 0; j < TN; ++j) {
            acc[i][j] = 0.0f;
        }
    }

    // Loop over K tiles
    for (int k_tile = 0; k_tile < K; k_tile += BK) {
        // Load A tile cooperatively
        for (int i = tid; i < BM * BK; i += NUM_THREADS) {
            int sm_row = i / BK;
            int sm_col = i % BK;
            int gm_row = blockIdx.x * BM + sm_row;
            int gm_col = k_tile + sm_col;

            if (gm_row < M && gm_col < K) {
                smem_A[sm_row * BK + sm_col] = A_batch[gm_row * K + gm_col];
            } else {
                smem_A[sm_row * BK + sm_col] = 0.0f;
            }
        }

        // Load B tile cooperatively
        for (int i = tid; i < BK * BN; i += NUM_THREADS) {
            int sm_row = i / BN;
            int sm_col = i % BN;
            int gm_row = k_tile + sm_row;
            int gm_col = blockIdx.y * BN + sm_col;

            if (gm_row < K && gm_col < N) {
                smem_B[sm_row * BN + sm_col] = B_batch[gm_row * N + gm_col];
            } else {
                smem_B[sm_row * BN + sm_col] = 0.0f;
            }
        }

        __syncthreads();

        // Compute
        for (int k = 0; k < BK; ++k) {
            float a_vals[TM];
            for (int i = 0; i < TM; ++i) {
                a_vals[i] = smem_A[(thread_m * TM + i) * BK + k];
            }

            float b_vals[TN];
            for (int j = 0; j < TN; ++j) {
                b_vals[j] = smem_B[k * BN + thread_n * TN + j];
            }

            for (int i = 0; i < TM; ++i) {
                for (int j = 0; j < TN; ++j) {
                    acc[i][j] += a_vals[i] * b_vals[j];
                }
            }
        }

        __syncthreads();
    }

    // Write results
    for (int i = 0; i < TM; ++i) {
        for (int j = 0; j < TN; ++j) {
            int row = row_base + i;
            int col = col_base + j;

            if (row < M && col < N) {
                int idx = row * N + col;
                if (beta == 0.0f) {
                    C_batch[idx] = alpha * acc[i][j];
                } else {
                    C_batch[idx] = alpha * acc[i][j] + beta * C_batch[idx];
                }
            }
        }
    }
}

void init_batched_matrices(float* data, int batch, int rows, int cols) {
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);

    for (int i = 0; i < batch * rows * cols; ++i) {
        data[i] = dis(gen);
    }
}

float verify_batched_result(const float* C, const float* C_ref,
                             int batch, int M, int N) {
    float max_error = 0.0f;
    for (int i = 0; i < batch * M * N; ++i) {
        float error = std::abs(C[i] - C_ref[i]);
        max_error = std::max(max_error, error);
    }
    return max_error;
}

void test_batched_gemm(int batch_size, int M, int N, int K) {
    std::cout << "\n" << std::string(70, '=') << std::endl;
    std::cout << "Batched GEMM: batch=" << batch_size
              << ", M=" << M << ", N=" << N << ", K=" << K << std::endl;
    std::cout << std::string(70, '=') << std::endl;

    size_t size_A = batch_size * M * K;
    size_t size_B = batch_size * K * N;
    size_t size_C = batch_size * M * N;

    std::vector<float> h_A(size_A);
    std::vector<float> h_B(size_B);
    std::vector<float> h_C(size_C, 0.0f);
    std::vector<float> h_C_ref(size_C, 0.0f);

    init_batched_matrices(h_A.data(), batch_size, M, K);
    init_batched_matrices(h_B.data(), batch_size, K, N);

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

    constexpr int BM = 64, BN = 64, BK = 8;
    constexpr int TM = 8, TN = 8;
    constexpr int THREADS = (BM / TM) * (BN / TN);

    dim3 block(THREADS);
    dim3 grid((M + BM - 1) / BM, (N + BN - 1) / BN, batch_size);

    // Benchmark
    const int num_runs = 100;
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < num_runs; ++i) {
        batched_gemm_kernel<BM, BN, BK, TM, TN><<<grid, block>>>(
            d_A, d_B, d_C, batch_size, M, N, K, alpha, beta);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    float avg_ms = ms / num_runs;

    // cuBLAS
    cublasHandle_t handle;
    cublasCreate(&handle);

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < num_runs; ++i) {
        cublasGemmStridedBatchedEx(
            handle, CUBLAS_OP_N, CUBLAS_OP_N,
            N, M, K, &alpha,
            d_B, CUDA_R_32F, N, K * N,
            d_A, CUDA_R_32F, K, M * K,
            &beta,
            d_C_ref, CUDA_R_32F, N, M * N,
            batch_size,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float cublas_ms;
    CUDA_CHECK(cudaEventElapsedTime(&cublas_ms, start, stop));
    float avg_cublas_ms = cublas_ms / num_runs;

    // Verify
    CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, size_C * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_C_ref.data(), d_C_ref, size_C * sizeof(float), cudaMemcpyDeviceToHost));

    float max_error = verify_batched_result(h_C.data(), h_C_ref.data(), batch_size, M, N);

    double flops = 2.0 * batch_size * M * N * K;
    double tflops = flops / (avg_ms * 1e9);
    double cublas_tflops = flops / (avg_cublas_ms * 1e9);

    std::cout << "\nResults:" << std::endl;
    std::cout << "  Custom kernel: " << avg_ms << " ms, " << tflops << " TFLOPS" << std::endl;
    std::cout << "  cuBLAS: " << avg_cublas_ms << " ms, " << cublas_tflops << " TFLOPS" << std::endl;
    std::cout << "  Efficiency: " << (tflops / cublas_tflops * 100.0) << "% of cuBLAS" << std::endl;
    std::cout << "  Max error: " << max_error << (max_error < 1e-3f ? " ✓" : " ✗") << std::endl;

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    cublasDestroy(handle);
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaFree(d_C_ref));
}

int main() {
    test_batched_gemm(10, 64, 64, 64);
    test_batched_gemm(5, 256, 256, 256);
    test_batched_gemm(2, 512, 512, 512);
    return 0;
}
