#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>
#include <iomanip>
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

#define CUBLAS_CHECK(call) \
    do { \
        cublasStatus_t status = call; \
        if (status != CUBLAS_STATUS_SUCCESS) { \
            std::cerr << "cuBLAS error: " << status << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// Tiled GEMM kernel using shared memory
template <int BM, int BN, int BK, int TM, int TN>
__global__ void gemm_tiled_kernel(
    const float* __restrict__ A_ptr, int M, int K,
    const float* __restrict__ B_ptr, int N,
    float* __restrict__ C_ptr,
    float alpha, float beta)
{
    // Shared memory for tiles
    __shared__ float smem_A[BM * BK];
    __shared__ float smem_B[BK * BN];

    // Thread and block indices
    int tid = threadIdx.x;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Threads per block dimension
    constexpr int THREADS_M = BM / TM;
    constexpr int THREADS_N = BN / TN;
    constexpr int NUM_THREADS = THREADS_M * THREADS_N;

    int thread_m = tid / THREADS_N;
    int thread_n = tid % THREADS_N;

    // Global row and column for this thread
    int row_base = bx * BM + thread_m * TM;
    int col_base = by * BN + thread_n * TN;

    // Accumulator registers
    float acc[TM][TN];
    for (int i = 0; i < TM; ++i) {
        for (int j = 0; j < TN; ++j) {
            acc[i][j] = 0.0f;
        }
    }

    // Loop over K dimension in tiles
    for (int k_tile = 0; k_tile < K; k_tile += BK) {
        // Cooperatively load A tile to shared memory
        // Each thread loads multiple elements
        for (int i = tid; i < BM * BK; i += NUM_THREADS) {
            int sm_row = i / BK;
            int sm_col = i % BK;
            int gm_row = bx * BM + sm_row;
            int gm_col = k_tile + sm_col;

            if (gm_row < M && gm_col < K) {
                smem_A[sm_row * BK + sm_col] = A_ptr[gm_row * K + gm_col];
            } else {
                smem_A[sm_row * BK + sm_col] = 0.0f;
            }
        }

        // Cooperatively load B tile to shared memory
        for (int i = tid; i < BK * BN; i += NUM_THREADS) {
            int sm_row = i / BN;
            int sm_col = i % BN;
            int gm_row = k_tile + sm_row;
            int gm_col = by * BN + sm_col;

            if (gm_row < K && gm_col < N) {
                smem_B[sm_row * BN + sm_col] = B_ptr[gm_row * N + gm_col];
            } else {
                smem_B[sm_row * BN + sm_col] = 0.0f;
            }
        }

        __syncthreads();

        // Compute using shared memory
        for (int k = 0; k < BK; ++k) {
            // Load A values for this thread
            float a_vals[TM];
            for (int i = 0; i < TM; ++i) {
                a_vals[i] = smem_A[(thread_m * TM + i) * BK + k];
            }

            // Load B values for this thread
            float b_vals[TN];
            for (int j = 0; j < TN; ++j) {
                b_vals[j] = smem_B[k * BN + thread_n * TN + j];
            }

            // Outer product
            for (int i = 0; i < TM; ++i) {
                for (int j = 0; j < TN; ++j) {
                    acc[i][j] += a_vals[i] * b_vals[j];
                }
            }
        }

        __syncthreads();
    }

    // Write results to global memory
    for (int i = 0; i < TM; ++i) {
        for (int j = 0; j < TN; ++j) {
            int row = row_base + i;
            int col = col_base + j;

            if (row < M && col < N) {
                int idx = row * N + col;
                if (beta == 0.0f) {
                    C_ptr[idx] = alpha * acc[i][j];
                } else {
                    C_ptr[idx] = alpha * acc[i][j] + beta * C_ptr[idx];
                }
            }
        }
    }
}

void init_matrix(float* mat, int rows, int cols) {
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    for (int i = 0; i < rows * cols; ++i) {
        mat[i] = dis(gen);
    }
}

float verify_result(const float* C, const float* C_ref, int M, int N) {
    float max_error = 0.0f;
    for (int i = 0; i < M * N; ++i) {
        float error = std::abs(C[i] - C_ref[i]);
        max_error = std::max(max_error, error);
    }
    return max_error;
}

void run_gemm_test(int M, int N, int K) {
    std::cout << "\n" << std::string(70, '=') << std::endl;
    std::cout << "Testing Tiled GEMM: M=" << M << ", N=" << N << ", K=" << K << std::endl;
    std::cout << std::string(70, '=') << std::endl;

    // Allocate host memory
    std::vector<float> h_A(M * K);
    std::vector<float> h_B(K * N);
    std::vector<float> h_C(M * N, 0.0f);
    std::vector<float> h_C_ref(M * N, 0.0f);

    init_matrix(h_A.data(), M, K);
    init_matrix(h_B.data(), K, N);

    // Allocate device memory
    float *d_A, *d_B, *d_C, *d_C_ref;
    CUDA_CHECK(cudaMalloc(&d_A, M * K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B, K * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C, M * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C_ref, M * N * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), K * N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_C, h_C.data(), M * N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_C_ref, h_C_ref.data(), M * N * sizeof(float), cudaMemcpyHostToDevice));

    float alpha = 1.0f, beta = 0.0f;

    // Kernel configuration
    constexpr int BM = 128, BN = 128, BK = 8;
    constexpr int TM = 8, TN = 8;
    constexpr int THREADS = (BM / TM) * (BN / TN);

    dim3 block(THREADS);
    dim3 grid((M + BM - 1) / BM, (N + BN - 1) / BN);

    std::cout << "\nConfiguration:" << std::endl;
    std::cout << "  Block tile: " << BM << "x" << BN << "x" << BK << std::endl;
    std::cout << "  Thread tile: " << TM << "x" << TN << std::endl;
    std::cout << "  Threads per block: " << THREADS << std::endl;
    std::cout << "  Grid: " << grid.x << "x" << grid.y << std::endl;

    // Warmup
    gemm_tiled_kernel<BM, BN, BK, TM, TN><<<grid, block>>>(d_A, M, K, d_B, N, d_C, alpha, beta);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Benchmark
    const int num_runs = 100;
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < num_runs; ++i) {
        gemm_tiled_kernel<BM, BN, BK, TM, TN><<<grid, block>>>(d_A, M, K, d_B, N, d_C, alpha, beta);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    float avg_ms = ms / num_runs;

    // cuBLAS reference
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < num_runs; ++i) {
        CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                 N, M, K, &alpha, d_B, N, d_A, K, &beta, d_C_ref, N));
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float cublas_ms;
    CUDA_CHECK(cudaEventElapsedTime(&cublas_ms, start, stop));
    float avg_cublas_ms = cublas_ms / num_runs;

    // Verify
    CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_C_ref.data(), d_C_ref, M * N * sizeof(float), cudaMemcpyDeviceToHost));
    float max_error = verify_result(h_C.data(), h_C_ref.data(), M, N);

    // Results
    double flops = 2.0 * M * N * K;
    double tflops = flops / (avg_ms * 1e9);
    double cublas_tflops = flops / (avg_cublas_ms * 1e9);

    std::cout << "\nResults:" << std::endl;
    std::cout << "  Tiled GEMM:" << std::endl;
    std::cout << "    Time: " << std::fixed << std::setprecision(3) << avg_ms << " ms" << std::endl;
    std::cout << "    Performance: " << std::setprecision(2) << tflops << " TFLOPS" << std::endl;
    std::cout << "\n  cuBLAS:" << std::endl;
    std::cout << "    Time: " << std::fixed << std::setprecision(3) << avg_cublas_ms << " ms" << std::endl;
    std::cout << "    Performance: " << std::setprecision(2) << cublas_tflops << " TFLOPS" << std::endl;
    std::cout << "\n  Efficiency: " << std::setprecision(1)
              << (tflops / cublas_tflops * 100.0) << "% of cuBLAS" << std::endl;
    std::cout << "  Max error: " << std::scientific << max_error << std::endl;

    if (max_error < 1e-3f) {
        std::cout << "  ✓ Verification passed!" << std::endl;
    } else {
        std::cout << "  ✗ Verification failed!" << std::endl;
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUBLAS_CHECK(cublasDestroy(handle));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaFree(d_C_ref));
}

int main() {
    std::cout << "\n" << std::string(70, '=') << std::endl;
    std::cout << "Tiled GEMM with CuTe and Shared Memory" << std::endl;
    std::cout << std::string(70, '=') << std::endl;

    int device;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    std::cout << "\nDevice: " << prop.name << std::endl;
    std::cout << "Compute Capability: " << prop.major << "." << prop.minor << std::endl;
    std::cout << "Shared Memory per Block: " << prop.sharedMemPerBlock / 1024 << " KB" << std::endl;

    run_gemm_test(512, 512, 512);
    run_gemm_test(1024, 1024, 1024);
    run_gemm_test(2048, 2048, 2048);

    std::cout << "\n" << std::string(70, '=') << std::endl;
    std::cout << "Summary:" << std::endl;
    std::cout << "  - Tiled GEMM uses shared memory for data reuse" << std::endl;
    std::cout << "  - Target: 60-80% of cuBLAS for FP32" << std::endl;
    std::cout << "  - For higher performance, try FP16 with Tensor Cores (example 04)" << std::endl;
    std::cout << std::string(70, '=') << std::endl;

    return EXIT_SUCCESS;
}
