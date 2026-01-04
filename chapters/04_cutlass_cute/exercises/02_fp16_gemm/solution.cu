// Solution for FP16 GEMM with Tensor Cores
// Reference implementation using WMMA

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cublas_v2.h>
#include <iostream>
#include <vector>
#include <random>

using namespace nvcuda;

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;

template <int BM, int BN, int BK>
__global__ void fp16_gemm_wmma_kernel(
    const half* __restrict__ A, int M, int K,
    const half* __restrict__ B, int N,
    half* __restrict__ C)
{
    // Warp and block organization
    constexpr int WARPS_M = BM / WMMA_M;
    constexpr int WARPS_N = BN / WMMA_N;

    int warp_id = threadIdx.x / 32;
    int warp_m = warp_id / WARPS_N;
    int warp_n = warp_id % WARPS_N;

    int block_m = blockIdx.x * BM;
    int block_n = blockIdx.y * BN;
    int warp_m_offset = block_m + warp_m * WMMA_M;
    int warp_n_offset = block_n + warp_n * WMMA_N;

    // Shared memory
    __shared__ half smem_A[BM * BK];
    __shared__ half smem_B[BK * BN];

    // WMMA fragments
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;

    // Initialize accumulator
    wmma::fill_fragment(acc_frag, 0.0f);

    // Main loop over K
    for (int k_tile = 0; k_tile < K; k_tile += BK) {
        // Cooperatively load A tile
        int tid = threadIdx.x;
        int num_threads = blockDim.x;

        for (int i = tid; i < BM * BK; i += num_threads) {
            int row = i / BK;
            int col = i % BK;
            int gm_row = block_m + row;
            int gm_col = k_tile + col;

            if (gm_row < M && gm_col < K) {
                smem_A[row * BK + col] = A[gm_row * K + gm_col];
            } else {
                smem_A[row * BK + col] = __float2half(0.0f);
            }
        }

        // Cooperatively load B tile
        for (int i = tid; i < BK * BN; i += num_threads) {
            int row = i / BN;
            int col = i % BN;
            int gm_row = k_tile + row;
            int gm_col = block_n + col;

            if (gm_row < K && gm_col < N) {
                smem_B[row * BN + col] = B[gm_row * N + gm_col];
            } else {
                smem_B[row * BN + col] = __float2half(0.0f);
            }
        }

        __syncthreads();

        // Each warp computes its 16x16 tile
        for (int k = 0; k < BK; k += WMMA_K) {
            int a_offset = warp_m * WMMA_M * BK + k;
            int b_offset = k * BN + warp_n * WMMA_N;

            wmma::load_matrix_sync(a_frag, smem_A + a_offset, BK);
            wmma::load_matrix_sync(b_frag, smem_B + b_offset, BN);
            wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        }

        __syncthreads();
    }

    // Store result
    if (warp_m_offset < M && warp_n_offset < N) {
        // Convert FP32 accumulator to FP16
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> c_frag;

        for (int i = 0; i < acc_frag.num_elements; ++i) {
            c_frag.x[i] = __float2half(acc_frag.x[i]);
        }

        wmma::store_matrix_sync(C + warp_m_offset * N + warp_n_offset,
                                c_frag, N, wmma::mem_row_major);
    }
}

void init_matrix(half* mat, int rows, int cols) {
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    for (int i = 0; i < rows * cols; ++i) {
        mat[i] = __float2half(dis(gen));
    }
}

float verify_result(const half* C, const half* C_ref, int M, int N) {
    float max_error = 0.0f;
    for (int i = 0; i < M * N; ++i) {
        float error = std::abs(__half2float(C[i]) - __half2float(C_ref[i]));
        max_error = std::max(max_error, error);
    }
    return max_error;
}

void test_fp16_gemm(int M, int N, int K) {
    std::cout << "\nFP16 GEMM: M=" << M << ", N=" << N << ", K=" << K << std::endl;

    std::vector<half> h_A(M * K);
    std::vector<half> h_B(K * N);
    std::vector<half> h_C(M * N);
    std::vector<half> h_C_ref(M * N);

    init_matrix(h_A.data(), M, K);
    init_matrix(h_B.data(), K, N);

    half *d_A, *d_B, *d_C, *d_C_ref;
    CUDA_CHECK(cudaMalloc(&d_A, M * K * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_B, K * N * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_C, M * N * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_C_ref, M * N * sizeof(half)));

    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), M * K * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), K * N * sizeof(half), cudaMemcpyHostToDevice));

    constexpr int BM = 128, BN = 128, BK = 32;
    constexpr int WARPS = (BM / WMMA_M) * (BN / WMMA_N);
    dim3 block(WARPS * 32);
    dim3 grid((M + BM - 1) / BM, (N + BN - 1) / BN);

    // Benchmark
    const int num_runs = 100;
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < num_runs; ++i) {
        fp16_gemm_wmma_kernel<BM, BN, BK><<<grid, block>>>(d_A, M, K, d_B, N, d_C);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    float avg_ms = ms / num_runs;

    // cuBLAS
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);

    float alpha = 1.0f, beta = 0.0f;

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < num_runs; ++i) {
        cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                     N, M, K, &alpha,
                     d_B, CUDA_R_16F, N,
                     d_A, CUDA_R_16F, K,
                     &beta,
                     d_C_ref, CUDA_R_16F, N,
                     CUBLAS_COMPUTE_32F,
                     CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float cublas_ms;
    CUDA_CHECK(cudaEventElapsedTime(&cublas_ms, start, stop));
    float avg_cublas_ms = cublas_ms / num_runs;

    // Verify
    CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, M * N * sizeof(half), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_C_ref.data(), d_C_ref, M * N * sizeof(half), cudaMemcpyDeviceToHost));

    float max_error = verify_result(h_C.data(), h_C_ref.data(), M, N);

    double flops = 2.0 * M * N * K;
    double tflops = flops / (avg_ms * 1e9);
    double cublas_tflops = flops / (avg_cublas_ms * 1e9);

    std::cout << "  WMMA: " << avg_ms << " ms, " << tflops << " TFLOPS" << std::endl;
    std::cout << "  cuBLAS: " << avg_cublas_ms << " ms, " << cublas_tflops << " TFLOPS" << std::endl;
    std::cout << "  Efficiency: " << (tflops / cublas_tflops * 100.0) << "%" << std::endl;
    std::cout << "  Max error: " << max_error << (max_error < 0.1f ? " ✓" : " ✗") << std::endl;

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    cublasDestroy(handle);
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaFree(d_C_ref));
}

int main() {
    int device;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    std::cout << "Device: " << prop.name << std::endl;

    if (prop.major < 7) {
        std::cerr << "Requires SM70+" << std::endl;
        return 1;
    }

    test_fp16_gemm(1024, 1024, 1024);
    test_fp16_gemm(2048, 2048, 2048);
    test_fp16_gemm(4096, 4096, 4096);

    return 0;
}
