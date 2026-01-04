// Starter code for FP16 GEMM with Tensor Cores
// TODO: Implement WMMA-based FP16 GEMM

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

// TODO: Define tile sizes
constexpr int BM = 128;  // Block M
constexpr int BN = 128;  // Block N
constexpr int BK = 32;   // Block K

// WMMA tile size (fixed by hardware)
constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;

// TODO: Implement FP16 GEMM kernel using WMMA
template <int BM, int BN, int BK>
__global__ void fp16_gemm_wmma_kernel(
    const half* __restrict__ A, int M, int K,
    const half* __restrict__ B, int N,
    half* __restrict__ C)
{
    // TODO: Calculate warp ID and position
    // int warp_id = threadIdx.x / 32;
    // int warp_m = warp_id / (BN / WMMA_N);
    // int warp_n = warp_id % (BN / WMMA_N);

    // TODO: Allocate shared memory
    // __shared__ half smem_A[BM * BK];
    // __shared__ half smem_B[BK * BN];

    // TODO: Declare WMMA fragments
    // wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    // wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
    // wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;

    // TODO: Initialize accumulator
    // wmma::fill_fragment(acc_frag, 0.0f);

    // TODO: Main loop over K
    // for (int k_tile = 0; k_tile < K; k_tile += BK) {
    //     // 1. Load tiles to shared memory cooperatively
    //     // 2. __syncthreads()
    //     // 3. Each warp loads fragments and accumulates
    //     // 4. __syncthreads()
    // }

    // TODO: Store result
    // Convert FP32 accumulator to FP16 and store
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
    std::cout << "\nTesting FP16 GEMM: M=" << M << ", N=" << N << ", K=" << K << std::endl;

    // Allocate host memory
    std::vector<half> h_A(M * K);
    std::vector<half> h_B(K * N);
    std::vector<half> h_C(M * N);
    std::vector<half> h_C_ref(M * N);

    init_matrix(h_A.data(), M, K);
    init_matrix(h_B.data(), K, N);

    // Allocate device memory
    half *d_A, *d_B, *d_C, *d_C_ref;
    CUDA_CHECK(cudaMalloc(&d_A, M * K * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_B, K * N * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_C, M * N * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_C_ref, M * N * sizeof(half)));

    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), M * K * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), K * N * sizeof(half), cudaMemcpyHostToDevice));

    // TODO: Configure kernel launch
    // constexpr int WARPS = (BM / WMMA_M) * (BN / WMMA_N);
    // dim3 block(WARPS * 32);
    // dim3 grid((M + BM - 1) / BM, (N + BN - 1) / BN);

    // TODO: Launch kernel
    // fp16_gemm_wmma_kernel<BM, BN, BK><<<grid, block>>>(d_A, M, K, d_B, N, d_C);

    // cuBLAS reference
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);

    float alpha = 1.0f, beta = 0.0f;
    cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                 N, M, K, &alpha,
                 d_B, CUDA_R_16F, N,
                 d_A, CUDA_R_16F, K,
                 &beta,
                 d_C_ref, CUDA_R_16F, N,
                 CUBLAS_COMPUTE_32F,
                 CUBLAS_GEMM_DEFAULT_TENSOR_OP);

    // Verify
    CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, M * N * sizeof(half), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_C_ref.data(), d_C_ref, M * N * sizeof(half), cudaMemcpyDeviceToHost));

    float max_error = verify_result(h_C.data(), h_C_ref.data(), M, N);
    std::cout << "  Max error: " << max_error << std::endl;

    if (max_error < 0.1f) {
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
    std::cout << "FP16 GEMM with Tensor Cores - Starter Code" << std::endl;
    std::cout << "Complete the TODOs to implement WMMA-based GEMM" << std::endl;

    int device;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    std::cout << "\nDevice: " << prop.name << std::endl;
    std::cout << "Compute Capability: " << prop.major << "." << prop.minor << std::endl;

    if (prop.major < 7) {
        std::cerr << "Error: Tensor Cores require SM70+ (Volta or newer)" << std::endl;
        return 1;
    }

    test_fp16_gemm(1024, 1024, 1024);
    test_fp16_gemm(2048, 2048, 2048);

    return 0;
}
