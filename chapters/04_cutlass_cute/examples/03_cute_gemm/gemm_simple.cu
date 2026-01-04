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

// Simple GEMM kernel: each thread computes one element of C
// C(i, j) = sum_k A(i, k) * B(k, j)
template <typename TA, typename TB, typename TC>
__global__ void gemm_simple_kernel(
    TA const* A_ptr, int M, int K,
    TB const* B_ptr, int N,
    TC* C_ptr,
    float alpha, float beta)
{
    // Thread index maps to output element
    int i = blockIdx.x * blockDim.x + threadIdx.x;  // Row in C
    int j = blockIdx.y * blockDim.y + threadIdx.y;  // Column in C

    if (i < M && j < N) {
        // Create CuTe tensors for global memory
        auto A = make_tensor(A_ptr, make_layout(make_shape(M, K), make_stride(K, 1)));
        auto B = make_tensor(B_ptr, make_layout(make_shape(K, N), make_stride(N, 1)));
        auto C = make_tensor(C_ptr, make_layout(make_shape(M, N), make_stride(N, 1)));

        // Compute dot product
        float acc = 0.0f;
        for (int k = 0; k < K; ++k) {
            acc += static_cast<float>(A(i, k)) * static_cast<float>(B(k, j));
        }

        // Update C
        C(i, j) = alpha * acc + beta * C(i, j);
    }
}

// Initialize matrix with random values
void init_matrix(float* mat, int rows, int cols) {
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);

    for (int i = 0; i < rows * cols; ++i) {
        mat[i] = dis(gen);
    }
}

// Verify result against cuBLAS
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
    std::cout << "Testing Simple GEMM: M=" << M << ", N=" << N << ", K=" << K << std::endl;
    std::cout << std::string(70, '=') << std::endl;

    // Allocate host memory
    std::vector<float> h_A(M * K);
    std::vector<float> h_B(K * N);
    std::vector<float> h_C(M * N, 0.0f);
    std::vector<float> h_C_ref(M * N, 0.0f);

    // Initialize
    init_matrix(h_A.data(), M, K);
    init_matrix(h_B.data(), K, N);

    // Allocate device memory
    float *d_A, *d_B, *d_C, *d_C_ref;
    CUDA_CHECK(cudaMalloc(&d_A, M * K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B, K * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C, M * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C_ref, M * N * sizeof(float)));

    // Copy to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), K * N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_C, h_C.data(), M * N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_C_ref, h_C_ref.data(), M * N * sizeof(float), cudaMemcpyHostToDevice));

    float alpha = 1.0f, beta = 0.0f;

    // Run simple GEMM kernel
    dim3 block(16, 16);
    dim3 grid((M + block.x - 1) / block.x, (N + block.y - 1) / block.y);

    // Warmup
    gemm_simple_kernel<<<grid, block>>>(d_A, M, K, d_B, N, d_C, alpha, beta);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Benchmark
    const int num_runs = 10;
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < num_runs; ++i) {
        gemm_simple_kernel<<<grid, block>>>(d_A, M, K, d_B, N, d_C, alpha, beta);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    float avg_ms = ms / num_runs;

    // Run cuBLAS reference
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < num_runs; ++i) {
        // cuBLAS uses column-major, so we compute C = B^T * A^T
        CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                 N, M, K,
                                 &alpha,
                                 d_B, N,
                                 d_A, K,
                                 &beta,
                                 d_C_ref, N));
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float cublas_ms;
    CUDA_CHECK(cudaEventElapsedTime(&cublas_ms, start, stop));
    float avg_cublas_ms = cublas_ms / num_runs;

    // Copy results back
    CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_C_ref.data(), d_C_ref, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    // Verify
    float max_error = verify_result(h_C.data(), h_C_ref.data(), M, N);

    // Calculate TFLOPS
    double flops = 2.0 * M * N * K;
    double tflops = flops / (avg_ms * 1e9);
    double cublas_tflops = flops / (avg_cublas_ms * 1e9);

    // Print results
    std::cout << "\nResults:" << std::endl;
    std::cout << "  Simple GEMM:" << std::endl;
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

    // Cleanup
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
    std::cout << "Simple GEMM with CuTe" << std::endl;
    std::cout << std::string(70, '=') << std::endl;

    // Check device
    int device;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    std::cout << "\nDevice: " << prop.name << std::endl;
    std::cout << "Compute Capability: " << prop.major << "." << prop.minor << std::endl;

    // Run tests with different sizes
    run_gemm_test(512, 512, 512);
    run_gemm_test(1024, 1024, 1024);
    run_gemm_test(2048, 2048, 2048);

    std::cout << "\n" << std::string(70, '=') << std::endl;
    std::cout << "Notes:" << std::endl;
    std::cout << "  - Simple GEMM has no tiling or shared memory" << std::endl;
    std::cout << "  - Expected: 5-15% of cuBLAS performance" << std::endl;
    std::cout << "  - Each thread computes one output element" << std::endl;
    std::cout << "  - Poor data reuse and cache locality" << std::endl;
    std::cout << "\nNext: Try gemm_tiled for optimized implementation!" << std::endl;
    std::cout << std::string(70, '=') << std::endl;

    return EXIT_SUCCESS;
}
