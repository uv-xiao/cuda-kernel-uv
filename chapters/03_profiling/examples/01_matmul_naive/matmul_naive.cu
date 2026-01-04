#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

// Matrix multiplication kernel: C = A * B
// Each thread computes one element of C
// This is the naive implementation with poor memory access patterns
__global__ void matmul_naive(const float* A, const float* B, float* C, int N) {
    // Calculate global thread position
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Check bounds
    if (row < N && col < N) {
        float sum = 0.0f;

        // Compute dot product of row from A and column from B
        for (int k = 0; k < N; k++) {
            // A is accessed in a coalesced manner (consecutive threads access consecutive memory)
            // B is accessed in a strided manner (consecutive threads access non-consecutive memory)
            // This causes poor memory bandwidth utilization
            sum += A[row * N + k] * B[k * N + col];
        }

        // Write result
        C[row * N + col] = sum;
    }
}

// Initialize matrix with random values
void init_matrix(float* mat, int N) {
    for (int i = 0; i < N * N; i++) {
        mat[i] = (float)(rand() % 100) / 10.0f;
    }
}

// Verify result against CPU computation
bool verify_result(const float* A, const float* B, const float* C, int N) {
    const float epsilon = 1e-3f;

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < N; k++) {
                sum += A[i * N + k] * B[k * N + j];
            }

            float diff = fabs(C[i * N + j] - sum);
            if (diff > epsilon) {
                printf("Mismatch at (%d, %d): GPU=%.3f, CPU=%.3f, diff=%.3f\n",
                       i, j, C[i * N + j], sum, diff);
                return false;
            }
        }
    }
    return true;
}

// CUDA error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            printf("CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                   cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

int main(int argc, char** argv) {
    // Matrix size (default 2048x2048)
    int N = 2048;
    if (argc > 1) {
        N = atoi(argv[1]);
    }

    printf("=== Naive Matrix Multiplication ===\n");
    printf("Matrix size: %d x %d\n", N, N);
    printf("Total elements: %d\n", N * N);
    printf("Memory per matrix: %.2f MB\n", (N * N * sizeof(float)) / (1024.0 * 1024.0));

    // Allocate host memory
    size_t bytes = N * N * sizeof(float);
    float *h_A = (float*)malloc(bytes);
    float *h_B = (float*)malloc(bytes);
    float *h_C = (float*)malloc(bytes);

    // Initialize matrices
    srand(42);  // Fixed seed for reproducibility
    init_matrix(h_A, N);
    init_matrix(h_B, N);

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_C, bytes));

    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    // Configure kernel launch parameters
    // Using 32x32 thread blocks (1024 threads per block)
    dim3 blockDim(32, 32);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x,
                 (N + blockDim.y - 1) / blockDim.y);

    printf("\nKernel configuration:\n");
    printf("  Block size: %d x %d = %d threads\n", blockDim.x, blockDim.y,
           blockDim.x * blockDim.y);
    printf("  Grid size: %d x %d = %d blocks\n", gridDim.x, gridDim.y,
           gridDim.x * gridDim.y);
    printf("  Total threads: %d\n", blockDim.x * blockDim.y * gridDim.x * gridDim.y);

    // Warm-up run
    matmul_naive<<<gridDim, blockDim>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Benchmark
    int num_runs = 10;
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < num_runs; i++) {
        matmul_naive<<<gridDim, blockDim>>>(d_A, d_B, d_C, N);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    float avg_time = milliseconds / num_runs;

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));

    // Calculate performance metrics
    // For NxN matrix multiplication: 2*N^3 FLOPs (N^3 multiplies + N^3 adds)
    double flops = 2.0 * N * N * N;
    double gflops = (flops / (avg_time / 1000.0)) / 1e9;

    // Memory bandwidth: 3 matrices read/written (but C is only written once)
    // Actually: A is read N times per row, B is read N times per column
    // Total memory traffic = N^2 reads from A + N^2 reads from B + N^2 writes to C
    // But this is minimum - actual traffic is much higher due to cache misses
    double memory_gb = (3.0 * bytes) / 1e9;
    double bandwidth = memory_gb / (avg_time / 1000.0);

    printf("\n=== Performance Results ===\n");
    printf("Average execution time: %.3f ms\n", avg_time);
    printf("Performance: %.2f GFLOPS\n", gflops);
    printf("Effective bandwidth: %.2f GB/s (minimum theoretical)\n", bandwidth);

    // Get device properties for comparison
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    double peak_bandwidth = (prop.memoryClockRate * 1000.0 * (prop.memoryBusWidth / 8) * 2) / 1e9;
    printf("Peak memory bandwidth: %.2f GB/s\n", peak_bandwidth);
    printf("Bandwidth utilization: %.1f%%\n", (bandwidth / peak_bandwidth) * 100.0);

    // Verify correctness (only for small matrices to avoid long CPU computation)
    if (N <= 512) {
        printf("\n=== Verification ===\n");
        printf("Verifying result against CPU computation...\n");
        if (verify_result(h_A, h_B, h_C, N)) {
            printf("SUCCESS: Results match!\n");
        } else {
            printf("FAILURE: Results do not match!\n");
        }
    } else {
        printf("\nSkipping verification for large matrix (N > 512)\n");
    }

    // Analysis and recommendations
    printf("\n=== Analysis ===\n");
    printf("Issues with naive implementation:\n");
    printf("1. Poor memory access pattern for matrix B (strided access)\n");
    printf("2. No data reuse - each element fetched from global memory\n");
    printf("3. Low arithmetic intensity (FLOP/byte ratio)\n");
    printf("4. Each thread performs %d FLOPs but loads %d elements\n", 2 * N, 2 * N);
    printf("\nExpected performance: ~1-2%% of cuBLAS (300-500 GFLOPS on A100)\n");
    printf("Next optimization: Memory coalescing (example 02)\n");

    // Cleanup
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
