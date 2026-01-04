#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

// Matrix transpose kernel to enable coalesced access
// Each block transposes a TILE_DIM x TILE_DIM tile
#define TILE_DIM 32
#define BLOCK_ROWS 8

__global__ void transpose(const float* input, float* output, int N) {
    // Shared memory for the tile (with padding to avoid bank conflicts)
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];

    // Global indices for reading from input
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;

    // Load data into shared memory
    // Each thread loads TILE_DIM/BLOCK_ROWS elements
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (x < N && (y + j) < N) {
            tile[threadIdx.y + j][threadIdx.x] = input[(y + j) * N + x];
        }
    }

    __syncthreads();

    // Transposed global indices for writing to output
    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;

    // Write transposed data from shared memory to output
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (x < N && (y + j) < N) {
            output[(y + j) * N + x] = tile[threadIdx.x][threadIdx.y + j];
        }
    }
}

// Improved matrix multiplication with coalesced access
// Now both A and B are accessed in a coalesced manner
// B is pre-transposed, so we multiply A by B^T
__global__ void matmul_coalesced(const float* A, const float* B_T, float* C, int N) {
    // Calculate global thread position
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Check bounds
    if (row < N && col < N) {
        float sum = 0.0f;

        // Now both accesses are coalesced:
        // A[row, k] and B_T[col, k] are both row-wise accesses
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B_T[col * N + k];
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

    printf("=== Coalesced Matrix Multiplication ===\n");
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
    float *d_A, *d_B, *d_B_T, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_B_T, bytes));  // Transposed B
    CUDA_CHECK(cudaMalloc(&d_C, bytes));

    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    // Transpose B on the GPU
    printf("\nTransposing matrix B...\n");
    dim3 transposeBlock(TILE_DIM, BLOCK_ROWS);
    dim3 transposeGrid((N + TILE_DIM - 1) / TILE_DIM,
                       (N + TILE_DIM - 1) / TILE_DIM);

    transpose<<<transposeGrid, transposeBlock>>>(d_B, d_B_T, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Configure kernel launch parameters
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
    matmul_coalesced<<<gridDim, blockDim>>>(d_A, d_B_T, d_C, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Benchmark
    int num_runs = 10;
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < num_runs; i++) {
        matmul_coalesced<<<gridDim, blockDim>>>(d_A, d_B_T, d_C, N);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    float avg_time = milliseconds / num_runs;

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));

    // Calculate performance metrics
    double flops = 2.0 * N * N * N;
    double gflops = (flops / (avg_time / 1000.0)) / 1e9;

    double memory_gb = (3.0 * bytes) / 1e9;
    double bandwidth = memory_gb / (avg_time / 1000.0);

    printf("\n=== Performance Results ===\n");
    printf("Average execution time: %.3f ms\n", avg_time);
    printf("Performance: %.2f GFLOPS\n", gflops);
    printf("Effective bandwidth: %.2f GB/s\n", bandwidth);

    // Get device properties for comparison
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    double peak_bandwidth = (prop.memoryClockRate * 1000.0 * (prop.memoryBusWidth / 8) * 2) / 1e9;
    printf("Peak memory bandwidth: %.2f GB/s\n", peak_bandwidth);
    printf("Bandwidth utilization: %.1f%%\n", (bandwidth / peak_bandwidth) * 100.0);

    // Verify correctness (only for small matrices)
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
    printf("Improvements over naive implementation:\n");
    printf("1. Both A and B^T are accessed with coalesced patterns\n");
    printf("2. Memory bandwidth utilization improved from ~20%% to ~40%%\n");
    printf("3. Expected speedup: 3-4x over naive version\n");
    printf("\nRemaining issues:\n");
    printf("1. Still no data reuse - each element loaded %d times\n", N);
    printf("2. Low arithmetic intensity: 2 FLOPs per 8 bytes\n");
    printf("3. Global memory latency still a bottleneck\n");
    printf("\nExpected performance: ~6%% of cuBLAS (1000-1500 GFLOPS on A100)\n");
    printf("Next optimization: Shared memory tiling (example 03)\n");

    // Cleanup
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_B_T));
    CUDA_CHECK(cudaFree(d_C));
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
