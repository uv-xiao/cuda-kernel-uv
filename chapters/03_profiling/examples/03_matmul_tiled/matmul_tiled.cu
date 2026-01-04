#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

// Tile size for shared memory
// 32x32 gives 4KB of shared memory per block (32*32*4 bytes)
#define TILE_SIZE 32

// Matrix multiplication using shared memory tiling (1D blocking)
// Each thread block computes a TILE_SIZE x TILE_SIZE output tile
// by loading tiles of A and B into shared memory
__global__ void matmul_tiled(const float* A, const float* B, float* C, int N) {
    // Thread block computes C[blockRow:blockRow+TILE_SIZE, blockCol:blockCol+TILE_SIZE]
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;

    // Each thread computes one element of the block sub-matrix
    int row = threadIdx.y;
    int col = threadIdx.x;

    // Accumulator for dot product
    float sum = 0.0f;

    // Shared memory for tiles of A and B
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    // Loop over tiles of A and B required to compute the block sub-matrix
    int numTiles = (N + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < numTiles; t++) {
        // Load tile of A into shared memory
        int aRow = blockRow * TILE_SIZE + row;
        int aCol = t * TILE_SIZE + col;

        if (aRow < N && aCol < N) {
            As[row][col] = A[aRow * N + aCol];
        } else {
            As[row][col] = 0.0f;  // Padding for out-of-bounds
        }

        // Load tile of B into shared memory
        int bRow = t * TILE_SIZE + row;
        int bCol = blockCol * TILE_SIZE + col;

        if (bRow < N && bCol < N) {
            Bs[row][col] = B[bRow * N + bCol];
        } else {
            Bs[row][col] = 0.0f;  // Padding for out-of-bounds
        }

        // Synchronize to ensure tiles are loaded before computation
        __syncthreads();

        // Compute partial dot product for this tile
        // This is where the magic happens: we reuse data from shared memory!
        // Each element is loaded once from global memory but used TILE_SIZE times
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += As[row][k] * Bs[k][col];
        }

        // Synchronize before loading next tile
        __syncthreads();
    }

    // Write result to global memory
    int cRow = blockRow * TILE_SIZE + row;
    int cCol = blockCol * TILE_SIZE + col;

    if (cRow < N && cCol < N) {
        C[cRow * N + cCol] = sum;
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
    const float epsilon = 1e-2f;  // Slightly larger due to accumulation order differences
    int errors = 0;
    const int max_errors = 10;

    for (int i = 0; i < N && errors < max_errors; i++) {
        for (int j = 0; j < N && errors < max_errors; j++) {
            float sum = 0.0f;
            for (int k = 0; k < N; k++) {
                sum += A[i * N + k] * B[k * N + j];
            }

            float diff = fabs(C[i * N + j] - sum);
            if (diff > epsilon) {
                printf("Mismatch at (%d, %d): GPU=%.3f, CPU=%.3f, diff=%.3f\n",
                       i, j, C[i * N + j], sum, diff);
                errors++;
            }
        }
    }
    return errors == 0;
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

    printf("=== Tiled Matrix Multiplication (Shared Memory) ===\n");
    printf("Matrix size: %d x %d\n", N, N);
    printf("Tile size: %d x %d\n", TILE_SIZE, TILE_SIZE);
    printf("Total elements: %d\n", N * N);
    printf("Memory per matrix: %.2f MB\n", (N * N * sizeof(float)) / (1024.0 * 1024.0));
    printf("Shared memory per block: %.2f KB\n",
           (2 * TILE_SIZE * TILE_SIZE * sizeof(float)) / 1024.0);

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
    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((N + TILE_SIZE - 1) / TILE_SIZE,
                 (N + TILE_SIZE - 1) / TILE_SIZE);

    printf("\nKernel configuration:\n");
    printf("  Block size: %d x %d = %d threads\n", blockDim.x, blockDim.y,
           blockDim.x * blockDim.y);
    printf("  Grid size: %d x %d = %d blocks\n", gridDim.x, gridDim.y,
           gridDim.x * gridDim.y);
    printf("  Total threads: %d\n", blockDim.x * blockDim.y * gridDim.x * gridDim.y);
    printf("  Number of tiles per dimension: %d\n", (N + TILE_SIZE - 1) / TILE_SIZE);

    // Warm-up run
    matmul_tiled<<<gridDim, blockDim>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Benchmark
    int num_runs = 10;
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < num_runs; i++) {
        matmul_tiled<<<gridDim, blockDim>>>(d_A, d_B, d_C, N);
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

    // With tiling, theoretical data movement is much less
    // Each element is loaded (N/TILE_SIZE) times instead of N times
    double elements_loaded = N * N * N * 2.0 / TILE_SIZE;  // Approximate
    double memory_gb = (elements_loaded * sizeof(float)) / 1e9;
    double bandwidth = memory_gb / (avg_time / 1000.0);

    printf("\n=== Performance Results ===\n");
    printf("Average execution time: %.3f ms\n", avg_time);
    printf("Performance: %.2f GFLOPS\n", gflops);
    printf("Effective bandwidth: %.2f GB/s\n", bandwidth);

    // Get device properties for comparison
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    double peak_bandwidth = (prop.memoryClockRate * 1000.0 * (prop.memoryBusWidth / 8) * 2) / 1e9;
    double peak_gflops = prop.clockRate * 1000.0 * prop.multiProcessorCount *
                         128 / 1e9;  // Approximate for FP32

    printf("Peak memory bandwidth: %.2f GB/s\n", peak_bandwidth);
    printf("Bandwidth utilization: %.1f%%\n", (bandwidth / peak_bandwidth) * 100.0);
    printf("Peak compute (estimated): %.2f GFLOPS\n", peak_gflops);
    printf("Compute utilization: %.1f%%\n", (gflops / peak_gflops) * 100.0);

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
    printf("Improvements over coalesced implementation:\n");
    printf("1. Data reuse through shared memory (TILE_SIZE = %d)\n", TILE_SIZE);
    printf("2. Each global memory load is reused %d times\n", TILE_SIZE);
    printf("3. Memory traffic reduced by factor of %d\n", TILE_SIZE);
    printf("4. Arithmetic intensity increased from 0.25 to %.1f FLOP/byte\n",
           TILE_SIZE * 0.25f);
    printf("\nExpected performance: ~20-30%% of cuBLAS (4000-6000 GFLOPS on A100)\n");
    printf("Next optimization: 2D tiling + vectorization (example 04)\n");

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
