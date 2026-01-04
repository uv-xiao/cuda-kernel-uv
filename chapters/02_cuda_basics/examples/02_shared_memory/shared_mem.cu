#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define CHECK_CUDA(call)                                                     \
    {                                                                        \
        cudaError_t err = call;                                              \
        if (err != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err));                                \
            exit(EXIT_FAILURE);                                              \
        }                                                                    \
    }

#define TILE_DIM 32
#define BLOCK_ROWS 8

/**
 * Naive matrix transpose - no optimization
 * Non-coalesced writes lead to poor performance
 */
__global__ void transposeNaive(float *input, float *output, int width, int height) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < height && col < width) {
        // Coalesced read, non-coalesced write
        output[col * height + row] = input[row * width + col];
    }
}

/**
 * Matrix transpose using shared memory - basic version
 * Uses shared memory to enable coalesced reads and writes
 */
__global__ void transposeShared(float *input, float *output, int width, int height) {
    // Shared memory tile - one per block
    __shared__ float tile[TILE_DIM][TILE_DIM];

    // Global input coordinates
    int col = blockIdx.x * TILE_DIM + threadIdx.x;
    int row = blockIdx.y * TILE_DIM + threadIdx.y;

    // Load tile into shared memory (coalesced read)
    if (row < height && col < width) {
        tile[threadIdx.y][threadIdx.x] = input[row * width + col];
    }

    // Wait for all threads in block to finish loading
    __syncthreads();

    // Global output coordinates (transposed)
    col = blockIdx.y * TILE_DIM + threadIdx.x;
    row = blockIdx.x * TILE_DIM + threadIdx.y;

    // Write transposed tile to output (coalesced write)
    if (row < width && col < height) {
        output[row * height + col] = tile[threadIdx.x][threadIdx.y];
    }
}

/**
 * Optimized transpose with padding to avoid bank conflicts
 * Adds one extra column to shared memory to shift bank accesses
 */
__global__ void transposeSharedNoBankConflicts(float *input, float *output,
                                                int width, int height) {
    // Shared memory with padding (+1 to avoid bank conflicts)
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];

    int col = blockIdx.x * TILE_DIM + threadIdx.x;
    int row = blockIdx.y * TILE_DIM + threadIdx.y;

    // Load tile into shared memory (coalesced read)
    if (row < height && col < width) {
        tile[threadIdx.y][threadIdx.x] = input[row * width + col];
    }

    __syncthreads();

    // Global output coordinates (transposed)
    col = blockIdx.y * TILE_DIM + threadIdx.x;
    row = blockIdx.x * TILE_DIM + threadIdx.y;

    // Write transposed tile to output (coalesced write)
    if (row < width && col < height) {
        output[row * height + col] = tile[threadIdx.x][threadIdx.y];
    }
}

/**
 * Demonstrates shared memory for data reuse
 * Each thread loads once but reuses data multiple times
 */
__global__ void matrixMulShared(float *A, float *B, float *C, int N) {
    __shared__ float As[TILE_DIM][TILE_DIM];
    __shared__ float Bs[TILE_DIM][TILE_DIM];

    int row = blockIdx.y * TILE_DIM + threadIdx.y;
    int col = blockIdx.x * TILE_DIM + threadIdx.x;

    float sum = 0.0f;

    // Loop over tiles
    for (int t = 0; t < (N + TILE_DIM - 1) / TILE_DIM; t++) {
        // Load tile into shared memory
        if (row < N && (t * TILE_DIM + threadIdx.x) < N) {
            As[threadIdx.y][threadIdx.x] = A[row * N + t * TILE_DIM + threadIdx.x];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }

        if ((t * TILE_DIM + threadIdx.y) < N && col < N) {
            Bs[threadIdx.y][threadIdx.x] = B[(t * TILE_DIM + threadIdx.y) * N + col];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        // Compute partial product using shared memory
        for (int k = 0; k < TILE_DIM; k++) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}

void initMatrix(float *mat, int width, int height) {
    for (int i = 0; i < width * height; i++) {
        mat[i] = (float)(rand() % 100) / 10.0f;
    }
}

void printMatrix(const char *name, float *matrix, int width, int height, int max_display = 8) {
    printf("\n%s (%dx%d):\n", name, width, height);
    int display_rows = (height < max_display) ? height : max_display;
    int display_cols = (width < max_display) ? width : max_display;

    for (int row = 0; row < display_rows; row++) {
        for (int col = 0; col < display_cols; col++) {
            printf("%.1f ", matrix[row * width + col]);
        }
        if (width > max_display) printf("...");
        printf("\n");
    }
    if (height > max_display) printf("...\n");
}

bool verifyTranspose(float *input, float *output, int width, int height) {
    for (int row = 0; row < height; row++) {
        for (int col = 0; col < width; col++) {
            int input_idx = row * width + col;
            int output_idx = col * height + row;
            if (fabs(input[input_idx] - output[output_idx]) > 1e-5) {
                printf("Mismatch at (%d,%d): input=%.2f, output=%.2f\n",
                       row, col, input[input_idx], output[output_idx]);
                return false;
            }
        }
    }
    return true;
}

float measureBandwidth(float ms, int width, int height) {
    // Each transpose: read + write = 2 * size
    float bytes = 2.0f * width * height * sizeof(float);
    float bandwidth = (bytes / (1024 * 1024 * 1024)) / (ms / 1000.0f);
    return bandwidth;
}

int main(int argc, char **argv) {
    printf("=== CUDA Shared Memory Examples ===\n\n");

    srand(time(NULL));

    // Matrix dimensions (use power of 2 for simplicity)
    int width = 2048;
    int height = 2048;
    size_t bytes = width * height * sizeof(float);

    printf("Matrix size: %d x %d (%.2f MB)\n",
           width, height, bytes / (1024.0 * 1024.0));
    printf("Tile size: %d x %d\n\n", TILE_DIM, TILE_DIM);

    // Allocate host memory
    float *h_input = (float *)malloc(bytes);
    float *h_output = (float *)malloc(bytes);

    // Initialize input matrix
    initMatrix(h_input, width, height);

    // Allocate device memory
    float *d_input, *d_output;
    CHECK_CUDA(cudaMalloc(&d_input, bytes));
    CHECK_CUDA(cudaMalloc(&d_output, bytes));
    CHECK_CUDA(cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice));

    // Setup execution configuration
    dim3 blockSize(TILE_DIM, TILE_DIM);
    dim3 gridSize((width + TILE_DIM - 1) / TILE_DIM,
                  (height + TILE_DIM - 1) / TILE_DIM);

    printf("Execution Configuration:\n");
    printf("  Block size: (%d, %d) = %d threads\n",
           blockSize.x, blockSize.y, blockSize.x * blockSize.y);
    printf("  Grid size:  (%d, %d) = %d blocks\n\n",
           gridSize.x, gridSize.y, gridSize.x * gridSize.y);

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // ===== Benchmark 1: Naive Transpose =====
    printf("1. Naive Transpose (no optimization)\n");
    printf("-------------------------------------\n");

    CHECK_CUDA(cudaEventRecord(start));
    transposeNaive<<<gridSize, blockSize>>>(d_input, d_output, width, height);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms_naive;
    CHECK_CUDA(cudaEventElapsedTime(&ms_naive, start, stop));
    float bw_naive = measureBandwidth(ms_naive, width, height);

    CHECK_CUDA(cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost));

    bool naive_correct = verifyTranspose(h_input, h_output, width, height);
    printf("  Correctness: %s\n", naive_correct ? "PASSED" : "FAILED");
    printf("  Time: %.3f ms\n", ms_naive);
    printf("  Bandwidth: %.2f GB/s\n\n", bw_naive);

    // ===== Benchmark 2: Shared Memory Transpose =====
    printf("2. Shared Memory Transpose\n");
    printf("---------------------------\n");

    CHECK_CUDA(cudaEventRecord(start));
    transposeShared<<<gridSize, blockSize>>>(d_input, d_output, width, height);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms_shared;
    CHECK_CUDA(cudaEventElapsedTime(&ms_shared, start, stop));
    float bw_shared = measureBandwidth(ms_shared, width, height);

    CHECK_CUDA(cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost));

    bool shared_correct = verifyTranspose(h_input, h_output, width, height);
    printf("  Correctness: %s\n", shared_correct ? "PASSED" : "FAILED");
    printf("  Time: %.3f ms\n", ms_shared);
    printf("  Bandwidth: %.2f GB/s\n", bw_shared);
    printf("  Speedup: %.2fx\n\n", ms_naive / ms_shared);

    // ===== Benchmark 3: No Bank Conflicts =====
    printf("3. Shared Memory + No Bank Conflicts\n");
    printf("-------------------------------------\n");

    CHECK_CUDA(cudaEventRecord(start));
    transposeSharedNoBankConflicts<<<gridSize, blockSize>>>(d_input, d_output, width, height);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms_optimized;
    CHECK_CUDA(cudaEventElapsedTime(&ms_optimized, start, stop));
    float bw_optimized = measureBandwidth(ms_optimized, width, height);

    CHECK_CUDA(cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost));

    bool optimized_correct = verifyTranspose(h_input, h_output, width, height);
    printf("  Correctness: %s\n", optimized_correct ? "PASSED" : "FAILED");
    printf("  Time: %.3f ms\n", ms_optimized);
    printf("  Bandwidth: %.2f GB/s\n", bw_optimized);
    printf("  Speedup over naive: %.2fx\n", ms_naive / ms_optimized);
    printf("  Speedup over shared: %.2fx\n\n", ms_shared / ms_optimized);

    // ===== Example 4: Matrix Multiplication =====
    printf("4. Matrix Multiplication with Shared Memory\n");
    printf("--------------------------------------------\n");

    int N = 512;  // Smaller for demonstration
    size_t mat_bytes = N * N * sizeof(float);

    float *h_A = (float *)malloc(mat_bytes);
    float *h_B = (float *)malloc(mat_bytes);
    float *h_C = (float *)malloc(mat_bytes);

    initMatrix(h_A, N, N);
    initMatrix(h_B, N, N);

    float *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc(&d_A, mat_bytes));
    CHECK_CUDA(cudaMalloc(&d_B, mat_bytes));
    CHECK_CUDA(cudaMalloc(&d_C, mat_bytes));

    CHECK_CUDA(cudaMemcpy(d_A, h_A, mat_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, mat_bytes, cudaMemcpyHostToDevice));

    dim3 matmul_block(TILE_DIM, TILE_DIM);
    dim3 matmul_grid((N + TILE_DIM - 1) / TILE_DIM,
                     (N + TILE_DIM - 1) / TILE_DIM);

    CHECK_CUDA(cudaEventRecord(start));
    matrixMulShared<<<matmul_grid, matmul_block>>>(d_A, d_B, d_C, N);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms_matmul;
    CHECK_CUDA(cudaEventElapsedTime(&ms_matmul, start, stop));

    CHECK_CUDA(cudaMemcpy(h_C, d_C, mat_bytes, cudaMemcpyDeviceToHost));

    printf("  Matrix size: %d x %d\n", N, N);
    printf("  Time: %.3f ms\n", ms_matmul);
    printf("  GFLOPS: %.2f\n\n",
           (2.0f * N * N * N / 1e9) / (ms_matmul / 1000.0f));

    // Show small example
    printf("Sample matrices (8x8):\n");
    printMatrix("A", h_A, 8, 8, 8);
    printMatrix("B", h_B, 8, 8, 8);
    printMatrix("C = A * B", h_C, 8, 8, 8);

    // ===== Summary =====
    printf("\n=== Performance Summary ===\n");
    printf("%-30s %10s %12s\n", "Kernel", "Time (ms)", "Bandwidth");
    printf("%-30s %10.3f %9.2f GB/s\n", "Naive Transpose", ms_naive, bw_naive);
    printf("%-30s %10.3f %9.2f GB/s\n", "Shared Memory", ms_shared, bw_shared);
    printf("%-30s %10.3f %9.2f GB/s\n", "No Bank Conflicts", ms_optimized, bw_optimized);
    printf("\n");

    printf("Key Insights:\n");
    printf("1. Shared memory enables coalesced reads AND writes\n");
    printf("2. Bank conflict padding provides additional speedup\n");
    printf("3. Shared memory enables data reuse (matrix multiply)\n");
    printf("4. __syncthreads() is essential for correctness\n");

    // Cleanup
    free(h_input);
    free(h_output);
    free(h_A);
    free(h_B);
    free(h_C);

    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    printf("\n=== All tests completed successfully! ===\n");

    return 0;
}
