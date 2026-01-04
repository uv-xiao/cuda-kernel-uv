#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

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

/**
 * Naive transpose: Simple but has non-coalesced writes
 */
__global__ void transposeNaive(float *input, float *output, int width, int height) {
    // Calculate global position
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    // Boundary check
    if (row < height && col < width) {
        // Coalesced read from input
        int input_idx = row * width + col;

        // Non-coalesced write to output (strided by height)
        int output_idx = col * height + row;

        output[output_idx] = input[input_idx];
    }
}

/**
 * Shared memory transpose: Enables coalesced reads AND writes
 */
__global__ void transposeShared(float *input, float *output, int width, int height) {
    // Shared memory tile
    __shared__ float tile[TILE_DIM][TILE_DIM];

    // Global coordinates for input
    int col = blockIdx.x * TILE_DIM + threadIdx.x;
    int row = blockIdx.y * TILE_DIM + threadIdx.y;

    // Load input tile into shared memory (coalesced read)
    if (row < height && col < width) {
        tile[threadIdx.y][threadIdx.x] = input[row * width + col];
    }

    // Wait for all threads to finish loading
    __syncthreads();

    // Global coordinates for output (note: blockIdx swapped)
    col = blockIdx.y * TILE_DIM + threadIdx.x;
    row = blockIdx.x * TILE_DIM + threadIdx.y;

    // Write transposed tile to output (coalesced write)
    // Note: reading tile with swapped indices performs the transpose
    if (row < width && col < height) {
        output[row * height + col] = tile[threadIdx.x][threadIdx.y];
    }
}

/**
 * Optimized transpose: Avoids bank conflicts with padding
 */
__global__ void transposeOptimized(float *input, float *output, int width, int height) {
    // Shared memory with padding to avoid bank conflicts
    // The +1 shifts the column addresses to avoid conflicts
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];

    // Global coordinates for input
    int col = blockIdx.x * TILE_DIM + threadIdx.x;
    int row = blockIdx.y * TILE_DIM + threadIdx.y;

    // Load input tile (coalesced read)
    if (row < height && col < width) {
        tile[threadIdx.y][threadIdx.x] = input[row * width + col];
    }

    __syncthreads();

    // Global coordinates for output
    col = blockIdx.y * TILE_DIM + threadIdx.x;
    row = blockIdx.x * TILE_DIM + threadIdx.y;

    // Write transposed tile (coalesced write, no bank conflicts)
    if (row < width && col < height) {
        output[row * height + col] = tile[threadIdx.x][threadIdx.y];
    }
}

// Host verification function
bool verifyTranspose(float *input, float *output, int width, int height) {
    for (int row = 0; row < height; row++) {
        for (int col = 0; col < width; col++) {
            int input_idx = row * width + col;
            int output_idx = col * height + row;
            if (fabs(input[input_idx] - output[output_idx]) > 1e-5) {
                printf("Error at (%d, %d): input=%.2f, output=%.2f\n",
                       row, col, input[input_idx], output[output_idx]);
                return false;
            }
        }
    }
    return true;
}

float measureBandwidth(float ms, int width, int height) {
    // Transpose: read once, write once = 2x data size
    float bytes = 2.0f * width * height * sizeof(float);
    return (bytes / (1024 * 1024 * 1024)) / (ms / 1000.0f);
}

int main(int argc, char **argv) {
    printf("=== Matrix Transpose Exercise - SOLUTION ===\n\n");

    // Matrix dimensions
    int width = 4096;
    int height = 4096;
    size_t bytes = width * height * sizeof(float);

    printf("Matrix: %d x %d (%.2f MB)\n", width, height,
           bytes / (1024.0 * 1024.0));
    printf("Tile size: %d x %d\n\n", TILE_DIM, TILE_DIM);

    // Allocate and initialize host memory
    float *h_input = (float *)malloc(bytes);
    float *h_output = (float *)malloc(bytes);

    for (int i = 0; i < width * height; i++) {
        h_input[i] = (float)i;
    }

    // Allocate device memory
    float *d_input, *d_output;
    CHECK_CUDA(cudaMalloc(&d_input, bytes));
    CHECK_CUDA(cudaMalloc(&d_output, bytes));
    CHECK_CUDA(cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice));

    // Setup execution configuration
    dim3 blockSize(TILE_DIM, TILE_DIM);
    dim3 gridSize((width + TILE_DIM - 1) / TILE_DIM,
                  (height + TILE_DIM - 1) / TILE_DIM);

    printf("Grid: (%d, %d), Block: (%d, %d)\n\n",
           gridSize.x, gridSize.y, blockSize.x, blockSize.y);

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    float ms;

    // Test 1: Naive transpose
    printf("Test 1: Naive Transpose\n");
    printf("------------------------\n");

    CHECK_CUDA(cudaEventRecord(start));
    transposeNaive<<<gridSize, blockSize>>>(d_input, d_output, width, height);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

    CHECK_CUDA(cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost));

    bool naive_correct = verifyTranspose(h_input, h_output, width, height);
    float naive_bw = measureBandwidth(ms, width, height);
    float naive_ms = ms;

    printf("Result: %s\n", naive_correct ? "PASSED" : "FAILED");
    printf("Time: %.3f ms\n", ms);
    printf("Bandwidth: %.2f GB/s\n\n", naive_bw);

    // Test 2: Shared memory transpose
    printf("Test 2: Shared Memory Transpose\n");
    printf("--------------------------------\n");

    CHECK_CUDA(cudaEventRecord(start));
    transposeShared<<<gridSize, blockSize>>>(d_input, d_output, width, height);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

    CHECK_CUDA(cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost));

    bool shared_correct = verifyTranspose(h_input, h_output, width, height);
    float shared_bw = measureBandwidth(ms, width, height);

    printf("Result: %s\n", shared_correct ? "PASSED" : "FAILED");
    printf("Time: %.3f ms\n", ms);
    printf("Bandwidth: %.2f GB/s\n", shared_bw);
    printf("Speedup: %.2fx\n\n", naive_ms / ms);

    // Test 3: Optimized transpose
    printf("Test 3: Optimized Transpose (No Bank Conflicts)\n");
    printf("------------------------------------------------\n");

    CHECK_CUDA(cudaEventRecord(start));
    transposeOptimized<<<gridSize, blockSize>>>(d_input, d_output, width, height);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

    CHECK_CUDA(cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost));

    bool opt_correct = verifyTranspose(h_input, h_output, width, height);
    float opt_bw = measureBandwidth(ms, width, height);

    printf("Result: %s\n", opt_correct ? "PASSED" : "FAILED");
    printf("Time: %.3f ms\n", ms);
    printf("Bandwidth: %.2f GB/s\n", opt_bw);
    printf("Speedup vs naive: %.2fx\n\n", naive_ms / ms);

    // Summary
    printf("=== Summary ===\n");
    printf("All tests: %s\n",
           (naive_correct && shared_correct && opt_correct) ? "PASSED" : "FAILED");

    printf("\nKey Learnings:\n");
    printf("1. Naive transpose has non-coalesced writes\n");
    printf("2. Shared memory enables coalesced reads AND writes\n");
    printf("3. Padding eliminates bank conflicts\n");
    printf("4. Memory access patterns dominate performance\n");

    // Cleanup
    free(h_input);
    free(h_output);
    cudaFree(d_input);
    cudaFree(d_output);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
