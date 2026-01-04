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

/**
 * Kernel demonstrating 1D indexing for a flattened 2D array
 * Each thread processes one matrix element
 */
__global__ void matrixAdd1D(float *A, float *B, float *C, int width, int height) {
    // Calculate 1D thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = width * height;

    if (idx < total_elements) {
        C[idx] = A[idx] + B[idx];
    }
}

/**
 * Kernel demonstrating 2D indexing for matrix operations
 * Uses 2D blocks and grids for intuitive row/column mapping
 */
__global__ void matrixAdd2D(float *A, float *B, float *C, int width, int height) {
    // Calculate row and column from 2D thread indices
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    // Convert 2D indices to 1D array index (row-major order)
    int idx = row * width + col;

    // Boundary check
    if (row < height && col < width) {
        C[idx] = A[idx] + B[idx];
    }
}

/**
 * Kernel demonstrating different indexing patterns
 * Shows column-major vs row-major access
 */
__global__ void matrixTranspose(float *input, float *output, int width, int height) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < height && col < width) {
        // Row-major read
        int input_idx = row * width + col;

        // Column-major write (transpose)
        int output_idx = col * height + row;

        output[output_idx] = input[input_idx];
    }
}

/**
 * Kernel showing thread and block ID usage
 * Useful for debugging and understanding thread organization
 */
__global__ void printThreadInfo(int width, int height) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < height && col < width) {
        printf("Thread (%d,%d) in Block (%d,%d): Global position (%d,%d)\n",
               threadIdx.x, threadIdx.y,
               blockIdx.x, blockIdx.y,
               col, row);
    }
}

void initializeMatrix(float *matrix, int width, int height, float value) {
    for (int i = 0; i < width * height; i++) {
        matrix[i] = value;
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

bool verifyResults(float *A, float *B, float *C, int size) {
    for (int i = 0; i < size; i++) {
        float expected = A[i] + B[i];
        if (fabs(C[i] - expected) > 1e-5) {
            printf("Verification failed at index %d: expected %.2f, got %.2f\n",
                   i, expected, C[i]);
            return false;
        }
    }
    return true;
}

int main(int argc, char **argv) {
    printf("=== CUDA 2D Indexing Examples ===\n\n");

    // Matrix dimensions
    int width = 1024;
    int height = 1024;
    int size = width * height;
    size_t bytes = size * sizeof(float);

    printf("Matrix size: %d x %d (%d elements, %.2f MB)\n\n",
           width, height, size, bytes / (1024.0 * 1024.0));

    // Allocate host memory
    float *h_A = (float *)malloc(bytes);
    float *h_B = (float *)malloc(bytes);
    float *h_C = (float *)malloc(bytes);

    // Initialize matrices
    initializeMatrix(h_A, width, height, 1.0f);
    initializeMatrix(h_B, width, height, 2.0f);

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc(&d_A, bytes));
    CHECK_CUDA(cudaMalloc(&d_B, bytes));
    CHECK_CUDA(cudaMalloc(&d_C, bytes));

    // Copy data to device
    CHECK_CUDA(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    // ===== Example 1: 1D Indexing =====
    printf("Example 1: 1D Indexing\n");
    printf("------------------------\n");

    int blockSize1D = 256;
    int gridSize1D = (size + blockSize1D - 1) / blockSize1D;

    printf("Configuration:\n");
    printf("  Block size: %d threads\n", blockSize1D);
    printf("  Grid size:  %d blocks\n", gridSize1D);
    printf("  Total threads: %d\n\n", blockSize1D * gridSize1D);

    matrixAdd1D<<<gridSize1D, blockSize1D>>>(d_A, d_B, d_C, width, height);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));

    if (verifyResults(h_A, h_B, h_C, size)) {
        printf("1D Indexing: PASSED\n\n");
    }

    // ===== Example 2: 2D Indexing =====
    printf("Example 2: 2D Indexing\n");
    printf("------------------------\n");

    dim3 blockSize2D(16, 16);  // 256 threads per block
    dim3 gridSize2D((width + blockSize2D.x - 1) / blockSize2D.x,
                     (height + blockSize2D.y - 1) / blockSize2D.y);

    printf("Configuration:\n");
    printf("  Block size: (%d, %d) = %d threads\n",
           blockSize2D.x, blockSize2D.y, blockSize2D.x * blockSize2D.y);
    printf("  Grid size:  (%d, %d) = %d blocks\n",
           gridSize2D.x, gridSize2D.y, gridSize2D.x * gridSize2D.y);
    printf("  Total threads: %d\n\n",
           blockSize2D.x * blockSize2D.y * gridSize2D.x * gridSize2D.y);

    matrixAdd2D<<<gridSize2D, blockSize2D>>>(d_A, d_B, d_C, width, height);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));

    if (verifyResults(h_A, h_B, h_C, size)) {
        printf("2D Indexing: PASSED\n\n");
    }

    // ===== Example 3: Thread Info (Small Matrix) =====
    printf("Example 3: Thread Organization (4x4 matrix)\n");
    printf("--------------------------------------------\n");

    int small_width = 4;
    int small_height = 4;

    dim3 small_block(2, 2);  // 2x2 threads per block
    dim3 small_grid(
        (small_width + small_block.x - 1) / small_block.x,
        (small_height + small_block.y - 1) / small_block.y
    );

    printf("Configuration:\n");
    printf("  Matrix: %dx%d\n", small_width, small_height);
    printf("  Block size: (%d, %d)\n", small_block.x, small_block.y);
    printf("  Grid size:  (%d, %d)\n\n", small_grid.x, small_grid.y);

    printf("Thread organization:\n");
    printThreadInfo<<<small_grid, small_block>>>(small_width, small_height);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    printf("\n");

    // ===== Example 4: Matrix Transpose =====
    printf("Example 4: Matrix Transpose (different access patterns)\n");
    printf("--------------------------------------------------------\n");

    int trans_size = 8;
    size_t trans_bytes = trans_size * trans_size * sizeof(float);

    float *h_input = (float *)malloc(trans_bytes);
    float *h_output = (float *)malloc(trans_bytes);

    // Initialize with sequential values
    for (int i = 0; i < trans_size * trans_size; i++) {
        h_input[i] = (float)i;
    }

    float *d_input, *d_output;
    CHECK_CUDA(cudaMalloc(&d_input, trans_bytes));
    CHECK_CUDA(cudaMalloc(&d_output, trans_bytes));
    CHECK_CUDA(cudaMemcpy(d_input, h_input, trans_bytes, cudaMemcpyHostToDevice));

    dim3 trans_block(4, 4);
    dim3 trans_grid(
        (trans_size + trans_block.x - 1) / trans_block.x,
        (trans_size + trans_block.y - 1) / trans_block.y
    );

    matrixTranspose<<<trans_grid, trans_block>>>(d_input, d_output, trans_size, trans_size);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(h_output, d_output, trans_bytes, cudaMemcpyDeviceToHost));

    printMatrix("Input Matrix", h_input, trans_size, trans_size, trans_size);
    printMatrix("Transposed Matrix", h_output, trans_size, trans_size, trans_size);

    // Verify transpose
    bool transpose_correct = true;
    for (int row = 0; row < trans_size; row++) {
        for (int col = 0; col < trans_size; col++) {
            int input_idx = row * trans_size + col;
            int output_idx = col * trans_size + row;
            if (h_input[input_idx] != h_output[output_idx]) {
                transpose_correct = false;
                break;
            }
        }
    }

    printf("\nTranspose: %s\n\n", transpose_correct ? "PASSED" : "FAILED");

    // ===== Key Concepts Summary =====
    printf("Key Concepts:\n");
    printf("=============\n");
    printf("1. 1D Indexing: idx = blockIdx.x * blockDim.x + threadIdx.x\n");
    printf("2. 2D Indexing: col = blockIdx.x * blockDim.x + threadIdx.x\n");
    printf("                row = blockIdx.y * blockDim.y + threadIdx.y\n");
    printf("3. 2D to 1D:    idx = row * width + col (row-major)\n");
    printf("4. Thread blocks are typically 256 threads (e.g., 16x16)\n");
    printf("5. Always perform boundary checks!\n");

    // Cleanup
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_input);
    free(h_output);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_input);
    cudaFree(d_output);

    printf("\n=== All tests completed successfully! ===\n");

    return 0;
}
