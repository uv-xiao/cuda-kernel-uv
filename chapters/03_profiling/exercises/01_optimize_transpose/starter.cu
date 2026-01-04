#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define TILE_DIM 32
#define BLOCK_ROWS 8

// Naive transpose: simple but slow (uncoalesced writes)
__global__ void transpose_naive(float* out, const float* in, int N) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < N && y < N) {
        // Read: coalesced
        // Write: uncoalesced (consecutive threads write to addresses N apart)
        out[x * N + y] = in[y * N + x];
    }
}

// Shared memory version with bank conflicts (needs optimization!)
__global__ void transpose_shared(float* out, const float* in, int N) {
    // TODO: Fix the bank conflicts!
    // Hint: What's wrong with this shared memory declaration?
    __shared__ float tile[TILE_DIM][TILE_DIM];

    // Calculate global indices for input
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;

    // Read from input into shared memory (coalesced)
    if (x < N && y < N) {
        tile[threadIdx.y][threadIdx.x] = in[y * N + x];
    }

    __syncthreads();

    // Calculate global indices for output (note the swap of blockIdx.x and blockIdx.y)
    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;

    // Write to output from shared memory (coalesced)
    // But reading from shared memory has bank conflicts!
    if (x < N && y < N) {
        out[y * N + x] = tile[threadIdx.x][threadIdx.y];
    }
}

// TODO: Implement your optimized version here
__global__ void transpose_optimized(float* out, const float* in, int N) {
    // Your code here!
    // Fix the bank conflicts in transpose_shared

    // Hint: The only change needed is in the shared memory declaration
}

// Initialize matrix
void init_matrix(float* mat, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            mat[i * N + j] = i * N + j;
        }
    }
}

// Verify transpose
bool verify_transpose(const float* A, const float* B, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (fabs(A[i * N + j] - B[j * N + i]) > 1e-5) {
                printf("Mismatch at (%d, %d): A[%d][%d]=%.1f, B[%d][%d]=%.1f\n",
                       i, j, i, j, A[i * N + j], j, i, B[j * N + i]);
                return false;
            }
        }
    }
    return true;
}

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
    int N = 4096;
    if (argc > 1) {
        N = atoi(argv[1]);
    }

    printf("=== Matrix Transpose Exercise ===\n");
    printf("Matrix size: %d x %d\n", N, N);

    size_t bytes = N * N * sizeof(float);
    float *h_A = (float*)malloc(bytes);
    float *h_B = (float*)malloc(bytes);

    init_matrix(h_A, N);

    float *d_A, *d_B;
    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));

    dim3 blockDim(TILE_DIM, TILE_DIM);
    dim3 gridDim((N + TILE_DIM - 1) / TILE_DIM, (N + TILE_DIM - 1) / TILE_DIM);

    // Benchmark naive version
    printf("\n=== Naive Transpose ===\n");
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    transpose_naive<<<gridDim, blockDim>>>(d_B, d_A, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    int num_runs = 10;
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < num_runs; i++) {
        transpose_naive<<<gridDim, blockDim>>>(d_B, d_A, N);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    float avg_time_naive = milliseconds / num_runs;

    CUDA_CHECK(cudaMemcpy(h_B, d_B, bytes, cudaMemcpyDeviceToHost));

    // Verify
    if (!verify_transpose(h_A, h_B, N)) {
        printf("ERROR: Naive transpose incorrect!\n");
        return 1;
    }
    printf("Correctness: PASSED\n");

    // Calculate bandwidth
    double bytes_transferred = 2.0 * bytes;  // Read + write
    double bandwidth_naive = (bytes_transferred / (avg_time_naive / 1000.0)) / 1e9;
    printf("Time: %.3f ms\n", avg_time_naive);
    printf("Bandwidth: %.2f GB/s\n", bandwidth_naive);

    // Get peak bandwidth
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    double peak_bandwidth = (prop.memoryClockRate * 1000.0 * (prop.memoryBusWidth / 8) * 2) / 1e9;
    printf("Peak bandwidth: %.2f GB/s\n", peak_bandwidth);
    printf("Utilization: %.1f%%\n", (bandwidth_naive / peak_bandwidth) * 100.0);

    // Benchmark shared memory version
    printf("\n=== Shared Memory Transpose (with bank conflicts) ===\n");

    transpose_shared<<<gridDim, blockDim>>>(d_B, d_A, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < num_runs; i++) {
        transpose_shared<<<gridDim, blockDim>>>(d_B, d_A, N);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    float avg_time_shared = milliseconds / num_runs;

    CUDA_CHECK(cudaMemcpy(h_B, d_B, bytes, cudaMemcpyDeviceToHost));

    if (!verify_transpose(h_A, h_B, N)) {
        printf("ERROR: Shared memory transpose incorrect!\n");
        return 1;
    }
    printf("Correctness: PASSED\n");

    double bandwidth_shared = (bytes_transferred / (avg_time_shared / 1000.0)) / 1e9;
    printf("Time: %.3f ms\n", avg_time_shared);
    printf("Bandwidth: %.2f GB/s\n", bandwidth_shared);
    printf("Utilization: %.1f%%\n", (bandwidth_shared / peak_bandwidth) * 100.0);
    printf("Speedup vs. naive: %.2fx\n", avg_time_naive / avg_time_shared);

    printf("\n=== Your Task ===\n");
    printf("Implement transpose_optimized to:\n");
    printf("1. Achieve >80%% bandwidth utilization (>%.0f GB/s)\n", peak_bandwidth * 0.8);
    printf("2. Eliminate all shared memory bank conflicts\n");
    printf("3. Maintain correctness\n");
    printf("\nHint: Add padding to the shared memory array!\n");

    // Cleanup
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    free(h_A);
    free(h_B);

    return 0;
}
