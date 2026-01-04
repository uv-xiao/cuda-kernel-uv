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
        out[x * N + y] = in[y * N + x];
    }
}

// Shared memory version with bank conflicts
__global__ void transpose_shared(float* out, const float* in, int N) {
    __shared__ float tile[TILE_DIM][TILE_DIM];

    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;

    if (x < N && y < N) {
        tile[threadIdx.y][threadIdx.x] = in[y * N + x];
    }

    __syncthreads();

    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;

    if (x < N && y < N) {
        out[y * N + x] = tile[threadIdx.x][threadIdx.y];
    }
}

// Optimized version: padding eliminates bank conflicts!
__global__ void transpose_optimized(float* out, const float* in, int N) {
    // The key optimization: +1 padding to avoid bank conflicts
    // When threads read tile[threadIdx.x][threadIdx.y], consecutive threads
    // access consecutive columns. Without padding, these map to the same bank.
    // With padding, the column stride is 33 instead of 32, avoiding conflicts.
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];

    // Read from global memory (coalesced)
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;

    if (x < N && y < N) {
        tile[threadIdx.y][threadIdx.x] = in[y * N + x];
    }

    __syncthreads();

    // Transpose block indices for output
    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;

    // Write to global memory (coalesced)
    // Read from shared memory is now bank-conflict-free!
    if (x < N && y < N) {
        out[y * N + x] = tile[threadIdx.x][threadIdx.y];
    }
}

// Advanced version: rectangular tiles for better performance
__global__ void transpose_rectangular(float* out, const float* in, int N) {
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];

    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;

    // Each thread loads TILE_DIM / BLOCK_ROWS elements
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (x < N && (y + j) < N) {
            tile[threadIdx.y + j][threadIdx.x] = in[(y + j) * N + x];
        }
    }

    __syncthreads();

    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (x < N && (y + j) < N) {
            out[(y + j) * N + x] = tile[threadIdx.x][threadIdx.y + j];
        }
    }
}

void init_matrix(float* mat, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            mat[i * N + j] = i * N + j;
        }
    }
}

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

void benchmark_kernel(const char* name, void (*kernel)(float*, const float*, int),
                      float* d_A, float* d_B, float* h_A, float* h_B, int N,
                      dim3 gridDim, dim3 blockDim) {
    printf("\n=== %s ===\n", name);

    size_t bytes = N * N * sizeof(float);

    // Warmup
    kernel<<<gridDim, blockDim>>>(d_B, d_A, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Benchmark
    int num_runs = 10;
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < num_runs; i++) {
        kernel<<<gridDim, blockDim>>>(d_B, d_A, N);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    float avg_time = milliseconds / num_runs;

    // Verify
    CUDA_CHECK(cudaMemcpy(h_B, d_B, bytes, cudaMemcpyDeviceToHost));
    if (!verify_transpose(h_A, h_B, N)) {
        printf("ERROR: Transpose incorrect!\n");
        return;
    }
    printf("Correctness: PASSED\n");

    // Performance
    double bytes_transferred = 2.0 * bytes;
    double bandwidth = (bytes_transferred / (avg_time / 1000.0)) / 1e9;

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    double peak_bandwidth = (prop.memoryClockRate * 1000.0 * (prop.memoryBusWidth / 8) * 2) / 1e9;

    printf("Time: %.3f ms\n", avg_time);
    printf("Bandwidth: %.2f GB/s\n", bandwidth);
    printf("Utilization: %.1f%%\n", (bandwidth / peak_bandwidth) * 100.0);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}

int main(int argc, char** argv) {
    int N = 4096;
    if (argc > 1) {
        N = atoi(argv[1]);
    }

    printf("=== Matrix Transpose Solution ===\n");
    printf("Matrix size: %d x %d\n", N, N);

    size_t bytes = N * N * sizeof(float);
    float *h_A = (float*)malloc(bytes);
    float *h_B = (float*)malloc(bytes);

    init_matrix(h_A, N);

    float *d_A, *d_B;
    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("GPU: %s\n", prop.name);
    double peak_bandwidth = (prop.memoryClockRate * 1000.0 * (prop.memoryBusWidth / 8) * 2) / 1e9;
    printf("Peak memory bandwidth: %.2f GB/s\n", peak_bandwidth);

    // Test all versions
    dim3 blockDim(TILE_DIM, TILE_DIM);
    dim3 gridDim((N + TILE_DIM - 1) / TILE_DIM, (N + TILE_DIM - 1) / TILE_DIM);

    benchmark_kernel("Naive Transpose", transpose_naive, d_A, d_B, h_A, h_B, N, gridDim, blockDim);
    benchmark_kernel("Shared Memory (with conflicts)", transpose_shared, d_A, d_B, h_A, h_B, N, gridDim, blockDim);
    benchmark_kernel("Optimized (padded)", transpose_optimized, d_A, d_B, h_A, h_B, N, gridDim, blockDim);

    // Rectangular version
    dim3 blockDim2(TILE_DIM, BLOCK_ROWS);
    dim3 gridDim2((N + TILE_DIM - 1) / TILE_DIM, (N + TILE_DIM - 1) / TILE_DIM);
    benchmark_kernel("Rectangular tiles", transpose_rectangular, d_A, d_B, h_A, h_B, N, gridDim2, blockDim2);

    printf("\n=== Summary ===\n");
    printf("Key insight: Adding +1 padding to shared memory eliminates bank conflicts!\n");
    printf("  __shared__ float tile[32][33];  // Instead of [32][32]\n");
    printf("\nWhy? Consecutive threads reading tile[threadIdx.x][threadIdx.y] now access\n");
    printf("different banks because stride is 33 (not divisible by 32) instead of 32.\n");
    printf("\nTo verify zero bank conflicts, run:\n");
    printf("  ncu --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum ./transpose\n");

    // Cleanup
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    free(h_A);
    free(h_B);

    return 0;
}
