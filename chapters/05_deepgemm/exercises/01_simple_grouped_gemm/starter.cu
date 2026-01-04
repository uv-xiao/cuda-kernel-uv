#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            printf("CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                   cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// ============================================================================
// TODO: Implement this kernel
// ============================================================================

template<int TILE_SIZE>
__global__ void grouped_gemm_kernel(
    const float* __restrict__ A_concat,
    const float* __restrict__ B_concat,
    float* __restrict__ C_concat,
    const int* __restrict__ group_offsets,
    const int* __restrict__ M_sizes,
    int K, int N, int num_groups) {

    // TODO: Your implementation here
    //
    // High-level structure:
    // 1. Determine which group and tile this block processes
    // 2. Load input tiles into shared memory
    // 3. Compute partial products
    // 4. Write results to global memory
    //
    // Tips:
    // - Use shared memory for tiling
    // - Handle boundary conditions (M < TILE_SIZE)
    // - Minimize warp divergence
}

// ============================================================================
// Host function to launch grouped GEMM
// ============================================================================

void launch_grouped_gemm(
    const float* A_concat,
    const float* B_concat,
    float* C_concat,
    const int* group_offsets,
    const int* M_sizes,
    int K, int N, int num_groups) {

    const int TILE_SIZE = 16;

    // TODO: Calculate number of tiles needed
    int total_tiles = 0;
    // Hint: Sum up tiles across all groups

    // TODO: Launch kernel with appropriate grid/block dimensions
    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim(/* TODO: fill in */);

    grouped_gemm_kernel<TILE_SIZE><<<gridDim, blockDim>>>(
        A_concat, B_concat, C_concat, group_offsets, M_sizes, K, N, num_groups);

    CHECK_CUDA(cudaGetLastError());
}

// ============================================================================
// Reference CPU implementation for validation
// ============================================================================

void grouped_gemm_cpu(
    const float* A_concat,
    const float* B_concat,
    float* C_concat,
    const int* group_offsets,
    const int* M_sizes,
    int K, int N, int num_groups) {

    for (int g = 0; g < num_groups; g++) {
        int M = M_sizes[g];
        int offset = group_offsets[g];

        const float* A = A_concat + offset * K;
        const float* B = B_concat + g * K * N;
        float* C = C_concat + offset * N;

        // Simple matrix multiplication
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                float sum = 0.0f;
                for (int k = 0; k < K; k++) {
                    sum += A[i * K + k] * B[k * N + j];
                }
                C[i * N + j] = sum;
            }
        }
    }
}

// ============================================================================
// Test harness
// ============================================================================

void test_grouped_gemm(const int* M_sizes, int num_groups, int K, int N) {
    printf("\n=== Test: num_groups=%d, K=%d, N=%d ===\n", num_groups, K, N);
    printf("M sizes: [");
    for (int i = 0; i < num_groups; i++) {
        printf("%d", M_sizes[i]);
        if (i < num_groups - 1) printf(", ");
    }
    printf("]\n");

    // Compute offsets and total size
    int* group_offsets = (int*)malloc((num_groups + 1) * sizeof(int));
    group_offsets[0] = 0;
    for (int i = 0; i < num_groups; i++) {
        group_offsets[i + 1] = group_offsets[i] + M_sizes[i];
    }
    int total_M = group_offsets[num_groups];

    printf("Total M: %d\n", total_M);

    // Allocate host memory
    float* h_A = (float*)malloc(total_M * K * sizeof(float));
    float* h_B = (float*)malloc(num_groups * K * N * sizeof(float));
    float* h_C_gpu = (float*)malloc(total_M * N * sizeof(float));
    float* h_C_cpu = (float*)malloc(total_M * N * sizeof(float));

    // Initialize with random data
    srand(42);
    for (int i = 0; i < total_M * K; i++) {
        h_A[i] = (rand() / (float)RAND_MAX) * 2.0f - 1.0f;
    }
    for (int i = 0; i < num_groups * K * N; i++) {
        h_B[i] = (rand() / (float)RAND_MAX) * 2.0f - 1.0f;
    }

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    int *d_group_offsets, *d_M_sizes;

    CHECK_CUDA(cudaMalloc(&d_A, total_M * K * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_B, num_groups * K * N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_C, total_M * N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_group_offsets, (num_groups + 1) * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_M_sizes, num_groups * sizeof(int)));

    CHECK_CUDA(cudaMemcpy(d_A, h_A, total_M * K * sizeof(float),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, num_groups * K * N * sizeof(float),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_group_offsets, group_offsets,
                          (num_groups + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_M_sizes, M_sizes, num_groups * sizeof(int),
                          cudaMemcpyHostToDevice));

    // Run GPU kernel
    launch_grouped_gemm(d_A, d_B, d_C, d_group_offsets, d_M_sizes, K, N, num_groups);
    CHECK_CUDA(cudaDeviceSynchronize());

    // Copy result back
    CHECK_CUDA(cudaMemcpy(h_C_gpu, d_C, total_M * N * sizeof(float),
                          cudaMemcpyDeviceToHost));

    // Run CPU reference
    grouped_gemm_cpu(h_A, h_B, h_C_cpu, group_offsets, M_sizes, K, N, num_groups);

    // Verify correctness
    float max_error = 0.0f;
    for (int i = 0; i < total_M * N; i++) {
        float error = fabsf(h_C_gpu[i] - h_C_cpu[i]);
        max_error = fmaxf(max_error, error);
    }

    printf("Max error: %.6e\n", max_error);
    if (max_error < 1e-3f) {
        printf("PASSED\n");
    } else {
        printf("FAILED\n");
    }

    // Cleanup
    free(h_A);
    free(h_B);
    free(h_C_gpu);
    free(h_C_cpu);
    free(group_offsets);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_group_offsets);
    cudaFree(d_M_sizes);
}

int main() {
    printf("=== Grouped GEMM Exercise ===\n");

    // Test 1: Uniform sizes
    {
        int M_sizes[] = {128, 128, 128, 128};
        test_grouped_gemm(M_sizes, 4, 256, 256);
    }

    // Test 2: Variable sizes
    {
        int M_sizes[] = {64, 128, 32, 256, 16, 192, 96, 48};
        test_grouped_gemm(M_sizes, 8, 512, 512);
    }

    // Test 3: Extreme imbalance
    {
        int M_sizes[] = {1, 10, 100, 1000};
        test_grouped_gemm(M_sizes, 4, 1024, 1024);
    }

    // Test 4: Small groups
    {
        int M_sizes[] = {8, 12, 16, 24, 32, 8, 16, 12, 20, 28, 16, 8, 24, 32, 16, 12};
        test_grouped_gemm(M_sizes, 16, 128, 128);
    }

    return 0;
}
