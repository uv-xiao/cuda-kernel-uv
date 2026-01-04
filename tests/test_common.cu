/**
 * Test file for common utilities
 */

#include "check.cuh"
#include "cuda_utils.cuh"
#include "timer.cuh"
#include <stdio.h>

// Simple test kernel
__global__ void test_kernel(float *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = data[idx] * 2.0f;
    }
}

int main() {
    printf("Testing common utilities...\n\n");

    // Test device info
    printf("=== Device Info ===\n");
    print_device_info();

    // Test timer
    printf("=== Timer Test ===\n");
    const int N = 1024 * 1024;
    float *d_data;
    CUDA_CHECK(cudaMalloc(&d_data, N * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_data, 0, N * sizeof(float)));

    GpuTimer timer;
    timer.start();
    test_kernel<<<(N + 255) / 256, 256>>>(d_data, N);
    timer.stop();
    printf("Kernel time: %.3f ms\n\n", timer.elapsed_ms());

    // Test verification utilities
    printf("=== Verification Test ===\n");
    float h_expected[10] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
                            6.0f, 7.0f, 8.0f, 9.0f, 10.0f};
    float h_actual[10] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
                          6.0f, 7.0f, 8.0f, 9.0f, 10.0f};

    if (verify_result(h_expected, h_actual, 10)) {
        printf("Verification utility works correctly!\n\n");
    }

    // Test print_matrix
    printf("=== Print Matrix Test ===\n");
    print_matrix(h_expected, 2, 5, "Test Matrix");

    // Cleanup
    CUDA_CHECK(cudaFree(d_data));

    printf("All tests passed!\n");
    return 0;
}
