#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>

#define CHECK_CUDA_ERROR(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// TODO: Implement warp-level max reduction using __shfl_xor_sync
// This function should find the maximum value across all 32 threads in a warp
// The result should be available in ALL threads (not just lane 0)
__device__ int warp_reduce_max(int value) {
    // TODO: Implement using butterfly reduction with __shfl_xor_sync
    // Hint: Loop with offsets 16, 8, 4, 2, 1
    // Hint: Use max(value, shuffled_value) in each iteration

    return value; // Replace with your implementation
}

// Test kernel
__global__ void test_warp_reduce_max(int *output, const int *input) {
    int tid = threadIdx.x;
    int value = input[tid];

    int max_val = warp_reduce_max(value);

    output[tid] = max_val;
}

int main() {
    const int n = 32; // One warp
    const int bytes = n * sizeof(int);

    // Test case 1: Sequential values
    printf("Test 1: Sequential values [0, 1, 2, ..., 31]\n");
    int *h_input = (int*)malloc(bytes);
    int *h_output = (int*)malloc(bytes);

    for (int i = 0; i < n; i++) {
        h_input[i] = i;
    }

    int *d_input, *d_output;
    CHECK_CUDA_ERROR(cudaMalloc(&d_input, bytes));
    CHECK_CUDA_ERROR(cudaMalloc(&d_output, bytes));
    CHECK_CUDA_ERROR(cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice));

    test_warp_reduce_max<<<1, 32>>>(d_output, d_input);
    CHECK_CUDA_ERROR(cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost));

    int expected = 31;
    bool pass = true;
    for (int i = 0; i < n; i++) {
        if (h_output[i] != expected) {
            pass = false;
            break;
        }
    }
    printf("  Expected: %d, Got: %d, Result: %s\n\n", expected, h_output[0], pass ? "PASS" : "FAIL");

    // Test case 2: All same values
    printf("Test 2: All same values (42)\n");
    for (int i = 0; i < n; i++) {
        h_input[i] = 42;
    }
    CHECK_CUDA_ERROR(cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice));
    test_warp_reduce_max<<<1, 32>>>(d_output, d_input);
    CHECK_CUDA_ERROR(cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost));

    expected = 42;
    pass = true;
    for (int i = 0; i < n; i++) {
        if (h_output[i] != expected) {
            pass = false;
            break;
        }
    }
    printf("  Expected: %d, Got: %d, Result: %s\n\n", expected, h_output[0], pass ? "PASS" : "FAIL");

    // Test case 3: Random values
    printf("Test 3: Random values\n");
    int max_input = INT_MIN;
    for (int i = 0; i < n; i++) {
        h_input[i] = rand() % 1000 - 500; // Random values from -500 to 499
        if (h_input[i] > max_input) max_input = h_input[i];
    }
    CHECK_CUDA_ERROR(cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice));
    test_warp_reduce_max<<<1, 32>>>(d_output, d_input);
    CHECK_CUDA_ERROR(cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost));

    pass = true;
    for (int i = 0; i < n; i++) {
        if (h_output[i] != max_input) {
            pass = false;
            break;
        }
    }
    printf("  Expected: %d, Got: %d, Result: %s\n\n", max_input, h_output[0], pass ? "PASS" : "FAIL");

    // Test case 4: Negative values
    printf("Test 4: All negative values\n");
    max_input = INT_MIN;
    for (int i = 0; i < n; i++) {
        h_input[i] = -1000 + i; // -1000, -999, ..., -969
        if (h_input[i] > max_input) max_input = h_input[i];
    }
    CHECK_CUDA_ERROR(cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice));
    test_warp_reduce_max<<<1, 32>>>(d_output, d_input);
    CHECK_CUDA_ERROR(cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost));

    pass = true;
    for (int i = 0; i < n; i++) {
        if (h_output[i] != max_input) {
            pass = false;
            break;
        }
    }
    printf("  Expected: %d, Got: %d, Result: %s\n", max_input, h_output[0], pass ? "PASS" : "FAIL");

    // Cleanup
    free(h_input);
    free(h_output);
    CHECK_CUDA_ERROR(cudaFree(d_input));
    CHECK_CUDA_ERROR(cudaFree(d_output));

    return 0;
}
