#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define CHECK_CUDA_ERROR(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Warp-level inclusive scan (provided)
__device__ int warp_scan_inclusive(int value) {
    for (int offset = 1; offset < warpSize; offset *= 2) {
        int neighbor = __shfl_up_sync(0xffffffff, value, offset);
        if ((threadIdx.x % warpSize) >= offset) {
            value += neighbor;
        }
    }
    return value;
}

// TODO: Implement block-level inclusive scan
// Use warp_scan_inclusive and shared memory for coordination
__global__ void block_scan_inclusive(int *output, const int *input, int n) {
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // TODO: Load value from global memory
    int value = 0;

    // TODO: Perform warp-level scan
    // Hint: Use warp_scan_inclusive(value)

    // TODO: Share warp totals in shared memory
    // Hint: Create __shared__ array for warp totals
    // Hint: Last thread in each warp writes the total

    // TODO: Scan the warp totals (first warp only)
    // Hint: First warp scans the warp totals array

    // TODO: Add warp offset to get final result
    // Hint: Each thread adds the sum of all previous warps

    // TODO: Write result to global memory
    if (i < n) {
        output[i] = value;
    }
}

// CPU reference
void cpu_scan_inclusive(int *output, const int *input, int n) {
    output[0] = input[0];
    for (int i = 1; i < n; i++) {
        output[i] = output[i - 1] + input[i];
    }
}

bool verify(const int *gpu, const int *cpu, int n) {
    for (int i = 0; i < n; i++) {
        if (gpu[i] != cpu[i]) {
            printf("Mismatch at %d: GPU=%d, CPU=%d\n", i, gpu[i], cpu[i]);
            return false;
        }
    }
    return true;
}

int main() {
    const int n = 256;
    const int bytes = n * sizeof(int);

    int *h_input = (int*)malloc(bytes);
    int *h_output_gpu = (int*)malloc(bytes);
    int *h_output_cpu = (int*)malloc(bytes);

    // Test 1: All ones
    printf("Test 1: All ones\n");
    for (int i = 0; i < n; i++) h_input[i] = 1;

    int *d_input, *d_output;
    CHECK_CUDA_ERROR(cudaMalloc(&d_input, bytes));
    CHECK_CUDA_ERROR(cudaMalloc(&d_output, bytes));
    CHECK_CUDA_ERROR(cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice));

    cpu_scan_inclusive(h_output_cpu, h_input, n);
    block_scan_inclusive<<<1, 256>>>(d_output, d_input, n);
    CHECK_CUDA_ERROR(cudaMemcpy(h_output_gpu, d_output, bytes, cudaMemcpyDeviceToHost));

    printf("  Expected: [1, 2, 3, ..., %d]\n", n);
    printf("  Got: [%d, %d, %d, ..., %d]\n",
           h_output_gpu[0], h_output_gpu[1], h_output_gpu[2], h_output_gpu[n-1]);
    printf("  Result: %s\n\n", verify(h_output_gpu, h_output_cpu, n) ? "PASS" : "FAIL");

    // Test 2: Sequential values
    printf("Test 2: Sequential [1, 2, 3, ...]\n");
    for (int i = 0; i < n; i++) h_input[i] = i + 1;

    CHECK_CUDA_ERROR(cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice));
    cpu_scan_inclusive(h_output_cpu, h_input, n);
    block_scan_inclusive<<<1, 256>>>(d_output, d_input, n);
    CHECK_CUDA_ERROR(cudaMemcpy(h_output_gpu, d_output, bytes, cudaMemcpyDeviceToHost));

    int expected_last = n * (n + 1) / 2;
    printf("  Last element expected: %d\n", expected_last);
    printf("  Last element got: %d\n", h_output_gpu[n-1]);
    printf("  Result: %s\n\n", verify(h_output_gpu, h_output_cpu, n) ? "PASS" : "FAIL");

    // Test 3: Mixed positive/negative
    printf("Test 3: Alternating +1/-1\n");
    for (int i = 0; i < n; i++) h_input[i] = (i % 2 == 0) ? 1 : -1;

    CHECK_CUDA_ERROR(cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice));
    cpu_scan_inclusive(h_output_cpu, h_input, n);
    block_scan_inclusive<<<1, 256>>>(d_output, d_input, n);
    CHECK_CUDA_ERROR(cudaMemcpy(h_output_gpu, d_output, bytes, cudaMemcpyDeviceToHost));

    printf("  Result: %s\n", verify(h_output_gpu, h_output_cpu, n) ? "PASS" : "FAIL");

    free(h_input);
    free(h_output_gpu);
    free(h_output_cpu);
    CHECK_CUDA_ERROR(cudaFree(d_input));
    CHECK_CUDA_ERROR(cudaFree(d_output));

    return 0;
}
