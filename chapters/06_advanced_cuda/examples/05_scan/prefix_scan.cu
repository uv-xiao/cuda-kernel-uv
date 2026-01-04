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

// Warp-level inclusive scan using shuffle
__device__ int warp_scan_inclusive(int value) {
    for (int offset = 1; offset < warpSize; offset *= 2) {
        int neighbor = __shfl_up_sync(0xffffffff, value, offset);
        if ((threadIdx.x % warpSize) >= offset) {
            value += neighbor;
        }
    }
    return value;
}

// Warp-level exclusive scan using shuffle
__device__ int warp_scan_exclusive(int value) {
    int inclusive = warp_scan_inclusive(value);
    int exclusive = __shfl_up_sync(0xffffffff, inclusive, 1);
    return ((threadIdx.x % warpSize) == 0) ? 0 : exclusive;
}

// Block-level inclusive scan (Blelloch algorithm)
__global__ void block_scan_inclusive(int *output, const int *input, int n) {
    extern __shared__ int temp[];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Load input
    temp[tid] = (i < n) ? input[i] : 0;
    __syncthreads();

    // Up-sweep (reduce) phase
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        int index = (tid + 1) * stride * 2 - 1;
        if (index < blockDim.x) {
            temp[index] += temp[index - stride];
        }
        __syncthreads();
    }

    // Down-sweep phase
    for (int stride = blockDim.x / 4; stride > 0; stride /= 2) {
        int index = (tid + 1) * stride * 2 - 1;
        if (index + stride < blockDim.x) {
            temp[index + stride] += temp[index];
        }
        __syncthreads();
    }

    // Write output
    if (i < n) {
        output[i] = temp[tid];
    }
}

// Block-level exclusive scan
__global__ void block_scan_exclusive(int *output, const int *input, int n) {
    extern __shared__ int temp[];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Load input
    temp[tid] = (i < n) ? input[i] : 0;
    __syncthreads();

    // Up-sweep phase
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        int index = (tid + 1) * stride * 2 - 1;
        if (index < blockDim.x) {
            temp[index] += temp[index - stride];
        }
        __syncthreads();
    }

    // Store total and clear last element
    if (tid == 0) {
        temp[blockDim.x] = temp[blockDim.x - 1];
        temp[blockDim.x - 1] = 0;
    }
    __syncthreads();

    // Down-sweep phase
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        int index = (tid + 1) * stride * 2 - 1;
        if (index < blockDim.x) {
            int temp_val = temp[index];
            temp[index] += temp[index + stride];
            temp[index + stride] = temp_val;
        }
        __syncthreads();
    }

    // Write output
    if (i < n) {
        output[i] = temp[tid];
    }
}

// Simplified block scan using warp primitives
__global__ void block_scan_warp_based(int *output, const int *input, int n) {
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    int value = (i < n) ? input[i] : 0;

    // Warp-level scan
    int warp_scan = warp_scan_inclusive(value);

    // Share warp totals
    __shared__ int warp_totals[32];
    int lane_id = tid % warpSize;
    int warp_id = tid / warpSize;

    if (lane_id == warpSize - 1) {
        warp_totals[warp_id] = warp_scan;
    }
    __syncthreads();

    // First warp scans the totals
    if (warp_id == 0) {
        int warp_count = (blockDim.x + warpSize - 1) / warpSize;
        int total = (tid < warp_count) ? warp_totals[tid] : 0;
        total = warp_scan_inclusive(total);
        warp_totals[tid] = total;
    }
    __syncthreads();

    // Add warp offset
    int warp_offset = (warp_id > 0) ? warp_totals[warp_id - 1] : 0;
    int result = warp_scan + warp_offset;

    if (i < n) {
        output[i] = result;
    }
}

// CPU reference implementations
void cpu_scan_inclusive(int *output, const int *input, int n) {
    output[0] = input[0];
    for (int i = 1; i < n; i++) {
        output[i] = output[i - 1] + input[i];
    }
}

void cpu_scan_exclusive(int *output, const int *input, int n) {
    output[0] = 0;
    for (int i = 1; i < n; i++) {
        output[i] = output[i - 1] + input[i - 1];
    }
}

bool verify_result(const int *gpu, const int *cpu, int n) {
    for (int i = 0; i < n; i++) {
        if (gpu[i] != cpu[i]) {
            printf("Mismatch at index %d: GPU=%d, CPU=%d\n", i, gpu[i], cpu[i]);
            return false;
        }
    }
    return true;
}

void print_array(const char *name, const int *arr, int n) {
    printf("%s: [", name);
    int print_n = (n > 16) ? 16 : n;
    for (int i = 0; i < print_n; i++) {
        printf("%d", arr[i]);
        if (i < print_n - 1) printf(", ");
    }
    if (n > 16) printf(", ...");
    printf("]\n");
}

int main() {
    printf("========== Parallel Prefix Scan (Prefix Sum) ==========\n\n");

    const int n = 256;
    const int bytes = n * sizeof(int);

    int *h_input = (int*)malloc(bytes);
    int *h_output_gpu = (int*)malloc(bytes);
    int *h_output_cpu = (int*)malloc(bytes);

    // Initialize with ones for easy verification
    for (int i = 0; i < n; i++) {
        h_input[i] = 1;
    }

    int *d_input, *d_output;
    CHECK_CUDA_ERROR(cudaMalloc(&d_input, bytes));
    CHECK_CUDA_ERROR(cudaMalloc(&d_output, bytes));
    CHECK_CUDA_ERROR(cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice));

    printf("Input size: %d elements (all ones)\n\n", n);

    // Test 1: Inclusive scan
    printf("1. Inclusive Scan:\n");
    cpu_scan_inclusive(h_output_cpu, h_input, n);

    block_scan_inclusive<<<1, n, (n + 1) * sizeof(int)>>>(d_output, d_input, n);
    CHECK_CUDA_ERROR(cudaMemcpy(h_output_gpu, d_output, bytes, cudaMemcpyDeviceToHost));

    print_array("   CPU", h_output_cpu, n);
    print_array("   GPU", h_output_gpu, n);
    printf("   Verification: %s\n", verify_result(h_output_gpu, h_output_cpu, n) ? "PASS" : "FAIL");

    // Test 2: Exclusive scan
    printf("\n2. Exclusive Scan:\n");
    cpu_scan_exclusive(h_output_cpu, h_input, n);

    block_scan_exclusive<<<1, n, (n + 1) * sizeof(int)>>>(d_output, d_input, n);
    CHECK_CUDA_ERROR(cudaMemcpy(h_output_gpu, d_output, bytes, cudaMemcpyDeviceToHost));

    print_array("   CPU", h_output_cpu, n);
    print_array("   GPU", h_output_gpu, n);
    printf("   Verification: %s\n", verify_result(h_output_gpu, h_output_cpu, n) ? "PASS" : "FAIL");

    // Test 3: Warp-based scan
    printf("\n3. Warp-Based Scan (Inclusive):\n");
    cpu_scan_inclusive(h_output_cpu, h_input, n);

    block_scan_warp_based<<<1, n>>>(d_output, d_input, n);
    CHECK_CUDA_ERROR(cudaMemcpy(h_output_gpu, d_output, bytes, cudaMemcpyDeviceToHost));

    print_array("   CPU", h_output_cpu, n);
    print_array("   GPU", h_output_gpu, n);
    printf("   Verification: %s\n", verify_result(h_output_gpu, h_output_cpu, n) ? "PASS" : "FAIL");

    // Test 4: Different input pattern
    printf("\n4. Scan with Sequential Input [1,2,3,...]:\n");
    for (int i = 0; i < n; i++) {
        h_input[i] = i + 1;
    }
    CHECK_CUDA_ERROR(cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice));

    cpu_scan_inclusive(h_output_cpu, h_input, n);
    block_scan_warp_based<<<1, n>>>(d_output, d_input, n);
    CHECK_CUDA_ERROR(cudaMemcpy(h_output_gpu, d_output, bytes, cudaMemcpyDeviceToHost));

    print_array("   Input", h_input, n);
    print_array("   Output", h_output_gpu, n);
    printf("   Last element: %d (sum of 1 to %d)\n", h_output_gpu[n-1], n);
    printf("   Expected: %d\n", n * (n + 1) / 2);
    printf("   Verification: %s\n", verify_result(h_output_gpu, h_output_cpu, n) ? "PASS" : "FAIL");

    // Cleanup
    free(h_input);
    free(h_output_gpu);
    free(h_output_cpu);
    CHECK_CUDA_ERROR(cudaFree(d_input));
    CHECK_CUDA_ERROR(cudaFree(d_output));

    printf("\n========== All Tests Completed Successfully ==========\n");
    printf("\nApplications of Scan:\n");
    printf("- Stream compaction (filtering)\n");
    printf("- Radix sort\n");
    printf("- Quicksort partitioning\n");
    printf("- Sparse matrix operations\n");
    printf("- Building data structures (trees, tries)\n");

    return 0;
}
