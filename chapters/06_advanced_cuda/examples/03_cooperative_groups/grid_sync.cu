#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <stdio.h>
#include <stdlib.h>

namespace cg = cooperative_groups;

// Error checking macro
#define CHECK_CUDA_ERROR(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Example 1: Grid-wide synchronization for iterative algorithms
__global__ void jacobi_iteration(float *output, const float *input,
                                 int width, int height, int iterations) {
    // Get grid group - represents all threads in the kernel
    cg::grid_group grid = cg::this_grid();

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;
    if (x == 0 || x == width - 1 || y == 0 || y == height - 1) return; // Skip boundaries

    int idx = y * width + x;

    // Perform multiple iterations with grid-wide synchronization
    for (int iter = 0; iter < iterations; iter++) {
        // Compute new value from neighbors (5-point stencil)
        float center = input[idx];
        float left = input[idx - 1];
        float right = input[idx + 1];
        float up = input[idx - width];
        float down = input[idx + width];

        output[idx] = 0.2f * (center + left + right + up + down);

        // Synchronize all threads across entire grid
        grid.sync();

        // Swap buffers (conceptually - here we just prepare for next iteration)
        // In a real implementation, you'd swap pointers
        if (iter < iterations - 1) {
            output[idx] = input[idx]; // Copy for next iteration
            grid.sync();
        }
    }
}

// Example 2: Histogram with grid-wide coordination
__global__ void histogram_grid_sync(int *histogram, const int *data,
                                    int n, int num_bins) {
    cg::grid_group grid = cg::this_grid();

    // Each block processes a portion of the data
    extern __shared__ int local_hist[];

    int tid = threadIdx.x;
    int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    int grid_size = gridDim.x * blockDim.x;

    // Initialize local histogram
    for (int i = tid; i < num_bins; i += blockDim.x) {
        local_hist[i] = 0;
    }
    __syncthreads();

    // Build local histogram
    for (int i = global_tid; i < n; i += grid_size) {
        int bin = data[i] % num_bins;
        atomicAdd(&local_hist[bin], 1);
    }
    __syncthreads();

    // Merge local histograms into global histogram
    for (int i = tid; i < num_bins; i += blockDim.x) {
        atomicAdd(&histogram[i], local_hist[i]);
    }

    // Grid-wide sync ensures all blocks have finished
    grid.sync();

    // Now all threads can safely read the complete histogram
    // For example, normalize the histogram
    if (global_tid == 0) {
        // Could compute statistics, normalize, etc.
        printf("Histogram complete. Total bins: %d\n", num_bins);
    }
}

// Example 3: Multi-phase algorithm with grid synchronization
__global__ void multi_phase_algorithm(int *data, int n, int *phase_complete) {
    cg::grid_group grid = cg::this_grid();

    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= n) return;

    // Phase 1: Increment all elements
    data[tid] += 1;
    grid.sync();

    // Phase 2: Multiply by sum of neighbors (needs phase 1 complete)
    int left = (tid > 0) ? data[tid - 1] : 0;
    int right = (tid < n - 1) ? data[tid + 1] : 0;
    int factor = left + right;
    data[tid] *= factor;
    grid.sync();

    // Phase 3: Subtract global mean (needs phase 2 complete)
    // First, compute sum
    __shared__ int block_sum;
    if (threadIdx.x == 0) {
        block_sum = 0;
    }
    __syncthreads();

    atomicAdd(&block_sum, data[tid]);
    __syncthreads();

    if (threadIdx.x == 0) {
        atomicAdd(phase_complete, block_sum);
    }
    grid.sync();

    // All threads can now read the global sum
    int global_sum = *phase_complete;
    int mean = global_sum / n;
    data[tid] -= mean;
}

// Example 4: Global reduction with grid synchronization
__global__ void global_reduction(int *result, const int *input, int n) {
    cg::grid_group grid = cg::this_grid();
    cg::thread_block block = cg::this_thread_block();

    extern __shared__ int sdata[];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int grid_size = gridDim.x * blockDim.x;

    // Grid-stride loop for input elements
    int sum = 0;
    for (int idx = i; idx < n; idx += grid_size) {
        sum += input[idx];
    }

    // Block-level reduction
    sdata[tid] = sum;
    block.sync();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        block.sync();
    }

    // First thread in each block contributes to global result
    if (tid == 0) {
        atomicAdd(result, sdata[0]);
    }

    // Ensure all blocks have contributed
    grid.sync();

    // Now all threads can read the final result
    if (grid.thread_rank() == 0) {
        printf("Global sum: %d\n", *result);
    }
}

// Example 5: Convergence check across entire grid
__global__ void iterative_solver(float *x, const float *b, int n,
                                int max_iterations, float tolerance,
                                int *converged) {
    cg::grid_group grid = cg::this_grid();

    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= n) return;

    __shared__ int block_converged;

    for (int iter = 0; iter < max_iterations; iter++) {
        // Save old value
        float old_x = x[tid];

        // Compute new value (simple example: x = 0.5 * (x + b))
        float new_x = 0.5f * (old_x + b[tid]);
        x[tid] = new_x;

        // Check local convergence
        float diff = fabsf(new_x - old_x);
        int local_converged = (diff < tolerance) ? 1 : 0;

        // Block-level all reduce
        if (threadIdx.x == 0) {
            block_converged = 1;
        }
        __syncthreads();

        if (!local_converged) {
            block_converged = 0;
        }
        __syncthreads();

        // Accumulate to global convergence flag
        if (threadIdx.x == 0) {
            if (!block_converged) {
                atomicAnd(converged, 0);
            }
        }

        grid.sync();

        // Check if entire grid has converged
        if (*converged == 1) {
            if (grid.thread_rank() == 0) {
                printf("Converged after %d iterations\n", iter + 1);
            }
            break;
        }

        // Reset convergence flag for next iteration
        if (grid.thread_rank() == 0) {
            *converged = 1;
        }
        grid.sync();
    }
}

void test_jacobi_iteration() {
    printf("\n1. Jacobi Iteration with Grid Sync:\n");

    const int width = 32;
    const int height = 32;
    const int size = width * height;
    const int iterations = 5;

    float *h_input = (float*)malloc(size * sizeof(float));
    float *h_output = (float*)malloc(size * sizeof(float));

    // Initialize with some pattern
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            h_input[y * width + x] = (x == width / 2 && y == height / 2) ? 100.0f : 0.0f;
        }
    }

    float *d_input, *d_output;
    CHECK_CUDA_ERROR(cudaMalloc(&d_input, size * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_output, size * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMemcpy(d_input, h_input, size * sizeof(float), cudaMemcpyHostToDevice));

    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    // Launch with cooperative groups support
    void *args[] = {&d_output, &d_input, &width, &height, &iterations};

    // Need to use cooperative launch API
    CHECK_CUDA_ERROR(cudaLaunchCooperativeKernel(
        (void*)jacobi_iteration, grid, block, args, 0, 0));

    CHECK_CUDA_ERROR(cudaMemcpy(h_output, d_output, size * sizeof(float), cudaMemcpyDeviceToHost));

    printf("   Center value after %d iterations: %.2f\n", iterations,
           h_output[height / 2 * width + width / 2]);
    printf("   Grid sync allowed %d iterations in single kernel launch\n", iterations);

    free(h_input);
    free(h_output);
    CHECK_CUDA_ERROR(cudaFree(d_input));
    CHECK_CUDA_ERROR(cudaFree(d_output));
}

void test_histogram() {
    printf("\n2. Histogram with Grid Sync:\n");

    const int n = 10000;
    const int num_bins = 256;

    int *h_data = (int*)malloc(n * sizeof(int));
    for (int i = 0; i < n; i++) {
        h_data[i] = rand() % num_bins;
    }

    int *d_data, *d_histogram;
    CHECK_CUDA_ERROR(cudaMalloc(&d_data, n * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_histogram, num_bins * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMemset(d_histogram, 0, num_bins * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMemcpy(d_data, h_data, n * sizeof(int), cudaMemcpyHostToDevice));

    int block_size = 256;
    int num_blocks = 32;
    size_t shared_mem = num_bins * sizeof(int);

    void *args[] = {&d_histogram, &d_data, &n, &num_bins};

    CHECK_CUDA_ERROR(cudaLaunchCooperativeKernel(
        (void*)histogram_grid_sync,
        dim3(num_blocks), dim3(block_size),
        args, shared_mem, 0));

    int *h_histogram = (int*)malloc(num_bins * sizeof(int));
    CHECK_CUDA_ERROR(cudaMemcpy(h_histogram, d_histogram, num_bins * sizeof(int),
                                 cudaMemcpyDeviceToHost));

    int total = 0;
    for (int i = 0; i < num_bins; i++) {
        total += h_histogram[i];
    }

    printf("   Total histogram count: %d (expected: %d) - %s\n",
           total, n, total == n ? "PASS" : "FAIL");

    free(h_data);
    free(h_histogram);
    CHECK_CUDA_ERROR(cudaFree(d_data));
    CHECK_CUDA_ERROR(cudaFree(d_histogram));
}

int main() {
    printf("========== Grid-Wide Synchronization Examples ==========\n");

    // Check if device supports cooperative launch
    int device;
    CHECK_CUDA_ERROR(cudaGetDevice(&device));

    cudaDeviceProp props;
    CHECK_CUDA_ERROR(cudaGetDeviceProperties(&props, device));

    if (!props.cooperativeLaunch) {
        printf("ERROR: Device does not support cooperative launch\n");
        printf("Cooperative groups grid sync requires compute capability 6.0+\n");
        return 1;
    }

    printf("Device: %s\n", props.name);
    printf("Compute Capability: %d.%d\n", props.major, props.minor);
    printf("Cooperative Launch: Supported\n\n");

    test_jacobi_iteration();
    test_histogram();

    printf("\n========== Tests Completed Successfully ==========\n");
    printf("\nNote: Grid synchronization allows multiple phases in a single\n");
    printf("kernel launch, reducing CPU-GPU synchronization overhead.\n");
    printf("Use cudaLaunchCooperativeKernel() for grid-wide sync.\n");

    return 0;
}
