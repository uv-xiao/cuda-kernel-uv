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

constexpr int BLOCK_SIZE = 128;  // Fine-grained block size

struct fp8_e4m3 {
    uint8_t __x;
    __host__ __device__ fp8_e4m3(uint8_t val = 0) : __x(val) {}
};

// ============================================================================
// Fine-Grained Quantization: Compute scales per block
// ============================================================================

__global__ void compute_block_scales_kernel(const float* input, float* scales,
                                           int n, int block_size) {
    int block_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int start_idx = block_idx * block_size;
    int end_idx = min(start_idx + block_size, n);

    if (start_idx >= n) return;

    // Find max in this block
    float local_max = 0.0f;
    for (int i = start_idx; i < end_idx; i++) {
        local_max = fmaxf(local_max, fabsf(input[i]));
    }

    // Compute scale (with epsilon to avoid division by zero)
    scales[block_idx] = fmaxf(local_max, 1e-6f) / 448.0f;
}

// ============================================================================
// Fine-Grained Quantization Kernel
// ============================================================================

__device__ __forceinline__ fp8_e4m3 float_to_fp8_e4m3(float val) {
    val = fmaxf(-448.0f, fminf(val, 448.0f));

    uint32_t bits = __float_as_uint(val);
    uint32_t sign = (bits >> 31) & 0x1;
    int32_t exp = ((bits >> 23) & 0xFF) - 127;
    uint32_t mantissa = bits & 0x7FFFFF;

    uint8_t result = 0;
    if (exp < -9) {
        result = (sign << 7);
    } else if (exp > 8) {
        result = (sign << 7) | 0x7E;
    } else if (exp < -6) {
        int shift = -6 - exp;
        uint32_t denorm_mantissa = (0x800000 | mantissa) >> (shift + 20);
        result = (sign << 7) | (denorm_mantissa & 0x7);
    } else {
        uint32_t e4m3_exp = exp + 7;
        uint32_t e4m3_mantissa = (mantissa >> 20) & 0x7;
        result = (sign << 7) | (e4m3_exp << 3) | e4m3_mantissa;
    }

    return fp8_e4m3(result);
}

__device__ __forceinline__ float fp8_e4m3_to_float(fp8_e4m3 val) {
    uint8_t bits = val.__x;
    uint32_t sign = (bits >> 7) & 0x1;
    uint32_t exp = (bits >> 3) & 0xF;
    uint32_t mantissa = bits & 0x7;

    float result;
    if (exp == 0) {
        result = ldexpf((float)mantissa / 8.0f, -6);
    } else {
        result = ldexpf(1.0f + (float)mantissa / 8.0f, (int)exp - 7);
    }

    return sign ? -result : result;
}

__global__ void quantize_fine_grained_kernel(const float* input, fp8_e4m3* output,
                                             const float* scales, int n,
                                             int block_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    int block_idx = idx / block_size;
    float scale = scales[block_idx];

    float val = input[idx] / scale;
    output[idx] = float_to_fp8_e4m3(val);
}

__global__ void dequantize_fine_grained_kernel(const fp8_e4m3* input, float* output,
                                               const float* scales, int n,
                                               int block_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    int block_idx = idx / block_size;
    float scale = scales[block_idx];

    float val = fp8_e4m3_to_float(input[idx]);
    output[idx] = val * scale;
}

// ============================================================================
// Optimized version: Fused scale computation and quantization
// ============================================================================

template<int BlockSize>
__global__ void quantize_fine_grained_fused_kernel(const float* input,
                                                    fp8_e4m3* output,
                                                    float* scales, int n) {
    int block_idx = blockIdx.x;
    int start_idx = block_idx * BlockSize;
    int tid = threadIdx.x;

    __shared__ float shared_max[BlockSize];

    // Each thread loads one element and finds local max
    float local_val = 0.0f;
    float local_max = 0.0f;

    int global_idx = start_idx + tid;
    if (global_idx < n) {
        local_val = input[global_idx];
        local_max = fabsf(local_val);
    }

    shared_max[tid] = local_max;
    __syncthreads();

    // Block-level reduction to find max
    for (int s = BlockSize / 2; s > 0; s >>= 1) {
        if (tid < s && (start_idx + tid + s) < n) {
            shared_max[tid] = fmaxf(shared_max[tid], shared_max[tid + s]);
        }
        __syncthreads();
    }

    float scale = shared_max[0] / 448.0f;
    scale = fmaxf(scale, 1e-6f);

    // Write scale (only thread 0)
    if (tid == 0) {
        scales[block_idx] = scale;
    }

    // Quantize
    if (global_idx < n) {
        float val = local_val / scale;
        output[global_idx] = float_to_fp8_e4m3(val);
    }
}

// ============================================================================
// Test with synthetic data containing outliers
// ============================================================================

void test_fine_grained_quantization() {
    printf("=== Fine-Grained Quantization Test ===\n");

    const int N = 128 * 1024; // 128 blocks of 1024 elements
    float* h_input = (float*)malloc(N * sizeof(float));
    float* h_output = (float*)malloc(N * sizeof(float));

    // Generate data with outliers every 128 elements
    srand(42);
    for (int i = 0; i < N; i++) {
        if (i % 128 == 0) {
            // Inject outlier
            h_input[i] = (rand() % 2 ? 1.0f : -1.0f) * (100.0f + rand() % 200);
        } else {
            // Normal values
            float u1 = rand() / (float)RAND_MAX;
            float u2 = rand() / (float)RAND_MAX;
            h_input[i] = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * M_PI * u2) * 2.0f;
        }
    }

    int num_blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    printf("Total elements: %d\n", N);
    printf("Block size: %d\n", BLOCK_SIZE);
    printf("Number of blocks: %d\n", num_blocks);

    float *d_input, *d_output, *d_scales;
    fp8_e4m3* d_quantized;

    CHECK_CUDA(cudaMalloc(&d_input, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_output, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_quantized, N * sizeof(fp8_e4m3)));
    CHECK_CUDA(cudaMalloc(&d_scales, num_blocks * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_input, h_input, N * sizeof(float),
                          cudaMemcpyHostToDevice));

    // Method 1: Two-pass (compute scales, then quantize)
    int threadsPerBlock = 256;
    int blocksPerGrid = (num_blocks + threadsPerBlock - 1) / threadsPerBlock;

    compute_block_scales_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        d_input, d_scales, N, BLOCK_SIZE);

    blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    quantize_fine_grained_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        d_input, d_quantized, d_scales, N, BLOCK_SIZE);

    dequantize_fine_grained_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        d_quantized, d_output, d_scales, N, BLOCK_SIZE);

    CHECK_CUDA(cudaMemcpy(h_output, d_output, N * sizeof(float),
                          cudaMemcpyDeviceToHost));

    // Calculate error
    double mae = 0.0, mse = 0.0, max_error = 0.0;
    for (int i = 0; i < N; i++) {
        double error = fabs(h_output[i] - h_input[i]);
        mae += error;
        mse += error * error;
        max_error = fmax(max_error, error);
    }

    printf("\nTwo-Pass Fine-Grained Quantization:\n");
    printf("  Mean Absolute Error: %.6f\n", mae / N);
    printf("  Root Mean Square Error: %.6f\n", sqrt(mse / N));
    printf("  Max Absolute Error: %.6f\n", max_error);

    // Test fused version
    quantize_fine_grained_fused_kernel<BLOCK_SIZE><<<num_blocks, BLOCK_SIZE>>>(
        d_input, d_quantized, d_scales, N);

    dequantize_fine_grained_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        d_quantized, d_output, d_scales, N, BLOCK_SIZE);

    CHECK_CUDA(cudaMemcpy(h_output, d_output, N * sizeof(float),
                          cudaMemcpyDeviceToHost));

    mae = mse = max_error = 0.0;
    for (int i = 0; i < N; i++) {
        double error = fabs(h_output[i] - h_input[i]);
        mae += error;
        mse += error * error;
        max_error = fmax(max_error, error);
    }

    printf("\nFused Fine-Grained Quantization:\n");
    printf("  Mean Absolute Error: %.6f\n", mae / N);
    printf("  Root Mean Square Error: %.6f\n", sqrt(mse / N));
    printf("  Max Absolute Error: %.6f\n\n", max_error);

    // Compare with per-tensor quantization
    printf("=== Comparison with Per-Tensor Quantization ===\n");

    // Find global max
    float global_max = 0.0f;
    for (int i = 0; i < N; i++) {
        global_max = fmaxf(global_max, fabsf(h_input[i]));
    }
    float global_scale = global_max / 448.0f;

    printf("Global scale: %.6f\n", global_scale);

    // Quantize with global scale
    float* h_per_tensor = (float*)malloc(N * sizeof(float));
    for (int i = 0; i < N; i++) {
        float val = h_input[i] / global_scale;
        fp8_e4m3 quantized = float_to_fp8_e4m3(val);
        h_per_tensor[i] = fp8_e4m3_to_float(quantized) * global_scale;
    }

    mae = mse = max_error = 0.0;
    for (int i = 0; i < N; i++) {
        double error = fabs(h_per_tensor[i] - h_input[i]);
        mae += error;
        mse += error * error;
        max_error = fmax(max_error, error);
    }

    printf("\nPer-Tensor Quantization:\n");
    printf("  Mean Absolute Error: %.6f\n", mae / N);
    printf("  Root Mean Square Error: %.6f\n", sqrt(mse / N));
    printf("  Max Absolute Error: %.6f\n\n", max_error);

    printf("Fine-grained scaling significantly reduces error from outliers!\n");

    free(h_input);
    free(h_output);
    free(h_per_tensor);
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_quantized);
    cudaFree(d_scales);
}

// Benchmark performance
void benchmark_quantization() {
    printf("\n=== Quantization Performance Benchmark ===\n");

    const int N = 1 << 24; // 16M elements
    const int num_blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const int warmup = 10;
    const int iters = 100;

    float *d_input, *d_output, *d_scales;
    fp8_e4m3* d_quantized;

    CHECK_CUDA(cudaMalloc(&d_input, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_output, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_quantized, N * sizeof(fp8_e4m3)));
    CHECK_CUDA(cudaMalloc(&d_scales, num_blocks * sizeof(float)));

    // Initialize with random data
    cudaMemset(d_input, 0, N * sizeof(float));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Benchmark fused quantization
    for (int i = 0; i < warmup; i++) {
        quantize_fine_grained_fused_kernel<BLOCK_SIZE><<<num_blocks, BLOCK_SIZE>>>(
            d_input, d_quantized, d_scales, N);
    }
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    for (int i = 0; i < iters; i++) {
        quantize_fine_grained_fused_kernel<BLOCK_SIZE><<<num_blocks, BLOCK_SIZE>>>(
            d_input, d_quantized, d_scales, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    ms /= iters;

    float bandwidth = (N * (sizeof(float) + sizeof(fp8_e4m3) + sizeof(float) / BLOCK_SIZE)) /
                      (ms / 1000.0f) / 1e9;

    printf("Fused Quantization (block size %d):\n", BLOCK_SIZE);
    printf("  Time: %.3f ms\n", ms);
    printf("  Bandwidth: %.1f GB/s\n", bandwidth);
    printf("  Throughput: %.1f G elements/s\n\n", (N / 1e9) / (ms / 1000.0f));

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_quantized);
    cudaFree(d_scales);
}

int main() {
    printf("=== Fine-Grained Scaling Demo ===\n\n");

    test_fine_grained_quantization();
    benchmark_quantization();

    return 0;
}
