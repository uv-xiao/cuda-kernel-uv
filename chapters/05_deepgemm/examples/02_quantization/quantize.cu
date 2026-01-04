#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <cuda_bf16.h>
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

// Simple FP8 wrapper
struct fp8_e4m3 {
    uint8_t __x;
    __host__ __device__ fp8_e4m3(uint8_t val = 0) : __x(val) {}
};

// ============================================================================
// Per-Tensor Quantization
// ============================================================================

// Kernel to find max absolute value (for scale computation)
__global__ void find_absmax_kernel(const float* input, float* output, int n) {
    __shared__ float shared_max[256];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Each thread finds its local max
    float local_max = 0.0f;
    for (int i = idx; i < n; i += gridDim.x * blockDim.x) {
        local_max = fmaxf(local_max, fabsf(input[i]));
    }

    shared_max[tid] = local_max;
    __syncthreads();

    // Reduce within block
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_max[tid] = fmaxf(shared_max[tid], shared_max[tid + s]);
        }
        __syncthreads();
    }

    // Write block result
    if (tid == 0) {
        atomicMax((int*)output, __float_as_int(shared_max[0]));
    }
}

// Quantize with per-tensor scale
__global__ void quantize_per_tensor_kernel(const float* input, fp8_e4m3* output,
                                           float scale, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = input[idx] / scale;

        // Saturate to [-448, 448]
        val = fmaxf(-448.0f, fminf(val, 448.0f));

        // Convert to FP8 (simplified)
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

        output[idx] = fp8_e4m3(result);
    }
}

// Dequantize with per-tensor scale
__global__ void dequantize_per_tensor_kernel(const fp8_e4m3* input, float* output,
                                             float scale, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        uint8_t bits = input[idx].__x;
        uint32_t sign = (bits >> 7) & 0x1;
        uint32_t exp = (bits >> 3) & 0xF;
        uint32_t mantissa = bits & 0x7;

        float result;
        if (exp == 0) {
            result = ldexpf((float)mantissa / 8.0f, -6);
        } else {
            result = ldexpf(1.0f + (float)mantissa / 8.0f, (int)exp - 7);
        }

        output[idx] = (sign ? -result : result) * scale;
    }
}

// ============================================================================
// Per-Channel Quantization
// ============================================================================

// Find max per row (for weight matrix MxN, M channels)
__global__ void find_absmax_per_row_kernel(const float* input, float* row_max,
                                           int M, int N) {
    int row = blockIdx.x;
    if (row >= M) return;

    __shared__ float shared_max[256];
    int tid = threadIdx.x;

    float local_max = 0.0f;
    for (int col = tid; col < N; col += blockDim.x) {
        local_max = fmaxf(local_max, fabsf(input[row * N + col]));
    }

    shared_max[tid] = local_max;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_max[tid] = fmaxf(shared_max[tid], shared_max[tid + s]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        row_max[row] = shared_max[0];
    }
}

// Quantize per-channel
__global__ void quantize_per_channel_kernel(const float* input, fp8_e4m3* output,
                                            const float* scales, int M, int N) {
    int row = blockIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        int idx = row * N + col;
        float val = input[idx] / scales[row];
        val = fmaxf(-448.0f, fminf(val, 448.0f));

        // Simplified FP8 conversion
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

        output[idx] = fp8_e4m3(result);
    }
}

// ============================================================================
// Test and benchmark
// ============================================================================

void test_per_tensor_quantization() {
    printf("=== Per-Tensor Quantization Test ===\n");

    const int N = 1024 * 1024;
    float* h_input = (float*)malloc(N * sizeof(float));
    float* h_output = (float*)malloc(N * sizeof(float));

    // Generate test data (Gaussian N(0, 10))
    srand(42);
    for (int i = 0; i < N; i++) {
        float u1 = rand() / (float)RAND_MAX;
        float u2 = rand() / (float)RAND_MAX;
        h_input[i] = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * M_PI * u2) * 10.0f;
    }

    // Allocate device memory
    float *d_input, *d_output, *d_absmax;
    fp8_e4m3* d_quantized;

    CHECK_CUDA(cudaMalloc(&d_input, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_output, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_quantized, N * sizeof(fp8_e4m3)));
    CHECK_CUDA(cudaMalloc(&d_absmax, sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_input, h_input, N * sizeof(float),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(d_absmax, 0, sizeof(float)));

    // Find max
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;
    find_absmax_kernel<<<gridSize, blockSize>>>(d_input, d_absmax, N);

    float h_absmax;
    CHECK_CUDA(cudaMemcpy(&h_absmax, d_absmax, sizeof(float),
                          cudaMemcpyDeviceToHost));

    // Compute scale (to fill FP8 range)
    float scale = h_absmax / 448.0f;
    printf("Input absmax: %.6f\n", h_absmax);
    printf("Scale factor: %.6f\n", scale);

    // Quantize
    quantize_per_tensor_kernel<<<gridSize, blockSize>>>(d_input, d_quantized,
                                                        scale, N);

    // Dequantize
    dequantize_per_tensor_kernel<<<gridSize, blockSize>>>(d_quantized, d_output,
                                                          scale, N);

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

    printf("Mean Absolute Error: %.6f\n", mae / N);
    printf("Root Mean Square Error: %.6f\n", sqrt(mse / N));
    printf("Max Absolute Error: %.6f\n\n", max_error);

    free(h_input);
    free(h_output);
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_quantized);
    cudaFree(d_absmax);
}

void test_per_channel_quantization() {
    printf("=== Per-Channel Quantization Test ===\n");

    const int M = 512;  // Channels
    const int N = 2048; // Features per channel
    const int total = M * N;

    float* h_input = (float*)malloc(total * sizeof(float));
    float* h_output = (float*)malloc(total * sizeof(float));

    // Generate data with different scales per channel
    srand(42);
    for (int i = 0; i < M; i++) {
        float channel_scale = 1.0f + (rand() % 100) / 10.0f; // [1.0, 11.0]
        for (int j = 0; j < N; j++) {
            float u1 = rand() / (float)RAND_MAX;
            float u2 = rand() / (float)RAND_MAX;
            h_input[i * N + j] = sqrtf(-2.0f * logf(u1)) *
                                 cosf(2.0f * M_PI * u2) * channel_scale;
        }
    }

    float *d_input, *d_output, *d_scales;
    fp8_e4m3* d_quantized;

    CHECK_CUDA(cudaMalloc(&d_input, total * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_output, total * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_quantized, total * sizeof(fp8_e4m3)));
    CHECK_CUDA(cudaMalloc(&d_scales, M * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_input, h_input, total * sizeof(float),
                          cudaMemcpyHostToDevice));

    // Find max per row
    find_absmax_per_row_kernel<<<M, 256>>>(d_input, d_scales, M, N);

    // Compute scales (on GPU or CPU - doing on CPU for simplicity)
    float* h_scales = (float*)malloc(M * sizeof(float));
    CHECK_CUDA(cudaMemcpy(h_scales, d_scales, M * sizeof(float),
                          cudaMemcpyDeviceToHost));

    for (int i = 0; i < M; i++) {
        h_scales[i] = h_scales[i] / 448.0f;
    }

    CHECK_CUDA(cudaMemcpy(d_scales, h_scales, M * sizeof(float),
                          cudaMemcpyHostToDevice));

    // Quantize per-channel
    dim3 blockDim(256);
    dim3 gridDim((N + 255) / 256, M);
    quantize_per_channel_kernel<<<gridDim, blockDim>>>(d_input, d_quantized,
                                                        d_scales, M, N);

    // Dequantize (same kernel structure)
    // For simplicity, using host loop here
    CHECK_CUDA(cudaDeviceSynchronize());

    // Calculate error (simplified - doing on CPU)
    float* h_quantized_float = (float*)malloc(total * sizeof(float));
    fp8_e4m3* h_quantized = (fp8_e4m3*)malloc(total * sizeof(fp8_e4m3));
    CHECK_CUDA(cudaMemcpy(h_quantized, d_quantized, total * sizeof(fp8_e4m3),
                          cudaMemcpyDeviceToHost));

    for (int i = 0; i < total; i++) {
        uint8_t bits = h_quantized[i].__x;
        uint32_t sign = (bits >> 7) & 0x1;
        uint32_t exp = (bits >> 3) & 0xF;
        uint32_t mantissa = bits & 0x7;

        float result;
        if (exp == 0) {
            result = ldexpf((float)mantissa / 8.0f, -6);
        } else {
            result = ldexpf(1.0f + (float)mantissa / 8.0f, (int)exp - 7);
        }

        int row = i / N;
        h_quantized_float[i] = (sign ? -result : result) * h_scales[row];
    }

    double mae = 0.0, mse = 0.0, max_error = 0.0;
    for (int i = 0; i < total; i++) {
        double error = fabs(h_quantized_float[i] - h_input[i]);
        mae += error;
        mse += error * error;
        max_error = fmax(max_error, error);
    }

    printf("Mean Absolute Error: %.6f\n", mae / total);
    printf("Root Mean Square Error: %.6f\n", sqrt(mse / total));
    printf("Max Absolute Error: %.6f\n\n", max_error);

    free(h_input);
    free(h_output);
    free(h_scales);
    free(h_quantized);
    free(h_quantized_float);
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_quantized);
    cudaFree(d_scales);
}

int main() {
    printf("=== Quantization Techniques Demo ===\n\n");

    test_per_tensor_quantization();
    test_per_channel_quantization();

    printf("Compare the error metrics:\n");
    printf("Per-channel should have lower error due to adaptive scaling.\n");

    return 0;
}
