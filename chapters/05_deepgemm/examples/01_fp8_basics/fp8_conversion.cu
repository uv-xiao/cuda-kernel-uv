#include <cuda_fp8.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            printf("CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                   cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Simple FP8 E4M3 wrapper for compatibility
struct __align__(1) fp8_e4m3 {
    uint8_t __x;

    __host__ __device__ fp8_e4m3(uint8_t val = 0) : __x(val) {}
};

struct __align__(1) fp8_e5m2 {
    uint8_t __x;

    __host__ __device__ fp8_e5m2(uint8_t val = 0) : __x(val) {}
};

// ============================================================================
// Conversion Kernels: FP32 <-> FP8 E4M3
// ============================================================================

__device__ __forceinline__ fp8_e4m3 float_to_fp8_e4m3(float val) {
    // Saturation to [-448, 448]
    const float kMaxVal = 448.0f;
    val = fmaxf(-kMaxVal, fminf(val, kMaxVal));

    // Simple bit manipulation conversion
    uint32_t bits = __float_as_uint(val);
    uint32_t sign = (bits >> 31) & 0x1;
    int32_t exp = ((bits >> 23) & 0xFF) - 127;
    uint32_t mantissa = bits & 0x7FFFFF;

    // E4M3: bias = 7, 3 mantissa bits
    uint8_t result = 0;

    if (exp < -9) {
        // Underflow to zero
        result = (sign << 7);
    } else if (exp > 8) {
        // Overflow to max
        result = (sign << 7) | 0x7E;
    } else if (exp < -6) {
        // Denormal range
        int shift = -6 - exp;
        uint32_t denorm_mantissa = (0x800000 | mantissa) >> (shift + 20);
        result = (sign << 7) | (denorm_mantissa & 0x7);
    } else {
        // Normal range
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
        // Zero or denormal
        result = ldexpf((float)mantissa / 8.0f, -6);
    } else {
        // Normal
        result = ldexpf(1.0f + (float)mantissa / 8.0f, (int)exp - 7);
    }

    return sign ? -result : result;
}

// ============================================================================
// Conversion Kernels: FP32 <-> FP8 E5M2
// ============================================================================

__device__ __forceinline__ fp8_e5m2 float_to_fp8_e5m2(float val) {
    uint32_t bits = __float_as_uint(val);
    uint32_t sign = (bits >> 31) & 0x1;
    int32_t exp = ((bits >> 23) & 0xFF) - 127;
    uint32_t mantissa = bits & 0x7FFFFF;

    uint8_t result = 0;

    if (isinf(val)) {
        result = (sign << 7) | 0x7C; // Infinity
    } else if (isnan(val)) {
        result = 0x7F; // NaN
    } else if (exp < -17) {
        result = (sign << 7); // Underflow to zero
    } else if (exp > 15) {
        result = (sign << 7) | 0x7C; // Overflow to infinity
    } else if (exp < -14) {
        // Denormal
        int shift = -14 - exp;
        uint32_t denorm_mantissa = (0x800000 | mantissa) >> (shift + 21);
        result = (sign << 7) | (denorm_mantissa & 0x3);
    } else {
        // Normal
        uint32_t e5m2_exp = exp + 15;
        uint32_t e5m2_mantissa = (mantissa >> 21) & 0x3;
        result = (sign << 7) | (e5m2_exp << 2) | e5m2_mantissa;
    }

    return fp8_e5m2(result);
}

__device__ __forceinline__ float fp8_e5m2_to_float(fp8_e5m2 val) {
    uint8_t bits = val.__x;
    uint32_t sign = (bits >> 7) & 0x1;
    uint32_t exp = (bits >> 2) & 0x1F;
    uint32_t mantissa = bits & 0x3;

    if (exp == 31) {
        if (mantissa == 0) {
            return sign ? -INFINITY : INFINITY;
        } else {
            return NAN;
        }
    }

    float result;
    if (exp == 0) {
        result = ldexpf((float)mantissa / 4.0f, -14);
    } else {
        result = ldexpf(1.0f + (float)mantissa / 4.0f, (int)exp - 15);
    }

    return sign ? -result : result;
}

// ============================================================================
// Conversion Kernels: BF16 <-> FP8
// ============================================================================

__device__ __forceinline__ fp8_e4m3 bf16_to_fp8_e4m3(__nv_bfloat16 val) {
    return float_to_fp8_e4m3(__bfloat162float(val));
}

__device__ __forceinline__ __nv_bfloat16 fp8_e4m3_to_bf16(fp8_e4m3 val) {
    return __float2bfloat16(fp8_e4m3_to_float(val));
}

// ============================================================================
// Vectorized Conversion Kernels
// ============================================================================

__global__ void fp32_to_fp8_e4m3_kernel(const float* input, fp8_e4m3* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = float_to_fp8_e4m3(input[idx]);
    }
}

__global__ void fp8_e4m3_to_fp32_kernel(const fp8_e4m3* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = fp8_e4m3_to_float(input[idx]);
    }
}

__global__ void fp32_to_fp8_e5m2_kernel(const float* input, fp8_e5m2* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = float_to_fp8_e5m2(input[idx]);
    }
}

__global__ void fp8_e5m2_to_fp32_kernel(const fp8_e5m2* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = fp8_e5m2_to_float(input[idx]);
    }
}

__global__ void bf16_to_fp8_e4m3_kernel(const __nv_bfloat16* input,
                                        fp8_e4m3* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = bf16_to_fp8_e4m3(input[idx]);
    }
}

// ============================================================================
// Vectorized conversion with float4 for better memory bandwidth
// ============================================================================

__global__ void fp32_to_fp8_e4m3_vec4_kernel(const float4* input,
                                             fp8_e4m3* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int vec_idx = idx * 4;

    if (vec_idx < n) {
        float4 in = input[idx];

        output[vec_idx + 0] = float_to_fp8_e4m3(in.x);
        output[vec_idx + 1] = float_to_fp8_e4m3(in.y);
        output[vec_idx + 2] = float_to_fp8_e4m3(in.z);
        output[vec_idx + 3] = float_to_fp8_e4m3(in.w);
    }
}

// ============================================================================
// Benchmark utilities
// ============================================================================

float benchmark_conversion(void (*kernel)(const float*, fp8_e4m3*, int),
                          const float* d_input, fp8_e4m3* d_output,
                          int n, int warmup, int iters) {
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;

    // Warmup
    for (int i = 0; i < warmup; i++) {
        kernel<<<gridSize, blockSize>>>(d_input, d_output, n);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    // Benchmark
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < iters; i++) {
        kernel<<<gridSize, blockSize>>>(d_input, d_output, n);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return milliseconds / iters;
}

void run_conversion_benchmarks() {
    const int N = 1 << 20; // 1M elements
    const int warmup = 10;
    const int iters = 100;

    // Allocate host memory
    float* h_input = (float*)malloc(N * sizeof(float));
    float* h_output = (float*)malloc(N * sizeof(float));

    // Initialize with random data
    srand(42);
    for (int i = 0; i < N; i++) {
        h_input[i] = (rand() / (float)RAND_MAX) * 200.0f - 100.0f;
    }

    // Allocate device memory
    float *d_input, *d_output;
    fp8_e4m3* d_fp8_e4m3;
    fp8_e5m2* d_fp8_e5m2;
    __nv_bfloat16* d_bf16;

    CHECK_CUDA(cudaMalloc(&d_input, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_output, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_fp8_e4m3, N * sizeof(fp8_e4m3)));
    CHECK_CUDA(cudaMalloc(&d_fp8_e5m2, N * sizeof(fp8_e5m2)));
    CHECK_CUDA(cudaMalloc(&d_bf16, N * sizeof(__nv_bfloat16)));

    CHECK_CUDA(cudaMemcpy(d_input, h_input, N * sizeof(float),
                          cudaMemcpyHostToDevice));

    printf("Conversion Performance (%d elements):\n", N);

    // Benchmark FP32 -> E4M3
    float time_ms = benchmark_conversion(fp32_to_fp8_e4m3_kernel, d_input,
                                        d_fp8_e4m3, N, warmup, iters);
    float bandwidth_gb = (N * (sizeof(float) + sizeof(fp8_e4m3)) / 1e9) / (time_ms / 1000.0f);
    printf("  FP32 -> E4M3: %.3f ms (%.1f GB/s)\n", time_ms, bandwidth_gb);

    // Benchmark E4M3 -> FP32
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;

    for (int i = 0; i < warmup; i++) {
        fp8_e4m3_to_fp32_kernel<<<gridSize, blockSize>>>(d_fp8_e4m3, d_output, N);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < iters; i++) {
        fp8_e4m3_to_fp32_kernel<<<gridSize, blockSize>>>(d_fp8_e4m3, d_output, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&time_ms, start, stop);
    time_ms /= iters;
    bandwidth_gb = (N * (sizeof(float) + sizeof(fp8_e4m3)) / 1e9) / (time_ms / 1000.0f);
    printf("  E4M3 -> FP32: %.3f ms (%.1f GB/s)\n", time_ms, bandwidth_gb);

    // Benchmark FP32 -> E5M2
    time_ms = 0;
    cudaEventRecord(start);
    for (int i = 0; i < iters; i++) {
        fp32_to_fp8_e5m2_kernel<<<gridSize, blockSize>>>(d_input, d_fp8_e5m2, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_ms, start, stop);
    time_ms /= iters;
    bandwidth_gb = (N * (sizeof(float) + sizeof(fp8_e5m2)) / 1e9) / (time_ms / 1000.0f);
    printf("  FP32 -> E5M2: %.3f ms (%.1f GB/s)\n", time_ms, bandwidth_gb);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Cleanup
    free(h_input);
    free(h_output);
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_fp8_e4m3);
    cudaFree(d_fp8_e5m2);
    cudaFree(d_bf16);
}

void test_accuracy() {
    const int N = 1000;
    float h_input[N], h_e4m3_output[N], h_e5m2_output[N];

    // Generate Gaussian distribution N(0, 10)
    srand(42);
    for (int i = 0; i < N; i++) {
        float u1 = rand() / (float)RAND_MAX;
        float u2 = rand() / (float)RAND_MAX;
        h_input[i] = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * M_PI * u2) * 10.0f;
    }

    float *d_input, *d_e4m3_output, *d_e5m2_output;
    fp8_e4m3* d_fp8_e4m3;
    fp8_e5m2* d_fp8_e5m2;

    CHECK_CUDA(cudaMalloc(&d_input, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_fp8_e4m3, N * sizeof(fp8_e4m3)));
    CHECK_CUDA(cudaMalloc(&d_fp8_e5m2, N * sizeof(fp8_e5m2)));
    CHECK_CUDA(cudaMalloc(&d_e4m3_output, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_e5m2_output, N * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_input, h_input, N * sizeof(float),
                          cudaMemcpyHostToDevice));

    // Convert and back
    fp32_to_fp8_e4m3_kernel<<<(N + 255) / 256, 256>>>(d_input, d_fp8_e4m3, N);
    fp8_e4m3_to_fp32_kernel<<<(N + 255) / 256, 256>>>(d_fp8_e4m3, d_e4m3_output, N);

    fp32_to_fp8_e5m2_kernel<<<(N + 255) / 256, 256>>>(d_input, d_fp8_e5m2, N);
    fp8_e5m2_to_fp32_kernel<<<(N + 255) / 256, 256>>>(d_fp8_e5m2, d_e5m2_output, N);

    CHECK_CUDA(cudaMemcpy(h_e4m3_output, d_e4m3_output, N * sizeof(float),
                          cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_e5m2_output, d_e5m2_output, N * sizeof(float),
                          cudaMemcpyDeviceToHost));

    // Calculate errors
    double e4m3_mae = 0.0, e5m2_mae = 0.0;
    double e4m3_mse = 0.0, e5m2_mse = 0.0;

    for (int i = 0; i < N; i++) {
        double e4m3_err = fabs(h_e4m3_output[i] - h_input[i]);
        double e5m2_err = fabs(h_e5m2_output[i] - h_input[i]);

        e4m3_mae += e4m3_err;
        e5m2_mae += e5m2_err;
        e4m3_mse += e4m3_err * e4m3_err;
        e5m2_mse += e5m2_err * e5m2_err;
    }

    printf("\nAccuracy Test (Gaussian N(0, 10), %d samples):\n", N);
    printf("  E4M3:\n");
    printf("    Mean Absolute Error: %.6f\n", e4m3_mae / N);
    printf("    Root Mean Square Error: %.6f\n", sqrt(e4m3_mse / N));
    printf("  E5M2:\n");
    printf("    Mean Absolute Error: %.6f\n", e5m2_mae / N);
    printf("    Root Mean Square Error: %.6f\n", sqrt(e5m2_mse / N));

    cudaFree(d_input);
    cudaFree(d_fp8_e4m3);
    cudaFree(d_fp8_e5m2);
    cudaFree(d_e4m3_output);
    cudaFree(d_e5m2_output);
}

int main() {
    printf("=== FP8 Conversion Benchmarks ===\n\n");

    run_conversion_benchmarks();
    test_accuracy();

    return 0;
}
