#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdint.h>
#include <math.h>

// Manual FP8 E4M3 implementation for educational purposes
struct FP8_E4M3 {
    uint8_t data;

    __host__ __device__ FP8_E4M3(uint8_t bits = 0) : data(bits) {}

    // Convert from float
    __host__ __device__ static FP8_E4M3 from_float(float value) {
        // Handle special cases
        if (isnan(value)) {
            return FP8_E4M3(0x7F); // NaN pattern: 0111 1111
        }

        // E4M3 max value is 448
        const float max_val = 448.0f;
        if (value > max_val) value = max_val;
        if (value < -max_val) value = -max_val;

        // Extract sign
        uint8_t sign = (value < 0) ? 1 : 0;
        value = fabsf(value);

        if (value == 0.0f) {
            return FP8_E4M3((sign << 7));
        }

        // Convert to exponent and mantissa
        int exp_bits;
        float mantissa = frexpf(value, &exp_bits);

        // Adjust exponent (bias = 7 for E4M3)
        int exponent = exp_bits + 6; // bias - 1 because frexp returns [0.5, 1)

        // Handle denormal numbers
        if (exponent <= 0) {
            // Denormal
            mantissa = ldexpf(mantissa, exponent);
            exponent = 0;
        } else if (exponent >= 15) {
            // Overflow to max
            return FP8_E4M3((sign << 7) | 0x7E);
        }

        // Round mantissa to 3 bits
        // Mantissa is in [0.5, 1) or [0, 0.5) for denormals
        int mantissa_int = (int)(mantissa * 16.0f); // Scale to [0, 16)
        if (exponent > 0) {
            mantissa_int -= 8; // Remove implicit leading bit
        }
        mantissa_int = mantissa_int < 0 ? 0 : (mantissa_int > 7 ? 7 : mantissa_int);

        uint8_t result = (sign << 7) | (exponent << 3) | mantissa_int;
        return FP8_E4M3(result);
    }

    // Convert to float
    __host__ __device__ float to_float() const {
        uint8_t sign = (data >> 7) & 0x1;
        uint8_t exponent = (data >> 3) & 0xF;
        uint8_t mantissa = data & 0x7;

        // Check for NaN
        if (exponent == 15 && mantissa == 7) {
            return NAN;
        }

        float result;
        if (exponent == 0) {
            // Denormal or zero
            result = ldexpf((float)mantissa / 8.0f, -6);
        } else {
            // Normal
            result = ldexpf(1.0f + (float)mantissa / 8.0f, exponent - 7);
        }

        return sign ? -result : result;
    }
};

// Manual FP8 E5M2 implementation
struct FP8_E5M2 {
    uint8_t data;

    __host__ __device__ FP8_E5M2(uint8_t bits = 0) : data(bits) {}

    __host__ __device__ static FP8_E5M2 from_float(float value) {
        if (isnan(value)) {
            return FP8_E5M2(0x7F); // NaN pattern
        }

        if (isinf(value)) {
            return FP8_E5M2(value > 0 ? 0x7C : 0xFC); // Infinity
        }

        uint8_t sign = (value < 0) ? 1 : 0;
        value = fabsf(value);

        // E5M2 max finite value is 57344
        const float max_val = 57344.0f;
        if (value >= max_val) {
            return FP8_E5M2((sign << 7) | 0x7C); // Infinity
        }

        if (value == 0.0f) {
            return FP8_E5M2((sign << 7));
        }

        int exp_bits;
        float mantissa = frexpf(value, &exp_bits);
        int exponent = exp_bits + 14; // bias - 1

        if (exponent <= 0) {
            mantissa = ldexpf(mantissa, exponent);
            exponent = 0;
        } else if (exponent >= 31) {
            return FP8_E5M2((sign << 7) | 0x7C); // Infinity
        }

        int mantissa_int = (int)(mantissa * 8.0f);
        if (exponent > 0) {
            mantissa_int -= 4; // Remove implicit bit
        }
        mantissa_int = mantissa_int < 0 ? 0 : (mantissa_int > 3 ? 3 : mantissa_int);

        uint8_t result = (sign << 7) | (exponent << 2) | mantissa_int;
        return FP8_E5M2(result);
    }

    __host__ __device__ float to_float() const {
        uint8_t sign = (data >> 7) & 0x1;
        uint8_t exponent = (data >> 2) & 0x1F;
        uint8_t mantissa = data & 0x3;

        // Check for infinity
        if (exponent == 31 && mantissa == 0) {
            return sign ? -INFINITY : INFINITY;
        }

        // Check for NaN
        if (exponent == 31 && mantissa != 0) {
            return NAN;
        }

        float result;
        if (exponent == 0) {
            result = ldexpf((float)mantissa / 4.0f, -14);
        } else {
            result = ldexpf(1.0f + (float)mantissa / 4.0f, exponent - 15);
        }

        return sign ? -result : result;
    }
};

// Test FP8 E4M3 range and precision
void test_e4m3_range() {
    printf("FP8 E4M3 Range Test:\n");

    // Max value
    FP8_E4M3 max_val = FP8_E4M3::from_float(448.0f);
    printf("  Max value: %.6f\n", max_val.to_float());

    // Min positive normalized
    FP8_E4M3 min_norm = FP8_E4M3(0x08); // exponent=1, mantissa=0
    printf("  Min positive normal: %.10f\n", min_norm.to_float());

    // Min positive denormal
    FP8_E4M3 min_denorm = FP8_E4M3(0x01); // exponent=0, mantissa=1
    printf("  Min positive denormal: %.10f\n", min_denorm.to_float());

    // Values near 1.0 (show quantization)
    printf("  Values near 1.0: [");
    for (int i = 0; i < 8; i++) {
        FP8_E4M3 val = FP8_E4M3((7 << 3) | i); // exponent=7 (bias), varying mantissa
        printf("%.3f", val.to_float());
        if (i < 7) printf(", ");
    }
    printf("]\n\n");
}

// Test FP8 E5M2 range and precision
void test_e5m2_range() {
    printf("FP8 E5M2 Range Test:\n");

    // Max finite value
    FP8_E5M2 max_val = FP8_E5M2::from_float(57344.0f);
    printf("  Max finite value: %.1f\n", max_val.to_float());

    // Min positive normalized
    FP8_E5M2 min_norm = FP8_E5M2(0x04); // exponent=1, mantissa=0
    printf("  Min positive normal: %.10f\n", min_norm.to_float());

    // Min positive denormal
    FP8_E5M2 min_denorm = FP8_E5M2(0x01); // exponent=0, mantissa=1
    printf("  Min positive denormal: %.10f\n", min_denorm.to_float());

    // Infinity
    FP8_E5M2 inf_val = FP8_E5M2(0x7C);
    printf("  Infinity: %f\n", inf_val.to_float());

    // Values near 1.0
    printf("  Values near 1.0: [");
    for (int i = 0; i < 4; i++) {
        FP8_E5M2 val = FP8_E5M2((15 << 2) | i); // exponent=15 (bias)
        printf("%.2f", val.to_float());
        if (i < 3) printf(", ");
    }
    printf("]\n\n");
}

// Kernel to test conversion accuracy
__global__ void test_conversion_kernel(float* input, float* e4m3_output,
                                       float* e5m2_output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = input[idx];

        // E4M3 round-trip
        FP8_E4M3 e4m3 = FP8_E4M3::from_float(val);
        e4m3_output[idx] = e4m3.to_float();

        // E5M2 round-trip
        FP8_E5M2 e5m2 = FP8_E5M2::from_float(val);
        e5m2_output[idx] = e5m2.to_float();
    }
}

void test_conversion_accuracy() {
    const int N = 1000;
    float h_input[N], h_e4m3[N], h_e5m2[N];

    // Generate test values in different ranges
    for (int i = 0; i < N; i++) {
        if (i < 300) {
            // Normal range [-10, 10]
            h_input[i] = -10.0f + (20.0f * i / 300);
        } else if (i < 600) {
            // E4M3 limits
            h_input[i] = -400.0f + (800.0f * (i - 300) / 300);
        } else {
            // E5M2 extreme range
            h_input[i] = -50000.0f + (100000.0f * (i - 600) / 400);
        }
    }

    float *d_input, *d_e4m3, *d_e5m2;
    cudaMalloc(&d_input, N * sizeof(float));
    cudaMalloc(&d_e4m3, N * sizeof(float));
    cudaMalloc(&d_e5m2, N * sizeof(float));

    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);

    test_conversion_kernel<<<(N + 255) / 256, 256>>>(d_input, d_e4m3, d_e5m2, N);

    cudaMemcpy(h_e4m3, d_e4m3, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_e5m2, d_e5m2, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Calculate errors
    double e4m3_total_error = 0.0, e5m2_total_error = 0.0;
    int e4m3_valid = 0, e5m2_valid = 0;

    for (int i = 0; i < N; i++) {
        if (fabsf(h_input[i]) <= 448.0f) {
            e4m3_total_error += fabsf(h_e4m3[i] - h_input[i]);
            e4m3_valid++;
        }
        if (fabsf(h_input[i]) <= 57344.0f) {
            e5m2_total_error += fabsf(h_e5m2[i] - h_input[i]);
            e5m2_valid++;
        }
    }

    printf("Conversion Accuracy Test (%d values):\n", N);
    printf("  E4M3 Mean Absolute Error: %.6f (valid: %d)\n",
           e4m3_total_error / e4m3_valid, e4m3_valid);
    printf("  E5M2 Mean Absolute Error: %.6f (valid: %d)\n\n",
           e5m2_total_error / e5m2_valid, e5m2_valid);

    cudaFree(d_input);
    cudaFree(d_e4m3);
    cudaFree(d_e5m2);
}

// Demonstrate precision around specific values
void show_precision_examples() {
    printf("Precision Examples:\n");

    printf("  Around 1.0 (E4M3):\n");
    for (float f = 0.9f; f <= 1.1f; f += 0.02f) {
        FP8_E4M3 e4m3 = FP8_E4M3::from_float(f);
        printf("    %.3f -> 0x%02X -> %.6f (error: %.6f)\n",
               f, e4m3.data, e4m3.to_float(), e4m3.to_float() - f);
    }

    printf("\n  Around 100.0 (E4M3):\n");
    for (float f = 96.0f; f <= 104.0f; f += 2.0f) {
        FP8_E4M3 e4m3 = FP8_E4M3::from_float(f);
        printf("    %.1f -> 0x%02X -> %.6f (error: %.6f)\n",
               f, e4m3.data, e4m3.to_float(), e4m3.to_float() - f);
    }

    printf("\n  Large values (E5M2):\n");
    for (float f = 10000.0f; f <= 50000.0f; f += 10000.0f) {
        FP8_E5M2 e5m2 = FP8_E5M2::from_float(f);
        printf("    %.1f -> 0x%02X -> %.1f (error: %.1f)\n",
               f, e5m2.data, e5m2.to_float(), e5m2.to_float() - f);
    }
    printf("\n");
}

int main() {
    printf("=== FP8 Data Types Exploration ===\n\n");

    test_e4m3_range();
    test_e5m2_range();
    test_conversion_accuracy();
    show_precision_examples();

    printf("=== Hardware FP8 Support Check ===\n");
    int device = 0;
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    printf("Device: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    if (prop.major >= 9) {
        printf("Status: FP8 Tensor Cores SUPPORTED\n");
    } else {
        printf("Status: FP8 Tensor Cores NOT supported (need 9.0+)\n");
    }

    return 0;
}
