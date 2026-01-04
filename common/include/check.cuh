#pragma once

#include <cmath>
#include <stdio.h>

// Verify two arrays are equal within tolerance
template <typename T>
bool verify_result(const T *expected, const T *actual, size_t n,
                   T rtol = 1e-5, T atol = 1e-8) {
    int errors = 0;
    int max_errors_to_print = 10;

    for (size_t i = 0; i < n; i++) {
        T diff = std::abs(expected[i] - actual[i]);
        T tolerance = atol + rtol * std::abs(expected[i]);

        if (diff > tolerance) {
            if (errors < max_errors_to_print) {
                printf("Mismatch at index %zu: expected %.6f, got %.6f "
                       "(diff=%.6e)\n",
                       i, (float)expected[i], (float)actual[i], (float)diff);
            }
            errors++;
        }
    }

    if (errors > 0) {
        printf("Total errors: %d / %zu (%.2f%%)\n", errors, n,
               100.0f * errors / n);
        return false;
    }

    printf("Verification PASSED: all %zu elements match\n", n);
    return true;
}

// Verify 2D matrix
template <typename T>
bool verify_matrix(const T *expected, const T *actual, int rows, int cols,
                   T rtol = 1e-5, T atol = 1e-8) {
    int errors = 0;
    int max_errors_to_print = 10;

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            size_t idx = i * cols + j;
            T diff = std::abs(expected[idx] - actual[idx]);
            T tolerance = atol + rtol * std::abs(expected[idx]);

            if (diff > tolerance) {
                if (errors < max_errors_to_print) {
                    printf("Mismatch at (%d, %d): expected %.6f, got %.6f "
                           "(diff=%.6e)\n",
                           i, j, (float)expected[idx], (float)actual[idx],
                           (float)diff);
                }
                errors++;
            }
        }
    }

    if (errors > 0) {
        printf("Total errors: %d / %d (%.2f%%)\n", errors, rows * cols,
               100.0f * errors / (rows * cols));
        return false;
    }

    printf("Verification PASSED: all %d elements match\n", rows * cols);
    return true;
}

// Print a small portion of a matrix for debugging
template <typename T>
void print_matrix(const T *data, int rows, int cols, const char *name,
                  int max_rows = 6, int max_cols = 6) {
    printf("%s (%d x %d):\n", name, rows, cols);

    int print_rows = (rows < max_rows) ? rows : max_rows;
    int print_cols = (cols < max_cols) ? cols : max_cols;

    for (int i = 0; i < print_rows; i++) {
        printf("  ");
        for (int j = 0; j < print_cols; j++) {
            printf("%8.4f ", (float)data[i * cols + j]);
        }
        if (cols > max_cols)
            printf("...");
        printf("\n");
    }
    if (rows > max_rows)
        printf("  ...\n");
    printf("\n");
}

// Fill array with random values
template <typename T>
void fill_random(T *data, size_t n, T min_val = 0.0, T max_val = 1.0) {
    for (size_t i = 0; i < n; i++) {
        data[i] = min_val + (max_val - min_val) * ((T)rand() / RAND_MAX);
    }
}

// Fill array with sequential values
template <typename T>
void fill_sequential(T *data, size_t n, T start = 0) {
    for (size_t i = 0; i < n; i++) {
        data[i] = start + (T)i;
    }
}

// Fill array with constant value
template <typename T>
void fill_constant(T *data, size_t n, T val) {
    for (size_t i = 0; i < n; i++) {
        data[i] = val;
    }
}

// Compute reference GEMM on CPU (for verification)
template <typename T>
void cpu_gemm(const T *A, const T *B, T *C, int M, int N, int K,
              T alpha = 1.0, T beta = 0.0) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            T sum = 0.0;
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = alpha * sum + beta * C[i * N + j];
        }
    }
}
