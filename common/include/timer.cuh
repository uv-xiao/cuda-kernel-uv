#pragma once

#include <cuda_runtime.h>
#include <stdio.h>

// GPU Timer using CUDA events
class GpuTimer {
  public:
    GpuTimer() {
        cudaEventCreate(&start_);
        cudaEventCreate(&stop_);
    }

    ~GpuTimer() {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }

    void start(cudaStream_t stream = 0) {
        cudaEventRecord(start_, stream);
    }

    void stop(cudaStream_t stream = 0) {
        cudaEventRecord(stop_, stream);
    }

    float elapsed_ms() {
        cudaEventSynchronize(stop_);
        float ms;
        cudaEventElapsedTime(&ms, start_, stop_);
        return ms;
    }

    float elapsed_us() { return elapsed_ms() * 1000.0f; }

  private:
    cudaEvent_t start_, stop_;
};

// Benchmark helper - runs kernel multiple times and reports statistics
template <typename KernelFunc>
void benchmark_kernel(const char *name, KernelFunc kernel, int warmup_runs = 5,
                      int timed_runs = 20) {
    GpuTimer timer;

    // Warmup
    for (int i = 0; i < warmup_runs; i++) {
        kernel();
    }
    cudaDeviceSynchronize();

    // Timed runs
    float total_ms = 0.0f;
    float min_ms = 1e10f;
    float max_ms = 0.0f;

    for (int i = 0; i < timed_runs; i++) {
        timer.start();
        kernel();
        timer.stop();
        float ms = timer.elapsed_ms();
        total_ms += ms;
        min_ms = fminf(min_ms, ms);
        max_ms = fmaxf(max_ms, ms);
    }

    float avg_ms = total_ms / timed_runs;
    printf("%s: avg=%.3f ms, min=%.3f ms, max=%.3f ms\n", name, avg_ms, min_ms,
           max_ms);
}

// Calculate and print performance metrics for GEMM
inline void print_gemm_performance(int M, int N, int K, float elapsed_ms) {
    // FLOPS for GEMM: 2 * M * N * K (multiply-add)
    double flops = 2.0 * M * N * K;
    double gflops = flops / (elapsed_ms * 1e6); // Convert ms to s, flops to GFLOPS

    printf("  Matrix size: M=%d, N=%d, K=%d\n", M, N, K);
    printf("  Time: %.3f ms\n", elapsed_ms);
    printf("  Performance: %.2f GFLOPS\n", gflops);
}

// Calculate and print memory bandwidth
inline void print_bandwidth(size_t bytes_transferred, float elapsed_ms) {
    double gb = bytes_transferred / (1024.0 * 1024.0 * 1024.0);
    double seconds = elapsed_ms / 1000.0;
    double bandwidth = gb / seconds;

    printf("  Data transferred: %.2f GB\n", gb);
    printf("  Time: %.3f ms\n", elapsed_ms);
    printf("  Bandwidth: %.2f GB/s\n", bandwidth);
}

// Calculate effective bandwidth for GEMM
inline void print_gemm_bandwidth(int M, int N, int K, size_t element_size,
                                  float elapsed_ms) {
    // Minimum data movement: read A (M*K), read B (K*N), write C (M*N)
    size_t bytes = (size_t)M * K * element_size + (size_t)K * N * element_size +
                   (size_t)M * N * element_size;
    print_bandwidth(bytes, elapsed_ms);
}
