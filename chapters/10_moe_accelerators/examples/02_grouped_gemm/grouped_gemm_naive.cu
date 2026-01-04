/*
 * Naive Grouped GEMM for MoE
 *
 * Sequential processing of experts using standard cuBLAS calls.
 * Demonstrates the performance bottleneck of irregular batch sizes.
 */

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <vector>
#include <random>

#define CHECK_CUDA(call) \
    do { \
        cudaError_t status = call; \
        if (status != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(status)); \
            exit(1); \
        } \
    } while(0)

#define CHECK_CUBLAS(call) \
    do { \
        cublasStatus_t status = call; \
        if (status != CUBLAS_STATUS_SUCCESS) { \
            fprintf(stderr, "cuBLAS error at %s:%d: %d\n", __FILE__, __LINE__, status); \
            exit(1); \
        } \
    } while(0)

// Configuration
struct MoEConfig {
    int num_experts;
    int hidden_dim;
    int ffn_dim;
    int total_tokens;
    int top_k;
};

// Expert assignment data
struct ExpertAssignment {
    std::vector<int> token_counts;  // Tokens per expert
    std::vector<std::vector<int>> token_indices;  // Token indices for each expert
};

// Simulate expert routing with random assignment
ExpertAssignment simulate_routing(const MoEConfig& config) {
    ExpertAssignment assignment;
    assignment.token_counts.resize(config.num_experts, 0);
    assignment.token_indices.resize(config.num_experts);

    // Random number generator
    std::mt19937 gen(42);
    std::uniform_int_distribution<> expert_dist(0, config.num_experts - 1);

    // Assign each token to top_k random experts
    for (int token_id = 0; token_id < config.total_tokens; ++token_id) {
        for (int k = 0; k < config.top_k; ++k) {
            int expert_idx = expert_dist(gen);
            assignment.token_counts[expert_idx]++;
            assignment.token_indices[expert_idx].push_back(token_id);
        }
    }

    return assignment;
}

// Naive grouped GEMM: process each expert sequentially
float naive_grouped_gemm(
    const MoEConfig& config,
    const ExpertAssignment& assignment,
    const float* d_input,           // [total_tokens, hidden_dim]
    const float** d_expert_weights, // [num_experts][hidden_dim, ffn_dim]
    float** d_expert_outputs,       // [num_experts][num_tokens_i, ffn_dim]
    cublasHandle_t cublas_handle
) {
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    const float alpha = 1.0f;
    const float beta = 0.0f;

    CHECK_CUDA(cudaEventRecord(start));

    // Process each expert sequentially
    for (int expert_idx = 0; expert_idx < config.num_experts; ++expert_idx) {
        int num_tokens = assignment.token_counts[expert_idx];

        if (num_tokens == 0) {
            continue;  // Skip empty experts
        }

        // Gather input tokens for this expert (simplified: assume contiguous)
        // In practice, you'd need a gather kernel here
        const float* expert_input = d_input;  // Placeholder

        // GEMM: expert_output = expert_input @ expert_weights
        // C = A @ B where:
        //   A: [num_tokens, hidden_dim]
        //   B: [hidden_dim, ffn_dim]
        //   C: [num_tokens, ffn_dim]
        CHECK_CUBLAS(cublasSgemm(
            cublas_handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            config.ffn_dim,        // n
            num_tokens,            // m (variable!)
            config.hidden_dim,     // k
            &alpha,
            d_expert_weights[expert_idx], config.ffn_dim,  // lda
            expert_input, config.hidden_dim,               // ldb
            &beta,
            d_expert_outputs[expert_idx], config.ffn_dim   // ldc
        ));
    }

    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float elapsed_ms;
    CHECK_CUDA(cudaEventElapsedTime(&elapsed_ms, start, stop));

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return elapsed_ms;
}

// Compute theoretical FLOPs
double compute_flops(const MoEConfig& config, const ExpertAssignment& assignment) {
    double total_flops = 0.0;

    for (int expert_idx = 0; expert_idx < config.num_experts; ++expert_idx) {
        int m = assignment.token_counts[expert_idx];
        int k = config.hidden_dim;
        int n = config.ffn_dim;

        // GEMM FLOPs: 2 * m * k * n
        total_flops += 2.0 * m * k * n;
    }

    return total_flops;
}

int main() {
    printf("================================================\n");
    printf("Grouped GEMM Benchmark (Naive)\n");
    printf("================================================\n\n");

    // Configuration (similar to DeepSeek-V3 scaled down)
    MoEConfig config = {
        .num_experts = 8,
        .hidden_dim = 4096,
        .ffn_dim = 14336,
        .total_tokens = 16384,  // 32 batch * 512 seq_len
        .top_k = 2,
    };

    printf("Configuration:\n");
    printf("  - Num Experts: %d\n", config.num_experts);
    printf("  - Hidden Dim: %d\n", config.hidden_dim);
    printf("  - FFN Dim: %d\n", config.ffn_dim);
    printf("  - Total Tokens: %d\n", config.total_tokens);
    printf("  - Top-k: %d\n\n", config.top_k);

    // Simulate expert routing
    ExpertAssignment assignment = simulate_routing(config);

    printf("Expert Token Distribution:\n");
    for (int i = 0; i < config.num_experts; ++i) {
        printf("  Expert %d: %d tokens\n", i, assignment.token_counts[i]);
    }
    printf("\n");

    // Initialize cuBLAS
    cublasHandle_t cublas_handle;
    CHECK_CUBLAS(cublasCreate(&cublas_handle));

    // Allocate device memory
    float* d_input;
    CHECK_CUDA(cudaMalloc(&d_input, config.total_tokens * config.hidden_dim * sizeof(float)));

    std::vector<float*> h_expert_weights(config.num_experts);
    std::vector<float*> h_expert_outputs(config.num_experts);

    for (int i = 0; i < config.num_experts; ++i) {
        CHECK_CUDA(cudaMalloc(&h_expert_weights[i],
                             config.hidden_dim * config.ffn_dim * sizeof(float)));

        if (assignment.token_counts[i] > 0) {
            CHECK_CUDA(cudaMalloc(&h_expert_outputs[i],
                                 assignment.token_counts[i] * config.ffn_dim * sizeof(float)));
        }
    }

    // Copy expert weight pointers to device
    float** d_expert_weights;
    float** d_expert_outputs;
    CHECK_CUDA(cudaMalloc(&d_expert_weights, config.num_experts * sizeof(float*)));
    CHECK_CUDA(cudaMalloc(&d_expert_outputs, config.num_experts * sizeof(float*)));
    CHECK_CUDA(cudaMemcpy(d_expert_weights, h_expert_weights.data(),
                         config.num_experts * sizeof(float*), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_expert_outputs, h_expert_outputs.data(),
                         config.num_experts * sizeof(float*), cudaMemcpyHostToDevice));

    // Initialize input (random data)
    std::vector<float> h_input(config.total_tokens * config.hidden_dim);
    for (auto& val : h_input) {
        val = static_cast<float>(rand()) / RAND_MAX;
    }
    CHECK_CUDA(cudaMemcpy(d_input, h_input.data(),
                         config.total_tokens * config.hidden_dim * sizeof(float),
                         cudaMemcpyHostToDevice));

    // Warmup
    for (int i = 0; i < 3; ++i) {
        naive_grouped_gemm(config, assignment, d_input, (const float**)d_expert_weights,
                          d_expert_outputs, cublas_handle);
    }

    // Benchmark
    const int num_iters = 10;
    float total_time = 0.0f;

    for (int i = 0; i < num_iters; ++i) {
        float elapsed = naive_grouped_gemm(config, assignment, d_input,
                                          (const float**)d_expert_weights,
                                          d_expert_outputs, cublas_handle);
        total_time += elapsed;
    }

    float avg_time = total_time / num_iters;

    // Compute performance metrics
    double total_flops = compute_flops(config, assignment);
    double tflops = total_flops / 1e12;
    double tflops_per_sec = tflops / (avg_time / 1000.0);

    printf("Performance:\n");
    printf("  - Sequential Time: %.3f ms\n", avg_time);
    printf("  - Total FLOPs: %.3f TFLOPs\n", tflops);
    printf("  - Throughput: %.1f TFLOPs/s\n", tflops_per_sec);

    // Get GPU specs for utilization calculation
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));

    // Estimate SM utilization (very rough)
    // Assumes sequential execution prevents parallelism
    float sm_utilization = 100.0f / config.num_experts;  // Simplified estimate
    printf("  - Estimated SM Utilization: %.1f%%\n", sm_utilization);

    printf("\nBottleneck Analysis:\n");
    printf("  - Sequential processing limits parallelism\n");
    printf("  - Small batch sizes underutilize GPU\n");
    printf("  - Multiple kernel launches add overhead\n");

    // Cleanup
    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_expert_weights));
    CHECK_CUDA(cudaFree(d_expert_outputs));
    for (auto ptr : h_expert_weights) {
        CHECK_CUDA(cudaFree(ptr));
    }
    for (auto ptr : h_expert_outputs) {
        if (ptr) CHECK_CUDA(cudaFree(ptr));
    }
    CHECK_CUBLAS(cublasDestroy(cublas_handle));

    printf("\n================================================\n");

    return 0;
}
