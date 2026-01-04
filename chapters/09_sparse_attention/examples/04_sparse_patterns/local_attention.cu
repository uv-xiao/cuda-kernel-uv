#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>
#include <float.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            printf("CUDA error at %s %d: %s\n", __FILE__, __LINE__, \
                   cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

/**
 * Local (Sliding Window) Attention Kernel
 *
 * Each query attends to a window of size window_size around itself.
 * For causal attention, only attends to previous window_size tokens.
 *
 * Complexity: O(L * window_size * d) instead of O(L^2 * d)
 */
__global__ void local_attention_kernel(
    const float* __restrict__ Q,  // [batch, heads, seq_len, head_dim]
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ O,
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim,
    int window_size,
    bool is_causal,
    float scale
) {
    // Each thread block handles one query position
    int b = blockIdx.z;
    int h = blockIdx.y;
    int q_pos = blockIdx.x;

    if (q_pos >= seq_len) return;

    int tid = threadIdx.x;
    int bhd_offset = (b * num_heads + h) * seq_len * head_dim;
    const float* Q_base = Q + bhd_offset;
    const float* K_base = K + bhd_offset;
    const float* V_base = V + bhd_offset;
    float* O_base = O + bhd_offset;

    // Determine window boundaries
    int k_start, k_end;
    if (is_causal) {
        // Causal: attend to [q_pos - window_size + 1, q_pos]
        k_start = max(0, q_pos - window_size + 1);
        k_end = q_pos + 1;
    } else {
        // Non-causal: attend to [q_pos - window_size/2, q_pos + window_size/2]
        int half_window = window_size / 2;
        k_start = max(0, q_pos - half_window);
        k_end = min(seq_len, q_pos + half_window + 1);
    }

    int window_len = k_end - k_start;

    // Shared memory for local window of K and V
    extern __shared__ float shared_mem[];
    float* K_local = shared_mem;
    float* V_local = K_local + window_size * head_dim;
    float* scores = V_local + window_size * head_dim;

    // Load Q for this position (in registers)
    float q_vec[64];  // Assuming head_dim <= 64
    for (int d = tid; d < head_dim; d += blockDim.x) {
        q_vec[d] = Q_base[q_pos * head_dim + d];
    }

    // Load K and V window into shared memory
    for (int i = tid; i < window_len * head_dim; i += blockDim.x) {
        int local_pos = i / head_dim;
        int d = i % head_dim;
        int global_pos = k_start + local_pos;
        K_local[local_pos * head_dim + d] = K_base[global_pos * head_dim + d];
        V_local[local_pos * head_dim + d] = V_base[global_pos * head_dim + d];
    }
    __syncthreads();

    // Compute attention scores: Q @ K^T
    for (int i = tid; i < window_len; i += blockDim.x) {
        float sum = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            sum += q_vec[d] * K_local[i * head_dim + d];
        }
        scores[i] = sum * scale;
    }
    __syncthreads();

    // Softmax over window
    // Step 1: Find max
    __shared__ float max_score;
    if (tid == 0) {
        max_score = -FLT_MAX;
        for (int i = 0; i < window_len; i++) {
            max_score = fmaxf(max_score, scores[i]);
        }
    }
    __syncthreads();

    // Step 2: Exp and sum
    __shared__ float sum_exp;
    if (tid == 0) {
        sum_exp = 0.0f;
        for (int i = 0; i < window_len; i++) {
            scores[i] = expf(scores[i] - max_score);
            sum_exp += scores[i];
        }
    }
    __syncthreads();

    // Step 3: Normalize
    for (int i = tid; i < window_len; i += blockDim.x) {
        scores[i] /= sum_exp;
    }
    __syncthreads();

    // Compute output: attention @ V
    for (int d = tid; d < head_dim; d += blockDim.x) {
        float sum = 0.0f;
        for (int i = 0; i < window_len; i++) {
            sum += scores[i] * V_local[i * head_dim + d];
        }
        O_base[q_pos * head_dim + d] = sum;
    }
}

// Host wrapper
void local_attention(
    const float* Q,
    const float* K,
    const float* V,
    float* O,
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim,
    int window_size,
    bool is_causal = true
) {
    float scale = 1.0f / sqrtf((float)head_dim);

    dim3 grid(seq_len, num_heads, batch_size);
    dim3 block(128);

    // Shared memory: K_window + V_window + scores
    size_t shared_mem = (2 * window_size * head_dim + window_size) * sizeof(float);

    local_attention_kernel<<<grid, block, shared_mem>>>(
        Q, K, V, O, batch_size, num_heads, seq_len, head_dim,
        window_size, is_causal, scale
    );

    CUDA_CHECK(cudaGetLastError());
}

// Test function
void test_local_attention() {
    printf("=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=\n");
    printf("Local (Sliding Window) Attention\n");
    printf("=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=\n\n");

    int batch_size = 4;
    int num_heads = 8;
    int seq_len = 2048;
    int head_dim = 64;
    int window_size = 256;

    printf("Configuration:\n");
    printf("  Batch: %d, Heads: %d\n", batch_size, num_heads);
    printf("  Sequence length: %d\n", seq_len);
    printf("  Head dimension: %d\n", head_dim);
    printf("  Window size: %d\n", window_size);
    printf("  Sparsity: %.1f%% (only computing %d out of %d)\n\n",
           100.0f * (1.0f - (float)window_size / seq_len),
           window_size, seq_len);

    size_t qkv_size = (size_t)batch_size * num_heads * seq_len * head_dim;

    // Allocate memory
    float* h_Q = new float[qkv_size];
    float* h_K = new float[qkv_size];
    float* h_V = new float[qkv_size];
    float* h_O = new float[qkv_size];

    // Initialize
    srand(42);
    for (size_t i = 0; i < qkv_size; i++) {
        h_Q[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
        h_K[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
        h_V[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
    }

    float *d_Q, *d_K, *d_V, *d_O;
    CUDA_CHECK(cudaMalloc(&d_Q, qkv_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_K, qkv_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_V, qkv_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_O, qkv_size * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_Q, h_Q, qkv_size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_K, h_K, qkv_size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_V, h_V, qkv_size * sizeof(float), cudaMemcpyHostToDevice));

    // Warmup
    for (int i = 0; i < 10; i++) {
        local_attention(d_Q, d_K, d_V, d_O, batch_size, num_heads,
                       seq_len, head_dim, window_size, true);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Benchmark
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    int num_iters = 100;
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < num_iters; i++) {
        local_attention(d_Q, d_K, d_V, d_O, batch_size, num_heads,
                       seq_len, head_dim, window_size, true);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    float avg_time = milliseconds / num_iters;

    // FLOPs: 4 * B * H * L * window_size * d (sparse instead of L^2)
    double flops = 4.0 * batch_size * num_heads * (double)seq_len * window_size * head_dim;
    double tflops = flops / (avg_time * 1e-3) / 1e12;

    printf("Performance:\n");
    printf("  Average time: %.3f ms\n", avg_time);
    printf("  Throughput: %.2f TFLOPs/s\n", tflops);

    // Compare with dense attention FLOPs
    double dense_flops = 4.0 * batch_size * num_heads * (double)seq_len * seq_len * head_dim;
    printf("  Speedup vs dense (FLOPs): %.1fx\n", dense_flops / flops);

    CUDA_CHECK(cudaMemcpy(h_O, d_O, qkv_size * sizeof(float), cudaMemcpyDeviceToHost));

    // Sanity check
    float sum = 0.0f;
    for (size_t i = 0; i < qkv_size; i++) {
        sum += h_O[i];
    }
    printf("\nOutput mean: %.6f\n", sum / qkv_size);

    // Cleanup
    delete[] h_Q;
    delete[] h_K;
    delete[] h_V;
    delete[] h_O;
    CUDA_CHECK(cudaFree(d_Q));
    CUDA_CHECK(cudaFree(d_K));
    CUDA_CHECK(cudaFree(d_V));
    CUDA_CHECK(cudaFree(d_O));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    printf("\n✓ Test completed successfully!\n");
}

int main() {
    test_local_attention();
    return 0;
}
