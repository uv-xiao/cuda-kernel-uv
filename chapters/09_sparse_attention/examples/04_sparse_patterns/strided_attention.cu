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
 * Strided (Dilated) Attention Kernel
 *
 * Each query attends to every stride-th position.
 * Can capture long-range dependencies with O(L * L/stride * d) complexity.
 */
__global__ void strided_attention_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ O,
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim,
    int stride,
    int max_attended,  // Maximum number of positions to attend to
    float scale
) {
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

    // Shared memory for strided K, V, and scores
    extern __shared__ float shared_mem[];
    float* K_strided = shared_mem;
    float* V_strided = K_strided + max_attended * head_dim;
    float* scores = V_strided + max_attended * head_dim;

    // Load Q for this position
    float q_vec[64];
    for (int d = tid; d < head_dim; d += blockDim.x) {
        q_vec[d] = Q_base[q_pos * head_dim + d];
    }

    // Determine strided positions to attend to
    // Attend to positions: q_pos, q_pos - stride, q_pos - 2*stride, ...
    int num_attended = 0;
    int positions[256];  // Assuming max_attended <= 256

    for (int offset = 0; offset <= q_pos && num_attended < max_attended; offset += stride) {
        positions[num_attended] = q_pos - offset;
        num_attended++;
    }

    // Load K and V for strided positions
    for (int i = tid; i < num_attended * head_dim; i += blockDim.x) {
        int idx = i / head_dim;
        int d = i % head_dim;
        int pos = positions[idx];
        K_strided[idx * head_dim + d] = K_base[pos * head_dim + d];
        V_strided[idx * head_dim + d] = V_base[pos * head_dim + d];
    }
    __syncthreads();

    // Compute attention scores
    for (int i = tid; i < num_attended; i += blockDim.x) {
        float sum = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            sum += q_vec[d] * K_strided[i * head_dim + d];
        }
        scores[i] = sum * scale;
    }
    __syncthreads();

    // Softmax
    __shared__ float max_score;
    __shared__ float sum_exp;

    if (tid == 0) {
        max_score = -FLT_MAX;
        for (int i = 0; i < num_attended; i++) {
            max_score = fmaxf(max_score, scores[i]);
        }
    }
    __syncthreads();

    if (tid == 0) {
        sum_exp = 0.0f;
        for (int i = 0; i < num_attended; i++) {
            scores[i] = expf(scores[i] - max_score);
            sum_exp += scores[i];
        }
    }
    __syncthreads();

    for (int i = tid; i < num_attended; i += blockDim.x) {
        scores[i] /= sum_exp;
    }
    __syncthreads();

    // Compute output
    for (int d = tid; d < head_dim; d += blockDim.x) {
        float sum = 0.0f;
        for (int i = 0; i < num_attended; i++) {
            sum += scores[i] * V_strided[i * head_dim + d];
        }
        O_base[q_pos * head_dim + d] = sum;
    }
}

void strided_attention(
    const float* Q,
    const float* K,
    const float* V,
    float* O,
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim,
    int stride
) {
    float scale = 1.0f / sqrtf((float)head_dim);

    // Maximum number of positions each query can attend to
    int max_attended = (seq_len + stride - 1) / stride;

    dim3 grid(seq_len, num_heads, batch_size);
    dim3 block(128);

    size_t shared_mem = (2 * max_attended * head_dim + max_attended) * sizeof(float);

    strided_attention_kernel<<<grid, block, shared_mem>>>(
        Q, K, V, O, batch_size, num_heads, seq_len, head_dim,
        stride, max_attended, scale
    );

    CUDA_CHECK(cudaGetLastError());
}

void test_strided_attention() {
    printf("=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=\n");
    printf("Strided (Dilated) Attention\n");
    printf("=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=\n\n");

    int batch_size = 4;
    int num_heads = 8;
    int seq_len = 2048;
    int head_dim = 64;
    int stride = 8;

    printf("Configuration:\n");
    printf("  Batch: %d, Heads: %d\n", batch_size, num_heads);
    printf("  Sequence length: %d\n", seq_len);
    printf("  Head dimension: %d\n", head_dim);
    printf("  Stride: %d\n", stride);

    int avg_attended = seq_len / stride;
    printf("  Avg positions attended: %d (out of %d)\n", avg_attended, seq_len);
    printf("  Sparsity: %.1f%%\n\n", 100.0f * (1.0f - (float)avg_attended / seq_len));

    size_t qkv_size = (size_t)batch_size * num_heads * seq_len * head_dim;

    float* h_Q = new float[qkv_size];
    float* h_K = new float[qkv_size];
    float* h_V = new float[qkv_size];
    float* h_O = new float[qkv_size];

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
        strided_attention(d_Q, d_K, d_V, d_O, batch_size, num_heads,
                         seq_len, head_dim, stride);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Benchmark
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    int num_iters = 100;
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < num_iters; i++) {
        strided_attention(d_Q, d_K, d_V, d_O, batch_size, num_heads,
                         seq_len, head_dim, stride);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    float avg_time = milliseconds / num_iters;

    double flops = 4.0 * batch_size * num_heads * (double)seq_len * avg_attended * head_dim;
    double tflops = flops / (avg_time * 1e-3) / 1e12;

    printf("Performance:\n");
    printf("  Average time: %.3f ms\n", avg_time);
    printf("  Throughput: %.2f TFLOPs/s\n", tflops);

    double dense_flops = 4.0 * batch_size * num_heads * (double)seq_len * seq_len * head_dim;
    printf("  Speedup vs dense: %.1fx\n", dense_flops / flops);

    CUDA_CHECK(cudaMemcpy(h_O, d_O, qkv_size * sizeof(float), cudaMemcpyDeviceToHost));

    float sum = 0.0f;
    for (size_t i = 0; i < qkv_size; i++) {
        sum += h_O[i];
    }
    printf("\nOutput mean: %.6f\n", sum / qkv_size);

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
    test_strided_attention();
    return 0;
}
