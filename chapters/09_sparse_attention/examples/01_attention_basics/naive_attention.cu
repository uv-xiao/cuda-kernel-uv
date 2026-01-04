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

// Kernel 1: Compute QK^T and scale
// Output S: [batch_size, num_heads, seq_len, seq_len]
__global__ void qk_kernel(
    const float* Q,  // [batch_size, num_heads, seq_len, head_dim]
    const float* K,  // [batch_size, num_heads, seq_len, head_dim]
    float* S,        // [batch_size, num_heads, seq_len, seq_len]
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim,
    float scale
) {
    int b = blockIdx.z;
    int h = blockIdx.y;
    int i = blockIdx.x * blockDim.x + threadIdx.x; // query position
    int j = threadIdx.y; // key position (handle multiple per thread if needed)

    if (b >= batch_size || h >= num_heads || i >= seq_len) return;

    // Compute dot product Q[b,h,i,:] · K[b,h,j,:]
    for (int j_pos = j; j_pos < seq_len; j_pos += blockDim.y) {
        float sum = 0.0f;
        int q_offset = ((b * num_heads + h) * seq_len + i) * head_dim;
        int k_offset = ((b * num_heads + h) * seq_len + j_pos) * head_dim;

        for (int d = 0; d < head_dim; d++) {
            sum += Q[q_offset + d] * K[k_offset + d];
        }

        // Scale and store
        int s_offset = ((b * num_heads + h) * seq_len + i) * seq_len + j_pos;
        S[s_offset] = sum * scale;
    }
}

// Kernel 2: Softmax over rows
// In-place operation on S
__global__ void softmax_kernel(
    float* S,           // [batch_size, num_heads, seq_len, seq_len]
    const bool* mask,   // [seq_len, seq_len] or nullptr (causal mask)
    int batch_size,
    int num_heads,
    int seq_len,
    bool is_causal
) {
    int b = blockIdx.z;
    int h = blockIdx.y;
    int i = blockIdx.x * blockDim.x + threadIdx.x; // row to normalize

    if (b >= batch_size || h >= num_heads || i >= seq_len) return;

    int row_offset = ((b * num_heads + h) * seq_len + i) * seq_len;

    // Step 1: Find max (for numerical stability)
    float max_val = -FLT_MAX;
    for (int j = 0; j < seq_len; j++) {
        // Apply causal mask if needed
        if (is_causal && j > i) continue;
        if (mask != nullptr && !mask[i * seq_len + j]) continue;

        max_val = fmaxf(max_val, S[row_offset + j]);
    }

    // Step 2: Compute exp and sum
    float sum_exp = 0.0f;
    for (int j = 0; j < seq_len; j++) {
        if (is_causal && j > i) {
            S[row_offset + j] = 0.0f;
            continue;
        }
        if (mask != nullptr && !mask[i * seq_len + j]) {
            S[row_offset + j] = 0.0f;
            continue;
        }

        float exp_val = expf(S[row_offset + j] - max_val);
        S[row_offset + j] = exp_val;
        sum_exp += exp_val;
    }

    // Step 3: Normalize
    for (int j = 0; j < seq_len; j++) {
        S[row_offset + j] /= sum_exp;
    }
}

// Kernel 3: Compute output O = S @ V
// Output O: [batch_size, num_heads, seq_len, head_dim]
__global__ void sv_kernel(
    const float* S,  // [batch_size, num_heads, seq_len, seq_len]
    const float* V,  // [batch_size, num_heads, seq_len, head_dim]
    float* O,        // [batch_size, num_heads, seq_len, head_dim]
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim
) {
    int b = blockIdx.z;
    int h = blockIdx.y;
    int i = blockIdx.x * blockDim.x + threadIdx.x; // output position
    int d = threadIdx.y; // head dimension (handle multiple per thread if needed)

    if (b >= batch_size || h >= num_heads || i >= seq_len) return;

    // Compute weighted sum O[b,h,i,d] = sum_j S[b,h,i,j] * V[b,h,j,d]
    for (int d_pos = d; d_pos < head_dim; d_pos += blockDim.y) {
        float sum = 0.0f;
        int s_offset = ((b * num_heads + h) * seq_len + i) * seq_len;

        for (int j = 0; j < seq_len; j++) {
            int v_offset = ((b * num_heads + h) * seq_len + j) * head_dim;
            sum += S[s_offset + j] * V[v_offset + d_pos];
        }

        int o_offset = ((b * num_heads + h) * seq_len + i) * head_dim;
        O[o_offset + d_pos] = sum;
    }
}

// Host function to run naive attention
void naive_attention(
    const float* Q,
    const float* K,
    const float* V,
    float* O,
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim,
    bool is_causal = false
) {
    // Allocate temporary storage for attention scores
    size_t S_size = batch_size * num_heads * seq_len * seq_len;
    float* d_S;
    CUDA_CHECK(cudaMalloc(&d_S, S_size * sizeof(float)));

    // Scale factor
    float scale = 1.0f / sqrtf(static_cast<float>(head_dim));

    // Kernel 1: Compute QK^T
    dim3 qk_blocks((seq_len + 31) / 32, num_heads, batch_size);
    dim3 qk_threads(32, 8);
    qk_kernel<<<qk_blocks, qk_threads>>>(
        Q, K, d_S, batch_size, num_heads, seq_len, head_dim, scale
    );
    CUDA_CHECK(cudaGetLastError());

    // Kernel 2: Softmax
    dim3 sm_blocks((seq_len + 255) / 256, num_heads, batch_size);
    dim3 sm_threads(256);
    softmax_kernel<<<sm_blocks, sm_threads>>>(
        d_S, nullptr, batch_size, num_heads, seq_len, is_causal
    );
    CUDA_CHECK(cudaGetLastError());

    // Kernel 3: Compute SV
    dim3 sv_blocks((seq_len + 31) / 32, num_heads, batch_size);
    dim3 sv_threads(32, 4);
    sv_kernel<<<sv_blocks, sv_threads>>>(
        d_S, V, O, batch_size, num_heads, seq_len, head_dim
    );
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaFree(d_S));
}

// Test function
void test_attention() {
    int batch_size = 2;
    int num_heads = 4;
    int seq_len = 128;
    int head_dim = 64;

    size_t qkv_size = batch_size * num_heads * seq_len * head_dim;

    // Allocate and initialize host memory
    float* h_Q = new float[qkv_size];
    float* h_K = new float[qkv_size];
    float* h_V = new float[qkv_size];
    float* h_O = new float[qkv_size];

    // Initialize with random values
    for (size_t i = 0; i < qkv_size; i++) {
        h_Q[i] = (float)rand() / RAND_MAX - 0.5f;
        h_K[i] = (float)rand() / RAND_MAX - 0.5f;
        h_V[i] = (float)rand() / RAND_MAX - 0.5f;
    }

    // Allocate device memory
    float *d_Q, *d_K, *d_V, *d_O;
    CUDA_CHECK(cudaMalloc(&d_Q, qkv_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_K, qkv_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_V, qkv_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_O, qkv_size * sizeof(float)));

    // Copy to device
    CUDA_CHECK(cudaMemcpy(d_Q, h_Q, qkv_size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_K, h_K, qkv_size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_V, h_V, qkv_size * sizeof(float), cudaMemcpyHostToDevice));

    // Run attention
    printf("Running naive attention...\n");
    printf("Config: batch=%d, heads=%d, seq_len=%d, head_dim=%d\n",
           batch_size, num_heads, seq_len, head_dim);

    // Warmup
    naive_attention(d_Q, d_K, d_V, d_O, batch_size, num_heads, seq_len, head_dim);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Benchmark
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    int num_iters = 100;
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < num_iters; i++) {
        naive_attention(d_Q, d_K, d_V, d_O, batch_size, num_heads, seq_len, head_dim);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    printf("Average time: %.3f ms\n", milliseconds / num_iters);

    // Copy result back
    CUDA_CHECK(cudaMemcpy(h_O, d_O, qkv_size * sizeof(float), cudaMemcpyDeviceToHost));

    // Basic sanity check
    float sum = 0.0f;
    for (size_t i = 0; i < qkv_size; i++) {
        sum += h_O[i];
    }
    printf("Output sum: %.6f (sanity check)\n", sum);

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

    printf("Test completed successfully!\n");
}

int main() {
    test_attention();
    return 0;
}
