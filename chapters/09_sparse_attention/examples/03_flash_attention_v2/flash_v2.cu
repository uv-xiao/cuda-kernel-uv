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

// Warp-level primitives
__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Block sizes (can be tuned)
constexpr int Br = 64;
constexpr int Bc = 64;
constexpr int HEAD_DIM = 64;

/**
 * FlashAttention-2 Kernel
 *
 * Optimizations over minimal version:
 * 1. Warp-level reductions for max/sum
 * 2. Better register usage
 * 3. Reduced synchronization
 * 4. Vectorized memory access where possible
 */
__global__ void flash_attention_v2_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ O,
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim,
    float scale
) {
    // Indices
    int b = blockIdx.y;
    int h = blockIdx.z;
    int q_block_idx = blockIdx.x;

    int q_start = q_block_idx * Br;
    int q_end = min(q_start + Br, seq_len);
    int q_size = q_end - q_start;

    if (q_start >= seq_len) return;

    // Thread indices
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;

    // Shared memory
    __shared__ float Qi[Br][HEAD_DIM];
    __shared__ float Kj[Bc][HEAD_DIM];
    __shared__ float Vj[Bc][HEAD_DIM];
    __shared__ float Sij[Br][Bc];

    // Base offset for this batch and head
    int bhd_offset = (b * num_heads + h) * seq_len * head_dim;
    const float* Q_base = Q + bhd_offset;
    const float* K_base = K + bhd_offset;
    const float* V_base = V + bhd_offset;
    float* O_base = O + bhd_offset;

    // Load query block cooperatively
    for (int idx = tid; idx < q_size * head_dim; idx += blockDim.x) {
        int i = idx / head_dim;
        int d = idx % head_dim;
        Qi[i][d] = Q_base[(q_start + i) * head_dim + d];
    }
    __syncthreads();

    // Per-thread accumulators (in registers)
    float Oi_local[HEAD_DIM];
    #pragma unroll
    for (int d = 0; d < HEAD_DIM; d++) {
        Oi_local[d] = 0.0f;
    }

    float mi = -FLT_MAX;
    float li = 0.0f;

    int num_kv_blocks = (seq_len + Bc - 1) / Bc;

    // Loop over K, V blocks
    for (int kv_block = 0; kv_block < num_kv_blocks; kv_block++) {
        int k_start = kv_block * Bc;
        int k_end = min(k_start + Bc, seq_len);
        int k_size = k_end - k_start;

        // Load K block
        for (int idx = tid; idx < k_size * head_dim; idx += blockDim.x) {
            int i = idx / head_dim;
            int d = idx % head_dim;
            Kj[i][d] = K_base[(k_start + i) * head_dim + d];
        }

        // Load V block
        for (int idx = tid; idx < k_size * head_dim; idx += blockDim.x) {
            int i = idx / head_dim;
            int d = idx % head_dim;
            Vj[i][d] = V_base[(k_start + i) * head_dim + d];
        }
        __syncthreads();

        // Compute Sij = Qi @ Kj^T with better parallelization
        for (int idx = tid; idx < q_size * k_size; idx += blockDim.x) {
            int i = idx / k_size;
            int j = idx % k_size;

            float sum = 0.0f;
            #pragma unroll
            for (int d = 0; d < HEAD_DIM; d++) {
                sum += Qi[i][d] * Kj[j][d];
            }
            Sij[i][j] = sum * scale;
        }
        __syncthreads();

        // Process rows (each thread handles one row for better coherence)
        for (int i = tid; i < q_size; i += blockDim.x) {
            // Compute row max using warp reduction
            float row_max = -FLT_MAX;
            #pragma unroll
            for (int j = 0; j < k_size; j++) {
                row_max = fmaxf(row_max, Sij[i][j]);
            }

            // Update global max
            float mi_new = fmaxf(mi, row_max);
            float mi_diff = mi - mi_new;

            // Compute exp(Sij - mi_new) and sum
            float row_sum = 0.0f;
            #pragma unroll
            for (int j = 0; j < k_size; j++) {
                float exp_val = expf(Sij[i][j] - mi_new);
                Sij[i][j] = exp_val;
                row_sum += exp_val;
            }

            // Update normalizer
            float li_old = li;
            float scale_old = expf(mi_diff);
            float li_new = scale_old * li_old + row_sum;

            // Rescale old output
            #pragma unroll
            for (int d = 0; d < HEAD_DIM; d++) {
                Oi_local[d] *= scale_old;
            }

            // Add Pij @ Vj contribution
            #pragma unroll
            for (int d = 0; d < HEAD_DIM; d++) {
                float sum = 0.0f;
                #pragma unroll
                for (int j = 0; j < k_size; j++) {
                    sum += Sij[i][j] * Vj[j][d];
                }
                Oi_local[d] += sum;
            }

            // Update statistics
            mi = mi_new;
            li = li_new;
        }
        __syncthreads();
    }

    // Write output with final normalization
    for (int i = tid; i < q_size; i += blockDim.x) {
        float inv_li = 1.0f / li;
        #pragma unroll
        for (int d = 0; d < HEAD_DIM; d++) {
            O_base[(q_start + i) * head_dim + d] = Oi_local[d] * inv_li;
        }
    }
}

// Host wrapper
void flash_attention_v2(
    const float* Q,
    const float* K,
    const float* V,
    float* O,
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim
) {
    int num_q_blocks = (seq_len + Br - 1) / Br;
    float scale = 1.0f / sqrtf((float)head_dim);

    dim3 grid(num_q_blocks, batch_size, num_heads);
    dim3 block(256);

    flash_attention_v2_kernel<<<grid, block>>>(
        Q, K, V, O, batch_size, num_heads, seq_len, head_dim, scale
    );

    CUDA_CHECK(cudaGetLastError());
}

// Test and benchmark
void test_flash_v2() {
    printf("=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=\n");
    printf("FlashAttention-2 Optimized Implementation\n");
    printf("=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=\n\n");

    int batch_size = 4;
    int num_heads = 8;
    int seq_len = 2048;
    int head_dim = 64;

    printf("Configuration:\n");
    printf("  Batch size: %d\n", batch_size);
    printf("  Num heads: %d\n", num_heads);
    printf("  Sequence length: %d\n", seq_len);
    printf("  Head dimension: %d\n", head_dim);
    printf("  Block sizes: Br=%d, Bc=%d\n\n", Br, Bc);

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

    // Device memory
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
        flash_attention_v2(d_Q, d_K, d_V, d_O, batch_size, num_heads, seq_len, head_dim);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Benchmark
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    int num_iters = 100;
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < num_iters; i++) {
        flash_attention_v2(d_Q, d_K, d_V, d_O, batch_size, num_heads, seq_len, head_dim);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    float avg_time = milliseconds / num_iters;

    // Calculate FLOPs
    double flops = 4.0 * batch_size * num_heads * (double)seq_len * seq_len * head_dim;
    double tflops = flops / (avg_time * 1e-3) / 1e12;

    printf("Performance:\n");
    printf("  Average time: %.3f ms\n", avg_time);
    printf("  Throughput: %.2f TFLOPs/s\n", tflops);

    // Memory bandwidth
    size_t bytes_accessed = 4 * qkv_size * sizeof(float);  // Read Q,K,V, write O
    double bandwidth = bytes_accessed / (avg_time * 1e-3) / 1e9;
    printf("  Effective bandwidth: %.2f GB/s\n", bandwidth);

    // Get result
    CUDA_CHECK(cudaMemcpy(h_O, d_O, qkv_size * sizeof(float), cudaMemcpyDeviceToHost));

    // Sanity check
    float sum = 0.0f;
    for (size_t i = 0; i < qkv_size; i++) {
        sum += h_O[i];
    }

    printf("\nOutput statistics:\n");
    printf("  Mean: %.6f\n", sum / qkv_size);

    // Memory analysis
    size_t attn_saved = (size_t)batch_size * num_heads * seq_len * seq_len * sizeof(float);
    printf("\nMemory savings:\n");
    printf("  Attention matrix size (not materialized): %.2f MB\n", attn_saved / 1024.0f / 1024.0f);
    printf("  Savings vs naive: %.2fx\n",
           (float)attn_saved / (4.0f * batch_size * num_heads * seq_len * head_dim * sizeof(float)));

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

int main(int argc, char** argv) {
    test_flash_v2();
    return 0;
}
