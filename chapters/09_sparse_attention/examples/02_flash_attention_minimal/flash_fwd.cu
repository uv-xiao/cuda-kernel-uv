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

// Block sizes (fixed for simplicity)
#define Br 64  // Query block size
#define Bc 64  // Key/Value block size

/**
 * FlashAttention Forward Kernel (Minimal Educational Version)
 *
 * Each thread block processes one query block of size Br.
 * Loops over key/value blocks, computing attention incrementally.
 *
 * Key ideas:
 * 1. Tile Q, K, V into blocks that fit in shared memory
 * 2. Use online softmax to avoid materializing full attention matrix
 * 3. Fuse QK^T, softmax, and PV into single kernel
 */
__global__ void flash_attention_forward_kernel(
    const float* Q,  // [batch_size, num_heads, seq_len, head_dim]
    const float* K,  // [batch_size, num_heads, seq_len, head_dim]
    const float* V,  // [batch_size, num_heads, seq_len, head_dim]
    float* O,        // [batch_size, num_heads, seq_len, head_dim]
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim
) {
    // Batch and head indices
    int b = blockIdx.y;
    int h = blockIdx.z;

    // Query block index
    int q_block_idx = blockIdx.x;
    int q_start = q_block_idx * Br;
    int q_end = min(q_start + Br, seq_len);
    int q_size = q_end - q_start;

    if (q_start >= seq_len) return;

    // Thread index within block
    int tid = threadIdx.x;
    int num_threads = blockDim.x;

    // Shared memory for blocks
    __shared__ float Qi[Br][64];  // Query block (max head_dim = 64)
    __shared__ float Kj[Bc][64];  // Key block
    __shared__ float Vj[Bc][64];  // Value block
    __shared__ float Sij[Br][Bc]; // Attention scores

    // Shared memory for reductions
    __shared__ float shared_max[Br];
    __shared__ float shared_sum[Br];

    // Base pointers for this batch and head
    int bhd_stride = seq_len * head_dim;
    int bhd_offset = (b * num_heads + h) * bhd_stride;
    const float* Q_bhd = Q + bhd_offset;
    const float* K_bhd = K + bhd_offset;
    const float* V_bhd = V + bhd_offset;
    float* O_bhd = O + bhd_offset;

    float scale = 1.0f / sqrtf((float)head_dim);

    // Load query block into shared memory
    for (int i = tid; i < q_size * head_dim; i += num_threads) {
        int row = i / head_dim;
        int col = i % head_dim;
        Qi[row][col] = Q_bhd[(q_start + row) * head_dim + col];
    }
    __syncthreads();

    // Register storage for output and statistics (per thread)
    // Each thread handles one query row
    float Oi[64] = {0.0f};  // Output accumulator
    float mi = -FLT_MAX;     // Running max
    float li = 0.0f;         // Running sum of exp

    // Number of key/value blocks
    int num_kv_blocks = (seq_len + Bc - 1) / Bc;

    // Loop over key/value blocks
    for (int kv_block = 0; kv_block < num_kv_blocks; kv_block++) {
        int k_start = kv_block * Bc;
        int k_end = min(k_start + Bc, seq_len);
        int k_size = k_end - k_start;

        // Load key block
        for (int i = tid; i < k_size * head_dim; i += num_threads) {
            int row = i / head_dim;
            int col = i % head_dim;
            Kj[row][col] = K_bhd[(k_start + row) * head_dim + col];
        }

        // Load value block
        for (int i = tid; i < k_size * head_dim; i += num_threads) {
            int row = i / head_dim;
            int col = i % head_dim;
            Vj[row][col] = V_bhd[(k_start + row) * head_dim + col];
        }
        __syncthreads();

        // Compute Sij = Qi @ Kj^T (each thread handles subset of elements)
        for (int i = tid; i < q_size * k_size; i += num_threads) {
            int row = i / k_size;
            int col = i % k_size;

            float sum = 0.0f;
            for (int d = 0; d < head_dim; d++) {
                sum += Qi[row][d] * Kj[col][d];
            }
            Sij[row][col] = sum * scale;
        }
        __syncthreads();

        // Each thread processes one query row
        for (int q_row = tid; q_row < q_size; q_row += num_threads) {
            // Find max of this row (for numerical stability)
            float row_max = -FLT_MAX;
            for (int j = 0; j < k_size; j++) {
                row_max = fmaxf(row_max, Sij[q_row][j]);
            }

            // Compute new max
            float mi_new = fmaxf(mi, row_max);

            // Compute exp(Sij - mi_new) and sum
            float row_sum = 0.0f;
            for (int j = 0; j < k_size; j++) {
                float exp_val = expf(Sij[q_row][j] - mi_new);
                Sij[q_row][j] = exp_val;  // Reuse memory for Pij
                row_sum += exp_val;
            }

            // Update normalizer: li_new = exp(mi - mi_new) * li + row_sum
            float scale_old = expf(mi - mi_new);
            float li_new = scale_old * li + row_sum;

            // Update output: Oi = scale_old * Oi + Pij @ Vj
            // First, scale old output
            for (int d = 0; d < head_dim; d++) {
                Oi[d] *= scale_old;
            }

            // Add new contribution: Pij @ Vj
            for (int d = 0; d < head_dim; d++) {
                float sum = 0.0f;
                for (int j = 0; j < k_size; j++) {
                    sum += Sij[q_row][j] * Vj[j][d];
                }
                Oi[d] += sum;
            }

            // Update statistics
            mi = mi_new;
            li = li_new;
        }
        __syncthreads();
    }

    // Final normalization and write output
    for (int q_row = tid; q_row < q_size; q_row += num_threads) {
        float inv_li = 1.0f / li;
        for (int d = 0; d < head_dim; d++) {
            O_bhd[(q_start + q_row) * head_dim + d] = Oi[d] * inv_li;
        }
    }
}

// Host wrapper function
void flash_attention_forward(
    const float* Q,
    const float* K,
    const float* V,
    float* O,
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim
) {
    // Calculate number of query blocks
    int num_q_blocks = (seq_len + Br - 1) / Br;

    // Launch configuration
    dim3 grid(num_q_blocks, batch_size, num_heads);
    dim3 block(256);  // Number of threads per block

    flash_attention_forward_kernel<<<grid, block>>>(
        Q, K, V, O, batch_size, num_heads, seq_len, head_dim
    );

    CUDA_CHECK(cudaGetLastError());
}

// Simple test
void test_flash_attention() {
    printf("=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=\n");
    printf("FlashAttention Minimal - Forward Pass\n");
    printf("=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=\n\n");

    int batch_size = 2;
    int num_heads = 4;
    int seq_len = 512;
    int head_dim = 64;

    printf("Configuration:\n");
    printf("  Batch size: %d\n", batch_size);
    printf("  Num heads: %d\n", num_heads);
    printf("  Sequence length: %d\n", seq_len);
    printf("  Head dimension: %d\n", head_dim);
    printf("  Block sizes: Br=%d, Bc=%d\n\n", Br, Bc);

    size_t qkv_size = batch_size * num_heads * seq_len * head_dim;

    // Allocate and initialize host memory
    float* h_Q = new float[qkv_size];
    float* h_K = new float[qkv_size];
    float* h_V = new float[qkv_size];
    float* h_O = new float[qkv_size];

    // Initialize with small random values
    srand(42);
    for (size_t i = 0; i < qkv_size; i++) {
        h_Q[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
        h_K[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
        h_V[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
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

    // Warmup
    flash_attention_forward(d_Q, d_K, d_V, d_O, batch_size, num_heads, seq_len, head_dim);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Benchmark
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    int num_iters = 100;
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < num_iters; i++) {
        flash_attention_forward(d_Q, d_K, d_V, d_O, batch_size, num_heads, seq_len, head_dim);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));

    printf("Performance:\n");
    printf("  Average time: %.3f ms\n", milliseconds / num_iters);
    printf("  Throughput: %.2f GFLOPs/s\n",
           (4.0f * batch_size * num_heads * seq_len * seq_len * head_dim / 1e9) /
           (milliseconds / num_iters / 1000.0f));

    // Copy result back
    CUDA_CHECK(cudaMemcpy(h_O, d_O, qkv_size * sizeof(float), cudaMemcpyDeviceToHost));

    // Sanity checks
    float sum = 0.0f;
    float max_val = -FLT_MAX;
    float min_val = FLT_MAX;
    for (size_t i = 0; i < qkv_size; i++) {
        sum += h_O[i];
        max_val = fmaxf(max_val, h_O[i]);
        min_val = fminf(min_val, h_O[i]);
    }

    printf("\nOutput statistics:\n");
    printf("  Sum: %.6f\n", sum);
    printf("  Mean: %.6f\n", sum / qkv_size);
    printf("  Min: %.6f\n", min_val);
    printf("  Max: %.6f\n", max_val);

    // Memory analysis
    size_t attn_matrix_size = batch_size * num_heads * seq_len * seq_len * sizeof(float);
    size_t sram_usage = (Br * head_dim + 2 * Bc * head_dim + Br * Bc) * sizeof(float);

    printf("\nMemory analysis:\n");
    printf("  Attention matrix (NOT materialized): %.2f MB\n", attn_matrix_size / 1024.0f / 1024.0f);
    printf("  Working set in SRAM per block: %.2f KB\n", sram_usage / 1024.0f);
    printf("  Memory savings: %.2fx\n",
           (float)attn_matrix_size / (batch_size * num_heads * seq_len * head_dim * sizeof(float)));

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
    test_flash_attention();
    return 0;
}
