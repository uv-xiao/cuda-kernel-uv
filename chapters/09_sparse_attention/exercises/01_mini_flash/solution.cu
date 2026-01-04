// Mini FlashAttention - Solution
// Complete implementation with detailed comments

#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>
#include <float.h>

#define Br 32
#define Bc 32
#define HEAD_DIM 64

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            printf("CUDA error: %s\n", cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while(0)

__global__ void mini_flash_attention_kernel(
    const float* Q,
    const float* K,
    const float* V,
    float* O,
    int seq_len,
    int head_dim,
    float scale
) {
    int q_block_idx = blockIdx.x;
    int q_start = q_block_idx * Br;
    int q_end = min(q_start + Br, seq_len);
    int q_size = q_end - q_start;

    if (q_start >= seq_len) return;

    int tid = threadIdx.x;
    int num_threads = blockDim.x;

    // Shared memory
    __shared__ float Qi[Br][HEAD_DIM];
    __shared__ float Kj[Bc][HEAD_DIM];
    __shared__ float Vj[Bc][HEAD_DIM];
    __shared__ float Sij[Br][Bc];

    // Load query block cooperatively
    for (int idx = tid; idx < q_size * head_dim; idx += num_threads) {
        int i = idx / head_dim;
        int d = idx % head_dim;
        Qi[i][d] = Q[(q_start + i) * head_dim + d];
    }
    __syncthreads();

    // Per-thread accumulators (each thread handles one query row)
    float Oi[HEAD_DIM];
    for (int d = 0; d < HEAD_DIM; d++) {
        Oi[d] = 0.0f;
    }
    float mi = -FLT_MAX;
    float li = 0.0f;

    int num_kv_blocks = (seq_len + Bc - 1) / Bc;

    for (int kv_block = 0; kv_block < num_kv_blocks; kv_block++) {
        int k_start = kv_block * Bc;
        int k_end = min(k_start + Bc, seq_len);
        int k_size = k_end - k_start;

        // Load K and V blocks
        for (int idx = tid; idx < k_size * head_dim; idx += num_threads) {
            int i = idx / head_dim;
            int d = idx % head_dim;
            Kj[i][d] = K[(k_start + i) * head_dim + d];
            Vj[i][d] = V[(k_start + i) * head_dim + d];
        }
        __syncthreads();

        // Compute Sij = Qi @ Kj^T
        for (int idx = tid; idx < q_size * k_size; idx += num_threads) {
            int i = idx / k_size;
            int j = idx % k_size;
            float sum = 0.0f;
            for (int d = 0; d < head_dim; d++) {
                sum += Qi[i][d] * Kj[j][d];
            }
            Sij[i][j] = sum * scale;
        }
        __syncthreads();

        // Each thread processes one query row
        for (int i = tid; i < q_size; i += num_threads) {
            // Compute row max
            float row_max = -FLT_MAX;
            for (int j = 0; j < k_size; j++) {
                row_max = fmaxf(row_max, Sij[i][j]);
            }

            // Update global max
            float mi_new = fmaxf(mi, row_max);

            // Compute exp(Sij - mi_new) and row sum
            float row_sum = 0.0f;
            for (int j = 0; j < k_size; j++) {
                Sij[i][j] = expf(Sij[i][j] - mi_new);
                row_sum += Sij[i][j];
            }

            // Update normalizer
            float scale_old = expf(mi - mi_new);
            float li_new = scale_old * li + row_sum;

            // Rescale old output
            for (int d = 0; d < HEAD_DIM; d++) {
                Oi[d] *= scale_old;
            }

            // Add new contribution
            for (int d = 0; d < HEAD_DIM; d++) {
                float sum = 0.0f;
                for (int j = 0; j < k_size; j++) {
                    sum += Sij[i][j] * Vj[j][d];
                }
                Oi[d] += sum;
            }

            mi = mi_new;
            li = li_new;
        }
        __syncthreads();
    }

    // Final normalization and write
    for (int i = tid; i < q_size; i += num_threads) {
        for (int d = 0; d < HEAD_DIM; d++) {
            O[(q_start + i) * head_dim + d] = Oi[d] / li;
        }
    }
}

void mini_flash_attention(const float* Q, const float* K, const float* V,
                         float* O, int seq_len, int head_dim) {
    float scale = 1.0f / sqrtf((float)head_dim);
    int num_q_blocks = (seq_len + Br - 1) / Br;
    mini_flash_attention_kernel<<<num_q_blocks, 256>>>(
        Q, K, V, O, seq_len, head_dim, scale);
    CUDA_CHECK(cudaGetLastError());
}

int main() {
    printf("Mini FlashAttention - Solution\n");
    printf("See implementation above!\n");
    return 0;
}
