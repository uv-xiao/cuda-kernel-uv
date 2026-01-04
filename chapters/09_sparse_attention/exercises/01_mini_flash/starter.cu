// Mini FlashAttention - Starter Code
// Fill in the TODOs to implement FlashAttention forward pass

#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>
#include <float.h>

#define Br 32
#define Bc 32
#define HEAD_DIM 64

__global__ void mini_flash_attention_kernel(
    const float* Q,
    const float* K,
    const float* V,
    float* O,
    int seq_len,
    int head_dim,
    float scale
) {
    // Block processes one query block
    int q_block_idx = blockIdx.x;
    int q_start = q_block_idx * Br;
    int q_end = min(q_start + Br, seq_len);
    int q_size = q_end - q_start;

    if (q_start >= seq_len) return;

    int tid = threadIdx.x;

    // TODO: Allocate shared memory for Qi, Kj, Vj, Sij
    __shared__ float Qi[Br][HEAD_DIM];
    __shared__ float Kj[Bc][HEAD_DIM];
    __shared__ float Vj[Bc][HEAD_DIM];
    __shared__ float Sij[Br][Bc];

    // TODO: Load query block Qi into shared memory
    // Hint: Use cooperative loading with all threads


    // TODO: Initialize output accumulator in registers
    // Each thread should maintain:
    // - Oi[HEAD_DIM]: output values
    // - mi: running max
    // - li: running normalizer


    // TODO: Loop over key/value blocks
    int num_kv_blocks = (seq_len + Bc - 1) / Bc;

    for (int kv_block = 0; kv_block < num_kv_blocks; kv_block++) {
        int k_start = kv_block * Bc;
        int k_end = min(k_start + Bc, seq_len);
        int k_size = k_end - k_start;

        // TODO: Load Kj and Vj blocks


        // TODO: Compute Sij = Qi @ Kj^T
        // Hint: Each thread computes subset of elements


        // TODO: Compute row max of Sij
        // For each row i, find max_j(Sij[i][j])


        // TODO: Update global max mi_new = max(mi, row_max)


        // TODO: Compute Pij = exp(Sij - mi_new)


        // TODO: Compute row sum of Pij


        // TODO: Update normalizer li
        // li_new = exp(mi - mi_new) * li + row_sum


        // TODO: Rescale old output Oi
        // Oi = exp(mi - mi_new) * Oi


        // TODO: Add new contribution: Oi += Pij @ Vj


        // TODO: Update mi and li

    }

    // TODO: Final normalization: Oi = Oi / li


    // TODO: Write output to global memory

}

// Host wrapper
void mini_flash_attention(
    const float* Q,
    const float* K,
    const float* V,
    float* O,
    int seq_len,
    int head_dim
) {
    float scale = 1.0f / sqrtf((float)head_dim);
    int num_q_blocks = (seq_len + Br - 1) / Br;

    dim3 grid(num_q_blocks);
    dim3 block(256);

    mini_flash_attention_kernel<<<grid, block>>>(
        Q, K, V, O, seq_len, head_dim, scale
    );
}

int main() {
    printf("Implement the TODOs in the kernel!\n");
    return 0;
}
