"""
Task 2: Implement Mini FlashAttention in Triton

Implement the online softmax FlashAttention algorithm using Triton.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def flash_attention_kernel(
    # Pointers to tensors
    Q, K, V, Out,
    # Strides for Q [batch, seq_q, num_heads, head_dim]
    stride_qb, stride_qq, stride_qh, stride_qd,
    # Strides for K [batch, seq_kv, num_kv_heads, head_dim]
    stride_kb, stride_kk, stride_kh, stride_kd,
    # Strides for V [batch, seq_kv, num_kv_heads, head_dim]
    stride_vb, stride_vk, stride_vh, stride_vd,
    # Strides for Out [batch, seq_q, num_heads, head_dim]
    stride_ob, stride_oq, stride_oh, stride_od,
    # Dimensions
    seq_q, seq_kv,
    num_heads, num_kv_heads,
    # Attention scale (1 / sqrt(head_dim))
    scale,
    # Causal masking flag
    causal: tl.constexpr,
    # Block sizes
    BLOCK_Q: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    """
    FlashAttention kernel with online softmax.

    TODO: Implement the kernel following these steps:

    1. Get program IDs for batch, head, and q_tile
       - batch_idx = tl.program_id(0)
       - head_idx = tl.program_id(1)
       - q_tile_idx = tl.program_id(2)

    2. Compute KV head index for GQA
       - kv_head_idx = head_idx // (num_heads // num_kv_heads)

    3. Compute pointers to Q, K, V, Out for this thread block

    4. Load Q tile (stays in registers, reused across KV tiles)

    5. Initialize online softmax state:
       - m = -infinity (per-row max)
       - l = 0 (per-row sum of exp)
       - O_acc = 0 (output accumulator)

    6. Loop over KV tiles:
       a. Load K tile
       b. Compute S = Q @ K.T * scale
       c. Apply causal mask if needed
       d. Update online softmax: m_new, l_new
       e. Load V tile
       f. Update O_acc with rescaling
       g. Update m, l

    7. Final normalization: O = O_acc / l

    8. Store output
    """
    # TODO: Your implementation here
    pass


def run(q, k, v, causal=True):
    """
    Launch the FlashAttention kernel.

    Args:
        q: Query tensor [batch, seq_q, num_heads, head_dim]
        k: Key tensor [batch, seq_kv, num_kv_heads, head_dim]
        v: Value tensor [batch, seq_kv, num_kv_heads, head_dim]
        causal: Whether to apply causal masking

    Returns:
        out: Output tensor [batch, seq_q, num_heads, head_dim]
    """
    batch, seq_q, num_heads, head_dim = q.shape
    _, seq_kv, num_kv_heads, _ = k.shape

    # Allocate output
    out = torch.empty_like(q)

    # Block sizes (tune these for your GPU)
    BLOCK_Q = 64
    BLOCK_KV = 64

    # Compute grid dimensions
    num_q_tiles = triton.cdiv(seq_q, BLOCK_Q)
    grid = (batch, num_heads, num_q_tiles)

    # Attention scale
    scale = 1.0 / (head_dim ** 0.5)

    # TODO: Launch kernel
    # flash_attention_kernel[grid](
    #     q, k, v, out,
    #     q.stride(0), q.stride(1), q.stride(2), q.stride(3),
    #     k.stride(0), k.stride(1), k.stride(2), k.stride(3),
    #     v.stride(0), v.stride(1), v.stride(2), v.stride(3),
    #     out.stride(0), out.stride(1), out.stride(2), out.stride(3),
    #     seq_q, seq_kv,
    #     num_heads, num_kv_heads,
    #     scale,
    #     causal,
    #     BLOCK_Q=BLOCK_Q,
    #     BLOCK_KV=BLOCK_KV,
    #     HEAD_DIM=head_dim,
    # )

    return out


if __name__ == "__main__":
    # Test the implementation
    torch.manual_seed(42)

    batch, seq_q, seq_kv = 2, 256, 256
    num_heads, num_kv_heads, head_dim = 32, 8, 128

    q = torch.randn(batch, seq_q, num_heads, head_dim, dtype=torch.float16, device="cuda")
    k = torch.randn(batch, seq_kv, num_kv_heads, head_dim, dtype=torch.float16, device="cuda")
    v = torch.randn(batch, seq_kv, num_kv_heads, head_dim, dtype=torch.float16, device="cuda")

    # Run kernel
    out = run(q, k, v, causal=True)

    print(f"Output shape: {out.shape}")
    print(f"Output dtype: {out.dtype}")
