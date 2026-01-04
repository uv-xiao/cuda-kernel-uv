"""
Simplified Fused Attention

Attention: softmax(Q @ K^T) @ V

This demonstrates fusing matrix multiply with softmax,
inspired by Flash Attention but simplified for learning.
"""

import torch
import triton
import triton.language as tl
import time


@triton.jit
def fused_attention_kernel(
    Q, K, V, Out,
    stride_qm, stride_qk,
    stride_kn, stride_kk,
    stride_vm, stride_vk,
    stride_om, stride_ok,
    M, N, K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Simplified fused attention kernel.
    Fuses: matmul(Q, K^T) + softmax + matmul(scores, V)
    """
    pid_m = tl.program_id(0)

    # Offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)

    # Initialize output accumulator
    acc = tl.zeros((BLOCK_M, K), dtype=tl.float32)

    # Loop over N dimension
    for n in range(0, N, BLOCK_N):
        # Load Q block (BLOCK_M x K)
        q = tl.load(Q + offs_m[:, None] * stride_qm + tl.arange(0, K)[None, :] * stride_qk,
                    mask=(offs_m[:, None] < M), other=0.0)

        # Load K block (BLOCK_N x K)
        offs_n_cur = n + offs_n
        k = tl.load(K + offs_n_cur[:, None] * stride_kn + tl.arange(0, K)[None, :] * stride_kk,
                    mask=(offs_n_cur[:, None] < N), other=0.0)

        # Compute scores: Q @ K^T (BLOCK_M x BLOCK_N)
        scores = tl.dot(q, tl.trans(k))

        # Softmax (simplified - per row)
        scores_max = tl.max(scores, axis=1)[:, None]
        scores_exp = tl.exp(scores - scores_max)
        scores_sum = tl.sum(scores_exp, axis=1)[:, None]
        probs = scores_exp / scores_sum

        # Load V block (BLOCK_N x K)
        v = tl.load(V + offs_n_cur[:, None] * stride_vm + tl.arange(0, K)[None, :] * stride_vk,
                    mask=(offs_n_cur[:, None] < N), other=0.0)

        # Accumulate: probs @ V
        acc += tl.dot(probs.to(tl.float16), v.to(tl.float16))

    # Store output
    tl.store(Out + offs_m[:, None] * stride_om + tl.arange(0, K)[None, :] * stride_ok,
             acc.to(tl.float16), mask=(offs_m[:, None] < M))


def triton_attention(Q, K, V):
    """Simplified fused attention."""
    M, K_dim = Q.shape
    N, _ = K.shape

    Out = torch.empty((M, K_dim), device=Q.device, dtype=Q.dtype)

    BLOCK_M = 64
    BLOCK_N = 64

    grid = (triton.cdiv(M, BLOCK_M),)

    fused_attention_kernel[grid](
        Q, K, V, Out,
        Q.stride(0), Q.stride(1),
        K.stride(0), K.stride(1),
        V.stride(0), V.stride(1),
        Out.stride(0), Out.stride(1),
        M, N, K_dim,
        BLOCK_M, BLOCK_N,
    )

    return Out


def pytorch_attention(Q, K, V):
    """Reference implementation."""
    scores = Q @ K.T
    probs = torch.softmax(scores, dim=-1)
    output = probs @ V
    return output


def test_correctness():
    """Test correctness."""
    print("\nTesting Correctness")
    print("=" * 70)

    test_cases = [
        (64, 64, 32),
        (128, 128, 64),
        (256, 256, 64),
    ]

    for M, N, K in test_cases:
        Q = torch.randn(M, K, device='cuda', dtype=torch.float16)
        K_mat = torch.randn(N, K, device='cuda', dtype=torch.float16)
        V = torch.randn(N, K, device='cuda', dtype=torch.float16)

        result_triton = triton_attention(Q, K_mat, V)
        result_torch = pytorch_attention(Q, K_mat, V)

        assert torch.allclose(result_triton, result_torch, rtol=1e-2, atol=1e-2), \
            f"Mismatch at shape M={M}, N={N}, K={K}"

        print(f"  Shape (M={M:3d}, N={N:3d}, K={K:2d}): PASS")

    print("All tests passed!")


if __name__ == "__main__":
    print("Simplified Fused Attention")
    print("=" * 70)

    if not torch.cuda.is_available():
        print("CUDA not available. This example requires a GPU.")
        exit(1)

    test_correctness()

    print("\nNote: This is a simplified version for learning.")
    print("Production Flash Attention is more complex and handles:")
    print("  - Causal masking")
    print("  - Multi-head attention")
    print("  - Better memory tiling")
    print("  - Backward pass optimization")
