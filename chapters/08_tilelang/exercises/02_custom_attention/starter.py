"""
Exercise 2: Sliding Window Attention - Starter Code
====================================================

Implement sliding window attention where each query only attends to
nearby keys within a specified window.

TODO: Complete the implementation following problem.md
"""

import tilelang as T
import torch
import math


@T.prim_func
def sliding_window_attention(
    Q: T.Buffer((2048, 64), "float16"),
    K: T.Buffer((2048, 64), "float16"),
    V: T.Buffer((2048, 64), "float16"),
    O: T.Buffer((2048, 64), "float32"),
    window_size: T.int32  # e.g., 128
):
    """
    Sliding window attention.

    TODO: Implement attention where each query position i only attends
    to keys in range [i - window_size, i + window_size].
    """
    SEQ_LEN = 2048
    HEAD_DIM = 64
    BLOCK_M = 64  # Query tile size
    BLOCK_N = 64  # Key/Value tile size
    SCALE = 1.0 / math.sqrt(HEAD_DIM)

    with T.block("root"):
        # TODO: Get block index for query tiles
        # block_m = T.thread_binding(...)

        # TODO: Allocate shared memory for Q, K, V tiles
        # Q_shared = T.alloc_shared([BLOCK_M, HEAD_DIM], "float16")
        # K_shared = T.alloc_shared([BLOCK_N, HEAD_DIM], "float16")
        # V_shared = T.alloc_shared([BLOCK_N, HEAD_DIM], "float16")

        # TODO: Allocate output accumulator and softmax stats
        # O_local = T.alloc_fragment([BLOCK_M, HEAD_DIM], "float32")
        # m_stat = T.alloc_fragment([BLOCK_M], "float32")  # Max scores
        # l_stat = T.alloc_fragment([BLOCK_M], "float32")  # Sum exp

        # TODO: Initialize
        # T.fill(O_local, 0.0)
        # T.fill(m_stat, -1e10)
        # T.fill(l_stat, 0.0)

        # TODO: Load Q tile
        # T.copy(Q[block_m * BLOCK_M : (block_m + 1) * BLOCK_M, :], Q_shared)
        # T.sync_threads()

        # TODO: Determine window boundaries for this query block
        # q_start = block_m * BLOCK_M
        # q_end = (block_m + 1) * BLOCK_M
        # kv_start = max(0, q_start - window_size)
        # kv_end = min(SEQ_LEN, q_end + window_size)

        # TODO: Loop over K/V tiles within window
        # num_kv_tiles = (kv_end - kv_start + BLOCK_N - 1) // BLOCK_N
        # for tile in T.serial(num_kv_tiles):
        #     kv_pos = kv_start + tile * BLOCK_N
        #
        #     # Load K, V tiles
        #     # Compute attention scores with masking
        #     # Apply online softmax
        #     # Accumulate output

        # TODO: Write output
        # T.copy(O_local, O[block_m * BLOCK_M : (block_m + 1) * BLOCK_M, :])

        pass  # Remove when implementing


def reference_sliding_window_attention(Q, K, V, window_size):
    """
    Reference implementation using PyTorch for testing.
    """
    seq_len = Q.shape[0]
    scale = 1.0 / math.sqrt(Q.shape[-1])

    # Compute full attention matrix
    scores = torch.matmul(Q, K.transpose(-1, -2)) * scale

    # Create sliding window mask
    mask = torch.zeros(seq_len, seq_len, device=Q.device)
    for i in range(seq_len):
        start = max(0, i - window_size)
        end = min(seq_len, i + window_size + 1)
        mask[i, start:end] = 1.0

    # Apply mask (set non-window positions to -inf)
    scores = scores.masked_fill(mask == 0, -1e10)

    # Softmax and output
    attn = torch.softmax(scores, dim=-1)
    output = torch.matmul(attn, V)

    return output


def test_sliding_window():
    """Test sliding window attention."""
    print("="*60)
    print("Testing Sliding Window Attention")
    print("="*60)

    SEQ_LEN = 512  # Smaller for testing
    HEAD_DIM = 64
    WINDOW_SIZE = 64

    Q = torch.randn(SEQ_LEN, HEAD_DIM, device="cuda", dtype=torch.float16)
    K = torch.randn(SEQ_LEN, HEAD_DIM, device="cuda", dtype=torch.float16)
    V = torch.randn(SEQ_LEN, HEAD_DIM, device="cuda", dtype=torch.float16)
    O = torch.zeros(SEQ_LEN, HEAD_DIM, device="cuda", dtype=torch.float32)

    # Reference
    expected = reference_sliding_window_attention(Q, K, V, WINDOW_SIZE)

    # TODO: Compile and run your implementation
    # mod = T.compile(sliding_window_attention, target="cuda")
    # mod(Q, K, V, O, WINDOW_SIZE)

    # TODO: Compare results
    # max_diff = (O - expected.float()).abs().max().item()
    # print(f"Max difference: {max_diff:.6f}")
    # assert max_diff < 0.1, "Test failed!"

    print("\nTODO: Implement the kernel!")
    print("See problem.md for requirements.")


def main():
    """Main function."""
    test_sliding_window()


if __name__ == "__main__":
    main()
