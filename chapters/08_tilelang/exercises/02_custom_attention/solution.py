"""
Exercise 2: Sliding Window Attention - Solution
================================================

Complete implementation of sliding window attention using TileLang.
"""

import tilelang as T
import torch
import math
import time


@T.prim_func
def sliding_window_attention(
    Q: T.Buffer((2048, 64), "float16"),
    K: T.Buffer((2048, 64), "float16"),
    V: T.Buffer((2048, 64), "float16"),
    O: T.Buffer((2048, 64), "float32"),
    window_size: T.int32
):
    """
    Sliding window attention implementation.

    Each query position i attends only to keys in [i-W, i+W].
    """
    SEQ_LEN = 2048
    HEAD_DIM = 64
    BLOCK_M = 64
    BLOCK_N = 64
    SCALE = 1.0 / math.sqrt(HEAD_DIM)

    with T.block("root"):
        block_m = T.thread_binding(0, SEQ_LEN // BLOCK_M, "blockIdx.x")

        # Shared memory
        Q_shared = T.alloc_shared([BLOCK_M, HEAD_DIM], "float16")
        K_shared = T.alloc_shared([BLOCK_N, HEAD_DIM], "float16")
        V_shared = T.alloc_shared([BLOCK_N, HEAD_DIM], "float16")

        # Output and statistics
        O_local = T.alloc_fragment([BLOCK_M, HEAD_DIM], "float32")
        m_stat = T.alloc_fragment([BLOCK_M], "float32")
        l_stat = T.alloc_fragment([BLOCK_M], "float32")

        # Initialize
        T.fill(O_local, 0.0)
        T.fill(m_stat, -1e10)
        T.fill(l_stat, 0.0)

        # Load Q tile
        T.copy(Q[block_m * BLOCK_M : (block_m + 1) * BLOCK_M, :], Q_shared)
        T.sync_threads()

        # Determine window boundaries
        q_start = block_m * BLOCK_M
        q_end = (block_m + 1) * BLOCK_M

        # Conservative window boundaries for this block
        kv_start = T.max(0, q_start - window_size)
        kv_end = T.min(SEQ_LEN, q_end + window_size)

        # Number of K/V tiles to process
        num_kv_tiles = (kv_end - kv_start + BLOCK_N - 1) // BLOCK_N

        # Process K/V tiles within window
        for tile_idx in T.serial(num_kv_tiles):
            kv_pos = kv_start + tile_idx * BLOCK_N

            # Load K and V tiles
            tile_start = kv_pos
            tile_end = T.min(kv_pos + BLOCK_N, kv_end)

            T.copy(K[tile_start:tile_end, :], K_shared)
            T.copy(V[tile_start:tile_end, :], V_shared)
            T.sync_threads()

            # Compute attention scores with masking
            scores = T.alloc_fragment([BLOCK_M, BLOCK_N], "float32")

            for i in T.serial(BLOCK_M):
                q_pos = q_start + i

                for j in T.serial(BLOCK_N):
                    k_pos = kv_pos + j

                    # Check if within window and valid position
                    in_window = (k_pos >= T.max(0, q_pos - window_size)) and \
                                (k_pos <= T.min(SEQ_LEN - 1, q_pos + window_size)) and \
                                (k_pos < SEQ_LEN)

                    if in_window:
                        # Compute score: Q[i] @ K[j]
                        score = 0.0
                        for d in T.serial(HEAD_DIM):
                            score += T.cast(Q_shared[i, d], "float32") * \
                                     T.cast(K_shared[j, d], "float32")
                        scores[i, j] = score * SCALE
                    else:
                        # Masked position
                        scores[i, j] = -1e10

            # Online softmax update
            m_new = T.alloc_fragment([BLOCK_M], "float32")
            l_new = T.alloc_fragment([BLOCK_M], "float32")

            for i in T.serial(BLOCK_M):
                # Compute new max
                m_new[i] = m_stat[i]
                for j in T.serial(BLOCK_N):
                    m_new[i] = T.max(m_new[i], scores[i, j])

                # Compute new sum
                alpha = T.exp(m_stat[i] - m_new[i])
                l_new[i] = alpha * l_stat[i]

                for j in T.serial(BLOCK_N):
                    l_new[i] += T.exp(scores[i, j] - m_new[i])

                # Rescale previous output
                scale_factor = alpha * (l_stat[i] / l_new[i]) if l_new[i] > 0 else 0.0
                for d in T.serial(HEAD_DIM):
                    O_local[i, d] *= scale_factor

                # Add new contribution
                for d in T.serial(HEAD_DIM):
                    for j in T.serial(BLOCK_N):
                        p_val = T.exp(scores[i, j] - m_new[i]) / l_new[i] if l_new[i] > 0 else 0.0
                        O_local[i, d] += p_val * T.cast(V_shared[j, d], "float32")

                # Update statistics
                m_stat[i] = m_new[i]
                l_stat[i] = l_new[i]

            T.sync_threads()

        # Write output
        T.copy(O_local, O[block_m * BLOCK_M : (block_m + 1) * BLOCK_M, :])


def reference_sliding_window_attention(Q, K, V, window_size):
    """Reference implementation."""
    seq_len = Q.shape[0]
    scale = 1.0 / math.sqrt(Q.shape[-1])

    scores = torch.matmul(Q, K.transpose(-1, -2)) * scale

    # Create sliding window mask
    mask = torch.zeros(seq_len, seq_len, device=Q.device)
    for i in range(seq_len):
        start = max(0, i - window_size)
        end = min(seq_len, i + window_size + 1)
        mask[i, start:end] = 1.0

    scores = scores.masked_fill(mask == 0, -1e10)
    attn = torch.softmax(scores, dim=-1)
    output = torch.matmul(attn, V)

    return output


def test_correctness():
    """Test correctness."""
    print("="*60)
    print("Correctness Tests")
    print("="*60)

    test_configs = [
        (512, 64, 32),
        (512, 64, 64),
        (1024, 64, 128),
    ]

    for seq_len, head_dim, window_size in test_configs:
        print(f"\nTesting: seq_len={seq_len}, window={window_size}")

        Q = torch.randn(seq_len, head_dim, device="cuda", dtype=torch.float16)
        K = torch.randn(seq_len, head_dim, device="cuda", dtype=torch.float16)
        V = torch.randn(seq_len, head_dim, device="cuda", dtype=torch.float16)
        O = torch.zeros(seq_len, head_dim, device="cuda", dtype=torch.float32)

        # Reference
        expected = reference_sliding_window_attention(Q, K, V, window_size)

        # Note: Would need to compile with dynamic shapes or adjust kernel
        # For demonstration, assume seq_len matches kernel size
        if seq_len <= 2048:
            # Pad to 2048 if needed for kernel
            Q_padded = torch.zeros(2048, head_dim, device="cuda", dtype=torch.float16)
            K_padded = torch.zeros(2048, head_dim, device="cuda", dtype=torch.float16)
            V_padded = torch.zeros(2048, head_dim, device="cuda", dtype=torch.float16)
            O_padded = torch.zeros(2048, head_dim, device="cuda", dtype=torch.float32)

            Q_padded[:seq_len] = Q
            K_padded[:seq_len] = K
            V_padded[:seq_len] = V

            mod = T.compile(sliding_window_attention, target="cuda")
            mod(Q_padded, K_padded, V_padded, O_padded, window_size)

            O = O_padded[:seq_len]

        # Compare
        max_diff = (O - expected.float()).abs().max().item()
        avg_diff = (O - expected.float()).abs().mean().item()

        print(f"  Max difference: {max_diff:.6f}")
        print(f"  Avg difference: {avg_diff:.6f}")

        assert max_diff < 0.5, f"Test failed! Max diff: {max_diff}"
        print(f"  ✓ Passed")


def benchmark_performance():
    """Benchmark performance."""
    print("\n" + "="*60)
    print("Performance Benchmark")
    print("="*60)

    configs = [
        (512, 64),
        (1024, 128),
        (2048, 128),
    ]

    for seq_len, window_size in configs:
        print(f"\nSeq Length: {seq_len}, Window: {window_size}")
        print("-" * 60)

        head_dim = 64
        Q = torch.randn(seq_len, head_dim, device="cuda", dtype=torch.float16)
        K = torch.randn(seq_len, head_dim, device="cuda", dtype=torch.float16)
        V = torch.randn(seq_len, head_dim, device="cuda", dtype=torch.float16)

        # Pad for kernel if needed
        Q_padded = torch.zeros(2048, head_dim, device="cuda", dtype=torch.float16)
        K_padded = torch.zeros(2048, head_dim, device="cuda", dtype=torch.float16)
        V_padded = torch.zeros(2048, head_dim, device="cuda", dtype=torch.float16)
        O_padded = torch.zeros(2048, head_dim, device="cuda", dtype=torch.float32)

        Q_padded[:seq_len] = Q
        K_padded[:seq_len] = K
        V_padded[:seq_len] = V

        mod = T.compile(sliding_window_attention, target="cuda")

        # Warmup
        for _ in range(10):
            _ = reference_sliding_window_attention(Q, K, V, window_size)
            mod(Q_padded, K_padded, V_padded, O_padded, window_size)
        torch.cuda.synchronize()

        # Benchmark reference (full attention)
        start = time.time()
        for _ in range(100):
            _ = reference_sliding_window_attention(Q, K, V, window_size)
        torch.cuda.synchronize()
        ref_time = (time.time() - start) / 100

        # Benchmark sliding window
        start = time.time()
        for _ in range(100):
            mod(Q_padded, K_padded, V_padded, O_padded, window_size)
        torch.cuda.synchronize()
        sw_time = (time.time() - start) / 100

        print(f"Full Attention:     {ref_time*1e3:.3f} ms")
        print(f"Sliding Window:     {sw_time*1e3:.3f} ms")
        print(f"Speedup:            {ref_time/sw_time:.2f}×")
        print(f"Complexity:         O(N²) vs O(N×W)")
        print(f"Memory savings:     {seq_len / (2*window_size):.1f}×")


def main():
    """Run all tests."""
    print("="*60)
    print("Exercise 2: Sliding Window Attention - Solution")
    print("="*60)

    test_correctness()
    benchmark_performance()

    print("\n✓ All tests passed!")
    print("\nKey Insights:")
    print("1. Sliding window reduces complexity from O(N²) to O(N×W)")
    print("2. Masking ensures each query only sees nearby keys")
    print("3. Online softmax handles variable window sizes")
    print("4. Significant speedup for long sequences")


if __name__ == "__main__":
    main()
