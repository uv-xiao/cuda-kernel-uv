# Exercise 2: Custom Attention Pattern

## Objective

Implement a custom attention mechanism with a **sliding window** pattern using TileLang. This exercises your understanding of tiling, memory management, and attention computation.

## Background

While standard attention has quadratic complexity O(N²), many applications use structured attention patterns:

1. **Sliding Window**: Each query only attends to nearby keys (local context)
2. **Block Diagonal**: Attention within blocks only
3. **Dilated**: Skip connections with gaps

Sliding window attention is particularly useful for:
- Long sequence processing (10K+ tokens)
- Streaming applications
- Hierarchical models

## Problem Statement

Implement sliding window attention where each query position i attends only to keys in range [i - window_size, i + window_size].

### Attention Formula

```
For each query position i:
  scores[j] = Q[i] @ K[j]^T  for j in [i - W, i + W]
  attention[j] = softmax(scores)[j]
  output[i] = sum(attention[j] * V[j])  for j in [i - W, i + W]
```

Where W is the window size.

### Requirements

1. **Implement sliding window attention**:
   - Window size: configurable (e.g., 128)
   - Handle boundary conditions (beginning/end of sequence)
   - Tiled implementation using shared memory

2. **Optimization goals**:
   - Load Q, K, V tiles into shared memory
   - Compute attention scores efficiently
   - Use online softmax for numerical stability
   - Achieve 2-3× speedup over full attention for long sequences

3. **Constraints**:
   - Sequence length: 2048
   - Head dimension: 64
   - Window size: 128
   - Block size: 64

## Starter Code Structure

```python
@T.prim_func
def sliding_window_attention(
    Q: T.Buffer((2048, 64), "float16"),
    K: T.Buffer((2048, 64), "float16"),
    V: T.Buffer((2048, 64), "float16"),
    O: T.Buffer((2048, 64), "float32"),
    window_size: T.int32
):
    """
    Sliding window attention.

    For each query position i:
    - Attend only to keys in [i - window_size, i + window_size]
    - Compute softmax over this window
    - Accumulate output from values in window
    """
    SEQ_LEN = 2048
    HEAD_DIM = 64
    BLOCK_M = 64  # Query tile size

    with T.block("root"):
        # TODO: Implement sliding window attention
        pass
```

## Implementation Hints

### 1. Determine Window Boundaries

```python
# For query block at position block_m
q_start = block_m * BLOCK_M
q_end = (block_m + 1) * BLOCK_M

# Window boundaries for this block
kv_start = T.max(0, q_start - window_size)
kv_end = T.min(SEQ_LEN, q_end + window_size)
```

### 2. Load Only Relevant K/V Tiles

```python
# Only process K/V tiles within the window
for block_n in T.serial((kv_end - kv_start) // BLOCK_N):
    kv_pos = kv_start + block_n * BLOCK_N

    # Check if this K/V tile overlaps with window
    # Load and process only if it does
```

### 3. Masking for Partial Windows

```python
# For each query in the block
for i in T.serial(BLOCK_M):
    q_pos = q_start + i

    # For each key in the tile
    for j in T.serial(BLOCK_N):
        k_pos = kv_pos + j

        # Check if within window
        if abs(k_pos - q_pos) <= window_size:
            # Valid attention position
            score = compute_score(...)
        else:
            # Masked position
            score = -1e10  # Will be zero after softmax
```

### 4. Online Softmax

Use the same online softmax trick as FlashAttention:

```python
# Update statistics
m_new = max(m_prev, max(scores))
l_new = exp(m_prev - m_new) * l_prev + sum(exp(scores - m_new))

# Rescale previous output
O = O * exp(m_prev - m_new) * (l_prev / l_new)

# Add new contribution
O = O + sum(exp(scores - m_new) / l_new * V)
```

## Test Cases

```python
# Test 1: Verify window constraint
Q, K, V = create_test_data(seq_len=256, dim=64)
O = sliding_window_attention(Q, K, V, window_size=32)

# Check: each position should only use nearby keys
attn_weights = compute_attention_weights(Q, K, O)
for i in range(256):
    for j in range(256):
        if abs(i - j) > 32:
            assert attn_weights[i, j] < 1e-5

# Test 2: Numerical correctness (within window)
# Compare with reference implementation
assert torch.allclose(O, expected, rtol=1e-2)

# Test 3: Memory efficiency
# Should use O(N * W) memory, not O(N²)
```

## Performance Target

On RTX 3090 / A100:

| Sequence Length | Full Attention | Sliding Window (W=128) | Speedup |
|-----------------|----------------|------------------------|---------|
| 512 | 0.5 ms | 0.3 ms | 1.7× |
| 1024 | 2.0 ms | 0.8 ms | 2.5× |
| 2048 | 8.0 ms | 2.5 ms | 3.2× |

Memory usage should be O(N * W) instead of O(N²).

## Bonus Challenges

1. **Causal Sliding Window**: Combine with causal masking (future positions masked)
2. **Dilated Windows**: Attend to positions i ± W, i ± 2W, i ± 4W, etc.
3. **Global + Local**: Combine global attention (few positions) with local window
4. **Adaptive Windows**: Different window sizes for different positions

## Visualization

```
Full Attention (N=8):
Q0 → [K0 K1 K2 K3 K4 K5 K6 K7]  (all keys)
Q1 → [K0 K1 K2 K3 K4 K5 K6 K7]
Q2 → [K0 K1 K2 K3 K4 K5 K6 K7]
...

Sliding Window (W=2):
Q0 → [K0 K1 K2 -- -- -- -- --]  (only nearby)
Q1 → [K0 K1 K2 K3 -- -- -- --]
Q2 → [K0 K1 K2 K3 K4 -- -- --]
Q3 → [-- K1 K2 K3 K4 K5 -- --]
Q4 → [-- -- K2 K3 K4 K5 K6 --]
...
```

## Learning Objectives

After completing this exercise, you should understand:
- How to implement custom attention patterns
- Memory-efficient attention for long sequences
- Masking strategies and boundary handling
- Trade-offs between expressiveness and efficiency

## Submission

Include:
1. `solution.py` - Your implementation
2. `test.py` - Correctness and performance tests
3. Comments explaining your approach
4. Analysis of complexity and memory usage

Good luck!
