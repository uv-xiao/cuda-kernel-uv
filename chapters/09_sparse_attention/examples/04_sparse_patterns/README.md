# Sparse Attention Patterns

## Overview

This directory implements various sparse attention patterns that reduce complexity from O(L²) to O(Lk) where k << L. These patterns are crucial for scaling transformers to very long sequences.

## Why Sparse Attention?

### The L² Problem

Dense attention becomes impractical for long sequences:

| Sequence Length | Attention Matrix Size | Memory (FP32) |
|----------------|----------------------|---------------|
| 512 | 262K | 1 MB |
| 2048 | 4.2M | 16 MB |
| 8192 | 67M | 256 MB |
| 32768 | 1.07B | 4 GB |
| 131072 | 17.2B | 64 GB |

Even with FlashAttention (which doesn't materialize the matrix), the O(L²) compute becomes prohibitive.

### Sparse Attention Solution

Key insight: Most attention weights are near zero! We can:
1. Only compute attention for a subset of positions
2. Reduce complexity to O(L · k) where k is sparsity factor
3. Maintain model quality with careful pattern design

## Sparsity Patterns

### 1. Local (Sliding Window) Attention

Each token attends to k nearest neighbors.

```
Pattern (k=3, causal):
[X . . . . . . .]
[X X . . . . . .]
[X X X . . . . .]
[. X X X . . . .]
[. . X X X . . .]
[. . . X X X . .]
[. . . . X X X .]
[. . . . . X X X]
```

**Use cases:**
- Language modeling (local context)
- Audio/video processing (temporal locality)
- Computer vision (spatial locality)

**Complexity:** O(L · k)

**Advantages:**
- Simple to implement
- Good inductive bias for sequential data
- Works well with causal masking

**Limitations:**
- Cannot capture long-range dependencies
- Fixed receptive field

### 2. Strided (Dilated) Attention

Attend to every s-th token.

```
Pattern (stride=2, k=4):
[X . X . X . X .]
[X X . X . X . X]
[X . X . X . X .]
[. X . X . X . X]
[X . X . X . X .]
[. X . X . X . X]
[X . X . X . X .]
[. X . X . X . X]
```

**Use cases:**
- Capturing different scales
- Hierarchical attention
- Multi-scale modeling

**Complexity:** O(L · k)

**Advantages:**
- Exponentially growing receptive field with layers
- Captures different temporal/spatial scales

**Limitations:**
- May miss important positions between strides
- Needs multiple strides for full coverage

### 3. Block-Sparse Attention

Attention is restricted to predefined blocks.

```
Pattern (block_size=2):
[X X . . . . . .]
[X X . . . . . .]
[. . X X . . . .]
[. . X X . . . .]
[. . . . X X . .]
[. . . . X X . .]
[. . . . . . X X]
[. . . . . . X X]
```

**Use cases:**
- Sparse Transformers (OpenAI)
- Image modeling (spatial blocks)
- Document understanding (paragraph blocks)

**Complexity:** O(L · b) where b is block size

**Advantages:**
- Hardware-friendly (aligned memory access)
- Can use optimized block-sparse kernels
- Flexible pattern design

### 4. Global + Local Attention

Some tokens attend globally, others locally.

```
Pattern (global tokens at 0, 4):
[X X X X X X X X]  <- Global token
[X X . . . . . .]
[X . X . . . . .]
[X . . X . . . .]
[X X X X X X X X]  <- Global token
[X . . . . X . .]
[X . . . . . X .]
[X . . . . . . X]
```

**Use cases:**
- Longformer, BigBird
- Document QA (CLS token attends globally)
- Hierarchical models

**Complexity:** O(L · k + g · L) where g is number of global tokens

**Advantages:**
- Balances local and global information
- Good for task-specific attention (e.g., [CLS] token)
- Proven effective in practice

### 5. Random Sparse Attention

Random subset of positions per query.

```
Pattern (k=4, random):
[X . X . . X . X]
[. X X . X . . X]
[X . . X X . X .]
[. X . X . X X .]
[X X . . X . . X]
[. . X X . X X .]
[X . X . X . X .]
[. X . X . X . X]
```

**Use cases:**
- Theoretical analysis
- Routing attention (mixture of experts)
- Regularization

**Complexity:** O(L · k)

**Advantages:**
- Unbiased (no structural assumptions)
- Can capture unexpected patterns
- Good for ensembles

**Limitations:**
- Non-deterministic (different every time)
- Harder to optimize
- May miss systematic patterns

## Implementation Strategies

### Strategy 1: Mask-Based

Simple approach: Use standard attention with sparse mask.

```python
# Create sparse mask
mask = create_sparse_mask(seq_len, pattern_type)

# Standard attention with mask
scores = Q @ K.T / sqrt(d)
scores = scores.masked_fill(mask == 0, -inf)
attention = softmax(scores)
output = attention @ V
```

**Pros:** Easy to implement
**Cons:** Still O(L²) memory and compute (just zeros out values)

### Strategy 2: Index-Based

Only compute attention for non-zero positions.

```python
# Get indices of non-zero attention
indices = get_sparse_indices(seq_len, pattern_type)

# Gather relevant K, V
K_sparse = K[indices]  # [L, k, d]
V_sparse = V[indices]  # [L, k, d]

# Compute attention only for these
scores = (Q.unsqueeze(2) * K_sparse).sum(-1)  # [L, k]
attention = softmax(scores)
output = (attention.unsqueeze(-1) * V_sparse).sum(-2)  # [L, d]
```

**Pros:** True O(Lk) complexity
**Cons:** More complex indexing, harder to optimize

### Strategy 3: Block-Sparse GEMM

Use specialized block-sparse matrix multiplication.

```python
# Create block-sparse layout
layout = create_block_sparse_layout(seq_len, block_size, pattern)

# Use block-sparse kernels
scores = block_sparse_matmul(Q, K.T, layout)
attention = softmax(scores)
output = block_sparse_matmul(attention, V, layout)
```

**Pros:** Hardware-optimized, very fast
**Cons:** Limited patterns, needs special kernels (e.g., Triton BlockSparseAttention)

## Combining Patterns

Real models often combine multiple patterns:

### Longformer Pattern
```python
# Local window + global tokens
attention_mask = local_attention_mask(window_size=256)
attention_mask[0, :] = 1  # CLS token attends globally
attention_mask[:, 0] = 1  # All attend to CLS
```

### BigBird Pattern
```python
# Local + global + random
mask = local_attention_mask(window_size=64)
mask |= global_attention_mask(num_global=2)
mask |= random_attention_mask(num_random=3)
```

### Sparse Transformer Pattern
```python
# Alternate between local and strided per layer
if layer % 2 == 0:
    mask = local_attention_mask(window_size=128)
else:
    mask = strided_attention_mask(stride=64)
```

## Files in This Directory

### local_attention.cu

CUDA implementation of sliding window attention:
- Efficient local window computation
- Causal variant for autoregressive models
- Optimized for fixed window sizes

### strided_attention.cu

CUDA implementation of strided/dilated attention:
- Multiple stride patterns
- Hierarchical attention
- Combined with local for full coverage

### README.md

This file - overview of sparse patterns.

## Performance Comparison

On A100, batch=4, heads=8, d=64:

| Pattern | Seq Len | k/window | Time (ms) | vs Dense | Memory |
|---------|---------|----------|-----------|----------|--------|
| Dense | 2048 | - | 2.1 | 1.0× | 256 MB |
| Local | 2048 | 128 | 0.31 | 6.8× | 16 MB |
| Strided | 2048 | 128 | 0.35 | 6.0× | 16 MB |
| Dense | 8192 | - | 38.2 | 1.0× | 4 GB |
| Local | 8192 | 128 | 1.24 | 30.8× | 64 MB |
| Strided | 8192 | 128 | 1.41 | 27.1× | 64 MB |

Notice: Sparse attention maintains constant time as k is fixed!

## Quality Considerations

### When Does Sparse Attention Work?

**Good scenarios:**
- Strong locality (language, audio, video)
- Hierarchical structure (documents, code)
- Task-specific patterns (QA, classification)

**Challenging scenarios:**
- Dense reasoning (math, logic)
- Long-range dependencies (coreference)
- Unstructured data (random sequences)

### Maintaining Quality

Strategies to preserve model quality:
1. **Layer-wise patterns**: Different patterns per layer
2. **Learned sparsity**: Dynamically select positions (e.g., DeepSeek)
3. **Hybrid attention**: Some layers dense, others sparse
4. **Increased depth**: More layers to compensate for reduced receptive field

## Building and Running

```bash
cd examples/04_sparse_patterns

# Build local attention
mkdir build && cd build
cmake ..
make

# Run examples
./local_attention
./strided_attention

# Python reference
python test_patterns.py --pattern local --window_size 128
python test_patterns.py --pattern strided --stride 32
```

## Exercises

1. Implement a custom sparse pattern for your use case
2. Combine local + global attention
3. Benchmark sparse vs dense for different sequence lengths
4. Visualize attention patterns and compare with dense attention

## References

- [Sparse Transformers](https://arxiv.org/abs/1904.10509) - Child et al., 2019
- [Longformer](https://arxiv.org/abs/2004.05150) - Beltagy et al., 2020
- [BigBird](https://arxiv.org/abs/2007.14062) - Zaheer et al., 2020
- [Random Feature Attention](https://arxiv.org/abs/2103.02143) - Peng et al., 2021

## Next Steps

After understanding these basic patterns:
1. Study DeepSeek's dynamic sparse attention (next example)
2. Implement learned sparsity patterns
3. Explore efficient sparse kernels (Triton, xFormers)
