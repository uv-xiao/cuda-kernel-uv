# DeepSeek Sparse Attention (DSA)

## Overview

This directory implements DeepSeek-V3's Dynamic Sparse Attention (DSA), which uses learned token selection to achieve both efficiency and quality. Unlike fixed sparse patterns, DSA dynamically selects which tokens to attend to based on query-key similarity.

## DeepSeek-V3 Background

DeepSeek-V3 is a 671B parameter mixture-of-experts (MoE) model that uses:
- **Multi-head Latent Attention (MLA)**: Compressed KV cache
- **Dynamic Sparse Attention (DSA)**: Learned sparse patterns
- **Load balancing**: For efficient MoE routing

Key innovation: **Lightning Indexer** - O(1) index computation for sparse attention.

## DeepSeek Sparse Attention (DSA)

### Core Idea

Instead of fixed patterns (local, strided), DSA:
1. Computes **approximate** attention scores for all positions
2. Selects **top-k** most relevant tokens per query
3. Computes **exact** attention only for selected tokens

**Complexity**: O(L · d · k) where k is typically 128-512 (vs L = 32768+)

### Algorithm

```python
# Step 1: Compute low-rank approximation of attention scores
Q_approx = Q @ W_q_approx  # [L, d] -> [L, r] where r << d
K_approx = K @ W_k_approx  # [L, d] -> [L, r]

S_approx = Q_approx @ K_approx.T  # [L, L] but in low rank

# Step 2: Select top-k positions per query
indices = topk(S_approx, k, dim=-1)  # [L, k]

# Step 3: Compute exact attention for selected positions
Q_full = Q  # [L, d]
K_selected = K[indices]  # [L, k, d]
V_selected = V[indices]  # [L, k, d]

S_exact = (Q_full.unsqueeze(1) * K_selected).sum(-1) / sqrt(d)  # [L, k]
P = softmax(S_exact)  # [L, k]
O = (P.unsqueeze(-1) * V_selected).sum(1)  # [L, d]
```

### Three-Stage Process

**Stage 1: Lightning Indexer (Selection)**
- Input: Q, K (or low-rank versions)
- Output: indices [L, k] of top-k tokens per query
- Complexity: O(L · r²) where r is rank (typically 128)

**Stage 2: Gather**
- Input: K, V, indices
- Output: K_selected [L, k, d], V_selected [L, k, d]
- Complexity: O(L · k · d)

**Stage 3: Sparse Attention**
- Input: Q, K_selected, V_selected
- Output: O [L, d]
- Complexity: O(L · k · d)

**Total: O(L · (r² + k · d))** - much better than O(L² · d) for r, k << L!

## Lightning Indexer

### The Challenge

Naive top-k selection:
```python
# For each query
for i in range(L):
    scores = Q[i] @ K.T  # O(L · d) per query
    top_indices = topk(scores, k)  # O(L log k) per query

# Total: O(L² · d) - no speedup!
```

### DeepSeek's Solution

Use **low-rank approximation** for selection:

```python
# Compress Q, K to rank r
Q_approx = Q @ W_q  # [L, d] @ [d, r] -> [L, r]
K_approx = K @ W_k  # [L, d] @ [d, r] -> [L, r]

# Compute approximate scores
S_approx = Q_approx @ K_approx.T  # [L, r] @ [r, L] -> [L, L]
                                   # BUT: r << d so much faster!

# Select top-k
indices = topk(S_approx, k, dim=-1)  # [L, k]

# Total: O(L · d · r + L · r² + L² log k)
# For r << d: Dominated by O(L² log k) for top-k
# BUT: Can use approximations like MIPS for O(L · k)!
```

### Further Optimization: Blocked Indexer

DeepSeek uses **block-wise top-k**:

```python
# Divide sequence into blocks
for block_i in range(num_blocks):
    Q_block = Q_approx[block_i]  # [block_size, r]

    # Only compute scores for nearby blocks (locality bias)
    nearby_blocks = get_nearby_blocks(block_i, radius)

    for block_j in nearby_blocks:
        K_block = K_approx[block_j]  # [block_size, r]
        S_block = Q_block @ K_block.T  # [block_size, block_size]

    # Top-k within nearby blocks
    indices_block = topk(S_block, k)

# Complexity: O(L · r · w) where w is window size
```

## Multi-Head Latent Attention (MLA)

DeepSeek combines DSA with MLA for further efficiency:

```python
# Standard multi-head: K, V shape [B, H, L, d]
# Memory: O(H · L · d) for KV cache

# MLA: Compress to latent space
K_latent = K @ W_k_down  # [B, H, L, d] -> [B, L, d_latent]
V_latent = V @ W_v_down  # [B, H, L, d] -> [B, L, d_latent]

# Memory: O(L · d_latent) where d_latent << H · d
# Typical: d_latent = 512, H·d = 8192 -> 16× compression!

# At inference, expand per head
K_h = K_latent @ W_k_up[h]  # [B, L, d_latent] -> [B, L, d]
V_h = V_latent @ W_v_up[h]  # [B, L, d_latent] -> [B, L, d]
```

## Implementation Components

### 1. lightning_indexer.py

Implements efficient top-k token selection:
- Low-rank score computation
- Blocked indexing for locality
- Approximate nearest neighbors (FAISS integration)

### 2. token_selector.py

Top-k selection strategies:
- Exact top-k (for validation)
- Approximate top-k (LSH, MIPS)
- Hybrid: exact for nearby, approximate for distant

### 3. sparse_attention.py

Full DSA pipeline:
- Integrates indexer + sparse attention
- Supports MLA compression
- Causal masking for autoregressive models

## Performance Analysis

### Complexity Comparison

For L=32768, d=128, H=8, k=512, r=128:

| Component | Dense | DSA |
|-----------|-------|-----|
| Score computation | O(L² · d) = 137B | O(L · r²) = 537M |
| Top-k selection | - | O(L · k log k) = 151M |
| Sparse attention | - | O(L · k · d) = 2.1B |
| **Total FLOPs** | **137B** | **2.8B** |
| **Speedup** | **1×** | **49×** |

### Memory Comparison

| Component | Dense | DSA |
|-----------|-------|-----|
| Attention matrix | L² = 1GB | L · k = 16MB |
| KV cache (full) | H·L·d = 32MB | d_latent·L = 4MB |
| **Total** | **1GB** | **20MB** |
| **Reduction** | **1×** | **50×** |

## Quality Considerations

### Does Sparsity Hurt Quality?

DeepSeek-V3 maintains quality through:

1. **Adaptive k**: More important layers use larger k
2. **Learned selection**: W_q, W_k trained end-to-end
3. **Hybrid patterns**: Combine top-k with local window
4. **Per-layer tuning**: Different sparsity patterns per layer

### Example Patterns

```python
# Early layers: More local (syntax)
pattern_early = local_window(256) + topk_global(64)

# Middle layers: Balanced (semantics)
pattern_mid = local_window(128) + topk_global(256)

# Late layers: More global (reasoning)
pattern_late = local_window(64) + topk_global(512)
```

## Files in This Directory

### lightning_indexer.py
- Low-rank score approximation
- Efficient top-k selection
- Blocked indexing

### token_selector.py
- Various top-k strategies
- Benchmarking utilities
- Accuracy vs speed trade-offs

### sparse_attention.py
- End-to-end DSA implementation
- Integration with FlashAttention
- Performance benchmarks

## Usage

```python
# Basic usage
from sparse_attention import DeepSeekSparseAttention

model = DeepSeekSparseAttention(
    dim=128,
    num_heads=8,
    approx_rank=128,
    topk=512
)

output = model(Q, K, V)  # Automatic sparse attention

# Advanced: Custom indexer
from lightning_indexer import LightningIndexer

indexer = LightningIndexer(
    dim=128,
    approx_rank=128,
    block_size=256,
    use_faiss=True
)

indices = indexer.select_tokens(Q, K, k=512)  # [L, k]
```

## Benchmarks

```bash
# Compare dense vs sparse
python sparse_attention.py --benchmark

# Test different k values
python sparse_attention.py --topk 128 256 512 1024

# Measure indexer overhead
python lightning_indexer.py --profile
```

## Extensions and Variants

### 1. Hierarchical Indexing

```python
# Two-stage selection
coarse_indices = indexer.select(Q, K, k=1024)  # Coarse
fine_indices = indexer.select(Q, K[coarse_indices], k=256)  # Fine
```

### 2. Cache-Aware Selection

```python
# Prefer cached tokens (for incremental decoding)
scores = S_approx.clone()
scores[:, cached_positions] += cache_bonus
indices = topk(scores, k)
```

### 3. Task-Specific Patterns

```python
# Question answering: Attend to question tokens
question_mask = (token_types == QUESTION)
scores[:, question_mask] += question_bonus
```

## References

### DeepSeek Papers
- [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437) - Main reference
- See Section 3.2: Multi-head Latent Attention
- See Appendix: Implementation details

### Related Work
- [Linformer](https://arxiv.org/abs/2006.04768) - Low-rank attention approximation
- [Reformer](https://arxiv.org/abs/2001.04451) - LSH-based sparse attention
- [Routing Transformer](https://arxiv.org/abs/2003.05997) - Learned sparse patterns

### Efficient Search
- [FAISS](https://github.com/facebookresearch/faiss) - Fast approximate nearest neighbors
- [ScaNN](https://github.com/google-research/google-research/tree/master/scann) - Scalable nearest neighbors

## Key Takeaways

1. **Dynamic > Static**: Learned sparsity outperforms fixed patterns
2. **Low-rank approximation**: Enables O(r²) instead of O(d²) for selection
3. **Multi-head compression**: MLA reduces KV cache by 10-20×
4. **Quality maintenance**: Careful design preserves model capability
5. **Practical speedup**: 30-50× for very long sequences (L > 16K)

## Next Steps

After understanding DSA:
1. Implement your own sparse pattern
2. Experiment with different k values
3. Combine with FlashAttention for maximum efficiency
4. Profile on your hardware
