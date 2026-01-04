# Attention Basics: Naive Implementation

## Overview

This directory contains baseline implementations of standard scaled dot-product attention. These serve as reference implementations to compare against optimized versions.

## The Attention Mechanism

### Mathematical Definition

Scaled dot-product attention is defined as:

```
Attention(Q, K, V) = softmax(QK^T / √d_k) V
```

Where:
- Q (Query): shape [B, L, d] - "What am I looking for?"
- K (Key): shape [B, L, d] - "What do I contain?"
- V (Value): shape [B, L, d] - "What information do I carry?"
- B: batch size
- L: sequence length
- d: head dimension

### Step-by-Step Algorithm

```python
# Step 1: Compute attention scores
S = Q @ K.T  # [B, L, L] - similarity matrix
S = S / sqrt(d)  # Scale by √d for numerical stability

# Step 2: Apply mask (optional, for causal/padding)
if mask is not None:
    S = S.masked_fill(mask == 0, -inf)

# Step 3: Compute attention weights
P = softmax(S, dim=-1)  # [B, L, L] - normalized probabilities

# Step 4: Apply dropout (during training)
P = dropout(P)

# Step 5: Compute output
O = P @ V  # [B, L, d] - weighted sum of values
```

### Computational Complexity

**Time Complexity:**
- QK^T: O(L² · d) - dominant term for long sequences
- Softmax: O(L²)
- PV: O(L² · d)
- **Total: O(L² · d)**

**Space Complexity:**
- Store S and P: O(L²) - memory bottleneck!
- For L=4096: 16M elements per sample
- For L=32768: 1B elements per sample (4GB in FP32!)

### Why is This Naive?

1. **Materialization**: Stores full L×L attention matrix in memory
2. **Memory bandwidth**: Multiple passes over data (separate kernels)
3. **No fusion**: QK^T, softmax, and PV are separate operations
4. **Poor scaling**: L² term makes long sequences impractical

## Files

### naive_attention.cu

CUDA implementation with:
- Standard three-pass algorithm (QK^T, softmax, PV)
- Separate kernels for each operation
- No optimization (educational baseline)
- Supports causal masking

**Expected performance**: ~10× slower than FlashAttention

### naive_attention.py

PyTorch reference implementation:
- Pure Python/PyTorch for clarity
- Identical algorithm to CUDA version
- Useful for testing and validation
- Includes visualization utilities

## Usage

### Building the CUDA Version

```bash
cd examples/01_attention_basics
mkdir build && cd build
cmake ..
make
```

### Running Examples

```bash
# CUDA version
./naive_attention

# Python version
python naive_attention.py --seq_len 512 --batch_size 4 --num_heads 8
```

### Testing Correctness

```python
import torch
from naive_attention import naive_attention_pytorch

# Create random inputs
B, L, d = 2, 128, 64
Q = torch.randn(B, L, d)
K = torch.randn(B, L, d)
V = torch.randn(B, L, d)

# Compute attention
output = naive_attention_pytorch(Q, K, V)

# Compare with PyTorch's built-in
expected = torch.nn.functional.scaled_dot_product_attention(Q, K, V)
assert torch.allclose(output, expected, atol=1e-5)
```

## Numerical Stability

### Why Scale by √d?

Without scaling, for large d:
- Dot products QK^T have variance proportional to d
- Softmax outputs become near one-hot (high variance)
- Gradients vanish for non-max positions

Scaling by √d normalizes variance to 1.

### Softmax Numerical Stability

Naive softmax can overflow:

```python
# Unstable
exp_scores = torch.exp(scores)
probs = exp_scores / exp_scores.sum(dim=-1)
```

Stable version subtracts max:

```python
# Stable
max_score = scores.max(dim=-1, keepdim=True)
exp_scores = torch.exp(scores - max_score)
probs = exp_scores / exp_scores.sum(dim=-1, keepdim=True)
```

## Memory Layout

### Multi-Head Attention

For multi-head attention, Q, K, V have shape [B, H, L, d]:
- B: batch size
- H: number of heads
- L: sequence length
- d: head dimension (typically d_model / H)

Memory layout options:
1. **BSNH** (batch, seq, num_heads, head_dim): Better for QKV projection
2. **BNSH** (batch, num_heads, seq, head_dim): Better for attention computation

Our implementation uses BNSH for simplicity.

## Benchmarking

Expected performance on A100:

| Seq Length | Time (ms) | Memory (MB) |
|------------|-----------|-------------|
| 128 | 0.08 | 1 |
| 512 | 0.9 | 16 |
| 1024 | 3.5 | 64 |
| 2048 | 14.1 | 256 |
| 4096 | 56.8 | 1024 |

Notice the quadratic scaling in both time and memory!

## Visualizing Attention

The Python implementation includes utilities to visualize attention patterns:

```python
import matplotlib.pyplot as plt

# Compute attention
output, attention_weights = naive_attention_pytorch(Q, K, V, return_attention=True)

# Visualize
plt.imshow(attention_weights[0, 0].cpu(), cmap='viridis')
plt.colorbar()
plt.title('Attention Pattern (Head 0)')
plt.xlabel('Key Position')
plt.ylabel('Query Position')
plt.show()
```

## Common Patterns

### Causal (Autoregressive) Attention

```python
# Create causal mask (lower triangular)
mask = torch.tril(torch.ones(L, L))
output = naive_attention_pytorch(Q, K, V, mask=mask)
```

### Padding Mask

```python
# Mask out padding tokens
# padding_mask: [B, L] where 1 = valid, 0 = padding
padding_mask = (tokens != PAD_TOKEN)
mask = padding_mask.unsqueeze(1) * padding_mask.unsqueeze(2)  # [B, L, L]
output = naive_attention_pytorch(Q, K, V, mask=mask)
```

## Next Steps

1. Understand this baseline implementation thoroughly
2. Profile memory usage and identify bottlenecks
3. Move to `02_flash_attention_minimal` to see optimizations
4. Compare performance differences

## References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Original Transformer paper
- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/) - Excellent visual explanation
- [PyTorch scaled_dot_product_attention docs](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)
