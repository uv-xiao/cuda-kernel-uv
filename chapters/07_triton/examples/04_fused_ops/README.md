# Example 04 - Fused Operations

Kernel fusion combines multiple operations into a single kernel, reducing memory traffic and improving performance.

## Files

1. **fused_add_mul.py** - Simple fusion: `(x + y) * z`
2. **fused_layernorm.py** - LayerNorm with all operations fused
3. **fused_attention.py** - Simplified fused attention mechanism

## Why Kernel Fusion?

### Memory Traffic Problem

Consider computing `output = (x + y) * z`:

**Unfused (2 kernels):**
```python
temp = x + y      # Kernel 1: read x, y; write temp
output = temp * z  # Kernel 2: read temp, z; write output
```
- Total memory: 6N elements (4 reads + 2 writes)

**Fused (1 kernel):**
```python
output = (x + y) * z  # read x, y, z; write output
```
- Total memory: 4N elements (3 reads + 1 write)
- **33% reduction in memory traffic!**

### Benefits

1. **Reduced Memory Bandwidth**: Fewer DRAM accesses
2. **Eliminated Intermediate Storage**: Results stay in registers
3. **Lower Launch Overhead**: One kernel instead of multiple
4. **Better Cache Utilization**: Data loaded once, used multiple times

### When Fusion Helps Most

- Memory-bound operations (low arithmetic intensity)
- Element-wise operation chains
- Small to medium tensors
- Operations with data dependencies

## Example: fused_add_mul.py

Demonstrates basic fusion benefits:

```python
@triton.jit
def fused_add_mul_kernel(x_ptr, y_ptr, z_ptr, output_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n

    # Load all inputs
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    z = tl.load(z_ptr + offsets, mask=mask)

    # Fused computation (no intermediate storage)
    output = (x + y) * z

    # Store result
    tl.store(output_ptr + offsets, output, mask=mask)
```

**Expected speedup:** 1.5-3x vs unfused

## Example: fused_layernorm.py

LayerNorm is a perfect candidate for fusion:

```
y = (x - mean(x)) / sqrt(var(x) + eps) * gamma + beta
```

Unfused would require 4-5 separate kernels:
1. Compute mean
2. Compute variance
3. Normalize
4. Scale (multiply by gamma)
5. Shift (add beta)

Fused version does all in one pass:

```python
@triton.jit
def layernorm_kernel(...):
    # Load row
    x = tl.load(...)

    # Compute mean and variance
    mean = tl.sum(x) / n_cols
    var = tl.sum((x - mean) ** 2) / n_cols

    # Normalize
    x_normed = (x - mean) / tl.sqrt(var + eps)

    # Affine transform
    y = x_normed * gamma + beta

    # Store (one write instead of 5!)
    tl.store(..., y)
```

**Expected speedup:** 2-4x vs unfused

## Example: fused_attention.py

Simplified attention fusion inspired by Flash Attention:

```
Attention(Q, K, V) = softmax(Q @ K^T) @ V
```

Unfused:
1. Matmul: `scores = Q @ K^T`
2. Softmax: `probs = softmax(scores)`
3. Matmul: `output = probs @ V`

Fused:
- Compute one block at a time
- Never materialize full attention matrix
- Memory: O(n) instead of O(n²)

## Performance Comparison

Typical speedups on A100 GPU:

| Operation | Unfused (ms) | Fused (ms) | Speedup |
|-----------|--------------|------------|---------|
| Add+Mul (100M) | 0.42 | 0.18 | 2.3x |
| LayerNorm (4096x1024) | 0.31 | 0.12 | 2.6x |
| Softmax (1024x512) | 0.089 | 0.087 | 1.02x |

Note: Softmax in PyTorch is already fused, so less improvement.

## Common Fusion Patterns

### 1. Element-wise Chain
```python
# Fuse: y = f(g(h(x)))
y = tl.exp(tl.relu(x + bias))
```

### 2. Reduction + Element-wise
```python
# Fuse: LayerNorm, BatchNorm, etc.
mean = tl.sum(x) / n
normalized = (x - mean) / tl.sqrt(var)
```

### 3. Matmul + Activation
```python
# Fuse: Linear layer + activation
result = tl.dot(a, b)
output = tl.maximum(result, 0)  # ReLU
```

### 4. Multi-step Reductions
```python
# Fuse: softmax (max + sum)
max_val = tl.max(x)
exp_vals = tl.exp(x - max_val)
output = exp_vals / tl.sum(exp_vals)
```

## Best Practices

1. **Identify Bottlenecks**: Profile to find memory-bound operations
2. **Start Simple**: Fuse 2-3 operations first
3. **Verify Correctness**: Test against unfused baseline
4. **Measure Performance**: Ensure fusion actually helps
5. **Consider Reuse**: Don't fuse if intermediate result is reused elsewhere

## Anti-patterns

Avoid fusing when:
- Intermediate results needed elsewhere
- Operations are compute-bound (already maxing out FLOPs)
- Very large tensors (may exceed SRAM capacity)
- Complex control flow (makes fusion difficult)

## Exercises

1. **Fuse GELU**: Implement fused GELU activation
2. **Fuse Dropout**: Combine dropout with another operation
3. **Fuse Bias+Activation**: Extend matmul to include bias and ReLU
4. **Custom Fusion**: Find bottleneck in your code and fuse it

## References

- [Flash Attention Paper](https://arxiv.org/abs/2205.14135)
- [Triton Fused Softmax Tutorial](https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html)
- [Kernel Fusion in PyTorch](https://pytorch.org/docs/stable/generated/torch.jit.fuse.html)

## Key Takeaways

1. Fusion reduces memory traffic by eliminating intermediate storage
2. Most beneficial for memory-bound, element-wise operations
3. Triton makes fusion easy - just combine operations in one kernel
4. Expected speedups: 1.5-4x for common patterns
5. Always verify correctness and measure performance
