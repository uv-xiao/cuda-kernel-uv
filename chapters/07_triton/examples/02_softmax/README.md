# Example 02 - Softmax in Triton

This example demonstrates reduction operations and numerical stability in Triton through the implementation of softmax.

## Learning Objectives

- Understand reduction operations (`tl.max()`, `tl.sum()`)
- Learn row-wise parallel processing patterns
- Master numerical stability techniques
- Implement online algorithms for memory efficiency

## Softmax Background

Softmax is a fundamental operation in deep learning used to convert logits to probabilities:

```
softmax(x)_i = exp(x_i) / sum_j(exp(x_j))
```

**Properties:**
- Output values in [0, 1]
- Sum of outputs = 1
- Preserves ordering (larger input → larger output)
- Commonly used in classification, attention, etc.

## Numerical Stability Problem

### Naive Implementation (WRONG)

```python
def naive_softmax(x):
    exp_x = torch.exp(x)  # Can overflow!
    return exp_x / exp_x.sum()
```

**Problem:** `exp(1000) = inf` causes overflow

### Stable Implementation (CORRECT)

```python
def stable_softmax(x):
    x_max = x.max()
    exp_shifted = torch.exp(x - x_max)  # Safe!
    return exp_shifted / exp_shifted.sum()
```

**Solution:** Subtract max before exp

**Why it works:**
```
exp(x_i) / sum(exp(x_j))
= exp(x_i - c) / sum(exp(x_j - c))  for any constant c
= exp(x_i - max(x)) / sum(exp(x_j - max(x)))  let c = max(x)
```

Now the largest exponent is `exp(0) = 1`, preventing overflow.

## Triton Implementation

### Standard Softmax Kernel

```python
@triton.jit
def softmax_kernel(output_ptr, input_ptr, input_row_stride,
                   output_row_stride, n_cols, BLOCK_SIZE: tl.constexpr):
    # Each program handles one row
    row_idx = tl.program_id(0)

    # Load row
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    row_start_ptr = input_ptr + row_idx * input_row_stride
    row = tl.load(row_start_ptr + col_offsets, mask=mask, other=-float('inf'))

    # Find max (for stability)
    row_max = tl.max(row, axis=0)

    # Compute exp(x - max)
    row_shifted = row - row_max
    numerator = tl.exp(row_shifted)

    # Sum and normalize
    denominator = tl.sum(numerator, axis=0)
    softmax_output = numerator / denominator

    # Store
    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    tl.store(output_row_start_ptr + col_offsets, softmax_output, mask=mask)
```

### Key Concepts

#### 1. Row-Wise Parallelism

```
Input matrix (n_rows x n_cols):
Row 0: [x00, x01, x02, ...] → Program 0
Row 1: [x10, x11, x12, ...] → Program 1
Row 2: [x20, x21, x22, ...] → Program 2
...
```

Each program processes one complete row independently.

#### 2. Reduction Operations

```python
row_max = tl.max(row, axis=0)  # Reduce along axis 0
row_sum = tl.sum(row, axis=0)  # Reduce along axis 0
```

Reductions collapse a dimension:
- Input: `row` has shape `(BLOCK_SIZE,)`
- Output: `row_max` is a scalar

#### 3. Broadcast Operations

```python
row_shifted = row - row_max  # Broadcasting
# row:      [x0, x1, x2, ...]
# row_max:  scalar
# result:   [x0-max, x1-max, x2-max, ...]
```

#### 4. Row Stride

```python
row_start_ptr = input_ptr + row_idx * input_row_stride
```

For contiguous memory: `stride = n_cols`
For transposed: `stride = 1`

## Online Softmax Algorithm

For very large rows that don't fit in SRAM, use the **online** (streaming) algorithm:

```python
@triton.jit
def softmax_kernel_online(...):
    running_max = -float('inf')
    running_sum = 0.0

    # First pass: compute max and sum
    for block in chunks:
        block_vals = load(block)
        block_max = tl.max(block_vals)

        # Update running max
        old_max = running_max
        running_max = tl.maximum(running_max, block_max)

        # Update running sum (rescale for new max)
        running_sum = running_sum * tl.exp(old_max - running_max)
        running_sum += tl.sum(tl.exp(block_vals - running_max))

    # Second pass: normalize and store
    for block in chunks:
        block_vals = load(block)
        softmax_vals = tl.exp(block_vals - running_max) / running_sum
        store(softmax_vals)
```

**Advantage:** Processes one chunk at a time (lower memory)
**Disadvantage:** Two passes over data (more memory traffic)

## Running the Example

```bash
python softmax.py
```

### Expected Output

```
Softmax Algorithm Analysis
======================================================================
Input: tensor([[1., 2., 3., 4.]], device='cuda:0')

Step 1: Find maximum (for stability)
  max(x) = 4.0

Step 2: Subtract max
  x - max(x) = tensor([[-3., -2., -1.,  0.]], device='cuda:0')

Step 3: Compute exponential
  exp(x - max(x)) = tensor([[0.0498, 0.1353, 0.3679, 1.0000]], device='cuda:0')

Step 4: Compute sum
  sum(exp(x - max(x))) = 1.5529

Step 5: Normalize
  softmax(x) = tensor([[0.0321, 0.0871, 0.2369, 0.6439]], device='cuda:0')
  sum(softmax(x)) = 1.0000000000

Numerical Stability Test
======================================================================
Input: tensor([[1000., 1001., 1002.]], device='cuda:0')
Max value: 1002.0

Naive approach: exp(x) / sum(exp(x))
  exp(x) = tensor([[inf, inf, inf]], device='cuda:0')
  Result: OVERFLOW (inf values)

Stable approach: exp(x - max(x)) / sum(exp(x - max(x)))
  x - max(x) = tensor([[-2., -1.,  0.]], device='cuda:0')
  exp(x - max(x)) = tensor([[0.1353, 0.3679, 1.0000]], device='cuda:0')
  Result: tensor([[0.0900, 0.2447, 0.6652]], device='cuda:0')
  Sum: 1.0000000000 (should be 1.0)

Testing Correctness
======================================================================
  Shape (   1,    4): PASS
  Shape (   4,    1): PASS
  Shape ( 128,  256): PASS
  Shape ( 512,  512): PASS
  Shape (1024,   64): PASS
All tests passed!

Benchmarking softmax (shape=1024x512)
======================================================================
Triton (standard):  0.087 ms
Triton (online):    0.142 ms
PyTorch:            0.089 ms
Speedup (std):     1.02x
Speedup (online):  0.63x
```

## Performance Analysis

### Memory Access Pattern

Softmax requires:
1. Read entire row (n_cols elements)
2. Compute max (reduction)
3. Read row again (for exp)
4. Compute sum (reduction)
5. Write row (n_cols elements)

**Total memory:** 2 reads + 1 write = `3 * n_cols * sizeof(float)`

### Arithmetic Intensity

- Operations: 2*n_cols (one exp, one div) + reductions
- Bytes: 3*n_cols*4
- **Intensity: ~0.17 ops/byte** (memory-bound)

### Standard vs Online

| Algorithm | Memory Passes | SRAM Usage | Best For |
|-----------|---------------|------------|----------|
| Standard | 1 | Full row | n_cols < ~8K |
| Online | 2 | Block-sized | Large n_cols |

For typical cases (n_cols ≤ 2048), standard is faster.

## Common Patterns

### 1. Per-Row Processing

```python
# Launch one program per row
grid = (n_rows,)

# Each program handles one row
row_idx = tl.program_id(0)
```

### 2. Loading with Other Value

```python
# Use 'other' to set value for masked elements
row = tl.load(ptrs, mask=mask, other=-float('inf'))
# Ensures max() ignores out-of-bounds values
```

### 3. Next Power of 2

```python
# Round up to power of 2 for efficiency
BLOCK_SIZE = triton.next_power_of_2(n_cols)
```

This improves:
- Memory coalescing
- Warp utilization
- Compiler optimizations

## Exercises

1. **Numerical Stability**: Remove the `- row_max` line and test with large values. What happens?

2. **LogSumExp**: Implement `logsumexp(x) = log(sum(exp(x)))` using the same stability trick.

3. **Column-wise Softmax**: Modify to compute softmax along columns instead of rows.

4. **Temperature Scaling**: Add a temperature parameter: `softmax(x/T)`.

5. **Masked Softmax**: Support a mask where certain elements should be -inf before softmax (useful for attention).

## Comparison with CUDA

CUDA softmax would require:
- Explicit shared memory for reductions
- Manual warp-level primitives (`__shfl_down_sync`)
- Careful synchronization (`__syncthreads()`)
- More complex indexing

Triton handles these automatically:
- `tl.max()` and `tl.sum()` compile to efficient reductions
- Block-level operations are optimized automatically
- Less code, fewer bugs

## Advanced: Flash Attention Connection

Flash Attention uses a similar online algorithm to compute softmax during attention:

```
Attention(Q, K, V) = softmax(Q @ K^T) @ V
```

The online algorithm enables:
- Processing large sequence lengths
- Reducing memory from O(n²) to O(n)
- Fusing operations for speed

See `examples/04_fused_ops/fused_attention.py` for simplified version.

## Key Takeaways

1. **Numerical Stability**: Always subtract max before exp
2. **Row-wise Parallelism**: One program per row
3. **Reductions**: `tl.max()` and `tl.sum()` for aggregation
4. **Online Algorithm**: Better for very large rows
5. **Memory-Bound**: Performance limited by bandwidth, not compute

## Next Steps

- [Example 03 - Matrix Multiplication](../03_matmul/): Learn 2D tiling
- [Example 04 - Fused Ops](../04_fused_ops/): Combine softmax with other ops
- Try exercises to deepen understanding

## References

- [Triton Softmax Tutorial](https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html)
- [Online Normalizer Calculation](https://arxiv.org/abs/1805.02867)
- [Flash Attention](https://arxiv.org/abs/2205.14135)
