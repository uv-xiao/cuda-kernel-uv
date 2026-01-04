# Chapter 07 - Triton Kernel Design

Triton is a language and compiler for parallel programming that makes it easier to write highly efficient GPU kernels. Unlike CUDA, which requires explicit memory management and optimization, Triton uses a Python-based DSL with automatic optimizations.

## Learning Goals

By the end of this chapter, you will understand:

1. **Triton Programming Model**
   - Block-based parallel computation
   - Program ID and indexing schemes
   - Memory loading and storing patterns
   - Mask-based boundary handling

2. **Kernel Optimization**
   - Autotuning configurations
   - Tiling and blocking strategies
   - Memory coalescing and reuse
   - Performance profiling

3. **Operator Fusion**
   - Benefits of kernel fusion
   - Implementing fused operations
   - Memory bandwidth optimization
   - Common fusion patterns

4. **Comparison with CUDA**
   - When to use Triton vs CUDA
   - Performance characteristics
   - Development productivity trade-offs

## Key Concepts

### Block-Based Programming Model

Triton uses a **block-based** programming model where:
- Each program instance processes a block of elements
- `tl.program_id(axis)` identifies which block this instance handles
- Block size is a compile-time constant for optimization

```python
@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)  # Which block am I?
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)
```

### Core Triton Operations

| Operation | Description | Example |
|-----------|-------------|---------|
| `tl.program_id(axis)` | Get current program ID | `pid = tl.program_id(0)` |
| `tl.arange(start, end)` | Create range tensor | `offsets = tl.arange(0, BLOCK_SIZE)` |
| `tl.load(ptr, mask)` | Load from memory | `x = tl.load(x_ptr + offsets, mask=mask)` |
| `tl.store(ptr, value, mask)` | Store to memory | `tl.store(out_ptr + offsets, result, mask=mask)` |
| `tl.dot(a, b)` | Matrix multiplication | `c = tl.dot(a, b)` |
| `tl.sum(x, axis)` | Reduction sum | `total = tl.sum(x, axis=0)` |
| `tl.max(x, axis)` | Reduction max | `max_val = tl.max(x, axis=0)` |

### Memory Access Patterns

**Coalesced Access:**
```python
# Good: contiguous access
offsets = block_start + tl.arange(0, BLOCK_SIZE)
data = tl.load(ptr + offsets)
```

**2D Tiling:**
```python
# Matrix multiplication tiling
offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)[:, None])
offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)[None, :])
offs_k = tl.arange(0, BLOCK_SIZE_K)
a_ptrs = a_ptr + offs_am * stride_am + offs_k[None, :] * stride_ak
b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_bn * stride_bn
```

### Autotuning

Triton supports automatic performance tuning:

```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
    ],
    key=['n_elements'],
)
@triton.jit
def kernel(...):
    ...
```

The compiler benchmarks all configurations and selects the fastest.

## Triton vs CUDA Comparison

| Aspect | CUDA | Triton |
|--------|------|--------|
| **Language** | C++ extension | Python DSL |
| **Memory Management** | Manual (shared mem, registers) | Automatic optimization |
| **Optimization** | Explicit tuning required | Autotuning built-in |
| **Development Speed** | Slower, more verbose | Faster prototyping |
| **Performance Ceiling** | Highest (full control) | Very high (95-100% of CUDA) |
| **Learning Curve** | Steep | Gentler |
| **Debugging** | Standard C++ tools | Python + Triton tools |
| **Portability** | NVIDIA GPUs only | NVIDIA, AMD (limited), CPU |
| **Use Case** | Maximum performance | Rapid development, research |

### When to Use Each

**Choose CUDA when:**
- You need absolute maximum performance
- Using complex GPU features (dynamic parallelism, etc.)
- Integrating with existing CUDA codebases
- Building low-level libraries

**Choose Triton when:**
- Rapid prototyping and iteration
- Research and experimentation
- Custom operators for PyTorch
- Performance is critical but not absolute maximum
- Team prefers Python ecosystem

## Chapter Organization

### Examples

1. **01_vector_add/** - Introduction to Triton basics
2. **02_softmax/** - Reductions and numerical stability
3. **03_matmul/** - Matrix multiplication from naive to optimized
4. **04_fused_ops/** - Operator fusion techniques
5. **05_autotuning/** - Comprehensive autotuning guide

### Puzzles

Interactive coding challenges inspired by [Triton-Puzzles-Lite](https://github.com/srush/Triton-Puzzles):
- Fill-in-the-blank kernel implementations
- Progressive difficulty
- Immediate feedback with tests

### Exercises

1. **01_relu_fusion/** - Fuse ReLU with matrix multiply
2. **02_gelu/** - Implement GELU activation function

## Resources

### Official Documentation
- [Triton Documentation](https://triton-lang.org/)
- [Triton GitHub](https://github.com/openai/triton)
- [Triton Tutorial](https://triton-lang.org/main/getting-started/tutorials/index.html)

### Learning Resources
- [Triton-Puzzles](https://github.com/srush/Triton-Puzzles) - Interactive learning
- [PyTorch Triton Integration](https://pytorch.org/tutorials/recipes/torch_compile_backend_ipex.html)
- [GPU MODE Lectures](https://github.com/gpu-mode/lectures) - Community tutorials

### Research Papers
- [Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations](https://www.eecs.harvard.edu/~htk/publication/2019-mapl-tillet-kung-cox.pdf)

### Comparison Studies
- [Triton vs CUDA Performance](https://openai.com/research/triton)
- [Flash Attention Triton Implementation](https://github.com/Dao-AILab/flash-attention)

## Installation

```bash
pip install triton
```

For development:
```bash
pip install triton pytest torch
```

## Running Examples

All examples are standalone Python scripts:

```bash
cd examples/01_vector_add
python vector_add.py
```

Most examples include:
- Triton kernel implementation
- PyTorch baseline for comparison
- Performance benchmarking
- Correctness verification

## Testing

Run all tests:
```bash
cd exercises/01_relu_fusion
python test.py
```

## Performance Notes

Expected performance tiers:
- **Tier 1**: Naive Triton ≈ 50-70% of optimal
- **Tier 2**: Blocked Triton ≈ 80-90% of optimal
- **Tier 3**: Autotuned Triton ≈ 95-100% of optimal

Triton typically achieves 95-100% of hand-tuned CUDA performance with significantly less code.

## Next Steps

After completing this chapter:
1. Review CUDA chapters (01-06) for comparison
2. Explore Chapter 08 - Advanced Patterns
3. Implement custom operators for your models
4. Contribute to Triton ecosystem

## Quick Reference Card

```python
# Basic kernel template
import triton
import triton.language as tl

@triton.jit
def kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load
    x = tl.load(input_ptr + offsets, mask=mask)

    # Compute
    y = x * 2

    # Store
    tl.store(output_ptr + offsets, y, mask=mask)

# Launch
def launch(input, output, n_elements):
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    kernel[grid](input, output, n_elements, BLOCK_SIZE)
```
