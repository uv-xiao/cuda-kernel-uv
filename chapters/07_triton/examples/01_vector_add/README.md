# Example 01 - Vector Addition in Triton

This example introduces the fundamental concepts of Triton programming through a simple vector addition kernel.

## Learning Objectives

- Understand Triton's block-based programming model
- Learn to use `tl.program_id()`, `tl.arange()`, `tl.load()`, and `tl.store()`
- Master masked memory operations for boundary handling
- Compare Triton with CUDA thread-based model

## Triton Basics

### Block-Based Programming

Unlike CUDA where each thread typically processes one element, Triton programs process **blocks** of elements:

```
CUDA Model:                 Triton Model:
Thread 0 → Element 0        Program 0 → Elements [0, 1, ..., BLOCK_SIZE-1]
Thread 1 → Element 1        Program 1 → Elements [BLOCK_SIZE, ..., 2*BLOCK_SIZE-1]
Thread 2 → Element 2        Program 2 → Elements [2*BLOCK_SIZE, ...]
...                         ...
```

### Core Concepts

#### 1. Program ID

```python
pid = tl.program_id(axis=0)  # Which block am I processing?
```

- `axis=0`: First dimension (for 1D problems)
- `axis=1`: Second dimension (for 2D problems like matrices)

#### 2. Creating Offsets

```python
block_start = pid * BLOCK_SIZE
offsets = block_start + tl.arange(0, BLOCK_SIZE)
```

This creates a vector of indices: `[block_start, block_start+1, ..., block_start+BLOCK_SIZE-1]`

#### 3. Masked Memory Operations

```python
mask = offsets < n_elements
x = tl.load(x_ptr + offsets, mask=mask)  # Only load valid elements
tl.store(output_ptr + offsets, result, mask=mask)  # Only store valid elements
```

Masks are crucial for handling:
- Boundary conditions (when size not divisible by BLOCK_SIZE)
- Conditional operations
- Sparse access patterns

#### 4. Compile-Time Constants

```python
BLOCK_SIZE: tl.constexpr  # Compile-time constant for optimization
```

The compiler uses this to:
- Unroll loops
- Optimize register allocation
- Generate specialized code

## Code Walkthrough

### The Kernel

```python
@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # 1. Identify which block to process
    pid = tl.program_id(axis=0)

    # 2. Calculate indices for this block
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # 3. Create boundary mask
    mask = offsets < n_elements

    # 4. Load data (vectorized)
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)

    # 5. Compute (vectorized)
    output = x + y

    # 6. Store result (vectorized)
    tl.store(output_ptr + offsets, output, mask=mask)
```

### The Wrapper

```python
def triton_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    output = torch.empty_like(x)
    n_elements = x.numel()
    BLOCK_SIZE = 1024

    # Calculate grid size
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)

    # Launch kernel
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE)

    return output
```

## Running the Example

```bash
python vector_add.py
```

### Expected Output

```
Triton Vector Addition Tutorial
======================================================================
Testing correctness...
  Size       1: PASS
  Size     127: PASS
  Size     128: PASS
  Size    1023: PASS
  Size    1024: PASS
  Size   10000: PASS
  Size 1000000: PASS
All tests passed!

Block-based Computation Pattern
======================================================================
Total elements: 5000
Block size:     1024
Num blocks:     5

Block 0: elements [   0:1024]  (1024 valid elements)
Block 1: elements [1024:2048]  (1024 valid elements)
Block 2: elements [2048:3072]  (1024 valid elements)
Block 3: elements [3072:4096]  (1024 valid elements)
Block 4: elements [4096:5000]  ( 904 valid elements)

Benchmarking vector addition (size=1,000,000)
======================================================================
Triton:    0.035 ms  (342.86 GB/s)
PyTorch:   0.034 ms  (352.94 GB/s)
Speedup: 0.97x
```

## Performance Analysis

### Memory Bandwidth

Vector addition is **memory-bound**:
- Arithmetic intensity: 1 FLOP / 12 bytes (2 reads + 1 write, fp32)
- Peak performance limited by memory bandwidth, not compute

For A100 GPU:
- Theoretical bandwidth: ~2000 GB/s (HBM2e)
- Achieved: ~340 GB/s (17% of peak)
- Limited by: Kernel launch overhead, small size

### Why Triton ≈ PyTorch?

Both use highly optimized memory operations:
- PyTorch: Calls cuBLAS/custom CUDA kernels
- Triton: Compiles to optimized CUDA code
- Similar performance for memory-bound ops

## Comparison with CUDA

| Aspect | CUDA | Triton |
|--------|------|--------|
| **Abstraction** | Thread-level | Block-level |
| **Indexing** | Manual (`blockIdx.x * blockDim.x + threadIdx.x`) | Automatic (`tl.arange`) |
| **Vectorization** | Explicit (vector types) | Implicit (array operations) |
| **Boundary Checks** | Manual `if` statement | Masked loads/stores |
| **Code Length** | ~15 lines | ~12 lines |
| **Type Safety** | Static (C++) | Dynamic (Python) |

### CUDA Version (Chapter 01)

```cpp
__global__ void add_kernel(float* x, float* y, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = x[idx] + y[idx];
    }
}
```

### Triton Version

```python
@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + y, mask=mask)
```

## Common Patterns

### 1. Grid Size Calculation

```python
# Always round up
grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

# Equivalent to:
grid = ((n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)
```

### 2. Pointer Arithmetic

```python
# Base pointer + offsets
data = tl.load(ptr + offsets, mask=mask)

# Equivalent to:
# for i in offsets where mask[i]:
#     data[i] = ptr[i]
```

### 3. Boundary Handling

```python
# Always use masks for variable-sized inputs
mask = offsets < n_elements

# Alternative: pad inputs to multiple of BLOCK_SIZE
# (avoids mask overhead but wastes memory)
```

## Exercises

1. **Modify Block Size**: Try BLOCK_SIZE = 256, 512, 2048. How does performance change?

2. **Add Constant**: Modify to compute `output = x + y + c` where `c` is a scalar constant.

3. **In-Place Operation**: Modify to compute `x = x + y` (in-place).

4. **Multiple Outputs**: Compute both `sum = x + y` and `diff = x - y`.

5. **Conditional**: Only add elements where `x > 0`.

## Key Takeaways

1. **Block-Based Model**: Each Triton program processes BLOCK_SIZE elements
2. **Masks are Essential**: Handle boundaries and conditional operations
3. **Vectorized Operations**: `x + y` operates on entire blocks
4. **Compile-Time Constants**: Use `tl.constexpr` for optimization
5. **Python Integration**: Seamless PyTorch interop

## Next Steps

- [Example 02 - Softmax](../02_softmax/): Learn reductions and numerical stability
- [Triton Tutorial](https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html): Official vector add tutorial
- Try the exercises above to solidify understanding

## References

- [Triton Language Reference](https://triton-lang.org/main/python-api/triton.language.html)
- [Triton Best Practices](https://triton-lang.org/main/getting-started/tutorials/index.html)
