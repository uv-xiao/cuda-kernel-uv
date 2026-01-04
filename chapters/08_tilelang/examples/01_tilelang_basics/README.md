# TileLang Basics

This directory contains introductory examples for TileLang, covering fundamental concepts and memory hierarchy abstractions.

## Contents

1. **hello_tilelang.py** - First TileLang kernels
   - Vector addition (basic and tiled)
   - Matrix transpose
   - Parallel reduction
   - Element-wise operations

2. **memory_hierarchy.py** - Memory hierarchy exploration
   - Global, shared, and register memory
   - Tiling strategies
   - Bandwidth benchmarks
   - Cooperative memory operations

## Setup Instructions

### Prerequisites

```bash
# Ensure you have Python 3.8+ and CUDA installed
python --version  # Should be 3.8+
nvcc --version    # Should be 11.4+
```

### Installation

Install TileLang via BitBLAS:

```bash
# Option 1: Install from PyPI
pip install bitblas

# Option 2: Install from source (for latest features)
git clone https://github.com/microsoft/BitBLAS.git
cd BitBLAS
pip install -e .
```

### Additional Dependencies

```bash
pip install torch  # PyTorch for tensor operations and testing
pip install numpy  # For numerical computations
```

### Verify Installation

```python
import tilelang as T
import torch

print(f"TileLang version: {T.__version__}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
```

## Running the Examples

### Hello TileLang

Run all basic examples:

```bash
cd examples/01_tilelang_basics
python hello_tilelang.py
```

Expected output:
```
============================================================
TileLang Hello World Examples
============================================================
Testing vector_add...
✓ vector_add passed
Testing vector_add_tiled...
✓ vector_add_tiled passed
Testing matrix_transpose...
✓ matrix_transpose passed
Testing block_reduce_sum...
✓ block_reduce_sum passed
Testing vector_scalar_op...
✓ vector_scalar_op passed

============================================================
Performance Comparison: TileLang vs PyTorch
============================================================
Vector size: 4096
TileLang:    12.45 μs
PyTorch:     11.23 μs
Overhead:    10.9%
============================================================

✓ All tests passed!
```

### Memory Hierarchy

Explore memory hierarchy:

```bash
python memory_hierarchy.py
```

Expected output:
```
============================================================
TileLang Memory Hierarchy Examples
============================================================
Testing memory_hierarchy_demo...
✓ memory_hierarchy_demo passed
Testing matrix_multiply_shared...
✓ matrix_multiply_shared passed
Testing fragment_operations...
✓ fragment_operations passed
Testing cooperative_copy_demo...
✓ cooperative_copy_demo passed

============================================================
Memory Bandwidth Benchmarks
============================================================
Array size: 1048576 elements (4.19 MB)

Global Memory Only:
  Time:      0.234 ms
  Bandwidth: 35.82 GB/s

With Shared Memory:
  Time:      0.256 ms
  Bandwidth: 32.73 GB/s

Overhead: 9.4%
============================================================
```

## Key Concepts Covered

### 1. Kernel Definition

TileLang kernels are defined using the `@T.prim_func` decorator:

```python
@T.prim_func
def my_kernel(A: T.Buffer((N,), "float32"),
              B: T.Buffer((N,), "float32")):
    with T.block("root"):
        # Kernel code here
        pass
```

### 2. Thread Binding

Map thread indices to computation:

```python
# 1D indexing
tx = T.thread_binding(0, 256, "threadIdx.x")
bx = T.thread_binding(0, num_blocks, "blockIdx.x")

# 2D indexing (for matrices)
tx = T.thread_binding(0, TILE_SIZE, "threadIdx.x")
ty = T.thread_binding(0, TILE_SIZE, "threadIdx.y")
```

### 3. Memory Allocation

Three levels of memory:

```python
# Shared memory (cooperative, on-chip)
shared = T.alloc_shared([BLOCK_SIZE], "float32")

# Register fragment (per-thread, fastest)
fragment = T.alloc_fragment([SIZE], "float16")

# Global memory (implicit via buffers)
# A[idx] = value  # Write to global memory
```

### 4. Synchronization

Coordinate threads within a block:

```python
T.sync_threads()  # Barrier synchronization
```

### 5. Data Types

TileLang supports standard data types:

```python
"float32"   # 32-bit float
"float16"   # 16-bit float (half precision)
"int32"     # 32-bit integer
"int8"      # 8-bit integer
```

### 6. Compilation and Execution

```python
# Compile kernel for CUDA
mod = T.compile(my_kernel, target="cuda")

# Execute
mod(tensor_a, tensor_b)
```

## Memory Hierarchy Summary

| Level | Size (A100) | Latency | Bandwidth | TileLang API |
|-------|-------------|---------|-----------|--------------|
| Registers | 256 KB/SM | ~1 cycle | ~20 TB/s | `T.alloc_fragment()` |
| Shared Memory | 164 KB/SM | ~20 cycles | ~15 TB/s | `T.alloc_shared()` |
| L1 Cache | Unified | ~30 cycles | ~10 TB/s | Automatic |
| L2 Cache | 40 MB | ~200 cycles | ~3 TB/s | Automatic |
| Global (HBM) | 40-80 GB | ~400 cycles | ~1.5 TB/s | `T.Buffer()` |

**Key Insight**: Tiling is essential! Moving from global to shared memory gives ~10× bandwidth improvement.

## Common Patterns

### Vector Operation

```python
@T.prim_func
def vector_op(A: T.Buffer, B: T.Buffer, C: T.Buffer):
    with T.block("root"):
        tx = T.thread_binding(0, N, "threadIdx.x")
        C[tx] = A[tx] + B[tx]
```

### Reduction

```python
@T.prim_func
def reduction(A: T.Buffer, result: T.Buffer):
    BLOCK_SIZE = 256
    with T.block("root"):
        tx = T.thread_binding(0, BLOCK_SIZE, "threadIdx.x")

        shared = T.alloc_shared([BLOCK_SIZE], "float32")
        shared[tx] = A[tx]
        T.sync_threads()

        # Tree reduction
        stride = BLOCK_SIZE // 2
        while stride > 0:
            if tx < stride:
                shared[tx] += shared[tx + stride]
            T.sync_threads()
            stride = stride // 2

        if tx == 0:
            result[0] = shared[0]
```

### Tiled Matrix Operation

```python
@T.prim_func
def tiled_matrix_op(A: T.Buffer, B: T.Buffer):
    TILE_SIZE = 32
    with T.block("root"):
        bx = T.thread_binding(0, N // TILE_SIZE, "blockIdx.x")
        tx = T.thread_binding(0, TILE_SIZE, "threadIdx.x")

        tile = T.alloc_shared([TILE_SIZE], "float32")

        offset = bx * TILE_SIZE
        tile[tx] = A[offset + tx]
        T.sync_threads()

        # Process tile...
        B[offset + tx] = tile[tx]
```

## Performance Tips

1. **Use Shared Memory for Reuse**
   - When data is accessed multiple times by different threads
   - Example: Matrix tiles in GEMM

2. **Minimize Global Memory Access**
   - Load once into shared memory, reuse many times
   - Each global access costs ~400 cycles!

3. **Vectorize Loads/Stores**
   - Use fragments for multiple elements per thread
   - Enables 64-bit or 128-bit memory transactions

4. **Avoid Bank Conflicts**
   - Add padding to shared memory dimensions
   - Example: `[TILE, TILE+1]` instead of `[TILE, TILE]`

5. **Thread Block Size**
   - Typically use multiples of 32 (warp size)
   - Common sizes: 128, 256, 512 threads

## Debugging Tips

### Check Compilation

```python
# Compile and inspect generated code
mod = T.compile(kernel, target="cuda")
print(mod.imported_modules[0].get_source())  # View CUDA source
```

### Verify Results

```python
# Compare with PyTorch reference
result_tilelang = torch.zeros_like(expected)
mod(input_a, input_b, result_tilelang)

result_pytorch = input_a + input_b
assert torch.allclose(result_tilelang, result_pytorch, rtol=1e-5)
```

### Profile Performance

```python
import time

# Warmup
for _ in range(10):
    mod(a, b, c)
torch.cuda.synchronize()

# Benchmark
start = time.time()
for _ in range(100):
    mod(a, b, c)
torch.cuda.synchronize()
elapsed = (time.time() - start) / 100
print(f"Time: {elapsed*1e6:.2f} μs")
```

## Next Steps

After mastering these basics, proceed to:

1. **examples/02_gemm/** - Matrix multiplication with tiling and pipelining
2. **examples/03_attention/** - FlashAttention implementations
3. **examples/04_mla_decoding/** - Production-grade MLA kernels

## Resources

- **TileLang Documentation**: https://microsoft.github.io/BitBLAS/tilelang/
- **BitBLAS Repository**: https://github.com/microsoft/BitBLAS
- **CUDA Programming Guide**: https://docs.nvidia.com/cuda/cuda-c-programming-guide/
- **GPU Memory Hierarchy**: Understanding GPU memory is crucial for optimization

## Common Issues

### Import Error

```
ImportError: cannot import name 'tilelang'
```

**Solution**: Install BitBLAS: `pip install bitblas`

### CUDA Not Available

```
AssertionError: CUDA is not available
```

**Solution**: Ensure PyTorch is installed with CUDA support:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### Out of Memory

```
RuntimeError: CUDA out of memory
```

**Solution**: Reduce tile sizes or batch sizes in examples.

## Questions?

- Open an issue: https://github.com/microsoft/BitBLAS/issues
- Discussion forum: https://github.com/microsoft/BitBLAS/discussions

---

**Happy coding with TileLang!** The abstractions may seem simple, but they unlock powerful optimizations.
