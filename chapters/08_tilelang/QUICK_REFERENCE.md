# TileLang Quick Reference

A cheat sheet for TileLang syntax and common patterns.

## Basic Kernel Structure

```python
import tilelang as T

@T.prim_func
def my_kernel(
    A: T.Buffer((M, N), "float16"),  # Input buffer
    B: T.Buffer((M, N), "float32")   # Output buffer
):
    with T.block("root"):
        # Kernel code here
        pass
```

## Data Types

```python
"float32"   # 32-bit float
"float16"   # 16-bit float (half)
"bfloat16"  # BFloat16
"int32"     # 32-bit integer
"int8"      # 8-bit integer
"bool"      # Boolean
```

## Thread Organization

```python
# 1D thread organization
tx = T.thread_binding(0, BLOCK_SIZE, "threadIdx.x")
bx = T.thread_binding(0, NUM_BLOCKS, "blockIdx.x")

# 2D thread organization
tx = T.thread_binding(0, TILE_X, "threadIdx.x")
ty = T.thread_binding(0, TILE_Y, "threadIdx.y")
bx = T.thread_binding(0, NUM_BLOCKS_X, "blockIdx.x")
by = T.thread_binding(0, NUM_BLOCKS_Y, "blockIdx.y")

# 3D (add blockIdx.z, threadIdx.z)
```

## Memory Allocation

```python
# Shared memory (cooperative across thread block)
A_shared = T.alloc_shared([BLOCK_M, BLOCK_K], "float16")

# Register fragment (per-thread)
A_frag = T.alloc_fragment([M, K], "float16")

# Accumulator (higher precision)
C_frag = T.alloc_fragment([M, N], "float32")
```

## Memory Operations

```python
# Copy data
T.copy(source, destination)

# Fill with value
T.fill(fragment, 0.0)

# Synchronize threads
T.sync_threads()
```

## Loops

```python
# Serial loop (sequential)
for i in T.serial(N):
    # Loop body

# Parallel loop (mapped to threads)
for i in T.thread_binding(0, N, "threadIdx.x"):
    # Loop body

# Vectorized loop
for i in T.vectorized(4):
    # Processes 4 elements at once
```

## Compute Operations

```python
# Matrix multiplication (uses Tensor Cores)
T.gemm(A_frag, B_frag, C_frag, transpose_A=False, transpose_B=False)

# Element-wise operations
c = a + b
c = a * b
c = T.exp(a)
c = T.max(a, b)
c = T.min(a, b)

# Reductions
max_val = T.max(fragment, axis=1)
sum_val = T.reduce_sum(fragment, axis=0)

# Type casting
b = T.cast(a, "float32")
```

## Software Pipelining

```python
# Automatic pipelining
with T.pipeline(num_stages=2):
    for k in T.serial(K // BLOCK_K):
        # Load and compute operations
        # TileLang automatically overlaps them
        T.copy(A[...], A_shared)
        T.gemm(A_frag, B_frag, C_frag)
```

## Buffer Indexing

```python
# 1D indexing
A[idx]

# 2D indexing
A[i, j]

# Slicing
A[start:end, :]
A[:, start:end]

# Block-based indexing
A[block_m * BLOCK_M : (block_m + 1) * BLOCK_M, :]
```

## Conditionals

```python
# If statement
if condition:
    # True branch
else:
    # False branch

# Ternary operator
result = value_if_true if condition else value_if_false
```

## Math Functions

```python
T.exp(x)        # Exponential
T.log(x)        # Natural log
T.sqrt(x)       # Square root
T.sin(x)        # Sine
T.cos(x)        # Cosine
T.max(x, y)     # Maximum
T.min(x, y)     # Minimum
T.abs(x)        # Absolute value
T.pow(x, y)     # Power
```

## Common Patterns

### Vector Addition

```python
@T.prim_func
def vector_add(A: T.Buffer((N,), "float32"),
               B: T.Buffer((N,), "float32"),
               C: T.Buffer((N,), "float32")):
    with T.block("root"):
        tx = T.thread_binding(0, N, "threadIdx.x")
        C[tx] = A[tx] + B[tx]
```

### Matrix Transpose

```python
@T.prim_func
def transpose(A: T.Buffer((M, N), "float32"),
              B: T.Buffer((N, M), "float32")):
    TILE = 32
    with T.block("root"):
        bx = T.thread_binding(0, M // TILE, "blockIdx.x")
        by = T.thread_binding(0, N // TILE, "blockIdx.y")
        tx = T.thread_binding(0, TILE, "threadIdx.x")
        ty = T.thread_binding(0, TILE, "threadIdx.y")

        tile = T.alloc_shared([TILE, TILE], "float32")

        # Load
        tile[ty, tx] = A[by * TILE + ty, bx * TILE + tx]
        T.sync_threads()

        # Store transposed
        B[bx * TILE + ty, by * TILE + tx] = tile[tx, ty]
```

### Reduction

```python
@T.prim_func
def reduce_sum(A: T.Buffer((N,), "float32"),
               result: T.Buffer((1,), "float32")):
    BLOCK_SIZE = 256
    with T.block("root"):
        tx = T.thread_binding(0, BLOCK_SIZE, "threadIdx.x")
        shared = T.alloc_shared([BLOCK_SIZE], "float32")

        # Each thread loads
        shared[tx] = A[tx]
        T.sync_threads()

        # Tree reduction
        stride = BLOCK_SIZE // 2
        while stride > 0:
            if tx < stride:
                shared[tx] += shared[tx + stride]
            T.sync_threads()
            stride //= 2

        if tx == 0:
            result[0] = shared[0]
```

### GEMM (Basic)

```python
@T.prim_func
def gemm(A: T.Buffer((M, K), "float16"),
         B: T.Buffer((K, N), "float16"),
         C: T.Buffer((M, N), "float32")):
    BLOCK_M, BLOCK_N, BLOCK_K = 128, 128, 32

    with T.block("root"):
        # Shared memory
        A_shared = T.alloc_shared([BLOCK_M, BLOCK_K], "float16")
        B_shared = T.alloc_shared([BLOCK_K, BLOCK_N], "float16")

        # Fragments
        A_frag = T.alloc_fragment([BLOCK_M, BLOCK_K], "float16")
        B_frag = T.alloc_fragment([BLOCK_K, BLOCK_N], "float16")
        C_frag = T.alloc_fragment([BLOCK_M, BLOCK_N], "float32")

        T.fill(C_frag, 0.0)

        for k in T.serial(K // BLOCK_K):
            T.copy(A[...], A_shared)
            T.copy(B[...], B_shared)

            T.copy(A_shared, A_frag)
            T.copy(B_shared, B_frag)

            T.gemm(A_frag, B_frag, C_frag)

        T.copy(C_frag, C[...])
```

## Compilation and Execution

```python
# Compile kernel
mod = T.compile(my_kernel, target="cuda")

# Execute
mod(tensor_a, tensor_b)

# With PyTorch tensors
import torch
A = torch.randn(M, N, device="cuda", dtype=torch.float16)
B = torch.zeros(M, N, device="cuda", dtype=torch.float32)
mod(A, B)
```

## Debugging Tips

```python
# Print generated code
mod = T.compile(kernel, target="cuda")
print(mod.imported_modules[0].get_source())

# Check buffer shapes
assert A.shape == (M, N), "Wrong shape!"

# Use CPU for debugging
mod = T.compile(kernel, target="llvm")  # CPU execution
```

## Performance Tips

1. **Use Tensor Cores**: `T.gemm()` for FP16 matrix multiplication
2. **Pipeline**: Wrap loops with `T.pipeline(num_stages=2)`
3. **Shared Memory**: Use for data reuse across threads
4. **Vectorize**: Load/store multiple elements per thread
5. **Avoid Bank Conflicts**: Pad shared memory dimensions
6. **Minimize Sync**: Reduce `T.sync_threads()` calls when safe

## Common Tile Sizes

```python
# For GEMM (good for most GPUs)
BLOCK_M = 128
BLOCK_N = 128
BLOCK_K = 32

# For FlashAttention
BLOCK_M = 64  # Query tile
BLOCK_N = 64  # Key/Value tile

# For reductions
BLOCK_SIZE = 256  # Threads per block
```

## Error Messages

| Error | Cause | Solution |
|-------|-------|----------|
| "out of shared memory" | Too much shared memory | Reduce tile sizes |
| "invalid buffer access" | Out of bounds indexing | Add bounds checking |
| "type mismatch" | Wrong dtype | Use `T.cast()` |
| "synchronization error" | Missing sync | Add `T.sync_threads()` |

## Resources

- **Documentation**: https://microsoft.github.io/BitBLAS/tilelang/
- **Examples**: https://github.com/microsoft/BitBLAS/tree/main/examples
- **Issues**: https://github.com/microsoft/BitBLAS/issues

---

**Keep this reference handy while coding!**
