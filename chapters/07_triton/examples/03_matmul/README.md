# Example 03 - Matrix Multiplication in Triton

This example demonstrates the progressive optimization of matrix multiplication in Triton, from naive to production-ready implementations.

## Files

1. **matmul_naive.py** - Basic implementation, loads entire K dimension
2. **matmul_blocked.py** - Tiled in all dimensions, handles large matrices
3. **matmul_autotuned.py** - Automatic performance tuning with advanced optimizations

## Learning Progression

### Step 1: Naive Implementation

The simplest approach loads entire rows/columns:

```python
# Each program handles one output tile (BLOCK_M x BLOCK_N)
a = load_entire_row()  # Size: BLOCK_M x K
b = load_entire_col()  # Size: K x BLOCK_N
c = tl.dot(a, b)       # Single matrix multiply
```

**Limitations:**
- K must fit in SRAM (limited to ~1024-2048)
- No data reuse across tiles
- Inefficient for large matrices

### Step 2: Blocked Implementation

Tile K dimension and accumulate:

```python
accumulator = zeros(BLOCK_M, BLOCK_N)
for k_block in range(0, K, BLOCK_K):
    a_block = load(BLOCK_M x BLOCK_K)
    b_block = load(BLOCK_K x BLOCK_N)
    accumulator += tl.dot(a_block, b_block)
```

**Improvements:**
- Handles arbitrary K sizes
- Better memory reuse
- 2-10x faster than naive

### Step 3: Autotuned Implementation

Add automatic optimization:

```python
@triton.autotune(configs=[...])  # Try multiple configs
def matmul_kernel(...):
    # Same algorithm as blocked
    # + Swizzling for cache optimization
    # + Pipeline stages
    # + Warp configuration
```

**Additional Optimizations:**
- Best block sizes selected automatically
- Swizzled tile ordering for L2 cache reuse
- Software pipelining (num_stages)
- Optimal thread block size (num_warps)

## Matrix Multiplication Basics

### Algorithm

```
C[i,j] = sum(A[i,k] * B[k,j] for k in range(K))
```

For matrices:
- A: M x K
- B: K x N
- C: M x N

### Computational Cost

- **Operations:** 2*M*N*K (multiply-add for each element)
- **Memory:** (M*K + K*N + M*N) * sizeof(element)
- **Arithmetic Intensity:** O(K) ops/byte (compute-bound for large K)

### Why It's Important

Matrix multiplication is:
- Foundation of neural networks (>90% of compute)
- Highly optimizable (tensor cores, tiling, etc.)
- Good benchmark for GPU performance

## Tiling Strategy

### 2D Output Tiling

Divide output matrix C into tiles:

```
C = [Tile(0,0)  Tile(0,1)  Tile(0,2) ...]
    [Tile(1,0)  Tile(1,1)  Tile(1,2) ...]
    [...]
```

Each program computes one tile.

### 3D Tiling (with K blocking)

For each output tile, accumulate contributions:

```
Tile(i,j) = sum(A_tile(i,k) @ B_tile(k,j) for k in K_blocks)
```

Visualization:
```
    N
  ┌─────────┐
K │    B    │
  └─────────┘

M ┌─────┐   ┌─────┐
  │  A  │ @ │  C  │
K └─────┘   └─────┘
         M       N
```

## Performance Analysis

### Memory Access Pattern

For one output tile (BLOCK_M x BLOCK_N):

**Naive approach:**
- Load A: BLOCK_M x K elements
- Load B: K x BLOCK_N elements
- Total: (BLOCK_M + BLOCK_N) * K elements

**Blocked approach:**
- Loop iterations: K / BLOCK_K
- Per iteration:
  - Load A: BLOCK_M x BLOCK_K
  - Load B: BLOCK_K x BLOCK_N
- Total: (BLOCK_M + BLOCK_N) * K elements (same!)

**So why is blocking faster?**
1. SRAM capacity: Can fit blocks even for large K
2. Data reuse: Same A/B blocks used for multiple tiles
3. Pipelining: Overlap compute and memory access

### Arithmetic Intensity

```
AI = Operations / Memory_Bytes
   = (2 * BLOCK_M * BLOCK_N * K) / ((BLOCK_M * K + K * BLOCK_N) * 4)
   = (BLOCK_M * BLOCK_N) / (2 * (BLOCK_M + BLOCK_N))
```

For BLOCK_M = BLOCK_N = 128:
- AI = 128 / 4 = 32 ops/byte
- Very high! Matmul is compute-bound for reasonable tile sizes

### Roofline Analysis

GPU performance limited by:
1. **Memory bandwidth** (for low arithmetic intensity)
2. **Compute throughput** (for high arithmetic intensity)

For A100 GPU:
- Memory BW: 2 TB/s
- FP16 Compute: 312 TFLOPS (with tensor cores)

Matmul with AI = 32 ops/byte:
- BW-limited: 2 TB/s * 32 = 64 TFLOPS
- Compute-limited: 312 TFLOPS
- **Bottleneck: Compute** ✓ (good!)

## Autotuning Deep Dive

### Configuration Parameters

| Parameter | Options | Impact |
|-----------|---------|--------|
| BLOCK_SIZE_M | 32, 64, 128, 256 | Output tile height |
| BLOCK_SIZE_N | 32, 64, 128, 256 | Output tile width |
| BLOCK_SIZE_K | 16, 32, 64, 128 | Accumulation chunk size |
| GROUP_SIZE_M | 4, 8, 16 | Cache optimization |
| num_stages | 2, 3, 4, 5 | Pipeline depth |
| num_warps | 2, 4, 8 | Parallelism |

### Trade-offs

**Larger Blocks (128+ vs 32-64):**
- ✓ More compute per memory access
- ✓ Better utilization of tensor cores
- ✗ Higher register pressure
- ✗ Lower occupancy
- **Best for:** Large matrices, compute-bound

**More Stages (4-5 vs 2-3):**
- ✓ Better hiding of memory latency
- ✓ Higher throughput
- ✗ More registers/shared memory
- ✗ May reduce occupancy
- **Best for:** Memory-bound cases

**More Warps (8 vs 2-4):**
- ✓ More parallelism
- ✓ Better occupancy
- ✗ More synchronization overhead
- ✗ More shared memory usage
- **Best for:** Smaller tiles, irregular sizes

### How Autotuning Works

1. **First call:** Triton benchmarks all configs
2. **Caching:** Stores best config for (M, N, K)
3. **Subsequent calls:** Uses cached config
4. **Cache key:** Rounded matrix dimensions

Example:
```python
# First call: benchmarks 8 configs (takes ~1 second)
result1 = matmul(a_2048, b_2048)

# Second call with same dims: instant (uses cache)
result2 = matmul(a_2048, b_2048)

# Different dims: benchmarks again
result3 = matmul(a_4096, b_4096)
```

## Running the Examples

```bash
# Run individually
python matmul_naive.py
python matmul_blocked.py
python matmul_autotuned.py

# Compare all versions (from matmul_autotuned.py)
python matmul_autotuned.py
```

## Expected Performance

On A100 GPU with FP16:

| Size | PyTorch | Naive | Blocked | Autotuned |
|------|---------|-------|---------|-----------|
| 512³ | 0.15 ms | 0.45 ms | 0.18 ms | 0.15 ms |
| 1024³ | 0.95 ms | N/A | 1.10 ms | 0.96 ms |
| 2048³ | 7.2 ms | N/A | 8.1 ms | 7.3 ms |
| 4096³ | 57 ms | N/A | 63 ms | 58 ms |

**Notes:**
- Naive fails for K > ~1024 (SRAM limit)
- Blocked achieves ~90% of PyTorch
- Autotuned achieves ~98-100% of PyTorch
- PyTorch uses cuBLAS (highly optimized)

## Exercises

1. **Custom Blocking**: Modify blocked version to use BLOCK_K=64. How does performance change?

2. **Transpose**: Implement `C = A^T @ B` (transpose A before multiply).

3. **Batch Matmul**: Extend to batch matrix multiplication `C[b] = A[b] @ B[b]`.

4. **Mixed Precision**: Load FP16, accumulate in FP32, store FP16.

5. **Rectangular Tiles**: Try BLOCK_M ≠ BLOCK_N. When is this beneficial?

## Advanced Topics

### Tensor Cores

`tl.dot()` automatically uses tensor cores when:
- Data type is FP16, BF16, or INT8
- Block sizes are multiples of 16
- GPU supports tensor cores (Volta+)

Tensor cores provide:
- 8x throughput vs FP32 CUDA cores
- Specialized matrix multiply-accumulate units
- Key to modern deep learning performance

### Swizzling

Reorder tile processing for better cache reuse:

```python
# Without swizzling: (0,0), (0,1), (0,2), (1,0), (1,1), ...
# With swizzling:    (0,0), (1,0), (2,0), (0,1), (1,1), ...
#                     └──────group──────┘  └────group────┘
```

Benefits:
- Same B tiles loaded for GROUP_SIZE_M output tiles
- Improved L2 cache hit rate
- 5-15% speedup

### Software Pipelining

Overlap computation and memory access:

```
Stage 0: Load A0, B0
Stage 1: Load A1, B1 | Compute C0 = A0 @ B0
Stage 2: Load A2, B2 | Compute C1 = A1 @ B1
...
```

Controlled by `num_stages` parameter.

## Comparison with CUDA

Equivalent CUDA matmul requires:
- Explicit shared memory management
- Manual double buffering for pipelining
- Complex indexing for swizzling
- 200-500 lines of code

Triton autotuned:
- Automatic memory management
- Built-in pipelining support
- ~50 lines of code
- Similar performance

**Development time: 1 day (Triton) vs 1-2 weeks (CUDA)**

## Key Takeaways

1. **Progressive Optimization**
   - Start simple (naive)
   - Add blocking (production)
   - Enable autotuning (performance)

2. **Tiling is Essential**
   - Fits data in SRAM
   - Enables data reuse
   - Improves compute intensity

3. **Autotuning Works**
   - Eliminates manual tuning
   - Adapts to different GPUs
   - Achieves near-optimal performance

4. **Triton Advantage**
   - Much simpler than CUDA
   - Comparable performance
   - Faster development iteration

## Next Steps

- [Example 04 - Fused Operations](../04_fused_ops/): Combine matmul with other ops
- [Example 05 - Autotuning](../05_autotuning/): Deep dive into autotune decorator
- Try implementing matrix transpose, batch matmul

## References

- [Triton Matmul Tutorial](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html)
- [CUDA Matmul Optimization](https://siboehm.com/articles/22/CUDA-MMM)
- [Roofline Model](https://en.wikipedia.org/wiki/Roofline_model)
