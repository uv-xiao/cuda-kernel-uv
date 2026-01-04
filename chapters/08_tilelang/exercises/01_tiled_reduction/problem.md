# Exercise 1: Tiled Reduction

## Objective

Implement a tiled reduction operation using TileLang that computes the sum of a large array using hierarchical tiling across the memory hierarchy.

## Background

Reductions are fundamental operations in deep learning (e.g., computing loss, norms, statistics). Naive reductions suffer from poor performance due to:
1. Limited parallelism (sequential aggregation)
2. Poor memory locality
3. Warp divergence

Tiled reductions solve these problems by:
1. Using shared memory for block-level reductions
2. Minimizing global memory access
3. Maximizing warp-level parallelism

## Problem Statement

Implement `tiled_reduction(A, result)` that computes `result = sum(A)` where:
- `A` is a 1D array of size `N` (e.g., 1048576 elements)
- `result` is a scalar output

### Requirements

1. **Three-level hierarchy**:
   - Thread-level: Each thread accumulates elements
   - Block-level: Shared memory reduction within block
   - Grid-level: Final reduction across blocks

2. **Optimization goals**:
   - Each thread should process multiple elements (e.g., 4-8)
   - Use shared memory for block reduction
   - Avoid warp divergence in reduction tree
   - Achieve >80% of theoretical memory bandwidth

3. **Implementation constraints**:
   - Block size: 256 threads
   - Elements per thread: 4
   - Use TileLang's memory abstractions

## Starter Code Structure

```python
@T.prim_func
def tiled_reduction(
    A: T.Buffer((1048576,), "float32"),
    result: T.Buffer((1,), "float32")
):
    """
    Compute sum of all elements in A.

    Your implementation should:
    1. Have each thread accumulate multiple elements
    2. Use shared memory for block-level reduction
    3. Write partial sums to global memory
    4. Final reduction on CPU or separate kernel
    """
    N = 1048576
    BLOCK_SIZE = 256
    ELEMENTS_PER_THREAD = 4

    with T.block("root"):
        # TODO: Implement tiled reduction
        pass
```

## Hints

1. **Thread-level accumulation**:
   ```python
   # Each thread processes ELEMENTS_PER_THREAD elements
   thread_sum = T.alloc_fragment([1], "float32")
   thread_sum[0] = 0.0

   for i in T.serial(ELEMENTS_PER_THREAD):
       idx = (bx * BLOCK_SIZE + tx) * ELEMENTS_PER_THREAD + i
       if idx < N:
           thread_sum[0] = thread_sum[0] + A[idx]
   ```

2. **Block-level reduction**:
   ```python
   # Use shared memory for parallel reduction
   shared = T.alloc_shared([BLOCK_SIZE], "float32")
   shared[tx] = thread_sum[0]
   T.sync_threads()

   # Tree reduction
   stride = BLOCK_SIZE // 2
   while stride > 0:
       if tx < stride:
           shared[tx] = shared[tx] + shared[tx + stride]
       T.sync_threads()
       stride = stride // 2
   ```

3. **Handle partial blocks**:
   ```python
   # Check bounds when loading
   if idx < N:
       thread_sum[0] = thread_sum[0] + A[idx]
   else:
       thread_sum[0] = 0.0  # Identity element
   ```

## Test Cases

Your implementation should pass these tests:

```python
# Test 1: All ones
A = torch.ones(1048576, device="cuda", dtype=torch.float32)
expected = 1048576.0

# Test 2: Sequential
A = torch.arange(1048576, device="cuda", dtype=torch.float32)
expected = (1048576 * 1048575) / 2

# Test 3: Random
A = torch.randn(1048576, device="cuda", dtype=torch.float32)
expected = A.sum().item()
```

## Performance Target

On a modern GPU (RTX 3090 / A100):
- Array size: 1M elements (4 MB)
- Target time: <0.1 ms
- Memory bandwidth: >80% of theoretical peak

## Bonus Challenges

1. **Warp-level primitives**: Use warp shuffle operations for the final 32 elements
2. **Multi-stage reduction**: Handle very large arrays with multiple kernel launches
3. **Vectorized loads**: Load 4 floats at once (float4)
4. **Template for different types**: Support float16, int32, etc.

## Submission

Your solution should include:
1. `solution.py` - Your TileLang implementation
2. `test.py` - Tests demonstrating correctness
3. Brief explanation of your approach (comments)

## Learning Objectives

After completing this exercise, you should understand:
- Hierarchical reduction patterns
- Shared memory usage for cooperative operations
- Trade-offs between parallelism and synchronization
- Memory bandwidth optimization

Good luck!
