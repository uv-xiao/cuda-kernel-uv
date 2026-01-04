# Exercise 02: SAXPY

## Objective

Implement the SAXPY operation: **Y = α·X + Y**

SAXPY stands for "Scalar A times X Plus Y" and is a fundamental BLAS (Basic Linear Algebra Subprograms) Level 1 operation. This exercise introduces working with scalar parameters alongside vectors.

## Difficulty

**Beginner** - Slightly more complex than vector subtraction due to scalar multiplication

## Prerequisites

- Complete Exercise 01: Vector Subtraction
- Understand kernel parameters
- Familiarity with floating-point arithmetic

## Problem Description

Given:
- Scalar value **α** (alpha)
- Input vector **X** of size **n**
- Input/output vector **Y** of size **n**

Compute:
```
Y[i] = α * X[i] + Y[i]  for all i in [0, n)
```

**Note**: This is an in-place operation - Y is both input and output!

### Example

```
Input:
  α = 2.0
  X = [1.0, 2.0, 3.0, 4.0]
  Y = [5.0, 6.0, 7.0, 8.0]

Computation:
  Y[0] = 2.0 * 1.0 + 5.0 = 7.0
  Y[1] = 2.0 * 2.0 + 6.0 = 10.0
  Y[2] = 2.0 * 3.0 + 7.0 = 13.0
  Y[3] = 2.0 * 4.0 + 8.0 = 16.0

Output:
  Y = [7.0, 10.0, 13.0, 16.0]
```

## Mathematical Background

SAXPY is used in:
- Linear algebra computations
- Numerical methods (conjugate gradient, etc.)
- Machine learning (gradient updates)
- Physics simulations

The operation combines:
1. **Scalar-vector multiplication**: α·X
2. **Vector addition**: (α·X) + Y

## Requirements

Your implementation must:

1. **Accept Scalar Parameter**
   - Pass alpha as a parameter to the kernel
   - Apply it to each element of X

2. **Implement Kernel**
   - Write `saxpyKernel` that computes Y[i] = α * X[i] + Y[i]
   - Handle the in-place update correctly
   - Include bounds checking

3. **Memory Management**
   - Allocate memory for X and Y
   - Y needs both initial values AND will store results
   - Only need to copy Y back (X doesn't change)

4. **Validate Results**
   - CPU reference implementation
   - Verify GPU matches CPU
   - Test with different alpha values

5. **Test Edge Cases**
   - α = 0 (Y should remain unchanged)
   - α = 1 (Y = X + Y, simple addition)
   - α = -1 (Y = -X + Y, subtraction)

## Function Signatures

### Kernel (Device Code)

```cuda
__global__ void saxpyKernel(float alpha, const float* x, float* y, int n)
{
    // TODO: Implement this
}
```

### CPU Reference (Host Code)

```cuda
void saxpyCPU(float alpha, const float* x, float* y, int n)
{
    // TODO: Implement this
}
```

## Starter Code

See `starter.cu` for a template with TODO markers.

Key differences from previous exercises:
- Scalar parameter (alpha)
- In-place operation (Y is modified)
- Only need to copy Y back to host

## Testing

### Manual Testing

```bash
# Compile
nvcc -o saxpy solution.cu

# Run with default parameters
./saxpy

# Run with custom size and alpha
./saxpy 100000 2.5
```

### Automated Testing

```bash
python3 test.py

# Expected output:
# Test 1 (α=2.0, small): PASSED
# Test 2 (α=1.0, medium): PASSED
# Test 3 (α=-1.0, large): PASSED
# Test 4 (α=0.0, edge case): PASSED
# All tests passed!
```

## Expected Output

```
=== CUDA SAXPY (Y = α·X + Y) ===

Vector size: 1000000 elements
Alpha (α): 2.500000

Allocating and initializing memory...
  X initialized with random values
  Y initialized with random values

Transferring data to device...
Launching kernel...
  Grid: 3907 blocks
  Block: 256 threads
  Computing: Y = 2.50 * X + Y

Transferring results back...
Computing CPU reference...
Verifying results...

SUCCESS: GPU results match CPU!

Sample results (first 5):
  Y[0] = 2.50 * 0.8415 + 0.9093 = 3.0130 (CPU: 3.0130)
  Y[1] = 2.50 * 0.5403 + 0.5985 = 1.9492 (CPU: 1.9492)
  ...

Cleanup complete.
```

## Hints

1. **Kernel Implementation**
   ```cuda
   int i = blockIdx.x * blockDim.x + threadIdx.x;
   if (i < n) {
       y[i] = alpha * x[i] + y[i];
   }
   ```

2. **Order of Operations**
   - Read original Y value
   - Multiply alpha by X
   - Add to original Y
   - Write result back to Y

3. **Memory Pattern**
   ```cuda
   // Y is both input and output
   cudaMemcpy(d_y, h_y, bytes, cudaMemcpyHostToDevice);  // Send initial Y
   saxpyKernel<<<grid, block>>>(alpha, d_x, d_y, n);     // Modify Y
   cudaMemcpy(h_y, d_y, bytes, cudaMemcpyDeviceToHost);  // Get modified Y
   ```

4. **CPU Reference**
   ```cuda
   void saxpyCPU(float alpha, const float* x, float* y, int n) {
       for (int i = 0; i < n; i++) {
           y[i] = alpha * x[i] + y[i];
       }
   }
   ```

## Common Mistakes

1. **Wrong Order of Operations**
   ```cuda
   // WRONG: Reads modified Y instead of original
   y[i] = y[i] + alpha * x[i];  // This works but less clear

   // CORRECT: Shows intent clearly
   y[i] = alpha * x[i] + y[i];
   ```

2. **Forgetting In-Place Update**
   ```cuda
   // WRONG: Doesn't add original Y
   y[i] = alpha * x[i];

   // CORRECT: Adds to original Y
   y[i] = alpha * x[i] + y[i];
   ```

3. **Not Preserving Original Y for CPU**
   ```cuda
   // Make a copy of Y before GPU modifies it
   memcpy(y_original, h_y, bytes);

   // GPU computation modifies h_y

   // CPU computation uses original Y
   saxpyCPU(alpha, h_x, y_original, n);
   ```

## Validation Criteria

Your solution is correct if:

1. **Correctness**: Results match CPU for various alpha values
2. **Edge Cases**: Works correctly for α=0, α=1, α=-1
3. **In-Place**: Y is properly updated (not overwritten)
4. **Error Handling**: All CUDA calls checked
5. **Memory**: No leaks, proper cleanup

## Test Cases

The test script will verify:

| Test | Alpha | Vector Size | Purpose |
|------|-------|-------------|---------|
| 1 | 2.0 | 1,000 | Basic functionality |
| 2 | 1.0 | 100,000 | Simple addition (Y = X + Y) |
| 3 | -1.0 | 1,000,000 | Subtraction (Y = -X + Y) |
| 4 | 0.0 | 10,000 | Edge case (Y unchanged) |
| 5 | 0.5 | 50,000 | Fractional scalar |

## Extension Challenges

### Easy Extensions

1. **Multiple Alpha Values**
   - Test with α = {0.0, 0.5, 1.0, 2.0, -1.0}
   - Verify each case separately

2. **Add Timing**
   - Measure kernel execution time
   - Compare with CPU performance

### Medium Extensions

3. **Generalized AXPY**
   - Template version that works with double precision
   ```cuda
   template<typename T>
   __global__ void axpyKernel(T alpha, const T* x, T* y, int n)
   ```

4. **Fused Operations**
   - Compute Z = α·X + β·Y (two scalars)
   - Requires additional parameter and vector

### Hard Extensions

5. **Strided SAXPY**
   - Support non-unit stride: `Y[i*stride] = α·X[i*stride] + Y[i*stride]`
   - Useful for matrix rows/columns

6. **Batched SAXPY**
   - Multiple independent SAXPY operations
   - Process multiple vector pairs in parallel

## Learning Outcomes

After completing this exercise, you should:

- ✓ Pass scalar parameters to kernels
- ✓ Implement in-place operations
- ✓ Handle multiple arithmetic operations per thread
- ✓ Understand BLAS-style operations
- ✓ Test edge cases systematically

## Real-World Applications

SAXPY is used in:

1. **Linear Solvers**
   ```
   Conjugate Gradient: p = r + β·p
   ```

2. **Gradient Descent**
   ```
   weights = weights - learning_rate * gradients
   (α = -learning_rate, X = gradients, Y = weights)
   ```

3. **Physics Simulations**
   ```
   velocity = velocity + dt * acceleration
   (α = dt, X = acceleration, Y = velocity)
   ```

4. **Image Processing**
   ```
   blended = α·image1 + (1-α)·image2
   ```

## Performance Considerations

SAXPY is **memory-bound**:
- Reads: 2n values (X and Y)
- Writes: 1n values (Y)
- Compute: 2 operations per element (multiply, add)
- **Arithmetic Intensity**: 2 ops / 3 memory accesses = 0.67 ops/byte

For modern GPUs:
- Compute: 10-30 TFLOPS
- Memory: 500-900 GB/s
- **Bottleneck**: Memory bandwidth

This means:
- Increasing compute won't help
- Focus on memory access patterns
- Coalescing is critical (covered in later chapters)

## Common Questions

**Q: Why is Y both input and output?**

A: SAXPY is defined as an update operation. This is common in numerical algorithms where you're iteratively refining a solution.

**Q: What if I want to keep the original Y?**

A: You'd need a separate output vector Z, computing `Z = α·X + Y`. That would be AXPBY.

**Q: Does the order of operations matter?**

A: For exact results, yes (due to floating-point precision). But for correctness, `α·X + Y` and `Y + α·X` are equivalent.

**Q: What's the difference between SAXPY and DAXPY?**

A: SAXPY uses single precision (float), DAXPY uses double precision (double). The 'S' and 'D' prefixes come from BLAS naming.

## Resources

- **BLAS Documentation**: [SAXPY](http://www.netlib.org/lapack/explore-html/d8/daf/saxpy_8f.html)
- **CUBLAS**: NVIDIA's GPU-accelerated BLAS library
- **CUDA Samples**: SimpleCUBLAS example

## Next Steps

After completing SAXPY:

1. **Move to Chapter 02**: Memory optimization techniques
2. **Explore CUBLAS**: NVIDIA's optimized library implementation
3. **Implement GEMV**: Matrix-vector multiplication (next level up)

---

**SAXPY is a fundamental building block!** Many complex algorithms are built from simple operations like this. Master it well!
