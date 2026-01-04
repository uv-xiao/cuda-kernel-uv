# Exercise 01: Vector Subtraction

## Objective

Implement a CUDA kernel that performs element-wise vector subtraction: `C[i] = A[i] - B[i]`

This exercise reinforces the fundamental CUDA workflow you learned in the vector addition example. You'll practice:
- Kernel implementation
- Memory management
- Data transfer
- Result validation

## Difficulty

**Beginner** - This is a straightforward modification of vector addition

## Prerequisites

- Complete Example 02: Vector Addition
- Understand basic CUDA workflow
- Familiarity with kernel launching syntax

## Problem Description

Given two input vectors **A** and **B** of size **n**, compute the output vector **C** where:

```
C[i] = A[i] - B[i]  for all i in [0, n)
```

### Example

```
Input:
  A = [5.0, 8.0, 3.0, 9.0]
  B = [2.0, 3.0, 1.0, 4.0]

Output:
  C = [3.0, 5.0, 2.0, 5.0]
```

## Requirements

Your implementation must:

1. **Allocate Memory**
   - Host memory for input vectors A, B and output C
   - Device memory for d_A, d_B, d_C

2. **Initialize Data**
   - Initialize A and B with test values
   - Can use random values or sequential values

3. **Implement Kernel**
   - Write `vectorSubtractKernel` that computes C[i] = A[i] - B[i]
   - Include proper bounds checking
   - Calculate correct global thread ID

4. **Transfer Data**
   - Copy A and B from host to device
   - Copy C from device to host

5. **Validate Results**
   - Implement CPU version for verification
   - Compare GPU results with CPU results
   - Report success or failure

6. **Error Handling**
   - Check all CUDA API calls
   - Verify kernel launch succeeded
   - Handle allocation failures

7. **Cleanup**
   - Free all device memory
   - Free all host memory

## Function Signatures

### Kernel (Device Code)

```cuda
__global__ void vectorSubtractKernel(const float* a, const float* b,
                                      float* c, int n)
{
    // TODO: Implement this
}
```

### CPU Reference (Host Code)

```cuda
void vectorSubtractCPU(const float* a, const float* b, float* c, int n)
{
    // TODO: Implement this
}
```

## Starter Code

See `starter.cu` for a template with TODO markers indicating where you need to add code.

The starter code provides:
- Basic structure and includes
- Error checking macro
- Helper functions
- Main function outline with TODOs

## Testing

### Manual Testing

```bash
# Compile
nvcc -o vector_subtract solution.cu

# Run with default size (1M elements)
./vector_subtract

# Run with custom size
./vector_subtract 100000
```

### Automated Testing

```bash
# Run Python test script
python3 test.py

# Should output:
# Test 1 (small vector): PASSED
# Test 2 (medium vector): PASSED
# Test 3 (large vector): PASSED
# All tests passed!
```

## Expected Output

```
=== CUDA Vector Subtraction ===

Vector size: 1000000 elements

Allocating and initializing memory...
Transferring data to device...
Launching kernel...
  Grid: 3907 blocks
  Block: 256 threads
Transferring results back...
Computing CPU reference...
Verifying results...

SUCCESS: GPU results match CPU!

Sample results:
  A[0] - B[0] = C[0]: 0.8415 - 0.2536 = 0.5879
  A[1] - B[1] = C[1]: 0.9093 - 0.4536 = 0.4557
  ...

Cleanup complete.
```

## Hints

1. **Kernel Implementation**
   ```cuda
   int i = blockIdx.x * blockDim.x + threadIdx.x;
   if (i < n) {
       // What operation goes here?
   }
   ```

2. **Grid Size Calculation**
   ```cuda
   int threadsPerBlock = 256;
   int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
   ```

3. **Verification**
   - The only difference from vector add is the operation
   - CPU: `c[i] = a[i] - b[i]`
   - GPU: Same operation in kernel

4. **Common Mistakes**
   - Forgetting bounds check `if (i < n)`
   - Using `+` instead of `-`
   - Wrong cudaMemcpy direction
   - Not checking for errors

## Validation Criteria

Your solution is correct if:

1. **Compilation**: Code compiles without errors
2. **Execution**: Program runs without crashes
3. **Correctness**: GPU results match CPU results within epsilon (1e-5)
4. **Error Handling**: All CUDA calls are checked
5. **Cleanup**: No memory leaks

## Grading Rubric (Self-Assessment)

- **25%**: Kernel correctly implements subtraction
- **25%**: Proper memory management (allocation, transfer, free)
- **20%**: Error checking for all CUDA calls
- **15%**: CPU verification implementation
- **15%**: Code organization and comments

## Extension Challenges

Once you complete the basic exercise, try these:

### Easy Extensions

1. **Add Timing**
   - Measure kernel execution time
   - Compare GPU vs CPU performance
   - Calculate effective bandwidth

2. **Multiple Operations**
   - Compute both addition and subtraction
   - Launch two kernels with same data

### Medium Extensions

3. **Unified Kernel**
   - Single kernel that does add OR subtract based on parameter
   ```cuda
   __global__ void vectorOpKernel(float* a, float* b, float* c,
                                  int n, char op)
   ```

4. **In-Place Operation**
   - Modify to compute `A[i] = A[i] - B[i]` (result in A)
   - Requires only 2 vectors instead of 3

### Hard Extensions

5. **Multi-Vector**
   - Subtract multiple vectors: `D = A - B - C`
   - Requires kernel to read 3 inputs

6. **Fused Operation**
   - Implement: `D[i] = (A[i] - B[i]) * C[i]`
   - Combines subtraction and multiplication

## Learning Outcomes

After completing this exercise, you should be able to:

- ✓ Modify existing CUDA kernels
- ✓ Implement simple element-wise operations
- ✓ Debug CUDA programs
- ✓ Verify GPU computations against CPU
- ✓ Manage CUDA memory lifecycle

## Common Questions

**Q: Why does my GPU give different results than CPU?**

A: Check:
- Are you using the same data for both?
- Did you transfer data correctly (H2D before kernel, D2H after)?
- Are floating-point values within tolerance (use epsilon comparison)?

**Q: What if vector size isn't divisible by block size?**

A: That's why we need `if (i < n)` - extra threads do nothing.

**Q: How do I choose block size?**

A: Use 128, 256, or 512. All are multiples of warp size (32).

**Q: Should I optimize for performance?**

A: For this exercise, correctness is more important. We'll focus on optimization in later chapters.

## Resources

- Review Example 02: Vector Addition
- CUDA Programming Guide: Section 3.2 (Programming Interface)
- Your device_query output for hardware limits

## Next Exercise

After completing this, move to **Exercise 02: SAXPY** which adds scalar multiplication to the mix.

---

**Good luck!** Remember to start with the starter code and fill in the TODOs one at a time. Test frequently!
