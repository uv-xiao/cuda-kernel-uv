# Exercise 1: Mini FlashAttention

## Objective

Implement a simplified version of FlashAttention to understand the core tiling and online softmax algorithm.

## Background

FlashAttention avoids materializing the full L×L attention matrix by:
1. Processing Q, K, V in blocks that fit in shared memory
2. Computing softmax incrementally using online statistics
3. Updating output on-the-fly

## Your Task

Implement the `mini_flash_attention` kernel with these specifications:

### Input
- Q, K, V: [batch_size, seq_len, head_dim]
- Block size: Br = 32 (query block), Bc = 32 (key/value block)
- head_dim = 64 (fixed for simplicity)

### Algorithm

```
For each query block Qi (size Br):
  1. Load Qi into shared memory
  2. Initialize Oi = 0, li = 0, mi = -inf

  For each key/value block (Kj, Vj) (size Bc):
    3. Load Kj, Vj into shared memory
    4. Compute Sij = Qi @ Kj^T / sqrt(d)
    5. Update max: mi_new = max(mi, row_max(Sij))
    6. Compute Pij = exp(Sij - mi_new)
    7. Update normalizer: li_new = exp(mi - mi_new) * li + row_sum(Pij)
    8. Update output: Oi = exp(mi - mi_new) * Oi + Pij @ Vj
    9. mi = mi_new, li = li_new

  10. Final normalization: Oi = Oi / li
  11. Write Oi to HBM
```

### Key Requirements

1. **Shared Memory Usage**:
   - Qi: [Br][head_dim]
   - Kj: [Bc][head_dim]
   - Vj: [Bc][head_dim]
   - Sij: [Br][Bc]
   - Total: ~20KB for Br=Bc=32, d=64

2. **Online Softmax**:
   - Track running max (mi) and sum (li) for each row
   - Rescale previous output when new max is found
   - Never store full attention matrix

3. **Numerical Stability**:
   - Always subtract max before exp()
   - Use FP32 for accumulation
   - Handle edge cases (partial blocks)

## Starter Code

See `starter.cu` for the skeleton implementation.

## Testing

Your implementation should:
1. Match PyTorch's output within 1e-5 (FP32)
2. Not materialize L×L attention matrix
3. Complete for L=512 in <5ms on A100

## Hints

1. **Thread organization**: Each thread can handle one row of Qi or compute subset of Sij
2. **Synchronization**: Use `__syncthreads()` after loading shared memory
3. **Edge cases**: Handle when seq_len is not divisible by block size
4. **Reduction**: Use shared memory for row max/sum reductions

## Bonus Challenges

1. Support causal masking
2. Implement backward pass
3. Optimize with warp-level primitives
4. Add FP16 support

## Expected Output

```
Testing Mini FlashAttention...
Config: seq_len=512, head_dim=64, Br=32, Bc=32

Correctness:
  Max error vs PyTorch: 8.42e-06
  Mean error: 1.23e-07
  ✓ PASSED

Performance:
  Time: 1.23 ms
  Memory saved: 1024 KB (attention matrix not materialized)
```

## Resources

- FlashAttention paper: https://arxiv.org/abs/2205.14135
- Online softmax: https://arxiv.org/abs/1805.02867
- Solution available in `solution.cu`
