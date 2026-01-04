# FlashAttention Minimal: Educational Implementation

## Overview

This directory contains a minimal, educational implementation of FlashAttention. The goal is to illustrate the core algorithmic ideas in ~100 lines of well-commented code, prioritizing clarity over performance.

## The FlashAttention Algorithm

### Key Insight

Standard attention materializes the full L×L attention matrix, requiring O(L²) memory. FlashAttention avoids this by:

1. **Tiling** Q, K, V into blocks that fit in SRAM (GPU shared memory)
2. **Online softmax** computing softmax incrementally without storing full scores
3. **Kernel fusion** combining QK^T, softmax, and PV into a single kernel

### Algorithm Overview

```
Input: Q, K, V [B, H, L, d] in HBM
Output: O [B, H, L, d] in HBM

1. Divide Q into blocks Q₁, Q₂, ..., Qₜ of size Bᵣ
2. Divide K, V into blocks K₁, K₂, ..., Kₜ of size Bᶜ
3. Initialize O = 0, ℓ = 0, m = -∞ (output, normalizer, max)

4. For each query block Qᵢ:
   a. Load Qᵢ from HBM to SRAM
   b. Initialize Oᵢ = 0, ℓᵢ = 0, mᵢ = -∞

   For each key/value block (Kⱼ, Vⱼ):
      c. Load Kⱼ, Vⱼ from HBM to SRAM
      d. Compute Sᵢⱼ = Qᵢ Kⱼᵀ (on-chip)
      e. Update statistics: mᵢ_new = max(mᵢ, max(Sᵢⱼ))
      f. Compute Pᵢⱼ = exp(Sᵢⱼ - mᵢ_new) (on-chip)
      g. Update normalizer: ℓᵢ_new = exp(mᵢ - mᵢ_new)·ℓᵢ + sum(Pᵢⱼ)
      h. Update output: Oᵢ = exp(mᵢ - mᵢ_new)·Oᵢ + Pᵢⱼ Vⱼ
      i. Update mᵢ = mᵢ_new, ℓᵢ = ℓᵢ_new

   j. Final normalization: Oᵢ = Oᵢ / ℓᵢ
   k. Write Oᵢ to HBM
```

### Why This Works: Online Softmax

The key is computing softmax incrementally. For softmax over concatenated vectors [x₁, x₂]:

```
Old softmax: softmax(x₁) with m₁ = max(x₁), ℓ₁ = Σexp(x₁ - m₁)

New data arrives: x₂
m₂ = max(x₁, x₂) = max(m₁, max(x₂))
ℓ₂ = Σexp([x₁, x₂] - m₂)
   = exp(m₁ - m₂)·ℓ₁ + Σexp(x₂ - m₂)

Output update:
O_new = (exp(m₁ - m₂)·ℓ₁·O_old + Pᵢⱼ·Vⱼ) / ℓ₂

where Pᵢⱼ = exp(Sᵢⱼ - m₂)
```

This allows us to process K, V in blocks without storing full attention matrix!

## Memory Access Analysis

### Standard Attention

```
HBM accesses per attention layer:
1. Load Q, K, V: 3BHLd
2. Write S = QKᵀ: BHL²
3. Load S: BHL²
4. Write P = softmax(S): BHL²
5. Load P, V: BHL² + BHLd
6. Write O: BHLd

Total: 4BHL² + 4BHLd ≈ O(BHL²) for large L
```

### FlashAttention

```
HBM accesses:
1. Load Q, K, V: 3BHLd
2. Write O: BHLd

Total: 4BHLd = O(BHLd)

SRAM operations:
- Each block of size Bᵣ × Bᶜ processed (L/Bᵣ) × (L/Bᶜ) times
- Total SRAM ops: O(L²d²/M) where M is SRAM size
```

**Speedup**: For L=4096, d=64, M=192KB:
- HBM access reduction: ~60×
- Wall-clock speedup: ~4-8× (due to SRAM compute)

## Block Size Selection

Optimal block sizes depend on:
- **SRAM size**: Blocks must fit in shared memory
- **Occupancy**: Larger blocks → better occupancy
- **Arithmetic intensity**: Balance compute vs memory

For A100 (192KB shared memory per SM):

```python
# Constraints:
# Qᵢ: Bᵣ × d
# Kⱼ: Bᶜ × d
# Vⱼ: Bᶜ × d
# Sᵢⱼ: Bᵣ × Bᶜ
# Pᵢⱼ: Bᵣ × Bᶜ (can reuse S memory)

# Memory: (Bᵣ + 2Bᶜ)d + 2BᵣBᶜ ≤ M
# For d=64, M=192KB:
# Typical: Bᵣ = 64, Bᶜ = 64
# Memory: (64 + 128)·64·4 + 2·64·64·4 = 82KB ✓
```

## Implementation Details

### Flash Forward Kernel Structure

```cpp
__global__ void flash_attention_forward(
    const float* Q, const float* K, const float* V, float* O,
    int B, int H, int L, int d
) {
    // 1. Thread block processes one query block Qᵢ
    int block_idx = blockIdx.x;

    // 2. Allocate shared memory for Q block, K block, V block, and S/P
    __shared__ float Qi[Br][d];
    __shared__ float Kj[Bc][d];
    __shared__ float Vj[Bc][d];
    __shared__ float Sij[Br][Bc];

    // 3. Load Q block into SRAM
    load_block(Q, Qi, block_idx);

    // 4. Initialize output accumulators (in registers)
    float Oi[d] = {0};
    float li = 0.0f;
    float mi = -INFINITY;

    // 5. Loop over K, V blocks
    for (int j = 0; j < num_kv_blocks; j++) {
        // a. Load K, V blocks
        load_block(K, Kj, j);
        load_block(V, Vj, j);
        __syncthreads();

        // b. Compute Sij = Qi @ Kj^T
        matmul(Qi, Kj, Sij);  // In SRAM

        // c. Online softmax update
        float mi_new = max(mi, row_max(Sij));
        float li_old = li;

        // d. Compute Pij = exp(Sij - mi_new)
        elementwise_exp(Sij, mi_new, Sij);  // Reuse memory

        // e. Update normalizer
        float li_new = exp(mi - mi_new) * li_old + row_sum(Sij);

        // f. Update output
        // Oi = exp(mi - mi_new) * Oi + Pij @ Vj
        float scale = exp(mi - mi_new);
        scale_vector(Oi, scale, d);
        matmul_add(Sij, Vj, Oi);  // Oi += Pij @ Vj

        // g. Update statistics
        mi = mi_new;
        li = li_new;
        __syncthreads();
    }

    // 6. Final normalization
    scale_vector(Oi, 1.0f / li, d);

    // 7. Write output to HBM
    store_block(Oi, O, block_idx);
}
```

## Numerical Stability

### Challenge

Computing exp(x - max) can still overflow if values are too large.

### Solution

1. **Always subtract max before exp**:
   ```cpp
   float m = max(scores);
   for (int i = 0; i < n; i++) {
       exp_scores[i] = expf(scores[i] - m);
   }
   ```

2. **Track running max and rescale**:
   ```cpp
   // When new max m_new > old max m_old:
   scale = exp(m_old - m_new);
   output *= scale;  // Rescale old contributions
   normalizer *= scale;
   ```

3. **Use float for accumulation** (even if Q,K,V are half):
   - Accumulate in FP32
   - Cast to FP16 only when writing output

## Backward Pass

FlashAttention uses **recomputation** for backward pass:

```
Forward: Store only O, m, ℓ (not S or P)
Memory: O(BHLd) instead of O(BHL²)

Backward:
1. Recompute S, P on-the-fly in blocks
2. Compute gradients using online algorithm
3. Still O(BHLd) memory
```

Trade-off: ~1.3× more compute, but massive memory savings.

## Comparison with Naive Attention

| Aspect | Naive | FlashAttention |
|--------|-------|----------------|
| HBM access | O(BHL²) | O(BHLd) |
| SRAM compute | Minimal | O(L²d²/M) |
| Memory stored | S, P matrices | Only O, m, ℓ |
| Kernel launches | 3 (QK, softmax, PV) | 1 (fused) |
| Backward memory | O(BHL²) | O(BHLd) |

## Files

### flash_fwd.cu

Minimal forward pass implementation (~100-150 lines):
- Single kernel for entire attention
- Well-commented for educational purposes
- Fixed block sizes for simplicity
- Supports batch, multi-head

### CMakeLists.txt

Build configuration for the kernel.

## Usage

```bash
cd examples/02_flash_attention_minimal
mkdir build && cd build
cmake ..
make
./flash_fwd
```

## Expected Output

```
Running FlashAttention (minimal)...
Config: batch=2, heads=4, seq_len=512, head_dim=64
Block sizes: Br=64, Bc=64

Correctness check:
  Max error vs PyTorch: 1.23e-05
  Mean error: 3.45e-07
  ✓ PASSED

Performance:
  FlashAttention: 0.834 ms
  PyTorch SDPA: 0.621 ms
  Ratio: 1.34x (expected for minimal impl)

Memory:
  Attention matrix (not materialized): 1024 KB
  Working set in SRAM: 82 KB
```

## Optimization Opportunities

This minimal version is ~1.5-2× slower than production FlashAttention because:

1. **No warp-level optimization**: Uses thread-level operations
2. **Simple matmul**: Doesn't use tensor cores
3. **Fixed block sizes**: Not tuned per GPU/problem size
4. **No prefetching**: Loads blocks sequentially
5. **FP32 only**: No half-precision support

See `03_flash_attention_v2` for optimized version.

## Key Takeaways

1. **Tiling breaks L² memory barrier**: Blocks fit in fast SRAM
2. **Online softmax enables fusion**: No need to store full attention matrix
3. **Kernel fusion reduces HBM traffic**: Single pass through data
4. **Recomputation trades compute for memory**: Essential for long sequences

## References

- [FlashAttention Paper](https://arxiv.org/abs/2205.14135)
- [Online Normalizer Calculation](https://arxiv.org/abs/1805.02867)
- [flash-attention-minimal repo](https://github.com/tspeterkim/flash-attention-minimal)
- [Official FlashAttention Implementation](https://github.com/Dao-AILab/flash-attention)

## Next Steps

1. Understand this minimal version thoroughly
2. Trace through the algorithm with small examples (L=4, Br=2, Bc=2)
3. Implement the exercise: `exercises/01_mini_flash`
4. Study optimized version: `03_flash_attention_v2`
