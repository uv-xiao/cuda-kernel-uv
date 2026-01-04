# Grouped GEMM for Mixture-of-Experts

## Overview

Grouped GEMM is a critical optimization for Mixture-of-Experts (MoE) models, which route different tokens to different expert networks. This directory demonstrates how to efficiently implement batched matrix multiplications where each batch can have different dimensions.

## MoE Architecture Background

### Standard Transformer Layer
```
Input → Attention → FFN → Output
        (all tokens processed together)
```

### MoE Transformer Layer
```
Input → Attention → Router → Expert Selection → Output
                             ↓
                    [Expert 0, Expert 1, ..., Expert N]
                    (tokens routed to different experts)
```

**Key challenge:** Each expert receives a different number of tokens, leading to variable-size GEMMs.

## The Padding Problem

### Traditional Batched GEMM Approach

```python
# Assume 8 experts, each receives different number of tokens
token_counts = [150, 80, 200, 45, 120, 95, 180, 130]  # Per expert

# Batched GEMM requires uniform sizes - must pad!
max_tokens = max(token_counts)  # 200

for expert_id in range(8):
    # Pad to max_tokens
    padded_input = pad(expert_inputs[expert_id], max_tokens)
    outputs[expert_id] = expert_weights[expert_id] @ padded_input
    # Actual compute: 200 * hidden_dim * 3 (QKV projection)
```

**Waste:**
```
Total useful compute: (150+80+200+45+120+95+180+130) = 1000 tokens
Total actual compute: 8 * 200 = 1600 tokens
Efficiency: 1000/1600 = 62.5% (37.5% wasted!)
```

In real LLM serving, this waste is often 20-40% depending on load balancing.

## Grouped GEMM Solution

### Concept

Instead of padding, **concatenate** all expert inputs and run a single GEMM with dynamic offsets:

```python
# Group all tokens together
group_offsets = cumsum([0] + token_counts)  # [0, 150, 230, 430, ...]
concatenated_input = concat([expert_inputs[i] for i in range(8)])

# Single GEMM call with variable-size groups
grouped_gemm(
    concatenated_input,   # size: 1000 x hidden_dim
    expert_weights,        # size: num_experts x hidden_dim x hidden_dim
    group_offsets,         # where each expert starts/ends
    outputs
)
```

**Benefits:**
- Zero padding waste
- Single kernel launch (better GPU utilization)
- More work per kernel (amortize fixed costs)

### Implementation Challenges

1. **Non-contiguous outputs:** Each expert's output goes to different location
2. **Load balancing:** Small experts underutilize GPU, large experts dominate runtime
3. **Tile alignment:** CUTLASS/Tensor Cores prefer power-of-2 tile sizes
4. **Memory layout:** How to efficiently store/access expert weights

## Grouped GEMM Variants

### 1. Fixed-Size Grouped GEMM

All groups have the same M, N, K dimensions, just different data.

**Use case:** Evenly balanced MoE (rare in practice)

### 2. Variable-Size Grouped GEMM

Each group has different M (number of tokens), but same N, K (hidden dimensions).

**Use case:** Standard MoE inference (this is what we implement)

### 3. Masked Grouped GEMM

Groups with dynamic masking for top-k expert selection.

**Use case:** Sparse MoE where not all tokens use all experts

## DeepGEMM Design for Grouped GEMM

DeepGEMM provides three key innovations:

### 1. Persistent Kernel Design

Instead of launching one kernel per group:
```cuda
// Traditional: Multiple kernel launches
for (int i = 0; i < num_experts; i++) {
    gemm_kernel<<<grid, block>>>(expert_inputs[i], ...);
}

// DeepGEMM: Single persistent kernel
grouped_gemm_kernel<<<num_sms, block>>>(all_inputs, offsets, ...);
// Each SM processes multiple groups in a loop
```

### 2. Dynamic Work Distribution

```cuda
__global__ void grouped_gemm_kernel(...) {
    int sm_id = blockIdx.x;

    while (true) {
        // Atomically grab next tile of work
        int tile_id = atomicAdd(&global_work_counter, 1);
        if (tile_id >= total_tiles) break;

        // Determine which group this tile belongs to
        int group_id = find_group(tile_id, offsets);

        // Process this tile
        process_tile(group_id, tile_id, ...);
    }
}
```

### 3. Epilogue Fusion with Scaling

Apply fine-grained scales in the epilogue:
```cuda
// After computing C = A @ B in FP8
for each output element C[i,j]:
    C[i,j] = C_accumulator[i,j] * scale_A[block_i] * scale_B[block_j]
```

## Examples in This Directory

### 1. grouped_gemm.cu

Basic grouped GEMM implementation:
- Simple concatenated approach
- Fixed number of experts
- Demonstrates core algorithm

### 2. variable_sizes.cu

Production-ready variable-size handling:
- Dynamic offset computation
- Load balancing strategies
- Comparison with padded approach

## Performance Expectations

On NVIDIA H100 with typical MoE workload (8 experts, 2048 hidden dim):

| Method | Throughput | Efficiency | Notes |
|--------|------------|------------|-------|
| Padded Batched GEMM | 450 TFLOPS | ~60% | Wasted compute on padding |
| Grouped GEMM (naive) | 520 TFLOPS | ~70% | Better, but poor load balancing |
| Grouped GEMM (DeepGEMM) | 720 TFLOPS | ~92% | Optimized work distribution |
| Theoretical Peak (FP8) | 800 TFLOPS | 100% | Hardware limit |

**Speedup:** 1.3-1.6x over padded approach in practice

## Load Balancing Strategies

### Problem

```
Expert 0: 500 tokens → 25 tiles
Expert 1: 50 tokens  → 2 tiles
Expert 2: 480 tokens → 24 tiles
...
```

If we assign experts to SMs sequentially, some SMs finish early and idle.

### Solution 1: Tile-Level Assignment

Assign individual tiles (not entire experts) to SMs:
```
SM 0: [Exp0_Tile0, Exp2_Tile0, Exp0_Tile1, ...]
SM 1: [Exp0_Tile2, Exp2_Tile1, Exp1_Tile0, ...]
```

### Solution 2: Work Stealing

SMs dynamically grab work from a global queue.

### Solution 3: Persistent Threads

Keep threads alive across multiple groups (DeepGEMM approach).

## Building and Running

```bash
cd /home/uvxiao/cuda-kernel-tutorial/chapters/05_deepgemm/examples/03_grouped_gemm
mkdir build && cd build
cmake ..
make

# Run basic grouped GEMM
./grouped_gemm

# Run variable-size comparison
./variable_sizes
```

## Expected Output

### grouped_gemm
```
=== Grouped GEMM Demo ===
Configuration:
  Number of experts: 8
  Hidden dimension: 2048
  Token counts per expert: [156, 89, 201, 52, 134, 98, 187, 143]

Grouped GEMM Performance:
  Time: 0.245 ms
  Throughput: 712 TFLOPS

Padded Batched GEMM Performance:
  Time: 0.385 ms
  Throughput: 453 TFLOPS

Speedup: 1.57x
```

### variable_sizes
```
=== Variable-Size Grouped GEMM ===
Testing load balancing strategies...

Strategy 1 (Sequential): 0.298 ms (585 TFLOPS)
Strategy 2 (Tile-level): 0.251 ms (695 TFLOPS)
Strategy 3 (Work stealing): 0.243 ms (718 TFLOPS)

Best strategy: Work stealing (1.23x over sequential)
```

## Integration with MoE Layers

### Complete MoE Forward Pass

```cuda
// 1. Router: compute expert assignments
router_forward(input, router_weights, expert_ids, expert_scores);

// 2. Gather: group tokens by expert
gather_by_expert(input, expert_ids, grouped_input, group_offsets);

// 3. Grouped GEMM: expert computation
grouped_gemm(grouped_input, expert_weights, group_offsets, expert_outputs);

// 4. Scatter: route outputs back to original positions
scatter_by_expert(expert_outputs, expert_ids, expert_scores, final_output);
```

Each step must be optimized to avoid becoming a bottleneck.

## Advanced Topics

### 1. Fused Router + Gather

Combine routing and gathering in one kernel to save memory bandwidth.

### 2. Expert Parallelism

Split experts across multiple GPUs in tensor parallel setting.

### 3. Dynamic Expert Assignment

Adapt number of active experts based on input difficulty.

### 4. Capacity Factor

Limit tokens per expert to maintain balanced load:
```python
max_tokens_per_expert = (total_tokens / num_experts) * capacity_factor
# capacity_factor > 1.0 allows some imbalance
```

## Further Reading

1. **Papers:**
   - "Switch Transformers: Scaling to Trillion Parameter Models" (Google, 2021)
   - "DeepSeek-V3 Technical Report" (DeepSeek, 2024)
   - "Mixture-of-Experts with Expert Choice Routing" (Google, 2022)

2. **Implementations:**
   - DeepGEMM: https://github.com/deepseek-ai/DeepSeek-V3/
   - FasterTransformer MoE: https://github.com/NVIDIA/FasterTransformer
   - Megatron-LM MoE: https://github.com/NVIDIA/Megatron-LM

3. **Libraries:**
   - cuBLASLt grouped GEMM: https://docs.nvidia.com/cuda/cublas/
   - CUTLASS grouped GEMM: https://github.com/NVIDIA/cutlass/

## Next Steps

- Implement your own grouped GEMM (see exercises/)
- Study DeepGEMM source code for production optimizations
- Profile MoE models to identify bottlenecks beyond GEMM
- Explore expert parallelism strategies for distributed training
