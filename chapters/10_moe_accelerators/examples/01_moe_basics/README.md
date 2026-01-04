# Example 01: MoE Basics

## Overview

This example implements a simple Mixture of Experts (MoE) layer in PyTorch to understand the fundamental architecture before diving into CUDA optimizations.

## MoE Architecture

### Core Components

1. **Experts**: Multiple feed-forward networks (FFNs) operating in parallel
2. **Router**: Selects which experts process each token (Top-k selection)
3. **Gating**: Combines expert outputs using learned weights

### Mathematical Formulation

For input `x` with shape `[seq_len, hidden_dim]`:

```
1. Router computes scores: scores = x @ W_router
2. Select top-k experts: top_k_indices, top_k_weights = topk(softmax(scores), k)
3. For each token, route to k experts:
   expert_outputs = [Expert_i(x) for i in top_k_indices]
4. Weighted combination:
   output = sum(top_k_weights[i] * expert_outputs[i])
```

### Why MoE?

**Advantages:**
- **Scalability**: Add more experts without increasing per-token computation
- **Specialization**: Experts learn different patterns (syntax, semantics, domains)
- **Conditional Computation**: Only activate a subset of parameters

**Challenges:**
- **Load Balancing**: Popular experts get overloaded
- **Communication**: All-to-all token routing across GPUs
- **Memory**: Must store all expert weights even if not all are active

## Files

### `moe_layer.py`
Complete MoE layer implementation with:
- Configurable number of experts and Top-k
- Load balancing auxiliary loss
- Performance profiling

### `expert_routing.py`
Detailed routing mechanisms:
- Top-k selection with temperature scaling
- Capacity-based token dropping
- Expert load statistics

## Running the Examples

```bash
# Basic MoE layer
python moe_layer.py

# Routing analysis
python expert_routing.py
```

## Expected Output

```
MoE Layer Configuration:
  - Hidden Dim: 4096
  - Expert FFN Dim: 14336
  - Num Experts: 8
  - Top-k: 2
  - Batch Size: 32, Sequence Length: 512

Forward Pass:
  - Input: [32, 512, 4096]
  - Router Logits: [32, 512, 8]
  - Expert Assignments: [32, 512, 2]
  - Output: [32, 512, 4096]
  - Time: 12.4ms

Expert Load Distribution:
  Expert 0: 3210 tokens (19.8%)
  Expert 1: 2145 tokens (13.2%)
  Expert 2: 2890 tokens (17.8%)
  ...
  Load Balance Loss: 0.0234
```

## Key Observations

1. **Irregular Workloads**: Experts receive different numbers of tokens
2. **Memory Overhead**: All expert weights loaded even if expert is idle
3. **Sequential Execution**: Experts run one after another (no parallelism yet)

## Next Steps

- **Example 02**: Implement grouped GEMM for parallel expert execution
- **Example 04**: Add tile-aware token rounding to balance loads
