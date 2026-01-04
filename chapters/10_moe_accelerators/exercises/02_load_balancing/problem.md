# Exercise 02: Expert Load Balancing

## Objective

Implement load balancing mechanisms to ensure experts are utilized evenly in MoE layers.

## Problem Description

In MoE systems, some experts can become "popular" and receive most tokens, while others remain idle. This creates:
- GPU underutilization (idle experts)
- Increased latency (overloaded experts)
- Training instability (gradient imbalance)

**Your task**: Implement two load balancing strategies:

1. **Auxiliary Loss** - Penalize imbalanced expert usage during training
2. **Expert Capacity** - Enforce hard limits on tokens per expert

## Background

### Load Imbalance Example

Without load balancing:
```
Expert 0: 1842 tokens (28%)
Expert 1: 2341 tokens (36%)
Expert 2: 156 tokens (2%)   <- Underutilized!
Expert 3: 845 tokens (13%)
...
```

With load balancing:
```
Expert 0: 1024 tokens (16%)
Expert 1: 1128 tokens (17%)
Expert 2: 982 tokens (15%)  <- Better!
Expert 3: 1056 tokens (16%)
...
```

## Part 1: Auxiliary Loss

Implement the load balancing loss from GShard/Switch Transformer:

```
L_balance = α * num_experts * Σ(f_i * P_i)

where:
  f_i = fraction of tokens assigned to expert i
  P_i = fraction of routing probability mass for expert i
  α = loss weight (typically 0.01)
```

This encourages experts to have balanced probability mass and token assignments.

## Part 2: Expert Capacity

Implement capacity-based token dropping:

1. Set expert capacity: `capacity = (num_tokens / num_experts) * capacity_factor`
2. Track tokens assigned to each expert
3. Drop tokens when expert reaches capacity
4. Reallocate dropped tokens or skip them

## Requirements

### Auxiliary Loss Implementation
- Compute per-expert token fractions
- Compute per-expert routing probability mass
- Calculate balanced loss
- Add to total loss during training

### Capacity Implementation
- Enforce capacity limits during routing
- Gracefully handle dropped tokens
- Report drop statistics

### Testing
- Verify load balancing improves expert distribution
- Ensure capacity limits are enforced
- Check that auxiliary loss reduces imbalance over training

## Starter Code

See `starter.py` for template with TODOs.

## Expected Behavior

```python
# Without load balancing
moe = MoEWithLoadBalancing(num_experts=8, top_k=2, use_load_balancing=False)
for epoch in range(10):
    output, stats = moe(x)
    # Expert usage remains imbalanced

# With load balancing
moe = MoEWithLoadBalancing(num_experts=8, top_k=2, use_load_balancing=True)
for epoch in range(10):
    output, stats = moe(x)
    # Expert usage becomes more balanced over time
```

## Evaluation Metrics

1. **Gini Coefficient** - Measures inequality (0 = perfect balance, 1 = maximally imbalanced)
2. **Coefficient of Variation** - Std/Mean of expert loads
3. **Min/Max Ratio** - Ratio of least to most loaded expert

## Hints

1. Use `torch.bincount()` to count tokens per expert
2. Track routing probabilities during forward pass
3. For capacity, maintain a counter per expert in the routing loop
4. Test with small batch sizes first to verify correctness

## Bonus Challenges

1. Implement expert dropout during training
2. Add dynamic capacity adjustment
3. Implement expert merging for very imbalanced cases
4. Visualize expert load over training

## Grading Criteria

- **Correctness (50%)**: Load balancing measurably improves distribution
- **Implementation (30%)**: Clean, efficient code
- **Analysis (20%)**: Insightful comparison of strategies

## References

- GShard: https://arxiv.org/abs/2006.16668
- Switch Transformer: https://arxiv.org/abs/2101.03961
- Expert Choice: https://arxiv.org/abs/2202.09368

## Time Estimate

- Auxiliary Loss: 30-40 minutes
- Expert Capacity: 40-50 minutes
- Testing and Analysis: 30 minutes
- **Total**: 90-120 minutes
