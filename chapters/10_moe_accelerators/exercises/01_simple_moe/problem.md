# Exercise 01: Implement Simple MoE Forward Pass

## Objective

Implement a complete MoE layer forward pass with:
- Expert routing (Top-k selection)
- Expert computation
- Output aggregation

## Problem Description

Given:
- Input tokens: `[batch_size, seq_len, hidden_dim]`
- Router network: Linear layer mapping to expert logits
- Multiple expert networks (FFN layers)
- Top-k: Number of experts per token

Implement:
1. Router forward pass to get expert selection
2. Token-to-expert assignment
3. Expert computation
4. Weighted aggregation of expert outputs

## Starter Code

See `starter.py` for the template.

## Requirements

### 1. Router Implementation
- Compute router logits for all experts
- Apply softmax to get probabilities
- Select top-k experts per token
- Normalize top-k probabilities

### 2. Expert Assignment
- Group tokens by assigned expert
- Handle variable number of tokens per expert

### 3. Expert Computation
- Process each expert's assigned tokens
- Apply FFN (Linear → GELU → Linear)

### 4. Output Aggregation
- Combine expert outputs using router weights
- Ensure output shape matches input shape

## Expected Behavior

```python
moe = SimpleMoE(hidden_dim=512, ffn_dim=2048, num_experts=4, top_k=2)
x = torch.randn(8, 64, 512)  # [batch, seq_len, hidden]
output = moe(x)

assert output.shape == x.shape
assert not torch.isnan(output).any()
```

## Testing

Run the test suite:
```bash
python test.py
```

Tests check:
- Correct output shape
- No NaN/Inf values
- Load balancing (expert usage is reasonable)
- Router probabilities sum to 1.0

## Hints

1. Use `torch.topk()` for top-k selection
2. Use `torch.where()` to find tokens assigned to each expert
3. Consider creating a helper function for expert computation
4. Remember to normalize router probabilities after top-k selection

## Bonus Challenges

1. Add load balancing auxiliary loss
2. Implement expert capacity constraints
3. Optimize with fused kernels
4. Add dropout for regularization

## Grading Criteria

- **Correctness (60%)**: Tests pass, correct output
- **Code Quality (20%)**: Clean, readable, documented
- **Efficiency (20%)**: Minimal redundant operations

## Time Estimate

- Basic implementation: 30-45 minutes
- With bonus challenges: 60-90 minutes
