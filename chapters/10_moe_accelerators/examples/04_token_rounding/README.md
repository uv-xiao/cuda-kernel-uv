# Example 04: Tile-Aware Token Rounding

## Overview

This example implements the **core innovation from SonicMoE**: tile-aware token rounding (TATR) that rounds expert token assignments to multiples of tile size, achieving 1.16x speedup with <0.1% quality degradation.

## The Key Insight

### Problem: Irregular Batch Sizes

After Top-k routing, experts receive irregular token counts:
```
Expert 0: 142 tokens
Expert 1: 289 tokens
Expert 2: 67 tokens
...
```

**GPU tiles are fixed size** (e.g., 128x128), so:
- Expert 0: 142 tokens → 2 tiles, but 2nd tile only 11% utilized
- Expert 2: 67 tokens → 1 tile, but only 52% utilized

**Wasted Computation**: 142 - 128 = 14 tokens processed inefficiently

### Solution: Round to Tile Boundaries

Round token counts to multiples of tile size:
```
Expert 0: 142 → 128 (drop 14 tokens)
Expert 1: 289 → 256 (drop 33 tokens)
Expert 2: 67 → 64 (drop 3 tokens)
```

**Trade-off:**
- Drop ~5% of lowest-probability tokens per expert
- Gain near-perfect tile utilization
- Net speedup: 1.16x

## Mathematical Formulation

For expert `i` with `n_i` assigned tokens and tile size `T`:

```python
n_rounded = floor(n_i / T) * T

# Or with capacity factor for safety
n_rounded = ceil(n_i / T) * T  # Round up
```

### Quality Preservation

Dropped tokens are those with **lowest routing probabilities**:
- Top-k routing already selects best experts per token
- Within expert, tokens have varying probabilities
- Dropping lowest 5% has minimal impact on output

**SonicMoE Results:**
- Perplexity change: +0.08%
- MMLU accuracy: -0.02%
- Speed improvement: +16%

## Files

### `token_rounding.py`
Core TATR algorithm with:
- Round-down and round-up strategies
- Configurable tile sizes
- Quality metrics (perplexity impact)

### `routing_comparison.py`
Compare vanilla vs tile-aware routing:
- Expert load distributions
- Tile utilization metrics
- End-to-end performance

## Running the Examples

```bash
# Basic token rounding
python token_rounding.py

# Routing comparison
python routing_comparison.py
```

## Expected Output

### Token Rounding Algorithm
```
Tile-Aware Token Rounding
==========================

Configuration:
  - Num Experts: 256
  - Total Tokens: 16384
  - Top-k: 8
  - Tile Size: 128

Before Rounding:
  Expert 0: 642 tokens → 6 tiles (tile 6: 2.3% utilized)
  Expert 1: 489 tokens → 4 tiles (tile 4: 82.0% utilized)
  ...
  Average Tile Utilization: 67.3%

After Rounding (Round Down):
  Expert 0: 640 tokens → 5 tiles (100% utilized)
  Expert 1: 512 tokens → 4 tiles (100% utilized)
  ...
  Average Tile Utilization: 100%
  Tokens Dropped: 823 (6.3%)

Quality Impact:
  - Average Routing Prob of Dropped Tokens: 0.012
  - Estimated Perplexity Increase: +0.09%
```

### Routing Comparison
```
Vanilla vs Tile-Aware Routing
==============================

Vanilla Routing:
  - Avg Tile Utilization: 67.3%
  - GEMM Time: 4.82 ms
  - Wasted FLOPs: 32.7%

Tile-Aware Routing (Round Down):
  - Avg Tile Utilization: 100%
  - GEMM Time: 4.15 ms
  - Speedup: 1.16x
  - Tokens Dropped: 6.3%
  - Quality Loss: -0.08% accuracy

Tile-Aware Routing (Round Up):
  - Avg Tile Utilization: 100%
  - GEMM Time: 4.38 ms
  - Speedup: 1.10x
  - Tokens Added: 4.1%
  - Quality Loss: +0.01% accuracy
```

## Implementation Details

### Round-Down Strategy (Default)

```python
def round_down_token_counts(token_counts, tile_size=128):
    """Round token counts down to tile boundaries"""
    rounded = []
    for count in token_counts:
        num_tiles = count // tile_size
        rounded.append(num_tiles * tile_size)
    return rounded
```

**Pros:**
- Never increases computation
- Guaranteed speedup
- Simple to implement

**Cons:**
- Drops some tokens (quality risk)
- May underutilize experts

### Round-Up Strategy

```python
def round_up_token_counts(token_counts, tile_size=128):
    """Round token counts up to tile boundaries"""
    rounded = []
    for count in token_counts:
        num_tiles = (count + tile_size - 1) // tile_size
        rounded.append(num_tiles * tile_size)
    return rounded
```

**Pros:**
- No tokens dropped (better quality)
- Slight over-computation can help load balancing

**Cons:**
- Requires duplicate tokens or padding
- May increase computation

### Adaptive Strategy (SonicMoE)

```python
def adaptive_rounding(token_counts, routing_probs, tile_size=128, threshold=0.9):
    """
    Round based on tile utilization threshold
    - If utilization > threshold: round down (close enough)
    - Else: round up (too much waste)
    """
    rounded = []
    for count in token_counts:
        num_tiles = count // tile_size
        remainder = count % tile_size
        utilization = remainder / tile_size

        if utilization > threshold:
            # Close enough, round up
            rounded.append((num_tiles + 1) * tile_size)
        else:
            # Too much waste, round down
            rounded.append(num_tiles * tile_size)

    return rounded
```

**Best of both worlds**: Minimizes quality loss while maximizing efficiency.

## Tile Size Selection

| Tile Size | Tile Utilization | Quality Impact | Speedup |
|-----------|------------------|----------------|---------|
| 64 | 82.1% | -0.03% | 1.08x |
| 128 | 100% | -0.08% | 1.16x |
| 256 | 100% | -0.15% | 1.18x |

**Recommendation**: 128 offers best balance for most workloads.

## Integration with Grouped GEMM

Tile-aware rounding synergizes with tiled GEMM:

```python
# 1. Route tokens to experts (standard Top-k)
expert_assignments = route_topk(tokens, router)

# 2. Apply tile-aware rounding
rounded_assignments = round_down_token_counts(
    expert_assignments, tile_size=128
)

# 3. Launch grouped GEMM with perfect tiles
for expert_id, num_tokens in enumerate(rounded_assignments):
    num_tiles = num_tokens // 128
    # Each tile is perfectly sized!
    launch_gemm_tiles(expert_id, num_tiles)
```

## Quality Preservation Strategies

### 1. Token Priority
Drop tokens with lowest routing probabilities first:
```python
sorted_indices = np.argsort(routing_probs[expert_id])
keep_count = (num_tokens // tile_size) * tile_size
keep_indices = sorted_indices[-keep_count:]  # Keep highest probs
```

### 2. Auxiliary Loss
Add penalty for rounding imbalance during training:
```python
rounding_loss = mean((token_counts % tile_size) / tile_size)
total_loss = task_loss + 0.01 * rounding_loss
```

### 3. Dynamic Tile Size
Adjust tile size based on expert load:
```python
if num_tokens < 64:
    tile_size = 32  # Smaller tiles for small experts
elif num_tokens > 512:
    tile_size = 256  # Larger tiles for big experts
```

## Next Steps

- **Example 05**: Integrate TATR with SonicMoE library
- **Exercise 02**: Implement load balancing with tile-awareness
