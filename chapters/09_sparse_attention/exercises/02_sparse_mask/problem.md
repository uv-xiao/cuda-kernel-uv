# Exercise 2: Custom Sparse Attention Mask

## Objective

Design and implement a custom sparse attention pattern for a specific use case.

## Background

Different tasks benefit from different sparsity patterns:
- Language modeling: local + some global
- Document QA: question tokens attend to all, others local
- Code understanding: indentation-aware patterns
- Math reasoning: equation-aware patterns

## Your Task

Choose ONE scenario and implement an efficient sparse attention pattern:

### Option 1: Document Question Answering

**Scenario**: Long document with question at the beginning.

**Requirements**:
- Question tokens (first K tokens) attend to ALL document tokens
- Document tokens attend to:
  - All question tokens (K)
  - Local window (256 tokens)
  - Sentence boundaries (if available)

**Expected sparsity**: ~95% for L=4096, K=64

### Option 2: Hierarchical Code Attention

**Scenario**: Source code with indentation levels.

**Requirements**:
- Each token attends to:
  - Same indentation level (throughout file)
  - Parent indentation level (one level up)
  - Local window (128 tokens)

**Expected sparsity**: ~90% for typical code files

### Option 3: Multi-Scale Vision Attention

**Scenario**: Image patches with different resolutions.

**Requirements**:
- Each patch attends to:
  - Same resolution patches (local 8x8 grid)
  - Corresponding patches at lower resolutions
  - Border patches for context

**Expected sparsity**: ~80% for 256x256 image

### Option 4: Your Custom Pattern

Design your own sparse pattern for your application!

## Implementation Requirements

1. **Mask Creation**:
   ```python
   def create_sparse_mask(seq_len, **kwargs):
       """Returns [seq_len, seq_len] boolean mask where 1 = attend."""
       pass
   ```

2. **Efficient Index Generation**:
   ```python
   def get_sparse_indices(seq_len, **kwargs):
       """Returns [seq_len, k] indices of tokens to attend to."""
       pass
   ```

3. **Sparse Attention**:
   ```python
   def sparse_attention(Q, K, V, indices):
       """Compute attention using only indices."""
       pass
   ```

## Evaluation Criteria

1. **Sparsity**: What % of attention is pruned?
2. **Efficiency**: How much faster than dense?
3. **Quality**: How much accuracy is lost (if applicable)?
4. **Practicality**: Is the pattern easy to implement in real systems?

## Starter Code

See `starter.py` for template implementation.

## Testing

Your implementation should:
1. Generate valid sparse mask/indices
2. Run significantly faster than dense attention
3. (Optional) Maintain quality on downstream task

## Bonus Challenges

1. Implement in CUDA for maximum performance
2. Support dynamic pattern based on input content
3. Combine multiple patterns (e.g., local + learned)
4. Benchmark on real task

## Example Output

```
Custom Sparse Attention Pattern: Document QA

Configuration:
  Sequence length: 4096
  Question tokens: 64
  Local window: 256

Sparsity Analysis:
  Total attention pairs: 16,777,216
  Active pairs: 1,245,184
  Sparsity: 92.6%

Performance:
  Dense attention: 38.2 ms
  Sparse attention: 3.1 ms
  Speedup: 12.3x

Quality (if applicable):
  Task: SQuAD QA
  Dense F1: 87.3
  Sparse F1: 86.9
  Degradation: 0.4 points
```

## Resources

- Longformer paper for global + local pattern
- BigBird paper for combined patterns
- Your creativity!
