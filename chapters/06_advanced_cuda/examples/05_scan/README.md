# Parallel Prefix Scan (Prefix Sum)

Parallel prefix scan is a fundamental building block for many parallel algorithms.

## What is Scan?

Given input array `[x₀, x₁, x₂, ..., xₙ₋₁]`:

**Inclusive Scan:** `[x₀, x₀+x₁, x₀+x₁+x₂, ..., Σxᵢ]`
**Exclusive Scan:** `[0, x₀, x₀+x₁, ..., Σxᵢ for i<n-1]`

## Applications

1. **Stream Compaction**: Remove elements from array
2. **Radix Sort**: Compute output positions
3. **Sparse Matrix**: Build row/column pointers
4. **Partitioning**: Split array by predicate
5. **Resource Allocation**: Assign IDs/positions

## Building and Running

```bash
mkdir build && cd build
cmake ..
make
./prefix_scan
```
