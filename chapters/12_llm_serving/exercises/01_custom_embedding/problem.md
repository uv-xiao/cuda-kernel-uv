# Exercise: Custom Embedding Kernel

## Difficulty: Medium

## Learning Goal

Implement a custom embedding kernel with vocabulary masking for tensor parallelism.

## Problem Statement

Create a CUDA kernel that performs embedding lookup with optional vocabulary range masking. This is essential for vocabulary parallelism where each GPU holds a portion of the embedding table.

## Input Specification

- `weights`: Embedding table, `[vocab_size, embed_dim]` float16
- `indices`: Token IDs, `[batch_size]` int32
- `vocab_range`: Optional `(start, length)` tuple for masking

## Output Specification

- `output`: Embeddings, `[batch_size, embed_dim]` float16

## Requirements

1. **Vectorized Access**: Use `uint4` loads for memory efficiency
2. **Warp-Level Processing**: Each warp handles one embedding
3. **Masking**: If `vocab_range` is provided:
   - Indices outside `[start, start+length)` should produce zeros
   - Implement efficiently without branching in hot path

## Starter Code

```cpp
template <typename T, int EMBED_DIM>
__global__ void embedding_kernel(
    const T* __restrict__ weights,     // [vocab_size, embed_dim]
    const int* __restrict__ indices,   // [batch_size]
    T* __restrict__ output,            // [batch_size, embed_dim]
    int batch_size,
    int vocab_start,                   // 0 if no masking
    int vocab_length                   // vocab_size if no masking
) {
    // TODO: Implement embedding lookup with masking

    // Hints:
    // 1. Compute warp_id from threadIdx.x and blockIdx.x
    // 2. Get index for this warp's embedding
    // 3. Check if index is in range (if masking enabled)
    // 4. Use vectorized copy from weights to output
}
```

## Hints

<details>
<summary>Hint 1: Warp Assignment</summary>

```cpp
const int warp_id = (threadIdx.x / 32) + blockIdx.x * (blockDim.x / 32);
const int lane_id = threadIdx.x % 32;
if (warp_id >= batch_size) return;
```
</details>

<details>
<summary>Hint 2: Masking Logic</summary>

```cpp
int idx = indices[warp_id];
bool valid = (idx >= vocab_start) && (idx < vocab_start + vocab_length);
idx = valid ? (idx - vocab_start) : 0;  // Remap to local index
```
</details>

<details>
<summary>Hint 3: Vectorized Copy</summary>

```cpp
constexpr int ELEMENTS_PER_THREAD = EMBED_DIM / 32 / (sizeof(uint4) / sizeof(T));

const uint4* src = reinterpret_cast<const uint4*>(weights + idx * EMBED_DIM);
uint4* dst = reinterpret_cast<uint4*>(output + warp_id * EMBED_DIM);

#pragma unroll
for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
    int offset = i * 32 + lane_id;
    dst[offset] = valid ? src[offset] : make_uint4(0, 0, 0, 0);
}
```
</details>

## Testing

```bash
python test.py
```

Expected output:
```
Test without masking: PASSED
Test with masking: PASSED
Performance: XX GB/s (YY% of peak)
```

## Performance Target

Achieve at least 80% of theoretical memory bandwidth:
- A100: 2039 GB/s → Target: 1630 GB/s
- RTX 4090: 1008 GB/s → Target: 806 GB/s

## Bonus Challenges

1. **PDL Support**: Add Programmatic Dependent Launch for Hopper GPUs
2. **Async Copy**: Use `cp.async` for better pipelining
3. **Multiple Embeddings**: Extend to handle multiple embedding tables (e.g., token + position)
