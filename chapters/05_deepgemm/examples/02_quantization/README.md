# Quantization Techniques

## Overview

This directory explores quantization techniques essential for FP8 GEMM operations. We focus on **fine-grained scaling**, which maintains numerical accuracy while enabling 2x throughput gains from FP8 hardware.

## Quantization Fundamentals

### Basic Quantization Formula

```
quantized_value = round(original_value / scale_factor)
dequantized_value = quantized_value * scale_factor
```

The scale factor determines the mapping between high-precision (FP32/BF16) and low-precision (FP8) representations.

### Why Scaling Matters

FP8 has limited dynamic range:
- **E4M3:** [-448, 448]
- **E5M2:** [-57344, 57344]

Neural network weights and activations often have values outside these ranges or concentrated in a small subrange. Without scaling, we would either:
1. Saturate (clip) many values → loss of information
2. Use only a small portion of FP8's representable range → wasted precision

## Scaling Strategies

### 1. Per-Tensor Scaling (Coarse-Grained)

**Single scale factor for entire tensor:**
```
scale = max(abs(tensor)) / max_fp8_value
```

**Pros:**
- Minimal memory overhead (1 float per tensor)
- Fast to compute
- Simple implementation

**Cons:**
- Outliers force small scale → poor precision for typical values
- Not robust to activation spikes in specific regions

**Example:**
```
Tensor: [0.1, 0.2, ..., 100.0]  (max = 100.0)
Scale: 100.0 / 448 = 0.223
Quantized: [0, 1, ..., 448]
→ Values near 0.1 have very poor precision (quantized to 0)
```

### 2. Per-Channel Scaling

**One scale factor per output channel:**
```
scale[c] = max(abs(tensor[:, c])) / max_fp8_value
```

**Pros:**
- Better than per-tensor for weight matrices (channels often have different ranges)
- Moderate memory overhead (~1KB for 1024 channels)

**Cons:**
- Still vulnerable to outliers within a channel
- Not ideal for activations (dynamic per-sample)

### 3. Fine-Grained Scaling (Recommended)

**Scale factor per small block (e.g., 128 elements):**
```
For each block of size B:
    scale[block_idx] = max(abs(tensor[block])) / max_fp8_value
```

**Pros:**
- Robust to outliers (isolated to specific blocks)
- Maintains high precision for typical values
- Memory overhead ~1% (for block size 128)
- Hardware-friendly (blocks fit in registers/shared memory)

**Cons:**
- Slightly more complex implementation
- Need to manage scale factors in GEMM kernels

**Example:**
```
Block 1: [0.1, 0.15, 0.2] → scale = 0.2/448 = 0.000446
Block 2: [50, 60, 100]    → scale = 100/448 = 0.223

After quantization:
Block 1: [224, 336, 448]  (high precision!)
Block 2: [224, 269, 448]  (full range used)
```

## Implementation Considerations

### Block Size Selection

Typical choices: 64, 128, 256 elements

**Tradeoffs:**
- Smaller blocks (64): Better outlier isolation, more overhead
- Larger blocks (256): Less overhead, more vulnerability to outliers
- **128 is sweet spot** for most LLM workloads

### Scale Factor Storage

**Option 1: Separate tensor**
```cpp
float* scales;  // size = (N + block_size - 1) / block_size
```

**Option 2: Interleaved with data**
```cpp
struct BlockData {
    fp8_e4m3 values[128];
    float scale;
};
```

Separate storage is more common (better memory coalescing).

### Computing Scale Factors

**Static (offline):**
- Compute once during model quantization
- Used for weights
- Allows calibration over representative data

**Dynamic (online):**
- Compute during forward pass
- Required for activations
- Must be fast (avoid becoming bottleneck)

## Examples in This Directory

### 1. quantize.cu
Basic quantization and dequantization:
- Per-tensor quantization
- Per-channel quantization
- Accuracy comparison

### 2. fine_grained_scaling.cu
Fine-grained scaling implementation:
- Block-wise scale computation
- Quantized GEMM with scaling
- Performance analysis

## Fine-Grained Scaling in GEMM

For matrix multiplication `C = A @ B`:

**Standard approach:**
```
1. Quantize A with scale_A
2. Quantize B with scale_B
3. Compute C_q = A_q @ B_q  (FP8 GEMM)
4. Dequantize: C = C_q * scale_A * scale_B
```

**Fine-grained approach:**
```
1. Quantize A with scale_A[i] per block
2. Quantize B with scale_B[j] per block
3. Compute C_q = A_q @ B_q  (FP8 GEMM)
4. Apply scales in epilogue:
   C[i,j] = C_q[i,j] * scale_A[i] * scale_B[j]
```

The epilogue fusion is critical for performance - we'll explore this in grouped GEMM examples.

## Expected Results

### Accuracy (vs FP32 baseline)

For typical LLM layers (Linear projections):

| Method | Mean Abs Error | Max Abs Error | Inference Accuracy |
|--------|----------------|---------------|-------------------|
| Per-Tensor E4M3 | 0.015 | 2.5 | 95% |
| Per-Channel E4M3 | 0.008 | 1.2 | 98% |
| Fine-Grained E4M3 (B=128) | 0.003 | 0.4 | 99.5% |
| BF16 (reference) | - | - | 100% |

### Performance

On NVIDIA H100:

| Operation | Size | Method | Throughput |
|-----------|------|--------|------------|
| Quantization | 4096x4096 | Per-Tensor | 2.5 TB/s |
| Quantization | 4096x4096 | Fine-Grained (B=128) | 2.1 TB/s |
| GEMM | 4096x4096x4096 | FP8 + Fine-Grained | 750 TFLOPS |
| GEMM | 4096x4096x4096 | BF16 | 400 TFLOPS |

Fine-grained scaling overhead: ~5% (well worth the accuracy gain)

## Building and Running

```bash
cd /home/uvxiao/cuda-kernel-tutorial/chapters/05_deepgemm/examples/02_quantization
mkdir build && cd build
cmake ..
make

# Run quantization examples
./quantize

# Run fine-grained scaling demo
./fine_grained_scaling
```

## Best Practices

### 1. Calibration for Static Quantization (Weights)
```python
# Collect statistics over calibration dataset
max_vals = []
for batch in calibration_data:
    activations = model.forward(batch)
    max_vals.append(activations.abs().max())

# Use 99.9th percentile to handle outliers
scale = np.percentile(max_vals, 99.9) / 448.0
```

### 2. Dynamic Quantization (Activations)
```cuda
// Fast max reduction in shared memory
__shared__ float block_max[NUM_WARPS];

float thread_max = abs(value);
// Warp reduce
for (int offset = 16; offset > 0; offset >>= 1) {
    thread_max = max(thread_max, __shfl_down_sync(0xFFFFFFFF, thread_max, offset));
}

// Block reduce
if (lane_id == 0) {
    block_max[warp_id] = thread_max;
}
__syncthreads();

float scale = block_max[0] / 448.0f;
```

### 3. Handling Zeros and Small Values
```cuda
// Avoid division by zero
float scale = max(max_abs_val, 1e-6f) / 448.0f;

// Or use epsilon
float scale = (max_abs_val + 1e-6f) / 448.0f;
```

### 4. Mixed Precision Strategies
Not all layers need FP8:
- **FP8:** Large matmuls (QKV projection, FFN, output projection)
- **BF16:** LayerNorm, Softmax, small operations
- **FP32:** Reductions, loss computation

## Further Reading

1. **FP8 Quantization Papers:**
   - "FP8 Formats for Deep Learning" (NVIDIA/ARM, 2022)
   - "LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale"

2. **Production Implementations:**
   - TransformerEngine: https://github.com/NVIDIA/TransformerEngine
   - DeepGEMM: https://github.com/deepseek-ai/DeepSeek-V3/tree/main/inference/kernels/DeepGEMM

3. **Quantization Techniques:**
   - GPTQ (post-training quantization)
   - AWQ (activation-aware weight quantization)
   - SmoothQuant (migration of difficulty from activations to weights)

## Next Steps

- Implement fine-grained scaling in your own kernels
- Explore grouped GEMM for MoE workloads (03_grouped_gemm/)
- Study DeepGEMM's production implementation (04_deepgemm_usage/)
