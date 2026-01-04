# FP8 Basics

## Overview

This directory introduces FP8 (8-bit floating point) data types, which are critical for efficient LLM inference on modern GPUs. FP8 provides approximately 2x compute throughput compared to BF16/FP16 while maintaining acceptable accuracy for most LLM workloads.

## FP8 Format Specification

### E4M3 (4-bit Exponent, 3-bit Mantissa)

**Format:** `S EEEE MMM`
- Sign: 1 bit
- Exponent: 4 bits (bias = 7)
- Mantissa: 3 bits

**Properties:**
- Max value: 448
- Min positive normalized: 2^(-6) ≈ 0.015625
- Min positive denormalized: 2^(-9) ≈ 0.001953
- Precision: ~8 representable values per power of 2

**Special values:**
- No infinity (largest exponent used for normal numbers)
- NaN: S 1111 111 (sign bit can vary)

**Use cases:** Forward pass activations, weights

### E5M2 (5-bit Exponent, 2-bit Mantissa)

**Format:** `S EEEEE MM`
- Sign: 1 bit
- Exponent: 5 bits (bias = 15)
- Mantissa: 2 bits

**Properties:**
- Max value: 57344
- Min positive normalized: 2^(-14) ≈ 0.000061
- Min positive denormalized: 2^(-16) ≈ 0.000015
- Precision: ~4 representable values per power of 2

**Special values:**
- Infinity: S 11111 00
- NaN: S 11111 [01, 10, 11]

**Use cases:** Gradients (requires wider dynamic range)

## Format Comparison

| Property | FP8 E4M3 | FP8 E5M2 | BF16 | FP16 |
|----------|----------|----------|------|------|
| Total bits | 8 | 8 | 16 | 16 |
| Exponent bits | 4 | 5 | 8 | 5 |
| Mantissa bits | 3 | 2 | 7 | 10 |
| Max value | 448 | 57344 | 3.4e38 | 65504 |
| Min normal | 0.015625 | 0.000061 | 1.2e-38 | 0.000061 |
| Has infinity | No | Yes | Yes | Yes |
| Precision (bits) | ~10 bits | ~7 bits | ~7 bits | ~11 bits |

**Key insights:**
- E4M3 has better precision in [-448, 448] range
- E5M2 can represent extreme values (gradients with outliers)
- BF16 exponent range matches FP32 (easy conversion)
- FP16 has best precision but limited range

## Conversion Strategies

### 1. Saturation (Default)
```
if (value > max_fp8):
    return max_fp8
elif (value < -max_fp8):
    return -max_fp8
else:
    return round_to_nearest_fp8(value)
```

### 2. Stochastic Rounding
```
frac = value - floor(value)
if (random() < frac):
    return ceil(value)
else:
    return floor(value)
```
Better for training (reduces bias), slower.

### 3. Scaled Conversion
```
scale = max(abs(value)) / max_fp8
fp8_value = round(value / scale)
```
Preserves relative magnitudes, requires storing scale factors.

## Examples in This Directory

### 1. fp8_types.cu
Demonstrates basic FP8 representation:
- Manual bit manipulation for E4M3/E5M2
- Special value handling (NaN, saturation)
- Range and precision experiments

### 2. fp8_conversion.cu
Conversion kernels between formats:
- FP32 → FP8 (E4M3/E5M2)
- FP8 → FP32
- BF16 ↔ FP8
- Performance comparison of conversion strategies

## Building and Running

```bash
cd /home/uvxiao/cuda-kernel-tutorial/chapters/05_deepgemm/examples/01_fp8_basics
mkdir build && cd build
cmake ..
make

# Run FP8 type demonstrations
./fp8_types

# Run conversion benchmarks
./fp8_conversion
```

## Expected Output

### fp8_types
```
FP8 E4M3 Range Test:
  Max value: 448.000000
  Min positive normal: 0.015625
  Min positive denormal: 0.001953125
  Values near 1.0: [0.875, 1.0, 1.125, 1.25]

FP8 E5M2 Range Test:
  Max value: 57344.000000
  Min positive normal: 0.000061
  Min positive denormal: 0.000015
  Values near 1.0: [0.75, 1.0, 1.5, 2.0]
```

### fp8_conversion
```
Conversion Performance (1M elements):
  FP32 -> E4M3: 0.125 ms (8.0 GB/s)
  E4M3 -> FP32: 0.098 ms (10.2 GB/s)
  BF16 -> E4M3: 0.115 ms (8.7 GB/s)

Accuracy Test (Gaussian N(0,1)):
  E4M3 Mean Abs Error: 0.0023
  E5M2 Mean Abs Error: 0.0089
```

## Accuracy Considerations

### When FP8 Works Well
- Normal distributions centered near 0
- Values within [-100, 100] for E4M3
- Relative precision matters more than absolute

### When FP8 Struggles
- Extreme outliers (>1000x difference in magnitudes)
- Tasks requiring high precision (e.g., solving linear systems)
- Values outside typical range

### Mitigation Strategies
1. **Fine-grained scaling:** Normalize per block/channel
2. **Mixed precision:** Keep some tensors in BF16 (e.g., LayerNorm scales)
3. **Outlier handling:** Clip or quantize outliers separately
4. **Calibration:** Collect statistics to determine optimal scale factors

## Hardware Support

### NVIDIA
- **H100/H200:** Native FP8 Tensor Cores (up to 1979 TFLOPS)
- **Ada Lovelace (RTX 40-series):** FP8 support in some SKUs
- **Older GPUs:** Software emulation (no speedup)

### AMD
- **MI300X:** FP8 support (up to 1307 TFLOPS)
- **MI250X and older:** No FP8 support

### Check GPU Support
```bash
nvidia-smi --query-gpu=compute_cap --format=csv
# Need compute capability >= 9.0 for FP8 Tensor Cores
```

## Further Reading

1. **NVIDIA FP8 Primer:** https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/fp8_primer.html
2. **FP8 Formats for Deep Learning (arXiv:2209.05433)**
3. **CUTLASS FP8 Documentation:** https://github.com/NVIDIA/cutlass/blob/main/media/docs/fundamental_types.md

## Next Steps

After understanding FP8 formats:
1. Explore quantization techniques (02_quantization/)
2. Learn how FP8 accelerates GEMM operations
3. Implement fine-grained scaling for accuracy
