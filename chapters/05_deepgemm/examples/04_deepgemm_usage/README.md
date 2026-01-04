# Using DeepGEMM Library

## Overview

This directory demonstrates how to use DeepGEMM, the production FP8 GEMM library from DeepSeek. DeepGEMM provides highly optimized kernels for:
- Dense FP8 GEMM
- Grouped GEMM for MoE
- Fine-grained scaling support

## Installation

### Prerequisites

```bash
# CUDA Toolkit 12.0+
nvidia-smi

# Python 3.8+
python --version

# PyTorch with CUDA support
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### Install DeepGEMM

```bash
# Clone DeepSeek-V3 repository
git clone https://github.com/deepseek-ai/DeepSeek-V3.git
cd DeepSeek-V3/inference/kernels/DeepGEMM

# Build and install
mkdir build && cd build
cmake .. -DCMAKE_CUDA_ARCHITECTURES="80;86;89;90"
make -j
pip install ..
```

### Verify Installation

```python
import torch
import deepgemm

print(f"DeepGEMM version: {deepgemm.__version__}")
print(f"FP8 support: {deepgemm.has_fp8_support()}")
```

## Examples in This Directory

### 1. dense_example.py
Basic FP8 dense GEMM usage:
- Matrix initialization and quantization
- Computing GEMM with FP8
- Performance comparison with BF16

### 2. moe_example.py
MoE workload with grouped GEMM:
- Token routing simulation
- Grouped GEMM execution
- End-to-end MoE layer

## API Reference

### Dense GEMM

```python
import deepgemm

# Quantize tensors to FP8
A_fp8, scale_A = deepgemm.quantize_fp8(A_bf16, dtype='e4m3')
B_fp8, scale_B = deepgemm.quantize_fp8(B_bf16, dtype='e4m3')

# Compute GEMM
C = deepgemm.fp8_gemm(
    A_fp8, B_fp8,
    scale_A, scale_B,
    out_dtype=torch.bfloat16
)
```

### Grouped GEMM

```python
# Prepare grouped inputs
grouped_inputs = [expert_inputs_i for i in range(num_experts)]
expert_weights = [weights_i for i in range(num_experts)]

# Run grouped GEMM
outputs = deepgemm.grouped_gemm(
    grouped_inputs,
    expert_weights,
    scales_A,
    scales_B,
    out_dtype=torch.bfloat16
)
```

### Fine-Grained Quantization

```python
# Quantize with per-block scaling (default block size: 128)
A_fp8, scales_A = deepgemm.quantize_fp8_fine_grained(
    A_bf16,
    block_size=128,
    dtype='e4m3'
)

# GEMM with fine-grained scales
C = deepgemm.fp8_gemm_fine_grained(
    A_fp8, B_fp8,
    scales_A, scales_B,
    block_size=128
)
```

## Performance Tips

### 1. Batch Size Selection

FP8 Tensor Cores perform best with certain sizes:
```python
# Good sizes (multiples of 256)
M, N, K = 2048, 2048, 2048  # Optimal
M, N, K = 4096, 4096, 4096  # Optimal

# Avoid odd sizes
M, N, K = 2049, 2049, 2049  # Poor performance
```

### 2. Memory Layout

Keep tensors contiguous:
```python
# Good
A = torch.randn(M, K, dtype=torch.bfloat16).contiguous()

# Bad (non-contiguous after transpose)
A = torch.randn(K, M, dtype=torch.bfloat16).t()

# Fix
A = A.contiguous()
```

### 3. Quantization Strategy

Choose block size based on data distribution:
```python
# Normal activations: larger blocks
scales = deepgemm.compute_scales(A, block_size=256)

# Activations with outliers: smaller blocks
scales = deepgemm.compute_scales(A, block_size=64)
```

### 4. Workspace Preallocation

Reuse workspace buffers:
```python
# Create workspace once
workspace = deepgemm.create_workspace(max_M, max_N, max_K)

# Reuse in loop
for batch in batches:
    C = deepgemm.fp8_gemm(..., workspace=workspace)
```

## Benchmarking

### Dense GEMM Benchmark

```bash
python dense_example.py --size 4096 --warmup 10 --iters 100
```

Expected output on H100:
```
Size: 4096x4096x4096
FP8 GEMM: 0.152 ms (1145 TFLOPS)
BF16 GEMM: 0.298 ms (584 TFLOPS)
Speedup: 1.96x
```

### MoE Benchmark

```bash
python moe_example.py --experts 8 --hidden 2048 --tokens 1024
```

Expected output:
```
Grouped GEMM: 0.245 ms (712 TFLOPS)
Padded GEMM: 0.385 ms (453 TFLOPS)
Speedup: 1.57x
```

## Common Issues

### Issue 1: CUDA Out of Memory

```python
# Problem: Allocating too many FP8 tensors at once

# Solution: Process in smaller batches
for i in range(0, len(data), batch_size):
    batch = data[i:i+batch_size]
    output = process_batch(batch)
    torch.cuda.empty_cache()
```

### Issue 2: Numerical Instability

```python
# Problem: Large activation outliers

# Solution: Use smaller quantization blocks
A_fp8, scales = deepgemm.quantize_fp8_fine_grained(
    A, block_size=64  # Smaller blocks handle outliers better
)
```

### Issue 3: Slow First Iteration

```python
# Problem: CUDA kernel compilation on first call

# Solution: Warm up before timing
for _ in range(10):
    deepgemm.fp8_gemm(A_fp8, B_fp8, scale_A, scale_B)
torch.cuda.synchronize()

# Now benchmark
start = time.time()
for _ in range(100):
    deepgemm.fp8_gemm(A_fp8, B_fp8, scale_A, scale_B)
torch.cuda.synchronize()
elapsed = time.time() - start
```

## Integration with Transformer Models

### Example: FP8 Linear Layer

```python
import torch.nn as nn
import deepgemm

class FP8Linear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(
            torch.randn(out_features, in_features, dtype=torch.bfloat16)
        )

        # Quantize weights once (static)
        self.weight_fp8, self.weight_scale = \
            deepgemm.quantize_fp8_fine_grained(self.weight)

    def forward(self, x):
        # Quantize activations (dynamic)
        x_fp8, x_scale = deepgemm.quantize_fp8_fine_grained(x)

        # FP8 GEMM
        output = deepgemm.fp8_gemm_fine_grained(
            x_fp8, self.weight_fp8.t(),
            x_scale, self.weight_scale
        )

        return output
```

### Example: FP8 MoE Layer

```python
class FP8MoELayer(nn.Module):
    def __init__(self, hidden_dim, num_experts, expert_dim):
        super().__init__()
        self.num_experts = num_experts

        # Expert weights
        self.experts_w1 = nn.Parameter(
            torch.randn(num_experts, hidden_dim, expert_dim, dtype=torch.bfloat16)
        )
        self.experts_w2 = nn.Parameter(
            torch.randn(num_experts, expert_dim, hidden_dim, dtype=torch.bfloat16)
        )

        # Router
        self.router = nn.Linear(hidden_dim, num_experts)

    def forward(self, x):
        # Router: assign tokens to experts
        router_logits = self.router(x)
        expert_ids = torch.argmax(router_logits, dim=-1)

        # Group tokens by expert
        grouped_inputs = self.group_by_expert(x, expert_ids)

        # Grouped GEMM for all experts
        expert_outputs = deepgemm.grouped_gemm_moe(
            grouped_inputs,
            self.experts_w1,
            self.experts_w2,
            use_fp8=True
        )

        # Scatter back to original positions
        output = self.scatter(expert_outputs, expert_ids)
        return output
```

## Further Reading

1. **DeepGEMM Documentation:**
   - GitHub: https://github.com/deepseek-ai/DeepSeek-V3/tree/main/inference/kernels/DeepGEMM
   - API Docs: (included in repository)

2. **Related Papers:**
   - DeepSeek-V3 Technical Report
   - FP8 Formats for Deep Learning (NVIDIA)

3. **Tutorials:**
   - NVIDIA TransformerEngine FP8 Guide
   - CUTLASS FP8 Examples

## Running the Examples

```bash
# Make sure DeepGEMM is installed
cd /home/uvxiao/cuda-kernel-tutorial/chapters/05_deepgemm/examples/04_deepgemm_usage

# Run dense GEMM example
python dense_example.py

# Run MoE example
python moe_example.py
```

Note: These examples require DeepGEMM to be properly installed. If you don't have access to the library, study the code structure to understand the API design and implement similar interfaces in your own kernels.
