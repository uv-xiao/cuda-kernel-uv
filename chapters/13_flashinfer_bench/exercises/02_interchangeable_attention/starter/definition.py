"""
Task 1: Create the Definition for Mini FlashAttention

Complete the Definition below following the FlashInfer-Bench schema.
"""

from flashinfer_bench.data import (
    Definition, AxisConst, AxisVar, TensorSpec, DType
)


# TODO: Complete this Definition
attention_def = Definition(
    name="mini_flash_attention_prefill",
    op_type="gqa_ragged",

    # TODO: Define axes
    # - batch: variable (determined at runtime)
    # - seq_q: variable (query sequence length)
    # - seq_kv: variable (key/value sequence length)
    # - num_heads: constant 32
    # - num_kv_heads: constant 8 (for GQA)
    # - head_dim: constant 128
    axes={
        # Your code here
    },

    # TODO: Define inputs
    # - q: [batch, seq_q, num_heads, head_dim] float16
    # - k: [batch, seq_kv, num_kv_heads, head_dim] float16
    # - v: [batch, seq_kv, num_kv_heads, head_dim] float16
    # - causal: scalar bool
    inputs={
        # Your code here
    },

    # TODO: Define outputs
    # - out: [batch, seq_q, num_heads, head_dim] float16
    outputs={
        # Your code here
    },

    # TODO: Implement reference using PyTorch
    # This should be a simple, correct implementation (not optimized)
    reference='''
import torch

def run(q, k, v, causal):
    """Reference attention implementation using PyTorch."""
    # Your code here
    # Hints:
    # 1. Compute scale = 1 / sqrt(head_dim)
    # 2. Handle GQA by repeating KV heads
    # 3. Compute attention: softmax(Q @ K.T * scale) @ V
    # 4. Apply causal mask if causal=True
    pass
''',

    tags=["stage:prefill", "attention:flash", "exercise:interchangeable"],
)


if __name__ == "__main__":
    # Validate the definition
    print(f"Definition: {attention_def.name}")
    print(f"Op type: {attention_def.op_type}")
    print(f"Axes: {attention_def.axes}")
    print(f"Inputs: {list(attention_def.inputs.keys())}")
    print(f"Outputs: {list(attention_def.outputs.keys())}")
