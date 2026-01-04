"""
Reference Implementation - Mini LLM Inference Engine

This is a reference implementation using PyTorch native operations.
Use this to validate your custom kernel implementations.

Note: This is NOT optimized - it's for correctness checking only.
Your custom implementation should outperform this significantly.
"""

import torch
import torch.nn as nn
from typing import Optional
import time
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from starter.utils import LayerNorm, create_causal_mask


class ReferenceAttention(nn.Module):
    """Reference multi-head attention using PyTorch native operations"""

    def __init__(self, hidden_dim: int, num_heads: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = 1.0 / (self.head_dim ** 0.5)

        self.qkv_proj = nn.Linear(hidden_dim, 3 * hidden_dim, bias=False)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, S, H = x.shape

        # Project to Q, K, V
        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)

        # Reshape to (B, num_heads, S, head_dim)
        q = q.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # Softmax
        attn_weights = torch.softmax(scores, dim=-1)

        # Apply attention to values
        output = torch.matmul(attn_weights, v)

        # Reshape and project
        output = output.transpose(1, 2).contiguous().view(B, S, H)
        output = self.out_proj(output)

        return output


class ReferenceFFN(nn.Module):
    """Reference feedforward network"""

    def __init__(self, hidden_dim: int, intermediate_dim: int):
        super().__init__()
        self.up_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.down_proj = nn.Linear(intermediate_dim, hidden_dim, bias=False)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.up_proj(x)
        x = self.activation(x)
        x = self.down_proj(x)
        return x


class ReferenceTransformerBlock(nn.Module):
    """Reference transformer block"""

    def __init__(self, hidden_dim: int, num_heads: int, intermediate_dim: int):
        super().__init__()
        self.attention = ReferenceAttention(hidden_dim, num_heads)
        self.ffn = ReferenceFFN(hidden_dim, intermediate_dim)
        self.norm1 = LayerNorm(hidden_dim)
        self.norm2 = LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Attention with residual
        x = x + self.attention(self.norm1(x), mask)
        # FFN with residual
        x = x + self.ffn(self.norm2(x))
        return x


class ReferenceInferenceEngine(nn.Module):
    """Reference inference engine"""

    def __init__(
        self,
        vocab_size: int = 32000,
        hidden_dim: int = 4096,
        num_layers: int = 32,
        num_heads: int = 32,
        intermediate_dim: int = 11008,
        max_seq_len: int = 8192,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len

        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.position_embedding = nn.Embedding(max_seq_len, hidden_dim)

        # Transformer blocks
        self.layers = nn.ModuleList([
            ReferenceTransformerBlock(hidden_dim, num_heads, intermediate_dim)
            for _ in range(num_layers)
        ])

        # Output
        self.final_norm = LayerNorm(hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, S = input_ids.shape

        # Get embeddings
        token_embeds = self.token_embedding(input_ids)
        positions = torch.arange(S, device=input_ids.device).unsqueeze(0).expand(B, -1)
        position_embeds = self.position_embedding(positions)

        x = token_embeds + position_embeds

        # Pass through layers
        for layer in self.layers:
            x = layer(x, attention_mask)

        # Output projection
        x = self.final_norm(x)
        logits = self.lm_head(x)

        return logits


def test_reference_implementation():
    """Test the reference implementation"""
    print("Testing Reference Implementation...")

    # Small model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = ReferenceInferenceEngine(
        vocab_size=1000,
        hidden_dim=256,
        num_layers=4,
        num_heads=4,
        intermediate_dim=1024,
        max_seq_len=512,
    ).to(device)

    # Test input
    batch_size, seq_len = 2, 128
    input_ids = torch.randint(0, 1000, (batch_size, seq_len), device=device)

    # Forward pass
    print("Running forward pass...")
    output = model(input_ids)

    print(f"  Input shape: {input_ids.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Output range: [{output.min().item():.2f}, {output.max().item():.2f}]")

    # Benchmark
    print("\nBenchmarking...")
    num_iterations = 10
    warmup = 2

    for _ in range(warmup):
        _ = model(input_ids)

    torch.cuda.synchronize()
    start = time.perf_counter()

    for _ in range(num_iterations):
        _ = model(input_ids)

    torch.cuda.synchronize()
    end = time.perf_counter()

    avg_time = (end - start) / num_iterations
    throughput = (batch_size * seq_len) / avg_time

    print(f"  Average time: {avg_time * 1000:.2f} ms")
    print(f"  Throughput: {throughput:.1f} tokens/s")

    print("\nReference implementation test complete!")


if __name__ == "__main__":
    test_reference_implementation()
