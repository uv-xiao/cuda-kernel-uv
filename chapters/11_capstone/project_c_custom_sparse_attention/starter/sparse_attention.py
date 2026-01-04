"""
Sparse Attention Implementation

Implement efficient sparse attention using your custom pattern.
"""

import torch
import torch.nn as nn
from typing import Optional
import math

from sparse_pattern import SparseAttentionPattern


class SparseAttention(nn.Module):
    """
    Sparse attention layer using custom sparsity pattern
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        pattern: SparseAttentionPattern,
        dropout: float = 0.0,
        use_custom_kernel: bool = False,
    ):
        """
        Initialize sparse attention

        Args:
            hidden_dim: Hidden dimension
            num_heads: Number of attention heads
            pattern: Sparsity pattern object
            dropout: Dropout probability
            use_custom_kernel: Use custom CUDA kernel if True
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.pattern = pattern
        self.dropout = dropout
        self.use_custom_kernel = use_custom_kernel

        # QKV projection
        self.qkv_proj = nn.Linear(hidden_dim, 3 * hidden_dim, bias=False)

        # Output projection
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # Dropout
        if dropout > 0:
            self.dropout_layer = nn.Dropout(dropout)

        # Scaling
        self.scale = 1.0 / math.sqrt(self.head_dim)

        # TODO: Load custom CUDA kernel if use_custom_kernel is True
        if use_custom_kernel:
            # self.sparse_kernel = load_sparse_attention_kernel()
            pass

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass with sparse attention

        Args:
            x: Input tensor (batch_size, seq_len, hidden_dim)
            attention_mask: Optional additional mask

        Returns:
            Output tensor (batch_size, seq_len, hidden_dim)
        """
        batch_size, seq_len, _ = x.shape

        # Project to Q, K, V
        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)

        # Reshape to (batch, heads, seq_len, head_dim)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute sparse attention
        if self.use_custom_kernel:
            # TODO: Use custom CUDA kernel
            output = self._sparse_attention_custom(q, k, v, attention_mask)
        else:
            # Use PyTorch with masking
            output = self._sparse_attention_pytorch(q, k, v, attention_mask)

        # Reshape and project
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        output = self.out_proj(output)

        if self.dropout > 0:
            output = self.dropout_layer(output)

        return output

    def _sparse_attention_pytorch(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Sparse attention using PyTorch (with masking)

        Args:
            q, k, v: Query, key, value (batch, heads, seq_len, head_dim)
            attention_mask: Optional mask

        Returns:
            Output (batch, heads, seq_len, head_dim)
        """
        batch_size, num_heads, seq_len, head_dim = q.shape

        # Generate sparsity mask
        sparse_mask = self.pattern.generate_mask(seq_len, device=q.device)

        # Expand mask for batch and heads
        sparse_mask = sparse_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)
        sparse_mask = sparse_mask.expand(batch_size, num_heads, -1, -1)

        # Combine with attention_mask if provided
        if attention_mask is not None:
            sparse_mask = sparse_mask & attention_mask

        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Apply sparse mask (set masked positions to -inf)
        scores = scores.masked_fill(~sparse_mask, float('-inf'))

        # Softmax
        attn_weights = torch.softmax(scores, dim=-1)

        # Replace NaN (from -inf) with 0
        attn_weights = torch.where(
            torch.isnan(attn_weights),
            torch.zeros_like(attn_weights),
            attn_weights
        )

        # Apply dropout
        if self.dropout > 0:
            attn_weights = self.dropout_layer(attn_weights)

        # Weighted sum of values
        output = torch.matmul(attn_weights, v)

        return output

    def _sparse_attention_custom(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Sparse attention using custom CUDA kernel

        TODO: Implement this function to call your custom kernel

        Args:
            q, k, v: Query, key, value tensors
            attention_mask: Optional mask

        Returns:
            Output tensor
        """
        # TODO: Call custom CUDA kernel
        # Example:
        # output = self.sparse_kernel(q, k, v, pattern_indices, ...)

        raise NotImplementedError("Custom CUDA kernel not implemented")

    def get_attention_map(
        self,
        x: torch.Tensor,
        head_idx: int = 0,
    ) -> torch.Tensor:
        """
        Get attention weights for visualization

        Args:
            x: Input tensor (batch_size, seq_len, hidden_dim)
            head_idx: Which attention head to visualize

        Returns:
            Attention weights (batch_size, seq_len, seq_len)
        """
        with torch.no_grad():
            batch_size, seq_len, _ = x.shape

            # Project to Q, K
            qkv = self.qkv_proj(x)
            q, k, _ = qkv.chunk(3, dim=-1)

            q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

            # Select head
            q = q[:, head_idx]  # (batch, seq_len, head_dim)
            k = k[:, head_idx]

            # Compute scores
            scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

            # Apply sparse mask
            sparse_mask = self.pattern.generate_mask(seq_len, device=x.device)
            sparse_mask = sparse_mask.unsqueeze(0).expand(batch_size, -1, -1)
            scores = scores.masked_fill(~sparse_mask, float('-inf'))

            # Softmax
            attn_weights = torch.softmax(scores, dim=-1)
            attn_weights = torch.where(
                torch.isnan(attn_weights),
                torch.zeros_like(attn_weights),
                attn_weights
            )

            return attn_weights


class EfficientSparseAttention(SparseAttention):
    """
    Optimized sparse attention using sparse tensor operations

    This version uses PyTorch's sparse tensor support for better efficiency
    when sparsity is very high (>90%).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _sparse_attention_pytorch(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Efficient sparse attention using sparse tensors

        TODO: Implement this using sparse tensor operations
        Useful when sparsity is very high (>90%)
        """
        # TODO: Convert to sparse format
        # - Get indices of non-zero elements from pattern
        # - Compute only those attention scores
        # - Use sparse matrix operations

        # For now, fall back to dense masked version
        return super()._sparse_attention_pytorch(q, k, v, attention_mask)


def test_sparse_attention():
    """Test sparse attention implementation"""
    print("Testing Sparse Attention...")

    from sparse_pattern import LocalAttentionPattern

    # Configuration
    batch_size = 2
    seq_len = 512
    hidden_dim = 512
    num_heads = 8

    # Create pattern
    pattern = LocalAttentionPattern(window_size=64)

    # Create attention layer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    attn = SparseAttention(
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        pattern=pattern,
    ).to(device)

    # Create input
    x = torch.randn(batch_size, seq_len, hidden_dim, device=device)

    # Forward pass
    output = attn(x)

    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Sparsity: {pattern.get_sparsity(pattern.generate_mask(seq_len)):.1%}")

    # Test attention map extraction
    attn_map = attn.get_attention_map(x, head_idx=0)
    print(f"  Attention map shape: {attn_map.shape}")

    print("  Test passed!")


def benchmark_sparse_attention():
    """Benchmark sparse vs dense attention"""
    import time

    print("\nBenchmarking Sparse vs Dense Attention...")

    from sparse_pattern import LocalAttentionPattern

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    configs = [
        (512, 64),   # seq_len, window_size
        (1024, 128),
        (2048, 128),
        (4096, 256),
    ]

    for seq_len, window_size in configs:
        print(f"\nSeq Length: {seq_len}, Window: {window_size}")

        hidden_dim = 512
        num_heads = 8
        batch_size = 4

        # Sparse attention
        pattern = LocalAttentionPattern(window_size=window_size)
        sparse_attn = SparseAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            pattern=pattern,
        ).to(device)

        # Dense attention (for comparison)
        dense_attn = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            batch_first=True,
        ).to(device)

        # Create input
        x = torch.randn(batch_size, seq_len, hidden_dim, device=device)

        # Benchmark sparse
        sparse_attn.eval()
        with torch.no_grad():
            # Warmup
            for _ in range(10):
                _ = sparse_attn(x)

            torch.cuda.synchronize()
            start = time.perf_counter()

            num_iterations = 100
            for _ in range(num_iterations):
                _ = sparse_attn(x)

            torch.cuda.synchronize()
            end = time.perf_counter()

            sparse_time = (end - start) / num_iterations

        # Benchmark dense
        dense_attn.eval()
        with torch.no_grad():
            # Warmup
            for _ in range(10):
                _ = dense_attn(x, x, x)

            torch.cuda.synchronize()
            start = time.perf_counter()

            for _ in range(num_iterations):
                _ = dense_attn(x, x, x)

            torch.cuda.synchronize()
            end = time.perf_counter()

            dense_time = (end - start) / num_iterations

        # Results
        speedup = dense_time / sparse_time
        sparsity = pattern.get_sparsity(pattern.generate_mask(seq_len))

        print(f"  Sparse: {sparse_time*1000:.2f} ms")
        print(f"  Dense:  {dense_time*1000:.2f} ms")
        print(f"  Speedup: {speedup:.2f}x")
        print(f"  Sparsity: {sparsity:.1%}")


if __name__ == "__main__":
    test_sparse_attention()
    benchmark_sparse_attention()
