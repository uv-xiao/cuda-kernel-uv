"""
Attention Layer Interface

Implement either Flash Attention or Sparse Attention here.
"""

import torch
import torch.nn as nn
from typing import Optional
import math


class AttentionLayer(nn.Module):
    """
    Multi-head attention layer with optimized kernel

    Choose ONE implementation:
    1. Flash Attention (recommended for dense attention)
    2. Sparse Attention (for long sequences with sparsity pattern)
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        head_dim: int,
        max_seq_len: int = 8192,
        dropout: float = 0.0,
        use_flash: bool = True,
        sparse_pattern: Optional[str] = None,
    ):
        """
        Initialize attention layer

        Args:
            hidden_dim: Hidden dimension
            num_heads: Number of attention heads
            head_dim: Dimension per head
            max_seq_len: Maximum sequence length
            dropout: Dropout probability
            use_flash: Use Flash Attention if True
            sparse_pattern: Sparse attention pattern (if not using flash)
                          Options: 'block_sparse', 'strided', 'local', etc.
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.dropout = dropout
        self.use_flash = use_flash
        self.sparse_pattern = sparse_pattern

        # QKV projection
        self.qkv_proj = nn.Linear(hidden_dim, 3 * num_heads * head_dim, bias=False)

        # Output projection
        self.out_proj = nn.Linear(num_heads * head_dim, hidden_dim, bias=False)

        # Dropout
        if dropout > 0:
            self.dropout_layer = nn.Dropout(dropout)

        # Scaling factor for attention scores
        self.scale = 1.0 / math.sqrt(head_dim)

        # TODO: Initialize custom CUDA kernel
        # Load your compiled CUDA kernel here
        # self.attention_kernel = load_custom_kernel(...)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: Input tensor (batch_size, seq_len, hidden_dim)
            attention_mask: Optional mask (batch_size, seq_len, seq_len)

        Returns:
            Output tensor (batch_size, seq_len, hidden_dim)
        """
        batch_size, seq_len, _ = x.shape

        # TODO: Implement attention forward pass
        # 1. Project to Q, K, V
        # 2. Reshape to (batch, heads, seq_len, head_dim)
        # 3. Compute attention (using your custom kernel)
        # 4. Project output

        # Placeholder implementation (replace with your kernel)
        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)

        # Reshape
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        if self.use_flash:
            # TODO: Call Flash Attention kernel
            # output = flash_attention_forward(q, k, v, self.scale, attention_mask)
            raise NotImplementedError("Flash Attention kernel not implemented")
        else:
            # TODO: Call Sparse Attention kernel
            # output = sparse_attention_forward(q, k, v, self.scale, attention_mask, self.sparse_pattern)
            raise NotImplementedError("Sparse Attention kernel not implemented")

        # Reshape and project output
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        output = self.out_proj(output)

        if self.dropout > 0:
            output = self.dropout_layer(output)

        return output

    def naive_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Naive attention implementation for correctness testing

        Args:
            q, k, v: Query, key, value tensors (batch, heads, seq_len, head_dim)
            attention_mask: Optional mask

        Returns:
            Output tensor (batch, heads, seq_len, head_dim)
        """
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Apply mask if provided
        if attention_mask is not None:
            scores = scores.masked_fill(attention_mask == 0, float('-inf'))

        # Softmax
        attn_weights = torch.softmax(scores, dim=-1)

        # Apply dropout
        if self.dropout > 0:
            attn_weights = self.dropout_layer(attn_weights)

        # Weighted sum of values
        output = torch.matmul(attn_weights, v)

        return output


class FlashAttention(nn.Module):
    """
    Flash Attention implementation

    Implements the Flash Attention algorithm:
    - Tiling for memory efficiency
    - Online softmax
    - Fused kernel for speed
    """

    def __init__(
        self,
        head_dim: int,
        block_size: int = 128,
    ):
        """
        Initialize Flash Attention

        Args:
            head_dim: Dimension per head
            block_size: Block size for tiling (tune based on GPU)
        """
        super().__init__()
        self.head_dim = head_dim
        self.block_size = block_size
        self.scale = 1.0 / math.sqrt(head_dim)

        # TODO: Load CUDA kernel
        # self.kernel = load_flash_attention_kernel()

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        causal: bool = False,
    ) -> torch.Tensor:
        """
        Flash Attention forward pass

        Args:
            q, k, v: Query, key, value (batch, heads, seq_len, head_dim)
            causal: Whether to use causal masking

        Returns:
            Output (batch, heads, seq_len, head_dim)
        """
        # TODO: Implement Flash Attention
        # 1. Tile Q into blocks
        # 2. For each Q block:
        #    - Load K, V blocks
        #    - Compute attention with online softmax
        #    - Update output
        # 3. Return final output

        raise NotImplementedError("Flash Attention kernel not implemented")


class SparseAttention(nn.Module):
    """
    Sparse Attention implementation

    Implements sparse attention with configurable sparsity pattern
    """

    def __init__(
        self,
        head_dim: int,
        pattern: str = 'block_sparse',
        block_size: int = 64,
        stride: int = 128,
    ):
        """
        Initialize Sparse Attention

        Args:
            head_dim: Dimension per head
            pattern: Sparsity pattern ('block_sparse', 'strided', 'local')
            block_size: Block size for sparse pattern
            stride: Stride for strided attention
        """
        super().__init__()
        self.head_dim = head_dim
        self.pattern = pattern
        self.block_size = block_size
        self.stride = stride
        self.scale = 1.0 / math.sqrt(head_dim)

        # TODO: Load sparse attention kernel
        # self.kernel = load_sparse_attention_kernel()

    def create_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Create sparsity mask based on pattern

        Args:
            seq_len: Sequence length
            device: Device to create mask on

        Returns:
            Mask tensor (seq_len, seq_len) with 1s for attended positions
        """
        # TODO: Implement mask creation based on self.pattern
        # Examples:
        # - block_sparse: Block diagonal pattern
        # - strided: Every stride-th position
        # - local: Local window attention

        if self.pattern == 'block_sparse':
            # Block sparse pattern
            raise NotImplementedError("Block sparse mask not implemented")
        elif self.pattern == 'strided':
            # Strided pattern
            raise NotImplementedError("Strided mask not implemented")
        elif self.pattern == 'local':
            # Local window pattern
            raise NotImplementedError("Local mask not implemented")
        else:
            raise ValueError(f"Unknown pattern: {self.pattern}")

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        """
        Sparse Attention forward pass

        Args:
            q, k, v: Query, key, value (batch, heads, seq_len, head_dim)

        Returns:
            Output (batch, heads, seq_len, head_dim)
        """
        # TODO: Implement Sparse Attention
        # 1. Create sparsity mask
        # 2. Compute attention only for non-zero mask entries
        # 3. Use sparse kernel for efficiency

        raise NotImplementedError("Sparse Attention kernel not implemented")


# Helper function to test attention implementation
def test_attention():
    """Test attention layer for correctness"""
    print("Testing Attention Layer...")

    # Configuration
    batch_size = 2
    seq_len = 512
    hidden_dim = 1024
    num_heads = 8
    head_dim = 128

    # Create attention layer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    attn = AttentionLayer(
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        head_dim=head_dim,
        use_flash=True,
    ).to(device)

    # Create random input
    x = torch.randn(batch_size, seq_len, hidden_dim, device=device)

    # Test forward pass
    try:
        output = attn(x)
        print(f"  Input shape: {x.shape}")
        print(f"  Output shape: {output.shape}")
        assert output.shape == x.shape, "Output shape mismatch"
        print("  Test passed!")
    except NotImplementedError as e:
        print(f"  Not implemented: {e}")
    except Exception as e:
        print(f"  Error: {e}")


if __name__ == "__main__":
    test_attention()
