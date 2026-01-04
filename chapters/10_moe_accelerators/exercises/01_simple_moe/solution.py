"""
Exercise 01: Simple MoE Forward Pass - Solution

Complete implementation of a basic MoE layer.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class Expert(nn.Module):
    """Single expert FFN"""

    def __init__(self, hidden_dim: int, ffn_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, hidden_dim)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [num_tokens, hidden_dim]
        Returns:
            output: [num_tokens, hidden_dim]
        """
        h = self.activation(self.fc1(x))
        return self.fc2(h)


class SimpleMoE(nn.Module):
    """Simple Mixture of Experts Layer"""

    def __init__(
        self,
        hidden_dim: int = 512,
        ffn_dim: int = 2048,
        num_experts: int = 4,
        top_k: int = 2,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.ffn_dim = ffn_dim
        self.num_experts = num_experts
        self.top_k = top_k

        # Create experts
        self.experts = nn.ModuleList([
            Expert(hidden_dim, ffn_dim) for _ in range(num_experts)
        ])

        # Router network
        self.router = nn.Linear(hidden_dim, num_experts, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        MoE forward pass

        Args:
            x: [batch_size, seq_len, hidden_dim]

        Returns:
            output: [batch_size, seq_len, hidden_dim]
        """
        batch_size, seq_len, hidden_dim = x.shape
        num_tokens = batch_size * seq_len

        # Flatten batch and sequence dimensions
        x_flat = x.view(num_tokens, hidden_dim)  # [num_tokens, hidden_dim]

        # Compute router logits
        router_logits = self.router(x_flat)  # [num_tokens, num_experts]

        # Apply softmax to get probabilities
        router_probs = F.softmax(router_logits, dim=-1)

        # Select top-k experts
        top_k_probs, top_k_indices = torch.topk(
            router_probs, self.top_k, dim=-1
        )  # [num_tokens, top_k]

        # Normalize top-k probabilities to sum to 1
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)

        # Initialize output tensor
        output = torch.zeros_like(x_flat)

        # Process each expert
        for expert_idx in range(self.num_experts):
            # Find tokens assigned to this expert
            expert_mask = (top_k_indices == expert_idx)  # [num_tokens, top_k]
            token_indices, k_indices = torch.where(expert_mask)

            # Skip if no tokens assigned
            if len(token_indices) == 0:
                continue

            # Gather input tokens for this expert
            expert_input = x_flat[token_indices]  # [num_expert_tokens, hidden_dim]

            # Forward pass through expert
            expert_output = self.experts[expert_idx](expert_input)

            # Get routing weights for these tokens
            expert_weights = top_k_probs[token_indices, k_indices]  # [num_expert_tokens]

            # Add weighted expert output to result
            output[token_indices] += expert_weights.unsqueeze(-1) * expert_output

        # Reshape output to match input
        output = output.view(batch_size, seq_len, hidden_dim)

        return output


def demonstrate_usage():
    """Demonstrate MoE layer usage"""
    print("=" * 60)
    print("SimpleMoE Layer - Solution Demo")
    print("=" * 60 + "\n")

    # Create MoE layer
    moe = SimpleMoE(
        hidden_dim=512,
        ffn_dim=2048,
        num_experts=8,
        top_k=2
    )

    # Move to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    moe = moe.to(device)

    # Create input
    batch_size, seq_len = 4, 32
    x = torch.randn(batch_size, seq_len, 512, device=device)

    print(f"Configuration:")
    print(f"  - Hidden Dim: {moe.hidden_dim}")
    print(f"  - FFN Dim: {moe.ffn_dim}")
    print(f"  - Num Experts: {moe.num_experts}")
    print(f"  - Top-k: {moe.top_k}")
    print(f"  - Device: {device}\n")

    print(f"Input:")
    print(f"  - Shape: {x.shape}")
    print(f"  - Total Tokens: {batch_size * seq_len}\n")

    # Forward pass
    output = moe(x)

    print(f"Output:")
    print(f"  - Shape: {output.shape}")
    print(f"  - Mean: {output.mean().item():.4f}")
    print(f"  - Std: {output.std().item():.4f}")

    # Verify correctness
    assert output.shape == x.shape, "Shape mismatch!"
    assert not torch.isnan(output).any(), "NaN in output!"
    assert not torch.isinf(output).any(), "Inf in output!"

    print("\n✓ All checks passed!")


if __name__ == "__main__":
    demonstrate_usage()
