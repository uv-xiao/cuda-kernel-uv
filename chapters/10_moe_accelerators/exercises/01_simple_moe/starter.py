"""
Exercise 01: Simple MoE Forward Pass - Starter Code

Complete the TODOs to implement a working MoE layer.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class Expert(nn.Module):
    """Single expert FFN"""

    def __init__(self, hidden_dim: int, ffn_dim: int):
        super().__init__()
        # TODO: Initialize two linear layers and activation
        # Hint: Use nn.Linear and nn.GELU
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [num_tokens, hidden_dim]
        Returns:
            output: [num_tokens, hidden_dim]
        """
        # TODO: Implement FFN forward pass
        # fc1 → GELU → fc2
        pass


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

        # TODO: Initialize experts
        # Hint: Use nn.ModuleList
        self.experts = None

        # TODO: Initialize router
        # Hint: Linear layer mapping hidden_dim → num_experts
        self.router = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        MoE forward pass

        Args:
            x: [batch_size, seq_len, hidden_dim]

        Returns:
            output: [batch_size, seq_len, hidden_dim]
        """
        batch_size, seq_len, hidden_dim = x.shape

        # TODO: Flatten batch and sequence dimensions
        # Shape should be [num_tokens, hidden_dim] where num_tokens = batch_size * seq_len
        x_flat = None

        # TODO: Compute router logits
        # Shape: [num_tokens, num_experts]
        router_logits = None

        # TODO: Apply softmax to get probabilities
        router_probs = None

        # TODO: Select top-k experts
        # Use torch.topk to get top_k probabilities and indices
        # Shapes: both [num_tokens, top_k]
        top_k_probs = None
        top_k_indices = None

        # TODO: Normalize top-k probabilities
        # Hint: Divide by sum to ensure they sum to 1
        top_k_probs = None

        # TODO: Initialize output tensor
        output = torch.zeros_like(x_flat)

        # TODO: Process each expert
        for expert_idx in range(self.num_experts):
            # TODO: Find tokens assigned to this expert
            # Hint: Use torch.where on top_k_indices
            # This gives you (token_indices, k_indices) tuple
            pass

            # TODO: Skip if no tokens assigned
            pass

            # TODO: Gather input tokens for this expert
            # Shape: [num_expert_tokens, hidden_dim]
            expert_input = None

            # TODO: Forward pass through expert
            expert_output = None

            # TODO: Get routing weights for these tokens
            # Shape: [num_expert_tokens]
            expert_weights = None

            # TODO: Add weighted expert output to result
            # Hint: Multiply expert_output by weights and add to output[token_indices]
            pass

        # TODO: Reshape output to match input
        output = None

        return output


def test_basic():
    """Test basic functionality"""
    print("Testing basic MoE forward pass...")

    moe = SimpleMoE(hidden_dim=512, ffn_dim=2048, num_experts=4, top_k=2)
    x = torch.randn(8, 64, 512)

    try:
        output = moe(x)
        print(f"✓ Forward pass successful")
        print(f"  Input shape: {x.shape}")
        print(f"  Output shape: {output.shape}")

        assert output.shape == x.shape, "Output shape mismatch!"
        print(f"✓ Output shape correct")

        assert not torch.isnan(output).any(), "NaN values in output!"
        assert not torch.isinf(output).any(), "Inf values in output!"
        print(f"✓ No NaN/Inf values")

        print("\nBasic tests passed! Run test.py for full test suite.")

    except Exception as e:
        print(f"✗ Error: {e}")
        print("\nComplete the TODOs in the code above.")


if __name__ == "__main__":
    test_basic()
