"""
Expert Routing Mechanisms for MoE

Demonstrates different routing strategies:
- Top-k with temperature scaling
- Capacity-based token dropping
- Expert load statistics and visualization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict
import numpy as np


class TopKRouter(nn.Module):
    """
    Standard Top-k routing with configurable temperature

    Args:
        hidden_dim: Input hidden dimension
        num_experts: Total number of experts
        top_k: Number of experts to select per token
        temperature: Softmax temperature (higher = more uniform)
    """

    def __init__(
        self,
        hidden_dim: int,
        num_experts: int,
        top_k: int = 2,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.temperature = temperature

        self.gate = nn.Linear(hidden_dim, num_experts, bias=False)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Args:
            x: [num_tokens, hidden_dim]

        Returns:
            top_k_indices: [num_tokens, top_k] - Selected expert indices
            top_k_weights: [num_tokens, top_k] - Normalized routing weights
            stats: Dictionary with routing statistics
        """
        # Compute router logits
        logits = self.gate(x)  # [num_tokens, num_experts]

        # Apply temperature scaling
        logits = logits / self.temperature

        # Softmax to get probabilities
        probs = F.softmax(logits, dim=-1)  # [num_tokens, num_experts]

        # Top-k selection
        top_k_probs, top_k_indices = torch.topk(probs, self.top_k, dim=-1)

        # Renormalize top-k probabilities
        top_k_weights = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)

        # Compute statistics
        stats = self._compute_stats(probs, top_k_indices)

        return top_k_indices, top_k_weights, stats

    def _compute_stats(
        self,
        probs: torch.Tensor,
        top_k_indices: torch.Tensor
    ) -> Dict:
        """Compute routing statistics"""

        num_tokens = probs.shape[0]

        # Expert load (number of tokens assigned)
        expert_counts = torch.zeros(self.num_experts, device=probs.device)
        for expert_idx in range(self.num_experts):
            expert_counts[expert_idx] = (top_k_indices == expert_idx).sum()

        # Router probability distribution
        avg_probs = probs.mean(dim=0)  # [num_experts]

        # Entropy of routing (higher = more diverse)
        entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1).mean()

        # Gini coefficient (load imbalance measure)
        sorted_counts = torch.sort(expert_counts)[0]
        n = self.num_experts
        index = torch.arange(1, n + 1, device=probs.device, dtype=torch.float32)
        gini = ((2 * index - n - 1) * sorted_counts).sum() / (n * sorted_counts.sum() + 1e-10)

        return {
            'expert_counts': expert_counts.cpu().numpy(),
            'avg_probs': avg_probs.cpu().numpy(),
            'entropy': entropy.item(),
            'gini_coefficient': gini.item(),
            'num_tokens': num_tokens,
        }


class CapacityRouter(TopKRouter):
    """
    Top-k routing with expert capacity constraints

    Drops tokens when an expert exceeds its capacity.
    Common in Switch Transformer and GShard.

    Args:
        capacity_factor: Capacity as multiple of average load
                        capacity = (num_tokens / num_experts) * capacity_factor
    """

    def __init__(
        self,
        hidden_dim: int,
        num_experts: int,
        top_k: int = 2,
        capacity_factor: float = 1.25,
        temperature: float = 1.0,
    ):
        super().__init__(hidden_dim, num_experts, top_k, temperature)
        self.capacity_factor = capacity_factor

    def forward(
        self,
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        """
        Args:
            x: [num_tokens, hidden_dim]

        Returns:
            top_k_indices: [num_tokens, top_k]
            top_k_weights: [num_tokens, top_k]
            capacity_mask: [num_tokens, top_k] - 1 if kept, 0 if dropped
            stats: Dictionary with routing and capacity statistics
        """
        # Get base routing
        top_k_indices, top_k_weights, stats = super().forward(x)

        num_tokens = x.shape[0]

        # Compute expert capacity
        capacity = int((num_tokens * self.top_k / self.num_experts) * self.capacity_factor)

        # Track tokens assigned to each expert
        expert_counters = torch.zeros(self.num_experts, device=x.device, dtype=torch.long)

        # Capacity mask (1 = keep, 0 = drop)
        capacity_mask = torch.ones_like(top_k_indices, dtype=torch.float32)

        num_dropped = 0

        # Process tokens sequentially (can be parallelized with cumsum tricks)
        for token_idx in range(num_tokens):
            for k_idx in range(self.top_k):
                expert_idx = top_k_indices[token_idx, k_idx]

                if expert_counters[expert_idx] >= capacity:
                    # Drop this token-expert assignment
                    capacity_mask[token_idx, k_idx] = 0.0
                    num_dropped += 1
                else:
                    expert_counters[expert_idx] += 1

        # Renormalize weights after dropping
        mask_sum = capacity_mask.sum(dim=-1, keepdim=True)
        top_k_weights = top_k_weights * capacity_mask
        top_k_weights = top_k_weights / (mask_sum + 1e-10)

        # Update statistics
        stats['capacity'] = capacity
        stats['num_dropped'] = num_dropped
        stats['drop_rate'] = num_dropped / (num_tokens * self.top_k)
        stats['expert_counters'] = expert_counters.cpu().numpy()

        return top_k_indices, top_k_weights, capacity_mask, stats


def visualize_routing(router: TopKRouter, num_samples: int = 1000):
    """Visualize routing patterns"""

    device = next(router.parameters()).device

    # Generate random inputs
    x = torch.randn(num_samples, router.hidden_dim, device=device)

    # Route
    if isinstance(router, CapacityRouter):
        indices, weights, mask, stats = router(x)
    else:
        indices, weights, stats = router(x)

    print("=" * 60)
    print("Routing Statistics")
    print("=" * 60)

    print(f"\nTotal Tokens: {stats['num_tokens']}")
    print(f"Top-k: {router.top_k}")
    print(f"Routing Entropy: {stats['entropy']:.3f}")
    print(f"Gini Coefficient: {stats['gini_coefficient']:.3f}")
    print("  (0.0 = perfectly balanced, 1.0 = maximally imbalanced)")

    print("\nExpert Load Distribution:")
    expert_counts = stats['expert_counts']
    avg_probs = stats['avg_probs']

    for i in range(router.num_experts):
        count = expert_counts[i]
        prob = avg_probs[i]
        percentage = (count / (stats['num_tokens'] * router.top_k)) * 100
        print(f"  Expert {i:2d}: {int(count):4d} tokens ({percentage:5.2f}%) | Avg Prob: {prob:.4f}")

    # Load imbalance metrics
    expected_load = (stats['num_tokens'] * router.top_k) / router.num_experts
    max_load = expert_counts.max()
    min_load = expert_counts.min()

    print(f"\nLoad Imbalance:")
    print(f"  Expected Load: {expected_load:.1f} tokens/expert")
    print(f"  Max Load: {int(max_load)} ({max_load/expected_load:.2f}x expected)")
    print(f"  Min Load: {int(min_load)} ({min_load/expected_load:.2f}x expected)")
    print(f"  Range: {int(max_load - min_load)} tokens")

    # Capacity-specific stats
    if isinstance(router, CapacityRouter):
        print(f"\nCapacity Constraints:")
        print(f"  Capacity Factor: {router.capacity_factor}")
        print(f"  Expert Capacity: {stats['capacity']} tokens")
        print(f"  Dropped Assignments: {stats['num_dropped']} ({stats['drop_rate']*100:.2f}%)")


def compare_temperatures():
    """Compare routing behavior at different temperatures"""

    config = {
        'hidden_dim': 512,
        'num_experts': 8,
        'top_k': 2,
        'num_samples': 1000,
    }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    temperatures = [0.1, 0.5, 1.0, 2.0, 5.0]

    print("=" * 60)
    print("Temperature Comparison")
    print("=" * 60 + "\n")

    for temp in temperatures:
        print(f"\nTemperature: {temp}")
        print("-" * 40)

        router = TopKRouter(
            hidden_dim=config['hidden_dim'],
            num_experts=config['num_experts'],
            top_k=config['top_k'],
            temperature=temp,
        ).to(device)

        x = torch.randn(config['num_samples'], config['hidden_dim'], device=device)
        _, _, stats = router(x)

        print(f"  Entropy: {stats['entropy']:.3f}")
        print(f"  Gini Coefficient: {stats['gini_coefficient']:.3f}")

        expert_counts = stats['expert_counts']
        print(f"  Load Range: {int(expert_counts.min())} - {int(expert_counts.max())} tokens")


def compare_capacity_factors():
    """Compare capacity-based routing at different capacity factors"""

    config = {
        'hidden_dim': 512,
        'num_experts': 8,
        'top_k': 2,
        'num_samples': 1000,
    }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    capacity_factors = [1.0, 1.25, 1.5, 2.0]

    print("\n" + "=" * 60)
    print("Capacity Factor Comparison")
    print("=" * 60 + "\n")

    for cf in capacity_factors:
        print(f"\nCapacity Factor: {cf}")
        print("-" * 40)

        router = CapacityRouter(
            hidden_dim=config['hidden_dim'],
            num_experts=config['num_experts'],
            top_k=config['top_k'],
            capacity_factor=cf,
        ).to(device)

        x = torch.randn(config['num_samples'], config['hidden_dim'], device=device)
        _, _, _, stats = router(x)

        print(f"  Expert Capacity: {stats['capacity']} tokens")
        print(f"  Dropped Assignments: {stats['num_dropped']} ({stats['drop_rate']*100:.2f}%)")
        print(f"  Gini Coefficient: {stats['gini_coefficient']:.3f}")


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Example 1: Basic Top-k routing
    print("Example 1: Basic Top-k Routing")
    print("=" * 60 + "\n")

    router = TopKRouter(
        hidden_dim=1024,
        num_experts=16,
        top_k=4,
        temperature=1.0,
    ).to(device)

    visualize_routing(router, num_samples=2000)

    # Example 2: Capacity-based routing
    print("\n\n" + "=" * 60)
    print("Example 2: Capacity-Based Routing")
    print("=" * 60 + "\n")

    capacity_router = CapacityRouter(
        hidden_dim=1024,
        num_experts=16,
        top_k=4,
        capacity_factor=1.25,
        temperature=1.0,
    ).to(device)

    visualize_routing(capacity_router, num_samples=2000)

    # Example 3: Temperature comparison
    compare_temperatures()

    # Example 4: Capacity factor comparison
    compare_capacity_factors()
