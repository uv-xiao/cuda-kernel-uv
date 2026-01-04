"""
Simple Mixture of Experts (MoE) Layer in PyTorch

This implementation demonstrates the core MoE architecture:
- Multiple expert FFN networks
- Top-k token routing
- Load balancing with auxiliary loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
import time


class Expert(nn.Module):
    """Single expert: a two-layer feed-forward network"""

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


class MoELayer(nn.Module):
    """
    Mixture of Experts Layer

    Args:
        hidden_dim: Hidden dimension of input/output
        ffn_dim: Feed-forward network intermediate dimension
        num_experts: Total number of experts
        top_k: Number of experts to activate per token
        capacity_factor: Expert capacity as multiple of average load
    """

    def __init__(
        self,
        hidden_dim: int = 4096,
        ffn_dim: int = 14336,
        num_experts: int = 8,
        top_k: int = 2,
        capacity_factor: float = 1.25,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.ffn_dim = ffn_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.capacity_factor = capacity_factor

        # Create experts
        self.experts = nn.ModuleList([
            Expert(hidden_dim, ffn_dim) for _ in range(num_experts)
        ])

        # Router network
        self.router = nn.Linear(hidden_dim, num_experts, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        return_stats: bool = False
    ) -> Tuple[torch.Tensor, dict]:
        """
        Forward pass with top-k routing

        Args:
            x: Input tensor [batch_size, seq_len, hidden_dim]
            return_stats: Whether to return routing statistics

        Returns:
            output: [batch_size, seq_len, hidden_dim]
            stats: Dictionary with routing and performance stats
        """
        batch_size, seq_len, hidden_dim = x.shape
        num_tokens = batch_size * seq_len

        # Flatten batch and sequence dimensions
        x_flat = x.view(num_tokens, hidden_dim)  # [num_tokens, hidden_dim]

        # Router forward pass
        router_logits = self.router(x_flat)  # [num_tokens, num_experts]
        router_probs = F.softmax(router_logits, dim=-1)

        # Top-k selection
        top_k_probs, top_k_indices = torch.topk(
            router_probs, self.top_k, dim=-1
        )  # [num_tokens, top_k]

        # Normalize top-k probabilities
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)

        # Initialize output
        output = torch.zeros_like(x_flat)

        # Statistics
        expert_token_counts = torch.zeros(self.num_experts, device=x.device)

        # Process each expert (naive sequential implementation)
        for expert_idx in range(self.num_experts):
            # Find all tokens assigned to this expert
            expert_mask = (top_k_indices == expert_idx)  # [num_tokens, top_k]
            token_indices, k_indices = torch.where(expert_mask)

            if len(token_indices) == 0:
                continue

            # Gather tokens for this expert
            expert_input = x_flat[token_indices]  # [num_expert_tokens, hidden_dim]

            # Expert forward pass
            expert_output = self.experts[expert_idx](expert_input)

            # Gather corresponding weights
            expert_weights = top_k_probs[token_indices, k_indices]  # [num_expert_tokens]

            # Weighted accumulation
            output[token_indices] += expert_weights.unsqueeze(-1) * expert_output

            # Statistics
            expert_token_counts[expert_idx] = len(token_indices)

        # Reshape output
        output = output.view(batch_size, seq_len, hidden_dim)

        # Compute statistics
        stats = {}
        if return_stats:
            stats = {
                'expert_token_counts': expert_token_counts.cpu().numpy(),
                'total_tokens': num_tokens,
                'load_balance_loss': self._compute_load_balance_loss(
                    router_probs, expert_token_counts, num_tokens
                ),
            }

        return output, stats

    def _compute_load_balance_loss(
        self,
        router_probs: torch.Tensor,
        expert_token_counts: torch.Tensor,
        num_tokens: int
    ) -> float:
        """
        Compute load balancing auxiliary loss

        Encourages balanced expert utilization:
        L_balance = num_experts * sum_i(P_i * f_i)

        where:
        - P_i = fraction of router probability mass for expert i
        - f_i = fraction of tokens assigned to expert i
        """
        # Average router probability per expert
        avg_router_probs = router_probs.mean(dim=0)  # [num_experts]

        # Fraction of tokens per expert
        expert_fractions = expert_token_counts / num_tokens  # [num_experts]

        # Load balance loss (should be minimized)
        loss = self.num_experts * torch.sum(avg_router_probs * expert_fractions)

        return loss.item()


def benchmark_moe():
    """Benchmark MoE layer performance"""

    # Configuration similar to DeepSeek-V3 (scaled down)
    config = {
        'hidden_dim': 4096,
        'ffn_dim': 14336,
        'num_experts': 8,
        'top_k': 2,
        'batch_size': 32,
        'seq_len': 512,
    }

    print("MoE Layer Configuration:")
    print(f"  - Hidden Dim: {config['hidden_dim']}")
    print(f"  - Expert FFN Dim: {config['ffn_dim']}")
    print(f"  - Num Experts: {config['num_experts']}")
    print(f"  - Top-k: {config['top_k']}")
    print(f"  - Batch Size: {config['batch_size']}, Sequence Length: {config['seq_len']}\n")

    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MoELayer(
        hidden_dim=config['hidden_dim'],
        ffn_dim=config['ffn_dim'],
        num_experts=config['num_experts'],
        top_k=config['top_k'],
    ).to(device)

    # Random input
    x = torch.randn(
        config['batch_size'],
        config['seq_len'],
        config['hidden_dim'],
        device=device
    )

    # Warmup
    for _ in range(3):
        output, _ = model(x)

    # Benchmark
    if device.type == 'cuda':
        torch.cuda.synchronize()

    start = time.perf_counter()
    output, stats = model(x, return_stats=True)

    if device.type == 'cuda':
        torch.cuda.synchronize()

    elapsed_ms = (time.perf_counter() - start) * 1000

    # Print results
    print("Forward Pass:")
    print(f"  - Input: {list(x.shape)}")
    print(f"  - Output: {list(output.shape)}")
    print(f"  - Time: {elapsed_ms:.2f}ms\n")

    print("Expert Load Distribution:")
    total_tokens = stats['total_tokens']
    for i, count in enumerate(stats['expert_token_counts']):
        percentage = (count / total_tokens) * 100
        print(f"  Expert {i}: {int(count):4d} tokens ({percentage:5.2f}%)")

    print(f"\nLoad Balance Loss: {stats['load_balance_loss']:.4f}")

    # Memory statistics
    if device.type == 'cuda':
        mem_mb = torch.cuda.max_memory_allocated() / 1024**2
        print(f"\nPeak GPU Memory: {mem_mb:.2f} MB")

    # Compute theoretical FLOPs
    num_tokens = config['batch_size'] * config['seq_len']
    tokens_per_expert = num_tokens * config['top_k'] / config['num_experts']

    # Each expert: 2 linear layers (up-proj and down-proj)
    flops_per_expert = 2 * tokens_per_expert * (
        config['hidden_dim'] * config['ffn_dim'] +  # fc1
        config['ffn_dim'] * config['hidden_dim']     # fc2
    )
    total_flops = flops_per_expert * config['num_experts']
    tflops = total_flops / 1e12
    tflops_per_sec = tflops / (elapsed_ms / 1000)

    print(f"\nTheoretical Compute:")
    print(f"  - Total FLOPs: {tflops:.3f} TFLOPs")
    print(f"  - Throughput: {tflops_per_sec:.2f} TFLOPs/s")


def analyze_routing_patterns():
    """Analyze routing patterns and load imbalance"""

    config = {
        'hidden_dim': 1024,
        'ffn_dim': 4096,
        'num_experts': 16,
        'top_k': 4,
        'batch_size': 8,
        'seq_len': 128,
    }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MoELayer(
        hidden_dim=config['hidden_dim'],
        ffn_dim=config['ffn_dim'],
        num_experts=config['num_experts'],
        top_k=config['top_k'],
    ).to(device)

    # Run multiple forward passes
    num_runs = 10
    all_counts = []

    print(f"Analyzing routing patterns over {num_runs} batches...\n")

    for run_idx in range(num_runs):
        x = torch.randn(
            config['batch_size'],
            config['seq_len'],
            config['hidden_dim'],
            device=device
        )

        _, stats = model(x, return_stats=True)
        all_counts.append(stats['expert_token_counts'])

    # Compute statistics
    all_counts = torch.tensor(all_counts)  # [num_runs, num_experts]
    mean_counts = all_counts.mean(dim=0)
    std_counts = all_counts.std(dim=0)

    print("Average Expert Load (across batches):")
    for i in range(config['num_experts']):
        print(f"  Expert {i:2d}: {mean_counts[i]:6.1f} ± {std_counts[i]:5.1f} tokens")

    # Coefficient of variation (CV) as load imbalance metric
    cv = (std_counts / (mean_counts + 1e-6)).mean().item()
    print(f"\nLoad Imbalance (Coefficient of Variation): {cv:.3f}")
    print("(Lower is better, 0.0 = perfectly balanced)")


if __name__ == "__main__":
    print("=" * 60)
    print("MoE Layer Benchmark")
    print("=" * 60 + "\n")

    benchmark_moe()

    print("\n" + "=" * 60)
    print("Routing Pattern Analysis")
    print("=" * 60 + "\n")

    analyze_routing_patterns()
