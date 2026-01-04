"""
Mixture of Experts (MoE) Layer

Implement efficient MoE with:
1. Router for expert selection (TopK)
2. Batched expert computation
3. Output combination
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional
import math


class MoELayer(nn.Module):
    """
    Mixture of Experts layer

    Architecture:
    1. Router: Select top-k experts per token
    2. Expert dispatch: Group tokens by expert
    3. Expert computation: Batched GEMM for each expert
    4. Combine: Weighted combination of expert outputs
    """

    def __init__(
        self,
        hidden_dim: int,
        intermediate_dim: int,
        num_experts: int,
        num_experts_per_token: int = 2,
        expert_capacity: Optional[int] = None,
        use_custom_kernels: bool = True,
    ):
        """
        Initialize MoE layer

        Args:
            hidden_dim: Hidden dimension
            intermediate_dim: Intermediate dimension in experts
            num_experts: Total number of experts
            num_experts_per_token: Number of experts to route each token to
            expert_capacity: Maximum tokens per expert (for load balancing)
            use_custom_kernels: Use custom CUDA kernels if True
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.num_experts = num_experts
        self.num_experts_per_token = num_experts_per_token
        self.expert_capacity = expert_capacity
        self.use_custom_kernels = use_custom_kernels

        # Router: Linear layer + TopK
        self.router = nn.Linear(hidden_dim, num_experts, bias=False)

        # Experts: Each expert is a simple FFN (up projection + activation + down projection)
        self.experts = nn.ModuleList([
            Expert(hidden_dim, intermediate_dim) for _ in range(num_experts)
        ])

        # TODO: Load custom CUDA kernels
        # self.topk_kernel = load_topk_kernel()
        # self.dispatch_kernel = load_dispatch_kernel()
        # self.combine_kernel = load_combine_kernel()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        Forward pass through MoE layer

        Args:
            x: Input tensor (batch_size, seq_len, hidden_dim)

        Returns:
            output: Output tensor (batch_size, seq_len, hidden_dim)
            aux_loss: Auxiliary losses for load balancing
        """
        batch_size, seq_len, hidden_dim = x.shape
        num_tokens = batch_size * seq_len

        # Reshape to (num_tokens, hidden_dim)
        x_flat = x.view(-1, hidden_dim)

        # TODO: Implement MoE forward pass
        # 1. Route tokens to experts
        # 2. Dispatch tokens to experts
        # 3. Compute expert outputs
        # 4. Combine expert outputs

        # Step 1: Router - compute routing weights and select top-k experts
        router_logits = self.router(x_flat)  # (num_tokens, num_experts)
        routing_weights, selected_experts = self._route(router_logits)

        # Step 2: Dispatch tokens to experts
        # Group tokens going to the same expert for efficient batched computation
        expert_outputs, dispatch_mask = self._dispatch_and_compute(
            x_flat, selected_experts, routing_weights
        )

        # Step 3: Combine expert outputs
        output = self._combine(expert_outputs, dispatch_mask, routing_weights, selected_experts)

        # Reshape back to (batch_size, seq_len, hidden_dim)
        output = output.view(batch_size, seq_len, hidden_dim)

        # Compute auxiliary loss for load balancing
        aux_loss = self._compute_aux_loss(router_logits)

        return output, aux_loss

    def _route(
        self,
        router_logits: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Route tokens to top-k experts

        Args:
            router_logits: Router logits (num_tokens, num_experts)

        Returns:
            routing_weights: Routing weights (num_tokens, num_experts_per_token)
            selected_experts: Selected expert indices (num_tokens, num_experts_per_token)
        """
        # TODO: Implement efficient TopK routing
        # Option 1: Use PyTorch topk (baseline)
        # Option 2: Use custom CUDA kernel for better performance

        # Baseline implementation
        routing_weights, selected_experts = torch.topk(
            router_logits, self.num_experts_per_token, dim=-1
        )

        # Normalize routing weights (softmax over selected experts)
        routing_weights = torch.softmax(routing_weights, dim=-1)

        return routing_weights, selected_experts

    def _dispatch_and_compute(
        self,
        x: torch.Tensor,
        selected_experts: torch.Tensor,
        routing_weights: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Dispatch tokens to experts and compute outputs

        Args:
            x: Input tokens (num_tokens, hidden_dim)
            selected_experts: Selected expert indices (num_tokens, num_experts_per_token)
            routing_weights: Routing weights (num_tokens, num_experts_per_token)

        Returns:
            expert_outputs: Expert outputs (num_tokens, num_experts_per_token, hidden_dim)
            dispatch_mask: Mask indicating which tokens go to which experts
        """
        num_tokens = x.shape[0]

        # TODO: Implement efficient expert dispatch
        # Key optimization: Batch tokens going to the same expert

        # Naive implementation (replace with batched version)
        expert_outputs = torch.zeros(
            num_tokens,
            self.num_experts_per_token,
            self.hidden_dim,
            device=x.device,
            dtype=x.dtype,
        )

        for i in range(self.num_experts_per_token):
            for expert_idx in range(self.num_experts):
                # Find tokens routed to this expert
                mask = selected_experts[:, i] == expert_idx

                if mask.any():
                    # Get tokens for this expert
                    expert_input = x[mask]

                    # Compute expert output
                    expert_output = self.experts[expert_idx](expert_input)

                    # Store output
                    expert_outputs[mask, i] = expert_output

        dispatch_mask = torch.ones_like(expert_outputs[:, :, 0])

        return expert_outputs, dispatch_mask

    def _combine(
        self,
        expert_outputs: torch.Tensor,
        dispatch_mask: torch.Tensor,
        routing_weights: torch.Tensor,
        selected_experts: torch.Tensor,
    ) -> torch.Tensor:
        """
        Combine expert outputs with routing weights

        Args:
            expert_outputs: Expert outputs (num_tokens, num_experts_per_token, hidden_dim)
            dispatch_mask: Dispatch mask (num_tokens, num_experts_per_token)
            routing_weights: Routing weights (num_tokens, num_experts_per_token)
            selected_experts: Selected expert indices (num_tokens, num_experts_per_token)

        Returns:
            output: Combined output (num_tokens, hidden_dim)
        """
        # TODO: Implement efficient combining
        # Weighted sum of expert outputs

        # Expand routing weights for multiplication
        routing_weights = routing_weights.unsqueeze(-1)  # (num_tokens, num_experts_per_token, 1)

        # Weighted sum
        output = torch.sum(expert_outputs * routing_weights * dispatch_mask.unsqueeze(-1), dim=1)

        return output

    def _compute_aux_loss(self, router_logits: torch.Tensor) -> dict:
        """
        Compute auxiliary losses for load balancing

        Args:
            router_logits: Router logits (num_tokens, num_experts)

        Returns:
            Dictionary of auxiliary losses
        """
        # TODO: Implement load balancing loss
        # Common approach: Encourage uniform expert usage

        # Compute expert usage
        expert_probs = torch.softmax(router_logits, dim=-1)
        expert_usage = expert_probs.mean(dim=0)  # (num_experts,)

        # Ideal usage is 1/num_experts for each expert
        target_usage = 1.0 / self.num_experts

        # Load balancing loss (encourage uniform usage)
        load_balance_loss = torch.sum((expert_usage - target_usage) ** 2)

        return {
            'load_balance_loss': load_balance_loss,
            'expert_usage': expert_usage,
        }


class Expert(nn.Module):
    """
    Single expert network (simple FFN)
    """

    def __init__(self, hidden_dim: int, intermediate_dim: int):
        super().__init__()
        self.up_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.down_proj = nn.Linear(intermediate_dim, hidden_dim, bias=False)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through expert

        Args:
            x: Input (num_tokens, hidden_dim)

        Returns:
            Output (num_tokens, hidden_dim)
        """
        # TODO: Consider fusing operations for better performance
        x = self.up_proj(x)
        x = self.activation(x)
        x = self.down_proj(x)
        return x


class OptimizedMoELayer(MoELayer):
    """
    Optimized MoE layer with custom CUDA kernels

    Optimizations:
    1. Fused TopK routing kernel
    2. Efficient token batching by expert
    3. Batched GEMM using CUTLASS or cuBLAS
    4. Fused combining kernel
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # TODO: Load optimized CUDA kernels
        # self.fused_topk_kernel = load_fused_topk_kernel()
        # self.batched_expert_kernel = load_batched_expert_kernel()
        # self.fused_combine_kernel = load_fused_combine_kernel()

    def _route(self, router_logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Optimized routing with custom kernel"""
        # TODO: Use custom fused TopK + softmax kernel
        # return self.fused_topk_kernel(router_logits, self.num_experts_per_token)

        # Fall back to parent implementation
        return super()._route(router_logits)

    def _dispatch_and_compute(
        self,
        x: torch.Tensor,
        selected_experts: torch.Tensor,
        routing_weights: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Optimized dispatch with batched expert computation"""
        # TODO: Implement optimized batching
        # 1. Sort tokens by expert (use counting sort for efficiency)
        # 2. Compute start/end indices for each expert's tokens
        # 3. Launch batched GEMM for each expert
        # 4. Unsort tokens back to original order

        # Fall back to parent implementation
        return super()._dispatch_and_compute(x, selected_experts, routing_weights)

    def _combine(
        self,
        expert_outputs: torch.Tensor,
        dispatch_mask: torch.Tensor,
        routing_weights: torch.Tensor,
        selected_experts: torch.Tensor,
    ) -> torch.Tensor:
        """Optimized combining with fused kernel"""
        # TODO: Use fused kernel for weighted combination
        # return self.fused_combine_kernel(expert_outputs, routing_weights, dispatch_mask)

        # Fall back to parent implementation
        return super()._combine(expert_outputs, dispatch_mask, routing_weights, selected_experts)


def test_moe():
    """Test MoE layer for correctness"""
    print("Testing MoE Layer...")

    # Configuration
    batch_size = 4
    seq_len = 128
    hidden_dim = 1024
    intermediate_dim = 4096
    num_experts = 8
    num_experts_per_token = 2

    # Create MoE layer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    moe = MoELayer(
        hidden_dim=hidden_dim,
        intermediate_dim=intermediate_dim,
        num_experts=num_experts,
        num_experts_per_token=num_experts_per_token,
    ).to(device)

    # Create random input
    x = torch.randn(batch_size, seq_len, hidden_dim, device=device)

    # Test forward pass
    try:
        output, aux_loss = moe(x)
        print(f"  Input shape: {x.shape}")
        print(f"  Output shape: {output.shape}")
        print(f"  Load balance loss: {aux_loss['load_balance_loss'].item():.6f}")
        print(f"  Expert usage: {aux_loss['expert_usage']}")
        assert output.shape == x.shape, "Output shape mismatch"
        print("  Test passed!")
    except Exception as e:
        print(f"  Error: {e}")

    # Benchmark
    print("\nBenchmarking MoE layer...")
    import time

    num_iterations = 100
    warmup = 10

    # Warmup
    for _ in range(warmup):
        _ = moe(x)

    # Benchmark
    torch.cuda.synchronize()
    start = time.perf_counter()

    for _ in range(num_iterations):
        _ = moe(x)

    torch.cuda.synchronize()
    end = time.perf_counter()

    avg_time = (end - start) / num_iterations
    throughput = (batch_size * seq_len) / avg_time

    print(f"  Average time: {avg_time * 1000:.2f} ms")
    print(f"  Throughput: {throughput:.1f} tokens/s")


if __name__ == "__main__":
    test_moe()
