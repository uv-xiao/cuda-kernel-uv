#!/usr/bin/env python3
"""
DeepGEMM MoE Grouped GEMM Example

Demonstrates:
1. Token routing for MoE
2. Grouped GEMM execution
3. Comparison with padded batched GEMM
"""

import torch
import time
import argparse
from collections import defaultdict

try:
    import deepgemm
    HAS_DEEPGEMM = True
except ImportError:
    HAS_DEEPGEMM = False
    print("Warning: DeepGEMM not installed. Using reference implementation.")


# ============================================================================
# MoE Router
# ============================================================================

class SimpleRouter(torch.nn.Module):
    """Simple top-1 router for MoE"""

    def __init__(self, hidden_dim, num_experts):
        super().__init__()
        self.gate = torch.nn.Linear(hidden_dim, num_experts, bias=False)

    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, hidden_dim)
        Returns:
            expert_ids: (batch_size * seq_len,)
            expert_scores: (batch_size * seq_len,)
        """
        logits = self.gate(x)  # (batch, seq_len, num_experts)
        logits = logits.view(-1, logits.size(-1))  # (batch*seq_len, num_experts)

        expert_scores, expert_ids = torch.max(logits, dim=-1)
        return expert_ids, expert_scores


# ============================================================================
# Token Grouping
# ============================================================================

def group_tokens_by_expert(tokens, expert_ids, num_experts):
    """
    Group tokens by their assigned expert.

    Args:
        tokens: (num_tokens, hidden_dim)
        expert_ids: (num_tokens,)
        num_experts: int

    Returns:
        grouped_tokens: list of tensors, one per expert
        group_sizes: list of ints
        token_indices: list of lists (for scatter back)
    """
    grouped_tokens = []
    group_sizes = []
    token_indices = []

    for expert_id in range(num_experts):
        mask = (expert_ids == expert_id)
        expert_tokens = tokens[mask]

        grouped_tokens.append(expert_tokens)
        group_sizes.append(expert_tokens.size(0))
        token_indices.append(torch.where(mask)[0])

    return grouped_tokens, group_sizes, token_indices


def scatter_expert_outputs(expert_outputs, token_indices, num_tokens, hidden_dim):
    """
    Scatter expert outputs back to original token positions.

    Args:
        expert_outputs: list of tensors
        token_indices: list of lists
        num_tokens: int
        hidden_dim: int

    Returns:
        output: (num_tokens, hidden_dim)
    """
    output = torch.zeros(num_tokens, hidden_dim,
                        dtype=expert_outputs[0].dtype,
                        device=expert_outputs[0].device)

    for expert_output, indices in zip(expert_outputs, token_indices):
        if len(indices) > 0:
            output[indices] = expert_output

    return output


# ============================================================================
# MoE Layer
# ============================================================================

class MoELayer(torch.nn.Module):
    """Mixture-of-Experts layer with grouped GEMM"""

    def __init__(self, hidden_dim, num_experts, expert_dim=None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.expert_dim = expert_dim or hidden_dim * 4

        # Router
        self.router = SimpleRouter(hidden_dim, num_experts)

        # Expert weights (all stacked)
        self.w1 = torch.nn.Parameter(
            torch.randn(num_experts, hidden_dim, self.expert_dim,
                       dtype=torch.bfloat16) * 0.02
        )
        self.w2 = torch.nn.Parameter(
            torch.randn(num_experts, self.expert_dim, hidden_dim,
                       dtype=torch.bfloat16) * 0.02
        )

    def forward_grouped(self, x):
        """Forward pass using grouped GEMM"""
        batch_size, seq_len, hidden_dim = x.shape
        num_tokens = batch_size * seq_len

        # Flatten tokens
        tokens = x.view(num_tokens, hidden_dim)

        # Route tokens
        expert_ids, expert_scores = self.router(x)

        # Group by expert
        grouped_tokens, group_sizes, token_indices = \
            group_tokens_by_expert(tokens, expert_ids, self.num_experts)

        # Process each expert (simplified - real grouped GEMM would be single call)
        expert_outputs = []

        for i, expert_tokens in enumerate(grouped_tokens):
            if expert_tokens.size(0) == 0:
                expert_outputs.append(torch.empty(0, hidden_dim,
                                                 dtype=x.dtype, device=x.device))
                continue

            # FFN: w2 @ gelu(w1 @ x)
            h = torch.matmul(expert_tokens, self.w1[i])
            h = torch.nn.functional.gelu(h)
            output = torch.matmul(h, self.w2[i])

            expert_outputs.append(output)

        # Scatter back
        output = scatter_expert_outputs(expert_outputs, token_indices,
                                       num_tokens, hidden_dim)

        return output.view(batch_size, seq_len, hidden_dim), group_sizes

    def forward_padded(self, x):
        """Forward pass using padded batched GEMM (baseline)"""
        batch_size, seq_len, hidden_dim = x.shape
        num_tokens = batch_size * seq_len

        tokens = x.view(num_tokens, hidden_dim)
        expert_ids, expert_scores = self.router(x)

        # Group by expert
        grouped_tokens, group_sizes, token_indices = \
            group_tokens_by_expert(tokens, expert_ids, self.num_experts)

        # Find max tokens per expert
        max_tokens = max(group_sizes)

        # Pad and stack
        padded_tokens = torch.zeros(self.num_experts, max_tokens, hidden_dim,
                                    dtype=x.dtype, device=x.device)

        for i, expert_tokens in enumerate(grouped_tokens):
            if expert_tokens.size(0) > 0:
                padded_tokens[i, :expert_tokens.size(0)] = expert_tokens

        # Batched GEMM
        # (num_experts, max_tokens, expert_dim)
        h = torch.matmul(padded_tokens, self.w1)
        h = torch.nn.functional.gelu(h)
        # (num_experts, max_tokens, hidden_dim)
        padded_outputs = torch.matmul(h, self.w2)

        # Extract actual outputs
        expert_outputs = []
        for i in range(self.num_experts):
            if group_sizes[i] > 0:
                expert_outputs.append(padded_outputs[i, :group_sizes[i]])
            else:
                expert_outputs.append(torch.empty(0, hidden_dim,
                                                 dtype=x.dtype, device=x.device))

        # Scatter back
        output = scatter_expert_outputs(expert_outputs, token_indices,
                                       num_tokens, hidden_dim)

        return output.view(batch_size, seq_len, hidden_dim), max_tokens


# ============================================================================
# Benchmarking
# ============================================================================

def benchmark_moe(batch_size=4, seq_len=256, hidden_dim=2048,
                 num_experts=8, expert_dim=8192, warmup=10, iters=100):
    """Benchmark grouped vs padded MoE"""

    print(f"\n{'='*60}")
    print(f"MoE Layer Benchmark")
    print(f"{'='*60}")
    print(f"Batch size: {batch_size}")
    print(f"Sequence length: {seq_len}")
    print(f"Hidden dimension: {hidden_dim}")
    print(f"Number of experts: {num_experts}")
    print(f"Expert dimension: {expert_dim}")
    print(f"{'='*60}\n")

    # Create model
    model = MoELayer(hidden_dim, num_experts, expert_dim).cuda()

    # Input
    torch.manual_seed(42)
    x = torch.randn(batch_size, seq_len, hidden_dim,
                   dtype=torch.bfloat16, device='cuda')

    # Get token distribution
    with torch.no_grad():
        _, group_sizes = model.forward_grouped(x)

    print("Token distribution per expert:")
    total_tokens = sum(group_sizes)
    for i, size in enumerate(group_sizes):
        print(f"  Expert {i}: {size} tokens ({size/total_tokens*100:.1f}%)")
    print(f"  Total: {total_tokens} tokens\n")

    # Benchmark grouped GEMM
    print("Benchmarking Grouped GEMM...")
    for _ in range(warmup):
        with torch.no_grad():
            output, _ = model.forward_grouped(x)
    torch.cuda.synchronize()

    start = time.time()
    for _ in range(iters):
        with torch.no_grad():
            output, _ = model.forward_grouped(x)
    torch.cuda.synchronize()
    elapsed_grouped = time.time() - start

    time_grouped = elapsed_grouped / iters * 1000  # ms

    # Benchmark padded GEMM
    print("Benchmarking Padded Batched GEMM...")
    for _ in range(warmup):
        with torch.no_grad():
            output_padded, _ = model.forward_padded(x)
    torch.cuda.synchronize()

    start = time.time()
    for _ in range(iters):
        with torch.no_grad():
            output_padded, max_tokens = model.forward_padded(x)
    torch.cuda.synchronize()
    elapsed_padded = time.time() - start

    time_padded = elapsed_padded / iters * 1000  # ms

    # Calculate FLOPS
    # Two GEMMs per expert: (M, K) @ (K, N) and (M, N) @ (N, K)
    useful_flops = 0
    for size in group_sizes:
        useful_flops += 2 * size * hidden_dim * expert_dim  # w1
        useful_flops += 2 * size * expert_dim * hidden_dim  # w2

    padded_flops = 2 * num_experts * max_tokens * hidden_dim * expert_dim * 2

    tflops_grouped = (useful_flops / 1e12) / (time_grouped / 1000)
    tflops_padded = (padded_flops / 1e12) / (time_padded / 1000)

    # Results
    print(f"\n{'='*60}")
    print(f"Results")
    print(f"{'='*60}")
    print(f"Grouped GEMM:")
    print(f"  Time: {time_grouped:.3f} ms")
    print(f"  Throughput: {tflops_grouped:.1f} TFLOPS")
    print(f"  Useful FLOPs: {useful_flops/1e9:.2f} GFLOPS\n")

    print(f"Padded Batched GEMM:")
    print(f"  Time: {time_padded:.3f} ms")
    print(f"  Throughput: {tflops_padded:.1f} TFLOPS")
    print(f"  Total FLOPs: {padded_flops/1e9:.2f} GFLOPS")
    print(f"  Compute efficiency: {useful_flops/padded_flops*100:.1f}%\n")

    print(f"Speedup: {time_padded/time_grouped:.2f}x")
    print(f"{'='*60}\n")

    # Verify correctness
    error = (output - output_padded).abs().mean().item()
    print(f"Numerical difference: {error:.6f}")


def main():
    parser = argparse.ArgumentParser(description='MoE Grouped GEMM Example')
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--seq-len', type=int, default=256)
    parser.add_argument('--hidden', type=int, default=2048)
    parser.add_argument('--experts', type=int, default=8)
    parser.add_argument('--expert-dim', type=int, default=8192)
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--iters', type=int, default=100)

    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("Error: CUDA not available")
        return

    print(f"Device: {torch.cuda.get_device_name()}")

    benchmark_moe(
        batch_size=args.batch,
        seq_len=args.seq_len,
        hidden_dim=args.hidden,
        num_experts=args.experts,
        expert_dim=args.expert_dim,
        warmup=args.warmup,
        iters=args.iters
    )


if __name__ == '__main__':
    main()
