"""
Tile-Aware Token Rounding (TATR)

Implements the core SonicMoE optimization:
- Round expert token counts to tile boundaries
- Maximize GPU tile utilization
- Minimal quality degradation
"""

import torch
import numpy as np
from typing import List, Tuple, Dict
import time


def round_down_token_counts(
    token_counts: List[int],
    tile_size: int = 128
) -> Tuple[List[int], int]:
    """
    Round token counts down to nearest tile boundary

    Args:
        token_counts: Number of tokens assigned to each expert
        tile_size: Tile size (e.g., 128)

    Returns:
        rounded_counts: Rounded token counts
        total_dropped: Total number of dropped tokens
    """
    rounded_counts = []
    total_dropped = 0

    for count in token_counts:
        num_tiles = count // tile_size
        rounded = num_tiles * tile_size
        rounded_counts.append(rounded)
        total_dropped += (count - rounded)

    return rounded_counts, total_dropped


def round_up_token_counts(
    token_counts: List[int],
    tile_size: int = 128
) -> Tuple[List[int], int]:
    """
    Round token counts up to nearest tile boundary

    Args:
        token_counts: Number of tokens assigned to each expert
        tile_size: Tile size (e.g., 128)

    Returns:
        rounded_counts: Rounded token counts
        total_added: Total number of padded tokens
    """
    rounded_counts = []
    total_added = 0

    for count in token_counts:
        num_tiles = (count + tile_size - 1) // tile_size
        rounded = num_tiles * tile_size
        rounded_counts.append(rounded)
        total_added += (rounded - count)

    return rounded_counts, total_added


def adaptive_rounding(
    token_counts: List[int],
    tile_size: int = 128,
    threshold: float = 0.9
) -> Tuple[List[int], int, int]:
    """
    Adaptive rounding based on tile utilization

    Args:
        token_counts: Number of tokens assigned to each expert
        tile_size: Tile size
        threshold: Utilization threshold for rounding up

    Returns:
        rounded_counts: Rounded token counts
        total_dropped: Tokens dropped
        total_added: Tokens added (via padding)
    """
    rounded_counts = []
    total_dropped = 0
    total_added = 0

    for count in token_counts:
        num_tiles = count // tile_size
        remainder = count % tile_size
        utilization = remainder / tile_size if remainder > 0 else 1.0

        if utilization >= threshold:
            # High utilization, round up
            rounded = (num_tiles + 1) * tile_size
            total_added += (rounded - count)
        else:
            # Low utilization, round down
            rounded = num_tiles * tile_size
            total_dropped += (count - rounded)

        rounded_counts.append(rounded)

    return rounded_counts, total_dropped, total_added


def compute_tile_utilization(
    token_counts: List[int],
    tile_size: int = 128
) -> Dict[str, float]:
    """Compute tile utilization statistics"""

    total_tiles = 0
    total_utilized = 0

    for count in token_counts:
        if count == 0:
            continue

        num_full_tiles = count // tile_size
        remainder = count % tile_size

        total_tiles += num_full_tiles
        total_utilized += num_full_tiles * tile_size

        if remainder > 0:
            total_tiles += 1
            total_utilized += remainder

    avg_utilization = total_utilized / (total_tiles * tile_size) if total_tiles > 0 else 0

    return {
        'total_tiles': total_tiles,
        'avg_utilization': avg_utilization,
        'wasted_percentage': (1.0 - avg_utilization) * 100,
    }


def simulate_routing(
    num_experts: int,
    total_tokens: int,
    top_k: int,
    seed: int = 42
) -> Tuple[List[int], np.ndarray]:
    """
    Simulate expert routing (random for demonstration)

    Returns:
        token_counts: Tokens per expert
        routing_probs: Routing probabilities for each token-expert pair
    """
    np.random.seed(seed)

    token_counts = [0] * num_experts
    routing_probs = np.random.uniform(0, 1, (total_tokens, num_experts))

    # Normalize to get probabilities
    routing_probs = routing_probs / routing_probs.sum(axis=1, keepdims=True)

    # Top-k selection
    for token_idx in range(total_tokens):
        top_k_indices = np.argsort(routing_probs[token_idx])[-top_k:]
        for expert_idx in top_k_indices:
            token_counts[expert_idx] += 1

    return token_counts, routing_probs


def analyze_quality_impact(
    original_counts: List[int],
    rounded_counts: List[int],
    routing_probs: np.ndarray,
    num_experts: int
) -> Dict[str, float]:
    """
    Analyze quality impact of token rounding

    Estimates perplexity increase based on dropped tokens' routing probabilities
    """
    total_dropped_prob = 0.0
    total_tokens = sum(original_counts)
    tokens_dropped = 0

    expert_token_map = {i: [] for i in range(num_experts)}

    # Simulate token assignments
    token_idx = 0
    for expert_idx in range(num_experts):
        for _ in range(original_counts[expert_idx]):
            expert_token_map[expert_idx].append(token_idx)
            token_idx += 1

    # Calculate impact
    for expert_idx in range(num_experts):
        original = original_counts[expert_idx]
        rounded = rounded_counts[expert_idx]

        if rounded < original:
            dropped = original - rounded
            tokens_dropped += dropped

            # Assume we drop lowest probability tokens
            expert_tokens = expert_token_map[expert_idx]
            if len(expert_tokens) >= dropped:
                # Simplified: use average routing prob for expert
                avg_prob = routing_probs[:, expert_idx].mean()
                total_dropped_prob += dropped * avg_prob

    avg_dropped_prob = total_dropped_prob / tokens_dropped if tokens_dropped > 0 else 0
    drop_percentage = (tokens_dropped / total_tokens) * 100

    # Estimate perplexity increase (simplified)
    perplexity_increase = avg_dropped_prob * drop_percentage / 100 * 0.1  # Heuristic

    return {
        'tokens_dropped': tokens_dropped,
        'drop_percentage': drop_percentage,
        'avg_dropped_prob': avg_dropped_prob,
        'estimated_perplexity_increase': perplexity_increase,
    }


def benchmark_comparison(
    num_experts: int = 256,
    total_tokens: int = 16384,
    top_k: int = 8,
    tile_size: int = 128
):
    """Compare vanilla vs tile-aware routing performance"""

    print("=" * 60)
    print("Tile-Aware Token Rounding Benchmark")
    print("=" * 60 + "\n")

    print(f"Configuration:")
    print(f"  - Num Experts: {num_experts}")
    print(f"  - Total Tokens: {total_tokens}")
    print(f"  - Top-k: {top_k}")
    print(f"  - Tile Size: {tile_size}\n")

    # Simulate routing
    token_counts, routing_probs = simulate_routing(num_experts, total_tokens, top_k)

    # Vanilla routing
    print("Before Rounding:")
    vanilla_util = compute_tile_utilization(token_counts, tile_size)
    print(f"  - Total Tiles: {vanilla_util['total_tiles']}")
    print(f"  - Avg Tile Utilization: {vanilla_util['avg_utilization']*100:.1f}%")
    print(f"  - Wasted Computation: {vanilla_util['wasted_percentage']:.1f}%\n")

    # Show some examples
    print("  Example Expert Token Counts:")
    for i in range(min(5, num_experts)):
        count = token_counts[i]
        num_tiles = (count + tile_size - 1) // tile_size
        remainder = count % tile_size
        util = (remainder / tile_size * 100) if remainder > 0 else 100
        print(f"    Expert {i}: {count} tokens → {num_tiles} tiles " +
              f"(last tile: {util:.1f}% utilized)")
    print()

    # Round-down strategy
    rounded_down, dropped = round_down_token_counts(token_counts, tile_size)
    print("After Rounding (Round Down):")
    down_util = compute_tile_utilization(rounded_down, tile_size)
    print(f"  - Total Tiles: {down_util['total_tiles']}")
    print(f"  - Avg Tile Utilization: {down_util['avg_utilization']*100:.1f}%")
    print(f"  - Tokens Dropped: {dropped} ({dropped/sum(token_counts)*100:.1f}%)\n")

    # Quality impact
    quality = analyze_quality_impact(token_counts, rounded_down, routing_probs, num_experts)
    print("Quality Impact (Round Down):")
    print(f"  - Tokens Dropped: {quality['tokens_dropped']}")
    print(f"  - Drop Percentage: {quality['drop_percentage']:.2f}%")
    print(f"  - Avg Routing Prob of Dropped: {quality['avg_dropped_prob']:.4f}")
    print(f"  - Estimated Perplexity Increase: +{quality['estimated_perplexity_increase']:.2f}%\n")

    # Round-up strategy
    rounded_up, added = round_up_token_counts(token_counts, tile_size)
    print("After Rounding (Round Up):")
    up_util = compute_tile_utilization(rounded_up, tile_size)
    print(f"  - Total Tiles: {up_util['total_tiles']}")
    print(f"  - Avg Tile Utilization: {up_util['avg_utilization']*100:.1f}%")
    print(f"  - Tokens Added (padding): {added} ({added/sum(token_counts)*100:.1f}%)\n")

    # Adaptive strategy
    rounded_adaptive, dropped_adp, added_adp = adaptive_rounding(token_counts, tile_size, 0.9)
    print("After Rounding (Adaptive, threshold=0.9):")
    adaptive_util = compute_tile_utilization(rounded_adaptive, tile_size)
    print(f"  - Total Tiles: {adaptive_util['total_tiles']}")
    print(f"  - Avg Tile Utilization: {adaptive_util['avg_utilization']*100:.1f}%")
    print(f"  - Tokens Dropped: {dropped_adp}")
    print(f"  - Tokens Added: {added_adp}\n")

    # Performance estimation
    print("Performance Estimation:")
    print(f"  - Vanilla Relative Time: 1.00x")
    speedup_down = 1.0 / vanilla_util['avg_utilization']
    print(f"  - Round Down Speedup: {speedup_down:.2f}x")
    speedup_up = (down_util['total_tiles'] / vanilla_util['total_tiles']) * speedup_down
    print(f"  - Round Up Speedup: {speedup_up:.2f}x")
    speedup_adaptive = (adaptive_util['total_tiles'] / vanilla_util['total_tiles']) * speedup_down
    print(f"  - Adaptive Speedup: {speedup_adaptive:.2f}x")

    print("\n" + "=" * 60)


def tile_size_sensitivity():
    """Analyze sensitivity to tile size choice"""

    print("\nTile Size Sensitivity Analysis")
    print("=" * 60 + "\n")

    num_experts = 256
    total_tokens = 16384
    top_k = 8

    tile_sizes = [32, 64, 128, 256, 512]

    print(f"Configuration: {num_experts} experts, {total_tokens} tokens, top-{top_k}\n")
    print(f"{'Tile Size':<12} {'Utilization':<15} {'Dropped %':<12} {'Speedup':<10}")
    print("-" * 60)

    token_counts, routing_probs = simulate_routing(num_experts, total_tokens, top_k)

    for tile_size in tile_sizes:
        vanilla_util = compute_tile_utilization(token_counts, tile_size)
        rounded, dropped = round_down_token_counts(token_counts, tile_size)
        rounded_util = compute_tile_utilization(rounded, tile_size)

        drop_pct = (dropped / sum(token_counts)) * 100
        speedup = 1.0 / vanilla_util['avg_utilization']

        print(f"{tile_size:<12} {rounded_util['avg_utilization']*100:>6.1f}% " +
              f"        {drop_pct:>6.2f}%      {speedup:>6.2f}x")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    # Main benchmark
    benchmark_comparison(
        num_experts=256,
        total_tokens=16384,
        top_k=8,
        tile_size=128
    )

    # Tile size sensitivity
    tile_size_sensitivity()
