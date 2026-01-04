"""
Routing Comparison: Vanilla vs Tile-Aware

Compare expert routing strategies and their impact on performance.
"""

import torch
import numpy as np
import time
from typing import Dict, List
import sys
from pathlib import Path

# Import from token_rounding module
sys.path.append(str(Path(__file__).parent))
from token_rounding import (
    simulate_routing,
    round_down_token_counts,
    compute_tile_utilization,
    analyze_quality_impact
)


def visualize_expert_distribution(
    token_counts: List[int],
    title: str = "Expert Load Distribution"
):
    """Simple text-based visualization of expert loads"""

    max_count = max(token_counts) if token_counts else 1
    bar_width = 50

    print(f"\n{title}")
    print("-" * 60)

    for i, count in enumerate(token_counts):
        bar_len = int((count / max_count) * bar_width)
        bar = "█" * bar_len
        print(f"Expert {i:3d}: {bar:<{bar_width}} {count:4d}")

    print("-" * 60)


def compare_routing_strategies(
    num_experts: int = 64,
    total_tokens: int = 16384,
    top_k: int = 4,
    tile_size: int = 128
):
    """Compare vanilla vs tile-aware routing"""

    print("=" * 70)
    print("Routing Strategy Comparison")
    print("=" * 70 + "\n")

    print(f"Configuration:")
    print(f"  Num Experts: {num_experts}")
    print(f"  Total Tokens: {total_tokens}")
    print(f"  Top-k: {top_k}")
    print(f"  Tile Size: {tile_size}\n")

    # Simulate routing
    token_counts, routing_probs = simulate_routing(num_experts, total_tokens, top_k)

    # Vanilla routing
    print("Strategy 1: Vanilla Routing (No Rounding)")
    print("-" * 70)

    vanilla_util = compute_tile_utilization(token_counts, tile_size)

    print(f"Tile Utilization: {vanilla_util['avg_utilization']*100:.1f}%")
    print(f"Wasted Computation: {vanilla_util['wasted_percentage']:.1f}%")
    print(f"Total Tiles: {vanilla_util['total_tiles']}")

    # Show distribution for first 10 experts
    visualize_expert_distribution(token_counts[:10], "Expert Load (First 10 Experts)")

    # Compute metrics
    vanilla_std = np.std(token_counts)
    vanilla_cv = vanilla_std / (np.mean(token_counts) + 1e-6)
    vanilla_gini = compute_gini(token_counts)

    print(f"\nLoad Balance Metrics:")
    print(f"  Coefficient of Variation: {vanilla_cv:.3f}")
    print(f"  Gini Coefficient: {vanilla_gini:.3f}")
    print(f"  (Lower is better, 0 = perfectly balanced)")

    # Tile-aware routing
    print("\n\nStrategy 2: Tile-Aware Routing (Round Down)")
    print("-" * 70)

    rounded_counts, dropped = round_down_token_counts(token_counts, tile_size)
    tiled_util = compute_tile_utilization(rounded_counts, tile_size)

    print(f"Tile Utilization: {tiled_util['avg_utilization']*100:.1f}%")
    print(f"Wasted Computation: {tiled_util['wasted_percentage']:.1f}%")
    print(f"Total Tiles: {tiled_util['total_tiles']}")
    print(f"Tokens Dropped: {dropped} ({dropped/sum(token_counts)*100:.1f}%)")

    visualize_expert_distribution(rounded_counts[:10], "Expert Load (First 10 Experts)")

    # Metrics
    tiled_std = np.std(rounded_counts)
    tiled_cv = tiled_std / (np.mean(rounded_counts) + 1e-6)
    tiled_gini = compute_gini(rounded_counts)

    print(f"\nLoad Balance Metrics:")
    print(f"  Coefficient of Variation: {tiled_cv:.3f}")
    print(f"  Gini Coefficient: {tiled_gini:.3f}")

    # Quality impact
    quality = analyze_quality_impact(token_counts, rounded_counts, routing_probs, num_experts)
    print(f"\nQuality Impact:")
    print(f"  Estimated Perplexity Increase: +{quality['estimated_perplexity_increase']:.2f}%")

    # Performance comparison
    print("\n\nPerformance Comparison")
    print("=" * 70)

    # Simulated GEMM time (proportional to number of tiles and utilization)
    vanilla_time = vanilla_util['total_tiles'] / vanilla_util['avg_utilization']
    tiled_time = tiled_util['total_tiles']  # Perfect utilization

    speedup = vanilla_time / tiled_time

    print(f"{'Metric':<30} {'Vanilla':<15} {'Tile-Aware':<15} {'Change':<15}")
    print("-" * 70)
    print(f"{'Tile Utilization':<30} {vanilla_util['avg_utilization']*100:>8.1f}%      "
          f"{tiled_util['avg_utilization']*100:>8.1f}%      "
          f"{(tiled_util['avg_utilization']/vanilla_util['avg_utilization']-1)*100:>+8.1f}%")
    print(f"{'Relative Compute Time':<30} {vanilla_time:>11.2f}      "
          f"{tiled_time:>11.2f}      {speedup:>11.2f}x")
    print(f"{'Load Balance (CV)':<30} {vanilla_cv:>11.3f}      "
          f"{tiled_cv:>11.3f}      {(tiled_cv-vanilla_cv):>+11.3f}")
    print(f"{'Token Drop Rate':<30} {'0.0%':>11}      "
          f"{dropped/sum(token_counts)*100:>10.1f}%      "
          f"{dropped/sum(token_counts)*100:>+10.1f}%")

    print("\n" + "=" * 70)


def compute_gini(values: List[int]) -> float:
    """Compute Gini coefficient for load imbalance"""

    values = np.array(values, dtype=float)
    n = len(values)

    if n == 0 or values.sum() == 0:
        return 0.0

    sorted_values = np.sort(values)
    index = np.arange(1, n + 1)

    gini = ((2 * index - n - 1) * sorted_values).sum() / (n * sorted_values.sum())

    return gini


def sensitivity_analysis():
    """Analyze sensitivity to different parameters"""

    print("\n\n" + "=" * 70)
    print("Sensitivity Analysis")
    print("=" * 70 + "\n")

    # Vary number of experts
    print("Effect of Number of Experts:")
    print("-" * 70)
    print(f"{'Num Experts':<15} {'Tile Util':<15} {'Speedup':<15} {'Drop Rate':<15}")
    print("-" * 70)

    for num_experts in [16, 32, 64, 128, 256]:
        token_counts, _ = simulate_routing(num_experts, 16384, 4)
        vanilla_util = compute_tile_utilization(token_counts, 128)

        rounded, dropped = round_down_token_counts(token_counts, 128)
        tiled_util = compute_tile_utilization(rounded, 128)

        speedup = 1.0 / vanilla_util['avg_utilization']
        drop_rate = dropped / sum(token_counts) * 100

        print(f"{num_experts:<15} {tiled_util['avg_utilization']*100:>8.1f}%      "
              f"{speedup:>11.2f}x      {drop_rate:>10.1f}%")

    print()


if __name__ == "__main__":
    # Main comparison
    compare_routing_strategies(
        num_experts=64,
        total_tokens=16384,
        top_k=4,
        tile_size=128
    )

    # Sensitivity analysis
    sensitivity_analysis()
