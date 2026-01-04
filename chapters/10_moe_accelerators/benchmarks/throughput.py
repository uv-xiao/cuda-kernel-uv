"""
Throughput Calculation for MoE Models

Calculate:
- Tokens per second
- Tokens per day (training throughput)
- Multi-GPU scaling
- Cost analysis
"""

import argparse
from typing import Dict


# Model configurations
CONFIGS = {
    'small': {
        'name': 'Small MoE',
        'num_experts': 8,
        'hidden_dim': 1024,
        'ffn_dim': 4096,
        'top_k': 2,
        'num_layers': 12,
        'total_params': '1.2B',
    },
    'medium': {
        'name': 'Medium MoE',
        'num_experts': 64,
        'hidden_dim': 4096,
        'ffn_dim': 14336,
        'top_k': 4,
        'num_layers': 32,
        'total_params': '52B',
    },
    'deepseek-v3': {
        'name': 'DeepSeek-V3.2-Exp',
        'num_experts': 256,
        'hidden_dim': 7168,
        'ffn_dim': 14336,
        'top_k': 8,
        'num_layers': 61,
        'total_params': '685B',
    },
}

# Hardware specs
HARDWARE = {
    'a100': {
        'name': 'NVIDIA A100 (80GB)',
        'peak_tflops_fp16': 312,
        'memory_gb': 80,
        'cost_per_hour': 2.50,
    },
    'h100': {
        'name': 'NVIDIA H100 (80GB)',
        'peak_tflops_fp16': 990,
        'memory_gb': 80,
        'cost_per_hour': 3.50,
    },
    'h200': {
        'name': 'NVIDIA H200 (141GB)',
        'peak_tflops_fp16': 990,
        'memory_gb': 141,
        'cost_per_hour': 4.00,
    },
}


def calculate_moe_layer_latency(
    config: Dict,
    hardware: Dict,
    batch_size: int = 32,
    seq_len: int = 512,
    use_sonicmoe: bool = False
) -> float:
    """
    Calculate MoE layer latency in milliseconds

    Simplified model based on GEMM time:
    time ≈ (total_flops / hardware_tflops) * efficiency_factor
    """

    num_tokens = batch_size * seq_len
    hidden_dim = config['hidden_dim']
    ffn_dim = config['ffn_dim']
    top_k = config['top_k']

    # Average tokens per expert
    tokens_per_expert = (num_tokens * top_k) / config['num_experts']

    # FLOPs per expert (up-projection + down-projection)
    flops_per_expert = 2 * tokens_per_expert * (
        hidden_dim * ffn_dim +  # fc1
        ffn_dim * hidden_dim    # fc2
    )

    # Total FLOPs
    total_flops = flops_per_expert * config['num_experts']
    total_tflops = total_flops / 1e12

    # Hardware efficiency
    # Baseline: ~45% of peak (typical for irregular MoE)
    # SonicMoE: ~78% of peak (optimized)
    efficiency = 0.78 if use_sonicmoe else 0.45

    # Calculate time
    peak_tflops = hardware['peak_tflops_fp16']
    achieved_tflops = peak_tflops * efficiency

    latency_ms = (total_tflops / achieved_tflops) * 1000

    return latency_ms


def calculate_full_model_latency(
    config: Dict,
    hardware: Dict,
    batch_size: int = 32,
    seq_len: int = 512,
    use_sonicmoe: bool = False
) -> float:
    """Calculate full model forward pass latency"""

    # MoE layer latency
    moe_latency = calculate_moe_layer_latency(
        config, hardware, batch_size, seq_len, use_sonicmoe
    )

    # Attention and other layers (simplified)
    # Roughly 30% of total time for MoE models
    attention_latency = moe_latency * 0.3

    # Per-layer latency
    layer_latency = moe_latency + attention_latency

    # Full model
    total_latency = layer_latency * config['num_layers']

    return total_latency


def calculate_throughput(
    config: Dict,
    hardware: Dict,
    batch_size: int = 32,
    seq_len: int = 512,
    num_gpus: int = 1,
    use_sonicmoe: bool = False
) -> Dict:
    """Calculate training/inference throughput"""

    # Single GPU latency
    single_gpu_latency_ms = calculate_full_model_latency(
        config, hardware, batch_size, seq_len, use_sonicmoe
    )

    # Multi-GPU scaling (simplified, assumes good parallelism)
    # Pipeline parallelism efficiency: ~0.85
    # Tensor parallelism efficiency: ~0.90
    scaling_efficiency = 0.85 if num_gpus > 1 else 1.0

    effective_latency_ms = single_gpu_latency_ms / (num_gpus * scaling_efficiency)

    # Tokens per forward pass
    tokens_per_pass = batch_size * seq_len

    # Throughput
    tokens_per_second = (tokens_per_pass / effective_latency_ms) * 1000
    tokens_per_day = tokens_per_second * 86400

    # Training (forward + backward ≈ 3x forward)
    training_tokens_per_day = tokens_per_day / 3

    # Calculate time to 1T tokens
    one_trillion = 1e12
    days_to_1t = one_trillion / training_tokens_per_day

    return {
        'latency_ms': effective_latency_ms,
        'tokens_per_second': tokens_per_second,
        'tokens_per_day': tokens_per_day,
        'training_tokens_per_day': training_tokens_per_day,
        'days_to_1t_tokens': days_to_1t,
    }


def cost_analysis(
    config: Dict,
    hardware: Dict,
    num_gpus: int = 8,
    target_tokens: float = 1e12,  # 1T tokens
    use_sonicmoe: bool = False
):
    """Analyze training cost"""

    # Calculate throughput
    throughput = calculate_throughput(
        config, hardware,
        batch_size=32, seq_len=512,
        num_gpus=num_gpus,
        use_sonicmoe=use_sonicmoe
    )

    # Training time
    training_days = target_tokens / throughput['training_tokens_per_day']
    training_hours = training_days * 24

    # GPU hours
    gpu_hours = training_hours * num_gpus

    # Cost
    cost_per_hour = hardware['cost_per_hour']
    total_cost = gpu_hours * cost_per_hour

    return {
        'training_days': training_days,
        'training_hours': training_hours,
        'gpu_hours': gpu_hours,
        'total_cost': total_cost,
        'cost_per_b_tokens': (total_cost / (target_tokens / 1e9)),
    }


def main():
    parser = argparse.ArgumentParser(description='MoE Throughput Calculator')
    parser.add_argument('--config', choices=CONFIGS.keys(), default='medium',
                        help='Model configuration')
    parser.add_argument('--hardware', choices=HARDWARE.keys(), default='h100',
                        help='Hardware platform')
    parser.add_argument('--num-gpus', type=int, default=8,
                        help='Number of GPUs')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--seq-len', type=int, default=512,
                        help='Sequence length')

    args = parser.parse_args()

    config = CONFIGS[args.config]
    hardware = HARDWARE[args.hardware]

    print("=" * 80)
    print("MoE Throughput Analysis")
    print("=" * 80 + "\n")

    print(f"Model: {config['name']} ({config['total_params']} parameters)")
    print(f"  - Num Experts: {config['num_experts']}")
    print(f"  - Top-k: {config['top_k']}")
    print(f"  - Num Layers: {config['num_layers']}\n")

    print(f"Hardware: {hardware['name']}")
    print(f"  - Num GPUs: {args.num_gpus}")
    print(f"  - Peak TFLOPs (FP16): {hardware['peak_tflops_fp16']}")
    print(f"  - Memory: {hardware['memory_gb']} GB\n")

    print(f"Workload:")
    print(f"  - Batch Size: {args.batch_size}")
    print(f"  - Sequence Length: {args.seq_len}")
    print(f"  - Tokens per Pass: {args.batch_size * args.seq_len}\n")

    # Baseline throughput
    print("=" * 80)
    print("Baseline (Vanilla PyTorch)")
    print("=" * 80)

    baseline_tp = calculate_throughput(
        config, hardware, args.batch_size, args.seq_len,
        args.num_gpus, use_sonicmoe=False
    )

    print(f"Latency: {baseline_tp['latency_ms']:.2f} ms")
    print(f"Tokens/sec: {baseline_tp['tokens_per_second']:,.0f}")
    print(f"Tokens/day (inference): {baseline_tp['tokens_per_day']/1e9:.2f}B")
    print(f"Tokens/day (training): {baseline_tp['training_tokens_per_day']/1e9:.2f}B")
    print(f"Days to 1T tokens: {baseline_tp['days_to_1t_tokens']:.1f}\n")

    # SonicMoE throughput
    print("=" * 80)
    print("Optimized (SonicMoE)")
    print("=" * 80)

    optimized_tp = calculate_throughput(
        config, hardware, args.batch_size, args.seq_len,
        args.num_gpus, use_sonicmoe=True
    )

    print(f"Latency: {optimized_tp['latency_ms']:.2f} ms")
    print(f"Tokens/sec: {optimized_tp['tokens_per_second']:,.0f}")
    print(f"Tokens/day (inference): {optimized_tp['tokens_per_day']/1e9:.2f}B")
    print(f"Tokens/day (training): {optimized_tp['training_tokens_per_day']/1e9:.2f}B")
    print(f"Days to 1T tokens: {optimized_tp['days_to_1t_tokens']:.1f}\n")

    # Improvement
    speedup = baseline_tp['latency_ms'] / optimized_tp['latency_ms']
    print(f"Speedup: {speedup:.2f}x")
    print(f"Time Saved: {baseline_tp['days_to_1t_tokens'] - optimized_tp['days_to_1t_tokens']:.1f} days\n")

    # Cost analysis
    print("=" * 80)
    print("Cost Analysis (1T tokens)")
    print("=" * 80)

    baseline_cost = cost_analysis(config, hardware, args.num_gpus, 1e12, False)
    optimized_cost = cost_analysis(config, hardware, args.num_gpus, 1e12, True)

    print(f"{'Metric':<30} {'Baseline':<20} {'SonicMoE':<20}")
    print("-" * 80)
    print(f"{'Training Days':<30} {baseline_cost['training_days']:>15.1f}      "
          f"{optimized_cost['training_days']:>15.1f}")
    print(f"{'GPU Hours':<30} {baseline_cost['gpu_hours']:>15.0f}      "
          f"{optimized_cost['gpu_hours']:>15.0f}")
    print(f"{'Total Cost':<30} ${baseline_cost['total_cost']:>14,.0f}      "
          f"${optimized_cost['total_cost']:>14,.0f}")
    print(f"{'Cost per B tokens':<30} ${baseline_cost['cost_per_b_tokens']:>14.2f}      "
          f"${optimized_cost['cost_per_b_tokens']:>14.2f}")

    savings = baseline_cost['total_cost'] - optimized_cost['total_cost']
    print(f"\nCost Savings: ${savings:,.0f} ({savings/baseline_cost['total_cost']*100:.1f}%)")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
