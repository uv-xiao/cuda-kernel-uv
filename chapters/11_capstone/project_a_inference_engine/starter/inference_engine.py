"""
Mini LLM Inference Engine - Main Implementation

This is the skeleton code for the inference engine. You need to implement:
1. Attention mechanism (Flash Attention or Sparse Attention)
2. MoE layer with optimized routing and expert dispatch
3. Integration into end-to-end pipeline
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
import time

from attention import AttentionLayer
from moe import MoELayer
from utils import LayerNorm, get_device_properties


class TransformerConfig:
    """Configuration for transformer model"""
    def __init__(
        self,
        vocab_size: int = 32000,
        hidden_dim: int = 4096,
        num_layers: int = 32,
        num_heads: int = 32,
        head_dim: int = 128,
        intermediate_dim: int = 11008,
        num_experts: int = 8,
        num_experts_per_token: int = 2,
        max_seq_len: int = 8192,
        use_moe: bool = True,
        use_flash_attention: bool = True,
    ):
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.intermediate_dim = intermediate_dim
        self.num_experts = num_experts
        self.num_experts_per_token = num_experts_per_token
        self.max_seq_len = max_seq_len
        self.use_moe = use_moe
        self.use_flash_attention = use_flash_attention


class TransformerBlock(nn.Module):
    """Single transformer block with attention and MoE/FFN"""

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config

        # TODO: Initialize attention layer
        # self.attention = AttentionLayer(...)
        self.attention = None

        # TODO: Initialize MoE or standard FFN
        if config.use_moe:
            # self.moe = MoELayer(...)
            self.moe = None
        else:
            # Standard FFN as fallback
            self.ffn = nn.Sequential(
                nn.Linear(config.hidden_dim, config.intermediate_dim),
                nn.GELU(),
                nn.Linear(config.intermediate_dim, config.hidden_dim),
            )

        # Layer normalization
        self.norm1 = LayerNorm(config.hidden_dim)
        self.norm2 = LayerNorm(config.hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through transformer block

        Args:
            x: Input tensor (batch_size, seq_len, hidden_dim)
            attention_mask: Optional attention mask

        Returns:
            Output tensor (batch_size, seq_len, hidden_dim)
        """
        # TODO: Implement forward pass
        # 1. Self-attention with residual connection
        #    x = x + self.attention(self.norm1(x), attention_mask)
        # 2. MoE/FFN with residual connection
        #    x = x + self.moe(self.norm2(x))

        raise NotImplementedError("Forward pass not implemented")


class InferenceEngine(nn.Module):
    """Main inference engine for LLM"""

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config

        # Token embedding
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_dim)

        # Positional embedding (learned or RoPE)
        self.position_embedding = nn.Embedding(config.max_seq_len, config.hidden_dim)

        # Transformer blocks
        self.layers = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.num_layers)
        ])

        # Final layer norm and projection
        self.final_norm = LayerNorm(config.hidden_dim)
        self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)

        # Statistics
        self.reset_stats()

    def reset_stats(self):
        """Reset performance statistics"""
        self.stats = {
            'forward_time': [],
            'attention_time': [],
            'moe_time': [],
            'total_tokens': 0,
        }

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through entire model

        Args:
            input_ids: Input token IDs (batch_size, seq_len)
            attention_mask: Optional attention mask (batch_size, seq_len)

        Returns:
            logits: Output logits (batch_size, seq_len, vocab_size)
        """
        batch_size, seq_len = input_ids.shape

        # TODO: Implement forward pass
        # 1. Get token embeddings
        # 2. Add positional embeddings
        # 3. Pass through transformer blocks
        # 4. Apply final norm and projection

        raise NotImplementedError("Forward pass not implemented")

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Generate text autoregressively

        Args:
            prompt_ids: Prompt token IDs (batch_size, prompt_len)
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Nucleus sampling parameter

        Returns:
            generated_ids: Generated token IDs (batch_size, prompt_len + max_new_tokens)
        """
        # TODO: Implement autoregressive generation
        # 1. Start with prompt
        # 2. For each new token:
        #    - Run forward pass
        #    - Sample next token
        #    - Append to sequence

        raise NotImplementedError("Generation not implemented")

    def benchmark(
        self,
        batch_sizes: list = [1, 4, 16, 32],
        seq_lengths: list = [128, 512, 2048, 8192],
        num_iterations: int = 100,
        warmup: int = 10,
    ):
        """
        Benchmark inference performance

        Args:
            batch_sizes: List of batch sizes to test
            seq_lengths: List of sequence lengths to test
            num_iterations: Number of iterations for each configuration
            warmup: Number of warmup iterations
        """
        print("Running benchmarks...")
        print(f"Device: {next(self.parameters()).device}")

        results = []

        for batch_size in batch_sizes:
            for seq_len in seq_lengths:
                # Create dummy input
                input_ids = torch.randint(
                    0, self.config.vocab_size,
                    (batch_size, seq_len),
                    device=next(self.parameters()).device
                )

                # Warmup
                for _ in range(warmup):
                    _ = self.forward(input_ids)

                # Benchmark
                torch.cuda.synchronize()
                start_time = time.perf_counter()

                for _ in range(num_iterations):
                    _ = self.forward(input_ids)

                torch.cuda.synchronize()
                end_time = time.perf_counter()

                # Calculate metrics
                avg_time = (end_time - start_time) / num_iterations
                throughput = (batch_size * seq_len) / avg_time

                result = {
                    'batch_size': batch_size,
                    'seq_len': seq_len,
                    'avg_time_ms': avg_time * 1000,
                    'throughput_tokens_per_sec': throughput,
                }
                results.append(result)

                print(f"Batch={batch_size}, SeqLen={seq_len}: "
                      f"{avg_time*1000:.2f}ms, {throughput:.1f} tokens/s")

        return results

    def profile(
        self,
        batch_size: int = 16,
        seq_len: int = 2048,
    ):
        """
        Profile model to identify bottlenecks

        Args:
            batch_size: Batch size for profiling
            seq_len: Sequence length for profiling
        """
        print(f"\nProfiling with batch_size={batch_size}, seq_len={seq_len}")

        # Create dummy input
        input_ids = torch.randint(
            0, self.config.vocab_size,
            (batch_size, seq_len),
            device=next(self.parameters()).device
        )

        # TODO: Add detailed profiling
        # 1. Time each component
        # 2. Measure memory usage
        # 3. Compute FLOPS utilization

        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        ) as prof:
            _ = self.forward(input_ids)

        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

        return prof


def main():
    """Example usage and testing"""

    # Print device info
    print("Device Information:")
    props = get_device_properties()
    for key, value in props.items():
        print(f"  {key}: {value}")

    # Create small model for testing
    config = TransformerConfig(
        vocab_size=32000,
        hidden_dim=2048,
        num_layers=4,
        num_heads=16,
        head_dim=128,
        intermediate_dim=5504,
        num_experts=8,
        num_experts_per_token=2,
        max_seq_len=4096,
        use_moe=True,
        use_flash_attention=True,
    )

    print(f"\nModel Configuration:")
    print(f"  Hidden dim: {config.hidden_dim}")
    print(f"  Num layers: {config.num_layers}")
    print(f"  Num heads: {config.num_heads}")
    print(f"  Num experts: {config.num_experts}")
    print(f"  Use MoE: {config.use_moe}")
    print(f"  Use Flash Attention: {config.use_flash_attention}")

    # Create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = InferenceEngine(config).to(device)

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {num_params:,}")

    # Test forward pass
    print("\nTesting forward pass...")
    batch_size, seq_len = 2, 128
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)

    try:
        output = model(input_ids)
        print(f"  Input shape: {input_ids.shape}")
        print(f"  Output shape: {output.shape}")
        print(f"  Success!")
    except NotImplementedError as e:
        print(f"  Not implemented yet: {e}")
    except Exception as e:
        print(f"  Error: {e}")

    # Run benchmarks
    print("\nRunning benchmarks...")
    try:
        results = model.benchmark(
            batch_sizes=[1, 4],
            seq_lengths=[128, 512],
            num_iterations=10,
            warmup=2,
        )
    except NotImplementedError:
        print("  Benchmarking not implemented yet")

    # Profile
    print("\nProfiling...")
    try:
        prof = model.profile(batch_size=4, seq_len=512)
    except NotImplementedError:
        print("  Profiling not implemented yet")


if __name__ == "__main__":
    main()
