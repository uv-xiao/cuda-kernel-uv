"""
Sparse Attention Pattern Definition

Define your custom sparse attention pattern here.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional


class SparseAttentionPattern:
    """
    Base class for sparse attention patterns

    Your custom pattern should inherit from this class and implement
    the generate_mask method.
    """

    def __init__(self, config: dict):
        """
        Initialize sparse pattern

        Args:
            config: Configuration dictionary with pattern-specific parameters
        """
        self.config = config

    def generate_mask(
        self,
        seq_len: int,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """
        Generate sparse attention mask

        Args:
            seq_len: Sequence length
            device: Device to create mask on

        Returns:
            mask: (seq_len, seq_len) boolean tensor where True means attend
        """
        raise NotImplementedError("Subclass must implement generate_mask")

    def get_sparsity(self, mask: torch.Tensor) -> float:
        """
        Calculate sparsity ratio

        Args:
            mask: Attention mask

        Returns:
            sparsity: Fraction of zero entries (0.0 = dense, 1.0 = fully sparse)
        """
        return 1.0 - mask.float().mean().item()

    def visualize(self, seq_len: int = 512, save_path: Optional[str] = None):
        """
        Visualize the sparsity pattern

        Args:
            seq_len: Sequence length to visualize
            save_path: Optional path to save figure
        """
        mask = self.generate_mask(seq_len, device=torch.device('cpu'))
        mask_np = mask.numpy()

        plt.figure(figsize=(10, 10))
        plt.imshow(mask_np, cmap='Blues', interpolation='nearest')
        plt.xlabel('Key Position')
        plt.ylabel('Query Position')
        plt.title(f'{self.__class__.__name__} (Sparsity: {self.get_sparsity(mask):.2%})')
        plt.colorbar(label='Attention Mask')

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved visualization to {save_path}")
        else:
            plt.show()

    def get_attention_indices(
        self,
        seq_len: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get sparse attention indices

        Returns:
            query_indices: (num_nonzero,) tensor of query positions
            key_indices: (num_nonzero,) tensor of key positions

        Example:
            For attention from query 0 to keys [0, 1, 2]:
            query_indices = [0, 0, 0]
            key_indices = [0, 1, 2]
        """
        mask = self.generate_mask(seq_len)
        query_indices, key_indices = torch.where(mask)
        return query_indices, key_indices


class LocalAttentionPattern(SparseAttentionPattern):
    """
    Local (sliding window) attention pattern

    Each token attends to a fixed window of neighbors.
    """

    def __init__(self, window_size: int):
        """
        Args:
            window_size: Size of local window on each side
        """
        super().__init__({'window_size': window_size})
        self.window_size = window_size

    def generate_mask(self, seq_len: int, device: Optional[torch.device] = None) -> torch.Tensor:
        if device is None:
            device = torch.device('cpu')

        # Create mask
        mask = torch.zeros(seq_len, seq_len, dtype=torch.bool, device=device)

        # Fill in local windows
        for i in range(seq_len):
            start = max(0, i - self.window_size)
            end = min(seq_len, i + self.window_size + 1)
            mask[i, start:end] = True

        return mask


class StridedAttentionPattern(SparseAttentionPattern):
    """
    Strided attention pattern

    Each token attends to every stride-th token.
    """

    def __init__(self, stride: int, local_window: int = 0):
        """
        Args:
            stride: Stride for global attention
            local_window: Optional local window size
        """
        super().__init__({'stride': stride, 'local_window': local_window})
        self.stride = stride
        self.local_window = local_window

    def generate_mask(self, seq_len: int, device: Optional[torch.device] = None) -> torch.Tensor:
        if device is None:
            device = torch.device('cpu')

        mask = torch.zeros(seq_len, seq_len, dtype=torch.bool, device=device)

        # Add strided attention
        for i in range(seq_len):
            # Strided positions
            for j in range(0, seq_len, self.stride):
                mask[i, j] = True

            # Optional local window
            if self.local_window > 0:
                start = max(0, i - self.local_window)
                end = min(seq_len, i + self.local_window + 1)
                mask[i, start:end] = True

        return mask


class BlockSparsePattern(SparseAttentionPattern):
    """
    Block-sparse attention pattern

    Attention is computed at block granularity.
    """

    def __init__(self, block_size: int):
        """
        Args:
            block_size: Size of attention blocks
        """
        super().__init__({'block_size': block_size})
        self.block_size = block_size

    def generate_mask(self, seq_len: int, device: Optional[torch.device] = None) -> torch.Tensor:
        if device is None:
            device = torch.device('cpu')

        mask = torch.zeros(seq_len, seq_len, dtype=torch.bool, device=device)

        num_blocks = (seq_len + self.block_size - 1) // self.block_size

        # Each block attends to itself and neighboring blocks
        for i in range(num_blocks):
            for j in range(max(0, i - 1), min(num_blocks, i + 2)):
                # Fill block
                i_start = i * self.block_size
                i_end = min(seq_len, (i + 1) * self.block_size)
                j_start = j * self.block_size
                j_end = min(seq_len, (j + 1) * self.block_size)

                mask[i_start:i_end, j_start:j_end] = True

        return mask


class CustomSparsePattern(SparseAttentionPattern):
    """
    YOUR CUSTOM SPARSE PATTERN

    Implement your novel sparsity pattern here!

    Example: Adaptive Local-Global Pattern
    - Local window for nearby context
    - Global landmark tokens for distant context
    - Adaptive selection of landmarks
    """

    def __init__(self, **kwargs):
        """
        Initialize your custom pattern

        Args:
            **kwargs: Pattern-specific parameters
                For example:
                - local_window: int
                - num_landmarks: int
                - landmark_stride: int
                - adaptive: bool
        """
        super().__init__(kwargs)

        # TODO: Initialize your pattern parameters
        # Example:
        # self.local_window = kwargs.get('local_window', 128)
        # self.num_landmarks = kwargs.get('num_landmarks', 64)
        # self.landmark_stride = kwargs.get('landmark_stride', 128)

        raise NotImplementedError("Implement your custom pattern")

    def generate_mask(self, seq_len: int, device: Optional[torch.device] = None) -> torch.Tensor:
        """
        Generate your custom sparse mask

        TODO: Implement your pattern logic
        """
        if device is None:
            device = torch.device('cpu')

        # TODO: Implement your custom pattern
        # Example structure:
        # 1. Create empty mask
        # mask = torch.zeros(seq_len, seq_len, dtype=torch.bool, device=device)

        # 2. Add local attention
        # for i in range(seq_len):
        #     start = max(0, i - self.local_window)
        #     end = min(seq_len, i + self.local_window + 1)
        #     mask[i, start:end] = True

        # 3. Add global/landmark attention
        # landmark_indices = self._select_landmarks(seq_len)
        # for i in range(seq_len):
        #     mask[i, landmark_indices] = True
        # for l in landmark_indices:
        #     mask[l, :] = True  # Landmarks attend to all

        # return mask

        raise NotImplementedError("Implement generate_mask for your pattern")

    def _select_landmarks(self, seq_len: int) -> torch.Tensor:
        """
        Select landmark tokens for global attention

        TODO: Implement landmark selection strategy
        Options:
        - Fixed stride
        - Random sampling
        - Content-based (requires input tokens)
        - Structural markers
        """
        raise NotImplementedError("Implement landmark selection")


def compare_patterns(seq_len: int = 512):
    """Compare different sparse patterns"""

    patterns = [
        ('Local (window=64)', LocalAttentionPattern(window_size=64)),
        ('Strided (stride=64)', StridedAttentionPattern(stride=64)),
        ('Block Sparse (block=64)', BlockSparsePattern(block_size=64)),
    ]

    fig, axes = plt.subplots(1, len(patterns), figsize=(15, 5))

    for ax, (name, pattern) in zip(axes, patterns):
        mask = pattern.generate_mask(seq_len)
        sparsity = pattern.get_sparsity(mask)

        ax.imshow(mask.numpy(), cmap='Blues', interpolation='nearest')
        ax.set_xlabel('Key Position')
        ax.set_ylabel('Query Position')
        ax.set_title(f'{name}\nSparsity: {sparsity:.1%}')

    plt.tight_layout()
    plt.savefig('pattern_comparison.png', dpi=150)
    print("Saved comparison to pattern_comparison.png")


if __name__ == "__main__":
    print("Testing sparse attention patterns...\n")

    # Test local pattern
    print("1. Local Attention Pattern")
    local = LocalAttentionPattern(window_size=64)
    mask = local.generate_mask(512)
    print(f"   Sparsity: {local.get_sparsity(mask):.2%}")
    local.visualize(512, save_path='local_pattern.png')

    # Test strided pattern
    print("\n2. Strided Attention Pattern")
    strided = StridedAttentionPattern(stride=64, local_window=16)
    mask = strided.generate_mask(512)
    print(f"   Sparsity: {strided.get_sparsity(mask):.2%}")
    strided.visualize(512, save_path='strided_pattern.png')

    # Test block sparse
    print("\n3. Block Sparse Pattern")
    block = BlockSparsePattern(block_size=64)
    mask = block.generate_mask(512)
    print(f"   Sparsity: {block.get_sparsity(mask):.2%}")
    block.visualize(512, save_path='block_pattern.png')

    # Compare all patterns
    print("\n4. Comparing all patterns...")
    compare_patterns(512)

    print("\nDone!")
