"""
Solutions to All Triton Puzzles

SPOILER ALERT: Only look at this after attempting the puzzles!
"""

import torch
import triton
import triton.language as tl


# ============================================================================
# PUZZLE 01: Vector Addition
# ============================================================================

@triton.jit
def add_kernel_solution(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)  # BLANK 1
    block_start = pid * BLOCK_SIZE  # BLANK 2
    offsets = block_start + tl.arange(0, BLOCK_SIZE)  # BLANK 3
    mask = offsets < n_elements  # BLANK 4
    x = tl.load(x_ptr + offsets, mask=mask)  # BLANK 5
    y = tl.load(y_ptr + offsets, mask=mask)  # BLANK 6
    output = x + y  # BLANK 7
    tl.store(output_ptr + offsets, output, mask=mask)  # BLANK 8


# ============================================================================
# PUZZLE 02: Broadcasting
# ============================================================================

@triton.jit
def broadcast_add_kernel_solution(
    x_ptr, y_ptr, output_ptr,
    M, N,
    stride_xm, stride_xn,
    stride_om, stride_on,
    BLOCK_SIZE: tl.constexpr
):
    """Add scalar y to each element of matrix x."""
    pid = tl.program_id(0)

    # Calculate row and column indices
    row = pid // triton.cdiv(N, BLOCK_SIZE)
    col_block = pid % triton.cdiv(N, BLOCK_SIZE)

    col_start = col_block * BLOCK_SIZE
    col_offsets = col_start + tl.arange(0, BLOCK_SIZE)

    mask = (row < M) & (col_offsets < N)

    # Load matrix element
    x = tl.load(x_ptr + row * stride_xm + col_offsets * stride_xn, mask=mask)

    # Load scalar (broadcast)
    y = tl.load(y_ptr)

    # Add
    output = x + y

    # Store
    tl.store(output_ptr + row * stride_om + col_offsets * stride_on, output, mask=mask)


# ============================================================================
# PUZZLE 03: Matrix Multiplication
# ============================================================================

@triton.jit
def matmul_kernel_solution(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_SIZE_K):
        offs_k = k + tl.arange(0, BLOCK_SIZE_K)

        a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
        b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

        mask_a = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        mask_b = (offs_k[:, None] < K) & (offs_n[None, :] < N)

        a = tl.load(a_ptrs, mask=mask_a, other=0.0)
        b = tl.load(b_ptrs, mask=mask_b, other=0.0)

        accumulator += tl.dot(a, b)

    c = accumulator.to(tl.float16)

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + (offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn)

    mask_c = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=mask_c)


# ============================================================================
# PUZZLE 04: Softmax
# ============================================================================

@triton.jit
def softmax_kernel_solution(
    output_ptr, input_ptr,
    input_row_stride, output_row_stride,
    n_cols, BLOCK_SIZE: tl.constexpr
):
    row_idx = tl.program_id(0)

    row_start_ptr = input_ptr + row_idx * input_row_stride

    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    row = tl.load(row_start_ptr + col_offsets, mask=mask, other=-float('inf'))

    row_max = tl.max(row, axis=0)
    row_shifted = row - row_max
    numerator = tl.exp(row_shifted)
    denominator = tl.sum(numerator, axis=0)
    softmax_output = numerator / denominator

    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    tl.store(output_row_start_ptr + col_offsets, softmax_output, mask=mask)


# ============================================================================
# Test Functions
# ============================================================================

def test_all_solutions():
    """Test all puzzle solutions."""
    print("Testing Puzzle Solutions")
    print("=" * 70)

    # Test Puzzle 01
    n = 10000
    x = torch.randn(n, device='cuda')
    y = torch.randn(n, device='cuda')
    output = torch.empty_like(x)

    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    add_kernel_solution[grid](x, y, output, n, BLOCK_SIZE)

    assert torch.allclose(output, x + y)
    print("✓ Puzzle 01: Vector Addition")

    # Test Puzzle 03
    M, N, K = 128, 128, 128
    a = torch.randn(M, K, device='cuda', dtype=torch.float16)
    b = torch.randn(K, N, device='cuda', dtype=torch.float16)
    c = torch.empty(M, N, device='cuda', dtype=torch.float16)

    BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    matmul_kernel_solution[grid](
        a, b, c, M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M, BLOCK_N, BLOCK_K
    )

    assert torch.allclose(c, torch.matmul(a, b), rtol=1e-2, atol=1e-2)
    print("✓ Puzzle 03: Matrix Multiplication")

    # Test Puzzle 04
    x = torch.randn(128, 256, device='cuda')
    output = torch.empty_like(x)

    BLOCK_SIZE = triton.next_power_of_2(256)
    grid = (128,)

    softmax_kernel_solution[grid](
        output, x,
        x.stride(0), output.stride(0),
        256, BLOCK_SIZE
    )

    expected = torch.softmax(x, dim=-1)
    assert torch.allclose(output, expected, rtol=1e-5, atol=1e-6)
    print("✓ Puzzle 04: Softmax")

    print("\nAll puzzles solved correctly!")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available. These solutions require a GPU.")
        exit(1)

    test_all_solutions()
