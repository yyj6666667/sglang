"""Fused grouped MXFP8 block-scaled GEMM kernel for MoE on Hopper (SM90).

Replaces per-expert Python loop with a single Triton kernel launch.
Uses manual FP8 dequant + tl.dot (no tl.dot_scaled) to avoid
packed scale layout complexity.

Layout:
  a:       [num_experts, max_m, K]  float8_e4m3fn
  a_scale: [num_experts, max_m, K//32]  uint8 (UE8M0)
  b:       [num_experts, N, K]  float8_e4m3fn
  b_scale: [num_experts, N, K//32]  uint8 (UE8M0)
  c:       [num_experts, max_m, N]  bfloat16
  masked_m: [num_experts]  int32
"""

from typing import Optional

import torch
import triton
import triton.language as tl


@triton.jit
def _grouped_mxfp8_gemm_kernel(
    a_ptr,
    a_scale_ptr,
    b_ptr,
    b_scale_ptr,
    c_ptr,
    masked_m_ptr,
    stride_a_e: tl.constexpr,
    stride_a_m: tl.constexpr,
    stride_a_k: tl.constexpr,
    stride_as_e: tl.constexpr,
    stride_as_m: tl.constexpr,
    stride_as_k: tl.constexpr,
    stride_b_e: tl.constexpr,
    stride_b_n: tl.constexpr,
    stride_b_k: tl.constexpr,
    stride_bs_e: tl.constexpr,
    stride_bs_n: tl.constexpr,
    stride_bs_k: tl.constexpr,
    stride_c_e: tl.constexpr,
    stride_c_m: tl.constexpr,
    stride_c_n: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    expert_id = tl.program_id(axis=2)
    actual_m = tl.load(masked_m_ptr + expert_id)

    pid = tl.program_id(axis=0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    start_m = pid_m * BLOCK_M
    if start_m >= actual_m:
        return

    start_n = pid_n * BLOCK_N

    offs_m = start_m + tl.arange(0, BLOCK_M)
    offs_n = start_n + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_base = a_ptr + expert_id * stride_a_e
    b_base = b_ptr + expert_id * stride_b_e
    as_base = a_scale_ptr + expert_id * stride_as_e
    bs_base = b_scale_ptr + expert_id * stride_bs_e

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    num_k_blocks = tl.cdiv(K, BLOCK_K)
    for k_idx in range(num_k_blocks):
        k_start = k_idx * BLOCK_K

        # Load A block: [BLOCK_M, BLOCK_K] fp8
        a_ptrs = a_base + offs_m[:, None] * stride_a_m + (k_start + offs_k[None, :]) * stride_a_k
        mask_a = (offs_m[:, None] < actual_m) & ((k_start + offs_k[None, :]) < K)
        a_block = tl.load(a_ptrs, mask=mask_a, other=0.0)

        # Load B block: [BLOCK_N, BLOCK_K] fp8 (weight is [N, K])
        b_ptrs = b_base + offs_n[:, None] * stride_b_n + (k_start + offs_k[None, :]) * stride_b_k
        mask_b = (offs_n[:, None] < N) & ((k_start + offs_k[None, :]) < K)
        b_block = tl.load(b_ptrs, mask=mask_b, other=0.0)

        # Load A scales: [BLOCK_M, GROUPS_PER_BLOCK] uint8
        GROUPS_PER_BLOCK_LOAD: tl.constexpr = BLOCK_K // GROUP_SIZE
        offs_sg = tl.arange(0, GROUPS_PER_BLOCK_LOAD)
        k_scale_start = k_idx * GROUPS_PER_BLOCK_LOAD
        K_GROUPS: tl.constexpr = K // GROUP_SIZE
        as_ptrs = as_base + offs_m[:, None] * stride_as_m + (k_scale_start + offs_sg[None, :]) * stride_as_k
        mask_as = (offs_m[:, None] < actual_m) & ((k_scale_start + offs_sg[None, :]) < K_GROUPS)
        a_scale_block = tl.load(as_ptrs, mask=mask_as, other=127).to(tl.uint8)

        # Load B scales: [BLOCK_N, GROUPS_PER_BLOCK] uint8
        bs_ptrs = bs_base + offs_n[:, None] * stride_bs_n + (k_scale_start + offs_sg[None, :]) * stride_bs_k
        mask_bs = (offs_n[:, None] < N) & ((k_scale_start + offs_sg[None, :]) < K_GROUPS)
        b_scale_block = tl.load(bs_ptrs, mask=mask_bs, other=127).to(tl.uint8)

        # Dequant FP8 to float32
        a_f32 = a_block.to(tl.float32)
        b_f32 = b_block.to(tl.float32)

        # Apply per-group UE8M0 scale: reshape to [M, num_groups, group_size],
        # multiply by scale, reshape back
        GROUPS_PER_BLOCK: tl.constexpr = BLOCK_K // GROUP_SIZE
        a_f32 = tl.reshape(a_f32, (BLOCK_M, GROUPS_PER_BLOCK, GROUP_SIZE))
        b_f32 = tl.reshape(b_f32, (BLOCK_N, GROUPS_PER_BLOCK, GROUP_SIZE))

        # UE8M0 uint8 -> float32: 2^(val - 127)
        a_scale_f32 = (a_scale_block.to(tl.int32) << 23).to(tl.float32, bitcast=True)
        b_scale_f32 = (b_scale_block.to(tl.int32) << 23).to(tl.float32, bitcast=True)

        # Scale: [M, groups] -> [M, groups, 1] broadcast multiply
        a_f32 = a_f32 * a_scale_f32[:, :, None]
        b_f32 = b_f32 * b_scale_f32[:, :, None]

        a_f32 = tl.reshape(a_f32, (BLOCK_M, BLOCK_K))
        b_f32 = tl.reshape(b_f32, (BLOCK_N, BLOCK_K))

        # Matmul: [BLOCK_M, BLOCK_K] x [BLOCK_K, BLOCK_N]
        accumulator += tl.dot(a_f32.to(tl.bfloat16), tl.trans(b_f32).to(tl.bfloat16))

    # Store output
    c_base = c_ptr + expert_id * stride_c_e
    c_ptrs = c_base + offs_m[:, None] * stride_c_m + offs_n[None, :] * stride_c_n
    mask_c = (offs_m[:, None] < actual_m) & (offs_n[None, :] < N)
    tl.store(c_ptrs, accumulator.to(tl.bfloat16), mask=mask_c)


def grouped_mxfp8_block_scaled_matmul(
    a: torch.Tensor,
    a_scale: torch.Tensor,
    b: torch.Tensor,
    b_scale: torch.Tensor,
    output: torch.Tensor,
    masked_m: torch.Tensor,
    group_size: int = 32,
    block_m: int = 64,
    block_n: int = 64,
    block_k: int = 64,
) -> None:
    from sglang.srt.debug_utils.m3_hidden_dump import enabled as _dump_on
    from sglang.srt.debug_utils.m3_hidden_dump import record as _dump

    if _dump_on():
        _dump("grouped_mxfp8.in.a", a)
        _dump("grouped_mxfp8.in.a_scale", a_scale)
        _dump("grouped_mxfp8.in.b", b)
        _dump("grouped_mxfp8.in.b_scale", b_scale)
        _dump("grouped_mxfp8.in.masked_m", masked_m)
    """Fused grouped MXFP8 GEMM for MoE.

    Args:
        a: [num_experts, max_m, K] float8_e4m3fn
        a_scale: [num_experts, max_m, K//group_size] uint8 (UE8M0)
        b: [num_experts, N, K] float8_e4m3fn
        b_scale: [num_experts, N, K//group_size] uint8 (UE8M0)
        output: [num_experts, max_m, N] bfloat16
        masked_m: [num_experts] int32
        group_size: MXFP8 block size (default 32)
    """
    num_experts, max_m, K = a.shape
    N = b.shape[1]

    assert a.shape == (num_experts, max_m, K)
    assert b.shape == (num_experts, N, K)
    assert output.shape == (num_experts, max_m, N)
    assert a_scale.shape[0] == num_experts
    assert b_scale.shape[0] == num_experts
    assert K % block_k == 0
    assert block_k % group_size == 0

    # Ensure scales are uint8
    if a_scale.dtype == torch.float32:
        a_scale = (a_scale.view(torch.int32) >> 23).to(torch.uint8)
    if b_scale.dtype == torch.float32:
        b_scale = (b_scale.view(torch.int32) >> 23).to(torch.uint8)

    a = a.contiguous()
    b = b.contiguous()
    a_scale = a_scale.contiguous()
    b_scale = b_scale.contiguous()

    grid = (
        triton.cdiv(max_m, block_m) * triton.cdiv(N, block_n),
        1,
        num_experts,
    )

    _grouped_mxfp8_gemm_kernel[grid](
        a,
        a_scale,
        b,
        b_scale,
        output,
        masked_m,
        a.stride(0), a.stride(1), a.stride(2),
        a_scale.stride(0), a_scale.stride(1), a_scale.stride(2),
        b.stride(0), b.stride(1), b.stride(2),
        b_scale.stride(0), b_scale.stride(1), b_scale.stride(2),
        output.stride(0), output.stride(1), output.stride(2),
        N=N,
        K=K,
        GROUP_SIZE=group_size,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
    )

    if _dump_on():
        _dump("grouped_mxfp8.out.output", output)
