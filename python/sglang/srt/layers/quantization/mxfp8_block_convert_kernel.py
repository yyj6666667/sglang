"""Triton GPU kernel: MXFP8 (e4m3fn + 1x32 UE8M0) -> block-FP8 [128,128] (e4m3fn + fp32).

This is the GPU-resident equivalent of the Python reference
``mxfp8_block_convert.convert_mxfp8_weight_to_block_fp8`` =
``dequant_mxfp8_2d_to_bf16`` ∘ ``bf16_to_block_fp8_128``. The output must be
**bit-identical** to that reference, including the lossy intermediate
bf16 round-trip (dequant produces bf16 before re-quantizing).

Used by KT-EP layerwise prefill (kt_ep_wrapper.py:_prepare_weight_mxfp8):
after kt-kernel C++ writes raw MXFP8 bytes into the shadow gpu_layer, this
kernel converts the per-expert weights+scales into the block-FP8 layout
that ``Fp8MoEMethod.apply`` / deep_gemm expect on Hopper.

Layout (per expert):
  in  w_fp8      : [N, K]        float8_e4m3fn (MXFP8 quantized weight)
  in  s_u8       : [N, K // 32]  uint8         (MXFP8 UE8M0 scale)
  out w_fp8      : [N, K]        float8_e4m3fn (re-quantized in-place)
  out s_fp32     : [N // 128, K // 128] float32 (block scale, sf = amax / 448)

Constraints:
  * N % 128 == 0, K % 128 == 0  (M3 satisfies; padding branch dropped)
  * K % 32  == 0                (MXFP8 group size)
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _mxfp8_to_block_fp8_kernel(
    w_fp8_ptr,        # [N, K]              float8_e4m3fn (in-place modified)
    s_u8_ptr,         # [N, K // GROUP]     uint8 UE8M0
    s_fp32_out_ptr,   # [N // BLOCK, K // BLOCK] float32
    N,
    K,
    stride_w_n,
    stride_w_k,
    stride_s_n,
    stride_s_g,
    stride_o_n,
    stride_o_k,
    BLOCK: tl.constexpr,   # 128
    GROUP: tl.constexpr,   # 32
):
    """One program == one [BLOCK, BLOCK] output block.

    Bit-equal target: dequant_mxfp8_2d_to_bf16 + bf16_to_block_fp8_128.
    The reference round-trips through bf16 between dequant and block quant,
    so we replicate that lossy cast inside the kernel.
    """
    pid_n = tl.program_id(axis=0)
    pid_k = tl.program_id(axis=1)

    # Row / column indices for this output block.
    offs_n = pid_n * BLOCK + tl.arange(0, BLOCK)            # [BLOCK]
    offs_k = pid_k * BLOCK + tl.arange(0, BLOCK)            # [BLOCK]

    # Load [BLOCK, BLOCK] fp8 weights.
    w_ptrs = w_fp8_ptr + offs_n[:, None] * stride_w_n + offs_k[None, :] * stride_w_k
    w_fp8 = tl.load(w_ptrs)                                 # float8_e4m3fn
    w_fp32 = w_fp8.to(tl.float32)                           # [BLOCK, BLOCK]

    # Load [BLOCK, BLOCK // GROUP] uint8 UE8M0 scales (per-row K-grouped).
    GROUPS_PER_BLOCK: tl.constexpr = BLOCK // GROUP         # = 4
    offs_g = pid_k * GROUPS_PER_BLOCK + tl.arange(0, GROUPS_PER_BLOCK)  # [4]
    s_ptrs = s_u8_ptr + offs_n[:, None] * stride_s_n + offs_g[None, :] * stride_s_g
    s_u8 = tl.load(s_ptrs)                                  # [BLOCK, 4] uint8

    # UE8M0 (biased exponent, bias 127) -> fp32 = 2^(v - 127).
    # Mirrors `_ue8m0_to_fp32`: (s.to(int32) << 23).view(float32).
    descale = (s_u8.to(tl.int32) << 23).to(tl.float32, bitcast=True)  # [BLOCK, 4]

    # Broadcast descale across each 32-element K group: reshape (BLOCK, GROUPS)
    # to (BLOCK, GROUPS, 1) then expand to (BLOCK, GROUPS, GROUP) then flatten
    # back to (BLOCK, BLOCK). We do this with explicit reshape + multiply.
    w_fp32_grouped = tl.reshape(w_fp32, (BLOCK, GROUPS_PER_BLOCK, GROUP))
    deq = w_fp32_grouped * descale[:, :, None]              # [BLOCK, GROUPS, GROUP]
    deq = tl.reshape(deq, (BLOCK, BLOCK))                   # [BLOCK, BLOCK] fp32

    # Round-trip through bf16 to match the Python reference's lossy cast.
    deq_bf16 = deq.to(tl.bfloat16)
    deq_fp32 = deq_bf16.to(tl.float32)

    # amax over the full BLOCK*BLOCK tile, clamp(min=1e-4), sf = amax / 448.
    amax = tl.max(tl.abs(deq_fp32))                         # scalar fp32
    amax = tl.maximum(amax, 1e-4)
    sf = amax / 448.0                                       # scalar fp32

    # Re-quantize: xq = (deq_fp32 / sf).to(float8_e4m3fn).
    # Use IEEE round-to-nearest division (tl.math.div_rn) instead of the
    # default `a / b` which Triton may lower to reciprocal-based fast
    # division (~1 ULP off). Reference PyTorch division is IEEE-conformant,
    # so the fp32 result going into the fp8 cast must match bit-for-bit.
    xq = tl.math.div_rn(deq_fp32, sf).to(tl.float8e4nv)     # [BLOCK, BLOCK]

    # Write fp8 back in place.
    tl.store(w_ptrs, xq)

    # Write the scalar block scale.
    out_ptr = s_fp32_out_ptr + pid_n * stride_o_n + pid_k * stride_o_k
    tl.store(out_ptr, sf)


def mxfp8_to_block_fp8_per_expert(
    w_fp8: torch.Tensor,
    s_u8: torch.Tensor,
    s_fp32_out: torch.Tensor,
) -> None:
    """Per-expert MXFP8 -> block-FP8 [128, 128] convert (in-place weight).

    Args:
        w_fp8:      [N, K] float8_e4m3fn — MXFP8-quantized weight, modified
                    in place to hold the block-FP8 weight.
        s_u8:       [N, K // 32] uint8 — MXFP8 UE8M0 scale (read only).
        s_fp32_out: [N // 128, K // 128] float32 — block scale output slot.

    Constraints:
        - N % 128 == 0, K % 128 == 0 (M3 satisfies all MoE shapes)
        - K % 32 == 0
        - All tensors on the same CUDA device.
    """
    assert w_fp8.is_cuda and s_u8.is_cuda and s_fp32_out.is_cuda
    assert w_fp8.dtype == torch.float8_e4m3fn, f"got {w_fp8.dtype}"
    assert s_u8.dtype == torch.uint8, f"got {s_u8.dtype}"
    assert s_fp32_out.dtype == torch.float32, f"got {s_fp32_out.dtype}"

    N, K = w_fp8.shape
    assert N % 128 == 0, f"N={N} must be divisible by 128"
    assert K % 128 == 0, f"K={K} must be divisible by 128"
    assert K % 32 == 0, f"K={K} must be divisible by 32"
    assert s_u8.shape == (N, K // 32), (
        f"s_u8.shape={tuple(s_u8.shape)} != ({N}, {K // 32})"
    )
    assert s_fp32_out.shape == (N // 128, K // 128), (
        f"s_fp32_out.shape={tuple(s_fp32_out.shape)} != ({N // 128}, {K // 128})"
    )

    grid = (N // 128, K // 128)
    _mxfp8_to_block_fp8_kernel[grid](
        w_fp8,
        s_u8,
        s_fp32_out,
        N,
        K,
        w_fp8.stride(0),
        w_fp8.stride(1),
        s_u8.stride(0),
        s_u8.stride(1),
        s_fp32_out.stride(0),
        s_fp32_out.stride(1),
        BLOCK=128,
        GROUP=32,
    )


def mxfp8_to_block_fp8_batched(
    w_fp8: torch.Tensor,
    s_u8: torch.Tensor,
    s_fp32_out: torch.Tensor,
) -> None:
    """Apply :func:`mxfp8_to_block_fp8_per_expert` to every expert.

    Args:
        w_fp8:      [E, N, K] float8_e4m3fn — modified in place.
        s_u8:       [E, N, K // 32] uint8.
        s_fp32_out: [E, N // 128, K // 128] float32.

    v1 simple loop. v2 can fuse the expert dim into the grid (one kernel
    launch). The loop overhead is ~10 µs per expert; for M3 E=128 the
    fixed-cost is ~1.3 ms per call, negligible against the cudaMemcpy and
    the kernel work itself.
    """
    E = w_fp8.shape[0]
    assert s_u8.shape[0] == E and s_fp32_out.shape[0] == E
    for e in range(E):
        mxfp8_to_block_fp8_per_expert(w_fp8[e], s_u8[e], s_fp32_out[e])
