# SPDX-License-Identifier: Apache-2.0
"""V4-Flash MXFP4 GPU MoE — FP8 MMA 路径 + matmul_ogs fallback.

SM_120 (RTX 5090) 上 tl.dot_scaled 退化为 bf16 MMA；本模块提供自定义
FP8 MMA kernel (tl.dot(fp8,fp8) → mma.m16n8k32.e4m3)，吞吐量 2×。
其余 GPU 走 triton_kernels.matmul_ogs fallback。

路径选择:
  SM_120 + env!=0     → FP8 自定义 kernel（默认）
  env SGLANG_V4_FP8_MMA=0 → 强制 matmul_ogs fallback
  其余 GPU            → matmul_ogs (StridedLayout + simulated MXFP)
"""

from __future__ import annotations

import os
import logging
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Section 1: 环境与路径检测
# ---------------------------------------------------------------------------

_TRTLLM_FP4_CAPS = {(10, 0)}


def use_v4_triton_kernels() -> bool:
    """SGLANG_V4_USE_TRITON_KERNELS=1 强制走本模块（含 SM_100）。"""
    return os.environ.get("SGLANG_V4_USE_TRITON_KERNELS") == "1"


def force_disable_v4_triton_kernels() -> bool:
    """SGLANG_V4_USE_TRITON_KERNELS=0 强制 trtllm（诊断用）。"""
    return os.environ.get("SGLANG_V4_USE_TRITON_KERNELS") == "0"


def _use_strided_layout() -> bool:
    if not torch.cuda.is_available():
        return False
    return torch.cuda.get_device_capability() not in _TRTLLM_FP4_CAPS


def _use_fp8_mma() -> bool:
    """SM_120 默认开启 FP8 MMA；SGLANG_V4_FP8_MMA=0 降级。"""
    if os.environ.get("SGLANG_V4_FP8_MMA") == "0":
        return False
    if not torch.cuda.is_available():
        return False
    return torch.cuda.get_device_capability() == (12, 0)


# ---------------------------------------------------------------------------
# Section 2: Triton 工具函数
# ---------------------------------------------------------------------------

@triton.jit
def _fp4_nibble_to_fp8(nib):
    """单个 FP4 e2m1 nibble (4 bits in uint8) → FP8 e4m3 (uint8).

    e2m1 {0,±0.5,±1,±1.5,±2,±3,±4,±6} 全可被 e4m3 无损表示。
    normal: stored_exp = orig_exp + 6, mantissa 左移 2 bit。
    subnormal(±0.5): 直接映射到 e4m3 的 0110_000。
    zero: 保留符号位。
    """
    sign = (nib & 0x08) << 4        # bit3 → bit7
    exp = (nib >> 1) & 0x03         # 2-bit exponent
    man = nib & 0x01                # 1-bit mantissa
    # exp>0: (exp+6)<<3 | man<<2;  exp==0,man==1: 0x30;  exp==0,man==0: 0
    body = tl.where(exp != 0,
                    ((exp + 6) << 3) | (man << 2),
                    man * 0x30)
    return (sign | body).to(tl.uint8)


@triton.jit
def _unpack_fp4_to_fp8(packed):
    """[K_packed, N] uint8 (2×FP4 per byte) → [2*K_packed, N] float8e4nv.

    低 nibble = even K index, 高 nibble = odd K index.
    用 tl.interleave 沿 K 维交叉拼接。
    """
    fp8_lo = _fp4_nibble_to_fp8(packed & 0x0F)
    fp8_hi = _fp4_nibble_to_fp8((packed >> 4) & 0x0F)
    # interleave 沿最后一维工作，转置后 K 变成最后一维
    w_fp8 = tl.interleave(fp8_lo.trans(), fp8_hi.trans()).trans()
    return w_fp8.to(tl.float8e4nv, bitcast=True)


@triton.jit
def _e8m0_to_bf16(scale_u8):
    """e8m0 (uint8) → bfloat16 乘子: 2^(val-127)."""
    return (scale_u8.to(tl.uint16) << 7).to(tl.bfloat16, bitcast=True)


# ---------------------------------------------------------------------------
# Section 3: Bitmatrix 路由
# ---------------------------------------------------------------------------

@triton.jit
def _pack_bitmatrix_v4(
    bitmatrix_ptr,
    topk_ids_ptr,
    n_rows,
    bm_cols: tl.constexpr,
    n_expts_act,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """topk_ids → bitmatrix: bitmatrix[r, e//32] |= (1 << (e%32))."""
    pid_m = tl.program_id(0)
    rows = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    row_mask = rows < n_rows
    for k in range(n_expts_act):
        ids = tl.load(
            topk_ids_ptr + rows * n_expts_act + k,
            mask=row_mask, other=-1,
        ).to(tl.int32)
        valid = (ids >= 0) & row_mask
        col = ids // BLOCK_SIZE_K
        bit = ids - col * BLOCK_SIZE_K
        ptrs = bitmatrix_ptr + rows * bm_cols + col
        tl.atomic_or(ptrs, (1 << bit).to(tl.uint32), mask=valid)


def _make_routing_data_v4(
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    num_local_experts: int,
):
    """sglang topk → triton_kernels (RoutingData, GatherIndx, ScatterIndx).

    V4 top-6 不是 2 的幂，pad 到 8 以满足 routing kernel 要求。
    """
    from triton_kernels.routing import routing_from_bitmatrix
    from triton_kernels.tensor import Bitmatrix

    topk_ids_i16 = topk_ids.to(torch.int16).contiguous()
    topk_weights_bf = topk_weights.to(torch.bfloat16).contiguous()
    n_rows, n_topk_raw = topk_ids_i16.shape

    n_topk = 1
    while n_topk < n_topk_raw:
        n_topk *= 2
    if n_topk != n_topk_raw:
        pad_len = n_topk - n_topk_raw
        pad_ids = torch.full(
            (n_rows, pad_len), -1, dtype=torch.int16, device=topk_ids_i16.device
        )
        pad_w = torch.full(
            (n_rows, pad_len), -1.0, dtype=torch.bfloat16, device=topk_ids_i16.device,
        )
        topk_ids_i16 = torch.cat([topk_ids_i16, pad_ids], dim=1).contiguous()
        topk_weights_bf = torch.cat([topk_weights_bf, pad_w], dim=1).contiguous()

    BLOCK_SIZE_M = 512
    BLOCK_SIZE_K = 32
    bm_cols = triton.cdiv(num_local_experts, BLOCK_SIZE_K)
    bitmatrix_data = torch.zeros(
        (n_rows, bm_cols), dtype=torch.uint32, device=topk_ids_i16.device,
    )
    grid = (triton.cdiv(n_rows, BLOCK_SIZE_M),)
    _pack_bitmatrix_v4[grid](
        bitmatrix_data, topk_ids_i16, n_rows, bm_cols, n_topk,
        BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    bitmatrix = Bitmatrix(
        bitmatrix_data, shape=[n_rows, bm_cols * 32], shape_max=[n_rows, None],
        scratchpad=None,
    )
    topk_weights_bf = topk_weights_bf.masked_fill(topk_ids_i16 == -1, -1.0)
    return routing_from_bitmatrix(
        bitmatrix, topk_weights_bf, topk_ids_i16, num_local_experts, n_topk
    )


# ---------------------------------------------------------------------------
# Section 4: FP8 MoE GEMM kernel
# ---------------------------------------------------------------------------

@triton.jit
def _v4_fp8_moe_gemm(
    # 输出 (expanded array, 非最终 scatter 输出)
    Y, stride_y_m, stride_y_n,
    # 输入 (bf16)
    X, stride_x_m, stride_x_k,
    # 权重 (packed FP4, uint8, transposed: [E, K//2, N])
    W, stride_w_e, stride_w_k, stride_w_n,
    # 权重 scale (e8m0, uint8, [E, K//32, N])
    WS, stride_ws_e, stride_ws_k, stride_ws_n,
    # 路由
    ExptData,       # [grid_m] int32, packed (block_id:i16 << 16 | expt_id:i16)
    ExptHist,       # [E] int32
    ExptOffs,       # [E] int32 (token_offs_raw)
    GatherIndx,     # [expanded] int32 or None
    Gammas,         # [expanded] bf16 or None (topk weights, applied in-kernel)
    # 形状
    N, K,
    # 常量
    N_EXPTS_ACT: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    MX_GROUP: tl.constexpr,
    HAS_GATHER: tl.constexpr,
):
    """FP8 MMA MoE GEMM: FP4→FP8, bf16→FP8, per-32-group scale 后乘.

    始终写到 expanded array; scatter 归并由 Python 层处理。
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # --- expert 分发 ---
    expt_packed = tl.load(ExptData + pid_m)
    if expt_packed == -1:
        return
    expt_id = (expt_packed & 0x0000FFFF).to(tl.int32)
    block_id = (expt_packed >> 16).to(tl.int32)
    M_expert = tl.load(ExptHist + expt_id)
    start_m = tl.load(ExptOffs + expt_id)

    offs_m = BLOCK_M * block_id + tl.arange(0, BLOCK_M)
    offs_m_safe = offs_m % M_expert
    mask_m = offs_m < M_expert
    offs_n = BLOCK_N * pid_n + tl.arange(0, BLOCK_N)
    mask_n = offs_n < N

    # --- X 行索引 ---
    if HAS_GATHER:
        src_idx = tl.load(GatherIndx + start_m + offs_m_safe, mask=mask_m, other=0)
        x_rows = src_idx // N_EXPTS_ACT
    else:
        x_rows = start_m + offs_m_safe

    # --- K 循环: FP8 MMA + per-group scale ---
    N_GROUPS: tl.constexpr = BLOCK_K // MX_GROUP
    HALF_GROUP: tl.constexpr = MX_GROUP // 2
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    for k_outer in range(0, K, BLOCK_K):
        for g in tl.static_range(N_GROUPS):
            k_off = k_outer + g * MX_GROUP
            k_packed = k_off // 2

            x_ptrs = (X + x_rows[:, None] * stride_x_m
                      + (k_off + tl.arange(0, MX_GROUP))[None, :] * stride_x_k)
            x = tl.load(x_ptrs, mask=mask_m[:, None], other=0.0)
            x_fp8 = x.to(tl.float8e4nv)

            w_ptrs = (W + expt_id * stride_w_e
                      + (k_packed + tl.arange(0, HALF_GROUP))[:, None] * stride_w_k
                      + offs_n[None, :] * stride_w_n)
            w_packed = tl.load(w_ptrs, mask=mask_n[None, :], other=0)
            w_fp8 = _unpack_fp4_to_fp8(w_packed)

            partial = tl.dot(x_fp8, w_fp8)

            s_ptrs = (WS + expt_id * stride_ws_e
                      + (k_off // MX_GROUP) * stride_ws_k
                      + offs_n * stride_ws_n)
            scale = tl.load(s_ptrs, mask=mask_n, other=127)
            acc += partial * _e8m0_to_bf16(scale)[None, :].to(tl.float32)

    # --- 输出 ---
    out = acc.to(tl.bfloat16)
    if Gammas is not None:
        gammas = tl.load(Gammas + start_m + offs_m_safe, mask=mask_m, other=0.0)
        out = out * gammas[:, None]

    y_ptrs = (Y + (start_m + offs_m)[:, None] * stride_y_m
              + offs_n[None, :] * stride_y_n)
    tl.store(y_ptrs, out, mask=mask_m[:, None] & mask_n[None, :])


def _launch_fp8_gemm(
    x: torch.Tensor,
    w_data: torch.Tensor,
    w_scale_data: torch.Tensor,
    routing_data,
    N: int,
    K: int,
    gather_indx=None,
    gammas=None,
) -> torch.Tensor:
    """调度 _v4_fp8_moe_gemm, 写到 expanded output array."""
    expt_data = routing_data.expt_data
    has_gather = gather_indx is not None

    BLOCK_M = 64
    BLOCK_N = 128
    BLOCK_K = 128
    MX_GROUP = 32

    block_pid_map = expt_data.block_pid_map[BLOCK_M]
    grid_m = block_pid_map.shape[0]
    grid_n = triton.cdiv(N, BLOCK_N)

    M_expanded = gather_indx.src_indx.shape[0] if has_gather else x.shape[0]
    y = torch.empty(M_expanded, N, dtype=x.dtype, device=x.device)

    _v4_fp8_moe_gemm[(grid_m, grid_n)](
        y, y.stride(0), y.stride(1),
        x, x.stride(0), x.stride(1),
        w_data, w_data.stride(0), w_data.stride(1), w_data.stride(2),
        w_scale_data, w_scale_data.stride(0), w_scale_data.stride(1), w_scale_data.stride(2),
        block_pid_map,
        expt_data.hist,
        expt_data.token_offs_raw,
        gather_indx.src_indx if has_gather else None,
        gammas,
        N, K,
        N_EXPTS_ACT=routing_data.n_expts_act,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        MX_GROUP=MX_GROUP,
        HAS_GATHER=has_gather,
    )
    return y


def _scatter_reduce(
    expanded: torch.Tensor,
    scatter_indx,
    n_expts_act: int,
    M_out: int,
) -> torch.Tensor:
    """expanded [M_expanded, K] → output [M, K]: 按 src_indx 分组求和.

    gammas 已在 kernel 内乘好，这里只做归并。
    """
    K = expanded.shape[1]
    group_indx = scatter_indx.src_indx.view(-1, n_expts_act)  # [M, n_act]
    flat = group_indx.flatten()                                 # [M * n_act]
    valid = flat >= 0
    safe_idx = flat.clamp(min=0)
    gathered = expanded[safe_idx]                               # [M*n_act, K]
    gathered[~valid] = 0.0
    return gathered.view(M_out, n_expts_act, K).sum(dim=1)      # [M, K]


# ---------------------------------------------------------------------------
# Section 5: matmul_ogs fallback (非 FP8 GPU)
# ---------------------------------------------------------------------------

def _patch_strided_mxfp():
    """matmul_ogs 需要的 monkey-patch: has_native_mxfp→False + is_persistent→False."""
    if not _use_strided_layout():
        return
    import triton_kernels.target_info as target_info
    if not getattr(target_info, "_v4_strided_patched", False):
        original = target_info.has_native_mxfp
        def has_native_mxfp_strided():
            return False if _use_strided_layout() else original()
        target_info.has_native_mxfp = has_native_mxfp_strided
        target_info._v4_strided_patched = True
    try:
        import triton_kernels.matmul_ogs_details.opt_flags as _of
        if hasattr(_of, "has_native_mxfp"):
            _of.has_native_mxfp = target_info.has_native_mxfp
        _of.update_opt_flags_constraints({"is_persistent": False})
        if (
            hasattr(_of, "make_default_opt_flags_nvidia")
            and not getattr(_of, "_v4_assert_patched", False)
        ):
            import inspect as _inspect, textwrap as _textwrap
            _src = _textwrap.dedent(
                _inspect.getsource(_of.make_default_opt_flags_nvidia)
            )
            _patched_src = _src.replace(
                "assert num_stages >= 1",
                "num_stages = max(num_stages, 1)  # v4-flash patch",
            )
            if _patched_src != _src:
                _patched_src = _patched_src.replace(
                    "def make_default_opt_flags_nvidia",
                    "def _v4_make_default_opt_flags_nvidia", 1,
                )
                exec(_patched_src, _of.__dict__)
                _of.make_default_opt_flags_nvidia = _of._v4_make_default_opt_flags_nvidia
                _of._v4_assert_patched = True
    except Exception:
        pass


def _swizzle_mxfp4_strided(quant_tensor: torch.Tensor, scale: torch.Tensor):
    """StridedLayout 包装: 转置 + wrap 为 triton_kernels.Tensor (无硬件 swizzle)."""
    from triton_kernels.numerics import InFlexData
    from triton_kernels.tensor import FP4, convert_layout, wrap_torch_tensor
    from triton_kernels.tensor_details.layout import StridedLayout

    quant_tensor = convert_layout(
        wrap_torch_tensor(quant_tensor.transpose(-2, -1).contiguous(), dtype=FP4), StridedLayout
    )
    scale = convert_layout(wrap_torch_tensor(scale.transpose(-2, -1).contiguous()), StridedLayout)
    return quant_tensor, InFlexData(), scale


def convert_v4_weights_to_triton_kernels(
    w13: torch.Tensor,
    w13_scale: torch.Tensor,
    w2: torch.Tensor,
    w2_scale: torch.Tensor,
    *,
    num_warps: int = 4,
) -> Tuple:
    """转换 checkpoint FP4 权重 → matmul_ogs 格式 (+ FP8 路径复用同一 storage)."""
    from triton_kernels.matmul_ogs import FlexCtx, PrecisionConfig

    _patch_strided_mxfp()

    # dtype 规范化
    if w13_scale.dtype != torch.float8_e8m0fnu:
        w13_scale = (w13_scale.view(torch.float8_e8m0fnu)
                     if w13_scale.dtype == torch.uint8
                     else w13_scale.to(torch.float8_e8m0fnu))
    if w2_scale.dtype != torch.float8_e8m0fnu:
        w2_scale = (w2_scale.view(torch.float8_e8m0fnu)
                    if w2_scale.dtype == torch.uint8
                    else w2_scale.to(torch.float8_e8m0fnu))
    if w13.dtype != torch.uint8:
        w13 = w13.view(torch.uint8)
    if w2.dtype != torch.uint8:
        w2 = w2.view(torch.uint8)

    if _use_strided_layout():
        w13_swiz, w13_flex, w13_scale_swiz = _swizzle_mxfp4_strided(w13, w13_scale)
        w2_swiz, w2_flex, w2_scale_swiz = _swizzle_mxfp4_strided(w2, w2_scale)
    else:
        from sglang.srt.layers.quantization.mxfp4 import _swizzle_mxfp4
        w13_swiz, w13_flex, w13_scale_swiz = _swizzle_mxfp4(w13, w13_scale, num_warps)
        w2_swiz, w2_flex, w2_scale_swiz = _swizzle_mxfp4(w2, w2_scale, num_warps)

    w13_pcg = PrecisionConfig(
        weight_scale=w13_scale_swiz, flex_ctx=FlexCtx(rhs_data=w13_flex),
    )
    w2_pcg = PrecisionConfig(
        weight_scale=w2_scale_swiz, flex_ctx=FlexCtx(rhs_data=w2_flex),
    )
    return w13_swiz, w13_pcg, w2_swiz, w2_pcg


# ---------------------------------------------------------------------------
# Section 6: MoE forward
# ---------------------------------------------------------------------------

def _extract_raw_tensor(tk_tensor) -> torch.Tensor:
    """从 triton_kernels.Tensor 提取底层 torch.Tensor."""
    return tk_tensor.storage.data


def apply_v4_triton_kernels_moe(
    *,
    hidden_states: torch.Tensor,
    w13_swiz,
    w13_pcg,
    w2_swiz,
    w2_pcg,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    intermediate_size: int,
    num_experts: int,
    routed_scaling_factor: float = 1.0,
    swiglu_limit: Optional[float] = None,
) -> torch.Tensor:
    """V4 sparse MoE forward: FP8 kernel (SM_120) 或 matmul_ogs fallback."""
    from sgl_kernel import silu_and_mul

    _patch_strided_mxfp()

    M, K = hidden_states.shape
    N = intermediate_size

    routing_data, gather_indx, scatter_indx = _make_routing_data_v4(
        topk_ids, topk_weights, num_experts
    )

    fp8 = _use_fp8_mma()

    if fp8:
        w13_data = _extract_raw_tensor(w13_swiz)
        w13_scale_data = _extract_raw_tensor(w13_pcg.weight_scale).view(torch.uint8)
        w2_data = _extract_raw_tensor(w2_swiz)
        w2_scale_data = _extract_raw_tensor(w2_pcg.weight_scale).view(torch.uint8)

        intermediate1 = _launch_fp8_gemm(
            hidden_states, w13_data, w13_scale_data,
            routing_data, 2 * N, K,
            gather_indx=gather_indx,
        )
    else:
        from triton_kernels.matmul_ogs import matmul_ogs

        intermediate1 = matmul_ogs(
            hidden_states, w13_swiz, None,
            routing_data, gather_indx=gather_indx,
            precision_config=w13_pcg,
        )

    # SwiGLU clamp (2604B)
    if swiglu_limit is not None:
        N_int = intermediate1.shape[-1] // 2
        intermediate1[..., :N_int].clamp_(max=swiglu_limit)
        intermediate1[..., N_int:].clamp_(min=-swiglu_limit, max=swiglu_limit)

    M_topk = intermediate1.shape[0]
    intermediate2 = torch.empty(
        (M_topk, N), device=hidden_states.device, dtype=hidden_states.dtype
    )
    silu_and_mul(intermediate1.view(-1, 2 * N), intermediate2)

    if fp8:
        # gemm2 写到 expanded array (gammas 在 kernel 内乘)，再 scatter 归并
        expanded = _launch_fp8_gemm(
            intermediate2, w2_data, w2_scale_data,
            routing_data, K, N,
            gammas=routing_data.gate_scal,
        )
        output = _scatter_reduce(
            expanded, scatter_indx, routing_data.n_expts_act, M,
        )
    else:
        output = matmul_ogs(
            intermediate2, w2_swiz, None,
            routing_data, scatter_indx=scatter_indx,
            precision_config=w2_pcg,
            gammas=routing_data.gate_scal,
        )

    return output
