"""Bit-equality test: Triton MXFP8->block-FP8 kernel vs Python reference.

The Triton kernel in ``mxfp8_block_convert_kernel.py`` MUST produce
output bit-identical to the Python reference
``mxfp8_block_convert.convert_mxfp8_weight_to_block_fp8`` (which is the
load-time conversion used by Fp8MoEMethod on Hopper today). KT-EP
layerwise prefill relies on this equivalence: shadow gpu_layer's
weights/scales must be in the same block-FP8 [128, 128] form as the
hybrid path's original layer so the downstream deep_gemm GEMM produces
consistent output.

Run with the M3 routed-expert shapes:
  w13 per expert: N = 2 * intermediate = 1536, K = hidden = 6144
  w2  per expert: N = hidden = 6144,           K = intermediate = 768
"""

import pytest
import torch

from sglang.srt.layers.quantization.mxfp8_block_convert import (
    convert_mxfp8_weight_to_block_fp8,
)
from sglang.srt.layers.quantization.mxfp8_block_convert_kernel import (
    mxfp8_to_block_fp8_per_expert,
)


@pytest.mark.parametrize("N,K", [(1536, 6144), (6144, 768), (256, 256)])
@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_bit_equal_per_expert(N: int, K: int, seed: int) -> None:
    """Triton kernel output must byte-for-byte match Python reference."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    torch.manual_seed(seed)

    # Random fp8 weights and uint8 UE8M0 scales (realistic exponent range).
    w_fp8 = (torch.randn(N, K, device="cuda") * 0.3).to(torch.float8_e4m3fn)
    # UE8M0 ~ exponent biased by 127. Use [120, 134] which corresponds to
    # multipliers 2^-7 .. 2^7, plenty of dynamic range.
    s_u8 = torch.randint(120, 135, (N, K // 32), device="cuda", dtype=torch.uint8)

    # --- Python reference (runs on the same CUDA tensors) ---
    ref_w, ref_s = convert_mxfp8_weight_to_block_fp8(
        w_fp8.clone(), s_u8.clone(), block=128
    )
    assert ref_w.dtype == torch.float8_e4m3fn
    assert ref_s.dtype == torch.float32
    assert ref_w.shape == (N, K)
    assert ref_s.shape == (N // 128, K // 128)

    # --- Triton kernel ---
    out_w = w_fp8.clone().contiguous()
    out_s = torch.empty(N // 128, K // 128, dtype=torch.float32, device="cuda")
    mxfp8_to_block_fp8_per_expert(out_w, s_u8.contiguous(), out_s)

    # Bit-equal check.
    # fp8 e4m3 has no NaN/Inf representation we'd want to distinguish; we just
    # compare the raw byte payload via int8 view (fp8 is 1 byte/element).
    w_diff_max = (
        (out_w.view(torch.int8).to(torch.int32) - ref_w.view(torch.int8).to(torch.int32))
        .abs()
        .max()
        .item()
    )
    assert torch.equal(out_w.view(torch.int8), ref_w.view(torch.int8)), (
        f"weight bytes differ: seed={seed} N={N} K={K} max_byte_diff={w_diff_max}"
    )

    # fp32 scale comparison: bit-exact (no rounding involved on either side,
    # both compute amax / 448.0 in fp32 from the same fp32 intermediates).
    if not torch.equal(out_s, ref_s):
        diff = (out_s - ref_s).abs()
        rel = diff / ref_s.abs().clamp(min=1e-30)
        raise AssertionError(
            f"scale fp32 differs: seed={seed} N={N} K={K} "
            f"max_abs={diff.max().item():.3e} max_rel={rel.max().item():.3e}"
        )


if __name__ == "__main__":
    # Quick smoke without pytest infra.
    for seed in range(5):
        for N, K in [(1536, 6144), (6144, 768), (256, 256)]:
            test_bit_equal_per_expert(N, K, seed)
            print(f"PASS seed={seed} N={N} K={K}")
    print("ALL PASS")
