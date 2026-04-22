#!/usr/bin/env python3
"""Benchmark sgl_kernel.apply_rope_with_cos_sin_cache_inplace vs a torch reference."""

import argparse
from typing import List

import numpy as np
import torch

import sgl_kernel
from flashinfer.testing.utils import bench_gpu_time


def _torch_rope_ref(
    positions: torch.Tensor,  # [num_tokens], int64
    query: torch.Tensor,  # [num_tokens, num_q_heads * head_size]
    key: torch.Tensor,  # [num_tokens, num_k_heads * head_size]
    head_size: int,
    cos_sin_cache: torch.Tensor,  # [max_seq, head_size]
    is_neox: bool,
):
    # Reference cribbed from vLLM's native torch RoPE (not optimised).
    cos_sin = cos_sin_cache[positions]  # [T, head_size]
    cos, sin = cos_sin.chunk(2, dim=-1)  # [T, head_size/2] each

    def _apply(t):
        # t: [T, num_heads * head_size] -> [T, num_heads, head_size]
        shape = t.shape
        t = t.view(shape[0], -1, head_size)
        if is_neox:
            x1, x2 = t[..., : head_size // 2], t[..., head_size // 2 :]
            # Broadcast cos/sin over heads.
            c = cos.unsqueeze(1)
            s = sin.unsqueeze(1)
            o1 = x1 * c - x2 * s
            o2 = x2 * c + x1 * s
            out = torch.cat([o1, o2], dim=-1)
        else:  # GPT-J: interleaved
            x1 = t[..., ::2]
            x2 = t[..., 1::2]
            c = cos.unsqueeze(1)
            s = sin.unsqueeze(1)
            o1 = x1 * c - x2 * s
            o2 = x2 * c + x1 * s
            out = torch.stack([o1, o2], dim=-1).flatten(-2)
        return out.view(shape)

    return _apply(query), _apply(key)


@torch.inference_mode()
def bench_one(num_tokens: int, num_q_heads: int, num_kv_heads: int, head_size: int,
              max_pos: int, dtype: torch.dtype, is_neox: bool, num_iters: int):
    positions = torch.randint(0, max_pos, (num_tokens,), device="cuda", dtype=torch.int64)

    # Build a cos/sin cache: [max_pos, head_size] where first half is cos, second half sin.
    theta = torch.linspace(0, 1, head_size // 2, device="cuda", dtype=torch.float32).unsqueeze(0)
    pos_range = torch.arange(max_pos, device="cuda", dtype=torch.float32).unsqueeze(1)
    freqs = pos_range * theta
    # sgl_kernel requires cos_sin_cache in float32 regardless of q/k dtype.
    cos_sin_cache = torch.cat([freqs.cos(), freqs.sin()], dim=-1)

    q = torch.randn((num_tokens, num_q_heads * head_size), dtype=dtype, device="cuda")
    k = torch.randn((num_tokens, num_kv_heads * head_size), dtype=dtype, device="cuda")

    q_t, k_t = q.clone(), k.clone()
    q_s, k_s = q.clone(), k.clone()

    q_ref, k_ref = _torch_rope_ref(positions, q_t, k_t, head_size, cos_sin_cache, is_neox)
    q_ref = q_ref.to(dtype); k_ref = k_ref.to(dtype)
    sgl_kernel.apply_rope_with_cos_sin_cache_inplace(
        positions, q_s, k_s, head_size, cos_sin_cache, is_neox=is_neox,
    )
    torch.testing.assert_close(q_s, q_ref, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(k_s, k_ref, rtol=1e-2, atol=1e-2)

    def torch_run():
        q2 = q.clone(); k2 = k.clone()
        _torch_rope_ref(positions, q2, k2, head_size, cos_sin_cache, is_neox)

    def sgl_run():
        q2 = q.clone(); k2 = k.clone()
        sgl_kernel.apply_rope_with_cos_sin_cache_inplace(
            positions, q2, k2, head_size, cos_sin_cache, is_neox=is_neox,
        )

    torch_ms = np.median(bench_gpu_time(torch_run, repeat_iters=num_iters))
    sgl_ms = np.median(bench_gpu_time(sgl_run, repeat_iters=num_iters))
    return torch_ms, sgl_ms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", nargs="+", type=int, default=[128, 1024, 8192])
    parser.add_argument("--q-heads", type=int, default=32)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--head-size", type=int, default=128)
    parser.add_argument("--max-pos", type=int, default=8192)
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="bfloat16")
    parser.add_argument("--is-neox", action="store_true", default=True)
    parser.add_argument("--num-iters", type=int, default=50)
    args = parser.parse_args()

    dtype = getattr(torch, args.dtype)
    print(f"Device: {torch.cuda.get_device_name()} | dtype: {args.dtype} | q_heads={args.q_heads} "
          f"kv_heads={args.kv_heads} head_size={args.head_size} is_neox={args.is_neox}")
    header = f"{'tokens':>7} {'torch (us)':>12} {'sgl_kernel (us)':>17} {'speedup':>9}"
    print("=" * len(header))
    print(header)
    print("=" * len(header))

    speedups: List[float] = []
    for n in args.tokens:
        t_ms, s_ms = bench_one(n, args.q_heads, args.kv_heads, args.head_size,
                               args.max_pos, dtype, args.is_neox, args.num_iters)
        sp = t_ms / s_ms
        speedups.append(sp)
        print(f"{n:>7} {t_ms * 1e3:>12.2f} {s_ms * 1e3:>17.2f} {sp:>8.2f}x")

    print("=" * len(header))
    print(f"speedup  avg={np.mean(speedups):.2f}x  median={np.median(speedups):.2f}x")


if __name__ == "__main__":
    main()
