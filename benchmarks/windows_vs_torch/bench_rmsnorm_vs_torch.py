#!/usr/bin/env python3
"""Benchmark comparing a PyTorch RMSNorm baseline vs sgl_kernel.rmsnorm.

Prints per-shape latency, speedup, and effective bandwidth. Modeled on
flashinfer's bench_rmsnorm_vs_torch.py (convention: Windows-port
benchmarks report speedup vs torch and use flashinfer.testing.bench_gpu_time
for CUPTI-backed timing).
"""

import argparse
from typing import List

import numpy as np
import torch

import sgl_kernel
from flashinfer.testing.utils import bench_gpu_time


def torch_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    rms = torch.sqrt(torch.mean(x.float() ** 2, dim=-1, keepdim=True) + eps)
    return (x.float() / rms * weight.float()).to(x.dtype)


@torch.inference_mode()
def bench_one(batch: int, hidden: int, dtype: torch.dtype, eps: float, num_iters: int):
    x = torch.randn((batch, hidden), dtype=dtype, device="cuda")
    weight = torch.randn(hidden, dtype=dtype, device="cuda")

    ref = torch_rmsnorm(x, weight, eps)
    out = sgl_kernel.rmsnorm(x, weight, eps=eps)
    torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-2)

    torch_ms = np.median(
        bench_gpu_time(lambda: torch_rmsnorm(x, weight, eps), repeat_iters=num_iters)
    )
    sgl_ms = np.median(
        bench_gpu_time(lambda: sgl_kernel.rmsnorm(x, weight, eps=eps), repeat_iters=num_iters)
    )

    bytes_moved = (
        x.numel() * x.element_size()
        + weight.numel() * weight.element_size()
        + x.numel() * x.element_size()
    )
    sgl_tbps = bytes_moved / (sgl_ms * 1e-3) / 1e12
    return torch_ms, sgl_ms, sgl_tbps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batches", nargs="+", type=int, default=[1, 99, 989])
    parser.add_argument("--hiddens", nargs="+", type=int, default=[1024, 4096, 8192, 16384])
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="bfloat16")
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--num-iters", type=int, default=100)
    args = parser.parse_args()

    dtype = getattr(torch, args.dtype)
    print(f"Device: {torch.cuda.get_device_name()} | dtype: {args.dtype}")
    header = (
        f"{'batch':>6} {'hidden':>7} {'torch (us)':>12} "
        f"{'sgl_kernel (us)':>17} {'speedup':>9} {'TB/s':>7}"
    )
    print("=" * len(header))
    print(header)
    print("=" * len(header))

    speedups: List[float] = []
    for b in args.batches:
        for h in args.hiddens:
            t_ms, s_ms, s_tbps = bench_one(b, h, dtype, args.eps, args.num_iters)
            sp = t_ms / s_ms
            speedups.append(sp)
            print(f"{b:>6} {h:>7} {t_ms * 1e3:>12.2f} {s_ms * 1e3:>17.2f} {sp:>8.2f}x {s_tbps:>7.2f}")

    print("=" * len(header))
    print(f"speedup  avg={np.mean(speedups):.2f}x  median={np.median(speedups):.2f}x  min={np.min(speedups):.2f}x  max={np.max(speedups):.2f}x")


if __name__ == "__main__":
    main()
