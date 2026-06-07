"""
M3 hidden-states / kernel-IO statistics dump (debug branch only).

Gated by environment variable M3_DEBUG_DUMP=1. Writes one line per
recorded tensor to /tmp/m3-hidden-dump.log with:
    shape, dtype, min, max, abs_mean, has_nan, has_inf

Used to localize the *first* layer or sub-op that produces NaN/Inf
without spamming 8x TP-ranks worth of noise (rank 0 only by default).
"""

from __future__ import annotations

import os
import threading
from typing import Optional

import torch

_DUMP_PATH = os.environ.get("M3_DEBUG_DUMP_PATH", "/tmp/m3-hidden-dump.log")
_ENABLED = os.environ.get("M3_DEBUG_DUMP") == "1"
_GLOBAL_COUNTER = 0
_GLOBAL_LOCK = threading.Lock()
_TP_RANK: Optional[int] = None
_FH = None


def enabled() -> bool:
    return _ENABLED


def _open_log():
    global _FH
    if _FH is None:
        _FH = open(_DUMP_PATH, "a", buffering=1)
        _FH.write("\n===== new run pid=%d =====\n" % os.getpid())
    return _FH


def set_tp_rank(rank: int) -> None:
    global _TP_RANK
    _TP_RANK = rank


def record(name: str, tensor, layer_idx=None) -> None:
    """Record one tensor's stats. Cheap if disabled; no-op for non-Tensor input."""
    if not _ENABLED:
        return
    if _TP_RANK is not None and _TP_RANK != 0:
        return
    if not isinstance(tensor, torch.Tensor):
        return

    global _GLOBAL_COUNTER
    with _GLOBAL_LOCK:
        idx = _GLOBAL_COUNTER
        _GLOBAL_COUNTER += 1
    try:
        shape = tuple(tensor.shape)
        dtype = str(tensor.dtype).replace("torch.", "")
        if tensor.numel() == 0:
            line = (
                f"#{idx:05d} [L{layer_idx}/{name}] shape={shape} dtype={dtype} EMPTY\n"
            )
        else:
            # Cast to float32 for stats so int/uint8/fp8 etc. don't blow up
            try:
                f = tensor.detach().float()
            except Exception:
                # FP8 isn't directly castable on all versions; reinterpret via bf16
                f = tensor.detach().to(torch.float32)
            has_nan = bool(torch.isnan(f).any().item())
            has_inf = bool(torch.isinf(f).any().item())
            mx = float(f.max().item()) if not has_nan else float("nan")
            mn = float(f.min().item()) if not has_nan else float("nan")
            am = float(f.abs().mean().item()) if not has_nan else float("nan")
            line = (
                f"#{idx:05d} [L{layer_idx}/{name}] shape={shape} dtype={dtype} "
                f"min={mn:.3e} max={mx:.3e} abs_mean={am:.3e} "
                f"nan={has_nan} inf={has_inf}\n"
            )
    except Exception as e:
        line = f"#{idx:05d} [L{layer_idx}/{name}] DUMP_ERROR: {e}\n"

    fh = _open_log()
    fh.write(line)


def _make_hook(name: str, layer_idx=None):
    def hook(module, inputs, output):
        if isinstance(output, tuple):
            for j, o in enumerate(output):
                record(f"{name}[{j}]", o, layer_idx=layer_idx)
        else:
            record(name, output, layer_idx=layer_idx)
    return hook


def register_all_hooks(model, tp_rank: int = 0) -> None:
    """Attach forward hooks at strategic points across the M3 model."""
    if not _ENABLED:
        return
    set_tp_rank(tp_rank)
    if tp_rank != 0:
        return  # only rank 0 hooks

    fh = _open_log()
    fh.write(f"==== register_all_hooks model={type(model).__name__} ====\n")

    # Walk to the actual decoder stack: model.model usually for CausalLM wrappers
    inner = getattr(model, "model", model)
    layers = getattr(inner, "layers", None)
    embed = getattr(inner, "embed_tokens", None)
    final_norm = getattr(inner, "norm", None)
    lm_head = getattr(model, "lm_head", None)

    n_hooks = 0
    if embed is not None:
        embed.register_forward_hook(_make_hook("embed_tokens"))
        n_hooks += 1

    if layers is not None:
        for i, layer in enumerate(layers):
            for attr in (
                "input_layernorm",
                "post_attention_layernorm",
            ):
                sub = getattr(layer, attr, None)
                if sub is not None:
                    sub.register_forward_hook(_make_hook(attr, layer_idx=i))
                    n_hooks += 1

            attn = getattr(layer, "self_attn", None)
            if attn is not None:
                attn.register_forward_hook(_make_hook("self_attn", layer_idx=i))
                n_hooks += 1
                for sub_name in ("qkv_proj", "o_proj"):
                    sub = getattr(attn, sub_name, None)
                    if sub is not None:
                        sub.register_forward_hook(
                            _make_hook(f"self_attn.{sub_name}", layer_idx=i)
                        )
                        n_hooks += 1

            mlp = getattr(layer, "mlp", None)
            if mlp is not None:
                mlp.register_forward_hook(_make_hook("mlp", layer_idx=i))
                n_hooks += 1
                for sub_name in ("gate", "experts", "shared_experts"):
                    sub = getattr(mlp, sub_name, None)
                    if sub is not None:
                        sub.register_forward_hook(
                            _make_hook(f"mlp.{sub_name}", layer_idx=i)
                        )
                        n_hooks += 1

            layer.register_forward_hook(_make_hook("layer", layer_idx=i))
            n_hooks += 1

    if final_norm is not None:
        final_norm.register_forward_hook(_make_hook("model.norm"))
        n_hooks += 1
    if lm_head is not None:
        lm_head.register_forward_hook(_make_hook("lm_head"))
        n_hooks += 1

    fh.write(f"==== hooks installed: {n_hooks} ====\n")
