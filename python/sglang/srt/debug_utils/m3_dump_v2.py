"""Hidden-state dumper for CPU vs GPU prefill comparison.

Usage:
  M3_DUMP=/tmp/m3.dump M3_DUMP_LAYER_MAX=10 python -m sglang.launch_server ...

For each `dump(tag, t)` call, writes one line to `${M3_DUMP}.pid<PID>` with:
  tag | shape | dtype | absmax | absmean | nan | inf | first5

Gates:
  - M3_DUMP env var must be set
  - skipped during cuda graph capture (.item() forbidden there)
  - per-process file (PID suffix) so 8 TPs don't race
  - per-call counter to bound output (default 5000)
  - layer_idx <= M3_DUMP_LAYER_MAX (default 10) when layer info given

The `first5` field captures the FIRST 5 values of the flattened tensor as
hex-formatted floats. CPU/GPU runs with same prompt + temp=0 + seed produce
bit-identical tensors UNTIL the first divergent stage — that line's first5
mismatch pinpoints the bug.
"""
import os
import sys
import threading

import torch


_MAX_CALLS = int(os.environ.get("M3_DUMP_MAX_CALLS", "20000"))
_LAYER_MAX = int(os.environ.get("M3_DUMP_LAYER_MAX", "10"))
_TP_ONLY = os.environ.get("M3_DUMP_TP", "0")  # only this TP rank dumps

_counter = 0
_lock = threading.Lock()


def _enabled():
    return os.environ.get("M3_DUMP") is not None


def _path():
    base = os.environ["M3_DUMP"]
    return f"{base}.pid{os.getpid()}"


def _filter_layer(layer):
    if layer is None:
        return True
    try:
        return int(layer) <= _LAYER_MAX
    except Exception:
        return True


def dump(tag, t=None, layer=None, **extras):
    if not _enabled():
        return
    global _counter
    if _counter >= _MAX_CALLS:
        return
    # Skip during cuda graph capture (item/sync forbidden)
    try:
        if torch.cuda.is_current_stream_capturing():
            return
    except Exception:
        pass
    if not _filter_layer(layer):
        return
    # TP filter — sglang sets local rank via LOCAL_RANK or by passing into worker
    try:
        rank = int(os.environ.get("LOCAL_RANK", "0"))
        if str(rank) != _TP_ONLY:
            return
    except Exception:
        pass

    with _lock:
        if _counter >= _MAX_CALLS:
            return
        _counter += 1
    parts = [f"#{_counter:05d}", tag]
    if layer is not None:
        parts.append(f"layer={layer}")
    if t is not None and hasattr(t, "shape"):
        try:
            parts.append(f"shape={tuple(t.shape)}")
            parts.append(f"dtype={t.dtype}")
            tf = (
                t.float()
                if t.dtype
                in (torch.float8_e4m3fn, torch.float8_e5m2, torch.bfloat16, torch.float16)
                else t
            )
            if tf.is_floating_point():
                parts.append(f"absmax={tf.abs().max().item():.6e}")
                parts.append(f"absmean={tf.abs().mean().item():.6e}")
                parts.append(f"nan={int(tf.isnan().sum())}")
                parts.append(f"inf={int(tf.isinf().sum())}")
                flat = tf.flatten()
                n = min(5, flat.numel())
                first5 = flat[:n].detach().cpu().tolist()
                parts.append(
                    "first5=[" + ",".join(f"{x:+.6e}" for x in first5) + "]"
                )
            else:
                parts.append(f"min={tf.min().item()}")
                parts.append(f"max={tf.max().item()}")
                parts.append(f"sum={tf.sum().item()}")
        except Exception as e:
            parts.append(f"stat_err={type(e).__name__}:{e}")
    else:
        parts.append(f"value={t}")
    for k, v in extras.items():
        parts.append(f"{k}={v}")

    line = " | ".join(parts) + "\n"
    try:
        with open(_path(), "a") as f:
            f.write(line)
    except Exception:
        pass


def reset():
    global _counter
    _counter = 0


def banner(msg):
    """Force-write a marker line (ignores counter)."""
    if not _enabled():
        return
    try:
        with open(_path(), "a") as f:
            f.write(f"=== {msg} ===\n")
    except Exception:
        pass
