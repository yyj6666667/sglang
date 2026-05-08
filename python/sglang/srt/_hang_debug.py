"""Cross-layer hang debugger. Single file, zero deps.

Set SGLANG_HANG_DEBUG=0 to disable. SGLANG_HANG_LOG=/path/to/file overrides
default /tmp/sglang_hang.log. Output is greppable on `[HANG]`.

Use:
    from sglang.srt._hang_debug import hlog, hbeat
    hlog("HTTP", "generate_enter", rid=obj.rid)
    hbeat("loop", "EVT_LOOP", "tick", waiting=len(self.waiting_queue))
"""
from __future__ import annotations

import os
import signal
import sys
import threading
import time

_ENABLED = os.environ.get("SGLANG_HANG_DEBUG", "1") != "0"

# Register SIGUSR1 -> dump Python tracebacks of all threads to stderr.
# This bypasses py-spy's ptrace requirement: a normal `kill -USR1 <pid>` from
# the same user works regardless of yama.ptrace_scope.
try:
    import faulthandler as _fh
    if not _fh.is_enabled():
        _fh.enable(file=sys.stderr, all_threads=True)
    _fh.register(signal.SIGUSR1, file=sys.stderr, all_threads=True, chain=False)
except Exception:
    pass

_LOG_PATH = os.environ.get("SGLANG_HANG_LOG", "/tmp/sglang_hang.log")
_PID = os.getpid()
_T0 = time.time()
_LOCK = threading.Lock()
_FH = None
_RANK_CACHE = None
_HEARTBEAT_LAST: dict = {}


def _open():
    global _FH
    if _FH is None:
        try:
            _FH = open(_LOG_PATH, "a", buffering=1)
        except Exception:
            _FH = False
    return _FH or None


def _rank() -> int:
    global _RANK_CACHE
    if _RANK_CACHE is not None:
        return _RANK_CACHE
    for k in ("SGLANG_TP_RANK", "RANK", "LOCAL_RANK"):
        v = os.environ.get(k)
        if v is not None:
            try:
                _RANK_CACHE = int(v)
                return _RANK_CACHE
            except Exception:
                pass
    try:
        import torch.distributed as dist  # type: ignore
        if dist.is_available() and dist.is_initialized():
            _RANK_CACHE = int(dist.get_rank())
            return _RANK_CACHE
    except Exception:
        pass
    _RANK_CACHE = -1
    return _RANK_CACHE


def hlog(layer: str, event: str, **kwargs) -> None:
    if not _ENABLED:
        return
    elapsed = time.time() - _T0
    parts = [
        "[HANG]",
        f"t={elapsed:09.3f}",
        f"pid={_PID}",
        f"rank={_rank()}",
        f"tid={threading.get_ident()}",
        f"layer={layer}",
        f"ev={event}",
    ]
    for k, v in kwargs.items():
        try:
            if isinstance(v, (int, float, bool, str, type(None))):
                sv = str(v)
            else:
                sv = repr(v)
                if len(sv) > 200:
                    sv = sv[:200] + "..."
        except Exception:
            sv = "<unrepr>"
        parts.append(f"{k}={sv}")
    line = " ".join(parts) + "\n"
    with _LOCK:
        try:
            sys.stderr.write(line)
            sys.stderr.flush()
        except Exception:
            pass
        fh = _open()
        if fh is not None:
            try:
                fh.write(line)
                fh.flush()
            except Exception:
                pass


def hbeat(key: str, layer: str, event: str, every_seconds: float = 1.0, **kwargs) -> None:
    """Heartbeat: at most one print per `every_seconds` per `key`. Always fires on the first call."""
    if not _ENABLED:
        return
    now = time.time()
    last = _HEARTBEAT_LAST.get(key, 0.0)
    if last and now - last < every_seconds:
        return
    _HEARTBEAT_LAST[key] = now
    hlog(layer, event, **kwargs)
