# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0
"""Low-overhead NVTX ranges for decode profiling.

Set ``SGLANG_DECODE_PROFILE=1`` before launching the server to enable the
markers. Fine-grained model ranges require eager execution; launch with
``--disable-cuda-graph`` when collecting them. The ranges never synchronize
CUDA, so profiling does not serialize the normal CPU/GPU overlap.
"""

import os

import torch

_DECODE_PROFILE_ENABLED = os.environ.get("SGLANG_DECODE_PROFILE") == "1"
_RANGE_PREFIX = "sglang.decode."


class _NoopRange:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc_value, traceback):
        return False


_NOOP_RANGE = _NoopRange()


def is_decode_profile_enabled(forward_batch=None) -> bool:
    if not _DECODE_PROFILE_ENABLED or not torch.cuda.is_available():
        return False
    if forward_batch is None:
        return True
    return forward_batch.forward_mode.is_decode()


def decode_profile_range(name: str, forward_batch=None):
    """Return an NVTX range when decode profiling is enabled, else a no-op."""
    if not is_decode_profile_enabled(forward_batch):
        return _NOOP_RANGE
    return torch.cuda.nvtx.range(_RANGE_PREFIX + name)
