"""DSV4 config traits and routed-expert checkpoint detection.

Imported lazily by :meth:`ModelConfig._set_dsv4_model_traits` so non-DSV4
models do not load the safetensors probing path.
"""

from __future__ import annotations

import json
import logging
import os
import re
import struct
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from sglang.srt.configs.model_config import ModelConfig

logger = logging.getLogger(__name__)


_DSV4_ARCHITECTURES = {
    "DeepseekV4ForCausalLM",
    "DeepseekV4ForCausalLMNextN",
}


def is_deepseek_v4_config(config: Any) -> bool:
    architectures = getattr(config, "architectures", None) or []
    return bool(architectures and architectures[0] in _DSV4_ARCHITECTURES)


def is_deepseek_v4_flash_config(config: Any) -> bool:
    """Return whether *config* uses the V4-Flash checkpoint layout.

    V4-Flash checkpoints carry ``head_dim`` and ``sliding_window`` while the
    earlier V4 layout carries ``qk_nope_head_dim``/``v_head_dim`` and
    ``window_size``.  Requiring the two Flash markers together prevents a
    partially overridden config from silently selecting the wrong layout.
    """
    if not is_deepseek_v4_config(config):
        return False

    has_head_dim = getattr(config, "head_dim", None) is not None
    has_sliding_window = getattr(config, "sliding_window", None) is not None
    if has_head_dim != has_sliding_window:
        raise ValueError(
            "Invalid DeepSeek V4 config: head_dim and sliding_window must be "
            "provided together for the V4-Flash layout."
        )
    return has_head_dim


def has_deepseek_v4_swiglu_clamp(config: Any) -> bool:
    return (
        is_deepseek_v4_config(config)
        and getattr(config, "swiglu_limit", None) is not None
    )


# Matches routed-expert weight keys in both HF-style layouts
# (``...mlp.experts.<N>.{gate,up,down}_proj.weight``) and V4-Flash-style
# layouts (``...ffn.experts.<N>.w{1,2,3}.weight``). ``shared_experts`` is
# excluded because the index segment requires a digit after ``.experts.``.
_ROUTED_EXPERT_KEY_RE = re.compile(
    r"\.experts\.\d+\.(?:w[123]|down_proj|up_proj|gate_proj)\.weight$"
)


def probe_routed_expert_weight_dtype(model_path: str) -> Optional[str]:
    """Return the safetensors dtype string (e.g. ``F8_E4M3``, ``U8``) of one
    routed-expert weight tensor, or ``None`` if the checkpoint is remote or has
    no matching key. Reads only the safetensors header of the relevant shard.
    """
    if not os.path.isdir(model_path):
        return None

    index_file = os.path.join(model_path, "model.safetensors.index.json")
    target_key = None
    target_shard_path = None

    if os.path.exists(index_file):
        with open(index_file) as f:
            index = json.load(f)
        weight_map = index.get("weight_map", {}) or {}
        for k, shard in weight_map.items():
            if _ROUTED_EXPERT_KEY_RE.search(k):
                target_key = k
                target_shard_path = os.path.join(model_path, shard)
                break
        if target_key is None:
            return None
    else:
        shards = sorted(Path(model_path).glob("*.safetensors"))
        if not shards:
            return None
        target_shard_path = str(shards[0])

    with open(target_shard_path, "rb") as f:
        (header_len,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(header_len))

    if target_key is not None:
        meta = header.get(target_key)
        return meta.get("dtype") if meta else None

    for k, meta in header.items():
        if k == "__metadata__" or not isinstance(meta, dict):
            continue
        if _ROUTED_EXPERT_KEY_RE.search(k):
            return meta.get("dtype")
    return None


def detect_dsv4_fp4_experts(model_config: "ModelConfig") -> bool:
    """Resolve the routed-expert layout once for this model instance.

    An explicit ``SGLANG_DSV4_FP4_EXPERTS`` remains a compatibility override.
    Otherwise the local safetensors header is probed without mutating process
    environment state.  Non-V4-Flash models always return ``False``.
    """
    from sglang.srt.environ import envs

    if not is_deepseek_v4_flash_config(model_config.hf_config):
        return False
    if envs.SGLANG_DSV4_FP4_EXPERTS.is_set():
        return envs.SGLANG_DSV4_FP4_EXPERTS.get()

    model_path = model_config.model_path
    if not os.path.isdir(model_path):
        try:
            from sglang.srt.utils import find_local_repo_dir

            model_path = (
                find_local_repo_dir(model_path, revision=model_config.revision)
                or model_path
            )
        except Exception:
            pass

    try:
        dtype = probe_routed_expert_weight_dtype(model_path)
    except Exception as e:
        logger.warning(
            "Failed to probe routed-expert dtype for %s; keeping "
            "SGLANG_DSV4_FP4_EXPERTS default. Reason: %s",
            model_path,
            e,
        )
        return envs.SGLANG_DSV4_FP4_EXPERTS.get()
    if dtype is None:
        return envs.SGLANG_DSV4_FP4_EXPERTS.get()
    if dtype in ("U8", "I8", "F4"):
        is_fp4_experts = True
    elif dtype == "F8_E4M3":
        is_fp4_experts = False
    else:
        logger.warning(
            "Unexpected routed-expert safetensors dtype=%s for V4-Flash; "
            "keeping SGLANG_DSV4_FP4_EXPERTS default.",
            dtype,
        )
        return envs.SGLANG_DSV4_FP4_EXPERTS.get()
    logger.info(
        "Auto-detected routed-expert safetensors dtype=%s; is_fp4_experts=%s",
        dtype,
        is_fp4_experts,
    )
    return is_fp4_experts
