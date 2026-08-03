import importlib.util
import json
import struct
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
TARGET_PATH = ROOT / "python/sglang/srt/configs/_model_config_dsv4.py"
SPEC = importlib.util.spec_from_file_location(
    "_dsv4_config_traits_test_target", TARGET_PATH
)
TARGET = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TARGET)


def _config(architecture="DeepseekV4ForCausalLM", **kwargs):
    return SimpleNamespace(architectures=[architecture], **kwargs)


def _write_safetensors_header(path: Path, dtype: str):
    header = {
        "model.layers.0.mlp.experts.0.w1.weight": {
            "dtype": dtype,
            "shape": [1],
            "data_offsets": [0, 0],
        }
    }
    encoded = json.dumps(header).encode("utf-8")
    with path.open("wb") as output:
        output.write(struct.pack("<Q", len(encoded)))
        output.write(encoded)


class _EnvBool:
    def __init__(self, value=False, is_set=False):
        self.value = value
        self.explicitly_set = is_set

    def get(self):
        return self.value

    def is_set(self):
        return self.explicitly_set


def _package(name):
    module = types.ModuleType(name)
    module.__path__ = []
    return module


class TestDeepSeekV4ConfigTraits(unittest.TestCase):
    def test_flash_layout_requires_both_markers(self):
        flash = _config(head_dim=512, sliding_window=128)
        legacy = _config(qk_nope_head_dim=448, v_head_dim=512, window_size=128)
        non_v4 = _config(
            architecture="DeepseekV3ForCausalLM",
            head_dim=512,
            sliding_window=128,
        )

        self.assertTrue(TARGET.is_deepseek_v4_flash_config(flash))
        self.assertTrue(
            TARGET.is_deepseek_v4_flash_config(
                _config(
                    architecture="DeepseekV4ForCausalLMNextN",
                    head_dim=512,
                    sliding_window=128,
                )
            )
        )
        self.assertFalse(TARGET.is_deepseek_v4_flash_config(legacy))
        self.assertFalse(TARGET.is_deepseek_v4_flash_config(non_v4))

        with self.assertRaisesRegex(ValueError, "must be provided together"):
            TARGET.is_deepseek_v4_flash_config(_config(head_dim=512))
        with self.assertRaisesRegex(ValueError, "must be provided together"):
            TARGET.is_deepseek_v4_flash_config(_config(sliding_window=128))

    def test_swiglu_clamp_comes_from_checkpoint_config(self):
        self.assertTrue(
            TARGET.has_deepseek_v4_swiglu_clamp(
                _config(head_dim=512, sliding_window=128, swiglu_limit=10.0)
            )
        )
        self.assertFalse(
            TARGET.has_deepseek_v4_swiglu_clamp(
                _config(head_dim=512, sliding_window=128, swiglu_limit=None)
            )
        )
        self.assertFalse(
            TARGET.has_deepseek_v4_swiglu_clamp(
                _config(architecture="OtherForCausalLM", swiglu_limit=10.0)
            )
        )

    def test_probe_reads_routed_expert_dtype(self):
        with tempfile.TemporaryDirectory() as directory:
            model_dir = Path(directory)
            _write_safetensors_header(model_dir / "model.safetensors", "U8")
            self.assertEqual(
                TARGET.probe_routed_expert_weight_dtype(str(model_dir)), "U8"
            )

    def test_detect_fp4_is_per_model_and_honors_override(self):
        env_flag = _EnvBool()
        environ = types.ModuleType("sglang.srt.environ")
        environ.envs = SimpleNamespace(SGLANG_DSV4_FP4_EXPERTS=env_flag)
        utils = types.ModuleType("sglang.srt.utils")
        stubs = {
            "sglang": _package("sglang"),
            "sglang.srt": _package("sglang.srt"),
            "sglang.srt.environ": environ,
            "sglang.srt.utils": utils,
        }

        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(
            sys.modules, stubs
        ):
            model_dir = Path(directory)
            weights = model_dir / "model.safetensors"
            model_config = SimpleNamespace(
                hf_config=_config(head_dim=512, sliding_window=128),
                model_path=str(model_dir),
                revision=None,
            )

            _write_safetensors_header(weights, "U8")
            self.assertTrue(TARGET.detect_dsv4_fp4_experts(model_config))

            _write_safetensors_header(weights, "F8_E4M3")
            self.assertFalse(TARGET.detect_dsv4_fp4_experts(model_config))

            env_flag.value = True
            env_flag.explicitly_set = True
            self.assertTrue(TARGET.detect_dsv4_fp4_experts(model_config))

            model_config.hf_config = _config(
                architecture="DeepseekV3ForCausalLM",
                head_dim=512,
                sliding_window=128,
            )
            self.assertFalse(TARGET.detect_dsv4_fp4_experts(model_config))

            model_config.hf_config = _config(head_dim=512, sliding_window=128)
            model_config.model_path = "organization/model"
            env_flag.value = False
            env_flag.explicitly_set = False
            utils.find_local_repo_dir = lambda _repo, revision=None: str(model_dir)
            _write_safetensors_header(weights, "U8")
            self.assertTrue(TARGET.detect_dsv4_fp4_experts(model_config))

    def test_removed_mode_envs_have_no_runtime_references(self):
        forbidden = ("SGLANG_DSV4_MODE", "SGLANG_DSV4_2604_SUBMODE")
        source_root = ROOT / "python/sglang/srt"
        references = []
        for path in source_root.rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            for name in forbidden:
                if name in text:
                    references.append(f"{path.relative_to(ROOT)}: {name}")
        self.assertEqual(references, [])


if __name__ == "__main__":
    unittest.main()
