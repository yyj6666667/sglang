# Windows Server (experimental)

> **Status:** in-progress port on `branch: windows`. Currently verified on
> RTX 5060 (SM120) + CUDA 12.9 + MSVC 14.44 + Python 3.12.8. Single-GPU
> only. Multi-GPU allreduce is not available on Windows (mscclpp is Linux-only).

## Prerequisites

- Windows 10 / 11 with a recent CUDA 12.x toolkit (tested 12.9).
- Visual Studio 2022 Build Tools with the C++ x64 compiler payload.
- A matched pair of `flashinfer` (editable install from `C:\flashinfer`) and
  this repo on `branch: windows`.
- A Python 3.10+ virtual environment that already has:
  - `torch` built against your CUDA (the existing `C:\flashinfer\.venv`
    ships `torch 2.11.0+cu128`, which satisfies this file's pin range).
  - `flashinfer` installed **editable** from `C:\flashinfer`.

## Install sglang-kt (Python package)

```powershell
cd C:\sglang\python
Copy-Item pyproject_win.toml pyproject.toml -Force
pip install -e . --no-build-isolation
```

## Install sgl-kernel

`sgl-kernel` needs a CUDA + MSVC build. The full native build steps live
in `sgl-kernel/` (Stage 4 / Stage 5 of the port plan). Once built:

```powershell
pip install -e C:\sglang\sgl-kernel --no-build-isolation
```

## What's different from Linux

Dependencies dropped on Windows (unavailable or POSIX-only):

- `uvloop` — Windows ships no wheel; HTTP server falls back to asyncio's
  default event loop.
- `torch_memory_saver` — POSIX `mmap` tricks; handled by a no-op adapter.
- `nvidia-cutlass-dsl`, `quack-kernels` — CuteDSL / quack paths are
  unavailable on Windows; the FA4 Cute-DSL MoE path is disabled.
- `torchcodec` — no Windows wheels; `torchvision`'s decoder handles video.
- `flashinfer_cubin` — JIT cubins from the local editable `flashinfer`
  cache are used instead.
- `decord2`, `st_attn`, `vsa`, `checkpoint-engine` — no Windows wheels or
  Linux-only.
- `triton` (official) — replaced by `triton-windows`, a third-party wheel
  that ships cp312 builds.

Functionality gated behind those dependencies is disabled at runtime
through a handful of small `try/except ImportError` / `sys.platform`
guards in the Python layer. No Linux behavior changes.

## Restore the Linux install

The Windows variant is a **copy** of `pyproject_win.toml` over
`pyproject.toml`; undo is a single file restore:

```bash
cd python
git checkout -- pyproject.toml
```

None of the other pyproject variants (`pyproject_cpu.toml`,
`pyproject_npu.toml`, `pyproject_xpu.toml`, `pyproject_other.toml`) are
modified.
