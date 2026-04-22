# Apply targeted in-place edits to the FetchContent-populated vendored
# headers that MSVC's nvcc path can't compile as-is. Call this from the
# main CMakeLists.txt inside an `if(WIN32)` branch AFTER FetchContent_Populate
# has run.
#
# In-place (instead of shadow-include) is required because several offending
# headers are included via relative paths (e.g. flashinfer internal files
# doing `#include "../vec_dtypes.cuh"`), which the compiler resolves against
# the source file's own directory — never consulting `-I`.
#
# Each patch is idempotent via a marker comment: the second run is a no-op.
# CMake re-runs this at every configure, so a fresh clone of the FetchContent
# tree re-applies the patch automatically.
#
# Patches applied:
# 1. flashinfer/vec_dtypes.cuh  — FLASHINFER_INLINE macro: rewrite
#    `inline __attribute__((always_inline)) __device__` -> `__forceinline__ __device__`.
#    MSVC's nvcc front-end mis-forwards the GCC attribute to a variable
#    template `flashinfer::__attribute__`, yielding C2143/C3861-style parse
#    errors throughout.
# 2. flashinfer/math.cuh  — replace stray `ushort` typedefs with
#    `unsigned short` (MSVC's CRT has no `ushort` alias).

function(_sgl_inplace_patch file_path marker label)
    if(NOT EXISTS "${file_path}")
        # Non-fatal: FA/DeepGEMM may not be populated on every build; skip.
        return()
    endif()
    file(READ "${file_path}" _content)
    if(_content MATCHES "${marker}")
        # already patched
        return()
    endif()
    set(_orig "${_content}")

    # The caller passes a function name via ${label} whose body does the
    # actual replacements on the local _content. We do this with one of
    # a fixed set of hard-coded transforms rather than dynamic dispatch.
    if(label STREQUAL "vec_dtypes")
        string(REPLACE
            "#define FLASHINFER_INLINE inline __attribute__((always_inline)) __device__"
            "#define FLASHINFER_INLINE __forceinline__ __device__   // ${marker}"
            _content "${_content}")
        string(REGEX REPLACE
            "inline __attribute__\\(\\(always_inline\\)\\) __device__"
            "__forceinline__ __device__"
            _content "${_content}")
    elseif(label STREQUAL "math")
        # Only replace `ushort` at word boundaries so `__half_as_ushort`,
        # `__ushort_as_half`, etc. stay intact. CMake's regex uses `[^A-Za-z_]`
        # lookarounds simulated via a captured prefix/suffix.
        # Match `([^A-Za-z0-9_])ushort([^A-Za-z0-9_])` globally.
        string(REGEX REPLACE
            "([^A-Za-z0-9_])ushort([^A-Za-z0-9_])"
            "\\1unsigned short\\2"
            _content "${_content}")
        # Leave a marker so the idempotency check fires next run.
        set(_content "// ${marker}\n${_content}")
    elseif(label STREQUAL "pytorch_ext_utils")
        # __attribute__((weak)) is GCC/Clang-only. On MSVC, make the
        # function `inline` so the compiler emits it with COMDAT linkage
        # and the linker de-duplicates the per-TU copies. Without this,
        # PyInit_TORCH_EXTENSION_NAME fires LNK2005 for every .obj beyond
        # the first.
        string(REGEX REPLACE
            "__attribute__\\(\\(weak\\)\\)"
            "inline"
            _content "${_content}")
        # Windows SDK pollution: rpcndr.h defines `small` as `char`, and
        # minwindef.h defines min/max. These bleed into torch's
        # CUDACachingAllocator.h (parameter named `small`) and cutlass
        # templates using std::min/max. Since this header is always
        # pulled in before torch/all.h, undef them here so every TU
        # downstream sees clean identifiers.
        set(_content "// ${marker}\n#ifdef _MSC_VER\n#ifdef small\n#undef small\n#endif\n#ifdef min\n#undef min\n#endif\n#ifdef max\n#undef max\n#endif\n#endif\n${_content}")
    elseif(label STREQUAL "alt_tokens")
        # Rewrite C++ alternative operator tokens (`and`, `or`, `not`) to
        # their punctuation spellings. nvcc's EDG front-end on MSVC does
        # NOT recognise the alternative tokens as keywords even with
        # /permissive-; patching the source is simpler than forcing an
        # include of <ciso646> (which causes corecrt redefinitions).
        #
        # Match only when surrounded by a `)` / `(` token so that
        # comments using "and" as English are unaffected. Applied to a
        # specific list of upstream headers; each file uses these tokens
        # sparingly and the patterns are unambiguous in code.
        string(REGEX REPLACE "\\) and \\(" ") && (" _content "${_content}")
        string(REGEX REPLACE "\\) or \\(" ") || (" _content "${_content}")
        string(REGEX REPLACE "! and " "! && " _content "${_content}")
        set(_content "// ${marker}\n${_content}")
    else()
        message(FATAL_ERROR "sgl-kernel Windows patch: unknown label '${label}'")
    endif()

    if(_content STREQUAL _orig)
        message(STATUS "sgl-kernel Windows patch: no replacements made in ${file_path} (source may already use MSVC-friendly form)")
        return()
    endif()

    file(WRITE "${file_path}" "${_content}")
    message(STATUS "sgl-kernel Windows patch: rewrote ${file_path}")
endfunction()

_sgl_inplace_patch(
    "${repo-flashinfer_SOURCE_DIR}/include/flashinfer/vec_dtypes.cuh"
    "SGL_KERNEL_WIN_PATCH_VEC_DTYPES"
    vec_dtypes
)

_sgl_inplace_patch(
    "${repo-flashinfer_SOURCE_DIR}/include/flashinfer/math.cuh"
    "SGL_KERNEL_WIN_PATCH_MATH"
    math
)

_sgl_inplace_patch(
    "${repo-flashinfer_SOURCE_DIR}/csrc/pytorch_extension_utils.h"
    "SGL_KERNEL_WIN_PATCH_PYEXT"
    pytorch_ext_utils
)

_sgl_inplace_patch(
    "${repo-flashinfer_SOURCE_DIR}/include/flashinfer/pos_enc.cuh"
    "SGL_KERNEL_WIN_PATCH_POS_ENC"
    alt_tokens
)

_sgl_inplace_patch(
    "${repo-flashinfer_SOURCE_DIR}/include/flashinfer/trtllm/batched_gemm/trtllmGen_bmm_export/KernelParams.h"
    "SGL_KERNEL_WIN_PATCH_KPARAMS"
    alt_tokens
)

# cutlass headers using alternative operator tokens (`and`/`or`/`not`).
# Nested inclusions (cute/int_tuple, cute/layout) are hit transitively by
# a lot of our kernels — patch aggressively.
set(_SGL_CUTLASS_ALT_TOKEN_FILES
    "cutlass/conv/threadblock/conv2d_dgrad_output_gradient_tile_access_iterator_analytic.h"
    "cutlass/conv/threadblock/conv2d_dgrad_output_gradient_tile_access_iterator_optimized.h"
    "cutlass/exmy_base.h"
    "cutlass/gemm/kernel/gemm_universal_decl.h"
    "cutlass/gemm/kernel/symm_universal.h"
    "cutlass/gemm/kernel/trmm_universal.h"
    "cutlass/transform/kernel/sm90_sparse_gemm_compressor.hpp"
    "cute/int_tuple.hpp"
    "cute/layout.hpp"
)
foreach(_rel IN LISTS _SGL_CUTLASS_ALT_TOKEN_FILES)
    string(REPLACE "/" "_" _markname "${_rel}")
    string(REPLACE "." "_" _markname "${_markname}")
    _sgl_inplace_patch(
        "${repo-cutlass_SOURCE_DIR}/include/${_rel}"
        "SGL_KERNEL_WIN_PATCH_CUTLASS_${_markname}"
        alt_tokens
    )
    # Flash-attention pulls its own embedded copy of cutlass — patch that
    # tree too. Path exists only if FA sources were fetched.
    _sgl_inplace_patch(
        "${repo-flash-attention_SOURCE_DIR}/csrc/cutlass/include/${_rel}"
        "SGL_KERNEL_WIN_PATCH_FA_CUTLASS_${_markname}"
        alt_tokens
    )
endforeach()
