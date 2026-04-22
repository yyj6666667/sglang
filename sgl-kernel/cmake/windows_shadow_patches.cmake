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
        message(WARNING "sgl-kernel Windows patch: target not found: ${file_path}")
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
        # __attribute__((weak)) is GCC/Clang-only; MSVC has no identical
        # semantics. For the Python module init stub it's only used to
        # allow multiple TUs to redefine the symbol without ODR errors,
        # which MSVC handles via single definition per DLL anyway. Drop
        # the attribute on Windows.
        string(REGEX REPLACE
            "__attribute__\\(\\(weak\\)\\)[ \t]*"
            ""
            _content "${_content}")
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
