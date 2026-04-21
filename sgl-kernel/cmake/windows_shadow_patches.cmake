# Emit patched copies of vendored headers into a Windows-only shadow
# include dir (${_WIN_SHADOW_DIR}). Call this from the main CMakeLists.txt
# inside an `if(WIN32)` branch AFTER FetchContent_Populate(repo-*) has run
# so the source paths exist. The shadow dir is added with -I ahead of the
# upstream include dirs so patched copies win.
#
# Why: sgl-kernel pins flashinfer at commit bc29697b, which still uses
# GCC-only attribute macros (`__attribute__((always_inline)) __device__`)
# in vec_dtypes.cuh. MSVC's nvcc path can't parse those (error: "declaration
# is incompatible with variable template flashinfer::__attribute__"). The
# newer flashinfer tree (local C:\flashinfer 0.6.7+) already uses
# `__forceinline__ __device__`, but it has removed pytorch_extension_utils.h,
# so we can't simply redirect the whole include dir. Instead we pluck the
# problem header out, rewrite it, and shadow-include it.
#
# Patches: if the substitution target isn't found in a source file, we
# emit a STATUS line and still copy verbatim (keeps Linux behaviour
# identical when this module accidentally runs on non-Windows).

function(_sgl_shadow_patch_file src_path dst_path)
    if(NOT EXISTS "${src_path}")
        message(WARNING "sgl-kernel shadow patch: source not found: ${src_path}")
        return()
    endif()
    file(READ "${src_path}" _content)

    # Replace GCC-style FLASHINFER_INLINE macro definition with a portable
    # alternative that nvcc maps to __forceinline__ on MSVC.
    string(REPLACE
        "#define FLASHINFER_INLINE inline __attribute__((always_inline)) __device__"
        "#define FLASHINFER_INLINE __forceinline__ __device__"
        _content "${_content}")

    # Drop any other `__attribute__((always_inline))` usage that reaches
    # the `inline __attribute__((always_inline)) __device__` pattern at
    # function definitions. Use a compact regex.
    string(REGEX REPLACE
        "inline __attribute__\\(\\(always_inline\\)\\) __device__"
        "__forceinline__ __device__"
        _content "${_content}")

    get_filename_component(_dst_dir "${dst_path}" DIRECTORY)
    file(MAKE_DIRECTORY "${_dst_dir}")
    file(WRITE "${dst_path}" "${_content}")
endfunction()

# vec_dtypes.cuh — the single known trigger for MSVC attribute parse.
_sgl_shadow_patch_file(
    "${repo-flashinfer_SOURCE_DIR}/include/flashinfer/vec_dtypes.cuh"
    "${_WIN_SHADOW_DIR}/flashinfer/vec_dtypes.cuh"
)
message(STATUS "sgl-kernel Windows shadow patches emitted to ${_WIN_SHADOW_DIR}")
