/* $Header: /Users/dlr/src/mdfind/gpu/RCS/gpu_kernel_cache.h,v 1.3 2026/05/04 07:54:52 dlr Exp dlr $
 *
 * gpu_kernel_cache — bare-files cache for OpenCL device-binary programs.
 *
 * Layout under <dir>/gpu-kernels/ where <dir> is dirname(MDXFIND_CACHE):
 *   .cache.lock            empty file used as the lock target for cache-wide
 *                          version invalidation (rm + write fresh cache.version)
 *   cache.version          one line: mdxfind binary RCS rev (e.g. "1.372")
 *   <key>.bin              kernel binary (raw clGetProgramInfo output)
 *   <key>.meta             provenance text + binary SHA-256, also the lock
 *                          target for compile-and-cache deduplication.
 *
 * Cache key (24 hex chars): first 12 bytes of
 *   SHA-256(source || device_name || driver_version || cl_platform_version || mdxfind_rev)
 *
 * Eviction policies:
 *   1. Cache-wide: mdxfind rev change → rm *.bin *.meta + write fresh cache.version.
 *   2. Per-entry: file load failure, SHA-256 mismatch, clCreateProgramWithBinary
 *      failure, or clBuildProgram-on-binary failure → unlink that entry.
 *
 * Concurrency: flock(LOCK_EX) on <key>.meta brackets the compile-and-store
 * window so concurrent mdxfind processes don't duplicate the JIT work.
 *
 * If MDXFIND_CACHE is unset on first call to gpu_kernel_cache_init():
 *   one stderr warning, cache disabled, no directory created, no probes.
 *   All subsequent gpu_kernel_cache_* calls become no-ops returning
 *   "miss" / NULL / -1 as appropriate.
 */

#ifndef GPU_KERNEL_CACHE_H
#define GPU_KERNEL_CACHE_H

#include <stddef.h>
#include <stdint.h>

/* Do NOT pull in <CL/cl.h> here. mdxfind.c only needs the init+enabled
 * entrypoints (no cl_* types) and including cl.h here propagates
 * Windows intrin.h via cl_platform.h on the win cross-compile path,
 * which then conflicts with GCC's __cpuid builtin macro. The cl_*
 * declarations below are only exposed if the caller has already
 * included cl.h itself (gpu_opencl.c does, mdxfind.c doesn't). */

#ifdef __cplusplus
extern "C" {
#endif

/* Initialize the cache. Reads MDXFIND_CACHE, derives cache dir via
 * path_dir_len(), validates cache.version against `mdxfind_rev`, and
 * (under .cache.lock) nukes the kernel directory contents on mismatch.
 *
 * Idempotent — second and later calls are a no-op.
 *
 * Returns 0 if the cache is enabled, -1 if disabled (no MDXFIND_CACHE,
 * mkdir failure, or other init error). On disabled, all subsequent
 * gpu_kernel_cache_* calls become no-ops. */
int gpu_kernel_cache_init(const char *mdxfind_rev);

/* True if the cache is initialized AND enabled. */
int gpu_kernel_cache_enabled(void);

/* Build (or load from cache) an OpenCL program for `dev`. This is the
 * single-call wrapper that does the entire cache-or-compile flow:
 *   1. compute key from sources + defines + device + driver + cl_ver + mdxfind_rev
 *   2. try cache load (with self-healing eviction on file/checksum/binary
 *      load/build failure)
 *   3. on miss: lock <key>.meta, recheck (peer may have just finished),
 *      compile from source, store binary + meta atomically, unlock
 *   4. return the built program object
 *
 * Build options string passed to both clBuildProgram paths.
 *
 * `defines_str` (Memo B B2 R3 mitigation): a comma-separated KEY=VAL
 * list (e.g. "HASH_WORDS=4,HASH_BLOCK_BYTES=64") that is hashed into
 * the cache key alongside the source set. The template kernel
 * (gpu_template.cl) generates IDENTICAL source text for tuples that
 * differ only in #define values; without including those defines in
 * the key, a cache hit could return the wrong-algorithm binary. Pass
 * NULL or "" for non-template builds — the cache key is bit-identical
 * to the pre-B2 key in that case (no eviction of existing cache).
 *
 * On any failure (including disabled cache), falls back to plain
 * clCreateProgramWithSource + clBuildProgram. The caller can then
 * retrieve build log via clGetProgramBuildInfo if program is NULL.
 *
 * Returns the cl_program on success (caller owns), or NULL on failure.
 * `*err_out` (if non-NULL) receives the last cl_int error code seen.
 *
 * Only declared when cl.h has already been included by the caller. */
#if defined(CL_VERSION_1_0) || defined(CL_PLATFORM_H) || defined(__OPENCL_CL_H)
cl_program gpu_kernel_cache_build_program(
    cl_context ctx, cl_device_id dev,
    cl_uint n_sources, const char **sources,
    const char *build_opts,
    cl_int *err_out);

/* Same as above but with an explicit defines_str included in the cache
 * key. The plain build_program() above is implemented as a thin wrapper
 * passing defines_str=NULL so existing call sites keep their current
 * cache keys. New template-instantiation call sites use this entry
 * point with their HASH_WORDS / HASH_BLOCK_BYTES defines string. */
cl_program gpu_kernel_cache_build_program_ex(
    cl_context ctx, cl_device_id dev,
    cl_uint n_sources, const char **sources,
    const char *build_opts,
    const char *defines_str,
    cl_int *err_out);
#endif

#ifdef __cplusplus
}
#endif

#endif /* GPU_KERNEL_CACHE_H */
