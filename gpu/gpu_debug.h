/*
 * $Revision: 1.1 $
 *
 * $Log: gpu_debug.h,v $
 * Revision 1.1  2026/05/17 13:15:44  dlr
 * Initial revision of compile-time GPU debug emit macro gated on MDXFIND_GPU_DEBUG
 *
 *
 * gpu_debug.h -- compile-time gated debug emit macro for GPU host code.
 *
 * Shared by Metal (gpu_metal.m, gpu/gpujob_metal.m) and OpenCL
 * (gpu/gpu_opencl.c, gpu/gpujob_opencl.c).
 *
 * Policy (user directive 2026-05-17): debug stderr emissions in GPU paths
 * MUST NOT live in the shipped binary. Wrapped sites are completely
 * elided in release builds -- both the call AND the format-string literal
 * are removed from the binary (verifiable via `strings`/`objdump`).
 * Debug builds re-enable everything for diagnostic visibility.
 *
 * Distinct from gpu/gpu_fatal.h: GPU_FATAL / MTL_FATAL_NSERR remain
 * UNCONDITIONAL in all builds. Those are runtime-failure reporters that
 * exit(1); they are not "debug chatter" and must always be visible at the
 * first occurrence of a real GPU error. See feedback_external_failures_-
 * are_fatal.md for the discipline.
 *
 * Classification reference (per the 2026-05-17 directive):
 *   PRODUCTION (UNCONDITIONAL plain fprintf):
 *     - GPU_FATAL / MTL_FATAL_NSERR runtime-failure reporters
 *     - One-shot device identity ("Metal: device = Apple M1")
 *     - Capability-gap summary ("STDERR: GPU admission: ... CPU-only")
 *     - End-of-job per-device stats ("Metal GPU[0]: N batches | ...")
 *     - BUG/defensive operator warnings (registry full, packed-without-
 *       rules_engine, etc.) -- low-noise, signal real misconfigurations
 *     - Non-fatal capability-check failures that return -1 (PSO compile
 *       failed, lib load failed, set_salt called before init, etc.)
 *       These already emit per-site diagnostics that the operator needs.
 *
 *   DEBUG (wrap with GPU_DEBUG_FPRINTF):
 *     - Per-(family, variant) "library JIT-compiled" markers
 *     - Per-(family, variant) "PSO ... created lazily" markers
 *     - Per-run init chatter ("library loaded from embedded metallib",
 *       "Metal GPU: N device initialized", "compact table registered",
 *       "overflow preload uploaded", "gpujob thread started",
 *       "salts uploaded", "rule_program uploaded", "mask binding",
 *       "buf_scratch_pool allocated", "first dispatch issued",
 *       "salt-chunked dispatch", "chunked dispatch summary")
 *
 * Build:
 *   default (release): no -DMDXFIND_GPU_DEBUG => GPU_DEBUG_FPRINTF expands
 *     to ((void)0); strings disappear from binary at compile time.
 *   debug:             make CFLAGS_EXTRA="-DMDXFIND_GPU_DEBUG=1"
 *     => GPU_DEBUG_FPRINTF expands to fprintf; matches today's chatty
 *     output (~420 stderr lines for a typical mid-size job).
 */

#ifndef GPU_DEBUG_H
#define GPU_DEBUG_H

#include <stdio.h>

#ifdef MDXFIND_GPU_DEBUG
#define GPU_DEBUG_FPRINTF(...) fprintf(__VA_ARGS__)
#else
/* ((void)0) elides the call AND the variadic format string at -O1+.
 * Confirmed via `strings mdxfind | grep -c "PSO template_phase0"` => 0
 * on release builds. */
#define GPU_DEBUG_FPRINTF(...) ((void)0)
#endif

#endif /* GPU_DEBUG_H */
