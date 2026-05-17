/*
 * gpu_fatal.h -- host-side fatal-error macros for GPU runtime failures.
 *
 * Replaces the previous in-line definition of GPU_FATAL in gpu_opencl.h.
 * Used by BOTH gpu/gpu_opencl.c (OpenCL host code) and gpu_metal.m / gpu/
 * gpujob_metal.m (Metal host code). The Metal-only variant MTL_FATAL_NSERR
 * is gated on __OBJC__ so the header is safe to include from plain C.
 *
 * Phase D5a 2026-05-16 (Task #281): user-enforced policy that ANY external
 * runtime failure (PSO/library create, dispatch error, buffer alloc,
 * enqueue error, etc.) must exit(1) with a clear diagnostic carrying
 * file:line + op identifier + error string. NEVER silent NULL return,
 * NEVER silent CPU fallback, NEVER drop data. See
 * feedback_external_failures_are_fatal.md.
 *
 * Rationale (from gpu_opencl.h Phase A discipline, mmt run #77 2026-05-01,
 * extended to Metal 2026-05-16): a CL_OUT_OF_RESOURCES or PSO-create
 * failure that gets logged-and-ignored produces wrong cracks output
 * silently. Operator sees `head -1 cracks.txt` looking fine; actual run
 * dropped 30% of words at first failing dispatch. The same workload
 * re-run clean produces bit-exact cracks (canonical truth). Fail-fast
 * here exposes the failure at the first occurrence with enough context
 * to investigate. _Exit (NOT exit) bypasses atexit handlers and stdio
 * buffered-output flushes that could mask the failure or paper over
 * partial state. stdout is flushed first so any cracks emitted before
 * the failure are preserved on the consumer's side.
 *
 * Distinct from "graceful capability check at init":
 *   - INIT: gpu_opencl_available / gpu_metal_available returning 0/1 --
 *     a deliberate query, not a failure.
 *   - INIT: gpu_metal_compile_families eager-PSO prune-on-failure --
 *     deliberate admission selection per architect Task #281 §6 Option B.
 *   - RUNTIME: any error code from API call MUST be fatal.
 * Capability checks happen BEFORE the operation is committed; failures
 * happen AFTER. Capability returning "not supported" -> fine, route
 * differently. Operation attempt that fails partway -> fatal.
 */

#ifndef GPU_FATAL_H
#define GPU_FATAL_H

#include <stdio.h>
#include <stdlib.h>   /* _Exit */

/* Generic fail-fast macro for GPU errors that do not carry an NSError.
 * Used by OpenCL host code (cl_int errcode-ret), the Metal newBuffer
 * sites (which take no NSError out-param), and any other "I have a
 * descriptive string but no Cocoa error object" call site.
 *
 * Sites that legitimately retry (probe paths, init-time device skip,
 * device tuning) MUST NOT use this macro -- they handle CL/Metal errors
 * via graceful early-return and document why retry is correct. Any new
 * caller of GPU_FATAL must be in a post-init production path where
 * an error genuinely indicates a corrupted GPU state.
 *
 * Async error callback usage (OpenCL): the OpenCL spec does not permit
 * longjmp-out-of-driver-thread, but _Exit is async-signal-safe and
 * terminates the process without stack unwinding -- the driver's
 * internal state never gets a chance to corrupt anything else. */
#define GPU_FATAL(fmt, ...) do {                                      \
    fflush(stdout);                                                   \
    fprintf(stderr, "FATAL: GPU error: " fmt "\n", ##__VA_ARGS__);    \
    fprintf(stderr, "FATAL: at %s:%d\n", __FILE__, __LINE__);         \
    fflush(stderr);                                                   \
    _Exit(1);                                                         \
} while (0)

/* Metal-specific fail-fast macro that knows how to unwrap an NSError
 * (Cocoa error object). Only defined in Objective-C translation units;
 * gated on __OBJC__ so plain-C consumers of this header don't see the
 * Objective-C syntax. Mirrors GPU_FATAL's flush+exit_immediate pattern,
 * with the addition of `[err localizedDescription]` for the
 * driver-supplied diagnostic. */
#ifdef __OBJC__
#define MTL_FATAL_NSERR(nserr, fmt, ...) do {                                \
    fflush(stdout);                                                          \
    fprintf(stderr, "FATAL: GPU error: " fmt " :: %s\n",                     \
        ##__VA_ARGS__,                                                       \
        (nserr) ? [[(nserr) localizedDescription] UTF8String] : "(no error)"); \
    fprintf(stderr, "FATAL: at %s:%d\n", __FILE__, __LINE__);                \
    fflush(stderr);                                                          \
    _Exit(1);                                                                \
} while (0)
#endif

#endif /* GPU_FATAL_H */
