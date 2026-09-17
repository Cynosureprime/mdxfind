/*
 * $Revision: 1.1 $
 * $Log: gpu_u32_host.c,v $
 * Revision 1.1  2026/09/14 19:30:05  dlr
 * Initial revision.
 *
 *
 */
/* gpu_u32_host.c -- the one translation unit that carries the UTF-32 host
 * module's bodies.
 *
 * Two lines of code and a page of reason.
 *
 * The module used to be header-only and all-`static`, with a single exported
 * wrapper in gpu_opencl.c.  Co-locating that wrapper with
 * gpu_opencl_u32_rules_lazy was deliberate: the pre-filter WRITES the build
 * cache and the lazy READS it, and `static` gives each including TU its own
 * copy, so putting them in the same TU made it one cache BY CONSTRUCTION
 * rather than by coincidence.
 *
 * That reasoning was right for one backend and wrong for two.  mdxfind.c calls
 * gpu_u32_membership_prefilter from its METAL rule-pack block as well as its
 * OpenCL one, and gpu_opencl.c is not compiled on a METAL_GPU build -- so on
 * Metal that call was an undefined symbol at link, and nothing had linked a
 * Metal-only build since it was added.  A latent break, found by reading the
 * Makefiles rather than by a failure.
 *
 * A backend-neutral TU is strictly better than the co-location it replaces:
 * there is exactly one TU that can hold the cache, so the one-instance
 * property is unconditional, and both backends' host files now read the same
 * cache instead of that being true only because there happened to be one
 * backend.
 *
 * This file must be in the link line of EVERY configuration -- OpenCL, Metal,
 * and a CPU-only build that still compiles mdxfind.c's GPU-gated blocks.  It
 * pulls in nothing backend-specific: ../ruleproc32.h, ../classify_utf8.h and
 * ../gpujob.h, all of which are already prerequisites.
 */
#define GPU_U32_HOST_IMPLEMENTATION 1
#include "gpu_u32_host.h"
