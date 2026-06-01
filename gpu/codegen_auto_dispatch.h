/* codegen_auto_dispatch.h -- in-engine capability+perf matrix that
 * selects the GPU rules backend (legacy vs codegen two-engine) per
 * (op, iter, rules, mask, bf, backend_kind).
 *
 * Replaces the user-visible MDXFIND_EXPERIMENT_RULES_CODEGEN_MD5
 * env-flag opt-in shipped 2026-05-29. Per spec
 * project_codegen_auto_dispatch_spec_2026-05-31.md (D1.c / D2.a /
 * D3.a / D4.b / D5.a) -- with the D5.a cell flipped from FATAL to
 * CODEGEN now that Metal kernel-A chunking shipped 2026-05-31.
 *
 * Decision authority:
 *   1. MDXFIND_GPU_BACKEND={auto|legacy|codegen}  developer/test FORCE override
 *   2. hardcoded capability table (consulted at the existing route gates)
 *
 * The OLD env var MDXFIND_EXPERIMENT_RULES_CODEGEN_MD5 is deprecated:
 * if set, emit a one-shot stderr WARNING and ignore the value (matrix
 * decision fires). Per spec §4 D1.c, kept for one minor version then
 * removed entirely.
 *
 * $Revision: 1.2 $
 * $Log: codegen_auto_dispatch.h,v $
 * Revision 1.2  2026/05/31 19:43:11  dlr
 * iter v1.2 (#386): admit JOB_SHA1 (e8) + JOB_SHA256 (e10) hex-feedback siblings of SHA1RAW/SHA256RAW into the codegen route gate at ANY iter; closes user item #5 from 2026-05-31 not-working list. Per #379 v1.1 widen option (a). Verified op-ids in job_types.h. CPU paths mdxfind.c:28666-28679 (SHA1) + :29088-29097 (SHA256) confirm hex-feedback (prmd5 between iters), distinct from binary-feedback RAW siblings (:27994, :29077). codegen/hx_emit_primitives.c adds 2 rows to unsalted_job_table (auto-propagates digest_bytes). gpu/gpujob_*.c widens admit + adds iter-aware full-digest recompute that walks N iters of mysha1/mysha256+prmd5 chain mirroring CPU loop. gpu_metal.m widens _is_exp_md5 admission gate. gpu/codegen_auto_dispatch.c+h add cells 9b/9c/9d for SHA1/SHA256 + 6 matrix-dump probes + docstring update. Apple Metal template_iterate empirically BROKEN for SHA1/SHA256 iter>1 (returns 0 cracks; same root cause as MD5: metal_template.metal:684 Phase 1 intentionally not called) — auto-dispatcher Metal SHA1/SHA256 iter>1 cell picks CODEGEN (flagship class). Latent iter-aware-recompute bug found+fixed during validation (was iter=1-only; broke immediately for new ops at i=2). 24-cell new-op parity matrix + 38-cell regression matrix ALL PASS byte-exact vs CPU oracle on dev1 M1 + fpga Pascal + hpi7 Maxwell. Cross-host CPU-oracle md5s match. Advisory dedup verified.
 *
 * Revision 1.1  2026/05/31 18:51:30  dlr
 * codegen auto-dispatcher: retire MDXFIND_EXPERIMENT_RULES_CODEGEN_MD5 env-flag opt-in; replace with in-engine capability-perf matrix that selects backend per (op, iter, rules, mask, bf, backend_kind). Per spec project_codegen_auto_dispatch_spec_2026-05-31 (D1.c retire + D2.a per-JOB dedup advisory + D3.a hardcoded table + D4.b direct ship + D5.a Metal-chunking-now-CODEGEN cell). New files gpu/codegen_auto_dispatch.h + gpu/codegen_auto_dispatch.c (~465 LOC); modified 6 existing files (route gates + accessor deletions + dispatcher integration). Two correctness-critical cells: (1) Apple Metal x MD5/MD4 x iter>1 x rules -> CODEGEN (legacy template_phase0 returns 0 cracks; flagship user-facing fix; works WITHOUT any env flag now); (2) OpenCL x MD5/MD4 x any-iter x rules -> LEGACY (1.39-2.28x faster preserved). New env MDXFIND_GPU_BACKEND=auto|legacy|codegen developer FORCE override; MDXFIND_GPU_BACKEND_QUIET=1 suppresses advisories. Deprecation shim for OLD flag prints WARNING + ignores. 7 gates PASS (A 4-host parity grid byte-exact vs CPU oracle, B Apple flagship no-env, C OpenCL perf preserved, D FORCE legacy on broken cell returns 0 confirming override + correctness of CODEGEN auto-pick, D-prime FORCE codegen on faster-legacy cell correct via slower path, E deprecation shim WARNING and ignore, F auto + Metal chunking composition 99K rules iter=5 correct, G advisory dedup). Per-host Makefiles edited IN PLACE on .205 + dev1 per [[feedback_makefile_edit_in_place_allowed]] to link codegen_auto_dispatch.o into mdxfind.
 *
 */
#ifndef CODEGEN_AUTO_DISPATCH_H
#define CODEGEN_AUTO_DISPATCH_H

#include <stdio.h>
#include <stdint.h>

/* Backend kinds: which engine the caller belongs to. The capability
 * matrix differs across backends -- e.g., OpenCL legacy md5_rules_phase0
 * is correct at iter>1; Metal legacy template_phase0 returns 0 cracks
 * at iter>1. */
#define GPU_BACKEND_KIND_OPENCL 1
#define GPU_BACKEND_KIND_METAL  2

/* Dispatch decision: which backend should serve this (op, config) cell.
 *   GPU_BACKEND_LEGACY  -- hand-tuned single-kernel path (md5_rules_phase0
 *                          on OpenCL; template_phase0 / template_iterate
 *                          on Metal). Default for cells where it is
 *                          correct AND not measurably slower.
 *   GPU_BACKEND_CODEGEN -- A1 -> codegen kernel B two-engine pipeline.
 *                          Default for cells where legacy is broken or
 *                          absent (Metal iter>1 with rules; e347; family).
 *   GPU_BACKEND_FATAL   -- neither backend can correctly serve this cell;
 *                          caller should exit(1) with explanatory text
 *                          (per feedback_external_failures_are_fatal.md).
 *                          Reserved for future cells; the post-Metal-
 *                          chunking matrix has no FATAL cells.
 */
typedef enum {
    GPU_BACKEND_LEGACY  = 0,
    GPU_BACKEND_CODEGEN = 1,
    GPU_BACKEND_FATAL   = 2
} gpu_backend_pick_t;

/* codegen_auto_dispatch_pick -- consult capability matrix + FORCE
 * override and return the backend pick for the (op, config) cell.
 *
 * Inputs:
 *   op            -- JOB_MD5 / JOB_MD4 / JOB_SHA1 / JOB_SHA256 /
 *                    JOB_SHA1RAW / JOB_SHA256RAW / JOB_MD5MD5SALT /
 *                    family op  (Iter v1.2 2026-05-31 added SHA1/SHA256
 *                    hex-feedback siblings of SHA1RAW/SHA256RAW.)
 *   max_iter      -- value of Maxiter (1 = single hash, N>1 = iterated)
 *   rule_count    -- value of gpu_rule_count (0 = no rules)
 *   mask_total    -- value of MaskTotal (0 = no mask)
 *   bf_chunk      -- nonzero when brute-force engine is active
 *   backend_kind  -- GPU_BACKEND_KIND_OPENCL or GPU_BACKEND_KIND_METAL
 *
 * Side effects:
 *   First call per process: reads MDXFIND_GPU_BACKEND env var (cached).
 *   First call per (op, pick) tuple: emits one-line stderr advisory
 *   unless MDXFIND_GPU_BACKEND_QUIET=1 suppresses (per spec D2.a).
 *
 * Returns the picked backend. Callers route their dispatch accordingly:
 *   GPU_BACKEND_LEGACY  -- skip the _exp_md5_route branch (fall through
 *                          to the legacy engine code below)
 *   GPU_BACKEND_CODEGEN -- set the _exp_md5_route flag (route through
 *                          the proto / codegen kernel B path)
 *   GPU_BACKEND_FATAL   -- print FATAL line + exit(1). No silent fallback.
 */
gpu_backend_pick_t codegen_auto_dispatch_pick(
    int op, int max_iter, int rule_count,
    unsigned long long mask_total, unsigned int bf_chunk,
    int backend_kind);

/* codegen_auto_dispatch_dump_matrix -- structured stderr dump of the
 * capability matrix for -V (verbose) output. Lists every (backend, op,
 * iter, rules) cell with the picked backend + reason. Per spec §3.
 *
 * Pure read; safe to call from -V handler. */
void codegen_auto_dispatch_dump_matrix(FILE *out);

/* codegen_auto_dispatch_deprecation_check -- one-shot check for the
 * OLD env var MDXFIND_EXPERIMENT_RULES_CODEGEN_MD5. If set, print
 * a stderr WARNING line and continue (the OLD flag is IGNORED, not
 * honored -- the matrix decision is what fires). Per spec §4.
 *
 * Safe to call repeatedly; advisory printed once per process. Called
 * automatically from codegen_auto_dispatch_pick() on first invocation. */
void codegen_auto_dispatch_deprecation_check(void);

#endif /* CODEGEN_AUTO_DISPATCH_H */
