/* codegen_auto_dispatch.c -- in-engine capability+perf matrix that
 * selects the GPU rules backend (legacy hand-tuned vs codegen two-engine
 * pipeline) per (op, iter, rules, mask, bf, backend_kind).
 *
 * Replaces the user-visible MDXFIND_EXPERIMENT_RULES_CODEGEN_MD5
 * env-flag opt-in shipped 2026-05-29. Per spec
 * project_codegen_auto_dispatch_spec_2026-05-31.md (D1.c / D2.a /
 * D3.a / D4.b / D5.a). The (Metal, e1, iter>1, large-rules) cell is
 * CODEGEN (not FATAL) because Metal kernel-A chunking shipped
 * 2026-05-31 (gpu_metal.m 1.128; metal_kernel_a_rules.metal 1.6).
 *
 * Decision authority (per spec §2):
 *   1. MDXFIND_GPU_BACKEND={auto|legacy|codegen}  developer/test FORCE
 *   2. hardcoded capability table (consulted at route gates)
 *
 * Advisory output (per spec §3): one stderr line per JOB-change, deduped
 * on (op, backend_pick). MDXFIND_GPU_BACKEND_QUIET=1 suppresses.
 *
 * Deprecation shim (per spec §4 R3): MDXFIND_EXPERIMENT_RULES_CODEGEN_MD5
 * triggers a one-shot stderr WARNING; flag value is IGNORED (matrix
 * decision fires). One-version retention; remove in v1.50x.
 *
 * $Revision: 1.2 $
 * $Log: codegen_auto_dispatch.c,v $
 * Revision 1.2  2026/05/31 19:43:11  dlr
 * iter v1.2 (#386): admit JOB_SHA1 (e8) + JOB_SHA256 (e10) hex-feedback siblings of SHA1RAW/SHA256RAW into the codegen route gate at ANY iter; closes user item #5 from 2026-05-31 not-working list. Per #379 v1.1 widen option (a). Verified op-ids in job_types.h. CPU paths mdxfind.c:28666-28679 (SHA1) + :29088-29097 (SHA256) confirm hex-feedback (prmd5 between iters), distinct from binary-feedback RAW siblings (:27994, :29077). codegen/hx_emit_primitives.c adds 2 rows to unsalted_job_table (auto-propagates digest_bytes). gpu/gpujob_*.c widens admit + adds iter-aware full-digest recompute that walks N iters of mysha1/mysha256+prmd5 chain mirroring CPU loop. gpu_metal.m widens _is_exp_md5 admission gate. gpu/codegen_auto_dispatch.c+h add cells 9b/9c/9d for SHA1/SHA256 + 6 matrix-dump probes + docstring update. Apple Metal template_iterate empirically BROKEN for SHA1/SHA256 iter>1 (returns 0 cracks; same root cause as MD5: metal_template.metal:684 Phase 1 intentionally not called) — auto-dispatcher Metal SHA1/SHA256 iter>1 cell picks CODEGEN (flagship class). Latent iter-aware-recompute bug found+fixed during validation (was iter=1-only; broke immediately for new ops at i=2). 24-cell new-op parity matrix + 38-cell regression matrix ALL PASS byte-exact vs CPU oracle on dev1 M1 + fpga Pascal + hpi7 Maxwell. Cross-host CPU-oracle md5s match. Advisory dedup verified.
 *
 * Revision 1.1  2026/05/31 19:02:25  dlr
 * codegen auto-dispatcher: retire MDXFIND_EXPERIMENT_RULES_CODEGEN_MD5 env-flag opt-in; replace with in-engine capability-perf matrix that selects backend per (op, iter, rules, mask, bf, backend_kind). Per spec project_codegen_auto_dispatch_spec_2026-05-31 (D1.c retire + D2.a per-JOB dedup advisory + D3.a hardcoded table + D4.b direct ship + D5.a Metal-chunking-now-CODEGEN cell). 7 gates PASS (4-host parity, Apple flagship no-env, OpenCL perf preserved, FORCE override semantics, deprecation shim WARNING, auto + Metal chunking composition, advisory dedup). Per-host Makefiles edited IN PLACE on .205 + dev1. See full report at project_codegen_auto_dispatch_2026-05-31_BUILD.md.
 *
 */

#include "codegen_auto_dispatch.h"
#include "job_types.h"
#include "gpu_codegen_eligible.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ---- internal state -------------------------------------------------- */

/* FORCE-override state, decoded from MDXFIND_GPU_BACKEND env var on
 * first call. -1 = uninspected; 0 = auto (default); 1 = force LEGACY;
 * 2 = force CODEGEN. Cached for the life of the process. */
static int _force_mode = -1;  /* -1 uninspected, 0 auto, 1 legacy, 2 codegen */

/* Advisory dedup: one stderr line per (op, pick) tuple. Bounded table;
 * a single mdxfind run touches O(10) distinct ops. Linear scan is fine. */
#define ADV_DEDUP_MAX 64
typedef struct {
    int op;
    int pick;
    int backend_kind;
} adv_dedup_entry_t;
static adv_dedup_entry_t _adv_dedup[ADV_DEDUP_MAX];
static int _adv_dedup_n = 0;

/* Quiet flag: MDXFIND_GPU_BACKEND_QUIET=1 suppresses advisories. */
static int _quiet_cached = -1;

/* OLD env var deprecation shim state: one-shot WARNING. */
static int _deprecation_warned = 0;

/* ---- helpers --------------------------------------------------------- */

/* Lazy-init the FORCE-override mode from MDXFIND_GPU_BACKEND. Per spec
 * §2: "auto" (or unrecognized) falls through to the capability table;
 * "legacy" forces LEGACY; "codegen" forces CODEGEN. Unrecognized values
 * print a stderr NOTICE and fall through to auto. */
static void _force_init(void)
{
    if (_force_mode != -1) return;
    const char *e = getenv("MDXFIND_GPU_BACKEND");
    if (e == NULL || e[0] == '\0' || !strcmp(e, "auto")) {
        _force_mode = 0;
        return;
    }
    if (!strcmp(e, "legacy")) {
        _force_mode = 1;
        fprintf(stderr,
            "mdxfind: GPU dispatcher FORCE-override active: "
            "MDXFIND_GPU_BACKEND=legacy. Capability table consultation "
            "bypassed; all eligible cells route through the legacy hand-"
            "tuned engine. Diagnostic/test mode -- may produce 0 cracks "
            "on cells where legacy is broken (e.g., Apple Metal -m e1 "
            "-i N>1 with rules).\n");
        return;
    }
    if (!strcmp(e, "codegen")) {
        _force_mode = 2;
        fprintf(stderr,
            "mdxfind: GPU dispatcher FORCE-override active: "
            "MDXFIND_GPU_BACKEND=codegen. Capability table consultation "
            "bypassed; all eligible cells route through the codegen two-"
            "engine pipeline. Diagnostic/test mode -- may FATAL or run "
            "slower than auto on cells where codegen is unsupported "
            "(mask, BF) or measurably slower (OpenCL JOB_MD5).\n");
        return;
    }
    fprintf(stderr,
        "mdxfind: NOTICE -- MDXFIND_GPU_BACKEND=%s unrecognized "
        "(expected auto|legacy|codegen); falling back to auto.\n", e);
    _force_mode = 0;
}

/* Lazy-init the QUIET flag from MDXFIND_GPU_BACKEND_QUIET. */
static int _quiet(void)
{
    if (_quiet_cached != -1) return _quiet_cached;
    const char *e = getenv("MDXFIND_GPU_BACKEND_QUIET");
    _quiet_cached = (e && e[0] == '1' && e[1] == '\0') ? 1 : 0;
    return _quiet_cached;
}

void codegen_auto_dispatch_deprecation_check(void)
{
    if (_deprecation_warned) return;
    const char *e = getenv("MDXFIND_EXPERIMENT_RULES_CODEGEN_MD5");
    if (e == NULL || e[0] == '\0') {
        _deprecation_warned = 1;
        return;
    }
    fprintf(stderr,
        "mdxfind: WARNING -- MDXFIND_EXPERIMENT_RULES_CODEGEN_MD5 is "
        "deprecated and IGNORED. The dispatcher now auto-selects the "
        "codegen path when needed (Apple Metal -m e1 -i N>1 with rules; "
        "JOB_MD5MD5SALT / MAKE_MD5PASS family). Set "
        "MDXFIND_GPU_BACKEND=codegen to force codegen for diagnostics "
        "(symmetric force-legacy is also available). This shim is "
        "scheduled for removal in the next minor version.\n");
    _deprecation_warned = 1;
}

/* Lookup reason string for a (backend_kind, op, iter, rule_count, pick)
 * cell. Inline in the table-style switch below; centralized so the same
 * text appears in advisories and -V dumps. Caller passes a buffer; the
 * function writes a short null-terminated description. */
static const char *_reason_text(
    int backend_kind, int op, int max_iter, int rule_count,
    unsigned long long mask_total, unsigned int bf_chunk,
    gpu_backend_pick_t pick)
{
    (void)rule_count;
    if (mask_total > 0 && pick == GPU_BACKEND_LEGACY) {
        return "mask active: codegen mask path is v2/v3 future work";
    }
    if (bf_chunk > 0 && pick == GPU_BACKEND_LEGACY) {
        return "brute-force active: codegen BF path is v2/v3 future work";
    }
    if (op == JOB_MD5MD5SALT && pick == GPU_BACKEND_CODEGEN) {
        return "legacy kernel B retired Phase 4a.2b: codegen is production";
    }
    if (gpu_codegen_kernelb_family_md5pass_eligible(op)
        && pick == GPU_BACKEND_CODEGEN) {
        return "MAKE_MD5PASS family: no legacy hand-tuned kernel; codegen is production";
    }
    if (backend_kind == GPU_BACKEND_KIND_METAL
        && (op == JOB_MD5 || op == JOB_MD4) && max_iter > 1
        && pick == GPU_BACKEND_CODEGEN) {
        return "Apple Metal legacy template iter>1 broken: codegen is the correct path";
    }
    if (backend_kind == GPU_BACKEND_KIND_OPENCL
        && (op == JOB_MD5 || op == JOB_MD4)
        && pick == GPU_BACKEND_LEGACY) {
        return "OpenCL legacy md5_rules_phase0 measured 1.39-2.28x faster than codegen (per #375/#377)";
    }
    if (backend_kind == GPU_BACKEND_KIND_METAL
        && (op == JOB_MD5 || op == JOB_MD4) && max_iter == 1
        && pick == GPU_BACKEND_LEGACY) {
        return "Apple Metal legacy template_phase0 iter=1 ~= codegen perf; legacy chosen for parity/stability";
    }
    if ((op == JOB_SHA1RAW || op == JOB_SHA256RAW)
        && max_iter > 1 && pick == GPU_BACKEND_LEGACY) {
        return "SHA*RAW iter>1: codegen uses hex feedback; CPU oracle uses binary feedback (divergence)";
    }
    if (backend_kind == GPU_BACKEND_KIND_METAL
        && (op == JOB_SHA1 || op == JOB_SHA256) && max_iter > 1
        && pick == GPU_BACKEND_CODEGEN) {
        return "Apple Metal legacy template_iterate intentionally not called (metal_template.metal:684); codegen is the correct path";
    }
    if (backend_kind == GPU_BACKEND_KIND_METAL
        && (op == JOB_SHA1 || op == JOB_SHA256) && max_iter == 1
        && pick == GPU_BACKEND_LEGACY) {
        return "Apple Metal legacy template_phase0 iter=1 correct; legacy chosen for parity/stability";
    }
    if (backend_kind == GPU_BACKEND_KIND_OPENCL
        && (op == JOB_SHA1 || op == JOB_SHA256)
        && pick == GPU_BACKEND_LEGACY) {
        return "OpenCL legacy template path handles hex-feedback iter loop correctly; legacy chosen for parity with JOB_MD5 default";
    }
    if (rule_count == 0 && pick == GPU_BACKEND_LEGACY) {
        return "rules unset: codegen route requires rules>0";
    }
    if (pick == GPU_BACKEND_LEGACY) {
        return "legacy: default for unmatched cells";
    }
    if (pick == GPU_BACKEND_CODEGEN) {
        return "codegen: matrix cell selected";
    }
    return "fatal: capability gap (see capability matrix)";
}

/* Maybe-emit an advisory for this (op, pick, backend_kind) tuple. Per
 * spec §3: one line per JOB-change, deduped. Riding the existing
 * stderr channel that already carries "Metal GPU[0]: first dispatch
 * issued" / "OpenCL: EXPERIMENT ..." markers (per
 * feedback_check_existing_traces_first.md). */
static void _maybe_emit_advisory(
    int backend_kind, int op, int max_iter, int rule_count,
    unsigned long long mask_total, unsigned int bf_chunk,
    gpu_backend_pick_t pick)
{
    if (_quiet()) return;
    /* Dedup scan. */
    for (int i = 0; i < _adv_dedup_n; i++) {
        if (_adv_dedup[i].op == op
            && _adv_dedup[i].pick == (int)pick
            && _adv_dedup[i].backend_kind == backend_kind) {
            return;
        }
    }
    if (_adv_dedup_n >= ADV_DEDUP_MAX) return;  /* bounded */
    _adv_dedup[_adv_dedup_n].op = op;
    _adv_dedup[_adv_dedup_n].pick = (int)pick;
    _adv_dedup[_adv_dedup_n].backend_kind = backend_kind;
    _adv_dedup_n++;
    const char *backend_name =
        (backend_kind == GPU_BACKEND_KIND_OPENCL) ? "opencl" :
        (backend_kind == GPU_BACKEND_KIND_METAL)  ? "metal"  : "unknown";
    const char *pick_name =
        (pick == GPU_BACKEND_LEGACY)  ? "legacy"  :
        (pick == GPU_BACKEND_CODEGEN) ? "codegen" :
        (pick == GPU_BACKEND_FATAL)   ? "fatal"   : "unknown";
    const char *reason = _reason_text(
        backend_kind, op, max_iter, rule_count, mask_total, bf_chunk, pick);
    fprintf(stderr,
        "mdxfind: GPU backend=%s op=%d iter=%d rules=%d "
        "mask_total=%llu bf=%u pick=%s reason=\"%s\"\n",
        backend_name, op, max_iter, rule_count,
        mask_total, bf_chunk, pick_name, reason);
}

/* ---- capability matrix ----------------------------------------------- */

/* The capability matrix is encoded directly as decision logic (per spec
 * D3.a: hardcoded static C; no config file per feedback_no_builtin_-
 * config_paths). Each branch carries an evidence-basis comment.
 *
 * Returns the unforced (auto) pick for (backend_kind, op, max_iter,
 * rule_count, mask_total, bf_chunk). FORCE-override is applied by the
 * caller before invoking this function. */
static gpu_backend_pick_t _auto_pick(
    int backend_kind, int op, int max_iter, int rule_count,
    unsigned long long mask_total, unsigned int bf_chunk)
{
    /* Cell 1: JOB_MD5MD5SALT (e347).
     * Legacy hand-tuned kernel B was retired Phase 4a.2b (2026-05-22).
     * Codegen is the only path on both backends. (Wired-by-construction:
     * gpu_opencl_kernelb_dispatch_proto / gpu_metal_kernelb_dispatch_-
     * proto have no legacy branch for this op.) */
    if (op == JOB_MD5MD5SALT) {
        return GPU_BACKEND_CODEGEN;
    }

    /* Cell 2: MAKE_MD5PASS family (e122/e159/e161/e163/e165/e167/e169
     * + Tier 1/2 widens). No legacy hand-tuned kernel; codegen is
     * production on both backends. */
    if (gpu_codegen_kernelb_family_md5pass_eligible(op)) {
        return GPU_BACKEND_CODEGEN;
    }

    /* Cell 3: mask-active or BF-active cells.
     * Codegen v1 scope explicitly excludes mask + BF (per
     * project_codegen_iteration_v1_spec_2026-05-31 §5). Legacy serves
     * these correctly on both backends. */
    if (mask_total > 0 || bf_chunk > 0) {
        return GPU_BACKEND_LEGACY;
    }

    /* Cell 4: rule_count == 0 (no-rules slab / template path).
     * Codegen route gate requires rules>0 (host-side admission asserts
     * the rule program is uploaded). Legacy handles no-rules via the
     * slab / template_phase0 path. */
    if (rule_count <= 0) {
        return GPU_BACKEND_LEGACY;
    }

    /* Cell 5: Apple Metal flagship correctness fix (MD5 + MD4).
     * Legacy template_phase0 iter==1 is correct; template_iterate iter>1
     * is BROKEN for ALL primitives sharing the template path -- returns
     * 0 cracks (#377 Gate D + spec §8 file-header comment: "Phase 2
     * promised to re-add"). Codegen iter-v1 produces correct cracks at
     * iter ∈ {1,2,5,10,100} (#377). The (Metal, JOB_MD5, iter>1) cell is
     * the flagship; the (Metal, JOB_MD4, iter>1) cell shares the same
     * broken-legacy fate per the user-direction brief 2026-05-31:
     *   "Apple Metal x MD5/MD4 x iter>1 x rules -> CODEGEN (legacy broken)"
     *
     * MD4 codegen uses HEX feedback matching CPU JOB_MD4 (which also
     * uses HEX feedback per mdxfind.c:27048-27057). Iter>1 byte-exact
     * vs CPU oracle verified by #379 cross-arch matrix (md4 i=2/5/10
     * sorted-md5 codegen-on-Pascal == codegen-on-Maxwell == codegen-
     * on-Metal-M1).
     *
     * Post-Metal-chunking-ship (gpu_metal.m 1.128, 2026-05-31): the
     * spec's FATAL escape at large rule counts is GONE -- chunked
     * dispatcher handles 99K rules cleanly (Gate A 99,074 rules at
     * dev1 M1 verified). These cells are CODEGEN unconditionally. */
    if (backend_kind == GPU_BACKEND_KIND_METAL
        && (op == JOB_MD5 || op == JOB_MD4)
        && max_iter > 1) {
        return GPU_BACKEND_CODEGEN;
    }

    /* Cell 6: Apple Metal JOB_MD5 / JOB_MD4 + iter==1 + rules.
     * Legacy template_phase0 is correct (#377 Gate D + small-fixture
     * 1.58s vs codegen 1.55s ~ 1.02x). Prefer legacy for parity with
     * the OpenCL JOB_MD5 default + battle-test stability. */
    if (backend_kind == GPU_BACKEND_KIND_METAL
        && (op == JOB_MD5 || op == JOB_MD4)
        && max_iter == 1) {
        return GPU_BACKEND_LEGACY;
    }

    /* Cell 7: OpenCL JOB_MD5 / JOB_MD4 + rules + any-iter.
     * Legacy md5_rules_phase0 is 1.39-2.28x FASTER than codegen at
     * iter=1 (#375) AND iter=10 (#377 Gate C). User-visible change
     * across this spec is ZERO for this cell (today's env-unset
     * default is preserved). Per user-direction brief 2026-05-31:
     *   "OpenCL x MD5/MD4 x any-iter x rules -> LEGACY (faster)"
     */
    if (backend_kind == GPU_BACKEND_KIND_OPENCL
        && (op == JOB_MD5 || op == JOB_MD4)) {
        return GPU_BACKEND_LEGACY;
    }

    /* Cell 9: JOB_SHA1RAW / JOB_SHA256RAW.
     * iter==1: legacy works; codegen also works; either picks LEGACY
     * for parity with OpenCL JOB_MD5 default.
     * iter>1: codegen uses HEX feedback; CPU oracle uses BINARY feedback
     * (mdxfind.c:27994 / 29077). At iter>1 codegen would silently
     * diverge from CPU. Legacy template_iterate handles binary feedback
     * correctly (per #377 v1.1 widen + B5 sub-batch 6 Tier A). */
    if (op == JOB_SHA1RAW || op == JOB_SHA256RAW) {
        return GPU_BACKEND_LEGACY;
    }

    /* Cell 9b: Apple Metal flagship correctness fix (SHA1 + SHA256
     * hex-feedback siblings of SHA1RAW/SHA256RAW). Per #379 v1.1
     * widen option a (2026-05-31):
     *
     * Metal legacy template_phase0 at iter>1 has the SAME
     * "template_iterate() intentionally NOT called" gap as MD5/MD4
     * (metal_template.metal:684 "Phase 1: template_iterate() is
     * intentionally NOT called. Phase 2 re-adds it inside
     * `if (iter < max_iter) { ... }`"). The gap is in template_-
     * phase0 itself, not the per-primitive cores -- so it affects
     * ALL primitives sharing the Metal template path including
     * SHA1 and SHA256. Empirically: Metal -m e8 -i 10 with rules
     * via legacy returns 0 cracks (same as MD5 iter>1).
     *
     * Codegen iter-v1 (codegen/hx_emit_opencl.c:4195+ + Metal twin)
     * emits per-primitive hex-feedback helpers for SHA1 (40-char
     * hex + 0x80 + len=320 -> single block) and SHA256 (64-char hex
     * + 0x80 + len=512 -> two blocks). Matches CPU JOB_SHA1
     * (mdxfind.c:28666-28679 using prmd5 lower-case hex) and CPU
     * JOB_SHA256 (mdxfind.c:29088-29097).
     *
     * Therefore: Metal × (SHA1, SHA256) × iter>1 × rules -> CODEGEN
     * (flagship class). */
    if (backend_kind == GPU_BACKEND_KIND_METAL
        && (op == JOB_SHA1 || op == JOB_SHA256)
        && max_iter > 1) {
        return GPU_BACKEND_CODEGEN;
    }

    /* Cell 9c: Apple Metal SHA1 / SHA256 + iter==1 + rules.
     * Legacy template_phase0 is correct at iter==1 (the iter loop body
     * fires once before the broken template_iterate gap matters). Prefer
     * LEGACY for parity with the (Metal, MD5, iter==1) decision (Cell 6)
     * and battle-test stability. */
    if (backend_kind == GPU_BACKEND_KIND_METAL
        && (op == JOB_SHA1 || op == JOB_SHA256)
        && max_iter == 1) {
        return GPU_BACKEND_LEGACY;
    }

    /* Cell 9d: OpenCL SHA1 / SHA256 + rules + any-iter.
     * Legacy template_phase0 + template_iterate is correct AND tuned
     * (gpu_template.cl:666-676 calls template_iterate every iter<max_iter
     * step; gpu_sha1_core.cl / gpu_sha256_core.cl supply hex-encoded
     * re-hash). Per [[feedback_codegen_two_engine_architecture]] "close
     * the gap, not surpass it" -- favor LEGACY for parity with the
     * (OpenCL, MD5, any-iter) Cell 7 decision. Mirror of Cell 7 for the
     * SHA1/SHA256 hex-feedback ops. */
    if (backend_kind == GPU_BACKEND_KIND_OPENCL
        && (op == JOB_SHA1 || op == JOB_SHA256)) {
        return GPU_BACKEND_LEGACY;
    }

    /* Cell 10: catch-all / unmatched op.
     * Codegen route gate (gpujob_*.c:1209+) requires op in the eligible
     * set; out-of-set ops fall through. Legacy serves everything else
     * (or is itself a no-op that downstream code handles). */
    return GPU_BACKEND_LEGACY;
}

/* ---- public entry point ---------------------------------------------- */

gpu_backend_pick_t codegen_auto_dispatch_pick(
    int op, int max_iter, int rule_count,
    unsigned long long mask_total, unsigned int bf_chunk,
    int backend_kind)
{
    /* Deprecation shim: one-shot WARNING if the OLD env var is set. */
    codegen_auto_dispatch_deprecation_check();
    /* FORCE-override init. */
    _force_init();

    /* Compute the unforced auto pick first; we still emit an advisory
     * tagged with the FINAL pick (force or auto). */
    gpu_backend_pick_t auto_pick = _auto_pick(
        backend_kind, op, max_iter, rule_count, mask_total, bf_chunk);

    gpu_backend_pick_t final_pick = auto_pick;
    if (_force_mode == 1) final_pick = GPU_BACKEND_LEGACY;
    else if (_force_mode == 2) final_pick = GPU_BACKEND_CODEGEN;

    _maybe_emit_advisory(
        backend_kind, op, max_iter, rule_count, mask_total, bf_chunk,
        final_pick);

    return final_pick;
}

/* ---- -V structured matrix dump --------------------------------------- */

void codegen_auto_dispatch_dump_matrix(FILE *out)
{
    if (out == NULL) out = stderr;
    fprintf(out, "GPU capability matrix (codegen vs legacy auto-dispatcher)\n");
    fprintf(out, "  Backend:   GPU_BACKEND_KIND_OPENCL=%d / GPU_BACKEND_KIND_METAL=%d\n",
            GPU_BACKEND_KIND_OPENCL, GPU_BACKEND_KIND_METAL);
    fprintf(out, "  Picks:     LEGACY=%d / CODEGEN=%d / FATAL=%d\n",
            GPU_BACKEND_LEGACY, GPU_BACKEND_CODEGEN, GPU_BACKEND_FATAL);
    fprintf(out, "  FORCE:     MDXFIND_GPU_BACKEND={auto|legacy|codegen} (default auto)\n");
    fprintf(out, "  Quiet:     MDXFIND_GPU_BACKEND_QUIET=1 suppresses per-JOB advisories\n");
    fprintf(out, "  Deprecated: MDXFIND_EXPERIMENT_RULES_CODEGEN_MD5 IGNORED (shim warns once)\n\n");

    static const struct {
        int backend;
        int op;
        int max_iter;
        int rule_count;
        unsigned long long mask_total;
        unsigned int bf_chunk;
        const char *label;
    } probes[] = {
        { GPU_BACKEND_KIND_OPENCL, JOB_MD5,         1, 100, 0, 0, "OpenCL MD5 iter=1 rules>0" },
        { GPU_BACKEND_KIND_OPENCL, JOB_MD5,         5, 100, 0, 0, "OpenCL MD5 iter=5 rules>0" },
        { GPU_BACKEND_KIND_OPENCL, JOB_MD5,        10, 100, 0, 0, "OpenCL MD5 iter=10 rules>0" },
        { GPU_BACKEND_KIND_OPENCL, JOB_MD5,         1,   0, 0, 0, "OpenCL MD5 iter=1 rules=0" },
        { GPU_BACKEND_KIND_OPENCL, JOB_MD5,         1, 100, 1, 0, "OpenCL MD5 iter=1 mask>0" },
        { GPU_BACKEND_KIND_OPENCL, JOB_MD5,         1, 100, 0, 1, "OpenCL MD5 iter=1 bf>0" },
        { GPU_BACKEND_KIND_OPENCL, JOB_MD4,         1, 100, 0, 0, "OpenCL MD4 iter=1 rules>0" },
        { GPU_BACKEND_KIND_OPENCL, JOB_MD4,        10, 100, 0, 0, "OpenCL MD4 iter=10 rules>0" },
        { GPU_BACKEND_KIND_OPENCL, JOB_SHA1RAW,     1, 100, 0, 0, "OpenCL SHA1RAW iter=1 rules>0" },
        { GPU_BACKEND_KIND_OPENCL, JOB_SHA1RAW,    10, 100, 0, 0, "OpenCL SHA1RAW iter=10 rules>0" },
        { GPU_BACKEND_KIND_OPENCL, JOB_SHA256RAW,   1, 100, 0, 0, "OpenCL SHA256RAW iter=1 rules>0" },
        { GPU_BACKEND_KIND_OPENCL, JOB_SHA256RAW,  10, 100, 0, 0, "OpenCL SHA256RAW iter=10 rules>0" },
        { GPU_BACKEND_KIND_OPENCL, JOB_SHA1,        1, 100, 0, 0, "OpenCL SHA1 iter=1 rules>0" },
        { GPU_BACKEND_KIND_OPENCL, JOB_SHA1,       10, 100, 0, 0, "OpenCL SHA1 iter=10 rules>0" },
        { GPU_BACKEND_KIND_OPENCL, JOB_SHA256,      1, 100, 0, 0, "OpenCL SHA256 iter=1 rules>0" },
        { GPU_BACKEND_KIND_OPENCL, JOB_SHA256,     10, 100, 0, 0, "OpenCL SHA256 iter=10 rules>0" },
        { GPU_BACKEND_KIND_OPENCL, JOB_MD5MD5SALT,  1, 100, 0, 0, "OpenCL MD5MD5SALT (e347)" },
        { GPU_BACKEND_KIND_METAL,  JOB_MD5,         1, 100, 0, 0, "Metal MD5 iter=1 rules>0" },
        { GPU_BACKEND_KIND_METAL,  JOB_MD5,         5, 100, 0, 0, "Metal MD5 iter=5 rules>0 (FLAGSHIP)" },
        { GPU_BACKEND_KIND_METAL,  JOB_MD5,        10, 100, 0, 0, "Metal MD5 iter=10 rules>0 (FLAGSHIP)" },
        { GPU_BACKEND_KIND_METAL,  JOB_MD5,         1,   0, 0, 0, "Metal MD5 iter=1 rules=0" },
        { GPU_BACKEND_KIND_METAL,  JOB_MD4,         1, 100, 0, 0, "Metal MD4 iter=1 rules>0" },
        { GPU_BACKEND_KIND_METAL,  JOB_MD4,        10, 100, 0, 0, "Metal MD4 iter=10 rules>0" },
        { GPU_BACKEND_KIND_METAL,  JOB_SHA1RAW,     1, 100, 0, 0, "Metal SHA1RAW iter=1 rules>0" },
        { GPU_BACKEND_KIND_METAL,  JOB_SHA256RAW,   1, 100, 0, 0, "Metal SHA256RAW iter=1 rules>0" },
        { GPU_BACKEND_KIND_METAL,  JOB_SHA1,        1, 100, 0, 0, "Metal SHA1 iter=1 rules>0" },
        { GPU_BACKEND_KIND_METAL,  JOB_SHA1,       10, 100, 0, 0, "Metal SHA1 iter=10 rules>0 (FLAGSHIP)" },
        { GPU_BACKEND_KIND_METAL,  JOB_SHA256,      1, 100, 0, 0, "Metal SHA256 iter=1 rules>0" },
        { GPU_BACKEND_KIND_METAL,  JOB_SHA256,     10, 100, 0, 0, "Metal SHA256 iter=10 rules>0 (FLAGSHIP)" },
        { GPU_BACKEND_KIND_METAL,  JOB_MD5MD5SALT,  1, 100, 0, 0, "Metal MD5MD5SALT (e347)" },
    };
    int nprobes = (int)(sizeof(probes) / sizeof(probes[0]));
    fprintf(out, "%-44s  %-7s  %s\n", "Cell", "Pick", "Reason");
    fprintf(out, "%-44s  %-7s  %s\n", "----", "----", "------");
    for (int i = 0; i < nprobes; i++) {
        gpu_backend_pick_t p = _auto_pick(
            probes[i].backend, probes[i].op, probes[i].max_iter,
            probes[i].rule_count, probes[i].mask_total, probes[i].bf_chunk);
        const char *pname =
            (p == GPU_BACKEND_LEGACY)  ? "LEGACY"  :
            (p == GPU_BACKEND_CODEGEN) ? "CODEGEN" :
            (p == GPU_BACKEND_FATAL)   ? "FATAL"   : "?";
        const char *reason = _reason_text(
            probes[i].backend, probes[i].op, probes[i].max_iter,
            probes[i].rule_count, probes[i].mask_total, probes[i].bf_chunk,
            p);
        fprintf(out, "%-44s  %-7s  %s\n", probes[i].label, pname, reason);
    }
    fprintf(out, "\n");
}
