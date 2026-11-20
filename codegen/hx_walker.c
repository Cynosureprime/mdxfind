/*
 * hx_walker.c -- in-process P4 state-machine walker for hx codegen.
 *
 * Sub-phase 2a.2 (per project_hx_codegen_phase2_3_spec_2026-05-21.md
 * D12.2.c REVISED-AGAIN -- Path A): the walker dispatches on the
 * 16-opcode production VM `hx_opcode` enum from hx_vm.h, NOT the
 * invented 2a.1 bytecode. The walker consumes `hx_program` directly
 * from the auto-generated codegen/hx_specs_data.c.
 *
 * State machine (small, no recursion):
 *
 *     EMIT_HEADER         -- file-level preamble + kernel signature
 *     EMIT_BYTECODE_BODY  -- walk prog->code[0..prog->ncode-1]
 *     EMIT_FOOTER         -- closing brace + trailing newline
 *
 * Each state appends to *out via hx_appendf. Each emitted state writes
 * a "// hx: state ..." annotation comment per R5 (walker output must
 * be human-debuggable on JIT failure).
 *
 * Comments emitted into the source use "//" only -- never block
 * comments -- per feedback_no_nested_block_comments_in_cl.md.
 *
 * 2a.2 implements the minimum-viable opcode set: OP_PUSH_VAR,
 * OP_PUSH_STR, OP_PUSH_INT, OP_STORE, OP_CALL (md5 only), OP_CONCAT,
 * OP_HALT. Control-flow opcodes (OP_JUMP*, OP_INC, OP_DUP, OP_POP)
 * are stubbed and return -1 -- real impl arrives in 2a.3+.
 *
 * For OP_CALL, the per-instruction function entry pointer is NULL in
 * the auto-generated data (cross-process pointers don't serialize);
 * the function name lives in the sidecar table emitted alongside
 * each program. In 2a.2 we accept the name via a separate
 * mechanism (caller passes through a name lookup); for the trivial
 * e1 MD5 fixture the only CALL target is "md5" hard-coded in the
 * emit helper.
 *
 * Per feedback_external_failures_are_fatal.md the walker itself never
 * silently drops or returns NULL for an external failure; allocation
 * failures bubble up as negative returns and the caller (mdxfind.c
 * harness) treats them as fatal.
 *
 * Sub-phase 2a.3 (2026-05-21): pattern-recognized fast-path dispatch
 * added at the head of hx_emit_kernel. hx_detect_pattern scans the
 * bytecode shape; on HX_PATTERN_E347_MD5MD5MD5SALT (OpenCL backend) the
 * walker delegates to the specialized emitter
 * hx_emit_e347_md5md5md5salt_opencl which produces a full tp0-pattern
 * kernel B source. (2026-11-16: the 2a.2 per-opcode placeholder walk is
 * removed; a program no pattern emitter takes is refused.)
 *
 * Sub-phase 2a.4 (2026-05-21): Metal backend dispatch added for the
 * same HX_PATTERN_E347_MD5MD5MD5SALT pattern. Walker now dispatches to
 * hx_emit_e347_md5md5md5salt_metal when (pattern, backend) matches.
 * The (pattern, backend) diagnostic markers distinguish OpenCL vs Metal
 * in stderr so dump-file traces are unambiguous when both backends are
 * exercised in the same session.
 *
 * Sub-phase 5a.1 (2026-05-22): walker recognises the new
 * HX_PATTERN_FAMILY_MD5PASS pattern (30 MAKE_MD5PASS algorithms) but
 * defers the per-primitive specialized emitter to sub-phase 5a.2
 * (OpenCL) / 5a.3 (Metal). For 5a.1 the walker logs a stderr marker on
 * family detection then falls through to the per-opcode generic
 * dispatch which produces a JIT-compilable but NOT byte-exact-correct
 * placeholder. Validates the detector framework end-to-end without
 * blocking 5a.2 emitter implementation.
 *
 * Sub-phase 5a.2 (2026-05-22): hx_emit_kernel signature gains a fourth
 * argument `const struct hx_spec_entry *entry` so the FAMILY_MD5PASS
 * arm can forward it to hx_emit_family_md5pass_opencl, which resolves
 * the per-call-site callnames sidecar via hx_callname_for_entry. The
 * walker FATALs on NULL entry when the pattern requires it (currently
 * just FAMILY_MD5PASS); e347 and the generic per-opcode dispatch are
 * entry-independent. All callers updated.
 *
 * Sub-phase 5a.3 (2026-05-22): FAMILY_MD5PASS Metal arm replaces the
 * 5a.2 fallthrough placeholder with the real hx_emit_family_md5pass_metal
 * dispatch. Both backends now share the same entry-required gate; on
 * NULL entry the walker FATALs uniformly. e347 and per-opcode generic
 * dispatch remain entry-independent.
 *
 * $Revision: 1.9 $
 * $Log: hx_walker.c,v $
 * Revision 1.9  2026/11/16 21:05:06  dlr
 * Remove the per-opcode placeholder walk. It emitted hx_trivial_kernel, which compiles and computes nothing, so a route that reached it would have run a GPU pass that found nothing and reported success. hx_emit_kernel now dispatches only to the E347, FAMILY_MD5PASS and UNSALTED_SINGLE emitters and returns -1 with a one-line reason for any other program; every caller already exits on -1. The walker state machine and its emit helpers go with it.
 *
 * Revision 1.8  2026/05/28 14:32:17  dlr
 * Phase 1b Batch 1: add UNSALTED_SINGLE dispatch arm to hx_emit_kernel routes the 3-op hash of pass shape to hx_emit_unsalted_single_opencl or _metal both backends ship in Batch 1; requires entry for the per-program callnames sidecar FATAL on NULL; placed after the FAMILY_MD5PASS arm
 *
 * Revision 1.7  2026/05/23 03:21:40  dlr
 * sub-phase 5a.3 walker FAMILY_MD5PASS Metal arm replaces 5a.2 fallthrough placeholder with real hx_emit_family_md5pass_metal dispatch both backends now share the same entry-required gate on NULL entry walker FATALs uniformly e347 and per-opcode generic dispatch remain entry-independent
 *
 * Revision 1.6  2026/05/23 02:02:34  dlr
 * sub-phase 5a.2 add HX_PATTERN_FAMILY_MD5PASS dispatch arm forwards prog spec and entry to hx_emit_family_md5pass_opencl when backend is OpenCL with FATAL on NULL entry per per-program callnames sidecar requirement; Metal arm still falls through to per-opcode generic until 5a.3 Metal twin lands; walker signature gains entry parameter so family emitters can reach the sidecar without O of N reverse lookup over hx_specs_data
 *
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include "hx_spec.h"
#include "hx_walker.h"
#include "hx_emit.h"
#include "hx_patterns.h"
#include "../hx_vm.h"

/* ------------------------------------------------------------------ */
/* hx_appendf -- printf-style append into a growable buffer.          */
/* ------------------------------------------------------------------ */
int hx_appendf(char **out, size_t *out_cap, size_t *out_len,
               const char *fmt, ...)
{
    if (!out || !out_cap || !out_len || !fmt) return -1;

    va_list ap, ap2;
    va_start(ap, fmt);
    va_copy(ap2, ap);
    int needed = vsnprintf(NULL, 0, fmt, ap);
    va_end(ap);
    if (needed < 0) { va_end(ap2); return -1; }

    size_t new_len = *out_len + (size_t)needed + 1; /* +1 for NUL */
    if (new_len > *out_cap) {
        size_t new_cap = *out_cap ? *out_cap : 4096;
        while (new_cap < new_len) new_cap *= 2;
        char *np = (char *)realloc(*out, new_cap);
        if (!np) { va_end(ap2); return -1; }
        *out = np;
        *out_cap = new_cap;
    }
    int written = vsnprintf(*out + *out_len, *out_cap - *out_len, fmt, ap2);
    va_end(ap2);
    if (written < 0) return -1;
    *out_len += (size_t)written;
    return 0;
}

/* ------------------------------------------------------------------ */
/* Public entry: state-machine loop.                                   */
/* ------------------------------------------------------------------ */
int hx_emit_kernel(const hx_program *prog,
                   const struct hx_specialization *zone,
                   enum hx_backend backend,
                   const struct hx_spec_entry *entry,
                   char **out, size_t *out_cap)
{
    if (!prog || !zone || !out || !out_cap) return -1;
    if (backend != HX_BACKEND_OPENCL && backend != HX_BACKEND_METAL)
        return -1;

    /* Sub-phase 2a.3 (2026-05-21): pattern-recognized fast path. The
     * detector scans prog->code[] for known hand-tuned shapes; on a
     * match for an implemented (pattern, backend) pair, dispatch to the
     * specialized emitter and skip the per-opcode generic walk.
     *
     * Implemented on both backends: E347, FAMILY_MD5PASS and
     * UNSALTED_SINGLE. Any other program is refused below. */
    hx_pattern_id pat = hx_detect_pattern(prog);
    if (pat == HX_PATTERN_E347_MD5MD5MD5SALT &&
        backend == HX_BACKEND_OPENCL) {
        fprintf(stderr,
                "hx codegen: e347 MD5MD5SALT tp0 pattern matched (opencl) "
                "(ncode=%d nvars=%d) -- dispatching to specialized emitter\n",
                prog->ncode, prog->nvars);
        return hx_emit_e347_md5md5md5salt_opencl(out, out_cap, prog, zone);
    }
    if (pat == HX_PATTERN_E347_MD5MD5MD5SALT &&
        backend == HX_BACKEND_METAL) {
        /* Sub-phase 2a.4 (2026-05-21): Metal twin of the e347 emitter. */
        fprintf(stderr,
                "hx codegen: e347 MD5MD5SALT tp0 pattern matched (metal) "
                "(ncode=%d nvars=%d) -- dispatching to specialized emitter\n",
                prog->ncode, prog->nvars);
        return hx_emit_e347_md5md5md5salt_metal(out, out_cap, prog, zone);
    }
    if (pat == HX_PATTERN_FAMILY_MD5PASS) {
        /* Sub-phase 5a.2 (2026-05-22): family pattern emitter wired
         * for OpenCL. Sub-phase 5a.3 (2026-05-22): Metal twin wired.
         * SHA1 outer (e161) is the first end-to-end primitive on both
         * backends; the other 7 5a-supported primitives are 5a.4 scope
         * and the emitter FATALs on those with a clean "deferred to 5a.4"
         * diagnostic. */
        if (!entry) {
            fprintf(stderr,
                    "FATAL: hx_walker.c hx_emit_kernel: "
                    "FAMILY_MD5PASS requires entry (per-program "
                    "callnames sidecar) but caller passed NULL. "
                    "Callers from production paths must pass the "
                    "entry returned by hx_specs_lookup(job_enum).\n");
            return -1;
        }
        if (backend == HX_BACKEND_OPENCL) {
            fprintf(stderr,
                    "hx codegen: FAMILY_MD5PASS pattern matched (opencl) "
                    "(ncode=%d nvars=%d) -- dispatching to family emitter\n",
                    prog->ncode, prog->nvars);
            return hx_emit_family_md5pass_opencl(out, out_cap, prog,
                                                 zone, entry);
        }
        if (backend == HX_BACKEND_METAL) {
            fprintf(stderr,
                    "hx codegen: FAMILY_MD5PASS pattern matched (metal) "
                    "(ncode=%d nvars=%d) -- dispatching to family emitter\n",
                    prog->ncode, prog->nvars);
            return hx_emit_family_md5pass_metal(out, out_cap, prog,
                                                zone, entry);
        }
    } else if (pat == HX_PATTERN_UNSALTED_SINGLE) {
        /* Phase 1b Batch 1 (2026-05-28): unsalted single-hash emitter
         * (hash(pass), category (a)). Requires entry (resolves the
         * single CALL primitive via the per-program callnames sidecar).
         * Both backends ship in Batch 1. */
        if (!entry) {
            fprintf(stderr,
                    "FATAL: hx_walker.c hx_emit_kernel: "
                    "UNSALTED_SINGLE requires entry (per-program "
                    "callnames sidecar) but caller passed NULL.\n");
            return -1;
        }
        if (backend == HX_BACKEND_OPENCL) {
            fprintf(stderr,
                    "hx codegen: UNSALTED_SINGLE pattern matched (opencl) "
                    "(ncode=%d nvars=%d) -- dispatching to unsalted emitter\n",
                    prog->ncode, prog->nvars);
            return hx_emit_unsalted_single_opencl(out, out_cap, prog,
                                                  zone, entry);
        }
        if (backend == HX_BACKEND_METAL) {
            fprintf(stderr,
                    "hx codegen: UNSALTED_SINGLE pattern matched (metal) "
                    "(ncode=%d nvars=%d) -- dispatching to unsalted emitter\n",
                    prog->ncode, prog->nvars);
            return hx_emit_unsalted_single_metal(out, out_cap, prog,
                                                 zone, entry);
        }
    }

    /* No pattern emitter took the program. The per-opcode placeholder walk
     * that used to run here emitted hx_trivial_kernel, which compiles and
     * computes nothing, so a route that reached it would have run a clean
     * GPU pass that found nothing. It is gone; the generic walker replaces
     * it, and until then an unmatched program is refused here and the
     * caller exits with its own diagnostic. */
    fprintf(stderr,
            "hx codegen: no emitter for this program "
            "(pattern %s, backend %d, ncode=%d)\n",
            hx_pattern_name(pat), (int)backend, prog->ncode);
    return -1;
}
