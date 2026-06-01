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
 * kernel B source. All other (pattern, backend) pairs fall through to
 * the 2a.2 per-opcode generic dispatch.
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
 * $Revision: 1.8 $
 * $Log: hx_walker.c,v $
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

/*
 * Internal walker state enum. Tiny, bounded, no recursion.
 */
enum walker_state {
    WST_EMIT_HEADER = 0,
    WST_EMIT_BYTECODE_BODY,
    WST_EMIT_FOOTER,
    WST_DONE,
    WST_ERROR
};

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
/* Per-backend skeleton-helper dispatch (preserved from 2a.1).        */
/* ------------------------------------------------------------------ */
static int emit_kernel_attribute(enum hx_backend backend,
                                 char **out, size_t *cap, size_t *len)
{
    if (backend == HX_BACKEND_OPENCL)
        return hx_emit_kernel_attribute_opencl(out, cap, len);
    if (backend == HX_BACKEND_METAL)
        return hx_emit_kernel_attribute_metal(out, cap, len);
    return -1;
}

static int emit_thread_id_load(enum hx_backend backend,
                               char **out, size_t *cap, size_t *len,
                               const char *var)
{
    if (backend == HX_BACKEND_OPENCL)
        return hx_emit_thread_id_load_opencl(out, cap, len, var);
    if (backend == HX_BACKEND_METAL)
        return hx_emit_thread_id_load_metal(out, cap, len, var);
    return -1;
}

static int emit_address_space_global(enum hx_backend backend,
                                     char **out, size_t *cap, size_t *len)
{
    if (backend == HX_BACKEND_OPENCL)
        return hx_emit_address_space_global_opencl(out, cap, len);
    if (backend == HX_BACKEND_METAL)
        return hx_emit_address_space_global_metal(out, cap, len);
    return -1;
}

/* ------------------------------------------------------------------ */
/* Per-opcode emit helpers (per-backend dispatch).                     */
/* ------------------------------------------------------------------ */

static int emit_op_push_var(enum hx_backend backend,
                            char **out, size_t *cap, size_t *len,
                            int slot, const char *varname)
{
    if (backend == HX_BACKEND_OPENCL)
        return hx_emit_push_var_opencl(out, cap, len, slot, varname);
    if (backend == HX_BACKEND_METAL)
        return hx_emit_push_var_metal(out, cap, len, slot, varname);
    return -1;
}

static int emit_op_push_str(enum hx_backend backend,
                            char **out, size_t *cap, size_t *len,
                            int stridx, const char *literal, int literal_len)
{
    if (backend == HX_BACKEND_OPENCL)
        return hx_emit_push_str_opencl(out, cap, len, stridx,
                                       literal, literal_len);
    if (backend == HX_BACKEND_METAL)
        return hx_emit_push_str_metal(out, cap, len, stridx,
                                      literal, literal_len);
    return -1;
}

static int emit_op_push_int(enum hx_backend backend,
                            char **out, size_t *cap, size_t *len,
                            int64_t ival)
{
    if (backend == HX_BACKEND_OPENCL)
        return hx_emit_push_int_opencl(out, cap, len, ival);
    if (backend == HX_BACKEND_METAL)
        return hx_emit_push_int_metal(out, cap, len, ival);
    return -1;
}

static int emit_op_store(enum hx_backend backend,
                         char **out, size_t *cap, size_t *len,
                         int slot, const char *varname)
{
    if (backend == HX_BACKEND_OPENCL)
        return hx_emit_store_opencl(out, cap, len, slot, varname);
    if (backend == HX_BACKEND_METAL)
        return hx_emit_store_metal(out, cap, len, slot, varname);
    return -1;
}

static int emit_op_call(enum hx_backend backend,
                        char **out, size_t *cap, size_t *len,
                        const char *fn_name, int nargs, uint8_t role)
{
    if (backend == HX_BACKEND_OPENCL)
        return hx_emit_call_opencl(out, cap, len, fn_name, nargs, role);
    if (backend == HX_BACKEND_METAL)
        return hx_emit_call_metal(out, cap, len, fn_name, nargs, role);
    return -1;
}

static int emit_op_concat(enum hx_backend backend,
                          char **out, size_t *cap, size_t *len)
{
    if (backend == HX_BACKEND_OPENCL)
        return hx_emit_concat_opencl(out, cap, len);
    if (backend == HX_BACKEND_METAL)
        return hx_emit_concat_metal(out, cap, len);
    return -1;
}

static int emit_op_halt(enum hx_backend backend,
                        char **out, size_t *cap, size_t *len)
{
    if (backend == HX_BACKEND_OPENCL)
        return hx_emit_halt_opencl(out, cap, len);
    if (backend == HX_BACKEND_METAL)
        return hx_emit_halt_metal(out, cap, len);
    return -1;
}

/* ------------------------------------------------------------------ */
/* State emit functions.                                              */
/* ------------------------------------------------------------------ */

static int emit_header(const hx_program *prog,
                       const struct hx_specialization *zone,
                       enum hx_backend backend,
                       char **out, size_t *cap, size_t *len)
{
    int rc;

    /* Annotation banner -- always // (never block-comment). */
    rc = hx_appendf(out, cap, len,
        "// hx: state EMIT_HEADER\n"
        "// hx: codegen sub-phase 2a.2 walker (production VM, 16 opcodes)\n"
        "// hx: program ncode=%d nvars=%d nstrings=%d max_stack=%d has_emit=%d\n"
        "// hx: specialization iter=%u rules=%u masks=%u bf=%u "
        "salt_minlen=%u salt_maxlen=%u salt_count_regime=%d emit_width=%u\n",
        prog->ncode, prog->nvars, prog->nstrings,
        prog->max_stack, prog->has_emit,
        zone->iter_count_if_fixed,
        (unsigned)zone->has_rules,
        (unsigned)zone->has_masks,
        (unsigned)zone->has_bf,
        zone->salt_minlen,
        zone->salt_maxlen,
        (int)zone->salt_count_regime,
        zone->emit_width);
    if (rc < 0) return rc;

    /* Kernel signature. Trivial: one global counter so the JIT has
     * something concrete to compile. The exact shape differs across
     * backends:
     *   - OpenCL: `__kernel void hx_trivial_kernel(__global volatile
     *             uint *counter)` + `const uint gid = get_global_id(0);`
     *   - Metal:  `kernel void hx_trivial_kernel(device atomic_uint
     *             *counter [[buffer(0)]], uint gid
     *             [[thread_position_in_grid]])` -- atomic_uint typing is
     *             required so per-opcode helpers can call
     *             atomic_fetch_add_explicit on counter[0]; gid binding
     *             is on the kernel argument list, not synthesised in
     *             the body.
     * Sub-phase 2a.4 (2026-05-21): Metal branch shaped from the e347
     * tp0 emitter discipline -- atomic typed args + per-arg attributes.
     * The OpenCL branch is byte-identical to the 2a.2 shape so existing
     * Pascal regression dumps remain identical. */
    if (backend == HX_BACKEND_METAL) {
        rc = hx_appendf(out, cap, len,
            "kernel void hx_trivial_kernel(\n"
            "    device atomic_uint *counter [[buffer(0)]],\n"
            "    uint gid [[thread_position_in_grid]])\n"
            "{\n");
        if (rc < 0) return rc;
        /* No body-side gid load needed (kernel arg already binds it);
         * emit a marker comment for the per-state // hx: trace. */
        rc = emit_thread_id_load(backend, out, cap, len, "gid");
        if (rc < 0) return rc;
        return 0;
    }

    /* OpenCL branch (unchanged from 2a.2). */
    rc = emit_kernel_attribute(backend, out, cap, len);  if (rc < 0) return rc;
    rc = hx_appendf(out, cap, len, "void hx_trivial_kernel(\n    ");
    if (rc < 0) return rc;
    rc = emit_address_space_global(backend, out, cap, len); if (rc < 0) return rc;
    rc = hx_appendf(out, cap, len, "volatile uint *counter)\n{\n");
    if (rc < 0) return rc;

    rc = emit_thread_id_load(backend, out, cap, len, "gid");
    if (rc < 0) return rc;

    return 0;
}

/*
 * Per-opcode dispatch. Sub-phase 2a.2 implements:
 *   OP_PUSH_VAR, OP_PUSH_STR, OP_PUSH_INT, OP_STORE,
 *   OP_CALL (md5 only), OP_CONCAT, OP_HALT
 * All other opcodes return -1 with an annotation comment.
 *
 * The walker pulls the function NAME for an OP_CALL out of the program's
 * varnames-style sidecar -- BUT the auto-generated data leaves
 * .u.call.entry = NULL (cross-process pointer). For 2a.2 we recognize
 * only the trivial e1 MD5 case where there's exactly one CALL with
 * nargs=1; we synthesize the name "md5" via the lookup helper, which
 * matches the sidecar emitted by tools/hx8_to_c. The runtime call-name
 * sidecar is consulted in 2a.3+ for the broader opcode coverage.
 */
static int emit_bytecode_op(const hx_program *prog, int i,
                            enum hx_backend backend,
                            char **out, size_t *cap, size_t *len)
{
    const hx_inst *ip = &prog->code[i];
    int rc = hx_appendf(out, cap, len,
                        "  // hx: op[%d] = %u (%s)\n", i, (unsigned)ip->op,
                        ip->op == OP_PUSH_VAR ? "PUSH_VAR" :
                        ip->op == OP_PUSH_STR ? "PUSH_STR" :
                        ip->op == OP_PUSH_INT ? "PUSH_INT" :
                        ip->op == OP_STORE    ? "STORE"    :
                        ip->op == OP_CALL     ? "CALL"     :
                        ip->op == OP_CONCAT   ? "CONCAT"   :
                        ip->op == OP_HALT     ? "HALT"     :
                        ip->op == OP_JUMP     ? "JUMP"     :
                        ip->op == OP_INC      ? "INC"      :
                        ip->op == OP_JUMP_LE  ? "JUMP_LE"  :
                        ip->op == OP_JUMP_LT  ? "JUMP_LT"  :
                        ip->op == OP_JUMP_GT  ? "JUMP_GT"  :
                        ip->op == OP_JUMP_GE  ? "JUMP_GE"  :
                        ip->op == OP_JUMP_EQ  ? "JUMP_EQ"  :
                        ip->op == OP_JUMP_NE  ? "JUMP_NE"  :
                        ip->op == OP_DUP      ? "DUP"      :
                        ip->op == OP_POP      ? "POP"      :
                                                "???");
    if (rc < 0) return rc;

    switch (ip->op) {
    case OP_PUSH_VAR: {
        const char *vn = (ip->u.slot >= 0 && ip->u.slot < prog->nvars
                          && prog->varnames[ip->u.slot])
                         ? prog->varnames[ip->u.slot] : "?";
        return emit_op_push_var(backend, out, cap, len, ip->u.slot, vn);
    }
    case OP_PUSH_STR: {
        const char *lit = NULL;
        int llen = 0;
        if (ip->u.stridx >= 0 && ip->u.stridx < prog->nstrings) {
            lit = prog->strings[ip->u.stridx];
            llen = prog->strlens[ip->u.stridx];
        }
        return emit_op_push_str(backend, out, cap, len,
                                ip->u.stridx, lit, llen);
    }
    case OP_PUSH_INT:
        return emit_op_push_int(backend, out, cap, len, ip->u.ival);

    case OP_STORE: {
        const char *vn = (ip->u.slot >= 0 && ip->u.slot < prog->nvars
                          && prog->varnames[ip->u.slot])
                         ? prog->varnames[ip->u.slot] : "?";
        return emit_op_store(backend, out, cap, len, ip->u.slot, vn);
    }
    case OP_CALL: {
        /* For 2a.2 the auto-generated data omits the function-entry
         * pointer (cross-process pointers don't serialize). We accept
         * either an entry pointer (when present, e.g. a program
         * compiled fresh via hx_compile_expr) or use a name lookup
         * fallback (currently "md5" hardcoded -- the only fn the e1
         * fixture exercises). 2a.3+ wires the call-name sidecar
         * properly. */
        const char *fn = "md5";
        if (ip->u.call.entry && ip->u.call.entry->name)
            fn = ip->u.call.entry->name;
        return emit_op_call(backend, out, cap, len,
                            fn, ip->u.call.nargs, ip->u.call.role);
    }
    case OP_CONCAT:
        return emit_op_concat(backend, out, cap, len);

    case OP_HALT:
        return emit_op_halt(backend, out, cap, len);

    /* Control-flow opcodes are stubbed for 2a.2 -- emit a comment
     * documenting the deferral and return -1. Walker reports this
     * up to caller for a clean FATAL. */
    case OP_JUMP:
    case OP_JUMP_LE:
    case OP_JUMP_LT:
    case OP_JUMP_GT:
    case OP_JUMP_GE:
    case OP_JUMP_EQ:
    case OP_JUMP_NE:
    case OP_INC:
    case OP_DUP:
    case OP_POP:
        hx_appendf(out, cap, len,
            "  // hx: opcode %u not yet implemented in 2a.2 (deferred)\n",
            (unsigned)ip->op);
        fprintf(stderr,
                "hx codegen: opcode %u not yet implemented in "
                "sub-phase 2a.2 (deferred to 2a.3+)\n",
                (unsigned)ip->op);
        return -1;

    default:
        fprintf(stderr,
                "hx codegen: unknown opcode %u at code[%d]\n",
                (unsigned)ip->op, i);
        return -1;
    }
}

static int emit_bytecode_body(const hx_program *prog,
                              enum hx_backend backend,
                              char **out, size_t *cap, size_t *len)
{
    int rc = hx_appendf(out, cap, len,
        "  // hx: state EMIT_BYTECODE_BODY ncode=%d\n",
        prog->ncode);
    if (rc < 0) return rc;

    if (prog->ncode <= 0) {
        fprintf(stderr, "hx codegen: empty program (ncode=%d)\n",
                prog->ncode);
        return -1;
    }
    for (int i = 0; i < prog->ncode; i++) {
        rc = emit_bytecode_op(prog, i, backend, out, cap, len);
        if (rc < 0) return rc;
    }
    return 0;
}

static int emit_footer(char **out, size_t *cap, size_t *len)
{
    return hx_appendf(out, cap, len,
        "  // hx: state EMIT_FOOTER\n"
        "}\n");
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
     * Currently implemented: HX_PATTERN_E347_MD5MD5MD5SALT for
     * HX_BACKEND_OPENCL only. Metal twin lands in sub-phase 2a.4.
     * Any other pattern OR Metal backend falls through to the per-opcode
     * generic dispatch (which currently emits 2a.2 placeholders for
     * non-trivial programs). */
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
        /* Unknown backend -- fall through to generic placeholder. */
        fprintf(stderr,
                "hx codegen: FAMILY_MD5PASS pattern matched (backend=%d) "
                "-- no emitter for this backend; falling through to "
                "per-opcode generic dispatch (placeholder, NOT byte-exact "
                "correct)\n",
                (int)backend);
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
        fprintf(stderr,
                "hx codegen: UNSALTED_SINGLE pattern matched (backend=%d) "
                "-- no emitter for this backend; falling through.\n",
                (int)backend);
    } else if (pat != HX_PATTERN_UNKNOWN) {
        /* Detected a pattern but backend not yet implemented; annotate
         * and fall through to generic. Diagnostic only -- helpful when
         * future patterns lack the requested backend's emitter. */
        fprintf(stderr,
                "hx codegen: pattern %s detected but no emitter for "
                "backend %d; falling through to generic dispatch\n",
                hx_pattern_name(pat), (int)backend);
    }

    size_t cur_len = 0;
    if (*out == NULL) *out_cap = 0;

    enum walker_state state = WST_EMIT_HEADER;
    int rc = 0;

    while (state != WST_DONE && state != WST_ERROR) {
        switch (state) {
        case WST_EMIT_HEADER:
            rc = emit_header(prog, zone, backend, out, out_cap, &cur_len);
            state = (rc < 0) ? WST_ERROR : WST_EMIT_BYTECODE_BODY;
            break;

        case WST_EMIT_BYTECODE_BODY:
            rc = emit_bytecode_body(prog, backend, out, out_cap, &cur_len);
            state = (rc < 0) ? WST_ERROR : WST_EMIT_FOOTER;
            break;

        case WST_EMIT_FOOTER:
            rc = emit_footer(out, out_cap, &cur_len);
            state = (rc < 0) ? WST_ERROR : WST_DONE;
            break;

        default:
            state = WST_ERROR;
        }
    }

    if (state == WST_ERROR) return (rc < 0) ? rc : -1;

    /* Defensive NUL terminator -- vsnprintf already null-terminated, */
    /* but if cur_len ended exactly at out_cap there's no slack.      */
    if (cur_len + 1 > *out_cap) {
        char *np = (char *)realloc(*out, cur_len + 1);
        if (!np) return -1;
        *out = np;
        *out_cap = cur_len + 1;
    }
    (*out)[cur_len] = '\0';
    return 0;
}
