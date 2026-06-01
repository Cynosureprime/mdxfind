/*
 * hx_program_cmp.h -- shared hx_program bytecode comparator + Layer 2
 *                     canonicalization, used by both
 *                     tools/hx_dedup_check.c (standalone dedup CLI) and
 *                     tools/hx8_to_c.c (build-time intra-catalog warning).
 *
 * Tier 2 (per architect spec project_hx_dedup_check_spec_2026-05-26.md
 * §7 + the D-points) lifted programs_equal() out of hx_dedup_check.c so
 * the build-time intra-catalog warning in hx8_to_c reuses the SAME
 * comparison logic the standalone tool uses. Header-only (static inline)
 * keeps the link surface identical to the existing tools/hx8_to_c line --
 * no new .o, no new library.
 *
 * Comparison definition (Tier 1 + Tier 2 Layer 2):
 *
 *   Two programs are "the same algorithm" iff, after canonicalization,
 *   their (op, operand) instruction streams are byte-equivalent:
 *     - ncode matches
 *     - per instruction: op matches, plus operand:
 *         OP_PUSH_VAR/STORE/INC : slot index matches
 *         OP_PUSH_STR           : string CONTENT + length match
 *                                 (NOT stridx -- table ordering can vary)
 *         OP_PUSH_INT           : ival matches
 *         OP_JUMP*              : addr matches
 *         OP_CALL               : nargs match, CANONICALIZED role matches,
 *                                 called function NAME matches
 *
 *   Layer 1 (Tier 1): bytecode-exact comparison as above.
 *
 *   Role-suffix canonicalization (Tier 1): ROLE_DEFAULT (bare name, no
 *   `_hex`/`_bin`/...) is collapsed to the function's documented
 *   default_role before comparison, so md5(pass) == md5_hex(pass).
 *
 *   Layer 2 normalization (Tier 2):
 *     (a) Commutative operator operand reordering -- canonicalize the
 *         operand order of commutative OP_CALL functions (xor) so
 *         xor(a,b) == xor(b,a). Implemented as a structural canonicalizer
 *         that, for each commutative 2-arg OP_CALL, identifies the two
 *         operand sub-sequences on the bytecode stream and swaps them
 *         into a deterministic order (lexicographic by a stable per-inst
 *         key). SAFE because xor is byte-wise commutative.
 *         NOT applied to OP_CONCAT (string concat a.b != b.a -- order is
 *         semantically significant) nor to any non-commutative OP_CALL.
 *     (b) Alpha-rename of named temporaries is ALREADY handled by Layer 1:
 *         the hx compiler assigns variable slots by first-use order
 *         (hx_compile.c resolve_var) and the comparator keys on u.slot,
 *         NOT on varnames[]. Two expressions differing only in the spelling
 *         of a `h = ...` temporary therefore already compare equal at
 *         Layer 1. See note "ALPHA-RENAME" below.
 *
 * Read-only: the comparator never mutates the catalog or hx.8. Per
 * feedback_catalog_aliases_are_historical_dont_alter.md the duplicates
 * it surfaces are INTENTIONAL historical artifacts -- the tool REPORTS,
 * the operator decides.
 *
 * $Revision: 1.1 $
 * $Log: hx_program_cmp.h,v $
 * Revision 1.1  2026/05/28 12:55:13  dlr
 * Tier 2: shared hx_program_cmp.h -- programs_equal bytecode comparator (Layer 1 plus role-suffix canonicalization) plus Layer 2 commutative-operator operand reordering (xor canonicalized, concat NOT reordered) and temporary alpha-rename (falls out of slot-based comparison). Lifted from hx_dedup_check.c so the hx8_to_c build-time warning shares one comparator. Header-only static inline; no new link surface.
 *
 */

#ifndef HX_PROGRAM_CMP_H
#define HX_PROGRAM_CMP_H

#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include "../hx_vm.h"

/* ------------------------------------------------------------------ *
 * ALPHA-RENAME (Layer 2 (b)) -- why it needs no extra code.
 *
 * The hx grammar supports named temporaries via assignment statements
 * (`h = md5(pass) ; sha1(h)`). The compiler (hx_compile.c resolve_var)
 * allocates user variable slots densely from HX_SLOT_USERVARS (5) in
 * FIRST-USE order and emits OP_STORE/OP_PUSH_VAR against the slot index,
 * not the name. programs_equal() compares ia->u.slot, never the
 * varnames[] table. Therefore two programs that differ ONLY in the
 * spelling of a temporary (`h = ...` vs `tmp = ...`) -- with identical
 * first-use order -- already produce identical bytecode and compare
 * equal at Layer 1. No canonicalization pass is required; the property
 * falls out of slot-based addressing.
 *
 * The ONLY residual alpha-rename case Layer 1 misses is when two
 * programs assign the SAME set of temporaries in a DIFFERENT first-use
 * order yet remain semantically identical (rare: requires independent
 * sub-expressions whose evaluation order the language leaves free). The
 * hx language fixes evaluation order (left-to-right), so reordering
 * independent assignments changes the OP stream deterministically and is
 * NOT a safe normalization (it can change emit() side-effect ordering).
 * We deliberately do NOT canonicalize assignment order. Documented gap.
 * ------------------------------------------------------------------ */

/* ---------- commutative-operator registry ---------- */

/*
 * A function is commutative for dedup purposes iff swapping its two
 * operands yields a byte-identical digest for ALL inputs. Byte-wise
 * `xor` qualifies. Concat is NOT a function (it is OP_CONCAT) and is
 * never commutative. Add names here only after verifying the underlying
 * fn_* is genuinely operand-symmetric.
 *
 * NOTE: xor in hx is variadic-ish (fn_xor folds all args), but the
 * catalog only ever uses the 2-arg form; we canonicalize the 2-arg form
 * only. A >2-arg commutative call would need full operand-set sorting --
 * deferred (no catalog usage).
 */
static const char *const HX_COMMUTATIVE_FUNCS[] = { "xor", NULL };

static inline int hx_func_is_commutative(const char *name)
{
    if (!name) return 0;
    for (int i = 0; HX_COMMUTATIVE_FUNCS[i]; i++)
        if (strcmp(name, HX_COMMUTATIVE_FUNCS[i]) == 0)
            return 1;
    return 0;
}

/* ---------- string / name accessors ---------- */

static inline const char *hx_cmp_get_str(const hx_program *prog, int idx,
                                          int *out_len)
{
    if (idx < 0 || idx >= prog->nstrings) {
        if (out_len) *out_len = 0;
        return "";
    }
    if (out_len) *out_len = prog->strlens[idx];
    return prog->strings[idx];
}

static inline const char *hx_cmp_call_name_at(const hx_program *prog,
                                              const char *const *call_names,
                                              int code_idx)
{
    const hx_inst *ip = &prog->code[code_idx];
    if (ip->u.call.entry && ip->u.call.entry->name)
        return ip->u.call.entry->name;
    if (call_names && call_names[code_idx])
        return call_names[code_idx];
    return "?";
}

/* ---------- role canonicalization (Tier 1) ---------- */

static inline uint8_t hx_cmp_default_role_for_call(const hx_program *prog,
                                                   const char *const *call_names,
                                                   int code_idx)
{
    hx_func_entry *fe = NULL;
    const hx_inst *ip = &prog->code[code_idx];

    if (ip->u.call.entry)
        fe = ip->u.call.entry;
    else if (call_names && call_names[code_idx])
        fe = hx_func_lookup(call_names[code_idx]);

    if (fe) {
        if (fe->default_role == ROLE_DEFAULT)
            return ROLE_HEX;            /* hard-default for transforms */
        return fe->default_role;
    }
    return ROLE_HEX;                    /* matches prescan stub default */
}

/* ------------------------------------------------------------------ *
 * Layer 2 (a) -- commutative operand reordering.
 *
 * The bytecode is a postfix (RPN) stream: each OP_CALL pops nargs
 * operands that were pushed by the immediately-preceding instructions.
 * For a 2-arg commutative call at index i, the two operand sub-sequences
 * are the two complete RPN sub-expressions that end just before i. We
 * locate their boundaries by walking BACKWARD from i-1 tracking the net
 * stack effect of each instruction, splitting off one balanced operand
 * at a time. If the two operands are out of canonical order (per a
 * stable comparison of their instruction sub-streams) we swap them.
 *
 * This produces a canonical form so xor(a,b) and xor(b,a) compile-equal
 * AFTER canonicalization. We do this on a COPY of the code array so the
 * caller's program is never mutated.
 *
 * Stack-effect model (operand count delta produced on the value stack):
 *   OP_PUSH_VAR / OP_PUSH_STR / OP_PUSH_INT : +1 (produce one value)
 *   OP_CONCAT                               : -1 (pop 2, push 1)
 *   OP_CALL nargs                           : 1 - nargs (pop nargs, push 1)
 *   OP_DUP                                  : +1
 *   OP_POP                                  : -1
 *   OP_STORE                                : 0 (peek-store in hx; value
 *                                                stays on stack -- see below)
 *   others (JUMP*, INC, HALT)               : 0 (control / no net value)
 *
 * Because commutative reordering of operands that contain control flow
 * (jumps/loops/stores) is risky (the addr operands are absolute and a
 * reorder would invalidate them), we ONLY reorder when BOTH operand
 * sub-sequences are "simple" -- contain no OP_STORE / OP_JUMP* / OP_INC /
 * OP_FOR-machinery. The catalog's xor usages are simple
 * (xor(sha1_bin(pass), salt) etc.), so this covers the real cases while
 * staying provably safe. Non-simple operands are left in source order
 * (documented conservative gap).
 * ------------------------------------------------------------------ */

static inline int hx_cmp_inst_stack_delta(const hx_inst *ip)
{
    switch (ip->op) {
    case OP_PUSH_VAR:
    case OP_PUSH_STR:
    case OP_PUSH_INT:
    case OP_DUP:
        return 1;
    case OP_CONCAT:
    case OP_POP:
        return -1;
    case OP_CALL:
        return 1 - ip->u.call.nargs;
    default:
        /* OP_STORE, OP_INC, jumps, OP_HALT: treat as net-zero AND mark
         * the operand non-simple via the caller's simple-check. */
        return 0;
    }
}

static inline int hx_cmp_inst_is_simple(const hx_inst *ip)
{
    switch (ip->op) {
    case OP_PUSH_VAR:
    case OP_PUSH_STR:
    case OP_PUSH_INT:
    case OP_CONCAT:
    case OP_CALL:
        return 1;
    default:
        /* OP_STORE, OP_INC, jumps, OP_DUP, OP_POP, OP_HALT */
        return 0;
    }
}

/*
 * Given the code array and the index `call_idx` of a 2-arg OP_CALL,
 * find the [start,end) ranges of its two operand sub-sequences on the
 * RPN stream. operand2 is the LAST one pushed (closest to call_idx),
 * operand1 the one before it. Returns 1 on success (both ranges found
 * and both simple), 0 if the operands are not cleanly splittable or not
 * simple (caller then leaves order unchanged).
 *
 *   o1_start .. o1_end   == operand 1 (first source arg)
 *   o2_start .. o2_end   == operand 2 (second source arg)
 *   where o1_end == o2_start and o2_end == call_idx.
 */
static inline int hx_cmp_find_two_operands(const hx_inst *code, int call_idx,
                                           int *o1_start, int *o1_end,
                                           int *o2_start, int *o2_end)
{
    /* Walk backward collecting one balanced operand: a sub-sequence
     * whose cumulative stack delta (read left-to-right within it) is
     * exactly +1 and never dips below 0 at its own left edge. We find
     * the boundary by scanning backward and stopping when the running
     * "values produced from the right edge" reaches 1. */
    int i;

    /* operand 2: ends at call_idx-1, walk back until net == +1 */
    int net = 0;
    int start2 = -1;
    for (i = call_idx - 1; i >= 0; i--) {
        if (!hx_cmp_inst_is_simple(&code[i])) return 0;
        net += hx_cmp_inst_stack_delta(&code[i]);
        if (net == 1) { start2 = i; break; }
        if (net > 1) return 0;   /* boundary not clean */
    }
    if (start2 < 0) return 0;

    /* operand 1: ends at start2-1, walk back until net == +1 */
    net = 0;
    int start1 = -1;
    for (i = start2 - 1; i >= 0; i--) {
        if (!hx_cmp_inst_is_simple(&code[i])) return 0;
        net += hx_cmp_inst_stack_delta(&code[i]);
        if (net == 1) { start1 = i; break; }
        if (net > 1) return 0;
    }
    if (start1 < 0) return 0;

    *o1_start = start1; *o1_end = start2;
    *o2_start = start2; *o2_end = call_idx;
    return 1;
}

/*
 * Stable comparison key for an operand sub-sequence so reordering is
 * deterministic. Compares instruction-by-instruction using (op, then a
 * canonical operand discriminator). For OP_PUSH_STR we compare string
 * content; for OP_CALL we compare callee name. Returns <0, 0, >0.
 *
 * call_names lets us key OP_CALL by name even for catalog (entry==NULL)
 * programs.
 */
static inline int hx_cmp_operand_key(const hx_program *prog,
                                     const char *const *call_names,
                                     int a_start, int a_end,
                                     int b_start, int b_end)
{
    int la = a_end - a_start, lb = b_end - b_start;
    int n = la < lb ? la : lb;
    for (int k = 0; k < n; k++) {
        const hx_inst *ia = &prog->code[a_start + k];
        const hx_inst *ib = &prog->code[b_start + k];
        if (ia->op != ib->op)
            return (int)ia->op - (int)ib->op;
        switch (ia->op) {
        case OP_PUSH_VAR:
            if (ia->u.slot != ib->u.slot) return ia->u.slot - ib->u.slot;
            break;
        case OP_PUSH_INT:
            if (ia->u.ival != ib->u.ival)
                return ia->u.ival < ib->u.ival ? -1 : 1;
            break;
        case OP_PUSH_STR: {
            int sal = 0, sbl = 0;
            const char *sa = hx_cmp_get_str(prog, ia->u.stridx, &sal);
            const char *sb = hx_cmp_get_str(prog, ib->u.stridx, &sbl);
            int m = sal < sbl ? sal : sbl;
            int c = memcmp(sa, sb, (size_t)m);
            if (c) return c;
            if (sal != sbl) return sal - sbl;
            break;
        }
        case OP_CALL: {
            const char *na = hx_cmp_call_name_at(prog, call_names, a_start + k);
            const char *nb = hx_cmp_call_name_at(prog, call_names, b_start + k);
            int c = strcmp(na, nb);
            if (c) return c;
            if (ia->u.call.nargs != ib->u.call.nargs)
                return ia->u.call.nargs - ib->u.call.nargs;
            break;
        }
        default:
            break;
        }
    }
    return la - lb;
}

/*
 * Produce a Layer-2-canonicalized COPY of prog->code AND a matching
 * reordered call_names sidecar. Commutative 2-arg OP_CALL operands are
 * reordered into canonical order. The code and call_names arrays are
 * permuted in LOCKSTEP so that, after canonicalization, accessing
 * out_code[i] / out_call_names[i] for the same i refers to the same
 * (instruction, callee-name) pairing it had before the move -- this is
 * what makes the post-reorder comparator's per-index OP_CALL name and
 * stridx lookups correct.
 *
 * String-table indices (stridx) are NOT rewritten -- we only move whole
 * instructions -- so callers MUST compare strings via the ORIGINAL
 * prog's string table (the canonicalized stream still carries each
 * instruction's original stridx).
 *
 * On success returns 1, fills *out_code (malloc'd hx_inst[ncode]) and
 * *out_call_names (malloc'd const char*[ncode], may hold NULLs); caller
 * frees both. On OOM returns 0 and the caller falls back to the
 * original (un-canonicalized) code+sidecar.
 *
 * Fixpoint: scan repeatedly until no swap occurs (catalog nesting is
 * shallow; converges in 1-2 passes). Guard caps at 8 passes.
 */
static inline int hx_cmp_canonicalize_code(const hx_program *prog,
                                           const char *const *call_names,
                                           hx_inst **out_code,
                                           const char ***out_call_names)
{
    int n = prog->ncode;
    int an = (n > 0 ? n : 1);
    hx_inst *code = (hx_inst *)malloc((size_t)an * sizeof(hx_inst));
    const char **names = (const char **)malloc((size_t)an * sizeof(char *));
    if (!code || !names) { free(code); free(names); return 0; }

    memcpy(code, prog->code, (size_t)n * sizeof(hx_inst));
    for (int i = 0; i < n; i++) {
        /* Snapshot each instruction's effective callee name by original
         * index so it travels with the instruction when reordered. */
        if (code[i].op == OP_CALL) {
            if (code[i].u.call.entry && code[i].u.call.entry->name)
                names[i] = code[i].u.call.entry->name;
            else if (call_names && call_names[i])
                names[i] = call_names[i];
            else
                names[i] = NULL;
        } else {
            names[i] = NULL;
        }
    }

    /* Program view pointing at the working copies for the comparator. */
    hx_program view = *prog;
    view.code = code;

    int changed = 1, guard = 0;
    while (changed && guard++ < 8) {
        changed = 0;
        for (int i = 0; i < n; i++) {
            if (code[i].op != OP_CALL) continue;
            if (code[i].u.call.nargs != 2) continue;
            const char *name = hx_cmp_call_name_at(&view,
                                                   (const char *const *)names, i);
            if (!hx_func_is_commutative(name)) continue;

            int o1s, o1e, o2s, o2e;
            if (!hx_cmp_find_two_operands(code, i, &o1s, &o1e, &o2s, &o2e))
                continue;

            int cmp = hx_cmp_operand_key(&view, (const char *const *)names,
                                         o1s, o1e, o2s, o2e);
            if (cmp <= 0) continue;     /* already canonical (o1 <= o2) */

            /* Swap operand1 and operand2 in BOTH arrays in lockstep:
             * [o1 block][o2 block] -> [o2 block][o1 block]. */
            int l1 = o1e - o1s, l2 = o2e - o2s;
            hx_inst *cscr = (hx_inst *)malloc((size_t)(l1 + l2) * sizeof(hx_inst));
            const char **nscr = (const char **)malloc((size_t)(l1 + l2) * sizeof(char *));
            if (!cscr || !nscr) {
                free(cscr); free(nscr); free(code); free(names);
                return 0;
            }
            memcpy(cscr,      &code[o2s], (size_t)l2 * sizeof(hx_inst));
            memcpy(cscr + l2, &code[o1s], (size_t)l1 * sizeof(hx_inst));
            memcpy(&code[o1s], cscr, (size_t)(l1 + l2) * sizeof(hx_inst));
            memcpy(nscr,      &names[o2s], (size_t)l2 * sizeof(char *));
            memcpy(nscr + l2, &names[o1s], (size_t)l1 * sizeof(char *));
            memcpy(&names[o1s], nscr, (size_t)(l1 + l2) * sizeof(char *));
            free(cscr); free(nscr);
            changed = 1;
        }
    }

    *out_code = code;
    *out_call_names = names;
    return 1;
}

/* ---------- the comparator ---------- */

/*
 * Compare two programs for "same algorithm" under Layer 1 + role-canon +
 * Layer 2 commutative reordering. call_names sidecars supply OP_CALL
 * names for catalog (entry==NULL) programs.
 */
static inline int programs_equal(const hx_program *a,
                                 const char *const *a_call_names,
                                 const hx_program *b,
                                 const char *const *b_call_names)
{
    if (!a || !b) return 0;
    if (a->ncode != b->ncode) return 0;

    /*
     * Layer 2: canonicalize commutative operand order on lockstep copies
     * of (code, call_names). After this, comparing acode[i]/aname[i] vs
     * bcode[i]/bname[i] index-for-index applies Layer 2 equivalence.
     * stridx survives the move untouched, so string lookups still use
     * the ORIGINAL prog->strings tables (a, b).
     */
    hx_inst *acode = NULL, *bcode = NULL;
    const char **aname = NULL, **bname = NULL;
    int a_canon = hx_cmp_canonicalize_code(a, a_call_names, &acode, &aname);
    int b_canon = hx_cmp_canonicalize_code(b, b_call_names, &bcode, &bname);

    /* OOM fallback: compare against original arrays (no Layer 2). */
    const hx_inst    *ac = a_canon ? acode : a->code;
    const hx_inst    *bc = b_canon ? bcode : b->code;
    const char *const *an = a_canon ? (const char *const *)aname : a_call_names;
    const char *const *bn = b_canon ? (const char *const *)bname : b_call_names;

    int equal = 1;
    for (int i = 0; i < a->ncode; i++) {
        const hx_inst *ia = &ac[i];
        const hx_inst *ib = &bc[i];
        if (ia->op != ib->op) { equal = 0; break; }

        if (ia->op == OP_PUSH_VAR || ia->op == OP_STORE || ia->op == OP_INC) {
            if (ia->u.slot != ib->u.slot) { equal = 0; break; }
        } else if (ia->op == OP_PUSH_STR) {
            int la = 0, lb = 0;
            const char *sa = hx_cmp_get_str(a, ia->u.stridx, &la);
            const char *sb = hx_cmp_get_str(b, ib->u.stridx, &lb);
            if (la != lb) { equal = 0; break; }
            if (la > 0 && memcmp(sa, sb, (size_t)la) != 0) { equal = 0; break; }
        } else if (ia->op == OP_PUSH_INT) {
            if (ia->u.ival != ib->u.ival) { equal = 0; break; }
        } else if (ia->op == OP_JUMP || ia->op == OP_JUMP_LE ||
                   ia->op == OP_JUMP_LT || ia->op == OP_JUMP_GT ||
                   ia->op == OP_JUMP_GE || ia->op == OP_JUMP_EQ ||
                   ia->op == OP_JUMP_NE) {
            if (ia->u.addr != ib->u.addr) { equal = 0; break; }
        } else if (ia->op == OP_CALL) {
            if (ia->u.call.nargs != ib->u.call.nargs) { equal = 0; break; }
            uint8_t ra = ia->u.call.role;
            uint8_t rb = ib->u.call.role;
            if (ra == ROLE_DEFAULT) {
                hx_program va = *a; va.code = acode ? acode : a->code;
                ra = hx_cmp_default_role_for_call(&va, an, i);
            }
            if (rb == ROLE_DEFAULT) {
                hx_program vb = *b; vb.code = bcode ? bcode : b->code;
                rb = hx_cmp_default_role_for_call(&vb, bn, i);
            }
            if (ra != rb) { equal = 0; break; }
            const char *na = an && an[i] ? an[i] :
                (ia->u.call.entry ? ia->u.call.entry->name : "?");
            const char *nb = bn && bn[i] ? bn[i] :
                (ib->u.call.entry ? ib->u.call.entry->name : "?");
            if (strcmp(na, nb) != 0) { equal = 0; break; }
        }
        /* OP_CONCAT / OP_HALT / OP_DUP / OP_POP: op-match is sufficient. */
    }

    free(acode); free(aname);
    free(bcode); free(bname);
    return equal;
}

#endif /* HX_PROGRAM_CMP_H */
