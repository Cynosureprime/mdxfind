/*
 * hx_spec_entry.h -- wrapper struct for compiled hx programs in the
 *                    auto-generated codegen/hx_specs_data.c array.
 *
 * Each entry in hx_specs_data[] pairs an hx algorithm row from hx.8
 * (job enum + canonical name + original expression text) with the
 * pre-compiled `hx_program` produced by the build-time tool
 * tools/hx8_to_c at make time. Entries marked is_outlier=1 (markup
 * in the EXPRESSION column instead of a real hx expression) carry
 * NULL .program and route to hx_override_table[] at codegen-dispatch
 * time. Entries marked compile_failed=1 (hx_compile_expr returned NULL
 * for a non-outlier expression) likewise carry NULL .program and route
 * to override. The codegen walker only consumes .program when both
 * flags are zero.
 *
 * Sub-phase 2a.2 (per project_hx_codegen_phase2_3_spec_2026-05-21.md
 * §9 D12.2.c REVISED AGAIN -- Path A): mdxfind contains NO hx parser;
 * the array here is the only handoff between the build-time hx.8
 * conversion and the runtime codegen walker.
 *
 * Sub-phase 5a.1 (2026-05-22): adds .call_names sidecar pointer to
 * each entry so codegen emitters can resolve per-call-site function
 * names via hx_callname_for_entry. tools/hx8_to_c emits the
 * initializer pointing at the per-program `_hx_callnames_NNN[]`
 * static array. Outliers and compile-fail entries carry NULL.
 *
 * 1.1
 * hx_spec_entry.h,v
 * Revision 1.1  2026/05/21 23:22:54  dlr
 * sub-phase 2a.2: hx_spec_entry wrapper struct that pairs hx.8 algorithm rows (job_enum, name, expression, hx8_line) with the compiled hx_program. Outliers and compile-fail entries carry NULL program and route to override at codegen-dispatch time. Auto-generated codegen/hx_specs_data.c declares the array.
 *
 */

#ifndef HX_SPEC_ENTRY_H
#define HX_SPEC_ENTRY_H

#include <stddef.h>
#include <stdint.h>
#include "../hx_vm.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Sub-phase 5c.1 (2026-05-27): multi-emit annotation.
 *
 * Most MAKE_MD5PASS family members produce ONE digest per password
 * (the canonical hash); they are HX_EMIT_SINGLE. A small "Note [N]"
 * class in hx.8 produces MORE THAN ONE digest per password, each
 * probed against the loaded hash table as an independent found-hash
 * candidate (the CPU calls checkhash() once per variant). e123
 * MD5MD5PASS is the proof-of-concept: it emits BOTH
 *   variant 0 (canonical): md5( hex32(md5(pass)) . pass )
 *   variant 1 (colon):      md5( hex32(md5(pass)) . ':' . pass )
 * The codegen family emitter consults emit_class to decide whether to
 * emit one probe+EMIT_HIT_4 block (SINGLE) or a compile-time-N
 * unrolled set of probe+EMIT_HIT_4 blocks (MULTI). The dedup machinery
 * (EMIT_HIT_4_DEDUP_OR_OVERFLOW keyed on the matched loaded-hash slot)
 * is ALREADY the correct multi-emit key -- no key widening; each
 * variant deduplicates against its own matched_idx.
 *
 * note_ref records the hx.8 "(see Note [N])" reference that drives the
 * variant set (24 = canonical+colon for e123). 0 = no Note reference.
 *
 * Default (C zero-init) is HX_EMIT_SINGLE (=0) / note_ref=0 for every
 * existing entry: the 29 prior family members + all non-family entries
 * take the single-emit body unchanged.
 */
enum hx_emit_class {
    HX_EMIT_SINGLE = 0,   /* one digest per password (default) */
    HX_EMIT_MULTI  = 1    /* N>1 digests per password (Note-[N] class) */
};

/*
 * One entry per algorithm row in hx.8. The build-time tool emits a
 * static const array `hx_specs_data[]` of these into the auto-
 * generated codegen/hx_specs_data.c.
 */
struct hx_spec_entry {
    int                 job_enum;       /* eN; matches mdxfind JOB_* */
    const char         *name;           /* canonical NAME (UPPER) */
    const char         *expression;     /* original hx text (NULL on outlier) */
    int                 hx8_line;       /* line in hx.8 for diagnostics */
    uint8_t             is_outlier;     /* 1 = troff markup, not real hx */
    uint8_t             compile_failed; /* 1 = hx_compile_expr returned NULL */
    const hx_program   *program;        /* compiled bytecode (NULL if either flag) */
    /*
     * Sub-phase 5a.1 (2026-05-22): pointer to the per-program callnames
     * sidecar array `_hx_callnames_NNN[]` (file-scope static in
     * hx_specs_data.c). Indexed by code-index; entries are the function-
     * name strings for OP_CALL opcodes (NULL for non-CALL positions).
     * NULL when either outlier or compile_failed. Used by codegen
     * emitters via hx_callname_for_entry(entry, idx) to look up the
     * function name at a specific call site (e.g. the outer-hash CALL
     * in MAKE_MD5PASS family kernels).
     */
    const char *const  *call_names;     /* per-code-index fn name sidecar */
    /*
     * Sub-phase 5c.1 (2026-05-27): multi-emit annotation (see enum
     * hx_emit_class above). The generator tools/hx8_to_c sets these by
     * parsing the "(see Note [N])" markup. Default-init zero =>
     * HX_EMIT_SINGLE / note_ref=0 for every existing entry.
     */
    int                 emit_class;     /* enum hx_emit_class */
    int                 note_ref;       /* hx.8 Note [N] ref; 0 = none */
};

/* Auto-generated table + count -- emitted into codegen/hx_specs_data.c. */
extern const struct hx_spec_entry hx_specs_data[];
extern const int                  hx_specs_count;

/* O(log N) or O(N) lookup helper. Returns NULL if no entry matches. */
const struct hx_spec_entry *hx_specs_lookup(int job_enum);

#ifdef __cplusplus
}
#endif

#endif /* HX_SPEC_ENTRY_H */
