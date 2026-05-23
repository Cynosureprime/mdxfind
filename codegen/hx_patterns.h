/*
 * hx_patterns.h -- pattern detector for the hx P4 walker.
 *
 * Sub-phase 2a.3 (per project_hx_codegen_phase2_3_spec_2026-05-21.md
 * §6 Phase 3 e347 target specifics): the walker has two complementary
 * emission paths:
 *
 *   (1) Per-opcode generic dispatch (2a.2 placeholders) -- adequate for
 *       simple algorithms that don't need hand-tuning.
 *   (2) Pattern-recognized fast paths (THIS FILE) -- the walker scans
 *       prog->code[] for known bytecode shapes and dispatches to a
 *       specialized emitter that produces a tp0-equivalent kernel.
 *
 * Pattern detection is a pure read-only walk over the hx_program; it
 * does NOT consume the bytecode. The walker's main dispatch first calls
 * hx_detect_pattern() and, on a non-UNKNOWN return AND a matching
 * backend, invokes the specialized emitter. Otherwise it falls through
 * to the per-opcode generic dispatch.
 *
 * For 2a.3 only the e347 (MD5MD5SALT, expression
 * "md5(md5(md5(pass)) . salt)") pattern is recognized. Future
 * sub-phases extend the enum and add detectors for additional hand-tuned
 * shapes.
 *
 * Sub-phase 5a.1 (2026-05-22): pattern framework upgrade. Adds the
 * HX_PATTERN_FAMILY_MD5PASS family pattern (30 algorithms eN where
 * outer in {md2,md4,md5,gost,hav*,rmd128,rmd160,sha1,sha224,sha256,
 * sha384,sha512,tiger,wrl,sne128,sne256}), table-driven dispatch
 * inside hx_detect_pattern, and the hx_callname_for_entry accessor
 * for per-call-site name lookup. Detector ships in 5a.1; per-primitive
 * emitter is sub-phase 5a.2.
 *
 * $Revision: 1.2 $
 * $Log: hx_patterns.h,v $
 * Revision 1.2  2026/05/22 23:52:27  dlr
 * sub-phase 5a.1 add HX_PATTERN_FAMILY_MD5PASS enum + hx_callname_for_entry accessor declaration with forward decl of struct hx_spec_entry
 *
 * Revision 1.1  2026/05/22 02:09:59  dlr
 * sub-phase 2a.3 initial pattern detector header. Implements HX_PATTERN_E347_MD5MD5MD5SALT only for now. Future patterns append to the enum.
 *
 *
 */

#ifndef HX_PATTERNS_H
#define HX_PATTERNS_H

#include "../hx_vm.h"

/* Forward decl -- full definition in hx_spec_entry.h. Used only by
 * hx_callname_for_entry below; consumers that need the full struct
 * include hx_spec_entry.h or hx_spec.h directly. */
struct hx_spec_entry;

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Pattern identifiers. Append-only; consumers may switch on this.
 *
 * Bytecode shape currently recognized for HX_PATTERN_E347_MD5MD5MD5SALT
 * (matches _hx_program_346 verbatim, ncode=7):
 *
 *   [0] OP_PUSH_VAR slot=0 (pass)
 *   [1] OP_CALL    md5  nargs=1
 *   [2] OP_CALL    md5  nargs=1
 *   [3] OP_PUSH_VAR slot=1 (salt)
 *   [4] OP_CONCAT
 *   [5] OP_CALL    md5  nargs=1
 *   [6] OP_HALT
 *
 * Note that this is TWO inner CALL md5 instructions, not three; the
 * hx compiler renders "md5(md5(md5(pass)) . salt)" with two inner MD5s
 * followed by the outer salt-concat MD5. The HEX32 expansion between
 * the two inner CALLs is implicit in the hx VM's CONCAT semantics
 * (binary-digest stack values stringify before concatenation).
 */
typedef enum {
    HX_PATTERN_UNKNOWN = 0,
    HX_PATTERN_E347_MD5MD5MD5SALT = 1,
    /* Sub-phase 5a.1 (2026-05-22): family pattern for MAKE_MD5PASS
     * (30 algorithms; bytecode `outer(md5(pass).pass)`; canonical 6-op
     * shape PUSH_VAR(pass)/CALL md5/PUSH_VAR(pass)/CONCAT/CALL outer/
     * HALT). Detector verifies structure only; per-algorithm dispatch
     * resolves the outer-CALL name via hx_callname_for_entry at
     * emitter time. Real emitter lands sub-phase 5a.2. */
    HX_PATTERN_FAMILY_MD5PASS = 2
    /* Future:
     * HX_PATTERN_E31_MD5MD5SALT,
     * HX_PATTERN_E386_SHA512PASSSALT,
     * ... */
} hx_pattern_id;

/* Return the pattern id matching prog, or HX_PATTERN_UNKNOWN if none.
 * Pure function over the bytecode; does not mutate prog. */
hx_pattern_id hx_detect_pattern(const hx_program *prog);

/* Diagnostic name for an hx_pattern_id (for stderr / dump annotations). */
const char *hx_pattern_name(hx_pattern_id id);

/*
 * Sub-phase 5a.1 (2026-05-22): per-call-site function-name accessor.
 *
 * Each compiled hx_program has a sidecar `_hx_callnames_NNN[]` table
 * emitted alongside it in codegen/hx_specs_data.c. The auto-generated
 * `hx_spec_entry` now carries a pointer to this table via .call_names;
 * the accessor below returns the function name (e.g. "md5", "sha256",
 * "tiger") at the requested code index, or NULL when:
 *   - entry is NULL
 *   - entry->call_names is NULL (no sidecar)
 *   - code_idx out of bounds (<0 or >= entry->program->ncode)
 *   - the opcode at code_idx is not OP_CALL (sidecar slot is NULL)
 *
 * Used by family emitters (5a.2+) to dispatch per-primitive on the
 * outer-call opcode (e.g. code[4] for MAKE_MD5PASS). The accessor takes
 * the entry directly (not just the program) to avoid an O(N) reverse-
 * lookup; callers already hold the entry via hx_specs_lookup.
 */
const char *hx_callname_for_entry(const struct hx_spec_entry *entry,
                                  int code_idx);

#ifdef __cplusplus
}
#endif

#endif /* HX_PATTERNS_H */
