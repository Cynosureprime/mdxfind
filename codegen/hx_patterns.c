/*
 * hx_patterns.c -- pattern detector for the hx P4 walker.
 *
 * Sub-phase 2a.3 ships a single recognizer:
 * HX_PATTERN_E347_MD5MD5MD5SALT, matching the 7-op shape produced by
 * the hx compiler for `md5(md5(md5(pass)) . salt)`. See hx_patterns.h
 * for the exact bytecode shape.
 *
 * Sub-phase 5a.1 (2026-05-22) framework upgrade:
 *   - Replaces the hand-coded if-chain in hx_detect_pattern with a
 *     table-driven dispatch over hx_pattern_table[]. More-specific
 *     patterns precede more-general; first-match-wins.
 *   - Adds HX_PATTERN_FAMILY_MD5PASS recognizer: matches the canonical
 *     MAKE_MD5PASS 6-op shape (PUSH_VAR pass / CALL md5 / PUSH_VAR pass
 *     / CONCAT / CALL outer / HALT). Detector is structural-only; the
 *     emitter validates code[1] callname == "md5" and resolves the
 *     outer-CALL primitive via hx_callname_for_entry(entry, 4).
 *   - Implements hx_callname_for_entry(entry, code_idx): reads
 *     entry->call_names[code_idx] with NULL/bounds guards.
 *
 * The call-name lookup for an OP_CALL pulls from the auto-generated
 * sidecar `_hx_callnames_NNN[]` table where present. The hx_spec_entry
 * struct carries .call_names pointing at the per-program sidecar
 * (extension 5a.1); the accessor below takes the entry directly to
 * avoid an O(N) reverse-lookup over hx_specs_data[].
 *
 * For 5a.1 the family pattern's structural-only check is sufficient
 * because the emitter (5a.2) validates names before dispatching to
 * per-primitive helpers; a non-family bytecode shape that happens to
 * pass the 6-op structural test (e.g. some hypothetical `xor(a(pass),
 * b(pass).pass)` variant) would be rejected by the emitter's name
 * verification.
 *
 * $Revision: 1.2 $
 * $Log: hx_patterns.c,v $
 * Revision 1.2  2026/05/22 23:52:30  dlr
 * sub-phase 5a.1 replace single-recognizer dispatch with table-driven hx_pattern_table; add matches_family_md5pass detector for 30-member MAKE_MD5PASS family; implement hx_callname_for_entry reading entry sidecar with NULL and bounds guards
 *
 * Revision 1.1  2026/05/22 02:09:59  dlr
 * sub-phase 2a.3 initial pattern detector implementation. Walks prog->code structurally checking opcodes and slot indices. Returns HX_PATTERN_UNKNOWN if no recognized shape matches.
 *
 *
 */

#include <stddef.h>
#include "hx_patterns.h"
#include "hx_spec_entry.h"

const char *hx_pattern_name(hx_pattern_id id)
{
    switch (id) {
    case HX_PATTERN_UNKNOWN:
        return "UNKNOWN";
    case HX_PATTERN_E347_MD5MD5MD5SALT:
        return "E347_MD5MD5MD5SALT";
    case HX_PATTERN_FAMILY_MD5PASS:
        return "FAMILY_MD5PASS";
    default:
        return "???";
    }
}

/*
 * Detect HX_PATTERN_E347_MD5MD5MD5SALT shape.
 *
 * Structural requirements:
 *   ncode == 7
 *   code[0].op == OP_PUSH_VAR && code[0].u.slot == HX_SLOT_PASS (=0)
 *   code[1].op == OP_CALL     && code[1].u.call.nargs == 1
 *   code[2].op == OP_CALL     && code[2].u.call.nargs == 1
 *   code[3].op == OP_PUSH_VAR && code[3].u.slot == HX_SLOT_SALT (=1)
 *   code[4].op == OP_CONCAT
 *   code[5].op == OP_CALL     && code[5].u.call.nargs == 1
 *   code[6].op == OP_HALT
 *
 * No string or integer literals required. nvars >= 2 (pass + salt).
 */
static int matches_e347(const hx_program *prog)
{
    if (!prog || !prog->code || prog->ncode != 7) return 0;
    if (prog->nvars < 2) return 0;

    const hx_inst *c = prog->code;

    if (c[0].op != OP_PUSH_VAR || c[0].u.slot != HX_SLOT_PASS) return 0;
    if (c[1].op != OP_CALL     || c[1].u.call.nargs != 1)      return 0;
    if (c[2].op != OP_CALL     || c[2].u.call.nargs != 1)      return 0;
    if (c[3].op != OP_PUSH_VAR || c[3].u.slot != HX_SLOT_SALT) return 0;
    if (c[4].op != OP_CONCAT)                                   return 0;
    if (c[5].op != OP_CALL     || c[5].u.call.nargs != 1)      return 0;
    if (c[6].op != OP_HALT)                                     return 0;

    return 1;
}

/*
 * Sub-phase 5a.1 (2026-05-22): detect HX_PATTERN_FAMILY_MD5PASS shape.
 *
 * Structural requirements (canonical MAKE_MD5PASS 6-op):
 *   ncode == 6
 *   code[0].op == OP_PUSH_VAR && code[0].u.slot == HX_SLOT_PASS (=0)
 *   code[1].op == OP_CALL     && code[1].u.call.nargs == 1   (inner md5)
 *   code[2].op == OP_PUSH_VAR && code[2].u.slot == HX_SLOT_PASS (=0)
 *   code[3].op == OP_CONCAT
 *   code[4].op == OP_CALL     && code[4].u.call.nargs == 1   (outer)
 *   code[5].op == OP_HALT
 *
 * Detector does NOT verify callnames. Verified against all 30 family
 * members (eidx 119,121,122,124,126,128,130,132,134,136,138,140,142,
 * 144,146,148,150,152,154,156,158,160,162,164,166,168,170,172,174,176)
 * in codegen/hx_specs_data.c -- every entry matches this shape.
 *
 * No string or integer literals required. nvars >= 1 (pass).
 *
 * The emitter (5a.2) MUST validate code[1] callname == "md5" and
 * code[4] callname is in the supported-primitive set for the chosen
 * backend before emitting per-primitive bodies.
 */
static int matches_family_md5pass(const hx_program *prog)
{
    if (!prog || !prog->code || prog->ncode != 6) return 0;
    if (prog->nvars < 1) return 0;

    const hx_inst *c = prog->code;

    if (c[0].op != OP_PUSH_VAR || c[0].u.slot != HX_SLOT_PASS) return 0;
    if (c[1].op != OP_CALL     || c[1].u.call.nargs != 1)      return 0;
    if (c[2].op != OP_PUSH_VAR || c[2].u.slot != HX_SLOT_PASS) return 0;
    if (c[3].op != OP_CONCAT)                                   return 0;
    if (c[4].op != OP_CALL     || c[4].u.call.nargs != 1)      return 0;
    if (c[5].op != OP_HALT)                                     return 0;

    return 1;
}

/*
 * Sub-phase 5a.1 (2026-05-22): table-driven pattern dispatch.
 *
 * Ordering rule: more-specific patterns FIRST. E347's 7-op shape and
 * the family's 6-op shape do not collide on ncode, so ordering is not
 * load-bearing yet -- but the convention matters for future patterns
 * (e.g. an e508 MD5MD5PASSSHA1 8-op recognizer that shares the family's
 * first 4 ops would need to precede FAMILY_MD5PASS in the table to
 * avoid false matches on the broader shape).
 *
 * Adding a new pattern: append a row, implement the matches_*
 * recognizer above, and update hx_pattern_name's switch.
 */
static const struct {
    hx_pattern_id id;
    int          (*match)(const hx_program *prog);
    const char    *name;
} hx_pattern_table[] = {
    { HX_PATTERN_E347_MD5MD5MD5SALT, matches_e347,           "E347_MD5MD5MD5SALT" },
    { HX_PATTERN_FAMILY_MD5PASS,     matches_family_md5pass, "FAMILY_MD5PASS"     },
};

hx_pattern_id hx_detect_pattern(const hx_program *prog)
{
    if (!prog) return HX_PATTERN_UNKNOWN;
    for (size_t i = 0;
         i < sizeof(hx_pattern_table) / sizeof(hx_pattern_table[0]);
         i++) {
        if (hx_pattern_table[i].match(prog))
            return hx_pattern_table[i].id;
    }
    return HX_PATTERN_UNKNOWN;
}

/*
 * Sub-phase 5a.1 (2026-05-22): per-call-site callname accessor.
 *
 * Returns the function name at code_idx (e.g. "md5", "sha256") or
 * NULL when:
 *   - entry is NULL
 *   - entry->call_names is NULL (outlier / compile_failed / generator
 *     did not populate the sidecar)
 *   - entry->program is NULL (outlier / compile_failed)
 *   - code_idx out of bounds (<0 or >= entry->program->ncode)
 *
 * Callers that need the name MUST treat NULL as fatal-or-unknown per
 * their dispatch contract; this accessor does not validate that
 * code[code_idx].op == OP_CALL (the sidecar slot is already NULL for
 * non-CALL positions per the generator's emission rule).
 */
const char *hx_callname_for_entry(const struct hx_spec_entry *entry,
                                  int code_idx)
{
    if (!entry) return NULL;
    if (!entry->call_names) return NULL;
    if (!entry->program) return NULL;
    if (code_idx < 0 || code_idx >= entry->program->ncode) return NULL;
    return entry->call_names[code_idx];
}
