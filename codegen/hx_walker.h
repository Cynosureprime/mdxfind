/*
 * hx_walker.h -- public API for the hx P4 state-machine walker.
 *
 * Sub-phase 2a.2 (per project_hx_codegen_phase2_3_spec_2026-05-21.md
 * D12.2.c REVISED-AGAIN -- Path A): the walker now consumes the
 * production VM `hx_program` directly (from hx_vm.h). The 2a.1 walker
 * took an invented `struct hx_spec` carrying invented bytecode; that
 * input shape is GONE.
 *
 * hx_emit_kernel walks an hx_program's instruction stream under a
 * given specialization context and emits a complete OpenCL or Metal
 * kernel source string into the caller-supplied dynamic buffer. The
 * buffer is grown via realloc as needed; caller frees it.
 *
 * hx_dump_source writes the emitted source verbatim to the path
 * named by env_var_name (if that env var is set). Idempotent if the
 * env var is unset.
 *
 * Return codes: 0 = success; negative = walker error (unimplemented
 * opcode for sub-phase 2a.2, unrecognized function CALL for 2a.2's
 * minimal helper set, etc.). External runtime failures (JIT) live in
 * the CALLER, not the walker -- per feedback_external_failures_are_fatal.md
 * any clBuildProgram or Metal newLibraryWithSource: failure must
 * exit(1) with full diagnostic at the call site, NOT propagate through
 * the walker.
 *
 * Sub-phase 2a.3 (2026-05-21): hx_emit_kernel now dispatches via
 * hx_detect_pattern (codegen/hx_patterns.h) before the per-opcode
 * generic walk. Implemented patterns short-circuit to specialized
 * emitters (e347 OpenCL); unimplemented patterns fall through.
 *
 * Sub-phase 5a.2 (2026-05-22): hx_emit_kernel takes a fourth argument
 * `const struct hx_spec_entry *entry` so family emitters (5a.2+) can
 * reach the per-program _hx_callnames_NNN[] sidecar via
 * hx_callname_for_entry(entry, code_idx). All existing callers updated
 * to pass the entry they obtained from hx_specs_lookup. NULL is allowed
 * for callers without an entry (test harnesses); patterns that need the
 * sidecar FATAL on NULL.
 *
 * $Revision: 1.4 $
 * $Log: hx_walker.h,v $
 * Revision 1.4  2026/05/23 02:02:34  dlr
 * sub-phase 5a.2 add HX_PATTERN_FAMILY_MD5PASS dispatch arm forwards prog spec and entry to hx_emit_family_md5pass_opencl when backend is OpenCL with FATAL on NULL entry per per-program callnames sidecar requirement; Metal arm still falls through to per-opcode generic until 5a.3 Metal twin lands; walker signature gains entry parameter so family emitters can reach the sidecar without O of N reverse lookup over hx_specs_data
 *
 *
 */

#ifndef HX_WALKER_H
#define HX_WALKER_H

#include <stddef.h>
#include "hx_spec.h"

/* Forward decl. Defined in hx_spec_entry.h. 5a.2 added to walker
 * signature so family emitters can reach the per-program callnames
 * sidecar via hx_callname_for_entry(entry, idx). */
struct hx_spec_entry;

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Emit a kernel source string for (program, specialization, backend).
 *
 * On entry: *out may be NULL (walker allocates); *out_cap is the current
 * allocated size of *out (0 if *out is NULL). On success: *out points to
 * a null-terminated source string, *out_cap is its allocated size, and
 * the return value is 0. On error: returns negative, *out and *out_cap
 * may have partial content (caller should free *out either way).
 *
 * Caller is responsible for free()ing *out.
 *
 * Sub-phase 2a.2 supports a minimum-viable opcode set:
 *   OP_PUSH_VAR, OP_PUSH_STR, OP_PUSH_INT, OP_STORE, OP_CALL (md5 only),
 *   OP_CONCAT, OP_HALT. Control-flow ops (OP_JUMP*, OP_INC, OP_DUP,
 *   OP_POP) and CALL targets other than md5 return -1 with an embedded
 *   `// hx: ... not yet implemented` comment in the emitted source so
 *   the failure is visible on dump. Real impl arrives in 2a.3+.
 */
/* Sub-phase 5a.2 (2026-05-22): signature now takes `entry` so family
 * emitters can reach the per-program _hx_callnames_NNN[] sidecar via
 * hx_callname_for_entry(entry, code_idx). `entry` MAY be NULL when the
 * caller doesn't have one in hand (e.g. ad-hoc test programs compiled
 * via hx_compile_expr); patterns that require the sidecar (i.e. all
 * family patterns) FATAL on NULL with a clean diagnostic. The legacy
 * pre-5a.2 single-pattern e347 emitter does NOT need entry. */
int hx_emit_kernel(const hx_program *prog,
                   const struct hx_specialization *spec,
                   enum hx_backend backend,
                   const struct hx_spec_entry *entry,
                   char **out, size_t *out_cap);

/*
 * If getenv(env_var_name) is set, write src to that path verbatim.
 * Returns 0 on success (including the "env var unset" case), negative
 * on I/O error.
 */
int hx_dump_source(const char *src, const char *env_var_name);

/*
 * printf-style appender that grows *out as needed. Exposed so the
 * per-backend emit helper files can share the same growth strategy
 * as the walker proper.
 *
 * Returns 0 on success, negative on allocation failure (caller treats
 * this as fatal; in practice realloc returning NULL is exit(1)
 * territory at the walker level too).
 */
int hx_appendf(char **out, size_t *out_cap, size_t *out_len,
               const char *fmt, ...)
#ifdef __GNUC__
    __attribute__((format(printf, 4, 5)))
#endif
    ;

#ifdef __cplusplus
}
#endif

#endif /* HX_WALKER_H */
