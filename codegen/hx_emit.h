/*
 * hx_emit.h -- internal header declaring per-backend emit helper
 *              signatures shared between the walker and the per-backend
 *              hx_emit_opencl.c / hx_emit_metal.c files.
 *
 * Both backends expose identical signatures so the walker dispatches
 * via a small switch (single walker, backend parameter, per spec
 * D12.3.a).
 *
 * Sub-phase 2a.2 adds per-opcode emit helpers for the minimum-viable
 * opcode set (PUSH_VAR/PUSH_STR/PUSH_INT/STORE/CALL[md5]/CONCAT/HALT)
 * needed to walk a trivial e1 MD5 hx_program end-to-end and produce
 * a JIT-compilable OpenCL source string.
 *
 * Sub-phase 2a.3 (2026-05-21): adds hx_emit_e347_md5md5md5salt_opencl
 * declaration for the pattern-recognized fast path. Metal twin lands
 * in 2a.4.
 *
 * Sub-phase 2a.4 (2026-05-21): adds hx_emit_e347_md5md5md5salt_metal
 * declaration for the Metal-backend twin of the pattern-recognized fast
 * path. Structural mirror of the OpenCL emitter; computes the SAME
 * algorithm chain MD5(hex32(MD5(MD5(pass))) || salt). 2a.5 validates
 * both backends against the hashpipe CPU oracle.
 *
 * Sub-phase 5a.2 (2026-05-22): adds hx_emit_family_md5pass_opencl
 * declaration for the MAKE_MD5PASS family emitter (30 algorithms; 8
 * GPU-eligible in 5a; 22 deferred to 5b). Takes `entry` for per-program
 * callnames-sidecar access. SHA1 outer (e161) is the first end-to-end
 * primitive; other 7 5a-supported primitives are 5a.4 scope.
 *
 * Sub-phase 5a.3 (2026-05-22): adds hx_emit_family_md5pass_metal
 * declaration -- Metal twin of the 5a.2 OpenCL family emitter. Same
 * structural shape and entry/spec contract; differs only in token
 * translation (device/thread address-space qualifiers, atomic_uint
 * counters, [[buffer(N)]] attribute binding, sequential MSL kernel
 * signature). SHA1 outer body MUST include the BE-to-LE state byte-swap
 * per the feedback memo. 5a.3 ships SHA1 only (e161); 5a.4 adds the
 * other 7 5a-supported primitives across both backends symmetrically.
 *
 * $Revision: 1.6 $
 * $Log: hx_emit.h,v $
 * Revision 1.6  2026/05/23 03:21:08  dlr
 * sub-phase 5a.3 add hx_emit_family_md5pass_metal declaration mirror of OpenCL 5a.2 twin same return contract structurally identical token translations device address space thread qualifier on out-state pointers atomic_uint counters sequential buffer attribute binding SHA1 outer body MUST include BE-to-LE state byte-swap per feedback memo only SHA1 ships in 5a.3 other 5a-supported primitives FATAL with deferred-to-5a4 diagnostic
 *
 * Revision 1.5  2026/05/23 02:02:48  dlr
 * sub-phase 5a.2 add hx_emit_family_md5pass_opencl decl plus forward decl of struct hx_spec_entry; family emitters take entry to reach per-program _hx_callnames_NNN sidecar via hx_callname_for_entry
 *
 *
 */

#ifndef HX_EMIT_H
#define HX_EMIT_H

#include <stddef.h>
#include "hx_spec.h"
#include "hx_walker.h"

/* Forward decl -- full def in hx_spec_entry.h. Used by 5a.2+ family
 * emitters to reach the per-program _hx_callnames_NNN[] sidecar via
 * hx_callname_for_entry(entry, code_idx). */
struct hx_spec_entry;

#ifdef __cplusplus
extern "C" {
#endif

/* ---- skeleton emit helpers (2a.1) ---- */

int hx_emit_kernel_attribute_opencl(char **out, size_t *out_cap,
                                    size_t *out_len);
int hx_emit_address_space_global_opencl(char **out, size_t *out_cap,
                                        size_t *out_len);
int hx_emit_thread_id_load_opencl(char **out, size_t *out_cap,
                                  size_t *out_len,
                                  const char *var_name);
int hx_emit_atomic_inc_opencl(char **out, size_t *out_cap, size_t *out_len,
                              const char *counter_expr);
int hx_emit_payload_load_opencl(char **out, size_t *out_cap, size_t *out_len);

int hx_emit_kernel_attribute_metal(char **out, size_t *out_cap,
                                   size_t *out_len);
int hx_emit_address_space_global_metal(char **out, size_t *out_cap,
                                       size_t *out_len);
int hx_emit_thread_id_load_metal(char **out, size_t *out_cap,
                                 size_t *out_len,
                                 const char *var_name);
int hx_emit_atomic_inc_metal(char **out, size_t *out_cap, size_t *out_len,
                             const char *counter_expr);
int hx_emit_payload_load_metal(char **out, size_t *out_cap, size_t *out_len);

/* ---- per-opcode emit helpers (2a.2) ---- */

/* Opcode emit signature: append source for a single opcode's effect.
 * Returns 0 on success, negative on emit error.
 *
 * For 2a.2, helpers emit minimum-viable scaffolding -- a comment
 * recording the opcode and operand plus a placeholder C declaration
 * sufficient for the JIT compiler to accept the file. Real semantics
 * arrive in 2a.3 (tp0 pattern for e347).
 */
int hx_emit_push_var_opencl(char **out, size_t *cap, size_t *len,
                            int slot, const char *varname);
int hx_emit_push_str_opencl(char **out, size_t *cap, size_t *len,
                            int stridx, const char *literal, int literal_len);
int hx_emit_push_int_opencl(char **out, size_t *cap, size_t *len,
                            int64_t ival);
int hx_emit_store_opencl(char **out, size_t *cap, size_t *len,
                         int slot, const char *varname);
int hx_emit_call_opencl(char **out, size_t *cap, size_t *len,
                        const char *fn_name, int nargs, uint8_t role);
int hx_emit_concat_opencl(char **out, size_t *cap, size_t *len);
int hx_emit_halt_opencl(char **out, size_t *cap, size_t *len);

int hx_emit_push_var_metal(char **out, size_t *cap, size_t *len,
                           int slot, const char *varname);
int hx_emit_push_str_metal(char **out, size_t *cap, size_t *len,
                           int stridx, const char *literal, int literal_len);
int hx_emit_push_int_metal(char **out, size_t *cap, size_t *len,
                           int64_t ival);
int hx_emit_store_metal(char **out, size_t *cap, size_t *len,
                        int slot, const char *varname);
int hx_emit_call_metal(char **out, size_t *cap, size_t *len,
                       const char *fn_name, int nargs, uint8_t role);
int hx_emit_concat_metal(char **out, size_t *cap, size_t *len);
int hx_emit_halt_metal(char **out, size_t *cap, size_t *len);

/* ---- pattern-recognized fast paths (2a.3) ----
 *
 * E347 (MD5MD5SALT, "md5(md5(md5(pass)) . salt)") OpenCL emitter.
 * Walks the recognized 7-op shape and emits a tp0-pattern kernel B
 * (per-thread serial SALT_BATCH=64 loop, register-held pre-state,
 * md5_buf_global for the inner MD5s, salt_pack_uint-equivalent for the
 * outer salt-concat, compact_fp probe + hit emit). Structurally mirrors
 * gpu/gpu_kernelb_md5md5salt_nocache.cl but computes the e347 chain
 * (which differs from that hand-written kernel's md5(md5_bin(pass) .
 * salt) chain by the hex32 encoding between inner MD5s).
 *
 * On entry: *out may be NULL (allocates), *out_cap is current alloc.
 * On success: *out is null-terminated source, *out_cap is alloc size,
 * return value 0. On error: negative; caller frees *out either way.
 *
 * Sub-phase 2a.4 (Metal twin) ports this to hx_emit_e347_*_metal.
 */
int hx_emit_e347_md5md5md5salt_opencl(
    char **out, size_t *out_cap,
    const hx_program *prog,
    const struct hx_specialization *spec);

/* Sub-phase 2a.4 (2026-05-21): Metal twin. Same return contract; emitted
 * source is JIT-compiled via gpu_metal_jit_compile_source_with_common()
 * which prepends metal_common_str. The two emitters compute the same
 * algorithm chain. */
int hx_emit_e347_md5md5md5salt_metal(
    char **out, size_t *out_cap,
    const hx_program *prog,
    const struct hx_specialization *spec);

/* ---- pattern-recognized family emitters (5a.2+) ----
 *
 * MAKE_MD5PASS family emitter for the canonical 6-op shape
 * (PUSH_VAR pass / CALL md5 / PUSH_VAR pass / CONCAT / CALL outer /
 * HALT). 30 algorithms in hx.8; 8 GPU-eligible in 5a (md4/md5/sha1/
 * sha224/sha256/sha384/sha512/rmd160 outer-primitive) -- the other 22
 * are 5b territory (need gpu_common.cl additions).
 *
 * Per-primitive emit is dispatched via hx_emit_primitives.h. 5a.2 ships
 * the SHA1 (e161) per-primitive body only; the other 7 5a-supported
 * primitives are 5a.4 scope and FATAL with a "deferred to 5a.4"
 * diagnostic if dispatched in this sub-phase.
 *
 * The emitter takes `entry` (NOT just `prog`) because resolving the
 * outer-call primitive name requires the per-program `_hx_callnames_*[]`
 * sidecar that's hung off entry->call_names; with only the program in
 * hand the emitter would have to do an O(N) reverse-lookup of
 * hx_specs_data[].
 *
 * On entry: *out may be NULL (allocates), *out_cap is current alloc.
 * On success: *out is null-terminated source, *out_cap is alloc size,
 * return value 0. On error: negative; caller frees *out either way.
 * Per feedback_external_failures_are_fatal.md the emitter exits(1) on
 * unrecoverable failures (e.g. callname missing, primitive not in 5a
 * set) -- those are NOT recoverable; production dispatcher rejects.
 */
int hx_emit_family_md5pass_opencl(
    char **out, size_t *out_cap,
    const hx_program *prog,
    const struct hx_specialization *spec,
    const struct hx_spec_entry *entry);

/* Sub-phase 5a.3 (2026-05-22): Metal twin of hx_emit_family_md5pass_opencl.
 * Same return contract; emitted source is JIT-compiled via
 * gpu_metal_jit_compile_source_with_common_keep which prepends
 * metal_common_str. Token translations: __global -> device, uint* state
 * passed to helpers becomes thread uint*, atomics typed device
 * atomic_uint*. SHA1 outer body MUST include BE-to-LE state byte-swap
 * (per feedback_be_state_primitives_need_byteswap_in_codegen.md). 5a.3
 * ships SHA1 only; other 5a-supported primitives FATAL with the same
 * "deferred to 5a.4" diagnostic the OpenCL twin emits. */
int hx_emit_family_md5pass_metal(
    char **out, size_t *out_cap,
    const hx_program *prog,
    const struct hx_specialization *spec,
    const struct hx_spec_entry *entry);

#ifdef __cplusplus
}
#endif

#endif /* HX_EMIT_H */
