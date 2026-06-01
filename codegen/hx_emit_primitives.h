/*
 * hx_emit_primitives.h -- per-primitive emit dispatch for the hx P4
 *                         codegen family emitters.
 *
 * Sub-phase 5a.2 (2026-05-22): the MAKE_MD5PASS family kernel emitter
 * dispatches on the outer-call primitive name (one of md5/md4/sha1/
 * sha224/sha256/sha384/sha512/rmd160 for the 5a-supported set, or one
 * of 22 deferred primitives that 5b will lift into gpu_common.cl).
 * This header centralizes the (name -> id -> digest size) mapping plus
 * the "is this primitive in 5a's supported set?" query so per-family
 * emitters share a single source of truth.
 *
 * 5a-supported set (8 primitives, all already present in gpu_common.cl /
 * metal_common.metal):
 *
 *   md4, md5, sha1, sha224, sha256, sha384, sha512, rmd160
 *
 * 5b-deferred set (22 primitives, gpu_common.cl additions needed):
 *
 *   md2, gost, gost_crypto, hav128_3/4/5, hav160_3/4/5, hav192_3/4/5,
 *   hav224_3/4/5, hav256_3/4/5, rmd128, tiger, wrl, sne128, sne256
 *
 * The family emitter (hx_emit_family_md5pass_opencl, 5a.2; Metal twin
 * 5a.3) calls hx_primitive_id_for_name() on hx_callname_for_entry(
 * entry, 4) to identify the outer-hash primitive; hx_primitive_is_-
 * supported_5a() FATALs the unsupported branches with a clean
 * "deferred to 5b" diagnostic. hx_primitive_digest_bytes() returns the
 * output digest width so harness code (oracle + diff) can size buffers.
 *
 * Per feedback_external_failures_are_fatal.md these helpers never
 * silently return wrong answers; unknown names map to HX_PRIM_UNKNOWN
 * and the caller treats as fatal.
 *
 * $Revision: 1.3 $
 * $Log: hx_emit_primitives.h,v $
 * Revision 1.3  2026/05/28 14:32:17  dlr
 * Phase 1b Batch 1: add hx_primitive_for_unsalted_job + hx_primitive_is_unsalted_single helpers backed by a SEPARATE unsalted_job_table 5 rows e1 MD5 e3 MD4 e33 MD5RAW e34 SHA1RAW e36 SHA256RAW; distinct from job_to_prim_table family map no JOB enum in both; RAW variants map to same primitive as hex siblings; single source of truth for the unsalted admit predicate and emit-helper dispatch; header NOT included for link portability same rationale as job_to_prim_table
 *
 * Revision 1.2  2026/05/28 02:09:17  dlr
 * sub-phase 5b3a03 D17.4.b refactor add hx_primitive_for_job lookup helper plus hx_primitive_is_family_md5pass convenience wrapper header declares new APIs for the single source of truth that collapses 4 widening sites onto one truth source admit predicate plus _proto_hexlen switch in OpenCL plus _proto_hexlen path in Metal plus 2 mdxfind harness OR-chains future Tier 4 ships flip prim_table supported_5a flag only no per-site edits comment block enumerates the four widening sites and the new query pattern
 *
 * Revision 1.1  2026/05/23 02:02:25  dlr
 * sub-phase 5a.2 initial header for per-primitive emit dispatch table
 *
 *
 */

#ifndef HX_EMIT_PRIMITIVES_H
#define HX_EMIT_PRIMITIVES_H

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Primitive identifiers. Append-only; consumers may switch on this.
 *
 * The 5a-supported set covers MD4/MD5/SHA1/SHA-2 family/RMD160 (all
 * already in gpu_common.cl + metal_common.metal). The 5b-deferred set
 * covers everything else seen in the MAKE_MD5PASS family expression
 * column of hx.8 — md2, gost variants, haval variants, rmd128, tiger,
 * whirlpool, snefru.
 */
enum hx_primitive_id {
    HX_PRIM_UNKNOWN = 0,
    /* 5a supported (currently present in gpu_common.cl) */
    HX_PRIM_MD5     = 1,
    HX_PRIM_MD4     = 2,
    HX_PRIM_SHA1    = 3,
    HX_PRIM_SHA224  = 4,
    HX_PRIM_SHA256  = 5,
    HX_PRIM_SHA384  = 6,
    HX_PRIM_SHA512  = 7,
    HX_PRIM_RMD160  = 8,
    /* 5b deferred (need gpu_common.cl additions) — enumerated so
     * dispatch can give a specific "deferred to 5b" diagnostic naming
     * the primitive. */
    HX_PRIM_MD2     = 9,
    HX_PRIM_RMD128  = 10,
    HX_PRIM_TIGER   = 11,
    HX_PRIM_WRL     = 12,        /* whirlpool */
    HX_PRIM_GOST    = 13,
    HX_PRIM_GOST_CRYPTO = 14,
    HX_PRIM_SNE128  = 15,        /* snefru-128 */
    HX_PRIM_SNE256  = 16,        /* snefru-256 */
    HX_PRIM_HAV128_3 = 17, HX_PRIM_HAV128_4 = 18, HX_PRIM_HAV128_5 = 19,
    HX_PRIM_HAV160_3 = 20, HX_PRIM_HAV160_4 = 21, HX_PRIM_HAV160_5 = 22,
    HX_PRIM_HAV192_3 = 23, HX_PRIM_HAV192_4 = 24, HX_PRIM_HAV192_5 = 25,
    HX_PRIM_HAV224_3 = 26, HX_PRIM_HAV224_4 = 27, HX_PRIM_HAV224_5 = 28,
    HX_PRIM_HAV256_3 = 29, HX_PRIM_HAV256_4 = 30, HX_PRIM_HAV256_5 = 31
};

/* Map an hx.8 / hx VM CALL name (e.g. "sha1", "rmd160") to a primitive
 * id. Returns HX_PRIM_UNKNOWN on NULL input or any unrecognized name.
 * Caller treats UNKNOWN as fatal. */
enum hx_primitive_id hx_primitive_id_for_name(const char *fn_name);

/* Canonical lowercase name string ("md5", "sha1", ...). Returns
 * "unknown" for HX_PRIM_UNKNOWN and out-of-range ids. */
const char *hx_primitive_name(enum hx_primitive_id id);

/* Output digest size in bytes for a primitive.
 *   MD4=16, MD5=16, MD2=16, RMD128=16
 *   SHA1=20, RMD160=20
 *   SHA224=28, HAV128_*=16, HAV160_*=20, HAV192_*=24, HAV224_*=28,
 *   HAV256_*=32
 *   SHA256=32, SNE256=32, GOST=32, GOST_CRYPTO=32, WRL=64
 *   SHA384=48
 *   SHA512=64
 *   TIGER=24
 *   SNE128=16
 * Returns 0 on HX_PRIM_UNKNOWN / out-of-range. */
int hx_primitive_digest_bytes(enum hx_primitive_id id);

/* Returns 1 if this primitive is in the 5a-supported set (currently
 * present in gpu_common.cl + metal_common.metal as a `*_block` function);
 * 0 otherwise. Family emitters FATAL on the 0 branch with a "deferred to
 * 5b" diagnostic naming the primitive. */
int hx_primitive_is_supported_5a(enum hx_primitive_id id);

/* Sub-phase 5b.3a.0.3 (2026-05-27) D17.4.b refactor:
 *
 * Map a MAKE_MD5PASS family JOB enum (e.g. JOB_SHA1MD5PASS = 161, JOB_-
 * HAV128MD5PASS = 127, etc.) to the corresponding outer-primitive id.
 * Returns HX_PRIM_UNKNOWN for any JOB enum that is NOT a MAKE_MD5PASS
 * family member.
 *
 * Single source of truth for the 4 widening sites that were previously
 * hand-coded per-family-member:
 *
 *   1. gpu_codegen_kernelb_family_md5pass_eligible() admit predicate
 *      (gpu/gpu_codegen_eligible.c)
 *   2. _proto_hexlen JOB switch (gpu/gpujob_opencl.c, gpu/gpujob_metal.m)
 *   3. mdxfind.c OpenCL harness OR-chain (line 39527 era)
 *   4. mdxfind.c Metal harness OR-chain (line 40037 era)
 *
 * After this refactor each site becomes a one-liner:
 *
 *   const enum hx_primitive_id pid = hx_primitive_for_job(op);
 *   if (pid != HX_PRIM_UNKNOWN && hx_primitive_is_supported_5a(pid)) ...
 *
 * Future Tier 4 ships only flip prim_table.supported_5a for the relevant
 * row -- no edits at any of the 4 sites. */
enum hx_primitive_id hx_primitive_for_job(int job_enum);

/* Returns 1 if this primitive is a member of the MAKE_MD5PASS family
 * (regardless of supported_5a state). Convenience wrapper for callers
 * that want to ask "is this a family member?" without enumerating all
 * 30 family JOB enums. Equivalent to:
 *   hx_primitive_for_job(op) != HX_PRIM_UNKNOWN.
 *
 * The MAKE_MD5PASS family consists of 30 members: e120 e122 e123
 * (deferred multi-emit) plus e127/e129/e131/e133/e135/e137/e139/e141/-
 * e143/e145/e147/e149/e151/e153/e155 (15 HAV variants) plus e157
 * e159 e161 e163 e165 e167 e169 e171 e173 (8 SHA-family/Tiger/Whirl-
 * pool/RIPEMD-family).
 *
 * NOTE: returning 1 here does NOT imply GPU eligibility; that requires
 * hx_primitive_is_supported_5a(hx_primitive_for_job(op)) == 1. */
int hx_primitive_is_family_md5pass(int job_enum);

/* Phase 1b Batch 1 (2026-05-28): unsalted single-hash family JOB->prim map.
 *
 * Map a category-(a) unsalted single-hash JOB enum (e.g. JOB_MD5=1,
 * JOB_MD4=3, JOB_SHA1RAW=34, JOB_MD5RAW=33, JOB_SHA256RAW=36) to its
 * primitive id. Returns HX_PRIM_UNKNOWN for any JOB enum NOT in the
 * unsalted-single wired set.
 *
 * This is a SEPARATE table from job_to_prim_table (the MAKE_MD5PASS
 * family map). The unsalted-single shape is `hash(pass)` (3-op bytecode),
 * structurally distinct from the family's `outer(md5_hex(pass).pass)`
 * (6-op). The two never overlap (no JOB enum is in both tables).
 *
 * The RAW variants (MD5RAW/SHA1RAW/SHA256RAW) map to the SAME primitive
 * as their hex siblings (MD5RAW->MD5, SHA1RAW->SHA1, SHA256RAW->SHA256);
 * the raw-vs-hex difference is host-side output formatting, not a
 * distinct primitive. Both are admitted equally here.
 *
 * Used by gpu_codegen_unsalted_eligible() (gpu/gpu_codegen_eligible.c)
 * and the unsalted emit helper's primitive dispatch. Single source of
 * truth: future unsalted-single ships add a row here + the emit-helper
 * arm; the admit predicate auto-propagates. */
enum hx_primitive_id hx_primitive_for_unsalted_job(int job_enum);

/* Returns 1 if this JOB enum is a wired unsalted-single-hash member
 * (hx_primitive_for_unsalted_job(op) != HX_PRIM_UNKNOWN). */
int hx_primitive_is_unsalted_single(int job_enum);

#ifdef __cplusplus
}
#endif

#endif /* HX_EMIT_PRIMITIVES_H */
