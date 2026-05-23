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
 * $Revision: 1.1 $
 * $Log: hx_emit_primitives.h,v $
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

#ifdef __cplusplus
}
#endif

#endif /* HX_EMIT_PRIMITIVES_H */
