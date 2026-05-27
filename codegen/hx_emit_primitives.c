/*
 * hx_emit_primitives.c -- per-primitive emit dispatch table.
 *
 * Sub-phase 5a.2 (2026-05-22): name -> id -> width mapping shared by
 * the MAKE_MD5PASS family emitter (5a.2+) and (future) other family
 * emitters. Pure data; no JIT or kernel emission here -- that's the
 * per-backend emit_outer_<primitive>_*_concat_then_hash helpers in
 * hx_emit_opencl.c / hx_emit_metal.c.
 *
 * Names are case-sensitive to match the hx.8 / hx VM convention
 * (lowercase). Per 5a.1 audit the hx compiler emits lowercase names
 * verbatim into _hx_callnames_NNN[] sidecar arrays.
 *
 * $Revision: 1.3 $
 * $Log: hx_emit_primitives.c,v $
 * Revision 1.3  2026/05/27 17:49:23  dlr
 * sub-phase 5b1b3 flip supported_5a flag 0 to 1 for HX_PRIM_RMD128 row promotes RMD128 into family emitter supported set rmd128_block lifted into gpu_common.cl rev 1.27 plus metal_common.metal rev 1.26 row moved from 5b-deferred comment block into 5b Tier 1 comment block alongside MD2 shipped 5b1a hx_primitive_is_supported_5a query now returns 1 for RMD128 making hx_emit_family_md5pass_opencl plus hx_emit_family_md5pass_metal accept e157 RMD128MD5PASS for kernel emission semantic of supported_5a flag remains is primitive resident in shared GPU helper sources as a _block function holds for RMD128 in Tier 1 even though Tier 1 is sub-phase 5b not 5a
 *
 * Revision 1.2  2026/05/27 17:03:18  dlr
 * sub-phase 5b1a3 flip supported_5a flag 0 to 1 for HX_PRIM_MD2 row promotes MD2 into family emitter supported set md2_block lifted into gpu_common.cl rev 1.26 plus metal_common.metal rev 1.25 row split from 5b-deferred comment block into new 5b Tier 1 comment block RMD128 remains in 5b-deferred block awaiting 5b1b lift hx_primitive_is_supported_5a query now returns 1 for MD2 making hx_emit_family_md5pass_opencl plus hx_emit_family_md5pass_metal accept e120 MD2MD5PASS for kernel emission semantic of supported_5a flag remains is primitive resident in shared GPU helper sources as a _block function holds for MD2 in Tier 1 even though Tier 1 is sub-phase 5b not 5a
 *
 * Revision 1.1  2026/05/23 02:02:19  dlr
 * sub-phase 5a.2 per-primitive emit dispatch table for family emitters; pure data lookup of (callname, primitive id, digest width, supported_5a flag); 31-row table covering 8 5a-supported primitives (md4 md5 sha1 sha224 sha256 sha384 sha512 rmd160) plus 22 5b-deferred primitives (md2 rmd128 tiger wrl gost gost_crypto sne128 sne256 hav128_3 hav128_4 hav128_5 hav160_3 hav160_4 hav160_5 hav192_3 hav192_4 hav192_5 hav224_3 hav224_4 hav224_5 hav256_3 hav256_4 hav256_5); shared by family emitters across backends
 *
 *
 */

#include <stddef.h>
#include <string.h>
#include "hx_emit_primitives.h"

struct prim_row {
    const char            *name;
    enum hx_primitive_id   id;
    int                    digest_bytes;
    int                    supported_5a;
};

static const struct prim_row prim_table[] = {
    /* 5a supported */
    { "md5",         HX_PRIM_MD5,     16, 1 },
    { "md4",         HX_PRIM_MD4,     16, 1 },
    { "sha1",        HX_PRIM_SHA1,    20, 1 },
    { "sha224",      HX_PRIM_SHA224,  28, 1 },
    { "sha256",      HX_PRIM_SHA256,  32, 1 },
    { "sha384",      HX_PRIM_SHA384,  48, 1 },
    { "sha512",      HX_PRIM_SHA512,  64, 1 },
    { "rmd160",      HX_PRIM_RMD160,  20, 1 },
    /* 5b Tier 1 (lifted into gpu_common.cl + metal_common.metal 2026-05-27) */
    { "md2",         HX_PRIM_MD2,     16, 1 },
    { "rmd128",      HX_PRIM_RMD128,  16, 1 },
    /* 5b deferred (need gpu_common.cl additions) */
    { "tiger",       HX_PRIM_TIGER,   24, 0 },
    { "wrl",         HX_PRIM_WRL,     64, 0 },
    { "gost",        HX_PRIM_GOST,    32, 0 },
    { "gost_crypto", HX_PRIM_GOST_CRYPTO, 32, 0 },
    { "sne128",      HX_PRIM_SNE128,  16, 0 },
    { "sne256",      HX_PRIM_SNE256,  32, 0 },
    { "hav128_3",    HX_PRIM_HAV128_3, 16, 0 },
    { "hav128_4",    HX_PRIM_HAV128_4, 16, 0 },
    { "hav128_5",    HX_PRIM_HAV128_5, 16, 0 },
    { "hav160_3",    HX_PRIM_HAV160_3, 20, 0 },
    { "hav160_4",    HX_PRIM_HAV160_4, 20, 0 },
    { "hav160_5",    HX_PRIM_HAV160_5, 20, 0 },
    { "hav192_3",    HX_PRIM_HAV192_3, 24, 0 },
    { "hav192_4",    HX_PRIM_HAV192_4, 24, 0 },
    { "hav192_5",    HX_PRIM_HAV192_5, 24, 0 },
    { "hav224_3",    HX_PRIM_HAV224_3, 28, 0 },
    { "hav224_4",    HX_PRIM_HAV224_4, 28, 0 },
    { "hav224_5",    HX_PRIM_HAV224_5, 28, 0 },
    { "hav256_3",    HX_PRIM_HAV256_3, 32, 0 },
    { "hav256_4",    HX_PRIM_HAV256_4, 32, 0 },
    { "hav256_5",    HX_PRIM_HAV256_5, 32, 0 },
};

static const int prim_table_count =
    (int)(sizeof(prim_table) / sizeof(prim_table[0]));

enum hx_primitive_id hx_primitive_id_for_name(const char *fn_name)
{
    if (!fn_name) return HX_PRIM_UNKNOWN;
    for (int i = 0; i < prim_table_count; i++) {
        if (strcmp(fn_name, prim_table[i].name) == 0)
            return prim_table[i].id;
    }
    return HX_PRIM_UNKNOWN;
}

const char *hx_primitive_name(enum hx_primitive_id id)
{
    for (int i = 0; i < prim_table_count; i++) {
        if (prim_table[i].id == id) return prim_table[i].name;
    }
    return "unknown";
}

int hx_primitive_digest_bytes(enum hx_primitive_id id)
{
    for (int i = 0; i < prim_table_count; i++) {
        if (prim_table[i].id == id) return prim_table[i].digest_bytes;
    }
    return 0;
}

int hx_primitive_is_supported_5a(enum hx_primitive_id id)
{
    for (int i = 0; i < prim_table_count; i++) {
        if (prim_table[i].id == id) return prim_table[i].supported_5a;
    }
    return 0;
}
