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
 * $Revision: 1.14 $
 * $Log: hx_emit_primitives.c,v $
 * Revision 1.14  2026/05/31 19:43:11  dlr
 * iter v1.2 (#386): admit JOB_SHA1 (e8) + JOB_SHA256 (e10) hex-feedback siblings of SHA1RAW/SHA256RAW into the codegen route gate at ANY iter; closes user item #5 from 2026-05-31 not-working list. Per #379 v1.1 widen option (a). Verified op-ids in job_types.h. CPU paths mdxfind.c:28666-28679 (SHA1) + :29088-29097 (SHA256) confirm hex-feedback (prmd5 between iters), distinct from binary-feedback RAW siblings (:27994, :29077). codegen/hx_emit_primitives.c adds 2 rows to unsalted_job_table (auto-propagates digest_bytes). gpu/gpujob_*.c widens admit + adds iter-aware full-digest recompute that walks N iters of mysha1/mysha256+prmd5 chain mirroring CPU loop. gpu_metal.m widens _is_exp_md5 admission gate. gpu/codegen_auto_dispatch.c+h add cells 9b/9c/9d for SHA1/SHA256 + 6 matrix-dump probes + docstring update. Apple Metal template_iterate empirically BROKEN for SHA1/SHA256 iter>1 (returns 0 cracks; same root cause as MD5: metal_template.metal:684 Phase 1 intentionally not called) — auto-dispatcher Metal SHA1/SHA256 iter>1 cell picks CODEGEN (flagship class). Latent iter-aware-recompute bug found+fixed during validation (was iter=1-only; broke immediately for new ops at i=2). 24-cell new-op parity matrix + 38-cell regression matrix ALL PASS byte-exact vs CPU oracle on dev1 M1 + fpga Pascal + hpi7 Maxwell. Cross-host CPU-oracle md5s match. Advisory dedup verified.
 *
 * Revision 1.13  2026/05/29 17:02:14  dlr
 * Hygiene: drop unused stddef.h include (size_t/NULL come from string.h).
 *
 * Revision 1.12  2026/05/28 14:32:17  dlr
 * Phase 1b Batch 1: add hx_primitive_for_unsalted_job + hx_primitive_is_unsalted_single helpers backed by a SEPARATE unsalted_job_table 5 rows e1 MD5 e3 MD4 e33 MD5RAW e34 SHA1RAW e36 SHA256RAW; distinct from job_to_prim_table family map no JOB enum in both; RAW variants map to same primitive as hex siblings; single source of truth for the unsalted admit predicate and emit-helper dispatch; header NOT included for link portability same rationale as job_to_prim_table
 *
 * Revision 1.11  2026/05/28 06:12:21  dlr
 * sub-phase 5c.2.2 add job_to_prim_table row 123 HX_PRIM_MD5 the single eligibility gate for e123 MD5MD5PASS multi-emit; HX_PRIM_MD5 was already supported_5a 1 as the inner hash so NO global flag change wrongly admitting other MD5-inner algos; adding only this row makes hx_primitive_for_job 123 return HX_PRIM_MD5 admitting ONLY job 123 since no other family member maps to an MD5 outer; multi-emit behavior keyed on the spec-entry emit_class HX_EMIT_MULTI in the family emitter not on this prim id; family now 30 of 30 GPU-eligible
 *
 * Revision 1.10  2026/05/28 04:49:21  dlr
 * 5b.4b.3: flip HX_PRIM_GOST supported_5a 0 to 1 promotes GOST into family emitter supported set; gost_block lifted gpu_common.cl 1.34 metal_common.metal 1.33 plus bespoke emit helper; job_to_prim_table 125 row pre-staged 5b.4a; D17.4.b auto-propagation now activates for e125; gost_crypto stays 0 forever non-family; family 29 of 30 GPU-eligible
 *
 * Revision 1.9  2026/05/28 04:32:05  dlr
 * sub-phase 5b4a4 flip supported_5a 0 to 1 for HX_PRIM_SNE128 and HX_PRIM_SNE256 prim_table rows promotes the 2 Snefru widths into family emitter supported set snefru_block lifted into gpu_common.cl rev 1.33 metal_common.metal rev 1.32 ONE parameterised block both widths plus sub-phase 5b4a1 ADD 3 job_to_prim_table rows the one-time non-flag-flip admit edit per D18.4.a GOST sne128 sne256 NOT pre-staged in 5b3a unlike HAVAL e125 GOST row pre-staged numeric-sorted after 122 before 127 supported_5a 0 until 5b4b harmless e175 SNE128 e177 SNE256 rows go live alongside the supported_5a flips after these rows land admit predicate OpenCL _proto_hexlen harness OR-chains chokepoint init-gate listing all auto-propagate via D17.4.b zero edits there gost_crypto stays supported_5a 0 forever not a MAKE_MD5PASS family member out of scope family now 28 of 30 GPU-eligible gost e125 follows in 5b4b for 29 of 30
 *
 * Revision 1.8  2026/05/28 03:52:49  dlr
 * sub-phase 5b3c3 flip 5 HAV*_5 supported_5a 0 to 1 hav128_5 hav160_5 hav192_5 hav224_5 hav256_5 in prim_table job_to_prim_table ALREADY had HAV*_5 rows 131 137 143 149 155 from 5b.3a no table edit needed completes 15-variant HAVAL family Tier 3 26 of 30 MAKE_MD5PASS members GPU-eligible Tier 4 comment updated gost sne128 sne256 gost_crypto remain deferred admit harness _proto_hexlen auto-propagate via D17.4.b no edits there
 *
 * Revision 1.7  2026/05/28 03:19:46  dlr
 * sub-phase 5b3b3 flip supported_5a 0 to 1 for 5 HAV*_4 prim_table rows hav128_4 hav160_4 hav192_4 hav224_4 hav256_4 split into new 5b Tier 3 HAVAL 4-pass sub-block 5-pass variants stay supported_5a 0 until 5b3c job_to_prim_table already had HAV*_4 rows from 5b3a so admit predicate harness gates _proto_hexlen auto-propagate via D17.4.b no edits there
 *
 * Revision 1.6  2026/05/28 02:09:24  dlr
 * sub-phase 5b3a04 NAME-FIX add 2 alias rows for bare hav128 and hav256 callnames mapping to canonical HX_PRIM_HAV128_3 and HX_PRIM_HAV256_3 ids resolves catalog mismatch where e127 and e151 use bare hav128 hav256 without _3 suffix but prim_table canonical names are hav128_3 hav256_3 without aliases emit FATAL with UNKNOWN id plus sub-phase 5b3a04 flip supported_5a flag 0 to 1 for 5 HX_PRIM_HAV*_3 rows hav128_3 hav160_3 hav192_3 hav224_3 hav256_3 promotes 3-pass HAVAL variants into family emitter supported set haval3_block to land in gpu_common.cl rev 1.30 in 5b3a1 plus sub-phase 5b3a03 D17.4.b refactor add hx_primitive_for_job lookup helper hand-built 28-row JOB-enum to outer-primitive mapping table single source of truth for all 4 widening sites plus hx_primitive_is_family_md5pass convenience wrapper API for future Tier 4 ships flag-flip only no per-site edits
 *
 * Revision 1.5  2026/05/27 23:09:46  dlr
 * sub-phase 5b2b4 flip supported_5a flag from 0 to 1 for HX_PRIM_TIGER row prim_table promotes Tiger into family emitter supported set tiger_block lifted into gpu_common.cl rev 1.29 metal_common.metal rev 1.28 4x256 ulong TIGER_SBOX 8 KB constant memory budget plus combined with WRL_SBOX 16 KB equals 24 KB total well within Pascal and Apple Silicon CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE 64 KB budget Tier 2b ship 11 of 11 wired primitives total via 5a.4 plus 5b.1a plus 5b.1b plus 5b.2a plus 5b.2b md2 md4 rmd128 sha1 sha224 sha256 sha384 sha512 rmd160 wrl tiger Tiger row moved from 5b-deferred block into new 5b Tier 2 Tiger sub-block comment haval snefru gost remain deferred Tiers 3 and 4
 *
 * Revision 1.4  2026/05/27 22:26:38  dlr
 * sub-phase 5b2a4 flip supported_5a flag from 0 to 1 for HX_PRIM_WRL row prim_table promotes WRL into family emitter supported set lifted into gpu_common.cl rev 1.28 metal_common.metal rev 1.27 wrl_block primitive 16 KB constant memory 8x256 ulong S-box plus 80 B WRL_RC well within Pascal and Apple Silicon CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE 64 KB budget Tier 2 ship 10 of 10 wired primitives total via 5a.4 plus 5b.1a plus 5b.1b plus 5b.2a md2 md4 rmd128 sha1 sha224 sha256 sha384 sha512 rmd160 wrl Tier 2 row block split with new Tier 2 sub-block comment Tiger row moved into deferred block pending Tier 2b
 *
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
    /* 5b Tier 2 Whirlpool (lifted 2026-05-27 sub-phase 5b.2a) */
    { "wrl",         HX_PRIM_WRL,     64, 1 },
    /* 5b Tier 2 Tiger (lifted 2026-05-27 sub-phase 5b.2b) */
    { "tiger",       HX_PRIM_TIGER,   24, 1 },
    /* 5b Tier 3 HAVAL 3-pass (sub-phase 5b.3a 2026-05-27): supported_5a
     * flipped 0 -> 1; haval3_block lifted into gpu_common.cl and
     * metal_common.metal in 5b.3a.1 + 5b.3a.2; emit_outer_haval_concat_-
     * then_hash parameterised over (passes, digest_bytes) lands in 5b.3a.3
     * for OpenCL twin and 5b.3a.3-metal for Metal twin. 4-pass + 5-pass
     * variants below stay supported_5a=0 until 5b.3b + 5b.3c ship. */
    { "hav128_3",    HX_PRIM_HAV128_3, 16, 1 },
    { "hav160_3",    HX_PRIM_HAV160_3, 20, 1 },
    { "hav192_3",    HX_PRIM_HAV192_3, 24, 1 },
    { "hav224_3",    HX_PRIM_HAV224_3, 28, 1 },
    { "hav256_3",    HX_PRIM_HAV256_3, 32, 1 },
    /* NAME-FIX (sub-phase 5b.3a.0.4 2026-05-27): catalog callnames for
     * e127 and e151 are `hav128` and `hav256` (no _3 suffix), but the
     * canonical prim_table names are `hav128_3` and `hav256_3`. Add 2
     * alias rows so hx_primitive_id_for_name("hav128") returns the
     * canonical HX_PRIM_HAV128_3 id (3-pass default per HAVAL paper).
     * Without these aliases e127 + e151 emit FATAL with UNKNOWN id. */
    { "hav128",      HX_PRIM_HAV128_3, 16, 1 },
    { "hav256",      HX_PRIM_HAV256_3, 32, 1 },
    /* 5b Tier 3 HAVAL 4-pass (sub-phase 5b.3b 2026-05-27): supported_5a
     * flipped 0 -> 1; haval4_block lifted into gpu_common.cl +
     * metal_common.metal in 5b.3b.2. The parameterised
     * emit_outer_haval_concat_then_hash already handles passes=4 (5b.3a);
     * 5b.3b wires the 5 HAV*_4 enums into the dispatch/filter/helper-name
     * switches. 5-pass variants below stay supported_5a=0 until 5b.3c. */
    { "hav128_4",    HX_PRIM_HAV128_4, 16, 1 },
    { "hav160_4",    HX_PRIM_HAV160_4, 20, 1 },
    { "hav192_4",    HX_PRIM_HAV192_4, 24, 1 },
    { "hav224_4",    HX_PRIM_HAV224_4, 28, 1 },
    { "hav256_4",    HX_PRIM_HAV256_4, 32, 1 },
    /* 5b Tier 3 HAVAL 5-pass (sub-phase 5b.3c 2026-05-27): supported_5a
     * flipped 0 -> 1; haval5_block lifted into gpu_common.cl +
     * metal_common.metal in 5b.3c.2. The parameterised
     * emit_outer_haval_concat_then_hash already handles passes=5 (5b.3a);
     * 5b.3c wires the 5 HAV*_5 enums into the dispatch/filter/helper-name
     * switches. This completes the 15-variant HAVAL family + closes Tier 3
     * (26/30 MAKE_MD5PASS family members GPU-eligible). */
    { "hav128_5",    HX_PRIM_HAV128_5, 16, 1 },
    { "hav160_5",    HX_PRIM_HAV160_5, 20, 1 },
    { "hav192_5",    HX_PRIM_HAV192_5, 24, 1 },
    { "hav224_5",    HX_PRIM_HAV224_5, 28, 1 },
    { "hav256_5",    HX_PRIM_HAV256_5, 32, 1 },
    /* 5b Tier 4 Snefru (sub-phase 5b.4a 2026-05-27): supported_5a flipped
     * 0 -> 1; ONE parameterised snefru_block lifted into gpu_common.cl +
     * metal_common.metal in 5b.4a.1 + 5b.4a.2 (handles both widths via
     * the is256 literal). emit_outer_snefru_concat_then_hash parameterised
     * over (is256, digest_bytes) lands in 5b.4a.3. After this ship the
     * MAKE_MD5PASS family is 28/30 GPU-eligible; gost (e125) follows in
     * 5b.4b -> 29/30. */
    { "sne128",      HX_PRIM_SNE128,  16, 1 },
    { "sne256",      HX_PRIM_SNE256,  32, 1 },
    /* 5b Tier 4: gost SHIPPED 5b.4b (2026-05-27) -- supported_5a 0->1, the
     * structurally-divergent block-cipher primitive (gost_block lift into
     * gpu_common.cl rev 1.34 + metal_common.metal rev 1.33 + bespoke emit
     * helper). The MAKE_MD5PASS family now reaches 29/30 GPU-eligible (only
     * e123 multi-emit remains CPU-only). gost_crypto is NOT a MAKE_MD5PASS
     * family member (e14 GOST-CRYPTO, separate non-family job) -- stays
     * supported_5a=0 forever, out of scope. */
    { "gost",        HX_PRIM_GOST,    32, 1 },
    { "gost_crypto", HX_PRIM_GOST_CRYPTO, 32, 0 },
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

/* Sub-phase 5b.3a.0.3 (2026-05-27) D17.4.b refactor:
 *
 * Hand-built JOB-enum -> outer-primitive-id table for the MAKE_MD5PASS
 * family. Values mirror mdxfind.c:5618..5647 (JOB_*MD5PASS macros).
 * Single source of truth for all 4 widening sites (admit predicate,
 * harness OR-chains, _proto_hexlen switch).
 *
 * 30 family members total: 28 wired into prim_table for the 4 sites to
 * route on; 1 (e123 MD5MD5PASS) deferred as multi-emit outlier and
 * intentionally excluded from this table so the admit predicate sees
 * UNKNOWN and routes to CPU; 1 (e123 again, same) is unique. The
 * MD5MD5*MD5 outliers are not present here.
 *
 * To keep this table mechanical and grep-friendly each row carries the
 * eNNN label as a comment.
 */
struct job_to_prim_row {
    int                  job_enum;
    enum hx_primitive_id prim_id;
};

/* Values are the literal JOB_*MD5PASS macro values from mdxfind.c
 * (5618..5647 era). The header is NOT included here to keep this TU
 * link-portable across CPU-only + OpenCL + Metal builds (same rationale
 * as gpu_codegen_eligible.c hard-coded constants per its rev 1.1
 * comment). */
static const struct job_to_prim_row job_to_prim_table[] = {
    { 120, HX_PRIM_MD2     },  /* e120 JOB_MD2MD5PASS    -- 5b.1a Tier 1 */
    { 122, HX_PRIM_MD4     },  /* e122 JOB_MD4MD5PASS    -- 5a */
    /* Sub-phase 5c.2 (2026-05-27): e123 JOB_MD5MD5PASS -- the FIRST
     * multi-emit member, now GPU-eligible. The eligibility GATE is THIS
     * table row, not a global supported_5a flag: HX_PRIM_MD5 was already
     * supported_5a=1 (it is the INNER hash used by every family member),
     * but no job mapped to MD5-as-OUTER until now. Adding this row makes
     * hx_primitive_for_job(123) return HX_PRIM_MD5 -> eligible, admitting
     * ONLY job 123 (no other family member maps to an MD5 outer). The
     * multi-emit behavior is keyed on the SPEC ENTRY's emit_class
     * (HX_EMIT_MULTI for e123) in the family emitter, NOT on this prim id.
     * Family = 30/30 GPU-eligible after this ship. */
    { 123, HX_PRIM_MD5     },  /* e123 JOB_MD5MD5PASS    -- 5c.2 multi-emit */
    { 125, HX_PRIM_GOST    },  /* e125 JOB_GOSTMD5PASS   -- 5b.4b Tier 4 (pre-staged 5b.4a; supported_5a=0 until 5b.4b) */
    { 127, HX_PRIM_HAV128_3 }, /* e127 JOB_HAV128MD5PASS   -- 5b.3a Tier 3 */
    { 129, HX_PRIM_HAV128_4 }, /* e129 JOB_HAV128_4MD5PASS -- 5b.3b */
    { 131, HX_PRIM_HAV128_5 }, /* e131 JOB_HAV128_5MD5PASS -- 5b.3c */
    { 133, HX_PRIM_HAV160_3 }, /* e133 JOB_HAV160_3MD5PASS -- 5b.3a Tier 3 */
    { 135, HX_PRIM_HAV160_4 }, /* e135 JOB_HAV160_4MD5PASS -- 5b.3b */
    { 137, HX_PRIM_HAV160_5 }, /* e137 JOB_HAV160_5MD5PASS -- 5b.3c */
    { 139, HX_PRIM_HAV192_3 }, /* e139 JOB_HAV192_3MD5PASS -- 5b.3a Tier 3 */
    { 141, HX_PRIM_HAV192_4 }, /* e141 JOB_HAV192_4MD5PASS -- 5b.3b */
    { 143, HX_PRIM_HAV192_5 }, /* e143 JOB_HAV192_5MD5PASS -- 5b.3c */
    { 145, HX_PRIM_HAV224_3 }, /* e145 JOB_HAV224_3MD5PASS -- 5b.3a Tier 3 */
    { 147, HX_PRIM_HAV224_4 }, /* e147 JOB_HAV224_4MD5PASS -- 5b.3b */
    { 149, HX_PRIM_HAV224_5 }, /* e149 JOB_HAV224_5MD5PASS -- 5b.3c */
    { 151, HX_PRIM_HAV256_3 }, /* e151 JOB_HAV256MD5PASS   -- 5b.3a Tier 3 */
    { 153, HX_PRIM_HAV256_4 }, /* e153 JOB_HAV256_4MD5PASS -- 5b.3b */
    { 155, HX_PRIM_HAV256_5 }, /* e155 JOB_HAV256_5MD5PASS -- 5b.3c */
    { 157, HX_PRIM_RMD128  },  /* e157 JOB_RMD128MD5PASS -- 5b.1b Tier 1 */
    { 159, HX_PRIM_RMD160  },  /* e159 JOB_RMD160MD5PASS -- 5a */
    { 161, HX_PRIM_SHA1    },  /* e161 JOB_SHA1MD5PASS   -- 5a */
    { 163, HX_PRIM_SHA224  },  /* e163 JOB_SHA224MD5PASS -- 5a */
    { 165, HX_PRIM_SHA256  },  /* e165 JOB_SHA256MD5PASS -- 5a */
    { 167, HX_PRIM_SHA384  },  /* e167 JOB_SHA384MD5PASS -- 5a */
    { 169, HX_PRIM_SHA512  },  /* e169 JOB_SHA512MD5PASS -- 5a */
    { 171, HX_PRIM_TIGER   },  /* e171 JOB_TIGERMD5PASS  -- 5b.2b Tier 2 */
    { 173, HX_PRIM_WRL     },  /* e173 JOB_WRLMD5PASS    -- 5b.2a Tier 2 */
    /* Phase 5b Tier 4 (sub-phase 5b.4a 2026-05-27): Snefru rows go live
     * here alongside the HX_PRIM_SNE128/SNE256 supported_5a flips. The
     * gost row (e125) is pre-staged numeric-sorted above (after 122,
     * before 127); it is HARMLESS until 5b.4b flips HX_PRIM_GOST's
     * supported_5a 0 -> 1. NOT pre-staged in 5b.3a (unlike HAVAL), so
     * Tier 4 adds all 3 rows (the one-time non-flag-flip admit edit per
     * D18.4.a). Once these rows land + supported_5a flips, the admit
     * predicate + OpenCL _proto_hexlen + harness OR-chains + chokepoint
     * + init-gate + listing all auto-propagate via D17.4.b. */
    { 175, HX_PRIM_SNE128  },  /* e175 JOB_SNE128MD5PASS -- 5b.4a Tier 4 */
    { 177, HX_PRIM_SNE256  },  /* e177 JOB_SNE256MD5PASS -- 5b.4a Tier 4 */
};

static const int job_to_prim_table_count =
    (int)(sizeof(job_to_prim_table) / sizeof(job_to_prim_table[0]));

enum hx_primitive_id hx_primitive_for_job(int job_enum)
{
    for (int i = 0; i < job_to_prim_table_count; i++) {
        if (job_to_prim_table[i].job_enum == job_enum)
            return job_to_prim_table[i].prim_id;
    }
    return HX_PRIM_UNKNOWN;
}

int hx_primitive_is_family_md5pass(int job_enum)
{
    return hx_primitive_for_job(job_enum) != HX_PRIM_UNKNOWN;
}

/* Phase 1b Batch 1 (2026-05-28): unsalted single-hash JOB-enum -> outer-
 * primitive-id table. Values mirror mdxfind.c JOB_* macros:
 *   JOB_MD5=1, JOB_MD4=3, JOB_MD5RAW=33, JOB_SHA1RAW=34, JOB_SHA256RAW=36.
 *
 * The bytecode shape for all of these is `hash(pass)` (3-op:
 * PUSH_VAR pass / CALL <prim> / HALT). The RAW variants share the SAME
 * primitive as their hex siblings; only the call role differs (ROLE_BIN
 * vs ROLE_DEFAULT), which is a host-side output-format concern.
 *
 * Header NOT included (same link-portability rationale as job_to_prim_-
 * table): the JOB_* values are literals matched against the int op. Each
 * row carries the eNNN label as a grep anchor.
 *
 * SEPARATE from job_to_prim_table (MAKE_MD5PASS family); no JOB enum is
 * in both. Batch 2/3 add rows here (SHA1/SHA224/SHA256/SHA384/SHA512/
 * SHA384RAW/SHA512RAW/RMD160/MD5UC/WRL...) once their emit-helper arm +
 * cross-arch validation land. */
struct unsalted_job_row {
    int                  job_enum;
    enum hx_primitive_id prim_id;
};

static const struct unsalted_job_row unsalted_job_table[] = {
    {  1, HX_PRIM_MD5    },  /* e1  JOB_MD5       -- Batch 1 */
    {  3, HX_PRIM_MD4    },  /* e3  JOB_MD4       -- Batch 1 */
    {  8, HX_PRIM_SHA1   },  /* e8  JOB_SHA1      -- Batch 1.2 (hex-feedback sibling of SHA1RAW) */
    { 10, HX_PRIM_SHA256 },  /* e10 JOB_SHA256    -- Batch 1.2 (hex-feedback sibling of SHA256RAW) */
    { 33, HX_PRIM_MD5    },  /* e33 JOB_MD5RAW    -- Batch 1 (raw; same prim as MD5) */
    { 34, HX_PRIM_SHA1   },  /* e34 JOB_SHA1RAW   -- Batch 1 (raw; same prim as SHA1) */
    { 36, HX_PRIM_SHA256 },  /* e36 JOB_SHA256RAW -- Batch 1 (raw; same prim as SHA256) */
};

static const int unsalted_job_table_count =
    (int)(sizeof(unsalted_job_table) / sizeof(unsalted_job_table[0]));

enum hx_primitive_id hx_primitive_for_unsalted_job(int job_enum)
{
    for (int i = 0; i < unsalted_job_table_count; i++) {
        if (unsalted_job_table[i].job_enum == job_enum)
            return unsalted_job_table[i].prim_id;
    }
    return HX_PRIM_UNKNOWN;
}

int hx_primitive_is_unsalted_single(int job_enum)
{
    return hx_primitive_for_unsalted_job(job_enum) != HX_PRIM_UNKNOWN;
}
