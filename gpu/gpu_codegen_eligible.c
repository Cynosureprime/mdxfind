/* gpu_codegen_eligible.c - admit-predicate helpers for the hx codegen
 * kernel B family routing. See gpu_codegen_eligible.h for full rationale.
 *
 * Sub-phase 5a.5 (2026-05-22): single switch over the seven 5a-eligible
 * MAKE_MD5PASS family JOB enums. Hard-coded constants (not symbolic
 * JOB_* references) because this TU is linked into BOTH the OpenCL and
 * Metal builds AND the iMac CPU-only build; pulling in mdxfind.h's
 * 600+ JOB_* macros from this tiny TU would add no clarity. The seven
 * values are referenced in code comments adjacent to each case label so
 * future maintainers can grep both directions.
 *
 * $Revision: 1.3 $
 * $Log: gpu_codegen_eligible.c,v $
 * Revision 1.3  2026/05/27 17:50:06  dlr
 * sub-phase 5b1b4 widen gpu_codegen_kernelb_family_md5pass_eligible from 8-arm to 9-arm switch add case 157 JOB_RMD128MD5PASS e157 admit for Tier 1 RMD128 ship 5a.4 era 7 wired primitives md4 rmd160 sha1 sha224 sha256 sha384 sha512 plus 5b.1a MD2 plus 5b.1b RMD128 now 9 widening uses incremental switch-arm pattern per D15.4.a recommendation supporting comment block updated RMD128 promoted out of deferred list into shipped Tier 1 list MD2 and RMD128 both now landed in Tier 1 remaining 20 family members tiger wrl haval 15 gost gost_crypto sne128 sne256 ship in Tier 2 to 4 after corresponding _block primitives land in gpu_common.cl
 *
 * Revision 1.2  2026/05/27 17:04:05  dlr
 * sub-phase 5b1a4 widen gpu_codegen_kernelb_family_md5pass_eligible from 7-arm to 8-arm switch add case 120 JOB_MD2MD5PASS e120 admit for Tier 1 MD2 ship 5a.4 era 7 wired primitives md4 rmd160 sha1 sha224 sha256 sha384 sha512 plus 5b.1a MD2 now 8 widening uses incremental switch-arm pattern per D15.4.a recommendation supporting comment block updated MD2 promoted out of deferred list into shipped Tier 1 list RMD128 stays in deferred list awaiting 5b1b lift remaining 20 family members tiger wrl haval 15 gost gost_crypto sne128 sne256 ship in Tier 2 to 4 after corresponding _block primitives land in gpu_common.cl
 *
 * Revision 1.1  2026/05/23 06:28:47  dlr
 * Initial check-in Sub-phase 5a.5 (2026-05-22): gpu_codegen_kernelb_family_md5pass_eligible admit-predicate helper. 7-arm switch over MAKE_MD5PASS family JOB enums 122 159 161 163 165 167 169. Excludes 123 multi-emit outlier deferred to future sub-phase. 22 5b-deferred members ship after gpu_common.cl primitive lifts. Used by mdxfind.c chokepoint admit and need_gpu override, gpu_opencl.c and gpu_metal.m kernelb_dispatch_proto early gates, gpujob_opencl.c and gpujob_metal.m route gates.
 *
 */

#include "gpu_codegen_eligible.h"

/* Per project_hx_codegen_phase5_family_md5pass_spec_2026-05-22.md §10
 * sub-phase 5a.5: the seven JOB enums correspond to:
 *
 *   122 = JOB_MD4MD5PASS    (e122)
 *   159 = JOB_RMD160MD5PASS (e159)
 *   161 = JOB_SHA1MD5PASS   (e161)
 *   163 = JOB_SHA224MD5PASS (e163)
 *   165 = JOB_SHA256MD5PASS (e165)
 *   167 = JOB_SHA384MD5PASS (e167)
 *   169 = JOB_SHA512MD5PASS (e169)
 *
 * Defined in mdxfind.c:5579..5626 (#define JOB_*MD5PASS <value>).
 *
 * Sub-phase 5b.1a (2026-05-27) added:
 *
 *   120 = JOB_MD2MD5PASS    (e120) -- MD2 outer primitive shipped Tier 1
 *
 * Sub-phase 5b.1b (2026-05-27) added:
 *
 *   157 = JOB_RMD128MD5PASS (e157) -- RMD128 outer primitive shipped Tier 1
 *
 * Intentionally excluded:
 *   123 = JOB_MD5MD5PASS    (e123) -- multi-emit outlier (canonical +
 *                                     colon variant); deferred to a
 *                                     future multi-emit codegen sub-phase.
 *                                     Per feedback_codegen_multi_emit_is_-
 *                                     deferred_not_excluded.md this is a
 *                                     DEFER, not a permanent exclusion.
 *
 * Phase 5b Tier 1 ships MD2 (5b.1a) and RMD128 (5b.1b) -- both now landed.
 * Remaining 20 family members (tiger / wrl / haval(15 variants) / gost /
 * gost_crypto / sne128 / sne256 outers) ship in Tier 2-4 after the
 * corresponding *_block primitives land in gpu_common.cl.
 */
int gpu_codegen_kernelb_family_md5pass_eligible(int op)
{
    switch (op) {
        case 120:  /* JOB_MD2MD5PASS    -- e120  (shipped 5b.1a) */
        case 122:  /* JOB_MD4MD5PASS    -- e122 */
        case 157:  /* JOB_RMD128MD5PASS -- e157  (shipped 5b.1b) */
        case 159:  /* JOB_RMD160MD5PASS -- e159 */
        case 161:  /* JOB_SHA1MD5PASS   -- e161 */
        case 163:  /* JOB_SHA224MD5PASS -- e163 */
        case 165:  /* JOB_SHA256MD5PASS -- e165 */
        case 167:  /* JOB_SHA384MD5PASS -- e167 */
        case 169:  /* JOB_SHA512MD5PASS -- e169 */
            return 1;
        default:
            return 0;
    }
}
