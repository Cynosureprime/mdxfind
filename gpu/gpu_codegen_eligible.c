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
 * $Revision: 1.8 $
 * $Log: gpu_codegen_eligible.c,v $
 * Revision 1.8  2026/05/28 14:32:26  dlr
 * Phase 1b Batch 1: add gpu_codegen_unsalted_eligible admit predicate SEPARATE from the family predicate not overloaded; table-driven via hx_primitive_for_unsalted_job plus hx_primitive_is_supported_5a; Batch-1 set e1 MD5 e3 MD4 e33 MD5RAW e34 SHA1RAW e36 SHA256RAW; COVERAGE-GAP CAVEAT a 1 here means the no-rule no-mask iter==1 compute is codegen-correct but does NOT authorize removing the op from the legacy rules-engine path which GPU-accelerates rules masks iter; cutover BLOCKED pending architect resolution; consumed only behind MDXFIND_HX_CODEGEN_UNSALTED env-flag gate not the default chokepoint admit
 *
 * Revision 1.7  2026/05/28 06:12:39  dlr
 * sub-phase 5c.2 update comment only e123 JOB_MD5MD5PASS now ADMITTED via table-driven path D17.4.b; the job_to_prim_table 123 HX_PRIM_MD5 row maps e123 to already-supported HX_PRIM_MD5 so this predicate returns 1 with ZERO logic change; multi-emit per-variant kernel body selected downstream by spec entry emit_class HX_EMIT_MULTI not by this id; e123 closes MAKE_MD5PASS family at 30 of 30 GPU-eligible
 *
 * Revision 1.6  2026/05/28 02:09:29  dlr
 * sub-phase 5b3a03 D17.4.b refactor admit predicate gpu_codegen_kernelb_family_md5pass_eligible from 11-arm switch to 2-line table-driven query via hx_primitive_for_job plus hx_primitive_is_supported_5a single source of truth future Tier 3 sub-phases 5b3b 4-pass and 5b3c 5-pass and Tier 4 gost sne128 sne256 ships flip prim_table supported_5a flag only no edit here new include path codegen hx_emit_primitives h linked into all builds via Makefile frag CODEGEN_OBJS post-refactor eligibility set as of 5b3a is md2 md4 rmd128 sha1 sha224 sha256 sha384 sha512 rmd160 wrl tiger hav128_3 hav160_3 hav192_3 hav224_3 hav256_3 16 primitives 16 GPU-eligible MAKE_MD5PASS family JOBs
 *
 * Revision 1.5  2026/05/27 23:10:38  dlr
 * sub-phase 5b2b4 widen gpu_codegen_kernelb_family_md5pass_eligible from 10-arm to 11-arm switch add case 171 JOB_TIGERMD5PASS e171 admit for Tier 2 Tiger ship 5a.4 era 7 wired primitives plus 5b.1a MD2 plus 5b.1b RMD128 plus 5b.2a WRL plus 5b.2b TIGER now 11 widening uses incremental switch-arm pattern per D16.4.a recommendation supporting comment block updated TIGER promoted out of deferred list into shipped Tier 2 list both Tier 2 ships complete remaining 18 family members haval 15 gost gost_crypto sne128 sne256 ship in Tier 3 to 4 after corresponding _block primitives land in gpu_common.cl case 171 inserted numerically between case 169 and case 173
 *
 * Revision 1.4  2026/05/27 22:27:19  dlr
 * sub-phase 5b2a4 widen gpu_codegen_kernelb_family_md5pass_eligible from 9-arm to 10-arm switch add case 173 JOB_WRLMD5PASS e173 admit for Tier 2 Whirlpool ship 5a.4 era 7 wired primitives plus 5b.1a MD2 plus 5b.1b RMD128 plus 5b.2a WRL now 10 widening uses incremental switch-arm pattern per D16.4.a recommendation supporting comment block updated WRL promoted out of deferred list into shipped Tier 2 list remaining 19 family members tiger haval 15 gost gost_crypto sne128 sne256 ship in Tier 2b to 4 after corresponding _block primitives land in gpu_common.cl
 *
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
#include "../codegen/hx_emit_primitives.h"

/* Sub-phase 5b.3a.0.3 (2026-05-27) D17.4.b refactor:
 *
 * The historical incremental-switch approach has been replaced with a
 * table-driven query against hx_emit_primitives.c's JOB->primitive map
 * + supported_5a flag. Future Tier 4 (and beyond) flips a single
 * supported_5a flag in prim_table[] and ALL 4 widening sites (this
 * predicate, the 2 mdxfind.c harness OR-chains, the _proto_hexlen
 * switches in gpujob_opencl.c + gpujob_metal.m) auto-propagate without
 * edits here.
 *
 * Eligibility rule:
 *   1. op must map to a MAKE_MD5PASS family member (hx_primitive_for_job
 *      returns non-UNKNOWN).
 *   2. The mapped primitive must have supported_5a = 1 in prim_table[]
 *      (i.e. its *_block primitive is already lifted into gpu_common.cl
 *      + metal_common.metal as a shared GPU helper).
 *   3. Sub-phase 5c.2 (2026-05-27): e123 JOB_MD5MD5PASS -- the FIRST
 *      multi-emit member -- is now ADMITTED via the table-driven path
 *      (D17.4.b). A { 123, HX_PRIM_MD5 } row in job_to_prim_table maps
 *      e123 to HX_PRIM_MD5 (already supported_5a=1, the inner hash), so
 *      this predicate returns 1 with ZERO logic change here. The
 *      multi-emit per-variant kernel body is selected downstream by the
 *      spec entry's emit_class == HX_EMIT_MULTI, NOT by this id. e123
 *      closes the MAKE_MD5PASS family at 30/30 GPU-eligible. (Previously
 *      e123 was excluded by the missing table row per
 *      feedback_codegen_multi_emit_is_deferred_not_excluded.md.)
 *
 * As of Phase 5b Tier 3 sub-phase 5b.3a (2026-05-27) the supported_5a
 * set is: md2 md4 rmd128 sha1 sha224 sha256 sha384 sha512 rmd160 wrl
 * tiger hav128_3 hav160_3 hav192_3 hav224_3 hav256_3 (16 primitives;
 * 16 GPU-eligible MAKE_MD5PASS family JOBs).
 *
 * The set will widen as Phase 5b Tier 3 sub-phases 5b.3b (4-pass HAVAL)
 * and 5b.3c (5-pass HAVAL) and Tier 4 (gost/sne128/sne256) ship.
 */
int gpu_codegen_kernelb_family_md5pass_eligible(int op)
{
    enum hx_primitive_id pid = hx_primitive_for_job(op);
    if (pid == HX_PRIM_UNKNOWN) return 0;
    return hx_primitive_is_supported_5a(pid);
}

/* Phase 1b Batch 1 (2026-05-28): admit predicate for the unsalted
 * single-hash codegen shape (HX_PATTERN_UNSALTED_SINGLE).
 *
 * SEPARATE predicate from gpu_codegen_kernelb_family_md5pass_eligible
 * (per the Batch-1 brief: do NOT overload the family predicate; the two
 * shapes are structurally distinct). Eligibility rule:
 *   1. op maps to a wired unsalted-single member (hx_primitive_for_-
 *      unsalted_job returns non-UNKNOWN).
 *   2. that primitive must be supported_5a=1 in prim_table[] (its *_block
 *      is resident in gpu_common.cl + metal_common.metal).
 *
 * As of Batch 1 the eligible set is: e1 MD5, e3 MD4, e33 MD5RAW,
 * e34 SHA1RAW, e36 SHA256RAW (5 JOBs; primitives md5/md4/sha1/sha256 all
 * already resident). Batch 2/3 widen by adding rows to unsalted_job_table
 * + the emit-helper arm; this predicate auto-propagates.
 *
 * IMPORTANT (Phase 1b coverage gap): this predicate answering 1 means the
 * NO-RULE NO-MASK ITER==1 compute is GPU-codegen-correct. It does NOT by
 * itself authorize REMOVING the op from the legacy rules-engine path,
 * which GPU-accelerates rules/masks/iter for these ops -- the codegen
 * unsalted-single kernel is a one-shot hash(pass). The production cutover
 * (gpu_ops[] removal + rules-engine OR-chain) is BLOCKED pending architect
 * resolution of that coverage gap (see Phase 1b Batch 1 handoff). This
 * predicate is currently consumed ONLY behind the MDXFIND_HX_CODEGEN_-
 * UNSALTED env-flag gate at the dispatcher, NOT in the default chokepoint
 * admit. */
int gpu_codegen_unsalted_eligible(int op)
{
    enum hx_primitive_id pid = hx_primitive_for_unsalted_job(op);
    if (pid == HX_PRIM_UNKNOWN) return 0;
    return hx_primitive_is_supported_5a(pid);
}
