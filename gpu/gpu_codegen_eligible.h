/* gpu_codegen_eligible.h - admit-predicate helpers for the hx codegen
 * kernel B family routing. Shared across OpenCL + Metal backends + the
 * mdxfind.c chokepoint admit / need_gpu override sites.
 *
 * Sub-phase 5a.5 (2026-05-22): introduces
 * gpu_codegen_kernelb_family_md5pass_eligible() so the seven GPU-eligible
 * MAKE_MD5PASS family JOB enums (e122, e159, e161, e163, e165, e167, e169)
 * route through the production codegen dispatcher alongside JOB_MD5MD5SALT
 * (e347).
 *
 * JOB_MD5MD5PASS (e123) is intentionally excluded; it is the multi-emit
 * outlier (canonical + colon-variant) deferred to a future multi-emit
 * codegen sub-phase. Per feedback_codegen_multi_emit_is_deferred_not_excluded.md
 * this is a DEFER, not a permanent exclusion.
 *
 * The other 22 MAKE_MD5PASS family members (md2, gost, haval*, rmd128,
 * tiger, wrl, sne128, sne256 outers) ship in Phase 5b once their block
 * primitives land in gpu_common.cl.
 *
 * $Revision: 1.1 $
 * $Log: gpu_codegen_eligible.h,v $
 * Revision 1.1  2026/05/23 06:28:47  dlr
 * Initial check-in Sub-phase 5a.5 (2026-05-22): gpu_codegen_kernelb_family_md5pass_eligible admit-predicate helper. 7-arm switch over MAKE_MD5PASS family JOB enums 122 159 161 163 165 167 169. Excludes 123 multi-emit outlier deferred to future sub-phase. 22 5b-deferred members ship after gpu_common.cl primitive lifts. Used by mdxfind.c chokepoint admit and need_gpu override, gpu_opencl.c and gpu_metal.m kernelb_dispatch_proto early gates, gpujob_opencl.c and gpujob_metal.m route gates.
 *
 */
#ifndef GPU_CODEGEN_ELIGIBLE_H
#define GPU_CODEGEN_ELIGIBLE_H

#ifdef __cplusplus
extern "C" {
#endif

/* Returns 1 if `op` is one of the seven 5a-eligible MAKE_MD5PASS family
 * JOB enums, 0 otherwise. Safe to call with any int (including invalid
 * op values).
 *
 * Used by:
 *   - mdxfind.c chokepoint admit (~line 11510) so default `-m e<N> -G 0`
 *     routes to the codegen kernel B dispatcher.
 *   - mdxfind.c need_gpu override (~line 39078) per
 *     feedback_gpu_init_gate_for_non_gpu_ops_dispatch.md so gpu_opencl_init
 *     fires even though these JOBs are absent from gpu_ops[].
 *   - gpu/gpujob_opencl.c proto route gate (~line 1147).
 *   - gpu/gpujob_metal.m proto route gate (~line 1190).
 *   - gpu/gpu_opencl.c gpu_opencl_kernelb_dispatch_proto early gate
 *     (~line 12520).
 *   - gpu_metal.m gpu_metal_kernelb_dispatch_proto early gate (~line 4403).
 */
int gpu_codegen_kernelb_family_md5pass_eligible(int op);

#ifdef __cplusplus
}
#endif

#endif /* GPU_CODEGEN_ELIGIBLE_H */
