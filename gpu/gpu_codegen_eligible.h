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
 * $Revision: 1.5 $
 * $Log: gpu_codegen_eligible.h,v $
 * Revision 1.5  2026/11/20 02:52:14  dlr
 * Generic hx walker stage 2 on both GPU backends: salted, user-keyed, HMAC and peppered walker types. A walker type with a channel is a salted op everywhere the host lists salted kernels, takes its salt snapshot from Typesalt or, for a user op, TYPEUSER or the -u pool, never uses the precomputed md5(salt), and replays each hit to cpu_confirm_enqueue with the raw salt. The worker sizes its snapshot for each selected op key set: more than 1024 users from -u overran saltsnap and crashed the worker, the hand-built e1063 included. A pepper op packs one entry per salt and pepper, salt . pepper . the pepper length in 2 bytes, with pack_map naming the snapshot salt, and the pack arrays are sized for the pairs. The templates clamp max_iter to 1 under HX_NO_ITERATE; without the define they preprocess identically. Verified on gp1 (Tahiti, OpenCL) and dev3 (M2 Max, Metal) against -G none: all new types SAME at x01 and x02 and deeper, rules and mask, -F and -s or -u, 2911 salts or users over several pages, 2911 salts by 3 peppers at 161270 hits, built-in, -j and -M -P peppers, and a 300-byte embedded pepper.
 *
 * Revision 1.4  2026/11/18 04:08:00  dlr
 * Generic hx walker stage 1a. OpenCL builds each walker type template program from the hx_gen dialect plus that type generated core, in per-device slots, cached on the source text; a build or slot overflow is FATAL. Both hit replays send walker hits to cpu_confirm_enqueue for the CPU to confirm. The template kernels compare from round 2 only under HX_X01_SILENT. Walker defaults for category and hash words; gpu_hx_generic declarations.
 *
 * Revision 1.3  2026/05/28 14:32:26  dlr
 * Phase 1b Batch 1: add gpu_codegen_unsalted_eligible admit predicate SEPARATE from the family predicate not overloaded; table-driven via hx_primitive_for_unsalted_job plus hx_primitive_is_supported_5a; Batch-1 set e1 MD5 e3 MD4 e33 MD5RAW e34 SHA1RAW e36 SHA256RAW; COVERAGE-GAP CAVEAT a 1 here means the no-rule no-mask iter==1 compute is codegen-correct but does NOT authorize removing the op from the legacy rules-engine path which GPU-accelerates rules masks iter; cutover BLOCKED pending architect resolution; consumed only behind MDXFIND_HX_CODEGEN_UNSALTED env-flag gate not the default chokepoint admit
 *
 * Revision 1.2  2026/05/28 01:18:59  dlr
 * Refresh doc-comment on gpu_codegen_kernelb_family_md5pass_eligible to reflect current eligibility set semantically rather than tracking arm count drop the stale 7-arm phrasing now stale post Tier 1 and Tier 2 widening switch body is the authoritative single source of truth list new bullet pointing at mdxfind.c gpu_op_advertise_for_h_listing as a third call site refreshed via the 2026-05-27 cleanup that lifted the legacy gpu_ops table to file scope and replaced the inline family override with a predicate call no API change
 *
 * Revision 1.1  2026/05/23 06:28:47  dlr
 * Initial check-in Sub-phase 5a.5 (2026-05-22): gpu_codegen_kernelb_family_md5pass_eligible admit-predicate helper. 7-arm switch over MAKE_MD5PASS family JOB enums 122 159 161 163 165 167 169. Excludes 123 multi-emit outlier deferred to future sub-phase. 22 5b-deferred members ship after gpu_common.cl primitive lifts. Used by mdxfind.c chokepoint admit and need_gpu override, gpu_opencl.c and gpu_metal.m kernelb_dispatch_proto early gates, gpujob_opencl.c and gpujob_metal.m route gates.
 *
 */
#ifndef GPU_CODEGEN_ELIGIBLE_H
#define GPU_CODEGEN_ELIGIBLE_H

#ifdef __cplusplus
extern "C" {
#endif

/* Returns 1 iff `op` is a JOB enum currently codegen-eligible for the
 * MAKE_MD5PASS family on GPU; 0 otherwise. Safe to call with any int
 * (including invalid op values).
 *
 * The authoritative list of admitted JOBs is the switch body in
 * gpu_codegen_eligible.c -- consult that file for the current set.
 * This predicate is the SINGLE SOURCE OF TRUTH used at every routing
 * site (chokepoint admit, need_gpu override, proto route gates, early
 * dispatch gates); never replicate its contents as an inline literal
 * table. Phase 5b Tier 3/4 widening (haval/gost/sne128/sne256 family
 * members) extends the switch and all callers pick up the change with
 * no further edits.
 *
 * Used by:
 *   - mdxfind.c chokepoint admit (~line 11510) so default `-m e<N> -G 0`
 *     routes to the codegen kernel B dispatcher.
 *   - mdxfind.c need_gpu override (~line 39200) per
 *     feedback_gpu_init_gate_for_non_gpu_ops_dispatch.md so gpu_opencl_init
 *     fires even though these JOBs are absent from gpu_ops[].
 *   - mdxfind.c gpu_op_advertise_for_h_listing (~line 45540) so the
 *     `mdxfind -h` listing tags the right entries [GPU] without a
 *     parallel mirror table.
 *   - gpu/gpujob_opencl.c proto route gate (~line 1147).
 *   - gpu/gpujob_metal.m proto route gate (~line 1190).
 *   - gpu/gpu_opencl.c gpu_opencl_kernelb_dispatch_proto early gate
 *     (~line 12520).
 *   - gpu_metal.m gpu_metal_kernelb_dispatch_proto early gate (~line 4403).
 */
int gpu_codegen_kernelb_family_md5pass_eligible(int op);

/* Phase 1b Batch 1 (2026-05-28): returns 1 iff `op` is a wired unsalted
 * single-hash codegen member (HX_PATTERN_UNSALTED_SINGLE) whose primitive
 * is GPU-resident; 0 otherwise. Safe with any int.
 *
 * SEPARATE from the family predicate above (distinct bytecode shape:
 * 3-op hash(pass) vs 6-op outer(md5_hex(pass).pass)). Batch-1 set:
 * e1 MD5, e3 MD4, e33 MD5RAW, e34 SHA1RAW, e36 SHA256RAW.
 *
 * COVERAGE-GAP CAVEAT: a 1 here means the no-rule/no-mask/iter==1 compute
 * is codegen-correct on GPU. It does NOT authorize migrating the op off
 * the legacy rules-engine template path (which GPU-accelerates rules/
 * masks/iter). The production cutover is BLOCKED pending architect
 * resolution of that gap. This predicate is consumed only behind the
 * MDXFIND_HX_CODEGEN_UNSALTED env-flag gate (validation), not the default
 * chokepoint admit. See Phase 1b Batch 1 handoff in mdx-team-state.md. */
int gpu_codegen_unsalted_eligible(int op);

/* Generic hx walker (stage 1a, mdxmain.c): 1 iff `op` is served in THIS run
 * by a template core generated from its hx program (codegen/hx_gen.c) --
 * a drift-checked catalog type, no hand-built kernel, and for a type that
 * reports nothing at x01 only when -i is 2 or more. Decided once at load
 * time; safe with any int from any thread. One term `|| gpu_hx_generic(op)`
 * per host site, never a per-type edit. */
int gpu_hx_generic(int op);

/* mdxfind.c: the pepper list op hashes with, and where its pepper comes
 * from (0 none, 1 that list, 2 the salt after a space, 3 after a colon).
 * A walker op with a pepper packs salt . pepper . plen (2 bytes LE). */
const char *type_pepper_list(int op);
int type_pepper_mode(int op);

/* The generated core text for a gpu_hx_generic op (backend-independent;
 * hx_gen_dialect() supplies the backend's macros), or NULL. */
const char *gpu_hx_generic_core(int op);

/* Its stored-form words (HASH_WORDS), 0 when not a walker op. */
int gpu_hx_generic_words(int op);

/* Its one variable channel besides the password (stage 2): 0 none, 1 salt
 * (Typesalt), 2 user (Typeuser, else the -u pool), and 0 when not a walker
 * op. A non-zero channel is a salted template program: the salt axis, the
 * salt snapshot and the salted hit replay, as for the hand salted types. */
int gpu_hx_generic_channel(int op);

#ifdef __cplusplus
}
#endif

#endif /* GPU_CODEGEN_ELIGIBLE_H */
