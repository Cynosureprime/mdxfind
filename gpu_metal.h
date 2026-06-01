/*
 * $Revision: 1.12 $
 * $Log: gpu_metal.h,v $
 * Revision 1.11  2026/05/22 00:00:00  dlr
 * Phase 4 sub-phase 4a.2b declare gpu_metal_kernelb_dispatch_proto Metal twin of gpu_opencl_kernelb_dispatch_proto. Two-kernel pipeline JOB_MD5MD5SALT only. Kernel A1 cand_rules_phase0 producer plus codegen-emitted kernelb_hx_e347_phase0 consumer. Returns persistent h_hits on dispatch fire NULL on early gate. nhits_out set to kernel B hit count. MDXFIND_HX_CODEGEN equal 0 opt out FATAL on Metal no legacy kernel B path exists. All failure modes FATAL per feedback_external_failures_are_fatal.md. Lazy-allocates persistent buf_hashes_shown plus mtl_buf_kernelb_ovr_set plus mtl_buf_kernelb_ovr_gid. Wired into gpu/gpujob_metal.m route gate for default mdxfind -m e347 -G 0 invocation. Header rev 1.57 to 1.58.
 *
 * Revision 1.10  2026/05/22 04:30:00  dlr
 * sub-phase 2a.6 declare three new Metal entry points for the e347 byte-exact validation harness; gpu_metal_jit_compile_source_with_common_keep returns the MTLLibrary and MTLComputePipelineState as opaque void pointers so the caller can dispatch against real data; gpu_metal_e347_validate_dispatch binds the kernel arg-list mirroring the OpenCL twin sets up MTLBuffers uploads packed candidates and chunk_index payload reuses device-resident salt and compact buffers from set_salt and set_compact_table dispatches via MTLCommandBuffer and reads hits back; gpu_metal_jit_release_keep tears down the retained handles idempotent on NULL; all three FATAL on any error per external-failures-are-fatal; Metal twin of OpenCL gpu_opencl_jit_compile_source_with_common_keep and gpu_opencl_e347_validate_dispatch in gpu_opencl.c rev 1.189
 *
 * Revision 1.9  2026/05/21 23:23:29  dlr
 * sub-phase 2a.4 declare gpu_metal_jit_compile_source_with_common for hx codegen Metal twin of OpenCL sister wrapper; FATAL on init or compile failure; harness mode only no PSO retained; called from mdxfind.c harness block under METAL_GPU guard.
 *
 * Revision 1.8  2026/05/12 18:00:00  dlr
 * Phase 2a row 5: additive declaration of gpu_metal_template_pso_lazy_md5_rules. Mirrors gpu_metal_template_pso_lazy_md5 signature (zero args, int return; 0 success, -1 failure). Caller (gpu_metal_dispatch_md5_rules in gpu_metal.m) invokes the new lazy creator on first dispatch when gpu_rule_count > 0; it JIT-compiles a second MTLLibrary from metal_common_str + metal_md5_core_str + metal_md5_rules_str + metal_template_str with preprocessorMacros = GPU_TEMPLATE_HAS_RULES=1, then creates the rules-variant PSO. No-rules PSO + embedded metallib path stays unchanged. No other API surface change.
 *
 * Revision 1.7  2026/05/12 13:40:22  dlr
 * Phase 1 Metal port host header fresh start (replaces retired 1.6). Minimal API surface per arch memo §11 row 11: init/shutdown/available/set_compact_table/set_overflow/set_op/set_max_iter/compile_families/dispatch_md5_rules. Signatures mirror gpu_opencl_* exactly for symmetric arms in mdxfind.c. extern "C" + #pragma once + METAL_GPU guard. Phase 2+ deferred APIs (_set_mask/_set_salts/_set_rules/_list_devices/_warm_probe/_dispatch_packed) not declared.
 *
 */
/* gpu_metal.h — Metal GPU acceleration for mdxfind (Phase 1 fresh start).
 *
 * Phase 1 scope:
 *   - macOS only; guarded by METAL_GPU at every call site (mdxfind.c).
 *   - Single device only (memo §1 non-goal: multi-device).
 *   - One kernel family (FAM_MD5UNSALTED), one PSO (template_phase0 for MD5).
 *   - metallib-first compile pipeline (embedded gpu_mdxfind_metallib[] bytes
 *     from gpu/mdxfind_metallib.h) with `MDXFIND_METAL_JIT=1` env override
 *     that recompiles from `metal_common_str + metal_md5_core_str +
 *     metal_template_str` (gpu/metal_*_str.h).
 *
 * Function signatures mirror the matching gpu_opencl_* API in
 * gpu/gpu_opencl.h, so the call sites in mdxfind.c are symmetric arms
 * (#elif defined(METAL_GPU) vs #if defined(OPENCL_GPU)).
 *
 * Surface this header pre-declares (memo §11 row 11):
 *   gpu_metal_init
 *   gpu_metal_shutdown
 *   gpu_metal_available
 *   gpu_metal_set_compact_table
 *   gpu_metal_set_overflow
 *   gpu_metal_set_op
 *   gpu_metal_set_max_iter
 *   gpu_metal_compile_families
 *   gpu_metal_dispatch_md5_rules
 *
 * Deferred to Phase 2+ (memo §1 non-goals): _set_mask / _set_salts /
 * _set_rules / _list_devices / _warm_probe / _dispatch_packed /
 * multi-device API extensions.
 *
 * NOT to be #include'd unless METAL_GPU is defined — the
 * `#if defined(METAL_GPU)` guard at the top of the body keeps a stray
 * include from a non-Metal build inert (matches gpu_opencl.h's
 * OPENCL_GPU guard).
 */

#pragma once
#ifndef GPU_METAL_H
#define GPU_METAL_H

#if defined(METAL_GPU)

#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Initialize the Metal GPU backend.
 *   - Acquire MTLDevice + MTLCommandQueue (single device).
 *   - Load embedded metallib (default) OR JIT from source strings if
 *     MDXFIND_METAL_JIT=1.
 *   - The PSO is created lazily on first dispatch (mirrors
 *     gpu_opencl_template_kernel_lazy_md5).
 * Returns 0 on success, -1 on failure (no device available, library
 * load failed, command queue create failed). Logs each failure path. */
int gpu_metal_init(void);

/* Tear down the Metal backend. Idempotent. ARC frees the strong refs;
 * we explicitly drain the command queue first so any in-flight buffer
 * autorelease doesn't fight the global destructor. */
void gpu_metal_shutdown(void);

/* Returns 1 if gpu_metal_init succeeded and the backend is usable.
 * Mirrors gpu_opencl_available(). */
int gpu_metal_available(void);

/* Per-device APIs. Phase 1 supports dev_idx == 0 only; any other value
 * returns -1 (mirrors gpu_opencl_set_compact_table's bounds check). The
 * dev_idx parameter is retained for signature parity with the OpenCL
 * mirror so mdxfind.c can use a single call site. */
int gpu_metal_set_compact_table(int dev_idx,
    uint32_t *compact_fp, uint32_t *compact_idx,
    uint64_t compact_size, uint64_t compact_mask,
    unsigned char *hash_data_buf, size_t hash_data_buf_size,
    size_t *hash_data_off, size_t hash_data_count,
    unsigned short *hash_data_len);

int gpu_metal_set_overflow(int dev_idx,
    const uint64_t *keys, const unsigned char *hashes,
    const uint32_t *offsets, const uint16_t *lengths, int count);

void gpu_metal_set_op(int dev_idx, int op);
void gpu_metal_set_max_iter(int dev_idx, int max_iter);

/* Compile the requested kernel families. Phase 1 stub:
 *   - If (fam_mask & (1u << FAM_MD5UNSALTED)) is set: no-op (library is
 *     already loaded at gpu_metal_init time; PSO is lazy at first
 *     dispatch).
 *   - Other family bits in fam_mask are silently ignored in Phase 1.
 * Mirrors gpu_opencl_compile_families. */
void gpu_metal_compile_families(unsigned int fam_mask);

/* Dispatch one packed batch of (word_offset, packed_words) through the
 * Metal MD5 template kernel. Phase 1 supports JOB_MD5 only; any other
 * op returns NULL with *nhits_out = 0.
 *
 * Returns a pointer to the static hits buffer (HIT_STRIDE-wide uint32
 * entries; *nhits_out hit records). Caller does NOT free the buffer.
 * Mirrors gpu_opencl_dispatch_md5_rules signature exactly so the
 * call site arms are symmetric.
 *
 * mask_start / mask_offset_per_word / bf_num_masks / inner_iter /
 * bf_fast_eligible are unused in Phase 1 but kept in the signature for
 * symmetry. */
uint32_t *gpu_metal_dispatch_md5_rules(int dev_idx,
    const char *packed_words, uint32_t packed_size,
    const uint32_t *word_offset, uint32_t num_words,
    int op, int *nhits_out,
    uint64_t mask_start, uint32_t mask_offset_per_word, uint32_t bf_num_masks,
    uint32_t inner_iter,
    int bf_fast_eligible);


/* Phase 2b row 3: bind a mask charset table + per-position sizes to the
 * device-side mask_charsets / mask_sizes MTLBuffers. Mirrors gpu/gpu_opencl.c
 * gpu_opencl_set_mask exactly:
 *   sizes[ntotal]            per-position character counts (uint8).
 *   tables[ntotal][256]      256-byte charset rows (only first sizes[i]
 *                            bytes are read on the GPU).
 *   npre, napp               prepend + append position counts. ntotal =
 *                            npre + napp; bounded by MASK_POS_CAP=16 each.
 * The host packs into uint8[MASK_TOTAL_CAP*256] charset buffer +
 * uint32[MASK_TOTAL_CAP] sizes buffer (MASK_TOTAL_CAP = 32), MTLResource-
 * StorageModeShared. Caches gpu_mask_n_prepend / gpu_mask_n_append /
 * gpu_mask_total / gpu_mask_sizes[] for hit-replay decode in
 * gpu/gpujob_metal.m. Idempotent (same call w/ same args yields no new
 * allocation, repopulates the existing buffers). Returns 0 on success;
 * -1 on bounds violation or MTLBuffer allocation failure. */
int gpu_metal_set_mask(const uint8_t *sizes, const uint8_t tables[][256],
                       int npre, int napp);

/* Phase 2c row 1: salt-variant PSO lazy creators. Four new variants
 * mirror the Phase 2b R/M/RM pattern with an additional GPU_TEMPLATE_HAS_SALT
 * macro defined. Each compiles a SEPARATE MTLLibrary that concatenates
 * metal_md5salt_core_str (replacing metal_md5_core_str — the two define
 * the same fn names; only one may live in any given MTLLibrary) with
 * the relevant ifdef macros set. Caller (gpu_metal_dispatch_md5_rules)
 * selects between the eight variants (none/R/M/RM/S/RS/MS/RMS) at
 * dispatch time based on (use_rules, use_mask, use_salt). Returns
 * 0 on success, -1 on failure. Idempotent. */
int gpu_metal_template_pso_lazy_md5_salt(void);
int gpu_metal_template_pso_lazy_md5_salt_rules(void);
int gpu_metal_template_pso_lazy_md5_salt_mask(void);
int gpu_metal_template_pso_lazy_md5_salt_rules_mask(void);

/* Phase 2c row 2: upload a salt list to the GPU. Mirrors
 * gpu_opencl_set_salts (gpu/gpu_opencl.c:3805) exactly:
 *   salts            concatenated salt bytes for the whole list (not
 *                    paged); salt N starts at salt_offsets[N], length is
 *                    salt_lens[N].
 *   salt_offsets     per-salt uint32 byte offset into `salts`.
 *   salt_lens        per-salt uint16 length.
 *   num_salts        entry count.
 * Allocates / re-allocates three MTLBuffers (buf_salt_data, buf_salt_off,
 * buf_salt_lens) at MTLResourceStorageModeShared. Cap-grow pattern: re-
 * allocate when current cap is exceeded; else memcpy in place. Per-session
 * lifetime; refreshed when nsalts_packed changes between dispatches.
 * Returns 0 on success, -1 on bounds violation or buffer allocation
 * failure. Logs `Metal: salts uploaded (N entries, M bytes)` marker on
 * each successful upload (per feedback_verify_gpu_fired_post_build.md). */
int gpu_metal_set_salt(const char *salts, const uint32_t *salt_offsets,
                       const uint16_t *salt_lens, int num_salts);


/* Phase 2d.2.4: md5passsalt family — FIRST salted family added via the
 * cl2metal codegen path (md5salt was hand-ported in Phase 2c).
 *
 * MD5PASSSALT (JOB_MD5PASSSALT=373, hashcat -m 10, mdxfind e373):
 *   MD5(pass || salt) — single-MD5 APPEND-salt variant.
 *
 * Four PSO variants (V_S, V_S|V_R, V_S|V_M, V_S|V_R|V_M) mirror the
 * md5salt salted-only resolver pattern. JOB_MD5PASSSALT is always salted;
 * the dispatcher's use_salt predicate (gpu_metal_dispatch_md5_rules)
 * widens to include JOB_MD5PASSSALT alongside JOB_MD5SALT.
 *
 * Lazy creators are file-scope STATIC inside gpu_metal.m (no exported
 * API needed — only the md5passsalt_pso_for_variant resolver references
 * them, and it lives in the same TU). The resolver hangs off the
 * registered gpu_metal_family pso_for_variant slot. Future salted
 * siblings (md5saltpass, sha1passsalt, sha256passsalt, ...) follow the
 * same internal-only pattern.
 *
 * NO PRESALT for md5passsalt (unlike md5salt's V_S|V_R PRESALT fold) —
 * gpu_md5passsalt_core.cl defines a single-MD5 template_finalize with no
 * template_pre_salt / template_finalize_post scaffolding. Phase 2e
 * presalt parity for md5passsalt is a separate future-work item. */

/* Phase 2d.2.5: md5saltpass family — LAST family in the Phase 2d.2 md5
 * family fan-out (md4, md4utf16, md5raw, md5passsalt, md5saltpass —
 * 5 codegen-translated families on top of md5+md5salt baselines).
 *
 * MD5SALTPASS (JOB_MD5SALTPASS=394, hashcat -m 20, mdxfind e394):
 *   MD5(salt || pass) — single-MD5 PREPEND-salt variant. Mirror image of
 *   md5passsalt's APPEND shape; same 4-PSO salted-only resolver pattern.
 *
 * Four PSO variants (V_S, V_S|V_R, V_S|V_M, V_S|V_R|V_M) mirror the
 * md5passsalt salted-only resolver pattern exactly. JOB_MD5SALTPASS is
 * always salted; the dispatcher's use_salt predicate widens to include
 * JOB_MD5SALTPASS alongside JOB_MD5SALT + JOB_MD5PASSSALT. gpujob_metal.m's
 * is_salted_op OR-list extends accordingly (Phase 2d.2.4 lesson per
 * feedback_metal_is_salted_op_widening.md).
 *
 * Lazy creators are file-scope STATIC inside gpu_metal.m (same as
 * md5passsalt — no exported API needed; only md5saltpass_pso_for_variant
 * references them and it lives in the same TU).
 *
 * NO PRESALT for md5saltpass (same rationale as md5passsalt) —
 * gpu_md5saltpass_core.cl defines a single-MD5 template_finalize with no
 * template_pre_salt / template_finalize_post scaffolding. Phase 2e
 * presalt parity for md5saltpass is a separate future-work item. */

/* Phase 2d.3.4: sha1passsalt family -- FIRST SHA-family SALTED port on the
 * cl2metal codegen path.
 *
 * SHA1PASSSALT (JOB_SHA1PASSSALT=405, hashcat -m 100, mdxfind e405):
 *   SHA-1(pass || salt) -- single-SHA-1 APPEND-salt variant.
 *
 * Four PSO variants (V_S, V_S|V_R, V_S|V_M, V_S|V_R|V_M) mirror the
 * md5passsalt salted-only resolver pattern exactly. JOB_SHA1PASSSALT is
 * always salted; the dispatcher's use_salt predicate (gpu_metal_dispatch_-
 * md5_rules) widens to include JOB_SHA1PASSSALT. gpujob_metal.m's
 * is_salted_op OR-list extends accordingly (forward-staged Phase 2d.3.2).
 * Hit-replay decode width: metal_gpu_hash_words(JOB_SHA1PASSSALT) returns
 * 5 (forward-staged Phase 2d.3.2).
 *
 * Distinct from md5passsalt: 5-word state (SHA-1) vs 4-word (MD5), BE
 * message word load, BE length encoding. Inherits sha1_block + bswap32
 * + EMIT_HIT_5_DEDUP_OR_OVERFLOW from metal_common.metal (Phase 2d.3.1).
 *
 * Lazy creators are file-scope STATIC inside gpu_metal.m (same as
 * md5passsalt -- no exported API needed; only sha1passsalt_pso_for_variant
 * references them).
 *
 * NO PRESALT (Phase 2e presalt hoist is md5salt-specific). */

/* Phase 2d.3.5: sha1saltpass family -- LAST family in Phase 2d.3 SHA-1
 * fan-out (sha1, sha1raw, sha1dru, sha1passsalt, sha1saltpass = 5 families
 * on the cl2metal codegen path).
 *
 * SHA1SALTPASS (JOB_SHA1SALTPASS=385, hashcat -m 110, mdxfind e385):
 *   SHA-1(salt || pass) -- single-SHA-1 PREPEND-salt variant. Mirror image
 *   of sha1passsalt's APPEND shape; same 4-PSO salted-only resolver pattern.
 *
 * Four PSO variants (V_S, V_S|V_R, V_S|V_M, V_S|V_R|V_M) mirror the
 * sha1passsalt salted-only resolver pattern exactly. JOB_SHA1SALTPASS is
 * always salted; the dispatcher's use_salt predicate widens to include
 * JOB_SHA1SALTPASS alongside JOB_SHA1PASSSALT (shipped together in Phase
 * 2d.3.4 widening).
 *
 * IMPORTANT: gpu_sha1saltpass_core.cl is a LARGE 50KB shared salted SHA-
 * family carrier hosting HMAC-SHA1/SHA224/SHA256 variants gated by runtime
 * if (HASH_WORDS == N) checks. At Metal JIT compile time HASH_WORDS
 * defaults to 5; only the raw SHA1SALTPASS path runs at runtime. The HMAC
 * branches must COMPILE but never execute. Per
 * feedback_runtime_gate_for_template_branches.md.
 *
 * Lazy creators are file-scope STATIC inside gpu_metal.m (same as
 * sha1passsalt -- no exported API needed).
 *
 * NO PRESALT (Phase 2e presalt hoist is md5salt-specific). */

/* Phase 2e: pre-salt hoist + SIMD lane-batching PSO variant. For the
 * salt+rules combination (op==JOB_MD5SALT && use_salt && use_rules &&
 * !use_mask) the kernel lifts the password-only inner MD5+hex32 out of
 * the per-salt loop and consumes it via template_finalize_post() with
 * SALT_BATCH-stride tile iteration. Compiles a SEPARATE MTLLibrary with
 * GPU_TEMPLATE_HAS_PRE_SALT=1 AND SALT_BATCH=N where N comes from
 * metal_select_salt_batch(MTLDevice) (8 for M1, 16 for M2/M2 Max, 32 for
 * M3+; env override MDXFIND_METAL_SALT_BATCH). The cache key
 * disambiguates on both macros so a SALT_BATCH change forces a fresh
 * JIT compile. Other salt combinations (salt-only / salt+mask /
 * salt+rules+mask) fall through to the non-presalt PSOs; Phase 2e.2
 * expands coverage. Returns 0 on success, -1 on failure. Idempotent. */
int gpu_metal_template_pso_lazy_md5_salt_rules_presalt(void);

/* Phase 2d.2.1a: per-family struct + op-keyed dispatcher pattern.
 *
 * Mirrors the OpenCL gpu_template_resolve_kernel pattern (gpu/gpu_opencl.c
 * line 8745) at a smaller, Metal-shaped granularity. Each registered family
 * advertises:
 *   - its CPU op (JOB_MD5, JOB_MD5SALT, ...)
 *   - its op_category (GPU_CAT_UNSALTED / GPU_CAT_MASK) returned via
 *     gpu_op_category() in gpu/gpujob_metal.m
 *   - the set of (rules, mask, salt) variant_bits combinations it supports
 *   - a pso_for_variant resolver that returns an already-lazy-created
 *     MTLComputePipelineState for the requested variant. The resolver
 *     internally invokes the appropriate per-variant lazy creator and
 *     also handles family-private optimizations (e.g., md5salt folds
 *     the Phase 2e PRESALT PSO inside the V_S|V_R-only arm — PRESALT
 *     is NOT exposed via variant_bits).
 *
 * variant_bits is a 3-bit field: V_R | V_M | V_S. The dispatcher computes
 * it from (use_rules, use_mask, use_salt) and passes it through the
 * registered family's resolver.
 *
 * Net effect at the dispatcher (gpu_metal_dispatch_md5_rules):
 *   fam = gpu_metal_lookup_family(op);
 *   if (!fam) return NULL;
 *   variant_bits = (use_rules?V_R:0) | (use_mask?V_M:0) | (use_salt?V_S:0);
 *   active_pso = fam->pso_for_variant(variant_bits);
 *
 * Per the Phase 2d.2.1a design memo (decisions 1+2 baked into the brief):
 *   - md5 registers with V_NONE|V_R|V_M|V_R|V_M support
 *   - md5salt registers with V_S|V_S|V_R|V_S|V_M|V_S|V_R|V_M support
 *     (PRESALT folded into V_S|V_R inside the md5salt resolver, not
 *      exposed in variant_bits)
 *
 * Subsequent algos (md4 phase 2d.2.1b, sha1, sha256, etc.) register
 * additional families via gpu_metal_register_family at init time without
 * touching the dispatcher cascade. The lookup is a linear scan over the
 * small N of registered families — fine for the expected algorithm count
 * (~10s, not 1000s). */

#define V_NONE  0u
#define V_R     (1u << 0)
#define V_M     (1u << 1)
#define V_S     (1u << 2)

/* Family registration. The Objective-C MTLComputePipelineState type is
 * intentionally not in the struct signature exposed via the header — the
 * pso_for_variant function pointer returns id<MTLComputePipelineState> in
 * Objective-C and `void *` here so this header stays usable in plain C
 * call sites. gpu_metal.m (the only caller of pso_for_variant) casts the
 * void* back to id<MTLComputePipelineState> at the call site. */
struct gpu_metal_family {
    int op;                 /* JOB_MD5, JOB_MD5SALT, etc. */
    const char *name;       /* short name for stderr/logs ("md5", "md5salt") */
    int op_category;        /* GPU_CAT_UNSALTED / GPU_CAT_MASK */
    uint8_t supported_variants;  /* bitmask of admissible variant_bits */
    void *(*pso_for_variant)(uint8_t variant_bits);

    /* === D5b Wave 1 shared-loader refactor 2026-05-16 ===
     *
     * Wave-1 migration-safety dual-pointer approach. Resolvers migrated to
     * the generic loader use pso_for_variant_v2 (carries the family pointer
     * so the generic infrastructure can index hidden parallel arrays via
     * fam_idx). Unmigrated resolvers continue to use the legacy
     * pso_for_variant slot. Dispatch site (gpu_metal.m and the
     * compile_families eager-probe loop) chooses v2 when non-NULL, else
     * legacy. This lets Wave 1 land the canary (md5) without churning the
     * 51 other resolver signatures; later waves migrate families one-by-one
     * and clear the legacy pointer once all are migrated.
     *
     * core_str:        per-family Metal source for the algo body (e.g.
     *                  metal_md5_core_str). Concatenated into the JIT source
     *                  by metal_load_library_generic.
     * base_macros:     opaque NSDictionary* (Obj-C in .m) of macro defaults
     *                  for this family (HASH_WORDS, BASE_ALGO, ...). The
     *                  generic loader copies+extends with variant-derived
     *                  macros (HAS_RULES/MASK/SALT) before passing to
     *                  MTLCompileOptions.preprocessorMacros. void* here so
     *                  the header stays C-pure.
     * dispatch_tg_size: per-family threadsPerThreadgroup override. 0 means
     *                  use the default policy (BCRYPT=8 today; eventually
     *                  the dispatch site stops hardcoding the op check and
     *                  reads this field instead).
     * fam_idx:         populated by a post-registration walk in
     *                  gpu_metal_init -- index into the parallel
     *                  metal_family_libs[][] / metal_family_psos[][] arrays
     *                  inside gpu_metal.m. -1 before the walk runs. */
    void *(*pso_for_variant_v2)(struct gpu_metal_family *fam,
                                uint8_t variant_bits);
    const char *core_str;
    void *base_macros;
    uint16_t dispatch_tg_size;
    int fam_idx;
};

/* Register a family. Idempotent on (op): repeated registration with the
 * same op silently overwrites the prior entry. Safe to call from
 * gpu_metal_init() (and only there in current code). */
void gpu_metal_register_family(struct gpu_metal_family *f);

/* Lookup a registered family by op. Returns NULL if no family is
 * registered for op. Called from the dispatch op-gate
 * (gpu_metal_dispatch_md5_rules) and from gpu_op_category
 * (gpu/gpujob_metal.m). */
struct gpu_metal_family *gpu_metal_lookup_family(int op);

/* Phase D5a (Task #281+#282) 2026-05-16: per-(family,variant) admission
 * query. After gpu_metal_compile_families() runs the eager-compile loop,
 * it populates an internal admission bitmap recording which (family,
 * variant_bits) combos PSO-compiled cleanly and which were rejected
 * (loudly logged to stderr).
 *
 * Returns 1 if at least one of the variants in `variant_bits_mask` was
 * admitted for `op`; 0 if no variant was admitted (the family is
 * effectively CPU-only on this device for this op).
 *
 * `variant_bits_mask` is a bitmask of which V_* combos to check. Pass
 * 0xFF to ask "any variant admitted at all"; pass VBIT(V_S) to ask
 * "salted-only variant admitted" specifically.
 *
 * Returns 0 (not admitted) for unregistered ops -- a benign answer that
 * lets callers safely query any op without pre-validation.
 *
 * §6 Option B per architect Task #281: this is a CAPABILITY check, NOT
 * a runtime failure. mdxfind.c may use this at init to prune gpu_ops[]
 * (deferred to D5c); current callers may use it to gate dispatch arms
 * without falling into the runtime-fatal path (see gpu_metal_dispatch_-
 * md5_rules). */
int gpu_metal_op_variant_admitted(int op, uint8_t variant_bits_mask);

/* Phase D5a (Task #281+#282) 2026-05-16: total admitted families count
 * (families with at least one admitted variant). Returned by the
 * compile_families summary on stderr; exposed here so mdxfind.c can
 * query it post-init if it wants to decide on GPU vs CPU routing
 * (deferred to D5c -- current call sites do not consult this).
 *
 * Returns 0 before gpu_metal_compile_families has been called. */
int gpu_metal_admitted_family_count(void);

/* Phase 1a sub-phase 1a.1b (2026-05-20): Metal twin of
 * gpu_opencl_kernel_a_proto_enabled() declared in gpu/gpu_opencl.h.
 *
 * Phase 0 of 1a.1b ships this as a STUB returning 0 so mdxfind.c (which
 * calls this symbol from BOTH backend-shared chokepoint admit and
 * need_gpu override sites at lines 11448-11450 and 38075-38081
 * respectively) links cleanly on Metal builds. Without the stub the
 * Metal build on dev1.local fails with "undefined reference to
 * gpu_opencl_kernel_a_proto_enabled" — the unconditional call landed
 * in mdxfind.c rev 1.488 (2026-05-19) under the assumption that the
 * symbol was backend-shared; it is OpenCL-only.
 *
 * Phase 2 of 1a.1b replaces the stub with the real variant validator
 * (port of gpu/gpu_opencl.c:9847-9880): reads MDXFIND_KERNEL_A_PROTO +
 * MDXFIND_KERNEL_A_VARIANT, caches the decision, emits a one-time
 * stderr advisory on first call. */
int gpu_metal_kernel_a_proto_enabled(void);

/* Phase 1a sub-phase 1a.1b-continued (2026-05-20): Metal twin of
 * gpu_opencl_kernelb_dispatch_proto (declared in gpu/gpu_opencl.h).
 *
 * A1-only sub-phase ship: dispatches kernel A1 (cand_rules_phase0) to
 * produce candidate plaintexts for the harness trace channel. Kernel B
 * is NOT wired here (separate follow-on sub-phase). Returns a non-NULL
 * pointer on dispatch fire, NULL when env-gate unset / not-ready.
 * *nhits_out is always 0 in this sub-phase (no hit-replay surface).
 *
 * Gated on (getenv("MDXFIND_KERNEL_B_PROTO") || gpu_metal_kernel_a_proto_-
 * enabled()) and op == JOB_MD5MD5SALT, matching OpenCL twin. */
uint32_t *gpu_metal_kernelA_dispatch_proto(int dev_idx,
    const char *packed_words, uint32_t packed_size,
    const uint32_t *word_offset, uint32_t num_words,
    int op, int *nhits_out);

/* Phase 1a sub-phase 1a.2 (2026-05-21): sibling of _proto_enabled per
 * D9.4.a. Returns numeric variant int (1/2/3/4) when proto enabled, 0
 * otherwise. Mirrors gpu_opencl_kernel_a_active_variant. */
int gpu_metal_kernel_a_active_variant(void);

/* Phase 1a sub-phase 1a.2 (2026-05-21): A2 (masks-only) top-level
 * dispatch entry. Metal twin of gpu_opencl_kernel_a_masks_dispatch.
 * Returns non-NULL on dispatch fire, NULL when variant != 2, no active
 * mask, build/upload fails, or device not ready. *nhits_out is always
 * 0 in this v1 ship (harness mode via MDXFIND_KERNEL_A_TRACE; no
 * kernel B wired for A2 yet). */
uint32_t *gpu_metal_kernelA_masks_dispatch(int dev_idx,
    const char *packed_words, uint32_t packed_size,
    const uint32_t *word_offset, uint32_t num_words,
    int op, int *nhits_out);

/* Phase 1a sub-phase 1a.3 (2026-05-21): A3 (rules + masks) top-level
 * dispatch entry. Metal twin of gpu_opencl_kernel_a_rules_masks_dispatch.
 * Compositional product of A1+A2 (cand_rules_masks_phase0 PSO, 10-arg
 * signature). Returns non-NULL on dispatch fire, NULL when variant != 3,
 * no active rules, no active mask, build/upload fails, or device not
 * ready. *nhits_out is always 0 in this v1 ship (harness mode; no
 * kernel B wired for A3 yet). */
uint32_t *gpu_metal_kernelA_rules_masks_dispatch(int dev_idx,
    const char *packed_words, uint32_t packed_size,
    const uint32_t *word_offset, uint32_t num_words,
    int op, int *nhits_out);

/* Phase 1a sub-phase 1a.4 (2026-05-21): A4 (brute-force) top-level
 * dispatch entry. Metal twin of gpu_opencl_kernel_a_bruteforce_dispatch.
 * Pure BF; no rule axis, no prepend mask axis (BF invariant
 * MaskPrependLen == 0 enforced). Routes to cand_bruteforce_phase0 (8-arg
 * PSO byte-identical to A2 signature; mask_pattern_prepend slot bound to
 * the zeroed mtl_buf_kern_a_mask_prepend buffer).
 *
 * Signature differs from A1/A2/A3: takes struct jobg *g directly (no
 * packed_words / word_offset / num_words args). Reads g->bf_chunk,
 * g->bf_mask_start, g->bf_offset_per_word, g->bf_num_masks,
 * g->packed_count + global MaskAppendLen + MaskAppendPattern +
 * MaskClasses[]. Reuses the 4 mask buffers already uploaded by A2/A3
 * (D11.2.a); zero new persistent buffers.
 *
 * Single-shot per dispatch (host pre-sized chunk via adaptive_bf_chunk_-
 * size servo). Host caps max_packed at 256 MB and FATAL-exits on
 * overflow.
 *
 * v1 ship is harness-mode: returns non-NULL on dispatch fire, NULL when
 * variant != 4, no active append mask, build/upload fails, or device
 * not ready. *nhits_out is always 0 (no kernel B wired). */
struct jobg;
uint32_t *gpu_metal_kernelA_bruteforce_dispatch(int dev_idx,
    struct jobg *g, int *nhits_out);

/* hx codegen sub-phase 2a.4 (2026-05-21): Metal twin of
 * gpu_opencl_jit_compile_source_with_common. Concatenate metal_common_str
 * + caller-provided emitted Metal source and JIT-compile via
 * -[MTLDevice newLibraryWithSource:options:error:]. Look up the known
 * entry point `kernelb_hx_e347_phase0` to verify the library contains
 * the expected kernel function. FATAL on any error per
 * feedback_external_failures_are_fatal.md. Harness mode only -- the
 * compiled library is released before return; no PSO retained (Phase 4
 * dispatcher will retain).
 *
 * dev_idx must be 0 (Phase 1 Metal is single-device). src must be
 * non-NULL. Returns 0 on success (only return path; failure paths
 * exit(1) before returning). */
int gpu_metal_jit_compile_source_with_common(int dev_idx, const char *src);

/* hx codegen sub-phase 2a.6 (2026-05-22): retain-the-PSO variant of the
 * 2a.4 JIT helper. Mirrors gpu_opencl_jit_compile_source_with_common_keep
 * shape but Metal-specific: prepends metal_common_str + emitted Metal
 * source, JIT-compiles via -[MTLDevice newLibraryWithSource:options:error:],
 * then creates an MTLComputePipelineState for `entry_point_name`. Returns
 * the MTLLibrary and MTLComputePipelineState as opaque void* through
 * out_library / out_pso so non-Objective-C TUs (mdxfind.c) can hold them
 * across the dispatch. Caller MUST invoke gpu_metal_jit_release_keep when
 * done. FATAL on any error per feedback_external_failures_are_fatal.md. */
int gpu_metal_jit_compile_source_with_common_keep(
    int dev_idx, const char *src, const char *entry_point_name,
    void **out_library, void **out_pso);

/* hx codegen sub-phase 2a.6 (2026-05-22): byte-exact validation dispatch
 * for the codegen-emitted e347 (MD5MD5SALT) Metal kernel. Metal twin of
 * gpu_opencl_e347_validate_dispatch.
 *
 * Caller must FIRST:
 *  - gpu_metal_init (during normal mdxfind startup)
 *  - gpu_metal_set_compact_table(...) with planted-hash compact table
 *  - gpu_metal_set_salt(...) with fixture salt table
 *  - gpu_metal_jit_compile_source_with_common_keep(..., "kernelb_hx_e347_phase0", ..., &lib, &pso)
 *
 * This function then allocates short-lived MTLBuffers for packed
 * candidates + word offsets + payload + hits, binds them along with the
 * device-resident compact/salt/overflow buffers, dispatches the kernel,
 * waits for completion, and copies hits back into out_hits
 * (GPU_HIT_STRIDE u32 per hit, up to max_hits). Returns 0 on success;
 * FATAL on any error. */
int gpu_metal_e347_validate_dispatch(
    int dev_idx, void *pso,
    const unsigned char *packed_words, uint32_t packed_size,
    const uint32_t *word_offset, uint32_t num_words,
    uint32_t num_salts,
    uint32_t *out_hits, int *out_n_hits, int max_hits);

/* hx codegen sub-phase 2a.6 (2026-05-22): cleanup pair for the keep
 * variant. Releases the MTLLibrary + MTLComputePipelineState that
 * gpu_metal_jit_compile_source_with_common_keep returned. Safe with NULL
 * inputs (idempotent). */
void gpu_metal_jit_release_keep(void *library, void *pso);

/* Phase 4 sub-phase 4a.2 (2026-05-21): D13.3.a opt-out accessor for the
 * hx codegen production path on Metal. Mirrors gpu_opencl_hx_codegen_-
 * enabled in gpu/gpu_opencl.h (rev 1.39).
 *
 * Returns 1 when MDXFIND_HX_CODEGEN is unset or set to any value other
 * than the literal "0"; returns 0 when MDXFIND_HX_CODEGEN equals the
 * literal "0".
 *
 * The Metal twin of the codegen production path is staged here in 4a.2
 * (PSO statics + lazy JIT helper) but the kernel-B dispatch wrapper
 * that the OpenCL twin embeds inside gpu_opencl_kernelb_dispatch_proto
 * does not exist on Metal yet -- prior sub-phases (1a.1b-continued
 * through 1a.4) shipped kernel A1/A2/A3/A4 only; no Metal kernel B
 * dispatch wrapper was wired. The 4a.2 ship therefore lays the codegen
 * scaffolding (accessor + statics + lazy JIT) so the Metal kernel-B
 * dispatcher (separate follow-on sub-phase) can swap in the codegen
 * PSO via the same shape as the OpenCL twin's swap in 4a.1.
 *
 * Sunsets in Phase 5 along with the hand-written kernel files. */
int gpu_metal_hx_codegen_enabled(void);

/* gpu_metal_experiment_rules_codegen_md5_enabled REMOVED 2026-05-31
 * per project_codegen_auto_dispatch_spec_2026-05-31.md (D1.c). The
 * user-visible MDXFIND_EXPERIMENT_RULES_CODEGEN_MD5 env flag is
 * RETIRED; the dispatcher now consults gpu/codegen_auto_dispatch.c
 * per (op, iter, rules, mask, bf, backend_kind). Developer FORCE
 * override: MDXFIND_GPU_BACKEND={auto|legacy|codegen}. */

/* Knob G (Metal twin, 2026-05-29): env-flag gate that activates the
 * coalesced uint4 (16-byte) candidate write path in Metal kernel A1
 * (cand_rules_phase0). Returns 1 when MDXFIND_METAL_EXPERIMENT_KNOBG_-
 * VEC_WRITE=1 is set; 0 otherwise. Decision cached.
 *
 * Unlike the OpenCL twin (which composes with the parent codegen
 * experiment flag), the Metal flag stands alone: Metal kernel A1 is
 * the production path for e347 + the 7 Phase-5a MAKE_MD5PASS family
 * algos (e122/e159/e161/e163/e165/e167/e169). Unset = byte-identical
 * to production-baseline Metal kernel (Phase 4a.2b, 2026-05-22).
 *
 * Consumed by metal_load_library_kernel_a_rules to thread
 * -DKNOBG_VEC_WRITE=1 into MTLCompileOptions.preprocessorMacros at
 * JIT library compile time. Spec: project_metal_knob_g_spec_2026-05-
 * 29.md. */
int gpu_metal_experiment_knobg_vec_write_enabled(void);

/* Phase 0 timing harness (Knob G spec D3, 2026-05-29; D2.b refactored
 * 2026-05-29 to auto-enable on MDXFIND_METAL_PROFILE_VARIANT): env-flag
 * gate that enables per-dispatch kernel-A wall time recording via the
 * MTLCommandBuffer's GPUStartTime/GPUEndTime, plus a shutdown-time
 * aggregate dump. Returns 1 when EITHER MDXFIND_METAL_KERNEL_A_TIMING=1
 * OR MDXFIND_METAL_PROFILE_VARIANT in 1..6 is set; 0 otherwise.
 * Decision cached. Used for end-to-end + kernel-A A/B measurement when
 * toggling Knob G on/off AND for PROFILE_VARIANT V0..V6 decomposition. */
int gpu_metal_kernel_a_timing_enabled(void);

/* PROFILE_VARIANT-for-Metal perf-decomposition env-flag gate (2026-05-29):
 * returns 0..6 if MDXFIND_METAL_PROFILE_VARIANT is set to "0".."6";
 * returns 0 (production-baseline V0) otherwise. Variants 1..5 are
 * throwaway timing stubs (no valid candidates); V6 produces valid
 * candidates with Metal Knob G FORCED ON (bit-equivalent to V0 +
 * KNOBG_VEC_WRITE=1). Consumed by metal_load_library_kernel_a_rules to
 * thread -DPROFILE_VARIANT=N into MTLCompileOptions.preprocessorMacros
 * at JIT library compile time; cache forks automatically per variant
 * (Apple MTLLibrary self-cache is keyed on source + macros).
 * Spec: project_metal_profile_variant_spec_2026-05-29.md. */
int gpu_metal_profile_variant(void);

/* gpu_metal_kernel_a4_profile_variant -- PROFILE_VARIANT-for-A4 env-flag
 * gate (Metal twin, 2026-05-30). Returns 1..5 when MDXFIND_METAL_KERNEL_-
 * A4_PROFILE_VARIANT is set to a valid value AND the active kernel-A
 * variant is 4 (host gate R5); returns 0 otherwise (legacy byte-identical
 * baseline V0 = production unchanged).
 *
 * Variant meanings (also documented in gpu/metal_kernel_a_bruteforce.metal
 * file-top A4_PROFILE_VARIANT block):
 *   V0 (unset / 0)  baseline -- full cand_bruteforce_phase0 (uint4 stores
 *                    by default post-2026-05-30 A4 C5 default-on refactor;
 *                    the env-gated KNOBG path + V6 cell were removed
 *                    because the A/B measurement was decisive)
 *   V1               no atomic claim (per-lane gid*256 strided; CONFOUNDED
 *                    on AGX per spec §9 R1)
 *   V2               no candidate write (atomic still runs)
 *   V3               stub charset walk -- EXPECTED A4 DOMINANT
 *   V4               decode-only (return after absolute_mask_idx ladder)
 *   V5               empty kernel (return immediately)
 *
 * Per spec §4 D1.a: separate per-engine env var (cross-engine pollution
 * avoidance). Apple MTLLibrary self-cache is keyed on source + macros
 * so JIT cache disambiguation per variant is automatic.
 * Spec: project_kernel_a_a4_profile_variant_spec_2026-05-30.md. */
int gpu_metal_kernel_a4_profile_variant(void);

/* Phase 4 sub-phase 4a.2 (2026-05-21): lazy-init helper that returns the
 * cached MTLComputePipelineState for the hx-codegen-emitted Metal kernel
 * `kernelb_hx_e347_phase0` on the requested device. On first call per
 * (device, process) the function runs the hx walker with the e347
 * specialization (iter=1, has_rules=1, SALT_BATCH=64, etc.), JIT-
 * compiles the emitted Metal source via gpu_metal_jit_compile_source_-
 * with_common_keep, and caches the resulting MTLLibrary + PSO in
 * module-local statics. On subsequent calls returns the cached PSO
 * pointer (as an opaque void *; cast back to id<MTLComputePipelineState>
 * at the call site).
 *
 * Returns non-NULL on success; FATAL on any failure (hx spec lookup,
 * walker, JIT, PSO creation) per feedback_external_failures_are_fatal.md.
 * A non-fatal source dump is written to /tmp/mdxfind-codegen-e347-<pid>-
 * dev<N>.metal so post-mortem inspection is possible. The cache is
 * process-lifetime; no explicit release entry point is exposed (process
 * exit reclaims). Apple driver does its own opaque PSO caching across
 * processes; the codegen-side cache here saves walker+dump+compile cost
 * for second-and-later dispatches in the same process. */
void *gpu_metal_kernelb_codegen_pso(int dev_idx);

/* Sub-phase 5a.3 (2026-05-22): per-JOB parameterized variant of
 * gpu_metal_kernelb_codegen_pso. Mirror of OpenCL twin
 * gpu_opencl_kernelb_codegen_kernel_for. Lazy-init on miss, linear scan
 * over the 16-slot module-local mtl_codegen_slots[] cache; FATAL on cap
 * exhaustion. job_enum selects the kernel + entry point:
 *   JOB_MD5MD5SALT  -> kernelb_hx_e347_phase0 (e347 hand-tuned chain)
 *   JOB_*MD5PASS    -> kernelb_hx_codegen_phase0 (family emit; SHA1 in
 *                      5a.3, other primitives 5a.4+)
 * Returns the cached PSO as opaque void *. FATAL on any failure. */
void *gpu_metal_kernelb_codegen_pso_for(int dev_idx, int job_enum);

/* hx codegen sub-phase 5a.3 (2026-05-22): byte-exact validation dispatch
 * for the codegen-emitted MAKE_MD5PASS family Metal kernel. Metal sibling
 * of gpu_opencl_kernelb_family_validate_dispatch.
 *
 * Differences from gpu_metal_e347_validate_dispatch:
 *   - num_salts implicitly 1 (kernel ignores; harness still calls
 *     gpu_metal_set_salt with a placeholder so the buffer slots bind).
 *   - Per-thread topology (no SALT_BATCH loop); grid sized to num_words
 *     padded up to tg_size=64 multiple.
 *   - Kernel signature is identical to e347 (18 args at the same
 *     [[buffer(N)]] indices); only the kernel BODY differs.
 *
 * Returns 0 on success; FATAL on any failure. */
int gpu_metal_kernelb_family_validate_dispatch(
    int dev_idx, void *pso,
    const unsigned char *packed_words, uint32_t packed_size,
    const uint32_t *word_offset, uint32_t num_words,
    uint32_t *out_hits, int *out_n_hits, int max_hits);

/* Phase 4 sub-phase 4a.2b (2026-05-22): Metal kernel-B production
 * dispatcher. Metal twin of gpu_opencl_kernelb_dispatch_proto
 * (gpu/gpu_opencl.h rev 1.39). Two-kernel pipeline for JOB_MD5MD5SALT:
 * kernel A1 (cand_rules_phase0) produces packed candidates from the
 * (word, rule) cross product; codegen-emitted kernelb_hx_e347_phase0
 * consumes the kernel-A output and emits hits via
 * EMIT_HIT_4_DEDUP_OR_OVERFLOW.
 *
 * Caller contract (parallel to OpenCL twin):
 *   - gpu_metal_init succeeded and the device is ready
 *   - gpu_metal_set_compact_table populated buf_compact_fp / buf_compact_idx /
 *     buf_hash_data / buf_hash_data_off
 *   - gpu_metal_set_salt populated buf_salt_data / buf_salt_off / buf_salt_lens
 *   - gpu_rule_count > 0 and gpu_rule_program / gpu_rule_offsets are non-NULL
 *   - op == JOB_MD5MD5SALT (other ops return NULL)
 *
 * Returns the persistent h_hits buffer on dispatch fire (caller does
 * NOT free); NULL when an early gate rejects the call (mismatched op,
 * not-ready, missing rule program). *nhits_out is set to the kernel B
 * hit count (0..GPU_MAX_HITS).
 *
 * MDXFIND_HX_CODEGEN=0 opt-out is NOT supported on Metal -- the Metal
 * dispatcher FATAL-exits with a documented message ("opt-out is not
 * supported on Metal -- no legacy kernel B path exists"). On Metal
 * the codegen IS the kernel B implementation.
 *
 * All failure modes (PSO acquire, ensure_proto_bufs, kernel A/B
 * cmdbuffer error, buffer alloc) are FATAL per feedback_external_-
 * failures_are_fatal.md. Lazy-allocates persistent buf_hashes_shown
 * (per feedback_proto_dispatch_lazy_alloc_dependency.md) plus two
 * persistent atomic_uint scratch buffers (ovr_set / ovr_gid) on first
 * dispatch; reuses on subsequent dispatches. */
uint32_t *gpu_metal_kernelb_dispatch_proto(int dev_idx,
    const char *packed_words, uint32_t packed_size,
    const uint32_t *word_offset, uint32_t num_words,
    int op, int *nhits_out);

/* Metal twin of gpu_opencl_kernelb_proto_plaintext (gpu/gpu_opencl.c:14483).
 * Recover the post-rule candidate plaintext for a hit emitted by the Metal
 * codegen kernel B (e347 + the seven Phase-5a MAKE_MD5PASS family algos).
 * The kernel B writes entry[0] = slot_idx (a packed-buffer slot index in the
 * kernel-A scratch buffer mtl_buf_proto_packed), NOT a host-side widx. This
 * accessor resolves slot_idx through the kernel-A index buffer
 * (mtl_buf_proto_chunk_index) into the kernel-A packed buffer
 * (mtl_buf_proto_packed) using the same [len][bytes] layout the OpenCL
 * twin's h_proto_packed_readback uses.
 *
 * Must be called AFTER the kernel-A + kernel-B command buffers have
 * completed (the dispatcher already waits on both via waitUntilCompleted;
 * the Apple Shared MTLBuffer contents are coherent by then).
 *
 * Returns a pointer into the Shared MTLBuffer's contents (caller does NOT
 * free; valid until the next kernel-A dispatch clobbers the buffer); sets
 * *plen_out to the plaintext byte count. Returns NULL on out-of-range
 * slot_idx, missing buffers, or device not ready. dev_idx must be 0 on
 * Metal (single-device). */
const char *gpu_metal_kernelb_proto_plaintext(int dev_idx,
    uint32_t hit_widx, int *plen_out);

#ifdef __cplusplus
}
#endif

#endif /* METAL_GPU */
#endif /* GPU_METAL_H */
