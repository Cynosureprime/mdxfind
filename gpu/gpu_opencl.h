/*
 * gpu_opencl.h — OpenCL GPU acceleration for mdxfind
 *
 * Cross-vendor GPU support via OpenCL runtime.
 * Supports multiple GPU devices.
 */

#ifndef GPU_OPENCL_H
#define GPU_OPENCL_H

#if defined(OPENCL_GPU)

#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>   /* _Exit */

/* Phase D5a 2026-05-16 (Task #281): GPU_FATAL macro relocated to
 * gpu/gpu_fatal.h so the Metal host code (gpu_metal.m, gpu/gpujob_metal.m)
 * can share the same fail-fast primitive without dragging in the rest of
 * the OpenCL public ABI. Backwards-compat: any existing GPU_FATAL caller
 * (gpu/gpu_opencl.c sites) continues to compile unchanged. */
#include "gpu_fatal.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Forward declaration: gpu_opencl_kernel_a_bruteforce_dispatch takes a
 * `struct jobg *` to read BF chunk fields (bf_chunk, bf_mask_start,
 * bf_offset_per_word, bf_num_masks). Full definition lives in gpujob.h
 * (which gpu_opencl.c #includes). Forward-decl here keeps gpu_opencl.h
 * usable from call sites that haven't pulled gpujob.h. */
struct jobg;

/* Host-side mirror of RULE_BUF_MAX in gpu/gpu_md5_rules.cl. MUST match
 * the kernel-side #define exactly. Bumping requires updating BOTH files
 * (gpu_opencl.h and gpu/gpu_md5_rules.cl) in the same commit. The
 * validator path uses this for record-buffer sizing and stack/heap
 * allocations; mismatch would cause stride mismatch and corrupted reads.
 *
 * Wire format for md5_rules_phase0_validate records (rev 1.23+):
 *   slot[0..1] = retlen as int16 little-endian
 *   slot[2..3] = outlen as uint16 little-endian
 *   slot[4..3+RULE_BUF_MAX_HOST] = post-rule buffer bytes
 * Total slot size = GPU_VALIDATE_RECORD_SZ_HOST = 4 + RULE_BUF_MAX_HOST. */
#define RULE_BUF_MAX_HOST           40960u
#define GPU_VALIDATE_RECORD_SZ_HOST (4u + RULE_BUF_MAX_HOST)


int gpu_opencl_init(void);

/* Memo C parallel device init (default on). Set to 1 by `-G serial`
 * CLI option to fall back to single-threaded init. Affects gpu_opencl_init()
 * only — the per-device set_compact_table / set_overflow / set_rules loops
 * in mdxfind.c are parallelized independently. */
void gpu_opencl_set_serial_init(int serial);
void gpu_opencl_compile_families(unsigned int fam_mask);
void gpu_opencl_shutdown(void);
int gpu_opencl_available(void);
int gpu_opencl_num_devices(void);
void gpu_opencl_list_devices(void);

/* hx codegen sub-phase 2a.1 (2026-05-21): JIT-compile arbitrary OpenCL
 * source against device dev_idx for the codegen harness. Returns 0 on
 * success; FATAL with exit(1) + full diagnostic on any failure per
 * feedback_external_failures_are_fatal.md. Harness-mode only -- the
 * compiled program is released before return. NOT wired to the runtime
 * dispatch path in 2a.1 (Phase 4 territory). */
int gpu_opencl_jit_compile_source(int dev_idx, const char *src,
                                  const char *build_opts);

/* hx codegen sub-phase 2a.3 (2026-05-21): same as above but PREPENDS
 * gpu_common_str to the source pair fed to clCreateProgramWithSource.
 * The codegen e347 tp0 emitter calls OCLParams, md5_block,
 * EMIT_HIT_4_DEDUP_OR_OVERFLOW, probe_compact_idx, etc. which all live
 * in gpu_common.cl; without the prepend the build would fail with
 * undeclared-identifier errors. Same FATAL-on-failure discipline as
 * gpu_opencl_jit_compile_source. */
int gpu_opencl_jit_compile_source_with_common(int dev_idx, const char *src,
                                              const char *build_opts);

/* hx codegen sub-phase 2a.5 (2026-05-21): retain-the-program variant of
 * the above. Keeps the cl_program + cl_kernel alive so the validation
 * harness can dispatch the kernel against real data after JIT. Caller
 * MUST clReleaseKernel + clReleaseProgram. FATAL on any error.
 *
 * Forward-declares cl_program/cl_kernel as void* in this header so that
 * non-OpenCL TUs that #include this file (callers that compile in
 * mdxfind.c outside the OPENCL_GPU branch) don't need the OpenCL
 * headers. The .c file casts back via the real types. */
struct _cl_program;
struct _cl_kernel;
int gpu_opencl_jit_compile_source_with_common_keep(
    int dev_idx, const char *src, const char *build_opts,
    const char *entry_point_name,
    struct _cl_program **out_program, struct _cl_kernel **out_kernel);

/* hx codegen sub-phase 2a.5 (2026-05-21): byte-exact validation dispatch
 * for the codegen-emitted e347 (MD5MD5SALT) kernel.
 *
 * Caller must FIRST:
 *  - gpu_opencl_init (during normal mdxfind startup)
 *  - gpu_opencl_set_compact_table(...) with planted-hash compact table
 *  - gpu_opencl_set_salts(...) with fixture salt table
 *  - gpu_opencl_jit_compile_source_with_common_keep(..., "kernelb_hx_e347_phase0", ..., &kern)
 *
 * This function then allocates short-lived input buffers (b_packed,
 * b_chunk, b_payload), reuses device-resident d->b_compact_fp / b_hits /
 * etc., binds all 16 kernel args, dispatches, and reads hits back into
 * out_hits (GPU_HIT_STRIDE u32 per hit, up to max_hits).
 *
 * Returns 0 on success; FATAL on any error. */
int gpu_opencl_e347_validate_dispatch(
    int dev_idx, struct _cl_kernel *kern,
    const unsigned char *packed_words, uint32_t packed_size,
    const uint32_t *word_offset, uint32_t num_words,
    uint32_t num_salts,
    uint32_t *out_hits, int *out_n_hits, int max_hits);

/* Sub-phase 5a.2 (2026-05-22): MAKE_MD5PASS family validate-dispatch.
 * Sibling to gpu_opencl_e347_validate_dispatch; per-thread (no
 * SALT_BATCH loop). Family is unsalted; kernel ignores num_salts but the
 * arg slot binds to device's existing salt buffer to satisfy
 * clSetKernelArg's nonNULL requirement. Returns 0 on success, FATAL on
 * error. Hits in 19-uint stride; sidx field is always 0 per the family
 * kernel emit. Caller compares h0..h3 (first 16 bytes of the 20-byte
 * SHA1 digest for e161). */
int gpu_opencl_kernelb_family_validate_dispatch(
    int dev_idx, struct _cl_kernel *kern,
    const unsigned char *packed_words, uint32_t packed_size,
    const uint32_t *word_offset, uint32_t num_words,
    uint32_t *out_hits, int *out_n_hits, int max_hits);

/* hx codegen sub-phase 2a.5 (2026-05-21): cleanup pair for the JIT
 * retain-the-program path. Releases kernel + program from a TU that
 * lacks the OpenCL headers (e.g. mdxfind.c). Safe with NULL inputs. */
void gpu_opencl_jit_release_keep(struct _cl_program *prog,
                                 struct _cl_kernel *kern);

/* Per-device disable accessors (rev 1.74+).
 *
 * gpu_opencl_device_disabled(d) returns 1 if device d failed compact-table
 * setup (insufficient VRAM or buffer alloc failure). All dispatch entry
 * points in gpu_opencl.c early-return on a disabled device; gpujob_init
 * does not spawn a worker for one.
 *
 * gpu_opencl_active_device_count() returns the number of non-disabled
 * devices. mdxfind.c checks this after the per-device set_compact_table
 * loop to decide whether GPU paths are usable at all — a return of 0
 * causes gpu_opencl_available() to flip back to false and mdxfind
 * routes everything to CPU FastRule (effectively -G none).
 *
 * gpu_opencl_finalize_active_count() must be called by the host AFTER
 * the per-device set_compact_table loop finishes. It computes the
 * active count, logs an "X of Y devices active" line, and (if the
 * count is 0) flips ocl_ready to false so subsequent
 * gpu_opencl_available() calls return 0. */
int gpu_opencl_device_disabled(int dev_idx);
int gpu_opencl_active_device_count(void);
void gpu_opencl_finalize_active_count(void);

/* Phase 1a sub-phase 1a.1 env-flag gate. Returns 1 when the operator has
 * opted into the kernel A production path via MDXFIND_KERNEL_A_PROTO=1
 * (with MDXFIND_KERNEL_A_VARIANT unset or set to a supported value).
 * Used at chokepoint-admit, GPU-init, and per-dispatch sites alongside
 * the legacy MDXFIND_KERNEL_B_PROTO env flag; the two flags are union'd
 * (per project_kernel_a_variants_phase1a_spec_2026-05-20.md decision
 * D3.b). Decision is cached after first call; unknown variant values
 * produce a one-time stderr warning and 0.
 *
 * Sub-phase 1a.2 (2026-05-21) extends accepted variants to {1, 2}: V1 =
 * rules-only A1, V2 = masks-only A2. */
int gpu_opencl_kernel_a_proto_enabled(void);

/* Phase 1a sub-phase 1a.2 (2026-05-21): sibling of _proto_enabled per
 * spec decision D9.4.a. Returns the numeric variant int (1/2/3/4) when
 * proto is enabled, 0 otherwise. Use this at sites that route to a
 * variant-specific dispatch (vs just the binary gate). Shares cache
 * + env-var inspection with gpu_opencl_kernel_a_proto_enabled. */
int gpu_opencl_kernel_a_active_variant(void);

/* Phase 4 sub-phase 4a.1 (2026-05-21): centralized MDXFIND_HX_CODEGEN
 * opt-out predicate. Returns 1 when the env is unset or anything other
 * than the literal string "0"; returns 0 when MDXFIND_HX_CODEGEN=0.
 *
 * Phase 4 sub-phase 4a.3 (2026-05-22): the legacy hand-written kernel B
 * source (gpu_kernelb_md5md5salt_nocache.cl) was deleted from the tree;
 * codegen is the only OpenCL path for JOB_MD5MD5SALT. The accessor is
 * retained so the dispatcher can FATAL on stale MDXFIND_HX_CODEGEN=0
 * settings rather than silently switching behavior. */
int gpu_opencl_hx_codegen_enabled(void);

/* Phase 1a sub-phase 1a.2 (2026-05-21): A2 (masks-only) top-level
 * dispatch entry. Mirrors gpu_opencl_kernelb_dispatch_proto shape but
 * routes to cand_masks_phase0 instead of cand_rules_phase0 + kernel B.
 * Returns NULL when variant != 2, no active mask, build/upload fails,
 * or device disabled. *nhits_out is always 0 in this v1 ship (no
 * kernel B wired for A2 yet; harness mode via MDXFIND_KERNEL_A_TRACE). */
uint32_t *gpu_opencl_kernel_a_masks_dispatch(int dev_idx,
    const char *packed_words, uint32_t packed_size,
    const uint32_t *word_offset, uint32_t num_words,
    int op, int *nhits_out);

/* Phase 1a sub-phase 1a.3 (2026-05-21): A3 (rules + masks) top-level
 * dispatch entry. Compositional product of A1 + A2. Routes to
 * cand_rules_masks_phase0 (10-arg kernel) when variant=3 and BOTH
 * gpu_n_rules > 0 AND MaskTotal > 0. Reuses all five buffers already
 * uploaded by A1 (rule_program/rule_offset) and A2 (mask prepend/append/
 * charsets/counts); no new persistent buffers. Per D10.1.a the host
 * passes num_rules = gpu_n_rules - 1 (source count); the kernel adds the
 * implicit no-rule pass via rule_idx==0 fast path. v1 ship is single-
 * shot (no cursor-restart chunking; sub-phase 1a.3.x); returns NULL
 * on success and *nhits_out is 0 in harness mode. */
uint32_t *gpu_opencl_kernel_a_rules_masks_dispatch(int dev_idx,
    const char *packed_words, uint32_t packed_size,
    const uint32_t *word_offset, uint32_t num_words,
    int op, int *nhits_out);

/* Phase 1a sub-phase 1a.4 (2026-05-21): A4 (brute-force) top-level
 * dispatch entry. Pure BF; no rule axis, no prepend mask axis (BF
 * invariant MaskPrependLen == 0 enforced). Routes to cand_bruteforce_-
 * phase0 (8-arg kernel byte-identical to A2 signature; mask_pattern_-
 * prepend slot bound to a zeroed buffer) when variant=4 AND g->bf_chunk
 * AND MaskAppendLen > 0.
 *
 * Signature differs from A1/A2/A3: takes struct jobg *g directly (no
 * packed_words / word_offset / num_words args). Reads g->bf_chunk,
 * g->bf_mask_start, g->bf_offset_per_word, g->bf_num_masks, g->num_words
 * + global MaskAppendLen + MaskAppendPattern + MaskClasses[]. Reuses the
 * 4 mask buffers already uploaded by A2/A3 (D11.2.a); zero new persistent
 * buffers.
 *
 * Single-shot per dispatch (host pre-sized chunk via adaptive_bf_chunk_-
 * size servo; no kernel-side cursor-restart). Host caps max_packed at
 * MAX_KERNEL_A_PACKED (256 MB) and FATAL-exits on overflow.
 *
 * v1 ship is harness-mode: returns NULL on success and *nhits_out is 0
 * (no kernel B wired for A4). */
uint32_t *gpu_opencl_kernel_a_bruteforce_dispatch(int dev_idx,
    struct jobg *g, int *nhits_out);

/* Per-device APIs — dev_idx from 0 to num_devices-1 */
int gpu_opencl_set_compact_table(int dev_idx,
    uint32_t *compact_fp, uint32_t *compact_idx,
    uint64_t compact_size, uint64_t compact_mask,
    unsigned char *hash_data_buf, size_t hash_data_buf_size,
    size_t *hash_data_off, size_t hash_data_count,
    unsigned short *hash_data_len);

int gpu_opencl_set_salts(int dev_idx,
    const char *salts, const uint32_t *salt_offsets,
    const uint16_t *salt_lens, int num_salts);

int gpu_opencl_set_overflow(int dev_idx,
    const uint64_t *keys, const unsigned char *hashes,
    const uint32_t *offsets, const uint16_t *lengths, int count);

void gpu_opencl_set_max_iter(int dev_idx, int max_iter);
/* BF Phase 3b Tranche C (2026-05-10): set_mask_resume / set_salt_resume /
 * has_resume / last_mask_start prototypes removed — implementations gone
 * with the slab arm (Tranche B). See gpu_opencl.c head comment. */
void gpu_opencl_set_op(int dev_idx, int op);
int gpu_opencl_get_op(int dev_idx);
int gpu_opencl_max_batch(int dev_idx);
int gpu_opencl_set_mask(const uint8_t *sizes, const uint8_t tables[][256],
                        int npre, int napp);

/* BF Phase 3 (2026-05-10): the multi-GPU atomic-cursor BF API
 * (gpu_opencl_bf_start/stop/active/set_partition/set_tail_start) has been
 * retired. BF on GPU now flows exclusively through the chunk-as-job
 * producer at mdxfind.c:~48590 + rules-engine path. See
 * project_bf_chunk_as_job.md Phase 3. RCS history retains the prior API
 * (gpu_opencl.h rev 1.22). */

/* Surviving accessor: per-(device, family) autotune rate. */
double gpu_opencl_fam_rate(int dev_idx, int fam);

/* Phase 6.1: warm-probe — eager, parallel autotune of all devices for op
 * family. Lets bf_partition_setup poll real fam_rate values instead of
 * always seeing zeros (autotune is normally lazy, runs on first dispatch).
 * gpu_opencl_warm_probe is the synchronous unit; the async pair below
 * spawns one pthread per device so the probe wall time = max device
 * probe time (~250-400ms) regardless of device count. */
void gpu_opencl_warm_probe(int dev_idx, int op);
void gpu_opencl_warm_probe_async(int op);
void gpu_opencl_warm_probe_wait(void);

/* BF Phase 3b Tranche B (2026-05-10): gpu_opencl_dispatch_batch declaration
 * retired. Function body deleted from gpu/gpu_opencl.c in same commit; sole
 * call site (slab arm in gpu/gpujob_opencl.c) deleted; sole producer of
 * slab-format slots (gpu_try_pack at mdxfind.c) deleted. RCS history retains
 * prior signature. */

/* B7.9 (2026-05-07): gpu_opencl_dispatch_packed declaration retired.
 * The chokepoint pack at mdxfind.c was removed; this function had no
 * other production callers. RCS history retains the prior signature. */

/* Phase 0/1 GPU rule expansion engine. See project_gpu_rule_engine_design.md
 * and the comment block above the implementations in gpu_opencl.c. */
int gpu_opencl_set_rules(int dev_idx,
    const unsigned char *rule_program, uint32_t prog_len,
    const uint32_t *rule_offset, int n_rules);

uint32_t *gpu_opencl_dispatch_md5_rules(int dev_idx,
    const char *packed_words, uint32_t packed_size,
    const uint32_t *word_offset, uint32_t num_words,
    int op, int *nhits_out,
    uint64_t mask_start, uint32_t mask_offset_per_word, uint32_t bf_num_masks,
    uint32_t inner_iter,
    /* Phase 1.9 Tranche A1 (2026-05-10): bf_fast_eligible: when 1, the
     * dispatch may use the BF-fast MD5 template kernel
     * (kern_template_phase0_md5_bf, gpu_md5_bf.cl); when 0, uses the
     * slow MD5 template (kern_template_phase0, gpu_md5_core.cl).
     * Threaded through from the host-side BF chunk producer (see
     * mdxfind.c BF activation site). Default 0 (slow path). */
    int bf_fast_eligible);


/* Two-kernel pipeline prototype dispatch (Phase 4, 2026-05-19).
 * Gated behind MDXFIND_KERNEL_B_PROTO env flag.
 * Returns d->h_hits on success (same buffer as gpu_opencl_dispatch_md5_rules);
 * sets *nhits_out to the hit count. Returns NULL if env flag is unset,
 * build fails (Gate 6 soft-gate), or op != JOB_MD5MD5SALT. */
uint32_t *gpu_opencl_kernelb_dispatch_proto(int dev_idx,
    const char *packed_words, uint32_t packed_size,
    const uint32_t *word_offset, uint32_t num_words,
    int op, int *nhits_out);

/* Accessor for post-kernel-A plaintext readback. Returns the post-rule
 * plaintext bytes for slot hit_widx (= entry[0] from the hit buffer).
 * Sets *plen_out to the byte count. Valid only after
 * gpu_opencl_kernelb_dispatch_proto returns non-NULL with nhits_out > 0.
 * Returns NULL if out of range or readback was not populated. */
const char *gpu_opencl_kernelb_proto_plaintext(
    int dev_idx, uint32_t hit_widx, int *plen_out);

/* Phase 5 stage-timing accessor (2026-05-20): per-thread last-dispatch
 * stage micros captured inside gpu_opencl_kernelb_dispatch_proto via
 * clGetEventProfilingInfo. Each output pointer is optional (NULL skips).
 * Returns 1 if the last proto dispatch produced valid timings, 0 otherwise
 * (queue not profiling-enabled OR profiling-info read failed OR no dispatch
 * fired on this thread). See gpu_opencl.c implementation comment for
 * per-stage definitions. */
int gpu_opencl_kernelb_last_stage_us(uint64_t *ka_us, uint64_t *gap_us,
                                     uint64_t *kb_us, uint64_t *qa_us);

/* Diagnostics accessors — used by gpujob_opencl.c end-of-run report. */
const char *gpu_opencl_device_name(int dev_idx);
void gpu_opencl_device_bdf(int dev_idx, char *out, size_t out_sz);

/* BF Phase 1.6 (2026-05-09): per-device stable identifier (16 hex chars,
 * NUL-terminated; out_sz must be >= 17). FNV-1a 64-bit over
 * (CL_DEVICE_NAME|CL_DRIVER_VERSION|CL_DEVICE_VENDOR). Mirrors the
 * existing dynsize sidecar UUID derivation. Used by mdxfind.c BF servo
 * sidecar persistence (~/.mdxfind/dynsize/<dev_uuid>/bf_<op>.txt).
 * Empty string on out-of-range dev_idx. */
void gpu_opencl_dev_uuid(int dev_idx, char *out, size_t out_sz);

#ifdef __cplusplus
}
#endif

#endif /* OPENCL_GPU */
#endif /* GPU_OPENCL_H */
