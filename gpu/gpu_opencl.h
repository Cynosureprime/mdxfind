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

#ifdef __cplusplus
extern "C" {
#endif

/* Fail-fast GPU error macro. Any cl_int != CL_SUCCESS in a production
 * (post-init) hot path triggers immediate process termination via _Exit(1).
 *
 * Rationale (mmt run #77, 2026-05-01): a CL_OUT_OF_RESOURCES on the first
 * dispatch was logged-and-ignored; the next dispatch then failed with
 * CL_INVALID_EVENT because the prior write event was corrupt; mdxfind
 * kept going and exited "successfully" with whatever cracks survived
 * non-failing paths. The same workload retried clean produces 21,289
 * cracks bit-exact (canonical truth). Some of the recent jitter
 * (21,285/21,287/21,288/21,289) was therefore silent dispatch failures,
 * not NVIDIA non-determinism.
 *
 * _Exit (NOT exit) bypasses atexit handlers and stdio buffered-output
 * flushes that could mask the failure or paper over partial state.
 * stdout is flushed first so any cracks emitted before the failure are
 * preserved on the consumer's side.
 *
 * Sites that legitimately retry (probe paths, init-time device skip,
 * device tuning) MUST NOT use this macro — they handle CL errors via
 * graceful early-return and document why retry is correct. Any new
 * caller of GPU_FATAL must be in a post-init production path where
 * a CL error genuinely indicates a corrupted GPU state.
 *
 * Async error callback usage: the OpenCL spec does not permit
 * longjmp-out-of-driver-thread, but _Exit is async-signal-safe and
 * terminates the process without stack unwinding — the driver's
 * internal state never gets a chance to corrupt anything else.
 */
#define GPU_FATAL(fmt, ...) do {                                      \
    fflush(stdout);                                                   \
    fprintf(stderr, "FATAL: GPU error: " fmt "\n", ##__VA_ARGS__);    \
    fprintf(stderr, "FATAL: at %s:%d\n", __FILE__, __LINE__);         \
    fflush(stderr);                                                   \
    _Exit(1);                                                         \
} while (0)

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
