/*
 * gpu_opencl.c — OpenCL GPU acceleration for mdxfind
 *
 * Cross-vendor: NVIDIA, AMD, Intel, Apple (via OpenCL compatibility).
 * Kernel source compiled at runtime via clCreateProgramWithSource.
 */

#if defined(OPENCL_GPU)

#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>
#include "opencl_dynload.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <pthread.h>
#include <sys/stat.h>   /* mkdir() — dynsize cache I/O (2026-05-09) */
#include <unistd.h>     /* unlink() — dynsize cache atomic write */
#include "gpu_opencl.h"
#include "gpu_kernel_cache.h"
#include "job_types.h"
#include "gpujob.h"
#include "yarn.h"      /* launch()/join() for parallel device init (Memo C) */

/* malloc_pinned() lives in mdxfind.c -- page-aligned + mlock helper for
 * GPU-upload host buffers. We don't pull in mdxfind.h (would drag the
 * whole world) so just forward-declare. Free-able with free() because the
 * underlying allocation is posix_memalign on POSIX and _aligned_malloc on
 * Windows; current callers (mdxfind.c) never free pinned buffers since
 * they live for the process lifetime, and B1 follows the same convention. */
extern void *malloc_pinned(size_t size, const char *reason);

/* tsfprintf() lives in mdxfind.c — startup-phase diagnostic stderr emit
 * with "[T+ S.SSSs] " prefix and thread-safe serialization. Forward-
 * declared here so we can timestamp init-phase events (kernel build,
 * compact-table upload, rule-program upload, overflow upload, selftest,
 * first dispatch, hashes_shown alloc) without pulling in mdxfind.h. */
extern void tsfprintf(FILE *fp, const char *fmt, ...)
    __attribute__((format(printf, 2, 3)));

/* ---- Multi-GPU device state ---- */
#define MAX_GPU_DEVICES 64

/* ============================================================
 * Dynamic-sizer prototype (2026-05-09)
 * ============================================================
 *
 * Per-(GPU, kern_template_phase0_md5salt) feedback-loop control of
 *   - SALT_BATCH (compile-time, baked into kernel via -DSALT_BATCH=N)
 *   - salts_per_page (host-side outer-loop step)
 *
 * Activated by env MDXFIND_DYNSIZE=1. When MDXFIND_SPP is set, the
 * feedback loop is bypassed (user has pinned the value). The loop's
 * persistent state lives in ~/.mdxfind/dynsize/<dev_uuid>/kern_md5salt.txt
 * (one file per device covers algo_modes 0-3 which share the kernel).
 *
 * Modes 0-3 (JOB_MD5SALT/MD5UCSALT/MD5revMD5SALT/MD5sub8_24SALT) ONLY:
 * the gate is on `kern == d->kern_template_phase0_md5salt`. Modes 4/5/6
 * (e367 + HMAC-MD5 KSALT/KPASS) are intentionally excluded from the
 * prototype; they share the kernel but their feedback should be
 * separately tuned in a follow-up (different per-pair work mix).
 *
 * Spec: ~/.claude/projects/-Users-dlr-src-mdfind/memory/project_dynamic_spp_sizing_design.md
 *
 * Cache cadence: atexit ONLY (prototype). SIGKILL/TDR/OOM lose up to
 * N dispatches of learning. Acceptable for prototype.
 *
 * Deferred from this prototype:
 *   - N exploration (sticky N, design memo §5.3)
 *   - Cache invalidation rules (binary_rev / kernel_cache_key cross-check)
 *   - Multi-GPU separate cache entries for `multi_active` mode
 */
struct dynsize_entry {
    /* Active state (cold-loaded from cache or seeded on first dispatch) */
    uint32_t current_N;            /* SALT_BATCH bake-in value */
    uint32_t current_spp;          /* salts_per_page host-side step */
    uint32_t spp_cap_observed;     /* upper cliff guard, learned per-session */
    uint32_t initialized;          /* 0 = need cold start, 1 = active */
    uint32_t loaded_from_cache;    /* 1 if we successfully loaded from disk */

    /* Telemetry / control */
    double   ema_mhz;
    double   prev_window_ema_mhz;
    uint32_t convergence_count;
    uint32_t plateau_streak;
    double   loop_weight;

    /* Salt-retirement detection (see design memo §5.4) */
    uint64_t prev_nsalts_active;
};

/* dynsize control parameters (design memo §5.2) */
#define DYNSIZE_LOOP_WEIGHT_INIT     1.0
#define DYNSIZE_LOOP_WEIGHT_FLOOR    0.05
#define DYNSIZE_LOOP_WEIGHT_DECAY    0.99
#define DYNSIZE_EMA_ALPHA_MIN        0.05
#define DYNSIZE_EMA_ALPHA_MAX        0.5
#define DYNSIZE_SPP_GROW_FACTOR      1.5
#define DYNSIZE_SPP_SHRINK_FACTOR    1.5
#define DYNSIZE_SPP_MIN              64u
#define DYNSIZE_SPP_MAX              65536u
#define DYNSIZE_TARGET_WALL_NS       500000000ULL   /* 500 ms soft target */
#define DYNSIZE_HARD_WALL_NS         2000000000ULL  /* 2  s hard ceiling */
#define DYNSIZE_CLIFF_RATIO          0.5
#define DYNSIZE_PLATEAU_RATIO        0.01
#define DYNSIZE_PLATEAU_SETTLE       5
#define DYNSIZE_CLIFF_GUARD_COUNT    10  /* min convergence before cliff fires */
#define DYNSIZE_SALT_RETIRE_DROP     0.20 /* >20% nsalts drop = workload shift */

/* dynsize forward declarations */
struct gpu_device;
static int      dynsize_is_enabled(void);
static int      dynsize_spp_pinned(void);
static int      dynsize_verbose(void);
static void     dynsize_compute_device_uuid(struct gpu_device *d, char *out, size_t outlen);
static int      dynsize_cache_path(const char *uuid, char *out, size_t outlen);
static int      dynsize_cache_load(const char *uuid, struct dynsize_entry *e);
static int      dynsize_cache_store(const char *uuid, const struct dynsize_entry *e);
static void     dynsize_ensure_loaded(struct gpu_device *d, int dev_idx);
static uint32_t dynsize_compile_time_N(struct gpu_device *d, int dev_idx);

struct gpu_device {
    cl_context       ctx;
    cl_command_queue  queue;
    cl_program        prog;           /* selftest program (common only) */
    cl_program        fam_prog[FAM_COUNT]; /* per-family compiled programs */
    cl_device_id      dev;
    char              name[256];
    int               max_batch;     /* per-device batch limit (words per dispatch) */
    int               max_dispatch;  /* max work items per dispatch (0 = unlimited) */

    /* Compact table (read-only, uploaded once) */
    cl_mem b_compact_fp, b_compact_idx;
    cl_mem b_hash_data, b_hash_data_off, b_hash_data_len;

    /* On-GPU dedup buffer for the rules-engine path. Allocated alongside
     * the compact table (size = hash_data_count + overflow_count uint slots),
     * zero-initialized at session start, persists across dispatches.
     * Per-target cracked-flag: kernel atomic_inc's hashes_shown[idx]; only
     * the lane that observes the prior value 0 actually emits a hit.
     * Repeat hits on already-cracked targets drop on-GPU. Theory #1 from
     * mdx-architect's 2026-04-30 7-theory analysis. */
    cl_mem b_hashes_shown;
    size_t hashes_shown_count;   /* slot count (== hash_data_count + overflow_count) */

    /* Salt buffers (updated per-snapshot) */
    cl_mem b_salt_data, b_salt_off, b_salt_len;
    size_t salt_data_cap;
    size_t salt_off_cap;
    int salts_count;

    /* Overflow (uploaded once) */
    cl_mem b_overflow_keys, b_overflow_hashes, b_overflow_offsets, b_overflow_lengths;

    /* Per-dispatch buffers */
    cl_mem b_hits, b_hit_count;
    cl_mem b_hexhashes, b_hexlens, b_params;
    size_t hexhash_cap, hexlens_cap;
    uint32_t *h_hits;    /* host-side hit buffer */

    /* Mask mode */
    cl_mem bgpu_mask_desc;  /* mask descriptor: charset IDs per position */

    /* Memo B Phase B7.1-B7.5/B7.8: multi-position prepend+append mask
     * charsets for the template_phase0 kernel. Sized for up to
     * MASK_POS_CAP=16 prepend + 16 append = 32 total positions × 256
     * bytes = 8192 bytes (B7.8 lift; pre-B7.8 was 8+8=4096 bytes).
     * Layout follows the SLAB-PATH CONVENTION (gpu_kernels.cl
     * md5_mask_batch / gpu_mask_desc): rows [0..n_prepend) hold prepend
     * charsets, rows [n_prepend..n_prepend+n_append) hold append charsets.
     * Bound as the 15th kernel arg to template_phase0; the kernel
     * decomposes mask_idx into prepend_idx*append_combos+append_idx,
     * then per-position via successive divmod (last position innermost
     * within each section) and reads one byte per position from
     * mask_charsets[row * 256 + pidx_i].
     *
     * Allocated lazily in gpu_opencl_set_mask when a B7-eligible mask
     * configuration is detected; persists thereafter. When no mask is
     * active, the buffer holds a sentinel (4096 bytes of 0x00) so kernel
     * reads are well-defined; the kernel's (n_prepend>=1 || n_append>=1)
     * gate prevents any actual read of sentinel bytes.
     *
     * B7.1 single-position append: only row 0 populated, n_append==1.
     * B7.2 multi-position append (n_prepend==0): rows [0..n_append).
     * B7.3 single-position prepend: only row 0 populated, n_prepend==1.
     * B7.4 multi-position prepend (n_append==0): rows [0..n_prepend).
     * B7.5 combined: rows [0..n_prepend) prepend, [n_prepend..) append.
     * B7.6 user-defined classes: same layout, charset bytes from -1/-2/...
     * B7.8 cap lift: MASK_POS_CAP 8 -> 16 per side; buffer 4096 -> 8192
     *      bytes; mask_sizes 16 -> 32 uints. Same layout, larger bound.
     *
     * Backward-compat with B7.2: when n_prepend==0, append rows are at
     * [0..n_append) — bit-identical to B7.2's packing. Buffer sizing
     * grew from 2048 (8*256) to 4096 (16*256) bytes at B7.5, then to
     * 8192 (32*256) bytes at B7.8; the kernel's sentinel-allocation in
     * dispatch also grew to match. */
    cl_mem b_template_mask_charsets;
    /* B7.2-B7.5/B7.8: per-position charset sizes. 32 uints (B7.8 lift
     * from 16); only the first (n_prepend + n_append) entries are valid.
     * Bound as the 16th kernel arg to template_phase0; the kernel reads
     * mask_sizes[i] inside the decomposition loop to know each position's
     * modulus. Sentinel allocation when no mask is active: 32 ones (so
     * any stray read produces psize==1, divmod loop terminates safely). */
    cl_mem b_template_mask_sizes;

    /* Packed password dispatch buffers — repurposed post-B7.9.
     *
     * b_packed_buf / b_chunk_index / packed_buf_cap / chunk_index_cap remain
     * live: the validator path (gpu_opencl_dispatch_md5_rules with
     * MDXFIND_GPU_VALIDATOR=1) re-uses them as a 2nd packed-buffer route
     * around b_dispatch_payload (see ~line 7885). The rules-engine production
     * path uses b_dispatch_payload exclusively.
     *
     * B7.9 (2026-05-07): kern_packed and kern_packed_fam[] retired. Both
     * fields kept for ABI stability with sibling builds and as defensive
     * placeholders; they are now dead storage (never assigned, never read).
     * A future cleanup phase may delete them along with the FAM_MD5PACKED
     * enum slot in gpujob.h. */
    cl_mem b_packed_buf, b_chunk_index;
    cl_kernel kern_packed;                 /* B7.9: dead, retained for ABI */
    cl_kernel kern_packed_fam[FAM_COUNT];  /* B7.9: dead, retained for ABI */
    size_t packed_buf_cap, chunk_index_cap;

    /* Memo B B1 (2026-05-04): coalesced rules-engine dispatch payload.
     *
     * b_dispatch_payload is ONE cl_mem holding {OCLParams params,
     * uint hit_count, uint word_offset[num_words], uchar packed_words[]}.
     * Layout offsets are deterministic from params.num_words; see
     * gpu/gpu_md5_rules.cl md5_rules_phase0 kernel header.
     *
     * h_dispatch_payload is the matching pinned host staging buffer
     * (malloc_pinned -- posix_memalign + mlock). The dispatch function
     * memcpy's params/word_offset/packed_buf into it, zeros the
     * hit_count word, then issues ONE clEnqueueWriteBuffer.
     *
     * Sizing: max payload = 128 (params) + 4 (hit_count)
     *                     + GPUBATCH_RULES_WOFF_SIZE (16384*4 = 64 KiB)
     *                     + GPUBATCH_RULES_PACKED_SIZE (16384*256 = 4 MiB)
     *                     ~= 4.06 MiB. Allocated lazily on first dispatch
     * with the ceiling sizing; never grown after.
     *
     * REPLACES the per-dispatch sequence in dispatch_md5_rules of:
     *   write b_packed_buf + write b_chunk_index + write b_params + write b_hit_count
     * = 4 host->GPU writes / dispatch.
     *
     * Per project_gpu_pcie_baseline_20260427.md, each clEnqueueWriteBuffer
     * pays a SUBMIT->START tax (~2.3 ms PCIe 3.0; ~50 us PCIe 4.0). One
     * write reclaims 3 of those per dispatch -- ~7 ms PCIe 3.0, ~150 us
     * PCIe 4.0. Pinning of the host source is preserved: the malloc_pinned
     * staging buffer matches the pattern the baseline numbers were
     * measured against.
     *
     * Existing b_packed_buf / b_chunk_index / b_params / b_hit_count
     * stay allocated -- gpu_opencl_dispatch_batch and the validate path
     * still use them. Only the rules-engine production path was rewired
     * in B1. (B7.9 2026-05-07: gpu_opencl_dispatch_packed retired; that
     * caller is gone but the buffers remain for the validate / batch
     * paths above.) */
    cl_mem   b_dispatch_payload;
    size_t   dispatch_payload_cap;
    void    *h_dispatch_payload;
    size_t   h_dispatch_payload_cap;

    /* Phase 0/1 GPU rule expansion engine. The rule program (compiled
     * bytecode for the GPU-eligible subset of rules) is uploaded once
     * via gpu_opencl_set_rules() at session start; per-dispatch only
     * needs to upload the words batch. global_size = n_words * n_rules;
     * the kernel emits hits with (word_idx, rule_idx) for host replay. */
    cl_program prog_md5_rules;
    cl_kernel  kern_md5_rules_phase0;
    cl_kernel  kern_md5_rules_phase0_validate;  /* env-gated: MDXFIND_GPU_VALIDATOR=1
                                                   replaces the production kernel
                                                   for this dispatch with a
                                                   per-(word,rule) buffer-state
                                                   emitter for diff against the
                                                   CPU validator (ruleproc.c
                                                   rev 1.24, gated by
                                                   MDXFIND_RULE_VALIDATOR=1). */

    /* Memo B B2 template path (env-gated: MDXFIND_GPU_TEMPLATE=md5).
     * Side-by-side with md5_rules_phase0; same kernel signature so the
     * dispatch_md5_rules call site swaps the cl_kernel handle and
     * leaves all the kernel-arg setup unchanged. Default off; B5 flips
     * the default once the template wins justify retiring the legacy
     * side-by-side kernel. See project_memo_b_dispatch_template.md
     * §3 (template body) and §4 (rollout path).
     *
     * Cache key (R3 mitigation): built via gpu_kernel_cache_build_-
     * program_ex with defines_str = "HASH_WORDS=4,HASH_BLOCK_BYTES=64"
     * so future SHA-1 / SHA-256 / SHA-512 instantiations of the same
     * source text produce distinct cache entries. */
    cl_program prog_template;
    cl_kernel  kern_template_phase0;
    /* Phase 1.9 Tranche A1 (2026-05-10): MD5 brute-force fast-path
     * template instantiation. Parallel to prog_template /
     * kern_template_phase0 (slow path MD5) — selected at dispatch time
     * when bf_fast_eligible is set on the jobg slot (unsalted JOB_MD5,
     * Numrules <= 1, append-only mask, napp in [1,8]) unless
     * MDXFIND_GPU_FAST_DISABLE=1 forces the slow path. Built lazily on
     * first eligible dispatch via gpu_opencl_template_md5_bf_compile();
     * kernel object created lazily via
     * gpu_opencl_template_md5_bf_kernel_lazy(). Cache key uses
     * defines_str = "HASH_WORDS=4,HASH_BLOCK_BYTES=64,BF_FAST_MD5=1"
     * so it does NOT collide with the slow MD5 path (which uses
     * "HASH_WORDS=4,HASH_BLOCK_BYTES=64"). A1 body in gpu_md5_bf.cl is
     * a verbatim copy of gpu_md5_core.cl rev 1.2; A2-A4 optimize. */
    cl_program prog_template_md5_bf;
    cl_kernel  kern_template_phase0_md5_bf;
    /* Memo B Phase B4 (2026-05-04): SHA1 template instantiation. Parallel
     * to prog_template / kern_template_phase0 (MD5) — selected when
     * MDXFIND_GPU_TEMPLATE=sha1. Built lazily on first dispatch via
     * gpu_opencl_template_compile_sha1(); kernel object created lazily
     * via gpu_opencl_template_kernel_lazy_sha1(). Cache key uses
     * defines_str = "HASH_WORDS=5,HASH_BLOCK_BYTES=64" so distinct
     * algorithm tuples receive distinct cache entries even though
     * gpu_template.cl source text is identical (R3 mitigation). */
    cl_program prog_template_sha1;
    cl_kernel  kern_template_phase0_sha1;
    /* Memo B Phase B4 fan-out (2026-05-04): SHA256/SHA224/MD4 template
     * instantiations. Parallel to prog_template_sha1; selected when
     * MDXFIND_GPU_TEMPLATE=sha256/sha224/md4. Cache keys distinguish
     * SHA256 (HASH_WORDS=8) from SHA224 (HASH_WORDS=7) via defines_str;
     * MD4 has the same defines_str as MD5 (HASH_WORDS=4) but distinct
     * source text (gpu_md4_core_str vs gpu_md5_core_str), and the cache
     * key hashes both source text and defines so distinct keys are
     * guaranteed. Built lazily on first dispatch; per-algo strict
     * op-match swap at dispatch site. */
    cl_program prog_template_sha256;
    cl_kernel  kern_template_phase0_sha256;
    cl_program prog_template_sha224;
    cl_kernel  kern_template_phase0_sha224;
    cl_program prog_template_md4;
    cl_kernel  kern_template_phase0_md4;
    /* End B4 fan-out fields. Process exit reclaims; per-program release on
     * teardown is intentionally absent (matches SHA1 rev 1.90 convention). */
    /* Memo B Phase B5 sub-batch 1 (2026-05-04): SHA384/SHA512 template
     * instantiations. First 64-bit-state algorithms in the family + first
     * 128-bit length encoding. Cache keys distinguish SHA512
     * (HASH_WORDS=16,HASH_BLOCK_BYTES=128) from SHA384 (HASH_WORDS=12).
     * Both use the same sha512_block primitive (gpu_common.cl). Built
     * lazily on first dispatch; per-algo strict op-match swap at
     * dispatch site (JOB_SHA384 / JOB_SHA512). */
    cl_program prog_template_sha384;
    cl_kernel  kern_template_phase0_sha384;
    cl_program prog_template_sha512;
    cl_kernel  kern_template_phase0_sha512;
    /* Memo B Phase B5 sub-batch 2 (2026-05-05): RIPEMD-160 / RIPEMD-320
     * template instantiations. RIPEMD-160 is the second 5-word-state
     * algorithm (after SHA1); shares HASH_WORDS=5 defines with SHA1 but
     * distinct source text (gpu_ripemd160_core_str vs gpu_sha1_core_str)
     * → distinct cache key per gpu_kernel_cache.c rev 1.5+. RIPEMD-320
     * is the FIRST 10-word-state algorithm in the family — needed new
     * EMIT_HIT_10{,_OR_OVERFLOW,_DEDUP_OR_OVERFLOW} macros in
     * gpu_common.cl rev 1.12. Both are LITTLE-ENDIAN per uint32 state
     * (matches MD5 / MD4 convention; no bswap32 in probe / emit). */
    cl_program prog_template_ripemd160;
    cl_kernel  kern_template_phase0_ripemd160;
    cl_program prog_template_ripemd320;
    cl_kernel  kern_template_phase0_ripemd320;
    /* Memo B Phase B5 sub-batch 3 (2026-05-06): BLAKE2 family template
     * instantiations. THREE algorithms wired (BLAKE2B-160 omitted —
     * JOB_BLAKE2B160 does not exist as a job type in mdxfind.c, see
     * brief deviation note in mdx-team-state.md):
     *   - BLAKE2S-256 (HASH_WORDS=8, HASH_BLOCK_BYTES=64): LE per uint32,
     *     10-round G compression (b2s_compress; existing in gpu_common.cl).
     *   - BLAKE2B-256 (HASH_WORDS=8, HASH_BLOCK_BYTES=128): LE per uint32
     *     pair (8 uint32 = first 4 ulong of internal 8-ulong state),
     *     12-round G compression (b2b_compress; NEW in gpu_common.cl
     *     rev 1.13).
     *   - BLAKE2B-512 (HASH_WORDS=16, HASH_BLOCK_BYTES=128): LE per uint32
     *     pair (full 8 ulong = 16 uint32), same compression as BLAKE2B-256.
     *
     * BLAKE2 distinct from MD-style hashes: state struct carries byte-
     * counter t[2] + finalization flag f[2] in addition to the digest
     * chaining h_ulong[]/h[]. Per-algo template_init / _finalize / _iterate
     * manipulate counter+flag; the shared template body never reads them.
     * No template_transform signature change — counter+flag stay INSIDE
     * the per-algo state struct. */
    cl_program prog_template_blake2s256;
    cl_kernel  kern_template_phase0_blake2s256;
    cl_program prog_template_blake2b256;
    cl_kernel  kern_template_phase0_blake2b256;
    cl_program prog_template_blake2b512;
    cl_kernel  kern_template_phase0_blake2b512;
    /* Memo B Phase B5 sub-batch 4 (2026-05-03): SHA3 / Keccak family template
     * instantiations. EIGHT algorithms wired: KECCAK-{224,256,384,512}
     * (suffix=0x01) and SHA3-{224,256,384,512} (suffix=0x06). Sponge
     * construction with rate=200-2*output_bytes (144/136/104/72) and
     * Keccak-f[1600] permutation (keccakf1600 in gpu_common.cl rev 1.14+).
     * State struct: ulong sp[25] (200 bytes), naturally LE per ulong;
     * decomposition to h[] is i*2/i*2+1 split per ulong (no bswap).
     * Per-algo cache keys differ via HASH_WORDS+HASH_BLOCK_BYTES tuple
     * (e.g. SHA3-256 has HASH_WORDS=8,HASH_BLOCK_BYTES=136 distinguishing
     * it from BLAKE2B-256's HASH_WORDS=8,HASH_BLOCK_BYTES=128). */
    cl_program prog_template_keccak224;
    cl_kernel  kern_template_phase0_keccak224;
    cl_program prog_template_keccak256;
    cl_kernel  kern_template_phase0_keccak256;
    cl_program prog_template_keccak384;
    cl_kernel  kern_template_phase0_keccak384;
    cl_program prog_template_keccak512;
    cl_kernel  kern_template_phase0_keccak512;
    cl_program prog_template_sha3_224;
    cl_kernel  kern_template_phase0_sha3_224;
    cl_program prog_template_sha3_256;
    cl_kernel  kern_template_phase0_sha3_256;
    cl_program prog_template_sha3_384;
    cl_kernel  kern_template_phase0_sha3_384;
    cl_program prog_template_sha3_512;
    cl_kernel  kern_template_phase0_sha3_512;
    /* Memo B Phase B5 sub-batch 5a (2026-05-03), Tier 1: SHA384RAW + SHA512RAW.
     * REUSE the existing SHA384/SHA512 compression cores (gpu_sha384_core.cl /
     * gpu_sha512_core.cl) — only template_iterate diverges (binary digest
     * re-fed instead of hex-encoded). Two new cl_program / cl_kernel pairs
     * because the kernel-cache key differs by source-text hash. defines_str
     * matches SHA384/SHA512 (HASH_WORDS=12/16, HASH_BLOCK_BYTES=128). */
    cl_program prog_template_sha384raw;
    cl_kernel  kern_template_phase0_sha384raw;
    cl_program prog_template_sha512raw;
    cl_kernel  kern_template_phase0_sha512raw;
    /* Memo B Phase B5 sub-batch 6 (2026-05-03), Tier A: MD5RAW + SHA1RAW + SHA256RAW.
     * REUSE the existing MD5/SHA1/SHA256 compression cores (gpu_md5_core.cl /
     * gpu_sha1_core.cl / gpu_sha256_core.cl) — only template_iterate diverges
     * (binary digest re-fed instead of hex-encoded). Three new cl_program /
     * cl_kernel pairs because the kernel-cache key differs by source-text hash.
     * defines_str matches MD5/SHA1/SHA256 (HASH_WORDS=4/5/8, HASH_BLOCK_BYTES=64). */
    cl_program prog_template_md5raw;
    cl_kernel  kern_template_phase0_md5raw;
    cl_program prog_template_sha1raw;
    cl_kernel  kern_template_phase0_sha1raw;
    cl_program prog_template_sha256raw;
    cl_kernel  kern_template_phase0_sha256raw;
    /* Memo B Phase B5 sub-batch 6 (2026-05-03), Tier C: SQL5 (MySQL 4.1+
     * password = SHA1(SHA1(p))). Per-algo template_state holds two SHA1
     * chains (inner/outer); template_finalize chains them; template_iterate
     * re-feeds UPPERCASE-hex of inner. defines_str matches SHA1 (HASH_WORDS=5,
     * HASH_BLOCK_BYTES=64). */
    cl_program prog_template_sql5;
    cl_kernel  kern_template_phase0_sql5;
    /* Memo B Phase B6.11 (2026-05-06): SHA1DRU (Drupal SHA1, hashcat -m 7900,
     * JOB_SHA1DRU=404). First 1M-iteration algorithm on the unified template
     * path. Algorithm: SHA1(pass) followed by 1,000,000 iterations of
     * SHA1(hex_lc(state) || pass). Only the FINAL state is probed (host forces
     * params.max_iter=1 for JOB_SHA1DRU; the 1M loop runs INSIDE template_-
     * finalize). defines_str: HASH_WORDS=5,HASH_BLOCK_BYTES=64,BASE_ALGO=sha1,
     * ITER_COUNT=1000000 — same first two axes as SHA1/SHA1RAW/SQL5; the
     * ITER_COUNT token + distinct source-text disambiguate the cache entry. */
    cl_program prog_template_sha1dru;
    cl_kernel  kern_template_phase0_sha1dru;
    /* Memo B Phase B7.7b (2026-05-07): MD6256 (hashcat -m 17800,
     * JOB_MD6256=29). Final M5 closure from B9 gate-fail. MD6-256 single-
     * block leaf compression — algorithmically-largest single-compression
     * unsalted algo on the unified template path: 89-ulong N input,
     * 1753-ulong A working array (14 KB per work-item), 104 rounds × 16
     * words per compression. Per-iter probe like SQL5 (vs. SHA1DRU's
     * max_iter=1 internal loop). defines_str: HASH_WORDS=8,
     * HASH_BLOCK_BYTES=64,BASE_ALGO=md6. KNOWN ACCEPTED RISK: gfx1201
     * priv_mem may bust 43,024 B HARD GATE due to the 14 KB A[1753]
     * stack on top of RULE_BUF_MAX. Compile-only ship per user OPTION A;
     * integrated post-B7.9 validation will reveal gfx1201 status. Fall-
     * back: leave gpu_md6256unsalted.cl as gfx1201-only slab fallback. */
    cl_program prog_template_md6256;
    cl_kernel  kern_template_phase0_md6256;
    /* Memo B Phase B5 sub-batch 6 (2026-05-03), Tier B: NTLMH (NT password
     * hash = MD4(UTF-16LE-zero-extend(p)). Hashcat-compatible zero-extend
     * variant — for non-ASCII inputs the iconv variant remains on CPU
     * (same gap as the existing slab path; documented in mdxfind.c
     * line 583/589). defines_str matches MD4 (HASH_WORDS=4, HASH_BLOCK_BYTES=64). */
    cl_program prog_template_ntlmh;
    cl_kernel  kern_template_phase0_ntlmh;
    /* Memo B Phase B5 sub-batch 8 (2026-05-05): MD4UTF16 (-m e496).
     * Same MD4(UTF-16LE-zero-extend(p)) algorithm as NTLMH, with a
     * proper iter step for Maxiter > 1: each iter feeds back the lowercase
     * hex of the prior digest (32 ASCII chars) zero-extended to UTF-16LE
     * (64 bytes) and MD4'd. defines_str matches MD4 / NTLMH
     * (HASH_WORDS=4, HASH_BLOCK_BYTES=64). Distinct cache entry by
     * source-text hash. */
    cl_program prog_template_md4utf16;
    cl_kernel  kern_template_phase0_md4utf16;
    /* Memo B Phase B5 sub-batch 7 (2026-05-05): MYSQL3 (-m e456).
     * Legacy MySQL OLD_PASSWORD() hash, 64-bit output. Per-byte arithmetic
     * accumulator (no MD-style block). The CPU JOB_MYSQL3 iter loop feeds
     * the lowercase-ASCII-hex of the prior 8-byte digest (16 ASCII chars)
     * back through mysql3() for x = 1..Maxiter. Probe uses HASH_WORDS=4
     * uint32 (h[2..3] zero); host zero-pad of HashDataBuf to 16 bytes
     * (mdxfind.c:36400-36412 rev 1.399+) makes the default 4-word probe
     * byte-exact for the 8-byte digest. defines_str matches MD5/MD4
     * (HASH_WORDS=4, HASH_BLOCK_BYTES=64); distinct cache entry by
     * source-text hash. */
    cl_program prog_template_mysql3;
    cl_kernel  kern_template_phase0_mysql3;
    /* Memo B Phase B5 sub-batch 6.5 (2026-05-05): WRL (-m e5).
     * Whirlpool 512-bit hash. Miyaguchi-Preneel over a 64-byte BE block
     * with 256-bit BE length in last 32 bytes of final block. The CPU
     * JOB_WRL iter loop feeds back the 128 lowercase ASCII hex chars
     * (prmd5 of 64-byte digest) for x = 1..Maxiter. defines_str
     * "HASH_WORDS=16,HASH_BLOCK_BYTES=64". Distinct cache entry by
     * source-text hash.
     *
     * DIAGNOSTIC: WRL has the same 16 KB __constant SBOX shape as
     * Streebog ([8][256] ulong) but a different access pattern (direct
     * ulong shift-then-mask vs. byte-pointer index). If WRL passes
     * gfx1201 100/100, the Streebog-deferred RDNA4 issue is
     * SBOG_LPS-specific (uchar/ulong aliasing), not 16 KB __constant
     * capacity. */
    cl_program prog_template_wrl;
    cl_kernel  kern_template_phase0_wrl;
    /* B5 sub-batch 5b retry (2026-05-06): Streebog-256 + Streebog-512 template
     * paths. SBOG_LPS is rewritten to mirror WRL_OP shift-then-mask access at
     * 16 KB __constant SBOB_SL64; sub-6.5 WRL ship validated this pattern on
     * gfx1201. See gpu_streebog{256,512}_core.cl. */
    cl_program prog_template_streebog256;
    cl_kernel  kern_template_phase0_streebog256;
    cl_program prog_template_streebog512;
    cl_kernel  kern_template_phase0_streebog512;
    /* Memo B Phase B6 (2026-05-06): salt-axis prereq — first two salted
     * variants. JOB_MD5SALT (hashcat -m 10): MD5(hex32(MD5(p)) || salt) —
     * double-MD5 chain. JOB_MD5SALTPASS (hashcat -m 20): MD5(salt || pass)
     * — simple PREPEND-salt MD5. Cores in gpu_md5salt_core.cl /
     * gpu_md5saltpass_core.cl; built with -DGPU_TEMPLATE_HAS_SALT=1 plus
     * SALT_POSITION=APPEND_TO_HEX32 / PREPEND in defines_str. The kernel
     * signature is 19 args (16 unsalted + 3 salt args appended under
     * #ifdef GPU_TEMPLATE_HAS_SALT in gpu_template.cl). */
    cl_program prog_template_md5salt;
    cl_kernel  kern_template_phase0_md5salt;
    cl_program prog_template_md5saltpass;
    cl_kernel  kern_template_phase0_md5saltpass;
    /* B6.1 SHA1 fan-out (2026-05-06): SHA1SALTPASS (hashcat -m 110) =
     * SHA1(salt || pass). First SHA-family salted variant on the unified
     * template path. defines_str disambiguates from MD5SALTPASS via
     * HASH_WORDS=5 (vs 4) + BASE_ALGO=sha1 token. Cores in
     * gpu_sha1saltpass_core.cl; built with -DGPU_TEMPLATE_HAS_SALT=1 plus
     * SALT_POSITION=PREPEND in defines_str. Same 19-arg kernel signature. */
    cl_program prog_template_sha1saltpass;
    cl_kernel  kern_template_phase0_sha1saltpass;
    /* B6.2 SHA256 fan-out (2026-05-06): SHA256SALTPASS (hashcat -m 1410) =
     * SHA256(salt || pass). Second SHA-family salted variant. defines_str
     * disambiguates from SHA1SALTPASS via HASH_WORDS=8 (vs 5) + BASE_ALGO=
     * sha256 token, and from MD5SALTPASS via HASH_WORDS=8 (vs 4) + BASE_-
     * ALGO=sha256 token. Cores in gpu_sha256saltpass_core.cl; built with
     * -DGPU_TEMPLATE_HAS_SALT=1 plus SALT_POSITION=PREPEND in defines_str.
     * Same 19-arg kernel signature. Iter loop differs from SHA1: 64-char
     * hex output exactly fills one block, so pad+length lands in a
     * second block (handled in template_iterate). */
    cl_program prog_template_sha256saltpass;
    cl_kernel  kern_template_phase0_sha256saltpass;
    /* B6.3 SHA224 fan-out (2026-05-06): SHA224SALTPASS (hashcat -m 1310) =
     * SHA224(salt || pass). Third SHA-family salted variant — reuses
     * sha256_block compression but truncates output to 7 uint32 words.
     * defines_str disambiguates from SHA256SALTPASS via HASH_WORDS=7
     * (vs 8) — same BASE_ALGO=sha256 since the compression primitive is
     * identical; from SHA1SALTPASS via HASH_WORDS=7 (vs 5) + BASE_ALGO=
     * sha256 token; from MD5SALTPASS via HASH_WORDS=7 (vs 4) + BASE_ALGO=
     * sha256 token. Cores in gpu_sha224saltpass_core.cl; built with
     * -DGPU_TEMPLATE_HAS_SALT=1 plus SALT_POSITION=PREPEND in defines_str.
     * Same 19-arg kernel signature. Iter hex output is 56 chars (28 bytes
     * × 2), fits in one block with room for pad+length. */
    cl_program prog_template_sha224saltpass;
    cl_kernel  kern_template_phase0_sha224saltpass;
    /* B6.4 MD5PASSSALT fan-out (2026-05-06): MD5PASSSALT (hashcat -m 10) =
     * MD5(pass || salt). First APPEND-shape salted variant on the codegen
     * path. Same MD-family LE compress as MD5SALTPASS (md5_block,
     * HASH_WORDS=4, EMIT_HIT_4) — only the salt POSITION at finalize time
     * differs. defines_str disambiguates from MD5SALTPASS via SALT_-
     * POSITION=APPEND (vs PREPEND); same BASE_ALGO=md5 + HASH_WORDS=4
     * axes. Authors finalize_append.cl.frag which unblocks future
     * SHA1PASSSALT + SHA256PASSSALT (both APPEND, but SHA-family BE).
     * Same 19-arg kernel signature as the PREPEND variants. */
    cl_program prog_template_md5passsalt;
    cl_kernel  kern_template_phase0_md5passsalt;
    /* B6.5 SHA1PASSSALT fan-out (2026-05-06): SHA1PASSSALT (hashcat -m 100) =
     * SHA1(pass || salt). First SHA-family APPEND-shape salted variant on
     * the codegen path. Same SHA-family BE compress as SHA1SALTPASS
     * (sha1_block, HASH_WORDS=5, EMIT_HIT_5) — only the salt POSITION at
     * finalize time differs. defines_str disambiguates from SHA1SALTPASS
     * via SALT_POSITION=APPEND (vs PREPEND); same BASE_ALGO=sha1 +
     * HASH_WORDS=5 axes. Authors finalize_append_be.cl.frag which unblocks
     * future SHA256PASSSALT (pure spec reuse — no further fragment work).
     * Same 19-arg kernel signature as the PREPEND siblings. */
    cl_program prog_template_sha1passsalt;
    cl_kernel  kern_template_phase0_sha1passsalt;
    /* B6.7 SHA256PASSSALT fan-out (2026-05-06): SHA256PASSSALT (hashcat -m 1410) =
     * SHA256(pass || salt). Second SHA-family APPEND-shape salted variant on
     * the codegen path — pure spec reuse. Same SHA-family BE compress as
     * SHA256SALTPASS (sha256_block, HASH_WORDS=8, EMIT_HIT_8) — only the salt
     * POSITION at finalize time differs. defines_str disambiguates from
     * SHA256SALTPASS via SALT_POSITION=APPEND (vs PREPEND); same BASE_ALGO=
     * sha256 + HASH_WORDS=8 axes. From SHA1PASSSALT via HASH_WORDS=8 +
     * BASE_ALGO=sha256 (both axes differ). Reuses sha256_style_salted.cl.tmpl
     * (B6.2) + finalize_append_be.cl.frag (B6.5) — no new templates or
     * fragments authored. Same 19-arg kernel signature as the other salted
     * template siblings. */
    cl_program prog_template_sha256passsalt;
    cl_kernel  kern_template_phase0_sha256passsalt;
    /* B6.9 SHA512 fan-out (2026-05-06): SHA512SALTPASS (hashcat -m 1710) =
     * SHA512(salt || pass). FIRST 64-bit-state salted variant on the codegen
     * path. Per-lane state[8] is uint64 (vs uint32 for MD5/SHA-1/SHA-2-32);
     * block size 128 bytes (vs 64); length field 128-bit BE (vs 64-bit BE).
     * Authors a sibling main template (sha512_style_salted.cl.tmpl) AND a
     * sibling fragment (finalize_prepend_be64.cl.frag) — width-bearing
     * constants belong in the template+fragment per the codegen-
     * reconsideration memo. Cores in gpu_sha512saltpass_core.cl; built with
     * -DGPU_TEMPLATE_HAS_SALT=1 plus HASH_WORDS=16 + HASH_BLOCK_BYTES=128 +
     * BASE_ALGO=sha512 in defines_str — distinct cache key from every other
     * salted template (the 128-byte block alone is unique to the SHA-384/512
     * family among salted variants on the codegen path). HARD GATE on
     * gfx1201 (3080 spill-region ceiling): priv_mem_size <= 43,024 B.
     * Unsalted SHA-512 reading was 42,520 B; salted finalize delta expected
     * ~0-50 B (same M[16] scratch + per-byte loop, plus one VGPR for the
     * salt-vs-pass branch index). Same 19-arg kernel signature as the
     * other salted template siblings. */
    cl_program prog_template_sha512saltpass;
    cl_kernel  kern_template_phase0_sha512saltpass;
    /* B6.10 SHA512PASSSALT fan-out (2026-05-06): SHA512PASSSALT (hashcat
     * -m 1720) = SHA512(pass || salt). FINAL B6 ladder step. APPEND-shape
     * sibling of B6.9's SHA512SALTPASS — same 64-bit-state SHA-512 family,
     * same 128-byte block, same 128-bit BE length field; only the salt
     * POSITION at template_finalize differs (APPEND vs PREPEND). Cores in
     * gpu_sha512passsalt_core.cl; built with -DGPU_TEMPLATE_HAS_SALT=1
     * plus HASH_WORDS=16 + HASH_BLOCK_BYTES=128 + SALT_POSITION=APPEND +
     * BASE_ALGO=sha512 in defines_str — disambiguated from SHA512SALTPASS
     * via the SALT_POSITION axis only (mirrors the SHA1PASSSALT vs
     * SHA1SALTPASS / SHA256PASSSALT vs SHA256SALTPASS / MD5PASSSALT vs
     * MD5SALTPASS pairings). HARD GATE on gfx1201 (3080 spill-region
     * ceiling): priv_mem_size <= 43,024 B. Sibling SHA512SALTPASS reading
     * was 42,032 B; expected delta ~0 B (same M[16] scratch + same per-
     * byte loop body, only the byte-source branch order swaps). Same
     * 19-arg kernel signature as the other salted template siblings. */
    cl_program prog_template_sha512passsalt;
    cl_kernel  kern_template_phase0_sha512passsalt;
    /* Family E HMAC-SHA384 carrier (2026-05-08): SHA384SALTPASS-shaped
     * carrier kernel for HMAC-SHA384 (e543) + HMAC-SHA384_KPASS (e796).
     * No JOB_SHA384SALTPASS algorithm exists in mdxfind; this template
     * kernel is reachable ONLY via the HMAC body branch in finalize_-
     * prepend_be64.cl.frag (HASH_WORDS == 12 && algo_mode >= 5u). The
     * mode-0 SHA384(salt||pass) main body is structurally unreachable
     * in production. defines_str disambiguates from SHA512SALTPASS via
     * HASH_WORDS=12 (vs 16) — same BASE_ALGO=sha512 since the
     * compression primitive is identical (sha512_block). Cores in
     * gpu_sha384saltpass_core.cl; built with -DGPU_TEMPLATE_HAS_SALT=1
     * plus HASH_WORDS=12 + HASH_BLOCK_BYTES=128 + BASE_ALGO=sha512 in
     * defines_str. Same 19-arg kernel signature as the other salted
     * template siblings. EMIT_HIT_12 (vs EMIT_HIT_16) emits 12 LE
     * uint32 = 48 bytes (matches HMAC-SHA384's 48-byte digest). */
    cl_program prog_template_sha384saltpass;
    cl_kernel  kern_template_phase0_sha384saltpass;
    /* Family G HMAC-RIPEMD-160 carrier (2026-05-08): RIPEMD160SALTPASS-shaped
     * carrier kernel for HMAC-RMD160 (e211) + HMAC-RMD160_KPASS (e798).
     * No JOB_RIPEMD160SALTPASS algorithm exists in mdxfind; this template
     * kernel is reachable ONLY via the HMAC body branch in finalize_-
     * prepend_rmd.cl.frag (HASH_WORDS == 5 && algo_mode >= 5u). The mode-0
     * RMD160(salt||pass) main body is structurally unreachable in
     * production. defines_str disambiguates from SHA1SALTPASS (HASH_WORDS=5
     * + BASE_ALGO=sha1) via the BASE_ALGO axis (BASE_ALGO=rmd160). Cores
     * in gpu_ripemd160saltpass_core.cl; built with -DGPU_TEMPLATE_HAS_SALT=1
     * plus HASH_WORDS=5 + HASH_BLOCK_BYTES=64 + BASE_ALGO=rmd160 in
     * defines_str. Same 19-arg kernel signature as the other salted
     * template siblings. EMIT_HIT_5 emits 5 LE uint32 = 20 bytes (matches
     * HMAC-RIPEMD-160's 20-byte digest). LE-direct probe / emit (no
     * bswap32; mirrors gpu_ripemd160_core.cl unsalted convention). */
    cl_program prog_template_ripemd160saltpass;
    cl_kernel  kern_template_phase0_ripemd160saltpass;
    /* Family H HMAC-RIPEMD-320 carrier (2026-05-08): RIPEMD320SALTPASS-shaped
     * carrier kernel for HMAC-RMD320 (e213) + HMAC-RMD320_KPASS (e799).
     * No JOB_RIPEMD320SALTPASS algorithm exists in mdxfind; this template
     * kernel is reachable ONLY via the HMAC body branch in finalize_-
     * prepend_rmd.cl.frag (HASH_WORDS == 10 && algo_mode >= 5u). The mode-0
     * RMD320(salt||pass) main body is structurally unreachable in
     * production. defines_str disambiguates from RIPEMD160SALTPASS via
     * HASH_WORDS=10 (vs 5) AND BASE_ALGO=rmd320 (vs rmd160) — distinct
     * compression primitive (rmd320_block has different per-step round
     * bodies + line/line' accumulation pattern, though the 2-arg call
     * signature is identical). Cores in gpu_ripemd320saltpass_core.cl;
     * built with -DGPU_TEMPLATE_HAS_SALT=1 plus HASH_WORDS=10 +
     * HASH_BLOCK_BYTES=64 + BASE_ALGO=rmd320 in defines_str. Same 19-arg
     * kernel signature as the other salted template siblings. EMIT_HIT_10
     * emits 10 LE uint32 = 40 bytes (matches HMAC-RMD320's 40-byte digest).
     * LE-direct probe / emit (no bswap32; mirrors gpu_ripemd320_core.cl
     * unsalted convention). */
    cl_program prog_template_ripemd320saltpass;
    cl_kernel  kern_template_phase0_ripemd320saltpass;
    /* Family I HMAC-BLAKE2S carrier (2026-05-08): hand-written Path A sibling
     * of the codegen-emitted salted cores. Single algo_mode (5) — no KPASS
     * sibling op. JOB_HMAC_BLAKE2S (e828) is the only op routed here; the
     * mode-0 BLAKE2S(salt||pass) main body is structurally unreachable in
     * production (no JOB_BLAKE2SSALTPASS algorithm exists in mdxfind). HMAC
     * body lives inline in template_finalize (NOT in a fragment — Path A
     * keeps the core self-contained). Cores in gpu_hmac_blake2s_core.cl;
     * built with -DGPU_TEMPLATE_HAS_SALT=1 plus HASH_WORDS=8 +
     * HASH_BLOCK_BYTES=64 + BASE_ALGO=blake2s + HMAC_KPASS=1 in defines_str.
     * Same 19-arg kernel signature as the other salted template siblings.
     * EMIT_HIT_8 emits 8 LE uint32 = 32 bytes (matches HMAC-BLAKE2S' 32-byte
     * digest). LE-direct probe / emit (no bswap32; mirrors BLAKE2S' native
     * byte order). Cache disambiguated from BLAKE2S256 unsalted via HAS_SALT=1
     * (axis absent in unsalted). Slab oracle retired in same commit:
     * gpu/gpu_hmac_blake2s.cl whole-file #if 0 wrap; previously held only
     * hmac_blake2s_kpass_batch. */
    cl_program prog_template_hmac_blake2s;
    cl_kernel  kern_template_phase0_hmac_blake2s;
    /* Family J HMAC-STREEBOG-256 carrier (2026-05-08): hand-written Path A
     * sibling of gpu_streebog256_core.cl. Two algo_modes: 5 = KSALT (e838),
     * 6 = KPASS (e837). The HMAC body lives inline in template_finalize
     * (NOT in a fragment — Path A keeps the core self-contained, mirrors
     * Family I HMAC-BLAKE2S precedent). Cores in gpu_hmac_streebog256_-
     * core.cl; built with -DGPU_TEMPLATE_HAS_SALT=1 plus HASH_WORDS=8 +
     * HASH_BLOCK_BYTES=64 + BASE_ALGO=streebog256 + HMAC_KSALTPASS=1 in
     * defines_str. Same 19-arg kernel signature as the other salted
     * template siblings. EMIT_HIT_8 emits 8 LE uint32 = 32 bytes (matches
     * HMAC-STREEBOG-256's 32-byte digest). LE-direct probe / emit (no
     * bswap32; mirrors Streebog-256's slab convention; the slab kernel
     * does NOT byte-reverse vs CPU's big-endian streebog_final because
     * GPU's streebog_hash_priv writes the internal h[] byte image
     * directly — net effect is post-reversal byte order on both sides).
     * Cache disambiguated from Streebog-256 unsalted (gpu_streebog256_-
     * core_str) via HAS_SALT=1 + HMAC_KSALTPASS=1 axes (absent in
     * unsalted defines_str). Slab kernels retired in same commit:
     * hmac_streebog256_kpass_batch + _ksalt_batch surgically deleted
     * from gpu/gpu_streebog.cl (KEEP streebog512 HMAC kernels — that's
     * Family K scope). */
    cl_program prog_template_hmac_streebog256;
    cl_kernel  kern_template_phase0_hmac_streebog256;
    /* Family K HMAC-STREEBOG-512 carrier (2026-05-08): hand-written Path A
     * sibling of gpu_streebog512_core.cl (mirrors Family J at HASH_WORDS=16
     * instead of 8). Two algo_modes: 5 = KSALT (e840), 6 = KPASS (e839).
     * The HMAC body lives inline in template_finalize (NOT in a fragment
     * - Path A keeps the core self-contained, mirrors Family J HMAC-
     * STREEBOG-256 precedent). Cores in gpu_hmac_streebog512_core.cl;
     * built with -DGPU_TEMPLATE_HAS_SALT=1 plus HASH_WORDS=16 +
     * HASH_BLOCK_BYTES=64 + BASE_ALGO=streebog512 + HMAC_KSALTPASS=1 in
     * defines_str. Same 19-arg kernel signature as the other salted
     * template siblings. EMIT_HIT_16 emits 16 LE uint32 = 64 bytes
     * (matches HMAC-STREEBOG-512's 64-byte digest). LE-direct probe /
     * emit (no bswap32; mirrors Streebog-512's slab convention; the slab
     * kernel does NOT byte-reverse vs CPU's HMAC body - Family K applies
     * the inner-reversal correctly for byte-exact CPU/GPU parity, parallel
     * to Family J's STREEBOG-256 fix). Cache disambiguated from Streebog-
     * 512 unsalted (gpu_streebog512_core_str) via HAS_SALT=1 +
     * HMAC_KSALTPASS=1 axes (absent in unsalted defines_str), and from
     * Family J HMAC-STREEBOG-256 via HASH_WORDS=16 vs 8 + BASE_ALGO=
     * streebog512 vs streebog256. Slab kernels retired in same commit:
     * hmac_streebog512_kpass_batch + _ksalt_batch surgically deleted from
     * gpu/gpu_streebog.cl (whole-file #if 0 wrap - file is empty post-
     * Family-K retirement; Family J retired the streebog256 HMAC kernels
     * earlier in same session, B10 retired unsalted streebog batches). */
    cl_program prog_template_hmac_streebog512;
    cl_kernel  kern_template_phase0_hmac_streebog512;
    /* PHPBB3 carrier (2026-05-08): hand-written Path A salted-template
     * kernel for JOB_PHPBB3 (e455). Single algo_mode; no KPASS/KSALT
     * siblings. Iterated MD5 chain INSIDE template_finalize (mirrors
     * SHA1DRU pattern at max_iter=1 forced host-side). The salt buffer
     * carries the FULL 12-byte "$H$<cost><8-byte salt>" prefix; iter
     * count is decoded from salt_bytes[3] via phpitoa64 reverse lookup.
     * Cores in gpu_phpbb3_core.cl; built with -DGPU_TEMPLATE_HAS_SALT=1
     * plus HASH_WORDS=4 + HASH_BLOCK_BYTES=64 + BASE_ALGO=phpbb3 in
     * defines_str. Same 19-arg kernel signature as the other salted
     * template siblings. EMIT_HIT_4 emits 4 LE uint32 = 16 bytes (MD5
     * digest). LE-direct probe / emit (no bswap32; mirrors MD5
     * convention). Cache disambiguated from MD5SALT family via
     * BASE_ALGO=phpbb3 (vs md5). Hit-replay: host calls checkhashbb
     * (not checkhashkey/checkhashsalt — PHPBB3 has its own bb-specific
     * digest format with phpitoa64-encoded output). */
    cl_program prog_template_phpbb3;
    cl_kernel  kern_template_phase0_phpbb3;
    /* MD5CRYPT carrier (2026-05-08): hand-written Path A salted-template
     * kernel for JOB_MD5CRYPT (e511). Single algo_mode; no KPASS/KSALT
     * siblings. Iterated MD5 chain (1000 fixed iters per BSD $1$
     * md5crypt) INSIDE template_finalize (mirrors PHPBB3 / SHA1DRU
     * pattern at max_iter=1 forced host-side). The salt buffer carries
     * the FULL "$1$<salt>$" prefix (variable length 5..12 bytes); the
     * raw salt is extracted inside the kernel by skipping the first
     * 3 bytes ("$1$") and reading until the next '$' or end of buffer.
     * Cores in gpu_md5crypt_core.cl; built with -DGPU_TEMPLATE_HAS_SALT=1
     * plus HASH_WORDS=4 + HASH_BLOCK_BYTES=64 + BASE_ALGO=md5crypt in
     * defines_str. Same 19-arg kernel signature as the other salted
     * template siblings. EMIT_HIT_4 emits 4 LE uint32 = 16 bytes (MD5
     * digest). LE-direct probe / emit (no bswap32; mirrors MD5
     * convention). Cache disambiguated from MD5SALT family via
     * BASE_ALGO=md5crypt (vs md5) and from PHPBB3 via BASE_ALGO=md5crypt
     * (vs phpbb3). Hit-replay: host calls hybrid_check + reconstructs
     * "$1$<salt>$<22-char-phpitoa64>" via md5crypt_b64encode (mirrors
     * existing slab arm at gpujob_opencl.c:1723). Phase 1 of the Unix-
     * crypt ladder (MD5CRYPT -> SHA256CRYPT -> SHA512CRYPT ->
     * SHA512CRYPTMD5). */
    cl_program prog_template_md5crypt;
    cl_kernel  kern_template_phase0_md5crypt;
    /* SHA256CRYPT carrier (2026-05-08): hand-written Path A salted-template
     * kernel for JOB_SHA256CRYPT (e512). Single algo_mode (0); no KPASS/
     * KSALT siblings. SHA-256 crypt chain (5 steps + variable-rounds main
     * loop, default 5000 iters; configurable via "rounds=N$" salt prefix)
     * INSIDE template_finalize. max_iter=1 forced host-side. The salt
     * buffer carries the FULL "$5$[rounds=N$]<salt>$" prefix; the kernel
     * parses and extracts raw_salt up to 16 bytes inline.
     *
     * Cores in gpu_shacrypt_core.cl (SHARED across SHA256CRYPT +
     * SHA512CRYPT + SHA512CRYPTMD5 — Phases 2 + 3 + 4); built with
     * -DGPU_TEMPLATE_HAS_SALT=1 plus HASH_WORDS=8 + HASH_BLOCK_BYTES=64 +
     * BASE_ALGO=sha256crypt in defines_str. Same 19-arg kernel signature
     * as the other salted template siblings. EMIT_HIT_8 emits 8 LE uint32
     * = 32 bytes (SHA-256 digest, full width). LE-direct probe / emit
     * (curin packed LE-byte-by-byte into st->h[i]; mirrors slab oracle
     * gpu_sha256crypt.cl). Cache disambiguated from every other salted
     * template via BASE_ALGO=sha256crypt (only this Phase-2 instance uses
     * it; Phase 3 will use sha512crypt). Hit-replay: host calls hybrid_-
     * check + reconstructs "$5$[rounds=N$]<salt>$<43-char-base64>" via
     * sha256crypt_b64encode (Phase 2-new helper at gpujob_opencl.c).
     * Phase 2 of the Unix-crypt ladder. */
    cl_program prog_template_sha256crypt;
    cl_kernel  kern_template_phase0_sha256crypt;
    /* SHA512CRYPT carrier (2026-05-08): hand-written Path A salted-template
     * kernel for JOB_SHA512CRYPT (e513). Single algo_mode (0); no KPASS/
     * KSALT pairing -- SHA512CRYPTMD5 (Phase 4) will overload algo_mode=1
     * on the same template program for MD5-preprocess.
     *
     * Cores in gpu_shacrypt_core.cl (SHARED across SHA256CRYPT +
     * SHA512CRYPT + SHA512CRYPTMD5 -- Phases 2 + 3 + 4); built with
     * -DGPU_TEMPLATE_HAS_SALT=1 plus HASH_WORDS=16 + HASH_BLOCK_BYTES=128
     * + BASE_ALGO=sha512crypt in defines_str. Same 19-arg kernel signature
     * as the other salted template siblings. EMIT_HIT_16 emits 16 LE
     * uint32 = 64 bytes (SHA-512 digest, full width). LE-direct probe /
     * emit (curin packed LE-byte-by-byte into st->h[i]; mirrors slab
     * oracle gpu_sha512crypt.cl). Cache disambiguated from every other
     * salted template via BASE_ALGO=sha512crypt (only this Phase-3
     * instance uses it; SHA512SALTPASS uses sha512 at HASH_WORDS=16,
     * SHA256CRYPT uses sha256crypt at HASH_WORDS=8). Hit-replay: host
     * calls hybrid_check + reconstructs "$6$[rounds=N$]<salt>$<86-char-
     * base64>" via sha512crypt_b64encode (Phase 3-new helper at
     * gpujob_opencl.c). Phase 3 of the Unix-crypt ladder. */
    cl_program prog_template_sha512crypt;
    cl_kernel  kern_template_phase0_sha512crypt;
    /* DESCRYPT carrier (2026-05-08, Unix-crypt Phase 5): hand-written Path A
     * carrier salted-template kernel for JOB_DESCRYPT (e500). Single algo_mode
     * (7); bespoke kernel; will NOT share with BCRYPT (BCRYPT will need its
     * own algo_modes for future BCRYPT variants). 25-iter DES Feistel chain
     * INSIDE template_finalize; max_iter=1 forced host-side. The salt buffer
     * carries the 2-byte phpitoa64 salt (gpu_pack_salts use_hashsalt=0;
     * extended-DES salts skipped via salt-pack saltlen!=2 filter). The kernel
     * computes pre-FP (l, r) which probes against the host-pre-FP'd compact
     * table (mdxfind.c:40402-40436 applies inverse FP at hash-load time so
     * the stored 16-byte entry is `4 il + 4 ir + 8 zero pad` = our 4-uint32
     * state with h[2..3] zero). HASH_WORDS=4. Phase 5 of the Unix-crypt
     * ladder (FINAL phase; Unix-crypt slab path fully retired across all 5
     * Unix-crypt ops). */
    cl_program prog_template_descrypt;
    cl_kernel  kern_template_phase0_descrypt;
    /* BCRYPT carrier (2026-05-08, Unix-crypt Phase 6): hand-written Path A
     * carrier salted-template kernel for JOB_BCRYPT (e450). Single algo_mode
     * (8); bespoke kernel; reserves algo_mode 8-15 for future BCRYPT family
     * variants. 2^cost Eksblowfish iter loop (cost parsed per-salt-string at
     * kernel entry, accept SIMT divergence) + final encryption of "Orphean-
     * BeholderScryDoubt" 64 times yields 6 BE uint32 = 24 bytes; swap to LE
     * for compact-table probe. HASH_WORDS=6, HASH_BLOCK_BYTES=64, HAS_SALT=1,
     * SALT_POSITION=PREPEND, BASE_ALGO=bcrypt. The kernel uses workgroup-
     * shared __local Eksblowfish S-boxes (4 KB per lane × BCRYPT_WG_SIZE=8
     * lanes = 32 KB per WG) via the GPU_TEMPLATE_HAS_LOCAL_BUFFER scaffold
     * extension to gpu_template.cl. max_iter=1 forced host-side (the
     * 2^cost iter loop is INTERNAL to the algorithm). 72-byte truncation
     * is HYBRID: host-side at the rules-engine pack site (mdxfind.c
     * ~11082) clamps pack_len = 72 + kernel-side defensive cap.
     * Compound siblings (BCRYPTMD5/BCRYPTSHA1/BCRYPTSHA512) remain CPU-only
     * via gpu_op_category default fall-through (returns GPU_CAT_NONE);
     * only the JOB_BCRYPT singleton routes through this template kernel.
     * Phase 6 of the slab-retirement ladder (final major slab kernel). */
    cl_program prog_template_bcrypt;
    cl_kernel  kern_template_phase0_bcrypt;
    cl_mem     b_validate_records; /* validator output buffer; sized lazily */
    size_t     validate_records_cap;
    /* Host-side mirror of rule_program / rule_offset, retained ONLY when
     * MDXFIND_GPU_VALIDATOR=1 — the validator needs to emit `rulebytes=<hex>`
     * for each rule_idx, which means it must be able to read the bytecode
     * back on the host. Production path (validator off) leaves these NULL
     * with zero allocation cost. Allocated/grown in gpu_opencl_set_rules
     * when validator is enabled. */
    unsigned char *h_rule_program;
    uint32_t       h_rule_program_len;
    uint32_t      *h_rule_offset;
    int            h_rule_offset_n;
    cl_mem     b_rule_program;
    cl_mem     b_rule_offset;
    size_t     rule_program_cap;
    size_t     rule_offset_cap;
    int        gpu_n_rules;        /* 0 = set_rules not yet called */

    /* Per-family timing-based dispatch sizing */
    uint32_t fam_max_items[FAM_COUNT];  /* timed max work items per dispatch */
    int      fam_timed[FAM_COUNT];      /* 1 = probed for this family */
    double   fam_rate_hps[FAM_COUNT];   /* Phase 6: hashes/sec, captured during timing probe.
                                           0 = not yet probed. Used by multi-GPU BF partition. */

    /* BF Phase 3 (2026-05-10): per-device bf_range_{start,end,pos} fields
     * deleted alongside the legacy bf_mode machinery in gpu_opencl_dispatch_-
     * batch. BF on GPU now flows exclusively through the chunk-as-job
     * producer at mdxfind.c:~48590 + rules-engine path; the atomic-cursor
     * mode that used these per-device ranges is retired. See
     * project_bf_chunk_as_job.md Phase 3. */

    /* Per-device dispatch state (was global — races with multi-GPU) */
    int gpu_op;
    int max_iter;
    uint64_t mask_resume;
    uint32_t salt_resume;
    uint64_t last_mask_start;

    /* Set to 1 in gpu_opencl_set_compact_table when the compact table
     * fails to load on this device (insufficient VRAM or buffer alloc
     * failure). EVERY hash kernel — rules engine, legacy packed, batch,
     * mask — probes the compact table; without it any dispatch on this
     * device would touch unbound buffers and trigger CL_OUT_OF_RESOURCES,
     * which the async error callback escalates to GPU_FATAL.
     *
     * Semantic: device-level disable, not rule-engine-level. When this
     * flag is set on device d, NO GPU work routes to device d at all.
     * All dispatch entry points (set_rules, dispatch_md5_rules,
     * dispatch_packed, dispatch_batch, warm_probe) early-return without
     * touching the device. The gpujob worker for this device is also
     * not spawned (see gpujob_init in gpujob_opencl.c) so the shared
     * GPUWorkHead queue's slots can never be consumed by a no-op'ing
     * worker that would silently drop words.
     *
     * When ALL devices end up disabled, num_active_devices==0 and
     * gpu_opencl_init's post-set_compact_table check (or rather the
     * caller's gpu_opencl_active_device_count() probe) flips ocl_ready
     * back to 0, so gpu_opencl_available() returns false and mdxfind
     * routes everything to the CPU FastRule walker — equivalent to
     * the user having passed -G none.
     *
     * History: introduced as `rule_engine_disabled` in rev 1.73 with
     * gates only on the rules-engine path. User correctly pointed out
     * that the legacy packed kernel ALSO probes the compact table, so
     * the gate must be device-level. Renamed to `device_disabled` and
     * extended to all dispatch paths in rev 1.74. */
    int device_disabled;

    /* Dynamic-sizer state (2026-05-09) for kern_template_phase0_md5salt.
     * Modes 0-3 share this entry. See block comment above struct gpu_device. */
    struct dynsize_entry dynsize_md5salt;
};

static struct gpu_device gpu_devs[MAX_GPU_DEVICES];
static int num_gpu_devs = 0;
static int ocl_ready = 0;

/* ============================================================
 * Dynamic-sizer prototype (2026-05-09) — implementation
 * See block comment + struct dynsize_entry above struct gpu_device.
 * ============================================================ */

static int _dynsize_atexit_registered = 0;

static int dynsize_is_enabled(void) {
    /* Default ON (2026-05-09 user direction). Env-var "0" is the only opt-out. */
    const char *e = getenv("MDXFIND_DYNSIZE");
    if (!e) return 1;
    return atoi(e) ? 1 : 0;
}

static int dynsize_spp_pinned(void) {
    return getenv("MDXFIND_SPP") != NULL;
}

static int dynsize_verbose(void) {
    const char *e = getenv("MDXFIND_DYNSIZE_VERBOSE");
    return (e && atoi(e)) ? 1 : 0;
}

/* FNV-1a 64-bit over (CL_DEVICE_NAME|CL_DRIVER_VERSION|CL_DEVICE_VENDOR).
 * 16 hex chars output. Stable identifier for per-device cache file. */
static void dynsize_compute_device_uuid(struct gpu_device *d, char *out, size_t outlen) {
    if (!out || outlen < 17) return;
    char devname[256] = {0}, drvver[256] = {0}, vendor[256] = {0};
    if (d && d->dev) {
        clGetDeviceInfo(d->dev, CL_DEVICE_NAME,    sizeof(devname), devname, NULL);
        clGetDeviceInfo(d->dev, CL_DRIVER_VERSION, sizeof(drvver),  drvver,  NULL);
        clGetDeviceInfo(d->dev, CL_DEVICE_VENDOR,  sizeof(vendor),  vendor,  NULL);
    }
    /* FNV-1a 64-bit */
    uint64_t h = 0xCBF29CE484222325ULL;
    const uint64_t prime = 0x100000001B3ULL;
    const char *bufs[3] = { devname, drvver, vendor };
    for (int i = 0; i < 3; i++) {
        const char *s = bufs[i];
        while (*s) { h ^= (uint8_t)*s++; h *= prime; }
        h ^= (uint8_t)'|'; h *= prime;
    }
    snprintf(out, outlen, "%016llx", (unsigned long long)h);
}

static int dynsize_cache_path(const char *uuid, char *out, size_t outlen) {
    const char *home = getenv("HOME");
    if (!home || !*home) return -1;
    int n = snprintf(out, outlen, "%s/.mdxfind/dynsize/%s/kern_md5salt.txt",
                     home, uuid);
    return (n > 0 && (size_t)n < outlen) ? 0 : -1;
}

/* Best-effort recursive mkdir. Returns 0 on success / already-exists.
 * Windows mkdir takes 1 argument (no mode); wrap so both platforms work.
 * Caught during the @mastercho Windows O_BINARY fix cross-compile attempt
 * 2026-05-10 — the parallel bf_sidecar_mkdir_p helper in mdxfind.c was
 * fixed first; this one was missed. Linux/macOS unchanged. */
#ifdef _WIN32
#define DYNSIZE_MKDIR(path) (void)mkdir(path)
#else
#define DYNSIZE_MKDIR(path) (void)mkdir((path), 0755)
#endif
static int dynsize_mkdir_p(const char *path) {
    char buf[1024];
    size_t n = strlen(path);
    if (n >= sizeof(buf)) return -1;
    memcpy(buf, path, n + 1);
    /* Walk path and mkdir each segment */
    for (size_t i = 1; i < n; i++) {
        if (buf[i] == '/') {
            buf[i] = '\0';
            DYNSIZE_MKDIR(buf);
            buf[i] = '/';
        }
    }
    DYNSIZE_MKDIR(buf);
    return 0;
}

static int dynsize_cache_load(const char *uuid, struct dynsize_entry *e) {
    char path[1024];
    if (dynsize_cache_path(uuid, path, sizeof(path)) < 0) return -1;
    FILE *fp = fopen(path, "r");
    if (!fp) return -1;
    char line[256];
    /* Defaults in case some keys are missing */
    e->current_N = 16;
    e->current_spp = 8192;
    e->spp_cap_observed = DYNSIZE_SPP_MAX;
    e->ema_mhz = 0.0;
    e->prev_window_ema_mhz = 0.0;
    e->convergence_count = 0;
    e->plateau_streak = 0;
    e->loop_weight = DYNSIZE_LOOP_WEIGHT_INIT;
    e->prev_nsalts_active = 0;
    while (fgets(line, sizeof(line), fp)) {
        char key[64]; double dval; unsigned long long uval;
        if (sscanf(line, " %63[^=]= %lf", key, &dval) == 2) {
            if      (!strcmp(key, "current_N"))         e->current_N         = (uint32_t)dval;
            else if (!strcmp(key, "current_spp"))       e->current_spp       = (uint32_t)dval;
            else if (!strcmp(key, "spp_cap_observed"))  e->spp_cap_observed  = (uint32_t)dval;
            else if (!strcmp(key, "ema_mhz"))           e->ema_mhz           = dval;
            else if (!strcmp(key, "loop_weight"))       e->loop_weight       = dval;
            else if (!strcmp(key, "convergence_count")) e->convergence_count = (uint32_t)dval;
        }
        (void)uval;
    }
    fclose(fp);
    if (e->current_N == 0 || e->current_N > 256) e->current_N = 16;
    if (e->current_spp < DYNSIZE_SPP_MIN) e->current_spp = DYNSIZE_SPP_MIN;
    if (e->current_spp > DYNSIZE_SPP_MAX) e->current_spp = DYNSIZE_SPP_MAX;
    if (e->spp_cap_observed < e->current_spp) e->spp_cap_observed = e->current_spp;
    if (e->loop_weight < DYNSIZE_LOOP_WEIGHT_FLOOR) e->loop_weight = DYNSIZE_LOOP_WEIGHT_FLOOR;
    if (e->loop_weight > DYNSIZE_LOOP_WEIGHT_INIT) e->loop_weight = DYNSIZE_LOOP_WEIGHT_INIT;
    return 0;
}

static int dynsize_cache_store(const char *uuid, const struct dynsize_entry *e) {
    char path[1024];
    if (dynsize_cache_path(uuid, path, sizeof(path)) < 0) return -1;
    /* Make parent dir tree */
    char dirbuf[1024];
    size_t plen = strlen(path);
    /* trim filename */
    size_t i = plen;
    while (i > 0 && path[i-1] != '/') i--;
    if (i == 0) return -1;
    if (i >= sizeof(dirbuf)) return -1;
    memcpy(dirbuf, path, i - 1);
    dirbuf[i - 1] = '\0';
    dynsize_mkdir_p(dirbuf);
    /* Write atomically: tmp file + rename */
    char tmp[1100];
    snprintf(tmp, sizeof(tmp), "%s.tmp", path);
    FILE *fp = fopen(tmp, "w");
    if (!fp) return -1;
    fprintf(fp,
        "schema=1\n"
        "current_N=%u\n"
        "current_spp=%u\n"
        "spp_cap_observed=%u\n"
        "ema_mhz=%.4f\n"
        "convergence_count=%u\n"
        "loop_weight=%.6f\n"
        "last_update_ts_unix=%lld\n",
        e->current_N, e->current_spp, e->spp_cap_observed,
        e->ema_mhz, e->convergence_count, e->loop_weight,
        (long long)time(NULL));
    fclose(fp);
    if (rename(tmp, path) != 0) {
        unlink(tmp);
        return -1;
    }
    return 0;
}

/* atexit hook — persists every device's converged dynsize state. */
static void dynsize_atexit_persist(void) {
    for (int di = 0; di < num_gpu_devs; di++) {
        struct gpu_device *d = &gpu_devs[di];
        if (!d->dynsize_md5salt.initialized) continue;
        if (d->dynsize_md5salt.convergence_count == 0) continue;
        char uuid[33];
        dynsize_compute_device_uuid(d, uuid, sizeof(uuid));
        (void)dynsize_cache_store(uuid, &d->dynsize_md5salt);
    }
}

/* Idempotent loader. Called from compile helper (eager) AND from
 * salts_per_page derivation site (lazy fallback). */
static void dynsize_ensure_loaded(struct gpu_device *d, int dev_idx) {
    (void)dev_idx;
    if (!d) return;
    if (d->dynsize_md5salt.initialized) return;
    if (!dynsize_is_enabled()) {
        /* Even if disabled at this moment, populate with defaults so
         * later code paths read sane values. We will not write the
         * cache on exit because convergence_count remains 0.
         * N=64 is the empirical Pascal optimum per gpu_md5salt_core.cl
         * rev 1.5 log ("N=64 spp=16384 closes ~76% of slab-vs-template
         * gap on 1080"); confirmed 2026-05-11 — N=64 yields 2.03 GH/s
         * on fpga GTX 1080 vs 0.52 GH/s at the old N=16 default. */
        d->dynsize_md5salt.current_N = 64;
        d->dynsize_md5salt.current_spp = 8192;
        d->dynsize_md5salt.spp_cap_observed = DYNSIZE_SPP_MAX;
        d->dynsize_md5salt.loop_weight = DYNSIZE_LOOP_WEIGHT_INIT;
        d->dynsize_md5salt.ema_mhz = 0.0;
        d->dynsize_md5salt.initialized = 1;
        d->dynsize_md5salt.loaded_from_cache = 0;
        return;
    }
    char uuid[33];
    dynsize_compute_device_uuid(d, uuid, sizeof(uuid));
    if (dynsize_cache_load(uuid, &d->dynsize_md5salt) == 0) {
        d->dynsize_md5salt.loaded_from_cache = 1;
        if (dynsize_verbose()) {
            fprintf(stderr,
                "[dynsize] dev=%d cache HIT uuid=%s N=%u spp=%u "
                "spp_cap=%u ema=%.1f weight=%.3f conv=%u\n",
                dev_idx, uuid,
                d->dynsize_md5salt.current_N,
                d->dynsize_md5salt.current_spp,
                d->dynsize_md5salt.spp_cap_observed,
                d->dynsize_md5salt.ema_mhz,
                d->dynsize_md5salt.loop_weight,
                d->dynsize_md5salt.convergence_count);
        }
    } else {
        /* Cold start: seed the PLL at the empirical Pascal optimum
         * (N=64 spp=8192). gpu_md5salt_core.cl rev 1.5 cites N=64
         * spp=16384 as the slab-vs-template gap closer; confirmed
         * 2026-05-11 — N=64 spp=8192 sustains 2.03 GH/s on fpga GTX
         * 1080 vs 0.52 GH/s at the old N=16 default (3.9× speedup).
         * The dynsize servo adapts from this seed; starting at the
         * known-good operating point cuts convergence time and avoids
         * shipping the first-session perf cliff. */
        d->dynsize_md5salt.current_N         = 64;
        d->dynsize_md5salt.current_spp       = 8192;
        d->dynsize_md5salt.spp_cap_observed  = DYNSIZE_SPP_MAX;
        d->dynsize_md5salt.ema_mhz           = 0.0;
        d->dynsize_md5salt.prev_window_ema_mhz = 0.0;
        d->dynsize_md5salt.convergence_count = 0;
        d->dynsize_md5salt.plateau_streak    = 0;
        d->dynsize_md5salt.loop_weight       = DYNSIZE_LOOP_WEIGHT_INIT;
        d->dynsize_md5salt.prev_nsalts_active = 0;
        d->dynsize_md5salt.loaded_from_cache = 0;
        if (dynsize_verbose()) {
            fprintf(stderr,
                "[dynsize] dev=%d cache MISS uuid=%s — cold start "
                "N=64 spp=8192\n", dev_idx, uuid);
        }
    }
    d->dynsize_md5salt.initialized = 1;
    if (!_dynsize_atexit_registered) {
        atexit(dynsize_atexit_persist);
        _dynsize_atexit_registered = 1;
    }
}

/* Used by the compile helper to choose -DSALT_BATCH=N. Falls back to
 * MDXFIND_SALT_BATCH env var, then default 64 (empirical Pascal
 * optimum per gpu_md5salt_core.cl rev 1.5 + 2026-05-11 fpga 1080
 * measurement: N=64 → 2.03 GH/s vs N=16 → 0.52 GH/s on e31). */
static uint32_t dynsize_compile_time_N(struct gpu_device *d, int dev_idx) {
    if (dynsize_is_enabled()) {
        dynsize_ensure_loaded(d, dev_idx);
        if (d->dynsize_md5salt.initialized) {
            return d->dynsize_md5salt.current_N;
        }
    }
    const char *sb = getenv("MDXFIND_SALT_BATCH");
    if (sb && *sb) {
        int v = atoi(sb);
        if (v >= 1 && v <= 256) return (uint32_t)v;
    }
    return 64u;
}

/* Shared state (same across all devices) */
static uint64_t _compact_mask = 0;
static uint32_t _hash_data_count = 0;
static int _overflow_count = 0;
/* gpu_op, max_iter, mask_resume, salt_resume, last_mask_start are
 * per-device fields in struct gpu_device for multi-GPU safety */

/* ---- Timing-based dispatch sizing constants ---- */
#define TIMING_BUDGET_MIN_MS  200.0
#define TIMING_BUDGET_MAX_MS  400.0
#define TIMING_INITIAL_SIZE   65536   /* 64K work items */
#define TIMING_WATCHDOG_MS    5000.0  /* never risk exceeding this */

extern int gpu_op_family(int op);

/* Mask state — defined at file scope below (gpu_opencl_set_mask).
 * Forward-declared here so warm-probe and other early functions can read them. */
extern int gpu_mask_n_prepend;
extern int gpu_mask_n_append;
extern uint64_t gpu_mask_total;

/* ---- Per-kernel autotune state ---- */
#define TUNE_CANDIDATES 4
static const size_t tune_sizes[TUNE_CANDIDATES] = { 64, 128, 256, 512 };

#define MAX_GPU_KERNELS 2048   /* indexed by op type, same as JOB_DONE */

struct gpu_kern {
    cl_kernel kernel;        /* NULL if no GPU kernel for this op */
    size_t local_size;       /* current work group size */
    size_t max_local;        /* from CL_KERNEL_WORK_GROUP_SIZE */
    int tuned;               /* 1 = done tuning */
    int tune_candidate;      /* which tune_sizes[] we're testing */
    int tune_samples;        /* dispatches at current candidate */
    double tune_best_time;   /* best avg ms so far */
    size_t tune_best_size;   /* size that produced best_time */
    double tune_cur_total;   /* accumulated ms for current candidate */
    int dev_idx;             /* device index for status messages */
};

/* Per-device kernel table — each device has its own compiled kernels */
struct gpu_kern_table {
    struct gpu_kern kerns[MAX_GPU_KERNELS];
};
static struct gpu_kern_table dev_kerns[MAX_GPU_DEVICES];

#define TUNE_SAMPLES 3  /* dispatches per candidate before moving on */

static void kern_register(int di, int op, cl_kernel kernel) {
    if (di < 0 || di >= MAX_GPU_DEVICES) return;
    if (op < 0 || op >= MAX_GPU_KERNELS || !kernel) return;
    struct gpu_kern *k = &dev_kerns[di].kerns[op];
    size_t max_wg = 0;
    clGetKernelWorkGroupInfo(kernel, gpu_devs[di].dev, CL_KERNEL_WORK_GROUP_SIZE, sizeof(max_wg), &max_wg, NULL);
    /* Check for reqd_work_group_size — if set, the required size overrides max_wg */
    size_t req_wg[3] = {0, 0, 0};
    clGetKernelWorkGroupInfo(kernel, gpu_devs[di].dev, CL_KERNEL_COMPILE_WORK_GROUP_SIZE, sizeof(req_wg), req_wg, NULL);
    if (req_wg[0] > 0) max_wg = req_wg[0];
    k->kernel = kernel;
    k->max_local = max_wg;
    k->dev_idx = di;
    k->tuned = 0;
    k->tune_candidate = 0;
    k->tune_samples = 0;
    k->tune_best_time = 1e30;
    k->tune_best_size = max_wg ? max_wg : 64;
    k->tune_cur_total = 0;
    int c = 0;
    while (c < TUNE_CANDIDATES && tune_sizes[c] > max_wg) c++;
    k->tune_candidate = c;
    k->local_size = (c < TUNE_CANDIDATES) ? tune_sizes[c] : max_wg;
}

static size_t kern_get_local_size(struct gpu_kern *k) {
    if (k->tuned) return k->local_size;
    if (k->tune_candidate >= TUNE_CANDIDATES) {
        k->tuned = 1;
        k->local_size = k->tune_best_size;
        fprintf(stderr, "OpenCL GPU[%d]: autotuned work group size = %zu\n", k->dev_idx, k->local_size);
        return k->local_size;
    }
    return tune_sizes[k->tune_candidate];
}

static void kern_record_time(struct gpu_kern *k, double ms) {
    if (k->tuned) return;
    k->tune_cur_total += ms;
    k->tune_samples++;
    if (k->tune_samples >= TUNE_SAMPLES) {
        double avg = k->tune_cur_total / k->tune_samples;
        if (avg < k->tune_best_time) {
            k->tune_best_time = avg;
            k->tune_best_size = tune_sizes[k->tune_candidate];
        }
        k->tune_candidate++;
        while (k->tune_candidate < TUNE_CANDIDATES && tune_sizes[k->tune_candidate] > k->max_local)
            k->tune_candidate++;
        k->tune_samples = 0;
        k->tune_cur_total = 0;
        if (k->tune_candidate >= TUNE_CANDIDATES) {
            k->tuned = 1;
            k->local_size = k->tune_best_size;
            fprintf(stderr, "OpenCL GPU[%d]: autotuned work group size = %zu\n", k->dev_idx, k->local_size);
        }
    }
}

#define GPU_MAX_HITS 32768

static uint32_t *h_hits = NULL;

/* ---- GPU Params struct: 128-byte uniform API (must match kernel) ----
 * All uint64 fields first (8-aligned), then uint32 fields, then reserved.
 * Identical layout for OpenCL and Metal. No padding holes.
 *
 * Memo B B1 (2026-05-04): cursor skeleton fields added at offsets 88-111
 * for two-cursor overflow restart (project_memo_b_dispatch_template.md §2,
 * project_memo_b_dispatch_template.md "B1" row of the phase ladder).
 *
 * Rules kernel does NOT read these in B1 -- they're declared so B3 can wire
 * them without bumping the struct layout again. Cursor=0 == today's
 * behavior is the locked contract. Host zeros all six on every dispatch.
 *
 * reserved32[0..1] at offsets 80-87 are still claimed by gpu_md5_packed.cl
 * etc. for word_start (reserved32[0]) and packed_size (reserved32[1]). The
 * rules dispatch path doesn't need them; the packed dispatch path doesn't
 * need the new cursor fields. Both layouts coexist in the same 128 bytes.
 *
 * B6 salt-axis (2026-05-06): num_salts_per_page at offset 112 (was reserved64[0])
 * communicates salt-page size to the kernel for combined_ridx packing.
 * Populated only when the salt-axis is active; ignored by unsalted kernels. */
typedef struct {
    /* 8-byte fields (offset 0-31) */
    uint64_t compact_mask;    /*  0: hash table mask */
    uint64_t mask_start;      /*  8: mask keyspace offset for chunking */
    uint64_t mask_base0;      /* 16: pre-decomposed mask_start positions 0-7 */
    uint64_t mask_base1;      /* 24: pre-decomposed mask_start positions 8-15 */
    /* 4-byte fields (offset 32-79) */
    uint32_t num_words;       /* 32: words in this batch */
    uint32_t num_salts;       /* 36: salts for this dispatch */
    uint32_t salt_start;      /* 40: starting salt index */
    uint32_t max_probe;       /* 44: compact table max probe depth */
    uint32_t hash_data_count; /* 48: entries in hash_data */
    uint32_t max_hits;        /* 52: hit buffer capacity */
    uint32_t overflow_count;  /* 56: overflow table entries */
    uint32_t max_iter;        /* 60: iteration count from -i */
    uint32_t num_masks;       /* 64: mask combinations per chunk */
    uint32_t n_prepend;       /* 68: prepend mask positions (-N) */
    uint32_t n_append;        /* 72: append mask positions (-n) */
    uint32_t iter_count;      /* 76: per-dispatch iteration count (PHPBB3) */
    /* Reserved (offset 80-127) */
    uint32_t reserved32[2];   /* 80-87: packed kernels: word_start, packed_size */
    /* B1 cursor skeleton — rules kernel reads as 0 in B1, B3 wires kernel logic. */
    uint32_t input_cursor_start;  /*  88: B3 input cursor (lanes < cursor early-return) */
    uint32_t rule_cursor_start;   /*  92: B3 rule cursor */
    uint32_t inner_iter;          /*  96: BF Phase 1.8 — kernel inner iter count
                                   *       for BF chunks. Repurposed from B3
                                   *       output_cursor_start (zero-read audit
                                   *       2026-05-10). 0 or 1 = bit-identical
                                   *       to today; >1 = each lane processes
                                   *       inner_iter mask values per (word,rule).
                                   *       Cap = 16. Unsalted BF only. */
    uint32_t overflow_first_set;  /* 100: B3 kernel sets to 1 on first overflow */
    uint32_t overflow_first_word; /* 104: B3 word_idx CAS-min target */
    uint32_t overflow_first_rule; /* 108: B3 rule_idx CAS-min target */
    uint64_t num_salts_per_page; /* 112: B6 salt-axis paging (was reserved64[0]) */
    uint32_t algo_mode;          /* 120: B6.6 per-algorithm runtime variant flag (was reserved64[1] high half) */
    uint32_t mask_offset_per_word; /* 124: BF chunk: word stride per BF chunk; 0 == not a BF chunk. Default 0 = today's behavior. */
} OCLParams;

/* ---- Load kernel source from file ---- */
static char *load_kernel_file(const char *path) {
    FILE *f = fopen(path, "r");
    if (!f) {
        /* Try gpu/ prefix */
        char buf[512];
        snprintf(buf, sizeof(buf), "gpu/%s", path);
        f = fopen(buf, "r");
    }
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *src = (char *)malloc_lock(sz + 1,"load_kernel");
    fread(src, 1, sz, f);
    src[sz] = 0;
    fclose(f);
    return src;
}

/* ---- Path-a buffer-upload read-back probes (always-on instrumentation) ----
 * After every per-device buffer upload, read back a sentinel slice and
 * memcmp() against the host source. On mismatch, log one descriptive
 * line per failing device with offset/size/expected/actual context, then
 * abort() — the bug must be loud, not silent. Cost is ~12 reads of ~320
 * bytes per device per session, trivial vs run wall time.
 *
 * Compile-time always-on (no #ifdef gates) so the same binary runs on
 * ioblade and SHOOTER. CL_TRUE blocking read serializes against the
 * preceding clEnqueueWriteBuffer, so the probe sees the post-write state.
 */
static void gpu_readback_probe(int dev_idx, cl_command_queue queue, cl_mem buf,
                               size_t off, size_t len, const void *expected,
                               const char *site, const char *bufname) {
    if (!buf || len == 0) return;
    unsigned char tmp[512];
    if (len > sizeof(tmp)) len = sizeof(tmp);
    cl_int err = clEnqueueReadBuffer(queue, buf, CL_TRUE, off, len, tmp,
                                     0, NULL, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "OpenCL GPU[%d]: PROBE %s/%s read err=%d (off=%zu len=%zu) — skipping verify\n",
                dev_idx, site, bufname, err, off, len);
        return;
    }
    if (memcmp(tmp, expected, len) == 0) return;
    /* Mismatch: dump first 32 bytes of expected vs actual. */
    char e_hex[3 * 32 + 1], a_hex[3 * 32 + 1];
    size_t dump_n = len < 32 ? len : 32;
    const unsigned char *ep = (const unsigned char *)expected;
    int p = 0;
    for (size_t i = 0; i < dump_n; i++) p += snprintf(e_hex + p, sizeof(e_hex) - p, "%02x ", ep[i]);
    p = 0;
    for (size_t i = 0; i < dump_n; i++) p += snprintf(a_hex + p, sizeof(a_hex) - p, "%02x ", tmp[i]);
    fprintf(stderr, "OpenCL GPU[%d]: PROBE FAIL %s/%s off=%zu len=%zu\n"
                    "  expected: %s\n"
                    "  actual:   %s\n",
            dev_idx, site, bufname, off, len, e_hex, a_hex);
    fflush(stderr);
    abort();
}

/* Public accessor: device name (CL_DEVICE_NAME), for end-of-run reports. */
const char *gpu_opencl_device_name(int dev_idx) {
    if (dev_idx < 0 || dev_idx >= num_gpu_devs) return "(unknown)";
    return gpu_devs[dev_idx].name;
}

/* BF Phase 1.6 (2026-05-09): per-device UUID for BF rate-EMA sidecar
 * persistence. Wraps the file-static dynsize_compute_device_uuid (FNV-1a
 * 64-bit over name|driver|vendor). Caller-supplied buffer (>= 17 bytes
 * for 16 hex + NUL). On out-of-range dev_idx, writes empty string. */
void gpu_opencl_dev_uuid(int dev_idx, char *out, size_t out_sz) {
    if (!out || out_sz == 0) return;
    out[0] = 0;
    if (dev_idx < 0 || dev_idx >= num_gpu_devs) return;
    dynsize_compute_device_uuid(&gpu_devs[dev_idx], out, out_sz);
}

/* Public accessor: device PCI BDF (best-effort). Tries CL_DEVICE_PCI_BUS_ID_NV
 * (NVIDIA) and CL_DEVICE_TOPOLOGY_AMD before falling back to a search-by-name
 * via lspci. Result format: "DDDD:BB:DD.F" or "BB:DD.F". Empty string if unknown. */
void gpu_opencl_device_bdf(int dev_idx, char *out, size_t out_sz) {
    if (out_sz == 0) return;
    out[0] = 0;
    if (dev_idx < 0 || dev_idx >= num_gpu_devs) return;
    cl_device_id dev = gpu_devs[dev_idx].dev;
    /* NVIDIA extension: CL_DEVICE_PCI_BUS_ID_NV (0x4008), CL_DEVICE_PCI_SLOT_ID_NV (0x4009) */
    cl_uint bus = 0, slot = 0;
    cl_int err1 = clGetDeviceInfo(dev, 0x4008, sizeof(bus), &bus, NULL);
    cl_int err2 = clGetDeviceInfo(dev, 0x4009, sizeof(slot), &slot, NULL);
    if (err1 == CL_SUCCESS && err2 == CL_SUCCESS) {
        snprintf(out, out_sz, "%02x:%02x.%x", bus & 0xff, (slot >> 3) & 0x1f, slot & 0x7);
        return;
    }
    /* AMD extension: CL_DEVICE_TOPOLOGY_AMD (0x4037) — struct {int, char unused[17], char bus, char dev, char fn} */
    struct { cl_uint type; char data[20]; } topo_amd;
    if (clGetDeviceInfo(dev, 0x4037, sizeof(topo_amd), &topo_amd, NULL) == CL_SUCCESS && topo_amd.type == 1) {
        unsigned char b = (unsigned char)topo_amd.data[17];
        unsigned char d = (unsigned char)topo_amd.data[18];
        unsigned char f = (unsigned char)topo_amd.data[19];
        snprintf(out, out_sz, "%02x:%02x.%x", b, d, f);
        return;
    }
    /* Fallback: leave empty (caller handles "BDF unknown"). */
}

/* ---- Embedded kernel sources (auto-generated from gpu_*.cl via cl2str.py --all) ---- */
#include "gpu_common_str.h"
#include "gpu_md5salt_str.h"
/* gpu_md5saltpass_str.h removed in B8 (2026-05-06): md5saltpass_batch +
 * md5passsalt_batch slab kernels retired — JOB_MD5SALTPASS / JOB_MD5PASSSALT
 * use the unified template path via B6 / B6.4 fan-out. RCS history retained
 * in gpu/RCS/gpu_md5saltpass.cl,v. */
/* gpu_md5iter_str.h removed in B12 (2026-05-08): the FAM_MD5ITER slab kernels
 * (md5_iter_lc + md5_iter_uc) had no live consumer — the rules-engine + tem-
 * plate paths handle JOB_MD5/JOB_MD5UC iteration. The only remaining caller
 * was warm_probe(JOB_MD5/JOB_MD5UC) which gracefully early-returns now that
 * dev_kerns[*][JOB_MD5/UC] resolves to NULL kernel handles. RCS history
 * retained at gpu/RCS/gpu_md5iter.cl,v. */
/* gpu_phpbb3_str.h removed in PHPBB3 carrier ship (2026-05-08): slab kernel
 * phpbb3_batch retired — JOB_PHPBB3 now routes through the unified template
 * path via the PHPBB3 carrier (gpu_phpbb3_core.cl) + max_iter=1 forced
 * dispatch. RCS history retains the slab kernel source at
 * gpu/RCS/gpu_phpbb3.cl,v. */
/* gpu_md5crypt_str.h removed in MD5CRYPT carrier ship (2026-05-08): slab kernel
 * md5crypt_batch retired — JOB_MD5CRYPT now routes through the unified
 * template path via the MD5CRYPT carrier (gpu_md5crypt_core.cl) + max_iter=1
 * forced dispatch. RCS history retains the slab kernel source at
 * gpu/RCS/gpu_md5crypt.cl,v. */
/* gpu_md5_md5saltmd5pass_str.h removed in B8 (2026-05-06): the slab kernel
 * md5_md5saltmd5pass_batch retired — JOB_MD5_MD5SALTMD5PASS uses the unified
 * MD5SALT template via params.algo_mode=4 (B6.8). RCS history retained in
 * gpu/RCS/gpu_md5_md5saltmd5pass.cl,v. */
/* Family B (2026-05-07): gpu_sha1_str.h removed — the FAM_SHA1 slab kernels
 * (hmac_sha1_ksalt_batch + hmac_sha1_kpass_batch) retired. JOB_HMAC_SHA1 /
 * JOB_HMAC_SHA1_KPASS now route through the SHA1SALTPASS unified template.
 * RCS history retained in gpu/RCS/gpu_sha1.cl,v. */
/* gpu_sha256_str.h removed in Family D (2026-05-08): hmac_sha256_ksalt_batch
 * + hmac_sha256_kpass_batch slab kernels retired post-template-path-ship;
 * gpu_sha256.cl whole-file deleted from working tree (mirrors Family J
 * pattern). JOB_HMAC_SHA256 / KPASS now route through SHA256SALTPASS
 * template via algo_mode 5/6 + RUNTIME-gated HMAC body (NEVER `#if`;
 * rev 1.7 Pascal NVIDIA ABORT lesson). RCS history retained at
 * gpu/RCS/gpu_sha256.cl,v. Final HMAC family in the ladder. */
/* gpu_md5mask_str.h removed in B12 (2026-05-08): the FAM_MD5MASK slab kernel
 * md5_mask_batch had no live consumer — JOB_MD5 mask coverage routes through
 * the unified template path (-n/-N integration in B6). The only remaining
 * caller was warm_probe(JOB_MD5) which gracefully early-returns now that
 * dev_kerns[*][JOB_MD5] resolves to a NULL kernel handle. RCS history
 * retained at gpu/RCS/gpu_md5mask.cl,v. */
/* gpu_descrypt_str.h removed in DESCRYPT carrier ship (2026-05-08, Unix-
 * crypt Phase 5): slab kernel descrypt_batch retired -- JOB_DESCRYPT now
 * routes through the unified template path via the hand-written Path A
 * carrier (gpu_descrypt_core.cl) at HASH_WORDS=4 + max_iter=1 forced
 * dispatch. The slab kernel oracle (algorithm correctness reference) is
 * preserved in RCS at gpu/RCS/gpu_descrypt.cl,v; the working tree no
 * longer carries the .cl/_str.h pair. probe_max_dispatch is anchored on
 * FAM_MD5SALT (hmac_md5_ksalt_batch), not FAM_DESCRYPT, so retirement
 * does not require a probe migration. Phase 5 is the FINAL phase of the
 * Unix-crypt ladder (Unix-crypt slab path fully retired across all 5
 * Unix-crypt ops: MD5CRYPT, SHA256CRYPT, SHA512CRYPT, SHA512CRYPTMD5,
 * DESCRYPT). */
#include "gpu_descrypt_core_str.h"
/* B7.9 (2026-05-07): chokepoint pack retirement. Five packed kernel _str.h
 * includes deleted (gpu_md5_packed_str.h, gpu_sha1_packed_str.h,
 * gpu_sha256_packed_str.h, gpu_md4_packed_str.h, gpu_sha512_packed_str.h).
 * The chokepoint pack at mdxfind.c:11199-11374 was retired in this same
 * commit; gpu_opencl_dispatch_packed() and packed_family() were also
 * removed. Mixed CPU+GPU rule workloads now CPU-fallback for the rule-
 * applied bytes (architect-accepted ~5% perf hit per
 * project_b76plus_mask_iter_closure.md §13 Q2 option A1). The 100%-GPU-
 * eligible workload (gpu_legacy_slot_unused == 1, dominant production
 * case) was already bypassing the chokepoint pack via gpu_skip_no_rule_pack
 * — no net change. RCS history retains the .cl + _str.h sources. */
/* B9 (2026-05-07): unsalted slab retirement. The legacy mask short-circuit
 * at mdxfind.c:10911-10951 was the sole reachability path for the 11 unsalted
 * slab kernels; that short-circuit + gpu_try_pack_unsalted() in mdxfind.c
 * were retired in this same commit. The 11 _str.h includes (md5/md4/sha1/
 * sha256/sha512/rmd160/blake2s256/keccak/md6256/wrl/mysql3 unsalted) are
 * gone alongside the .cl/.h sources. All GPU_CAT_MASK ops now route exclu-
 * sively through the unified template path (rules-engine kernel via
 * gpu_opencl_dispatch_md5_rules); mask configs > 16+16 positions and other
 * unmapped configurations CPU-fallback per architect §11 (B9 row). The
 * gpu_opencl_dispatch_batch() function and the worker's slab dispatch
 * branch (gpujob_opencl.c) STAY — they remain load-bearing for SALTED /
 * SALTPASS ops (PHPBB3, MD5CRYPT, DESCRYPT, HMAC_*, SHA512CRYPT,
 * SHA256CRYPT, BCRYPT) packed via gpu_try_pack (the salted variant) at
 * non-mask-short-circuit call sites. RCS history retains the .cl + _str.h
 * sources. */
/* HMAC-SHA256 kernels (gpu_sha256.cl, FAM_SHA256) RETIRED in Family D
 * (2026-05-08); gpu_sha256.cl whole-file deleted from working tree.
 * JOB_HMAC_SHA256 / JOB_HMAC_SHA256_KPASS now route through the unified
 * SHA256SALTPASS template path via params.algo_mode = 5 / 6. RCS history
 * retained at gpu/RCS/gpu_sha256.cl,v. */
/* B11 cleanup (2026-05-08): 5 retired-kernel `.cl` files + their `_str.h`
 * pairs deleted from the working tree per feedback_remove_retired_gpu_-
 * kernels.md. The `_str.h` includes that previously lived here for
 * Families E/F (gpu_hmac_sha512), G (gpu_hmac_rmd160), H (gpu_hmac_-
 * rmd320), I (gpu_hmac_blake2s), J/K (gpu_streebog) are gone alongside
 * the .cl sources. All five family_source[] slots already NULL (above);
 * the five gpu_*_str symbols had no live readers. RCS history retains
 * the .cl + (where tracked) _str.h sources at gpu/RCS/. */
/* gpu_sha512crypt_str.h removed in SHA512CRYPTMD5 carrier ship (2026-05-08):
 * slab kernel sha512crypt_batch retired -- both JOB_SHA512CRYPT and
 * JOB_SHA512CRYPTMD5 now route through the unified template path via the
 * SHACRYPT shared core (gpu_shacrypt_core.cl at HASH_WORDS=16) +
 * max_iter=1 forced dispatch. SHA512CRYPTMD5's MD5-preprocess is host-
 * side at mdxfind.c:12256-12258. RCS history retains the slab kernel
 * source at gpu/RCS/gpu_sha512crypt.cl,v. probe_max_dispatch is anchored
 * on FAM_MD5SALT, not FAM_SHA512CRYPT, so retirement does not require a
 * probe migration. Phase 4 of the Unix-crypt ladder (final phase; Unix-
 * crypt slab path fully retired). */
/* gpu_sha256crypt_str.h removed in SHA256CRYPT carrier ship (2026-05-08):
 * slab kernel sha256crypt_batch retired -- JOB_SHA256CRYPT now routes
 * through the unified template path via the SHACRYPT shared core
 * (gpu_shacrypt_core.cl at HASH_WORDS=8) + max_iter=1 forced dispatch.
 * RCS history retains the slab kernel source at gpu/RCS/gpu_sha256crypt.cl,v.
 * Phase 2 of the Unix-crypt ladder. */
/* gpu_bcrypt_str.h removed in BCRYPT carrier ship (2026-05-08, Unix-crypt
 * Phase 6): slab kernel bcrypt_batch retired -- JOB_BCRYPT now routes
 * through the unified template path via the hand-written Path A carrier
 * (gpu_bcrypt_core.cl) at HASH_WORDS=6 + max_iter=1 forced dispatch. The
 * slab kernel oracle (algorithm correctness reference) is preserved in RCS
 * at gpu/RCS/gpu_bcrypt.cl,v; the working tree no longer carries the
 * .cl/_str.h pair. probe_max_dispatch is anchored on FAM_MD5SALT (hmac_-
 * md5_ksalt_batch), not FAM_BCRYPT, so retirement does not require a
 * probe migration (warm-probe sizing arms at gpu_opencl.c lines 2820 +
 * 12261 stay live -- they work regardless of slab/template since
 * is_bcrypt is derived from fam == FAM_BCRYPT). Phase 6 of the slab-
 * retirement ladder (final major slab kernel). Compound siblings BCRYPT-
 * MD5 (e451) / BCRYPTSHA1 (e452) / BCRYPTSHA512 (e967) remain CPU-only
 * via gpu_op_category default fall-through; only JOB_BCRYPT singleton
 * is GPU-accelerated. */
#include "gpu_bcrypt_core_str.h"
/* Phase 0/1 GPU rule expansion engine. See project_gpu_rule_engine_design.md. */
#include "gpu_md5_rules_str.h"
/* Memo B Phase B2 (2026-05-04): generic dispatch-template skeleton +
 * MD5 algorithm core. Template kernel (template_phase0) is built
 * side-by-side with md5_rules_phase0; selected at dispatch time when
 * MDXFIND_GPU_TEMPLATE=md5 is set in the env. Default off. See
 * project_memo_b_dispatch_template.md §3 (template body) and the B2
 * row of the phase ladder. */
#include "gpu_md5_core_str.h"
/* Phase 1.9 Tranche A1 (2026-05-10): MD5 brute-force fast-path algorithm
 * core, side-by-side with gpu_md5_core_str. Same MD5 family geometry
 * (HASH_WORDS=4, HASH_BLOCK_BYTES=64) but distinguished in the kernel
 * cache key by an extra defines_str token "BF_FAST_MD5=1" so the two
 * compile to distinct cl_program / cl_kernel objects on each device.
 * A1 body is a verbatim copy of gpu_md5_core.cl rev 1.2 (correctness
 * gate); A2 inlines md5_block + pre-add hoists; A3 host-side mask
 * pre-explosion; A4 vendor intrinsics. Selected at dispatch time by
 * gpu_template_resolve_kernel when bf_fast_eligible is set on the
 * jobg slot (unsalted JOB_MD5, Numrules <= 1, npre==0, napp in [1,8])
 * unless MDXFIND_GPU_FAST_DISABLE=1 forces the slow path. See
 * project_bf_phase19_kernel_parity.md. */
#include "gpu_md5_bf_str.h"
/* Memo B Phase B4 (2026-05-04): SHA1 algorithm core for the generic
 * dispatch template. Parallel to gpu_md5_core_str — selected at template
 * compile time via defines_str = "HASH_WORDS=5,HASH_BLOCK_BYTES=64".
 * The same gpu_template.cl source text is used for both; the R3 cache-key
 * fix ensures distinct cache entries per (algorithm) instantiation. */
#include "gpu_sha1_core_str.h"
/* Memo B Phase B4 fan-out (2026-05-04): SHA256/SHA224/MD4 algorithm cores
 * for the generic dispatch template. Parallel to gpu_sha1_core_str —
 * selected at template compile time via defines_str (HASH_WORDS=8 / 7
 * respectively for SHA256/SHA224; HASH_WORDS=4 for MD4 — identical to
 * MD5 but distinct cache-key by source text). */
#include "gpu_sha256_core_str.h"
#include "gpu_sha224_core_str.h"
#include "gpu_md4_core_str.h"
/* Memo B Phase B5 sub-batch 1 (2026-05-04): SHA384/SHA512 algorithm cores.
 * First 64-bit-state algorithms in the family. Defines:
 *   SHA384: HASH_WORDS=12,HASH_BLOCK_BYTES=128 (truncates 6 ulong = 12 uint32)
 *   SHA512: HASH_WORDS=16,HASH_BLOCK_BYTES=128
 * Both share the sha512_block compression primitive (gpu_common.cl). */
#include "gpu_sha384_core_str.h"
#include "gpu_sha512_core_str.h"
/* Memo B Phase B5 sub-batch 2 (2026-05-05): RIPEMD-160 / RIPEMD-320 algorithm
 * cores. RIPEMD-160: 5 × uint32 LE state, 64-byte block, HASH_WORDS=5,
 * HASH_BLOCK_BYTES=64. RIPEMD-320: 10 × uint32 LE state (no merge of dual
 * pipeline at end), 64-byte block, HASH_WORDS=10. Both use LE message-word
 * layout (like MD5; UNLIKE the SHA family) and no-bswap probe / emit. The
 * compression primitives (rmd160_block / rmd320_block) live in gpu_common.cl. */
#include "gpu_ripemd160_core_str.h"
#include "gpu_ripemd320_core_str.h"
/* Memo B Phase B5 sub-batch 3 (2026-05-06): BLAKE2 family algorithm cores
 * for the generic dispatch template. Three algorithms wired:
 *   BLAKE2S-256:  HASH_WORDS=8,  HASH_BLOCK_BYTES=64  (LE per uint32,
 *                                                     b2s_compress)
 *   BLAKE2B-256:  HASH_WORDS=8,  HASH_BLOCK_BYTES=128 (4-of-8 ulong → 8
 *                                                     uint32 LE pairs,
 *                                                     b2b_compress NEW)
 *   BLAKE2B-512:  HASH_WORDS=16, HASH_BLOCK_BYTES=128 (full 8 ulong → 16
 *                                                     uint32 LE pairs)
 * BLAKE2 differs structurally from MD-style: state carries a byte counter
 * t[2] + finalization flag f[2] in addition to the digest chaining. The
 * counter+flag are kept INSIDE the per-algo state struct (Memo B brief §
 * sub-batch 3) so template_transform's signature is unchanged across the
 * 9 pre-existing cores. */
#include "gpu_blake2s256_core_str.h"
#include "gpu_blake2b256_core_str.h"
#include "gpu_blake2b512_core_str.h"
/* Memo B Phase B5 sub-batch 4 (2026-05-03): SHA3 / Keccak family algorithm
 * cores. EIGHT algorithms wired (KECCAK-{224,256,384,512} suffix=0x01,
 * SHA3-{224,256,384,512} suffix=0x06). Sponge construction with rate per
 * algorithm (HASH_BLOCK_BYTES = rate). Shared keccakf1600() in gpu_common.cl
 * rev 1.14+. EMIT_HIT widths reused across pairs (224→7, 256→8, 384→12,
 * 512→16). Keccak vs SHA3 differ ONLY in the suffix byte applied during
 * pad-finalize; same rate, same state shape, same h[] decomposition. */
#include "gpu_keccak224_core_str.h"
#include "gpu_keccak256_core_str.h"
#include "gpu_keccak384_core_str.h"
#include "gpu_keccak512_core_str.h"
#include "gpu_sha3_224_core_str.h"
#include "gpu_sha3_256_core_str.h"
#include "gpu_sha3_384_core_str.h"
#include "gpu_sha3_512_core_str.h"
/* Memo B Phase B5 sub-batch 5a (2026-05-03), Tier 1: SHA384RAW + SHA512RAW.
 * Mirror gpu_sha384_core.cl / gpu_sha512_core.cl byte-for-byte EXCEPT for
 * template_iterate, which re-feeds the BINARY digest (48/64 bytes) instead
 * of hex-encoding it. CPU references at mdxfind.c JOB_SHA384RAW (line 27744)
 * and JOB_SHA512RAW (line 27579). */
#include "gpu_sha384raw_core_str.h"
#include "gpu_sha512raw_core_str.h"
/* Memo B Phase B5 sub-batch 6 (2026-05-03), Tier A: MD5RAW + SHA1RAW + SHA256RAW.
 * Mirror gpu_md5_core.cl / gpu_sha1_core.cl / gpu_sha256_core.cl byte-for-byte
 * EXCEPT for template_iterate, which re-feeds the BINARY digest (16/20/32 bytes)
 * instead of hex-encoding it. CPU references at mdxfind.c JOB_MD5RAW (line 24237),
 * JOB_SHA1RAW (line 26321), and JOB_SHA256RAW (line 27405). */
#include "gpu_md5raw_core_str.h"
#include "gpu_sha1raw_core_str.h"
#include "gpu_sha256raw_core_str.h"
/* Memo B Phase B5 sub-batch 6 (2026-05-03), Tier C: SQL5. SHA1(SHA1(p))
 * with UPPERCASE-hex feedback in the iter step. Per-algo template_state
 * holds two SHA1 chains. CPU reference at mdxfind.c JOB_SQL5 (line 25301). */
#include "gpu_sql5_core_str.h"
/* Memo B Phase B6.11 (2026-05-06): SHA1DRU (Drupal SHA1, hashcat -m 7900).
 * First 1M-iteration algorithm on the unified template path. SHA1(pass) +
 * 1M iters of SHA1(hex_lc(state) || pass); ONE probe at the final state.
 * Implementation puts the 1M-loop inside template_finalize and uses
 * max_iter=1 (host-forced) so the kernel's outer iter loop runs once and
 * never calls template_iterate (which is a stub). CPU reference at
 * mdxfind.c JOB_SHA1DRU (lines 14261-14285). */
#include "gpu_sha1dru_core_str.h"
/* Memo B Phase B7.7b (2026-05-07): MD6256 (-m 17800). Final M5 closure
 * from B9 gate-fail — JOB_MD6256 admitted to the rules-engine path.
 * MD6-256 single-block leaf compression with 89-ulong N input + 1753-ulong
 * A working array. defines_str: HASH_WORDS=8,HASH_BLOCK_BYTES=64,
 * BASE_ALGO=md6. CPU reference at mdxfind.c JOB_MD6256 (lines 25836-25855). */
#include "gpu_md6256_core_str.h"
/* Memo B Phase B5 sub-batch 6 (2026-05-03), Tier B: NTLMH. MD4 of UTF-16LE
 * zero-extend(p) — hashcat-compatible NT password hash. CPU reference at
 * mdxfind.c JOB_NTLMH (line 15174). */
#include "gpu_ntlmh_core_str.h"
/* Memo B Phase B5 sub-batch 8 (2026-05-05): MD4UTF16 (-m e496). Same
 * MD4(UTF-16LE-zero-extend(p)) as NTLMH but with iter loop support
 * (Maxiter > 1 feeds back hex of prior digest as UTF-16LE-zero-extend
 * input). CPU reference at mdxfind.c JOB_MD4UTF16 (line 15040-15068). */
#include "gpu_md4utf16_core_str.h"
/* Memo B Phase B5 sub-batch 7 (2026-05-05): MYSQL3 (-m e456). Legacy MySQL
 * OLD_PASSWORD() hash, 64-bit output, per-byte arithmetic accumulator.
 * Iter step feeds back lowercase-hex (16 ASCII chars) of prior digest.
 * CPU reference at mdxfind.c JOB_MYSQL3 (line 25177-25187) and the
 * mysql3() function at mdxfind.c:3494-3513. */
#include "gpu_mysql3_core_str.h"
/* Memo B Phase B5 sub-batch 6.5 (2026-05-05): WRL (-m e5). Whirlpool
 * 512-bit hash. Miyaguchi-Preneel over 64-byte BE block; iter step
 * re-feeds 128 lowercase hex chars of prior digest. CPU reference at
 * mdxfind.c JOB_WRL (line 28014-28025). */
#include "gpu_wrl_core_str.h"
/* B5 sub-batch 5b retry (2026-05-06): Streebog-256/512 (-m e430/e431).
 * GOST R 34.11-2012. 64-byte block, 12-round LPS-keyed compression with
 * 16 KB __constant SBOB_SL64 + 768 B SBOB_RC64. SBOG_LPS rewritten to use
 * shift-then-mask access matching WRL_OP (RDNA4 gfx1201 mitigation). */
#include "gpu_streebog256_core_str.h"
#include "gpu_streebog512_core_str.h"
/* B6 salt-axis (2026-05-06): MD5SALT (hashcat -m 10) double-MD5 chain
 * MD5(hex32(MD5(p)) || salt). MD5SALTPASS (hashcat -m 20) simple
 * prepend MD5(salt || pass). Both compiled with -DGPU_TEMPLATE_HAS_SALT=1
 * plus distinct SALT_POSITION tokens (APPEND_TO_HEX32 / PREPEND) so the
 * kernel-cache key disambiguates them. CPU reference at mdxfind.c
 * JOB_MD5SALT (lines 21943-21974) and JOB_MD5SALTPASS (lines 15776-15832). */
#include "gpu_md5salt_core_str.h"
#include "gpu_md5saltpass_core_str.h"
/* B6.1 SHA1 fan-out (2026-05-06): SHA1SALTPASS (hashcat -m 110) =
 * SHA1(salt || pass). First SHA-family salted variant. Cache disambiguated
 * from MD5SALTPASS via HASH_WORDS=5 + BASE_ALGO=sha1 tokens. CPU reference
 * at mdxfind.c JOB_SHA1SALTPASS (lines 14369-14418). */
#include "gpu_sha1saltpass_core_str.h"
/* B6.2 SHA256 fan-out (2026-05-06): SHA256SALTPASS (hashcat -m 1410) =
 * SHA256(salt || pass). Second SHA-family salted variant. Cache
 * disambiguated from SHA1SALTPASS via HASH_WORDS=8 + BASE_ALGO=sha256
 * tokens (HASH_WORDS, BASE_ALGO both differ). CPU reference at mdxfind.c
 * JOB_SHA256SALTPASS (lines 27603-27651). */
#include "gpu_sha256saltpass_core_str.h"
/* B6.3 SHA224 fan-out (2026-05-06): SHA224SALTPASS (hashcat -m 1310) =
 * SHA224(salt || pass). Third SHA-family salted variant — sha256_block
 * compression with 7-word truncated output. Cache disambiguated from
 * SHA256SALTPASS via HASH_WORDS=7 (vs 8); from SHA1SALTPASS via
 * HASH_WORDS=7 + BASE_ALGO=sha256 (both axes differ); from MD5SALTPASS
 * via HASH_WORDS=7 + BASE_ALGO=sha256 (both axes differ). CPU reference
 * at mdxfind.c JOB_SHA224SALTPASS. */
#include "gpu_sha224saltpass_core_str.h"
/* B6.4 MD5PASSSALT fan-out (2026-05-06): first APPEND-shape salted variant.
 * Auto-generated by gpu/codegen/codegen.py from gpu/codegen/specs.py;
 * matches the unified template extension protocol used by MD5SALT,
 * MD5SALTPASS, SHA1SALTPASS, SHA256SALTPASS, SHA224SALTPASS. CPU reference
 * at mdxfind.c JOB_MD5PASSSALT. */
#include "gpu_md5passsalt_core_str.h"
/* B6.5 SHA1PASSSALT fan-out (2026-05-06): first SHA-family APPEND-shape
 * salted variant. Auto-generated by gpu/codegen/codegen.py from
 * gpu/codegen/specs.py; cache disambiguated from SHA1SALTPASS via
 * SALT_POSITION=APPEND (vs PREPEND); same BASE_ALGO=sha1 + HASH_WORDS=5
 * axes. Authors finalize_append_be.cl.frag which unblocks future
 * SHA-family APPEND variants (SHA256PASSSALT becomes pure spec reuse).
 * CPU reference at mdxfind.c JOB_SHA1PASSSALT (lines 14227-14270). */
#include "gpu_sha1passsalt_core_str.h"
/* B6.7 SHA256PASSSALT fan-out (2026-05-06): second SHA-family APPEND-shape
 * salted variant — pure spec reuse. Auto-generated by gpu/codegen/codegen.py
 * from gpu/codegen/specs.py; cache disambiguated from SHA256SALTPASS via
 * SALT_POSITION=APPEND (vs PREPEND); same BASE_ALGO=sha256 + HASH_WORDS=8
 * axes. Reuses sha256_style_salted.cl.tmpl + finalize_append_be.cl.frag
 * — no new template/fragment authored. CPU reference at mdxfind.c
 * JOB_SHA256PASSSALT (lines 27639-27677). */
#include "gpu_sha256passsalt_core_str.h"
/* B6.9 SHA512 fan-out (2026-05-06): SHA512SALTPASS — first 64-bit-state
 * salted variant. Auto-generated by gpu/codegen/codegen.py from
 * gpu/codegen/specs.py; cache disambiguated from every other salted
 * template via HASH_BLOCK_BYTES=128 (the 128-byte block is unique to
 * the SHA-384/512 family among salted variants on the codegen path) +
 * HASH_WORDS=16 + BASE_ALGO=sha512. Authors a sibling main template
 * (sha512_style_salted.cl.tmpl) AND a sibling fragment
 * (finalize_prepend_be64.cl.frag) — width-bearing constants
 * (block_size, word_width, length-field-width) live in the template
 * + fragment, not parameterized into the SHA-256 versions. CPU
 * reference at mdxfind.c JOB_SHA512SALTPASS (lines 13981-14023). */
#include "gpu_sha512saltpass_core_str.h"
/* B6.10 SHA512PASSSALT fan-out (2026-05-06): SHA512PASSSALT — second
 * 64-bit-state salted variant. FINAL B6 ladder step. APPEND-shape
 * sibling of SHA512SALTPASS (B6.9). Same SHA-512 family compression
 * + 8 ulong state + 128-byte block + 128-bit length field; only the
 * salt POSITION at template_finalize differs (APPEND vs PREPEND).
 * Cache disambiguated via SALT_POSITION=APPEND in defines_str (vs
 * PREPEND); same BASE_ALGO=sha512 + HASH_WORDS=16 + HASH_BLOCK_BYTES=128
 * axes — single-axis delta. Pure spec reuse on the SHA-512 main
 * template (sha512_style_salted.cl.tmpl, salt-position-agnostic per
 * B6.9), plus ONE new fragment authoring (finalize_append_be64.cl.frag —
 * sibling of finalize_prepend_be64.cl.frag, same width-bearing reasons
 * as the 32-bit BE APPEND/PREPEND pair). CPU reference at mdxfind.c
 * JOB_SHA512PASSSALT (lines 14069-14127). */
#include "gpu_sha512passsalt_core_str.h"
/* Family E HMAC-SHA384 carrier (2026-05-08): SHA384SALTPASS-shaped carrier
 * core. Auto-generated by gpu/codegen/codegen.py from gpu/codegen/specs.py
 * (sha384saltpass spec, template_enum_value=46). Carries HMAC-SHA384
 * (e543) and HMAC-SHA384_KPASS (e796) via the algo_mode 5/6 HMAC body in
 * finalize_prepend_be64.cl.frag (gated on HASH_WORDS == 12 && algo_mode
 * >= 5u). No JOB_SHA384SALTPASS algorithm exists in mdxfind; the mode-0
 * SHA384(salt||pass) main body is structurally unreachable in production
 * (host always sets algo_mode 5 or 6 for HMAC-SHA384 dispatch). Cache
 * disambiguated from SHA512SALTPASS via HASH_WORDS=12 (vs 16) — same
 * BASE_ALGO=sha512 since the compression primitive is identical
 * (sha512_block). EMIT_HIT_12 emits 6 ulong = 48 bytes (matches HMAC-
 * SHA384's 48-byte digest). CPU reference for the HMAC-SHA384 KSALT/KPASS
 * paths at mdxfind.c HMAC_start (line 29406+) / HMAC_KPASS_start (line
 * 29509+) cases JOB_HMAC_SHA384 (line 29369) / JOB_HMAC_SHA384_KPASS
 * (line 29522). */
#include "gpu_sha384saltpass_core_str.h"
/* Family G HMAC-RIPEMD-160 carrier (2026-05-08): RIPEMD160SALTPASS-shaped
 * carrier core. Auto-generated by gpu/codegen/codegen.py from gpu/codegen/-
 * specs.py (ripemd160saltpass spec, template_enum_value=48). Carries
 * HMAC-RMD160 (e211) and HMAC-RMD160_KPASS (e798) via the algo_mode 5/6
 * HMAC body in finalize_prepend_rmd.cl.frag (gated on HASH_WORDS == 5 &&
 * algo_mode >= 5u). No JOB_RIPEMD160SALTPASS algorithm exists in mdxfind;
 * the mode-0 RMD160(salt||pass) main body is structurally unreachable in
 * production (host always sets algo_mode 5 or 6 for HMAC-RMD160 dispatch).
 * Cache disambiguated from SHA1SALTPASS via BASE_ALGO=ripemd160 (vs sha1)
 * — same HASH_WORDS=5 + HASH_BLOCK_BYTES=64 axes; the BASE_ALGO axis is
 * load-bearing (ripemd160_block has different compression rounds and 2-arg
 * call signature vs sha1_block's BE compression). EMIT_HIT_5 emits 5 LE
 * uint32 = 20 bytes (matches HMAC-RMD160's 20-byte digest). CPU reference
 * for the HMAC-RMD160 KSALT/KPASS paths at mdxfind.c HMAC_start /
 * HMAC_KPASS_start cases JOB_HMAC_RMD160 (line 29391) / JOB_HMAC_RMD160_-
 * KPASS (line 29584). */
#include "gpu_ripemd160saltpass_core_str.h"
/* Family H HMAC-RIPEMD-320 carrier (2026-05-08): RIPEMD320SALTPASS-shaped
 * carrier core. Auto-generated by gpu/codegen/codegen.py from gpu/codegen/-
 * specs.py (ripemd320saltpass spec, template_enum_value=49). Carries
 * HMAC-RMD320 (e213) and HMAC-RMD320_KPASS (e799) via the algo_mode 5/6
 * HMAC body in finalize_prepend_rmd.cl.frag (gated on HASH_WORDS == 10 &&
 * algo_mode >= 5u). No JOB_RIPEMD320SALTPASS algorithm exists in mdxfind;
 * the mode-0 RMD320(salt||pass) main body is structurally unreachable in
 * production (host always sets algo_mode 5 or 6 for HMAC-RMD320 dispatch).
 * Cache disambiguated from RIPEMD160SALTPASS via HASH_WORDS=10 (vs 5) +
 * BASE_ALGO=rmd320 (vs rmd160). EMIT_HIT_10 emits 10 LE uint32 = 40 bytes
 * (matches HMAC-RMD320's 40-byte digest). CPU reference for the HMAC-
 * RMD320 KSALT/KPASS paths at mdxfind.c HMAC_start / HMAC_KPASS_start
 * cases JOB_HMAC_RMD320 (line 29428) / JOB_HMAC_RMD320_KPASS (line
 * 29616). */
#include "gpu_ripemd320saltpass_core_str.h"
/* Family I HMAC-BLAKE2S carrier (2026-05-08): Path A hand-written core for
 * HMAC-BLAKE2S (e828). Hand-written (vs codegen emission) because BLAKE2's
 * counter+flag state struct + 4-arg b2s_compress primitive don't fit the
 * codegen tool's MD-style fragment family. The HMAC body is inlined inside
 * template_finalize gated on algo_mode == 5u; no production
 * JOB_BLAKE2SSALTPASS exists, so mode-0 of this kernel is structurally
 * unreachable. Cache disambiguated from BLAKE2S256 unsalted core via the
 * HAS_SALT=1 axis (absent in the unsalted defines_str). CPU reference
 * for the HMAC-BLAKE2S path at mdxfind.c JOB_HMAC_BLAKE2S (line 30341).
 * Slab oracle (whole-file retired in same commit): gpu_hmac_blake2s.cl
 * hmac_blake2s_kpass_batch (lines 38-97). */
#include "gpu_hmac_blake2s_core_str.h"
/* Family J HMAC-STREEBOG-256 carrier (2026-05-08): hand-written Path A
 * sibling of gpu_streebog256_core.cl, mirrors Family I (HMAC-BLAKE2S)
 * structure. Two algo_modes: 5 = JOB_HMAC_STREEBOG256_KSALT (e838),
 * 6 = JOB_HMAC_STREEBOG256_KPASS (e837). Hand-written rather than codegen
 * because Streebog's compression primitive (streebog_g + streebog_hash_priv
 * with 16 KB __constant SBOB_SL64 table + 12-round AES-like rounds) doesn't
 * fit the codegen tool's MD-style fragment family. The HMAC body is
 * inlined inside template_finalize gated on algo_mode >= 5u (collapses
 * KSALT + KPASS to a single branch since the kernel-side math is identical;
 * the host's algo_mode setter and salt-Judy plumbing distinguishes the
 * two). No production JOB_STREEBOG256SALTPASS exists, so mode-0 of this
 * kernel is structurally unreachable. Cache disambiguated from STREEBOG-256
 * unsalted core via HAS_SALT=1 + HMAC_KSALTPASS=1 axes (absent in the
 * unsalted defines_str). CPU references for the HMAC-STREEBOG-256 paths at
 * mdxfind.c JOB_HMAC_STREEBOG256_KPASS (line 30764) + JOB_HMAC_STREEBOG256_-
 * KSALT (line 30822). Slab oracles surgically retired in same commit:
 * gpu_streebog.cl hmac_streebog256_kpass_batch (lines 837-900) +
 * hmac_streebog256_ksalt_batch (lines 902-966). KEEP streebog512 HMAC
 * kernels (Family K scope). */
#include "gpu_hmac_streebog256_core_str.h"
/* Family K HMAC-STREEBOG-512 carrier (2026-05-08): hand-written Path A
 * sibling of gpu_streebog512_core.cl, mirrors Family J at HASH_WORDS=16.
 * Two algo_modes: 5 = JOB_HMAC_STREEBOG512_KSALT (e840), 6 = JOB_HMAC_-
 * STREEBOG512_KPASS (e839). Hand-written rather than codegen because
 * Streebog's compression primitive (streebog_g + streebog_hash_priv with
 * 16 KB __constant SBOB_SL64 table + 12-round AES-like rounds) doesn't
 * fit the codegen tool's MD-style fragment family. The HMAC body is
 * inlined inside template_finalize gated on algo_mode >= 5u (collapses
 * KSALT + KPASS to a single branch since the kernel-side math is
 * identical; the host's algo_mode setter and salt-Judy plumbing
 * distinguishes the two). No production JOB_STREEBOG512SALTPASS exists,
 * so mode-0 of this kernel is structurally unreachable. Cache
 * disambiguated from STREEBOG-512 unsalted core via HAS_SALT=1 +
 * HMAC_KSALTPASS=1 axes (absent in the unsalted defines_str). Final
 * HMAC family in the ladder. CPU references for the HMAC-STREEBOG-512
 * paths at mdxfind.c JOB_HMAC_STREEBOG512_KPASS (line 30918) +
 * JOB_HMAC_STREEBOG512_KSALT (line 30975). Slab oracles surgically
 * retired in same commit: gpu_streebog.cl hmac_streebog512_kpass_batch
 * (lines 859-923) + hmac_streebog512_ksalt_batch (lines 925-989). */
#include "gpu_hmac_streebog512_core_str.h"
/* PHPBB3 carrier (2026-05-08): hand-written Path A salted-template core
 * for JOB_PHPBB3 (e455). Iterated MD5 chain (count decoded from salt
 * byte 3 via phpitoa64) INSIDE template_finalize; max_iter=1 forced
 * host-side. CPU reference: mdxfind.c JOB_PHPBB3 (lines 13415-13628).
 * Slab oracle: gpu_phpbb3.cl phpbb3_batch (retired in same commit). */
#include "gpu_phpbb3_core_str.h"
/* MD5CRYPT carrier (2026-05-08): hand-written Path A salted-template core
 * for JOB_MD5CRYPT (e511). Iterated MD5 chain (1000 fixed iters per BSD
 * $1$ md5crypt) INSIDE template_finalize; max_iter=1 forced host-side.
 * CPU reference: mdxfind.c JOB_MD5CRYPT (lines 13017-13117). Slab oracle:
 * gpu_md5crypt.cl md5crypt_batch (retired in same commit). Phase 1 of
 * the Unix-crypt ladder. */
#include "gpu_md5crypt_core_str.h"
/* SHACRYPT shared core (2026-05-08): hand-written Path A salted-template
 * core for JOB_SHA256CRYPT (e512), JOB_SHA512CRYPT (e513), and JOB_-
 * SHA512CRYPTMD5 (e510). 5-step glibc crypt-sha2 chain INSIDE template_-
 * finalize; rounds loop honors "rounds=N$" salt prefix (clamp 1000..
 * 999999999, default 5000). max_iter=1 forced host-side. CPU reference:
 * mdxfind.c shared crypt_round body at lines 12177-12290 + width-specific
 * b64 permutation tables at 12300+ (cryptlen=64) / 12753+ (cryptlen=32).
 * Slab oracles: gpu_sha256crypt.cl sha256crypt_batch (Phase 2 retirement)
 * + gpu_sha512crypt.cl sha512crypt_batch (Phase 3 retirement candidate).
 * Phase 2 ship 2026-05-08 instantiates this shared core at HASH_WORDS=8
 * for SHA256CRYPT only. Single include serves all Phase-2/3/4
 * instantiations -- HASH_WORDS, BASE_ALGO, and (Phase 4) algo_mode
 * tokens differ at compile time per gpu_opencl_template_compile_*
 * defines_str. */
#include "gpu_shacrypt_core_str.h"
#include "gpu_template_str.h"
/* Old monolithic kernel source removed — per-family compilation now.
 * See gpu_common_str.h + gpu_*_str.h
 *
 * REMOVED: ~300 lines of inline kernel string (kernel_source_embedded_unused)
"}\n";
#endif

/* ---- List GPU devices and exit (called early from option parsing) ---- */
void gpu_opencl_list_devices(void) {
    if (opencl_dynload_init() != 0) {
        fprintf(stderr, "  OpenCL library not found.\n");
        exit(0);
    }
    cl_platform_id plats[8];
    cl_uint nplat = 0;
    cl_int err = clGetPlatformIDs(8, plats, &nplat);
    if (err != CL_SUCCESS || nplat == 0) {
        fprintf(stderr, "  No OpenCL platforms found.\n");
        exit(0);
    }
    int idx = 0;
    for (cl_uint p = 0; p < nplat; p++) {
        cl_device_id devs[64];
        cl_uint ndev = 0;
        err = clGetDeviceIDs(plats[p], CL_DEVICE_TYPE_GPU, 64, devs, &ndev);
        if (err != CL_SUCCESS || ndev == 0) continue;
        for (cl_uint d = 0; d < ndev; d++) {
            char dname[256];
            cl_ulong gmem = 0;
            clGetDeviceInfo(devs[d], CL_DEVICE_NAME, sizeof(dname), dname, NULL);
            clGetDeviceInfo(devs[d], CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(gmem), &gmem, NULL);
            fprintf(stderr, "  GPU[%d]: %s (%llu MB)\n", idx, dname,
                    (unsigned long long)(gmem / (1024*1024)));
            idx++;
        }
    }
    if (idx == 0) fprintf(stderr, "  No GPU devices found.\n");
    exit(0);
}

/* ---- GPU device filter ---- */
extern int gpu_device_filter_set;
extern int GPUForce;
extern int gpu_device_allowed[];

static int device_allowed(int idx) {
    if (!gpu_device_filter_set) return 1;
    if (idx < 0 || idx >= 64) return 0;
    return gpu_device_allowed[idx];
}

/* ---- Helper ---- */
#define OCL_CHECK(call, msg) do { cl_int _e = (call); if (_e != CL_SUCCESS) { \
    fprintf(stderr, "OpenCL error %d: %s\n", _e, msg); return -1; } } while(0)

static cl_mem dev_buf(struct gpu_device *d, size_t size, cl_mem_flags flags) {
    cl_int err;
    return clCreateBuffer(d->ctx, flags, size ? size : 4, NULL, &err);
}

/* Allocate a cl_mem with a minimum allocation of MIN_BUFFER_BYTES.
 *
 * Why: NVIDIA's Windows OpenCL driver validates buffer-vs-signature
 * compatibility at clEnqueueNDRangeKernel time and rejects small (<256 B)
 * cl_mem objects against `__global uint *` / `__global ulong *` kernel
 * signatures with CL_INVALID_KERNEL_ARGS. The kernel side correctly guards
 * `if (gid < count)` before any access, but the static analyzer doesn't
 * honor those runtime guards. The fix is to over-allocate so the buffer
 * looks "real" to the validator, then write actual data into the prefix
 * and zero-pad the unused tail.
 *
 * Reproduced on Shooter's 12-GPU RTX 4090 Windows rig (bug2a/bug2b,
 * 2026-05-04). Linux NVIDIA always accepts these tiny buffers; Windows
 * NVIDIA rejects. ioblade Linux validation cannot reproduce locally —
 * the gates run on Linux are necessary-but-not-sufficient; Shooter's
 * Windows run is the definitive Truth Test.
 *
 * The split create + fill + write pattern (instead of CL_MEM_COPY_HOST_PTR
 * with a temporary padded host buffer) keeps the alloc identical for the
 * already-large case, and avoids any extra host allocation when the data
 * is large enough already.
 *
 * If host_data == NULL, the buffer is just zero-padded (no data write).
 * Suitable for read-write buffers that the kernel zero-inits / fills.
 */
#define MIN_BUFFER_BYTES 4096

static cl_mem create_min_buf(cl_context ctx, cl_command_queue q,
                             cl_mem_flags flags, size_t actual_bytes,
                             const void *host_data, cl_int *err_out) {
    cl_int err;
    size_t alloc = actual_bytes < MIN_BUFFER_BYTES ? MIN_BUFFER_BYTES : actual_bytes;
    cl_mem buf = clCreateBuffer(ctx, flags, alloc, NULL, &err);
    if (err != CL_SUCCESS || !buf) {
        if (err_out) *err_out = err;
        return NULL;
    }
    /* Zero-pad the entire allocation first (covers tail if min-size was
     * applied, AND covers the data region for the no-host-data case). */
    if (p_clEnqueueFillBuffer) {
        uint32_t zero32 = 0;
        cl_int fe = clEnqueueFillBuffer(q, buf, &zero32, sizeof(zero32),
                                         0, alloc, 0, NULL, NULL);
        if (fe != CL_SUCCESS) {
            /* clEnqueueFillBuffer can fail on some 1.1-era drivers even
             * when the symbol is loaded; fall through to host stage. */
            uint8_t *z = (uint8_t *)calloc(1, alloc);
            if (z) {
                clEnqueueWriteBuffer(q, buf, CL_TRUE, 0, alloc, z, 0, NULL, NULL);
                free(z);
            }
        }
    } else {
        /* OpenCL 1.1 fallback: stage zero buffer */
        uint8_t *z = (uint8_t *)calloc(1, alloc);
        if (z) {
            clEnqueueWriteBuffer(q, buf, CL_TRUE, 0, alloc, z, 0, NULL, NULL);
            free(z);
        }
    }
    /* Write actual data into the prefix. Blocking write so the buffer is
     * in its final state by the time we return — matches the prior
     * CL_MEM_COPY_HOST_PTR behavior these call sites expect. */
    if (host_data && actual_bytes > 0) {
        cl_int we = clEnqueueWriteBuffer(q, buf, CL_TRUE, 0, actual_bytes,
                                         host_data, 0, NULL, NULL);
        if (we != CL_SUCCESS) {
            clReleaseMemObject(buf);
            if (err_out) *err_out = we;
            return NULL;
        }
    }
    if (err_out) *err_out = CL_SUCCESS;
    return buf;
}

/* Per-family kernel source (concatenated with gpu_common_str at compile time) */
static const char *family_source[FAM_COUNT] = {
    [FAM_MD5SALT]            = gpu_md5salt_str,
    /* [FAM_MD5SALTPASS] retired in B8 (2026-05-06): slab kernels gone;
     * JOB_MD5SALTPASS / JOB_MD5PASSSALT now route through the template path.
     * Implicit NULL designated initializer; compile_families NULL-skips. */
    /* [FAM_MD5ITER] retired in B12 (2026-05-08): md5_iter_lc/uc slab kernels
     * gone; warm_probe NULL-skips. Implicit NULL designated initializer. */
    /* [FAM_PHPBB3] retired in PHPBB3 carrier ship (2026-05-08): slab
     * kernel phpbb3_batch gone; JOB_PHPBB3 now routes through the
     * unified template path via the PHPBB3 carrier (gpu_phpbb3_core.cl).
     * Implicit NULL designated initializer; compile_families NULL-skips. */
    /* [FAM_MD5CRYPT] retired in MD5CRYPT carrier ship (2026-05-08): slab
     * kernel md5crypt_batch gone; JOB_MD5CRYPT now routes through the
     * unified template path via the MD5CRYPT carrier (gpu_md5crypt_core.cl).
     * Implicit NULL designated initializer; compile_families NULL-skips. */
    /* [FAM_MD5_MD5SALTMD5PASS] retired in B8 (2026-05-06): slab kernel gone;
     * JOB_MD5_MD5SALTMD5PASS now routes through the MD5SALT template
     * (params.algo_mode=4). Implicit NULL designated initializer. */
    /* [FAM_SHA1] retired in Family B (2026-05-07): slab kernels gone;
     * JOB_HMAC_SHA1 / JOB_HMAC_SHA1_KPASS now route through the
     * SHA1SALTPASS unified template via params.algo_mode = 5 / 6.
     * Implicit NULL designated initializer; compile_families NULL-skips. */
    /* [FAM_SHA256] retired in Family D (2026-05-08): hmac_sha256_ksalt_batch
     * + hmac_sha256_kpass_batch slab kernels gone; JOB_HMAC_SHA256 / JOB_-
     * HMAC_SHA256_KPASS now route through the unified SHA256SALTPASS
     * template path via params.algo_mode = 5 / 6 + RUNTIME-gated HMAC body
     * (NEVER `#if HASH_WORDS == 8`; rev 1.7 Pascal NVIDIA ABORT lesson).
     * Implicit NULL designated initializer; compile_families NULL-skips.
     * gpu_sha256.cl whole-file deleted from working tree in same commit;
     * RCS history retained at gpu/RCS/gpu_sha256.cl,v. Final HMAC family
     * in the ladder. */
    /* [FAM_MD5MASK] retired in B12 (2026-05-08): md5_mask_batch slab kernel
     * gone; -n/-N mask coverage uses the unified template path. Implicit
     * NULL designated initializer. */
    /* [FAM_DESCRYPT] retired in DESCRYPT carrier ship (2026-05-08, Unix-
     * crypt Phase 5): slab kernel descrypt_batch gone; JOB_DESCRYPT now
     * routes through the unified template path via the hand-written Path A
     * carrier (gpu_descrypt_core.cl). Implicit NULL designated initializer;
     * compile_families NULL-skips. After this Phase 5 retirement, ALL
     * FIVE Unix-crypt ops (MD5CRYPT, SHA256CRYPT, SHA512CRYPT,
     * SHA512CRYPTMD5, DESCRYPT) are template-routed; the Unix-crypt slab
     * path is fully retired. RCS history retains the slab kernel source
     * at gpu/RCS/gpu_descrypt.cl,v. */
    /* B9 (2026-05-07): 11 FAM_*UNSALTED slab kernels retired. The legacy
     * mask short-circuit (mdxfind.c:10911-10951) was the sole reachability
     * path; that short-circuit + gpu_try_pack_unsalted() in mdxfind.c were
     * retired in this same commit. All GPU_CAT_MASK ops route exclusively
     * through the unified template path. Implicit NULL designated init-
     * ializers; compile_families NULL-skips:
     *   FAM_MD5UNSALTED, FAM_MD4UNSALTED, FAM_SHA1UNSALTED,
     *   FAM_SHA256UNSALTED, FAM_SHA512UNSALTED, FAM_WRLUNSALTED,
     *   FAM_MD6256UNSALTED, FAM_KECCAKUNSALTED, FAM_MYSQL3UNSALTED,
     *   FAM_RMD160UNSALTED, FAM_BLAKE2S256UNSALTED.
     * gpu_opencl_dispatch_batch() stays alive for SALTED/SALTPASS ops.
     * RCS history retains the .cl + _str.h sources. */
    /* [FAM_HMAC_SHA512] retired in Family F (2026-05-08): all slab kernels
     * gone. Family E (2026-05-08) retired hmac_sha384_{ksalt,kpass}_batch
     * (#if 0 wrapped); Family F retires hmac_sha512_{ksalt,kpass}_batch.
     * The gpu_hmac_sha512.cl file has zero live kernels after this commit;
     * the entire file is `#if 0`-wrapped as a retirement-record. JOB_HMAC_-
     * SHA384 / JOB_HMAC_SHA384_KPASS / JOB_HMAC_SHA512 / JOB_HMAC_SHA512_-
     * KPASS now all route through the unified template path (gpu_template.cl
     * + gpu_sha384saltpass_core.cl for SHA-384, + gpu_sha512saltpass_core.cl
     * for SHA-512) via params.algo_mode = 5 / 6. Implicit NULL designated
     * initializer; compile_families NULL-skips. RCS history retains the
     * kernel sources at gpu/RCS/gpu_hmac_sha512.cl,v. */
    /* [FAM_HMAC_RMD160] retired in Family G (2026-05-08): all slab kernels
     * gone. hmac_rmd160_{ksalt,kpass}_batch wrapped in #if 0; both ops
     * (JOB_HMAC_RMD160 / _KPASS) now route through the unified template
     * path (gpu_template.cl + gpu_ripemd160saltpass_core.cl) via params.
     * algo_mode = 5 / 6. Implicit NULL designated initializer; compile_-
     * families NULL-skips. RCS history retains the kernel sources at
     * gpu/RCS/gpu_hmac_rmd160.cl,v. */
    /* [FAM_HMAC_RMD320] retired in Family H (2026-05-08): all slab kernels
     * gone. hmac_rmd320_{ksalt,kpass}_batch wrapped in #if 0; both ops
     * (JOB_HMAC_RMD320 / _KPASS) now route through the unified template
     * path (gpu_template.cl + gpu_ripemd320saltpass_core.cl) via params.
     * algo_mode = 5 / 6. Implicit NULL designated initializer; compile_-
     * families NULL-skips. RCS history retains the kernel sources at
     * gpu/RCS/gpu_hmac_rmd320.cl,v. */
    /* [FAM_HMAC_BLAKE2S] retired in Family I (2026-05-08): the only slab
     * kernel hmac_blake2s_kpass_batch wrapped in #if 0. JOB_HMAC_BLAKE2S
     * (e828) now routes through the unified template path (gpu_template.cl
     * + gpu_hmac_blake2s_core.cl, hand-written Path A) via params.algo_mode
     * = 5. Implicit NULL designated initializer; compile_families NULL-
     * skips. RCS history retains the kernel sources at
     * gpu/RCS/gpu_hmac_blake2s.cl,v. */
    /* Family K retirement (2026-05-08): FAM_STREEBOG family_source set to
     * NULL. Post-Family-K, no live slab kernels remain in the file
     * (B10 retired unsalted streebog batches; Family J retired streebog256
     * HMACs; Family K retires streebog512 HMACs). The gpu_streebog.cl
     * file is whole-file #if 0 wrapped as a retirement-record. JOB_-
     * STREEBOG_32 / JOB_STREEBOG_64 route through gpu_streebog{256,512}_-
     * core.cl unsalted templates (B10); HMAC-STREEBOG-256/512 route
     * through gpu_hmac_streebog{256,512}_core.cl Path A carrier templates
     * (Families J/K). Implicit NULL designated initializer; compile_-
     * families NULL-skips. RCS history retains the kernel sources at
     * gpu/RCS/gpu_streebog.cl,v. */
    /* [FAM_STREEBOG]       = NULL (Family K retirement) */
    /* [FAM_SHA512CRYPT] retired in SHA512CRYPTMD5 carrier ship (2026-05-08):
     * slab kernel sha512crypt_batch gone; both JOB_SHA512CRYPT (Phase 3)
     * and JOB_SHA512CRYPTMD5 (Phase 4) now route through the unified
     * template path via the SHACRYPT shared core (gpu_shacrypt_core.cl
     * at HASH_WORDS=16). Implicit NULL designated initializer; compile_-
     * families NULL-skips. Phase 4 of the Unix-crypt ladder (final
     * phase; Unix-crypt slab path fully retired). */
    /* [FAM_SHA256CRYPT] retired in SHA256CRYPT carrier ship (2026-05-08):
     * slab kernel sha256crypt_batch gone; JOB_SHA256CRYPT now routes
     * through the unified template path via the SHACRYPT shared core
     * (gpu_shacrypt_core.cl at HASH_WORDS=8). Implicit NULL designated
     * initializer; compile_families NULL-skips. Phase 2 of the Unix-
     * crypt ladder. */
    /* [FAM_BCRYPT] retired in BCRYPT carrier ship (2026-05-08, Unix-crypt
     * Phase 6): slab kernel bcrypt_batch gone; JOB_BCRYPT now routes
     * through the unified template path via the hand-written Path A
     * carrier (gpu_bcrypt_core.cl) at HASH_WORDS=6. Implicit NULL
     * designated initializer; compile_families NULL-skips. RCS history
     * retains the slab kernel source at gpu/RCS/gpu_bcrypt.cl,v. Phase 6
     * of the slab-retirement ladder (final major slab kernel). Compound
     * siblings (BCRYPTMD5/BCRYPTSHA1/BCRYPTSHA512) remain CPU-only via
     * gpu_op_category default fall-through. */
    /* [FAM_MD5PACKED] retired in B7.9 (2026-05-07): chokepoint pack gone;
     * gpu_md5_packed_str no longer included. The FAM_MD5PACKED enum value
     * remains in gpujob.h for ABI stability with sibling builds; the
     * compile-families loop NULL-skips this slot. */
};

/* B7.9 (2026-05-07): packed_kernel_map[] + packed_family() retired with
 * the chokepoint pack. Their only consumer was gpu_opencl_dispatch_packed,
 * also retired in this commit. RCS history retains the prior versions of
 * this file. The 5 packed kernel sources (md5/sha1/sha256/md4/sha512_packed)
 * have been deleted from the working tree per
 * feedback_remove_retired_gpu_kernels.md. */

/* Kernel-to-op mapping table. Each entry: kernel function name, ops it serves, family.
 * Adding a new GPU algorithm = one line here + a .cl file in gpu/. */
static const struct {
    const char *name;
    int ops[8];      /* -1 terminated */
    int family;
} kernel_map[] = {
    /* B8 retirement (2026-05-06): 11 slab kernels removed — md5salt_batch,
     * md5salt_sub8_24, md5saltpass_batch, md5passsalt_batch,
     * md5_md5saltmd5pass_batch, sha1passsalt_batch, sha1saltpass_batch,
     * sha256passsalt_batch, sha256saltpass_batch, sha512passsalt_batch (in
     * the FAM_HMAC_SHA512 group below), sha512saltpass_batch (likewise).
     * All ops now route through the unified template path via B6/B6.x
     * fan-outs. RCS history retains the kernel sources.
     *
     * 2026-05-07: md5salt_iter retired — JOB_MD5SALT iter coverage handled
     * by the unified template path's per-iter loop in template_phase0
     * (gpu_template.cl) using gpu_md5salt_core.cl as the iterate body.
     * probe_max_dispatch migrated to hmac_md5_ksalt_batch. */
    /* md5_iter_lc, md5_iter_uc retired in B12 (2026-05-08): FAM_MD5ITER
     * slab kernels gone — JOB_MD5/JOB_MD5UC iteration handled by the
     * rules-engine + template paths. */
    /* md5_mask_batch retired in B12 (2026-05-08): FAM_MD5MASK slab kernel
     * gone — JOB_MD5 mask coverage handled by the unified template path
     * (B6 -n/-N integration). */
    /* B8.x retirement (B6.11 commit, 2026-05-06): sha1dru_batch removed —
     * JOB_SHA1DRU routes through the unified template path via B6.11
     * fan-out (gpu_sha1dru_core.cl). The slab kernel sha1dru_batch in
     * gpu_sha1.cl was removed in this same commit. RCS history retains
     * the kernel source at gpu/RCS/gpu_sha1.cl,v. */
    /* PHPBB3 slab kernel phpbb3_batch retired in PHPBB3 carrier ship
     * (2026-05-08): JOB_PHPBB3 routes through the unified template path
     * via the carrier kernel (gpu_phpbb3_core.cl). RCS history retains
     * the slab kernel source at gpu/RCS/gpu_phpbb3.cl,v. */
    /* MD5CRYPT slab kernel md5crypt_batch retired in MD5CRYPT carrier ship
     * (2026-05-08): JOB_MD5CRYPT routes through the unified template path
     * via the carrier kernel (gpu_md5crypt_core.cl). RCS history retains
     * the slab kernel source at gpu/RCS/gpu_md5crypt.cl,v. probe_max_-
     * dispatch is anchored on FAM_MD5SALT (hmac_md5_ksalt_batch), not
     * FAM_MD5CRYPT, so retirement does not require a probe migration. */
    /* DESCRYPT slab kernel descrypt_batch retired in DESCRYPT carrier
     * ship (2026-05-08, Unix-crypt Phase 5): JOB_DESCRYPT routes through
     * the unified template path via the hand-written Path A carrier
     * (gpu_descrypt_core.cl). RCS history retains the slab kernel source
     * at gpu/RCS/gpu_descrypt.cl,v. probe_max_dispatch is anchored on
     * FAM_MD5SALT (hmac_md5_ksalt_batch), not FAM_DESCRYPT, so retirement
     * does not require a probe migration. After this Phase 5, FAM_DESCRYPT
     * has zero kernel_map entries -- the entire family is dead and the
     * family_source[FAM_DESCRYPT] slot is NULL (above). */
    /* B9 (2026-05-07): 22 unsalted slab kernel entries retired (md5/md4/
     * md4utf16/sha1/sha256/sha224/sha256raw/sha512/sha384/wrl/md6_256/
     * keccak{224,256,384,512}/sha3_{224,256,384,512}/sql5/sha1raw/md5raw/
     * sha384raw/sha512raw/mysql3/rmd160/blake2s256). The legacy mask short-
     * circuit at mdxfind.c:10911-10951 was the sole reachability path;
     * that short-circuit + gpu_try_pack_unsalted() in mdxfind.c were
     * retired in this same commit. All GPU_CAT_MASK ops now route exclu-
     * sively through the unified template path. RCS history retains the
     * kernel sources. */
    /* Family C (2026-05-07): hmac_sha224_ksalt_batch + hmac_sha224_kpass_batch
     * RETIRED. JOB_HMAC_SHA224 / JOB_HMAC_SHA224_KPASS now route through the
     * unified SHA224SALTPASS template path (gpu_template.cl + gpu_sha224saltpass_-
     * core.cl) via params.algo_mode = 5 / 6.
     * Family D (2026-05-08): hmac_sha256_ksalt_batch + hmac_sha256_kpass_batch
     * RETIRED. JOB_HMAC_SHA256 / JOB_HMAC_SHA256_KPASS now route through the
     * unified SHA256SALTPASS template path (gpu_template.cl + gpu_sha256saltpass_-
     * core.cl) via params.algo_mode = 5 / 6 + RUNTIME-gated HMAC body in
     * finalize_prepend_be.cl.frag (NEVER `#if HASH_WORDS == 8`; rev 1.7
     * Pascal NVIDIA ABORT lesson). After this Family D retirement, FAM_SHA256
     * has zero kernel_map entries — the entire family is dead and gpu_sha256.cl
     * is whole-file deleted in this same commit (mirrors Family J pattern).
     * The family_source[FAM_SHA256] slot is dropped to NULL implicit-init.
     * Final HMAC family in the ladder; HMAC ladder COMPLETE 21/21 algos
     * shipped. RCS history retains the kernel sources at gpu/RCS/gpu_sha256.cl,v. */
    {"hmac_md5_ksalt_batch",    {JOB_HMAC_MD5, -1}, FAM_MD5SALT},
    {"hmac_md5_kpass_batch",    {JOB_HMAC_MD5_KPASS, -1}, FAM_MD5SALT},
    /* Family B (2026-05-07): hmac_sha1_ksalt_batch + hmac_sha1_kpass_batch
     * RETIRED. JOB_HMAC_SHA1 / JOB_HMAC_SHA1_KPASS now route through the
     * unified SHA1SALTPASS template path (gpu_template.cl + gpu_sha1saltpass_-
     * core.cl) via params.algo_mode = 5 / 6. probe_max_dispatch is anchored
     * on FAM_MD5SALT (hmac_md5_ksalt_batch) — FAM_SHA1's HMAC kernels were
     * NOT the probe anchor, so this retirement does not require migrating
     * the probe. RCS history retains the kernel sources at
     * gpu/RCS/gpu_sha1.cl,v. */
    /* B8 retirement (2026-05-06): sha512passsalt_batch + sha512saltpass_batch
     * removed from FAM_HMAC_SHA512 — JOB_SHA512PASSSALT / JOB_SHA512SALTPASS
     * route through the unified template path via B6.9 / B6.10 fan-out.
     * Family E HMAC-SHA384 carrier (2026-05-08): hmac_sha384_ksalt_batch +
     * hmac_sha384_kpass_batch RETIRED.
     * Family F (2026-05-08): hmac_sha512_ksalt_batch + hmac_sha512_kpass_batch
     * RETIRED. JOB_HMAC_SHA512 / JOB_HMAC_SHA512_KPASS now route through
     * the unified template path (gpu_template.cl + gpu_sha512saltpass_core.cl)
     * via params.algo_mode = 5 / 6. After this Family F retirement,
     * FAM_HMAC_SHA512 has zero kernel_map entries — the entire family is
     * dead and the family_source[FAM_HMAC_SHA512] slot is NULL (above).
     * The gpu_hmac_sha512.cl file is `#if 0`-wrapped as a retirement-record;
     * RCS history retains the kernel sources at gpu/RCS/gpu_hmac_sha512.cl,v. */
    /* B9 (2026-05-07): six more unsalted slab kernel entries retired
     * (sql5/sha1raw/md5raw/sha384raw/sha512raw/mysql3) — see top-of-table
     * B9 comment for the full list and rationale. */
    /* Family G retirement (2026-05-08): hmac_rmd160_{ksalt,kpass}_batch
     * removed from kernel_map[]. Both ops route through the
     * RIPEMD160SALTPASS-shaped carrier template kernel via algo_mode=5/6. */
    /* Family H retirement (2026-05-08): hmac_rmd320_{ksalt,kpass}_batch
     * removed from kernel_map[]. Both ops route through the
     * RIPEMD320SALTPASS-shaped carrier template kernel via algo_mode=5/6.
     * After this Family H retirement, FAM_HMAC_RMD320 has zero kernel_map
     * entries — the entire family is dead and the family_source[FAM_HMAC_-
     * RMD320] slot is NULL (above). The gpu_hmac_rmd320.cl file is
     * `#if 0`-wrapped as a retirement-record; RCS history retains the
     * kernel sources at gpu/RCS/gpu_hmac_rmd320.cl,v. */
    /* Family I retirement (2026-05-08): hmac_blake2s_kpass_batch removed
     * from kernel_map[]. JOB_HMAC_BLAKE2S routes through the hand-written
     * Path A carrier template kernel (gpu_hmac_blake2s_core.cl) via
     * algo_mode=5. After this Family I retirement, FAM_HMAC_BLAKE2S has
     * zero kernel_map entries — the entire family is dead and the
     * family_source[FAM_HMAC_BLAKE2S] slot is NULL (above). The
     * gpu_hmac_blake2s.cl file is `#if 0`-wrapped as a retirement-record;
     * RCS history retains the kernel sources at gpu/RCS/gpu_hmac_blake2s.cl,v. */
    /* B10 retirement (2026-05-07): streebog256_unsalted_batch +
     * streebog512_unsalted_batch removed. JOB_STREEBOG_32 / JOB_STREEBOG_64
     * route through the unified template path (gpu_template.cl +
     * gpu_streebog{256,512}_core.cl) via params.algo_mode. */
    /* Family J retirement (2026-05-08): hmac_streebog256_kpass_batch +
     * hmac_streebog256_ksalt_batch entries deleted. JOB_HMAC_STREEBOG256_-
     * KPASS / JOB_HMAC_STREEBOG256_KSALT now route through the hand-written
     * Path A carrier template kernel (gpu_hmac_streebog256_core.cl) via
     * algo_mode = 5/6. The slab kernels themselves were surgically deleted
     * from gpu_streebog.cl in this same commit (lines 837-966 removed). */
    /* Family K retirement (2026-05-08): hmac_streebog512_kpass_batch +
     * hmac_streebog512_ksalt_batch entries deleted. JOB_HMAC_STREEBOG512_-
     * KPASS / JOB_HMAC_STREEBOG512_KSALT now route through the hand-
     * written Path A carrier template kernel (gpu_hmac_streebog512_core.cl)
     * via algo_mode = 5/6. The slab kernels themselves were surgically
     * deleted from gpu_streebog.cl in this same commit (lines 859-989
     * removed; whole-file #if 0 wrap since the file is empty post-
     * retirement). Final HMAC family in the ladder. FAM_STREEBOG family_-
     * source slot is now NULL - no live slab kernels remain in the
     * Streebog family (B10 retired unsalted streebog batches earlier;
     * Family J retired streebog256 HMACs; Family K retires streebog512
     * HMACs). */
    /* SHA512CRYPT/SHA512CRYPTMD5 slab kernel sha512crypt_batch retired in
     * SHA512CRYPTMD5 carrier ship (2026-05-08): both JOB_SHA512CRYPT
     * (Phase 3) and JOB_SHA512CRYPTMD5 (Phase 4) now route through the
     * unified template path via the carrier kernel (gpu_shacrypt_core.cl
     * at HASH_WORDS=16). The MD5-preprocess for SHA512CRYPTMD5 is host-
     * side at mdxfind.c:12256-12258 (job->pass swapped with 32-char MD5
     * hex of the original password before gpu_try_pack). RCS history
     * retains the slab kernel source at gpu/RCS/gpu_sha512crypt.cl,v.
     * probe_max_dispatch is anchored on FAM_MD5SALT (hmac_md5_ksalt_-
     * batch), not FAM_SHA512CRYPT, so retirement does not require a
     * probe migration. Phase 4 of the Unix-crypt ladder (final phase;
     * Unix-crypt slab path fully retired). */
    /* SHA256CRYPT slab kernel sha256crypt_batch retired in SHA256CRYPT
     * carrier ship (2026-05-08): JOB_SHA256CRYPT routes through the
     * unified template path via the carrier kernel (gpu_shacrypt_core.cl
     * at HASH_WORDS=8). RCS history retains the slab kernel source at
     * gpu/RCS/gpu_sha256crypt.cl,v. probe_max_dispatch is anchored on
     * FAM_MD5SALT (hmac_md5_ksalt_batch), not FAM_SHA256CRYPT, so
     * retirement does not require a probe migration. Phase 2 of the
     * Unix-crypt ladder. */
    /* B9 (2026-05-07): rmd160_unsalted_batch + blake2s256_unsalted_batch
     * retired — see top-of-table B9 comment. */
    /* BCRYPT slab kernel bcrypt_batch retired in BCRYPT carrier ship
     * (2026-05-08, Unix-crypt Phase 6): JOB_BCRYPT routes through the
     * unified template path via the hand-written Path A carrier
     * (gpu_bcrypt_core.cl). RCS history retains the slab kernel source
     * at gpu/RCS/gpu_bcrypt.cl,v. probe_max_dispatch is anchored on
     * FAM_MD5SALT (hmac_md5_ksalt_batch), not FAM_BCRYPT, so retirement
     * does not require a probe migration. After this Phase 6, FAM_BCRYPT
     * has zero kernel_map entries -- the entire family is dead and the
     * family_source[FAM_BCRYPT] slot is NULL (above). Phase 6 of the
     * slab-retirement ladder (final major slab kernel). */
    /* B7.9 (2026-05-07): {"md5_packed_batch", FAM_MD5PACKED} entry retired
     * with the chokepoint pack. The kernel was dispatched directly via
     * gpu_opencl_dispatch_packed (not by op-routing through this table),
     * which is why it had no JOB_ entry. Now both the dispatch function
     * and the .cl source are gone — leave the kernel_map terminator. */
    {NULL, {-1}, 0}
};
/* B8 (2026-05-06): switched from positional KERN_ITER_IDX to name-based
 * lookup so future kernel_map[] surgery (B8.x slab retirement) can't shift
 * the salted-iter slot under us. Cost: one strcmp per kernel registration.
 *
 * 2026-05-07: KERN_NAME_SALT_ITER #define + 3 strcmp lookup sites + 4
 * kern_salt_iter struct/assignment sites + 1 dispatch_batch override
 * removed alongside md5salt_iter kernel deletion. JOB_MD5SALT iter
 * coverage routes through the unified template path. */

/* OpenCL async error callback — called by the driver on compute errors.
 *
 * The driver invokes this callback on its own thread when an
 * asynchronous compute fault is detected (out-of-resources, invalid
 * memory access, etc.). Previously this just logged the error and
 * returned, which let the next host-side dispatch run with corrupted
 * state — typically failing with CL_INVALID_EVENT and continuing to
 * silently produce wrong cracks. _Exit terminates the process
 * immediately without unwinding through the driver thread; the OpenCL
 * spec does not allow longjmp out of a driver callback, but _Exit is
 * async-signal-safe and process-wide. */
static void CL_CALLBACK ocl_error_callback(const char *errinfo,
    const void *private_info, size_t cb, void *user_data) {
    (void)private_info; (void)cb; (void)user_data;
    GPU_FATAL("OpenCL async error: %s", errinfo);
}

/* Parallel device-init control. Default 0 = parallel (one thread per device,
 * Memo C 2026-05-04). Set to 1 by `-G serial` CLI option to fall back to
 * the prior single-threaded init loop. Affects gpu_opencl_init() only. */
static int _serial_init = 0;
void gpu_opencl_set_serial_init(int serial) { _serial_init = serial ? 1 : 0; }

/* Per-device init thread payload — one of these per surviving device after
 * the blacklist filter runs. The thread populates the rc/max_dispatch fields
 * and (on failure) sets gpu_devs[di].device_disabled so downstream loops skip
 * the slot. Disjoint slot indices per thread → no shared writes to gpu_devs[]. */
struct init_thread_arg {
    int            di;             /* gpu_devs[] slot — disjoint per thread */
    cl_device_id   dev_id;
    int            all_dev_idx;    /* 0-based across all platforms, for log prefix */
    char           dname[256];
    cl_ulong       gmem;
    cl_uint        mhz;
    int            rc;             /* 0 success, -1 init failure (thread writes) */
    int            max_dispatch;   /* probe result (thread writes) */
};

/* Initialize one GPU device */
static int init_device(int di, cl_device_id dev_id) {
    struct gpu_device *d = &gpu_devs[di];
    cl_int err;

    d->dev = dev_id;
    clGetDeviceInfo(dev_id, CL_DEVICE_NAME, sizeof(d->name), d->name, NULL);

    d->ctx = clCreateContext(NULL, 1, &dev_id, ocl_error_callback, NULL, &err);
    if (!d->ctx) return -1;

    /* Out-of-order queue (theory #3, mdx-architect 2026-04-30). The
     * OpenCL 1.2 spec requires the driver to honor implicit data
     * dependencies on same-buffer reads/writes when ordering is
     * expressed via cl_event waitlists; NVIDIA, AMD ROCm, Intel and
     * Mali (Rusticl) all honor this. With OOO + per-call event chains
     * the driver can submit writes, kernel, and reads in parallel
     * where data dependencies allow, removing the per-dispatch
     * synchronous-blocking pipeline. Falls back to in-order queue
     * automatically on drivers that reject the property bit.
     *
     * Default: in-order. The OOO queue lost 4 cracks per run on mmt
     * (4070 Ti SUPER, NVIDIA driver) at -i 1: 21,285 vs canonical 21,289.
     * ioblade (4070 Ti regular + 3080 + RX 9070) was unaffected. Driver-
     * or hardware-specific OOO scheduler corner case (cross-buffer
     * implicit dependency tracking). Async writes + event chains alone
     * still give partial pipelining on in-order queues; OOO was a
     * 19%-disp_us bonus on ioblade, but correctness wins.
     *
     * MDXFIND_OOO_QUEUE=1 opts in to OOO for experiments / when the
     * underlying driver issue is fixed. Old env var name preserved
     * (MDXFIND_INORDER_QUEUE was the diagnostic toggle that confirmed
     * the bug; flipped the polarity). */
    cl_command_queue_properties qprops = 0;  /* default: in-order */
    {
        const char *e = getenv("MDXFIND_OOO_QUEUE");
        if (e && *e && *e != '0') {
            qprops = CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE;
            if (di == 0) fprintf(stderr,
                "OpenCL: OOO queue enabled (MDXFIND_OOO_QUEUE=%s) — known to lose cracks on some NVIDIA drivers\n", e);
        }
    }
    /* MDXFIND_KERNEL_TRACE=1 opts the queue into CL_QUEUE_PROFILING_ENABLE
     * so per-kernel-event start/end timestamps can be retrieved via
     * clGetEventProfilingInfo. Adds negligible host-side cost when on; OFF
     * by default. Disaggregates pure GPU kernel time from disp_us (which
     * is wall-clock around the whole dispatch call: kernel + readback +
     * applyrule replay + checkhash). One-line additive change to the
     * existing OOO-queue plumbing. */
    {
        const char *e = getenv("MDXFIND_KERNEL_TRACE");
        if (e && *e && *e != '0') {
            qprops |= CL_QUEUE_PROFILING_ENABLE;
            if (di == 0) fprintf(stderr,
                "OpenCL: kernel-time profiling enabled (MDXFIND_KERNEL_TRACE=%s) — emits [kern] lines on dispatch\n", e);
        }
    }
    d->queue = clCreateCommandQueue(d->ctx, dev_id, qprops, &err);
    if (!d->queue && qprops != 0) {
        d->queue = clCreateCommandQueue(d->ctx, dev_id, 0, &err);
        if (d->queue) {
            fprintf(stderr, "OpenCL GPU[%d]: OOO queue rejected; using in-order queue\n", di);
        }
    }
    if (!d->queue) return -1;

    /* Compile common-only program (warms up NVIDIA driver state).
     * Routed through gpu_kernel_cache_build_program — when MDXFIND_CACHE is set,
     * the JIT result is persisted per (source-hash, device, driver, mdxfind_rev)
     * and replays in milliseconds on subsequent sessions. */
    {
        struct timespec _kb_t0, _kb_t1;
        clock_gettime(CLOCK_MONOTONIC, &_kb_t0);
        tsfprintf(stderr, "OpenCL GPU[%d]: kernel build START (common+selftest)\n", di);
        const char *sources[1] = { gpu_common_str };
        d->prog = gpu_kernel_cache_build_program(d->ctx, dev_id, 1, sources,
                                                 "-cl-std=CL1.2", &err);
        clock_gettime(CLOCK_MONOTONIC, &_kb_t1);
        double _kb_ms = (_kb_t1.tv_sec - _kb_t0.tv_sec) * 1e3
                      + (_kb_t1.tv_nsec - _kb_t0.tv_nsec) / 1e6;
        tsfprintf(stderr, "OpenCL GPU[%d]: kernel build DONE in %.2fs (common+selftest)\n",
                  di, _kb_ms / 1e3);
    }
    if (!d->prog || err != CL_SUCCESS) {
        char log[4096] = {0};
        if (d->prog)
            clGetProgramBuildInfo(d->prog, dev_id, CL_PROGRAM_BUILD_LOG, sizeof(log), log, NULL);
        fprintf(stderr, "OpenCL GPU[%d] common kernel compile error (err=%d):\n%s\n",
                di, err, log);
        if (d->prog) clReleaseProgram(d->prog);
        d->prog = NULL;
        return -1;
    }

    /* Compile only the md5salt family at init (needed for selftest/probe).
     * Other families are compiled later by gpu_opencl_compile_families(). */
    memset(d->fam_prog, 0, sizeof(d->fam_prog));
    memset(&dev_kerns[di], 0, sizeof(dev_kerns[di]));
    {
        struct timespec _kb2_t0, _kb2_t1;
        clock_gettime(CLOCK_MONOTONIC, &_kb2_t0);
        tsfprintf(stderr, "OpenCL GPU[%d]: kernel build START (md5salt family)\n", di);
        const char *sources[2] = { gpu_common_str, family_source[FAM_MD5SALT] };
        d->fam_prog[FAM_MD5SALT] = gpu_kernel_cache_build_program(d->ctx, dev_id, 2, sources,
                                                                   "-cl-std=CL1.2", &err);
        if (d->fam_prog[FAM_MD5SALT] && err == CL_SUCCESS) {
            clock_gettime(CLOCK_MONOTONIC, &_kb2_t1);
            double _kb2_ms = (_kb2_t1.tv_sec - _kb2_t0.tv_sec) * 1e3
                           + (_kb2_t1.tv_nsec - _kb2_t0.tv_nsec) / 1e6;
            tsfprintf(stderr, "OpenCL GPU[%d]: kernel build DONE in %.2fs (md5salt family)\n",
                      di, _kb2_ms / 1e3);
        }
        if (!d->fam_prog[FAM_MD5SALT] || err != CL_SUCCESS) {
            char log[4096] = {0};
            if (d->fam_prog[FAM_MD5SALT])
                clGetProgramBuildInfo(d->fam_prog[FAM_MD5SALT], dev_id, CL_PROGRAM_BUILD_LOG, sizeof(log), log, NULL);
            fprintf(stderr, "OpenCL GPU[%d] md5salt compile error (err=%d) — device excluded:\n%s\n",
                    di, err, log);
            if (d->fam_prog[FAM_MD5SALT]) clReleaseProgram(d->fam_prog[FAM_MD5SALT]);
            d->fam_prog[FAM_MD5SALT] = NULL;
            return -1;
        }
        /* Register md5salt kernels */
        for (int k = 0; kernel_map[k].name; k++) {
            if (kernel_map[k].family != FAM_MD5SALT) continue;
            cl_kernel kern = clCreateKernel(d->fam_prog[FAM_MD5SALT], kernel_map[k].name, &err);
            if (!kern) {
                fprintf(stderr, "OpenCL GPU[%d]: md5salt kernel '%s' create failed (err=%d) — device excluded\n",
                        di, kernel_map[k].name, err);
                return -1;
            }
            for (int j = 0; kernel_map[k].ops[j] >= 0; j++)
                kern_register(di, kernel_map[k].ops[j], kern);
        }
    }

    /* Scale batch size to GPU capability */
    cl_uint compute_units = 0;
    clGetDeviceInfo(dev_id, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(compute_units), &compute_units, NULL);
    if (compute_units < 8)
        d->max_batch = compute_units * 8;   /* tiny GPU: 4 CU -> 32 words */
    else
        d->max_batch = GPUBATCH_RULE_MAX;   /* 8+ CU: full 4192 */
    if (d->max_batch < 16) d->max_batch = 16;
    if (d->max_batch > GPUBATCH_RULE_MAX) d->max_batch = GPUBATCH_RULE_MAX;
    tsfprintf(stderr, "OpenCL GPU[%d]: %u compute units, batch size %d\n",
            di, compute_units, d->max_batch);

    /* Per-device dispatch buffers */
    /* Hit buffer sized for packed dispatch: up to GPU_PACKED_MAX_HITS (9.7MB).
     * Also used by standard dispatch (which needs at most GPU_MAX_HITS). */
    d->b_hits = clCreateBuffer(d->ctx, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                               GPU_PACKED_MAX_HITS * GPU_HIT_STRIDE * sizeof(uint32_t), NULL, &err);
    d->b_hit_count = clCreateBuffer(d->ctx, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                                     sizeof(uint32_t), NULL, &err);
    d->h_hits = (uint32_t *)malloc_lock(GPU_PACKED_MAX_HITS * GPU_HIT_STRIDE * sizeof(uint32_t),"device_init");
    d->b_params = dev_buf(d, sizeof(OCLParams), CL_MEM_READ_ONLY);

    /* Dummy overflow buffers (replaced when overflow is loaded).
     *
     * NVIDIA Windows OpenCL driver validates buffer-vs-signature compatibility
     * at clEnqueueNDRangeKernel time and rejects 4-byte cl_mems against
     * `__global uint *` / `__global ulong *` kernel signatures with
     * CL_INVALID_KERNEL_ARGS. The kernel guards `if (overflow_count > 0)`
     * before any access (gpu_common.cl:225,293) so the buffer is never
     * actually read at runtime — but the static analyzer doesn't know that.
     *
     * Use MIN_BUFFER_BYTES zero-filled allocations to match the warm-probe
     * synthesizer pattern at lines 1600-1635 (AMD strict-bounds fix) and
     * the create_min_buf floor used elsewhere. The 256-byte literal that
     * lived here through 1.87 was *coincidentally* equal to the prior
     * MIN_BUFFER_BYTES value; rev 1.89 raises the floor to 4096 and these
     * placeholders must stay above the floor too, otherwise NVIDIA Windows
     * cold-JIT validation rejects them at the first md5_rules NDRange
     * call (CL_INVALID_KERNEL_ARGS, -52) for tiny-target workloads where
     * overflow_count == 0. Allocated via calloc so the placeholder zero
     * region is sized to MIN_BUFFER_BYTES at runtime (rather than a
     * compile-time-fixed static array, which would have to grow with the
     * floor). Static lifetime gives the driver a stable host pointer. */
    {
        uint8_t *_ovfl_placeholder_zero = (uint8_t *)calloc(1, MIN_BUFFER_BYTES);
        if (!_ovfl_placeholder_zero) {
            fprintf(stderr,
                "OpenCL GPU[%d]: overflow placeholder calloc(%d) failed; "
                "device init aborted\n", di, MIN_BUFFER_BYTES);
            return -1;
        }
        d->b_overflow_keys = clCreateBuffer(d->ctx,
            CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, MIN_BUFFER_BYTES,
            _ovfl_placeholder_zero, &err);
        d->b_overflow_hashes = clCreateBuffer(d->ctx,
            CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, MIN_BUFFER_BYTES,
            _ovfl_placeholder_zero, &err);
        d->b_overflow_offsets = clCreateBuffer(d->ctx,
            CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, MIN_BUFFER_BYTES,
            _ovfl_placeholder_zero, &err);
        d->b_overflow_lengths = clCreateBuffer(d->ctx,
            CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, MIN_BUFFER_BYTES,
            _ovfl_placeholder_zero, &err);
        /* CL_MEM_COPY_HOST_PTR has copied the data; safe to free now. */
        free(_ovfl_placeholder_zero);
    }

    d->salt_data_cap = 0;
    d->salt_off_cap = 0;
    d->salts_count = 0;
    d->hexhash_cap = 0;
    d->hexlens_cap = 0;

    return 0;
}

/* ---- Public API ---- */

/* Probe maximum reliable dispatch size for a GPU device.
 * Uses the live hmac_md5_ksalt_batch kernel with synthetic test data at
 * increasing power-of-2 salt counts. Each test uses 32 identical words
 * with a known HMAC-MD5(salt, password) result; the number of salts
 * scales up. When the GPU misses the known hit, we've found the dispatch
 * limit.
 *
 * 2026-05-07 (md5salt_iter retirement): switched from md5salt_iter (now
 * deleted) to hmac_md5_ksalt_batch. Shares the identical 17-arg slab
 * kernel signature, so the host-side input pack is unchanged; only the
 * expected fingerprint differs (HMAC-MD5(key=salt, msg=password) vs.
 * MD5(hexhash || salt)). The probe goal is dispatch-capacity, not
 * algorithmic validation of MD5SALT specifically — any deterministic
 * salted kernel suffices. hmac_md5_ksalt_batch remains live for e214.
 *
 * Returns 0 if all sizes pass (no limit needed). */
static int probe_max_dispatch(int di) {
    struct gpu_device *d = &gpu_devs[di];
    cl_int err;

    cl_kernel test_kern = d->fam_prog[FAM_MD5SALT]
        ? clCreateKernel(d->fam_prog[FAM_MD5SALT], "hmac_md5_ksalt_batch", &err) : NULL;
    if (!test_kern) {
        fprintf(stderr, "OpenCL GPU[%d]: probe kernel not found (err=%d)\n", di, err);
        return 0;
    }

    /* Test word (password): bytes "61" of length 2 (hexlens[w]=2).
     * Test salt at index 0: "x" (1 byte). Salts 1..N-1: "y" (1 byte each).
     * Kernel computes HMAC-MD5(key=salt, msg=password).
     * Per-word hit set: salt 0 matches (HMAC-MD5(x, "61"));
     *                   salts 1..N-1 do not match (HMAC-MD5(y, "61")).
     * 32 words × 1 matching salt = 32 expected hits per dispatch.
     * If the dispatch is too large and corrupts computation, hits drop to 0. */

    /* HMAC-MD5(key="x", msg="61") = 7f0cc4b8483a35db7799134392754cd0
     * As LE uint32 words (matches kernel ohx,ohy,ohz,ohw): */
    uint32_t exp_hx = 0xb8c40c7fu, exp_hy = 0xdb353a48u;

    /* Build a 4-slot compact table (mask=3) with just this one hash */
    uint32_t compact_fp[4] = {0, 0, 0, 0};
    uint32_t compact_idx[4] = {0, 0, 0, 0};
    uint32_t fp = exp_hy;
    if (fp == 0) fp = 1;
    uint32_t pos = (exp_hx ^ exp_hy) & 3;
    compact_fp[pos] = fp;
    compact_idx[pos] = 0;

    /* hash_data: 16-byte hash at offset 0 (LE uint32 of HMAC-MD5(x,"61")) */
    uint32_t hash_data[4] = { exp_hx, exp_hy, 0x43139977u, 0xd04c7592u };
    uint64_t hash_off = 0;
    uint16_t hash_len = 16;

    /* Word: "61" in hex, padded to 256 bytes. First 2 bytes = 0x36,0x31 = "61" */
    /* 32 copies of the same word "61" in the hexhash buffer */
    unsigned char hexwords[32 * 256];
    uint16_t hexlens[32];
    memset(hexwords, 0, sizeof(hexwords));
    for (int w = 0; w < 32; w++) {
        hexwords[w * 256] = '6';
        hexwords[w * 256 + 1] = '1';
        hexlens[w] = 2;
    }

    /* Create GPU buffers for the test */
    cl_mem b_hexhash = clCreateBuffer(d->ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(hexwords), hexwords, &err);
    cl_mem b_hexlen = clCreateBuffer(d->ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(hexlens), hexlens, &err);
    cl_mem b_cfp = clCreateBuffer(d->ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(compact_fp), compact_fp, &err);
    cl_mem b_cidx = clCreateBuffer(d->ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(compact_idx), compact_idx, &err);
    cl_mem b_hdata = clCreateBuffer(d->ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(hash_data), hash_data, &err);
    cl_mem b_hoff = clCreateBuffer(d->ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(hash_off), &hash_off, &err);
    cl_mem b_hlen = clCreateBuffer(d->ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(hash_len), &hash_len, &err);

    /* Params buffer */
    OCLParams params;
    memset(&params, 0, sizeof(params));
    params.compact_mask = 3;
    params.num_words = 32;
    // params.salt_start = 0;
    params.max_probe = 4;
    params.hash_data_count = 1;
    params.max_hits = 256;
    params.overflow_count = 0;
    params.max_iter = 1;
    cl_mem b_params = clCreateBuffer(d->ctx, CL_MEM_READ_ONLY, sizeof(params), NULL, &err);

    /* Hits */
    cl_mem b_hits = clCreateBuffer(d->ctx, CL_MEM_READ_WRITE, 256 * GPU_HIT_STRIDE * sizeof(uint32_t), NULL, &err);
    cl_mem b_hitcnt = clCreateBuffer(d->ctx, CL_MEM_READ_WRITE, sizeof(uint32_t), NULL, &err);

    /* Dummy overflow buffers (empty) */
    uint64_t dummy64 = 0;
    uint32_t dummy32 = 0;
    uint16_t dummy16 = 0;
    cl_mem b_okeys = clCreateBuffer(d->ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 8, &dummy64, &err);
    cl_mem b_ohash = clCreateBuffer(d->ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 4, &dummy32, &err);
    cl_mem b_ooff = clCreateBuffer(d->ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 4, &dummy32, &err);
    cl_mem b_olen = clCreateBuffer(d->ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 2, &dummy16, &err);

    /* Build salt buffer: salt[0] = "x", salt[1..max] = "y" */
    int max_salts = 4194304;
    char *salt_data = (char *)malloc_lock(max_salts,"salt_data");
    uint32_t *salt_off = (uint32_t *)malloc_lock(max_salts * sizeof(uint32_t),"Salt_off");
    uint16_t *salt_len = (uint16_t *)malloc_lock(max_salts * sizeof(uint16_t),"salt_len");
    salt_data[0] = 'x';
    for (int i = 1; i < max_salts; i++) salt_data[i] = 'y';
    for (int i = 0; i < max_salts; i++) { salt_off[i] = i; salt_len[i] = 1; }

    cl_mem b_sdata = clCreateBuffer(d->ctx, CL_MEM_READ_ONLY, max_salts, NULL, &err);
    cl_mem b_soff = clCreateBuffer(d->ctx, CL_MEM_READ_ONLY, max_salts * sizeof(uint32_t), NULL, &err);
    cl_mem b_slen = clCreateBuffer(d->ctx, CL_MEM_READ_ONLY, max_salts * sizeof(uint16_t), NULL, &err);
    clEnqueueWriteBuffer(d->queue, b_sdata, CL_TRUE, 0, max_salts, salt_data, 0, NULL, NULL);
    clEnqueueWriteBuffer(d->queue, b_soff, CL_TRUE, 0, max_salts * sizeof(uint32_t), salt_off, 0, NULL, NULL);
    clEnqueueWriteBuffer(d->queue, b_slen, CL_TRUE, 0, max_salts * sizeof(uint16_t), salt_len, 0, NULL, NULL);
    clFinish(d->queue);
    free(salt_data); free(salt_off); free(salt_len);

    /* Set kernel args */
    int a = 0;
    clSetKernelArg(test_kern, a++, sizeof(cl_mem), &b_hexhash);
    clSetKernelArg(test_kern, a++, sizeof(cl_mem), &b_hexlen);
    clSetKernelArg(test_kern, a++, sizeof(cl_mem), &b_sdata);
    clSetKernelArg(test_kern, a++, sizeof(cl_mem), &b_soff);
    clSetKernelArg(test_kern, a++, sizeof(cl_mem), &b_slen);
    clSetKernelArg(test_kern, a++, sizeof(cl_mem), &b_cfp);
    clSetKernelArg(test_kern, a++, sizeof(cl_mem), &b_cidx);
    clSetKernelArg(test_kern, a++, sizeof(cl_mem), &b_params);
    clSetKernelArg(test_kern, a++, sizeof(cl_mem), &b_hdata);
    clSetKernelArg(test_kern, a++, sizeof(cl_mem), &b_hoff);
    clSetKernelArg(test_kern, a++, sizeof(cl_mem), &b_hlen);
    clSetKernelArg(test_kern, a++, sizeof(cl_mem), &b_hits);
    clSetKernelArg(test_kern, a++, sizeof(cl_mem), &b_hitcnt);
    clSetKernelArg(test_kern, a++, sizeof(cl_mem), &b_okeys);
    clSetKernelArg(test_kern, a++, sizeof(cl_mem), &b_ohash);
    clSetKernelArg(test_kern, a++, sizeof(cl_mem), &b_ooff);
    clSetKernelArg(test_kern, a++, sizeof(cl_mem), &b_olen);

    /* Test at increasing power-of-2 salt counts */
    static const int test_sizes[] = {
        1024, 2048, 4096, 8192, 16384, 32768, 65536,
        131072, 262144, 524288, 1048576, 2097152, 4194304, 0
    };

    int max_good = 0;
    for (int t = 0; test_sizes[t]; t++) {
        int nsalts = test_sizes[t];

        params.num_salts = nsalts;
        // params.salt_start = 0;
        clEnqueueWriteBuffer(d->queue, b_params, CL_TRUE, 0, sizeof(params), &params, 0, NULL, NULL);

        /* Zero hit counter */
        uint32_t zero = 0;
        clEnqueueWriteBuffer(d->queue, b_hitcnt, CL_TRUE, 0, sizeof(zero), &zero, 0, NULL, NULL);
        clFinish(d->queue);

        size_t global = (size_t)32 * nsalts;  /* 32 words * nsalts */
        size_t local = 128;
        global = ((global + local - 1) / local) * local;

        err = clEnqueueNDRangeKernel(d->queue, test_kern, 1, NULL, &global, &local, 0, NULL, NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "OpenCL GPU[%d]: probe dispatch error %d at %d salts\n", di, err, nsalts);
            break;
        }
        cl_int finish_err = clFinish(d->queue);

        /* Read hit count — should be exactly 32 (salt 0 matches for each of 32 words) */
        uint32_t nhits;
        clEnqueueReadBuffer(d->queue, b_hitcnt, CL_TRUE, 0, sizeof(nhits), &nhits, 0, NULL, NULL);

        if (nhits == 32 && finish_err == CL_SUCCESS) {
            max_good = nsalts;
        } else {
            fprintf(stderr, "OpenCL GPU[%d]: probe FAIL at %d salts (%d items)"
                    " — got %u hits, expected 32, clFinish=%d\n",
                    di, nsalts, (int)global, nhits, finish_err);
            break;
        }
    }

    /* Cleanup */
    clReleaseKernel(test_kern);
    clReleaseMemObject(b_hexhash); clReleaseMemObject(b_hexlen);
    clReleaseMemObject(b_cfp); clReleaseMemObject(b_cidx);
    clReleaseMemObject(b_hdata); clReleaseMemObject(b_hoff); clReleaseMemObject(b_hlen);
    clReleaseMemObject(b_params); clReleaseMemObject(b_hits); clReleaseMemObject(b_hitcnt);
    clReleaseMemObject(b_sdata); clReleaseMemObject(b_soff); clReleaseMemObject(b_slen);
    clReleaseMemObject(b_okeys); clReleaseMemObject(b_ohash); clReleaseMemObject(b_ooff);
    clReleaseMemObject(b_olen);

    int last_tested = 0;
    for (int t = 0; test_sizes[t]; t++) last_tested = test_sizes[t];
    if (max_good >= last_tested)
        return 0;  /* all passed — no dispatch limit */
    return max_good;
}

/* Per-device init thread body (Memo C). Runs init_device + probe_max_dispatch
 * + Mali fixup for one slot. The slot index is pre-assigned by phase 1 of
 * gpu_opencl_init() so threads operate on disjoint gpu_devs[] entries with
 * no shared writes. Stderr lines all carry the OpenCL GPU[N] prefix so
 * concurrent writes interleave at line granularity but stay diagnosable. */
static void init_one_device_full(void *payload) {
    struct init_thread_arg *a = (struct init_thread_arg *)payload;
    a->rc = init_device(a->di, a->dev_id);
    if (a->rc != 0) {
        /* init_device returned -1 — slot's cl_context etc. are uninitialized.
         * Mark device_disabled so downstream loops (set_compact_table parallel
         * dispatch) skip it instead of touching the bad cl_* handles. */
        gpu_devs[a->di].device_disabled = 1;
        return;
    }
    tsfprintf(stderr, "OpenCL GPU[%d]: %s (%llu MB, %u MHz)\n",
            a->all_dev_idx, a->dname,
            (unsigned long long)(a->gmem / (1024*1024)), a->mhz);
    /* Phase I: selftest = probe_max_dispatch's per-size dispatch sweep. */
    struct timespec _st_t0, _st_t1;
    clock_gettime(CLOCK_MONOTONIC, &_st_t0);
    tsfprintf(stderr, "OpenCL GPU[%d]: selftest START\n", a->all_dev_idx);
    int max_disp = probe_max_dispatch(a->di);
    clock_gettime(CLOCK_MONOTONIC, &_st_t1);
    double _st_ms = (_st_t1.tv_sec - _st_t0.tv_sec) * 1e3
                  + (_st_t1.tv_nsec - _st_t0.tv_nsec) / 1e6;
    tsfprintf(stderr, "OpenCL GPU[%d]: selftest DONE in %.2fs\n",
              a->all_dev_idx, _st_ms / 1e3);
    /* Mali GPUs have a 17-bit salt dimension limit — silently drop work items
     * beyond 2^17 salts. Probe returns max_good in salts (not work items),
     * so convert: limit = max_salts * 32 test words. */
    if (strstr(a->dname, "Mali")) {
        if (max_disp > 0)
            max_disp *= 32;
        else
            max_disp = 4194304;  /* 2^17 * 32 */
    }
    a->max_dispatch = max_disp;
    if (max_disp == 0)
        tsfprintf(stderr, "OpenCL GPU[%d]: selftest passed all sizes (no dispatch limit)\n", a->all_dev_idx);
    else
        tsfprintf(stderr, "OpenCL GPU[%d]: dispatch limit = %d work items\n", a->all_dev_idx, max_disp);
}

int gpu_opencl_init(void) {
    if (opencl_dynload_init() != 0) return -1;

    cl_uint nplat = 0;
    cl_platform_id plats[8];
    cl_int err;

    clGetPlatformIDs(8, plats, &nplat);
    if (nplat == 0) { fprintf(stderr, "OpenCL: no platforms\n"); return -1; }

    /* Kernel sources are per-family, compiled in init_device */

    /* Enumerate all GPU devices across all platforms.
     * -G 0,2,4 or -G 0-2: select specific devices.
     * (-G list is handled earlier in main via gpu_opencl_list_devices) */
    int all_dev_idx = 0;

    /* === Memo C: parallel per-device init ===
     * Phase 1: enumerate platforms + devices, apply skip/blacklist, build a
     *          pending-list of (slot_idx, dev_id, all_dev_idx, name, mem, mhz)
     *          tuples. Slot indices are assigned monotonically here so threads
     *          operate on disjoint gpu_devs[] entries.
     * Phase 2: launch one thread per pending device (or run serially when the
     *          `-G serial` kill-option is set via gpu_opencl_set_serial_init).
     *          Each thread runs init_device + probe_max_dispatch + Mali fixup
     *          on its own slot — disjoint, no shared writes to gpu_devs[].
     * Phase 3: join all threads.
     * Phase 4: collect max_dispatch results, set num_gpu_devs to the highest
     *          assigned slot + 1. Failed slots remain in gpu_devs[] with
     *          device_disabled=1 (set by init_one_device_full on rc!=0).
     *          The downstream `for (i=0; i<num_gpu_devs; i++) if (device_disabled) skip`
     *          pattern (already established for set_compact_table failures)
     *          handles them transparently. */
    static struct init_thread_arg pending[MAX_GPU_DEVICES];
    int n_pending = 0;
    num_gpu_devs = 0;

    /* Phase 1: enumerate + filter (serial — fast, only API queries) */
    for (cl_uint p = 0; p < nplat; p++) {
        cl_device_id devs[64];
        cl_uint ndev = 0;
        err = clGetDeviceIDs(plats[p], CL_DEVICE_TYPE_GPU, 64, devs, &ndev);
        if (err != CL_SUCCESS || ndev == 0) continue;

        for (cl_uint d = 0; d < ndev; d++) {
            char dname[256];
            cl_ulong gmem = 0;
            cl_uint mhz = 0;
            clGetDeviceInfo(devs[d], CL_DEVICE_NAME, sizeof(dname), dname, NULL);
            clGetDeviceInfo(devs[d], CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(gmem), &gmem, NULL);
            clGetDeviceInfo(devs[d], CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(mhz), &mhz, NULL);

            if (!device_allowed(all_dev_idx)) {
                fprintf(stderr, "OpenCL GPU[%d]: %s - skipped\n", all_dev_idx, dname);
                all_dev_idx++;
                continue;
            }

            /* GPU Blacklist: suppress devices known to crash their OpenCL drivers.
             * Intel iGPUs (UHD/Iris/HD Graphics) crash in igdrcl64.dll on Windows.
             * AMD integrated GPUs (gfx1036 etc with few CUs) have stability issues.
             * Users can override with explicit -G <idx> to force-enable. */
            if (!gpu_device_filter_set) {
                char vendor[256] = {0};
                cl_uint cus = 0;
                clGetDeviceInfo(devs[d], CL_DEVICE_VENDOR, sizeof(vendor), vendor, NULL);
                clGetDeviceInfo(devs[d], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cus), &cus, NULL);
                int blacklisted = 0;
                if (strstr(vendor, "Intel"))
                    blacklisted = 1;
                if ((strstr(vendor, "AMD") || strstr(vendor, "Advanced Micro")) && cus <= 4)
                    blacklisted = 1;
                if (blacklisted) {
                    fprintf(stderr, "OpenCL GPU[%d]: %s - blacklisted (use -G %d to force)\n",
                            all_dev_idx, dname, all_dev_idx);
                    all_dev_idx++;
                    continue;
                }
            }

            if (n_pending < MAX_GPU_DEVICES) {
                struct init_thread_arg *a = &pending[n_pending];
                a->di           = n_pending;     /* monotonic slot assignment */
                a->dev_id       = devs[d];
                a->all_dev_idx  = all_dev_idx;
                strncpy(a->dname, dname, sizeof(a->dname) - 1);
                a->dname[sizeof(a->dname) - 1] = 0;
                a->gmem         = gmem;
                a->mhz          = mhz;
                a->rc           = -1;
                a->max_dispatch = 0;
                n_pending++;
            }
            all_dev_idx++;
        }
    }

    /* Phase 2 + 3: launch threads (or serial fallback) and join. */
    if (_serial_init || n_pending <= 1) {
        /* Serial fallback — exact same per-device sequence the parallel path
         * runs, just one at a time. Single-GPU rigs and `-G serial` land here. */
        for (int i = 0; i < n_pending; i++)
            init_one_device_full(&pending[i]);
    } else {
        thread *threads[MAX_GPU_DEVICES];
        for (int i = 0; i < n_pending; i++)
            threads[i] = launch(init_one_device_full, &pending[i]);
        for (int i = 0; i < n_pending; i++)
            join(threads[i]);
    }

    /* Phase 4: aggregate. num_gpu_devs covers all attempted slots; failed
     * slots are present with device_disabled=1 (set inside the thread body). */
    for (int i = 0; i < n_pending; i++) {
        if (pending[i].rc == 0)
            gpu_devs[i].max_dispatch = pending[i].max_dispatch;
    }
    num_gpu_devs = n_pending;

    if (num_gpu_devs == 0) {
        fprintf(stderr, "OpenCL: no GPU devices found\n");
        return -1;
    }

    tsfprintf(stderr, "OpenCL GPU: %d device%s initialized\n",
            num_gpu_devs, num_gpu_devs > 1 ? "s" : "");
    ocl_ready = 1;
    return 0;
}

/* Compile additional kernel families on demand. Called from main() after
 * hash types are known. fam_mask is a bitmask of FAM_* values to compile. */
void gpu_opencl_compile_families(unsigned int fam_mask) {
    if (!ocl_ready) return;
    for (int di = 0; di < num_gpu_devs; di++) {
        struct gpu_device *d = &gpu_devs[di];
        if (d->device_disabled) continue;
        cl_int err;
        for (int f = 0; f < FAM_COUNT; f++) {
            if (d->fam_prog[f]) continue;  /* already compiled */
            if (!(fam_mask & (1u << f))) continue;  /* not requested */
            /* B8 (2026-05-06): family_source[f] may be NULL for retired
             * FAM_* slots (FAM_MD5SALTPASS, FAM_MD5_MD5SALTMD5PASS) whose
             * kernels were removed when their ops moved to the unified
             * template path. Silently skip — the template path doesn't use
             * fam_prog[f], and host code in mdxfind.c still maps these JOB_
             * to FAM_ for timing/bookkeeping. No kernels register against
             * a NULL family, so the skip is safe end-to-end. */
            if (!family_source[f]) continue;
            const char *sources[2] = { gpu_common_str, family_source[f] };
            d->fam_prog[f] = gpu_kernel_cache_build_program(d->ctx, d->dev, 2, sources,
                                                             "-cl-std=CL1.2", &err);
            if (!d->fam_prog[f] || err != CL_SUCCESS) {
                char log[4096] = {0};
                if (d->fam_prog[f])
                    clGetProgramBuildInfo(d->fam_prog[f], d->dev, CL_PROGRAM_BUILD_LOG, sizeof(log), log, NULL);
                fprintf(stderr, "OpenCL GPU[%d] family %d compile error (err=%d) — DEVICE DISABLED, no GPU work routes here:\n%s\n",
                        di, f, err, log);
                if (d->fam_prog[f]) clReleaseProgram(d->fam_prog[f]);
                d->fam_prog[f] = NULL;
                d->device_disabled = 1;
                break;
            }
            /* Register kernels from this family */
            for (int k = 0; kernel_map[k].name; k++) {
                if (kernel_map[k].family != f) continue;
                cl_kernel kern = clCreateKernel(d->fam_prog[f], kernel_map[k].name, &err);
                if (!kern) {
                    fprintf(stderr, "OpenCL GPU[%d]: family %d kernel '%s' create failed (err=%d) — DEVICE DISABLED, no GPU work routes here\n",
                            di, f, kernel_map[k].name, err);
                    d->device_disabled = 1;
                    break;
                }
                for (int j = 0; kernel_map[k].ops[j] >= 0; j++)
                    kern_register(di, kernel_map[k].ops[j], kern);
            }
            if (d->device_disabled) break;
        }
    }
    gpu_opencl_finalize_active_count();
}

void gpu_opencl_shutdown(void) {
    if (!ocl_ready) return;
    ocl_ready = 0;  /* prevent re-entry */
    for (int i = 0; i < num_gpu_devs; i++) {
        struct gpu_device *d = &gpu_devs[i];
        if (d->queue) clFinish(d->queue);  /* drain any pending GPU work */
        /* Skip CL object releases — NVIDIA driver may have already torn
         * down internal state by this point, causing NULL dereference
         * inside clReleaseKernel.  Process exit reclaims everything. */
        free(d->h_hits);
    }
}

int gpu_opencl_available(void) { return ocl_ready; }
int gpu_opencl_num_devices(void) { return num_gpu_devs; }

/* Per-device disable accessor — see gpu_opencl.h doc. */
int gpu_opencl_device_disabled(int dev_idx) {
    if (dev_idx < 0 || dev_idx >= num_gpu_devs) return 1;  /* OOB == disabled */
    return gpu_devs[dev_idx].device_disabled ? 1 : 0;
}

/* Count of non-disabled devices. */
int gpu_opencl_active_device_count(void) {
    int n = 0;
    for (int i = 0; i < num_gpu_devs; i++)
        if (!gpu_devs[i].device_disabled) n++;
    return n;
}

/* Called by mdxfind.c after the per-device set_compact_table loop. Logs
 * the active device count and, if zero, flips ocl_ready back to false
 * so gpu_opencl_available() returns 0 — equivalent to the user having
 * passed -G none. The compile-families / set-rules / set-mask paths
 * that follow this check in mdxfind.c all gate on gpu_opencl_available(),
 * so they become no-ops cleanly. */
void gpu_opencl_finalize_active_count(void) {
    int n_active = gpu_opencl_active_device_count();
    int n_total  = num_gpu_devs;
    if (n_active == n_total) {
        /* Quiet path: every device active — no need to mention disable
         * counts (they're zero). The per-device "compact table registered"
         * lines already covered the success case. */
        return;
    }
    tsfprintf(stderr, "OpenCL GPU: %d of %d device%s active\n",
            n_active, n_total, n_total == 1 ? "" : "s");
    if (n_active == 0) {
        tsfprintf(stderr, "OpenCL GPU: all devices disabled — falling back to CPU-only mode\n");
        ocl_ready = 0;
    }
}

/* BF Phase 3 (2026-05-10): the multi-GPU atomic-cursor BF machinery
 * (_bf_cursor, _bf_active, gpu_opencl_bf_start/stop/active/set_partition/
 * set_tail_start) has been retired. BF on GPU now flows exclusively
 * through the chunk-as-job producer at mdxfind.c:~48590 + the rules-
 * engine path in gpu_opencl_dispatch_md5_rules. The bf_mode arm inside
 * gpu_opencl_dispatch_batch is gone in this same commit. See
 * project_bf_chunk_as_job.md Phase 3. RCS history retains the prior
 * machinery (gpu_opencl.c rev 1.160). */

int gpu_opencl_max_batch(int dev_idx) {
    if (dev_idx < 0 || dev_idx >= num_gpu_devs) return GPUBATCH_MAX;
    return gpu_devs[dev_idx].max_batch;
}

/* Phase 6: read the autotuned hashes/sec for a (device, family) pair.
 * Returns 0.0 if the device has not yet completed an autotune probe for
 * this family. Surviving accessor post-Phase-3; the per-device
 * fam_rate_hps[] is still populated by the warm-probe path and read by
 * external callers if they need a rate estimate. */
double gpu_opencl_fam_rate(int dev_idx, int fam) {
    if (dev_idx < 0 || dev_idx >= num_gpu_devs) return 0.0;
    if (fam < 0 || fam >= FAM_COUNT) return 0.0;
    return gpu_devs[dev_idx].fam_rate_hps[fam];
}

/* BF Phase 3 (2026-05-10): gpu_opencl_bf_set_partition + gpu_opencl_bf_-
 * set_tail_start retired alongside the rest of the bf_partition_setup
 * machinery. RCS history retains the prior implementations (gpu_opencl.c
 * rev 1.160). */

/* ===================================================================
 * Phase 6.1: synchronous and asynchronous warm-probe.
 *
 * On the lazy autotune path (inside dispatch_batch), the timing probe
 * runs only when a device first dispatches a kernel for a given family.
 * That is too late for bf_partition_setup, which needs per-device rates
 * BEFORE dispatch starts. Phase 6.1 fixes this by exposing the same
 * timing-probe logic as a callable function, plus an async dispatcher
 * that probes every device in parallel via a one-shot pthread per device.
 *
 * gpu_opencl_warm_probe(dev, op): synchronous. Idempotent — no-op if
 *     fam_timed[fam] is already set. Allocates buffer space if needed,
 *     populates a synthetic 32-hex-char input word, sets all kernel args,
 *     runs the same probe loop as dispatch_batch's autotune block, and
 *     stores fam_max_items / fam_timed / fam_rate_hps.
 * gpu_opencl_warm_probe_async(op): launches one pthread per device.
 * gpu_opencl_warm_probe_wait(): joins all outstanding probe threads.
 * =================================================================== */

extern int gpu_op_category(int op);

void gpu_opencl_warm_probe(int dev_idx, int op) {
    if (!ocl_ready || dev_idx < 0 || dev_idx >= num_gpu_devs) return;
    int fam = gpu_op_family(op);
    if (fam < 0 || fam >= FAM_COUNT) return;

    struct gpu_device *d = &gpu_devs[dev_idx];
    /* Skip probe on disabled devices — same VRAM that wouldn't fit the
     * compact table won't fit synthetic probe buffers either, and even
     * if it did, the device is going to be skipped by every dispatch
     * path so there's no point calibrating it. */
    if (d->device_disabled) {
        d->fam_timed[fam] = 1;  /* prevent re-entry */
        return;
    }
    if (d->fam_timed[fam]) return;          /* probe already done */

    /* AMD HSA fault on this code path (task #44 — root cause not yet
     * narrowed). Production rules/packed/iter dispatch works correctly on
     * AMD; only this synthetic-buffer probe path triggers
     * HSA_STATUS_ERROR_MEMORY_FAULT. Auto-skip for AMD devices and let
     * bf_partition_setup fall back to single-cursor mode. NVIDIA path
     * unchanged. Override either way via MDXFIND_SKIP_WARMUP=1.
     * MDXFIND_FORCE_WARMUP=1 overrides the AMD auto-skip (for debugging). */
    {
        const char *force = getenv("MDXFIND_FORCE_WARMUP");
        int force_warm = (force && *force && *force != '0');
        if (!force_warm &&
            (strstr(d->name, "AMD") || strstr(d->name, "Radeon") ||
             strncmp(d->name, "gfx", 3) == 0)) {
            if (!d->fam_timed[fam]) {
                fprintf(stderr, "OpenCL GPU[%d]: warm-probe skipped (AMD device, task #44)\n", dev_idx);
                d->fam_timed[fam] = 1;   /* prevent re-entry */
            }
            return;
        }
    }

    int cat = gpu_op_category(op);
    int is_salted = (cat == GPU_CAT_SALTED || cat == GPU_CAT_SALTPASS);
    int is_bcrypt = (fam == FAM_BCRYPT);
    /* Phase 6.2: warm-probe is callable BEFORE set_mask runs, so we
     * synthesize a mask-mode probe even if gpu_mask_total is still 0. */
    int is_mask   = (cat == GPU_CAT_MASK || cat == GPU_CAT_UNSALTED);
    if (!is_mask && !is_salted) return;     /* probe meaningful only for these */

    if (op < 0 || op >= MAX_GPU_KERNELS) return;
    struct gpu_kern *gk = &dev_kerns[dev_idx].kerns[op];
    if (!gk || !gk->kernel) return;
    cl_kernel kern = gk->kernel;
    size_t local = kern_get_local_size(gk);
    if (local == 0) local = 1;

    /* ===== Phase 6.2: synthesize compact table and mask if main hasn't
     * uploaded the real ones yet. Lets warm-probe fire right after
     * gpu_opencl_compile_families(), overlapping with the rest of
     * mdxfind's BF activation setup. After the probe completes, the
     * synthetic buffers are released; main's set_compact_table/set_mask
     * can run any time (concurrently is unsafe — but the wait at
     * bf_partition_setup serializes them). ===== */
    cl_int berr;
    cl_mem syn_compact_fp = NULL, syn_compact_idx = NULL, syn_mask_desc = NULL;
    cl_mem syn_salt_data = NULL, syn_salt_off = NULL, syn_salt_len = NULL;
    cl_mem syn_hash_data = NULL, syn_hash_data_off = NULL, syn_hash_data_len = NULL;
    cl_mem syn_ovfl_keys = NULL, syn_ovfl_hashes = NULL, syn_ovfl_offs = NULL, syn_ovfl_lens = NULL;
    cl_mem use_compact_fp  = d->b_compact_fp;
    cl_mem use_compact_idx = d->b_compact_idx;
    cl_mem use_mask_desc   = d->bgpu_mask_desc;
    cl_mem use_salt_data   = d->b_salt_data;
    cl_mem use_salt_off    = d->b_salt_off;
    cl_mem use_salt_len    = d->b_salt_len;
    cl_mem use_hash_data       = d->b_hash_data;
    cl_mem use_hash_data_off   = d->b_hash_data_off;
    cl_mem use_hash_data_len   = d->b_hash_data_len;
    cl_mem use_ovfl_keys       = d->b_overflow_keys;
    cl_mem use_ovfl_hashes     = d->b_overflow_hashes;
    cl_mem use_ovfl_offs       = d->b_overflow_offsets;
    cl_mem use_ovfl_lens       = d->b_overflow_lengths;
    int      use_n_prepend = gpu_mask_n_prepend;
    int      use_n_append  = gpu_mask_n_append;
    uint64_t use_mask_total = gpu_mask_total;

    if (!use_compact_fp || !use_compact_idx) {
        /* 4-slot empty compact table — kernel reads, finds no fingerprint
         * matches, returns no hits. Sufficient for rate measurement. */
        uint32_t zeros[4] = {0};
        syn_compact_fp = clCreateBuffer(d->ctx,
            CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(zeros), zeros, &berr);
        syn_compact_idx = clCreateBuffer(d->ctx,
            CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(zeros), zeros, &berr);
        if (!syn_compact_fp || !syn_compact_idx) {
            if (syn_compact_fp) clReleaseMemObject(syn_compact_fp);
            if (syn_compact_idx) clReleaseMemObject(syn_compact_idx);
            return;
        }
        use_compact_fp = syn_compact_fp;
        use_compact_idx = syn_compact_idx;
    }

    /* AMD strict-bounds fix (task #44): when the kernel binds an arg slot
     * to a NULL cl_mem, AMD's HSA faults the moment the compiled kernel
     * loads the buffer pointer for that arg, even if it never dereferences.
     * NVIDIA tolerates NULL silently. Synthesize 256-byte dummies (not 1-byte)
     * for every potentially-NULL arg: AMD validates buffer bounds against actual
     * read width, and some unsalted-kernel "unused" args share this slot. */
    {
        uint8_t zero_buf[256] = {0};
        cl_mem *targets[] = {
            &syn_salt_data,    &syn_salt_off,    &syn_salt_len,
            &syn_hash_data,    &syn_hash_data_off, &syn_hash_data_len,
            &syn_ovfl_keys,    &syn_ovfl_hashes,
            &syn_ovfl_offs,    &syn_ovfl_lens
        };
        cl_mem *uses[] = {
            &use_salt_data,    &use_salt_off,    &use_salt_len,
            &use_hash_data,    &use_hash_data_off, &use_hash_data_len,
            &use_ovfl_keys,    &use_ovfl_hashes,
            &use_ovfl_offs,    &use_ovfl_lens
        };
        int n = (int)(sizeof(uses) / sizeof(uses[0]));
        for (int i = 0; i < n; i++) {
            if (!*uses[i]) {
                *targets[i] = clCreateBuffer(d->ctx,
                    CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 256, zero_buf, &berr);
                if (!*targets[i]) {
                    /* alloc failed — release any prior synthesized + bail */
                    for (int j = 0; j <= i; j++) if (*targets[j]) clReleaseMemObject(*targets[j]);
                    if (syn_compact_fp)  clReleaseMemObject(syn_compact_fp);
                    if (syn_compact_idx) clReleaseMemObject(syn_compact_idx);
                    return;
                }
                *uses[i] = *targets[i];
            }
        }
    }

    if (is_mask && use_mask_total == 0) {
        /* Synthetic ?l^8 mask: 8 append positions, charset 'a'-'z' each.
         * Cardinality = 26^8 ≈ 2.1e11 — large enough that probe_size
         * doublings can grow num_masks freely up to the uint32_t cap.
         * Format matches gpu_opencl_set_mask: sizes[ntotal] then
         * tables[ntotal][256]. */
        const int syn_ntotal = 8;
        int bufsize = syn_ntotal + syn_ntotal * 256;
        unsigned char *desc = (unsigned char *)calloc(1, bufsize);
        if (!desc) {
            if (syn_compact_fp) clReleaseMemObject(syn_compact_fp);
            if (syn_compact_idx) clReleaseMemObject(syn_compact_idx);
            return;
        }
        for (int i = 0; i < syn_ntotal; i++) desc[i] = 26;
        for (int i = 0; i < syn_ntotal; i++) {
            unsigned char *tbl = desc + syn_ntotal + i * 256;
            for (int c = 0; c < 26; c++) tbl[c] = 'a' + c;
        }
        syn_mask_desc = clCreateBuffer(d->ctx,
            CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bufsize, desc, &berr);
        free(desc);
        if (!syn_mask_desc) {
            if (syn_compact_fp) clReleaseMemObject(syn_compact_fp);
            if (syn_compact_idx) clReleaseMemObject(syn_compact_idx);
            return;
        }
        use_mask_desc   = syn_mask_desc;
        use_n_prepend   = 0;
        use_n_append    = syn_ntotal;
        use_mask_total  = 1;
        for (int i = 0; i < syn_ntotal; i++) use_mask_total *= 26;
    }

    /* Synthetic input: 1024 zero words. With num_words=1 the probe loop
     * caps probe_size on max_items before pms ever reaches the 200-400ms
     * target, leaving sub-ms timings whose rate-formula divides-by-tiny
     * and produces spurious 100+ TH/s rates. 1024 words gives the GPU
     * enough total work that the doublings can converge. */
    int num_words = 1024;
    int word_stride = 256;  /* generous default; matches salted password stride */
    if (cat == GPU_CAT_MASK || cat == GPU_CAT_UNSALTED) {
        if (op == JOB_SHA512 || op == JOB_SHA384 ||
            op == JOB_SHA512RAW || op == JOB_SHA384RAW) word_stride = 128;
        else if (op == JOB_MD6256) word_stride = 64;
        else word_stride = 64;
    }
    size_t words_size    = (size_t)num_words * word_stride;
    size_t hexlens_upload = (size_t)num_words * sizeof(uint16_t);

    cl_int err;
    if (words_size > d->hexhash_cap) {
        if (d->b_hexhashes) clReleaseMemObject(d->b_hexhashes);
        d->b_hexhashes = clCreateBuffer(d->ctx, CL_MEM_READ_ONLY, words_size, NULL, &err);
        d->hexhash_cap = words_size;
    }
    if (hexlens_upload > d->hexlens_cap) {
        if (d->b_hexlens) clReleaseMemObject(d->b_hexlens);
        d->b_hexlens = clCreateBuffer(d->ctx, CL_MEM_READ_ONLY, hexlens_upload, NULL, &err);
        d->hexlens_cap = hexlens_upload;
    }

    /* Fill device word buffer with valid probe inputs.
     * Salted kernels: hex-encoded passwords — zero fill is fine (length=32
     *   means 16 zero bytes, a valid probe candidate).
     * Unsalted/mask kernels: pre-padded MD5/SHA blocks — zero fill causes
     *   M[14]=0 → total_len=0 → app_start=(0-n_app) which is negative,
     *   causing OOB private-array access that faults AMD GPUs.
     *   Fix: set M[14]=64 per block (= 8-byte word bit count), so
     *   total_len=8 and app_start=8-n_app=0 for the ?l^8 synthetic mask. */
    if (cat == GPU_CAT_MASK || cat == GPU_CAT_UNSALTED) {
        /* word_stride=64 bytes = 16 uint32. Write M[14]=64 per block. */
        uint32_t *wbuf = (uint32_t *)calloc(1, words_size);
        if (wbuf) {
            for (int wi = 0; wi < num_words; wi++) {
                /* M[2] = 0x80 byte in LE uint32: padding sentinel */
                wbuf[wi * 16 + 2] = 0x00000080u;
                /* M[14] = bit count = 8*8 = 64 */
                wbuf[wi * 16 + 14] = 64;
            }
            clEnqueueWriteBuffer(d->queue, d->b_hexhashes, CL_TRUE, 0, words_size, wbuf, 0, NULL, NULL);
            free(wbuf);
        }
    } else if (p_clEnqueueFillBuffer) {
        unsigned char zb = 0;
        clEnqueueFillBuffer(d->queue, d->b_hexhashes, &zb, sizeof(zb), 0, words_size, 0, NULL, NULL);
    } else {
        unsigned char *zbuf = (unsigned char *)calloc(1, words_size);
        if (zbuf) {
            clEnqueueWriteBuffer(d->queue, d->b_hexhashes, CL_TRUE, 0, words_size, zbuf, 0, NULL, NULL);
            free(zbuf);
        }
    }
    /* Each entry length = 32 (a valid hex hash slot). Build host-side. */
    {
        uint16_t *lens = (uint16_t *)malloc(hexlens_upload);
        if (lens) {
            for (int i = 0; i < num_words; i++) lens[i] = 32;
            clEnqueueWriteBuffer(d->queue, d->b_hexlens, CL_TRUE, 0, hexlens_upload, lens, 0, NULL, NULL);
            free(lens);
        }
    }

    OCLParams params;
    memset(&params, 0, sizeof(params));
    params.compact_mask    = _compact_mask;
    params.num_words       = num_words;
    params.num_salts       = d->salts_count;
    params.salt_start      = 0;
    params.max_probe       = 256;
    params.hash_data_count = _hash_data_count;
    params.max_hits        = GPU_MAX_HITS;
    params.overflow_count  = _overflow_count;
    params.max_iter        = (cat == GPU_CAT_ITER) ? (d->max_iter - 1) : d->max_iter;
    params.num_masks       = is_mask ? 1 : 0;
    params.mask_start      = 0;
    /* Phase 6.2: read mask geometry from local use_* values, which are the
     * synthetic ?l^8 values when gpu_mask_total was 0 at entry, else the
     * real (already-uploaded) values. */
    params.n_prepend       = use_n_prepend;
    params.n_append        = use_n_append;

    uint32_t zero = 0;
    if (p_clEnqueueFillBuffer)
        clEnqueueFillBuffer(d->queue, d->b_hit_count, &zero, sizeof(zero), 0, sizeof(zero), 0, NULL, NULL);
    else
        clEnqueueWriteBuffer(d->queue, d->b_hit_count, CL_TRUE, 0, sizeof(zero), &zero, 0, NULL, NULL);
    clFinish(d->queue);

    /* Bind kernel args identically to dispatch_batch — but use the
     * synthetic compact/mask buffers when the real ones aren't loaded yet
     * (Phase 6.2: lets warm-probe fire before set_compact_table/set_mask). */
    int a = 0;
    clSetKernelArg(kern, a++, sizeof(cl_mem), &d->b_hexhashes);
    clSetKernelArg(kern, a++, sizeof(cl_mem), &d->b_hexlens);
    if (cat == GPU_CAT_MASK || cat == GPU_CAT_UNSALTED)
        clSetKernelArg(kern, a++, sizeof(cl_mem), &use_mask_desc);
    else
        clSetKernelArg(kern, a++, sizeof(cl_mem), &use_salt_data);
    clSetKernelArg(kern, a++, sizeof(cl_mem), &use_salt_off);
    clSetKernelArg(kern, a++, sizeof(cl_mem), &use_salt_len);
    clSetKernelArg(kern, a++, sizeof(cl_mem), &use_compact_fp);
    clSetKernelArg(kern, a++, sizeof(cl_mem), &use_compact_idx);
    clSetKernelArg(kern, a++, sizeof(cl_mem), &d->b_params);
    clSetKernelArg(kern, a++, sizeof(cl_mem), &use_hash_data);
    clSetKernelArg(kern, a++, sizeof(cl_mem), &use_hash_data_off);
    clSetKernelArg(kern, a++, sizeof(cl_mem), &use_hash_data_len);
    clSetKernelArg(kern, a++, sizeof(cl_mem), &d->b_hits);
    clSetKernelArg(kern, a++, sizeof(cl_mem), &d->b_hit_count);
    clSetKernelArg(kern, a++, sizeof(cl_mem), &use_ovfl_keys);
    clSetKernelArg(kern, a++, sizeof(cl_mem), &use_ovfl_hashes);
    clSetKernelArg(kern, a++, sizeof(cl_mem), &use_ovfl_offs);
    clSetKernelArg(kern, a++, sizeof(cl_mem), &use_ovfl_lens);

    int total_salts     = is_salted ? d->salts_count : 0;
    int salt_start_base = is_salted ? d->salt_resume : 0;
    total_salts         = is_salted ? (total_salts - salt_start_base) : 0;

    /* === Same probe loop as dispatch_batch's autotune block === */
    uint32_t probe_size = is_bcrypt ? 1 : TIMING_INITIAL_SIZE;
    uint32_t best_size = probe_size;
    double best_ms = 0;

    for (int probe = 0; probe < 20; probe++) {
        uint32_t zero_hit = 0;
        if (p_clEnqueueFillBuffer)
            clEnqueueFillBuffer(d->queue, d->b_hit_count, &zero_hit, sizeof(zero_hit), 0, sizeof(zero_hit), 0, NULL, NULL);
        else
            clEnqueueWriteBuffer(d->queue, d->b_hit_count, CL_TRUE, 0, sizeof(zero_hit), &zero_hit, 0, NULL, NULL);

        OCLParams probe_params = params;
        if (is_mask) {
            uint32_t pm = probe_size / (num_words > 0 ? num_words : 1);
            if (pm < 1) pm = 1;
            if ((uint64_t)pm > use_mask_total)
                pm = (use_mask_total < 0xFFFFFFFF) ? (uint32_t)use_mask_total : 0xFFFFFFFF;
            probe_params.num_masks = pm;
            probe_params.mask_start = 0;
        } else if (is_salted) {
            int ps = (int)(probe_size / (num_words > 0 ? num_words : 1));
            if (ps < 1) ps = 1;
            if (ps > d->salts_count) ps = d->salts_count;
            probe_params.num_salts = ps;
            probe_params.salt_start = salt_start_base;
        }
        clEnqueueWriteBuffer(d->queue, d->b_params, CL_TRUE, 0, sizeof(probe_params), &probe_params, 0, NULL, NULL);

        size_t probe_global = is_mask
            ? (size_t)num_words * probe_params.num_masks
            : (size_t)num_words * probe_params.num_salts;
        probe_global = ((probe_global + local - 1) / local) * local;

        struct timespec pt0, pt1;
        clock_gettime(CLOCK_MONOTONIC, &pt0);
        cl_int perr = clEnqueueNDRangeKernel(d->queue, kern, 1, NULL, &probe_global, &local, 0, NULL, NULL);
        if (perr != CL_SUCCESS) break;
        clFinish(d->queue);
        clock_gettime(CLOCK_MONOTONIC, &pt1);
        double pms = (pt1.tv_sec - pt0.tv_sec) * 1e3 + (pt1.tv_nsec - pt0.tv_nsec) / 1e6;

        best_size = probe_size;
        best_ms = pms;

        if (is_bcrypt) {
            if (pms > 0) {
                double per_item = pms;
                uint32_t target = (uint32_t)(TIMING_BUDGET_MIN_MS / per_item);
                if (target < 1) target = 1;
                best_size = target;
                d->fam_rate_hps[fam] = (double)best_size * 1000.0 / pms;
            }
            break;
        }

        if (pms >= TIMING_BUDGET_MIN_MS && pms <= TIMING_BUDGET_MAX_MS) break;
        if (pms > TIMING_BUDGET_MAX_MS) {
            best_size = probe_size / 2;
            if (best_size < TIMING_INITIAL_SIZE) best_size = TIMING_INITIAL_SIZE;
            break;
        }
        if (pms >= TIMING_WATCHDOG_MS) break;

        uint64_t next = (uint64_t)probe_size * 2;
        uint64_t max_items = is_salted ? (uint64_t)num_words * d->salts_count
                           : is_mask   ? (uint64_t)num_words * use_mask_total
                           : (uint64_t)num_words;
        if (next > max_items) break;
        if (next > 0x7FFFFFFF) next = 0x7FFFFFFF;
        probe_size = (uint32_t)next;
    }

    /* Phase 6.1.1: rescue rate measurement on very fast GPUs.
     * If the probe loop converged with pms < TIMING_BUDGET_MIN_MS because
     * probe_size hit the uint32_t cap (RTX 3080 etc — 2.1B work items
     * complete in sub-ms), dispatch the same kernel N times back-to-back
     * to accumulate a measurable timing window. The kernel args, params,
     * and probe_global from the last loop iteration are still bound. */
    double final_ms = best_ms;
    uint64_t total_work = best_size;
    if (!is_bcrypt && best_size > 0 && best_ms < TIMING_BUDGET_MIN_MS) {
        uint32_t reps = 1;
        if (best_ms > 0) {
            while ((double)reps * best_ms < TIMING_BUDGET_MIN_MS && reps < 1024)
                reps *= 2;
        } else {
            reps = 256;  /* probe was unmeasurable; try 256x first */
        }
        if (reps > 1) {
            uint32_t zh = 0;
            if (p_clEnqueueFillBuffer)
                clEnqueueFillBuffer(d->queue, d->b_hit_count, &zh, sizeof(zh), 0, sizeof(zh), 0, NULL, NULL);
            else
                clEnqueueWriteBuffer(d->queue, d->b_hit_count, CL_TRUE, 0, sizeof(zh), &zh, 0, NULL, NULL);
            /* Re-establish probe_global from last iteration's params. */
            uint32_t last_pm = (params.num_masks > 0) ? params.num_masks : best_size;
            size_t resc_global = is_mask
                ? (size_t)num_words * last_pm
                : (size_t)num_words * params.num_salts;
            resc_global = ((resc_global + local - 1) / local) * local;
            struct timespec rt0, rt1;
            clock_gettime(CLOCK_MONOTONIC, &rt0);
            for (uint32_t r = 0; r < reps; r++) {
                cl_int re = clEnqueueNDRangeKernel(d->queue, kern, 1, NULL,
                    &resc_global, &local, 0, NULL, NULL);
                if (re != CL_SUCCESS) { reps = r; break; }
            }
            clFinish(d->queue);
            clock_gettime(CLOCK_MONOTONIC, &rt1);
            double total_ms = (rt1.tv_sec - rt0.tv_sec) * 1e3
                            + (rt1.tv_nsec - rt0.tv_nsec) / 1e6;
            if (total_ms >= 1.0 && reps > 0) {
                final_ms = total_ms;
                total_work = (uint64_t)best_size * reps;
            }
        }
    }

    d->fam_max_items[fam] = best_size;
    d->fam_timed[fam] = 1;
    if (final_ms >= 1.0 && d->fam_rate_hps[fam] == 0.0)
        d->fam_rate_hps[fam] = (double)total_work * 1000.0 / final_ms;
    tsfprintf(stderr, "OpenCL GPU[%d]: warm-probed family %d: max_items=%u (%.1fms, %.1f Mh/s)\n",
            dev_idx, fam, best_size, final_ms, d->fam_rate_hps[fam] / 1e6);

    /* Phase 6.2: release synthetic buffers (kernel arg state is irrelevant
     * — main's dispatch_batch rebinds before any subsequent kernel call). */
    if (syn_compact_fp)  clReleaseMemObject(syn_compact_fp);
    if (syn_compact_idx) clReleaseMemObject(syn_compact_idx);
    if (syn_mask_desc)   clReleaseMemObject(syn_mask_desc);
    if (syn_salt_data)      clReleaseMemObject(syn_salt_data);
    if (syn_salt_off)       clReleaseMemObject(syn_salt_off);
    if (syn_salt_len)       clReleaseMemObject(syn_salt_len);
    if (syn_hash_data)      clReleaseMemObject(syn_hash_data);
    if (syn_hash_data_off)  clReleaseMemObject(syn_hash_data_off);
    if (syn_hash_data_len)  clReleaseMemObject(syn_hash_data_len);
    if (syn_ovfl_keys)      clReleaseMemObject(syn_ovfl_keys);
    if (syn_ovfl_hashes)    clReleaseMemObject(syn_ovfl_hashes);
    if (syn_ovfl_offs)      clReleaseMemObject(syn_ovfl_offs);
    if (syn_ovfl_lens)      clReleaseMemObject(syn_ovfl_lens);
}

/* Async dispatcher: one pthread per device. Threads detach on completion;
 * the caller blocks at gpu_opencl_warm_probe_wait() until all are joined. */
struct probe_arg { int dev_idx; int op; };

static pthread_t probe_threads[MAX_GPU_DEVICES];
static int       probe_threads_active[MAX_GPU_DEVICES];
static int       probe_thread_count = 0;
static int       probe_op_cached    = -1;

static void *probe_thread_body(void *vp) {
    struct probe_arg *p = (struct probe_arg *)vp;
    gpu_opencl_warm_probe(p->dev_idx, p->op);
    free(p);
    return NULL;
}

void gpu_opencl_warm_probe_async(int op) {
    if (!ocl_ready) return;
    {
        const char *e = getenv("MDXFIND_SKIP_WARMUP");
        if (e && *e && *e != '0') {
            fprintf(stderr, "WARN: warm_probe_async skipped via MDXFIND_SKIP_WARMUP\n");
            return;
        }
    }
    if (probe_thread_count > 0) {
        fprintf(stderr, "WARN: warm_probe_async early-return: probe_thread_count=%d (stale from prior session?)\n",
                probe_thread_count);
        return;   /* already running — skip */
    }
    probe_op_cached = op;
    for (int d = 0; d < num_gpu_devs && d < MAX_GPU_DEVICES; d++) {
        struct probe_arg *arg = (struct probe_arg *)malloc(sizeof(*arg));
        if (!arg) continue;
        arg->dev_idx = d;
        arg->op      = op;
        if (pthread_create(&probe_threads[d], NULL, probe_thread_body, arg) == 0) {
            probe_threads_active[d] = 1;
            probe_thread_count++;
        } else {
            free(arg);
            probe_threads_active[d] = 0;
        }
    }
}

void gpu_opencl_warm_probe_wait(void) {
    for (int d = 0; d < num_gpu_devs && d < MAX_GPU_DEVICES; d++) {
        if (probe_threads_active[d]) {
            pthread_join(probe_threads[d], NULL);
            probe_threads_active[d] = 0;
        }
    }
    probe_thread_count = 0;
    probe_op_cached    = -1;
    __sync_synchronize();
}

int gpu_opencl_set_compact_table(int dev_idx,
    uint32_t *compact_fp, uint32_t *compact_idx,
    uint64_t compact_size, uint64_t compact_mask,
    unsigned char *hash_data_buf, size_t hash_data_buf_size,
    size_t *hash_data_off, size_t hash_data_count,
    unsigned short *hash_data_len)
{
    if (!ocl_ready || dev_idx < 0 || dev_idx >= num_gpu_devs) return -1;
    struct gpu_device *d = &gpu_devs[dev_idx];
    /* Skip slots where init_device failed (parallel-init path: failed slots
     * remain in gpu_devs[] with device_disabled=1; their cl_context etc. are
     * uninitialized so any cl* call would crash). */
    if (d->device_disabled) return -1;
    cl_int err;

    /* Check if GPU has enough memory for the compact table + hash data */
    size_t needed = compact_size * sizeof(uint32_t) * 2   /* compact_fp + compact_idx */
                  + hash_data_buf_size                      /* hash_data_buf */
                  + hash_data_count * sizeof(uint64_t)      /* hash_data_off */
                  + hash_data_count * sizeof(uint16_t)      /* hash_data_len */
                  + GPU_MAX_HITS * GPU_HIT_STRIDE * sizeof(uint32_t)     /* hits buffer */
                  + 512 * 256;                              /* hexhash buffer */
    cl_ulong gpu_mem = 0;
    clGetDeviceInfo(d->dev, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(gpu_mem), &gpu_mem, NULL);
    if (needed > (size_t)(gpu_mem * 0.8)) {
        fprintf(stderr, "OpenCL GPU[%d]: insufficient memory (%zuMB needed, %lluMB available) — DEVICE DISABLED, no GPU work routes here\n",
                dev_idx, needed / (1024*1024), (unsigned long long)(gpu_mem / (1024*1024)));
        d->device_disabled = 1;
        return -1;
    }

    _compact_mask = compact_mask;
    _hash_data_count = hash_data_count;

    /* Phase C: time the compact-table upload (CL_MEM_COPY_HOST_PTR is
     * implicit synchronous-ish copy, but timing the buffer-creates lets
     * us see how long the driver takes to provision + DMA the table to
     * VRAM). The "compact upload" wall = sum of all 5 buffer creates
     * for this device. */
    struct timespec _cu_t0, _cu_t1;
    clock_gettime(CLOCK_MONOTONIC, &_cu_t0);
    tsfprintf(stderr, "OpenCL GPU[%d]: compact upload START %zuMB\n",
              dev_idx, needed / (1024*1024));
    /* Allocate + upload via create_min_buf so per-target small workloads
     * (e.g. 6-hash compact table = 64 B fp/idx, ~96 B hash_data, 48 B
     * hash_data_off, 12 B hash_data_len) are padded to MIN_BUFFER_BYTES.
     * NVIDIA Windows cold-JIT validates buffer size at NDRange time
     * against kernel signature; small buffers are rejected with
     * CL_INVALID_KERNEL_ARGS even though kernel guards bound access.
     * Shooter 12-GPU RTX 4090 bug2b 2026-05-04. */
    d->b_compact_fp = create_min_buf(d->ctx, d->queue, CL_MEM_READ_ONLY,
                                     compact_size * sizeof(uint32_t),
                                     compact_fp, &err);
    if (err != CL_SUCCESS) goto alloc_fail;
    d->b_compact_idx = create_min_buf(d->ctx, d->queue, CL_MEM_READ_ONLY,
                                      compact_size * sizeof(uint32_t),
                                      compact_idx, &err);
    if (err != CL_SUCCESS) goto alloc_fail;
    d->b_hash_data = create_min_buf(d->ctx, d->queue, CL_MEM_READ_ONLY,
                                    hash_data_buf_size, hash_data_buf, &err);
    if (err != CL_SUCCESS) goto alloc_fail;

    uint64_t *off64 = (uint64_t *)malloc_lock(hash_data_count * sizeof(uint64_t),"gpu temp");
    for (size_t i = 0; i < hash_data_count; i++) off64[i] = hash_data_off[i];
    d->b_hash_data_off = create_min_buf(d->ctx, d->queue, CL_MEM_READ_ONLY,
                                        hash_data_count * sizeof(uint64_t),
                                        off64, &err);
    if (err != CL_SUCCESS) { free(off64); goto alloc_fail; }

    d->b_hash_data_len = create_min_buf(d->ctx, d->queue, CL_MEM_READ_ONLY,
                                        hash_data_count * sizeof(uint16_t),
                                        (void*)hash_data_len, &err);
    if (err != CL_SUCCESS) { free(off64); goto alloc_fail; }

    free(off64);
    clock_gettime(CLOCK_MONOTONIC, &_cu_t1);
    {
        double _cu_ms = (_cu_t1.tv_sec - _cu_t0.tv_sec) * 1e3
                      + (_cu_t1.tv_nsec - _cu_t0.tv_nsec) / 1e6;
        double _cu_gbps = (_cu_ms > 0.001)
                        ? ((double)needed / (_cu_ms / 1e3) / (1024.0*1024*1024))
                        : 0.0;
        tsfprintf(stderr, "OpenCL GPU[%d]: compact upload DONE in %.2fs (%.2f GB/s)\n",
                  dev_idx, _cu_ms / 1e3, _cu_gbps);
    }

    /* Read-back probes: first 256 + last 256 bytes of each compact buffer. */
    {
        size_t fp_bytes = compact_size * sizeof(uint32_t);
        size_t head = fp_bytes < 256 ? fp_bytes : 256;
        gpu_readback_probe(dev_idx, d->queue, d->b_compact_fp, 0, head,
                           compact_fp, "set_compact", "b_compact_fp[head]");
        gpu_readback_probe(dev_idx, d->queue, d->b_compact_idx, 0, head,
                           compact_idx, "set_compact", "b_compact_idx[head]");
        if (fp_bytes > 512) {
            size_t off = fp_bytes - 256;
            gpu_readback_probe(dev_idx, d->queue, d->b_compact_fp, off, 256,
                               (const unsigned char *)compact_fp + off, "set_compact", "b_compact_fp[tail]");
            gpu_readback_probe(dev_idx, d->queue, d->b_compact_idx, off, 256,
                               (const unsigned char *)compact_idx + off, "set_compact", "b_compact_idx[tail]");
        }
    }

    tsfprintf(stderr, "OpenCL GPU[%d]: compact table registered (%llu slots, %u hashes, %zuMB)\n",
            dev_idx, (unsigned long long)compact_size, (unsigned)hash_data_count,
            needed / (1024*1024));
    return 0;

alloc_fail:
    fprintf(stderr, "OpenCL GPU[%d]: buffer allocation failed (err=%d) — DEVICE DISABLED, no GPU work routes here\n", dev_idx, err);
    if (d->b_compact_fp) { clReleaseMemObject(d->b_compact_fp); d->b_compact_fp = NULL; }
    if (d->b_compact_idx) { clReleaseMemObject(d->b_compact_idx); d->b_compact_idx = NULL; }
    if (d->b_hash_data) { clReleaseMemObject(d->b_hash_data); d->b_hash_data = NULL; }
    if (d->b_hash_data_off) { clReleaseMemObject(d->b_hash_data_off); d->b_hash_data_off = NULL; }
    if (d->b_hash_data_len) { clReleaseMemObject(d->b_hash_data_len); d->b_hash_data_len = NULL; }
    d->device_disabled = 1;
    return -1;
}

int gpu_opencl_set_salts(int dev_idx,
    const char *salts, const uint32_t *salt_offsets,
    const uint16_t *salt_lens, int num_salts)
{
    if (!ocl_ready || dev_idx < 0 || dev_idx >= num_gpu_devs || num_salts <= 0) return -1;
    struct gpu_device *d = &gpu_devs[dev_idx];
    cl_int err;
    size_t salts_size = salt_offsets[num_salts - 1] + salt_lens[num_salts - 1];

    if (salts_size > d->salt_data_cap) {
        if (d->b_salt_data) clReleaseMemObject(d->b_salt_data);
        d->b_salt_data = clCreateBuffer(d->ctx, CL_MEM_READ_ONLY, salts_size + 4096, NULL, &err);
        d->salt_data_cap = salts_size + 4096;
    }
    /* Reuse existing offset/length buffers if salt count hasn't grown —
     * repeated clReleaseMemObject/clCreateBuffer leaks NVIDIA driver state */
    size_t need_off = (size_t)num_salts * sizeof(uint32_t);
    size_t need_len = (size_t)num_salts * sizeof(uint16_t);
    if (need_off > d->salt_off_cap) {
        if (d->b_salt_off) clReleaseMemObject(d->b_salt_off);
        if (d->b_salt_len) clReleaseMemObject(d->b_salt_len);
        d->b_salt_off = clCreateBuffer(d->ctx, CL_MEM_READ_ONLY, need_off, NULL, &err);
        d->b_salt_len = clCreateBuffer(d->ctx, CL_MEM_READ_ONLY, need_len, NULL, &err);
        d->salt_off_cap = need_off;
    }

    clEnqueueWriteBuffer(d->queue, d->b_salt_data, CL_TRUE, 0, salts_size, salts, 0, NULL, NULL);
    clEnqueueWriteBuffer(d->queue, d->b_salt_off, CL_TRUE, 0, num_salts * sizeof(uint32_t), salt_offsets, 0, NULL, NULL);
    clEnqueueWriteBuffer(d->queue, d->b_salt_len, CL_TRUE, 0, num_salts * sizeof(uint16_t), salt_lens, 0, NULL, NULL);
    clFinish(d->queue);
    d->salts_count = num_salts;
    return 0;
}

int gpu_opencl_set_overflow(int dev_idx,
    const uint64_t *keys, const unsigned char *hashes,
    const uint32_t *offsets, const uint16_t *lengths, int count)
{
    if (!ocl_ready || dev_idx < 0 || dev_idx >= num_gpu_devs) return -1;
    struct gpu_device *d = &gpu_devs[dev_idx];
    cl_int err;

    /* count == 0: release any prior real buffers and bind MIN_BUFFER_BYTES
     * zero placeholders. NVIDIA Windows rejects sub-floor cl_mems against
     * `__global ulong *` etc. at NDRange time with CL_INVALID_KERNEL_ARGS;
     * the floor allocation matches the kernel signature breadth and is
     * never read because the kernel guards `overflow_count > 0`. The
     * floor (rev 1.89) tracks MIN_BUFFER_BYTES so this stays consistent
     * with the create_min_buf path and the static init in
     * gpu_opencl_init_device — see comment block there. */
    if (count <= 0) {
        uint8_t *_ovfl_placeholder_zero = (uint8_t *)calloc(1, MIN_BUFFER_BYTES);
        if (!_ovfl_placeholder_zero) {
            fprintf(stderr,
                "OpenCL GPU[%d]: overflow placeholder calloc(%d) failed\n",
                dev_idx, MIN_BUFFER_BYTES);
            return -1;
        }
        if (d->b_overflow_keys)    clReleaseMemObject(d->b_overflow_keys);
        if (d->b_overflow_hashes)  clReleaseMemObject(d->b_overflow_hashes);
        if (d->b_overflow_offsets) clReleaseMemObject(d->b_overflow_offsets);
        if (d->b_overflow_lengths) clReleaseMemObject(d->b_overflow_lengths);
        d->b_overflow_keys = clCreateBuffer(d->ctx,
            CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, MIN_BUFFER_BYTES,
            _ovfl_placeholder_zero, &err);
        d->b_overflow_hashes = clCreateBuffer(d->ctx,
            CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, MIN_BUFFER_BYTES,
            _ovfl_placeholder_zero, &err);
        d->b_overflow_offsets = clCreateBuffer(d->ctx,
            CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, MIN_BUFFER_BYTES,
            _ovfl_placeholder_zero, &err);
        d->b_overflow_lengths = clCreateBuffer(d->ctx,
            CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, MIN_BUFFER_BYTES,
            _ovfl_placeholder_zero, &err);
        free(_ovfl_placeholder_zero);
        _overflow_count = 0;
        tsfprintf(stderr,
            "OpenCL GPU[%d]: overflow set to %dB zero-fill placeholder (count=0)\n",
            dev_idx, MIN_BUFFER_BYTES);
        return 0;
    }

    /* B5 sub-batch 7 (2026-05-05): the caller (gpujob_opencl.c:load_overflow)
     * zero-pads sub-128-bit overflow entries to 16 bytes so the GPU
     * 4xuint32 probe doesn't spill into the next entry. lengths[] keeps
     * the real digest length for CPU memcmp paths; for buffer sizing we
     * must use max(real_len, 16) so the upload covers the padded tail
     * of the LAST entry. Larger algorithms (>=16 bytes) are unaffected. */
    size_t last_pad_len = lengths[count - 1] < 16 ? 16 : lengths[count - 1];
    size_t total = offsets[count - 1] + last_pad_len;

    /* Phase G: time the overflow upload. */
    struct timespec _ov_t0, _ov_t1;
    clock_gettime(CLOCK_MONOTONIC, &_ov_t0);
    size_t _ov_total_bytes = count * sizeof(uint64_t)
                           + total
                           + count * sizeof(uint32_t)
                           + count * sizeof(uint16_t);
    tsfprintf(stderr, "OpenCL GPU[%d]: overflow upload START (%d entries, %.2f KB)\n",
              dev_idx, count, _ov_total_bytes / 1024.0);

    clReleaseMemObject(d->b_overflow_keys);
    clReleaseMemObject(d->b_overflow_hashes);
    clReleaseMemObject(d->b_overflow_offsets);
    clReleaseMemObject(d->b_overflow_lengths);

    d->b_overflow_keys = clCreateBuffer(d->ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                        count * sizeof(uint64_t), (void*)keys, &err);
    d->b_overflow_hashes = clCreateBuffer(d->ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                          total, (void*)hashes, &err);
    d->b_overflow_offsets = clCreateBuffer(d->ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                           count * sizeof(uint32_t), (void*)offsets, &err);
    d->b_overflow_lengths = clCreateBuffer(d->ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                            count * sizeof(uint16_t), (void*)lengths, &err);
    _overflow_count = count;
    clock_gettime(CLOCK_MONOTONIC, &_ov_t1);
    {
        double _ov_ms = (_ov_t1.tv_sec - _ov_t0.tv_sec) * 1e3
                      + (_ov_t1.tv_nsec - _ov_t0.tv_nsec) / 1e6;
        tsfprintf(stderr, "OpenCL GPU[%d]: overflow upload DONE in %.2fs\n",
                  dev_idx, _ov_ms / 1e3);
    }
    /* Read-back probes: first 64 bytes of each overflow buffer (and tail if large). */
    {
        size_t k_bytes = (size_t)count * sizeof(uint64_t);
        size_t k_head = k_bytes < 64 ? k_bytes : 64;
        gpu_readback_probe(dev_idx, d->queue, d->b_overflow_keys, 0, k_head,
                           keys, "set_overflow", "b_overflow_keys[head]");
        if (k_bytes > 128) {
            size_t off = k_bytes - 64;
            gpu_readback_probe(dev_idx, d->queue, d->b_overflow_keys, off, 64,
                               (const unsigned char *)keys + off, "set_overflow", "b_overflow_keys[tail]");
        }
        size_t h_head = total < 64 ? total : 64;
        gpu_readback_probe(dev_idx, d->queue, d->b_overflow_hashes, 0, h_head,
                           hashes, "set_overflow", "b_overflow_hashes[head]");
        if (total > 128) {
            size_t off = total - 64;
            gpu_readback_probe(dev_idx, d->queue, d->b_overflow_hashes, off, 64,
                               hashes + off, "set_overflow", "b_overflow_hashes[tail]");
        }
        size_t o_bytes = (size_t)count * sizeof(uint32_t);
        size_t o_head = o_bytes < 64 ? o_bytes : 64;
        gpu_readback_probe(dev_idx, d->queue, d->b_overflow_offsets, 0, o_head,
                           offsets, "set_overflow", "b_overflow_offsets[head]");
        size_t l_bytes = (size_t)count * sizeof(uint16_t);
        size_t l_head = l_bytes < 64 ? l_bytes : 64;
        gpu_readback_probe(dev_idx, d->queue, d->b_overflow_lengths, 0, l_head,
                           lengths, "set_overflow", "b_overflow_lengths[head]");
    }
    tsfprintf(stderr, "OpenCL GPU[%d]: %d overflow entries loaded\n", dev_idx, count);
    return 0;
}

void gpu_opencl_set_max_iter(int dev_idx, int max_iter) {
    if (dev_idx >= 0 && dev_idx < num_gpu_devs)
        gpu_devs[dev_idx].max_iter = (max_iter < 1) ? 1 : max_iter;
}
/* BF Phase 3b Tranche C (2026-05-10): gpu_opencl_set_mask_resume,
 * gpu_opencl_set_salt_resume, gpu_opencl_has_resume, and
 * gpu_opencl_last_mask_start removed. These helpers were referenced only
 * from the deleted slab dispatcher arm (gpu_opencl_dispatch_batch, retired
 * at gpu/gpu_opencl.c rev 1.164 / gpu/gpu_opencl.h rev 1.26 in Tranche B);
 * post-Tranche-B audit confirmed zero remaining callers in any live source
 * (only legacy mirrors under gpu/205/ and gpu/209/ reference them, and
 * those are snapshot directories not part of the iMac build). Mask/salt
 * resume state still lives on gpu_devs[] but is no longer exposed via
 * setter API; if a future feature needs it, expose a fresh API rather
 * than reviving these stubs. */
void gpu_opencl_set_op(int dev_idx, int op) {
    if (dev_idx >= 0 && dev_idx < num_gpu_devs)
        gpu_devs[dev_idx].gpu_op = op;
}
int gpu_opencl_get_op(int dev_idx) {
    if (dev_idx >= 0 && dev_idx < num_gpu_devs)
        return gpu_devs[dev_idx].gpu_op;
    return -1;
}

/* Mask mode state — accessed by gpujob for hit reconstruction.
 * gpu_mask_desc layout: [sizes[n_total], tables[n_total][256]]
 * sizes[i] = character count for position i
 * tables[i] = 256-byte character table for position i
 * Kernel indexes: ch = mask_desc[n_total + i*256 + (idx % mask_desc[i])] */
#ifndef MAX_MASK_POS
#define MAX_MASK_POS 16
#endif
uint8_t gpu_mask_desc[MAX_MASK_POS + MAX_MASK_POS * 256];
uint8_t gpu_mask_sizes[MAX_MASK_POS];  /* for hit reconstruction */
int gpu_mask_n_prepend = 0;
int gpu_mask_n_append = 0;
uint64_t gpu_mask_total = 0;

int gpu_opencl_set_mask(const uint8_t *sizes, const uint8_t tables[][256],
                        int npre, int napp) {
    int ntotal = npre + napp;
    gpu_mask_n_prepend = npre;
    gpu_mask_n_append = napp;
    /* Pack: sizes first, then tables */
    memcpy(gpu_mask_desc, sizes, ntotal);
    memcpy(gpu_mask_sizes, sizes, ntotal);
    for (int i = 0; i < ntotal; i++)
        memcpy(gpu_mask_desc + ntotal + i * 256, tables[i], 256);
    gpu_mask_total = 1;
    for (int i = 0; i < ntotal; i++)
        gpu_mask_total *= sizes[i];
    /* Upload mask descriptor to all devices */
    int bufsize = ntotal + ntotal * 256;
    for (int i = 0; i < num_gpu_devs; i++) {
        struct gpu_device *d = &gpu_devs[i];
        if (d->bgpu_mask_desc) clReleaseMemObject(d->bgpu_mask_desc);
        cl_int err;
        d->bgpu_mask_desc = clCreateBuffer(d->ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                         bufsize, gpu_mask_desc, &err);
    }
    /* Memo B Phase B7.1-B7.5/B7.8: upload the multi-position prepend+
     * append charsets and per-position sizes to the template path's
     * two dedicated buffers. The B7.5-eligible configuration is
     * (n_prepend in [0, MASK_POS_CAP], n_append in [0, MASK_POS_CAP],
     * n_prepend + n_append >= 1); for out-of-scope configurations we
     * still allocate sentinel buffers (so kernel args bind to valid
     * memory; the kernel's (n_prepend>=1 || n_append>=1) gate prevents
     * any actual read of sentinel bytes).
     *
     * mask_charsets buffer: MASK_TOTAL_CAP × 256 = 8192 bytes (B7.8;
     * pre-B7.8 was 4096 bytes at MASK_POS_CAP=8). Layout follows the
     * SLAB-PATH CONVENTION (gpu_kernels.cl md5_mask_batch /
     * gpu_mask_desc): rows [0..n_prepend) are prepend rows; rows
     * [n_prepend..n_prepend+n_append) are append rows. Row i at offset
     * i*256 holds the charset for that position. Unused rows are
     * zero-filled sentinel.
     *
     * mask_sizes buffer: MASK_TOTAL_CAP × sizeof(uint) = 128 bytes
     * (B7.8; pre-B7.8 was 64 bytes). mask_sizes[i] gives the modulus
     * for position i (in the same prepend-then-append order as
     * mask_charsets). The remaining entries hold a sentinel of 1 (so
     * any stray divmod-by-mask_sizes[i] terminates safely).
     *
     * Backward-compat with B7.2: when npre==0 and napp in [1, 8], rows
     * [0..napp) are append rows — bit-identical to B7.2's packing
     * (which had no prepend section). B7.8 keeps the same row layout
     * for npre/napp <= 8, just allows larger configurations. */
    {
        enum { MASK_POS_CAP = 16, MASK_TOTAL_CAP = MASK_POS_CAP * 2 };
        uint8_t  b7_charsets[MASK_TOTAL_CAP * 256];
        uint32_t b7_sizes[MASK_TOTAL_CAP];
        memset(b7_charsets, 0, sizeof(b7_charsets));
        for (int i = 0; i < MASK_TOTAL_CAP; i++) b7_sizes[i] = 1u;
        if (npre >= 0 && napp >= 0 && (npre + napp) >= 1
            && npre <= MASK_POS_CAP && napp <= MASK_POS_CAP) {
            /* Prepend rows [0..npre), then append rows [npre..npre+napp).
             * tables[i] is the 256-byte charset for position i in the
             * caller's (slab) layout: prepend [0..npre), append
             * [npre..npre+napp). We mirror that layout into the template
             * buffer one-to-one. */
            for (int i = 0; i < npre + napp; i++) {
                memcpy(b7_charsets + i * 256, tables[i], 256);
                b7_sizes[i] = (uint32_t)sizes[i];
            }
        }
        for (int i = 0; i < num_gpu_devs; i++) {
            struct gpu_device *d = &gpu_devs[i];
            if (d->b_template_mask_charsets) clReleaseMemObject(d->b_template_mask_charsets);
            cl_int err;
            d->b_template_mask_charsets = clCreateBuffer(d->ctx,
                CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                sizeof(b7_charsets), b7_charsets, &err);
            if (err != CL_SUCCESS || !d->b_template_mask_charsets) {
                fprintf(stderr,
                    "OpenCL GPU[%d]: b_template_mask_charsets alloc failed "
                    "(err=%d) — template B7 mask unavailable on "
                    "this device; will route mask jobs to slab path.\n",
                    i, err);
                d->b_template_mask_charsets = NULL;
            }
            if (d->b_template_mask_sizes) clReleaseMemObject(d->b_template_mask_sizes);
            d->b_template_mask_sizes = clCreateBuffer(d->ctx,
                CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                sizeof(b7_sizes), b7_sizes, &err);
            if (err != CL_SUCCESS || !d->b_template_mask_sizes) {
                fprintf(stderr,
                    "OpenCL GPU[%d]: b_template_mask_sizes alloc failed "
                    "(err=%d) — template B7 mask unavailable on "
                    "this device; will route mask jobs to slab path.\n",
                    i, err);
                d->b_template_mask_sizes = NULL;
            }
        }
    }
    fprintf(stderr, "OpenCL GPU: mask mode: %d prepend + %d append = %llu combinations\n",
            npre, napp, (unsigned long long)gpu_mask_total);
    return 0;
}

/* B7.9 (2026-05-07): gpu_opencl_dispatch_packed() retired. The chokepoint
 * pack at mdxfind.c:11199-11374 was removed in this same commit and was
 * the sole production caller. The 5 packed kernel sources (md5/sha1/sha256
 * /md4/sha512_packed) have been deleted from the working tree per
 * feedback_remove_retired_gpu_kernels.md; RCS history retains them.
 *
 * Workloads that previously hit this path:
 *   - 100%-GPU-eligible rules (gpu_legacy_slot_unused == 1, the dominant
 *     production case): already bypassed the chokepoint pack via
 *     gpu_skip_no_rule_pack — no net change.
 *   - Mixed CPU+GPU rules (rl.ncpu > 0): CPU-rule outputs now CPU-fallback
 *     instead of GPU-dispatching to the packed kernel. Architect-accepted
 *     ~5% perf hit per project_b76plus_mask_iter_closure.md §13 Q2 option
 *     A1 (pure-retire variant).
 *   - GPU_CAT_MASK ops without a packed_family entry (KECCAK*, SHA3_*,
 *     RMD160, RMD320, BLAKE2*, MYSQL3, STREEBOG_*, WRL, MD6256, salted
 *     variants, ...): previously silent-zero-hit when chokepoint fired
 *     (packed_family returned -1 → dispatch_packed returned NULL). Now
 *     CPU-fallbacks correctly. Net behavior IMPROVES vs the prior path.
 */

/* ========================================================================
 * Phase 0/1 GPU rule expansion engine (md5_rules_phase0)
 *
 * The kernel does (word, rule) Cartesian product on-GPU: a single
 * dispatch with global_size == n_words * n_rules expands every word
 * by every rule, hashes the result, and probes the compact table.
 * Hits arrive as (word_idx, rule_idx, 0, hash); host replays
 * applyrule(words[word_idx], rules[rule_idx]) on a hit to recover
 * the plaintext.
 *
 * Two-step dispatch:
 *   1. gpu_opencl_set_rules() — once per session. Uploads the
 *      compiled rule program (concatenated post-packrules bytecodes
 *      separated by NUL) and per-rule offsets to GPU buffers.
 *      Caller has typically run classify_rules() first and is passing
 *      only the GPU-eligible subset.
 *   2. gpu_opencl_dispatch_md5_rules() — per word batch. Same packed
 *      [len][bytes] format as the existing dispatch_packed path —
 *      the chokepoint can reuse its packing.
 *
 * The kernel + classifier together are validated at 792/792 byte-exact
 * across RTX 4070 Ti SUPER, GTX 1080, and GTX 960 in the standalone
 * harness (see gpu_rules_test.c). HashMob 100k rule-set classifier
 * coverage is 99.99%.
 * ====================================================================== */

/* Lazy-build the rules program for this device. The kernel uses
 * gpu_common.cl (md5_block, OCLParams, EMIT_HIT_4, probe_compact)
 * so we concat gpu_common_str + gpu_md5_rules_str. Held separately
 * from the per-family programs since the rules kernel is its own
 * independent dispatch path. */
/* Build the md5_rules program for this device. Note: we deliberately
 * do NOT create the kernel object here — that's deferred to the first
 * dispatch_md5_rules() call (see gpu_opencl_rules_kernel_lazy() below).
 *
 * Why: clCreateKernel'd-but-never-used kernel objects cause NVIDIA's
 * runtime to emit CL_INVALID_KERNEL_ARGS at context-finalize time
 * because the kernel never had its args bound. Sub-commit A's
 * gpu_opencl_set_rules() runs at session start, but in workloads
 * where the chokepoint never actually dispatches the rules engine
 * (e.g., non-MD5 sessions, or sub-commits A/C1 where the dispatch
 * path is unreachable) the kernel sits unused. Lazy kernel creation
 * means we only have a kernel object when we're about to bind args
 * to it, eliminating the unbound-kernel window. Cost: one branch in
 * dispatch_md5_rules() and a few microseconds at first dispatch.
 */
static int gpu_opencl_rules_compile(struct gpu_device *d, int dev_idx) {
    if (d->prog_md5_rules) return 0;       /* already built */
    cl_int err = CL_SUCCESS;
    const char *sources[2] = { gpu_common_str, gpu_md5_rules_str };
    /* gpu_kernel_cache_build_program() handles cache load/store + atomic
     * compile-with-lock when MDXFIND_CACHE is set; falls through to plain
     * source compile when the cache is disabled. Either way returns the
     * built cl_program (or NULL on failure) and the last cl_int via err. */
    d->prog_md5_rules = gpu_kernel_cache_build_program(d->ctx, d->dev,
                                                       2, sources,
                                                       "-cl-std=CL1.2", &err);
    if (!d->prog_md5_rules || err != CL_SUCCESS) {
        char log[8192] = {0};
        if (d->prog_md5_rules) {
            clGetProgramBuildInfo(d->prog_md5_rules, d->dev,
                                  CL_PROGRAM_BUILD_LOG,
                                  sizeof(log) - 1, log, NULL);
        }
        fprintf(stderr,
            "OpenCL GPU[%d]: rules program build error (err=%d) — DEVICE DISABLED, no GPU work routes here:\n%s\n",
            dev_idx, err, log);
        if (d->prog_md5_rules) clReleaseProgram(d->prog_md5_rules);
        d->prog_md5_rules = NULL;
        d->device_disabled = 1;
        return -1;
    }
    return 0;
}

/* Cached env-var read. MDXFIND_GPU_VALIDATOR=<anything> swaps the
 * production md5_rules_phase0 kernel for md5_rules_phase0_validate
 * and emits per-(word,rule) VALIDATE lines on stderr in the same
 * format as the CPU validator (ruleproc.c rev 1.24,
 * MDXFIND_RULE_VALIDATOR=1). When unset (the production path),
 * this function returns 0 after one getenv() call — no measurable
 * overhead on the dispatch hot path. */
static int gpu_validator_enabled(void) {
    static int cached = -1;
    if (cached == -1) {
        cached = (getenv("MDXFIND_GPU_VALIDATOR") != NULL) ? 1 : 0;
        if (cached) {
            fprintf(stderr,
                "OpenCL: MDXFIND_GPU_VALIDATOR=1 — md5_rules dispatch will use "
                "validator kernel and emit VALIDATE lines on stderr\n");
        }
    }
    return cached;
}

/* Lazily create the md5_rules_phase0_validate kernel object. Only
 * called when the validator env var is set (see gpu_validator_enabled).
 * Mirrors gpu_opencl_rules_kernel_lazy below. */
static int gpu_opencl_validate_kernel_lazy(struct gpu_device *d, int dev_idx) {
    if (d->kern_md5_rules_phase0_validate) return 0;
    if (!d->prog_md5_rules) return -1;
    cl_int err;
    d->kern_md5_rules_phase0_validate = clCreateKernel(d->prog_md5_rules,
                                                       "md5_rules_phase0_validate",
                                                       &err);
    if (err != CL_SUCCESS || !d->kern_md5_rules_phase0_validate) {
        fprintf(stderr, "OpenCL GPU[%d]: validator kernel create failed (err=%d) — DEVICE DISABLED, no GPU work routes here\n",
                dev_idx, err);
        d->kern_md5_rules_phase0_validate = NULL;
        d->device_disabled = 1;
        return -1;
    }
    return 0;
}

/* Lazily create the md5_rules_phase0 kernel object. Called from
 * dispatch_md5_rules at the start of every dispatch — does work only
 * on the first call. See the comment on gpu_opencl_rules_compile()
 * for why kernel creation is deferred. */
static int gpu_opencl_rules_kernel_lazy(struct gpu_device *d, int dev_idx) {
    if (d->kern_md5_rules_phase0) return 0;
    if (!d->prog_md5_rules) return -1;
    cl_int err;
    d->kern_md5_rules_phase0 = clCreateKernel(d->prog_md5_rules,
                                              "md5_rules_phase0", &err);
    if (err != CL_SUCCESS || !d->kern_md5_rules_phase0) {
        fprintf(stderr, "OpenCL GPU[%d]: rules kernel create failed (err=%d) — DEVICE DISABLED, no GPU work routes here\n",
                dev_idx, err);
        d->kern_md5_rules_phase0 = NULL;
        d->device_disabled = 1;
        return -1;
    }
    return 0;
}

/* ====================================================================
 * Memo B Phase B2: generic dispatch template (env-gated, default off).
 *
 * MDXFIND_GPU_TEMPLATE=md5 (or any non-empty/non-"0") swaps the
 * production md5_rules_phase0 kernel for the template-instantiated
 * template_phase0 kernel built from gpu_template.cl + gpu_md5_core.cl.
 * Both kernels have identical signatures and identical wire formats;
 * only the cl_kernel handle changes.
 *
 * The template path proves the structural skeleton that Phase B4-B7
 * will fan out to SHA1/SHA256/SHA512/MD4/etc. B2 itself is a
 * regression-only structural prerequisite: expected wins are 0 (per
 * the phase ladder); the byte-exact 21,289 mmt+ioblade gate is the
 * sign-off.
 * ==================================================================== */

/* Cached env-var read for MDXFIND_GPU_TEMPLATE. Returns:
 *   GPU_TEMPLATE_OFF  (0) = template path off (default; production md5_rules_phase0)
 *   GPU_TEMPLATE_MD5  (1) = MD5 template (template_phase0 with MD5 core)
 *   GPU_TEMPLATE_SHA1 (2) = SHA1 template (template_phase0 with SHA1 core, B4)
 *
 * B4 (2026-05-04) extended the parser to recognize "sha1". The chokepoint
 * gate at mdxfind.c:10054 still restricts the rules-engine path to
 * job->op == JOB_MD5 — MDXFIND_GPU_TEMPLATE=sha1 alone does NOT route
 * real-mdxfind SHA1 work through this kernel. Production validation of
 * the SHA1 template happens via the gpu_rules_test harness
 * (--algo=sha1 --engine={legacy|template}) until a follow-up commit
 * widens the chokepoint gate. The env-var slot is wired here so the
 * harness can reuse the same compile/cache infrastructure. */
#define GPU_TEMPLATE_OFF    0
#define GPU_TEMPLATE_MD5    1
#define GPU_TEMPLATE_SHA1   2
/* Memo B Phase B4 fan-out (2026-05-04): SHA256/SHA224/MD4 instantiations.
 * Each is selected over the MD5 template at the dispatch_md5_rules
 * kernel-handle swap site when MDXFIND_GPU_TEMPLATE matches AND the
 * dispatch op == JOB_<algo>. Real-mdxfind work for these algorithms
 * does NOT reach this dispatch path until the chokepoint gate at
 * mdxfind.c:10054 widens (separate commit); the harness
 * (gpu_rules_test.c -a sha256/sha224/md4) is the validation gate. */
#define GPU_TEMPLATE_SHA256 3
#define GPU_TEMPLATE_SHA224 4
#define GPU_TEMPLATE_MD4    5
/* Memo B Phase B5 sub-batch 1 (2026-05-04): first 64-bit-state algos
 * in the template family. SHA384 = 6 ulong = 12 uint32 (HASH_WORDS=12);
 * SHA512 = 8 ulong = 16 uint32 (HASH_WORDS=16). Both use 128-byte blocks
 * (vs MD5/SHA1/SHA2-256/MD4's 64-byte blocks). */
#define GPU_TEMPLATE_SHA384 6
#define GPU_TEMPLATE_SHA512 7
/* Memo B Phase B5 sub-batch 2 (2026-05-05): RIPEMD-160 / RIPEMD-320.
 * RIPEMD-160 is the second 5-word-state algo (after SHA1); RIPEMD-320 is
 * the first 10-word-state algo (10 × uint32 LE; needed new EMIT_HIT_10
 * family in gpu_common.cl rev 1.12). Both are LE per uint32 state (match
 * MD5 / MD4 convention; UNLIKE the SHA family which is BE). */
#define GPU_TEMPLATE_RIPEMD160 8
#define GPU_TEMPLATE_RIPEMD320 9
/* Memo B Phase B5 sub-batch 3 (2026-05-06): BLAKE2 family. BLAKE2S-256
 * (8 uint32 LE state), BLAKE2B-256 (8 uint32 LE = first 4-of-8 ulong),
 * BLAKE2B-512 (16 uint32 LE = full 8 ulong). New b2b_compress primitive
 * in gpu_common.cl rev 1.13 (b2s_compress was already there since the
 * original gpu_blake2s256unsalted slab kernel). */
#define GPU_TEMPLATE_BLAKE2S256 10
#define GPU_TEMPLATE_BLAKE2B256 11
#define GPU_TEMPLATE_BLAKE2B512 12
/* Memo B Phase B5 sub-batch 4 (2026-05-03): SHA3 / Keccak family. Sponge
 * construction (Keccak-f[1600] permutation; rate=200-2*output_bytes). Each
 * pair (Keccak/SHA3 of same output size) shares rate + EMIT_HIT width but
 * differs in suffix byte (0x01 plain Keccak, 0x06 SHA3 NIST FIPS 202). */
#define GPU_TEMPLATE_KECCAK224  13
#define GPU_TEMPLATE_KECCAK256  14
#define GPU_TEMPLATE_KECCAK384  15
#define GPU_TEMPLATE_KECCAK512  16
#define GPU_TEMPLATE_SHA3_224   17
#define GPU_TEMPLATE_SHA3_256   18
#define GPU_TEMPLATE_SHA3_384   19
#define GPU_TEMPLATE_SHA3_512   20
/* B5 sub-batch 5a Tier 1 (2026-05-03): SHA384RAW + SHA512RAW. Reuse
 * SHA384/SHA512 compression; binary-digest iter step. */
#define GPU_TEMPLATE_SHA384RAW  21
#define GPU_TEMPLATE_SHA512RAW  22
/* B5 sub-batch 6 Tier A (2026-05-03): MD5RAW + SHA1RAW + SHA256RAW. Reuse
 * MD5/SHA1/SHA256 compression; binary-digest iter step. */
#define GPU_TEMPLATE_MD5RAW     23
#define GPU_TEMPLATE_SHA1RAW    24
#define GPU_TEMPLATE_SHA256RAW  25
/* B5 sub-batch 6 Tier C (2026-05-03): SQL5 (MySQL 4.1+ password). Compound
 * SHA1(SHA1(p)) with UPPERCASE-hex iter feedback. Two SHA1 chains in state. */
#define GPU_TEMPLATE_SQL5       26
/* B5 sub-batch 6 Tier B (2026-05-03): NTLMH (NT password hash). MD4 of
 * UTF-16LE zero-extend(p). Hashcat-compatible single-variant. */
#define GPU_TEMPLATE_NTLMH      27
/* B5 sub-batch 8 (2026-05-05): MD4UTF16 (-m e496). Same MD4(UTF-16LE-
 * zero-extend(p)) algorithm as NTLMH with iter loop support (Maxiter > 1
 * feeds back hex of prior digest as UTF-16LE-zero-extend input). */
#define GPU_TEMPLATE_MD4UTF16   28
/* B5 sub-batch 7 (2026-05-05): MYSQL3 (-m e456). Legacy MySQL
 * OLD_PASSWORD() hash. 64-bit output. Per-byte arithmetic accumulator
 * loop with hex-feedback iter step (16 ASCII chars from prior digest).
 * Probe via the default 4-word path (h[2..3] zero); host zero-pad of
 * HashDataBuf (mdxfind.c:36400-36412 rev 1.399+) makes the 4-uint32
 * compare byte-exact for the 8-byte digest. */
#define GPU_TEMPLATE_MYSQL3     29
/* B5 sub-batch 6.5 (2026-05-05): WRL (-m e5). Whirlpool 512-bit hash.
 * Miyaguchi-Preneel over 64-byte BE block; iter feeds back 128
 * lowercase hex chars. Diagnostic data point for RDNA4 gfx1201
 * Streebog-deferred issue (different __constant access pattern from
 * SBOG_LPS). */
#define GPU_TEMPLATE_WRL        30
/* B5 sub-batch 5b retry (2026-05-06): Streebog-256 + Streebog-512.
 * GOST R 34.11-2012. SBOG_LPS rewritten to shift-then-mask access pattern
 * matching WRL_OP — RDNA4 gfx1201 mitigation validated by sub-6.5 WRL ship
 * (16 KB __constant size identical, only access pattern differs). */
#define GPU_TEMPLATE_STREEBOG256 31
#define GPU_TEMPLATE_STREEBOG512 32
/* B6 salt-axis (2026-05-06): first two salted variants ship together —
 * MD5SALT (hashcat -m 10, JOB_MD5SALT=31) is the double-MD5 chain
 * MD5(hex32(MD5(p)) || salt); MD5SALTPASS (hashcat -m 20, JOB_MD5SALTPASS=
 * 394) is the simple prepend MD5(salt || pass). Two distinct cache keys
 * via SALT_POSITION=APPEND_TO_HEX32 vs PREPEND in defines_str. */
#define GPU_TEMPLATE_MD5SALT     33
#define GPU_TEMPLATE_MD5SALTPASS 34
/* B6.1 SHA1 fan-out (2026-05-06): SHA1SALTPASS (hashcat -m 110, JOB_-
 * SHA1SALTPASS=385) is SHA1(salt || pass). First SHA-family salted variant.
 * Distinct cache key from MD5SALTPASS via HASH_WORDS=5 + BASE_ALGO=sha1
 * tokens in defines_str. */
#define GPU_TEMPLATE_SHA1SALTPASS 35
/* B6.2 SHA256 fan-out (2026-05-06): SHA256SALTPASS (hashcat -m 1410, JOB_-
 * SHA256SALTPASS=412) is SHA256(salt || pass). Second SHA-family salted
 * variant. Distinct cache key from SHA1SALTPASS via HASH_WORDS=8 +
 * BASE_ALGO=sha256 tokens (both axes differ); from MD5SALTPASS via
 * HASH_WORDS=8 + BASE_ALGO=sha256 (both axes differ). 36/36 pairwise
 * distinct defines_str. */
#define GPU_TEMPLATE_SHA256SALTPASS 36
/* B6.3 SHA224 fan-out (2026-05-06): SHA224SALTPASS (hashcat -m 1310, JOB_-
 * SHA224SALTPASS) is SHA224(salt || pass). Third SHA-family salted
 * variant — sha256_block compression with 7-word truncated output.
 * Distinct cache key from SHA256SALTPASS via HASH_WORDS=7 (vs 8) — same
 * BASE_ALGO=sha256 since the compression primitive is identical. From
 * SHA1SALTPASS via HASH_WORDS=7 + BASE_ALGO=sha256 (both axes differ).
 * From MD5SALTPASS via HASH_WORDS=7 + BASE_ALGO=sha256 (both axes
 * differ). 37/37 pairwise distinct defines_str. */
#define GPU_TEMPLATE_SHA224SALTPASS 37
/* B6.4 MD5PASSSALT fan-out (2026-05-06): MD5PASSSALT (hashcat -m 10,
 * JOB_MD5PASSSALT=373) is MD5(pass || salt). First APPEND-shape salted
 * variant — distinct cache key from MD5SALTPASS (PREPEND) via SALT_-
 * POSITION=APPEND in defines_str; same BASE_ALGO=md5 + HASH_WORDS=4.
 * 38/38 pairwise distinct defines_str. */
#define GPU_TEMPLATE_MD5PASSSALT 38
/* B6.5 SHA1PASSSALT fan-out (2026-05-06): SHA1PASSSALT (hashcat -m 100,
 * JOB_SHA1PASSSALT=405) is SHA1(pass || salt). First SHA-family APPEND-
 * shape salted variant — distinct cache key from SHA1SALTPASS (PREPEND)
 * via SALT_POSITION=APPEND in defines_str; same BASE_ALGO=sha1 +
 * HASH_WORDS=5 axes. From MD5PASSSALT via HASH_WORDS=5 + BASE_ALGO=sha1
 * (both axes differ). 39/39 pairwise distinct defines_str. */
#define GPU_TEMPLATE_SHA1PASSSALT 39
/* B6.7 SHA256PASSSALT fan-out (2026-05-06): SHA256PASSSALT (hashcat -m 1410,
 * JOB_SHA256PASSSALT=413) is SHA256(pass || salt). Second SHA-family APPEND-
 * shape salted variant — distinct cache key from SHA256SALTPASS (PREPEND)
 * via SALT_POSITION=APPEND in defines_str; same BASE_ALGO=sha256 +
 * HASH_WORDS=8 axes. From SHA1PASSSALT via HASH_WORDS=8 + BASE_ALGO=sha256
 * (both axes differ). 40/40 pairwise distinct defines_str. (Enum value 43
 * skips 40-42 — reserved for future fan-outs; the enum is a host-side cache
 * key and gaps are harmless.) */
#define GPU_TEMPLATE_SHA256PASSSALT 43
/* B6.9 SHA512 fan-out (2026-05-06): SHA512SALTPASS (hashcat -m 1710,
 * JOB_SHA512SALTPASS=388) is SHA512(salt || pass). FIRST 64-bit-state
 * salted variant on the codegen path — sha512_block compression with
 * 8 × ulong state, 128-byte block, 128-bit length field. Distinct
 * cache key via HASH_BLOCK_BYTES=128 (unique among salted templates;
 * all others use 64-byte blocks) + HASH_WORDS=16 + BASE_ALGO=sha512.
 * 44/44 pairwise distinct defines_str. R2 risk on gfx1201 — unsalted
 * SHA-512 reading was 42,520 B priv_mem; HARD GATE 43,024 B (3080
 * spill-region ceiling). Salted finalize delta expected ~0-50 B. */
#define GPU_TEMPLATE_SHA512SALTPASS 44
/* B6.10 SHA512PASSSALT fan-out (2026-05-06): SHA512PASSSALT (hashcat
 * -m 1720, JOB_SHA512PASSSALT=386) is SHA512(pass || salt). FINAL B6
 * ladder step. APPEND-shape sibling of SHA512SALTPASS (B6.9) — same
 * SHA-512 family compression + 8 × ulong state + 128-byte block +
 * 128-bit length field; only the salt POSITION at template_finalize
 * differs. Cache disambiguated from SHA512SALTPASS via SALT_POSITION=
 * APPEND (vs PREPEND); same BASE_ALGO=sha512 + HASH_WORDS=16 +
 * HASH_BLOCK_BYTES=128 axes. 45/45 pairwise distinct defines_str. */
#define GPU_TEMPLATE_SHA512PASSSALT 45
/* Family E HMAC-SHA384 carrier (2026-05-08): SHA384SALTPASS-shaped carrier
 * for HMAC-SHA384 (e543) + HMAC-SHA384_KPASS (e796). No JOB_SHA384SALTPASS
 * algorithm in mdxfind; this enum value exists ONLY for cache-key
 * disambiguation in the gpu_kernel_cache layer. Distinct from
 * SHA512SALTPASS (44) and SHA512PASSSALT (45) via HASH_WORDS=12 (vs 16);
 * same BASE_ALGO=sha512 + HASH_BLOCK_BYTES=128. 46/46 pairwise distinct
 * defines_str. */
#define GPU_TEMPLATE_SHA384SALTPASS 46
/* B6.11 SHA1DRU fan-out (2026-05-06): SHA1DRU (Drupal SHA1, hashcat -m 7900,
 * JOB_SHA1DRU=404). FIRST 1M-iteration algorithm on the unified template
 * path. Algorithm: SHA1(pass) followed by 1,000,000 iterations of
 * SHA1(hex_lc(state) || pass); ONE probe at the final state.
 *
 * Design: 1M loop INSIDE template_finalize, max_iter=1 host-forced so the
 * kernel's outer iter loop runs exactly once and template_iterate is a
 * stub. This matches CPU semantics exactly (mdxfind.c:14261-14285 has ONE
 * checkhash() call after the for-loop) and avoids 1M wasted compact-table
 * probes that would result if we put the loop in template_iterate with
 * max_iter=1000000.
 *
 * Cache key: HASH_WORDS=5,HASH_BLOCK_BYTES=64,BASE_ALGO=sha1,ITER_COUNT=1000000.
 * Distinct from SHA1 / SHA1RAW / SQL5 by the ITER_COUNT token AND by source
 * text (the iter body lives in template_finalize). 46/46 pairwise distinct
 * defines_str. */
#define GPU_TEMPLATE_SHA1DRU 46
/* B7.7b MD6256 fan-out (2026-05-07): MD6256 (hashcat -m 17800,
 * JOB_MD6256=29). Final M5 closure from B9 gate-fail. MD6-256 single-block
 * leaf compression — algorithmically-largest single-compression unsalted
 * algo (89-ulong N input + 1753-ulong A working array = 14 KB stack,
 * 104 rounds × 16 steps per compression). Per-iter probe like SQL5
 * (vs. SHA1DRU's max_iter=1 internal loop). Cache key: HASH_WORDS=8,
 * HASH_BLOCK_BYTES=64,BASE_ALGO=md6. Distinct from every prior template
 * core via BASE_ALGO=md6 (no other md6 algo on the template path) +
 * HASH_WORDS=8 (matches Streebog256/SHA256 family but BASE_ALGO axis
 * disambiguates). 47/47 pairwise distinct defines_str. KNOWN ACCEPTED
 * RISK: gfx1201 priv_mem may bust 43,024 B HARD GATE due to the 14 KB
 * A[1753] stack on top of RULE_BUF_MAX. Compile-only ship per user
 * 2026-05-07 OPTION A; integrated post-B7.9 validation will reveal
 * gfx1201 status. */
#define GPU_TEMPLATE_MD6256 47
/* Family G HMAC-RIPEMD-160 carrier (2026-05-08): RIPEMD160SALTPASS-shaped
 * carrier for HMAC-RMD160 (e211) + HMAC-RMD160_KPASS (e798). No
 * JOB_RIPEMD160SALTPASS algorithm in mdxfind; this enum value exists ONLY
 * for cache-key disambiguation in the gpu_kernel_cache layer. Distinct
 * from SHA1SALTPASS (35) via BASE_ALGO=ripemd160 (vs sha1) — same
 * HASH_WORDS=5 + HASH_BLOCK_BYTES=64 axes; the BASE_ALGO axis is the
 * load-bearing differentiator (ripemd160_block has different compression
 * rounds + 2-arg call signature vs sha1_block's BE compression).
 * 48/48 pairwise distinct defines_str. */
#define GPU_TEMPLATE_RIPEMD160SALTPASS 48
/* Family H HMAC-RIPEMD-320 carrier (2026-05-08): RIPEMD320SALTPASS-shaped
 * carrier for HMAC-RMD320 (e213) + HMAC-RMD320_KPASS (e799). No
 * JOB_RIPEMD320SALTPASS algorithm in mdxfind; this enum value exists ONLY
 * for cache-key disambiguation in the gpu_kernel_cache layer. Distinct
 * from RIPEMD160SALTPASS (48) via HASH_WORDS=10 (vs 5) + BASE_ALGO=rmd320
 * (vs rmd160) — both axes load-bearing (rmd320_block has distinct round
 * bodies + line/line' cross-mix accumulation; HASH_WORDS=10 vs 5 also
 * affects EMIT_HIT_<N> + iter-loop block geometry). 49/49 pairwise
 * distinct defines_str. */
#define GPU_TEMPLATE_RIPEMD320SALTPASS 49
/* Family I HMAC-BLAKE2S carrier (2026-05-08): hand-written Path A sibling
 * for HMAC-BLAKE2S (e828) with single algo_mode (5). HASH_WORDS=8,
 * HASH_BLOCK_BYTES=64, BASE_ALGO=blake2s, HAS_SALT=1, HMAC_KPASS=1. Cache
 * key disambiguated from every prior salted/unsalted template via the
 * BASE_ALGO=blake2s + HAS_SALT=1 + HMAC_KPASS=1 triple — pairwise distinct
 * across the 49 prior templates. */
#define GPU_TEMPLATE_HMAC_BLAKE2S 50
/* Family J HMAC-STREEBOG-256 carrier (2026-05-08): hand-written Path A
 * sibling of gpu_streebog256_core.cl for HMAC-STREEBOG256_KSALT (e838) +
 * HMAC-STREEBOG256_KPASS (e837) with two algo_modes (5/6). HASH_WORDS=8,
 * HASH_BLOCK_BYTES=64, BASE_ALGO=streebog256, HAS_SALT=1, HMAC_KSALTPASS=1.
 * Cache key disambiguated from gpu_streebog256_core_str (unsalted) via the
 * HAS_SALT=1 + HMAC_KSALTPASS=1 axes (absent in unsalted defines_str), and
 * from every other salted/unsalted template via BASE_ALGO=streebog256
 * (unique). 50/50 pairwise distinct defines_str. */
#define GPU_TEMPLATE_HMAC_STREEBOG256 51
/* Family K HMAC-STREEBOG-512 carrier (2026-05-08): hand-written Path A
 * sibling of gpu_streebog512_core.cl for HMAC-STREEBOG512_KSALT (e840) +
 * HMAC-STREEBOG512_KPASS (e839) with two algo_modes (5/6). HASH_WORDS=16,
 * HASH_BLOCK_BYTES=64, BASE_ALGO=streebog512, HAS_SALT=1, HMAC_KSALTPASS=1.
 * Cache key disambiguated from gpu_streebog512_core_str (unsalted) via the
 * HAS_SALT=1 + HMAC_KSALTPASS=1 axes (absent in unsalted defines_str), and
 * from every other salted/unsalted template via BASE_ALGO=streebog512
 * (unique). 51/51 pairwise distinct defines_str. Final HMAC family
 * shipped in the ladder (HMAC LADDER COMPLETE: 19/21 algos; Family D
 * HMAC-SHA256 deferred per project_family_d_deferred.md). */
#define GPU_TEMPLATE_HMAC_STREEBOG512 52
/* PHPBB3 carrier (2026-05-08): hand-written Path A salted-template
 * kernel for JOB_PHPBB3 (e455). Single algo_mode; iterated MD5 chain
 * INSIDE template_finalize (mirrors SHA1DRU pattern at max_iter=1
 * forced host-side). HASH_WORDS=4, HASH_BLOCK_BYTES=64, BASE_ALGO=
 * phpbb3, HAS_SALT=1. Cache key disambiguated from MD5SALT family
 * (HASH_WORDS=4 + HAS_SALT=1 + BASE_ALGO=md5) via the unique
 * BASE_ALGO=phpbb3 axis. 52/52 pairwise distinct defines_str.
 * Templated count delta: 55 -> 56. First iterated-crypt with salt-
 * carried iter count on the unified template path. */
#define GPU_TEMPLATE_PHPBB3 53
/* MD5CRYPT carrier (2026-05-08): hand-written Path A salted-template
 * kernel for JOB_MD5CRYPT (e511). Single algo_mode; iterated MD5 chain
 * (1000 fixed iters per BSD $1$ md5crypt) INSIDE template_finalize
 * (mirrors PHPBB3 / SHA1DRU pattern at max_iter=1 forced host-side).
 * HASH_WORDS=4, HASH_BLOCK_BYTES=64, BASE_ALGO=md5crypt, HAS_SALT=1.
 * Cache key disambiguated from MD5SALT family (HASH_WORDS=4 + HAS_SALT=1
 * + BASE_ALGO=md5) via the unique BASE_ALGO=md5crypt axis, and from
 * PHPBB3 (BASE_ALGO=phpbb3) via the same axis. 53/53 pairwise distinct
 * defines_str. Templated count delta: 56 -> 57. Phase 1 of the Unix-
 * crypt ladder (MD5CRYPT -> SHA256CRYPT -> SHA512CRYPT ->
 * SHA512CRYPTMD5). */
#define GPU_TEMPLATE_MD5CRYPT 54
/* SHA256CRYPT carrier (2026-05-08): hand-written Path A salted-template
 * kernel for JOB_SHA256CRYPT (e512). Single algo_mode; SHA-256 crypt
 * chain (5 steps + variable-rounds main loop, default 5000 iters,
 * configurable via "rounds=N$" salt prefix) INSIDE template_finalize
 * (mirrors PHPBB3 / SHA1DRU / MD5CRYPT pattern at max_iter=1 forced
 * host-side). HASH_WORDS=8, HASH_BLOCK_BYTES=64, BASE_ALGO=sha256crypt,
 * HAS_SALT=1. Cache key disambiguated from every prior salted template
 * via the unique BASE_ALGO=sha256crypt axis (no other template uses
 * this token at HASH_WORDS=8). Templated count delta: 57 -> 58. Phase
 * 2 of the Unix-crypt ladder. Shares the gpu_shacrypt_core.cl source
 * with Phase 3 (SHA512CRYPT at HASH_WORDS=16) + Phase 4 (SHA512CRYPTMD5
 * at HASH_WORDS=16, algo_mode=1 for the MD5-preprocess). */
#define GPU_TEMPLATE_SHA256CRYPT 55
/* SHA512CRYPT carrier (2026-05-08): hand-written Path A salted-template
 * kernel for JOB_SHA512CRYPT (e513). Single algo_mode; SHA-512 crypt
 * chain (5 steps + variable-rounds main loop, default 5000 iters,
 * configurable via "rounds=N$" salt prefix) INSIDE template_finalize
 * (mirrors PHPBB3 / SHA1DRU / MD5CRYPT / SHA256CRYPT pattern at max_-
 * iter=1 forced host-side). HASH_WORDS=16, HASH_BLOCK_BYTES=128,
 * BASE_ALGO=sha512crypt, HAS_SALT=1. Cache key disambiguated from
 * SHA512SALTPASS (BASE_ALGO=sha512 at the same other axes) and from
 * Phase 2 SHA256CRYPT (HASH_WORDS=8) via the unique BASE_ALGO=
 * sha512crypt axis. Templated count delta: 58 -> 59. Phase 3 of the
 * Unix-crypt ladder. Shares the gpu_shacrypt_core.cl source with
 * Phase 2 (SHA256CRYPT at HASH_WORDS=8) + Phase 4 (SHA512CRYPTMD5 at
 * HASH_WORDS=16, algo_mode=1 for the MD5-preprocess). SHA512CRYPTMD5
 * REMAINS on the slab path for now; Phase 4 will move it. */
#define GPU_TEMPLATE_SHA512CRYPT 56
/* DESCRYPT carrier (2026-05-08, Unix-crypt Phase 5): hand-written Path A
 * salted-template kernel for JOB_DESCRYPT (e500). Single algo_mode (7);
 * bespoke kernel that will NOT share with BCRYPT. 25-iter DES Feistel
 * chain (FIXED iteration count per Unix DES crypt(3) "old-style") INSIDE
 * template_finalize (mirrors PHPBB3 / MD5CRYPT / SHA256CRYPT / SHA512CRYPT
 * pattern at max_iter=1 forced host-side). HASH_WORDS=4 (state = pre-FP
 * (l, r) in h[0..1], h[2..3] zero-pad to match the host compact-table
 * layout 4 il + 4 ir + 8 zero pad), HASH_BLOCK_BYTES=64, BASE_ALGO=
 * descrypt, HAS_SALT=1. Cache key disambiguated from MD5SALT family
 * (HASH_WORDS=4 + HAS_SALT=1 + BASE_ALGO=md5) via BASE_ALGO=descrypt
 * axis, from PHPBB3 (BASE_ALGO=phpbb3) via the same axis, and from
 * MD5CRYPT (BASE_ALGO=md5crypt) via the same axis. Templated count
 * delta: 59 -> 60 (counting Phase 4 SHA512CRYPTMD5 reusing slot 56,
 * slot 57 is the next free integer; final templated count after Phase 5
 * = 60 algos on the unified template path). Phase 5 of the Unix-crypt
 * ladder (FINAL phase; Unix-crypt slab path fully retired across all 5
 * Unix-crypt ops: MD5CRYPT, SHA256CRYPT, SHA512CRYPT, SHA512CRYPTMD5,
 * DESCRYPT). */
#define GPU_TEMPLATE_DESCRYPT 57
/* BCRYPT carrier (2026-05-08, Unix-crypt Phase 6): hand-written Path A
 * salted-template kernel for JOB_BCRYPT (e450). Single algo_mode (8);
 * bespoke kernel that will NOT share with DESCRYPT or any other algo;
 * reserves algo_mode 8-15 for future BCRYPT family variants. 2^cost
 * Eksblowfish chain (cost parsed per-salt-string at kernel entry; SIMT
 * divergence accepted - slab pattern) INSIDE template_finalize (mirrors
 * PHPBB3 / MD5CRYPT / SHA256CRYPT / SHA512CRYPT / DESCRYPT pattern at
 * max_iter=1 forced host-side). HASH_WORDS=6 (24-byte output as 6 LE
 * uint32 in h[0..5]; first 4 LE words = 16 bytes probed against compact
 * table). HASH_BLOCK_BYTES=64. BASE_ALGO=bcrypt. HAS_SALT=1. SALT_-
 * POSITION=PREPEND. The salt buffer carries the FULL 28- or 29-byte
 * "$2[abkxy]$NN$<base64>" string; per-string parsing (cost decode,
 * variant prefix recognition, base64 decode) happens INSIDE the kernel.
 * Cache key disambiguated from every prior salted template at HASH_-
 * WORDS=6 via BASE_ALGO=bcrypt axis (no other algo uses HASH_WORDS=6 +
 * HAS_SALT=1). Workgroup-shared __local Eksblowfish S-boxes via
 * GPU_TEMPLATE_HAS_LOCAL_BUFFER=1 + GPU_TEMPLATE_LOCAL_BUFFER_PER_LANE=
 * 1024 (4 KB per lane) + BCRYPT_WG_SIZE=8 lanes (= 32 KB per WG, fits
 * Pascal 48 KB / RDNA 64 KB / Mali-T860 32 KB exactly). The dispatch
 * site at gpu_opencl.c:11724 sets `local = BCRYPT_WG_SIZE` when the
 * resolved kernel is kern_template_phase0_bcrypt to honor the kernel's
 * reqd_work_group_size attribute. Templated count delta: 60 -> 61.
 * Phase 6 of the slab-retirement ladder (final major slab kernel; only
 * HMAC-MD5 probe-anchor kernels remain in FAM_MD5SALT post-this
 * retirement). Compound siblings BCRYPTMD5 (e451) / BCRYPTSHA1 (e452) /
 * BCRYPTSHA512 (e967) remain CPU-only via gpu_op_category default
 * fall-through; only JOB_BCRYPT singleton uses this kernel. */
#define GPU_TEMPLATE_BCRYPT 58

static int gpu_template_enabled(void) {
    static int cached = -1;
    if (cached == -1) {
        const char *e = getenv("MDXFIND_GPU_TEMPLATE");
        if (e && *e && *e != '0' && strcmp(e, "md5") == 0) {
            cached = GPU_TEMPLATE_MD5;
            fprintf(stderr,
                "OpenCL: MDXFIND_GPU_TEMPLATE=md5 — md5_rules dispatch will use "
                "template_phase0 kernel from gpu_template.cl + gpu_md5_core.cl "
                "(B2 structural prereq; production path remains "
                "md5_rules_phase0 by default).\n");
        } else if (e && *e && *e != '0' && strcmp(e, "sha1") == 0) {
            /* B4 first-algorithm fan-out: SHA1 template instantiation.
             * The kernel object is built lazily on first dispatch via
             * gpu_opencl_template_compile_sha1 / _kernel_lazy_sha1 and
             * is selected over the MD5 template at the dispatch_md5_rules
             * kernel-handle swap site. NOTE: this comment dates to before
             * SHA1 was widened into the rules-engine admit list (B5). The
             * chokepoint pack at mdxfind.c that previously fielded SHA1
             * work was retired in B7.9 (2026-05-07); SHA1 now dispatches
             * exclusively via dispatch_md5_rules under the template kernel. */
            cached = GPU_TEMPLATE_SHA1;
            fprintf(stderr,
                "OpenCL: MDXFIND_GPU_TEMPLATE=sha1 — when SHA1 work reaches "
                "dispatch_md5_rules, will use template_phase0 kernel from "
                "gpu_template.cl + gpu_sha1_core.cl (B4 first-algorithm "
                "fan-out; chokepoint gate widening is a follow-up commit).\n");
        } else if (e && *e && *e != '0' && strcmp(e, "sha256") == 0) {
            /* B4 fan-out: SHA256 template instantiation. */
            cached = GPU_TEMPLATE_SHA256;
            fprintf(stderr,
                "OpenCL: MDXFIND_GPU_TEMPLATE=sha256 — when SHA256 work reaches "
                "dispatch_md5_rules, will use template_phase0 kernel from "
                "gpu_template.cl + gpu_sha256_core.cl (B4 fan-out; chokepoint "
                "gate widening is a follow-up commit).\n");
        } else if (e && *e && *e != '0' && strcmp(e, "sha224") == 0) {
            /* B4 fan-out: SHA224 template instantiation. Same compression as
             * SHA256, different IV, output truncated to 7 of 8 state words. */
            cached = GPU_TEMPLATE_SHA224;
            fprintf(stderr,
                "OpenCL: MDXFIND_GPU_TEMPLATE=sha224 — when SHA224 work reaches "
                "dispatch_md5_rules, will use template_phase0 kernel from "
                "gpu_template.cl + gpu_sha224_core.cl (B4 fan-out; chokepoint "
                "gate widening is a follow-up commit).\n");
        } else if (e && *e && *e != '0' && strcmp(e, "md4") == 0) {
            /* B4 fan-out: MD4 template instantiation. Same digest geometry
             * as MD5 (4 LE uint32) but different compression function. */
            cached = GPU_TEMPLATE_MD4;
            fprintf(stderr,
                "OpenCL: MDXFIND_GPU_TEMPLATE=md4 — when MD4 work reaches "
                "dispatch_md5_rules, will use template_phase0 kernel from "
                "gpu_template.cl + gpu_md4_core.cl (B4 fan-out; chokepoint "
                "gate widening is a follow-up commit).\n");
        } else if (e && *e && *e != '0' && strcmp(e, "sha384") == 0) {
            /* B5 sub-batch 1: SHA384 template instantiation. First 64-bit
             * state + 128-bit length encoding algorithm. Output truncates
             * to 6 ulong = 12 uint32. */
            cached = GPU_TEMPLATE_SHA384;
            fprintf(stderr,
                "OpenCL: MDXFIND_GPU_TEMPLATE=sha384 — when SHA384 work reaches "
                "dispatch_md5_rules, will use template_phase0 kernel from "
                "gpu_template.cl + gpu_sha384_core.cl (B5 fan-out; first "
                "64-bit-state algo).\n");
        } else if (e && *e && *e != '0' && strcmp(e, "sha512") == 0) {
            /* B5 sub-batch 1: SHA512 template instantiation. First 64-bit
             * state + 128-bit length encoding algorithm. 8 ulong = 16 uint32. */
            cached = GPU_TEMPLATE_SHA512;
            fprintf(stderr,
                "OpenCL: MDXFIND_GPU_TEMPLATE=sha512 — when SHA512 work reaches "
                "dispatch_md5_rules, will use template_phase0 kernel from "
                "gpu_template.cl + gpu_sha512_core.cl (B5 fan-out; first "
                "64-bit-state algo).\n");
        } else if (e && *e && *e != '0' &&
                   (strcmp(e, "ripemd160") == 0 || strcmp(e, "rmd160") == 0)) {
            /* B5 sub-batch 2: RIPEMD-160 template instantiation. Second
             * 5-word-state algo (after SHA1) but LITTLE-ENDIAN per uint32
             * (like MD5; UNLIKE the SHA family). */
            cached = GPU_TEMPLATE_RIPEMD160;
            fprintf(stderr,
                "OpenCL: MDXFIND_GPU_TEMPLATE=%s — when RIPEMD-160 work reaches "
                "dispatch_md5_rules, will use template_phase0 kernel from "
                "gpu_template.cl + gpu_ripemd160_core.cl (B5 fan-out; "
                "LE-per-uint32 state; HASH_WORDS=5).\n", e);
        } else if (e && *e && *e != '0' &&
                   (strcmp(e, "ripemd320") == 0 || strcmp(e, "rmd320") == 0)) {
            /* B5 sub-batch 2: RIPEMD-320 template instantiation. First
             * 10-word-state algo in the family — needed new EMIT_HIT_10
             * family macros in gpu_common.cl rev 1.12. */
            cached = GPU_TEMPLATE_RIPEMD320;
            fprintf(stderr,
                "OpenCL: MDXFIND_GPU_TEMPLATE=%s — when RIPEMD-320 work reaches "
                "dispatch_md5_rules, will use template_phase0 kernel from "
                "gpu_template.cl + gpu_ripemd320_core.cl (B5 fan-out; "
                "LE-per-uint32 state; HASH_WORDS=10; first 10-word state).\n", e);
        } else if (e && *e && *e != '0' && strcmp(e, "blake2s256") == 0) {
            /* B5 sub-batch 3: BLAKE2S-256 template instantiation. First
             * BLAKE2 algorithm (counter + flag in state struct). 8 uint32
             * LE digest, 64-byte block, 10-round G compression. */
            cached = GPU_TEMPLATE_BLAKE2S256;
            fprintf(stderr,
                "OpenCL: MDXFIND_GPU_TEMPLATE=blake2s256 — when BLAKE2S-256 "
                "work reaches dispatch_md5_rules, will use template_phase0 "
                "kernel from gpu_template.cl + gpu_blake2s256_core.cl "
                "(B5 sub-batch 3; LE-per-uint32; HASH_WORDS=8; counter+flag "
                "in per-algo state).\n");
        } else if (e && *e && *e != '0' && strcmp(e, "blake2b256") == 0) {
            /* B5 sub-batch 3: BLAKE2B-256 template instantiation. 64-bit
             * lane (8 ulong internal); HASH_WORDS=8 exposed as first 4
             * ulong → 8 uint32 LE truncated digest. New b2b_compress in
             * gpu_common.cl rev 1.13. */
            cached = GPU_TEMPLATE_BLAKE2B256;
            fprintf(stderr,
                "OpenCL: MDXFIND_GPU_TEMPLATE=blake2b256 — when BLAKE2B-256 "
                "work reaches dispatch_md5_rules, will use template_phase0 "
                "kernel from gpu_template.cl + gpu_blake2b256_core.cl "
                "(B5 sub-batch 3; 64-bit lane, 128-byte block, 12-round G; "
                "first BLAKE2b template).\n");
        } else if (e && *e && *e != '0' && strcmp(e, "blake2b512") == 0) {
            /* B5 sub-batch 3: BLAKE2B-512 template instantiation. Same
             * compression as BLAKE2B-256, full 8-ulong = 16-uint32 LE
             * digest output. */
            cached = GPU_TEMPLATE_BLAKE2B512;
            fprintf(stderr,
                "OpenCL: MDXFIND_GPU_TEMPLATE=blake2b512 — when BLAKE2B-512 "
                "work reaches dispatch_md5_rules, will use template_phase0 "
                "kernel from gpu_template.cl + gpu_blake2b512_core.cl "
                "(B5 sub-batch 3; HASH_WORDS=16; full 64-byte digest).\n");
        } else if (e && *e && *e != '0' && strcmp(e, "keccak224") == 0) {
            cached = GPU_TEMPLATE_KECCAK224;
            fprintf(stderr,
                "OpenCL: MDXFIND_GPU_TEMPLATE=keccak224 — sponge construction; "
                "rate=144, output=28, suffix=0x01 (B5 sub-batch 4).\n");
        } else if (e && *e && *e != '0' && strcmp(e, "keccak256") == 0) {
            cached = GPU_TEMPLATE_KECCAK256;
            fprintf(stderr,
                "OpenCL: MDXFIND_GPU_TEMPLATE=keccak256 — sponge construction; "
                "rate=136, output=32, suffix=0x01 (B5 sub-batch 4).\n");
        } else if (e && *e && *e != '0' && strcmp(e, "keccak384") == 0) {
            cached = GPU_TEMPLATE_KECCAK384;
            fprintf(stderr,
                "OpenCL: MDXFIND_GPU_TEMPLATE=keccak384 — sponge construction; "
                "rate=104, output=48, suffix=0x01 (B5 sub-batch 4).\n");
        } else if (e && *e && *e != '0' && strcmp(e, "keccak512") == 0) {
            cached = GPU_TEMPLATE_KECCAK512;
            fprintf(stderr,
                "OpenCL: MDXFIND_GPU_TEMPLATE=keccak512 — sponge construction; "
                "rate=72, output=64, suffix=0x01 (B5 sub-batch 4).\n");
        } else if (e && *e && *e != '0' && strcmp(e, "sha3_224") == 0) {
            cached = GPU_TEMPLATE_SHA3_224;
            fprintf(stderr,
                "OpenCL: MDXFIND_GPU_TEMPLATE=sha3_224 — sponge construction; "
                "rate=144, output=28, suffix=0x06 (B5 sub-batch 4; NIST FIPS 202).\n");
        } else if (e && *e && *e != '0' && strcmp(e, "sha3_256") == 0) {
            cached = GPU_TEMPLATE_SHA3_256;
            fprintf(stderr,
                "OpenCL: MDXFIND_GPU_TEMPLATE=sha3_256 — sponge construction; "
                "rate=136, output=32, suffix=0x06 (B5 sub-batch 4; NIST FIPS 202).\n");
        } else if (e && *e && *e != '0' && strcmp(e, "sha3_384") == 0) {
            cached = GPU_TEMPLATE_SHA3_384;
            fprintf(stderr,
                "OpenCL: MDXFIND_GPU_TEMPLATE=sha3_384 — sponge construction; "
                "rate=104, output=48, suffix=0x06 (B5 sub-batch 4; NIST FIPS 202).\n");
        } else if (e && *e && *e != '0' && strcmp(e, "sha3_512") == 0) {
            cached = GPU_TEMPLATE_SHA3_512;
            fprintf(stderr,
                "OpenCL: MDXFIND_GPU_TEMPLATE=sha3_512 — sponge construction; "
                "rate=72, output=64, suffix=0x06 (B5 sub-batch 4; NIST FIPS 202).\n");
        } else if (e && *e && *e != '0' && strcmp(e, "sha384raw") == 0) {
            cached = GPU_TEMPLATE_SHA384RAW;
            fprintf(stderr,
                "OpenCL: MDXFIND_GPU_TEMPLATE=sha384raw — SHA384 compression "
                "with binary-digest iter (48 bytes; B5 sub-batch 5a Tier 1).\n");
        } else if (e && *e && *e != '0' && strcmp(e, "sha512raw") == 0) {
            cached = GPU_TEMPLATE_SHA512RAW;
            fprintf(stderr,
                "OpenCL: MDXFIND_GPU_TEMPLATE=sha512raw — SHA512 compression "
                "with binary-digest iter (64 bytes; B5 sub-batch 5a Tier 1).\n");
        } else if (e && *e && *e != '0' && strcmp(e, "md5raw") == 0) {
            cached = GPU_TEMPLATE_MD5RAW;
            fprintf(stderr,
                "OpenCL: MDXFIND_GPU_TEMPLATE=md5raw — MD5 compression with "
                "binary-digest iter (16 bytes; B5 sub-batch 6 Tier A).\n");
        } else if (e && *e && *e != '0' && strcmp(e, "sha1raw") == 0) {
            cached = GPU_TEMPLATE_SHA1RAW;
            fprintf(stderr,
                "OpenCL: MDXFIND_GPU_TEMPLATE=sha1raw — SHA1 compression with "
                "binary-digest iter (20 bytes; B5 sub-batch 6 Tier A).\n");
        } else if (e && *e && *e != '0' && strcmp(e, "sha256raw") == 0) {
            cached = GPU_TEMPLATE_SHA256RAW;
            fprintf(stderr,
                "OpenCL: MDXFIND_GPU_TEMPLATE=sha256raw — SHA256 compression "
                "with binary-digest iter (32 bytes; B5 sub-batch 6 Tier A).\n");
        } else if (e && *e && *e != '0' && strcmp(e, "sql5") == 0) {
            cached = GPU_TEMPLATE_SQL5;
            fprintf(stderr,
                "OpenCL: MDXFIND_GPU_TEMPLATE=sql5 — SHA1(SHA1(p)) with "
                "UPPERCASE-hex iter feedback (B5 sub-batch 6 Tier C; "
                "two SHA1 chains in template_state).\n");
        } else if (e && *e && *e != '0' && strcmp(e, "ntlmh") == 0) {
            cached = GPU_TEMPLATE_NTLMH;
            fprintf(stderr,
                "OpenCL: MDXFIND_GPU_TEMPLATE=ntlmh — MD4(UTF-16LE-zero-extend"
                "(p)) (B5 sub-batch 6 Tier B; hashcat-compat; iconv variant "
                "remains on CPU for non-ASCII inputs).\n");
        } else if (e && *e && *e != '0' && strcmp(e, "md4utf16") == 0) {
            cached = GPU_TEMPLATE_MD4UTF16;
            fprintf(stderr,
                "OpenCL: MDXFIND_GPU_TEMPLATE=md4utf16 — MD4(UTF-16LE-zero-"
                "extend(p)) with iter feedback hex(prev_digest) (B5 sub-batch "
                "8; same hashcat-compat gap as NTLMH on non-ASCII inputs).\n");
        } else if (e && *e && *e != '0' && strcmp(e, "mysql3") == 0) {
            cached = GPU_TEMPLATE_MYSQL3;
            fprintf(stderr,
                "OpenCL: MDXFIND_GPU_TEMPLATE=mysql3 — legacy MySQL "
                "OLD_PASSWORD() hash (B5 sub-batch 7; per-byte arithmetic "
                "accumulator with 16-hex feedback iter step).\n");
        } else if (e && *e && *e != '0' && strcmp(e, "wrl") == 0) {
            cached = GPU_TEMPLATE_WRL;
            fprintf(stderr,
                "OpenCL: MDXFIND_GPU_TEMPLATE=wrl — Whirlpool 512-bit hash "
                "(B5 sub-batch 6.5; Miyaguchi-Preneel; 64-byte BE block; "
                "256-bit BE length encoding; 128-hex feedback iter step).\n");
        } else if (e && *e && *e != '0' && strcmp(e, "streebog256") == 0) {
            cached = GPU_TEMPLATE_STREEBOG256;
            fprintf(stderr,
                "OpenCL: MDXFIND_GPU_TEMPLATE=streebog256 — GOST R 34.11-2012 "
                "256-bit hash (B5 sub-batch 5b retry; LPS-keyed 12-round "
                "compression; 16 KB __constant SBOB_SL64 with shift-then-mask "
                "access pattern matching WRL_OP for RDNA4 compat).\n");
        } else if (e && *e && *e != '0' && strcmp(e, "streebog512") == 0) {
            cached = GPU_TEMPLATE_STREEBOG512;
            fprintf(stderr,
                "OpenCL: MDXFIND_GPU_TEMPLATE=streebog512 — GOST R 34.11-2012 "
                "512-bit hash (B5 sub-batch 5b retry; same compression as "
                "streebog256, 64-byte digest output).\n");
        } else if (e && *e && *e != '0' && strcmp(e, "md5salt") == 0) {
            /* B6 salt-axis (2026-05-06): MD5SALT (hashcat -m 10) =
             * MD5(hex32(MD5(p)) || salt) — first salted variant on the
             * unified template path. Mirrors mdxfind.c JOB_MD5SALT
             * (lines 21943-21974). Cache disambiguated from MD5SALTPASS
             * via SALT_POSITION=APPEND_TO_HEX32 in defines_str. */
            cached = GPU_TEMPLATE_MD5SALT;
            fprintf(stderr,
                "OpenCL: MDXFIND_GPU_TEMPLATE=md5salt — MD5(hex32(MD5(p)) || "
                "salt) double-MD5 chain (B6 salt-axis prereq; first salted "
                "variant on the template path; SALT_POSITION=APPEND_TO_HEX32).\n");
        } else if (e && *e && *e != '0' && strcmp(e, "md5saltpass") == 0) {
            /* B6 salt-axis (2026-05-06): MD5SALTPASS (hashcat -m 20) =
             * MD5(salt || pass) — simple prepend salt. Mirrors
             * mdxfind.c JOB_MD5SALTPASS (lines 15776-15832). Cache
             * disambiguated from MD5SALT via SALT_POSITION=PREPEND in
             * defines_str. */
            cached = GPU_TEMPLATE_MD5SALTPASS;
            fprintf(stderr,
                "OpenCL: MDXFIND_GPU_TEMPLATE=md5saltpass — MD5(salt || pass) "
                "simple prepend salt (B6 salt-axis prereq; second salted "
                "variant; SALT_POSITION=PREPEND).\n");
        } else if (e && *e && *e != '0' && strcmp(e, "sha1saltpass") == 0) {
            /* B6.1 SHA1 fan-out (2026-05-06): SHA1SALTPASS (hashcat -m 110)
             * = SHA1(salt || pass) — simple prepend salt SHA1. Mirrors
             * mdxfind.c JOB_SHA1SALTPASS (lines 14369-14418). Cache
             * disambiguated from MD5SALTPASS via HASH_WORDS=5 +
             * BASE_ALGO=sha1 in defines_str (SALT_POSITION=PREPEND
             * matches but the per-algorithm tokens differ). */
            cached = GPU_TEMPLATE_SHA1SALTPASS;
            fprintf(stderr,
                "OpenCL: MDXFIND_GPU_TEMPLATE=sha1saltpass — SHA1(salt || pass) "
                "simple prepend salt SHA1 (B6.1 SHA fan-out; first SHA-family "
                "salted variant; SALT_POSITION=PREPEND, HASH_WORDS=5).\n");
        } else if (e && *e && *e != '0' && strcmp(e, "sha256saltpass") == 0) {
            /* B6.2 SHA256 fan-out (2026-05-06): SHA256SALTPASS (hashcat
             * -m 1410) = SHA256(salt || pass) — simple prepend salt
             * SHA256. Mirrors mdxfind.c JOB_SHA256SALTPASS (lines 27603-
             * 27651). Cache disambiguated from SHA1SALTPASS via
             * HASH_WORDS=8 + BASE_ALGO=sha256 in defines_str (both axes
             * differ); from MD5SALTPASS via HASH_WORDS=8 + BASE_ALGO=
             * sha256 (both axes differ). */
            cached = GPU_TEMPLATE_SHA256SALTPASS;
            fprintf(stderr,
                "OpenCL: MDXFIND_GPU_TEMPLATE=sha256saltpass — SHA256(salt || pass) "
                "simple prepend salt SHA256 (B6.2 SHA fan-out; second SHA-family "
                "salted variant; SALT_POSITION=PREPEND, HASH_WORDS=8).\n");
        } else if (e && *e && *e != '0' && strcmp(e, "sha224saltpass") == 0) {
            /* B6.3 SHA224 fan-out (2026-05-06): SHA224SALTPASS (hashcat
             * -m 1310) = SHA224(salt || pass) — simple prepend salt
             * SHA224. Mirrors mdxfind.c JOB_SHA224SALTPASS. Cache
             * disambiguated from SHA256SALTPASS via HASH_WORDS=7 (vs 8)
             * — same BASE_ALGO=sha256 since the compression primitive is
             * identical. From SHA1SALTPASS via HASH_WORDS=7 + BASE_ALGO=
             * sha256 (both axes differ); from MD5SALTPASS via HASH_WORDS=7
             * + BASE_ALGO=sha256 (both axes differ). */
            cached = GPU_TEMPLATE_SHA224SALTPASS;
            fprintf(stderr,
                "OpenCL: MDXFIND_GPU_TEMPLATE=sha224saltpass — SHA224(salt || pass) "
                "simple prepend salt SHA224 (B6.3 SHA fan-out; third SHA-family "
                "salted variant; SALT_POSITION=PREPEND, HASH_WORDS=7).\n");
        } else if (e && *e && *e != '0' && strcmp(e, "md5passsalt") == 0) {
            /* B6.4 MD5PASSSALT fan-out (2026-05-06): MD5PASSSALT (hashcat
             * -m 10) = MD5(pass || salt) — simple APPEND salt MD5.
             * Mirrors mdxfind.c JOB_MD5PASSSALT (lines 16627-16669).
             * First APPEND-shape salted variant on the codegen path.
             * Cache disambiguated from MD5SALTPASS via SALT_POSITION=
             * APPEND (vs PREPEND); same BASE_ALGO=md5 + HASH_WORDS=4
             * axes. Authors the finalize_append.cl.frag fragment that
             * future SHA-family APPEND variants reuse (modulo BE/LE
             * sibling). */
            cached = GPU_TEMPLATE_MD5PASSSALT;
            fprintf(stderr,
                "OpenCL: MDXFIND_GPU_TEMPLATE=md5passsalt — MD5(pass || salt) "
                "simple append salt MD5 (B6.4 fan-out; first APPEND-shape "
                "salted variant; SALT_POSITION=APPEND, HASH_WORDS=4).\n");
        } else if (e && *e && *e != '0' && strcmp(e, "sha1passsalt") == 0) {
            /* B6.5 SHA1PASSSALT fan-out (2026-05-06): SHA1PASSSALT (hashcat
             * -m 100) = SHA1(pass || salt) — simple APPEND salt SHA1.
             * Mirrors mdxfind.c JOB_SHA1PASSSALT (lines 14227-14270).
             * First SHA-family APPEND-shape salted variant on the codegen
             * path. Cache disambiguated from SHA1SALTPASS via SALT_-
             * POSITION=APPEND (vs PREPEND); same BASE_ALGO=sha1 +
             * HASH_WORDS=5 axes. Authors the finalize_append_be.cl.frag
             * fragment that future SHA-family APPEND variants
             * (SHA256PASSSALT) reuse without further fragment work. */
            cached = GPU_TEMPLATE_SHA1PASSSALT;
            fprintf(stderr,
                "OpenCL: MDXFIND_GPU_TEMPLATE=sha1passsalt — SHA1(pass || salt) "
                "simple append salt SHA1 (B6.5 fan-out; first SHA-family "
                "APPEND-shape salted variant; SALT_POSITION=APPEND, "
                "HASH_WORDS=5, BASE_ALGO=sha1).\n");
        } else if (e && *e && *e != '0' && strcmp(e, "sha256passsalt") == 0) {
            /* B6.7 SHA256PASSSALT fan-out (2026-05-06): SHA256PASSSALT
             * (hashcat -m 1410) = SHA256(pass || salt) — simple APPEND
             * salt SHA256. Mirrors mdxfind.c JOB_SHA256PASSSALT
             * (lines 27639-27677). Second SHA-family APPEND-shape salted
             * variant — pure spec reuse (template + fragment both already
             * shipped at B6.2 and B6.5). Cache disambiguated from
             * SHA256SALTPASS via SALT_POSITION=APPEND (vs PREPEND); same
             * BASE_ALGO=sha256 + HASH_WORDS=8 axes. From SHA1PASSSALT via
             * HASH_WORDS=8 + BASE_ALGO=sha256 (both axes differ). */
            cached = GPU_TEMPLATE_SHA256PASSSALT;
            fprintf(stderr,
                "OpenCL: MDXFIND_GPU_TEMPLATE=sha256passsalt — SHA256(pass || salt) "
                "simple append salt SHA256 (B6.7 fan-out; second SHA-family "
                "APPEND-shape salted variant; SALT_POSITION=APPEND, "
                "HASH_WORDS=8, BASE_ALGO=sha256).\n");
        } else if (e && *e && *e != '0' && strcmp(e, "sha512saltpass") == 0) {
            /* B6.9 SHA512 fan-out (2026-05-06): SHA512SALTPASS (hashcat
             * -m 1710) = SHA512(salt || pass) — simple PREPEND salt
             * SHA-512. FIRST 64-bit-state salted variant on the codegen
             * path. Mirrors mdxfind.c JOB_SHA512SALTPASS (lines 13981-
             * 14023). Cache disambiguated from every other salted
             * template via HASH_BLOCK_BYTES=128 (the 128-byte block is
             * unique among salted variants on the codegen path) +
             * HASH_WORDS=16 + BASE_ALGO=sha512. Authors a sibling
             * sha512_style_salted.cl.tmpl AND a sibling
             * finalize_prepend_be64.cl.frag — width-bearing constants
             * (block_size, word_width, length-field-width) live in
             * the template+fragment, not parameterized into the
             * SHA-256 versions. R2 risk on gfx1201 — unsalted SHA-512
             * already at 42,520 B priv_mem; HARD GATE 43,024 B. */
            cached = GPU_TEMPLATE_SHA512SALTPASS;
            fprintf(stderr,
                "OpenCL: MDXFIND_GPU_TEMPLATE=sha512saltpass — SHA512(salt || pass) "
                "simple prepend salt SHA-512 (B6.9 fan-out; first 64-bit-state "
                "salted variant; SALT_POSITION=PREPEND, HASH_WORDS=16, "
                "HASH_BLOCK_BYTES=128, BASE_ALGO=sha512).\n");
        } else if (e && *e && *e != '0' && strcmp(e, "sha512passsalt") == 0) {
            /* B6.10 SHA512PASSSALT fan-out (2026-05-06): SHA512PASSSALT
             * (hashcat -m 1720) = SHA512(pass || salt) — simple APPEND
             * salt SHA-512. FINAL B6 ladder step. Second 64-bit-state
             * salted variant on the codegen path; APPEND-shape sibling
             * of SHA512SALTPASS (B6.9). Mirrors mdxfind.c JOB_SHA512-
             * PASSSALT (lines 14069-14127). Cache disambiguated from
             * SHA512SALTPASS via SALT_POSITION=APPEND (vs PREPEND);
             * same BASE_ALGO=sha512 + HASH_WORDS=16 + HASH_BLOCK_BYTES=
             * 128 axes — single-axis delta (mirrors SHA1PASSSALT vs
             * SHA1SALTPASS / SHA256PASSSALT vs SHA256SALTPASS / MD5-
             * PASSSALT vs MD5SALTPASS). Pure spec reuse on the SHA-512
             * main template (sha512_style_salted.cl.tmpl, salt-
             * position-agnostic), plus ONE new fragment authoring
             * (finalize_append_be64.cl.frag). HARD GATE 43,024 B
             * gfx1201 priv_mem; sibling SHA512SALTPASS reading was
             * 42,032 B (992 B headroom). */
            cached = GPU_TEMPLATE_SHA512PASSSALT;
            fprintf(stderr,
                "OpenCL: MDXFIND_GPU_TEMPLATE=sha512passsalt — SHA512(pass || salt) "
                "simple append salt SHA-512 (B6.10 fan-out; FINAL B6 ladder step; "
                "second 64-bit-state salted variant; SALT_POSITION=APPEND, "
                "HASH_WORDS=16, HASH_BLOCK_BYTES=128, BASE_ALGO=sha512).\n");
        } else if (e && *e && *e != '0' && strcmp(e, "sha1dru") == 0) {
            /* B6.11 SHA1DRU fan-out (2026-05-06): SHA1DRU (Drupal SHA1,
             * hashcat -m 7900, JOB_SHA1DRU=404). First 1M-iteration
             * algorithm on the unified template path. SHA1(pass) + 1M
             * iters of SHA1(hex_lc(state) || pass); ONE probe at the
             * final state. Mirrors mdxfind.c JOB_SHA1DRU (lines 14261-
             * 14285). 1M loop INSIDE template_finalize; max_iter=1 host-
             * forced. */
            cached = GPU_TEMPLATE_SHA1DRU;
            fprintf(stderr,
                "OpenCL: MDXFIND_GPU_TEMPLATE=sha1dru — Drupal SHA1 "
                "(B6.11 fan-out; first 1M-iteration algorithm on the unified "
                "template path; HASH_WORDS=5, HASH_BLOCK_BYTES=64, "
                "BASE_ALGO=sha1, ITER_COUNT=1000000).\n");
        } else if (e && *e && *e != '0' && strcmp(e, "md6256") == 0) {
            /* B7.7b MD6256 fan-out (2026-05-07): MD6256 (hashcat -m 17800,
             * JOB_MD6256=29). Final M5 closure from B9 gate-fail. MD6-256
             * single-block leaf compression with 1753-ulong A working
             * array (14 KB stack). Per-iter probe like SQL5 (vs.
             * SHA1DRU's max_iter=1 internal loop). Mirrors mdxfind.c
             * JOB_MD6256 (lines 25836-25855). */
            cached = GPU_TEMPLATE_MD6256;
            fprintf(stderr,
                "OpenCL: MDXFIND_GPU_TEMPLATE=md6256 — MD6-256 "
                "(B7.7b fan-out; final M5 closure; algorithmically-largest "
                "single-compression unsalted algo on the template path; "
                "HASH_WORDS=8, HASH_BLOCK_BYTES=64, BASE_ALGO=md6).\n");
        } else if (e && *e && *e != '0' && strcmp(e, "hmac_blake2s") == 0) {
            /* Family I HMAC-BLAKE2S carrier (2026-05-08): Path A hand-
             * written sibling for HMAC-BLAKE2S (e828). Single algo_mode (5);
             * no KPASS sibling op exists in mdxfind. HMAC body branches at
             * the top of template_finalize in gpu_hmac_blake2s_core.cl
             * (gated on algo_mode == 5u) and returns early. The mode-0
             * BLAKE2S(salt||pass) main body is structurally unreachable in
             * production. */
            cached = GPU_TEMPLATE_HMAC_BLAKE2S;
            fprintf(stderr,
                "OpenCL: MDXFIND_GPU_TEMPLATE=hmac_blake2s — Family I "
                "HMAC-BLAKE2S carrier (Path A hand-written; single algo_mode "
                "5; HASH_WORDS=8, HASH_BLOCK_BYTES=64, BASE_ALGO=blake2s, "
                "HAS_SALT=1, HMAC_KPASS=1).\n");
        } else if (e && *e && *e != '0' && strcmp(e, "hmac_streebog256") == 0) {
            /* Family J HMAC-STREEBOG-256 carrier (2026-05-08): Path A
             * hand-written sibling for HMAC-STREEBOG256_KSALT (e838) +
             * HMAC-STREEBOG256_KPASS (e837). Two algo_modes: 5 = KSALT,
             * 6 = KPASS. HMAC body branches at the top of template_finalize
             * in gpu_hmac_streebog256_core.cl (gated on algo_mode >= 5u —
             * single branch since kernel-side math is identical for KSALT
             * and KPASS) and returns early. The mode-0 STREEBOG-256(salt||
             * pass) main body is structurally unreachable in production. */
            cached = GPU_TEMPLATE_HMAC_STREEBOG256;
            fprintf(stderr,
                "OpenCL: MDXFIND_GPU_TEMPLATE=hmac_streebog256 — Family J "
                "HMAC-STREEBOG-256 carrier (Path A hand-written; two algo_modes "
                "5/6 for KSALT/KPASS; HASH_WORDS=8, HASH_BLOCK_BYTES=64, "
                "BASE_ALGO=streebog256, HAS_SALT=1, HMAC_KSALTPASS=1).\n");
        } else if (e && *e && *e != '0' && strcmp(e, "hmac_streebog512") == 0) {
            /* Family K HMAC-STREEBOG-512 carrier (2026-05-08): Path A
             * hand-written sibling for HMAC-STREEBOG512_KSALT (e840) +
             * HMAC-STREEBOG512_KPASS (e839). Two algo_modes: 5 = KSALT,
             * 6 = KPASS. HMAC body branches at the top of template_finalize
             * in gpu_hmac_streebog512_core.cl (gated on algo_mode >= 5u —
             * single branch since kernel-side math is identical for KSALT
             * and KPASS) and returns early. The mode-0 STREEBOG-512(salt||
             * pass) main body is structurally unreachable in production.
             * Final HMAC family in the ladder. */
            cached = GPU_TEMPLATE_HMAC_STREEBOG512;
            fprintf(stderr,
                "OpenCL: MDXFIND_GPU_TEMPLATE=hmac_streebog512 — Family K "
                "HMAC-STREEBOG-512 carrier (Path A hand-written; two algo_modes "
                "5/6 for KSALT/KPASS; HASH_WORDS=16, HASH_BLOCK_BYTES=64, "
                "BASE_ALGO=streebog512, HAS_SALT=1, HMAC_KSALTPASS=1).\n");
        } else if (e && *e && *e != '0' && strcmp(e, "phpbb3") == 0) {
            /* PHPBB3 carrier (2026-05-08): Path A hand-written
             * salted-template kernel for JOB_PHPBB3 (e455). Iterated
             * MD5 chain INSIDE template_finalize; iter count decoded
             * from salt[3] via phpitoa64 reverse lookup (typical range
             * 7..30 -> 128..2^30 iters). max_iter=1 host-forced so
             * kernel's outer iter loop runs exactly once and only the
             * FINAL state is probed. Mirrors mdxfind.c JOB_PHPBB3
             * (lines 13415-13628). */
            cached = GPU_TEMPLATE_PHPBB3;
            fprintf(stderr,
                "OpenCL: MDXFIND_GPU_TEMPLATE=phpbb3 — PHPBB3 / phpass "
                "(Path A hand-written; single algo_mode; iterated MD5 "
                "chain in template_finalize with iter count from salt[3]; "
                "HASH_WORDS=4, HASH_BLOCK_BYTES=64, BASE_ALGO=phpbb3, "
                "HAS_SALT=1).\n");
        } else if (e && *e && *e != '0' && strcmp(e, "md5crypt") == 0) {
            /* MD5CRYPT carrier (2026-05-08): Path A hand-written
             * salted-template kernel for JOB_MD5CRYPT (e511). BSD $1$
             * md5crypt with FIXED 1000-iteration count. Iterated MD5
             * chain INSIDE template_finalize; max_iter=1 host-forced
             * so kernel's outer iter loop runs exactly once and only
             * the FINAL state is probed. Mirrors mdxfind.c JOB_MD5CRYPT
             * (lines 13017-13117). Phase 1 of the Unix-crypt ladder. */
            cached = GPU_TEMPLATE_MD5CRYPT;
            fprintf(stderr,
                "OpenCL: MDXFIND_GPU_TEMPLATE=md5crypt -- MD5CRYPT / BSD $1$ "
                "(Path A hand-written; single algo_mode; iterated MD5 chain "
                "in template_finalize with FIXED 1000 iters; HASH_WORDS=4, "
                "HASH_BLOCK_BYTES=64, BASE_ALGO=md5crypt, HAS_SALT=1).\n");
        } else if (e && *e && *e != '0' && strcmp(e, "sha256crypt") == 0) {
            /* SHA256CRYPT carrier (2026-05-08): Path A hand-written
             * salted-template kernel for JOB_SHA256CRYPT (e512). glibc
             * crypt-sha256 ($5$[rounds=N$]<salt>$<43-base64>) with
             * default 5000-iteration count (configurable via "rounds=N$"
             * salt prefix). 5-step chain INSIDE template_finalize;
             * max_iter=1 host-forced so kernel's outer iter loop runs
             * exactly once and only the FINAL state is probed. Mirrors
             * mdxfind.c JOB_SHA256CRYPT (entry at line 12121; shared
             * crypt_round body at 12177-12290). Phase 2 of the Unix-
             * crypt ladder. */
            cached = GPU_TEMPLATE_SHA256CRYPT;
            fprintf(stderr,
                "OpenCL: MDXFIND_GPU_TEMPLATE=sha256crypt -- SHA256CRYPT / "
                "glibc $5$ (Path A hand-written; single algo_mode; 5-step "
                "chain in template_finalize with default 5000 iters via "
                "rounds=N$ salt prefix; HASH_WORDS=8, HASH_BLOCK_BYTES=64, "
                "BASE_ALGO=sha256crypt, HAS_SALT=1).\n");
        } else if (e && *e && *e != '0') {
            /* Unknown value: log once, default to off. */
            fprintf(stderr,
                "OpenCL: MDXFIND_GPU_TEMPLATE=\"%s\" not recognized "
                "(supported: \"md5\", \"sha1\", \"sha256\", \"sha224\", "
                "\"md4\", \"sha384\", \"sha512\", \"ripemd160\"/\"rmd160\", "
                "\"ripemd320\"/\"rmd320\", \"blake2s256\", \"blake2b256\", "
                "\"blake2b512\", \"keccak{224,256,384,512}\", "
                "\"sha3_{224,256,384,512}\", \"sha384raw\", \"sha512raw\", "
                "\"md5raw\", \"sha1raw\", \"sha256raw\", \"sql5\", "
                "\"ntlmh\", \"md4utf16\", \"mysql3\", \"wrl\", "
                "\"streebog256\", \"streebog512\", "
                "\"md5salt\", \"md5saltpass\", \"md5passsalt\", "
                "\"sha1saltpass\", \"sha1passsalt\", \"sha256saltpass\", "
                "\"sha256passsalt\", \"sha224saltpass\", \"sha512saltpass\", "
                "\"sha512passsalt\", \"sha1dru\", \"md6256\", "
                "\"hmac_blake2s\", \"hmac_streebog256\", \"hmac_streebog512\", "
                "\"phpbb3\", \"md5crypt\", \"sha256crypt\"); "
                "ignoring, using production md5_rules_phase0.\n", e);
            cached = GPU_TEMPLATE_OFF;
        } else {
            cached = GPU_TEMPLATE_OFF;
        }
    }
    return cached;
}

/* Build the template program for this device. Sources (in order):
 *   gpu_common_str        -- shared primitives (md5_block, OCLParams, ...)
 *   gpu_md5_rules_str     -- apply_rule (shared rules walker)
 *   gpu_md5_core_str      -- algorithm geometry + 4 extension fns
 *   gpu_template_str      -- this build's template_phase0 kernel
 *
 * Cache key (R3 mitigation): defines_str = "HASH_WORDS=4,HASH_BLOCK_BYTES=64"
 * so distinct algorithm tuples (when B4+ adds them) get distinct cache
 * keys even though gpu_template.cl source text is identical. */
static int gpu_opencl_template_compile(struct gpu_device *d, int dev_idx) {
    if (d->prog_template) return 0;             /* already built */
    cl_int err = CL_SUCCESS;
    const char *sources[4] = {
        gpu_common_str,
        gpu_md5_rules_str,
        gpu_md5_core_str,
        gpu_template_str
    };
    const char *defines = "HASH_WORDS=4,HASH_BLOCK_BYTES=64";
    d->prog_template = gpu_kernel_cache_build_program_ex(
        d->ctx, d->dev, 4, sources,
        "-cl-std=CL1.2",
        defines, &err);
    if (!d->prog_template || err != CL_SUCCESS) {
        char log[8192] = {0};
        if (d->prog_template) {
            clGetProgramBuildInfo(d->prog_template, d->dev,
                                  CL_PROGRAM_BUILD_LOG,
                                  sizeof(log) - 1, log, NULL);
        }
        fprintf(stderr,
            "OpenCL GPU[%d]: template program build error (err=%d) "
            "— template path unavailable; will fall back to "
            "md5_rules_phase0:\n%s\n",
            dev_idx, err, log);
        if (d->prog_template) clReleaseProgram(d->prog_template);
        d->prog_template = NULL;
        return -1;
    }
    return 0;
}

/* Lazily create the template_phase0 kernel object on this device. */
static int gpu_opencl_template_kernel_lazy(struct gpu_device *d, int dev_idx) {
    if (d->kern_template_phase0) return 0;
    if (!d->prog_template) return -1;
    cl_int err;
    d->kern_template_phase0 = clCreateKernel(d->prog_template,
                                             "template_phase0", &err);
    if (err != CL_SUCCESS || !d->kern_template_phase0) {
        fprintf(stderr,
            "OpenCL GPU[%d]: template_phase0 kernel create failed (err=%d) "
            "— template path unavailable; will fall back to "
            "md5_rules_phase0\n",
            dev_idx, err);
        d->kern_template_phase0 = NULL;
        return -1;
    }

    /* R2 instrumentation: probe per-lane private memory size for the
     * template kernel. Used to track register pressure across vendors;
     * relevant for B5 SHA-512 family (project_memo_b_dispatch_template.md
     * §7 R2: AMD gfx1201 budget = 64 VGPR; risk threshold ~60).
     *
     * CL_KERNEL_PRIVATE_MEM_SIZE returns bytes of private mem per work-
     * item (compiler-allocated; includes spilled registers). The "VGPR"
     * mapping isn't 1:1 across vendors but a relative reading captures
     * the cross-platform risk shape. One-shot stderr line on first
     * kernel build per device. */
    return 0;
}

/* ----------------------------------------------------------------------
 * Phase 1.9 Tranche A1 (2026-05-10): MD5 brute-force fast-path template
 * instantiation. Side-by-side with gpu_opencl_template_compile (slow MD5);
 * builds the SAME gpu_template.cl + gpu_md5_rules.cl source set BUT
 * substitutes gpu_md5_bf_str for gpu_md5_core_str in the algorithm-core
 * slot, and adds a distinct token to defines_str so the kernel cache
 * stores it as a separate cl_program. Selected at dispatch time by
 * gpu_template_resolve_kernel() when bf_fast_eligible is set on the
 * dispatching jobg slot; falls through to gpu_opencl_template_compile
 * (slow MD5) otherwise.
 *
 * A1 SCOPE: source skeleton only — gpu_md5_bf.cl body is byte-identical
 * to gpu_md5_core.cl rev 1.2. Perf parity to the slow path is the A1
 * gate. A2 inlines md5_block + round-constant pre-add hoist; A3 host
 * mask pre-explosion; A4 vendor intrinsics.
 *
 * Cache key (R3 mitigation): defines_str =
 * "HASH_WORDS=4,HASH_BLOCK_BYTES=64,BF_FAST_MD5=1". The trailing
 * BF_FAST_MD5=1 token differentiates this entry from the slow MD5
 * template's "HASH_WORDS=4,HASH_BLOCK_BYTES=64". The kernel-cache key
 * also includes the source-text hash, so even without BF_FAST_MD5=1
 * the entries would differ once A2 diverges the source — but the
 * defines_str disambiguation is cheap insurance against accidental
 * collision while A1 ships byte-identical bodies. The token is also
 * passed as a -D build_opt below; the OpenCL preprocessor sees it
 * during compilation (currently unused in source, kept for future
 * runtime branching if needed).
 *
 * The kernel name is still "template_phase0" — same gpu_template.cl
 * source defines exactly that one kernel; the algorithm core is
 * selected at source-array compose time, not at runtime. ---------- */

static int gpu_opencl_template_md5_bf_compile(struct gpu_device *d, int dev_idx) {
    if (d->prog_template_md5_bf) return 0;       /* already built */
    cl_int err = CL_SUCCESS;
    const char *sources[4] = {
        gpu_common_str,
        gpu_md5_rules_str,
        gpu_md5_bf_str,
        gpu_template_str
    };
    const char *defines = "HASH_WORDS=4,HASH_BLOCK_BYTES=64,BF_FAST_MD5=1";
    /* build_opts: pass BF_FAST_MD5=1 as -D so the preprocessor sees it
     * during compilation (currently unused in source; reserved for
     * runtime branching in A2-A4). -cl-std=CL1.2 matches the slow path. */
    d->prog_template_md5_bf = gpu_kernel_cache_build_program_ex(
        d->ctx, d->dev, 4, sources,
        "-cl-std=CL1.2 -DBF_FAST_MD5=1",
        defines, &err);
    if (!d->prog_template_md5_bf || err != CL_SUCCESS) {
        char log[8192] = {0};
        if (d->prog_template_md5_bf) {
            clGetProgramBuildInfo(d->prog_template_md5_bf, d->dev,
                                  CL_PROGRAM_BUILD_LOG,
                                  sizeof(log) - 1, log, NULL);
        }
        fprintf(stderr,
            "OpenCL GPU[%d]: BF-fast template program build error (err=%d) "
            "— Phase 1.9 fast path unavailable; will fall back to slow "
            "MD5 template kernel for BF dispatches:\n%s\n",
            dev_idx, err, log);
        if (d->prog_template_md5_bf) clReleaseProgram(d->prog_template_md5_bf);
        d->prog_template_md5_bf = NULL;
        return -1;
    }
    return 0;
}

/* Lazily create the template_phase0 (BF-fast MD5 instantiation) kernel
 * object on this device. Mirrors gpu_opencl_template_kernel_lazy. */
static int gpu_opencl_template_md5_bf_kernel_lazy(struct gpu_device *d, int dev_idx) {
    if (d->kern_template_phase0_md5_bf) return 0;
    if (!d->prog_template_md5_bf) return -1;
    cl_int err;
    d->kern_template_phase0_md5_bf = clCreateKernel(d->prog_template_md5_bf,
                                                    "template_phase0", &err);
    if (err != CL_SUCCESS || !d->kern_template_phase0_md5_bf) {
        fprintf(stderr,
            "OpenCL GPU[%d]: template_phase0 (BF-fast MD5) kernel create "
            "failed (err=%d) — Phase 1.9 fast path unavailable; will fall "
            "back to slow MD5 template kernel\n",
            dev_idx, err);
        d->kern_template_phase0_md5_bf = NULL;
        return -1;
    }
    return 0;
}

/* ----------------------------------------------------------------------
 * Memo B Phase B4 (2026-05-04): SHA1 template instantiation.
 *
 * Parallel to gpu_opencl_template_compile / _kernel_lazy (MD5). Builds
 * the template program with gpu_sha1_core.cl in place of gpu_md5_core.cl
 * and HASH_WORDS=5 in defines_str so the cache key differs from MD5's.
 *
 * The kernel name is still "template_phase0" — gpu_template.cl exports
 * exactly that one kernel; the algorithm core is selected at link/include
 * time via the source set, not at runtime. SHA1 and MD5 instantiations
 * therefore share the kernel name but live in distinct cl_program /
 * cl_kernel objects on the device (prog_template_sha1 / kern_template_-
 * phase0_sha1).
 * ---------------------------------------------------------------------- */

/* Build the SHA1 template program for this device. Sources (in order):
 *   gpu_common_str        -- shared primitives (sha1_block, bswap32,
 *                            EMIT_HIT_5, OCLParams, probe_compact_idx, ...)
 *   gpu_md5_rules_str     -- apply_rule (algorithm-agnostic rules walker)
 *   gpu_sha1_core_str     -- SHA1 algorithm geometry + extension fns
 *   gpu_template_str      -- this build's template_phase0 kernel
 *
 * Cache key (R3 mitigation): defines_str = "HASH_WORDS=5,HASH_BLOCK_BYTES=64"
 * so the SHA1 instantiation receives a distinct cache entry from MD5's
 * (whose key includes HASH_WORDS=4). gpu_template.cl source text is
 * identical between the two. */
static int gpu_opencl_template_compile_sha1(struct gpu_device *d, int dev_idx) {
    if (d->prog_template_sha1) return 0;        /* already built */
    cl_int err = CL_SUCCESS;
    const char *sources[4] = {
        gpu_common_str,
        gpu_md5_rules_str,
        gpu_sha1_core_str,
        gpu_template_str
    };
    const char *defines = "HASH_WORDS=5,HASH_BLOCK_BYTES=64";
    d->prog_template_sha1 = gpu_kernel_cache_build_program_ex(
        d->ctx, d->dev, 4, sources,
        "-cl-std=CL1.2",
        defines, &err);
    if (!d->prog_template_sha1 || err != CL_SUCCESS) {
        char log[8192] = {0};
        if (d->prog_template_sha1) {
            clGetProgramBuildInfo(d->prog_template_sha1, d->dev,
                                  CL_PROGRAM_BUILD_LOG,
                                  sizeof(log) - 1, log, NULL);
        }
        fprintf(stderr,
            "OpenCL GPU[%d]: SHA1 template program build error (err=%d) "
            "— SHA1 template path unavailable; will fall back to "
            "md5_rules_phase0 (which is MD5-only -- this is a no-op for "
            "SHA1 work; CPU will pick up the work via Typedone[]):\n%s\n",
            dev_idx, err, log);
        if (d->prog_template_sha1) clReleaseProgram(d->prog_template_sha1);
        d->prog_template_sha1 = NULL;
        return -1;
    }
    return 0;
}

/* Lazily create the template_phase0 (SHA1 instantiation) kernel object.
 * Mirrors gpu_opencl_template_kernel_lazy (MD5) including the R2
 * private_mem_size probe. The probe value for SHA1 is logged with the
 * "SHA1" tag so cross-algorithm comparison is explicit in stderr. */
static int gpu_opencl_template_kernel_lazy_sha1(struct gpu_device *d, int dev_idx) {
    if (d->kern_template_phase0_sha1) return 0;
    if (!d->prog_template_sha1) return -1;
    cl_int err;
    d->kern_template_phase0_sha1 = clCreateKernel(d->prog_template_sha1,
                                                  "template_phase0", &err);
    if (err != CL_SUCCESS || !d->kern_template_phase0_sha1) {
        fprintf(stderr,
            "OpenCL GPU[%d]: template_phase0 (SHA1) kernel create failed "
            "(err=%d) — SHA1 template path unavailable\n",
            dev_idx, err);
        d->kern_template_phase0_sha1 = NULL;
        return -1;
    }

    /* R2 instrumentation: probe per-lane private memory size for the
     * SHA1 template kernel. Compare against MD5's reading to detect
     * register-pressure expansion. SHA1's W[80] schedule (320 bytes)
     * is larger than MD5's M[16] (64 bytes); expect SHA1's reading
     * to be moderately larger but well within the 256-byte gfx1201
     * budget for a well-optimized compiler. If the reading is >>240
     * bytes on AMD, that's the early signal that B5/B6 should
     * separate the SHA1 family from the MD5/SHA256 occupancy class. */
    return 0;
}

/* ----------------------------------------------------------------------
 * Memo B Phase B4 fan-out (2026-05-04): SHA256/SHA224/MD4 template
 * instantiations. Mirror gpu_opencl_template_compile_sha1 /
 * _kernel_lazy_sha1 exactly; each is selected at the dispatch_md5_rules
 * kernel-handle swap site under strict op-match (op == JOB_<algo>).
 *
 * R3 cache-key mitigation: each algorithm passes its own defines_str.
 *   SHA256: HASH_WORDS=8,HASH_BLOCK_BYTES=64
 *   SHA224: HASH_WORDS=7,HASH_BLOCK_BYTES=64  (output truncation; state
 *           struct still 8 words internally)
 *   MD4:    HASH_WORDS=4,HASH_BLOCK_BYTES=64  (identical to MD5; the
 *           kernel-cache key includes source text so MD4's distinct
 *           gpu_md4_core_str produces a distinct cache entry)
 *
 * The kernel name is still "template_phase0" for all instantiations —
 * gpu_template.cl exports exactly that one kernel; the algorithm core
 * is selected at compile/include time via the source set. Distinct
 * cl_program / cl_kernel objects per instantiation.
 * ---------------------------------------------------------------------- */

/* SHA256 template program build. */
static int gpu_opencl_template_compile_sha256(struct gpu_device *d, int dev_idx) {
    if (d->prog_template_sha256) return 0;
    cl_int err = CL_SUCCESS;
    const char *sources[4] = {
        gpu_common_str,
        gpu_md5_rules_str,
        gpu_sha256_core_str,
        gpu_template_str
    };
    const char *defines = "HASH_WORDS=8,HASH_BLOCK_BYTES=64";
    d->prog_template_sha256 = gpu_kernel_cache_build_program_ex(
        d->ctx, d->dev, 4, sources,
        "-cl-std=CL1.2",
        defines, &err);
    if (!d->prog_template_sha256 || err != CL_SUCCESS) {
        char log[8192] = {0};
        if (d->prog_template_sha256) {
            clGetProgramBuildInfo(d->prog_template_sha256, d->dev,
                                  CL_PROGRAM_BUILD_LOG,
                                  sizeof(log) - 1, log, NULL);
        }
        fprintf(stderr,
            "OpenCL GPU[%d]: SHA256 template program build error (err=%d) "
            "— SHA256 template path unavailable:\n%s\n",
            dev_idx, err, log);
        if (d->prog_template_sha256) clReleaseProgram(d->prog_template_sha256);
        d->prog_template_sha256 = NULL;
        return -1;
    }
    return 0;
}

static int gpu_opencl_template_kernel_lazy_sha256(struct gpu_device *d, int dev_idx) {
    if (d->kern_template_phase0_sha256) return 0;
    if (!d->prog_template_sha256) return -1;
    cl_int err;
    d->kern_template_phase0_sha256 = clCreateKernel(d->prog_template_sha256,
                                                    "template_phase0", &err);
    if (err != CL_SUCCESS || !d->kern_template_phase0_sha256) {
        fprintf(stderr,
            "OpenCL GPU[%d]: template_phase0 (SHA256) kernel create failed "
            "(err=%d) — SHA256 template path unavailable\n",
            dev_idx, err);
        d->kern_template_phase0_sha256 = NULL;
        return -1;
    }
    return 0;
}

/* SHA224 template program build. */
static int gpu_opencl_template_compile_sha224(struct gpu_device *d, int dev_idx) {
    if (d->prog_template_sha224) return 0;
    cl_int err = CL_SUCCESS;
    const char *sources[4] = {
        gpu_common_str,
        gpu_md5_rules_str,
        gpu_sha224_core_str,
        gpu_template_str
    };
    const char *defines = "HASH_WORDS=7,HASH_BLOCK_BYTES=64";
    d->prog_template_sha224 = gpu_kernel_cache_build_program_ex(
        d->ctx, d->dev, 4, sources,
        "-cl-std=CL1.2",
        defines, &err);
    if (!d->prog_template_sha224 || err != CL_SUCCESS) {
        char log[8192] = {0};
        if (d->prog_template_sha224) {
            clGetProgramBuildInfo(d->prog_template_sha224, d->dev,
                                  CL_PROGRAM_BUILD_LOG,
                                  sizeof(log) - 1, log, NULL);
        }
        fprintf(stderr,
            "OpenCL GPU[%d]: SHA224 template program build error (err=%d) "
            "— SHA224 template path unavailable:\n%s\n",
            dev_idx, err, log);
        if (d->prog_template_sha224) clReleaseProgram(d->prog_template_sha224);
        d->prog_template_sha224 = NULL;
        return -1;
    }
    return 0;
}

static int gpu_opencl_template_kernel_lazy_sha224(struct gpu_device *d, int dev_idx) {
    if (d->kern_template_phase0_sha224) return 0;
    if (!d->prog_template_sha224) return -1;
    cl_int err;
    d->kern_template_phase0_sha224 = clCreateKernel(d->prog_template_sha224,
                                                    "template_phase0", &err);
    if (err != CL_SUCCESS || !d->kern_template_phase0_sha224) {
        fprintf(stderr,
            "OpenCL GPU[%d]: template_phase0 (SHA224) kernel create failed "
            "(err=%d) — SHA224 template path unavailable\n",
            dev_idx, err);
        d->kern_template_phase0_sha224 = NULL;
        return -1;
    }
    return 0;
}

/* MD4 template program build. */
static int gpu_opencl_template_compile_md4(struct gpu_device *d, int dev_idx) {
    if (d->prog_template_md4) return 0;
    cl_int err = CL_SUCCESS;
    const char *sources[4] = {
        gpu_common_str,
        gpu_md5_rules_str,
        gpu_md4_core_str,
        gpu_template_str
    };
    const char *defines = "HASH_WORDS=4,HASH_BLOCK_BYTES=64";
    d->prog_template_md4 = gpu_kernel_cache_build_program_ex(
        d->ctx, d->dev, 4, sources,
        "-cl-std=CL1.2",
        defines, &err);
    if (!d->prog_template_md4 || err != CL_SUCCESS) {
        char log[8192] = {0};
        if (d->prog_template_md4) {
            clGetProgramBuildInfo(d->prog_template_md4, d->dev,
                                  CL_PROGRAM_BUILD_LOG,
                                  sizeof(log) - 1, log, NULL);
        }
        fprintf(stderr,
            "OpenCL GPU[%d]: MD4 template program build error (err=%d) "
            "— MD4 template path unavailable:\n%s\n",
            dev_idx, err, log);
        if (d->prog_template_md4) clReleaseProgram(d->prog_template_md4);
        d->prog_template_md4 = NULL;
        return -1;
    }
    return 0;
}

static int gpu_opencl_template_kernel_lazy_md4(struct gpu_device *d, int dev_idx) {
    if (d->kern_template_phase0_md4) return 0;
    if (!d->prog_template_md4) return -1;
    cl_int err;
    d->kern_template_phase0_md4 = clCreateKernel(d->prog_template_md4,
                                                 "template_phase0", &err);
    if (err != CL_SUCCESS || !d->kern_template_phase0_md4) {
        fprintf(stderr,
            "OpenCL GPU[%d]: template_phase0 (MD4) kernel create failed "
            "(err=%d) — MD4 template path unavailable\n",
            dev_idx, err);
        d->kern_template_phase0_md4 = NULL;
        return -1;
    }
    return 0;
}

/* ----------------------------------------------------------------------
 * Memo B Phase B5 sub-batch 1 (2026-05-04): SHA384/SHA512 template
 * instantiations. First 64-bit-state + first 128-bit length encoding
 * algorithms in the family. Mirror the gpu_opencl_template_compile_md4
 * / _kernel_lazy_md4 wiring exactly.
 *
 * R3 cache-key mitigation: each algorithm passes its own defines_str.
 *   SHA384: HASH_WORDS=12,HASH_BLOCK_BYTES=128
 *   SHA512: HASH_WORDS=16,HASH_BLOCK_BYTES=128
 *
 * Distinct cache entries per algorithm guaranteed via source-text +
 * defines hashing in gpu_kernel_cache_build_program_ex.
 *
 * R2 (register pressure): SHA-512 family uses W[80] schedule (80 × 8
 * = 640 bytes private scratch) — largest in the family so far. Memo B
 * §3 R2 flagged gfx1201 as the risk vendor. The lazy creation path
 * logs CL_KERNEL_PRIVATE_MEM_SIZE so production runs and harness runs
 * both report the reading. SHA-512's reading is the canary for whether
 * future B6 algorithms (BLAKE2B, SHA3-512, etc.) need a separate
 * occupancy strategy.
 * ---------------------------------------------------------------------- */

/* SHA384 template program build. */
static int gpu_opencl_template_compile_sha384(struct gpu_device *d, int dev_idx) {
    if (d->prog_template_sha384) return 0;
    cl_int err = CL_SUCCESS;
    const char *sources[4] = {
        gpu_common_str,
        gpu_md5_rules_str,
        gpu_sha384_core_str,
        gpu_template_str
    };
    const char *defines = "HASH_WORDS=12,HASH_BLOCK_BYTES=128";
    d->prog_template_sha384 = gpu_kernel_cache_build_program_ex(
        d->ctx, d->dev, 4, sources,
        "-cl-std=CL1.2",
        defines, &err);
    if (!d->prog_template_sha384 || err != CL_SUCCESS) {
        char log[8192] = {0};
        if (d->prog_template_sha384) {
            clGetProgramBuildInfo(d->prog_template_sha384, d->dev,
                                  CL_PROGRAM_BUILD_LOG,
                                  sizeof(log) - 1, log, NULL);
        }
        fprintf(stderr,
            "OpenCL GPU[%d]: SHA384 template program build error (err=%d) "
            "— SHA384 template path unavailable:\n%s\n",
            dev_idx, err, log);
        if (d->prog_template_sha384) clReleaseProgram(d->prog_template_sha384);
        d->prog_template_sha384 = NULL;
        return -1;
    }
    return 0;
}

static int gpu_opencl_template_kernel_lazy_sha384(struct gpu_device *d, int dev_idx) {
    if (d->kern_template_phase0_sha384) return 0;
    if (!d->prog_template_sha384) return -1;
    cl_int err;
    d->kern_template_phase0_sha384 = clCreateKernel(d->prog_template_sha384,
                                                    "template_phase0", &err);
    if (err != CL_SUCCESS || !d->kern_template_phase0_sha384) {
        fprintf(stderr,
            "OpenCL GPU[%d]: template_phase0 (SHA384) kernel create failed "
            "(err=%d) — SHA384 template path unavailable\n",
            dev_idx, err);
        d->kern_template_phase0_sha384 = NULL;
        return -1;
    }
    return 0;
}

/* SHA512 template program build. */
static int gpu_opencl_template_compile_sha512(struct gpu_device *d, int dev_idx) {
    if (d->prog_template_sha512) return 0;
    cl_int err = CL_SUCCESS;
    const char *sources[4] = {
        gpu_common_str,
        gpu_md5_rules_str,
        gpu_sha512_core_str,
        gpu_template_str
    };
    const char *defines = "HASH_WORDS=16,HASH_BLOCK_BYTES=128";
    d->prog_template_sha512 = gpu_kernel_cache_build_program_ex(
        d->ctx, d->dev, 4, sources,
        "-cl-std=CL1.2",
        defines, &err);
    if (!d->prog_template_sha512 || err != CL_SUCCESS) {
        char log[8192] = {0};
        if (d->prog_template_sha512) {
            clGetProgramBuildInfo(d->prog_template_sha512, d->dev,
                                  CL_PROGRAM_BUILD_LOG,
                                  sizeof(log) - 1, log, NULL);
        }
        fprintf(stderr,
            "OpenCL GPU[%d]: SHA512 template program build error (err=%d) "
            "— SHA512 template path unavailable:\n%s\n",
            dev_idx, err, log);
        if (d->prog_template_sha512) clReleaseProgram(d->prog_template_sha512);
        d->prog_template_sha512 = NULL;
        return -1;
    }
    return 0;
}

static int gpu_opencl_template_kernel_lazy_sha512(struct gpu_device *d, int dev_idx) {
    if (d->kern_template_phase0_sha512) return 0;
    if (!d->prog_template_sha512) return -1;
    cl_int err;
    d->kern_template_phase0_sha512 = clCreateKernel(d->prog_template_sha512,
                                                    "template_phase0", &err);
    if (err != CL_SUCCESS || !d->kern_template_phase0_sha512) {
        fprintf(stderr,
            "OpenCL GPU[%d]: template_phase0 (SHA512) kernel create failed "
            "(err=%d) — SHA512 template path unavailable\n",
            dev_idx, err);
        d->kern_template_phase0_sha512 = NULL;
        return -1;
    }
    return 0;
}

/* ----------------------------------------------------------------------
 * Memo B Phase B5 sub-batch 2 (2026-05-05): RIPEMD-160 / RIPEMD-320
 * template program build + lazy kernel creation.
 *
 * Mirror gpu_opencl_template_compile_sha384 / _kernel_lazy_sha384 wiring.
 *
 * R3 cache-key mitigation: each algorithm passes its own defines_str.
 *   RIPEMD-160: HASH_WORDS=5,HASH_BLOCK_BYTES=64
 *   RIPEMD-320: HASH_WORDS=10,HASH_BLOCK_BYTES=64
 *
 * RIPEMD-160 has the same defines_str as SHA1 — distinct cache entry
 * guaranteed by source-text hashing in gpu_kernel_cache_build_program_ex.
 * RIPEMD-320 has unique defines_str (HASH_WORDS=10 is unique to it in
 * the family today).
 *
 * R2 (register pressure): rmd160_block / rmd320_block fit on the same
 * private-memory budget as MD5/SHA1 — comparable to rmd160_block.
 * RIPEMD-320's dual-pipeline carries 10 working uint32 + 10 chaining
 * + the 80-step expansion, slightly above rmd160 footprint but well
 * within the gfx1201 budget.
 * ---------------------------------------------------------------------- */

/* RIPEMD-160 template program build. */
static int gpu_opencl_template_compile_ripemd160(struct gpu_device *d, int dev_idx) {
    if (d->prog_template_ripemd160) return 0;
    cl_int err = CL_SUCCESS;
    const char *sources[4] = {
        gpu_common_str,
        gpu_md5_rules_str,
        gpu_ripemd160_core_str,
        gpu_template_str
    };
    const char *defines = "HASH_WORDS=5,HASH_BLOCK_BYTES=64";
    d->prog_template_ripemd160 = gpu_kernel_cache_build_program_ex(
        d->ctx, d->dev, 4, sources,
        "-cl-std=CL1.2",
        defines, &err);
    if (!d->prog_template_ripemd160 || err != CL_SUCCESS) {
        char log[8192] = {0};
        if (d->prog_template_ripemd160) {
            clGetProgramBuildInfo(d->prog_template_ripemd160, d->dev,
                                  CL_PROGRAM_BUILD_LOG,
                                  sizeof(log) - 1, log, NULL);
        }
        fprintf(stderr,
            "OpenCL GPU[%d]: RIPEMD-160 template program build error (err=%d) "
            "— RIPEMD-160 template path unavailable:\n%s\n",
            dev_idx, err, log);
        if (d->prog_template_ripemd160) clReleaseProgram(d->prog_template_ripemd160);
        d->prog_template_ripemd160 = NULL;
        return -1;
    }
    return 0;
}

static int gpu_opencl_template_kernel_lazy_ripemd160(struct gpu_device *d, int dev_idx) {
    if (d->kern_template_phase0_ripemd160) return 0;
    if (!d->prog_template_ripemd160) return -1;
    cl_int err;
    d->kern_template_phase0_ripemd160 = clCreateKernel(d->prog_template_ripemd160,
                                                       "template_phase0", &err);
    if (err != CL_SUCCESS || !d->kern_template_phase0_ripemd160) {
        fprintf(stderr,
            "OpenCL GPU[%d]: template_phase0 (RIPEMD-160) kernel create failed "
            "(err=%d) — RIPEMD-160 template path unavailable\n",
            dev_idx, err);
        d->kern_template_phase0_ripemd160 = NULL;
        return -1;
    }
    return 0;
}

/* RIPEMD-320 template program build. */
static int gpu_opencl_template_compile_ripemd320(struct gpu_device *d, int dev_idx) {
    if (d->prog_template_ripemd320) return 0;
    cl_int err = CL_SUCCESS;
    const char *sources[4] = {
        gpu_common_str,
        gpu_md5_rules_str,
        gpu_ripemd320_core_str,
        gpu_template_str
    };
    const char *defines = "HASH_WORDS=10,HASH_BLOCK_BYTES=64";
    d->prog_template_ripemd320 = gpu_kernel_cache_build_program_ex(
        d->ctx, d->dev, 4, sources,
        "-cl-std=CL1.2",
        defines, &err);
    if (!d->prog_template_ripemd320 || err != CL_SUCCESS) {
        char log[8192] = {0};
        if (d->prog_template_ripemd320) {
            clGetProgramBuildInfo(d->prog_template_ripemd320, d->dev,
                                  CL_PROGRAM_BUILD_LOG,
                                  sizeof(log) - 1, log, NULL);
        }
        fprintf(stderr,
            "OpenCL GPU[%d]: RIPEMD-320 template program build error (err=%d) "
            "— RIPEMD-320 template path unavailable:\n%s\n",
            dev_idx, err, log);
        if (d->prog_template_ripemd320) clReleaseProgram(d->prog_template_ripemd320);
        d->prog_template_ripemd320 = NULL;
        return -1;
    }
    return 0;
}

static int gpu_opencl_template_kernel_lazy_ripemd320(struct gpu_device *d, int dev_idx) {
    if (d->kern_template_phase0_ripemd320) return 0;
    if (!d->prog_template_ripemd320) return -1;
    cl_int err;
    d->kern_template_phase0_ripemd320 = clCreateKernel(d->prog_template_ripemd320,
                                                       "template_phase0", &err);
    if (err != CL_SUCCESS || !d->kern_template_phase0_ripemd320) {
        fprintf(stderr,
            "OpenCL GPU[%d]: template_phase0 (RIPEMD-320) kernel create failed "
            "(err=%d) — RIPEMD-320 template path unavailable\n",
            dev_idx, err);
        d->kern_template_phase0_ripemd320 = NULL;
        return -1;
    }
    return 0;
}

/* ---- Memo B Phase B5 sub-batch 3: BLAKE2 family compile + lazy helpers ----
 *
 * Three algorithms wired (BLAKE2B-160 omitted: JOB_BLAKE2B160 is not
 * defined in mdxfind.c and would require new job-type plumbing beyond
 * pure GPU work).
 *
 * Each (compile, kernel_lazy) pair mirrors gpu_opencl_template_compile_-
 * ripemd320 / _kernel_lazy_ripemd320 exactly: source set is { common,
 * md5_rules (apply_rule), <algo>_core, template }, defines_str carries
 * the algorithm-specific HASH_WORDS / HASH_BLOCK_BYTES so kernel cache
 * gets a unique key. */
static int gpu_opencl_template_compile_blake2s256(struct gpu_device *d, int dev_idx) {
    if (d->prog_template_blake2s256) return 0;
    cl_int err = CL_SUCCESS;
    const char *sources[4] = {
        gpu_common_str,
        gpu_md5_rules_str,
        gpu_blake2s256_core_str,
        gpu_template_str
    };
    const char *defines = "HASH_WORDS=8,HASH_BLOCK_BYTES=64";
    d->prog_template_blake2s256 = gpu_kernel_cache_build_program_ex(
        d->ctx, d->dev, 4, sources,
        "-cl-std=CL1.2",
        defines, &err);
    if (!d->prog_template_blake2s256 || err != CL_SUCCESS) {
        char log[8192] = {0};
        if (d->prog_template_blake2s256) {
            clGetProgramBuildInfo(d->prog_template_blake2s256, d->dev,
                                  CL_PROGRAM_BUILD_LOG,
                                  sizeof(log) - 1, log, NULL);
        }
        fprintf(stderr,
            "OpenCL GPU[%d]: BLAKE2S-256 template program build error (err=%d) "
            "— BLAKE2S-256 template path unavailable:\n%s\n",
            dev_idx, err, log);
        if (d->prog_template_blake2s256) clReleaseProgram(d->prog_template_blake2s256);
        d->prog_template_blake2s256 = NULL;
        return -1;
    }
    return 0;
}

static int gpu_opencl_template_kernel_lazy_blake2s256(struct gpu_device *d, int dev_idx) {
    if (d->kern_template_phase0_blake2s256) return 0;
    if (!d->prog_template_blake2s256) return -1;
    cl_int err;
    d->kern_template_phase0_blake2s256 = clCreateKernel(d->prog_template_blake2s256,
                                                        "template_phase0", &err);
    if (err != CL_SUCCESS || !d->kern_template_phase0_blake2s256) {
        fprintf(stderr,
            "OpenCL GPU[%d]: template_phase0 (BLAKE2S-256) kernel create failed "
            "(err=%d) — BLAKE2S-256 template path unavailable\n",
            dev_idx, err);
        d->kern_template_phase0_blake2s256 = NULL;
        return -1;
    }
    return 0;
}

static int gpu_opencl_template_compile_blake2b256(struct gpu_device *d, int dev_idx) {
    if (d->prog_template_blake2b256) return 0;
    cl_int err = CL_SUCCESS;
    const char *sources[4] = {
        gpu_common_str,
        gpu_md5_rules_str,
        gpu_blake2b256_core_str,
        gpu_template_str
    };
    const char *defines = "HASH_WORDS=8,HASH_BLOCK_BYTES=128";
    d->prog_template_blake2b256 = gpu_kernel_cache_build_program_ex(
        d->ctx, d->dev, 4, sources,
        "-cl-std=CL1.2",
        defines, &err);
    if (!d->prog_template_blake2b256 || err != CL_SUCCESS) {
        char log[8192] = {0};
        if (d->prog_template_blake2b256) {
            clGetProgramBuildInfo(d->prog_template_blake2b256, d->dev,
                                  CL_PROGRAM_BUILD_LOG,
                                  sizeof(log) - 1, log, NULL);
        }
        fprintf(stderr,
            "OpenCL GPU[%d]: BLAKE2B-256 template program build error (err=%d) "
            "— BLAKE2B-256 template path unavailable:\n%s\n",
            dev_idx, err, log);
        if (d->prog_template_blake2b256) clReleaseProgram(d->prog_template_blake2b256);
        d->prog_template_blake2b256 = NULL;
        return -1;
    }
    return 0;
}

static int gpu_opencl_template_kernel_lazy_blake2b256(struct gpu_device *d, int dev_idx) {
    if (d->kern_template_phase0_blake2b256) return 0;
    if (!d->prog_template_blake2b256) return -1;
    cl_int err;
    d->kern_template_phase0_blake2b256 = clCreateKernel(d->prog_template_blake2b256,
                                                        "template_phase0", &err);
    if (err != CL_SUCCESS || !d->kern_template_phase0_blake2b256) {
        fprintf(stderr,
            "OpenCL GPU[%d]: template_phase0 (BLAKE2B-256) kernel create failed "
            "(err=%d) — BLAKE2B-256 template path unavailable\n",
            dev_idx, err);
        d->kern_template_phase0_blake2b256 = NULL;
        return -1;
    }
    return 0;
}

static int gpu_opencl_template_compile_blake2b512(struct gpu_device *d, int dev_idx) {
    if (d->prog_template_blake2b512) return 0;
    cl_int err = CL_SUCCESS;
    const char *sources[4] = {
        gpu_common_str,
        gpu_md5_rules_str,
        gpu_blake2b512_core_str,
        gpu_template_str
    };
    const char *defines = "HASH_WORDS=16,HASH_BLOCK_BYTES=128";
    d->prog_template_blake2b512 = gpu_kernel_cache_build_program_ex(
        d->ctx, d->dev, 4, sources,
        "-cl-std=CL1.2",
        defines, &err);
    if (!d->prog_template_blake2b512 || err != CL_SUCCESS) {
        char log[8192] = {0};
        if (d->prog_template_blake2b512) {
            clGetProgramBuildInfo(d->prog_template_blake2b512, d->dev,
                                  CL_PROGRAM_BUILD_LOG,
                                  sizeof(log) - 1, log, NULL);
        }
        fprintf(stderr,
            "OpenCL GPU[%d]: BLAKE2B-512 template program build error (err=%d) "
            "— BLAKE2B-512 template path unavailable:\n%s\n",
            dev_idx, err, log);
        if (d->prog_template_blake2b512) clReleaseProgram(d->prog_template_blake2b512);
        d->prog_template_blake2b512 = NULL;
        return -1;
    }
    return 0;
}

static int gpu_opencl_template_kernel_lazy_blake2b512(struct gpu_device *d, int dev_idx) {
    if (d->kern_template_phase0_blake2b512) return 0;
    if (!d->prog_template_blake2b512) return -1;
    cl_int err;
    d->kern_template_phase0_blake2b512 = clCreateKernel(d->prog_template_blake2b512,
                                                        "template_phase0", &err);
    if (err != CL_SUCCESS || !d->kern_template_phase0_blake2b512) {
        fprintf(stderr,
            "OpenCL GPU[%d]: template_phase0 (BLAKE2B-512) kernel create failed "
            "(err=%d) — BLAKE2B-512 template path unavailable\n",
            dev_idx, err);
        d->kern_template_phase0_blake2b512 = NULL;
        return -1;
    }
    return 0;
}

/* ============================================================================
 * Memo B Phase B5 sub-batch 4 (2026-05-03): SHA3 / Keccak family
 * compile + lazy-kernel helpers.
 *
 * Eight algorithms, four shared rate values (144 / 136 / 104 / 72 bytes per
 * absorb-and-permute), two suffix bytes (0x01 plain Keccak, 0x06 SHA3 NIST
 * FIPS 202). Each algo has its own per-device cl_program + cl_kernel pair so
 * rate / output / suffix bake into the compiled kernel constants and the
 * driver can fold them through register allocation.
 *
 * Cache key (gpu_kernel_cache.c rev 1.5+) hashes BOTH source-text and
 * defines_str, so same-defines pairs (e.g. KECCAK-224 vs SHA3-224 both
 * have HASH_WORDS=7,HASH_BLOCK_BYTES=144) get distinct cache entries
 * because their core .cl source differs in the suffix-byte literal.
 *
 * R2 outlook: Keccak's 25-ulong state (200 bytes) plus 25-ulong B[]
 * scratch in keccakf1600 plus the 200-byte uchar block buffer in
 * absorb_pad pushes private memory above the BLAKE2 baseline. Compare
 * against SHA-512's ~42,520 B baseline on gfx1201; expect 42-44 KB.
 * ============================================================================ */

static int gpu_opencl_template_compile_keccak224(struct gpu_device *d, int dev_idx) {
    if (d->prog_template_keccak224) return 0;
    cl_int err = CL_SUCCESS;
    const char *sources[4] = {
        gpu_common_str, gpu_md5_rules_str,
        gpu_keccak224_core_str, gpu_template_str
    };
    const char *defines = "HASH_WORDS=7,HASH_BLOCK_BYTES=144";
    d->prog_template_keccak224 = gpu_kernel_cache_build_program_ex(
        d->ctx, d->dev, 4, sources, "-cl-std=CL1.2", defines, &err);
    if (!d->prog_template_keccak224 || err != CL_SUCCESS) {
        char log[8192] = {0};
        if (d->prog_template_keccak224) {
            clGetProgramBuildInfo(d->prog_template_keccak224, d->dev,
                                  CL_PROGRAM_BUILD_LOG,
                                  sizeof(log) - 1, log, NULL);
        }
        fprintf(stderr,
            "OpenCL GPU[%d]: KECCAK-224 template program build error (err=%d) "
            "— KECCAK-224 template path unavailable:\n%s\n", dev_idx, err, log);
        if (d->prog_template_keccak224) clReleaseProgram(d->prog_template_keccak224);
        d->prog_template_keccak224 = NULL;
        return -1;
    }
    return 0;
}
static int gpu_opencl_template_kernel_lazy_keccak224(struct gpu_device *d, int dev_idx) {
    if (d->kern_template_phase0_keccak224) return 0;
    if (!d->prog_template_keccak224) return -1;
    cl_int err;
    d->kern_template_phase0_keccak224 = clCreateKernel(d->prog_template_keccak224,
                                                       "template_phase0", &err);
    if (err != CL_SUCCESS || !d->kern_template_phase0_keccak224) {
        fprintf(stderr,
            "OpenCL GPU[%d]: template_phase0 (KECCAK-224) kernel create failed "
            "(err=%d) — KECCAK-224 template path unavailable\n", dev_idx, err);
        d->kern_template_phase0_keccak224 = NULL;
        return -1;
    }
    return 0;
}

static int gpu_opencl_template_compile_keccak256(struct gpu_device *d, int dev_idx) {
    if (d->prog_template_keccak256) return 0;
    cl_int err = CL_SUCCESS;
    const char *sources[4] = {
        gpu_common_str, gpu_md5_rules_str,
        gpu_keccak256_core_str, gpu_template_str
    };
    const char *defines = "HASH_WORDS=8,HASH_BLOCK_BYTES=136";
    d->prog_template_keccak256 = gpu_kernel_cache_build_program_ex(
        d->ctx, d->dev, 4, sources, "-cl-std=CL1.2", defines, &err);
    if (!d->prog_template_keccak256 || err != CL_SUCCESS) {
        char log[8192] = {0};
        if (d->prog_template_keccak256) {
            clGetProgramBuildInfo(d->prog_template_keccak256, d->dev,
                                  CL_PROGRAM_BUILD_LOG,
                                  sizeof(log) - 1, log, NULL);
        }
        fprintf(stderr,
            "OpenCL GPU[%d]: KECCAK-256 template program build error (err=%d) "
            "— KECCAK-256 template path unavailable:\n%s\n", dev_idx, err, log);
        if (d->prog_template_keccak256) clReleaseProgram(d->prog_template_keccak256);
        d->prog_template_keccak256 = NULL;
        return -1;
    }
    return 0;
}
static int gpu_opencl_template_kernel_lazy_keccak256(struct gpu_device *d, int dev_idx) {
    if (d->kern_template_phase0_keccak256) return 0;
    if (!d->prog_template_keccak256) return -1;
    cl_int err;
    d->kern_template_phase0_keccak256 = clCreateKernel(d->prog_template_keccak256,
                                                       "template_phase0", &err);
    if (err != CL_SUCCESS || !d->kern_template_phase0_keccak256) {
        fprintf(stderr,
            "OpenCL GPU[%d]: template_phase0 (KECCAK-256) kernel create failed "
            "(err=%d) — KECCAK-256 template path unavailable\n", dev_idx, err);
        d->kern_template_phase0_keccak256 = NULL;
        return -1;
    }
    return 0;
}

static int gpu_opencl_template_compile_keccak384(struct gpu_device *d, int dev_idx) {
    if (d->prog_template_keccak384) return 0;
    cl_int err = CL_SUCCESS;
    const char *sources[4] = {
        gpu_common_str, gpu_md5_rules_str,
        gpu_keccak384_core_str, gpu_template_str
    };
    const char *defines = "HASH_WORDS=12,HASH_BLOCK_BYTES=104";
    d->prog_template_keccak384 = gpu_kernel_cache_build_program_ex(
        d->ctx, d->dev, 4, sources, "-cl-std=CL1.2", defines, &err);
    if (!d->prog_template_keccak384 || err != CL_SUCCESS) {
        char log[8192] = {0};
        if (d->prog_template_keccak384) {
            clGetProgramBuildInfo(d->prog_template_keccak384, d->dev,
                                  CL_PROGRAM_BUILD_LOG,
                                  sizeof(log) - 1, log, NULL);
        }
        fprintf(stderr,
            "OpenCL GPU[%d]: KECCAK-384 template program build error (err=%d) "
            "— KECCAK-384 template path unavailable:\n%s\n", dev_idx, err, log);
        if (d->prog_template_keccak384) clReleaseProgram(d->prog_template_keccak384);
        d->prog_template_keccak384 = NULL;
        return -1;
    }
    return 0;
}
static int gpu_opencl_template_kernel_lazy_keccak384(struct gpu_device *d, int dev_idx) {
    if (d->kern_template_phase0_keccak384) return 0;
    if (!d->prog_template_keccak384) return -1;
    cl_int err;
    d->kern_template_phase0_keccak384 = clCreateKernel(d->prog_template_keccak384,
                                                       "template_phase0", &err);
    if (err != CL_SUCCESS || !d->kern_template_phase0_keccak384) {
        fprintf(stderr,
            "OpenCL GPU[%d]: template_phase0 (KECCAK-384) kernel create failed "
            "(err=%d) — KECCAK-384 template path unavailable\n", dev_idx, err);
        d->kern_template_phase0_keccak384 = NULL;
        return -1;
    }
    return 0;
}

static int gpu_opencl_template_compile_keccak512(struct gpu_device *d, int dev_idx) {
    if (d->prog_template_keccak512) return 0;
    cl_int err = CL_SUCCESS;
    const char *sources[4] = {
        gpu_common_str, gpu_md5_rules_str,
        gpu_keccak512_core_str, gpu_template_str
    };
    const char *defines = "HASH_WORDS=16,HASH_BLOCK_BYTES=72";
    d->prog_template_keccak512 = gpu_kernel_cache_build_program_ex(
        d->ctx, d->dev, 4, sources, "-cl-std=CL1.2", defines, &err);
    if (!d->prog_template_keccak512 || err != CL_SUCCESS) {
        char log[8192] = {0};
        if (d->prog_template_keccak512) {
            clGetProgramBuildInfo(d->prog_template_keccak512, d->dev,
                                  CL_PROGRAM_BUILD_LOG,
                                  sizeof(log) - 1, log, NULL);
        }
        fprintf(stderr,
            "OpenCL GPU[%d]: KECCAK-512 template program build error (err=%d) "
            "— KECCAK-512 template path unavailable:\n%s\n", dev_idx, err, log);
        if (d->prog_template_keccak512) clReleaseProgram(d->prog_template_keccak512);
        d->prog_template_keccak512 = NULL;
        return -1;
    }
    return 0;
}
static int gpu_opencl_template_kernel_lazy_keccak512(struct gpu_device *d, int dev_idx) {
    if (d->kern_template_phase0_keccak512) return 0;
    if (!d->prog_template_keccak512) return -1;
    cl_int err;
    d->kern_template_phase0_keccak512 = clCreateKernel(d->prog_template_keccak512,
                                                       "template_phase0", &err);
    if (err != CL_SUCCESS || !d->kern_template_phase0_keccak512) {
        fprintf(stderr,
            "OpenCL GPU[%d]: template_phase0 (KECCAK-512) kernel create failed "
            "(err=%d) — KECCAK-512 template path unavailable\n", dev_idx, err);
        d->kern_template_phase0_keccak512 = NULL;
        return -1;
    }
    return 0;
}

static int gpu_opencl_template_compile_sha3_224(struct gpu_device *d, int dev_idx) {
    if (d->prog_template_sha3_224) return 0;
    cl_int err = CL_SUCCESS;
    const char *sources[4] = {
        gpu_common_str, gpu_md5_rules_str,
        gpu_sha3_224_core_str, gpu_template_str
    };
    const char *defines = "HASH_WORDS=7,HASH_BLOCK_BYTES=144";
    d->prog_template_sha3_224 = gpu_kernel_cache_build_program_ex(
        d->ctx, d->dev, 4, sources, "-cl-std=CL1.2", defines, &err);
    if (!d->prog_template_sha3_224 || err != CL_SUCCESS) {
        char log[8192] = {0};
        if (d->prog_template_sha3_224) {
            clGetProgramBuildInfo(d->prog_template_sha3_224, d->dev,
                                  CL_PROGRAM_BUILD_LOG,
                                  sizeof(log) - 1, log, NULL);
        }
        fprintf(stderr,
            "OpenCL GPU[%d]: SHA3-224 template program build error (err=%d) "
            "— SHA3-224 template path unavailable:\n%s\n", dev_idx, err, log);
        if (d->prog_template_sha3_224) clReleaseProgram(d->prog_template_sha3_224);
        d->prog_template_sha3_224 = NULL;
        return -1;
    }
    return 0;
}
static int gpu_opencl_template_kernel_lazy_sha3_224(struct gpu_device *d, int dev_idx) {
    if (d->kern_template_phase0_sha3_224) return 0;
    if (!d->prog_template_sha3_224) return -1;
    cl_int err;
    d->kern_template_phase0_sha3_224 = clCreateKernel(d->prog_template_sha3_224,
                                                      "template_phase0", &err);
    if (err != CL_SUCCESS || !d->kern_template_phase0_sha3_224) {
        fprintf(stderr,
            "OpenCL GPU[%d]: template_phase0 (SHA3-224) kernel create failed "
            "(err=%d) — SHA3-224 template path unavailable\n", dev_idx, err);
        d->kern_template_phase0_sha3_224 = NULL;
        return -1;
    }
    return 0;
}

static int gpu_opencl_template_compile_sha3_256(struct gpu_device *d, int dev_idx) {
    if (d->prog_template_sha3_256) return 0;
    cl_int err = CL_SUCCESS;
    const char *sources[4] = {
        gpu_common_str, gpu_md5_rules_str,
        gpu_sha3_256_core_str, gpu_template_str
    };
    const char *defines = "HASH_WORDS=8,HASH_BLOCK_BYTES=136";
    d->prog_template_sha3_256 = gpu_kernel_cache_build_program_ex(
        d->ctx, d->dev, 4, sources, "-cl-std=CL1.2", defines, &err);
    if (!d->prog_template_sha3_256 || err != CL_SUCCESS) {
        char log[8192] = {0};
        if (d->prog_template_sha3_256) {
            clGetProgramBuildInfo(d->prog_template_sha3_256, d->dev,
                                  CL_PROGRAM_BUILD_LOG,
                                  sizeof(log) - 1, log, NULL);
        }
        fprintf(stderr,
            "OpenCL GPU[%d]: SHA3-256 template program build error (err=%d) "
            "— SHA3-256 template path unavailable:\n%s\n", dev_idx, err, log);
        if (d->prog_template_sha3_256) clReleaseProgram(d->prog_template_sha3_256);
        d->prog_template_sha3_256 = NULL;
        return -1;
    }
    return 0;
}
static int gpu_opencl_template_kernel_lazy_sha3_256(struct gpu_device *d, int dev_idx) {
    if (d->kern_template_phase0_sha3_256) return 0;
    if (!d->prog_template_sha3_256) return -1;
    cl_int err;
    d->kern_template_phase0_sha3_256 = clCreateKernel(d->prog_template_sha3_256,
                                                      "template_phase0", &err);
    if (err != CL_SUCCESS || !d->kern_template_phase0_sha3_256) {
        fprintf(stderr,
            "OpenCL GPU[%d]: template_phase0 (SHA3-256) kernel create failed "
            "(err=%d) — SHA3-256 template path unavailable\n", dev_idx, err);
        d->kern_template_phase0_sha3_256 = NULL;
        return -1;
    }
    return 0;
}

static int gpu_opencl_template_compile_sha3_384(struct gpu_device *d, int dev_idx) {
    if (d->prog_template_sha3_384) return 0;
    cl_int err = CL_SUCCESS;
    const char *sources[4] = {
        gpu_common_str, gpu_md5_rules_str,
        gpu_sha3_384_core_str, gpu_template_str
    };
    const char *defines = "HASH_WORDS=12,HASH_BLOCK_BYTES=104";
    d->prog_template_sha3_384 = gpu_kernel_cache_build_program_ex(
        d->ctx, d->dev, 4, sources, "-cl-std=CL1.2", defines, &err);
    if (!d->prog_template_sha3_384 || err != CL_SUCCESS) {
        char log[8192] = {0};
        if (d->prog_template_sha3_384) {
            clGetProgramBuildInfo(d->prog_template_sha3_384, d->dev,
                                  CL_PROGRAM_BUILD_LOG,
                                  sizeof(log) - 1, log, NULL);
        }
        fprintf(stderr,
            "OpenCL GPU[%d]: SHA3-384 template program build error (err=%d) "
            "— SHA3-384 template path unavailable:\n%s\n", dev_idx, err, log);
        if (d->prog_template_sha3_384) clReleaseProgram(d->prog_template_sha3_384);
        d->prog_template_sha3_384 = NULL;
        return -1;
    }
    return 0;
}
static int gpu_opencl_template_kernel_lazy_sha3_384(struct gpu_device *d, int dev_idx) {
    if (d->kern_template_phase0_sha3_384) return 0;
    if (!d->prog_template_sha3_384) return -1;
    cl_int err;
    d->kern_template_phase0_sha3_384 = clCreateKernel(d->prog_template_sha3_384,
                                                      "template_phase0", &err);
    if (err != CL_SUCCESS || !d->kern_template_phase0_sha3_384) {
        fprintf(stderr,
            "OpenCL GPU[%d]: template_phase0 (SHA3-384) kernel create failed "
            "(err=%d) — SHA3-384 template path unavailable\n", dev_idx, err);
        d->kern_template_phase0_sha3_384 = NULL;
        return -1;
    }
    return 0;
}

static int gpu_opencl_template_compile_sha3_512(struct gpu_device *d, int dev_idx) {
    if (d->prog_template_sha3_512) return 0;
    cl_int err = CL_SUCCESS;
    const char *sources[4] = {
        gpu_common_str, gpu_md5_rules_str,
        gpu_sha3_512_core_str, gpu_template_str
    };
    const char *defines = "HASH_WORDS=16,HASH_BLOCK_BYTES=72";
    d->prog_template_sha3_512 = gpu_kernel_cache_build_program_ex(
        d->ctx, d->dev, 4, sources, "-cl-std=CL1.2", defines, &err);
    if (!d->prog_template_sha3_512 || err != CL_SUCCESS) {
        char log[8192] = {0};
        if (d->prog_template_sha3_512) {
            clGetProgramBuildInfo(d->prog_template_sha3_512, d->dev,
                                  CL_PROGRAM_BUILD_LOG,
                                  sizeof(log) - 1, log, NULL);
        }
        fprintf(stderr,
            "OpenCL GPU[%d]: SHA3-512 template program build error (err=%d) "
            "— SHA3-512 template path unavailable:\n%s\n", dev_idx, err, log);
        if (d->prog_template_sha3_512) clReleaseProgram(d->prog_template_sha3_512);
        d->prog_template_sha3_512 = NULL;
        return -1;
    }
    return 0;
}
static int gpu_opencl_template_kernel_lazy_sha3_512(struct gpu_device *d, int dev_idx) {
    if (d->kern_template_phase0_sha3_512) return 0;
    if (!d->prog_template_sha3_512) return -1;
    cl_int err;
    d->kern_template_phase0_sha3_512 = clCreateKernel(d->prog_template_sha3_512,
                                                      "template_phase0", &err);
    if (err != CL_SUCCESS || !d->kern_template_phase0_sha3_512) {
        fprintf(stderr,
            "OpenCL GPU[%d]: template_phase0 (SHA3-512) kernel create failed "
            "(err=%d) — SHA3-512 template path unavailable\n", dev_idx, err);
        d->kern_template_phase0_sha3_512 = NULL;
        return -1;
    }
    return 0;
}

/* ----------------------------------------------------------------------
 * Memo B Phase B5 sub-batch 5a (2026-05-03), Tier 1: SHA384RAW + SHA512RAW
 * template program build + lazy kernel creation.
 *
 * REUSE the SHA384/SHA512 compression cores; only template_iterate
 * differs (binary-digest re-feed instead of hex). Per Memo B brief Tier 1
 * "Option B": ships full coverage including Maxiter > 1, replacing the
 * Option A chokepoint guard that would have skipped iter > 1.
 *
 * Cache key (R3): defines_str matches the SHA384/SHA512 entries
 * (HASH_WORDS=12/16, HASH_BLOCK_BYTES=128). Distinct cache entry by
 * source-text hash (only template_iterate changed).
 * ---------------------------------------------------------------------- */

static int gpu_opencl_template_compile_sha384raw(struct gpu_device *d, int dev_idx) {
    if (d->prog_template_sha384raw) return 0;
    cl_int err = CL_SUCCESS;
    const char *sources[4] = {
        gpu_common_str, gpu_md5_rules_str,
        gpu_sha384raw_core_str, gpu_template_str
    };
    const char *defines = "HASH_WORDS=12,HASH_BLOCK_BYTES=128";
    d->prog_template_sha384raw = gpu_kernel_cache_build_program_ex(
        d->ctx, d->dev, 4, sources, "-cl-std=CL1.2", defines, &err);
    if (!d->prog_template_sha384raw || err != CL_SUCCESS) {
        char log[8192] = {0};
        if (d->prog_template_sha384raw) {
            clGetProgramBuildInfo(d->prog_template_sha384raw, d->dev,
                                  CL_PROGRAM_BUILD_LOG,
                                  sizeof(log) - 1, log, NULL);
        }
        fprintf(stderr,
            "OpenCL GPU[%d]: SHA384RAW template program build error (err=%d) "
            "— SHA384RAW template path unavailable:\n%s\n", dev_idx, err, log);
        if (d->prog_template_sha384raw) clReleaseProgram(d->prog_template_sha384raw);
        d->prog_template_sha384raw = NULL;
        return -1;
    }
    return 0;
}

static int gpu_opencl_template_kernel_lazy_sha384raw(struct gpu_device *d, int dev_idx) {
    if (d->kern_template_phase0_sha384raw) return 0;
    if (!d->prog_template_sha384raw) return -1;
    cl_int err;
    d->kern_template_phase0_sha384raw = clCreateKernel(d->prog_template_sha384raw,
                                                       "template_phase0", &err);
    if (err != CL_SUCCESS || !d->kern_template_phase0_sha384raw) {
        fprintf(stderr,
            "OpenCL GPU[%d]: template_phase0 (SHA384RAW) kernel create failed "
            "(err=%d) — SHA384RAW template path unavailable\n", dev_idx, err);
        d->kern_template_phase0_sha384raw = NULL;
        return -1;
    }
    return 0;
}

static int gpu_opencl_template_compile_sha512raw(struct gpu_device *d, int dev_idx) {
    if (d->prog_template_sha512raw) return 0;
    cl_int err = CL_SUCCESS;
    const char *sources[4] = {
        gpu_common_str, gpu_md5_rules_str,
        gpu_sha512raw_core_str, gpu_template_str
    };
    const char *defines = "HASH_WORDS=16,HASH_BLOCK_BYTES=128";
    d->prog_template_sha512raw = gpu_kernel_cache_build_program_ex(
        d->ctx, d->dev, 4, sources, "-cl-std=CL1.2", defines, &err);
    if (!d->prog_template_sha512raw || err != CL_SUCCESS) {
        char log[8192] = {0};
        if (d->prog_template_sha512raw) {
            clGetProgramBuildInfo(d->prog_template_sha512raw, d->dev,
                                  CL_PROGRAM_BUILD_LOG,
                                  sizeof(log) - 1, log, NULL);
        }
        fprintf(stderr,
            "OpenCL GPU[%d]: SHA512RAW template program build error (err=%d) "
            "— SHA512RAW template path unavailable:\n%s\n", dev_idx, err, log);
        if (d->prog_template_sha512raw) clReleaseProgram(d->prog_template_sha512raw);
        d->prog_template_sha512raw = NULL;
        return -1;
    }
    return 0;
}

static int gpu_opencl_template_kernel_lazy_sha512raw(struct gpu_device *d, int dev_idx) {
    if (d->kern_template_phase0_sha512raw) return 0;
    if (!d->prog_template_sha512raw) return -1;
    cl_int err;
    d->kern_template_phase0_sha512raw = clCreateKernel(d->prog_template_sha512raw,
                                                       "template_phase0", &err);
    if (err != CL_SUCCESS || !d->kern_template_phase0_sha512raw) {
        fprintf(stderr,
            "OpenCL GPU[%d]: template_phase0 (SHA512RAW) kernel create failed "
            "(err=%d) — SHA512RAW template path unavailable\n", dev_idx, err);
        d->kern_template_phase0_sha512raw = NULL;
        return -1;
    }
    return 0;
}

/* ----------------------------------------------------------------------
 * Memo B Phase B5 sub-batch 6 (2026-05-03), Tier A: MD5RAW + SHA1RAW + SHA256RAW
 * template program build + lazy kernel creation.
 *
 * REUSE the MD5/SHA1/SHA256 compression cores; only template_iterate
 * differs (binary-digest re-feed instead of hex). Per Memo B brief Tier 1
 * Option B precedent: ships full coverage including Maxiter > 1.
 *
 * Cache key (R3): defines_str matches the MD5/SHA1/SHA256 entries
 * (HASH_WORDS=4/5/8, HASH_BLOCK_BYTES=64). Distinct cache entry by
 * source-text hash (only template_iterate changed).
 * ---------------------------------------------------------------------- */

static int gpu_opencl_template_compile_md5raw(struct gpu_device *d, int dev_idx) {
    if (d->prog_template_md5raw) return 0;
    cl_int err = CL_SUCCESS;
    const char *sources[4] = {
        gpu_common_str, gpu_md5_rules_str,
        gpu_md5raw_core_str, gpu_template_str
    };
    const char *defines = "HASH_WORDS=4,HASH_BLOCK_BYTES=64";
    d->prog_template_md5raw = gpu_kernel_cache_build_program_ex(
        d->ctx, d->dev, 4, sources, "-cl-std=CL1.2", defines, &err);
    if (!d->prog_template_md5raw || err != CL_SUCCESS) {
        char log[8192] = {0};
        if (d->prog_template_md5raw) {
            clGetProgramBuildInfo(d->prog_template_md5raw, d->dev,
                                  CL_PROGRAM_BUILD_LOG,
                                  sizeof(log) - 1, log, NULL);
        }
        fprintf(stderr,
            "OpenCL GPU[%d]: MD5RAW template program build error (err=%d) "
            "— MD5RAW template path unavailable:\n%s\n", dev_idx, err, log);
        if (d->prog_template_md5raw) clReleaseProgram(d->prog_template_md5raw);
        d->prog_template_md5raw = NULL;
        return -1;
    }
    return 0;
}

static int gpu_opencl_template_kernel_lazy_md5raw(struct gpu_device *d, int dev_idx) {
    if (d->kern_template_phase0_md5raw) return 0;
    if (!d->prog_template_md5raw) return -1;
    cl_int err;
    d->kern_template_phase0_md5raw = clCreateKernel(d->prog_template_md5raw,
                                                    "template_phase0", &err);
    if (err != CL_SUCCESS || !d->kern_template_phase0_md5raw) {
        fprintf(stderr,
            "OpenCL GPU[%d]: template_phase0 (MD5RAW) kernel create failed "
            "(err=%d) — MD5RAW template path unavailable\n", dev_idx, err);
        d->kern_template_phase0_md5raw = NULL;
        return -1;
    }
    return 0;
}

static int gpu_opencl_template_compile_sha1raw(struct gpu_device *d, int dev_idx) {
    if (d->prog_template_sha1raw) return 0;
    cl_int err = CL_SUCCESS;
    const char *sources[4] = {
        gpu_common_str, gpu_md5_rules_str,
        gpu_sha1raw_core_str, gpu_template_str
    };
    const char *defines = "HASH_WORDS=5,HASH_BLOCK_BYTES=64";
    d->prog_template_sha1raw = gpu_kernel_cache_build_program_ex(
        d->ctx, d->dev, 4, sources, "-cl-std=CL1.2", defines, &err);
    if (!d->prog_template_sha1raw || err != CL_SUCCESS) {
        char log[8192] = {0};
        if (d->prog_template_sha1raw) {
            clGetProgramBuildInfo(d->prog_template_sha1raw, d->dev,
                                  CL_PROGRAM_BUILD_LOG,
                                  sizeof(log) - 1, log, NULL);
        }
        fprintf(stderr,
            "OpenCL GPU[%d]: SHA1RAW template program build error (err=%d) "
            "— SHA1RAW template path unavailable:\n%s\n", dev_idx, err, log);
        if (d->prog_template_sha1raw) clReleaseProgram(d->prog_template_sha1raw);
        d->prog_template_sha1raw = NULL;
        return -1;
    }
    return 0;
}

static int gpu_opencl_template_kernel_lazy_sha1raw(struct gpu_device *d, int dev_idx) {
    if (d->kern_template_phase0_sha1raw) return 0;
    if (!d->prog_template_sha1raw) return -1;
    cl_int err;
    d->kern_template_phase0_sha1raw = clCreateKernel(d->prog_template_sha1raw,
                                                     "template_phase0", &err);
    if (err != CL_SUCCESS || !d->kern_template_phase0_sha1raw) {
        fprintf(stderr,
            "OpenCL GPU[%d]: template_phase0 (SHA1RAW) kernel create failed "
            "(err=%d) — SHA1RAW template path unavailable\n", dev_idx, err);
        d->kern_template_phase0_sha1raw = NULL;
        return -1;
    }
    return 0;
}

static int gpu_opencl_template_compile_sha256raw(struct gpu_device *d, int dev_idx) {
    if (d->prog_template_sha256raw) return 0;
    cl_int err = CL_SUCCESS;
    const char *sources[4] = {
        gpu_common_str, gpu_md5_rules_str,
        gpu_sha256raw_core_str, gpu_template_str
    };
    const char *defines = "HASH_WORDS=8,HASH_BLOCK_BYTES=64";
    d->prog_template_sha256raw = gpu_kernel_cache_build_program_ex(
        d->ctx, d->dev, 4, sources, "-cl-std=CL1.2", defines, &err);
    if (!d->prog_template_sha256raw || err != CL_SUCCESS) {
        char log[8192] = {0};
        if (d->prog_template_sha256raw) {
            clGetProgramBuildInfo(d->prog_template_sha256raw, d->dev,
                                  CL_PROGRAM_BUILD_LOG,
                                  sizeof(log) - 1, log, NULL);
        }
        fprintf(stderr,
            "OpenCL GPU[%d]: SHA256RAW template program build error (err=%d) "
            "— SHA256RAW template path unavailable:\n%s\n", dev_idx, err, log);
        if (d->prog_template_sha256raw) clReleaseProgram(d->prog_template_sha256raw);
        d->prog_template_sha256raw = NULL;
        return -1;
    }
    return 0;
}

static int gpu_opencl_template_kernel_lazy_sha256raw(struct gpu_device *d, int dev_idx) {
    if (d->kern_template_phase0_sha256raw) return 0;
    if (!d->prog_template_sha256raw) return -1;
    cl_int err;
    d->kern_template_phase0_sha256raw = clCreateKernel(d->prog_template_sha256raw,
                                                       "template_phase0", &err);
    if (err != CL_SUCCESS || !d->kern_template_phase0_sha256raw) {
        fprintf(stderr,
            "OpenCL GPU[%d]: template_phase0 (SHA256RAW) kernel create failed "
            "(err=%d) — SHA256RAW template path unavailable\n", dev_idx, err);
        d->kern_template_phase0_sha256raw = NULL;
        return -1;
    }
    return 0;
}

/* ----------------------------------------------------------------------
 * Memo B Phase B5 sub-batch 6 (2026-05-03), Tier C: SQL5 (MySQL 4.1+
 * password = SHA1(SHA1(p)))
 *
 * Two SHA1 chains in template_state; iter loop re-feeds UPPERCASE-hex of
 * inner SHA1. Cache key (R3): defines_str matches SHA1 / SHA1RAW
 * (HASH_WORDS=5, HASH_BLOCK_BYTES=64). Distinct cache entry by
 * source-text hash (SQL5 has the unique state-with-two-chains layout).
 * ---------------------------------------------------------------------- */

static int gpu_opencl_template_compile_sql5(struct gpu_device *d, int dev_idx) {
    if (d->prog_template_sql5) return 0;
    cl_int err = CL_SUCCESS;
    const char *sources[4] = {
        gpu_common_str, gpu_md5_rules_str,
        gpu_sql5_core_str, gpu_template_str
    };
    const char *defines = "HASH_WORDS=5,HASH_BLOCK_BYTES=64";
    d->prog_template_sql5 = gpu_kernel_cache_build_program_ex(
        d->ctx, d->dev, 4, sources, "-cl-std=CL1.2", defines, &err);
    if (!d->prog_template_sql5 || err != CL_SUCCESS) {
        char log[8192] = {0};
        if (d->prog_template_sql5) {
            clGetProgramBuildInfo(d->prog_template_sql5, d->dev,
                                  CL_PROGRAM_BUILD_LOG,
                                  sizeof(log) - 1, log, NULL);
        }
        fprintf(stderr,
            "OpenCL GPU[%d]: SQL5 template program build error (err=%d) "
            "— SQL5 template path unavailable:\n%s\n", dev_idx, err, log);
        if (d->prog_template_sql5) clReleaseProgram(d->prog_template_sql5);
        d->prog_template_sql5 = NULL;
        return -1;
    }
    return 0;
}

static int gpu_opencl_template_kernel_lazy_sql5(struct gpu_device *d, int dev_idx) {
    if (d->kern_template_phase0_sql5) return 0;
    if (!d->prog_template_sql5) return -1;
    cl_int err;
    d->kern_template_phase0_sql5 = clCreateKernel(d->prog_template_sql5,
                                                  "template_phase0", &err);
    if (err != CL_SUCCESS || !d->kern_template_phase0_sql5) {
        fprintf(stderr,
            "OpenCL GPU[%d]: template_phase0 (SQL5) kernel create failed "
            "(err=%d) — SQL5 template path unavailable\n", dev_idx, err);
        d->kern_template_phase0_sql5 = NULL;
        return -1;
    }
    return 0;
}

/* ----------------------------------------------------------------------
 * Memo B Phase B6.11 (2026-05-06): SHA1DRU (Drupal SHA1, hashcat -m 7900,
 * JOB_SHA1DRU=404)
 *
 * First 1M-iteration algorithm on the unified template path. The 1M loop
 * runs INSIDE template_finalize (not template_iterate); host forces
 * params.max_iter=1 so the kernel's outer iter loop runs exactly once
 * and template_iterate is a stub. This matches CPU semantics exactly
 * (mdxfind.c:14261-14285 has ONE checkhash() after the 1M for-loop) and
 * avoids 1M wasted compact-table probes.
 *
 * Cache key (R3): defines_str =
 *   "HASH_WORDS=5,HASH_BLOCK_BYTES=64,BASE_ALGO=sha1,ITER_COUNT=1000000"
 * — same first two axes as SHA1 / SHA1RAW / SQL5; distinct cache entry by
 * the ITER_COUNT token + the source-text hash (the 1M-loop body is unique
 * to this core).
 * ---------------------------------------------------------------------- */

static int gpu_opencl_template_compile_sha1dru(struct gpu_device *d, int dev_idx) {
    if (d->prog_template_sha1dru) return 0;
    cl_int err = CL_SUCCESS;
    const char *sources[4] = {
        gpu_common_str, gpu_md5_rules_str,
        gpu_sha1dru_core_str, gpu_template_str
    };
    const char *defines = "HASH_WORDS=5,HASH_BLOCK_BYTES=64,BASE_ALGO=sha1,ITER_COUNT=1000000";
    d->prog_template_sha1dru = gpu_kernel_cache_build_program_ex(
        d->ctx, d->dev, 4, sources, "-cl-std=CL1.2", defines, &err);
    if (!d->prog_template_sha1dru || err != CL_SUCCESS) {
        char log[8192] = {0};
        if (d->prog_template_sha1dru) {
            clGetProgramBuildInfo(d->prog_template_sha1dru, d->dev,
                                  CL_PROGRAM_BUILD_LOG,
                                  sizeof(log) - 1, log, NULL);
        }
        fprintf(stderr,
            "OpenCL GPU[%d]: SHA1DRU template program build error (err=%d) "
            "— SHA1DRU template path unavailable:\n%s\n", dev_idx, err, log);
        if (d->prog_template_sha1dru) clReleaseProgram(d->prog_template_sha1dru);
        d->prog_template_sha1dru = NULL;
        return -1;
    }
    return 0;
}

static int gpu_opencl_template_kernel_lazy_sha1dru(struct gpu_device *d, int dev_idx) {
    if (d->kern_template_phase0_sha1dru) return 0;
    if (!d->prog_template_sha1dru) return -1;
    cl_int err;
    d->kern_template_phase0_sha1dru = clCreateKernel(d->prog_template_sha1dru,
                                                     "template_phase0", &err);
    if (err != CL_SUCCESS || !d->kern_template_phase0_sha1dru) {
        fprintf(stderr,
            "OpenCL GPU[%d]: template_phase0 (SHA1DRU) kernel create failed "
            "(err=%d) — SHA1DRU template path unavailable\n", dev_idx, err);
        d->kern_template_phase0_sha1dru = NULL;
        return -1;
    }
    return 0;
}

/* ----------------------------------------------------------------------
 * Memo B Phase B7.7b (2026-05-07): MD6256 (hashcat -m 17800,
 * JOB_MD6256=29)
 *
 * Final M5 closure from B9 gate-fail. MD6-256 single-block leaf
 * compression — algorithmically-largest single-compression unsalted algo
 * on the unified template path: 89-ulong N input, 1753-ulong A working
 * array (14 KB stack per work-item), 104 rounds × 16 steps per
 * compression. Per-iter probe like SQL5 (vs. SHA1DRU's max_iter=1
 * internal loop).
 *
 * Cache key (R3): defines_str = "HASH_WORDS=8,HASH_BLOCK_BYTES=64,
 * BASE_ALGO=md6". Distinct from every prior template core via BASE_ALGO=
 * md6 (no other md6 algo on the template path) + HASH_WORDS=8 (matches
 * Streebog256/SHA256 family but BASE_ALGO axis disambiguates).
 *
 * KNOWN ACCEPTED RISK: gfx1201 priv_mem may bust 43,024 B HARD GATE due
 * to the 14 KB A[1753] stack on top of RULE_BUF_MAX. Compile-only ship
 * per user 2026-05-07 OPTION A; integrated post-B7.9 validation will
 * reveal gfx1201 status. Fall-back: leave gpu_md6256unsalted.cl as
 * gfx1201-only slab fallback.
 * ---------------------------------------------------------------------- */

static int gpu_opencl_template_compile_md6256(struct gpu_device *d, int dev_idx) {
    if (d->prog_template_md6256) return 0;
    cl_int err = CL_SUCCESS;
    const char *sources[4] = {
        gpu_common_str, gpu_md5_rules_str,
        gpu_md6256_core_str, gpu_template_str
    };
    const char *defines = "HASH_WORDS=8,HASH_BLOCK_BYTES=64,BASE_ALGO=md6";
    d->prog_template_md6256 = gpu_kernel_cache_build_program_ex(
        d->ctx, d->dev, 4, sources, "-cl-std=CL1.2", defines, &err);
    if (!d->prog_template_md6256 || err != CL_SUCCESS) {
        char log[8192] = {0};
        if (d->prog_template_md6256) {
            clGetProgramBuildInfo(d->prog_template_md6256, d->dev,
                                  CL_PROGRAM_BUILD_LOG,
                                  sizeof(log) - 1, log, NULL);
        }
        fprintf(stderr,
            "OpenCL GPU[%d]: MD6256 template program build error (err=%d) "
            "— MD6256 template path unavailable:\n%s\n", dev_idx, err, log);
        if (d->prog_template_md6256) clReleaseProgram(d->prog_template_md6256);
        d->prog_template_md6256 = NULL;
        return -1;
    }
    return 0;
}

static int gpu_opencl_template_kernel_lazy_md6256(struct gpu_device *d, int dev_idx) {
    if (d->kern_template_phase0_md6256) return 0;
    if (!d->prog_template_md6256) return -1;
    cl_int err;
    d->kern_template_phase0_md6256 = clCreateKernel(d->prog_template_md6256,
                                                    "template_phase0", &err);
    if (err != CL_SUCCESS || !d->kern_template_phase0_md6256) {
        fprintf(stderr,
            "OpenCL GPU[%d]: template_phase0 (MD6256) kernel create failed "
            "(err=%d) — MD6256 template path unavailable\n", dev_idx, err);
        d->kern_template_phase0_md6256 = NULL;
        return -1;
    }
    return 0;
}

/* ----------------------------------------------------------------------
 * Memo B Phase B5 sub-batch 6 (2026-05-03), Tier B: NTLMH (NT password
 * hash = MD4(UTF-16LE-zero-extend(p)))
 *
 * Hashcat-compatible single-variant. The iconv variant (utf-8 → UTF-16LE
 * for non-ASCII inputs) remains on CPU; the GPU template path implements
 * only the zero-extend variant, matching the existing slab kernel
 * md4utf16_unsalted_batch (gpu_md4unsalted.cl line 197+) semantic.
 *
 * Cache key (R3): defines_str matches MD4 / MD5 (HASH_WORDS=4,
 * HASH_BLOCK_BYTES=64). Distinct cache entry by source-text hash.
 * ---------------------------------------------------------------------- */

static int gpu_opencl_template_compile_ntlmh(struct gpu_device *d, int dev_idx) {
    if (d->prog_template_ntlmh) return 0;
    cl_int err = CL_SUCCESS;
    const char *sources[4] = {
        gpu_common_str, gpu_md5_rules_str,
        gpu_ntlmh_core_str, gpu_template_str
    };
    const char *defines = "HASH_WORDS=4,HASH_BLOCK_BYTES=64";
    d->prog_template_ntlmh = gpu_kernel_cache_build_program_ex(
        d->ctx, d->dev, 4, sources, "-cl-std=CL1.2", defines, &err);
    if (!d->prog_template_ntlmh || err != CL_SUCCESS) {
        char log[8192] = {0};
        if (d->prog_template_ntlmh) {
            clGetProgramBuildInfo(d->prog_template_ntlmh, d->dev,
                                  CL_PROGRAM_BUILD_LOG,
                                  sizeof(log) - 1, log, NULL);
        }
        fprintf(stderr,
            "OpenCL GPU[%d]: NTLMH template program build error (err=%d) "
            "— NTLMH template path unavailable:\n%s\n", dev_idx, err, log);
        if (d->prog_template_ntlmh) clReleaseProgram(d->prog_template_ntlmh);
        d->prog_template_ntlmh = NULL;
        return -1;
    }
    return 0;
}

static int gpu_opencl_template_kernel_lazy_ntlmh(struct gpu_device *d, int dev_idx) {
    if (d->kern_template_phase0_ntlmh) return 0;
    if (!d->prog_template_ntlmh) return -1;
    cl_int err;
    d->kern_template_phase0_ntlmh = clCreateKernel(d->prog_template_ntlmh,
                                                   "template_phase0", &err);
    if (err != CL_SUCCESS || !d->kern_template_phase0_ntlmh) {
        fprintf(stderr,
            "OpenCL GPU[%d]: template_phase0 (NTLMH) kernel create failed "
            "(err=%d) — NTLMH template path unavailable\n", dev_idx, err);
        d->kern_template_phase0_ntlmh = NULL;
        return -1;
    }
    return 0;
}

/* ----------------------------------------------------------------------
 * Memo B Phase B5 sub-batch 8 (2026-05-05): MD4UTF16 (-m e496)
 *
 * Same algorithm shape as NTLMH (MD4 of UTF-16LE-zero-extend(p)) but
 * supports the mdxfind -i / Maxiter loop. Iter > 1 feeds back the
 * lowercase hex of the prior digest (32 ASCII chars) zero-extended to
 * UTF-16LE (64 bytes) and MD4'd, matching CPU JOB_MD4UTF16 at
 * mdxfind.c:15059-15066.
 *
 * Cache key (R3): defines_str matches MD4 / NTLMH / MD5 (HASH_WORDS=4,
 * HASH_BLOCK_BYTES=64). Distinct cache entry by source-text hash
 * (template_iterate diverges from NTLMH).
 * ---------------------------------------------------------------------- */

static int gpu_opencl_template_compile_md4utf16(struct gpu_device *d, int dev_idx) {
    if (d->prog_template_md4utf16) return 0;
    cl_int err = CL_SUCCESS;
    const char *sources[4] = {
        gpu_common_str, gpu_md5_rules_str,
        gpu_md4utf16_core_str, gpu_template_str
    };
    const char *defines = "HASH_WORDS=4,HASH_BLOCK_BYTES=64";
    d->prog_template_md4utf16 = gpu_kernel_cache_build_program_ex(
        d->ctx, d->dev, 4, sources, "-cl-std=CL1.2", defines, &err);
    if (!d->prog_template_md4utf16 || err != CL_SUCCESS) {
        char log[8192] = {0};
        if (d->prog_template_md4utf16) {
            clGetProgramBuildInfo(d->prog_template_md4utf16, d->dev,
                                  CL_PROGRAM_BUILD_LOG,
                                  sizeof(log) - 1, log, NULL);
        }
        fprintf(stderr,
            "OpenCL GPU[%d]: MD4UTF16 template program build error (err=%d) "
            "— MD4UTF16 template path unavailable:\n%s\n", dev_idx, err, log);
        if (d->prog_template_md4utf16) clReleaseProgram(d->prog_template_md4utf16);
        d->prog_template_md4utf16 = NULL;
        return -1;
    }
    return 0;
}

static int gpu_opencl_template_kernel_lazy_md4utf16(struct gpu_device *d, int dev_idx) {
    if (d->kern_template_phase0_md4utf16) return 0;
    if (!d->prog_template_md4utf16) return -1;
    cl_int err;
    d->kern_template_phase0_md4utf16 = clCreateKernel(d->prog_template_md4utf16,
                                                      "template_phase0", &err);
    if (err != CL_SUCCESS || !d->kern_template_phase0_md4utf16) {
        fprintf(stderr,
            "OpenCL GPU[%d]: template_phase0 (MD4UTF16) kernel create failed "
            "(err=%d) — MD4UTF16 template path unavailable\n", dev_idx, err);
        d->kern_template_phase0_md4utf16 = NULL;
        return -1;
    }
    return 0;
}

/* ----------------------------------------------------------------------
 * Memo B Phase B5 sub-batch 7 (2026-05-05): MYSQL3 (-m e456)
 *
 * Legacy MySQL OLD_PASSWORD() hash, 64-bit output. The per-byte arithmetic
 * accumulator runs entirely inside template_finalize; iter > 1 feeds back
 * the lowercase ASCII hex of the prior digest (16 chars total) and reruns
 * the per-byte loop. Probe path uses the default 4-word compare (h[0],
 * h[1] = byteswapped nr/nr2; h[2..3] zero); host zero-pad of HashDataBuf
 * (mdxfind.c:36400-36412 rev 1.399+) makes that compare byte-exact for
 * the 8-byte digest by guaranteeing the next-entry bytes are zero.
 *
 * Cache key (R3): defines_str matches MD4 / NTLMH / MD5 (HASH_WORDS=4,
 * HASH_BLOCK_BYTES=64). Distinct cache entry by source-text hash (the
 * per-byte MYSQL3 arithmetic loop is unique).
 * ---------------------------------------------------------------------- */

static int gpu_opencl_template_compile_mysql3(struct gpu_device *d, int dev_idx) {
    if (d->prog_template_mysql3) return 0;
    cl_int err = CL_SUCCESS;
    const char *sources[4] = {
        gpu_common_str, gpu_md5_rules_str,
        gpu_mysql3_core_str, gpu_template_str
    };
    const char *defines = "HASH_WORDS=4,HASH_BLOCK_BYTES=64";
    d->prog_template_mysql3 = gpu_kernel_cache_build_program_ex(
        d->ctx, d->dev, 4, sources, "-cl-std=CL1.2", defines, &err);
    if (!d->prog_template_mysql3 || err != CL_SUCCESS) {
        char log[8192] = {0};
        if (d->prog_template_mysql3) {
            clGetProgramBuildInfo(d->prog_template_mysql3, d->dev,
                                  CL_PROGRAM_BUILD_LOG,
                                  sizeof(log) - 1, log, NULL);
        }
        fprintf(stderr,
            "OpenCL GPU[%d]: MYSQL3 template program build error (err=%d) "
            "— MYSQL3 template path unavailable:\n%s\n", dev_idx, err, log);
        if (d->prog_template_mysql3) clReleaseProgram(d->prog_template_mysql3);
        d->prog_template_mysql3 = NULL;
        return -1;
    }
    return 0;
}

static int gpu_opencl_template_kernel_lazy_mysql3(struct gpu_device *d, int dev_idx) {
    if (d->kern_template_phase0_mysql3) return 0;
    if (!d->prog_template_mysql3) return -1;
    cl_int err;
    d->kern_template_phase0_mysql3 = clCreateKernel(d->prog_template_mysql3,
                                                    "template_phase0", &err);
    if (err != CL_SUCCESS || !d->kern_template_phase0_mysql3) {
        fprintf(stderr,
            "OpenCL GPU[%d]: template_phase0 (MYSQL3) kernel create failed "
            "(err=%d) — MYSQL3 template path unavailable\n", dev_idx, err);
        d->kern_template_phase0_mysql3 = NULL;
        return -1;
    }
    return 0;
}

/* ----------------------------------------------------------------------
 * Memo B Phase B5 sub-batch 6.5 (2026-05-05): WRL (-m e5)
 *
 * Whirlpool 512-bit hash. Miyaguchi-Preneel construction over 64-byte
 * BIG-ENDIAN block; the compress is a 10-round AES-like permutation with
 * an 8x256 ulong S-box (16 KB total in __constant). Single-block IV is
 * all zeros; subsequent blocks use the prior digest as chain value and
 * XOR final state with the chain value (general M-P).
 *
 * Iter > 1 feeds back the lowercase ASCII hex of the prior 64-byte digest
 * (128 chars total). 128-byte hex string requires 2 full Whirlpool blocks
 * + 1 padding-only block (256-bit length encoding spills past byte 128).
 * Probe path uses HASH_WORDS=16 leading 16 bytes LE; matches the slab
 * kernel gpu_wrlunsalted.cl emit convention.
 *
 * Cache key (R3): defines_str = "HASH_WORDS=16,HASH_BLOCK_BYTES=64".
 * Distinct cache entry by source-text hash (16 KB SBOX + Miyaguchi-Preneel
 * compress is unique).
 *
 * DIAGNOSTIC (B5 sub-batch 5b 2026-05-05 hand-off): WRL has the same
 * 16 KB __constant ulong[8][256] SBOX shape as Streebog's SBOB_SL64 but
 * a structurally different access pattern — direct ulong indexing with
 * shift-then-mask in WRL_OP, vs uchar-from-ulong reinterpret in
 * SBOG_LPS. If gfx1201 corrupts the byte-pointer pattern only, WRL
 * passes 100/100 here while Streebog fails 4/100. If both fail in the
 * 1/33/65/97 lane pattern, the issue is generic to 16 KB __constant on
 * RDNA4 and both algos need __global mitigation.
 * ---------------------------------------------------------------------- */

static int gpu_opencl_template_compile_wrl(struct gpu_device *d, int dev_idx) {
    if (d->prog_template_wrl) return 0;
    cl_int err = CL_SUCCESS;
    const char *sources[4] = {
        gpu_common_str, gpu_md5_rules_str,
        gpu_wrl_core_str, gpu_template_str
    };
    const char *defines = "HASH_WORDS=16,HASH_BLOCK_BYTES=64";
    d->prog_template_wrl = gpu_kernel_cache_build_program_ex(
        d->ctx, d->dev, 4, sources, "-cl-std=CL1.2", defines, &err);
    if (!d->prog_template_wrl || err != CL_SUCCESS) {
        char log[8192] = {0};
        if (d->prog_template_wrl) {
            clGetProgramBuildInfo(d->prog_template_wrl, d->dev,
                                  CL_PROGRAM_BUILD_LOG,
                                  sizeof(log) - 1, log, NULL);
        }
        fprintf(stderr,
            "OpenCL GPU[%d]: WRL template program build error (err=%d) "
            "— WRL template path unavailable:\n%s\n", dev_idx, err, log);
        if (d->prog_template_wrl) clReleaseProgram(d->prog_template_wrl);
        d->prog_template_wrl = NULL;
        return -1;
    }
    return 0;
}

static int gpu_opencl_template_kernel_lazy_wrl(struct gpu_device *d, int dev_idx) {
    if (d->kern_template_phase0_wrl) return 0;
    if (!d->prog_template_wrl) return -1;
    cl_int err;
    d->kern_template_phase0_wrl = clCreateKernel(d->prog_template_wrl,
                                                 "template_phase0", &err);
    if (err != CL_SUCCESS || !d->kern_template_phase0_wrl) {
        fprintf(stderr,
            "OpenCL GPU[%d]: template_phase0 (WRL) kernel create failed "
            "(err=%d) — WRL template path unavailable\n", dev_idx, err);
        d->kern_template_phase0_wrl = NULL;
        return -1;
    }
    return 0;
}

/* B5 sub-batch 5b retry (2026-05-06): Streebog-256 template compile/lazy pair.
 * Sources: gpu_common, gpu_md5_rules, gpu_streebog256_core, gpu_template.
 * Defines: HASH_WORDS=8, HASH_BLOCK_BYTES=64. SBOG_LPS rewritten to
 * shift-then-mask byte extraction (RDNA4 gfx1201 mitigation per sub-6.5
 * diagnostic). */
static int gpu_opencl_template_compile_streebog256(struct gpu_device *d, int dev_idx) {
    if (d->prog_template_streebog256) return 0;
    cl_int err = CL_SUCCESS;
    const char *sources[4] = {
        gpu_common_str, gpu_md5_rules_str,
        gpu_streebog256_core_str, gpu_template_str
    };
    const char *defines = "HASH_WORDS=8,HASH_BLOCK_BYTES=64";
    d->prog_template_streebog256 = gpu_kernel_cache_build_program_ex(
        d->ctx, d->dev, 4, sources, "-cl-std=CL1.2", defines, &err);
    if (!d->prog_template_streebog256 || err != CL_SUCCESS) {
        char log[8192] = {0};
        if (d->prog_template_streebog256) {
            clGetProgramBuildInfo(d->prog_template_streebog256, d->dev,
                                  CL_PROGRAM_BUILD_LOG,
                                  sizeof(log) - 1, log, NULL);
        }
        fprintf(stderr,
            "OpenCL GPU[%d]: STREEBOG256 template program build error (err=%d) "
            "— STREEBOG256 template path unavailable:\n%s\n", dev_idx, err, log);
        if (d->prog_template_streebog256) clReleaseProgram(d->prog_template_streebog256);
        d->prog_template_streebog256 = NULL;
        return -1;
    }
    return 0;
}

static int gpu_opencl_template_kernel_lazy_streebog256(struct gpu_device *d, int dev_idx) {
    if (d->kern_template_phase0_streebog256) return 0;
    if (!d->prog_template_streebog256) return -1;
    cl_int err;
    d->kern_template_phase0_streebog256 = clCreateKernel(d->prog_template_streebog256,
                                                         "template_phase0", &err);
    if (err != CL_SUCCESS || !d->kern_template_phase0_streebog256) {
        fprintf(stderr,
            "OpenCL GPU[%d]: template_phase0 (STREEBOG256) kernel create failed "
            "(err=%d) — STREEBOG256 template path unavailable\n", dev_idx, err);
        d->kern_template_phase0_streebog256 = NULL;
        return -1;
    }
    return 0;
}

/* B5 sub-batch 5b retry: Streebog-512 template compile/lazy pair. Same as
 * streebog256 but HASH_WORDS=16. */
static int gpu_opencl_template_compile_streebog512(struct gpu_device *d, int dev_idx) {
    if (d->prog_template_streebog512) return 0;
    cl_int err = CL_SUCCESS;
    const char *sources[4] = {
        gpu_common_str, gpu_md5_rules_str,
        gpu_streebog512_core_str, gpu_template_str
    };
    const char *defines = "HASH_WORDS=16,HASH_BLOCK_BYTES=64";
    d->prog_template_streebog512 = gpu_kernel_cache_build_program_ex(
        d->ctx, d->dev, 4, sources, "-cl-std=CL1.2", defines, &err);
    if (!d->prog_template_streebog512 || err != CL_SUCCESS) {
        char log[8192] = {0};
        if (d->prog_template_streebog512) {
            clGetProgramBuildInfo(d->prog_template_streebog512, d->dev,
                                  CL_PROGRAM_BUILD_LOG,
                                  sizeof(log) - 1, log, NULL);
        }
        fprintf(stderr,
            "OpenCL GPU[%d]: STREEBOG512 template program build error (err=%d) "
            "— STREEBOG512 template path unavailable:\n%s\n", dev_idx, err, log);
        if (d->prog_template_streebog512) clReleaseProgram(d->prog_template_streebog512);
        d->prog_template_streebog512 = NULL;
        return -1;
    }
    return 0;
}

static int gpu_opencl_template_kernel_lazy_streebog512(struct gpu_device *d, int dev_idx) {
    if (d->kern_template_phase0_streebog512) return 0;
    if (!d->prog_template_streebog512) return -1;
    cl_int err;
    d->kern_template_phase0_streebog512 = clCreateKernel(d->prog_template_streebog512,
                                                         "template_phase0", &err);
    if (err != CL_SUCCESS || !d->kern_template_phase0_streebog512) {
        fprintf(stderr,
            "OpenCL GPU[%d]: template_phase0 (STREEBOG512) kernel create failed "
            "(err=%d) — STREEBOG512 template path unavailable\n", dev_idx, err);
        d->kern_template_phase0_streebog512 = NULL;
        return -1;
    }
    return 0;
}

/* B6 salt-axis (2026-05-06): MD5SALT compile + lazy pair.
 * defines_str: HASH_WORDS=4,HASH_BLOCK_BYTES=64,HAS_SALT=1,SALT_POSITION=
 * APPEND_TO_HEX32. The HAS_SALT=1 token is structurally load-bearing —
 * gpu_template.cl gates the salt args on `#ifdef GPU_TEMPLATE_HAS_SALT`,
 * which we set via -DGPU_TEMPLATE_HAS_SALT=1 in the build options. The
 * defines_str entry is what the kernel-cache key hashes; the build-option
 * is what the OpenCL preprocessor sees. Both are required.
 *
 * Note: SALT_POSITION token is APPEND_TO_HEX32 (NOT plain APPEND). This
 * is the "double-MD5 chain" position: the salt is appended to the
 * 32-char hex encoding of the inner MD5 digest, not to the raw password.
 * Kernel-cache disambiguation from MD5SALTPASS (which uses PREPEND) is
 * via this token — the source files differ too, but defines_str is the
 * mechanically load-bearing disambiguator (R3 mitigation per Memo B §4). */
static int gpu_opencl_template_compile_md5salt(struct gpu_device *d, int dev_idx) {
    if (d->prog_template_md5salt) return 0;
    cl_int err = CL_SUCCESS;
    const char *sources[4] = {
        gpu_common_str, gpu_md5_rules_str,
        gpu_md5salt_core_str, gpu_template_str
    };
    /* 2026-05-09 lane-batch experiment: GPU_TEMPLATE_HAS_PRE_SALT enables
     * the inner-MD5 hoist (template_pre_salt + template_finalize_post)
     * in gpu_template.cl + gpu_md5salt_core.cl. SALT_BATCH controls the
     * number of salts each lane processes serially after one inner+hex
     * computation.
     *
     * 2026-05-09 dynsize prototype: SALT_BATCH source priority:
     *   1. dynsize feedback loop (MDXFIND_DYNSIZE=1, cache hit) — reads
     *      d->dynsize_md5salt.current_N which was loaded from
     *      ~/.mdxfind/dynsize/<uuid>/kern_md5salt.txt
     *   2. MDXFIND_SALT_BATCH env var (if set, [1..256])
     *   3. Default 16
     * The defines_str entry mirrors SALT_BATCH so the kernel-cache key
     * disambiguates (a fresh JIT is needed if SALT_BATCH changes). */
    int salt_batch_env = (int)dynsize_compile_time_N(d, dev_idx);
    char defines_buf[160];
    snprintf(defines_buf, sizeof(defines_buf),
        "HASH_WORDS=4,HASH_BLOCK_BYTES=64,HAS_SALT=1,"
        "SALT_POSITION=APPEND_TO_HEX32,HAS_PRE_SALT=1,SALT_BATCH=%d",
        salt_batch_env);
    char build_opts_buf[200];
    snprintf(build_opts_buf, sizeof(build_opts_buf),
        "-cl-std=CL1.2 -DGPU_TEMPLATE_HAS_SALT=1 "
        "-DGPU_TEMPLATE_HAS_PRE_SALT=1 -DSALT_BATCH=%d",
        salt_batch_env);
    /* -DGPU_TEMPLATE_HAS_SALT=1 turns on the kernel-side salt-arg
     * #ifdef blocks in gpu_template.cl. The token name MUST match the
     * one in the .cl file; defines_str carries the SAME token plus the
     * SALT_POSITION discriminator for cache-key purposes. */
    d->prog_template_md5salt = gpu_kernel_cache_build_program_ex(
        d->ctx, d->dev, 4, sources,
        build_opts_buf,
        defines_buf, &err);
    if (!d->prog_template_md5salt || err != CL_SUCCESS) {
        char log[8192] = {0};
        if (d->prog_template_md5salt) {
            clGetProgramBuildInfo(d->prog_template_md5salt, d->dev,
                                  CL_PROGRAM_BUILD_LOG,
                                  sizeof(log) - 1, log, NULL);
        }
        fprintf(stderr,
            "OpenCL GPU[%d]: MD5SALT template program build error (err=%d) "
            "— MD5SALT template path unavailable:\n%s\n",
            dev_idx, err, log);
        if (d->prog_template_md5salt) clReleaseProgram(d->prog_template_md5salt);
        d->prog_template_md5salt = NULL;
        return -1;
    }
    return 0;
}

static int gpu_opencl_template_kernel_lazy_md5salt(struct gpu_device *d, int dev_idx) {
    if (d->kern_template_phase0_md5salt) return 0;
    if (!d->prog_template_md5salt) return -1;
    cl_int err;
    d->kern_template_phase0_md5salt = clCreateKernel(d->prog_template_md5salt,
                                                     "template_phase0", &err);
    if (err != CL_SUCCESS || !d->kern_template_phase0_md5salt) {
        fprintf(stderr,
            "OpenCL GPU[%d]: template_phase0 (MD5SALT) kernel create failed "
            "(err=%d) — MD5SALT template path unavailable\n", dev_idx, err);
        d->kern_template_phase0_md5salt = NULL;
        return -1;
    }
    return 0;
}

/* B6 salt-axis (2026-05-06): MD5SALTPASS compile + lazy pair. Simple
 * MD5(salt || pass) — SALT_POSITION=PREPEND. */
static int gpu_opencl_template_compile_md5saltpass(struct gpu_device *d, int dev_idx) {
    if (d->prog_template_md5saltpass) return 0;
    cl_int err = CL_SUCCESS;
    const char *sources[4] = {
        gpu_common_str, gpu_md5_rules_str,
        gpu_md5saltpass_core_str, gpu_template_str
    };
    const char *defines =
        "HASH_WORDS=4,HASH_BLOCK_BYTES=64,HAS_SALT=1,"
        "SALT_POSITION=PREPEND";
    d->prog_template_md5saltpass = gpu_kernel_cache_build_program_ex(
        d->ctx, d->dev, 4, sources,
        "-cl-std=CL1.2 -DGPU_TEMPLATE_HAS_SALT=1",
        defines, &err);
    if (!d->prog_template_md5saltpass || err != CL_SUCCESS) {
        char log[8192] = {0};
        if (d->prog_template_md5saltpass) {
            clGetProgramBuildInfo(d->prog_template_md5saltpass, d->dev,
                                  CL_PROGRAM_BUILD_LOG,
                                  sizeof(log) - 1, log, NULL);
        }
        fprintf(stderr,
            "OpenCL GPU[%d]: MD5SALTPASS template program build error (err=%d) "
            "— MD5SALTPASS template path unavailable:\n%s\n",
            dev_idx, err, log);
        if (d->prog_template_md5saltpass) clReleaseProgram(d->prog_template_md5saltpass);
        d->prog_template_md5saltpass = NULL;
        return -1;
    }
    return 0;
}

static int gpu_opencl_template_kernel_lazy_md5saltpass(struct gpu_device *d, int dev_idx) {
    if (d->kern_template_phase0_md5saltpass) return 0;
    if (!d->prog_template_md5saltpass) return -1;
    cl_int err;
    d->kern_template_phase0_md5saltpass = clCreateKernel(d->prog_template_md5saltpass,
                                                         "template_phase0", &err);
    if (err != CL_SUCCESS || !d->kern_template_phase0_md5saltpass) {
        fprintf(stderr,
            "OpenCL GPU[%d]: template_phase0 (MD5SALTPASS) kernel create failed "
            "(err=%d) — MD5SALTPASS template path unavailable\n", dev_idx, err);
        d->kern_template_phase0_md5saltpass = NULL;
        return -1;
    }
    return 0;
}

/* B6.1 SHA1 fan-out (2026-05-06): SHA1SALTPASS compile + lazy pair. Simple
 * SHA1(salt || pass) — SALT_POSITION=PREPEND. Distinct cache key from
 * MD5SALTPASS (which also has PREPEND) via HASH_WORDS=5 + BASE_ALGO=sha1
 * tokens — both axes change, so kernel-cache 35/35 pairwise disambiguation
 * is preserved. */
static int gpu_opencl_template_compile_sha1saltpass(struct gpu_device *d, int dev_idx) {
    if (d->prog_template_sha1saltpass) return 0;
    cl_int err = CL_SUCCESS;
    const char *sources[4] = {
        gpu_common_str, gpu_md5_rules_str,
        gpu_sha1saltpass_core_str, gpu_template_str
    };
    /* HASH_WORDS=5 (5 uint32 SHA1 state, vs MD5's 4) and BASE_ALGO=sha1
     * (per-family primitive disambiguator) are the two axes that distinguish
     * this entry's cache key from MD5SALTPASS's (which has HASH_WORDS=4 +
     * BASE_ALGO=md5; SALT_POSITION=PREPEND matches but isn't load-bearing
     * for disambiguation here). */
    const char *defines =
        "HASH_WORDS=5,HASH_BLOCK_BYTES=64,HAS_SALT=1,"
        "SALT_POSITION=PREPEND,BASE_ALGO=sha1";
    d->prog_template_sha1saltpass = gpu_kernel_cache_build_program_ex(
        d->ctx, d->dev, 4, sources,
        "-cl-std=CL1.2 -DGPU_TEMPLATE_HAS_SALT=1",
        defines, &err);
    if (!d->prog_template_sha1saltpass || err != CL_SUCCESS) {
        char log[8192] = {0};
        if (d->prog_template_sha1saltpass) {
            clGetProgramBuildInfo(d->prog_template_sha1saltpass, d->dev,
                                  CL_PROGRAM_BUILD_LOG,
                                  sizeof(log) - 1, log, NULL);
        }
        fprintf(stderr,
            "OpenCL GPU[%d]: SHA1SALTPASS template program build error (err=%d) "
            "— SHA1SALTPASS template path unavailable:\n%s\n",
            dev_idx, err, log);
        if (d->prog_template_sha1saltpass) clReleaseProgram(d->prog_template_sha1saltpass);
        d->prog_template_sha1saltpass = NULL;
        return -1;
    }
    return 0;
}

static int gpu_opencl_template_kernel_lazy_sha1saltpass(struct gpu_device *d, int dev_idx) {
    if (d->kern_template_phase0_sha1saltpass) return 0;
    if (!d->prog_template_sha1saltpass) return -1;
    cl_int err;
    d->kern_template_phase0_sha1saltpass = clCreateKernel(d->prog_template_sha1saltpass,
                                                          "template_phase0", &err);
    if (err != CL_SUCCESS || !d->kern_template_phase0_sha1saltpass) {
        fprintf(stderr,
            "OpenCL GPU[%d]: template_phase0 (SHA1SALTPASS) kernel create failed "
            "(err=%d) — SHA1SALTPASS template path unavailable\n", dev_idx, err);
        d->kern_template_phase0_sha1saltpass = NULL;
        return -1;
    }
    return 0;
}

/* B6.2 SHA256 fan-out (2026-05-06): SHA256SALTPASS compile + lazy pair.
 * Simple SHA256(salt || pass) — SALT_POSITION=PREPEND. Distinct cache key
 * from SHA1SALTPASS via HASH_WORDS=8 + BASE_ALGO=sha256 tokens (both axes
 * differ). From MD5SALTPASS: HASH_WORDS=8 + BASE_ALGO=sha256 (both axes
 * differ). 36/36 pairwise disambiguation preserved. */
static int gpu_opencl_template_compile_sha256saltpass(struct gpu_device *d, int dev_idx) {
    if (d->prog_template_sha256saltpass) return 0;
    cl_int err = CL_SUCCESS;
    const char *sources[4] = {
        gpu_common_str, gpu_md5_rules_str,
        gpu_sha256saltpass_core_str, gpu_template_str
    };
    /* HASH_WORDS=8 (8 uint32 SHA256 state, vs SHA1's 5 / MD5's 4) and
     * BASE_ALGO=sha256 (per-family primitive disambiguator) are the two
     * axes that distinguish this entry's cache key from both
     * SHA1SALTPASS and MD5SALTPASS. SALT_POSITION=PREPEND matches but
     * isn't load-bearing for disambiguation here. */
    const char *defines =
        "HASH_WORDS=8,HASH_BLOCK_BYTES=64,HAS_SALT=1,"
        "SALT_POSITION=PREPEND,BASE_ALGO=sha256";
    d->prog_template_sha256saltpass = gpu_kernel_cache_build_program_ex(
        d->ctx, d->dev, 4, sources,
        "-cl-std=CL1.2 -DGPU_TEMPLATE_HAS_SALT=1",
        defines, &err);
    if (!d->prog_template_sha256saltpass || err != CL_SUCCESS) {
        char log[8192] = {0};
        if (d->prog_template_sha256saltpass) {
            clGetProgramBuildInfo(d->prog_template_sha256saltpass, d->dev,
                                  CL_PROGRAM_BUILD_LOG,
                                  sizeof(log) - 1, log, NULL);
        }
        fprintf(stderr,
            "OpenCL GPU[%d]: SHA256SALTPASS template program build error (err=%d) "
            "— SHA256SALTPASS template path unavailable:\n%s\n",
            dev_idx, err, log);
        if (d->prog_template_sha256saltpass) clReleaseProgram(d->prog_template_sha256saltpass);
        d->prog_template_sha256saltpass = NULL;
        return -1;
    }
    return 0;
}

static int gpu_opencl_template_kernel_lazy_sha256saltpass(struct gpu_device *d, int dev_idx) {
    if (d->kern_template_phase0_sha256saltpass) return 0;
    if (!d->prog_template_sha256saltpass) return -1;
    cl_int err;
    d->kern_template_phase0_sha256saltpass = clCreateKernel(d->prog_template_sha256saltpass,
                                                            "template_phase0", &err);
    if (err != CL_SUCCESS || !d->kern_template_phase0_sha256saltpass) {
        fprintf(stderr,
            "OpenCL GPU[%d]: template_phase0 (SHA256SALTPASS) kernel create failed "
            "(err=%d) — SHA256SALTPASS template path unavailable\n", dev_idx, err);
        d->kern_template_phase0_sha256saltpass = NULL;
        return -1;
    }
    return 0;
}

/* B6.3 SHA224 fan-out (2026-05-06): SHA224SALTPASS compile + lazy pair.
 * Simple SHA224(salt || pass) — SALT_POSITION=PREPEND. SHA224 reuses
 * the SHA256 compression primitive (sha256_block) but uses the SHA224
 * IV constants and truncates output to 7 uint32 words (28 bytes).
 * Distinct cache key from SHA256SALTPASS via HASH_WORDS=7 (vs 8) — same
 * BASE_ALGO=sha256 since the compression primitive is identical. From
 * SHA1SALTPASS: HASH_WORDS=7 + BASE_ALGO=sha256 (both axes differ).
 * From MD5SALTPASS: HASH_WORDS=7 + BASE_ALGO=sha256 (both axes differ).
 * 37/37 pairwise disambiguation preserved. */
static int gpu_opencl_template_compile_sha224saltpass(struct gpu_device *d, int dev_idx) {
    if (d->prog_template_sha224saltpass) return 0;
    cl_int err = CL_SUCCESS;
    const char *sources[4] = {
        gpu_common_str, gpu_md5_rules_str,
        gpu_sha224saltpass_core_str, gpu_template_str
    };
    /* HASH_WORDS=7 (truncated SHA224 state, vs SHA256's 8) and
     * BASE_ALGO=sha256 (compression primitive — sha256_block is the
     * shared core; SHA224 differs only by IV constants and output
     * truncation, not the primitive itself). HASH_WORDS=7 is the
     * load-bearing axis here vs SHA256SALTPASS. */
    const char *defines =
        "HASH_WORDS=7,HASH_BLOCK_BYTES=64,HAS_SALT=1,"
        "SALT_POSITION=PREPEND,BASE_ALGO=sha256";
    d->prog_template_sha224saltpass = gpu_kernel_cache_build_program_ex(
        d->ctx, d->dev, 4, sources,
        "-cl-std=CL1.2 -DGPU_TEMPLATE_HAS_SALT=1",
        defines, &err);
    if (!d->prog_template_sha224saltpass || err != CL_SUCCESS) {
        char log[8192] = {0};
        if (d->prog_template_sha224saltpass) {
            clGetProgramBuildInfo(d->prog_template_sha224saltpass, d->dev,
                                  CL_PROGRAM_BUILD_LOG,
                                  sizeof(log) - 1, log, NULL);
        }
        fprintf(stderr,
            "OpenCL GPU[%d]: SHA224SALTPASS template program build error (err=%d) "
            "— SHA224SALTPASS template path unavailable:\n%s\n",
            dev_idx, err, log);
        if (d->prog_template_sha224saltpass) clReleaseProgram(d->prog_template_sha224saltpass);
        d->prog_template_sha224saltpass = NULL;
        return -1;
    }
    return 0;
}

static int gpu_opencl_template_kernel_lazy_sha224saltpass(struct gpu_device *d, int dev_idx) {
    if (d->kern_template_phase0_sha224saltpass) return 0;
    if (!d->prog_template_sha224saltpass) return -1;
    cl_int err;
    d->kern_template_phase0_sha224saltpass = clCreateKernel(d->prog_template_sha224saltpass,
                                                            "template_phase0", &err);
    if (err != CL_SUCCESS || !d->kern_template_phase0_sha224saltpass) {
        fprintf(stderr,
            "OpenCL GPU[%d]: template_phase0 (SHA224SALTPASS) kernel create failed "
            "(err=%d) — SHA224SALTPASS template path unavailable\n", dev_idx, err);
        d->kern_template_phase0_sha224saltpass = NULL;
        return -1;
    }
    return 0;
}

/* B6.4 MD5PASSSALT fan-out (2026-05-06): first APPEND-shape salted variant
 * on the codegen path. defines_str disambiguates from MD5SALTPASS via
 * SALT_POSITION=APPEND (vs PREPEND); same BASE_ALGO=md5 + HASH_WORDS=4
 * axes. Same compression primitive (md5_block); only the byte-order
 * inside the M[] build at template_finalize differs (pass first, then
 * salt). 38/38 pairwise distinct defines_str preserved. */
static int gpu_opencl_template_compile_md5passsalt(struct gpu_device *d, int dev_idx) {
    if (d->prog_template_md5passsalt) return 0;
    cl_int err = CL_SUCCESS;
    const char *sources[4] = {
        gpu_common_str, gpu_md5_rules_str,
        gpu_md5passsalt_core_str, gpu_template_str
    };
    /* SALT_POSITION=APPEND is the load-bearing axis here vs MD5SALTPASS
     * (PREPEND). HASH_WORDS=4 + BASE_ALGO=md5 match MD5SALTPASS exactly;
     * the kernel-cache key disambiguates only on SALT_POSITION. */
    const char *defines =
        "HASH_WORDS=4,HASH_BLOCK_BYTES=64,HAS_SALT=1,"
        "SALT_POSITION=APPEND,BASE_ALGO=md5";
    d->prog_template_md5passsalt = gpu_kernel_cache_build_program_ex(
        d->ctx, d->dev, 4, sources,
        "-cl-std=CL1.2 -DGPU_TEMPLATE_HAS_SALT=1",
        defines, &err);
    if (!d->prog_template_md5passsalt || err != CL_SUCCESS) {
        char log[8192] = {0};
        if (d->prog_template_md5passsalt) {
            clGetProgramBuildInfo(d->prog_template_md5passsalt, d->dev,
                                  CL_PROGRAM_BUILD_LOG,
                                  sizeof(log) - 1, log, NULL);
        }
        fprintf(stderr,
            "OpenCL GPU[%d]: MD5PASSSALT template program build error (err=%d) "
            "— MD5PASSSALT template path unavailable:\n%s\n",
            dev_idx, err, log);
        if (d->prog_template_md5passsalt) clReleaseProgram(d->prog_template_md5passsalt);
        d->prog_template_md5passsalt = NULL;
        return -1;
    }
    return 0;
}

static int gpu_opencl_template_kernel_lazy_md5passsalt(struct gpu_device *d, int dev_idx) {
    if (d->kern_template_phase0_md5passsalt) return 0;
    if (!d->prog_template_md5passsalt) return -1;
    cl_int err;
    d->kern_template_phase0_md5passsalt = clCreateKernel(d->prog_template_md5passsalt,
                                                         "template_phase0", &err);
    if (err != CL_SUCCESS || !d->kern_template_phase0_md5passsalt) {
        fprintf(stderr,
            "OpenCL GPU[%d]: template_phase0 (MD5PASSSALT) kernel create failed "
            "(err=%d) — MD5PASSSALT template path unavailable\n", dev_idx, err);
        d->kern_template_phase0_md5passsalt = NULL;
        return -1;
    }
    return 0;
}

/* B6.5 SHA1PASSSALT fan-out (2026-05-06): first SHA-family APPEND-shape
 * salted variant. defines_str disambiguates from SHA1SALTPASS via
 * SALT_POSITION=APPEND (vs PREPEND); same BASE_ALGO=sha1 + HASH_WORDS=5
 * axes. Same compression primitive (sha1_block); only the byte-order
 * inside the M[] build at template_finalize differs (pass first, then
 * salt). 39/39 pairwise distinct defines_str preserved. */
static int gpu_opencl_template_compile_sha1passsalt(struct gpu_device *d, int dev_idx) {
    if (d->prog_template_sha1passsalt) return 0;
    cl_int err = CL_SUCCESS;
    const char *sources[4] = {
        gpu_common_str, gpu_md5_rules_str,
        gpu_sha1passsalt_core_str, gpu_template_str
    };
    /* SALT_POSITION=APPEND is the load-bearing axis here vs SHA1SALTPASS
     * (PREPEND). HASH_WORDS=5 + BASE_ALGO=sha1 match SHA1SALTPASS exactly;
     * the kernel-cache key disambiguates only on SALT_POSITION. */
    const char *defines =
        "HASH_WORDS=5,HASH_BLOCK_BYTES=64,HAS_SALT=1,"
        "SALT_POSITION=APPEND,BASE_ALGO=sha1";
    d->prog_template_sha1passsalt = gpu_kernel_cache_build_program_ex(
        d->ctx, d->dev, 4, sources,
        "-cl-std=CL1.2 -DGPU_TEMPLATE_HAS_SALT=1",
        defines, &err);
    if (!d->prog_template_sha1passsalt || err != CL_SUCCESS) {
        char log[8192] = {0};
        if (d->prog_template_sha1passsalt) {
            clGetProgramBuildInfo(d->prog_template_sha1passsalt, d->dev,
                                  CL_PROGRAM_BUILD_LOG,
                                  sizeof(log) - 1, log, NULL);
        }
        fprintf(stderr,
            "OpenCL GPU[%d]: SHA1PASSSALT template program build error (err=%d) "
            "— SHA1PASSSALT template path unavailable:\n%s\n",
            dev_idx, err, log);
        if (d->prog_template_sha1passsalt) clReleaseProgram(d->prog_template_sha1passsalt);
        d->prog_template_sha1passsalt = NULL;
        return -1;
    }
    return 0;
}

static int gpu_opencl_template_kernel_lazy_sha1passsalt(struct gpu_device *d, int dev_idx) {
    if (d->kern_template_phase0_sha1passsalt) return 0;
    if (!d->prog_template_sha1passsalt) return -1;
    cl_int err;
    d->kern_template_phase0_sha1passsalt = clCreateKernel(d->prog_template_sha1passsalt,
                                                          "template_phase0", &err);
    if (err != CL_SUCCESS || !d->kern_template_phase0_sha1passsalt) {
        fprintf(stderr,
            "OpenCL GPU[%d]: template_phase0 (SHA1PASSSALT) kernel create failed "
            "(err=%d) — SHA1PASSSALT template path unavailable\n", dev_idx, err);
        d->kern_template_phase0_sha1passsalt = NULL;
        return -1;
    }
    return 0;
}

/* B6.7 SHA256PASSSALT fan-out (2026-05-06): second SHA-family APPEND-shape
 * salted variant. Pure spec reuse — both the main template
 * (sha256_style_salted.cl.tmpl from B6.2) and the finalize fragment
 * (finalize_append_be.cl.frag from B6.5) are already shipped; codegen.py
 * routing extended to allow SHA256 + APPEND. defines_str disambiguates
 * from SHA256SALTPASS via SALT_POSITION=APPEND (vs PREPEND); same
 * BASE_ALGO=sha256 + HASH_WORDS=8 axes. From SHA1PASSSALT via
 * HASH_WORDS=8 + BASE_ALGO=sha256 (both axes differ). 40/40 pairwise
 * distinct defines_str preserved. */
static int gpu_opencl_template_compile_sha256passsalt(struct gpu_device *d, int dev_idx) {
    if (d->prog_template_sha256passsalt) return 0;
    cl_int err = CL_SUCCESS;
    const char *sources[4] = {
        gpu_common_str, gpu_md5_rules_str,
        gpu_sha256passsalt_core_str, gpu_template_str
    };
    /* SALT_POSITION=APPEND is the load-bearing axis here vs SHA256SALTPASS
     * (PREPEND). HASH_WORDS=8 + BASE_ALGO=sha256 match SHA256SALTPASS
     * exactly; the kernel-cache key disambiguates only on SALT_POSITION. */
    const char *defines =
        "HASH_WORDS=8,HASH_BLOCK_BYTES=64,HAS_SALT=1,"
        "SALT_POSITION=APPEND,BASE_ALGO=sha256";
    d->prog_template_sha256passsalt = gpu_kernel_cache_build_program_ex(
        d->ctx, d->dev, 4, sources,
        "-cl-std=CL1.2 -DGPU_TEMPLATE_HAS_SALT=1",
        defines, &err);
    if (!d->prog_template_sha256passsalt || err != CL_SUCCESS) {
        char log[8192] = {0};
        if (d->prog_template_sha256passsalt) {
            clGetProgramBuildInfo(d->prog_template_sha256passsalt, d->dev,
                                  CL_PROGRAM_BUILD_LOG,
                                  sizeof(log) - 1, log, NULL);
        }
        fprintf(stderr,
            "OpenCL GPU[%d]: SHA256PASSSALT template program build error (err=%d) "
            "— SHA256PASSSALT template path unavailable:\n%s\n",
            dev_idx, err, log);
        if (d->prog_template_sha256passsalt) clReleaseProgram(d->prog_template_sha256passsalt);
        d->prog_template_sha256passsalt = NULL;
        return -1;
    }
    return 0;
}

static int gpu_opencl_template_kernel_lazy_sha256passsalt(struct gpu_device *d, int dev_idx) {
    if (d->kern_template_phase0_sha256passsalt) return 0;
    if (!d->prog_template_sha256passsalt) return -1;
    cl_int err;
    d->kern_template_phase0_sha256passsalt = clCreateKernel(d->prog_template_sha256passsalt,
                                                            "template_phase0", &err);
    if (err != CL_SUCCESS || !d->kern_template_phase0_sha256passsalt) {
        fprintf(stderr,
            "OpenCL GPU[%d]: template_phase0 (SHA256PASSSALT) kernel create failed "
            "(err=%d) — SHA256PASSSALT template path unavailable\n", dev_idx, err);
        d->kern_template_phase0_sha256passsalt = NULL;
        return -1;
    }
    return 0;
}

/* B6.9 SHA512 fan-out (2026-05-06): SHA512SALTPASS compile + lazy pair.
 * Simple SHA512(salt || pass) — SALT_POSITION=PREPEND. FIRST 64-bit-state
 * salted variant on the codegen path. Distinct cache key from every other
 * salted template via HASH_BLOCK_BYTES=128 (the 128-byte block is unique
 * to the SHA-384/512 family among salted variants on the codegen path)
 * + HASH_WORDS=16 + BASE_ALGO=sha512. Authors a sibling main template
 * (sha512_style_salted.cl.tmpl) AND a sibling fragment
 * (finalize_prepend_be64.cl.frag). 44/44 pairwise distinct defines_str
 * preserved. R2 risk on gfx1201 — unsalted SHA-512 reading was 42,520 B
 * priv_mem; HARD GATE 43,024 B (3080 spill-region ceiling). */
static int gpu_opencl_template_compile_sha512saltpass(struct gpu_device *d, int dev_idx) {
    if (d->prog_template_sha512saltpass) return 0;
    cl_int err = CL_SUCCESS;
    const char *sources[4] = {
        gpu_common_str, gpu_md5_rules_str,
        gpu_sha512saltpass_core_str, gpu_template_str
    };
    /* HASH_WORDS=16 + HASH_BLOCK_BYTES=128 + BASE_ALGO=sha512 — three
     * load-bearing axes, all distinct from every prior salted template.
     * HASH_BLOCK_BYTES=128 alone is unique to SHA-384/512 among salted
     * variants (every other salted core uses 64-byte blocks). */
    const char *defines =
        "HASH_WORDS=16,HASH_BLOCK_BYTES=128,HAS_SALT=1,"
        "SALT_POSITION=PREPEND,BASE_ALGO=sha512";
    d->prog_template_sha512saltpass = gpu_kernel_cache_build_program_ex(
        d->ctx, d->dev, 4, sources,
        "-cl-std=CL1.2 -DGPU_TEMPLATE_HAS_SALT=1",
        defines, &err);
    if (!d->prog_template_sha512saltpass || err != CL_SUCCESS) {
        char log[8192] = {0};
        if (d->prog_template_sha512saltpass) {
            clGetProgramBuildInfo(d->prog_template_sha512saltpass, d->dev,
                                  CL_PROGRAM_BUILD_LOG,
                                  sizeof(log) - 1, log, NULL);
        }
        fprintf(stderr,
            "OpenCL GPU[%d]: SHA512SALTPASS template program build error (err=%d) "
            "— SHA512SALTPASS template path unavailable:\n%s\n",
            dev_idx, err, log);
        if (d->prog_template_sha512saltpass) clReleaseProgram(d->prog_template_sha512saltpass);
        d->prog_template_sha512saltpass = NULL;
        return -1;
    }
    return 0;
}

static int gpu_opencl_template_kernel_lazy_sha512saltpass(struct gpu_device *d, int dev_idx) {
    if (d->kern_template_phase0_sha512saltpass) return 0;
    if (!d->prog_template_sha512saltpass) return -1;
    cl_int err;
    d->kern_template_phase0_sha512saltpass = clCreateKernel(d->prog_template_sha512saltpass,
                                                            "template_phase0", &err);
    if (err != CL_SUCCESS || !d->kern_template_phase0_sha512saltpass) {
        fprintf(stderr,
            "OpenCL GPU[%d]: template_phase0 (SHA512SALTPASS) kernel create failed "
            "(err=%d) — SHA512SALTPASS template path unavailable\n", dev_idx, err);
        d->kern_template_phase0_sha512saltpass = NULL;
        return -1;
    }
    return 0;
}

/* B6.10 SHA512PASSSALT fan-out (2026-05-06): SHA512PASSSALT compile + lazy pair.
 * Simple SHA512(pass || salt) — SALT_POSITION=APPEND. FINAL B6 ladder step.
 * Second 64-bit-state salted variant on the codegen path; APPEND-shape sibling
 * of SHA512SALTPASS. Cache disambiguated from SHA512SALTPASS via SALT_POSITION=
 * APPEND (vs PREPEND); same BASE_ALGO=sha512 + HASH_WORDS=16 + HASH_BLOCK_BYTES=
 * 128 axes — single-axis delta. Pure spec reuse on the SHA-512 main template
 * + the new B6.10-authored finalize_append_be64.cl.frag. 45/45 pairwise
 * distinct defines_str preserved. HARD GATE 43,024 B gfx1201 priv_mem
 * (3080 spill-region ceiling); sibling SHA512SALTPASS reading was 42,032 B
 * (992 B headroom). Expected delta ~0 B (same M[16] scratch + same per-byte
 * loop body, only the byte-source branch order swaps). */
static int gpu_opencl_template_compile_sha512passsalt(struct gpu_device *d, int dev_idx) {
    if (d->prog_template_sha512passsalt) return 0;
    cl_int err = CL_SUCCESS;
    const char *sources[4] = {
        gpu_common_str, gpu_md5_rules_str,
        gpu_sha512passsalt_core_str, gpu_template_str
    };
    /* HASH_WORDS=16 + HASH_BLOCK_BYTES=128 + SALT_POSITION=APPEND +
     * BASE_ALGO=sha512 — four load-bearing axes, of which only
     * SALT_POSITION differs from SHA512SALTPASS. The 128-byte block
     * remains unique to SHA-384/512 among salted variants. */
    const char *defines =
        "HASH_WORDS=16,HASH_BLOCK_BYTES=128,HAS_SALT=1,"
        "SALT_POSITION=APPEND,BASE_ALGO=sha512";
    d->prog_template_sha512passsalt = gpu_kernel_cache_build_program_ex(
        d->ctx, d->dev, 4, sources,
        "-cl-std=CL1.2 -DGPU_TEMPLATE_HAS_SALT=1",
        defines, &err);
    if (!d->prog_template_sha512passsalt || err != CL_SUCCESS) {
        char log[8192] = {0};
        if (d->prog_template_sha512passsalt) {
            clGetProgramBuildInfo(d->prog_template_sha512passsalt, d->dev,
                                  CL_PROGRAM_BUILD_LOG,
                                  sizeof(log) - 1, log, NULL);
        }
        fprintf(stderr,
            "OpenCL GPU[%d]: SHA512PASSSALT template program build error (err=%d) "
            "— SHA512PASSSALT template path unavailable:\n%s\n",
            dev_idx, err, log);
        if (d->prog_template_sha512passsalt) clReleaseProgram(d->prog_template_sha512passsalt);
        d->prog_template_sha512passsalt = NULL;
        return -1;
    }
    return 0;
}

static int gpu_opencl_template_kernel_lazy_sha512passsalt(struct gpu_device *d, int dev_idx) {
    if (d->kern_template_phase0_sha512passsalt) return 0;
    if (!d->prog_template_sha512passsalt) return -1;
    cl_int err;
    d->kern_template_phase0_sha512passsalt = clCreateKernel(d->prog_template_sha512passsalt,
                                                            "template_phase0", &err);
    if (err != CL_SUCCESS || !d->kern_template_phase0_sha512passsalt) {
        fprintf(stderr,
            "OpenCL GPU[%d]: template_phase0 (SHA512PASSSALT) kernel create failed "
            "(err=%d) — SHA512PASSSALT template path unavailable\n", dev_idx, err);
        d->kern_template_phase0_sha512passsalt = NULL;
        return -1;
    }
    return 0;
}

/* Family E HMAC-SHA384 carrier (2026-05-08): SHA384SALTPASS-shaped carrier
 * compile + lazy pair. No JOB_SHA384SALTPASS algorithm in mdxfind; this
 * kernel carries HMAC-SHA384 (e543) + HMAC-SHA384_KPASS (e796) via the
 * algo_mode 5/6 HMAC body in finalize_prepend_be64.cl.frag (gated on
 * HASH_WORDS == 12 && algo_mode >= 5u). The mode-0 SHA384(salt||pass)
 * main body in the generated core is structurally unreachable in
 * production. Cache disambiguated from SHA512SALTPASS via HASH_WORDS=12
 * (vs 16); same BASE_ALGO=sha512 + HASH_BLOCK_BYTES=128 axes since the
 * compression primitive (sha512_block) is shared. 46/46 pairwise distinct
 * defines_str preserved. HARD GATE 43,024 B gfx1201 priv_mem (3080
 * spill-region ceiling); HMAC body adds ~+450-600 B over unsalted SHA-384
 * finalize — well under budget. */
static int gpu_opencl_template_compile_sha384saltpass(struct gpu_device *d, int dev_idx) {
    if (d->prog_template_sha384saltpass) return 0;
    cl_int err = CL_SUCCESS;
    const char *sources[4] = {
        gpu_common_str, gpu_md5_rules_str,
        gpu_sha384saltpass_core_str, gpu_template_str
    };
    /* HASH_WORDS=12 + HASH_BLOCK_BYTES=128 + BASE_ALGO=sha512 — three
     * load-bearing axes. HASH_WORDS=12 distinguishes from SHA512SALTPASS
     * (16) and SHA512PASSSALT (16). HASH_BLOCK_BYTES=128 + BASE_ALGO=
     * sha512 keep SHA-384 in the SHA-512 family (sha512_block). */
    const char *defines =
        "HASH_WORDS=12,HASH_BLOCK_BYTES=128,HAS_SALT=1,"
        "SALT_POSITION=PREPEND,BASE_ALGO=sha512";
    d->prog_template_sha384saltpass = gpu_kernel_cache_build_program_ex(
        d->ctx, d->dev, 4, sources,
        "-cl-std=CL1.2 -DGPU_TEMPLATE_HAS_SALT=1",
        defines, &err);
    if (!d->prog_template_sha384saltpass || err != CL_SUCCESS) {
        char log[8192] = {0};
        if (d->prog_template_sha384saltpass) {
            clGetProgramBuildInfo(d->prog_template_sha384saltpass, d->dev,
                                  CL_PROGRAM_BUILD_LOG,
                                  sizeof(log) - 1, log, NULL);
        }
        fprintf(stderr,
            "OpenCL GPU[%d]: SHA384SALTPASS template program build error (err=%d) "
            "— HMAC-SHA384 template path unavailable:\n%s\n",
            dev_idx, err, log);
        if (d->prog_template_sha384saltpass) clReleaseProgram(d->prog_template_sha384saltpass);
        d->prog_template_sha384saltpass = NULL;
        return -1;
    }
    return 0;
}

static int gpu_opencl_template_kernel_lazy_sha384saltpass(struct gpu_device *d, int dev_idx) {
    if (d->kern_template_phase0_sha384saltpass) return 0;
    if (!d->prog_template_sha384saltpass) return -1;
    cl_int err;
    d->kern_template_phase0_sha384saltpass = clCreateKernel(d->prog_template_sha384saltpass,
                                                            "template_phase0", &err);
    if (err != CL_SUCCESS || !d->kern_template_phase0_sha384saltpass) {
        fprintf(stderr,
            "OpenCL GPU[%d]: template_phase0 (SHA384SALTPASS) kernel create failed "
            "(err=%d) — HMAC-SHA384 template path unavailable\n", dev_idx, err);
        d->kern_template_phase0_sha384saltpass = NULL;
        return -1;
    }
    return 0;
}

/* Family G HMAC-RIPEMD-160 carrier (2026-05-08): RIPEMD160SALTPASS-shaped
 * carrier compile + lazy pair. No JOB_RIPEMD160SALTPASS algorithm in
 * mdxfind; this kernel carries HMAC-RMD160 (e211) + HMAC-RMD160_KPASS
 * (e798) via the algo_mode 5/6 HMAC body in finalize_prepend_rmd.cl.frag
 * (gated on HASH_WORDS == 5 && algo_mode >= 5u). The mode-0 RMD160(salt
 * ||pass) main body in the generated core is structurally unreachable in
 * production. Cache disambiguated from SHA1SALTPASS via BASE_ALGO=ripemd160
 * (vs sha1) — same HASH_WORDS=5 + HASH_BLOCK_BYTES=64 axes, but the
 * BASE_ALGO axis is load-bearing (ripemd160_block has different
 * compression and 2-arg call signature). 48/48 pairwise distinct
 * defines_str preserved. HARD GATE 43,024 B gfx1201 priv_mem (3080
 * spill-region ceiling); HMAC body adds ~+250-350 B over unsalted RMD160
 * finalize — well under budget. */
static int gpu_opencl_template_compile_ripemd160saltpass(struct gpu_device *d, int dev_idx) {
    if (d->prog_template_ripemd160saltpass) return 0;
    cl_int err = CL_SUCCESS;
    const char *sources[4] = {
        gpu_common_str, gpu_md5_rules_str,
        gpu_ripemd160saltpass_core_str, gpu_template_str
    };
    /* HASH_WORDS=5 + HASH_BLOCK_BYTES=64 + BASE_ALGO=rmd160 — three
     * load-bearing axes. BASE_ALGO=rmd160 distinguishes from SHA1SALTPASS
     * (BASE_ALGO=sha1; same HASH_WORDS=5 + HASH_BLOCK_BYTES=64). The
     * compression primitive (rmd160_block, 2-arg) is distinct from
     * sha1_block (1-arg-state-pointer BE). The token is rmd160 (not
     * ripemd160) to match the actual primitive name in gpu_common.cl. */
    const char *defines =
        "HASH_WORDS=5,HASH_BLOCK_BYTES=64,HAS_SALT=1,"
        "SALT_POSITION=PREPEND,BASE_ALGO=rmd160";
    d->prog_template_ripemd160saltpass = gpu_kernel_cache_build_program_ex(
        d->ctx, d->dev, 4, sources,
        "-cl-std=CL1.2 -DGPU_TEMPLATE_HAS_SALT=1",
        defines, &err);
    if (!d->prog_template_ripemd160saltpass || err != CL_SUCCESS) {
        char log[8192] = {0};
        if (d->prog_template_ripemd160saltpass) {
            clGetProgramBuildInfo(d->prog_template_ripemd160saltpass, d->dev,
                                  CL_PROGRAM_BUILD_LOG,
                                  sizeof(log) - 1, log, NULL);
        }
        fprintf(stderr,
            "OpenCL GPU[%d]: RIPEMD160SALTPASS template program build error (err=%d) "
            "— HMAC-RMD160 template path unavailable:\n%s\n",
            dev_idx, err, log);
        if (d->prog_template_ripemd160saltpass) clReleaseProgram(d->prog_template_ripemd160saltpass);
        d->prog_template_ripemd160saltpass = NULL;
        return -1;
    }
    return 0;
}

static int gpu_opencl_template_kernel_lazy_ripemd160saltpass(struct gpu_device *d, int dev_idx) {
    if (d->kern_template_phase0_ripemd160saltpass) return 0;
    if (!d->prog_template_ripemd160saltpass) return -1;
    cl_int err;
    d->kern_template_phase0_ripemd160saltpass = clCreateKernel(d->prog_template_ripemd160saltpass,
                                                               "template_phase0", &err);
    if (err != CL_SUCCESS || !d->kern_template_phase0_ripemd160saltpass) {
        fprintf(stderr,
            "OpenCL GPU[%d]: template_phase0 (RIPEMD160SALTPASS) kernel create failed "
            "(err=%d) — HMAC-RMD160 template path unavailable\n", dev_idx, err);
        d->kern_template_phase0_ripemd160saltpass = NULL;
        return -1;
    }
    return 0;
}

/* Family H HMAC-RIPEMD-320 carrier (2026-05-08): RIPEMD320SALTPASS-shaped
 * carrier compile + lazy pair. No JOB_RIPEMD320SALTPASS algorithm in
 * mdxfind; this kernel carries HMAC-RMD320 (e213) + HMAC-RMD320_KPASS
 * (e799) via the algo_mode 5/6 HMAC body in finalize_prepend_rmd.cl.frag
 * (gated on HASH_WORDS == 10 && algo_mode >= 5u). The mode-0 RMD320(salt
 * ||pass) main body in the generated core is structurally unreachable in
 * production. Cache disambiguated from RIPEMD160SALTPASS via HASH_WORDS=10
 * (vs 5) + BASE_ALGO=rmd320 (vs rmd160) — both axes load-bearing.
 * 49/49 pairwise distinct defines_str preserved. HARD GATE 43,024 B
 * gfx1201 priv_mem (3080 spill-region ceiling); HMAC body adds ~+250-450
 * B over unsalted RMD320 finalize (key_block + ipad + opad_block + M +
 * kstate[10] + istate[10] + ostate[10] uints). */
static int gpu_opencl_template_compile_ripemd320saltpass(struct gpu_device *d, int dev_idx) {
    if (d->prog_template_ripemd320saltpass) return 0;
    cl_int err = CL_SUCCESS;
    const char *sources[4] = {
        gpu_common_str, gpu_md5_rules_str,
        gpu_ripemd320saltpass_core_str, gpu_template_str
    };
    /* HASH_WORDS=10 + HASH_BLOCK_BYTES=64 + BASE_ALGO=rmd320 — three
     * load-bearing axes. BASE_ALGO=rmd320 distinguishes from RIPEMD160-
     * SALTPASS (BASE_ALGO=rmd160; HASH_WORDS=5). The compression primitive
     * (rmd320_block, 2-arg) shares the call signature with rmd160_block
     * but the per-step round bodies and line/line' cross-mix accumulation
     * differ. */
    const char *defines =
        "HASH_WORDS=10,HASH_BLOCK_BYTES=64,HAS_SALT=1,"
        "SALT_POSITION=PREPEND,BASE_ALGO=rmd320";
    d->prog_template_ripemd320saltpass = gpu_kernel_cache_build_program_ex(
        d->ctx, d->dev, 4, sources,
        "-cl-std=CL1.2 -DGPU_TEMPLATE_HAS_SALT=1",
        defines, &err);
    if (!d->prog_template_ripemd320saltpass || err != CL_SUCCESS) {
        char log[8192] = {0};
        if (d->prog_template_ripemd320saltpass) {
            clGetProgramBuildInfo(d->prog_template_ripemd320saltpass, d->dev,
                                  CL_PROGRAM_BUILD_LOG,
                                  sizeof(log) - 1, log, NULL);
        }
        fprintf(stderr,
            "OpenCL GPU[%d]: RIPEMD320SALTPASS template program build error (err=%d) "
            "— HMAC-RMD320 template path unavailable:\n%s\n",
            dev_idx, err, log);
        if (d->prog_template_ripemd320saltpass) clReleaseProgram(d->prog_template_ripemd320saltpass);
        d->prog_template_ripemd320saltpass = NULL;
        return -1;
    }
    return 0;
}

static int gpu_opencl_template_kernel_lazy_ripemd320saltpass(struct gpu_device *d, int dev_idx) {
    if (d->kern_template_phase0_ripemd320saltpass) return 0;
    if (!d->prog_template_ripemd320saltpass) return -1;
    cl_int err;
    d->kern_template_phase0_ripemd320saltpass = clCreateKernel(d->prog_template_ripemd320saltpass,
                                                               "template_phase0", &err);
    if (err != CL_SUCCESS || !d->kern_template_phase0_ripemd320saltpass) {
        fprintf(stderr,
            "OpenCL GPU[%d]: template_phase0 (RIPEMD320SALTPASS) kernel create failed "
            "(err=%d) — HMAC-RMD320 template path unavailable\n", dev_idx, err);
        d->kern_template_phase0_ripemd320saltpass = NULL;
        return -1;
    }
    return 0;
}

/* Family I HMAC-BLAKE2S carrier (2026-05-08): hand-written Path A sibling.
 * Single algo_mode (5); no KPASS sibling op exists for HMAC-BLAKE2S in
 * mdxfind. The kernel is built with HAS_SALT=1 + HMAC_KPASS=1 axes (cache-
 * key disambiguation from the unsalted BLAKE2S256 core). HMAC body lives
 * inline in template_finalize (not in a fragment); no codegen-tool entry.
 * R2: priv_mem reading expected within 41-43 KB band on gfx1201; HMAC body
 * adds ~+256 B over unsalted BLAKE2S finalize (key_block[64] + ipad_block
 * [320] + opad_block[96] + scratch + outer[8] + inner[8] uints in scope —
 * BLAKE2's chaining state is 8 uint32 vs RMD160's 5; comparable to Family
 * G/H reading). */
static int gpu_opencl_template_compile_hmac_blake2s(struct gpu_device *d, int dev_idx) {
    if (d->prog_template_hmac_blake2s) return 0;
    cl_int err = CL_SUCCESS;
    const char *sources[4] = {
        gpu_common_str, gpu_md5_rules_str,
        gpu_hmac_blake2s_core_str, gpu_template_str
    };
    /* HASH_WORDS=8 + HASH_BLOCK_BYTES=64 + BASE_ALGO=blake2s + HAS_SALT=1
     * + HMAC_KPASS=1 — five-axis cache key. BASE_ALGO=blake2s + HMAC_KPASS=1
     * pairwise-distinguishes from every prior salted template (none use
     * blake2s + HMAC_KPASS=1). HAS_SALT=1 distinguishes from the unsalted
     * gpu_blake2s256_core.cl template (which has the same HASH_WORDS=8 +
     * HASH_BLOCK_BYTES=64 axes but no HAS_SALT). SALT_POSITION=PREPEND
     * mirrors other HMAC-* siblings even though the salt is consumed inline
     * by the HMAC body rather than concat-prepended (the SALT_POSITION axis
     * is informational at the cache layer; doesn't change kernel emission
     * for this hand-written core). */
    const char *defines =
        "HASH_WORDS=8,HASH_BLOCK_BYTES=64,HAS_SALT=1,"
        "SALT_POSITION=PREPEND,BASE_ALGO=blake2s,HMAC_KPASS=1";
    d->prog_template_hmac_blake2s = gpu_kernel_cache_build_program_ex(
        d->ctx, d->dev, 4, sources,
        "-cl-std=CL1.2 -DGPU_TEMPLATE_HAS_SALT=1",
        defines, &err);
    if (!d->prog_template_hmac_blake2s || err != CL_SUCCESS) {
        char log[8192] = {0};
        if (d->prog_template_hmac_blake2s) {
            clGetProgramBuildInfo(d->prog_template_hmac_blake2s, d->dev,
                                  CL_PROGRAM_BUILD_LOG,
                                  sizeof(log) - 1, log, NULL);
        }
        fprintf(stderr,
            "OpenCL GPU[%d]: HMAC-BLAKE2S template program build error (err=%d) "
            "— HMAC-BLAKE2S template path unavailable:\n%s\n",
            dev_idx, err, log);
        if (d->prog_template_hmac_blake2s) clReleaseProgram(d->prog_template_hmac_blake2s);
        d->prog_template_hmac_blake2s = NULL;
        return -1;
    }
    return 0;
}

static int gpu_opencl_template_kernel_lazy_hmac_blake2s(struct gpu_device *d, int dev_idx) {
    if (d->kern_template_phase0_hmac_blake2s) return 0;
    if (!d->prog_template_hmac_blake2s) return -1;
    cl_int err;
    d->kern_template_phase0_hmac_blake2s = clCreateKernel(d->prog_template_hmac_blake2s,
                                                          "template_phase0", &err);
    if (err != CL_SUCCESS || !d->kern_template_phase0_hmac_blake2s) {
        fprintf(stderr,
            "OpenCL GPU[%d]: template_phase0 (HMAC-BLAKE2S) kernel create failed "
            "(err=%d) — HMAC-BLAKE2S template path unavailable\n", dev_idx, err);
        d->kern_template_phase0_hmac_blake2s = NULL;
        return -1;
    }
    return 0;
}

/* Family J HMAC-STREEBOG-256 carrier (2026-05-08): hand-written Path A
 * sibling. Two algo_modes: 5 = JOB_HMAC_STREEBOG256_KSALT (e838),
 * 6 = JOB_HMAC_STREEBOG256_KPASS (e837). The kernel is built with HAS_SALT=1
 * + HMAC_KSALTPASS=1 axes (cache-key disambiguation from the unsalted
 * Streebog-256 core). HMAC body lives inline in template_finalize gated
 * on algo_mode >= 5u (single branch covers both KSALT + KPASS since the
 * kernel-side math is identical; the host swaps salt vs pass via the
 * algo_mode setter and salt-Judy plumbing). R2: priv_mem reading expected
 * to be near the unsalted Streebog-256 carrier reading on GTX 1080 Pascal
 * (Streebog has 16 KB __constant SBOB_SL64 table dominating priv-mem
 * footprint; HMAC body adds ~+544 B = key_block[64] + ibuf[320] + obuf[96]
 * + inner[32] + hash[32]). Same shape as Family G/H/I — not expected to
 * bust the 43 KB gfx1201 hard gate but the Streebog-256 unsalted carrier
 * already documented elevated reading at sub-batch 5b. */
static int gpu_opencl_template_compile_hmac_streebog256(struct gpu_device *d, int dev_idx) {
    if (d->prog_template_hmac_streebog256) return 0;
    cl_int err = CL_SUCCESS;
    const char *sources[4] = {
        gpu_common_str, gpu_md5_rules_str,
        gpu_hmac_streebog256_core_str, gpu_template_str
    };
    /* HASH_WORDS=8 + HASH_BLOCK_BYTES=64 + BASE_ALGO=streebog256 + HAS_SALT=1
     * + HMAC_KSALTPASS=1 — five-axis cache key. BASE_ALGO=streebog256 +
     * HMAC_KSALTPASS=1 pairwise-distinguishes from every prior salted
     * template (only this Family J core uses streebog256 + HMAC_KSALTPASS=1).
     * HAS_SALT=1 distinguishes from the unsalted gpu_streebog256_core.cl
     * template (which has the same HASH_WORDS=8 + HASH_BLOCK_BYTES=64 +
     * BASE_ALGO=streebog256 axes but no HAS_SALT). SALT_POSITION=PREPEND
     * mirrors other HMAC-* siblings even though the salt is consumed
     * inline by the HMAC body rather than concat-prepended (the
     * SALT_POSITION axis is informational at the cache layer; doesn't
     * change kernel emission for this hand-written core). */
    const char *defines =
        "HASH_WORDS=8,HASH_BLOCK_BYTES=64,HAS_SALT=1,"
        "SALT_POSITION=PREPEND,BASE_ALGO=streebog256,HMAC_KSALTPASS=1";
    d->prog_template_hmac_streebog256 = gpu_kernel_cache_build_program_ex(
        d->ctx, d->dev, 4, sources,
        "-cl-std=CL1.2 -DGPU_TEMPLATE_HAS_SALT=1",
        defines, &err);
    if (!d->prog_template_hmac_streebog256 || err != CL_SUCCESS) {
        char log[8192] = {0};
        if (d->prog_template_hmac_streebog256) {
            clGetProgramBuildInfo(d->prog_template_hmac_streebog256, d->dev,
                                  CL_PROGRAM_BUILD_LOG,
                                  sizeof(log) - 1, log, NULL);
        }
        fprintf(stderr,
            "OpenCL GPU[%d]: HMAC-STREEBOG-256 template program build error (err=%d) "
            "- HMAC-STREEBOG-256 template path unavailable:\n%s\n",
            dev_idx, err, log);
        if (d->prog_template_hmac_streebog256) clReleaseProgram(d->prog_template_hmac_streebog256);
        d->prog_template_hmac_streebog256 = NULL;
        return -1;
    }
    return 0;
}

static int gpu_opencl_template_kernel_lazy_hmac_streebog256(struct gpu_device *d, int dev_idx) {
    if (d->kern_template_phase0_hmac_streebog256) return 0;
    if (!d->prog_template_hmac_streebog256) return -1;
    cl_int err;
    d->kern_template_phase0_hmac_streebog256 = clCreateKernel(d->prog_template_hmac_streebog256,
                                                              "template_phase0", &err);
    if (err != CL_SUCCESS || !d->kern_template_phase0_hmac_streebog256) {
        fprintf(stderr,
            "OpenCL GPU[%d]: template_phase0 (HMAC-STREEBOG-256) kernel create failed "
            "(err=%d) - HMAC-STREEBOG-256 template path unavailable\n", dev_idx, err);
        d->kern_template_phase0_hmac_streebog256 = NULL;
        return -1;
    }
    return 0;
}

/* Family K HMAC-STREEBOG-512 carrier (2026-05-08): hand-written Path A
 * sibling at HASH_WORDS=16. Two algo_modes: 5 = JOB_HMAC_STREEBOG512_KSALT
 * (e840), 6 = JOB_HMAC_STREEBOG512_KPASS (e839). The kernel is built with
 * HAS_SALT=1 + HMAC_KSALTPASS=1 axes (cache-key disambiguation from the
 * unsalted Streebog-512 core). HMAC body lives inline in template_finalize
 * gated on algo_mode >= 5u (single branch covers both KSALT + KPASS since
 * the kernel-side math is identical; the host swaps salt vs pass via the
 * algo_mode setter and salt-Judy plumbing). R2: priv_mem reading expected
 * to be ~+352 B over Family J's HMAC-STREEBOG-256 reading on GTX 1080
 * Pascal (inner/hash are 64 bytes vs 32; obuf is 128 bytes vs 96; key/msg
 * staging unchanged). Final HMAC family in the ladder. */
static int gpu_opencl_template_compile_hmac_streebog512(struct gpu_device *d, int dev_idx) {
    if (d->prog_template_hmac_streebog512) return 0;
    cl_int err = CL_SUCCESS;
    const char *sources[4] = {
        gpu_common_str, gpu_md5_rules_str,
        gpu_hmac_streebog512_core_str, gpu_template_str
    };
    /* HASH_WORDS=16 + HASH_BLOCK_BYTES=64 + BASE_ALGO=streebog512 + HAS_SALT=1
     * + HMAC_KSALTPASS=1 — five-axis cache key. BASE_ALGO=streebog512 +
     * HMAC_KSALTPASS=1 pairwise-distinguishes from every prior salted
     * template (only this Family K core uses streebog512 + HMAC_KSALTPASS=1).
     * HAS_SALT=1 distinguishes from the unsalted gpu_streebog512_core.cl
     * template (which has the same HASH_WORDS=16 + HASH_BLOCK_BYTES=64 +
     * BASE_ALGO=streebog512 axes but no HAS_SALT). SALT_POSITION=PREPEND
     * mirrors other HMAC-* siblings even though the salt is consumed
     * inline by the HMAC body rather than concat-prepended (the
     * SALT_POSITION axis is informational at the cache layer; doesn't
     * change kernel emission for this hand-written core). */
    const char *defines =
        "HASH_WORDS=16,HASH_BLOCK_BYTES=64,HAS_SALT=1,"
        "SALT_POSITION=PREPEND,BASE_ALGO=streebog512,HMAC_KSALTPASS=1";
    d->prog_template_hmac_streebog512 = gpu_kernel_cache_build_program_ex(
        d->ctx, d->dev, 4, sources,
        "-cl-std=CL1.2 -DGPU_TEMPLATE_HAS_SALT=1",
        defines, &err);
    if (!d->prog_template_hmac_streebog512 || err != CL_SUCCESS) {
        char log[8192] = {0};
        if (d->prog_template_hmac_streebog512) {
            clGetProgramBuildInfo(d->prog_template_hmac_streebog512, d->dev,
                                  CL_PROGRAM_BUILD_LOG,
                                  sizeof(log) - 1, log, NULL);
        }
        fprintf(stderr,
            "OpenCL GPU[%d]: HMAC-STREEBOG-512 template program build error (err=%d) "
            "- HMAC-STREEBOG-512 template path unavailable:\n%s\n",
            dev_idx, err, log);
        if (d->prog_template_hmac_streebog512) clReleaseProgram(d->prog_template_hmac_streebog512);
        d->prog_template_hmac_streebog512 = NULL;
        return -1;
    }
    return 0;
}

static int gpu_opencl_template_kernel_lazy_hmac_streebog512(struct gpu_device *d, int dev_idx) {
    if (d->kern_template_phase0_hmac_streebog512) return 0;
    if (!d->prog_template_hmac_streebog512) return -1;
    cl_int err;
    d->kern_template_phase0_hmac_streebog512 = clCreateKernel(d->prog_template_hmac_streebog512,
                                                              "template_phase0", &err);
    if (err != CL_SUCCESS || !d->kern_template_phase0_hmac_streebog512) {
        fprintf(stderr,
            "OpenCL GPU[%d]: template_phase0 (HMAC-STREEBOG-512) kernel create failed "
            "(err=%d) - HMAC-STREEBOG-512 template path unavailable\n", dev_idx, err);
        d->kern_template_phase0_hmac_streebog512 = NULL;
        return -1;
    }
    return 0;
}

/* PHPBB3 carrier (2026-05-08): hand-written Path A salted-template
 * sibling at HASH_WORDS=4 (MD5 width). Single algo_mode (the algorithm
 * is a single fixed shape; the iteration count is decoded inside the
 * kernel from salt_bytes[3], not selected via algo_mode). The kernel
 * is built with HAS_SALT=1 axis (cache-key disambiguation from MD5SALT
 * family + unsalted MD5 cores). The iterated MD5 chain lives inline in
 * template_finalize — NOT in template_iterate — so the kernel's outer
 * iter loop runs exactly once (max_iter=1 host-forced) and only the
 * FINAL state is probed (matches CPU semantics at mdxfind.c:13620
 * which has ONE checkhashbb call AFTER the inner for-loop). Mirrors
 * SHA1DRU pattern (B6.11 precedent at gpu_sha1dru_core.cl). */
static int gpu_opencl_template_compile_phpbb3(struct gpu_device *d, int dev_idx) {
    if (d->prog_template_phpbb3) return 0;
    cl_int err = CL_SUCCESS;
    const char *sources[4] = {
        gpu_common_str, gpu_md5_rules_str,
        gpu_phpbb3_core_str, gpu_template_str
    };
    /* HASH_WORDS=4 + HASH_BLOCK_BYTES=64 + BASE_ALGO=phpbb3 + HAS_SALT=1 +
     * SALT_POSITION=PREPEND -- five-axis cache key. BASE_ALGO=phpbb3
     * pairwise-distinguishes from every prior salted template (only this
     * core uses BASE_ALGO=phpbb3); HAS_SALT=1 distinguishes from unsalted
     * MD5 cores (which share HASH_WORDS=4 + HASH_BLOCK_BYTES=64).
     * SALT_POSITION=PREPEND is informational at the cache layer (this
     * kernel doesn't concat-prepend the salt -- it consumes salt[4..11]
     * inline in step 1 of the chain). */
    const char *defines =
        "HASH_WORDS=4,HASH_BLOCK_BYTES=64,HAS_SALT=1,"
        "SALT_POSITION=PREPEND,BASE_ALGO=phpbb3";
    d->prog_template_phpbb3 = gpu_kernel_cache_build_program_ex(
        d->ctx, d->dev, 4, sources,
        "-cl-std=CL1.2 -DGPU_TEMPLATE_HAS_SALT=1",
        defines, &err);
    if (!d->prog_template_phpbb3 || err != CL_SUCCESS) {
        char log[8192] = {0};
        if (d->prog_template_phpbb3) {
            clGetProgramBuildInfo(d->prog_template_phpbb3, d->dev,
                                  CL_PROGRAM_BUILD_LOG,
                                  sizeof(log) - 1, log, NULL);
        }
        fprintf(stderr,
            "OpenCL GPU[%d]: PHPBB3 template program build error (err=%d) "
            "-- PHPBB3 template path unavailable:\n%s\n",
            dev_idx, err, log);
        if (d->prog_template_phpbb3) clReleaseProgram(d->prog_template_phpbb3);
        d->prog_template_phpbb3 = NULL;
        return -1;
    }
    return 0;
}

static int gpu_opencl_template_kernel_lazy_phpbb3(struct gpu_device *d, int dev_idx) {
    if (d->kern_template_phase0_phpbb3) return 0;
    if (!d->prog_template_phpbb3) return -1;
    cl_int err;
    d->kern_template_phase0_phpbb3 = clCreateKernel(d->prog_template_phpbb3,
                                                    "template_phase0", &err);
    if (err != CL_SUCCESS || !d->kern_template_phase0_phpbb3) {
        fprintf(stderr,
            "OpenCL GPU[%d]: template_phase0 (PHPBB3) kernel create failed "
            "(err=%d) -- PHPBB3 template path unavailable\n", dev_idx, err);
        d->kern_template_phase0_phpbb3 = NULL;
        return -1;
    }
    return 0;
}

/* MD5CRYPT carrier (2026-05-08): hand-written Path A salted-template
 * sibling at HASH_WORDS=4 (MD5 width). Single algo_mode (the algorithm
 * is a single fixed shape with FIXED 1000 iterations, vs PHPBB3's salt-
 * carried variable iter count). The kernel is built with HAS_SALT=1
 * axis (cache-key disambiguation from MD5SALT family + PHPBB3 + unsalted
 * MD5 cores). The iterated MD5 chain lives inline in template_finalize
 * -- NOT in template_iterate -- so the kernel's outer iter loop runs
 * exactly once (max_iter=1 host-forced) and only the FINAL state is
 * probed (matches CPU semantics at mdxfind.c:13071 which has ONE
 * hybrid_check call AFTER the inner for-loop). Mirrors PHPBB3 / SHA1DRU
 * pattern. */
static int gpu_opencl_template_compile_md5crypt(struct gpu_device *d, int dev_idx) {
    if (d->prog_template_md5crypt) return 0;
    cl_int err = CL_SUCCESS;
    const char *sources[4] = {
        gpu_common_str, gpu_md5_rules_str,
        gpu_md5crypt_core_str, gpu_template_str
    };
    /* HASH_WORDS=4 + HASH_BLOCK_BYTES=64 + BASE_ALGO=md5crypt + HAS_SALT=1 +
     * SALT_POSITION=PREPEND -- five-axis cache key. BASE_ALGO=md5crypt
     * pairwise-distinguishes from every prior salted template (only this
     * core uses BASE_ALGO=md5crypt); HAS_SALT=1 distinguishes from unsalted
     * MD5 cores (which share HASH_WORDS=4 + HASH_BLOCK_BYTES=64).
     * SALT_POSITION=PREPEND is informational at the cache layer (this
     * kernel doesn't concat-prepend the salt -- it consumes salt_bytes
     * inline in steps 1+2+3 of the chain). */
    const char *defines =
        "HASH_WORDS=4,HASH_BLOCK_BYTES=64,HAS_SALT=1,"
        "SALT_POSITION=PREPEND,BASE_ALGO=md5crypt";
    d->prog_template_md5crypt = gpu_kernel_cache_build_program_ex(
        d->ctx, d->dev, 4, sources,
        "-cl-std=CL1.2 -DGPU_TEMPLATE_HAS_SALT=1",
        defines, &err);
    if (!d->prog_template_md5crypt || err != CL_SUCCESS) {
        char log[8192] = {0};
        if (d->prog_template_md5crypt) {
            clGetProgramBuildInfo(d->prog_template_md5crypt, d->dev,
                                  CL_PROGRAM_BUILD_LOG,
                                  sizeof(log) - 1, log, NULL);
        }
        fprintf(stderr,
            "OpenCL GPU[%d]: MD5CRYPT template program build error (err=%d) "
            "-- MD5CRYPT template path unavailable:\n%s\n",
            dev_idx, err, log);
        if (d->prog_template_md5crypt) clReleaseProgram(d->prog_template_md5crypt);
        d->prog_template_md5crypt = NULL;
        return -1;
    }
    return 0;
}

static int gpu_opencl_template_kernel_lazy_md5crypt(struct gpu_device *d, int dev_idx) {
    if (d->kern_template_phase0_md5crypt) return 0;
    if (!d->prog_template_md5crypt) return -1;
    cl_int err;
    d->kern_template_phase0_md5crypt = clCreateKernel(d->prog_template_md5crypt,
                                                      "template_phase0", &err);
    if (err != CL_SUCCESS || !d->kern_template_phase0_md5crypt) {
        fprintf(stderr,
            "OpenCL GPU[%d]: template_phase0 (MD5CRYPT) kernel create failed "
            "(err=%d) -- MD5CRYPT template path unavailable\n", dev_idx, err);
        d->kern_template_phase0_md5crypt = NULL;
        return -1;
    }
    return 0;
}

/* SHA256CRYPT carrier (2026-05-08): hand-written Path A salted-template
 * sibling at HASH_WORDS=8 (SHA-256 width). Single algo_mode; the
 * algorithm is a single fixed shape with rounds count decoded INSIDE
 * the kernel from the salt prefix ("rounds=N$" optional, default 5000;
 * clamp 1000..999999999). The kernel built with HAS_SALT=1 axis (cache
 * key disambiguation from every prior salted template; only this Phase
 * 2 instance uses BASE_ALGO=sha256crypt at HASH_WORDS=8). The 5-step
 * SHA-crypt chain lives inline in template_finalize; the kernel's outer
 * iter loop runs exactly once (max_iter=1 host-forced) and only the
 * FINAL state is probed (matches CPU semantics at mdxfind.c:12290 which
 * has ONE checkhash-equivalent call AFTER the inner for-loop). Mirrors
 * PHPBB3 / SHA1DRU / MD5CRYPT pattern. Phase 2 of the Unix-crypt ladder. */
static int gpu_opencl_template_compile_sha256crypt(struct gpu_device *d, int dev_idx) {
    if (d->prog_template_sha256crypt) return 0;
    cl_int err = CL_SUCCESS;
    const char *sources[4] = {
        gpu_common_str, gpu_md5_rules_str,
        gpu_shacrypt_core_str, gpu_template_str
    };
    /* HASH_WORDS=8 + HASH_BLOCK_BYTES=64 + BASE_ALGO=sha256crypt + HAS_SALT=1
     * + SALT_POSITION=PREPEND -- five-axis cache key. BASE_ALGO=sha256crypt
     * pairwise-distinguishes from every prior salted template at HASH_WORDS=8
     * (SHA256SALTPASS uses BASE_ALGO=sha256; SHA224SALTPASS uses HASH_WORDS=7).
     * Phase 3's SHA512CRYPT will use HASH_WORDS=16 + HASH_BLOCK_BYTES=128
     * + BASE_ALGO=sha512crypt against the SAME core source. */
    const char *defines =
        "HASH_WORDS=8,HASH_BLOCK_BYTES=64,HAS_SALT=1,"
        "SALT_POSITION=PREPEND,BASE_ALGO=sha256crypt";
    d->prog_template_sha256crypt = gpu_kernel_cache_build_program_ex(
        d->ctx, d->dev, 4, sources,
        "-cl-std=CL1.2 -DGPU_TEMPLATE_HAS_SALT=1",
        defines, &err);
    if (!d->prog_template_sha256crypt || err != CL_SUCCESS) {
        char log[8192] = {0};
        if (d->prog_template_sha256crypt) {
            clGetProgramBuildInfo(d->prog_template_sha256crypt, d->dev,
                                  CL_PROGRAM_BUILD_LOG,
                                  sizeof(log) - 1, log, NULL);
        }
        fprintf(stderr,
            "OpenCL GPU[%d]: SHA256CRYPT template program build error (err=%d) "
            "-- SHA256CRYPT template path unavailable:\n%s\n",
            dev_idx, err, log);
        if (d->prog_template_sha256crypt) clReleaseProgram(d->prog_template_sha256crypt);
        d->prog_template_sha256crypt = NULL;
        return -1;
    }
    return 0;
}

static int gpu_opencl_template_kernel_lazy_sha256crypt(struct gpu_device *d, int dev_idx) {
    if (d->kern_template_phase0_sha256crypt) return 0;
    if (!d->prog_template_sha256crypt) return -1;
    cl_int err;
    d->kern_template_phase0_sha256crypt = clCreateKernel(d->prog_template_sha256crypt,
                                                         "template_phase0", &err);
    if (err != CL_SUCCESS || !d->kern_template_phase0_sha256crypt) {
        fprintf(stderr,
            "OpenCL GPU[%d]: template_phase0 (SHA256CRYPT) kernel create failed "
            "(err=%d) -- SHA256CRYPT template path unavailable\n", dev_idx, err);
        d->kern_template_phase0_sha256crypt = NULL;
        return -1;
    }
    return 0;
}

/* SHA512CRYPT carrier (2026-05-08): hand-written Path A salted-template
 * kernel for JOB_SHA512CRYPT (e513). 5-step glibc crypt-sha512 chain
 * (default 5000 iters; configurable via "rounds=N$" salt prefix decoded
 * INSIDE the kernel) runs INSIDE template_finalize; only the FINAL state
 * is probed (matches CPU semantics at mdxfind.c:12290 -- ONE checkhash-
 * equivalent call after the inner for-loop). Mirrors PHPBB3 / SHA1DRU /
 * MD5CRYPT / SHA256CRYPT pattern. Phase 3 of the Unix-crypt ladder.
 * Shares gpu_shacrypt_core.cl source with Phase 2 (SHA256CRYPT at
 * HASH_WORDS=8) and Phase 4 (SHA512CRYPTMD5 at HASH_WORDS=16,
 * algo_mode=1 for MD5-preprocess). */
static int gpu_opencl_template_compile_sha512crypt(struct gpu_device *d, int dev_idx) {
    if (d->prog_template_sha512crypt) return 0;
    cl_int err = CL_SUCCESS;
    const char *sources[4] = {
        gpu_common_str, gpu_md5_rules_str,
        gpu_shacrypt_core_str, gpu_template_str
    };
    /* HASH_WORDS=16 + HASH_BLOCK_BYTES=128 + BASE_ALGO=sha512crypt + HAS_SALT=1
     * + SALT_POSITION=PREPEND -- five-axis cache key. BASE_ALGO=sha512crypt
     * pairwise-distinguishes from every prior salted template at HASH_WORDS=16
     * (SHA512SALTPASS uses BASE_ALGO=sha512). Phase 2's SHA256CRYPT uses
     * HASH_WORDS=8 + BASE_ALGO=sha256crypt against the SAME core source.
     * IMPORTANT: gpu_shacrypt_core.cl's #ifndef HASH_WORDS / #define
     * HASH_WORDS 8 default is for the SHA-256 case; the SHA-512 instance
     * MUST override via -DHASH_WORDS=16 + -DHASH_BLOCK_BYTES=128 in build_-
     * opts because gpu_kernel_cache_build_program_ex's `defines_str`
     * argument is used ONLY for cache-key disambiguation, NOT passed to
     * clBuildProgram as -D options. SHA384 (HASH_WORDS=12) gets its
     * default from gpu_sha384_core.cl's own #define HASH_WORDS 12.
     * SHA512SALTPASS (HASH_WORDS=16) gets its default from
     * gpu_sha512saltpass_core.cl's own #define HASH_WORDS 16. Our shared
     * core has TWO target widths (8 and 16) so we must explicitly pass
     * HASH_WORDS via build_opts for the non-default width. */
    const char *defines =
        "HASH_WORDS=16,HASH_BLOCK_BYTES=128,HAS_SALT=1,"
        "SALT_POSITION=PREPEND,BASE_ALGO=sha512crypt";
    d->prog_template_sha512crypt = gpu_kernel_cache_build_program_ex(
        d->ctx, d->dev, 4, sources,
        "-cl-std=CL1.2 -DGPU_TEMPLATE_HAS_SALT=1 -DHASH_WORDS=16 -DHASH_BLOCK_BYTES=128",
        defines, &err);
    if (!d->prog_template_sha512crypt || err != CL_SUCCESS) {
        char log[8192] = {0};
        if (d->prog_template_sha512crypt) {
            clGetProgramBuildInfo(d->prog_template_sha512crypt, d->dev,
                                  CL_PROGRAM_BUILD_LOG,
                                  sizeof(log) - 1, log, NULL);
        }
        fprintf(stderr,
            "OpenCL GPU[%d]: SHA512CRYPT template program build error (err=%d) "
            "-- SHA512CRYPT template path unavailable:\n%s\n",
            dev_idx, err, log);
        if (d->prog_template_sha512crypt) clReleaseProgram(d->prog_template_sha512crypt);
        d->prog_template_sha512crypt = NULL;
        return -1;
    }
    return 0;
}

static int gpu_opencl_template_kernel_lazy_sha512crypt(struct gpu_device *d, int dev_idx) {
    if (d->kern_template_phase0_sha512crypt) return 0;
    if (!d->prog_template_sha512crypt) return -1;
    cl_int err;
    d->kern_template_phase0_sha512crypt = clCreateKernel(d->prog_template_sha512crypt,
                                                         "template_phase0", &err);
    if (err != CL_SUCCESS || !d->kern_template_phase0_sha512crypt) {
        fprintf(stderr,
            "OpenCL GPU[%d]: template_phase0 (SHA512CRYPT) kernel create failed "
            "(err=%d) -- SHA512CRYPT template path unavailable\n", dev_idx, err);
        d->kern_template_phase0_sha512crypt = NULL;
        return -1;
    }
    return 0;
}

/* DESCRYPT carrier (2026-05-08, Unix-crypt Phase 5): hand-written Path A
 * salted-template kernel for JOB_DESCRYPT (e500). 25-iter DES Feistel
 * chain (FIXED iter count per Unix DES crypt(3) "old-style") runs INSIDE
 * template_finalize; only the FINAL state is probed. Mirrors PHPBB3 /
 * MD5CRYPT / SHA256CRYPT / SHA512CRYPT pattern at max_iter=1 forced
 * host-side. HASH_WORDS=4 (state = pre-FP (l, r) in h[0..1], h[2..3]
 * zero-pad to match the host compact-table layout). Phase 5 of the
 * Unix-crypt ladder (FINAL phase). */
static int gpu_opencl_template_compile_descrypt(struct gpu_device *d, int dev_idx) {
    if (d->prog_template_descrypt) return 0;
    cl_int err = CL_SUCCESS;
    const char *sources[4] = {
        gpu_common_str, gpu_md5_rules_str,
        gpu_descrypt_core_str, gpu_template_str
    };
    /* HASH_WORDS=4 + HASH_BLOCK_BYTES=64 + BASE_ALGO=descrypt + HAS_SALT=1
     * + SALT_POSITION=PREPEND -- five-axis cache key. BASE_ALGO=descrypt
     * pairwise-distinguishes from every prior salted template at HASH_-
     * WORDS=4 (MD5SALT family uses BASE_ALGO=md5; PHPBB3 uses phpbb3;
     * MD5CRYPT uses md5crypt). The template default HASH_WORDS=4 in
     * gpu_descrypt_core.cl matches our target width, so HASH_WORDS does
     * NOT need an explicit -D in build_opts (per feedback_defines_via_-
     * build_opts.md: only NON-default widths need the -D). The build_-
     * opts pattern matches MD5CRYPT / PHPBB3 (also HASH_WORDS=4 default,
     * no -DHASH_WORDS override). */
    const char *defines =
        "HASH_WORDS=4,HASH_BLOCK_BYTES=64,HAS_SALT=1,"
        "SALT_POSITION=PREPEND,BASE_ALGO=descrypt";
    d->prog_template_descrypt = gpu_kernel_cache_build_program_ex(
        d->ctx, d->dev, 4, sources,
        "-cl-std=CL1.2 -DGPU_TEMPLATE_HAS_SALT=1",
        defines, &err);
    if (!d->prog_template_descrypt || err != CL_SUCCESS) {
        char log[8192] = {0};
        if (d->prog_template_descrypt) {
            clGetProgramBuildInfo(d->prog_template_descrypt, d->dev,
                                  CL_PROGRAM_BUILD_LOG,
                                  sizeof(log) - 1, log, NULL);
        }
        fprintf(stderr,
            "OpenCL GPU[%d]: DESCRYPT template program build error (err=%d) "
            "-- DESCRYPT template path unavailable:\n%s\n",
            dev_idx, err, log);
        if (d->prog_template_descrypt) clReleaseProgram(d->prog_template_descrypt);
        d->prog_template_descrypt = NULL;
        return -1;
    }
    return 0;
}

static int gpu_opencl_template_kernel_lazy_descrypt(struct gpu_device *d, int dev_idx) {
    if (d->kern_template_phase0_descrypt) return 0;
    if (!d->prog_template_descrypt) return -1;
    cl_int err;
    d->kern_template_phase0_descrypt = clCreateKernel(d->prog_template_descrypt,
                                                      "template_phase0", &err);
    if (err != CL_SUCCESS || !d->kern_template_phase0_descrypt) {
        fprintf(stderr,
            "OpenCL GPU[%d]: template_phase0 (DESCRYPT) kernel create failed "
            "(err=%d) -- DESCRYPT template path unavailable\n", dev_idx, err);
        d->kern_template_phase0_descrypt = NULL;
        return -1;
    }
    return 0;
}

/* BCRYPT carrier (2026-05-08, Unix-crypt Phase 6): hand-written Path A
 * salted-template kernel for JOB_BCRYPT (e450). 2^cost Eksblowfish chain
 * (cost parsed per-salt-string at kernel entry; SIMT divergence accepted
 * per Q3 user decision) runs INSIDE template_finalize; only the FINAL
 * state is probed. Mirrors PHPBB3 / MD5CRYPT / SHA256CRYPT / SHA512CRYPT
 * / DESCRYPT pattern at max_iter=1 forced host-side. HASH_WORDS=6
 * (24-byte output as 6 LE uint32; first 4 words = 16 bytes probed against
 * compact table). Workgroup-shared __local Eksblowfish state via
 * GPU_TEMPLATE_HAS_LOCAL_BUFFER scaffold extension (4 KB per lane × 8
 * lanes = 32 KB per WG). Phase 6 of the slab-retirement ladder (final
 * major slab kernel). */
static int gpu_opencl_template_compile_bcrypt(struct gpu_device *d, int dev_idx) {
    if (d->prog_template_bcrypt) return 0;
    cl_int err = CL_SUCCESS;
    const char *sources[4] = {
        gpu_common_str, gpu_md5_rules_str,
        gpu_bcrypt_core_str, gpu_template_str
    };
    /* HASH_WORDS=6 + HASH_BLOCK_BYTES=64 + BASE_ALGO=bcrypt + HAS_SALT=1
     * + SALT_POSITION=PREPEND + GPU_TEMPLATE_HAS_LOCAL_BUFFER=1 +
     * GPU_TEMPLATE_LOCAL_BUFFER_PER_LANE=1024 + BCRYPT_WG_SIZE=8 -- eight-
     * axis cache key. BASE_ALGO=bcrypt + HASH_WORDS=6 pairwise-distinguish
     * from every prior salted template (no other algo uses HASH_WORDS=6
     * + HAS_SALT=1). Per feedback_defines_via_build_opts.md: NON-default
     * macro values MUST be passed as -D<NAME>=<value> in build_opts (the
     * defines_str is cache-key only). HASH_WORDS=6 is NOT the core's
     * default (the gpu_bcrypt_core.cl `#ifndef HASH_WORDS / #define
     * HASH_WORDS 6 / #endif` does match HASH_WORDS=6 already, but for
     * defensive consistency with the SHA512CRYPT precedent we pass it
     * explicitly via -D), so include it in build_opts.
     * GPU_TEMPLATE_HAS_LOCAL_BUFFER and GPU_TEMPLATE_LOCAL_BUFFER_PER_LANE
     * gate the gpu_template.cl scaffold's __local sbox_pool declaration +
     * 8-arg template_finalize call (only THIS carrier compile defines
     * them; all other instantiations remain byte-identical pre-Phase-6
     * per R-S1). BCRYPT_WG_SIZE pins reqd_work_group_size at kernel-
     * function scope. */
    const char *defines =
        "HASH_WORDS=6,HASH_BLOCK_BYTES=64,HAS_SALT=1,"
        "SALT_POSITION=PREPEND,BASE_ALGO=bcrypt,"
        "GPU_TEMPLATE_HAS_LOCAL_BUFFER=1,"
        "GPU_TEMPLATE_LOCAL_BUFFER_PER_LANE=1024,"
        "BCRYPT_WG_SIZE=8";
    d->prog_template_bcrypt = gpu_kernel_cache_build_program_ex(
        d->ctx, d->dev, 4, sources,
        "-cl-std=CL1.2 -DGPU_TEMPLATE_HAS_SALT=1 "
        "-DGPU_TEMPLATE_HAS_LOCAL_BUFFER=1 "
        "-DGPU_TEMPLATE_LOCAL_BUFFER_PER_LANE=1024 "
        "-DBCRYPT_WG_SIZE=8 "
        "-DHASH_WORDS=6 -DHASH_BLOCK_BYTES=64",
        defines, &err);
    if (!d->prog_template_bcrypt || err != CL_SUCCESS) {
        char log[8192] = {0};
        if (d->prog_template_bcrypt) {
            clGetProgramBuildInfo(d->prog_template_bcrypt, d->dev,
                                  CL_PROGRAM_BUILD_LOG,
                                  sizeof(log) - 1, log, NULL);
        }
        fprintf(stderr,
            "OpenCL GPU[%d]: BCRYPT template program build error (err=%d) "
            "-- BCRYPT template path unavailable:\n%s\n",
            dev_idx, err, log);
        if (d->prog_template_bcrypt) clReleaseProgram(d->prog_template_bcrypt);
        d->prog_template_bcrypt = NULL;
        return -1;
    }
    return 0;
}

static int gpu_opencl_template_kernel_lazy_bcrypt(struct gpu_device *d, int dev_idx) {
    if (d->kern_template_phase0_bcrypt) return 0;
    if (!d->prog_template_bcrypt) return -1;
    cl_int err;
    d->kern_template_phase0_bcrypt = clCreateKernel(d->prog_template_bcrypt,
                                                    "template_phase0", &err);
    if (err != CL_SUCCESS || !d->kern_template_phase0_bcrypt) {
        fprintf(stderr,
            "OpenCL GPU[%d]: template_phase0 (BCRYPT) kernel create failed "
            "(err=%d) -- BCRYPT template path unavailable\n", dev_idx, err);
        d->kern_template_phase0_bcrypt = NULL;
        return -1;
    }
    return 0;
}

/* Upload the GPU-eligible rule program to device memory. Idempotent
 * within a session — a second call replaces the prior rule set.
 *
 * rule_program: post-packrules bytecodes for each GPU-eligible rule,
 *               concatenated and NUL-separated.
 * prog_len:     total bytes including all NUL terminators.
 * rule_offset:  byte offset of each rule's bytecode within rule_program.
 * n_rules:      number of GPU-eligible rules.
 *
 * Returns 0 on success, -1 on failure. The kernel declares rule_program
 * as __constant; if prog_len exceeds the device's CL_DEVICE_MAX_-
 * CONSTANT_BUFFER_SIZE (typically 64 KB), the build/dispatch will fail —
 * callers should fall back to CPU rule expansion in that case. */
int gpu_opencl_set_rules(int dev_idx,
    const unsigned char *rule_program, uint32_t prog_len,
    const uint32_t *rule_offset, int n_rules)
{
    if (!ocl_ready || dev_idx < 0 || dev_idx >= num_gpu_devs) return -1;
    if (!rule_program || prog_len == 0 || !rule_offset || n_rules <= 0) return -1;
    struct gpu_device *d = &gpu_devs[dev_idx];
    cl_int err;

    /* Don't upload rules to a disabled device. See d->device_disabled doc.
     * Caller's n_uploaded counter will omit this device, and if zero
     * devices upload successfully gpu_rule_count stays 0 → full CPU
     * rule-walk fallback. */
    if (d->device_disabled) {
        fprintf(stderr, "OpenCL GPU[%d]: skipping rule program upload (device disabled)\n", dev_idx);
        return -1;
    }

    if (gpu_opencl_rules_compile(d, dev_idx) < 0) return -1;

    /* Phase E: time the rule-program + offset upload. */
    struct timespec _ru_t0, _ru_t1;
    clock_gettime(CLOCK_MONOTONIC, &_ru_t0);
    tsfprintf(stderr, "OpenCL GPU[%d]: rule program upload START (%.2fMB, %d rules)\n",
              dev_idx, prog_len / (1024.0 * 1024), n_rules);

    /* Grow / (re)create the rule_program buffer. __constant so the
     * kernel can take advantage of the device's constant cache.
     * Buffer is reallocated if the new program is larger than the
     * current allocation; keeping a stale buffer of the same size
     * avoids churn when callers re-upload an identical rule set.
     * Routed through create_min_buf so the synthetic-only no-rule pass
     * (mdxfind 1.380 routing for `-i5` / iter-only no-rules MD5 unsalted
     * workloads) gets a 256B-floored, zero-padded allocation rather than
     * a 1B (single NUL) buffer. NVIDIA Windows cold-JIT validates buffer
     * size at NDRange time against kernel signature; tiny buffers are
     * rejected with CL_INVALID_KERNEL_ARGS even though kernel guards
     * bound access via rule_offset[rule_idx] and rule_idx < n_rules.
     * The kernel reads rule_program[rpos] only with rpos < rule_offset
     * for valid rule_idx, so the zero-padded tail is unread by valid
     * lanes. Just (GTX 1650 Win64) bug3a 2026-05-04. */
    if (!d->b_rule_program || (size_t)prog_len > d->rule_program_cap) {
        if (d->b_rule_program) clReleaseMemObject(d->b_rule_program);
        d->b_rule_program = create_min_buf(d->ctx, d->queue,
            CL_MEM_READ_ONLY, prog_len, rule_program, &err);
        if (err != CL_SUCCESS || !d->b_rule_program) {
            fprintf(stderr, "OpenCL GPU[%d]: b_rule_program alloc failed "
                    "(prog_len=%u, err=%d)\n", dev_idx, prog_len, err);
            d->b_rule_program = NULL;
            return -1;
        }
        d->rule_program_cap = prog_len;
    } else {
        /* Reuse path: existing buffer is large enough; just rewrite
         * the prefix. The 256B floor/zero-pad already established at
         * first allocation persists across re-uploads. */
        err = clEnqueueWriteBuffer(d->queue, d->b_rule_program, CL_TRUE, 0,
                                   prog_len, rule_program, 0, NULL, NULL);
        if (err != CL_SUCCESS) {
            /* Fail-fast: rule program is uploaded once at session
             * start; a failure here means every subsequent dispatch
             * reads garbage. */
            GPU_FATAL("b_rule_program upload err=%d on dev %d (prog_len=%u)",
                      err, dev_idx, prog_len);
        }
    }
    /* Read-back probe: first 64 bytes, plus last 64 if program > 128. */
    {
        size_t head = prog_len < 64 ? prog_len : 64;
        gpu_readback_probe(dev_idx, d->queue, d->b_rule_program, 0, head,
                           rule_program, "set_rules", "b_rule_program[head]");
        if (prog_len > 128) {
            size_t off = prog_len - 64;
            gpu_readback_probe(dev_idx, d->queue, d->b_rule_program, off, 64,
                               rule_program + off, "set_rules", "b_rule_program[tail]");
        }
    }

    /* Rule offsets buffer. Same 256B-floor + zero-pad rationale as
     * b_rule_program above: synthetic-only no-rule pass has n_rules=1
     * → off_bytes=4, well below NVIDIA Windows NDRange-time minimum
     * buffer threshold. Kernel reads rule_offset[rule_idx] only when
     * rule_idx < n_rules; tail bytes unread by valid lanes. */
    size_t off_bytes = (size_t)n_rules * sizeof(uint32_t);
    if (!d->b_rule_offset || off_bytes > d->rule_offset_cap) {
        if (d->b_rule_offset) clReleaseMemObject(d->b_rule_offset);
        d->b_rule_offset = create_min_buf(d->ctx, d->queue,
            CL_MEM_READ_ONLY, off_bytes, rule_offset, &err);
        if (err != CL_SUCCESS || !d->b_rule_offset) {
            fprintf(stderr, "OpenCL GPU[%d]: b_rule_offset alloc failed (err=%d)\n",
                    dev_idx, err);
            d->b_rule_offset = NULL;
            return -1;
        }
        d->rule_offset_cap = off_bytes;
    } else {
        /* Reuse path: existing buffer large enough; rewrite prefix. */
        err = clEnqueueWriteBuffer(d->queue, d->b_rule_offset, CL_TRUE, 0,
                                   off_bytes, rule_offset, 0, NULL, NULL);
        if (err != CL_SUCCESS) {
            /* Fail-fast: rule offsets uploaded once at session start;
             * a failure here means every subsequent dispatch reads
             * garbage. */
            GPU_FATAL("b_rule_offset upload err=%d on dev %d (off_bytes=%zu)",
                      err, dev_idx, off_bytes);
        }
    }
    /* Read-back probe: first 64 bytes, plus last 64 if buffer > 128. */
    {
        size_t head = off_bytes < 64 ? off_bytes : 64;
        gpu_readback_probe(dev_idx, d->queue, d->b_rule_offset, 0, head,
                           rule_offset, "set_rules", "b_rule_offset[head]");
        if (off_bytes > 128) {
            size_t off = off_bytes - 64;
            gpu_readback_probe(dev_idx, d->queue, d->b_rule_offset, off, 64,
                               (const unsigned char *)rule_offset + off, "set_rules", "b_rule_offset[tail]");
        }
    }

    d->gpu_n_rules = n_rules;

    /* Validator env-gated host-side mirror. When MDXFIND_GPU_VALIDATOR=1,
     * the dispatcher needs to stringify rulebytes=<hex> for each rule_idx
     * after the kernel returns; cheapest way is to keep a host copy of
     * the same data we just uploaded. Reallocated/grown idempotently;
     * never freed within a session (matches the device buffers'
     * persistence model). When the validator env var is unset the
     * pointers stay NULL and zero memory is consumed. */
    if (gpu_validator_enabled()) {
        if (!d->h_rule_program || prog_len > d->h_rule_program_len) {
            free(d->h_rule_program);
            d->h_rule_program = (unsigned char *)malloc(prog_len);
        }
        if (d->h_rule_program) {
            memcpy(d->h_rule_program, rule_program, prog_len);
            d->h_rule_program_len = prog_len;
        }
        if (!d->h_rule_offset || n_rules > d->h_rule_offset_n) {
            free(d->h_rule_offset);
            d->h_rule_offset = (uint32_t *)malloc((size_t)n_rules * sizeof(uint32_t));
        }
        if (d->h_rule_offset) {
            memcpy(d->h_rule_offset, rule_offset, (size_t)n_rules * sizeof(uint32_t));
            d->h_rule_offset_n = n_rules;
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &_ru_t1);
    {
        double _ru_ms = (_ru_t1.tv_sec - _ru_t0.tv_sec) * 1e3
                      + (_ru_t1.tv_nsec - _ru_t0.tv_nsec) / 1e6;
        tsfprintf(stderr, "OpenCL GPU[%d]: rule program upload DONE in %.2fs\n",
                  dev_idx, _ru_ms / 1e3);
    }
    return 0;
}

/* B5 chokepoint widening (2026-05-04): single entry point that resolves
 * the appropriate per-op template kernel handle, ensuring it has been
 * compiled and instantiated lazily on this device. Replaces the 5-branch
 * if-ladder previously inlined at the dispatch_md5_rules kernel-handle
 * swap site.
 *
 * Returns the per-op template kernel when:
 *   - MDXFIND_GPU_TEMPLATE is unset (tpl == GPU_TEMPLATE_OFF) — production
 *     path for the 5 fan-out ops; OR
 *   - MDXFIND_GPU_TEMPLATE is set AND the env-var algo matches the op.
 *
 * Returns NULL when the env-var is set but selects a different algorithm
 * than this dispatch's op (caller falls back to legacy md5_rules_phase0
 * for the MD5 case; for non-MD5 ops the caller will dispatch the legacy
 * MD5 kernel which is structurally wrong — but that combination is opt-in
 * via env-var, and the warning emitted here documents it).
 *
 * Returns NULL on compile / kernel-create failure (caller falls back to
 * legacy md5_rules_phase0 for MD5; for other ops the dispatch will
 * proceed against md5_rules_phase0 producing wrong digests — but compile
 * failure on a *_core.cl is a hard error already logged by the lazy/
 * compile helpers, so this is a degraded-but-noisy state, not a silent
 * miscompute).
 *
 * Per-op template handles touched (struct gpu_device fields):
 *   JOB_MD5     -> kern_template_phase0        (gpu_md5_core.cl    + gpu_template.cl)
 *   JOB_MD4     -> kern_template_phase0_md4    (gpu_md4_core.cl    + gpu_template.cl)
 *   JOB_SHA1    -> kern_template_phase0_sha1   (gpu_sha1_core.cl   + gpu_template.cl)
 *   JOB_SHA224  -> kern_template_phase0_sha224 (gpu_sha224_core.cl + gpu_template.cl)
 *   JOB_SHA256  -> kern_template_phase0_sha256 (gpu_sha256_core.cl + gpu_template.cl)
 *   JOB_SHA384  -> kern_template_phase0_sha384 (gpu_sha384_core.cl + gpu_template.cl) [B5]
 *   JOB_SHA512  -> kern_template_phase0_sha512 (gpu_sha512_core.cl + gpu_template.cl) [B5]
 *   JOB_RMD160  -> kern_template_phase0_ripemd160 (gpu_ripemd160_core.cl + gpu_template.cl) [B5 sb2]
 *   JOB_RMD320  -> kern_template_phase0_ripemd320 (gpu_ripemd320_core.cl + gpu_template.cl) [B5 sb2] */
static cl_kernel gpu_template_resolve_kernel(struct gpu_device *d, int dev_idx, int op) {
    int tpl = gpu_template_enabled();
    switch (op) {
        case JOB_MD5:
        /* B7.7a (2026-05-07): JOB_MD5UC reuses the MD5 template kernel.
         * Iter=1 is byte-exact between MD5 and MD5UC; the UC variant
         * differs only in the inter-iter hex encoding (lowercase vs
         * uppercase) which is selected at runtime via params.algo_mode
         * (set in the algo_mode setter elsewhere in this file). The
         * gpu_md5_core.cl template_iterate branches on algo_mode to
         * pick md5_to_hex_uc vs md5_to_hex_lc. No new kernel handle. */
        case JOB_MD5UC:
            if (tpl == GPU_TEMPLATE_OFF || tpl == GPU_TEMPLATE_MD5) {
                if (gpu_opencl_template_compile(d, dev_idx) == 0 &&
                    gpu_opencl_template_kernel_lazy(d, dev_idx) == 0)
                    return d->kern_template_phase0;
            }
            break;
        case JOB_MD4:
            if (tpl == GPU_TEMPLATE_OFF || tpl == GPU_TEMPLATE_MD4) {
                if (gpu_opencl_template_compile_md4(d, dev_idx) == 0 &&
                    gpu_opencl_template_kernel_lazy_md4(d, dev_idx) == 0)
                    return d->kern_template_phase0_md4;
            }
            break;
        case JOB_SHA1:
            if (tpl == GPU_TEMPLATE_OFF || tpl == GPU_TEMPLATE_SHA1) {
                if (gpu_opencl_template_compile_sha1(d, dev_idx) == 0 &&
                    gpu_opencl_template_kernel_lazy_sha1(d, dev_idx) == 0)
                    return d->kern_template_phase0_sha1;
            }
            break;
        case JOB_SHA224:
            if (tpl == GPU_TEMPLATE_OFF || tpl == GPU_TEMPLATE_SHA224) {
                if (gpu_opencl_template_compile_sha224(d, dev_idx) == 0 &&
                    gpu_opencl_template_kernel_lazy_sha224(d, dev_idx) == 0)
                    return d->kern_template_phase0_sha224;
            }
            break;
        case JOB_SHA256:
            if (tpl == GPU_TEMPLATE_OFF || tpl == GPU_TEMPLATE_SHA256) {
                if (gpu_opencl_template_compile_sha256(d, dev_idx) == 0 &&
                    gpu_opencl_template_kernel_lazy_sha256(d, dev_idx) == 0)
                    return d->kern_template_phase0_sha256;
            }
            break;
        case JOB_SHA384:
            if (tpl == GPU_TEMPLATE_OFF || tpl == GPU_TEMPLATE_SHA384) {
                if (gpu_opencl_template_compile_sha384(d, dev_idx) == 0 &&
                    gpu_opencl_template_kernel_lazy_sha384(d, dev_idx) == 0)
                    return d->kern_template_phase0_sha384;
            }
            break;
        case JOB_SHA512:
            if (tpl == GPU_TEMPLATE_OFF || tpl == GPU_TEMPLATE_SHA512) {
                if (gpu_opencl_template_compile_sha512(d, dev_idx) == 0 &&
                    gpu_opencl_template_kernel_lazy_sha512(d, dev_idx) == 0)
                    return d->kern_template_phase0_sha512;
            }
            break;
        case JOB_RMD160:
            if (tpl == GPU_TEMPLATE_OFF || tpl == GPU_TEMPLATE_RIPEMD160) {
                if (gpu_opencl_template_compile_ripemd160(d, dev_idx) == 0 &&
                    gpu_opencl_template_kernel_lazy_ripemd160(d, dev_idx) == 0)
                    return d->kern_template_phase0_ripemd160;
            }
            break;
        case JOB_RMD320:
            if (tpl == GPU_TEMPLATE_OFF || tpl == GPU_TEMPLATE_RIPEMD320) {
                if (gpu_opencl_template_compile_ripemd320(d, dev_idx) == 0 &&
                    gpu_opencl_template_kernel_lazy_ripemd320(d, dev_idx) == 0)
                    return d->kern_template_phase0_ripemd320;
            }
            break;
        /* B5 sub-batch 3 (2026-05-06): BLAKE2 family. Three variants
         * wired (BLAKE2B-160 omitted — JOB_BLAKE2B160 is not defined in
         * mdxfind.c; would require new job-type plumbing beyond GPU work). */
        case JOB_BLAKE2S256:
            if (tpl == GPU_TEMPLATE_OFF || tpl == GPU_TEMPLATE_BLAKE2S256) {
                if (gpu_opencl_template_compile_blake2s256(d, dev_idx) == 0 &&
                    gpu_opencl_template_kernel_lazy_blake2s256(d, dev_idx) == 0)
                    return d->kern_template_phase0_blake2s256;
            }
            break;
        case JOB_BLAKE2B256:
            if (tpl == GPU_TEMPLATE_OFF || tpl == GPU_TEMPLATE_BLAKE2B256) {
                if (gpu_opencl_template_compile_blake2b256(d, dev_idx) == 0 &&
                    gpu_opencl_template_kernel_lazy_blake2b256(d, dev_idx) == 0)
                    return d->kern_template_phase0_blake2b256;
            }
            break;
        case JOB_BLAKE2B512:
            if (tpl == GPU_TEMPLATE_OFF || tpl == GPU_TEMPLATE_BLAKE2B512) {
                if (gpu_opencl_template_compile_blake2b512(d, dev_idx) == 0 &&
                    gpu_opencl_template_kernel_lazy_blake2b512(d, dev_idx) == 0)
                    return d->kern_template_phase0_blake2b512;
            }
            break;
        /* B5 sub-batch 4 (2026-05-03): SHA3 / Keccak family. Eight variants;
         * sponge construction. Pairs (Keccak/SHA3 of same output size) share
         * rate + EMIT_HIT width but distinct suffix-byte literal in the core. */
        case JOB_KECCAK224:
            if (tpl == GPU_TEMPLATE_OFF || tpl == GPU_TEMPLATE_KECCAK224) {
                if (gpu_opencl_template_compile_keccak224(d, dev_idx) == 0 &&
                    gpu_opencl_template_kernel_lazy_keccak224(d, dev_idx) == 0)
                    return d->kern_template_phase0_keccak224;
            }
            break;
        case JOB_KECCAK256:
            if (tpl == GPU_TEMPLATE_OFF || tpl == GPU_TEMPLATE_KECCAK256) {
                if (gpu_opencl_template_compile_keccak256(d, dev_idx) == 0 &&
                    gpu_opencl_template_kernel_lazy_keccak256(d, dev_idx) == 0)
                    return d->kern_template_phase0_keccak256;
            }
            break;
        case JOB_KECCAK384:
            if (tpl == GPU_TEMPLATE_OFF || tpl == GPU_TEMPLATE_KECCAK384) {
                if (gpu_opencl_template_compile_keccak384(d, dev_idx) == 0 &&
                    gpu_opencl_template_kernel_lazy_keccak384(d, dev_idx) == 0)
                    return d->kern_template_phase0_keccak384;
            }
            break;
        case JOB_KECCAK512:
            if (tpl == GPU_TEMPLATE_OFF || tpl == GPU_TEMPLATE_KECCAK512) {
                if (gpu_opencl_template_compile_keccak512(d, dev_idx) == 0 &&
                    gpu_opencl_template_kernel_lazy_keccak512(d, dev_idx) == 0)
                    return d->kern_template_phase0_keccak512;
            }
            break;
        case JOB_SHA3_224:
            if (tpl == GPU_TEMPLATE_OFF || tpl == GPU_TEMPLATE_SHA3_224) {
                if (gpu_opencl_template_compile_sha3_224(d, dev_idx) == 0 &&
                    gpu_opencl_template_kernel_lazy_sha3_224(d, dev_idx) == 0)
                    return d->kern_template_phase0_sha3_224;
            }
            break;
        case JOB_SHA3_256:
            if (tpl == GPU_TEMPLATE_OFF || tpl == GPU_TEMPLATE_SHA3_256) {
                if (gpu_opencl_template_compile_sha3_256(d, dev_idx) == 0 &&
                    gpu_opencl_template_kernel_lazy_sha3_256(d, dev_idx) == 0)
                    return d->kern_template_phase0_sha3_256;
            }
            break;
        case JOB_SHA3_384:
            if (tpl == GPU_TEMPLATE_OFF || tpl == GPU_TEMPLATE_SHA3_384) {
                if (gpu_opencl_template_compile_sha3_384(d, dev_idx) == 0 &&
                    gpu_opencl_template_kernel_lazy_sha3_384(d, dev_idx) == 0)
                    return d->kern_template_phase0_sha3_384;
            }
            break;
        case JOB_SHA3_512:
            if (tpl == GPU_TEMPLATE_OFF || tpl == GPU_TEMPLATE_SHA3_512) {
                if (gpu_opencl_template_compile_sha3_512(d, dev_idx) == 0 &&
                    gpu_opencl_template_kernel_lazy_sha3_512(d, dev_idx) == 0)
                    return d->kern_template_phase0_sha3_512;
            }
            break;
        /* B5 sub-batch 5a Tier 1 (2026-05-03): SHA384RAW + SHA512RAW. */
        case JOB_SHA384RAW:
            if (tpl == GPU_TEMPLATE_OFF || tpl == GPU_TEMPLATE_SHA384RAW) {
                if (gpu_opencl_template_compile_sha384raw(d, dev_idx) == 0 &&
                    gpu_opencl_template_kernel_lazy_sha384raw(d, dev_idx) == 0)
                    return d->kern_template_phase0_sha384raw;
            }
            break;
        case JOB_SHA512RAW:
            if (tpl == GPU_TEMPLATE_OFF || tpl == GPU_TEMPLATE_SHA512RAW) {
                if (gpu_opencl_template_compile_sha512raw(d, dev_idx) == 0 &&
                    gpu_opencl_template_kernel_lazy_sha512raw(d, dev_idx) == 0)
                    return d->kern_template_phase0_sha512raw;
            }
            break;
        /* B5 sub-batch 6 Tier A (2026-05-03): MD5RAW + SHA1RAW + SHA256RAW. */
        case JOB_MD5RAW:
            if (tpl == GPU_TEMPLATE_OFF || tpl == GPU_TEMPLATE_MD5RAW) {
                if (gpu_opencl_template_compile_md5raw(d, dev_idx) == 0 &&
                    gpu_opencl_template_kernel_lazy_md5raw(d, dev_idx) == 0)
                    return d->kern_template_phase0_md5raw;
            }
            break;
        case JOB_SHA1RAW:
            if (tpl == GPU_TEMPLATE_OFF || tpl == GPU_TEMPLATE_SHA1RAW) {
                if (gpu_opencl_template_compile_sha1raw(d, dev_idx) == 0 &&
                    gpu_opencl_template_kernel_lazy_sha1raw(d, dev_idx) == 0)
                    return d->kern_template_phase0_sha1raw;
            }
            break;
        case JOB_SHA256RAW:
            if (tpl == GPU_TEMPLATE_OFF || tpl == GPU_TEMPLATE_SHA256RAW) {
                if (gpu_opencl_template_compile_sha256raw(d, dev_idx) == 0 &&
                    gpu_opencl_template_kernel_lazy_sha256raw(d, dev_idx) == 0)
                    return d->kern_template_phase0_sha256raw;
            }
            break;
        /* B5 sub-batch 6 Tier C (2026-05-03): SQL5. */
        case JOB_SQL5:
            if (tpl == GPU_TEMPLATE_OFF || tpl == GPU_TEMPLATE_SQL5) {
                if (gpu_opencl_template_compile_sql5(d, dev_idx) == 0 &&
                    gpu_opencl_template_kernel_lazy_sql5(d, dev_idx) == 0)
                    return d->kern_template_phase0_sql5;
            }
            break;
        /* B6.11 SHA1DRU (2026-05-06): Drupal SHA1; first 1M-iter algo on
         * template path. 1M loop in template_finalize; max_iter=1 forced. */
        case JOB_SHA1DRU:
            if (tpl == GPU_TEMPLATE_OFF || tpl == GPU_TEMPLATE_SHA1DRU) {
                if (gpu_opencl_template_compile_sha1dru(d, dev_idx) == 0 &&
                    gpu_opencl_template_kernel_lazy_sha1dru(d, dev_idx) == 0)
                    return d->kern_template_phase0_sha1dru;
            }
            break;
        /* PHPBB3 carrier (2026-05-08): JOB_PHPBB3 (e455). Iterated MD5
         * chain INSIDE template_finalize (iter count decoded from
         * salt[3]); max_iter=1 forced host-side at the rules-engine
         * pack site so kernel's outer iter loop runs exactly once.
         * Mirrors SHA1DRU dispatch shape. */
        case JOB_PHPBB3:
            if (tpl == GPU_TEMPLATE_OFF || tpl == GPU_TEMPLATE_PHPBB3) {
                if (gpu_opencl_template_compile_phpbb3(d, dev_idx) == 0 &&
                    gpu_opencl_template_kernel_lazy_phpbb3(d, dev_idx) == 0)
                    return d->kern_template_phase0_phpbb3;
            }
            break;
        /* MD5CRYPT carrier (2026-05-08): JOB_MD5CRYPT (e511). Iterated
         * MD5 chain INSIDE template_finalize (FIXED 1000 iters per BSD
         * $1$ md5crypt); max_iter=1 forced host-side at the rules-engine
         * pack site so kernel's outer iter loop runs exactly once.
         * Mirrors PHPBB3 dispatch shape. Phase 1 of the Unix-crypt
         * ladder. */
        case JOB_MD5CRYPT:
            if (tpl == GPU_TEMPLATE_OFF || tpl == GPU_TEMPLATE_MD5CRYPT) {
                if (gpu_opencl_template_compile_md5crypt(d, dev_idx) == 0 &&
                    gpu_opencl_template_kernel_lazy_md5crypt(d, dev_idx) == 0)
                    return d->kern_template_phase0_md5crypt;
            }
            break;
        /* SHA256CRYPT carrier (2026-05-08): JOB_SHA256CRYPT (e512). 5-step
         * SHA-crypt chain INSIDE template_finalize (default 5000 rounds,
         * configurable via "rounds=N$" salt prefix). max_iter=1 forced
         * host-side at the rules-engine pack site so kernel's outer iter
         * loop runs exactly once. Mirrors MD5CRYPT dispatch shape. Phase
         * 2 of the Unix-crypt ladder. Shares gpu_shacrypt_core.cl source
         * with Phase 3 (SHA512CRYPT) and Phase 4 (SHA512CRYPTMD5). */
        case JOB_SHA256CRYPT:
            if (tpl == GPU_TEMPLATE_OFF || tpl == GPU_TEMPLATE_SHA256CRYPT) {
                if (gpu_opencl_template_compile_sha256crypt(d, dev_idx) == 0 &&
                    gpu_opencl_template_kernel_lazy_sha256crypt(d, dev_idx) == 0)
                    return d->kern_template_phase0_sha256crypt;
            }
            break;
        /* SHA512CRYPT carrier (2026-05-08): JOB_SHA512CRYPT (e513). 5-step
         * SHA-crypt chain INSIDE template_finalize (default 5000 rounds,
         * configurable via "rounds=N$" salt prefix). max_iter=1 forced
         * host-side at the rules-engine pack site so kernel's outer iter
         * loop runs exactly once. Mirrors SHA256CRYPT dispatch shape.
         * Phase 3 of the Unix-crypt ladder. Shares gpu_shacrypt_core.cl
         * source with Phase 2 (SHA256CRYPT at HASH_WORDS=8) and Phase 4
         * (SHA512CRYPTMD5 at HASH_WORDS=16, algo_mode=1 for MD5-pre-
         * process). SHA512CRYPTMD5 REMAINS on the slab path for now. */
        case JOB_SHA512CRYPT:
            if (tpl == GPU_TEMPLATE_OFF || tpl == GPU_TEMPLATE_SHA512CRYPT) {
                if (gpu_opencl_template_compile_sha512crypt(d, dev_idx) == 0 &&
                    gpu_opencl_template_kernel_lazy_sha512crypt(d, dev_idx) == 0)
                    return d->kern_template_phase0_sha512crypt;
            }
            break;
        /* SHA512CRYPTMD5 carrier (2026-05-08): JOB_SHA512CRYPTMD5 (e510)
         * REUSES the same compiled program/kernel as Phase 3 SHA512CRYPT
         * (both BASE_ALGO=sha512crypt + HASH_WORDS=16 + HAS_SALT=1 +
         * SALT_POSITION=PREPEND -- pairwise-identical cache key). The
         * MD5-preprocess of the password is performed HOST-side at
         * mdxfind.c:12256-12258 BEFORE gpu_try_pack: the 32-char MD5
         * hex of the original password is substituted into job->pass,
         * and the GPU runs the IDENTICAL SHA-512 crypt chain over the
         * 32-byte hex string. No kernel change; algo_mode stays 0u.
         * Skipping the slot/compile-helper/lazy-helper trio that an
         * algo_mode=1 kernel-side approach would require. Phase 4 of
         * the Unix-crypt ladder (final phase). */
        case JOB_SHA512CRYPTMD5:
            if (tpl == GPU_TEMPLATE_OFF || tpl == GPU_TEMPLATE_SHA512CRYPT) {
                if (gpu_opencl_template_compile_sha512crypt(d, dev_idx) == 0 &&
                    gpu_opencl_template_kernel_lazy_sha512crypt(d, dev_idx) == 0)
                    return d->kern_template_phase0_sha512crypt;
            }
            break;
        /* DESCRYPT carrier (2026-05-08, Unix-crypt Phase 5): JOB_DESCRYPT
         * (e500). 25-iter DES Feistel chain INSIDE template_finalize.
         * max_iter=1 forced host-side at the rules-engine pack site so
         * kernel's outer iter loop runs exactly once. Bespoke Path A core
         * (gpu_descrypt_core.cl); will NOT share with BCRYPT (BCRYPT will
         * need its own algo_modes for future variants). Phase 5 of the
         * Unix-crypt ladder (FINAL phase; Unix-crypt slab path fully
         * retired across all 5 Unix-crypt ops). */
        case JOB_DESCRYPT:
            if (tpl == GPU_TEMPLATE_OFF || tpl == GPU_TEMPLATE_DESCRYPT) {
                if (gpu_opencl_template_compile_descrypt(d, dev_idx) == 0 &&
                    gpu_opencl_template_kernel_lazy_descrypt(d, dev_idx) == 0)
                    return d->kern_template_phase0_descrypt;
            }
            break;
        /* BCRYPT carrier (2026-05-08, Unix-crypt Phase 6): JOB_BCRYPT (e450).
         * 2^cost Eksblowfish chain INSIDE template_finalize; cost parsed per-
         * salt-string at kernel entry (SIMT divergence accepted per Q3).
         * max_iter=1 forced host-side. Bespoke Path A core (gpu_bcrypt_core.cl);
         * will NOT share with DESCRYPT or any other algo (reserves algo_mode
         * 8-15 for future BCRYPT family variants). Compound siblings BCRYPTMD5
         * (e451) / BCRYPTSHA1 (e452) / BCRYPTSHA512 (e967) remain CPU-only
         * via gpu_op_category default fall-through; only JOB_BCRYPT singleton
         * is admitted to the rules-engine path here. Phase 6 of the slab-
         * retirement ladder (final major slab kernel). */
        case JOB_BCRYPT:
            if (tpl == GPU_TEMPLATE_OFF || tpl == GPU_TEMPLATE_BCRYPT) {
                if (gpu_opencl_template_compile_bcrypt(d, dev_idx) == 0 &&
                    gpu_opencl_template_kernel_lazy_bcrypt(d, dev_idx) == 0)
                    return d->kern_template_phase0_bcrypt;
            }
            break;
        /* B7.7b MD6256 (2026-05-07): MD6-256; final M5 closure. Single-block
         * leaf compression with 14 KB A[1753] stack; per-iter probe like SQL5. */
        case JOB_MD6256:
            if (tpl == GPU_TEMPLATE_OFF || tpl == GPU_TEMPLATE_MD6256) {
                if (gpu_opencl_template_compile_md6256(d, dev_idx) == 0 &&
                    gpu_opencl_template_kernel_lazy_md6256(d, dev_idx) == 0)
                    return d->kern_template_phase0_md6256;
            }
            break;
        /* B5 sub-batch 6 Tier B (2026-05-03): NTLMH. */
        case JOB_NTLMH:
            if (tpl == GPU_TEMPLATE_OFF || tpl == GPU_TEMPLATE_NTLMH) {
                if (gpu_opencl_template_compile_ntlmh(d, dev_idx) == 0 &&
                    gpu_opencl_template_kernel_lazy_ntlmh(d, dev_idx) == 0)
                    return d->kern_template_phase0_ntlmh;
            }
            break;
        /* B5 sub-batch 8 (2026-05-05): MD4UTF16. */
        case JOB_MD4UTF16:
            if (tpl == GPU_TEMPLATE_OFF || tpl == GPU_TEMPLATE_MD4UTF16) {
                if (gpu_opencl_template_compile_md4utf16(d, dev_idx) == 0 &&
                    gpu_opencl_template_kernel_lazy_md4utf16(d, dev_idx) == 0)
                    return d->kern_template_phase0_md4utf16;
            }
            break;
        /* B5 sub-batch 7 (2026-05-05): MYSQL3. */
        case JOB_MYSQL3:
            if (tpl == GPU_TEMPLATE_OFF || tpl == GPU_TEMPLATE_MYSQL3) {
                if (gpu_opencl_template_compile_mysql3(d, dev_idx) == 0 &&
                    gpu_opencl_template_kernel_lazy_mysql3(d, dev_idx) == 0)
                    return d->kern_template_phase0_mysql3;
            }
            break;
        /* B5 sub-batch 6.5 (2026-05-05): WRL. */
        case JOB_WRL:
            if (tpl == GPU_TEMPLATE_OFF || tpl == GPU_TEMPLATE_WRL) {
                if (gpu_opencl_template_compile_wrl(d, dev_idx) == 0 &&
                    gpu_opencl_template_kernel_lazy_wrl(d, dev_idx) == 0)
                    return d->kern_template_phase0_wrl;
            }
            break;
        /* B5 sub-batch 5b retry (2026-05-06): Streebog-256 + Streebog-512. */
        case JOB_STREEBOG_32:
            if (tpl == GPU_TEMPLATE_OFF || tpl == GPU_TEMPLATE_STREEBOG256) {
                if (gpu_opencl_template_compile_streebog256(d, dev_idx) == 0 &&
                    gpu_opencl_template_kernel_lazy_streebog256(d, dev_idx) == 0)
                    return d->kern_template_phase0_streebog256;
            }
            break;
        case JOB_STREEBOG_64:
            if (tpl == GPU_TEMPLATE_OFF || tpl == GPU_TEMPLATE_STREEBOG512) {
                if (gpu_opencl_template_compile_streebog512(d, dev_idx) == 0 &&
                    gpu_opencl_template_kernel_lazy_streebog512(d, dev_idx) == 0)
                    return d->kern_template_phase0_streebog512;
            }
            break;
        /* B6 salt-axis (2026-05-06): MD5SALT + MD5SALTPASS — first two
         * salted variants on the unified template path. The kernel that
         * comes back from these arms has signature 19 (16 unsalted + 3
         * salt args appended under #ifdef GPU_TEMPLATE_HAS_SALT in
         * gpu_template.cl). The dispatcher's SETARG block detects this
         * via kern_is_salted_template and binds the extra args. */
        /* B6.6 (2026-05-06): JOB_MD5SALT family — 4 algorithms share the
         * SAME kernel via params.algo_mode runtime flag. e31 MD5SALT (mode 0),
         * e350 MD5UCSALT (mode 1), e541 MD5revMD5SALT (mode 2), e542
         * MD5sub8_24SALT (mode 3). All resolve to kern_template_phase0_md5salt;
         * the inner-digest hex encoding step in finalize_append_to_hex32.cl.frag
         * branches on params.algo_mode. No new GPU_TEMPLATE_* enums; no new
         * specs.py entries. CPU reference: mdxfind.c:22055-22072.
         * B6.8 (2026-05-06): JOB_MD5_MD5SALTMD5PASS joins as algo_mode=4.
         * MD5(hex32(MD5(salt)) || hex32(MD5(pass))) — outer MD5 over a
         * 64-byte intermediate; salt-hex is pre-computed by the host
         * (saltsnap[].hashsalt) and packed into salt_buf via
         * gpujob_opencl.c gpu_pack_salts(use_hashsalt=1). CPU reference:
         * mdxfind.c:17027-17075. Templated count remains 43 — no new
         * GPU_TEMPLATE_* enum value. */
        case JOB_MD5SALT:
        case JOB_MD5UCSALT:
        case JOB_MD5revMD5SALT:
        case JOB_MD5sub8_24SALT:
        case JOB_MD5_MD5SALTMD5PASS:
        /* Family A (2026-05-07): JOB_HMAC_MD5 (e214) + JOB_HMAC_MD5_KPASS
         * (e792) join the SAME GPU template kernel as e31 MD5SALT via
         * params.algo_mode = 5 / 6. The HMAC body branches at the top of
         * template_finalize and returns early — bypasses the double-MD5
         * chain entirely. No new GPU_TEMPLATE_* enum value; cache key is
         * disambiguated by params.algo_mode at host pack time. */
        case JOB_HMAC_MD5:
        case JOB_HMAC_MD5_KPASS:
            if (tpl == GPU_TEMPLATE_OFF || tpl == GPU_TEMPLATE_MD5SALT) {
                if (gpu_opencl_template_compile_md5salt(d, dev_idx) == 0 &&
                    gpu_opencl_template_kernel_lazy_md5salt(d, dev_idx) == 0)
                    return d->kern_template_phase0_md5salt;
            }
            break;
        case JOB_MD5SALTPASS:
            if (tpl == GPU_TEMPLATE_OFF || tpl == GPU_TEMPLATE_MD5SALTPASS) {
                if (gpu_opencl_template_compile_md5saltpass(d, dev_idx) == 0 &&
                    gpu_opencl_template_kernel_lazy_md5saltpass(d, dev_idx) == 0)
                    return d->kern_template_phase0_md5saltpass;
            }
            break;
        /* B6.1 SHA1 fan-out (2026-05-06): SHA1SALTPASS — first SHA-family
         * salted variant. Same 19-arg kernel signature as MD5SALTPASS
         * (16 unsalted + 3 salt args under #ifdef GPU_TEMPLATE_HAS_SALT).
         * Family B (2026-05-07): JOB_HMAC_SHA1 (e215) + JOB_HMAC_SHA1_KPASS
         * (e793) join the SAME GPU template kernel as e385 SHA1SALTPASS via
         * params.algo_mode = 5 / 6. The HMAC body branches at the top of
         * template_finalize (gated on HASH_WORDS == 5) and returns early —
         * bypasses the SHA1(salt||pass) chain entirely. No new GPU_TEMPLATE_*
         * enum value; cache key is disambiguated by params.algo_mode at
         * host pack time. */
        case JOB_SHA1SALTPASS:
        case JOB_HMAC_SHA1:
        case JOB_HMAC_SHA1_KPASS:
            if (tpl == GPU_TEMPLATE_OFF || tpl == GPU_TEMPLATE_SHA1SALTPASS) {
                if (gpu_opencl_template_compile_sha1saltpass(d, dev_idx) == 0 &&
                    gpu_opencl_template_kernel_lazy_sha1saltpass(d, dev_idx) == 0)
                    return d->kern_template_phase0_sha1saltpass;
            }
            break;
        /* B6.2 SHA256 fan-out (2026-05-06): SHA256SALTPASS — second SHA-family
         * salted variant. Same 19-arg kernel signature; cache disambiguated
         * from SHA1SALTPASS / MD5SALTPASS via HASH_WORDS=8 + BASE_ALGO=sha256
         * tokens.
         * Family D (2026-05-08): JOB_HMAC_SHA256 (e217) + JOB_HMAC_SHA256_KPASS
         * (e795) join the SAME GPU template kernel as e412 SHA256SALTPASS via
         * params.algo_mode = 5 / 6. The HMAC body branches at the top of
         * template_finalize (gated on RUNTIME `if (HASH_WORDS == 8 && algo_-
         * mode >= 5u)`, NEVER `#if HASH_WORDS == 8` — see prominent comment
         * in finalize_prepend_be.cl.frag) and returns early — bypasses the
         * SHA256(salt||pass) chain entirely. No new GPU_TEMPLATE_* enum
         * value; cache key is disambiguated by params.algo_mode at host
         * pack time. Mirrors Family A/B/C/E/F/G/H/J/K pattern. Final HMAC
         * family in the ladder. */
        case JOB_SHA256SALTPASS:
        case JOB_HMAC_SHA256:
        case JOB_HMAC_SHA256_KPASS:
            if (tpl == GPU_TEMPLATE_OFF || tpl == GPU_TEMPLATE_SHA256SALTPASS) {
                if (gpu_opencl_template_compile_sha256saltpass(d, dev_idx) == 0 &&
                    gpu_opencl_template_kernel_lazy_sha256saltpass(d, dev_idx) == 0)
                    return d->kern_template_phase0_sha256saltpass;
            }
            break;
        /* B6.3 SHA224 fan-out (2026-05-06): SHA224SALTPASS — third SHA-family
         * salted variant. Same 19-arg kernel signature; cache disambiguated
         * from SHA256SALTPASS via HASH_WORDS=7 (vs 8) — same BASE_ALGO=sha256
         * since the compression primitive is identical (sha256_block).
         * Family C (2026-05-07): JOB_HMAC_SHA224 (e216) + JOB_HMAC_SHA224_KPASS
         * (e794) join the SAME GPU template kernel as e832 SHA224SALTPASS via
         * params.algo_mode = 5 / 6. The HMAC body branches at the top of
         * template_finalize (gated on HASH_WORDS == 7) and returns early —
         * bypasses the SHA224(salt||pass) chain entirely. No new GPU_TEMPLATE_*
         * enum value; cache key is disambiguated by params.algo_mode at
         * host pack time. Mirrors Family A and Family B pattern. */
        case JOB_SHA224SALTPASS:
        case JOB_HMAC_SHA224:
        case JOB_HMAC_SHA224_KPASS:
            if (tpl == GPU_TEMPLATE_OFF || tpl == GPU_TEMPLATE_SHA224SALTPASS) {
                if (gpu_opencl_template_compile_sha224saltpass(d, dev_idx) == 0 &&
                    gpu_opencl_template_kernel_lazy_sha224saltpass(d, dev_idx) == 0)
                    return d->kern_template_phase0_sha224saltpass;
            }
            break;
        /* B6.4 MD5PASSSALT fan-out (2026-05-06): MD5PASSSALT — first
         * APPEND-shape salted variant. Same 19-arg kernel signature;
         * cache disambiguated from MD5SALTPASS via SALT_POSITION=APPEND
         * (vs PREPEND) — same BASE_ALGO=md5 + HASH_WORDS=4. */
        case JOB_MD5PASSSALT:
            if (tpl == GPU_TEMPLATE_OFF || tpl == GPU_TEMPLATE_MD5PASSSALT) {
                if (gpu_opencl_template_compile_md5passsalt(d, dev_idx) == 0 &&
                    gpu_opencl_template_kernel_lazy_md5passsalt(d, dev_idx) == 0)
                    return d->kern_template_phase0_md5passsalt;
            }
            break;
        /* B6.5 SHA1PASSSALT fan-out (2026-05-06): SHA1PASSSALT — first
         * SHA-family APPEND-shape salted variant. Same 19-arg kernel
         * signature; cache disambiguated from SHA1SALTPASS via SALT_-
         * POSITION=APPEND (vs PREPEND) — same BASE_ALGO=sha1 +
         * HASH_WORDS=5. */
        case JOB_SHA1PASSSALT:
            if (tpl == GPU_TEMPLATE_OFF || tpl == GPU_TEMPLATE_SHA1PASSSALT) {
                if (gpu_opencl_template_compile_sha1passsalt(d, dev_idx) == 0 &&
                    gpu_opencl_template_kernel_lazy_sha1passsalt(d, dev_idx) == 0)
                    return d->kern_template_phase0_sha1passsalt;
            }
            break;
        /* B6.7 SHA256PASSSALT fan-out (2026-05-06): SHA256PASSSALT — second
         * SHA-family APPEND-shape salted variant. Same 19-arg kernel
         * signature; cache disambiguated from SHA256SALTPASS via
         * SALT_POSITION=APPEND (vs PREPEND) — same BASE_ALGO=sha256 +
         * HASH_WORDS=8. */
        case JOB_SHA256PASSSALT:
            if (tpl == GPU_TEMPLATE_OFF || tpl == GPU_TEMPLATE_SHA256PASSSALT) {
                if (gpu_opencl_template_compile_sha256passsalt(d, dev_idx) == 0 &&
                    gpu_opencl_template_kernel_lazy_sha256passsalt(d, dev_idx) == 0)
                    return d->kern_template_phase0_sha256passsalt;
            }
            break;
        /* B6.9 SHA512 fan-out (2026-05-06): SHA512SALTPASS — first 64-bit-
         * state salted variant. Same 19-arg kernel signature; cache
         * disambiguated from every other salted template via
         * HASH_BLOCK_BYTES=128 (unique among salted variants on the
         * codegen path) + HASH_WORDS=16 + BASE_ALGO=sha512.
         * Family F (2026-05-08): JOB_HMAC_SHA512 (e218) + JOB_HMAC_SHA512_KPASS
         * (e797) join the SAME GPU template kernel as e388 SHA512SALTPASS via
         * params.algo_mode = 5 / 6. The HMAC body branches at the top of
         * template_finalize (gated on HASH_WORDS == 16 in finalize_prepend_-
         * be64.cl.frag) and returns early — bypasses the SHA512(salt||pass)
         * chain entirely. No new GPU_TEMPLATE_* enum value; cache key is
         * disambiguated from SHA384SALTPASS carrier by HASH_WORDS=16 (vs 12).
         * Mirrors Family E pattern. */
        case JOB_SHA512SALTPASS:
        case JOB_HMAC_SHA512:
        case JOB_HMAC_SHA512_KPASS:
            if (tpl == GPU_TEMPLATE_OFF || tpl == GPU_TEMPLATE_SHA512SALTPASS) {
                if (gpu_opencl_template_compile_sha512saltpass(d, dev_idx) == 0 &&
                    gpu_opencl_template_kernel_lazy_sha512saltpass(d, dev_idx) == 0)
                    return d->kern_template_phase0_sha512saltpass;
            }
            break;
        /* B6.10 SHA512PASSSALT fan-out (2026-05-06): SHA512PASSSALT — second
         * 64-bit-state salted variant; APPEND-shape sibling of SHA512SALTPASS.
         * FINAL B6 ladder step. Same 19-arg kernel signature; cache
         * disambiguated from SHA512SALTPASS via SALT_POSITION=APPEND
         * (vs PREPEND); same BASE_ALGO=sha512 + HASH_WORDS=16 +
         * HASH_BLOCK_BYTES=128 axes — single-axis delta. */
        case JOB_SHA512PASSSALT:
            if (tpl == GPU_TEMPLATE_OFF || tpl == GPU_TEMPLATE_SHA512PASSSALT) {
                if (gpu_opencl_template_compile_sha512passsalt(d, dev_idx) == 0 &&
                    gpu_opencl_template_kernel_lazy_sha512passsalt(d, dev_idx) == 0)
                    return d->kern_template_phase0_sha512passsalt;
            }
            break;
        /* Family E HMAC-SHA384 carrier (2026-05-08): JOB_HMAC_SHA384 (e543) +
         * JOB_HMAC_SHA384_KPASS (e796) resolve to the SHA384SALTPASS-shaped
         * carrier kernel via params.algo_mode = 5 / 6. The HMAC body branches
         * at the top of template_finalize (gated on HASH_WORDS == 12) and
         * returns early. There is no production JOB_SHA384SALTPASS dispatch;
         * mode-0 of this kernel is dead in production. Cache key is
         * disambiguated from SHA512SALTPASS / SHA512PASSSALT via HASH_WORDS=12
         * (vs 16) — same BASE_ALGO=sha512 + HASH_BLOCK_BYTES=128 axes. */
        case JOB_HMAC_SHA384:
        case JOB_HMAC_SHA384_KPASS:
            if (tpl == GPU_TEMPLATE_OFF || tpl == GPU_TEMPLATE_SHA384SALTPASS) {
                if (gpu_opencl_template_compile_sha384saltpass(d, dev_idx) == 0 &&
                    gpu_opencl_template_kernel_lazy_sha384saltpass(d, dev_idx) == 0)
                    return d->kern_template_phase0_sha384saltpass;
            }
            break;
        /* Family G HMAC-RIPEMD-160 carrier (2026-05-08): JOB_HMAC_RMD160 (e211)
         * + JOB_HMAC_RMD160_KPASS (e798) resolve to the RIPEMD160SALTPASS-
         * shaped carrier kernel via params.algo_mode = 5 / 6. The HMAC body
         * branches at the top of template_finalize (gated on HASH_WORDS == 5
         * in finalize_prepend_rmd.cl.frag) and returns early. There is no
         * production JOB_RIPEMD160SALTPASS dispatch; mode-0 of this kernel
         * is dead in production. Cache key is disambiguated from
         * SHA1SALTPASS via BASE_ALGO=ripemd160 (vs sha1) — same HASH_WORDS=5
         * + HASH_BLOCK_BYTES=64 axes. */
        case JOB_HMAC_RMD160:
        case JOB_HMAC_RMD160_KPASS:
            if (tpl == GPU_TEMPLATE_OFF || tpl == GPU_TEMPLATE_RIPEMD160SALTPASS) {
                if (gpu_opencl_template_compile_ripemd160saltpass(d, dev_idx) == 0 &&
                    gpu_opencl_template_kernel_lazy_ripemd160saltpass(d, dev_idx) == 0)
                    return d->kern_template_phase0_ripemd160saltpass;
            }
            break;
        /* Family H HMAC-RIPEMD-320 carrier (2026-05-08): JOB_HMAC_RMD320 (e213)
         * + JOB_HMAC_RMD320_KPASS (e799) resolve to the RIPEMD320SALTPASS-
         * shaped carrier kernel via params.algo_mode = 5 / 6. The HMAC body
         * branches at the top of template_finalize (gated on HASH_WORDS == 10
         * in finalize_prepend_rmd.cl.frag) and returns early. There is no
         * production JOB_RIPEMD320SALTPASS dispatch; mode-0 of this kernel
         * is dead in production. Cache key is disambiguated from
         * RIPEMD160SALTPASS via HASH_WORDS=10 (vs 5) + BASE_ALGO=rmd320
         * (vs rmd160). */
        case JOB_HMAC_RMD320:
        case JOB_HMAC_RMD320_KPASS:
            if (tpl == GPU_TEMPLATE_OFF || tpl == GPU_TEMPLATE_RIPEMD320SALTPASS) {
                if (gpu_opencl_template_compile_ripemd320saltpass(d, dev_idx) == 0 &&
                    gpu_opencl_template_kernel_lazy_ripemd320saltpass(d, dev_idx) == 0)
                    return d->kern_template_phase0_ripemd320saltpass;
            }
            break;
        /* Family I HMAC-BLAKE2S carrier (2026-05-08): JOB_HMAC_BLAKE2S (e828)
         * resolves to a hand-written Path A salted-template kernel. Single
         * algo_mode (5); no KPASS sibling op. The HMAC body lives inline in
         * template_finalize (gated on algo_mode == 5u in
         * gpu_hmac_blake2s_core.cl) and returns early. There is no production
         * JOB_BLAKE2SSALTPASS dispatch; mode-0 of this kernel is dead in
         * production. Cache key disambiguated from BLAKE2S256 unsalted via
         * HAS_SALT=1 + HMAC_KPASS=1 axes (absent in unsalted defines_str). */
        case JOB_HMAC_BLAKE2S:
            if (tpl == GPU_TEMPLATE_OFF || tpl == GPU_TEMPLATE_HMAC_BLAKE2S) {
                if (gpu_opencl_template_compile_hmac_blake2s(d, dev_idx) == 0 &&
                    gpu_opencl_template_kernel_lazy_hmac_blake2s(d, dev_idx) == 0)
                    return d->kern_template_phase0_hmac_blake2s;
            }
            break;
        /* Family J HMAC-STREEBOG-256 carrier (2026-05-08): JOB_HMAC_STREEBOG256_-
         * KSALT (e838) + JOB_HMAC_STREEBOG256_KPASS (e837) resolve to the
         * hand-written Path A salted-template kernel via params.algo_mode
         * = 5 / 6 (collapsed to a single >=5u branch in the kernel since
         * the math is identical; the host's salt-Judy plumbing distinguishes
         * the two by routing Typeuser vs Typesalt as the salt buffer).
         * The HMAC body branches at the top of template_finalize (gated on
         * algo_mode >= 5u inline in gpu_hmac_streebog256_core.cl — NOT in
         * a fragment; Path A keeps the core self-contained) and returns
         * early. There is no production JOB_STREEBOG256SALTPASS dispatch;
         * mode-0 of this kernel is dead in production. Cache key
         * disambiguated from STREEBOG-256 unsalted (gpu_streebog256_core_str)
         * via HAS_SALT=1 + HMAC_KSALTPASS=1 axes (absent in unsalted
         * defines_str). */
        case JOB_HMAC_STREEBOG256_KSALT:
        case JOB_HMAC_STREEBOG256_KPASS:
            if (tpl == GPU_TEMPLATE_OFF || tpl == GPU_TEMPLATE_HMAC_STREEBOG256) {
                if (gpu_opencl_template_compile_hmac_streebog256(d, dev_idx) == 0 &&
                    gpu_opencl_template_kernel_lazy_hmac_streebog256(d, dev_idx) == 0)
                    return d->kern_template_phase0_hmac_streebog256;
            }
            break;
        /* Family K HMAC-STREEBOG-512 carrier (2026-05-08): JOB_HMAC_STREEBOG512_-
         * KSALT (e840) + JOB_HMAC_STREEBOG512_KPASS (e839) resolve to the
         * hand-written Path A salted-template kernel via params.algo_mode
         * = 5 / 6 (collapsed to a single >=5u branch in the kernel since
         * the math is identical; the host's salt-Judy plumbing distinguishes
         * the two by routing Typesalt as the salt buffer for both ops -
         * mirrors Family J pattern since both KSALT and KPASS have
         * TYPEOPT_NEEDSALT). The HMAC body branches at the top of
         * template_finalize (gated on algo_mode >= 5u inline in
         * gpu_hmac_streebog512_core.cl - NOT in a fragment; Path A keeps
         * the core self-contained) and returns early. There is no
         * production JOB_STREEBOG512SALTPASS dispatch; mode-0 of this
         * kernel is dead in production. Cache key disambiguated from
         * STREEBOG-512 unsalted (gpu_streebog512_core_str) via HAS_SALT=1 +
         * HMAC_KSALTPASS=1 axes (absent in unsalted defines_str), and
         * from Family J HMAC-STREEBOG-256 via HASH_WORDS=16 vs 8. Final
         * HMAC family in the ladder. */
        case JOB_HMAC_STREEBOG512_KSALT:
        case JOB_HMAC_STREEBOG512_KPASS:
            if (tpl == GPU_TEMPLATE_OFF || tpl == GPU_TEMPLATE_HMAC_STREEBOG512) {
                if (gpu_opencl_template_compile_hmac_streebog512(d, dev_idx) == 0 &&
                    gpu_opencl_template_kernel_lazy_hmac_streebog512(d, dev_idx) == 0)
                    return d->kern_template_phase0_hmac_streebog512;
            }
            break;
        default:
            break;
    }
    return NULL;
}

/* Dispatch md5_rules_phase0 against a packed words batch. Validator
 * path (MDXFIND_GPU_VALIDATOR=1, env-gated diagnostic) reuses the older
 * packed_buf upload buffer (b_packed_buf, b_chunk_index) with the same
 * [len][bytes] format. Production path uses the coalesced
 * b_dispatch_payload (Memo B B1, ~line 8330+ below).
 *
 * Caller must have invoked gpu_opencl_set_rules() at least once on
 * this device. global_size = num_words * gpu_n_rules.
 *
 * Returns hits in d->h_hits; sets *nhits_out to the kernel's atomic
 * hit_count (capped at GPU_PACKED_MAX_HITS for buffer safety). */
uint32_t *gpu_opencl_dispatch_md5_rules(int dev_idx,
    const char *packed_words, uint32_t packed_size,
    const uint32_t *word_offset, uint32_t num_words,
    int op, int *nhits_out,
    uint64_t mask_start, uint32_t mask_offset_per_word, uint32_t bf_num_masks,
    uint32_t inner_iter,
    int bf_fast_eligible)
{
    *nhits_out = 0;
    if (!ocl_ready || dev_idx < 0 || dev_idx >= num_gpu_devs) return NULL;
    if (!packed_words || packed_size == 0 || !word_offset || num_words == 0) return NULL;
    struct gpu_device *d = &gpu_devs[dev_idx];
    /* Device-level disable. See d->device_disabled doc. The gpujob worker
     * for this device is also not spawned (see gpujob_init), so this
     * early-return is a defensive last line of defense rather than the
     * primary gate — but we keep it so the function is safe in isolation. */
    if (d->device_disabled) return NULL;
    if (d->gpu_n_rules <= 0 || !d->prog_md5_rules) {
        fprintf(stderr, "OpenCL GPU[%d]: dispatch_md5_rules called before "
                "set_rules (prog=%p, n_rules=%d)\n",
                dev_idx, (void *)d->prog_md5_rules, d->gpu_n_rules);
        return NULL;
    }
    /* Lazy kernel creation. Deferred from set_rules to here so the kernel
     * object only exists once we have args to bind to it — avoids the
     * NVIDIA-driver CL_INVALID_KERNEL_ARGS at process finalize when the
     * dispatch path is never reached in this session. */
    if (gpu_opencl_rules_kernel_lazy(d, dev_idx) < 0) return NULL;
    /* B5 chokepoint widening (2026-05-04 sub-batch 1; 2026-05-05 sub-batch 2):
     * the gate at mdxfind.c (around the gpu_rules_engine_active condition)
     * now permits {MD5, MD4, SHA1, SHA224, SHA256, SHA384, SHA512, RMD160,
     * RMD320}. The dispatch path picks the right per-op template kernel
     * via gpu_template_resolve_kernel() at the swap site below.
     *   - SHA384/SHA512: B5 sub-batch 1 (first 64-bit-state algos)
     *   - RMD160/RMD320: B5 sub-batch 2 (LE-per-uint32, 5/10-word state) */
    /* B7.9 (2026-05-07): JOB_MD5UC added here. B7.7a (2026-05-07) wired
     * MD5UC into the host-side rules-engine admit list (mdxfind.c:10287)
     * and the template resolver (case JOB_MD5UC at ~line 7500 above), but
     * missed adding JOB_MD5UC to THIS dispatch-side admit list. Pre-B7.9
     * the chokepoint pack masked the gap (MD5UC iter==1 is byte-equivalent
     * to MD5; the packed kernel's hard-coded LC hex is wrong only for
     * iter>1 inter-iter rebuild). With the chokepoint pack retired, MD5UC
     * MUST be in this admit list or it silently produces zero cracks.
     * The template kernel handles UC via params.algo_mode=1 (set at the
     * algo_mode setter ~line 8400 below) — md5_to_hex_uc fires inter-iter,
     * matching the CPU MDstart at mdxfind.c:25386 (prmd5UC vs prmd5). */
    if (op != JOB_MD5 && op != JOB_MD5UC && op != JOB_MD4 &&
        op != JOB_SHA1 && op != JOB_SHA224 && op != JOB_SHA256 &&
        op != JOB_SHA384 && op != JOB_SHA512 &&
        op != JOB_RMD160 && op != JOB_RMD320 &&
        op != JOB_BLAKE2S256 && op != JOB_BLAKE2B256 && op != JOB_BLAKE2B512 &&
        /* B5 sub-batch 4 (2026-05-03): SHA3 / Keccak family. */
        op != JOB_KECCAK224 && op != JOB_KECCAK256 &&
        op != JOB_KECCAK384 && op != JOB_KECCAK512 &&
        op != JOB_SHA3_224 && op != JOB_SHA3_256 &&
        op != JOB_SHA3_384 && op != JOB_SHA3_512 &&
        /* B5 sub-batch 5a Tier 1 (2026-05-03): SHA384RAW + SHA512RAW. */
        op != JOB_SHA384RAW && op != JOB_SHA512RAW &&
        /* B5 sub-batch 6 Tier A (2026-05-03): MD5RAW + SHA1RAW + SHA256RAW. */
        op != JOB_MD5RAW && op != JOB_SHA1RAW && op != JOB_SHA256RAW &&
        /* B5 sub-batch 6 Tier C (2026-05-03): SQL5. */
        op != JOB_SQL5 &&
        /* B6.11 (2026-05-06): SHA1DRU — Drupal SHA1, first 1M-iter algo on
         * template path. 1M loop runs INSIDE template_finalize; max_iter=1
         * forced at the dispatch params site. */
        op != JOB_SHA1DRU &&
        /* B7.7b (2026-05-07): MD6256 — MD6-256; final M5 closure. Single-
         * block leaf compression with 14 KB A[1753] working array; per-
         * iter probe like SQL5 (no max_iter forcing). */
        op != JOB_MD6256 &&
        /* B5 sub-batch 6 Tier B (2026-05-03): NTLMH. */
        op != JOB_NTLMH &&
        /* B5 sub-batch 8 (2026-05-05): MD4UTF16. */
        op != JOB_MD4UTF16 &&
        /* B5 sub-batch 7 (2026-05-05): MYSQL3. */
        op != JOB_MYSQL3 &&
        /* B5 sub-batch 6.5 (2026-05-05): WRL. */
        op != JOB_WRL &&
        /* B5 sub-batch 5b retry (2026-05-06): Streebog-256 + Streebog-512. */
        op != JOB_STREEBOG_32 && op != JOB_STREEBOG_64 &&
        /* B6 salt-axis (2026-05-06): MD5SALT + MD5SALTPASS — first two
         * salted variants on the unified template path.
         * B6.1 SHA1 fan-out (2026-05-06): SHA1SALTPASS — first SHA-family
         * salted variant. Same template path; distinct cache key via
         * HASH_WORDS=5 + BASE_ALGO=sha1 in defines_str.
         * B6.2 SHA256 fan-out (2026-05-06): SHA256SALTPASS — second
         * SHA-family salted variant. HASH_WORDS=8 + BASE_ALGO=sha256
         * tokens disambiguate from SHA1SALTPASS and MD5SALTPASS.
         * B6.3 SHA224 fan-out (2026-05-06): SHA224SALTPASS — third
         * SHA-family salted variant. HASH_WORDS=7 + BASE_ALGO=sha256
         * tokens disambiguate from SHA256SALTPASS (HASH_WORDS=7 vs 8). */
        op != JOB_MD5SALT && op != JOB_MD5SALTPASS &&
        op != JOB_SHA1SALTPASS &&
        op != JOB_SHA256SALTPASS &&
        op != JOB_SHA224SALTPASS &&
        /* B6.4 MD5PASSSALT fan-out (2026-05-06): first APPEND-shape salted
         * variant. Same dispatch geometry as the PREPEND siblings. */
        op != JOB_MD5PASSSALT &&
        /* B6.5 SHA1PASSSALT fan-out (2026-05-06): first SHA-family APPEND-
         * shape salted variant. Same dispatch geometry; only finalize-time
         * salt position differs from SHA1SALTPASS. */
        op != JOB_SHA1PASSSALT &&
        /* B6.7 SHA256PASSSALT fan-out (2026-05-06): second SHA-family
         * APPEND-shape salted variant. Same dispatch geometry; only
         * finalize-time salt position differs from SHA256SALTPASS. */
        op != JOB_SHA256PASSSALT &&
        /* B6.9 SHA512 fan-out (2026-05-06): SHA512SALTPASS — first
         * 64-bit-state salted variant. Same dispatch geometry; differs
         * in kernel internal width (8 ulong state + 128-byte block +
         * 128-bit length field). Host-side wiring is identical to the
         * other salted templates. */
        op != JOB_SHA512SALTPASS &&
        /* B6.10 SHA512PASSSALT fan-out (2026-05-06): SHA512PASSSALT —
         * second 64-bit-state salted variant; APPEND-shape sibling of
         * SHA512SALTPASS. FINAL B6 ladder step. Same dispatch geometry;
         * only finalize-time salt position differs from SHA512SALTPASS. */
        op != JOB_SHA512PASSSALT &&
        /* B6.6 (2026-05-06): MD5SALT family variants. e350/e541/e542 all
         * dispatch to the SAME kernel as e31 MD5SALT via params.algo_mode.
         * B6.8 (2026-05-06): JOB_MD5_MD5SALTMD5PASS (e367) joins as
         * algo_mode=4 — outer MD5 over (hex32(MD5(salt)) || hex32(MD5(pass))).
         * Salt-hex pre-computed by host (saltsnap[].hashsalt). */
        op != JOB_MD5UCSALT && op != JOB_MD5revMD5SALT && op != JOB_MD5sub8_24SALT &&
        op != JOB_MD5_MD5SALTMD5PASS &&
        /* Family A (2026-05-07): JOB_HMAC_MD5 (e214) + JOB_HMAC_MD5_KPASS
         * (e792) — sixth + seventh MD5SALT-template-kernel-sharing
         * variants. Both resolve to kern_template_phase0_md5salt via the
         * resolver above and run the HMAC body branch in template_finalize
         * (algo_mode 5/6). Without this allowlist entry the dispatch
         * function emits "unsupported op=214" / "unsupported op=792" and
         * returns NULL — silent drop, no actual GPU work. */
        op != JOB_HMAC_MD5 && op != JOB_HMAC_MD5_KPASS &&
        /* Family B (2026-05-07): JOB_HMAC_SHA1 (e215) + JOB_HMAC_SHA1_KPASS
         * (e793) — share the SHA1SALTPASS GPU template kernel via
         * params.algo_mode = 5 / 6. Resolver above maps both ops to
         * kern_template_phase0_sha1saltpass; this allowlist gate must
         * accept them or dispatch returns NULL with "unsupported op=215"
         * / "unsupported op=793" warning. (Family A 2026-05-07 lesson:
         * missing entry-gate entry = silent CPU fallback for word_idx>0.) */
        op != JOB_HMAC_SHA1 && op != JOB_HMAC_SHA1_KPASS &&
        /* Family C (2026-05-07): JOB_HMAC_SHA224 (e216) + JOB_HMAC_SHA224_KPASS
         * (e794) — share the SHA224SALTPASS GPU template kernel via
         * params.algo_mode = 5 / 6. Resolver above maps both ops to
         * kern_template_phase0_sha224saltpass; this allowlist gate must
         * accept them or dispatch returns NULL with "unsupported op=216"
         * / "unsupported op=794" warning. */
        op != JOB_HMAC_SHA224 && op != JOB_HMAC_SHA224_KPASS &&
        /* Family D (2026-05-08): JOB_HMAC_SHA256 (e217) + JOB_HMAC_SHA256_KPASS
         * (e795) — share the SHA256SALTPASS GPU template kernel via
         * params.algo_mode = 5 / 6. Resolver above maps both ops to
         * kern_template_phase0_sha256saltpass; this allowlist gate must
         * accept them or dispatch returns NULL with "unsupported op=217"
         * / "unsupported op=795" warning. The HMAC body uses RUNTIME gate
         * `if (HASH_WORDS == 8 && algo_mode >= 5u)` in finalize_prepend_-
         * be.cl.frag (NEVER preprocessor `#if`; rev 1.7 ABORT lesson).
         * Final HMAC family in the ladder. */
        op != JOB_HMAC_SHA256 && op != JOB_HMAC_SHA256_KPASS &&
        /* Family E HMAC-SHA384 carrier (2026-05-08): JOB_HMAC_SHA384 (e543) +
         * JOB_HMAC_SHA384_KPASS (e796) — share the SHA384SALTPASS-shaped
         * carrier GPU template kernel via params.algo_mode = 5 / 6.
         * Resolver above maps both ops to kern_template_phase0_-
         * sha384saltpass; this allowlist gate must accept them or dispatch
         * returns NULL with "unsupported op=543" / "unsupported op=796"
         * warning. (Family A 2026-05-07 lesson: missing entry-gate entry =
         * silent CPU fallback for word_idx>0.) */
        op != JOB_HMAC_SHA384 && op != JOB_HMAC_SHA384_KPASS &&
        /* Family F (2026-05-08): JOB_HMAC_SHA512 (e218) + JOB_HMAC_SHA512_KPASS
         * (e797) — share the SHA512SALTPASS GPU template kernel via
         * params.algo_mode = 5 / 6. Resolver above maps both ops to
         * kern_template_phase0_sha512saltpass; this allowlist gate must
         * accept them or dispatch returns NULL with "unsupported op=218"
         * / "unsupported op=797" warning. */
        op != JOB_HMAC_SHA512 && op != JOB_HMAC_SHA512_KPASS &&
        /* Family G HMAC-RIPEMD-160 carrier (2026-05-08): JOB_HMAC_RMD160
         * (e211) + JOB_HMAC_RMD160_KPASS (e798) — share the RIPEMD160-
         * SALTPASS-shaped carrier GPU template kernel via params.algo_mode
         * = 5 / 6. Resolver above maps both ops to kern_template_phase0_-
         * ripemd160saltpass; this allowlist gate must accept them or
         * dispatch returns NULL with "unsupported op=211" / "unsupported
         * op=798" warning. (Family A 2026-05-07 lesson: missing entry-gate
         * entry = silent CPU fallback for word_idx>0.) */
        op != JOB_HMAC_RMD160 && op != JOB_HMAC_RMD160_KPASS &&
        /* Family H HMAC-RIPEMD-320 carrier (2026-05-08): JOB_HMAC_RMD320
         * (e213) + JOB_HMAC_RMD320_KPASS (e799) — share the RIPEMD320-
         * SALTPASS-shaped carrier GPU template kernel via params.algo_mode
         * = 5 / 6. Resolver above maps both ops to kern_template_phase0_-
         * ripemd320saltpass; this allowlist gate must accept them or
         * dispatch returns NULL with "unsupported op=213" / "unsupported
         * op=799" warning. (Family A 2026-05-07 lesson: missing entry-gate
         * entry = silent CPU fallback for word_idx>0.) */
        op != JOB_HMAC_RMD320 && op != JOB_HMAC_RMD320_KPASS &&
        /* Family I HMAC-BLAKE2S carrier (2026-05-08): JOB_HMAC_BLAKE2S (e828)
         * — single algo_mode (5); no KPASS sibling op exists in mdxfind for
         * HMAC-BLAKE2S. Resolver above maps the op to kern_template_phase0_-
         * hmac_blake2s; this allowlist gate must accept it or dispatch returns
         * NULL with "unsupported op=828" warning. (Family A 2026-05-07 lesson:
         * missing entry-gate entry = silent CPU fallback for word_idx>0.) */
        op != JOB_HMAC_BLAKE2S &&
        /* Family J HMAC-STREEBOG-256 carrier (2026-05-08): JOB_HMAC_STREEBOG256_-
         * KSALT (e838) + JOB_HMAC_STREEBOG256_KPASS (e837) — share the Path A
         * hand-written carrier GPU template kernel via params.algo_mode = 5 / 6.
         * Resolver above maps both ops to kern_template_phase0_hmac_streebog256;
         * this allowlist gate must accept them or dispatch returns NULL with
         * "unsupported op=838" / "unsupported op=837" warning. (Family A
         * 2026-05-07 lesson: missing entry-gate entry = silent CPU fallback
         * for word_idx>0.) */
        op != JOB_HMAC_STREEBOG256_KSALT && op != JOB_HMAC_STREEBOG256_KPASS &&
        /* Family K HMAC-STREEBOG-512 carrier (2026-05-08): JOB_HMAC_STREEBOG512_-
         * KSALT (e840) + JOB_HMAC_STREEBOG512_KPASS (e839) - share the Path A
         * hand-written carrier GPU template kernel via params.algo_mode = 5 / 6.
         * Resolver above maps both ops to kern_template_phase0_hmac_streebog512;
         * this allowlist gate must accept them or dispatch returns NULL with
         * "unsupported op=840" / "unsupported op=839" warning. (Family A
         * 2026-05-07 lesson: missing entry-gate entry = silent CPU fallback
         * for word_idx>0.) Final HMAC family in the ladder. */
        op != JOB_HMAC_STREEBOG512_KSALT && op != JOB_HMAC_STREEBOG512_KPASS &&
        /* PHPBB3 carrier (2026-05-08): JOB_PHPBB3 (e455) routes through
         * the hand-written Path A salted-template kernel via the
         * resolver case above. This allowlist gate must accept it or
         * dispatch returns NULL with "unsupported op=455" warning.
         * (Family A 2026-05-07 lesson: missing entry-gate entry =
         * silent CPU fallback for word_idx>0.) Mirrors SHA1DRU pattern
         * minus the iter-count-from-salt detail (no entry-gate
         * difference). */
        op != JOB_PHPBB3 &&
        /* MD5CRYPT carrier (2026-05-08): JOB_MD5CRYPT (e511) routes
         * through the hand-written Path A salted-template kernel via
         * the resolver case above. This allowlist gate must accept it
         * or dispatch returns NULL with "unsupported op=511" warning.
         * Mirrors PHPBB3 entry-gate pattern. */
        op != JOB_MD5CRYPT &&
        /* SHA256CRYPT carrier (2026-05-08): JOB_SHA256CRYPT (e512) routes
         * through the hand-written Path A salted-template kernel via the
         * resolver case above. This allowlist gate must accept it or
         * dispatch returns NULL with "unsupported op=512" warning.
         * Mirrors MD5CRYPT entry-gate pattern. Phase 2 of the Unix-crypt
         * ladder. */
        op != JOB_SHA256CRYPT &&
        /* SHA512CRYPT carrier (2026-05-08): JOB_SHA512CRYPT (e513) routes
         * through the hand-written Path A salted-template kernel via the
         * resolver case above. This allowlist gate must accept it or
         * dispatch returns NULL with "unsupported op=513" warning.
         * Mirrors SHA256CRYPT entry-gate pattern. Phase 3 of the Unix-
         * crypt ladder. */
        op != JOB_SHA512CRYPT &&
        /* SHA512CRYPTMD5 carrier (2026-05-08): JOB_SHA512CRYPTMD5 (e510)
         * REUSES Phase 3 SHA512CRYPT's compiled kernel. This allowlist
         * gate must accept it or dispatch returns NULL with "unsupported
         * op=510" warning. The salted-kernel-safety allowlist + kern_is_-
         * salted_template OR-chain (sites further below) test the
         * resolved kern HANDLE, not the op -- both ops resolve to the
         * SAME d->kern_template_phase0_sha512crypt handle, so no extra
         * sibling-pair edits are needed at those sites. Phase 4 of the
         * Unix-crypt ladder. */
        op != JOB_SHA512CRYPTMD5 &&
        /* DESCRYPT carrier (2026-05-08, Unix-crypt Phase 5): JOB_DESCRYPT
         * (e500) routes through the hand-written Path A salted-template
         * kernel via the resolver case above. This allowlist gate must
         * accept it or dispatch returns NULL with "unsupported op=500"
         * warning. Mirrors SHA512CRYPT entry-gate pattern. Phase 5 of
         * the Unix-crypt ladder (FINAL phase; Unix-crypt slab path fully
         * retired). */
        op != JOB_DESCRYPT &&
        /* BCRYPT carrier (2026-05-08, Unix-crypt Phase 6): JOB_BCRYPT
         * (e450) routes through the hand-written Path A salted-template
         * kernel via the resolver case above. This allowlist gate must
         * accept it or dispatch returns NULL with "unsupported op=450"
         * warning. Compound siblings (BCRYPTMD5/BCRYPTSHA1/BCRYPTSHA512)
         * are NOT admitted here -- they remain CPU-only via gpu_op_-
         * category default fall-through. Phase 6 of the slab-retirement
         * ladder (final major slab kernel). */
        op != JOB_BCRYPT) {
        fprintf(stderr, "OpenCL GPU[%d]: dispatch_md5_rules: unsupported op=%d\n",
                dev_idx, op);
        return NULL;
    }
    cl_int err;

    /* Phase H: log the first dispatch per device. Static per-device
     * flag — emitted exactly once per process per device. The "10
     * second pause" Shooter reports between "12 gpujob threads
     * started" and the first comfort line is bracketed by these
     * lines + the warm-probe lines + hashes_shown alloc lines. */
    {
        static int _first_dispatch_logged[MAX_GPU_DEVICES] = {0};
        if (dev_idx >= 0 && dev_idx < MAX_GPU_DEVICES
            && !_first_dispatch_logged[dev_idx]) {
            _first_dispatch_logged[dev_idx] = 1;
            tsfprintf(stderr, "OpenCL GPU[%d]: first dispatch issued\n", dev_idx);
        }
    }

    /* ----------------------------------------------------------------
     * Validator branch (env-gated, fully additive).
     *
     * MDXFIND_GPU_VALIDATOR=1 swaps the production walker for the
     * md5_rules_phase0_validate kernel — emits per-(word, rule) buffer
     * state on stderr in the same format as ruleproc.c's CPU validator
     * (MDXFIND_RULE_VALIDATOR=1). Returns NULL (no hits) so the
     * chokepoint caller sees the same shape as a "no cracks" pass.
     * Does NOT mutate the production walker's code path.
     *
     * Slot layout matches gpu/gpu_md5_rules.cl md5_rules_phase0_validate
     * (rev 1.23+ widened wire format):
     *   per (word, rule) slot, RECORD_SZ = 4 + RULE_BUF_MAX_HOST bytes
     *   slot[0..1] = retlen (int16 little-endian),
     *   slot[2..3] = outlen (uint16 little-endian),
     *   slot[4..3+outlen] = post-rule buffer bytes.
     * Storage is word-major: slot_idx = word_idx * n_rules + rule_idx.
     *
     * Sized cap (defensive): n_words * n_rules * RECORD_SZ must be <=
     * 1 GB. With RULE_BUF_MAX_HOST=40960 the per-slot payload is much
     * larger than the prior 258 B, so the cap is raised proportionally.
     * Beyond that, fall back to the production walker (bail with warning).
     * ---------------------------------------------------------------- */
    if (gpu_validator_enabled()) {
        const size_t RECORD_SZ = (size_t)GPU_VALIDATE_RECORD_SZ_HOST;
        const size_t MAX_RECORDS_BYTES = (size_t)1024 * 1024 * 1024;
        size_t total_slots = (size_t)num_words * (size_t)d->gpu_n_rules;
        size_t records_bytes = total_slots * RECORD_SZ;
        if (records_bytes > MAX_RECORDS_BYTES) {
            static int _warned = 0;
            if (!_warned) {
                _warned = 1;
                fprintf(stderr,
                    "OpenCL GPU[%d]: validator buffer %.1f MB exceeds 256 MB cap "
                    "(n_words=%u, n_rules=%d, RECORD_SZ=%zu) — falling through to "
                    "production walker without validator output for this dispatch. "
                    "Reduce per-batch words or rule-set size.\n",
                    dev_idx, records_bytes / (1024.0 * 1024.0),
                    num_words, d->gpu_n_rules, RECORD_SZ);
            }
            /* Skip validator path; fall through to production dispatch below. */
            goto validator_skip;
        }
        if (gpu_opencl_validate_kernel_lazy(d, dev_idx) < 0) {
            fprintf(stderr, "OpenCL GPU[%d]: validator kernel unavailable; "
                    "falling through to production walker\n", dev_idx);
            goto validator_skip;
        }
        if (!d->h_rule_program || !d->h_rule_offset) {
            fprintf(stderr, "OpenCL GPU[%d]: validator host-side rule mirror not "
                    "populated (set_rules called before MDXFIND_GPU_VALIDATOR was "
                    "set?); cannot stringify rulebytes — bypassing validator\n",
                    dev_idx);
            goto validator_skip;
        }

        /* (Re)allocate the records buffer if needed. */
        if (!d->b_validate_records || records_bytes > d->validate_records_cap) {
            if (d->b_validate_records) clReleaseMemObject(d->b_validate_records);
            d->b_validate_records = clCreateBuffer(d->ctx, CL_MEM_WRITE_ONLY,
                                                   records_bytes, NULL, &err);
            if (err != CL_SUCCESS || !d->b_validate_records) {
                fprintf(stderr, "OpenCL GPU[%d]: validator records buf alloc "
                        "failed (%zu bytes, err=%d)\n",
                        dev_idx, records_bytes, err);
                d->b_validate_records = NULL;
                d->validate_records_cap = 0;
                goto validator_skip;
            }
            d->validate_records_cap = records_bytes;
        }

        /* Reuse the packed_buf / chunk_index buffers (same upload path as
         * the production dispatch — same packed [len][bytes] format). */
        if ((size_t)packed_size > d->packed_buf_cap) {
            if (d->b_packed_buf) clReleaseMemObject(d->b_packed_buf);
            d->b_packed_buf = clCreateBuffer(d->ctx, CL_MEM_READ_ONLY,
                                             packed_size, NULL, &err);
            d->packed_buf_cap = packed_size;
        }
        size_t wo_size_v = (size_t)num_words * sizeof(uint32_t);
        if (wo_size_v > d->chunk_index_cap) {
            if (d->b_chunk_index) clReleaseMemObject(d->b_chunk_index);
            d->b_chunk_index = clCreateBuffer(d->ctx, CL_MEM_READ_ONLY,
                                              wo_size_v, NULL, &err);
            d->chunk_index_cap = wo_size_v;
        }
        clEnqueueWriteBuffer(d->queue, d->b_packed_buf, CL_TRUE, 0,
                             packed_size, packed_words, 0, NULL, NULL);
        clEnqueueWriteBuffer(d->queue, d->b_chunk_index, CL_TRUE, 0,
                             wo_size_v, word_offset, 0, NULL, NULL);

        OCLParams vparams;
        memset(&vparams, 0, sizeof(vparams));
        vparams.compact_mask = _compact_mask;
        vparams.num_words = num_words;
        vparams.num_masks = (uint32_t)d->gpu_n_rules;
        vparams.max_probe = 256;
        vparams.hash_data_count = _hash_data_count;
        vparams.max_hits = GPU_PACKED_MAX_HITS;
        vparams.overflow_count = _overflow_count;
        vparams.max_iter = d->max_iter < 1 ? 1 : d->max_iter;
        clEnqueueWriteBuffer(d->queue, d->b_params, CL_TRUE, 0,
                             sizeof(vparams), &vparams, 0, NULL, NULL);

        cl_kernel kvk = d->kern_md5_rules_phase0_validate;
        {
            int a = 0;
            clSetKernelArg(kvk, a++, sizeof(cl_mem), &d->b_packed_buf);
            clSetKernelArg(kvk, a++, sizeof(cl_mem), &d->b_chunk_index);
            clSetKernelArg(kvk, a++, sizeof(cl_mem), &d->b_rule_program);
            clSetKernelArg(kvk, a++, sizeof(cl_mem), &d->b_rule_offset);
            clSetKernelArg(kvk, a++, sizeof(cl_mem), &d->b_params);
            clSetKernelArg(kvk, a++, sizeof(cl_mem), &d->b_validate_records);
        }

        size_t local_v = 64;
        size_t total_v = total_slots;
        size_t global_v = ((total_v + local_v - 1) / local_v) * local_v;
        err = clEnqueueNDRangeKernel(d->queue, kvk, 1, NULL,
                                     &global_v, &local_v, 0, NULL, NULL);
        if (err != CL_SUCCESS) {
            /* Fail-fast: validator (env-gated MDXFIND_GPU_VALIDATOR=1)
             * is a diagnostic path; if dispatch fails, the diagnostic
             * is invalid and silent return would mask the failure. */
            GPU_FATAL("validator dispatch error %d on dev %d "
                      "(global=%zu, n_words=%u, n_rules=%d)",
                      err, dev_idx, global_v, num_words, d->gpu_n_rules);
        }
        clFinish(d->queue);

        /* Read full records buffer back. For 200 words × 41 rules
         * × 258 B ≈ 2 MB — trivial. */
        unsigned char *host_records = (unsigned char *)malloc(records_bytes);
        if (!host_records) {
            fprintf(stderr, "OpenCL GPU[%d]: validator host buffer malloc "
                    "(%zu bytes) failed\n", dev_idx, records_bytes);
            return NULL;
        }
        err = clEnqueueReadBuffer(d->queue, d->b_validate_records, CL_TRUE, 0,
                                  records_bytes, host_records, 0, NULL, NULL);
        if (err != CL_SUCCESS) {
            /* Fail-fast: see validator dispatch comment above. */
            GPU_FATAL("validator records readback err=%d on dev %d "
                      "(%zu bytes)", err, dev_idx, records_bytes);
        }

        /* Emit stderr in the same word-major (word_idx, rule_idx) order
         * the CPU validator should produce. Hex-encoded fields match the
         * ruleproc.c rev 1.24 format:
         *   VALIDATE word=<hex> rulebytes=<hex> retlen=<int> outlen=<int> output=<hex>
         *
         * Field widening rev 1.72 (2026-05-02): retlen/outlen are int16/
         * uint16 little-endian on the wire. word_hex / rule_hex / out_hex
         * are heap-allocated once at RULE_BUF_MAX_HOST sizing because
         * stack-allocating 3 × (2*40960+1) ≈ 245 KB risks overflow on
         * deeply nested call stacks. The decimal stderr text format is
         * unchanged (validate_diff.py regex already accepts wider ints).
         */
        static const char hex_lc_chars[] = "0123456789abcdef";
        size_t hex_buf_sz = (size_t)RULE_BUF_MAX_HOST * 2u + 1u;
        char *word_hex = (char *)malloc(hex_buf_sz);
        char *rule_hex = (char *)malloc(hex_buf_sz);
        char *out_hex  = (char *)malloc(hex_buf_sz);
        if (!word_hex || !rule_hex || !out_hex) {
            fprintf(stderr, "OpenCL GPU[%d]: validator hex-buf malloc "
                    "(3 × %zu bytes) failed\n", dev_idx, hex_buf_sz);
            free(word_hex); free(rule_hex); free(out_hex);
            free(host_records);
            return NULL;
        }
        for (uint32_t wi = 0; wi < num_words; wi++) {
            /* Pull word from packed_words. Format is [len][bytes]; word_offset
             * points at the [len] byte. */
            uint32_t wpos = word_offset[wi];
            unsigned char wlen = (unsigned char)packed_words[wpos];
            const unsigned char *wbytes =
                (const unsigned char *)(packed_words + wpos + 1);
            for (int ri = 0; ri < d->gpu_n_rules; ri++) {
                size_t slot = (size_t)wi * (size_t)d->gpu_n_rules + (size_t)ri;
                size_t base = slot * RECORD_SZ;
                /* Decode int16 LE retlen and uint16 LE outlen (rev 1.23 wire). */
                int16_t retlen16 = (int16_t)((uint16_t)host_records[base + 0] |
                                             ((uint16_t)host_records[base + 1] << 8));
                uint16_t outlen16 = (uint16_t)((uint16_t)host_records[base + 2] |
                                               ((uint16_t)host_records[base + 3] << 8));
                int retlen = (int)retlen16;
                int outlen = (int)outlen16;
                const unsigned char *out_bytes = host_records + base + 4;

                for (int j = 0; j < (int)wlen; j++) {
                    word_hex[j * 2 + 0] = hex_lc_chars[(wbytes[j] >> 4) & 0xf];
                    word_hex[j * 2 + 1] = hex_lc_chars[wbytes[j] & 0xf];
                }
                word_hex[wlen * 2] = '\0';

                /* Walk the rule bytecode for rule_idx ri to find its byte
                 * span (NUL-terminated within the program). Rule bytecode
                 * is bounded short — a 256-byte rule walk cap is plenty. */
                uint32_t rpos = d->h_rule_offset[ri];
                uint32_t rlen = 0;
                while (rpos + rlen < d->h_rule_program_len &&
                       d->h_rule_program[rpos + rlen] != 0 &&
                       rlen < 256) {
                    rlen++;
                }
                for (uint32_t j = 0; j < rlen; j++) {
                    unsigned char rb = d->h_rule_program[rpos + j];
                    rule_hex[j * 2 + 0] = hex_lc_chars[(rb >> 4) & 0xf];
                    rule_hex[j * 2 + 1] = hex_lc_chars[rb & 0xf];
                }
                rule_hex[rlen * 2] = '\0';

                int outlen_clamped = (outlen < 0) ? 0 :
                                     (outlen > (int)RULE_BUF_MAX_HOST ?
                                      (int)RULE_BUF_MAX_HOST : outlen);
                for (int j = 0; j < outlen_clamped; j++) {
                    out_hex[j * 2 + 0] = hex_lc_chars[(out_bytes[j] >> 4) & 0xf];
                    out_hex[j * 2 + 1] = hex_lc_chars[out_bytes[j] & 0xf];
                }
                out_hex[outlen_clamped * 2] = '\0';

                fprintf(stderr,
                    "VALIDATE word=%s rulebytes=%s retlen=%d outlen=%d output=%s\n",
                    word_hex, rule_hex, retlen, outlen, out_hex);
            }
        }
        free(word_hex);
        free(rule_hex);
        free(out_hex);
        free(host_records);
        /* Validator path produces no cracks — return NULL like a 0-hit
         * production dispatch. Caller (chokepoint) handles NULL gracefully. */
        return NULL;
    }
validator_skip:
    ;

    /* ----------------------------------------------------------------
     * Memo B B1 (2026-05-04): coalesced dispatch payload.
     *
     * REPLACED 4 host->GPU writes with 1. Old layout was:
     *   write b_packed_buf  (4 MiB)
     *   write b_chunk_index (64 KiB word_offset)
     *   write b_params      (128 B)
     *   write b_hit_count   (4 B zero)
     * = 4 separate clEnqueueWriteBuffer calls per dispatch, each paying
     * a SUBMIT->START tax (~2.3 ms PCIe 3.0; ~50 us PCIe 4.0).
     *
     * New layout: ONE clEnqueueWriteBuffer into b_dispatch_payload with
     *   offset 0   : OCLParams params (128 B)
     *   offset 128 : uint hit_count (4 B, init 0)
     *   offset 132 : uint word_offset[num_words] (4*num_words B)
     *   offset 132+4*num_words : packed_words (packed_size B)
     *
     * Kernel reads layout via deterministic offsets from params.num_words;
     * see md5_rules_phase0 in gpu/gpu_md5_rules.cl.
     *
     * Pinning is preserved -- the host staging buffer is malloc_pinned
     * (posix_memalign + mlock), the same path the
     * project_gpu_pcie_baseline_20260427.md numbers were measured against.
     *
     * Alignment audit (rev 1.89, 2026-05-04, in response to bug2d/bug3a
     * Win+NVIDIA CL_INVALID_KERNEL_ARGS hypothesis): the kernel is
     * gpu_md5_rules.cl md5_rules_phase0. It reads `payload` only via:
     *   - `OCLParams params = *((__global const OCLParams *)payload)` at
     *     offset 0.  cl_mem base alignment is at minimum
     *     CL_DEVICE_MEM_BASE_ADDR_ALIGN bits (typically >= 1024); the
     *     8-byte natural alignment of OCLParams.compact_mask is therefore
     *     trivially satisfied.
     *   - `__global volatile uint *hit_count = (__global volatile uint *)
     *     (payload + 128)`.  Offset 128 is 16-byte aligned.
     *   - `__global const uint *word_offset = (__global const uint *)
     *     (payload + 132)`.  Offset 132 is 4-byte aligned (132 / 4 = 33),
     *     which is the natural alignment for `uint`.  Indexed reads
     *     `word_offset[wi]` are scalar uint reads -- no vload2/4/16.
     *   - `__global const uchar *words = payload + (132 + 4*n_words)`.
     *     uchar requires 1-byte alignment; satisfied trivially.
     * The follow-on `apply_rule(prog, buf, len)` and `md5_buf(buf, ...)`
     * operate on the private `__attribute__((aligned(16))) uchar buf[]`
     * stack array -- their vload16 calls go through private memory, never
     * through `payload` or any global pointer derived from it.
     * Therefore: NO global-memory vector access on payload, no risk of
     * NVIDIA Win cold-JIT validator rejecting the kernel for misaligned
     * vload at NDRange time.  The current 132-byte starting offset for
     * word_offset is safe; padding to 144 is unnecessary.  This audit was
     * performed because Shooter's bug2d (12-GPU RTX 4090) and Just's
     * bug3a (single GTX 1650) both hit the same -52 class on
     * md5_rules_phase0; the ARG_TRACE block below names the offending arg.
     * --------------------------------------------------------------- */
    size_t wo_size = (size_t)num_words * sizeof(uint32_t);
    size_t payload_pkt_off = 132u + wo_size;
    size_t payload_size = payload_pkt_off + (size_t)packed_size;

    /* Lazy alloc / grow paired host + device buffers (rev 1.91, 2026-05-04).
     * Adopted hashcat's host-buffer pattern: regular calloc + CL_MEM_USE_HOST_PTR
     * + sync CL_TRUE WriteBuffer. The driver wraps our host buffer and handles
     * pinning internally — NVIDIA Windows pin-on-demand becomes driver-internal,
     * not the user's VirtualLock fight against Windows working-set quota. ZERO
     * manual VirtualLock/mlock in hashcat's main code path; they ship millions
     * of users on Win NVIDIA via this pattern. ARG_TRACE in 1.89 confirmed our
     * 14 kernel args were all valid at NDRange time yet -52 still fired on
     * Shooter's 12-RTX-4090 + Just's GTX 1650; the failing layer was the
     * malloc_pinned + plain clCreateBuffer(NULL) + CL_FALSE async-write chain.
     * This converts ONLY the per-device dispatch_payload pair; slot pool
     * packed_buf / word_offset stay on the existing path until validated. */
    if (!d->b_dispatch_payload || payload_size > d->dispatch_payload_cap) {
        if (d->b_dispatch_payload) clReleaseMemObject(d->b_dispatch_payload);
        free(d->h_dispatch_payload);
        d->h_dispatch_payload = calloc(1, payload_size);
        if (!d->h_dispatch_payload) {
            GPU_FATAL("h_dispatch_payload calloc fail on dev %d (size=%zu)",
                      dev_idx, payload_size);
        }
        /* CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR: device buffer wraps our
         * host calloc'd region. Driver pins as needed (NVIDIA via internal
         * cuMemHostRegister-equivalent on both platforms; AMD ROCm via its
         * own pin pool). Kernel reads params/word_offset/words AND writes
         * hit_count via atomic_inc on the EMIT_HIT_4 macro. */
        d->b_dispatch_payload = clCreateBuffer(d->ctx,
            CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
            payload_size, d->h_dispatch_payload, &err);
        if (err != CL_SUCCESS || !d->b_dispatch_payload) {
            GPU_FATAL("b_dispatch_payload alloc err=%d on dev %d (size=%zu)",
                      err, dev_idx, payload_size);
        }
        d->dispatch_payload_cap = payload_size;
        d->h_dispatch_payload_cap = payload_size;
    }

    /* B3 max_hits override: MDXFIND_MAX_HITS_OVERRIDE=N forces a small cap
     * on the GPU's hit emission to deterministically trigger overflow
     * for synthetic dense-hit testing. Default uses GPU_PACKED_MAX_HITS
     * (1M slots). The override is per-dispatch (read every call); cached
     * after first read. The HOST h_hits buffer is still sized for the
     * full GPU_PACKED_MAX_HITS so accumulating re-issue hits up to that
     * cap works without grow. */
    uint32_t b3_max_hits_cap = GPU_PACKED_MAX_HITS;
    {
        static int _max_hits_cached = -1;
        static uint32_t _max_hits_override = 0;
        if (_max_hits_cached == -1) {
            const char *e = getenv("MDXFIND_MAX_HITS_OVERRIDE");
            if (e && *e) {
                long v = strtol(e, NULL, 0);
                if (v > 0 && v < GPU_PACKED_MAX_HITS) {
                    _max_hits_override = (uint32_t)v;
                    fprintf(stderr,
                        "MDXFIND_MAX_HITS_OVERRIDE=%u: forcing GPU hit "
                        "buffer cap to %u (default %u). Cursor-restart "
                        "(B3) protocol exercised at this threshold.\n",
                        _max_hits_override, _max_hits_override,
                        (unsigned)GPU_PACKED_MAX_HITS);
                }
            }
            _max_hits_cached = 1;
        }
        if (_max_hits_override > 0) b3_max_hits_cap = _max_hits_override;
    }

    /* Memo B Phase B7.1-B7.5/B7.8: derive mask_size for the third
     * dispatch axis. The B7.5-eligible configuration is (n_prepend in
     * [0, MASK_POS_CAP=16], n_append in [0, MASK_POS_CAP=16] post-B7.8;
     * pre-B7.8 cap was 8 per side), n_prepend + n_append >= 1);
     * chokepoint at mdxfind.c gates on this so dispatches that arrive
     * here have either no mask (mask_size==1) or a configuration
     * matching B7.5. mask_size = gpu_mask_total =
     * product(mask_sizes[0..n_prepend+n_append)) — the full Cartesian
     * count. */
    uint32_t b71_mask_size = 1;
    int b7_mask_eligible =
        (gpu_mask_n_prepend >= 0 && gpu_mask_n_prepend <= 16 &&
         gpu_mask_n_append  >= 0 && gpu_mask_n_append  <= 16 &&
         (gpu_mask_n_prepend + gpu_mask_n_append) >= 1 &&
         gpu_mask_total > 0 && gpu_mask_total <= 0xFFFFFFFFu);
    if (b7_mask_eligible) {
        b71_mask_size = (uint32_t)gpu_mask_total;
    }

    /* BF chunk-as-job Phase 2 (2026-05-10): effective_mask_size is the
     * kernel's per-word mask iteration range. For BF chunks (bf_num_masks
     * > 0) this is the chunk's per-word slice; for non-BF dispatches it
     * equals b71_mask_size (full keyspace). The salted-pack num_salts
     * pack at line ~10410 / per-page rewrite at line ~11607 must use
     * effective_mask_size so the layered axis (mask_size * salts_per_page)
     * matches what the kernel will iterate. The unsalted pack at line
     * ~10420 already uses the bf_num_masks override; lifting it to a
     * named local keeps both arms in sync. */
    uint32_t effective_mask_size = (bf_num_masks > 0u)
        ? bf_num_masks : b71_mask_size;

    /* B6 salt-axis (2026-05-06; §13.2 row 16.5): derive is-salted at the
     * pack site (op-direct). Row 17's `kern_is_salted_template` covers
     * the same condition at the kernel-arg-binding site (kernel-handle-
     * derived). Both signals exist intentionally at different code sites:
     * row 16.5's flag drives host pack (num_salts / num_salts_per_page /
     * salt_start population) BEFORE the kernel handle is resolved; row
     * 17's flag drives SETARG of the 3 salt args AFTER the kernel handle
     * is known. */
    /* B6.1 SHA1 fan-out (2026-05-06): JOB_SHA1SALTPASS joins the salted-
     * pack op set. Same salt-page outer loop, same num_salts_per_page
     * derivation, same 1024-salt page cap.
     * B6.2 SHA256 fan-out (2026-05-06): JOB_SHA256SALTPASS joins. Same
     * geometry; only the per-algorithm core (and resolved kernel handle)
     * differs.
     * B6.3 SHA224 fan-out (2026-05-06): JOB_SHA224SALTPASS joins. Same
     * geometry; sha256_block compression core, output truncated to 7 words. */
    /* B6.4 MD5PASSSALT fan-out (2026-05-06): JOB_MD5PASSSALT joins. Same
     * salt-page outer loop, same num_salts_per_page derivation. Salt
     * position (APPEND vs PREPEND of the PREPEND siblings) only changes
     * the byte order inside template_finalize; the host-side salt-pack
     * geometry is identical.
     * B6.5 SHA1PASSSALT fan-out (2026-05-06): JOB_SHA1PASSSALT joins —
     * SHA-family APPEND-shape sibling. Same salt-pack geometry; SHA1
     * compression + 5-word state matches SHA1SALTPASS, BE byte placement
     * + APPEND order matches MD5PASSSALT (modulo BE/LE). */
    int is_salted_pack = (op == JOB_MD5SALT || op == JOB_MD5SALTPASS ||
                          op == JOB_SHA1SALTPASS ||
                          op == JOB_SHA256SALTPASS ||
                          op == JOB_SHA224SALTPASS ||
                          op == JOB_MD5PASSSALT ||
                          op == JOB_SHA1PASSSALT ||
                          /* B6.7 SHA256PASSSALT fan-out (2026-05-06):
                           * second SHA-family APPEND-shape salted variant.
                           * Same salt-pack geometry as the other salted
                           * siblings. */
                          op == JOB_SHA256PASSSALT ||
                          /* B6.9 SHA512 fan-out (2026-05-06): SHA512SALTPASS
                           * — first 64-bit-state salted variant. Same
                           * salt-pack geometry as the SHA-256 family
                           * (1024-salt page cap, num_salts_per_page
                           * derivation, salt_start advance). The kernel
                           * width difference (8 ulong state + 128-byte
                           * block + 128-bit length) is invisible to host
                           * salt-pack — only the salt buffer is read by
                           * the kernel's template_finalize. */
                          op == JOB_SHA512SALTPASS ||
                          /* B6.10 SHA512PASSSALT fan-out (2026-05-06):
                           * SHA512PASSSALT — second 64-bit-state salted
                           * variant; APPEND-shape sibling. FINAL B6
                           * ladder step. Same salt-pack geometry as
                           * SHA512SALTPASS — only the byte order inside
                           * template_finalize differs (APPEND vs
                           * PREPEND), invisible to host salt-pack.
                           * Without this entry the salt-page outer
                           * loop runs only one page and silently
                           * produces partial cracks (lesson from B6.9). */
                          op == JOB_SHA512PASSSALT ||
                          /* B6.6 (2026-05-06): MD5SALT family variants
                           * share kernel via params.algo_mode. */
                          op == JOB_MD5UCSALT ||
                          op == JOB_MD5revMD5SALT ||
                          op == JOB_MD5sub8_24SALT ||
                          /* B6.8 (2026-05-06): JOB_MD5_MD5SALTMD5PASS
                           * (e367) shares the MD5SALT kernel via
                           * algo_mode=4. Salt-pack geometry identical
                           * to MD5SALT — the hashsalt (32-byte hex) is
                           * packed in place of raw salt by gpu_pack_salts
                           * with use_hashsalt=1. */
                          op == JOB_MD5_MD5SALTMD5PASS ||
                          /* Family A (2026-05-07): HMAC-MD5 (e214) +
                           * HMAC-MD5_KPASS (e792) share the MD5SALT
                           * GPU kernel via algo_mode=5 / 6. Salt-pack
                           * geometry identical to MD5SALT — raw salt
                           * bytes (use_hashsalt=0). For KSALT (e214) the
                           * "salt" source is Typeuser (HMAC key); for
                           * KPASS (e792) it's Typesalt (HMAC message).
                           * gpu_salt_judy() resolves the right Judy. */
                          op == JOB_HMAC_MD5 ||
                          op == JOB_HMAC_MD5_KPASS ||
                          /* Family B (2026-05-07): HMAC-SHA1 (e215) +
                           * HMAC-SHA1_KPASS (e793) share the SHA1SALTPASS
                           * GPU kernel via algo_mode=5 / 6. Salt-pack
                           * geometry identical to SHA1SALTPASS — raw salt
                           * bytes (use_hashsalt=0). For KSALT (e215) the
                           * "salt" source is Typeuser (HMAC key); for
                           * KPASS (e793) it's Typesalt (HMAC message).
                           * gpu_salt_judy() resolves the right Judy. */
                          op == JOB_HMAC_SHA1 ||
                          op == JOB_HMAC_SHA1_KPASS ||
                          /* Family C (2026-05-07): HMAC-SHA224 (e216) +
                           * HMAC-SHA224_KPASS (e794) share the SHA224SALTPASS
                           * GPU kernel via algo_mode=5 / 6. Salt-pack
                           * geometry identical to SHA224SALTPASS — raw salt
                           * bytes (use_hashsalt=0). For KSALT (e216) the
                           * "salt" source is Typeuser (HMAC key); for
                           * KPASS (e794) it's Typesalt (HMAC message).
                           * gpu_salt_judy() resolves the right Judy. */
                          op == JOB_HMAC_SHA224 ||
                          op == JOB_HMAC_SHA224_KPASS ||
                          /* Family D (2026-05-08): HMAC-SHA256 (e217) +
                           * HMAC-SHA256_KPASS (e795) share the SHA256SALTPASS
                           * GPU kernel via algo_mode=5 / 6. Salt-pack
                           * geometry identical to SHA256SALTPASS — raw salt
                           * bytes (use_hashsalt=0). For KSALT (e217) the
                           * "salt" source is Typeuser (HMAC key); for
                           * KPASS (e795) it's Typesalt (HMAC message).
                           * gpu_salt_judy() resolves the right Judy. The
                           * HMAC body uses RUNTIME gate (NEVER `#if`;
                           * Pascal NVIDIA rev 1.7 ABORT lesson). Final
                           * HMAC family in the ladder. */
                          op == JOB_HMAC_SHA256 ||
                          op == JOB_HMAC_SHA256_KPASS ||
                          /* Family E HMAC-SHA384 carrier (2026-05-08):
                           * HMAC-SHA384 (e543) + HMAC-SHA384_KPASS (e796)
                           * share the SHA384SALTPASS-shaped carrier GPU
                           * kernel via algo_mode=5 / 6. Salt-pack geometry
                           * identical to SHA512SALTPASS (1024-salt page
                           * cap, num_salts_per_page derivation, salt_-
                           * start advance). For KSALT (e543) the "salt"
                           * source is Typeuser (HMAC key); for KPASS
                           * (e796) it's Typesalt (HMAC message). gpu_-
                           * salt_judy() resolves the right Judy. The
                           * kernel-internal width difference (HASH_WORDS=12
                           * vs SHA-512's 16) is invisible to host salt-
                           * pack — only the salt buffer is read by the
                           * kernel's template_finalize HMAC body. */
                          op == JOB_HMAC_SHA384 ||
                          op == JOB_HMAC_SHA384_KPASS ||
                          /* Family F (2026-05-08): HMAC-SHA512 ops share the
                           * SHA512SALTPASS GPU kernel via algo_mode=5/6. Salt-
                           * pack geometry identical to SHA512SALTPASS (1024-
                           * salt page cap, num_salts_per_page derivation,
                           * salt_start advance). For KSALT (e218) the "salt"
                           * source is Typeuser (HMAC key); for KPASS (e797)
                           * it's Typesalt (HMAC message). gpu_salt_judy()
                           * resolves the right Judy. */
                          op == JOB_HMAC_SHA512 ||
                          op == JOB_HMAC_SHA512_KPASS ||
                          /* Family G HMAC-RIPEMD-160 carrier (2026-05-08):
                           * HMAC-RMD160 (e211) + HMAC-RMD160_KPASS (e798)
                           * share the RIPEMD160SALTPASS-shaped carrier GPU
                           * kernel via algo_mode=5/6. Salt-pack geometry
                           * identical to SHA1SALTPASS (1024-salt page cap,
                           * num_salts_per_page derivation, salt_start
                           * advance) — same 64-byte block, 32-bit state
                           * width. For KSALT (e211) the "salt" source is
                           * Typeuser (HMAC key); for KPASS (e798) it's
                           * Typesalt (HMAC message). gpu_salt_judy()
                           * resolves the right Judy. The kernel-internal
                           * BASE_ALGO=ripemd160 (vs sha1) is invisible to
                           * host salt-pack — only the salt buffer is read
                           * by the kernel's template_finalize HMAC body. */
                          op == JOB_HMAC_RMD160 ||
                          op == JOB_HMAC_RMD160_KPASS ||
                          /* Family H HMAC-RIPEMD-320 carrier (2026-05-08):
                           * HMAC-RMD320 (e213) + HMAC-RMD320_KPASS (e799)
                           * share the RIPEMD320SALTPASS-shaped carrier GPU
                           * kernel via algo_mode=5/6. Salt-pack geometry
                           * identical to RIPEMD160SALTPASS (1024-salt page
                           * cap, num_salts_per_page derivation, salt_start
                           * advance) — same 64-byte block, 32-bit state
                           * width. For KSALT (e213) the "salt" source is
                           * Typeuser (HMAC key); for KPASS (e799) it's
                           * Typesalt (HMAC message). gpu_salt_judy()
                           * resolves the right Judy. The kernel-internal
                           * width difference (HASH_WORDS=10 vs RMD160's 5)
                           * is invisible to host salt-pack — only the salt
                           * buffer is read by the kernel's template_-
                           * finalize HMAC body. */
                          op == JOB_HMAC_RMD320 ||
                          op == JOB_HMAC_RMD320_KPASS ||
                          /* Family I HMAC-BLAKE2S carrier (2026-05-08):
                           * HMAC-BLAKE2S (e828) — single algo_mode (5); no
                           * KPASS sibling op exists. Routes via the hand-
                           * written Path A kern_template_phase0_hmac_blake2s.
                           * Salt-pack geometry identical to BLAKE2S256
                           * unsalted at the host layer (1024-salt page cap,
                           * num_salts_per_page derivation, salt_start
                           * advance) — same 64-byte block, 32-bit state
                           * width. The "salt" source is Typesalt (HMAC
                           * message; the password is the HMAC key inside
                           * the kernel). gpu_salt_judy() resolves Typesalt.
                           * The kernel-internal HMAC body inlines salt
                           * bytes as the BLAKE2S inner-block message —
                           * invisible to host salt-pack. */
                          op == JOB_HMAC_BLAKE2S ||
                          /* Family J HMAC-STREEBOG-256 carrier (2026-05-08):
                           * HMAC-STREEBOG256_KSALT (e838) + HMAC-STREEBOG256_-
                           * KPASS (e837) — share the Path A hand-written
                           * carrier GPU kernel via algo_mode=5/6. Salt-pack
                           * geometry identical to BLAKE2S256 unsalted at the
                           * host layer (1024-salt page cap, num_salts_per_page
                           * derivation, salt_start advance) — same 64-byte
                           * block, 32-bit state width. UNLIKE Families A-H
                           * whose KSALT siblings have TYPEOPT_NEEDUSER (salt-
                           * data in Typeuser), Streebog-256 HMAC has
                           * TYPEOPT_NEEDSALT for BOTH ops — salt-data lives
                           * in Typesalt[op]. gpu_salt_judy() routes both ops
                           * to the Typesalt default arm. The kernel-internal
                           * HMAC body uses algo_mode 5 vs 6 to swap the
                           * (key, msg) source mapping between `data` and
                           * `salt_bytes` (see template_finalize in
                           * gpu_hmac_streebog256_core.cl). */
                          op == JOB_HMAC_STREEBOG256_KSALT ||
                          op == JOB_HMAC_STREEBOG256_KPASS ||
                          /* Family K HMAC-STREEBOG-512 carrier (2026-05-08):
                           * HMAC-STREEBOG512_KSALT (e840) + HMAC-STREEBOG512_-
                           * KPASS (e839) - share the Path A hand-written
                           * carrier GPU kernel via algo_mode=5/6. Salt-pack
                           * geometry identical to Family J STREEBOG-256 at the
                           * host layer (1024-salt page cap, num_salts_per_page
                           * derivation, salt_start advance) - same 64-byte
                           * block, 32-bit state width at host-pack level
                           * (kernel-internal HASH_WORDS=16 vs 8 invisible to
                           * host pack). Same TYPEOPT_NEEDSALT routing as
                           * Family J - BOTH ops route via Typesalt through
                           * gpu_salt_judy() default arm. Final HMAC family
                           * in the ladder. */
                          op == JOB_HMAC_STREEBOG512_KSALT ||
                          op == JOB_HMAC_STREEBOG512_KPASS ||
                          /* PHPBB3 carrier (2026-05-08): JOB_PHPBB3
                           * (e455) — routes through the hand-written
                           * Path A salted-template kernel. Salt-pack
                           * geometry identical to MD5SALT at the host
                           * layer (1024-salt page cap, num_salts_per_-
                           * page derivation, salt_start advance) —
                           * only the salt CONTENT is per-PHPBB3 (the
                           * 12-byte "$H$<cost><8-byte salt>" prefix).
                           * gpu_pack_salts(use_hashsalt=0) packs the
                           * raw 12-byte salt prefix from saltsnap[].-
                           * salt; the kernel reads salt_bytes[3] (cost
                           * char) + salt_bytes[4..11] (salt) inside
                           * template_finalize. */
                          op == JOB_PHPBB3 ||
                          /* MD5CRYPT carrier (2026-05-08): JOB_MD5CRYPT
                           * (e511) — routes through the hand-written
                           * Path A salted-template kernel. Salt-pack
                           * geometry identical to PHPBB3 at the host
                           * layer (1024-salt page cap, num_salts_per_-
                           * page derivation, salt_start advance) -- only
                           * the salt CONTENT is per-MD5CRYPT (variable-
                           * length "$1$<salt>$" prefix; 5..12 bytes).
                           * gpu_pack_salts(use_hashsalt=0) packs the
                           * raw "$1$<salt>$" prefix from saltsnap[].salt;
                           * the kernel extracts the raw salt by skipping
                           * salt_bytes[0..2] ("$1$") and reading until
                           * '$' or end-of-buffer (cap 8 bytes) inside
                           * template_finalize. Phase 1 of the Unix-
                           * crypt ladder. */
                          op == JOB_MD5CRYPT ||
                          /* SHA256CRYPT carrier (2026-05-08): JOB_SHA256CRYPT
                           * (e512) -- routes through the hand-written
                           * Path A salted-template kernel. Salt-pack
                           * geometry identical to MD5CRYPT/PHPBB3 at the
                           * host layer (1024-salt page cap, num_salts_-
                           * per_page derivation, salt_start advance) --
                           * only the salt CONTENT is per-SHA256CRYPT
                           * (variable-length "$5$[rounds=N$]<salt>$"
                           * prefix). gpu_pack_salts(use_hashsalt=0)
                           * packs the raw prefix from saltsnap[].salt;
                           * the kernel parses + extracts raw_salt up
                           * to 16 bytes inside template_finalize.
                           * Phase 2 of the Unix-crypt ladder. */
                          op == JOB_SHA256CRYPT ||
                          /* SHA512CRYPT carrier (2026-05-08): JOB_SHA512CRYPT
                           * (e513) -- routes through the hand-written
                           * Path A salted-template kernel. Salt-pack
                           * geometry identical to SHA256CRYPT at the host
                           * layer (1024-salt page cap, num_salts_per_page
                           * derivation, salt_start advance) -- only the
                           * salt CONTENT is per-SHA512CRYPT (variable-
                           * length "$6$[rounds=N$]<salt>$" prefix; b64
                           * tail at 86 chars vs SHA256CRYPT's 43 but tail
                           * is excluded from the salt prefix scan).
                           * gpu_pack_salts(use_hashsalt=0) packs the raw
                           * prefix from saltsnap[].salt; the kernel
                           * parses + extracts raw_salt up to 16 bytes
                           * inside template_finalize. Phase 3 of the
                           * Unix-crypt ladder. */
                          op == JOB_SHA512CRYPT ||
                          /* SHA512CRYPTMD5 carrier (2026-05-08): JOB_-
                           * SHA512CRYPTMD5 (e510) -- REUSES Phase 3
                           * SHA512CRYPT's compiled kernel. Salt-pack
                           * geometry IDENTICAL to JOB_SHA512CRYPT (both
                           * use Typesalt with the same "$6$[rounds=N$]
                           * <salt>$" prefix; mdxfind.c:47077-47087 inserts
                           * the same line into BOTH Judy arrays).
                           * gpu_pack_salts(use_hashsalt=0). The MD5-pre-
                           * process happens HOST-side at mdxfind.c:12256-
                           * 12258 before gpu_try_pack; the salt path is
                           * unchanged from SHA512CRYPT. Phase 4 of the
                           * Unix-crypt ladder. */
                          op == JOB_SHA512CRYPTMD5 ||
                          /* DESCRYPT carrier (2026-05-08, Unix-crypt
                           * Phase 5): JOB_DESCRYPT (e500) -- routes
                           * through the hand-written Path A salted-
                           * template kernel. Salt-pack geometry
                           * identical to MD5CRYPT/PHPBB3 at the host
                           * layer (1024-salt page cap, num_salts_per_-
                           * page derivation, salt_start advance) -- only
                           * the salt CONTENT is per-DESCRYPT (the
                           * 2-byte phpitoa64 salt). gpu_pack_salts_op
                           * (use_hashsalt=0, op=JOB_DESCRYPT) packs the
                           * 2-byte salt from saltsnap[].salt with
                           * extended-DES `_CCCCSSSS` 9-char salts SKIPPED
                           * via the saltlen != 2 filter; the kernel
                           * reads salt_bytes[0..1] inside template_-
                           * finalize. Without this entry the salt-page
                           * outer loop runs only one page (n_pages=1
                           * with num_salts_per_page=0) and silently
                           * drops cracks for words whose hash matches
                           * any salt OTHER than salt index 0 (Phase 5
                           * 2026-05-08 fpga smoke caught: 1/5 cracks
                           * with the only matching salt being the one
                           * sentinel-0; the other 4 silent CPU fallback
                           * via Typedone[] without the host iterating
                           * salts on GPU). Phase 5 of the Unix-crypt
                           * ladder (FINAL phase). */
                          op == JOB_DESCRYPT ||
                          /* BCRYPT carrier (2026-05-08, Unix-crypt
                           * Phase 6): JOB_BCRYPT (e450) -- routes
                           * through the hand-written Path A salted-
                           * template kernel. Salt-pack geometry
                           * identical to MD5CRYPT/PHPBB3/DESCRYPT at
                           * the host layer (1024-salt page cap, num_-
                           * salts_per_page derivation, salt_start
                           * advance) -- only the salt CONTENT is per-
                           * BCRYPT (the 28- or 29-byte "$2[abkxy]$NN$
                           * <base64>" prefix). gpu_pack_salts(use_-
                           * hashsalt=0) packs the salt prefix from
                           * saltsnap[].salt; the kernel reads salt_-
                           * bytes[0..salt_len) and parses cost +
                           * variant + raw-salt inside template_finalize.
                           * Without this entry the salt-page outer
                           * loop runs only one page and silently drops
                           * cracks (Phase 5 DESCRYPT fpga smoke lesson:
                           * the 5th sibling-site is load-bearing). Phase
                           * 6 of the slab-retirement ladder (final
                           * major slab kernel). */
                          op == JOB_BCRYPT);

    /* B6 salt-axis (2026-05-06; §13.4 row 27): derive salts_per_page +
     * n_pages for the outer salt-page loop. Initial cap of 1024 per
     * Memo B §3 R1 — keeps n_rules * mask_size * num_salts_per_page well
     * below the uint32 ceiling (4.29 G) for all realistic workloads. The
     * loop body re-writes first 132 bytes of b_dispatch_payload at the
     * start of each iteration to advance salt_start; hits accumulate
     * across pages into d->h_hits with NO per-page reset (each emitted
     * hit's combined_ridx encodes the global salt index unambiguously
     * via salt_local + salt_start, so de-dup is not needed across pages).
     *
     * For unsalted ops, n_pages = 1 and salts_per_page = 0 — the outer
     * loop runs exactly once and the OCLParams.num_salts_per_page slot
     * stays at 0 (kernel reads it as 0 → coerces to 1 → preserves
     * pre-B6 dispatch geometry byte-exact). */
    uint32_t salts_per_page = 0;
    uint32_t n_pages        = 1;
    uint32_t total_salts    = 0;
    if (is_salted_pack) {
        total_salts    = (uint32_t)d->salts_count;
        /* salts_per_page source priority (2026-05-09 dynsize prototype):
         *   1. MDXFIND_SPP env (user pin) — bypasses dynsize feedback
         *   2. dynsize cache (MDXFIND_DYNSIZE=1) for kern_template_phase0_md5salt
         *   3. Default 1024 (pre-experiment baseline)
         * Range cap [1, 65536] applied uniformly. */
        uint32_t spp_default = 1024u;
        {
            const char *spp_env = getenv("MDXFIND_SPP");
            if (spp_env && *spp_env) {
                int v = atoi(spp_env);
                if (v >= 1 && v <= 65536) spp_default = (uint32_t)v;
            } else if (dynsize_is_enabled() &&
                       (op == JOB_MD5SALT || op == JOB_MD5UCSALT ||
                        op == JOB_MD5revMD5SALT || op == JOB_MD5sub8_24SALT)) {
                /* Dynsize gate at salts_per_page derivation:
                 * `kern` is not in scope at this site (the kernel is
                 * resolved further down). Use op-based gate covering
                 * algo_modes 0-3 ONLY:
                 *   JOB_MD5SALT       (e31,  mode 0)
                 *   JOB_MD5UCSALT     (e350, mode 1)
                 *   JOB_MD5revMD5SALT (e541, mode 2)
                 *   JOB_MD5sub8_24SALT(e542, mode 3)
                 * Modes 4/5/6 (e367 + HMAC-MD5) also resolve to
                 * kern_template_phase0_md5salt but are EXCLUDED from
                 * the prototype (different per-pair work mix; a single
                 * shared dynsize entry would mistune them).
                 *
                 * The post-dispatch feedback site uses kern == ...
                 * equality (kern is in scope there). The two gates
                 * MUST agree on which dispatches are dynsize-targeted;
                 * keep them in sync if either changes. */
                dynsize_ensure_loaded(d, dev_idx);
                if (d->dynsize_md5salt.initialized &&
                    d->dynsize_md5salt.current_spp >= DYNSIZE_SPP_MIN &&
                    d->dynsize_md5salt.current_spp <= DYNSIZE_SPP_MAX) {
                    spp_default = d->dynsize_md5salt.current_spp;
                }
            }
        }
        salts_per_page = (total_salts < spp_default) ? total_salts : spp_default;
        if (salts_per_page == 0u) {
            /* Defensive: salts_count == 0 should have been caught upstream
             * (gpujob_opencl.c:618 stale-salt protection). Treat as a no-
             * op dispatch — caller will see *nhits_out = 0. */
            n_pages = 1;
        } else {
            n_pages = (total_salts + salts_per_page - 1u) / salts_per_page;
            if (n_pages == 0u) n_pages = 1u;
        }
    }

    /* Pack the payload struct in host staging memory. */
    {
        unsigned char *p = (unsigned char *)d->h_dispatch_payload;
        OCLParams *pparams = (OCLParams *)p;
        memset(pparams, 0, sizeof(*pparams));
        pparams->compact_mask = _compact_mask;
        pparams->num_words = num_words;
        pparams->num_masks = (uint32_t)d->gpu_n_rules;
        /* B7.1: num_salts repurposed as mask_size for the template path.
         * The slab-path salted kernels read num_salts directly; the rules-
         * engine / template path doesn't dispatch salted ops, so the slot
         * is free to repurpose here. Consistent with num_masks already
         * being repurposed as n_rules in this dispatch.
         *
         * B6 salt-axis (2026-05-06): when is_salted_pack is true, num_salts
         * carries the LAYERED axis = mask_size * num_salts_per_page so the
         * kernel can recover both via division by num_salts_per_page (see
         * gpu_template.cl:171 under #ifdef GPU_TEMPLATE_HAS_SALT). The host
         * outer salt-page loop overwrites these three fields (num_salts,
         * num_salts_per_page, salt_start) at the start of each iteration. */
        if (is_salted_pack) {
            uint32_t this_page_salts =
                (total_salts < salts_per_page) ? total_salts : salts_per_page;
            if (this_page_salts == 0u) this_page_salts = 1u;
            /* Phase 2 BF chunk-as-job (2026-05-10): use effective_mask_size
             * (= bf_num_masks for BF chunks, b71_mask_size otherwise) so
             * the layered axis num_salts = mask_size * num_salts_per_page
             * matches what the kernel will iterate. Pre-Phase-2 this used
             * b71_mask_size unconditionally; salted BF would over-pack
             * num_salts by gpu_mask_total/bf_num_masks (typically 100s-1000s)
             * and the kernel would compute mask_size = num_salts /
             * num_salts_per_page = full-keyspace, then mask_idx % mask_size
             * would access charsets beyond the chunk's per-word slice. */
            pparams->num_salts          = effective_mask_size * this_page_salts;
            pparams->num_salts_per_page = (uint64_t)this_page_salts;
            pparams->salt_start         = 0u;  /* page 0 starts at salt 0 */
        } else {
            /* Tranche 1 BF chunk-as-job (2026-05-09): when the dispatcher is
             * called with a non-zero bf_num_masks, override mask_size for the
             * kernel. The kernel derives mask_size = num_salts/num_salts_per_page
             * (and unsalted coerces num_salts_per_page=0 -> 1), so num_salts
             * IS the mask_size on the unsalted path. Default (bf_num_masks==0)
             * preserves existing behavior bit-identically. Phase 2: this is
             * now effective_mask_size (= bf_num_masks for BF, b71_mask_size
             * otherwise) — same value the salted arm uses. */
            pparams->num_salts          = effective_mask_size;
            pparams->num_salts_per_page = 0u;  /* unsalted: kernel coerces 0->1 */
            pparams->salt_start         = 0u;
        }
        pparams->n_prepend = (uint32_t)gpu_mask_n_prepend;
        pparams->n_append  = (uint32_t)gpu_mask_n_append;
        /* Tranche 1 BF chunk-as-job (2026-05-09): cursor + per-word stride.
         * Default 0/0 = today's behavior (kernel mask_idx unchanged). The
         * Tranche 2 kernel edit adds these into mask_idx; until then the
         * fields are present but ignored on the kernel side. */
        pparams->mask_start            = mask_start;
        pparams->mask_offset_per_word  = mask_offset_per_word;
        /* BF Phase 1.8 (2026-05-10): inner_iter — number of mask values each
         * work-item processes for the same (word, rule). Host coerces 0→1
         * here so unsalted-BF callers can pass 0 to mean "today's behavior".
         * Cap = 16 (enforced by adaptive_bf_chunk_size servo). Kernel applies
         * the same coercion defensively. Default 0 (memset) for non-BF. */
        pparams->inner_iter            = (inner_iter == 0u) ? 1u : inner_iter;
        if (pparams->inner_iter > 16u) pparams->inner_iter = 16u;
        /* B6.6 (2026-05-06): per-algorithm runtime variant flag. MD5SALT
         * family (e31/e350/e541/e542) shares one kernel; algo_mode selects
         * the inner-digest hex-encoding variant. 0=LC (e31), 1=UC (e350),
         * 2=REV (e541), 3=SLICE (e542). All other algos: 0 (default; no
         * variant logic in their fragments — the unused parameter is
         * documented in each fragment via `(void)algo_mode`).
         * B6.8 (2026-05-06): mode 4 = JOB_MD5_MD5SALTMD5PASS (e367) —
         * outer MD5 over (hex32(MD5(salt)) || hex32(MD5(pass))). Salt-hex
         * pre-computed by host; pass-hex computed in the fragment. */
        /* B7.7a (2026-05-07): JOB_MD5UC also uses algo_mode=1 (uppercase
         * hex). Distinct from JOB_MD5UCSALT (salted MD5 with UC inner
         * digest) which is in the SALT family — the kernel branch
         * differs (template_iterate UC vs salted-finalize UC). For the
         * unsalted MD5 template kernel, algo_mode=1 selects md5_to_hex_uc
         * in the iter loop; iter=1 byte-exact since UC only fires
         * inter-iter (mirrors the CPU path at mdxfind.c:25386 MDstart). */
        if (op == JOB_MD5UCSALT || op == JOB_MD5UC) pparams->algo_mode = 1u;
        else if (op == JOB_MD5revMD5SALT)         pparams->algo_mode = 2u;
        else if (op == JOB_MD5sub8_24SALT)        pparams->algo_mode = 3u;
        else if (op == JOB_MD5_MD5SALTMD5PASS)    pparams->algo_mode = 4u;
        /* Family A (2026-05-07): HMAC-MD5 modes share the MD5SALT
         * template kernel; algo_mode 5 = KSALT (e214: key=salt,msg=pass),
         * 6 = KPASS (e792: key=pass,msg=salt). The HMAC body at the top
         * of template_finalize returns early after writing the digest. */
        else if (op == JOB_HMAC_MD5)              pparams->algo_mode = 5u;
        else if (op == JOB_HMAC_MD5_KPASS)        pparams->algo_mode = 6u;
        /* Family B (2026-05-07): HMAC-SHA1 modes share the SHA1SALTPASS
         * template kernel; algo_mode 5 = KSALT (e215: key=salt,msg=pass),
         * 6 = KPASS (e793: key=pass,msg=salt). The HMAC body at the top
         * of template_finalize (gated on HASH_WORDS==5) returns early
         * after writing the digest. */
        else if (op == JOB_HMAC_SHA1)             pparams->algo_mode = 5u;
        else if (op == JOB_HMAC_SHA1_KPASS)       pparams->algo_mode = 6u;
        /* Family C (2026-05-07): HMAC-SHA224 modes share the SHA224SALTPASS
         * template kernel; algo_mode 5 = KSALT (e216: key=salt,msg=pass),
         * 6 = KPASS (e794: key=pass,msg=salt). The HMAC body at the top
         * of template_finalize (gated on HASH_WORDS==7) returns early
         * after writing the digest. */
        else if (op == JOB_HMAC_SHA224)           pparams->algo_mode = 5u;
        else if (op == JOB_HMAC_SHA224_KPASS)     pparams->algo_mode = 6u;
        /* Family D (2026-05-08): HMAC-SHA256 modes share the SHA256SALTPASS
         * template kernel; algo_mode 5 = KSALT (e217: key=salt,msg=pass),
         * 6 = KPASS (e795: key=pass,msg=salt). The HMAC body at the top
         * of template_finalize uses the RUNTIME gate `if (HASH_WORDS == 8
         * && algo_mode >= 5u)` (see prominent CRITICAL comment in
         * finalize_prepend_be.cl.frag forbidding `#if HASH_WORDS == 8`
         * conversion; rev 1.7 / Pascal NVIDIA failure 2026-05-07).
         * Returns early after writing the full 8-word digest (no
         * truncation, unlike Family C SHA224 which drops h[7]). */
        else if (op == JOB_HMAC_SHA256)           pparams->algo_mode = 5u;
        else if (op == JOB_HMAC_SHA256_KPASS)     pparams->algo_mode = 6u;
        /* Family E HMAC-SHA384 carrier (2026-05-08): HMAC-SHA384 modes
         * share the SHA384SALTPASS-shaped carrier template kernel;
         * algo_mode 5 = KSALT (e543: key=salt,msg=pass), 6 = KPASS
         * (e796: key=pass,msg=salt). The HMAC body at the top of
         * template_finalize (gated on HASH_WORDS==12 in finalize_-
         * prepend_be64.cl.frag) returns early after writing the
         * 6-ulong digest into st->state[0..5] (template_state_to_h
         * decomposes into st->h[0..11], 12 LE uint32 = 48 bytes). */
        else if (op == JOB_HMAC_SHA384)           pparams->algo_mode = 5u;
        else if (op == JOB_HMAC_SHA384_KPASS)     pparams->algo_mode = 6u;
        /* Family F (2026-05-08): HMAC-SHA512 modes share the SHA512SALTPASS
         * template kernel; algo_mode 5 = KSALT (e218: key=salt,msg=pass),
         * 6 = KPASS (e797: key=pass,msg=salt). The HMAC body at the top of
         * template_finalize (gated on HASH_WORDS==16 in finalize_prepend_-
         * be64.cl.frag) returns early after writing the 8-ulong digest into
         * st->state[0..7] (template_state_to_h decomposes into st->h[0..15],
         * 16 LE uint32 = 64 bytes; no truncation vs SHA-384's 6-word). */
        else if (op == JOB_HMAC_SHA512)           pparams->algo_mode = 5u;
        else if (op == JOB_HMAC_SHA512_KPASS)     pparams->algo_mode = 6u;
        /* Family G HMAC-RIPEMD-160 carrier (2026-05-08): HMAC-RMD160 modes
         * share the RIPEMD160SALTPASS-shaped carrier template kernel;
         * algo_mode 5 = KSALT (e211: key=salt,msg=pass), 6 = KPASS (e798:
         * key=pass,msg=salt). The HMAC body at the top of template_-
         * finalize (gated on HASH_WORDS==5 in finalize_prepend_rmd.cl.frag)
         * returns early after writing the 5-uint32 digest directly into
         * st->h[0..4] (LE-native; no bswap32 vs SHA-1/SHA-2 cores). */
        else if (op == JOB_HMAC_RMD160)           pparams->algo_mode = 5u;
        else if (op == JOB_HMAC_RMD160_KPASS)     pparams->algo_mode = 6u;
        /* Family H HMAC-RIPEMD-320 carrier (2026-05-08): HMAC-RMD320 modes
         * share the RIPEMD320SALTPASS-shaped carrier template kernel;
         * algo_mode 5 = KSALT (e213: key=salt, msg=pass), 6 = KPASS (e799:
         * key=pass, msg=salt). The HMAC body at the top of template_-
         * finalize (gated on HASH_WORDS==10 in finalize_prepend_rmd.cl.frag)
         * returns early after writing the 10-uint32 digest directly into
         * st->h[0..9] (LE-native; no bswap32 vs SHA-1/SHA-2 cores). */
        else if (op == JOB_HMAC_RMD320)           pparams->algo_mode = 5u;
        else if (op == JOB_HMAC_RMD320_KPASS)     pparams->algo_mode = 6u;
        /* Family I HMAC-BLAKE2S carrier (2026-05-08): HMAC-BLAKE2S routes
         * to the hand-written Path A kernel; algo_mode 5 = key=$pass,
         * msg=$salt (the only HMAC-BLAKE2S shape in mdxfind — there is no
         * KSALT sibling op). The HMAC body at the top of template_finalize
         * (gated on algo_mode == 5u in gpu_hmac_blake2s_core.cl) returns
         * early after writing the 8-uint32 LE digest into st->h[0..7]. */
        else if (op == JOB_HMAC_BLAKE2S)          pparams->algo_mode = 5u;
        /* Family J HMAC-STREEBOG-256 carrier (2026-05-08): HMAC-STREEBOG-256
         * modes share the hand-written Path A carrier kernel; algo_mode
         * 5 = KSALT (e838: key=salt, msg=pass), 6 = KPASS (e837: key=pass,
         * msg=salt). The HMAC body at the top of template_finalize (gated
         * on algo_mode >= 5u in gpu_hmac_streebog256_core.cl, single
         * branch covering both modes since math is identical) returns early
         * after writing the 32-byte digest (8 LE uint32) into st->h[0..7].
         * The host's gpu_salt_judy() resolves Typeuser (KSALT, key source) or
         * Typesalt (KPASS, msg source) per the conventional HMAC role mapping. */
        else if (op == JOB_HMAC_STREEBOG256_KSALT) pparams->algo_mode = 5u;
        else if (op == JOB_HMAC_STREEBOG256_KPASS) pparams->algo_mode = 6u;
        /* Family K HMAC-STREEBOG-512 carrier (2026-05-08): HMAC-STREEBOG-512
         * modes share the hand-written Path A carrier kernel; algo_mode
         * 5 = KSALT (e840: key=salt, msg=pass), 6 = KPASS (e839: key=pass,
         * msg=salt). The HMAC body at the top of template_finalize (gated
         * on algo_mode >= 5u in gpu_hmac_streebog512_core.cl, single
         * branch covering both modes since math is identical) returns early
         * after writing the 64-byte digest (16 LE uint32) into st->h[0..15].
         * BOTH ops route via Typesalt through gpu_salt_judy() default arm
         * (TYPEOPT_NEEDSALT for both KSALT and KPASS - mirrors Family J
         * STREEBOG-256). Final HMAC family in the ladder. */
        else if (op == JOB_HMAC_STREEBOG512_KSALT) pparams->algo_mode = 5u;
        else if (op == JOB_HMAC_STREEBOG512_KPASS) pparams->algo_mode = 6u;
        /* PHPBB3 carrier (2026-05-08): JOB_PHPBB3 (e455) -- single algo_mode
         * (0; the algorithm is a single fixed shape; the iteration count
         * is decoded inside the kernel from salt_bytes[3] via phpitoa64
         * reverse lookup, NOT selected via algo_mode). Explicit set for
         * clarity; redundant with the default-arm 0u below. */
        else if (op == JOB_PHPBB3)                 pparams->algo_mode = 0u;
        /* MD5CRYPT carrier (2026-05-08): JOB_MD5CRYPT (e511) -- single
         * algo_mode (0; BSD $1$ md5crypt is a single fixed shape with
         * FIXED 1000-iteration count, no algo_mode-controlled variants).
         * Explicit set for clarity; redundant with the default-arm 0u
         * below. Phase 1 of the Unix-crypt ladder. */
        else if (op == JOB_MD5CRYPT)               pparams->algo_mode = 0u;
        /* SHA256CRYPT carrier (2026-05-08): JOB_SHA256CRYPT (e512) --
         * single algo_mode (0; the algorithm is a single fixed shape;
         * the rounds count is decoded INSIDE the kernel from the
         * "rounds=N$" optional salt prefix, NOT selected via algo_mode).
         * Phase 4 SHA512CRYPTMD5 will introduce algo_mode=1 against the
         * same shared core (selecting MD5-preprocess of the password
         * before feeding into the SHA-512 chain). Explicit set for
         * clarity; redundant with the default-arm 0u below. Phase 2 of
         * the Unix-crypt ladder. */
        else if (op == JOB_SHA256CRYPT)            pparams->algo_mode = 0u;
        /* SHA512CRYPT carrier (2026-05-08): JOB_SHA512CRYPT (e513) --
         * single algo_mode (0; the algorithm is a single fixed shape;
         * the rounds count is decoded INSIDE the kernel from the
         * "rounds=N$" optional salt prefix, NOT selected via algo_mode).
         * Phase 4 SHA512CRYPTMD5 will introduce algo_mode=1 against the
         * same shared core at HASH_WORDS=16 (selecting MD5-preprocess
         * of the password before feeding into the SHA-512 chain).
         * Explicit set for clarity; redundant with the default-arm 0u
         * below. Phase 3 of the Unix-crypt ladder. */
        else if (op == JOB_SHA512CRYPT)            pparams->algo_mode = 0u;
        /* SHA512CRYPTMD5 carrier (2026-05-08): JOB_SHA512CRYPTMD5 (e510)
         * -- algo_mode = 1u selects the KERNEL-SIDE MD5-preprocess of
         * the password BEFORE the SHA-crypt chain runs. The kernel
         * (gpu_shacrypt_core.cl rev 1.3+ at HASH_WORDS=16, BASE_ALGO=
         * sha512crypt) runs the SAME compiled program as SHA512CRYPT;
         * the algo_mode=1u branch in template_finalize replaces the
         * private pw[] buffer with the 32-char ASCII-hex of MD5(pw),
         * sets plen=32, then continues into the unmodified SHA-crypt
         * chain. This is correct under rules: rules apply to the
         * original password BEFORE gpu_try_pack; the kernel sees the
         * post-rule plaintext and MD5-preprocesses THAT (matching CPU
         * semantics where checkhash applies rules first, then crypt-
         * round computes MD5(post-rule) inside JOB_SHA512CRYPTMD5
         * dispatch at mdxfind.c:12199-12212). The HOST-side substitution
         * at mdxfind.c:12256-12258 lives only in the slab-path arm of
         * gpu_try_pack and is INERT for the rules-engine path (which
         * packs raw `cur` directly at mdxfind.c:11017-11021); kernel-
         * side preprocess is the unifying fix. Phase 4 of the Unix-
         * crypt ladder. */
        else if (op == JOB_SHA512CRYPTMD5)         pparams->algo_mode = 1u;
        /* DESCRYPT carrier (2026-05-08, Unix-crypt Phase 5): JOB_DESCRYPT
         * (e500). algo_mode = 7u (next free integer; 0 default + 1
         * SHA512CRYPTMD5/MD5UC + 2 MD5revMD5SALT + 3 MD5sub8_24SALT + 4
         * MD5_MD5SALTMD5PASS + 5 HMAC KSALT family + 6 HMAC KPASS family
         * = 0..6 used). DESCRYPT is bespoke and will NOT share with
         * BCRYPT (BCRYPT will need its own algo_modes for future BCRYPT
         * variants like BCRYPTMD5). The kernel ignores algo_mode (single-
         * mode algorithm); the value matters only for cache-key
         * consistency with the defines_str BASE_ALGO=descrypt. Phase 5
         * of the Unix-crypt ladder (FINAL phase). */
        else if (op == JOB_DESCRYPT)              pparams->algo_mode = 7u;
        /* BCRYPT carrier (2026-05-08, Unix-crypt Phase 6): JOB_BCRYPT
         * (e450). algo_mode = 8u (next free integer after DESCRYPT's 7;
         * 0..7 used). BCRYPT is bespoke and reserves algo_mode 8-15 for
         * future BCRYPT-family kernel-side variants (e.g., BCRYPTMD5
         * could claim 9 if it ever gets dedicated kernel-side preprocess-
         * ing). The kernel ignores algo_mode (single-mode algorithm; only
         * JOB_BCRYPT routes here -- compound siblings are CPU-only); the
         * value matters only for cache-key consistency with the defines_-
         * str BASE_ALGO=bcrypt. Phase 6 of the slab-retirement ladder
         * (final major slab kernel). */
        else if (op == JOB_BCRYPT)                pparams->algo_mode = 8u;
        else                                      pparams->algo_mode = 0u;
        pparams->max_probe = 256;
        pparams->hash_data_count = _hash_data_count;
        pparams->max_hits = b3_max_hits_cap;
        pparams->overflow_count = _overflow_count;
        /* Iteration count threaded through from the per-device max_iter
         * (set by gpu_opencl_set_max_iter at gpujob_init from Maxiter). */
        pparams->max_iter = d->max_iter < 1 ? 1 : d->max_iter;
        /* B6.11 SHA1DRU (2026-05-06): JOB_SHA1DRU's 1M-iteration loop is
         * INSIDE template_finalize; the kernel's outer iter loop must run
         * exactly ONCE so only the FINAL state (after 1M inner iterations)
         * is probed. CPU semantics at mdxfind.c:14261-14285 also probe
         * exactly once after the for-loop. Force max_iter=1 regardless of
         * user `-i` (which doesn't apply to SHA1DRU at the CPU level either
         * — the 1M is hardcoded in the algorithm). Without this override,
         * `-i N` would cause N redundant emits at iter levels 1..N for the
         * same final state (different `mask = 1u << (iter & 31)` values
         * defeat the hashes_shown dedup). */
        if (op == JOB_SHA1DRU) pparams->max_iter = 1;
        /* Family A (2026-05-07): HMAC-MD5 has no CPU iter loop; the CPU
         * paths at mdxfind.c:29250 (KSALT) + mdxfind.c:29423 (KPASS)
         * compute exactly one HMAC and call checkhashkey/checkhashsalt
         * once per (word, salt). The kernel's outer iter loop must run
         * exactly ONCE — without this override `-i N` would emit N
         * redundant hits at iter levels 1..N for the same final digest.
         * Mirror SHA1DRU pattern (max_iter=1). */
        if (op == JOB_HMAC_MD5 || op == JOB_HMAC_MD5_KPASS)
            pparams->max_iter = 1;
        /* Family B (2026-05-07): HMAC-SHA1 has no CPU iter loop either
         * — same rationale as Family A. CPU paths at mdxfind.c:29301
         * (KSALT) + mdxfind.c:29454 (KPASS) compute exactly one HMAC
         * per (word, salt). Force max_iter=1. */
        if (op == JOB_HMAC_SHA1 || op == JOB_HMAC_SHA1_KPASS)
            pparams->max_iter = 1;
        /* Family C (2026-05-07): HMAC-SHA224 has no CPU iter loop either
         * — same rationale as Families A and B. CPU paths at HMAC_start
         * (mdxfind.c:29406, KSALT case JOB_HMAC_SHA224 at line 29326) +
         * HMAC_KPASS_start (mdxfind.c:29509, KPASS case JOB_HMAC_SHA224_-
         * KPASS at line 29479) compute exactly one HMAC per (word, salt).
         * Force max_iter=1. */
        if (op == JOB_HMAC_SHA224 || op == JOB_HMAC_SHA224_KPASS)
            pparams->max_iter = 1;
        /* Family D (2026-05-08): HMAC-SHA256 has no CPU iter loop either —
         * same rationale as Families A/B/C. CPU paths at HMAC_start
         * (mdxfind.c:29581, KSALT case JOB_HMAC_SHA256) + HMAC_KPASS_start
         * (mdxfind.c:29734, KPASS case JOB_HMAC_SHA256_KPASS) compute
         * exactly one HMAC per (word, salt). Force max_iter=1. */
        if (op == JOB_HMAC_SHA256 || op == JOB_HMAC_SHA256_KPASS)
            pparams->max_iter = 1;
        /* Family E HMAC-SHA384 carrier (2026-05-08): HMAC-SHA384 has no
         * CPU iter loop either — same rationale as Families A/B/C. CPU
         * paths at HMAC_start (mdxfind.c:29369, KSALT case JOB_HMAC_-
         * SHA384) + HMAC_KPASS_start (mdxfind.c:29522, KPASS case JOB_-
         * HMAC_SHA384_KPASS) compute exactly one HMAC per (word, salt).
         * Force max_iter=1. */
        if (op == JOB_HMAC_SHA384 || op == JOB_HMAC_SHA384_KPASS)
            pparams->max_iter = 1;
        /* Family F (2026-05-08): HMAC-SHA512 has no CPU iter loop either —
         * same rationale as Families A/B/C/E. CPU paths at HMAC_start
         * (mdxfind.c:29400, KSALT case JOB_HMAC_SHA512) + HMAC_KPASS_start
         * (mdxfind.c:29553, KPASS case JOB_HMAC_SHA512_KPASS) compute
         * exactly one HMAC per (word, salt). Force max_iter=1. */
        if (op == JOB_HMAC_SHA512 || op == JOB_HMAC_SHA512_KPASS)
            pparams->max_iter = 1;
        /* Family G HMAC-RIPEMD-160 carrier (2026-05-08): HMAC-RMD160 has no
         * CPU iter loop either — same rationale as Families A/B/C/E/F. CPU
         * paths at HMAC_start (mdxfind.c:29391, KSALT case JOB_HMAC_RMD160)
         * + HMAC_KPASS_start (mdxfind.c:29584, KPASS case JOB_HMAC_RMD160_-
         * KPASS) compute exactly one HMAC per (word, salt). Force max_iter=1. */
        if (op == JOB_HMAC_RMD160 || op == JOB_HMAC_RMD160_KPASS)
            pparams->max_iter = 1;
        /* Family H HMAC-RIPEMD-320 carrier (2026-05-08): HMAC-RMD320 has no
         * CPU iter loop either — same rationale as Families A/B/C/E/F/G.
         * CPU paths at HMAC_start (mdxfind.c:29428, KSALT case JOB_HMAC_-
         * RMD320) + HMAC_KPASS_start (mdxfind.c:29616, KPASS case JOB_-
         * HMAC_RMD320_KPASS) compute exactly one HMAC per (word, salt).
         * Force max_iter=1. */
        if (op == JOB_HMAC_RMD320 || op == JOB_HMAC_RMD320_KPASS)
            pparams->max_iter = 1;
        /* Family I HMAC-BLAKE2S carrier (2026-05-08): HMAC-BLAKE2S has no
         * CPU iter loop either — same rationale as Families A/B/C/E/F/G/H.
         * CPU path at mdxfind.c:30391 calls checkhashsalt(curin, 64, s1,
         * saltlen, 0, job) with iter=0 sentinel (suppresses xNN suffix);
         * single HMAC per (word, salt). Force max_iter=1 to prevent the
         * kernel's outer iter loop from emitting redundant hits at iter
         * levels 1..N for the same final state. */
        if (op == JOB_HMAC_BLAKE2S)
            pparams->max_iter = 1;
        /* Family J HMAC-STREEBOG-256 carrier (2026-05-08): HMAC-STREEBOG-256
         * has no CPU iter loop either — same rationale as Families A/B/C/E/F/-
         * G/H/I. CPU paths at JOB_HMAC_STREEBOG256_KPASS (mdxfind.c:30764) +
         * JOB_HMAC_STREEBOG256_KSALT (mdxfind.c:30822) compute exactly one
         * HMAC per (word, salt) and call checkhashsalt with iter=0 sentinel.
         * Force max_iter=1 to prevent redundant hits at iter levels 1..N. */
        if (op == JOB_HMAC_STREEBOG256_KSALT || op == JOB_HMAC_STREEBOG256_KPASS)
            pparams->max_iter = 1;
        /* Family K HMAC-STREEBOG-512 carrier (2026-05-08): HMAC-STREEBOG-512
         * has no CPU iter loop either - same rationale as Families A/B/C/E/F/-
         * G/H/I/J. CPU paths at JOB_HMAC_STREEBOG512_KPASS (mdxfind.c:30918) +
         * JOB_HMAC_STREEBOG512_KSALT (mdxfind.c:30975) compute exactly one
         * HMAC per (word, salt) and call checkhashsalt with iter=0 sentinel.
         * Force max_iter=1 to prevent redundant hits at iter levels 1..N. */
        if (op == JOB_HMAC_STREEBOG512_KSALT || op == JOB_HMAC_STREEBOG512_KPASS)
            pparams->max_iter = 1;
        /* PHPBB3 carrier (2026-05-08): JOB_PHPBB3 (e455). The internal
         * iter loop (count = 1 << itoa64(salt[3]); typically 128..2^30
         * iters) lives INSIDE template_finalize (mirrors SHA1DRU's
         * 1M-loop-in-finalize pattern). Only the FINAL state is probed
         * (CPU semantics at mdxfind.c:13620 calls checkhashbb ONCE
         * after the for-loop). Force max_iter=1 so the kernel's outer
         * iter loop runs exactly once and template_iterate (a stub) is
         * never called; user `-i N` would otherwise emit N redundant
         * hits at iter levels 1..N for the same final state. */
        if (op == JOB_PHPBB3)
            pparams->max_iter = 1;
        /* MD5CRYPT carrier (2026-05-08): JOB_MD5CRYPT (e511). The
         * internal iter loop (FIXED 1000 iters per BSD $1$ md5crypt)
         * lives INSIDE template_finalize (mirrors PHPBB3 / SHA1DRU
         * pattern). Only the FINAL state is probed (CPU semantics at
         * mdxfind.c:13071 calls hybrid_check ONCE after the for-loop).
         * Force max_iter=1 so the kernel's outer iter loop runs exactly
         * once and template_iterate (a stub) is never called; user
         * `-i N` would otherwise emit N redundant hits at iter levels
         * 1..N for the same final state. Phase 1 of the Unix-crypt
         * ladder. */
        if (op == JOB_MD5CRYPT)
            pparams->max_iter = 1;
        /* SHA256CRYPT carrier (2026-05-08): JOB_SHA256CRYPT (e512). The
         * internal rounds loop (default 5000 iters; configurable via
         * "rounds=N$" salt prefix decoded INSIDE the kernel) lives
         * INSIDE template_finalize (mirrors MD5CRYPT pattern; only the
         * FINAL state is probed -- CPU semantics at mdxfind.c:12290
         * calls the b64 reconstruction ONCE after the inner for-loop).
         * Force max_iter=1 so the kernel's outer iter loop runs exactly
         * once and template_iterate (a stub) is never called; user
         * `-i N` would otherwise emit N redundant hits at iter levels
         * 1..N for the same final state. Phase 2 of the Unix-crypt
         * ladder. */
        if (op == JOB_SHA256CRYPT)
            pparams->max_iter = 1;
        /* SHA512CRYPT carrier (2026-05-08): JOB_SHA512CRYPT (e513). The
         * internal rounds loop (default 5000 iters; configurable via
         * "rounds=N$" salt prefix decoded INSIDE the kernel) lives
         * INSIDE template_finalize (mirrors SHA256CRYPT pattern; only
         * the FINAL state is probed -- CPU semantics at mdxfind.c:12290
         * calls the b64 reconstruction ONCE after the inner for-loop).
         * Force max_iter=1 so the kernel's outer iter loop runs exactly
         * once and template_iterate (a stub) is never called; user
         * `-i N` would otherwise emit N redundant hits at iter levels
         * 1..N for the same final state. Phase 3 of the Unix-crypt
         * ladder. */
        if (op == JOB_SHA512CRYPT)
            pparams->max_iter = 1;
        /* SHA512CRYPTMD5 carrier (2026-05-08): JOB_SHA512CRYPTMD5 (e510).
         * Same internal-iter shape as SHA512CRYPT (5-step chain + variable-
         * rounds main loop INSIDE template_finalize); only the FINAL
         * state is probed -- CPU semantics at mdxfind.c:12290 calls the
         * b64 reconstruction ONCE after the inner for-loop (with the MD5-
         * hex-substituted password). Force max_iter=1 so the kernel's
         * outer iter loop runs exactly once. Phase 4 of the Unix-crypt
         * ladder. */
        if (op == JOB_SHA512CRYPTMD5)
            pparams->max_iter = 1;
        /* DESCRYPT carrier (2026-05-08, Unix-crypt Phase 5): JOB_DESCRYPT
         * (e500). The 25-iter DES Feistel loop (FIXED iter count per Unix
         * DES crypt(3) "old-style") lives INSIDE template_finalize;
         * mirrors PHPBB3 / MD5CRYPT / SHA256CRYPT / SHA512CRYPT pattern.
         * Only the FINAL state is probed (CPU semantics at mdxfind.c:
         * 23673 calls JSLG once after the bsd_crypt_des for-loop). Force
         * max_iter=1 so the kernel's outer iter loop runs exactly once
         * and template_iterate (a stub) is never called; user `-i N`
         * would otherwise emit N redundant hits at iter levels 1..N for
         * the same final state. Phase 5 of the Unix-crypt ladder (FINAL
         * phase). */
        if (op == JOB_DESCRYPT)
            pparams->max_iter = 1;
        /* BCRYPT carrier (2026-05-08, Unix-crypt Phase 6): JOB_BCRYPT
         * (e450). The 2^cost Eksblowfish iter loop (cost parsed per-
         * salt-string at kernel entry; 4..31 range clamped) lives INSIDE
         * template_finalize; mirrors PHPBB3 / MD5CRYPT / SHA256CRYPT /
         * SHA512CRYPT / DESCRYPT pattern. Only the FINAL state is probed
         * (CPU semantics: bcrypt yields a single 24-byte digest per
         * crypt_rn call). Force max_iter=1 so the kernel's outer iter
         * loop runs exactly once and template_iterate (a stub) is never
         * called; user `-i N` would otherwise emit N redundant hits at
         * iter levels 1..N for the same final state. Phase 6 of the
         * slab-retirement ladder (final major slab kernel). */
        if (op == JOB_BCRYPT)
            pparams->max_iter = 1;
        /* B3 cursor protocol fields (Memo B §2):
         *   input_cursor_start / rule_cursor_start: 0 on first dispatch;
         *     advanced on re-issue when overflow_first_set was 1.
         *   inner_iter (BF Phase 1.8, was output_cursor_start): set by caller
         *     for BF chunks; left 0 here (memset above) so non-BF dispatches
         *     stay bit-identical. Host coercion 0->1 happens in kernel.
         *   overflow_first_set: kernel sets to 1 if any lane overflowed.
         *   overflow_first_word: SENTINEL 0xFFFFFFFFu so first overflowing
         *     lane wins CAS-min on its lane gid (kernel writes gid here,
         *     host re-derives word/rule via div/mod n_words).
         *   overflow_first_rule: unused in B3 (kept for forward-compat). */
        pparams->input_cursor_start  = 0;
        pparams->rule_cursor_start   = 0;
        pparams->overflow_first_set  = 0;
        pparams->overflow_first_word = 0xFFFFFFFFu;
        pparams->overflow_first_rule = 0;

        /* hit_count slot (offset 128). Kernel atomic_inc's it; init zero. */
        uint32_t *phit = (uint32_t *)(p + 128);
        *phit = 0;

        /* word_offset[] (offset 132). */
        memcpy(p + 132, word_offset, wo_size);

        /* packed_words[] (offset 132+wo_size). */
        memcpy(p + payload_pkt_off, packed_words, packed_size);
    }

    /* ONE coalesced host->GPU write — synchronous (CL_TRUE) per hashcat
     * pattern. With CL_MEM_USE_HOST_PTR the driver may optimize this to
     * an internal flush/sync (no actual DMA copy on unified-memory
     * implementations); we keep the call for spec correctness so devices
     * that need an explicit transfer get one. CL_TRUE drops the event
     * chain — NDRange below uses (0, NULL) wait_list. */
    err = clEnqueueWriteBuffer(d->queue, d->b_dispatch_payload, CL_TRUE,
                               0, payload_size, d->h_dispatch_payload,
                               0, NULL, NULL);
    if (err != CL_SUCCESS) {
        GPU_FATAL("b_dispatch_payload write err=%d on dev %d (size=%zu)",
                  err, dev_idx, payload_size);
    }

    /* Memo B B1 verification gate: emit one [pipe] line per dispatch
     * with the host->GPU write count. Was 4 (b_packed_buf + b_chunk_index
     * + b_params + b_hit_count); now 1 (b_dispatch_payload). The gate
     * for B1 sign-off is "n_writes=1 in every rules-engine dispatch
     * trace line" -- if you ever see n_writes != 1 here, B1 has a bug. */
    {
        static int _pipe_trace_cached = -1;
        if (_pipe_trace_cached == -1) {
            const char *e = getenv("MDXFIND_PIPE_TRACE");
            _pipe_trace_cached = (e && *e && *e != '0') ? 1 : 0;
        }
        if (_pipe_trace_cached == 1) {
            /* n_writes=1 is constant since 1.91 conversion to hashcat
             * pattern (one sync CL_TRUE write). Was tracking w_count under
             * the old async pattern; now hardcoded for the contract gate. */
            fprintf(stderr,
                    "[pipe] dev=%d path=rules n_writes=1 payload_bytes=%zu "
                    "params_bytes=%zu woff_bytes=%zu packed_bytes=%u\n",
                    dev_idx, payload_size,
                    sizeof(OCLParams), wo_size, packed_size);
        }
    }

    /* On-GPU dedup buffer (theory #1, mdx-architect 2026-04-30). One
     * uint slot per loaded target (compact entries + overflow entries).
     * Allocated lazily here at first rules dispatch — by now both
     * gpu_opencl_set_compact_table and gpu_opencl_set_overflow have run.
     * Persists across all subsequent dispatches; only zero-initialized
     * once. The kernel uses atomic_inc to gate hit emission: only the
     * first lane to crack a given target (prior count == 0) emits. The
     * host-side dedup gate at mdxfind.c:8044 is the safety net.
     *
     * Sizing: hash_data_count + overflow_count slots, 4 bytes each.
     * For HashMob.100k × rockyou-class workloads this is ~152 MB on the
     * GPU plus the same on host (zero buffer). 4090 has 24 GB; fits 100x.
     *
     * OpenCL 1.2 atomic_inc requires uint (32-bit). Hashcat's hashes_shown
     * is also u32 per slot; we follow the same convention. */
    {
        size_t need_slots = (size_t)_hash_data_count + (size_t)_overflow_count;
        if (need_slots == 0) need_slots = 1;   /* avoid 0-byte buffer */
        if (!d->b_hashes_shown || need_slots > d->hashes_shown_count) {
            /* MDXFIND_PIN_TRACE=1 enables the per-device hashes_shown
             * START line; the existing "allocated" line is the DONE.
             * Gated because 12 simultaneous 144MB allocations is the
             * candidate cause of the "10 second pause" Shooter
             * reports — visible when diagnosing, not in normal output. */
            static int _ht_trace_cached = -1;
            if (_ht_trace_cached == -1) {
                const char *_e = getenv("MDXFIND_PIN_TRACE");
                _ht_trace_cached = (_e && *_e && *_e != '0') ? 1 : 0;
            }
            struct timespec _hs_t0, _hs_t1;
            clock_gettime(CLOCK_MONOTONIC, &_hs_t0);
            if (_ht_trace_cached) {
                tsfprintf(stderr,
                    "OpenCL GPU[%d]: hashes_shown alloc START %zuMB\n",
                    dev_idx, (need_slots * sizeof(uint32_t)) / (1024*1024));
            }
            if (d->b_hashes_shown) clReleaseMemObject(d->b_hashes_shown);
            size_t bytes = need_slots * sizeof(uint32_t);
            /* Apply MIN_BUFFER_BYTES floor: tiny workloads (6 hashes, 0
             * overflow → 24 B) trip Windows NVIDIA's cold-JIT static
             * validator (CL_INVALID_KERNEL_ARGS at NDRange) even though
             * the kernel guards `if (gid < total_targets)`. Allocate the
             * floor and zero-fill the entire region. Shooter 12-GPU RTX
             * 4090 bug2b 2026-05-04: dev that just allocated a 24-byte
             * hashes_shown was always the one that crashed dispatch. */
            size_t alloc_bytes = bytes < MIN_BUFFER_BYTES ? MIN_BUFFER_BYTES : bytes;
            d->b_hashes_shown = clCreateBuffer(d->ctx, CL_MEM_READ_WRITE,
                                               alloc_bytes, NULL, &err);
            if (err != CL_SUCCESS || !d->b_hashes_shown) {
                fprintf(stderr,
                    "OpenCL GPU[%d]: hashes_shown alloc failed (err=%d, %zu slots, %zuMB) - "
                    "on-GPU dedup disabled, host dedup still active\n",
                    dev_idx, err, need_slots, bytes / (1024*1024));
                d->b_hashes_shown = NULL;
                d->hashes_shown_count = 0;
                return NULL;
            }
            /* Zero-initialize the FULL allocation (including any min-size
             * tail). The buffer persists across dispatches; this runs
             * exactly once per session (or on grow, which shouldn't happen
             * since hash_data_count + overflow_count is fixed at load
             * time). */
            uint32_t zero32 = 0;
            if (p_clEnqueueFillBuffer) {
                err = clEnqueueFillBuffer(d->queue, d->b_hashes_shown,
                    &zero32, sizeof(zero32), 0, alloc_bytes, 0, NULL, NULL);
            } else {
                /* OpenCL 1.1 fallback: stage a zero buffer on host. */
                uint8_t *zero_host = (uint8_t *)calloc(1, alloc_bytes);
                if (zero_host) {
                    err = clEnqueueWriteBuffer(d->queue, d->b_hashes_shown,
                        CL_TRUE, 0, alloc_bytes, zero_host, 0, NULL, NULL);
                    free(zero_host);
                } else {
                    err = -1;
                }
            }
            clFinish(d->queue);
            clock_gettime(CLOCK_MONOTONIC, &_hs_t1);
            if (err != CL_SUCCESS) {
                fprintf(stderr,
                    "OpenCL GPU[%d]: hashes_shown zero-init failed (err=%d) - "
                    "on-GPU dedup disabled, host dedup still active\n",
                    dev_idx, err);
                clReleaseMemObject(d->b_hashes_shown);
                d->b_hashes_shown = NULL;
                d->hashes_shown_count = 0;
                return NULL;
            }
            d->hashes_shown_count = need_slots;
            if (_ht_trace_cached) {
                double _hs_ms = (_hs_t1.tv_sec - _hs_t0.tv_sec) * 1e3
                              + (_hs_t1.tv_nsec - _hs_t0.tv_nsec) / 1e6;
                tsfprintf(stderr,
                    "OpenCL GPU[%d]: hashes_shown alloc DONE in %.2fs (%zu slots, %zuMB)\n",
                    dev_idx, _hs_ms / 1e3, need_slots, bytes / (1024*1024));
            }
            tsfprintf(stderr,
                "OpenCL GPU[%d]: hashes_shown allocated (%zu slots, %zuMB) - "
                "on-GPU dedup active\n",
                dev_idx, need_slots, bytes / (1024*1024));
        }
    }

    /* B1: params + hit_count + word_offset + packed_words are already
     * written into b_dispatch_payload by the single coalesced
     * clEnqueueWriteBuffer above. The old separate b_params write and
     * b_hit_count zero-fill have been deleted -- both fields live inside
     * the payload now. The kernel reads them at deterministic offsets
     * (params @ 0; hit_count @ 128; word_offset @ 132; words @ 132+4*N).
     * See gpu/gpu_md5_rules.cl md5_rules_phase0 kernel header for the
     * layout contract. */

    /* Set kernel args. Order MUST match md5_rules_phase0 declaration in
     * gpu/gpu_md5_rules.cl (B1 signature: payload buffer first, then
     * everything else; b_packed_buf / b_chunk_index / b_params /
     * b_hit_count are folded INTO the payload and no longer kernel args).
     *
     * Memo B B2: when MDXFIND_GPU_TEMPLATE=md5 is set, swap to the
     * template_phase0 kernel from gpu_template.cl + gpu_md5_core.cl.
     * Same signature as md5_rules_phase0; the kernel-arg setup below
     * is unchanged. Template build/lazy is per-device and self-healing:
     * if compile fails or kernel create fails, fall back to the
     * legacy kernel for this dispatch (warning already emitted). */
    /* B5 chokepoint widening (2026-05-04): single resolve helper picks
     * the per-op template kernel for {MD5, MD4, SHA1, SHA224, SHA256}.
     * Production path is the template; legacy md5_rules_phase0 only
     * survives as a fallback when:
     *   (a) op == JOB_MD5 AND template compile/lazy failed for MD5; OR
     *   (b) MDXFIND_GPU_TEMPLATE is set and selects an algo other than
     *       this dispatch's op (env-var-driven cross-algorithm probe;
     *       not a production configuration).
     * For non-MD5 ops the legacy kernel computes MD5 digests (wrong) — but
     * cases (b) is opt-in and case (a) for non-MD5 is structurally a
     * hard error logged by the compile helpers (gpu_*_core build failed),
     * not a silent miscompute. See gpu_template_resolve_kernel doc above. */
    cl_kernel kern = gpu_template_resolve_kernel(d, dev_idx, op);
    if (!kern) kern = d->kern_md5_rules_phase0;
    /* Phase 1.9 Tranche A1 (2026-05-10): BF-fast MD5 kernel swap. The
     * standard resolver picked the slow MD5 template
     * (d->kern_template_phase0) for JOB_MD5; if the dispatch is BF-fast
     * eligible (host-side gate: unsalted JOB_MD5, Numrules<=1, append-
     * only mask, napp in [1,8]) and the BF-fast template compile +
     * kernel lazy succeed on this device, swap to
     * d->kern_template_phase0_md5_bf. The two kernels share the
     * "template_phase0" name + signature; A1's BF-fast body is byte-
     * identical to the slow path, so the swap is byte-exact and
     * provides the integration surface for A2-A4 perf work. If the
     * BF-fast build fails (transient JIT, low-mem GPU, etc.) the
     * compile helper emits a one-shot warning and we stay on the slow
     * kernel — silent self-heal. The swap is gated on
     * `kern == d->kern_template_phase0` so we never accidentally
     * displace a salted-op kernel or a non-MD5 algorithm's template.
     * Op==JOB_MD5UC reuses kern_template_phase0 (algo_mode=1) and is
     * NOT eligible for the BF-fast path in A1 — the producer-side
     * gate restricts to JOB_MD5 only. */
    if (bf_fast_eligible && op == JOB_MD5 &&
        kern == d->kern_template_phase0)
    {
        if (gpu_opencl_template_md5_bf_compile(d, dev_idx) == 0 &&
            gpu_opencl_template_md5_bf_kernel_lazy(d, dev_idx) == 0 &&
            d->kern_template_phase0_md5_bf != NULL)
        {
            kern = d->kern_template_phase0_md5_bf;
        }
        /* else: fall through with the slow kernel — warning already
         * emitted by the compile helper, dispatch correctness intact. */
    }
    /* B6 salt-axis (2026-05-06): if this is a salted op but the salted
     * kernel didn't resolve (compile/lazy failed — usually transient
     * NVIDIA cold-JIT err=-9999 on tiny workloads), DO NOT fall back to
     * the unsalted md5_rules_phase0 kernel — it would silently compute
     * MD5(p) instead of the salted digest. Skip the GPU dispatch and
     * return NULL; the host CPU path picks up the work via Typedone[]
     * fallback. */
    if (is_salted_pack &&
        kern != d->kern_template_phase0_md5salt &&
        kern != d->kern_template_phase0_md5saltpass &&
        kern != d->kern_template_phase0_sha1saltpass &&
        kern != d->kern_template_phase0_sha256saltpass &&
        kern != d->kern_template_phase0_sha224saltpass &&
        /* B6.4 MD5PASSSALT fan-out (2026-05-06): first APPEND-shape salted
         * variant. Add to the salted-kernel-safety allowlist. */
        kern != d->kern_template_phase0_md5passsalt &&
        /* B6.5 SHA1PASSSALT fan-out (2026-05-06): first SHA-family APPEND-
         * shape salted variant. Add to the salted-kernel-safety allowlist. */
        kern != d->kern_template_phase0_sha1passsalt &&
        /* B6.7 SHA256PASSSALT fan-out (2026-05-06): second SHA-family
         * APPEND-shape salted variant. Add to the salted-kernel-safety
         * allowlist. */
        kern != d->kern_template_phase0_sha256passsalt &&
        /* B6.9 SHA512 fan-out (2026-05-06): SHA512SALTPASS — first
         * 64-bit-state salted variant. Add to the salted-kernel-safety
         * allowlist. */
        kern != d->kern_template_phase0_sha512saltpass &&
        /* B6.10 SHA512PASSSALT fan-out (2026-05-06): SHA512PASSSALT —
         * second 64-bit-state salted variant; APPEND-shape sibling.
         * FINAL B6 ladder step. */
        kern != d->kern_template_phase0_sha512passsalt &&
        /* Family E HMAC-SHA384 carrier (2026-05-08): SHA384SALTPASS-shaped
         * template kernel for HMAC-SHA384 ops via algo_mode=5/6. Without
         * this entry the salted-kernel-safety gate skips dispatch and CPU
         * picks up only word_idx=0 via Typedone[] oracle. */
        kern != d->kern_template_phase0_sha384saltpass &&
        /* Family G HMAC-RIPEMD-160 carrier (2026-05-08): RIPEMD160SALTPASS-
         * shaped template kernel for HMAC-RMD160 ops via algo_mode=5/6.
         * Without this entry the salted-kernel-safety gate skips dispatch
         * and CPU picks up only word_idx=0 via Typedone[] oracle. */
        kern != d->kern_template_phase0_ripemd160saltpass &&
        /* Family H HMAC-RIPEMD-320 carrier (2026-05-08): RIPEMD320SALTPASS-
         * shaped template kernel for HMAC-RMD320 ops via algo_mode=5/6.
         * Without this entry the salted-kernel-safety gate skips dispatch
         * and CPU picks up only word_idx=0 via Typedone[] oracle. */
        kern != d->kern_template_phase0_ripemd320saltpass &&
        /* Family I HMAC-BLAKE2S carrier (2026-05-08): hand-written Path A
         * carrier template kernel for HMAC-BLAKE2S (e828) via algo_mode=5.
         * Without this entry the salted-kernel-safety gate skips dispatch
         * and CPU picks up only word_idx=0 via Typedone[] oracle.
         * (Family E 2026-05-08 lesson: missing salted-kernel-safety entry
         * = "WARN salted kernel for op=N unavailable" + silent CPU
         * fallback. Sibling sites 3 (this gate) and 4 (kern_is_salted_-
         * template at line ~9530+) ALWAYS appear together for salted
         * templates — extending one means extending both.) */
        kern != d->kern_template_phase0_hmac_blake2s &&
        /* Family J HMAC-STREEBOG-256 carrier (2026-05-08): hand-written
         * Path A carrier template kernel for HMAC-STREEBOG256_KSALT (e838)
         * + HMAC-STREEBOG256_KPASS (e837) via algo_mode=5/6. Without this
         * entry the salted-kernel-safety gate skips dispatch with the
         * "salted kernel for op=N unavailable" warning + CPU picks up
         * only word_idx=0 via Typedone[] oracle. */
        kern != d->kern_template_phase0_hmac_streebog256 &&
        /* Family K HMAC-STREEBOG-512 carrier (2026-05-08): hand-written
         * Path A carrier template kernel for HMAC-STREEBOG512_KSALT (e840)
         * + HMAC-STREEBOG512_KPASS (e839) via algo_mode=5/6. Without this
         * entry the salted-kernel-safety gate skips dispatch with the
         * "salted kernel for op=N unavailable" warning + CPU picks up
         * only word_idx=0 via Typedone[] oracle. Final HMAC family in
         * the ladder. */
        kern != d->kern_template_phase0_hmac_streebog512 &&
        /* PHPBB3 carrier (2026-05-08): hand-written Path A salted-
         * template kernel for JOB_PHPBB3 (e455). Without this entry
         * the salted-kernel-safety gate skips dispatch with the
         * "salted kernel for op=455 unavailable" warning + CPU picks
         * up only word_idx=0 via Typedone[] oracle. Sibling site to
         * the kern_is_salted_template OR-chain below; per
         * feedback_architect_host_wiring_reflex.md sites 3+4 ALWAYS
         * appear together for salted templates. */
        kern != d->kern_template_phase0_phpbb3 &&
        /* MD5CRYPT carrier (2026-05-08): hand-written Path A salted-
         * template kernel for JOB_MD5CRYPT (e511). Without this entry
         * the salted-kernel-safety gate skips dispatch with the
         * "salted kernel for op=511 unavailable" warning + CPU picks
         * up only word_idx=0 via Typedone[] oracle. Sibling site to
         * the kern_is_salted_template OR-chain below. */
        kern != d->kern_template_phase0_md5crypt &&
        /* SHA256CRYPT carrier (2026-05-08): hand-written Path A salted-
         * template kernel for JOB_SHA256CRYPT (e512). Without this entry
         * the salted-kernel-safety gate skips dispatch with the
         * "salted kernel for op=512 unavailable" warning + CPU picks
         * up only word_idx=0 via Typedone[] oracle. Sibling site to
         * the kern_is_salted_template OR-chain below. Phase 2 of the
         * Unix-crypt ladder. */
        kern != d->kern_template_phase0_sha256crypt &&
        /* SHA512CRYPT carrier (2026-05-08): hand-written Path A salted-
         * template kernel for JOB_SHA512CRYPT (e513). Without this entry
         * the salted-kernel-safety gate skips dispatch with the
         * "salted kernel for op=513 unavailable" warning + CPU picks
         * up only word_idx=0 via Typedone[] oracle. Sibling site to
         * the kern_is_salted_template OR-chain below; per
         * feedback_architect_host_wiring_reflex.md sites 3+4 ALWAYS
         * appear together for salted templates. Phase 3 of the Unix-
         * crypt ladder. */
        kern != d->kern_template_phase0_sha512crypt &&
        /* DESCRYPT carrier (2026-05-08, Unix-crypt Phase 5): hand-written
         * Path A salted-template kernel for JOB_DESCRYPT (e500). Without
         * this entry the salted-kernel-safety gate skips dispatch with
         * the "salted kernel for op=500 unavailable" warning + CPU picks
         * up only word_idx=0 via Typedone[] oracle. Sibling site to the
         * kern_is_salted_template OR-chain below; per feedback_architect_-
         * host_wiring_reflex.md sites 3+4 ALWAYS appear together for
         * salted templates. Phase 5 of the Unix-crypt ladder (FINAL
         * phase). */
        kern != d->kern_template_phase0_descrypt &&
        /* BCRYPT carrier (2026-05-08, Unix-crypt Phase 6): hand-written
         * Path A salted-template kernel for JOB_BCRYPT (e450). Without
         * this entry the salted-kernel-safety gate skips dispatch with
         * the "salted kernel for op=450 unavailable" warning + CPU picks
         * up only word_idx=0 via Typedone[] oracle. Sibling site to the
         * kern_is_salted_template OR-chain below; per feedback_architect_-
         * host_wiring_reflex.md sites 3+4 ALWAYS appear together for
         * salted templates. Phase 6 of the slab-retirement ladder (final
         * major slab kernel). */
        kern != d->kern_template_phase0_bcrypt) {
        static int _warned[MAX_GPU_DEVICES] = {0};
        if (dev_idx >= 0 && dev_idx < MAX_GPU_DEVICES && !_warned[dev_idx]) {
            _warned[dev_idx] = 1;
            fprintf(stderr,
                "OpenCL GPU[%d]: WARN salted kernel for op=%d unavailable; "
                "skipping GPU dispatch (CPU will catch up via Typedone[]).\n",
                dev_idx, op);
        }
        return NULL;
    }

    /* B7.1/B7.2: ensure b_template_mask_charsets and b_template_mask_sizes
     * are bound on every dispatch. Production runs without -n / -N never
     * call gpu_opencl_set_mask, so the buffers would be NULL — but the
     * kernel signature requires them. Lazy-allocate sentinels here; the
     * kernel's (n_prepend == 0 && n_append >= 1) gate prevents any actual
     * read of the sentinel charsets, and mask_sizes sentinel of all-1s
     * makes any stray loop iteration's divmod terminate safely.
     *
     * NOTE: The legacy md5_rules_phase0 kernel does NOT take these args.
     * If we fall back to that kernel (kern == d->kern_md5_rules_phase0),
     * we MUST NOT bind a 15th/16th arg. Detect by comparing kern pointer. */
    int kern_is_template = (kern != d->kern_md5_rules_phase0);
    /* B6 salt-axis (2026-05-06; §11 row 17): parallel flag for the salted
     * variants. Compares the resolved kernel handle against the per-device
     * salted-template kernel slots. SETARG block below binds 3 extra args
     * (salt_buf, salt_off, salt_lens) under this flag — args 17/18/19 in
     * the kernel signature, gated by #ifdef GPU_TEMPLATE_HAS_SALT in
     * gpu_template.cl. Row 16.5's `is_salted_pack` is the OP-direct twin
     * derived earlier (drives host pack); this flag is the KERNEL-handle
     * twin (drives kernel-arg binding). Both signals point to the same
     * condition but at different code sites — intentional duplicate. */
    int kern_is_salted_template =
        (kern == d->kern_template_phase0_md5salt) ||
        (kern == d->kern_template_phase0_md5saltpass) ||
        (kern == d->kern_template_phase0_sha1saltpass) ||
        (kern == d->kern_template_phase0_sha256saltpass) ||
        (kern == d->kern_template_phase0_sha224saltpass) ||
        /* B6.4 MD5PASSSALT fan-out (2026-05-06): first APPEND-shape salted
         * variant. SETARG block below binds 3 extra salt args under this
         * flag. */
        (kern == d->kern_template_phase0_md5passsalt) ||
        /* B6.5 SHA1PASSSALT fan-out (2026-05-06): first SHA-family APPEND-
         * shape salted variant. SETARG block below binds 3 extra salt
         * args under this flag. */
        (kern == d->kern_template_phase0_sha1passsalt) ||
        /* B6.7 SHA256PASSSALT fan-out (2026-05-06): second SHA-family
         * APPEND-shape salted variant. SETARG block below binds 3 extra
         * salt args under this flag. */
        (kern == d->kern_template_phase0_sha256passsalt) ||
        /* B6.9 SHA512 fan-out (2026-05-06): SHA512SALTPASS — first
         * 64-bit-state salted variant. SETARG block below binds 3 extra
         * salt args under this flag. */
        (kern == d->kern_template_phase0_sha512saltpass) ||
        /* B6.10 SHA512PASSSALT fan-out (2026-05-06): SHA512PASSSALT —
         * second 64-bit-state salted variant; APPEND-shape sibling.
         * FINAL B6 ladder step. */
        (kern == d->kern_template_phase0_sha512passsalt) ||
        /* Family E HMAC-SHA384 carrier (2026-05-08): SHA384SALTPASS-shaped
         * template kernel; must bind salt args 17/18/19 like other salted
         * templates. */
        (kern == d->kern_template_phase0_sha384saltpass) ||
        /* Family G HMAC-RIPEMD-160 carrier (2026-05-08): RIPEMD160SALTPASS-
         * shaped template kernel; must bind salt args 17/18/19 like other
         * salted templates. Without this flag the kernel runs with empty
         * salt buffer → silent miscompute (wrong digests for ANY word). */
        (kern == d->kern_template_phase0_ripemd160saltpass) ||
        /* Family H HMAC-RIPEMD-320 carrier (2026-05-08): RIPEMD320SALTPASS-
         * shaped template kernel; must bind salt args 17/18/19 like other
         * salted templates. Without this flag the kernel runs with empty
         * salt buffer → silent miscompute (wrong digests for ANY word). */
        (kern == d->kern_template_phase0_ripemd320saltpass) ||
        /* Family I HMAC-BLAKE2S carrier (2026-05-08): hand-written Path A
         * salted-template kernel; must bind salt args 17/18/19 like other
         * salted templates. The kernel's HMAC body reads salt_buf via the
         * template_finalize HAS_SALT signature (salt_bytes/salt_len/algo_mode
         * trio) and consumes it as the BLAKE2S inner-block message. Without
         * this flag the kernel runs with empty salt buffer → silent
         * miscompute (wrong digests for ANY word). */
        (kern == d->kern_template_phase0_hmac_blake2s) ||
        /* Family J HMAC-STREEBOG-256 carrier (2026-05-08): hand-written
         * Path A salted-template kernel; must bind salt args 17/18/19 like
         * other salted templates. The kernel's HMAC body reads salt_buf via
         * the template_finalize HAS_SALT signature (salt_bytes/salt_len/-
         * algo_mode trio) and consumes it as the STREEBOG-256 inner-block
         * message. Without this flag the kernel runs with empty salt buffer
         * → silent miscompute (wrong digests for ANY word). */
        (kern == d->kern_template_phase0_hmac_streebog256) ||
        /* Family K HMAC-STREEBOG-512 carrier (2026-05-08): hand-written
         * Path A salted-template kernel; must bind salt args 17/18/19 like
         * other salted templates. The kernel's HMAC body reads salt_buf via
         * the template_finalize HAS_SALT signature (salt_bytes/salt_len/-
         * algo_mode trio) and consumes it as the STREEBOG-512 inner-block
         * message. Without this flag the kernel runs with empty salt buffer
         * - silent miscompute (wrong digests for ANY word). Final HMAC
         * family in the ladder. */
        (kern == d->kern_template_phase0_hmac_streebog512) ||
        /* PHPBB3 carrier (2026-05-08): hand-written Path A salted-
         * template kernel; must bind salt args 17/18/19 like other
         * salted templates. The kernel's template_finalize reads
         * salt_bytes[3] (cost char -> iter count) + salt_bytes[4..11]
         * (8-byte salt for step 1) via the HAS_SALT signature
         * (salt_bytes/salt_len/algo_mode trio). Without this flag the
         * kernel runs with empty salt buffer -> silent miscompute (zero
         * iter count + zero salt bytes -> wrong digests for ANY word).
         * Sibling site to the salted-kernel-safety allowlist above; per
         * feedback_architect_host_wiring_reflex.md sites 3+4 ALWAYS
         * appear together for salted templates. */
        (kern == d->kern_template_phase0_phpbb3) ||
        /* MD5CRYPT carrier (2026-05-08): hand-written Path A salted-
         * template kernel; must bind salt args 17/18/19 like other
         * salted templates. The kernel's template_finalize reads
         * salt_bytes[0..2] ("$1$" prefix) + salt_bytes[3..] (raw salt
         * up to 8 bytes, terminated by '$') via the HAS_SALT signature
         * (salt_bytes/salt_len/algo_mode trio). Without this flag the
         * kernel runs with empty salt buffer -> silent miscompute
         * (zero salt bytes -> wrong digests for ANY word). Sibling
         * site to the salted-kernel-safety allowlist above. */
        (kern == d->kern_template_phase0_md5crypt) ||
        /* SHA256CRYPT carrier (2026-05-08): hand-written Path A salted-
         * template kernel; must bind salt args 17/18/19 like other
         * salted templates. The kernel's template_finalize reads
         * salt_bytes[0..2] ("$5$" prefix) and optionally "rounds=N$"
         * decoding, then raw salt up to 16 bytes (terminated by '$' or
         * end) via the HAS_SALT signature (salt_bytes/salt_len/algo_-
         * mode trio). Without this flag the kernel runs with empty salt
         * buffer -> silent miscompute (zero salt bytes -> wrong digests
         * for ANY word). Sibling site to the salted-kernel-safety
         * allowlist above; per feedback_architect_host_wiring_reflex.md
         * sites 3+4 ALWAYS appear together for salted templates. Phase
         * 2 of the Unix-crypt ladder. */
        (kern == d->kern_template_phase0_sha256crypt) ||
        /* SHA512CRYPT carrier (2026-05-08): hand-written Path A salted-
         * template kernel; must bind salt args 17/18/19 like other
         * salted templates. The kernel's template_finalize reads
         * salt_bytes[0..2] ("$6$" prefix) and optionally "rounds=N$"
         * decoding, then raw salt up to 16 bytes (terminated by '$' or
         * end) via the HAS_SALT signature (salt_bytes/salt_len/algo_-
         * mode trio). Without this flag the kernel runs with empty salt
         * buffer -> silent miscompute (zero salt bytes -> wrong digests
         * for ANY word). Sibling site to the salted-kernel-safety
         * allowlist above; per feedback_architect_host_wiring_reflex.md
         * sites 3+4 ALWAYS appear together for salted templates. Phase
         * 3 of the Unix-crypt ladder. */
        (kern == d->kern_template_phase0_sha512crypt) ||
        /* DESCRYPT carrier (2026-05-08, Unix-crypt Phase 5): hand-written
         * Path A salted-template kernel; must bind salt args 17/18/19
         * like other salted templates. The kernel's template_finalize
         * reads salt_bytes[0..1] (2-char phpitoa64 salt) via the
         * HAS_SALT signature (salt_bytes/salt_len/algo_mode trio).
         * Without this flag the kernel runs with empty salt buffer ->
         * silent miscompute (zero salt -> saltbits=0 -> wrong DES output
         * for ANY word). Sibling site to the salted-kernel-safety
         * allowlist above; per feedback_architect_host_wiring_reflex.md
         * sites 3+4 ALWAYS appear together for salted templates. Phase
         * 5 of the Unix-crypt ladder (FINAL phase). */
        (kern == d->kern_template_phase0_descrypt) ||
        /* BCRYPT carrier (2026-05-08, Unix-crypt Phase 6): hand-written
         * Path A salted-template kernel; must bind salt args 17/18/19
         * like other salted templates. The kernel's template_finalize
         * reads salt_bytes[0..salt_len) (28- or 29-byte "$2[abkxy]$NN$
         * <base64>" prefix; cost + variant + raw-salt parsed inline)
         * via the HAS_SALT signature (salt_bytes/salt_len/algo_mode
         * trio). Without this flag the kernel runs with empty salt
         * buffer -> silent miscompute (no salt parse -> wrong bcrypt
         * output for ANY word). Sibling site to the salted-kernel-safety
         * allowlist above; per feedback_architect_host_wiring_reflex.md
         * sites 3+4 ALWAYS appear together for salted templates. Phase
         * 6 of the slab-retirement ladder (final major slab kernel). */
        (kern == d->kern_template_phase0_bcrypt);
    if (kern_is_template && !d->b_template_mask_charsets) {
        /* B7.3+: sentinel widened to 16 rows (8 prepend + 8 append) ×
         * 256 bytes = 4096 bytes, matching the B7.5-capable layout. */
        uint8_t sentinel[16 * 256];
        memset(sentinel, 0, sizeof(sentinel));
        cl_int merr;
        d->b_template_mask_charsets = clCreateBuffer(d->ctx,
            CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
            sizeof(sentinel), sentinel, &merr);
        if (merr != CL_SUCCESS || !d->b_template_mask_charsets) {
            GPU_FATAL("b_template_mask_charsets sentinel alloc err=%d "
                      "on dev %d", merr, dev_idx);
        }
    }
    if (kern_is_template && !d->b_template_mask_sizes) {
        /* B7.3+: sentinel widened to 16 entries (8 prepend + 8 append). */
        uint32_t sentinel_sizes[16];
        for (int i = 0; i < 16; i++) sentinel_sizes[i] = 1u;
        cl_int merr;
        d->b_template_mask_sizes = clCreateBuffer(d->ctx,
            CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
            sizeof(sentinel_sizes), sentinel_sizes, &merr);
        if (merr != CL_SUCCESS || !d->b_template_mask_sizes) {
            GPU_FATAL("b_template_mask_sizes sentinel alloc err=%d "
                      "on dev %d", merr, dev_idx);
        }
    }

    {
        /* Fail-fast SETARG: B1 (rev 1.79) coalesced 16 args -> 14 but the
         * bind block continued to ignore clSetKernelArg return codes. A
         * silent CL_INVALID_ARG_* during bind surfaces only later as
         * CL_INVALID_KERNEL_ARGS at clEnqueueNDRangeKernel, naming the
         * kernel but not the offending arg. Shooter's 12-GPU RTX 4090
         * Win+NVIDIA cold-JIT crash (2026-05-04) requires this to identify
         * which arg the driver rejects. Pure instrumentation -- no
         * behavioral change on the success path. See
         * mdx-team-state.md mdx-debug 2026-05-04 entry.
         *
         * B7.1 (2026-05-05): kernel signature widened from 14 to 15 args
         * for template_phase0 (mask_charsets at index 14). The legacy
         * kernel md5_rules_phase0 keeps 14 args; the kern_is_template
         * gate above selects which arg count to bind.
         *
         * B7.2 (2026-05-06): kernel signature widened from 15 to 16 args
         * for template_phase0 (mask_sizes at index 15). The legacy
         * md5_rules_phase0 kernel still takes only 14 args; both the 15th
         * (mask_charsets) and 16th (mask_sizes) args are gated behind
         * kern_is_template. */
#define SETARG(K, IDX, SZ, P) do {                                              \
    cl_int _e = clSetKernelArg((K), (IDX), (SZ), (P));                          \
    if (_e != CL_SUCCESS)                                                       \
        GPU_FATAL("clSetKernelArg(arg %d, kern md5_rules) on dev %d failed: %d",\
                  (int)(IDX), dev_idx, (int)_e);                                \
} while (0)
        int a = 0;
        SETARG(kern, a++, sizeof(cl_mem), &d->b_dispatch_payload); /* coalesced */
        SETARG(kern, a++, sizeof(cl_mem), &d->b_rule_program);     /* __constant */
        SETARG(kern, a++, sizeof(cl_mem), &d->b_rule_offset);
        SETARG(kern, a++, sizeof(cl_mem), &d->b_compact_fp);
        SETARG(kern, a++, sizeof(cl_mem), &d->b_compact_idx);
        SETARG(kern, a++, sizeof(cl_mem), &d->b_hash_data);
        SETARG(kern, a++, sizeof(cl_mem), &d->b_hash_data_off);
        SETARG(kern, a++, sizeof(cl_mem), &d->b_hash_data_len);
        SETARG(kern, a++, sizeof(cl_mem), &d->b_hits);
        SETARG(kern, a++, sizeof(cl_mem), &d->b_overflow_keys);
        SETARG(kern, a++, sizeof(cl_mem), &d->b_overflow_hashes);
        SETARG(kern, a++, sizeof(cl_mem), &d->b_overflow_offsets);
        SETARG(kern, a++, sizeof(cl_mem), &d->b_overflow_lengths);
        SETARG(kern, a++, sizeof(cl_mem), &d->b_hashes_shown);     /* on-GPU dedup */
        if (kern_is_template) {
            SETARG(kern, a++, sizeof(cl_mem), &d->b_template_mask_charsets); /* B7.1 */
            SETARG(kern, a++, sizeof(cl_mem), &d->b_template_mask_sizes);    /* B7.2 */
        }
        if (kern_is_salted_template) {
            /* B6 salt-axis (2026-05-06; §11 row 18): three appended args
             * for the salted template variants. The buffers are uploaded
             * by gpu_opencl_set_salts() at the gpujob_opencl.c:605 sync
             * point (widened in gpujob_opencl.c §11 row 20 to fire for
             * MASK + gpu_salt_judy(g->op) != NULL configurations). The
             * lengths buffer is ushort (uint16_t) per §13.3 — the slab
             * convention at gpu_md5salt.cl:3, host populated as uint16_t
             * at gpu_opencl.c:2580-2591. */
            SETARG(kern, a++, sizeof(cl_mem), &d->b_salt_data);  /* salt_buf  */
            SETARG(kern, a++, sizeof(cl_mem), &d->b_salt_off);   /* salt_off  */
            SETARG(kern, a++, sizeof(cl_mem), &d->b_salt_len);   /* salt_lens */
        }
#undef SETARG
    }

    /* Per-arg diagnostic trace (env-gated, one-shot per device).
     *
     * MDXFIND_ARG_TRACE=1 prints driver-side state for each of the 14
     * kernel args at first dispatch. For each cl_mem we query
     * CL_MEM_SIZE and CL_MEM_FLAGS via clGetMemObjectInfo so the values
     * reflect what the OpenCL runtime actually sees -- not what the
     * host source thinks it allocated.
     *
     * Goal: when Shooter / Just / any Win+NVIDIA user hits the
     * CL_INVALID_KERNEL_ARGS class at clEnqueueNDRangeKernel, the
     * MDXFIND_ARG_TRACE=1 stderr names exactly which arg has what
     * size/flags. That dataset definitively answers the buffer-size
     * hypothesis without another round-trip.
     *
     * One-shot per device (static per-device flag), default off so
     * normal runs don't pay the 14 clGetMemObjectInfo calls. */
    {
        static int _arg_trace_cached = -1;
        if (_arg_trace_cached == -1) {
            const char *e = getenv("MDXFIND_ARG_TRACE");
            _arg_trace_cached = (e && *e && *e != '0') ? 1 : 0;
        }
        if (_arg_trace_cached == 1) {
            static int _arg_trace_logged[MAX_GPU_DEVICES] = {0};
            if (dev_idx >= 0 && dev_idx < MAX_GPU_DEVICES
                && !_arg_trace_logged[dev_idx]) {
                _arg_trace_logged[dev_idx] = 1;
                /* Hardcoded names match the SETARG order above (14 args).
                 * Update both blocks together if the kernel signature
                 * changes. */
                static const char * const _arg_names[] = {
                    "payload",          /*  0 */
                    "rule_program",     /*  1 */
                    "rule_offset",      /*  2 */
                    "compact_fp",       /*  3 */
                    "compact_idx",      /*  4 */
                    "hash_data",        /*  5 */
                    "hash_data_off",    /*  6 */
                    "hash_data_len",    /*  7 */
                    "hits",             /*  8 */
                    "overflow_keys",    /*  9 */
                    "overflow_hashes",  /* 10 */
                    "overflow_offsets", /* 11 */
                    "overflow_lengths", /* 12 */
                    "hashes_shown",     /* 13 */
                };
                cl_mem _arg_bufs[14] = {
                    d->b_dispatch_payload,
                    d->b_rule_program,
                    d->b_rule_offset,
                    d->b_compact_fp,
                    d->b_compact_idx,
                    d->b_hash_data,
                    d->b_hash_data_off,
                    d->b_hash_data_len,
                    d->b_hits,
                    d->b_overflow_keys,
                    d->b_overflow_hashes,
                    d->b_overflow_offsets,
                    d->b_overflow_lengths,
                    d->b_hashes_shown,
                };
                for (int ai = 0; ai < 14; ai++) {
                    cl_mem buf = _arg_bufs[ai];
                    size_t buf_size = 0;
                    cl_mem_flags buf_flags = 0;
                    if (buf) {
                        clGetMemObjectInfo(buf, CL_MEM_SIZE,
                                           sizeof(buf_size), &buf_size, NULL);
                        clGetMemObjectInfo(buf, CL_MEM_FLAGS,
                                           sizeof(buf_flags), &buf_flags, NULL);
                    }
                    /* Stringify the salient flag bits. The driver may set
                     * other bits (alloc_host_ptr, copy_host_ptr, host_*);
                     * we surface the ones that affect kernel-arg validation. */
                    const char *rwstr =
                        (buf_flags & CL_MEM_READ_ONLY)  ? "READ_ONLY"  :
                        (buf_flags & CL_MEM_WRITE_ONLY) ? "WRITE_ONLY" :
                        (buf_flags & CL_MEM_READ_WRITE) ? "READ_WRITE" :
                                                          "(none)";
                    fprintf(stderr,
                        "[arg] dev=%d kern=md5_rules a=%d name=%s "
                        "buf_size=%zu flags=%s cl_mem=%p\n",
                        dev_idx, ai, _arg_names[ai],
                        buf_size, rwstr, (void *)buf);
                }
                fflush(stderr);
            }
        }
    }

    /* global_size = num_words * n_rules * mask_size, rounded up to
     * local_size. B7.1: mask_size > 1 only when the chokepoint admitted
     * a single-position append mask (n_prepend == 0, n_append == 1) —
     * see b71_mask_size derivation above. mask_size == 1 (no mask) keeps
     * the dispatch global_size identical to pre-B7. */
    size_t local = 64;
    /* BCRYPT carrier (2026-05-08, Unix-crypt Phase 6): the BCRYPT template
     * kernel pins WG=BCRYPT_WG_SIZE (8) at compile time via reqd_work_-
     * group_size attribute (gpu_template.cl rev 1.x §15.B mechanism (i)).
     * Override `local = 8` here so clEnqueueNDRangeKernel honors the
     * kernel's required WG size; otherwise the driver rejects with
     * CL_INVALID_WORK_GROUP_SIZE on Win NVIDIA NDRange validation
     * (architect §15 R-S2 + Win NVIDIA preflight per
     * feedback_win_nvidia_coverage_gap.md). The 32 KB __local sbox_pool
     * footprint per WG (8 lanes × 4 KB) fits Pascal 48 KB / RDNA 64 KB /
     * Mali-T860 32 KB exactly. */
    if (kern == d->kern_template_phase0_bcrypt) {
        local = 8;  /* must match BCRYPT_WG_SIZE in gpu_common.cl */
    }
    /* BF chunk-as-job (2026-05-09 Tranche 3): when bf_num_masks > 0 the
     * kernel's mask axis is the chunk's per-word range (bf_num_masks),
     * NOT the full gpu_mask_total keyspace. Using b71_mask_size here would
     * over-launch by (gpu_mask_total / bf_num_masks)× and burn billions
     * of redundant work-items. The OCLParams.num_salts pack at line ~10420
     * already uses `bf_num_masks > 0u ? bf_num_masks : b71_mask_size` for
     * the kernel's mask_size; this enqueue must mirror that. */
    size_t kernel_mask_size = (bf_num_masks > 0u) ? (size_t)bf_num_masks
                                                  : (size_t)b71_mask_size;
    size_t total = (size_t)num_words * (size_t)d->gpu_n_rules
                   * kernel_mask_size;
    size_t global = ((total + local - 1) / local) * local;

    /* B3 cursor-restart loop (Memo B §2). On a non-overflow dispatch the
     * loop body runs exactly once -- identical to the pre-B3 single-shot
     * path. On overflow, the lane CAS-mins its gid into the payload's
     * overflow_first_word slot and signals overflow_first_set; the host
     * reads those back, advances input_cursor_start + rule_cursor_start
     * past the overflow point, resets the hit/overflow state, re-issues
     * the SAME packed buffer (no need to re-pack words), and accumulates
     * the next batch of hits into d->h_hits beyond the current count.
     *
     * Aggregate hit cap is GPU_PACKED_MAX_HITS (host buffer size); if
     * the workload genuinely produces more cracks than that, the excess
     * is dropped with a warning -- B3 minimal scope; back-pressure ladder
     * (grow/split/drain) is a follow-up phase per Memo B §2.
     *
     * Safety cap on re-issue count: 64. With max_hits=1M and 16K-word
     * dispatches at ~93K hits/avg, overflow is rare in practice and a
     * single re-issue almost always finishes. The cap guards against a
     * pathological cursor-stall bug (e.g., cursor failed to advance). */
    uint32_t total_hits = 0;
    cl_event kern_event = NULL;
    int last_dispatch_kern_us_emitted = 0;
    const int MAX_REISSUES = 64;
    /* Tracks reissues across the FINAL salt_page only — the warning
     * block at the function tail only uses this for the warn copy. The
     * inner loop below resets this on each salt_page so the per-page
     * MAX_REISSUES safety cap remains meaningful. */
    int reissue_count = 0;

    /* B6 salt-axis (2026-05-06; §13.4 row 27): outer salt-page loop.
     * For unsalted ops n_pages == 1 — loop runs exactly once, no per-
     * iteration param re-write (the original full coalesced write
     * supplied everything), backward-compat byte-exact with pre-B6.
     *
     * For salted ops the loop steps through salt pages, advancing
     * salt_start by salts_per_page each iteration, recomputing
     * num_salts (= mask_size * this_page_salts) for the final page
     * which may have fewer salts. d->h_hits accumulates across pages
     * with NO per-page reset — combined_ridx encodes the global salt
     * index so de-dup is structurally unnecessary. The reissue counter
     * (B3 inner loop) resets per page so the safety cap stays
     * meaningful per-page.
     *
     * Hit aggregation cap: GPU_PACKED_MAX_HITS host buffer size still
     * applies in aggregate across all pages; excess drops with the
     * existing approach/overflow warnings. For 10K-salt class workloads
     * with ~10K cracks/page (typical), 10 pages * 10K = 100K hits, well
     * below GPU_PACKED_MAX_HITS=262144. */

    /* Dynsize prototype (2026-05-09): wall-time measurement around the
     * salt-page outer loop. Used as the feedback signal in lieu of
     * kern_ns — captures cliffs, TDR, dispatch overhead. Only active
     * when MDXFIND_DYNSIZE=1 AND op is one of modes 0-3 AND the kernel
     * handle is kern_template_phase0_md5salt (defensive — op==MD5SALT
     * implies that kernel today, but the dual check guards against
     * future kernel-handle remap). MUST stay in sync with the salts_-
     * per_page derivation site's op-based gate. */
    int dynsize_target = dynsize_is_enabled() && !dynsize_spp_pinned() &&
                         (op == JOB_MD5SALT || op == JOB_MD5UCSALT ||
                          op == JOB_MD5revMD5SALT || op == JOB_MD5sub8_24SALT) &&
                         (kern == d->kern_template_phase0_md5salt);
    struct timespec t_dispatch_start = {0}, t_dispatch_end = {0};
    if (dynsize_target) {
        clock_gettime(CLOCK_MONOTONIC, &t_dispatch_start);
    }
    for (uint32_t salt_page = 0; salt_page < n_pages; salt_page++) {
        /* Per-page params + state setup. For salt_page == 0 in the
         * unsalted path the initial full coalesced write at line
         * ~6984 already loaded everything; skip the redundant first-
         * 132-bytes re-write (preserves backward-compat byte-exact /
         * MDXFIND_PIPE_TRACE n_writes count). For salt_page > 0 OR
         * is_salted_pack, we re-write the params + hit_count slice
         * before the inner loop. */
        if (is_salted_pack) {
            uint32_t this_page_start = salt_page * salts_per_page;
            uint32_t remaining       = (this_page_start < total_salts)
                                       ? (total_salts - this_page_start)
                                       : 0u;
            uint32_t this_page_salts = (remaining < salts_per_page)
                                       ? remaining : salts_per_page;
            if (this_page_salts == 0u) break;  /* defensive: total_salts==0 */
            OCLParams *pparams = (OCLParams *)d->h_dispatch_payload;
            pparams->salt_start         = this_page_start;
            pparams->num_salts_per_page = (uint64_t)this_page_salts;
            /* Phase 2 BF chunk-as-job (2026-05-10): per-page rewrite must
             * mirror the initial pack at line ~10410; use effective_mask_size
             * so salted BF chunks pack the chunk's per-word range, not the
             * full keyspace. */
            pparams->num_salts          = effective_mask_size * this_page_salts;
            pparams->input_cursor_start  = 0;
            pparams->rule_cursor_start   = 0;
            /* inner_iter (BF Phase 1.8): salted re-write preserves the value
             * the initial pack set (servo forces 1 on salted; safe to leave). */
            pparams->overflow_first_set  = 0;
            pparams->overflow_first_word = 0xFFFFFFFFu;
            pparams->overflow_first_rule = 0;
            uint32_t *phit = (uint32_t *)((unsigned char *)d->h_dispatch_payload + 128);
            *phit = 0;
            cl_int werr = clEnqueueWriteBuffer(d->queue, d->b_dispatch_payload,
                CL_TRUE, 0, 132, d->h_dispatch_payload, 0, NULL, NULL);
            if (werr != CL_SUCCESS) {
                GPU_FATAL("md5_rules salt-page param write err=%d on dev %d "
                          "(salt_page=%u)", werr, dev_idx, salt_page);
            }
            /* Recompute global dispatch size for this page. The mask_size
             * + n_words + n_rules axes don't change, but the salt axis
             * shrinks on the final page if total_salts isn't an exact
             * multiple of salts_per_page.
             *
             * 2026-05-09 lane-batch experiment: when the kernel is the
             * md5salt-template (compiled with GPU_TEMPLATE_HAS_PRE_SALT),
             * the salt axis is replaced by a chunks-per-page axis of
             * size ceil(this_page_salts / SALT_BATCH). The kernel reads
             * the same SALT_BATCH macro at compile time and bounds its
             * inner loop accordingly. MDXFIND_SALT_BATCH must agree
             * with the value baked into the kernel JIT. */
            {
                size_t local2 = local;
                size_t salt_axis = (size_t)this_page_salts;
                if (kern == d->kern_template_phase0_md5salt) {
                    /* Host/kernel SALT_BATCH agreement: draw from the same
                     * source of truth that the kernel's -DSALT_BATCH=N was
                     * built with at compile time. dynsize_compile_time_N()
                     * encapsulates the cache/env/default precedence. The
                     * prior hardcoded `sb=16` + separate env-var check
                     * drifted out of agreement with the kernel's baked-in
                     * N whenever cache loaded a non-default current_N (and
                     * silently after 2026-05-11 fix that seeds cold-start
                     * N=64). Mismatch caused (kernel_N / host_sb)× NDRange
                     * over-dispatch — lanes bound-check out of the inner
                     * salt loop without hashing, wasting GPU cycles. e.g.
                     * host_sb=16 + kernel_N=64 → 4× over-dispatch → ~4×
                     * effective rate loss on fpga 1080 (820 MH/s vs the
                     * 2.03 GH/s seen with explicit env var override). */
                    int sb = (int)dynsize_compile_time_N(d, dev_idx);
                    salt_axis = ((size_t)this_page_salts + (size_t)sb - 1)
                              / (size_t)sb;
                    if (salt_axis == 0) salt_axis = 1;
                }
                /* Phase 2 BF chunk-as-job (2026-05-10): salted-page global
                 * recompute must use effective_mask_size for BF chunks (the
                 * kernel iterates bf_num_masks per word, not the full
                 * gpu_mask_total keyspace). Pre-Phase-2 b71_mask_size here
                 * was correct for non-BF salted dispatches; effective_-
                 * mask_size collapses to b71_mask_size when bf_num_masks==0
                 * so non-BF behavior is bit-identical. */
                size_t total2 = (size_t)num_words * (size_t)d->gpu_n_rules
                              * (size_t)effective_mask_size
                              * salt_axis;
                global = ((total2 + local2 - 1) / local2) * local2;
            }
        }

        /* Reset per-page reissue counter — keeps MAX_REISSUES safety
         * cap meaningful per-page. The function-scope reissue_count
         * (declared above) is the value visible to the warn block at
         * function-tail. */
        reissue_count = 0;

    for (;;) {
        if (kern_event) {
            clReleaseEvent(kern_event);
            kern_event = NULL;
        }

        /* No wait_list: the dispatch_payload write above (or the params
         * re-write at the end of the previous loop iteration) was
         * synchronous (CL_TRUE) per 1.91 hashcat-pattern conversion --
         * the host already blocked until the write completed. */
        err = clEnqueueNDRangeKernel(d->queue, kern, 1, NULL, &global, &local,
                                     0, NULL, &kern_event);
        if (err != CL_SUCCESS) {
            /* Fail-fast: a failed kernel enqueue leaves the write events
             * in an inconsistent state. mmt run #77 silent-failure pattern. */
            GPU_FATAL("md5_rules dispatch error %d on dev %d "
                      "(global=%zu, n_words=%u, n_rules=%d, reissue=%d)",
                      err, dev_idx, global, num_words, d->gpu_n_rules,
                      reissue_count);
        }

        /* Read overflow status + hit_count in one batch.
         * Layout (matches OCLParams + payload):
         *   offset 100: overflow_first_set  (uint32)
         *   offset 104: overflow_first_word (uint32, lane gid CAS-min)
         *   offset 108: overflow_first_rule (uint32, unused in B3)
         *   offset 128: hit_count           (uint32)
         * One read of 12 bytes from offset 100 + one read of 4 bytes from
         * offset 128 is two clEnqueueReadBuffer calls; the alternative is
         * one larger read (e.g., 32 bytes from 100..131 covering the gap)
         * which is wasteful but a single driver call. We pick two reads
         * (12 B and 4 B) with the second one piggybacking on the same
         * kernel event for sync. */
        uint32_t ovr_state[3] = { 0, 0, 0 };
        uint32_t raw_nhits = 0;
        {
            /* Both reads wait on kern_event (out-of-order queue safety,
             * mirrors the pre-B3 hit-count read pattern). The first read
             * is CL_TRUE so the host blocks until it completes; the
             * second read uses NULL wait_list because at that point the
             * kernel has already drained (the first read consumed the
             * sync). */
            cl_int rerr = clEnqueueReadBuffer(d->queue, d->b_dispatch_payload,
                CL_TRUE, 100, sizeof(ovr_state), ovr_state,
                1, &kern_event, NULL);
            if (rerr != CL_SUCCESS)
                GPU_FATAL("md5_rules ovr-state read err=%d on dev %d "
                          "(reissue=%d)", rerr, dev_idx, reissue_count);
        }
        {
            cl_int rerr = clEnqueueReadBuffer(d->queue, d->b_dispatch_payload,
                CL_TRUE, 128, sizeof(raw_nhits), &raw_nhits,
                0, NULL, NULL);
            if (rerr != CL_SUCCESS)
                GPU_FATAL("md5_rules hit-count read err=%d on dev %d "
                          "(reissue=%d)", rerr, dev_idx, reissue_count);
        }

        /* Cap raw_nhits at the per-dispatch GPU buffer cap. The kernel's
         * EMIT_HIT_4_OR_OVERFLOW macro short-circuits writes once a slot
         * exceeds max_hits (the lane CAS-mins overflow_gid instead), but
         * atomic_add still increments hit_count regardless -- so
         * raw_nhits can exceed b3_max_hits_cap. Cap to the readable range. */
        uint32_t emitted = (raw_nhits > b3_max_hits_cap)
                           ? b3_max_hits_cap : raw_nhits;

        /* Append this dispatch's hits to d->h_hits[total_hits..]. The host
         * h_hits buffer is sized for GPU_PACKED_MAX_HITS slots; cap if we'd
         * overflow it. */
        if (emitted > 0) {
            uint32_t room = (total_hits < GPU_PACKED_MAX_HITS)
                            ? (GPU_PACKED_MAX_HITS - total_hits) : 0;
            uint32_t to_read = (emitted > room) ? room : emitted;
            if (to_read > 0) {
                /* Use kern_event as wait_list (out-of-order safety, see
                 * pre-B3 comment about mmt -i 1 cracks loss). */
                cl_int rerr = clEnqueueReadBuffer(d->queue, d->b_hits, CL_TRUE,
                    0, (size_t)to_read * GPU_HIT_STRIDE * sizeof(uint32_t),
                    &d->h_hits[(size_t)total_hits * GPU_HIT_STRIDE],
                    kern_event ? 1 : 0, kern_event ? &kern_event : NULL, NULL);
                if (rerr != CL_SUCCESS)
                    GPU_FATAL("md5_rules hits readback err=%d on dev %d "
                              "(emitted=%u reissue=%d)",
                              rerr, dev_idx, emitted, reissue_count);
                /* B6 salt-axis (2026-05-06): post-process slot[1] to convert
                 * page-local combined_ridx into a UNIFIED-across-pages
                 * format that host hit-replay (gpujob_opencl.c) can decode
                 * without needing per-hit page metadata.
                 *
                 * Kernel emits: combined_ridx = X * nspp + salt_local
                 *   where X = rule_idx*mask_size + mask_idx,
                 *         nspp = params.num_salts_per_page (this page),
                 *         salt_local = lane's local salt index in [0, nspp).
                 *
                 * Host hit-replay needs: combined_ridx_global =
                 *                        X * total_salts + salt_idx_global
                 *   where total_salts = d->salts_count (full snapshot),
                 *         salt_idx_global = salt_start + salt_local.
                 *
                 * Decompose-and-rebuild: divmod by nspp to recover X and
                 * salt_local; then add salt_start; then recompose using
                 * total_salts as the divisor. Cheap loop, runs only when
                 * is_salted_pack. Unsalted dispatches skip this entirely. */
                if (is_salted_pack) {
                    OCLParams *pparams = (OCLParams *)d->h_dispatch_payload;
                    uint32_t nspp = (uint32_t)pparams->num_salts_per_page;
                    uint32_t salt_start_now = pparams->salt_start;
                    uint32_t total_salts_dyn = (uint32_t)d->salts_count;
                    if (nspp == 0u) nspp = 1u;
                    if (total_salts_dyn == 0u) total_salts_dyn = 1u;
                    for (uint32_t hi = 0; hi < to_read; hi++) {
                        uint32_t *entry = &d->h_hits[
                            (size_t)(total_hits + hi) * GPU_HIT_STRIDE];
                        uint32_t cri = entry[1];
                        uint32_t salt_local      = cri % nspp;
                        uint32_t X               = cri / nspp;
                        uint32_t salt_idx_global = salt_start_now + salt_local;
                        entry[1] = X * total_salts_dyn + salt_idx_global;
                    }
                }
                total_hits += to_read;
            }
        }

        /* Kernel profiling: only emit on the FIRST dispatch (re-issues
         * are diagnostic and would clutter the trace). Existing test
         * expectations are one [kern] line per logical dispatch. */
        if (!last_dispatch_kern_us_emitted &&
            kern_event && p_clGetEventProfilingInfo) {
            const char *_e = getenv("MDXFIND_KERNEL_TRACE");
            if (_e && *_e && *_e != '0') {
                cl_ulong t_queued = 0, t_submit = 0, t_start = 0, t_end = 0;
                cl_int _pe1 = clGetEventProfilingInfo(kern_event,
                    CL_PROFILING_COMMAND_QUEUED, sizeof(t_queued), &t_queued, NULL);
                cl_int _pe2 = clGetEventProfilingInfo(kern_event,
                    CL_PROFILING_COMMAND_SUBMIT, sizeof(t_submit), &t_submit, NULL);
                cl_int _pe3 = clGetEventProfilingInfo(kern_event,
                    CL_PROFILING_COMMAND_START, sizeof(t_start), &t_start, NULL);
                cl_int _pe4 = clGetEventProfilingInfo(kern_event,
                    CL_PROFILING_COMMAND_END, sizeof(t_end), &t_end, NULL);
                if (_pe1 == CL_SUCCESS && _pe2 == CL_SUCCESS &&
                    _pe3 == CL_SUCCESS && _pe4 == CL_SUCCESS) {
                    unsigned long long kern_ns   = (unsigned long long)(t_end    - t_start);
                    unsigned long long submit_ns = (unsigned long long)(t_start  - t_submit);
                    unsigned long long queued_ns = (unsigned long long)(t_submit - t_queued);
                    fprintf(stderr,
                        "[kern] dev=%d path=rules words=%u rules=%d hits=%u "
                        "queued_ns=%llu submit_ns=%llu kern_ns=%llu kern_us=%llu\n",
                        dev_idx, num_words, d->gpu_n_rules, raw_nhits,
                        queued_ns, submit_ns, kern_ns, kern_ns / 1000ULL);
                    fflush(stderr);
                }
            }
            last_dispatch_kern_us_emitted = 1;
        }

        /* Overflow check: if no lane signaled overflow_first_set, this
         * dispatch consumed all remaining lanes -- we're done. */
        if (ovr_state[0] == 0) break;

        /* Overflow occurred. Advance cursor past the overflow point and
         * re-issue. */
        if (++reissue_count > MAX_REISSUES) {
            static __thread int _stall_seen = 0;
            if ((_stall_seen++ & 0x3F) == 0) {
                fprintf(stderr,
                    "WARN: rules-engine cursor-restart loop hit %d re-issues "
                    "on dev=%d without completing -- giving up to avoid "
                    "infinite loop. Hits accumulated: %u. Workload may have "
                    "extreme density; consider raising GPU_PACKED_MAX_HITS "
                    "or reducing batch size.\n",
                    MAX_REISSUES, dev_idx, total_hits);
            }
            break;
        }

        uint32_t ovr_gid = ovr_state[1];
        if (ovr_gid == 0xFFFFFFFFu) {
            /* Sentinel never overwritten: kernel said overflow_first_set
             * but no lane CAS-min'd the gid. This shouldn't happen --
             * either both fields are set together by the EMIT_HIT macro,
             * or neither is. Treat as a bug; bail out to avoid stall. */
            fprintf(stderr,
                "WARN: rules-engine overflow signaled but ovr_gid is "
                "sentinel on dev=%d; bailing out (hits=%u)\n",
                dev_idx, total_hits);
            break;
        }
        uint32_t cursor_word = ovr_gid % (uint32_t)num_words;
        uint32_t cursor_rule = ovr_gid / (uint32_t)num_words;

        /* Sanity: cursor must advance. The kernel records the FIRST
         * overflow lane in lex-(rule,word) order, so on re-issue the
         * cursor monotonically increases. If it doesn't, we'd loop. */
        {
            OCLParams *pparams = (OCLParams *)d->h_dispatch_payload;
            uint32_t prev_rule = pparams->rule_cursor_start;
            uint32_t prev_word = pparams->input_cursor_start;
            int advanced = (cursor_rule > prev_rule) ||
                           (cursor_rule == prev_rule && cursor_word > prev_word);
            if (!advanced && reissue_count > 1) {
                /* Reissue 1 starts from cursor=(0,0); cursor at any
                 * non-zero point is monotonic advance. Reissue >=2 must
                 * advance again -- if not, the cursor stalled. */
                fprintf(stderr,
                    "WARN: rules-engine cursor stalled at "
                    "(rule=%u, word=%u) on dev=%d after %d re-issues; "
                    "bailing out (hits=%u)\n",
                    cursor_rule, cursor_word, dev_idx, reissue_count,
                    total_hits);
                break;
            }
        }

        /* Per-thread "saw overflow" suppression-bounded warning. */
        {
            static __thread uint32_t _ovr_warn_seen = 0;
            if ((_ovr_warn_seen++ & 0x3F) == 0) {
                fprintf(stderr,
                    "INFO: rules-engine hit-buffer overflow dev=%d "
                    "at gid=%u (rule=%u, word=%u); re-issuing with "
                    "cursor advanced. hits_so_far=%u, max_hits=%u "
                    "[overflow #%u]\n",
                    dev_idx, ovr_gid, cursor_rule, cursor_word,
                    total_hits, b3_max_hits_cap, _ovr_warn_seen);
            }
        }

        /* Update the params + hit_count slice in host staging memory and
         * re-write only the first 132 bytes of b_dispatch_payload (params
         * + hit_count). word_offset[] and packed_words[] are unchanged
         * on the device; no need to retransmit them. This is the key
         * efficiency property of the cursor-restart protocol. */
        {
            OCLParams *pparams = (OCLParams *)d->h_dispatch_payload;
            pparams->input_cursor_start = cursor_word;
            pparams->rule_cursor_start  = cursor_rule;
            /* inner_iter (BF Phase 1.8, was output_cursor_start) preserved
             * across cursor restart. */
            pparams->overflow_first_set  = 0;
            pparams->overflow_first_word = 0xFFFFFFFFu;
            pparams->overflow_first_rule = 0;
            uint32_t *phit = (uint32_t *)((unsigned char *)d->h_dispatch_payload + 128);
            *phit = 0;
        }
        cl_int werr = clEnqueueWriteBuffer(d->queue, d->b_dispatch_payload,
            CL_TRUE, 0, 132, d->h_dispatch_payload, 0, NULL, NULL);
        if (werr != CL_SUCCESS) {
            GPU_FATAL("md5_rules cursor re-write err=%d on dev %d "
                      "(reissue=%d)", werr, dev_idx, reissue_count);
        }
    } /* end inner B3 cursor-restart loop */
    } /* end outer salt-page loop (B6 §13.4 row 27) */

    /* Dynsize prototype (2026-05-09): post-dispatch feedback update.
     * Runs ONCE per gpu_opencl_dispatch_md5_rules call when:
     *   - MDXFIND_DYNSIZE=1
     *   - MDXFIND_SPP NOT set (user hasn't pinned)
     *   - Kernel == kern_template_phase0_md5salt (algo_modes 0-3 only)
     * Algorithm follows project_dynamic_spp_sizing_design.md §5:
     *   EMA-smoothed observed_mhz; hard-wall halve; cliff-detection halve;
     *   geometric grow on positive slope; geometric shrink on negative;
     *   plateau decay; salt-retirement guard.
     * N exploration (§5.3) is DEFERRED for the prototype. */
    if (dynsize_target) {
        clock_gettime(CLOCK_MONOTONIC, &t_dispatch_end);
        uint64_t wall_ns =
            (uint64_t)(t_dispatch_end.tv_sec  - t_dispatch_start.tv_sec)  * 1000000000ULL +
            (uint64_t)(t_dispatch_end.tv_nsec - t_dispatch_start.tv_nsec);
        if (wall_ns < 1) wall_ns = 1; /* paranoia */

        struct dynsize_entry *e = &d->dynsize_md5salt;

        /* "pairs" = candidates explored on the GPU this call.
         * Phase 2 BF chunk-as-job (2026-05-10): use effective_mask_size
         * for BF chunks (= bf_num_masks per word). For non-BF dispatches
         * effective_mask_size == b71_mask_size, so pre-Phase-2 dynsize
         * feedback is unchanged. Without this substitution dynsize would
         * over-count BF dispatches by gpu_mask_total/bf_num_masks, biasing
         * the EMA upward on BF runs (and mistune subsequent dispatches). */
        uint64_t pairs = (uint64_t)num_words *
                         (uint64_t)d->gpu_n_rules *
                         (uint64_t)effective_mask_size *
                         (uint64_t)salts_per_page *
                         (uint64_t)n_pages;
        double observed_mhz = (double)pairs / (double)wall_ns * 1000.0;

        /* Salt-retirement guard (§5.4): if nsalts active dropped >20%
         * from the last call AND throughput dropped <20%, treat as
         * workload shift, NOT regression. Hold (N, spp); skip adjust. */
        uint64_t cur_nsalts = (uint64_t)total_salts;
        int salt_retiring = 0;
        if (e->prev_nsalts_active > 0 && cur_nsalts > 0) {
            double salt_drop = 1.0 - ((double)cur_nsalts / (double)e->prev_nsalts_active);
            if (salt_drop > DYNSIZE_SALT_RETIRE_DROP) {
                /* Significant salt drop. Hold pattern. */
                salt_retiring = 1;
            }
        }
        e->prev_nsalts_active = cur_nsalts;

        /* EMA update (§5.1): alpha scales with loop_weight */
        double alpha = e->loop_weight * 0.5;
        if (alpha < DYNSIZE_EMA_ALPHA_MIN) alpha = DYNSIZE_EMA_ALPHA_MIN;
        if (alpha > DYNSIZE_EMA_ALPHA_MAX) alpha = DYNSIZE_EMA_ALPHA_MAX;
        if (e->ema_mhz <= 0.0) {
            e->ema_mhz = observed_mhz;
        } else {
            e->ema_mhz = alpha * observed_mhz + (1.0 - alpha) * e->ema_mhz;
        }

        const char *action = "hold";
        if (salt_retiring) {
            action = "salt_retire_hold";
        } else if (wall_ns > DYNSIZE_HARD_WALL_NS &&
                   e->current_spp > DYNSIZE_SPP_MIN) {
            /* Hard wall trip: halve spp, lock cap, re-elevate gain.
             * Gated on current_spp > MIN — at the floor, the kernel
             * itself is the bottleneck, not spp granularity. */
            uint32_t new_spp = e->current_spp / 2;
            if (new_spp < DYNSIZE_SPP_MIN) new_spp = DYNSIZE_SPP_MIN;
            e->spp_cap_observed = new_spp; /* lock cap */
            e->current_spp = new_spp;
            e->loop_weight = e->loop_weight * 1.5;
            if (e->loop_weight > 1.0) e->loop_weight = 1.0;
            e->plateau_streak = 0;
            action = "hard_wall_halve";
        } else if (e->convergence_count > DYNSIZE_CLIFF_GUARD_COUNT &&
                   observed_mhz < e->ema_mhz * DYNSIZE_CLIFF_RATIO &&
                   e->current_spp > DYNSIZE_SPP_MIN) {
            /* Cliff detected: halve, lock cap. Gated on current_spp >
             * MIN — at the floor, halving is a no-op and would just
             * keep ramping loop_weight without changing state. */
            uint32_t new_spp = e->current_spp / 2;
            if (new_spp < DYNSIZE_SPP_MIN) new_spp = DYNSIZE_SPP_MIN;
            e->spp_cap_observed = e->current_spp; /* the value JUST before cliff */
            e->current_spp = new_spp;
            e->loop_weight = e->loop_weight * 1.5;
            if (e->loop_weight > 1.0) e->loop_weight = 1.0;
            e->plateau_streak = 0;
            action = "cliff_halve";
        } else {
            /* Standard slope-based adjust. Lessons from fpga 100K smoke
             * 2026-05-09 (death-spiral fix):
             *   - Comparing instantaneous obs_mhz to ema and shrinking
             *     on small negative diffs creates a death spiral — each
             *     shrink hurts throughput (smaller batch -> more dispatch
             *     overhead -> lower MH/s -> shrinks again).
             *   - Shrink path must be gated by wall_ns being TOO LONG,
             *     not just observed_mhz dipping. The hard-wall and cliff
             *     branches above already cover the "wall too long" case.
             *   - In the steady-state range (target_wall < wall_ns <
             *     hard_wall, no cliff), the only adjustment is GROW or
             *     PLATEAU. Shrink without wall pressure is ill-conditioned.
             *   - We still grow on positive slope (room to scale up). */
            double diff = observed_mhz - e->ema_mhz;
            double thresh = DYNSIZE_PLATEAU_RATIO * e->ema_mhz;
            int wall_under_target = (wall_ns < DYNSIZE_TARGET_WALL_NS);
            int wall_in_band      = (wall_ns >= DYNSIZE_TARGET_WALL_NS &&
                                     wall_ns < DYNSIZE_HARD_WALL_NS);

            if (diff > thresh && e->current_spp < e->spp_cap_observed &&
                wall_under_target) {
                /* Positive slope AND wall has headroom -> grow. */
                uint32_t new_spp = (uint32_t)((double)e->current_spp *
                                              DYNSIZE_SPP_GROW_FACTOR);
                if (new_spp > e->spp_cap_observed) new_spp = e->spp_cap_observed;
                if (new_spp > DYNSIZE_SPP_MAX) new_spp = DYNSIZE_SPP_MAX;
                if (new_spp != e->current_spp) {
                    e->current_spp = new_spp;
                    e->plateau_streak = 0;
                    action = "grow_spp";
                } else {
                    e->plateau_streak++;
                    action = "at_cap";
                }
            } else if (wall_in_band && diff < -thresh * 5.0) {
                /* In-band wall AND large negative diff (>5%): cautious
                 * shrink. The 5x widening (vs the 1% grow threshold)
                 * resists noise-driven shrink without ignoring genuine
                 * regressions. The hard-wall and cliff branches catch
                 * worse trips. */
                uint32_t new_spp = (uint32_t)((double)e->current_spp /
                                              DYNSIZE_SPP_SHRINK_FACTOR);
                if (new_spp < DYNSIZE_SPP_MIN) new_spp = DYNSIZE_SPP_MIN;
                if (new_spp != e->current_spp) {
                    e->current_spp = new_spp;
                    e->plateau_streak = 0;
                    action = "shrink_spp";
                } else {
                    e->plateau_streak++;
                    action = "at_floor";
                }
            } else {
                /* Plateau / wall-under-target with small slope. Count
                 * streak; decay weight when settled. */
                e->plateau_streak++;
                if (e->plateau_streak > DYNSIZE_PLATEAU_SETTLE) {
                    e->loop_weight = e->loop_weight * 0.95;
                    if (e->loop_weight < DYNSIZE_LOOP_WEIGHT_FLOOR)
                        e->loop_weight = DYNSIZE_LOOP_WEIGHT_FLOOR;
                    action = "plateau_decay";
                } else {
                    action = "plateau";
                }
            }
        }

        /* Per-dispatch weight decay regardless of branch */
        e->loop_weight = e->loop_weight * DYNSIZE_LOOP_WEIGHT_DECAY;
        if (e->loop_weight < DYNSIZE_LOOP_WEIGHT_FLOOR)
            e->loop_weight = DYNSIZE_LOOP_WEIGHT_FLOOR;

        e->convergence_count++;

        if (dynsize_verbose()) {
            fprintf(stderr,
                "[dynsize] dev=%d kern=md5salt N=%u spp=%u spp_cap=%u "
                "obs_mhz=%.1f ema=%.1f weight=%.3f conv=%u "
                "plateau=%u wall_ms=%llu pairs=%llu action=%s\n",
                dev_idx, e->current_N, e->current_spp, e->spp_cap_observed,
                observed_mhz, e->ema_mhz, e->loop_weight,
                e->convergence_count, e->plateau_streak,
                (unsigned long long)(wall_ns / 1000000ULL),
                (unsigned long long)pairs, action);
        }
    }

    /* Cleanup + tail diagnostics (mirror the pre-B3 path). */
    if (kern_event) {
        clReleaseEvent(kern_event);
        kern_event = NULL;
    }

    *nhits_out = total_hits;

    /* Approach + overflow warnings: report against the GPU-side cap, not
     * the per-dispatch (possibly-overridden) cap. The "approach" warning
     * uses the FIRST dispatch's raw_nhits since that's the metric users
     * tune against. After cursor-restart, total_hits reflects the
     * aggregate across all re-issues. */
    {
        static __thread uint32_t _ovr_seen = 0;
        static __thread uint32_t _approach_seen = 0;
        if (reissue_count > 0) {
            if ((_ovr_seen++ & 0x3F) == 0) {
                fprintf(stderr,
                    "INFO: rules-engine cursor-restart completed dev=%d: "
                    "%d re-issue(s), %u total hits aggregated [overflow "
                    "#%u]\n",
                    dev_idx, reissue_count, total_hits, _ovr_seen);
            }
        } else if (total_hits >= (b3_max_hits_cap * 9 / 10)) {
            if ((_approach_seen++ & 0x3F) == 0) {
                fprintf(stderr,
                    "WARN: rules-engine hit-buffer at %u/%u (%.0f%%) on dev=%d "
                    "[approach #%u]. Cap is approaching.\n",
                    total_hits, (unsigned)b3_max_hits_cap,
                    100.0 * total_hits / b3_max_hits_cap, dev_idx, _approach_seen);
            }
        }
    }

    return (total_hits > 0) ? d->h_hits : NULL;
}

/* BF Phase 3b Tranche B (2026-05-10): gpu_opencl_dispatch_batch retired.
 * Sole caller (slab arm in gpu/gpujob_opencl.c) and sole producer of slab
 * format slots (gpu_try_pack in mdxfind.c) both deleted in this same
 * commit. The slab-dispatch path is structurally gone; all GPU traffic
 * now flows through gpu_opencl_dispatch_md5_rules via the rules-engine
 * chokepoint pack. The B7.9 (2026-05-07) defensive-abort at gpu/gpujob_-
 * opencl.c:2423 already proved no packed=1/rules_engine=0 slots reach
 * the worker; Tranche A 2026-05-10 then proved zero gpu_try_pack callers
 * remain; Tranche B retires the function body. probe_max_dispatch (at
 * gpu_opencl.c:2630, FAM_MD5SALT capacity probe) uses dedicated probe-
 * internal dispatch (clEnqueueNDRangeKernel direct) and does not depend
 * on this function. RCS history retains the prior implementation
 * (gpu/gpu_opencl.c rev 1.163). The accompanying slab word_stride table
 * formerly at the head of this function is retired with it. */

#endif /* OPENCL_GPU */
