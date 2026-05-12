/*
 * gpujob.h — Dedicated GPU worker thread for mdxfind Metal acceleration
 *
 * Receives batches of pre-hashed words from procjob() threads,
 * dispatches them to the GPU as a single Metal command.
 * Only available on Apple Silicon with METAL_GPU defined.
 */

#ifndef GPUJOB_H
#define GPUJOB_H

#if (defined(__APPLE__) && defined(METAL_GPU)) || defined(CUDA_GPU) || defined(OPENCL_GPU)

#include <stdio.h>
#include <stdint.h>

/* Kernel family IDs for gpu_opencl_compile_families() bitmask */
enum {
    FAM_MD5SALT, FAM_MD5SALTPASS, FAM_MD5ITER, FAM_PHPBB3,
    FAM_MD5CRYPT, FAM_MD5_MD5SALTMD5PASS, FAM_SHA1, FAM_SHA256,
    FAM_MD5MASK, FAM_DESCRYPT, FAM_MD5UNSALTED, FAM_MD4UNSALTED,
    FAM_SHA1UNSALTED, FAM_SHA256UNSALTED, FAM_SHA512UNSALTED,
    FAM_WRLUNSALTED, FAM_MD6256UNSALTED, FAM_KECCAKUNSALTED,
    FAM_HMAC_SHA512, FAM_MYSQL3UNSALTED, FAM_HMAC_RMD160, FAM_HMAC_RMD320,
    FAM_HMAC_BLAKE2S, FAM_STREEBOG, FAM_SHA512CRYPT, FAM_SHA256CRYPT,
    FAM_RMD160UNSALTED, FAM_BLAKE2S256UNSALTED, FAM_BCRYPT, FAM_MD5PACKED, FAM_COUNT
};

/* Forward declarations for mdxfind types */
struct job;
union HashU;

/* Slot pool identity: two disjoint pools to prevent buffer-size mismatch
 * across the legacy chokepoint and rules-engine packing paths. */
enum jobg_kind { JOBG_KIND_LEGACY = 0, JOBG_KIND_RULES = 1 };

#ifdef __cplusplus
extern "C" {
#endif

/* Checked malloc: zeroes memory, exits on failure */
void *malloc_lock(size_t size, const char *reason);

#define GPUBATCH_MAX   512
#define GPUBATCH_RULE_MAX 4192  /* max words for rule-packing MD5 path */
#define GPUBATCH_PASS  (GPUBATCH_RULE_MAX * 64)  /* 262KB: fits GPUBATCH_RULE_MAX at stride 64 */

/* Packed password buffer for GPU rule dispatch */
#define GPUBATCH_PACKED_SIZE (128 * 1024 * 1024)  /* 128MB per buffer */
#define GPUBATCH_PACKED_MAXWORDS (128 * 1024 * 1024)  /* limited by buffer, not word count */

/* GPU algorithm categories */
#define GPU_CAT_NONE     0   /* not GPU-capable */
#define GPU_CAT_SALTED   1   /* salted: GPU does MD5(hex_hash + salt) per salt */
#define GPU_CAT_ITER     2   /* unsalted iterated: GPU does MD5(hex) iterations */
#define GPU_CAT_SALTPASS 3   /* salted: GPU does MD5(salt + raw_password) per salt */
#define GPU_CAT_MASK     4   /* unsalted + mask: GPU generates mask candidates */
#define GPU_CAT_UNSALTED 5   /* unsalted pre-padded: GPU fills masks into M[] */

#define GPU_MAX_PASSLEN  55  /* max password length for GPU (single MD5 block with salt) */
#define GPU_PACKED_MAX_HITS 1048576 /* hit buffer size for packed dispatch.
                                     * Sized for 16K-word rules-engine batches at
                                     * 100k rules — observed avg 57 hits/Mcandidates
                                     * on rockyou×HashMob.100k, so 16K×100K=1.638B
                                     * candidates ≈ 93K hits avg / ~465K hits at 5×
                                     * worst-case density. The rules-engine path
                                     * (gpujob_opencl.c:603) is single-shot: no
                                     * chunking or retry; if the kernel emits more
                                     * hits than this cap, slots beyond the cap
                                     * have their data silently dropped (cracks
                                     * lost). Per-device GPU buffer: 1M × 19 ×
                                     * 4 bytes = 76MB; host pinned buffer same.
                                     * Was 262144 prior to rev 1.364; cracks were
                                     * never observed to be lost at the smaller
                                     * cap (max batch was 64 words pre-rev-1.362).
                                     * gpu_opencl.c warns on stderr if a dispatch
                                     * comes within 90% of this cap. */

/* Universal GPU hit entry stride: 19 uint32 words per hit.
 * Layout: [0]=word_idx [1]=salt_idx [2]=iter_num [3..18]=hash[0..15]
 * All kernels emit this format. No exceptions. */
#define GPU_HIT_STRIDE   19

/* Returns GPU category for an op code, or GPU_CAT_NONE */
int gpu_op_category(int op);

/* Returns 1 if op has any GPU support */
int is_gpu_op(int op);

/* Returns the number of uint32 hash words this algorithm produces.
 * Used to determine hit_stride, hash_words, and hexlen for GPU hits.
 * hexlen = hash_words * 8; hit_stride = 2 + hash_words (+ 1 if iter). */
int gpu_hash_words(int op);

struct jobg {
    /* === 8-byte fields: pointers + size_t + uint64 (10 × 8 = 80 bytes) === */
    struct jobg *next;
    char        *filename;
    int         *doneprint;
    char        *packed_buf;               /* 128MB malloc'd buffer: [len][data]... */
    uint32_t    *word_offset;              /* byte offset per word */
    size_t       packed_buf_size;          /* allocation in bytes for packed_buf */
    /* Pipeline-phase timestamps (microseconds since process start, set by
     * gpu_now_us()). Only meaningful when MDXFIND_PIPE_TRACE is on; otherwise
     * the fields are stamped but not printed. Each phase prints one [pipe]
     * stderr line keyed by the slot's address (g=0x...) so deltas can be
     * correlated post-run via grep:
     *   t_acquired     = procjob got this slot from the rules-engine free pool.
     *   t_added        = gpujob_submit() pushed this slot onto GPUWorkWaiting.
     *   t_dispatched   = gpujob worker popped this slot off GPUWorkWaiting.
     *   t_return_start = gpujob worker began returning the slot to its pool.
     * Phase deltas: fill_us = t_added-t_acquired, queue_us = t_dispatched-t_added,
     *   disp_us = t_return_start-t_dispatched, ret_us = now-t_return_start. */
    uint64_t     t_acquired;
    uint64_t     t_added;
    uint64_t     t_dispatched;
    uint64_t     t_return_start;
    /* BF chunk-as-job (Tranche 1 plumbing, 2026-05-09): host-side cursor base
     * accumulated as uint64 across chunks; kernel sees only uint32 in-chunk
     * indices. bf_chunk == 0 (default) preserves existing behavior; activation
     * arrives in Tranche 3. */
    uint64_t     bf_mask_start;             /* BF chunk: base mask cursor */

    /* === 4-byte fields === */
    int          flags;
    int          op;                        /* JOB_MD5SALT etc. */
    int          count;                     /* entries filled (0..GPUBATCH_RULE_MAX) */
    int          max_count;                 /* max entries for this batch (stride-dependent) */
    uint32_t     passbuf_pos;               /* fill cursor into passbuf */
    uint32_t     word_stride;               /* bytes per word slot (64, 128, 256) */
    uint32_t     packed_count;              /* words packed so far */
    uint32_t     packed_pos;                /* byte fill cursor in packed_buf */
    uint32_t     word_offset_entries;       /* allocation in entries for word_offset */
    int          packed;                    /* 1 = packed format, 0 = legacy */
    int          rules_engine;              /* 1 = dispatch via gpu_opencl_dispatch_md5_rules;
                                             * decode hits via applyrule(words[widx], gpu_rule_origin[ridx]).
                                             * 0 = existing packed dispatch via gpu_opencl_dispatch_packed.
                                             * Mutually exclusive with packed dispatch semantics, but
                                             * shares packed_buf/word_offset/packed_count/packed_pos. */
    uint32_t     bf_offset_per_word;        /* BF chunk: per-word stride (==mask_size_per_word*inner_iter for contiguous coverage) */
    uint32_t     bf_num_masks;              /* BF chunk: per-iter mask range size (kernel mask_size) */
    uint32_t     bf_inner_iter;             /* BF Phase 1.8 chunk: per-lane mask iterations.
                                             * 0 or 1 = today's behavior (bit-identical). Cap=16.
                                             * Set by adaptive_bf_chunk_size servo; copied into
                                             * OCLParams.inner_iter at dispatch. Unsalted BF only;
                                             * servo forces 1 when salts_per_page > 1. */

    /* === 1-byte === */
    unsigned char slot_kind;                /* JOBG_KIND_LEGACY or JOBG_KIND_RULES; set once at
                                             * gpujob_init, never mutated. Determines which free-list
                                             * the slot returns to and which buffer sizes apply. */
    unsigned char bf_chunk;                 /* BF chunk: 1 = this slot is a BF chunk; 0 = normal. Default 0. */
    /* Phase 1.9 Tranche A1 (2026-05-10): when 1, this dispatch may use
     * the gpu_md5_bf.cl fast template kernel
     * (kern_template_phase0_md5_bf) instead of the generic slow MD5
     * template (kern_template_phase0). Set by the BF chunk producer in
     * mdxfind.c when (op==JOB_MD5) && (Numrules<=1) && unsalted &&
     * (npre==0) && (napp in [1,8]) AND env MDXFIND_GPU_FAST_DISABLE is
     * unset. The dispatch path in gpu_opencl_dispatch_md5_rules swaps
     * to the fast kernel after the standard kernel resolve, only if
     * the fast kernel compile/lazy succeeded; otherwise it stays on the
     * slow kernel (silent self-heal, warning already emitted by the
     * compile helper). Default 0 from slot-reset memsets — same shape
     * as bf_chunk. */
    unsigned char bf_fast_eligible;

    /* === Per-batch metadata arrays — 4-byte elements === */
    uint32_t     passoff[GPUBATCH_RULE_MAX];
    int          clen[GPUBATCH_RULE_MAX];
    int          ruleindex[GPUBATCH_RULE_MAX];

    /* === Per-batch metadata arrays — 2-byte elements === */
    uint16_t     hexlen[GPUBATCH_RULE_MAX];
    uint16_t     passlen[GPUBATCH_RULE_MAX];

    /* === Large word-data buffer (16-aligned for SIMD), placed last for
     *     better cache locality of the metadata fields above. GPUBATCH_RULE_MAX
     *     * 64 = 262KB for MD5 rule-packing path. Legacy accesses first half
     *     as passbuf, second half as hexhash[]. === */
    union {
        char     raw[GPUBATCH_PASS + GPUBATCH_MAX * 256]; /* legacy size */
        struct { char passbuf[GPUBATCH_PASS]; char hexhash[GPUBATCH_MAX][256]; };
    } __attribute__((aligned(16)));
};

/* Initialize GPU work queue, allocate JOBG structs, launch gpujob thread.
 * Call from main() after metal_md5salt_init() and set_compact_table().
 * Returns 0 on success, -1 on failure. */
int gpujob_init(int num_jobg);

/* Pre-load overflow data to every GPU device at session start, before
 * procjob threads launch. Eliminates the worker-thread CPU-contention
 * window between warm-probe and first dispatch. Idempotent. Call from
 * main() after build_compact_table() has run gpu_opencl_init(). */
void gpujob_overflow_preload_all(void);

/* Rules-engine slot buffer sizing. The chokepoint hook clamps
 * max_words_per_batch <= 16384 (per mdx-gpu's 1M-lane target divided
 * by gpu_rule_count, floor 64 for very-large rule sets). Buffers are
 * sized to that ceiling; rules-engine slots only hold ORIGINAL words,
 * not expanded (word, rule) plaintexts, so they're 32x smaller than
 * the legacy GPUBATCH_PACKED_SIZE. Used only when gpu_legacy_slot_unused
 * is set (rl.ncpu == 0 + auto-`:` injected). For mixed-mode workloads
 * the legacy GPUBATCH_PACKED_SIZE is still required because slots are
 * pool-shared and may be used as legacy for CPU-rule expansions. */
#define GPU_RULES_MAX_WORDS_PER_BATCH 16384
#define GPUBATCH_RULES_PACKED_SIZE   (GPU_RULES_MAX_WORDS_PER_BATCH * 256)
/* Maximum input-word length accepted by the GPU rules-engine pack path
 * in mdxfind.c. MUST match RULE_BUF_LIMIT in gpu/gpu_md5_rules.cl
 * (currently 40959 = RULE_BUF_MAX(40960) - 1). The walker's private buf
 * is sized RULE_BUF_MAX bytes; longer inputs would overflow apply_rule's
 * per-op bounds checks. Bumping here requires bumping RULE_BUF_LIMIT
 * in gpu_md5_rules.cl in the same commit. Bumped 256→512 on 2026-05-01
 * to fix 303-byte rule-output truncation (ioblade run #76, one missed
 * crack vs canonical 21,289). Bumped 512→40959 on 2026-05-02 to track
 * CPU's MAXLINE=40*1024 in mdxfind.h:63 — eliminates GPU/CPU clamp
 * divergence on long rule outputs (e.g. p9 × 67-char input = 670 B). */
#define GPU_RULES_MAX_INPUT_LEN 40959
#define GPUBATCH_RULES_WOFF_SIZE     (GPU_RULES_MAX_WORDS_PER_BATCH * sizeof(uint32_t))

/* Send JOB_DONE to GPU queue and join the gpujob thread.
 * Call from main() after all procjob threads have exited. */
void gpujob_shutdown(void);

/* Get a free JOBG from the legacy chokepoint pool. Priority scheduling
 * ensures earlier lines in the file get GPU buffers first. Pass NULL
 * filename for shutdown sentinels (bypasses scheduling). */
struct jobg *gpujob_get_free(char *filename, unsigned long long startline);

/* Get a free JOBG from the rules-engine pool. Same semantics as
 * gpujob_get_free, but services a separate pool sized for the rules
 * engine path. Returns NULL if the rules pool is unavailable (e.g.,
 * gpu_rule_count == 0 at init time). */
struct jobg *gpujob_get_free_rules(char *filename, unsigned long long startline);

/* Non-blocking: returns NULL immediately if no free buffer.
 * Used by hybrid types where CPU fallback is preferred over waiting. */
struct jobg *gpujob_try_get_free(void);

/* Submit a filled JOBG to the GPU work queue. */
void gpujob_submit(struct jobg *g);

/* Return a JOBG to the free list without submitting for work. */
void gpujob_return_free(struct jobg *g);

/* Returns 1 if GPU job system is initialized and ready. */
int gpujob_available(void);

/* Returns per-device batch limit (min across all GPUs). */
int gpujob_batch_max(void);

/* Returns number of batches waiting in GPU work queue. */
int gpujob_queue_depth(void);

/* Returns number of free jobg buffers available. */
int gpujob_free_count(void);

/* Emit one-line per-device dispatch share to fp, e.g.:
 *   GPU share: [0]=56.1% [1]=43.9%
 * Computed from per-device hash totals (words × rule_count + legacy
 * entries + slab hashes, all × Maxiter). Devices with zero dispatches
 * are skipped. If >8 devices, abbreviates to top 4 + "..." + bottom 1.
 * Safe to call after gpujob_shutdown(). */
void gpujob_print_share_line(FILE *fp);

#ifdef __cplusplus
}
#endif

#endif /* (__APPLE__ && METAL_GPU) || CUDA_GPU || OPENCL_GPU */
#endif /* GPUJOB_H */
