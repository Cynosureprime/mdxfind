/*
 * gpujob.h — Dedicated GPU worker thread for mdxfind GPU acceleration
 *
 * Receives batches of pre-hashed words from procjob() threads,
 * dispatches them to the GPU as a single command.
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
     * (npre==0) && (napp in [1,8]).  An MDXFIND_GPU_FAST_DISABLE env
     * var could also veto it; removed 2026-09-16.  The dispatch path
     * in gpu_opencl_dispatch_md5_rules swaps
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
 * Call from main() after the GPU backend's init + set_compact_table.
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

/* PER-DISPATCH ROUND BUDGET (A1131, 2026-10-30).
 *
 * The full-slot packing above is a WORD count, and every measurement behind
 * it was taken at -i 1, where one dispatch costs num_words * n_rules. But
 * gpu_md5_rules.cl md5_rules_phase0 runs the iteration loop INSIDE each work
 * item ("for (iter = 1; iter <= max_iter; iter++)"), so the real cost is
 *
 *     num_words * n_rules * max_iter
 *
 * and the iteration multiplier was never folded into the batch size. At
 * 16384 words x 25000 rules x -i 9999 that is 4.1e12 hash rounds in ONE
 * dispatch: minutes long on a 3060 Ti, with no progress line and no point at
 * which the run can be interrupted or reported on.
 *
 * Shrinking the word axis here does NOT reintroduce the host-side overhead
 * that the 16384 removed. That overhead (~670us per dispatch) matters only
 * when a dispatch is MILLISECONDS; a large max_iter makes it seconds whatever
 * the word count, so there is nothing left for a big batch to amortise. The
 * budget therefore serves the intent of the full-slot change rather than
 * reverting it -- it binds only where that change was never measured.
 *
 * The budget is set ABOVE the largest dispatch any tuned configuration
 * produces today (16384 x 100K rules at -i 1 = 1.638e9 rounds), so no
 * measured workload changes shape: every -i 1 run, and every run whose
 * rules x iter product stays under the budget, still packs to 16384. Only
 * the pathological product moves.
 *
 * The floor exists ONLY so the division can never yield a zero-word batch.
 * It is deliberately 1 and not some "worth a dispatch" figure: when
 * n_rules * max_iter alone exceeds the budget, a single word already carries
 * a full budget of work, so any larger floor just multiplies a dispatch that
 * was already too big. A 32-word floor let 100K rules at -i 9999 through at
 * 3.2e10 rounds, 4x over budget, for no benefit. Occupancy is not the reason
 * to raise it either -- the budget keeps the grid wide whenever the rule count
 * is what is large, and the existing "below lane target" warning already
 * covers a genuinely narrow grid.
 */
#define GPU_RULES_DISPATCH_ROUND_BUDGET  8000000000LL
#define GPU_RULES_MIN_WORDS_PER_BATCH    1

/* The 1-D NDRange ceiling, 2 GiW.  Moved here from mdxfind.c so the host-side
 * dot-width clamp in gpu/gpu_opencl.c reads the SAME number the mask fan-out
 * producer enforces -- two copies of a grid ceiling is how a dispatch gets
 * admitted that the kernel's own 32-bit arithmetic cannot represent, which is
 * the silent-wrong-plaintext failure plan section 6 is about. */
#ifndef MDX_NDRANGE_HARD
#define MDX_NDRANGE_HARD ((unsigned long long)1 << 31)
#endif

/* ---- word_offset bit layout, HOST side ------------------------------------
 *
 * These MUST equal the device definitions in gpu/gpu_common.cl and
 * gpu/metal_common.metal.  They cannot be shared by #include -- a .cl is not
 * a C translation unit -- so the layout is described in two places and the
 * regress suite asserts the two agree (regress/regress-email.sh).  Do not
 * change one without the other.
 *
 *   bits  0-21  offset into the packed-words region (2^22 = 16384 x 256)
 *   bits 22-29  head_len + 1, or 0 when untagged
 *   bits 30-31  UTF-32 word class
 */
#define WOFF_H_OFF_MASK    0x003fffffu
#define WOFF_H_HEAD_SHIFT  22u
#define WOFF_H_HEAD_MASK   0xffu
#define WOFF_H_CLASS_SHIFT 30u
/* The head_len field, kept as-is -- for a write-back that must not disturb it. */
#define WOFF_H_HEAD_FIELD  (WOFF_H_HEAD_MASK << WOFF_H_HEAD_SHIFT)

/* ---- -a: normalize an address, and measure its sweep width ----------------
 *
 * ONE definition, two callers: procjob's -a setup and the chokepoint pack site.
 * They must agree, and they did not have to before -a reached the GPU -- the
 * chokepoint packs at mdxfind.c ~15846 while the -a setup normalizes at ~15979,
 * so the device would otherwise be handed the RAW word and would sweep dots
 * onto an already-dotted local.  That is exactly the defect phase 2 fixed on
 * the CPU side (a reversed fastcopy discarding the normalization), and it is
 * not a bug worth having twice.
 *
 * Writes the dotless local part followed by the domain verbatim, returns the
 * NORMALIZED total length, and reports the local length through *head.  dst
 * needs room for srclen bytes; normalizing only shrinks. */
static inline int gpu_email_normalize(const char *src, int srclen,
                                      char *dst, int *head)
{
    int i, n = 0, at = srclen;
    for (i = 0; i < srclen; i++) {
        if (src[i] == '@') { at = i; break; }
        if (src[i] == '.') continue;
        dst[n++] = src[i];
    }
    if (head) *head = n;
    for (; i < srclen; i++) dst[n++] = src[i];   /* '@' and the domain, verbatim */
    return n;
}

/* The dot-sweep width a word needs: its normalized local length, capped.  The
 * cap matches EMAIL_DOT_WIDTH_MAX in mdxfind.c and WOFF_DOT_WIDTH_MAX in the
 * two device commons; the regress suite asserts they agree. */
#ifndef GPU_EMAIL_DOT_WIDTH_MAX
#define GPU_EMAIL_DOT_WIDTH_MAX 24

/* The mask-axis size the LAST dispatch on a device actually used -- including
 * the -a dot radix and AFTER the grid clamp.  Indexed by device slot.
 *
 * The hit replay has to decode combined_ridx against the same base the kernel
 * packed it with, and it cannot re-derive that base: the dot width comes from
 * the widest local part in the batch and is then clamped against the grid, so
 * repeating both steps in the replay would be a second opinion about which
 * dotting ran -- and when the two opinions differ the run emits a CORRECT
 * digest beside a plaintext that does not hash to it.  Written once where it is
 * computed, read once where it is needed.
 *
 * Defined in mdxfind.c, which is in every build, rather than in a backend: the
 * two replays and the two dispatches are four translation units and only
 * mdxfind.c is common to all of them. */
#define GPUJOB_MAX_DEVS 64
extern uint64_t gpu_dispatch_mask_size[GPUJOB_MAX_DEVS];
/* The global salt index of each hit of a salted template dispatch, parallel to
 * the returned hit records, and how many of them are valid for the dispatch the
 * replay is decoding (0 = this dispatch published none; decode slot[1] as
 * before).  Slot[1] then holds only rule*mask_size + mask_idx.
 *
 * It used to hold that times the FULL salt count plus the salt index, in one
 * uint32.  A brute-force chunk has mask_size 131072, so the product wrapped once
 * a run had more than 32768 salts: the digest matched, the replay decoded the
 * wrong salt and candidate, printed a line that does not verify, and the hash
 * was marked solved.  The two axes need ~37 bits together; neither 32-bit field
 * spare in the 19-word hit record (iter is the -i depth) can carry the rest.
 * The workers zero the count before every dispatch, so a proto/kernel-B
 * dispatch, which never sets it, can never decode against a stale array. */
extern uint32_t *gpu_dispatch_hit_salt[GPUJOB_MAX_DEVS];
extern uint32_t gpu_dispatch_hit_salt_n[GPUJOB_MAX_DEVS];
#endif
static inline int gpu_email_dot_width(const char *src, int srclen)
{
    int i, n = 0;
    for (i = 0; i < srclen; i++) {
        if (src[i] == '@') break;
        if (src[i] != '.') n++;
    }
    return n > GPU_EMAIL_DOT_WIDTH_MAX ? GPU_EMAIL_DOT_WIDTH_MAX : n;
}

/* ---- -E / -a: tag each word's local-part length into word_offset ----------
 *
 * The device needs to know, per word, how much of it mangling may touch: under
 * -E and -a the rules, the mask and the dot axis act on the LOCAL PART and the
 * domain is appended unchanged.  That length rides word_offset bits 22-29.
 * The full layout, and the reason every reader must mask, is documented once
 * in gpu/gpu_common.cl and mirrored in gpu/metal_common.metal.
 *
 * STORED BIASED BY ONE.  0 means untagged, so an untouched array reads as
 * "the whole word is the head" and a non-email run needs no test at all.  The
 * bias is load-bearing rather than tidy: an address beginning with '@' has a
 * genuine head_len of ZERO, so an unbiased field could not tell "the local
 * part is empty" from "there is no tag", and would hand the kernel the whole
 * address as the head for every such word.
 *
 * CALL IT ON THE PAYLOAD COPY, NEVER ON THE JOBG SLOT'S ARRAY.  The slot's
 * word_offset[] is reused across dispatches and is read back by the host hit
 * replay, which masks nothing; tagging it in place would corrupt the plaintext
 * of every emitted crack.  gpu_u32_tag_word_offsets carries the same warning
 * for the class tag and is the pattern to follow -- tag the staging buffer at
 * staging + 132, which is where both callers already point.
 *
 * `packed` is the packed-words region: each word a 2-byte little-endian length
 * followed by its bytes, exactly as the pack site writes it.
 *
 * Byte 0x40 cannot occur inside a multi-byte UTF-8 sequence -- lead bytes are
 * 0xC2 and above, continuations 0x80 to 0xBF -- so scanning bytes for '@' is
 * exactly as correct as scanning codepoints, and -8 needs no variant. */
static inline void gpu_tag_head_lens(uint32_t *woff, uint32_t n_words,
                                     const unsigned char *packed)
{
    uint32_t i;
    for (i = 0; i < n_words; i++) {
        uint32_t raw  = woff[i];
        uint32_t off  = raw & WOFF_H_OFF_MASK;
        uint32_t wlen = (uint32_t)packed[off] | ((uint32_t)packed[off + 1] << 8);
        const unsigned char *w = packed + off + 2;
        uint32_t j, head = wlen;
        for (j = 0; j < wlen; j++) {
            if (w[j] == '@') { head = j; break; }
        }
        if (head == wlen) continue;      /* no '@': leave the field 0 */
        if (head > 254u) continue;       /* 8 bits biased: 0..254 only */
        woff[i] = (raw & ~WOFF_H_HEAD_FIELD) | ((head + 1u) << WOFF_H_HEAD_SHIFT);
    }
}

/* RULE-AXIS CHUNKING (Option A, 2026-05-29) — codegen two-kernel proto path
 * candidate-buffer cap. The codegen pipeline (gpu_opencl_kernelb_dispatch_proto)
 * materializes post-rule candidates in a global buffer sized
 * num_words * n_rules * 256. At 16384 words x 99K rules (dive.rule, an ORDINARY
 * ruleset) that is ~415 GB -> OOM FATAL. The dispatcher caps the per-dispatch
 * candidate buffer to GPU_RULES_CAND_CAP_FRACTION of device VRAM (clamped to
 * [GPU_RULES_CAND_CAP_MIN_MB, GPU_RULES_CAND_CAP_MAX_MB]) and loops disjoint
 * rule sub-ranges (rule chunks) of K = cap_bytes/(num_words*256) rules each.
 * Env override MDXFIND_GPU_CAND_CAP_MB sets cap_bytes directly (0/unset =
 * fraction). The legacy streaming rules-engine (gpu_opencl_dispatch_md5_rules)
 * hashes inline and never materializes candidates, so it has no such cap and
 * is unaffected by these. */
#define GPU_RULES_CAND_CAP_FRACTION  0.25
#define GPU_RULES_CAND_CAP_MIN_MB    64
#define GPU_RULES_CAND_CAP_MAX_MB    1024

/* RULE-AXIS CHUNKING (Metal twin, 2026-05-31) — gpu_metal_kernelb_dispatch_-
 * proto_chunked candidate-buffer cap. Apple unified memory: shared with macOS
 * + every other process; cap fraction conservative-lower than OpenCL's
 * (0.15 vs 0.25). Env override MDXFIND_METAL_GPU_CAND_CAP_MB sets cap_bytes
 * directly. Hard ceiling MAX_MB mirrors OpenCL. GPU_RULES_CAND_CAP_MIN_MB
 * (64) above is reused unchanged for the floor. */
#define GPU_RULES_METAL_CAND_CAP_FRACTION  0.15
#define GPU_RULES_METAL_CAND_CAP_MAX_MB    1024
/* ---- INPUT-WORD GATE (operator ruling 2026-09-11) ----------------------
 *
 * The ONLY length check on the GPU rules path.  A word longer than this is
 * not packed; procjob leaves word_packed_by_rules_engine 0, the C2.3 skip
 * therefore does not fire, and the CPU applies every rule to that word.
 *
 * DELIBERATELY INDEPENDENT of the walker buffer size below.  Those were one
 * constant, and the conflation is why discussion of it slid between "which
 * words go to the GPU" and "how big is the scratch buffer".  Nothing is
 * checked about what a rule can PRODUCE -- that is unbounded in principle,
 * and exceeding the buffer is defined behaviour: the rule does what it can,
 * stops, and whatever was generated becomes the candidate.  Both engines
 * behave that way, at their own limits, which is why GPU and CPU results
 * diverge above the GPU walker's limit.  `-G none` is the answer to anyone
 * who needs CPU-exact output.
 *
 * History: this was 256, went to 512 on 2026-05-01 to fix a 303-byte rule
 * OUTPUT truncation that cost one crack against the canonical 21,289, then to
 * 40959 on 2026-05-02 to track MAXLINE.  The 40959 value also silently
 * outran the one-byte packed-length wire field (fixed 2026-09-11, now two
 * bytes), so words of 256..40959 were hashed as their first len mod 256
 * bytes.  1024 is a SET limit, not a derived one.
 */
#define GPU_RULES_MAX_INPUT_LEN 1024

/* Walker scratch size in ELEMENTS, = 2 x the input gate.  Single source of
 * truth: injected into every GPU program as -D RULE_BUF_MAX (OpenCL: the one
 * funnel in gpu_kernel_cache.c; Metal: metal_compile_opts in gpu_metal.m), so
 * the kernels' #ifndef fallbacks are only for out-of-band compiles.
 *
 * ELEMENTS, not bytes, on purpose: the byte walker has 1 element = 1 uchar,
 * and a future UTF-32 walker keeps the same element count at 4x the bytes,
 * so this does not need re-deriving for that port.
 *
 * The usable limit is 15 BELOW the allocation (RULE_BUF_LIMIT in the
 * kernels), covering a NUL and the n+1 boundary conditions in the per-op
 * bounds checks.  Not MAX-1. */
#define GPU_RULES_WALKER_BUF_ELEMS (2 * GPU_RULES_MAX_INPUT_LEN)
#define GPU_RULES_WALKER_BUF_SLACK 15
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

/* Hard ceiling on PACKED SALT BYTES in a single snapshot upload.
 *
 * 2^32 - 10. The per-salt offset array handed to both backends is uint32_t,
 * and gpu_pack_salts_op uses the same running counter BOTH as the recorded
 * offset and as the memcpy destination:
 *
 *     soff[packed] = gsp;
 *     memcpy(salts_packed + gsp, s, sl);
 *     gsp += sl;
 *
 * so wrapping it does not merely record wrong offsets -- it writes later
 * salts back over earlier ones, inside the allocation, with no fault and no
 * diagnostic. Every digest computed against an overwritten salt is wrong and
 * the run still exits 0.
 *
 * Reaching it is not hypothetical at the volumes these types imply. hashcat
 * mode 2811 (e367 MD5-MD5SALTMD5PASS) draws its salt from a 95-character set
 * over 5 positions, so 95^5 = 7,737,809,375 distinct salts exist. At 5 raw
 * bytes each the counter wraps at 858,993,459 salts; for the types that pack
 * the 32-byte MD5(salt) instead of the raw salt it wraps at 134,217,728.
 *
 * The limit is also NOT the same as what the device can hold, and the faster
 * machine is the more dangerous one: measured 2026-10-06, maxBufferLength is
 * 4096 MB on an Apple M1 and 39813 MB on an M2 Max. The M1 fails the
 * allocation first and dies loudly; the M2 Max allocates happily and would
 * run straight into the silent wrap. */
#define GPU_SALT_PACK_MAX 4294967286u   /* 2^32 - 10 */

#ifdef __cplusplus
}
#endif

#endif /* (__APPLE__ && METAL_GPU) || CUDA_GPU || OPENCL_GPU */
#endif /* GPUJOB_H */
