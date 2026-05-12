/*
 * $Revision: 1.5 $
 * $Log: gpu_md5_core.cl,v $
 * Revision 1.5  2026/05/11 05:22:01  dlr
 * Backfill $Revision/$Log RCS keyword stanzas per feedback_rcs_keyword_stanzas.md. Passive 4-line comment block at top of file; no behavioral change. Hand-authored .cl file was missing required stanzas (per memory: all hand-authored .c/.h/.cl/.frag/.tmpl/.py/.sh files MUST contain $Revision/$Log keyword stanzas). Build green on .205 against the post-add files; OpenCL compile strips comments so no kernel behavior change.
 *
 */
/* gpu_md5_core.cl — MD5 algorithm extension functions for the generic
 * dispatch template (Memo B Phase B2).
 *
 * This file defines the per-algorithm pieces that the generic template
 * (gpu_template.cl) calls into:
 *
 *   HASH_WORDS         — digest size in 32-bit words (4 for MD5)
 *   HASH_BLOCK_BYTES   — compress-block size (64 for MD5)
 *   template_state     — opaque per-lane state struct
 *   template_init      — initialize state to algorithm IV
 *   template_transform — absorb one HASH_BLOCK_BYTES block
 *   template_finalize  — pad and finalize a buffer; produces final state
 *   template_digest_compare — probe the compact table from final state and
 *                             return 1+matched_idx on hit, 0 on miss
 *   template_iterate   — re-hash the digest as 32-byte hex_lc (-i loop)
 *
 * Reuses primitives from gpu_common.cl: md5_block, md5_to_hex_lc,
 * probe_compact_idx. The kernel proper lives in gpu_template.cl; the
 * source order at compile time is:
 *
 *   [ gpu_common_str, gpu_md5_rules_str, gpu_md5_core_str, gpu_template_str ]
 *
 * gpu_md5_rules_str provides apply_rule() for the rules walker; this
 * file provides the per-algorithm hooks; gpu_template_str ties them
 * into the template_phase0 kernel.
 *
 * Bytecast invariants:
 *   - apply_rule + buf in/out is identical to gpu_md5_rules.cl rev 1.28+.
 *   - md5 padding + multi-block logic is identical to md5_buf in
 *     gpu_md5_rules.cl (we do not call md5_buf directly to keep this
 *     file self-contained for future algorithm forks; the bytewise
 *     equivalence is the byte-exact gate).
 *   - template_finalize() builds the M[16] message words directly
 *     from the input bytes (mirrors md5_buf) and calls md5_block()
 *     once per block. It does NOT route through template_transform()
 *     — that wrapper exists for the public API but is bypassed in
 *     the hot path because the byte-buffer round-trip cost it would
 *     introduce was 12.3% of wall time on Memo B B2's first-cut
 *     template path. See the comment on template_finalize for detail.
 *
 * R1 mitigation (AMD ROCm comgr addrspace fragility): the template
 * passes a `__private uchar *buf` and `__private template_state *st`.
 * No addrspace-cast ternary pointers, no helper that takes a generic
 * pointer to private state. Pattern matches the safe single-private-
 * buffer model from gpu_md5_rules.cl r28.
 */

/* Per-algorithm geometry. The template uses these as structural
 * compile-time constants; the kernel-cache key (gpu_kernel_cache R3
 * fix) hashes the defines_str ("HASH_WORDS=4,HASH_BLOCK_BYTES=64")
 * alongside the source set so distinct instantiations get distinct
 * cache entries even though source text would otherwise be identical. */
#ifndef HASH_WORDS
#define HASH_WORDS 4
#endif
#ifndef HASH_BLOCK_BYTES
#define HASH_BLOCK_BYTES 64
#endif

/* Per-lane state struct: MD5 carries 4 uint32 chaining values.
 * SHA1 will carry 5; SHA256 8; SHA384/SHA512 8 ulong. The template
 * only reads/writes the digest words via template_finalize's output
 * + template_digest_compare; it does not introspect the struct. */
typedef struct {
    uint h[HASH_WORDS];
} template_state;

/* template_init: install algorithm IV into state. */
static inline void template_init(template_state *st) {
    st->h[0] = 0x67452301u;
    st->h[1] = 0xEFCDAB89u;
    st->h[2] = 0x98BADCFEu;
    st->h[3] = 0x10325476u;
}

/* template_transform: absorb one HASH_BLOCK_BYTES (64) byte block.
 * `block` points into the working buffer at the absorb position;
 * the caller must guarantee >=64 bytes are readable. */
static inline void template_transform(template_state *st,
                                      const uchar *block)
{
    uint M[16];
    /* MD5 reads message words little-endian. */
    for (int j = 0; j < 16; j++) {
        int b = j * 4;
        M[j] = (uint)block[b]
             | ((uint)block[b + 1] << 8)
             | ((uint)block[b + 2] << 16)
             | ((uint)block[b + 3] << 24);
    }
    md5_block(&st->h[0], &st->h[1], &st->h[2], &st->h[3], M);
}

/* template_finalize: process the tail, append the 0x80 padding marker
 * + length-in-bits, and absorb. Caller passes the full buffer; we run
 * complete blocks then build the final padded block(s) here. After
 * return, st->h[0..HASH_WORDS-1] holds the final digest words.
 *
 * Mirrors md5_buf() in gpu_md5_rules.cl byte-for-byte. We do NOT
 * route through template_transform() here — instead we build uint
 * M[16] directly via in-place OR-merge of the input bytes, exactly
 * as md5_buf does, and call md5_block() once per block.
 *
 * Why not use template_transform: the previous version built a
 * private uchar pad[64] tail block, populated it with 64+rem byte
 * stores, then handed it to template_transform which read the same
 * 64 bytes BACK out as little-endian uints into a fresh M[16].
 * That round-trip (byte-stores -> byte-loads -> uint-pack) was the
 * 12.3% wall regression vs the legacy md5_buf path on ioblade
 * (Memo B Phase B2 perf-fix, 2026-05-04). For full blocks the
 * function-call boundary plus rebuilt-M[] cost is non-trivial too;
 * for the final block the extra 64-byte zeroing + byte-by-byte
 * tail copy + 4-byte length store dominates because rem is small
 * for typical wordlist inputs.
 *
 * template_transform() remains exposed as the public extension API
 * (a future algorithm core might prefer the function-style entry).
 * For MD5 we ignore it in the hot path. SHA1/SHA256 cores in B4
 * SHOULD follow this same in-place pattern in their finalize.
 *
 * R1 mitigation preserved: single private buffer (just the M[16]
 * uints + the input data pointer), no addrspace-cast helpers, no
 * __private uchar* helper that takes a private buffer pointer. */
static inline void template_finalize(template_state *st,
                                     const uchar *data, int len)
{
    uint M[16];
    int pos = 0;

    /* Process complete 64-byte blocks. Build M[] directly from
     * bytes (no intermediate pad[]). Identical to md5_buf's main loop. */
    while (len - pos >= HASH_BLOCK_BYTES) {
        for (int j = 0; j < 16; j++) {
            int b = pos + j * 4;
            M[j] = (uint)data[b]
                 | ((uint)data[b + 1] << 8)
                 | ((uint)data[b + 2] << 16)
                 | ((uint)data[b + 3] << 24);
        }
        md5_block(&st->h[0], &st->h[1], &st->h[2], &st->h[3], M);
        pos += HASH_BLOCK_BYTES;
    }

    /* Build final block(s): tail bytes + 0x80 marker + zeros + length.
     * Same direct-into-M[] approach as md5_buf — no intermediate uchar
     * pad[]. */
    int rem = len - pos;  /* 0..63 */

    /* Zero scratch. */
    for (int j = 0; j < 16; j++) M[j] = 0;

    /* Copy remaining tail bytes into M[] little-endian. */
    for (int i = 0; i < rem; i++) {
        uint v = (uint)data[pos + i];
        M[i >> 2] |= v << ((i & 3) * 8);
    }
    /* 0x80 padding marker. */
    M[rem >> 2] |= (uint)0x80 << ((rem & 3) * 8);

    if (rem < 56) {
        /* Length fits in this block. */
        M[14] = (uint)(len * 8);
        M[15] = 0;
        md5_block(&st->h[0], &st->h[1], &st->h[2], &st->h[3], M);
    } else {
        /* Need one extra padding-only block to hold the length. */
        md5_block(&st->h[0], &st->h[1], &st->h[2], &st->h[3], M);
        for (int j = 0; j < 16; j++) M[j] = 0;
        M[14] = (uint)(len * 8);
        M[15] = 0;
        md5_block(&st->h[0], &st->h[1], &st->h[2], &st->h[3], M);
    }
}

/* template_iterate: -i loop step. Re-encode the digest as 32-byte
 * hex ASCII and rehash. Algorithm-specific because the
 * digest geometry differs (MD5: 4 words, 32 chars; SHA1: 5 words,
 * 40 chars; SHA256: 8 words, 64 chars). For MD5 we mirror
 * gpu_md5_rules.cl rev 1.28's iter loop exactly so byte-exact.
 *
 * B7.7a (2026-05-07): MD5UC variant via algo_mode. The CPU iter loop
 * (mdxfind.c:25386 MDstart) selects prmd5UC vs prmd5 based on
 * job->op == JOB_MD5UC at every iter step, so the inter-iter hex
 * encoding is uppercase for JOB_MD5UC. Iter=1 is byte-exact between
 * MD5 and MD5UC (the UC path only fires inter-iter). The
 * GPU_TEMPLATE_ITERATE_HAS_ALGO_MODE define (set ONLY in this core)
 * extends the signature to accept algo_mode; gpu_template.cl uses an
 * #ifdef at the call site to invoke the right shape. Other cores'
 * template_iterate stay at the legacy `(st)` signature.
 *
 * algo_mode == 0: lowercase hex (JOB_MD5 — default).
 * algo_mode == 1: uppercase hex (JOB_MD5UC). */
#define GPU_TEMPLATE_ITERATE_HAS_ALGO_MODE 1
static inline void template_iterate(template_state *st, uint algo_mode)
{
    uint M[16];
    if (algo_mode == 1u) {
        md5_to_hex_uc(st->h[0], st->h[1], st->h[2], st->h[3], M);
    } else {
        md5_to_hex_lc(st->h[0], st->h[1], st->h[2], st->h[3], M);
    }
    M[8] = 0x80u;
    for (int j = 9; j < 14; j++) M[j] = 0u;
    M[14] = 32u * 8u;     /* 32 hex chars = 256 bits */
    M[15] = 0u;
    /* Reinitialize state to IV, then absorb the prepared M[]. */
    st->h[0] = 0x67452301u;
    st->h[1] = 0xEFCDAB89u;
    st->h[2] = 0x98BADCFEu;
    st->h[3] = 0x10325476u;
    md5_block(&st->h[0], &st->h[1], &st->h[2], &st->h[3], M);
}

/* template_digest_compare: probe the compact table with the final
 * digest. On a hit, *out_idx is set to the matched target's hash_data
 * index (mirrors probe_compact_idx semantics). Return 1 on hit, 0 on
 * miss.
 *
 * Wrapper around probe_compact_idx — the template body calls this to
 * stay algorithm-agnostic about which words of the digest form the
 * (hx, hy, hz, hw) probe key. For MD5 it's the first 4 uint words.
 * SHA1 will have a 5-word variant; SHA256 will pull (h[0]..h[3]) for
 * the same 16-byte probe geometry (compact table is keyed on the
 * leading 16 bytes regardless of algorithm digest length). */
static inline int template_digest_compare(
    const template_state *st,
    __global const uint *compact_fp, __global const uint *compact_idx,
    ulong compact_mask, uint max_probe, uint hash_data_count,
    __global const uchar *hash_data_buf, __global const ulong *hash_data_off,
    __global const ulong *overflow_keys, __global const uchar *overflow_hashes,
    __global const uint *overflow_offsets, uint overflow_count,
    uint *out_idx)
{
    return probe_compact_idx(
        st->h[0], st->h[1], st->h[2], st->h[3],
        compact_fp, compact_idx,
        compact_mask, max_probe, hash_data_count,
        hash_data_buf, hash_data_off,
        overflow_keys, overflow_hashes, overflow_offsets, overflow_count,
        out_idx);
}

/* template_emit_hit: emit a hit into the global hits buffer. Wraps
 * EMIT_HIT_4 (MD5 = 4 uint32 digest words). The template body invokes
 * this through the algorithm core so HASH_WORDS-specific EMIT_HIT_N
 * macros stay encapsulated. Input order matches md5_rules_phase0's
 * EMIT_HIT_4 call site. */
#define template_emit_hit(hits, hit_count, max_hits, st, widx, sidx, iter) \
    EMIT_HIT_4((hits), (hit_count), (max_hits), \
               (widx), (sidx), (iter), \
               (st)->h[0], (st)->h[1], (st)->h[2], (st)->h[3])

/* B3 dedup+overflow-aware variant. Wraps EMIT_HIT_4_DEDUP_OR_OVERFLOW
 * to keep the hashes_shown[] dedup bit consistent across overflow
 * re-issues (see gpu_common.cl §B3 protocol notes). The macro takes
 * the dedup state (hashes_shown ptr, matched_idx, dedup_mask) PLUS
 * the overflow channel (ovr_set, ovr_gid, lane_gid). The kernel must
 * NOT pre-set the dedup bit; the macro does that atomically. */
#define template_emit_hit_or_overflow(hits, hit_count, max_hits, st, widx, sidx, iter, \
                                      hashes_shown, matched_idx, dedup_mask, \
                                      ovr_set, ovr_gid, lane_gid) \
    EMIT_HIT_4_DEDUP_OR_OVERFLOW((hits), (hit_count), (max_hits), \
               (widx), (sidx), (iter), \
               (st)->h[0], (st)->h[1], (st)->h[2], (st)->h[3], \
               (hashes_shown), (matched_idx), (dedup_mask), \
               (ovr_set), (ovr_gid), (lane_gid))
