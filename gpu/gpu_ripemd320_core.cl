/*
 * $Revision: 1.2 $
 * $Log: gpu_ripemd320_core.cl,v $
 * Revision 1.2  2026/05/11 05:22:02  dlr
 * Backfill $Revision/$Log RCS keyword stanzas per feedback_rcs_keyword_stanzas.md. Passive 4-line comment block at top of file; no behavioral change. Hand-authored .cl file was missing required stanzas (per memory: all hand-authored .c/.h/.cl/.frag/.tmpl/.py/.sh files MUST contain $Revision/$Log keyword stanzas). Build green on .205 against the post-add files; OpenCL compile strips comments so no kernel behavior change.
 *
 */
/* gpu_ripemd320_core.cl — RIPEMD-320 algorithm extension functions for the
 * generic dispatch template (Memo B Phase B5 sub-batch 2).
 *
 * RIPEMD-320 is a parallel-pipeline extension of RIPEMD-160 that DOES
 * NOT merge the two lines at the end of compression. The result is a
 * 320-bit digest = 10 × uint32 LE state.
 *
 * Geometry / byte ordering:
 *   - 64-byte block, LE message-word layout (same as MD5 / RIPEMD-160)
 *   - 10 × uint32 state (5 from line A + 5 from line B; cross-swapped
 *     between rounds per RIPEMD-320 spec, accumulated into hash[0..9]
 *     at end of compression)
 *   - 64-bit LE length encoding in M[14..15]
 *
 *   HASH_WORDS         — digest size in 32-bit words (10 for RIPEMD-320)
 *   HASH_BLOCK_BYTES   — compress-block size (64)
 *   template_state     — 10 × uint32 chaining
 *   template_init      — install RIPEMD-320 IV
 *   template_transform — absorb one 64-byte block (LE word load,
 *                        rmd320_block from gpu_common.cl)
 *   template_finalize  — pad and finalize (LE, in-place M[16]; same
 *                        perf-fix lesson as MD5/SHA1/RMD160)
 *   template_iterate   — re-hash digest as 80-byte hex_lc input. Two
 *                        blocks: 80 hex chars + 1 (0x80) + 8 (length) =
 *                        89 bytes > 64 ⇒ FIRST block holds 64 hex
 *                        chars (state[0..7]), SECOND block holds the
 *                        remaining 16 hex chars (state[8..9]) + 0x80 +
 *                        length.
 *   template_digest_compare — probe leading 16 bytes (h[0..3]) directly,
 *                             NO bswap32 (LE state convention; same as
 *                             RMD160 / MD5)
 *   template_emit_hit       — EMIT_HIT_10 wrapper (NEW; added to
 *                             gpu_common.cl in this sub-batch).
 *
 * The dual-pipeline compression body lives in gpu_common.cl as
 * `rmd320_block` (added in this sub-batch; mirrors the body in
 * gpu_hmac_rmd320.cl which already implemented it for HMAC). The hash[]
 * accumulation pattern at end of compression:
 *
 *   hash[0] += A;  hash[1] += B;  hash[2] += C;  hash[3] += D;  hash[4] += EE;
 *   hash[5] += AA; hash[6] += BB; hash[7] += CC; hash[8] += DD; hash[9] += E;
 *
 * (cross-mixed E/EE in slots [4] and [9] is part of the spec — copied
 * directly from gpu_hmac_rmd320.cl rmd320_block.)
 *
 * Source order at compile time:
 *
 *   [ gpu_common_str, gpu_md5_rules_str, gpu_ripemd320_core_str, gpu_template_str ]
 *
 * Bytecast invariants:
 *   - Final state[0..9] (after rmd320_block) is LITTLE-ENDIAN per uint32.
 *     Direct probe + direct emit; no bswap32.
 *   - apply_rule + buf in/out is identical to gpu_md5_rules.cl rev 1.28+.
 *
 * R1 mitigation: single-private-buffer pattern. R2 (register pressure):
 * rmd320_block carries 10 uint32 chaining + 10 uint32 working (A..E +
 * AA..EE) plus the 80-step expansion = comparable to but slightly more
 * than rmd160_block. Expected priv_mem_size on gfx1201 ~ similar to or
 * slightly above the RMD160 reading.
 *
 * PERF-FIX LESSON (carried from gpu_md5_core.cl rev 1.2): the in-place
 * M[16] pattern is preserved here — no intermediate uchar pad[] buffer.
 */

/* Per-algorithm geometry. Cache key: "HASH_WORDS=10,HASH_BLOCK_BYTES=64". */
#ifndef HASH_WORDS
#define HASH_WORDS 10
#endif
#ifndef HASH_BLOCK_BYTES
#define HASH_BLOCK_BYTES 64
#endif

/* Per-lane state struct: RIPEMD-320 carries 10 uint32 chaining values. */
typedef struct {
    uint h[HASH_WORDS];
} template_state;

/* template_init: install RIPEMD-320 IV.
 * Standard RIPEMD-320 initial hash values (RIPEMD-320 spec / mhash;
 * matches gpu_hmac_rmd320.cl line 228-229 init exactly). */
static inline void template_init(template_state *st) {
    st->h[0] = 0x67452301u;
    st->h[1] = 0xEFCDAB89u;
    st->h[2] = 0x98BADCFEu;
    st->h[3] = 0x10325476u;
    st->h[4] = 0xC3D2E1F0u;
    st->h[5] = 0x76543210u;
    st->h[6] = 0xFEDCBA98u;
    st->h[7] = 0x89ABCDEFu;
    st->h[8] = 0x01234567u;
    st->h[9] = 0x3C2D1E0Fu;
}

/* template_transform: absorb one 64-byte block, LE message words. */
static inline void template_transform(template_state *st,
                                      const uchar *block)
{
    uint M[16];
    for (int j = 0; j < 16; j++) {
        int b = j * 4;
        M[j] = (uint)block[b]
             | ((uint)block[b + 1] << 8)
             | ((uint)block[b + 2] << 16)
             | ((uint)block[b + 3] << 24);
    }
    rmd320_block(&st->h[0], M);
}

/* template_finalize: process tail, append 0x80 + 64-bit LE length-in-
 * bits, absorb. After return, st->h[0..9] holds the final RIPEMD-320
 * digest words in LITTLE-ENDIAN form. */
static inline void template_finalize(template_state *st,
                                     const uchar *data, int len)
{
    uint M[16];
    int pos = 0;

    while (len - pos >= HASH_BLOCK_BYTES) {
        for (int j = 0; j < 16; j++) {
            int b = pos + j * 4;
            M[j] = (uint)data[b]
                 | ((uint)data[b + 1] << 8)
                 | ((uint)data[b + 2] << 16)
                 | ((uint)data[b + 3] << 24);
        }
        rmd320_block(&st->h[0], M);
        pos += HASH_BLOCK_BYTES;
    }

    int rem = len - pos;  /* 0..63 */
    for (int j = 0; j < 16; j++) M[j] = 0;

    for (int i = 0; i < rem; i++) {
        uint v = (uint)data[pos + i];
        M[i >> 2] |= v << ((i & 3) * 8);
    }
    M[rem >> 2] |= (uint)0x80u << ((rem & 3) * 8);

    if (rem < 56) {
        M[14] = (uint)((uint)len * 8u);
        M[15] = 0u;
        rmd320_block(&st->h[0], M);
    } else {
        rmd320_block(&st->h[0], M);
        for (int j = 0; j < 16; j++) M[j] = 0;
        M[14] = (uint)((uint)len * 8u);
        M[15] = 0u;
        rmd320_block(&st->h[0], M);
    }
}

/* template_iterate: -i loop step. Re-encode the 320-bit digest as 80
 * lowercase hex chars and rehash. 80 + 1 (0x80) + 8 (length) = 89 bytes
 * > 64 ⇒ requires a SECOND padding block.
 *
 * Layout:
 *   block 1 (M[0..15], 64 bytes):
 *     M[0..15] = first 16 LE-packed hex pairs = 64 hex chars (state[0..7])
 *
 *   block 2 (M[0..15], 64 bytes):
 *     M[0..3]  = remaining 16 hex chars (state[8..9])
 *     M[4]     = 0x80 (LE byte at offset 80)
 *     M[5..13] = 0
 *     M[14]    = 80 * 8 = 640 (low 32 bits of bit count)
 *     M[15]    = 0
 *
 * Snapshot state[8..9] BEFORE the IV reset so block 2 can hex-encode
 * the original digest's last two words. */
static inline void template_iterate(template_state *st)
{
    uint M[16];
    /* Snapshot state[8..9] for block 2 before the IV reset. */
    uint snap8 = st->h[8];
    uint snap9 = st->h[9];

    /* Block 1: hex-encode state[0..7] into M[0..15]. */
    for (int i = 0; i < 8; i++) {
        uint s = st->h[i];
        uint b0 = s & 0xff,        b1 = (s >> 8) & 0xff;
        uint b2 = (s >> 16) & 0xff, b3 = (s >> 24) & 0xff;
        M[i*2]     = hex_byte_lc(b0) | (hex_byte_lc(b1) << 16);
        M[i*2 + 1] = hex_byte_lc(b2) | (hex_byte_lc(b3) << 16);
    }
    /* Reinitialize state to RIPEMD-320 IV; absorb block 1. */
    st->h[0] = 0x67452301u;
    st->h[1] = 0xEFCDAB89u;
    st->h[2] = 0x98BADCFEu;
    st->h[3] = 0x10325476u;
    st->h[4] = 0xC3D2E1F0u;
    st->h[5] = 0x76543210u;
    st->h[6] = 0xFEDCBA98u;
    st->h[7] = 0x89ABCDEFu;
    st->h[8] = 0x01234567u;
    st->h[9] = 0x3C2D1E0Fu;
    rmd320_block(&st->h[0], M);

    /* Block 2: hex-encode (snap8, snap9) into M[0..3]; 0x80 at M[4];
     * bit length in M[14]. */
    {
        uint s8 = snap8, s9 = snap9;
        uint b0 = s8 & 0xff,        b1 = (s8 >> 8) & 0xff;
        uint b2 = (s8 >> 16) & 0xff, b3 = (s8 >> 24) & 0xff;
        M[0] = hex_byte_lc(b0) | (hex_byte_lc(b1) << 16);
        M[1] = hex_byte_lc(b2) | (hex_byte_lc(b3) << 16);
        b0 = s9 & 0xff;        b1 = (s9 >> 8) & 0xff;
        b2 = (s9 >> 16) & 0xff; b3 = (s9 >> 24) & 0xff;
        M[2] = hex_byte_lc(b0) | (hex_byte_lc(b1) << 16);
        M[3] = hex_byte_lc(b2) | (hex_byte_lc(b3) << 16);
    }
    M[4] = 0x80u;                   /* LE 0x80 at byte offset 80 */
    for (int j = 5; j < 14; j++) M[j] = 0u;
    M[14] = 80u * 8u;               /* 80 hex chars = 640 bits */
    M[15] = 0u;
    rmd320_block(&st->h[0], M);
}

/* template_digest_compare: probe leading 16 bytes (h[0..3]). LE state
 * convention; no bswap32 (same as MD5 / RIPEMD-160). */
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

/* template_emit_hit: EMIT_HIT_10 wrapper (RIPEMD-320 = 10 uint32 LE
 * digest words). State words flow directly, no bswap32. */
#define template_emit_hit(hits, hit_count, max_hits, st, widx, sidx, iter) \
    EMIT_HIT_10((hits), (hit_count), (max_hits), \
                (widx), (sidx), (iter), ((st)->h))

/* B3 dedup+overflow-aware variant. */
#define template_emit_hit_or_overflow(hits, hit_count, max_hits, st, widx, sidx, iter, \
                                      hashes_shown, matched_idx, dedup_mask, \
                                      ovr_set, ovr_gid, lane_gid) \
    EMIT_HIT_10_DEDUP_OR_OVERFLOW((hits), (hit_count), (max_hits), \
               (widx), (sidx), (iter), ((st)->h), \
               (hashes_shown), (matched_idx), (dedup_mask), \
               (ovr_set), (ovr_gid), (lane_gid))
