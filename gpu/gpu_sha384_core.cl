/*
 * $Revision: 1.2 $
 * $Log: gpu_sha384_core.cl,v $
 * Revision 1.2  2026/05/11 05:22:02  dlr
 * Backfill $Revision/$Log RCS keyword stanzas per feedback_rcs_keyword_stanzas.md. Passive 4-line comment block at top of file; no behavioral change. Hand-authored .cl file was missing required stanzas (per memory: all hand-authored .c/.h/.cl/.frag/.tmpl/.py/.sh files MUST contain $Revision/$Log keyword stanzas). Build green on .205 against the post-add files; OpenCL compile strips comments so no kernel behavior change.
 *
 */
/* gpu_sha384_core.cl — SHA384 algorithm extension functions for the
 * generic dispatch template (Memo B Phase B5 sub-batch 1).
 *
 * SHA384 uses the SAME compression function as SHA512 — only the IV
 * and the output truncation differ:
 *   - IV: 8 distinct ulong constants (FIPS 180-4 §5.3.4)
 *   - Output: first 6 of 8 state words (384 bits = 6 × 64 = 12 × 32)
 *
 * Mirrors gpu_sha224_core.cl rev 1.2 (the analogous SHA256→SHA224
 * truncation pattern), scaled to 64-bit state and 128-byte blocks.
 *
 *   HASH_WORDS         — digest size in 32-bit words (12 for SHA384;
 *                        truncates 6 ulong = 12 uint32)
 *   HASH_BLOCK_BYTES   — compress-block size (128, same as SHA512)
 *   template_state     — 8 × ulong INTERNAL chaining + 12 × uint exposed
 *                        digest. The 8th & 7th internal ulong (state[6],
 *                        state[7]) participate in compression but are
 *                        DROPPED from h[]: only state[0..5] decompose
 *                        into h[0..11].
 *   template_init      — install SHA384 IV (8 distinct ulong values)
 *   template_transform — same as SHA512 (BIG-ENDIAN ulong block load)
 *   template_finalize  — same compression loop as SHA512; 128-bit length
 *                        encoding; only state[0..5] decompose into h[]
 *   template_iterate   — re-hash digest as 96-byte hex_lc (-i loop;
 *                        96 hex chars + 0x80 + 16 length = 113 bytes
 *                        > 128? NO — 113 ≤ 128: single-block iter,
 *                        with 0x80 at byte 96, length at M[14..15])
 *
 *   ACTUALLY: 96 + 1 + 16 = 113. 113 < 128 ⇒ FITS IN ONE BLOCK.
 *   But 96 hex bytes occupy M[0..11] (12 ulong = 96 bytes); M[12..13]
 *   are zero-pad; M[14..15] are length. The 0x80 marker lands at
 *   byte 96, which is M[12] high octet ⇒ M[12] = 0x8000000000000000UL.
 *   So actually layout is:
 *     M[0..11] = 96 hex bytes packed BE
 *     M[12] = 0x8000000000000000UL (0x80 at byte 96)
 *     M[13] = 0
 *     M[14] = 0 (high 64 bits of length)
 *     M[15] = 96 * 8 = 768 (low 64 bits)
 *   Single block. Different from SHA512's two-block iter.
 *
 *   template_digest_compare — probe leading 16 bytes (h[0..3], LE uint32
 *                             pairs from state[0..1])
 *   template_emit_hit       — EMIT_HIT_12 wrapper (12 LE uint32 from
 *                             state[0..5])
 *
 * Source order at compile time:
 *
 *   [ gpu_common_str, gpu_md5_rules_str, gpu_sha384_core_str, gpu_template_str ]
 *
 * gpu_common_str provides sha512_block(), bswap64, EMIT_HIT_12{,_DEDUP_OR_OVERFLOW},
 * and probe_compact_idx. We do NOT use sha512_to_hex_lc directly because
 * SHA384's hex iter writes only 96 bytes (12 ulong) — sha512_to_hex_lc
 * writes 16 ulong (128 bytes); an inline 12-word version is provided.
 *
 * Bytecast invariants:
 *   - Final state[0..7] (after compression) is BIG-ENDIAN. h[] holds
 *     LE-byteswapped uint32 pairs from state[0..5] only. state[6..7]
 *     are not exposed to the template body but participate in chaining.
 *   - template_emit_hit writes 12 LE uint32 to the hits buffer.
 *
 * R1 mitigation: single private buffer pattern. R2: same compression
 * footprint as SHA512 (W[80] schedule, 80 × 8 = 640 bytes scratch),
 * minus 4 uint32 (h[12..15] not allocated). Expected priv_mem_size on
 * gfx1201 ~ same as SHA512.
 *
 * Why a separate file (vs sharing sha512_block via #include): the SHA384
 * core only needs a different IV and a smaller h[] decomposition.
 * Inlining the entire compression body would duplicate ~80 lines vs
 * keeping the sha512_block primitive in gpu_common.cl and just changing
 * IV + output. The factoring sha512_block-as-shared-primitive is the
 * cleaner option; this file consumes it directly.
 */

/* Per-algorithm geometry. Cache key (R3 fix) hashes the defines_str
 * "HASH_WORDS=12,HASH_BLOCK_BYTES=128" alongside source text so SHA384
 * gets a distinct cache entry from SHA512 (HASH_WORDS=16). Same source
 * text + different defines is the same setup as SHA224/SHA256. */
#ifndef HASH_WORDS
#define HASH_WORDS 12
#endif
#ifndef HASH_BLOCK_BYTES
#define HASH_BLOCK_BYTES 128
#endif

/* Per-lane state struct. SHA384 carries 8 × uint64 chaining INTERNALLY
 * (state[]) — same as SHA512's compression. Output truncates to 6 ulong
 * = 12 uint32 in h[]. The 7th & 8th ulong (state[6], state[7]) are
 * NEVER exposed; they exist only to feed sha512_block correctly. */
typedef struct {
    ulong state[8];   /* internal compression state, BIG-ENDIAN */
    uint  h[HASH_WORDS]; /* exposed digest words, LE-byteswapped uint32 (12 only) */
} template_state;

/* template_init: install SHA384 IV.
 * Standard SHA-384 initial hash values (FIPS 180-4 §5.3.4). */
static inline void template_init(template_state *st) {
    st->state[0] = 0xcbbb9d5dc1059ed8UL;
    st->state[1] = 0x629a292a367cd507UL;
    st->state[2] = 0x9159015a3070dd17UL;
    st->state[3] = 0x152fecd8f70e5939UL;
    st->state[4] = 0x67332667ffc00b31UL;
    st->state[5] = 0x8eb44a8768581511UL;
    st->state[6] = 0xdb0c2e0d64f98fa7UL;
    st->state[7] = 0x47b5481dbefa4fa4UL;
}

/* Internal helper: decompose state[0..5] (BE ulong) into h[0..11]
 * (LE uint32 pairs). state[6..7] dropped per SHA384 truncation. */
static inline void template_state_to_h(template_state *st) {
    for (int i = 0; i < 6; i++) {
        ulong s = bswap64(st->state[i]);
        st->h[i*2]   = (uint)s;
        st->h[i*2+1] = (uint)(s >> 32);
    }
}

/* template_transform: absorb one HASH_BLOCK_BYTES (128) byte block.
 * Same compression as SHA512. */
static inline void template_transform(template_state *st,
                                      const uchar *block)
{
    ulong M[16];
    for (int j = 0; j < 16; j++) {
        int b = j * 8;
        M[j] = ((ulong)block[b]     << 56)
             | ((ulong)block[b + 1] << 48)
             | ((ulong)block[b + 2] << 40)
             | ((ulong)block[b + 3] << 32)
             | ((ulong)block[b + 4] << 24)
             | ((ulong)block[b + 5] << 16)
             | ((ulong)block[b + 6] << 8)
             |  (ulong)block[b + 7];
    }
    sha512_block(&st->state[0], M);
}

/* template_finalize: identical to SHA512 finalize for compression and
 * 128-bit length encoding; only the post-compression state-to-h
 * decomposition differs (truncates to 12 uint32 instead of 16). */
static inline void template_finalize(template_state *st,
                                     const uchar *data, int len)
{
    ulong M[16];
    int pos = 0;

    while (len - pos >= HASH_BLOCK_BYTES) {
        for (int j = 0; j < 16; j++) {
            int b = pos + j * 8;
            M[j] = ((ulong)data[b]     << 56)
                 | ((ulong)data[b + 1] << 48)
                 | ((ulong)data[b + 2] << 40)
                 | ((ulong)data[b + 3] << 32)
                 | ((ulong)data[b + 4] << 24)
                 | ((ulong)data[b + 5] << 16)
                 | ((ulong)data[b + 6] << 8)
                 |  (ulong)data[b + 7];
        }
        sha512_block(&st->state[0], M);
        pos += HASH_BLOCK_BYTES;
    }

    int rem = len - pos;  /* 0..127 */

    for (int j = 0; j < 16; j++) M[j] = 0UL;

    for (int i = 0; i < rem; i++) {
        int wi = i >> 3;
        int bi = 7 - (i & 7);
        M[wi] |= ((ulong)data[pos + i]) << (bi * 8);
    }
    {
        int wi = rem >> 3;
        int bi = 7 - (rem & 7);
        M[wi] |= ((ulong)0x80UL) << (bi * 8);
    }

    if (rem < 112) {
        M[14] = 0UL;
        M[15] = (ulong)((ulong)len * 8UL);
        sha512_block(&st->state[0], M);
    } else {
        sha512_block(&st->state[0], M);
        for (int j = 0; j < 16; j++) M[j] = 0UL;
        M[14] = 0UL;
        M[15] = (ulong)((ulong)len * 8UL);
        sha512_block(&st->state[0], M);
    }

    template_state_to_h(st);
}

/* template_iterate: -i loop step. SHA384 hex output is 96 lowercase
 * hex chars (48 digest bytes × 2). 96 + 1 (0x80) + 16 (length) = 113
 * bytes — fits in one HASH_BLOCK_BYTES (128) block.
 *
 * Layout:
 *   M[0..11] = 96 hex bytes packed BE into 12 ulong words (each ulong
 *              holds 8 hex chars BE)
 *   M[12]    = 0x8000000000000000UL  (0x80 marker at byte 96)
 *   M[13]    = 0
 *   M[14]    = 0 (high 64 bits of bit count)
 *   M[15]    = 96 * 8 = 768 (low 64 bits of bit count)
 *
 * Inlined per-byte hex encoder over state[0..5] only (12 words instead
 * of sha512_to_hex_lc's 16). */
static inline void template_iterate(template_state *st)
{
    ulong M[16];
    /* Encode 6 state words → 12 hex ulong words. Same byte-by-byte
     * pattern as sha512_to_hex_lc (gpu_common.cl line 780) but limited
     * to 6 input words. */
    for (int i = 0; i < 6; i++) {
        ulong s = st->state[i];
        uint b0 = (uint)((s >> 56) & 0xff), b1 = (uint)((s >> 48) & 0xff);
        uint b2 = (uint)((s >> 40) & 0xff), b3 = (uint)((s >> 32) & 0xff);
        uint b4 = (uint)((s >> 24) & 0xff), b5 = (uint)((s >> 16) & 0xff);
        uint b6 = (uint)((s >> 8)  & 0xff), b7 = (uint)(s & 0xff);
        M[i*2]   = (hex_byte_be64(b0) << 48) | (hex_byte_be64(b1) << 32)
                  | (hex_byte_be64(b2) << 16) | hex_byte_be64(b3);
        M[i*2+1] = (hex_byte_be64(b4) << 48) | (hex_byte_be64(b5) << 32)
                  | (hex_byte_be64(b6) << 16) | hex_byte_be64(b7);
    }
    M[12] = 0x8000000000000000UL;   /* 0x80 BE at byte position 96 */
    M[13] = 0UL;
    M[14] = 0UL;                    /* high 64 bits of bit count */
    M[15] = 96UL * 8UL;             /* 96 hex chars = 768 bits */

    /* Reset state to SHA384 IV; absorb the hex block. */
    st->state[0] = 0xcbbb9d5dc1059ed8UL;
    st->state[1] = 0x629a292a367cd507UL;
    st->state[2] = 0x9159015a3070dd17UL;
    st->state[3] = 0x152fecd8f70e5939UL;
    st->state[4] = 0x67332667ffc00b31UL;
    st->state[5] = 0x8eb44a8768581511UL;
    st->state[6] = 0xdb0c2e0d64f98fa7UL;
    st->state[7] = 0x47b5481dbefa4fa4UL;
    sha512_block(&st->state[0], M);

    template_state_to_h(st);
}

/* template_digest_compare: probe leading 16 bytes (h[0..3]). Same
 * convention as SHA512 — h[] is already LE-decomposed. */
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

/* template_emit_hit: EMIT_HIT_12 (SHA384 = 12 uint32 = 6 ulong LE
 * decomposed). state[6..7] dropped per SHA384 truncation. */
#define template_emit_hit(hits, hit_count, max_hits, st, widx, sidx, iter) \
    EMIT_HIT_12((hits), (hit_count), (max_hits), \
                (widx), (sidx), (iter), ((st)->h))

/* B3 dedup+overflow-aware variant. */
#define template_emit_hit_or_overflow(hits, hit_count, max_hits, st, widx, sidx, iter, \
                                      hashes_shown, matched_idx, dedup_mask, \
                                      ovr_set, ovr_gid, lane_gid) \
    EMIT_HIT_12_DEDUP_OR_OVERFLOW((hits), (hit_count), (max_hits), \
               (widx), (sidx), (iter), ((st)->h), \
               (hashes_shown), (matched_idx), (dedup_mask), \
               (ovr_set), (ovr_gid), (lane_gid))
