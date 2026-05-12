/*
 * $Revision: 1.3 $
 * $Log: gpu_sha224_core.cl,v $
 * Revision 1.3  2026/05/11 05:22:02  dlr
 * Backfill $Revision/$Log RCS keyword stanzas per feedback_rcs_keyword_stanzas.md. Passive 4-line comment block at top of file; no behavioral change. Hand-authored .cl file was missing required stanzas (per memory: all hand-authored .c/.h/.cl/.frag/.tmpl/.py/.sh files MUST contain $Revision/$Log keyword stanzas). Build green on .205 against the post-add files; OpenCL compile strips comments so no kernel behavior change.
 *
 */
/* gpu_sha224_core.cl — SHA224 algorithm extension functions for the
 * generic dispatch template (Memo B Phase B4 fan-out).
 *
 * Mirrors gpu_sha256_core.cl rev 1.1 structure. SHA224 uses the same
 * compression function as SHA256 — only the IV and the output truncation
 * differ:
 *   - IV: 8 distinct constants (FIPS 180-4 §5.3.4)
 *   - Output: first 7 of 8 state words (224 bits = 7 × 32)
 *
 * KEY DIFFERENCE FROM SHA256: HASH_WORDS = 7 (digest output size in
 * 32-bit words), but the INTERNAL state struct still holds 8 words —
 * the SHA256 compression function (sha256_block) reads and writes all
 * 8 chaining values regardless of which output truncation is requested.
 *
 * To handle this asymmetry without breaking the template's
 * `digests_out[gid * HASH_WORDS + i] = st.h[i]` loop (gpu_template.cl
 * template_phase0_test), the state struct is sized as a literal `uint
 * h[8]` rather than `uint h[HASH_WORDS]`. The template body only
 * iterates `i < HASH_WORDS` so it correctly writes 7 words per lane;
 * the 8th word is internal-only and never crosses the kernel boundary.
 *
 *   HASH_WORDS         — digest size in 32-bit words (7 for SHA224)
 *   HASH_BLOCK_BYTES   — compress-block size (64 — same as SHA256)
 *   template_state     — opaque per-lane state struct (8 uint32 chaining;
 *                        9th-and-beyond never accessed)
 *   template_init      — install SHA224 IV (8 standard values)
 *   template_transform — absorb one 64-byte block (BIG-ENDIAN word load)
 *   template_finalize  — pad and finalize (in-place M[16] pattern)
 *   template_iterate   — re-hash digest as 56-byte hex_lc (-i loop)
 *   template_digest_compare — probe compact table with leading 16 bytes
 *                             (h[0..3], LE-byteswapped)
 *   template_emit_hit       — EMIT_HIT_7 wrapper (7 LE digest words)
 *
 * KEY POINTS:
 *
 * 1. State width vs digest width. State struct = 8 uint32 (literal);
 *    digest output = 7 uint32 words (HASH_WORDS = 7). The 8th word
 *    (st->h[7]) is the SHA256 'h' chaining value — it participates in
 *    every compression step but is dropped from the final digest.
 *
 * 2. Iter loop. SHA224 hex output is 56 lowercase hex chars. The pad
 *    fits in the first block: M[14] = 0 = high bits, M[15] = 56 * 8 =
 *    448 (bit count). 56 < 64 so single-block pad-and-length. That's
 *    different from SHA256 (which fills exactly one block and needs a
 *    second pad-only block).
 *
 * 3. Compact table probe. Same leading-16-byte LE convention as the
 *    other algos: bswap32 the leading 4 state words. Works for SHA224
 *    because the truncation happens at output time, not at chaining.
 *
 * 4. EMIT_HIT_7: 7-word emit wrapper (gpu_common.cl line 71). The 8th
 *    state word st->h[7] is intentionally NOT emitted to match SHA224's
 *    output format.
 *
 * Source order at compile time:
 *
 *   [ gpu_common_str, gpu_md5_rules_str, gpu_sha224_core_str, gpu_template_str ]
 *
 * gpu_common_str provides sha256_block(), bswap32, EMIT_HIT_7, and
 * probe_compact_idx. gpu_md5_rules_str provides apply_rule.
 *
 * Bytecast invariants:
 *   - Final digest h[0..6] (after template_finalize) is BIG-ENDIAN —
 *     same as SHA256 / SHA1 (the compression function's natural form).
 *   - template_digest_compare byte-swaps the leading 4 words to LE
 *     before probing.
 *   - template_emit_hit writes 7 LE-byteswapped digest words.
 *   - Existing legacy SHA224 path (gpu_sha256_packed.cl already covers
 *     SHA224 per the file header comment) uses the same wire format.
 *
 * R1 mitigation: single private buffer pattern. No __private uchar*
 * helpers; no addrspace-cast ternaries.
 *
 * R2 (register pressure): SHA224 is structurally identical to SHA256
 * for the compression function — same W[64] schedule, same round
 * constants. Expected priv_mem_size matches SHA256.
 *
 * PERF-FIX LESSON (from gpu_md5_core.cl rev 1.2): template_finalize
 * builds M[16] DIRECTLY from input bytes via shifts/OR-merge, not via
 * a uchar pad[64] tail-block round-trip. Same pattern as SHA1/SHA256.
 */

/* Per-algorithm geometry. The kernel-cache key (R3 fix) hashes the
 * defines_str ("HASH_WORDS=7,HASH_BLOCK_BYTES=64") so this instantiation
 * receives a distinct cache entry from SHA256's (HASH_WORDS=8). */
#ifndef HASH_WORDS
#define HASH_WORDS 7
#endif
#ifndef HASH_BLOCK_BYTES
#define HASH_BLOCK_BYTES 64
#endif

/* Per-lane state struct: SHA224 carries 8 uint32 chaining values
 * INTERNALLY (same compression function as SHA256). The struct is
 * literally sized at 8 — NOT HASH_WORDS — because the compression
 * function reads/writes 8 words per call. Output truncation (HASH_WORDS
 * = 7) is enforced at digest-out / digest_compare / emit_hit time by
 * iterating to 7, not by shrinking the struct.
 *
 * This is the only algorithm in the B4 fan-out that has this width
 * asymmetry; B5/B6 SHA384 (state 8 ulong, output 6 ulong) will follow
 * the same convention. */
typedef struct {
    uint h[8];
} template_state;

/* template_init: install SHA224 IV into state.
 * Standard SHA224 initial hash values (FIPS 180-4 §5.3.2). 8 values —
 * the compression function uses all 8 as the chaining IV. */
static inline void template_init(template_state *st) {
    st->h[0] = 0xc1059ed8u;
    st->h[1] = 0x367cd507u;
    st->h[2] = 0x3070dd17u;
    st->h[3] = 0xf70e5939u;
    st->h[4] = 0xffc00b31u;
    st->h[5] = 0x68581511u;
    st->h[6] = 0x64f98fa7u;
    st->h[7] = 0xbefa4fa4u;
}

/* template_transform: absorb one HASH_BLOCK_BYTES (64) byte block.
 * SHA224 reads message words BIG-ENDIAN (same as SHA256). */
static inline void template_transform(template_state *st,
                                      const uchar *block)
{
    uint M[16];
    for (int j = 0; j < 16; j++) {
        int b = j * 4;
        M[j] = ((uint)block[b]     << 24)
             | ((uint)block[b + 1] << 16)
             | ((uint)block[b + 2] << 8)
             |  (uint)block[b + 3];
    }
    sha256_block(&st->h[0], M);
}

/* template_finalize: identical to SHA256's — same compression, same
 * BE message-word load, same 64-bit BE length encoding. The output
 * truncation (7 words instead of 8) happens at digest-out time, not
 * here. After return, st->h[0..7] holds 8 SHA256-style chaining words
 * in BIG-ENDIAN; the template body emits only the leading 7. */
static inline void template_finalize(template_state *st,
                                     const uchar *data, int len)
{
    uint M[16];
    int pos = 0;

    while (len - pos >= HASH_BLOCK_BYTES) {
        for (int j = 0; j < 16; j++) {
            int b = pos + j * 4;
            M[j] = ((uint)data[b]     << 24)
                 | ((uint)data[b + 1] << 16)
                 | ((uint)data[b + 2] << 8)
                 |  (uint)data[b + 3];
        }
        sha256_block(&st->h[0], M);
        pos += HASH_BLOCK_BYTES;
    }

    int rem = len - pos;  /* 0..63 */

    for (int j = 0; j < 16; j++) M[j] = 0;

    /* Tail bytes BIG-ENDIAN. */
    for (int i = 0; i < rem; i++) {
        int wi = i >> 2;
        int bi = 3 - (i & 3);
        M[wi] |= ((uint)data[pos + i]) << (bi * 8);
    }
    /* 0x80 padding marker. */
    {
        int wi = rem >> 2;
        int bi = 3 - (rem & 3);
        M[wi] |= ((uint)0x80u) << (bi * 8);
    }

    if (rem < 56) {
        M[14] = 0;
        M[15] = (uint)((uint)len * 8u);
        sha256_block(&st->h[0], M);
    } else {
        sha256_block(&st->h[0], M);
        for (int j = 0; j < 16; j++) M[j] = 0;
        M[14] = 0;
        M[15] = (uint)((uint)len * 8u);
        sha256_block(&st->h[0], M);
    }
}

/* template_iterate: -i loop step. SHA224 hex output is 56 lowercase
 * hex chars (28 digest bytes × 2). 56 < 64 so the pad fits in the
 * SAME block — single-block iter (vs SHA256 which needs two).
 *
 * Layout:
 *   M[0..13] = 56 hex bytes packed BE into 14 uint32 words (each word
 *              holds 4 hex chars BE)
 *   M[14] = high 32 bits of bit count = 0
 *   M[15] = low 32 bits of bit count = 56 * 8 = 448
 *   But we also need the 0x80 pad byte right after the hex. 56 hex
 *   chars fill M[0..13] exactly (14 * 4 = 56), so 0x80 lands in M[14]
 *   at the high octet — UNLESS we'd overrun. Wait: 14 uint32 = 56
 *   bytes. 0x80 must go at byte 56, which is M[14] high octet.
 *   Then the length must go in M[14..15]. That's a conflict.
 *
 * Resolution: 56 + 1 (for 0x80) + 8 (length) = 65 bytes. Exceeds 64,
 * so we DO need a second block.
 *
 * Block 1: M[0..13] = 56 hex bytes BE; M[14] = 0x80000000u (BE 0x80
 * at byte position 56); M[15] = 0.
 * Block 2: M[0..13] = 0; M[14] = 0; M[15] = 56 * 8 = 448. */
static inline void template_iterate(template_state *st)
{
    uint M[16];
    /* Inlined hex encoder — per-byte BE hex. 7 state words → 14 hex
     * uint32 words (M[0..13]). Note: we ENCODE only the FIRST 7 state
     * words (HASH_WORDS) since SHA224's output is 28 bytes / 56 hex. */
    for (int i = 0; i < 7; i++) {
        uint s = st->h[i];
        uint b0 = (s >> 24) & 0xff;
        uint b1 = (s >> 16) & 0xff;
        uint b2 = (s >> 8)  & 0xff;
        uint b3 = s & 0xff;
        uint hi0 = (b0 >> 4) & 0xf, lo0 = b0 & 0xf;
        uint hi1 = (b1 >> 4) & 0xf, lo1 = b1 & 0xf;
        uint hi2 = (b2 >> 4) & 0xf, lo2 = b2 & 0xf;
        uint hi3 = (b3 >> 4) & 0xf, lo3 = b3 & 0xf;
        uint h0 = ((hi0 + ((hi0 < 10) ? '0' : ('a' - 10))) << 8)
                |  (lo0 + ((lo0 < 10) ? '0' : ('a' - 10)));
        uint h1 = ((hi1 + ((hi1 < 10) ? '0' : ('a' - 10))) << 8)
                |  (lo1 + ((lo1 < 10) ? '0' : ('a' - 10)));
        uint h2 = ((hi2 + ((hi2 < 10) ? '0' : ('a' - 10))) << 8)
                |  (lo2 + ((lo2 < 10) ? '0' : ('a' - 10)));
        uint h3 = ((hi3 + ((hi3 < 10) ? '0' : ('a' - 10))) << 8)
                |  (lo3 + ((lo3 < 10) ? '0' : ('a' - 10)));
        M[i * 2]     = (h0 << 16) | h1;
        M[i * 2 + 1] = (h2 << 16) | h3;
    }
    M[14] = 0x80000000u;   /* 0x80 BE at byte position 56 */
    M[15] = 0u;
    /* Reset state to SHA224 IV; absorb first hex block. */
    st->h[0] = 0xc1059ed8u; st->h[1] = 0x367cd507u;
    st->h[2] = 0x3070dd17u; st->h[3] = 0xf70e5939u;
    st->h[4] = 0xffc00b31u; st->h[5] = 0x68581511u;
    st->h[6] = 0x64f98fa7u; st->h[7] = 0xbefa4fa4u;
    sha256_block(&st->h[0], M);

    /* Second block: pad + length only. */
    for (int j = 0; j < 14; j++) M[j] = 0u;
    M[14] = 0u;
    M[15] = 56u * 8u;       /* 56 hex chars = 448 bits */
    sha256_block(&st->h[0], M);
}

/* template_digest_compare: probe the compact table with the final
 * digest. Same leading-16-byte LE convention as SHA256: bswap32 the
 * leading 4 state words before probe_compact_idx. */
static inline int template_digest_compare(
    const template_state *st,
    __global const uint *compact_fp, __global const uint *compact_idx,
    ulong compact_mask, uint max_probe, uint hash_data_count,
    __global const uchar *hash_data_buf, __global const ulong *hash_data_off,
    __global const ulong *overflow_keys, __global const uchar *overflow_hashes,
    __global const uint *overflow_offsets, uint overflow_count,
    uint *out_idx)
{
    uint h0 = bswap32(st->h[0]);
    uint h1 = bswap32(st->h[1]);
    uint h2 = bswap32(st->h[2]);
    uint h3 = bswap32(st->h[3]);
    return probe_compact_idx(
        h0, h1, h2, h3,
        compact_fp, compact_idx,
        compact_mask, max_probe, hash_data_count,
        hash_data_buf, hash_data_off,
        overflow_keys, overflow_hashes, overflow_offsets, overflow_count,
        out_idx);
}

/* template_emit_hit: emit a hit into the global hits buffer. Wraps
 * EMIT_HIT_7 (SHA224 = 7 uint32 digest words; the 8th internal word
 * st->h[7] is intentionally dropped). Hits buffer convention matches
 * gpu_sha256_packed.cl SHA224 path. */
#define template_emit_hit(hits, hit_count, max_hits, st, widx, sidx, iter) \
    do { \
        uint _h[7]; \
        _h[0] = bswap32((st)->h[0]); \
        _h[1] = bswap32((st)->h[1]); \
        _h[2] = bswap32((st)->h[2]); \
        _h[3] = bswap32((st)->h[3]); \
        _h[4] = bswap32((st)->h[4]); \
        _h[5] = bswap32((st)->h[5]); \
        _h[6] = bswap32((st)->h[6]); \
        EMIT_HIT_7((hits), (hit_count), (max_hits), \
                   (widx), (sidx), (iter), _h) \
    } while (0)

/* B3 dedup+overflow-aware variant. See gpu_md5_core.cl for protocol
 * notes. */
#define template_emit_hit_or_overflow(hits, hit_count, max_hits, st, widx, sidx, iter, \
                                      hashes_shown, matched_idx, dedup_mask, \
                                      ovr_set, ovr_gid, lane_gid) \
    do { \
        uint _h[7]; \
        _h[0] = bswap32((st)->h[0]); \
        _h[1] = bswap32((st)->h[1]); \
        _h[2] = bswap32((st)->h[2]); \
        _h[3] = bswap32((st)->h[3]); \
        _h[4] = bswap32((st)->h[4]); \
        _h[5] = bswap32((st)->h[5]); \
        _h[6] = bswap32((st)->h[6]); \
        EMIT_HIT_7_DEDUP_OR_OVERFLOW((hits), (hit_count), (max_hits), \
                   (widx), (sidx), (iter), _h, \
                   (hashes_shown), (matched_idx), (dedup_mask), \
                   (ovr_set), (ovr_gid), (lane_gid)); \
    } while (0)
