/*
 * $Revision: 1.3 $
 * $Log: gpu_sha256_core.cl,v $
 * Revision 1.3  2026/05/11 05:22:02  dlr
 * Backfill $Revision/$Log RCS keyword stanzas per feedback_rcs_keyword_stanzas.md. Passive 4-line comment block at top of file; no behavioral change. Hand-authored .cl file was missing required stanzas (per memory: all hand-authored .c/.h/.cl/.frag/.tmpl/.py/.sh files MUST contain $Revision/$Log keyword stanzas). Build green on .205 against the post-add files; OpenCL compile strips comments so no kernel behavior change.
 *
 */
/* gpu_sha256_core.cl — SHA256 algorithm extension functions for the
 * generic dispatch template (Memo B Phase B4 fan-out).
 *
 * Mirrors gpu_sha1_core.cl rev 1.1 structure exactly. The template body
 * (gpu_template.cl) is algorithm-agnostic; this file provides the
 * SHA256-specific extension functions:
 *
 *   HASH_WORDS         — digest size in 32-bit words (8 for SHA256)
 *   HASH_BLOCK_BYTES   — compress-block size (64 — same Merkle-Damgård
 *                        block geometry as MD5/SHA1)
 *   template_state     — opaque per-lane state struct (8 uint32 chaining)
 *   template_init      — install SHA256 IV (8 standard values)
 *   template_transform — absorb one 64-byte block (BIG-ENDIAN word load)
 *   template_finalize  — pad and finalize (in-place M[16] pattern; see
 *                        the perf-fix lesson at the head of gpu_md5_core.cl)
 *   template_iterate   — re-hash digest as 64-byte hex_lc (-i loop)
 *   template_digest_compare — probe compact table with leading 16 bytes
 *                             (h[0..3], LE-byteswapped to match the
 *                             host-side compact table key encoding)
 *   template_emit_hit       — EMIT_HIT_8 wrapper (8 LE digest words)
 *
 * KEY POINTS vs gpu_sha1_core.cl:
 *
 * 1. State width. SHA256 carries 8 uint32 chaining values (vs SHA1's 5).
 *    The state struct is uint h[HASH_WORDS] = uint h[8].
 *
 * 2. Message schedule. SHA256 uses W[0..63] with the recurrence
 *    W[t] = SSIG1(W[t-2]) + W[t-7] + SSIG0(W[t-15]) + W[t-16];
 *    sha256_block (gpu_common.cl line 484) handles this.
 *
 * 3. BIG-ENDIAN identical to SHA1: message words BE-loaded, length
 *    encoded as 64-bit BE in M[14..15] (high 32 in M[14], low in M[15]).
 *    For our wordlist inputs len < 2^29 bytes always, so M[14] = 0 and
 *    M[15] = len * 8. Same convention as gpu_sha256_packed.cl line 76-78.
 *
 * 4. Iter loop. SHA256 hex output is 64 lowercase hex chars, which
 *    exactly fills one HASH_BLOCK_BYTES block. Pad lands in the SECOND
 *    block: M[0]=0x80000000 BE, M[1..14]=0, M[15]=64*8=512. Mirrors
 *    gpu_sha256_packed.cl rev 1's iter step (lines 100-112) byte-for-byte.
 *
 * 5. Compact table probe. Same leading-16-byte LE convention as MD5 and
 *    SHA1: bswap32 the leading 4 state words before probe_compact_idx.
 *    The remaining 4 words (h[4..7]) are NOT used by the probe — only
 *    by EMIT_HIT_8.
 *
 * 6. EMIT_HIT_8: 8-word emit; widx/sidx/iter triple prepend stays the same.
 *
 * Source order at compile time (mirrors SHA1 build):
 *
 *   [ gpu_common_str, gpu_md5_rules_str, gpu_sha256_core_str, gpu_template_str ]
 *
 * gpu_common_str provides sha256_block(), bswap32, EMIT_HIT_8, and
 * probe_compact_idx. gpu_md5_rules_str provides apply_rule (the rules
 * walker is algorithm-agnostic). gpu_sha256_core_str (this file) provides
 * the per-algorithm hooks. gpu_template_str ties them into template_phase0.
 *
 * Bytecast invariants:
 *   - Final digest h[0..7] (after template_finalize) is BIG-ENDIAN —
 *     the same byte order sha256_block writes back into state[0..7].
 *     gpu_sha256_packed.cl matches this.
 *   - template_digest_compare byte-swaps the leading 4 words to LE
 *     before probing. Matches gpu_sha256_packed.cl line 90's
 *     `for j: h[j] = bswap32(state[j])` pattern.
 *   - template_emit_hit writes the LE-byteswapped digest into the hits
 *     buffer (same as gpu_sha256_packed.cl), so host hit-replay matches
 *     the legacy SHA256 packed kernel.
 *
 * R1 mitigation (AMD ROCm comgr addrspace fragility): single private
 * buffer pattern. No __private uchar* helpers; no addrspace-cast
 * ternaries. apply_rule + buf are unchanged from MD5; the algorithm
 * core only operates on the post-rule buf and emits LE digest words.
 *
 * R2 (register pressure): SHA256's W[64] schedule is built on-stack
 * inside sha256_block (gpu_common.cl line 485) — 64 uint32 = 256 bytes
 * private, plus 8-word state, plus 4-word transient probe_h. Per-lane
 * VGPR footprint is moderately larger than SHA1's W[80] (320 bytes)
 * but with simpler round constants. Expected priv_mem_size on gfx1201
 * comparable to SHA1's reading, well below the 240-byte risk threshold
 * (the ~41 KB measurements seen on SHA1/MD5 are dominated by the
 * RULE_BUF_MAX = 40960 byte buf[] array on the rules walker, not the
 * compression scratch). Watching this for SHA256 is a B5 prep for the
 * SHA512 family.
 *
 * PERF-FIX LESSON (from gpu_md5_core.cl rev 1.2, repeated for SHA1
 * rev 1.1, applied here):
 *
 * template_finalize MUST build M[16] DIRECTLY from input bytes via
 * shifts/OR-merge — NOT via a private uchar pad[64] tail-block buffer
 * that is then read BACK out as message words. This file applies the
 * same pattern: BE byte ordering of M[] words and 64-bit BE length
 * encoding in M[14..15].
 */

/* Per-algorithm geometry. The template uses these as structural
 * compile-time constants; the kernel-cache key (gpu_kernel_cache R3
 * fix) hashes the defines_str ("HASH_WORDS=8,HASH_BLOCK_BYTES=64")
 * alongside the source set so distinct instantiations get distinct
 * cache entries even though source text would otherwise be identical. */
#ifndef HASH_WORDS
#define HASH_WORDS 8
#endif
#ifndef HASH_BLOCK_BYTES
#define HASH_BLOCK_BYTES 64
#endif

/* Per-lane state struct: SHA256 carries 8 uint32 chaining values. The
 * template only reads/writes the digest words via template_finalize's
 * output + template_digest_compare; it does not introspect the struct. */
typedef struct {
    uint h[HASH_WORDS];
} template_state;

/* template_init: install SHA256 IV into state.
 * Standard SHA256 initial hash values (FIPS 180-4 §5.3.3). */
static inline void template_init(template_state *st) {
    st->h[0] = 0x6a09e667u;
    st->h[1] = 0xbb67ae85u;
    st->h[2] = 0x3c6ef372u;
    st->h[3] = 0xa54ff53au;
    st->h[4] = 0x510e527fu;
    st->h[5] = 0x9b05688cu;
    st->h[6] = 0x1f83d9abu;
    st->h[7] = 0x5be0cd19u;
}

/* template_transform: absorb one HASH_BLOCK_BYTES (64) byte block.
 * `block` points into the working buffer at the absorb position;
 * the caller must guarantee >=64 bytes are readable. SHA256 reads
 * message words BIG-ENDIAN.
 *
 * Public extension API; the hot-path template_finalize bypasses this
 * via an in-place M[16] pattern (see perf-fix lesson at file head).
 * Kept symmetric with gpu_md5_core.cl/gpu_sha1_core.cl for cores that
 * prefer the function-style entry. */
static inline void template_transform(template_state *st,
                                      const uchar *block)
{
    uint M[16];
    /* SHA256 reads message words big-endian. */
    for (int j = 0; j < 16; j++) {
        int b = j * 4;
        M[j] = ((uint)block[b]     << 24)
             | ((uint)block[b + 1] << 16)
             | ((uint)block[b + 2] << 8)
             |  (uint)block[b + 3];
    }
    sha256_block(&st->h[0], M);
}

/* template_finalize: process the tail, append the 0x80 padding marker
 * + 64-bit BE length-in-bits, and absorb. Caller passes the full
 * buffer; we run complete blocks then build the final padded block(s)
 * here. After return, st->h[0..7] holds the final SHA256 digest words
 * in BIG-ENDIAN form (sha256_block's natural output).
 *
 * Mirrors gpu_sha1_core.cl rev 1.1's in-place M[16] pattern. The
 * difference vs SHA1 is 8 chaining words (vs 5) and the SHA256 round
 * constants/schedule (handled by sha256_block).
 *
 * R1 mitigation preserved: single private buffer (M[16] uints + the
 * input data pointer), no addrspace-cast helpers. */
static inline void template_finalize(template_state *st,
                                     const uchar *data, int len)
{
    uint M[16];
    int pos = 0;

    /* Process complete 64-byte blocks. Build M[] BIG-ENDIAN directly
     * from bytes. */
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

    /* Build final block(s): tail bytes + 0x80 marker + zeros + 64-bit
     * BE length. Direct-into-M[] approach — no intermediate uchar
     * pad[]. */
    int rem = len - pos;  /* 0..63 */

    /* Zero scratch. */
    for (int j = 0; j < 16; j++) M[j] = 0;

    /* Copy remaining tail bytes into M[] big-endian. Each byte's
     * position within its 32-bit word is determined by 3-(byte_idx & 3)
     * shift (BE: byte 0 is high octet of word). */
    for (int i = 0; i < rem; i++) {
        int wi = i >> 2;
        int bi = 3 - (i & 3);
        M[wi] |= ((uint)data[pos + i]) << (bi * 8);
    }
    /* 0x80 padding marker, BE byte position. */
    {
        int wi = rem >> 2;
        int bi = 3 - (rem & 3);
        M[wi] |= ((uint)0x80u) << (bi * 8);
    }

    if (rem < 56) {
        /* Length fits in this block. SHA256 BE: M[14] = high 32 bits
         * of bit count, M[15] = low 32 bits. For len < 2^29 bytes
         * (always true for our wordlist inputs), high 32 bits = 0. */
        M[14] = 0;
        M[15] = (uint)((uint)len * 8u);
        sha256_block(&st->h[0], M);
    } else {
        /* Need one extra padding-only block to hold the length. */
        sha256_block(&st->h[0], M);
        for (int j = 0; j < 16; j++) M[j] = 0;
        M[14] = 0;
        M[15] = (uint)((uint)len * 8u);
        sha256_block(&st->h[0], M);
    }
}

/* template_iterate: -i loop step. Re-encode the digest as 64-byte
 * lowercase hex ASCII and rehash. SHA256's hex output exactly fills
 * one 64-byte block, so the pad lands in a SECOND block.
 *
 * Mirrors gpu_sha256_packed.cl rev 1 lines 100-112 byte-for-byte:
 *   - 8 state words -> 16 BE-encoded hex M[] words (2 words per state
 *     word, 4 hex chars each — matches sha256_to_hex_lc_p)
 *   - state reset to SHA256 IV; sha256_block(M)
 *   - second block: M[0] = 0x80000000u BE, M[1..14] = 0, M[15] = 64*8 = 512
 *   - sha256_block(M) again. */
static inline void template_iterate(template_state *st)
{
    uint M[16];
    /* Inlined sha256_to_hex_lc_p body — keeps gpu_sha256_core
     * self-contained. Per-byte BE hex encoding. */
    for (int i = 0; i < 8; i++) {
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
    /* Reset state to SHA256 IV; absorb first hex block (full 64 bytes). */
    st->h[0] = 0x6a09e667u; st->h[1] = 0xbb67ae85u;
    st->h[2] = 0x3c6ef372u; st->h[3] = 0xa54ff53au;
    st->h[4] = 0x510e527fu; st->h[5] = 0x9b05688cu;
    st->h[6] = 0x1f83d9abu; st->h[7] = 0x5be0cd19u;
    sha256_block(&st->h[0], M);

    /* Second block: pad + length only. */
    M[0] = 0x80000000u;
    for (int j = 1; j < 15; j++) M[j] = 0u;
    M[15] = 64u * 8u;       /* 64 hex chars = 512 bits */
    /* M[14] left from the previous block — zero it explicitly to be safe. */
    M[14] = 0u;
    sha256_block(&st->h[0], M);
}

/* template_digest_compare: probe the compact table with the final
 * digest. On a hit, *out_idx is set to the matched target's hash_data
 * index. Return 1 on hit, 0 on miss.
 *
 * SHA256 state is stored big-endian (sha256_block's natural form). The
 * compact table is keyed on the leading 16 BYTES of the digest in
 * little-endian uint32 form (host-side compact_table_register reads
 * the digest bytes as 4 LE uint32 to form the key). We byte-swap the
 * leading 4 words to LE before calling probe_compact_idx.
 *
 * Mirrors gpu_sha256_packed.cl line 90's `for j: h[j] = bswap32(state[j])`
 * pattern verbatim. The remaining state words (h[4..7]) are NOT used
 * by the compact-table probe — only by EMIT_HIT_8 below. */
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
 * EMIT_HIT_8 (SHA256 = 8 uint32 digest words). Hits buffer convention
 * matches gpu_sha256_packed.cl: 8 LE-byteswapped digest words after the
 * widx/sidx/iter triple.
 *
 * The macro builds a private uint h[8] of LE-byteswapped state, then
 * passes it to EMIT_HIT_8 (which expects an array, not 8 individual
 * args). Same wire format as gpu_sha256_packed.cl line 98 emit. */
#define template_emit_hit(hits, hit_count, max_hits, st, widx, sidx, iter) \
    do { \
        uint _h[8]; \
        _h[0] = bswap32((st)->h[0]); \
        _h[1] = bswap32((st)->h[1]); \
        _h[2] = bswap32((st)->h[2]); \
        _h[3] = bswap32((st)->h[3]); \
        _h[4] = bswap32((st)->h[4]); \
        _h[5] = bswap32((st)->h[5]); \
        _h[6] = bswap32((st)->h[6]); \
        _h[7] = bswap32((st)->h[7]); \
        EMIT_HIT_8((hits), (hit_count), (max_hits), \
                   (widx), (sidx), (iter), _h) \
    } while (0)

/* B3 dedup+overflow-aware variant. See gpu_md5_core.cl for protocol
 * notes. */
#define template_emit_hit_or_overflow(hits, hit_count, max_hits, st, widx, sidx, iter, \
                                      hashes_shown, matched_idx, dedup_mask, \
                                      ovr_set, ovr_gid, lane_gid) \
    do { \
        uint _h[8]; \
        _h[0] = bswap32((st)->h[0]); \
        _h[1] = bswap32((st)->h[1]); \
        _h[2] = bswap32((st)->h[2]); \
        _h[3] = bswap32((st)->h[3]); \
        _h[4] = bswap32((st)->h[4]); \
        _h[5] = bswap32((st)->h[5]); \
        _h[6] = bswap32((st)->h[6]); \
        _h[7] = bswap32((st)->h[7]); \
        EMIT_HIT_8_DEDUP_OR_OVERFLOW((hits), (hit_count), (max_hits), \
                   (widx), (sidx), (iter), _h, \
                   (hashes_shown), (matched_idx), (dedup_mask), \
                   (ovr_set), (ovr_gid), (lane_gid)); \
    } while (0)
