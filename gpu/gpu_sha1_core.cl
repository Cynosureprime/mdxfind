/*
 * $Revision: 1.3 $
 * $Log: gpu_sha1_core.cl,v $
 * Revision 1.3  2026/05/11 05:22:02  dlr
 * Backfill $Revision/$Log RCS keyword stanzas per feedback_rcs_keyword_stanzas.md. Passive 4-line comment block at top of file; no behavioral change. Hand-authored .cl file was missing required stanzas (per memory: all hand-authored .c/.h/.cl/.frag/.tmpl/.py/.sh files MUST contain $Revision/$Log keyword stanzas). Build green on .205 against the post-add files; OpenCL compile strips comments so no kernel behavior change.
 *
 */
/* gpu_sha1_core.cl — SHA1 algorithm extension functions for the generic
 * dispatch template (Memo B Phase B4 — first-algorithm fan-out).
 *
 * Mirrors gpu_md5_core.cl rev 1.2 structure exactly. The template body
 * (gpu_template.cl) is algorithm-agnostic; this file provides the
 * SHA1-specific extension functions:
 *
 *   HASH_WORDS         — digest size in 32-bit words (5 for SHA1)
 *   HASH_BLOCK_BYTES   — compress-block size (64 — same as MD5/SHA-256
 *                        Merkle-Damgård 64-byte block)
 *   template_state     — opaque per-lane state struct (5 uint32 chaining)
 *   template_init      — install SHA1 IV (5 standard values)
 *   template_transform — absorb one 64-byte block (SHA1 BIG-ENDIAN word load)
 *   template_finalize  — pad and finalize (in-place M[16] pattern; see
 *                        the perf-fix lesson at the head of gpu_md5_core.cl)
 *   template_iterate   — re-hash digest as 40-byte hex_lc (-i loop)
 *   template_digest_compare — probe compact table with leading 16 bytes
 *                             (h[0..3], LE-byteswapped to match the
 *                             host-side compact table key encoding)
 *   template_emit_hit       — EMIT_HIT_5 wrapper (5 LE digest words)
 *
 * KEY DIFFERENCES FROM gpu_md5_core.cl:
 *
 * 1. Endianness. SHA1 reads message words in BIG-ENDIAN (RFC 3174); MD5
 *    reads LITTLE-ENDIAN (RFC 1321). The template_finalize in-place M[]
 *    build below packs bytes BE, length-encodes the bit count BE in
 *    M[15], and leaves M[14] = 0 (high-order 32 bits of the 64-bit BE
 *    bit count). md5's was M[14] LE, M[15] = 0.
 *
 * 2. State width. SHA1 = 5 uint32; MD5 = 4. This file uses HASH_WORDS = 5
 *    (compile-time defined via -D from gpu_kernel_cache_build_program_ex's
 *    defines_str). The state struct is uint h[HASH_WORDS]; the template
 *    body and digest_compare helper read at most HASH_WORDS entries.
 *
 * 3. Compact table probe. The compact table is keyed on the leading 16
 *    BYTES of the digest, in the algorithm's NATIVE byte order. SHA1's
 *    state is stored big-endian internally (matches sha1_block's chaining
 *    output convention); to match the host-side compact_table_register
 *    keying (which reads bytes from the digest as little-endian uint32
 *    pairs), we byte-swap state[0..3] -> probe_h[0..3] LE before calling
 *    probe_compact_idx. This matches gpu_sha1_packed.cl's existing path.
 *
 * 4. Iter loop. SHA1 hex output is 40 lowercase hex chars (vs MD5's 32).
 *    template_iterate fills 10 uint M[10] with the hex bytes (BE word
 *    encoding to match SHA1's BE message word layout), pads, and
 *    re-runs sha1_block. The iter step matches gpu_sha1_packed.cl's
 *    sha1_to_hex_lc_p + state-reset + sha1_compress sequence.
 *
 * 5. EMIT_HIT_5 vs EMIT_HIT_4. 5-word emit; widx/sidx/iter triple
 *     prepend stays the same.
 *
 * Source order at compile time (mirrors MD5 build):
 *
 *   [ gpu_common_str, gpu_md5_rules_str, gpu_sha1_core_str, gpu_template_str ]
 *
 * gpu_common_str provides sha1_block(), bswap32, EMIT_HIT_5, and
 * probe_compact_idx. gpu_md5_rules_str provides apply_rule (the rules
 * walker is algorithm-agnostic — it only mutates a uchar buffer in
 * place). gpu_sha1_core_str (this file) provides the per-algorithm
 * hooks. gpu_template_str ties them into template_phase0.
 *
 * Bytecast invariants:
 *   - Final digest h[0..4] (after template_finalize) is BIG-ENDIAN —
 *     the same byte order sha1_block writes back into state[0..4].
 *     gpu_sha1_packed.cl matches this.
 *   - template_digest_compare byte-swaps the leading 4 words to LE
 *     before probing. This matches the host's compact_table key encoding
 *     and gpu_sha1_packed.cl's existing `for j: h[j] = bswap32(state[j])`
 *     pattern at line 88-92.
 *   - template_emit_hit writes the LE-byteswapped digest into the hits
 *     buffer (same as gpu_sha1_packed.cl), so the host hit-replay
 *     matches the legacy SHA1 packed kernel.
 *
 * R1 mitigation (AMD ROCm comgr addrspace fragility): single private
 * buffer pattern. No __private uchar* helpers; no addrspace-cast
 * ternaries. apply_rule + buf are unchanged from MD5; the algorithm
 * core only operates on the post-rule buf and emits LE digest words.
 *
 * R2 (register pressure): SHA1's W[80] schedule is built on-stack inside
 * sha1_block (gpu_common.cl line 437) — 80 uint32 = 320 bytes private,
 * plus 5-word state, plus 5-word transient probe_h. Per-lane VGPR
 * footprint is comparable to MD5's M[16] schedule + 4-word state.
 * Expected priv_mem_size on gfx1201 ~ similar to MD5's reading.
 *
 * PERF-FIX LESSON (from gpu_md5_core.cl rev 1.2):
 *
 * template_finalize MUST build M[16] DIRECTLY from input bytes via
 * shifts/OR-merge — NOT via a private uchar pad[64] tail-block buffer
 * that is then read BACK out as message words. The byte-store /
 * byte-load round-trip cost was the 12.3% wall regression in B2's
 * first-cut MD5 template path. This file applies the same pattern for
 * SHA1: only difference is BE byte ordering of the M[] words and the
 * length encoding being BE in M[15] (low 32 bits of 64-bit count).
 */

/* Per-algorithm geometry. The template uses these as structural
 * compile-time constants; the kernel-cache key (gpu_kernel_cache R3
 * fix) hashes the defines_str ("HASH_WORDS=5,HASH_BLOCK_BYTES=64")
 * alongside the source set so distinct instantiations get distinct
 * cache entries even though source text would otherwise be identical. */
#ifndef HASH_WORDS
#define HASH_WORDS 5
#endif
#ifndef HASH_BLOCK_BYTES
#define HASH_BLOCK_BYTES 64
#endif

/* Per-lane state struct: SHA1 carries 5 uint32 chaining values (vs MD5's
 * 4). The template only reads/writes the digest words via
 * template_finalize's output + template_digest_compare; it does not
 * introspect the struct. */
typedef struct {
    uint h[HASH_WORDS];
} template_state;

/* template_init: install SHA1 IV into state.
 * Standard SHA1 initial hash values (RFC 3174 §6.1). */
static inline void template_init(template_state *st) {
    st->h[0] = 0x67452301u;
    st->h[1] = 0xEFCDAB89u;
    st->h[2] = 0x98BADCFEu;
    st->h[3] = 0x10325476u;
    st->h[4] = 0xC3D2E1F0u;
}

/* template_transform: absorb one HASH_BLOCK_BYTES (64) byte block.
 * `block` points into the working buffer at the absorb position;
 * the caller must guarantee >=64 bytes are readable. SHA1 reads
 * message words BIG-ENDIAN.
 *
 * Public extension API; the hot-path template_finalize bypasses this
 * via an in-place M[16] pattern (see perf-fix lesson at file head).
 * Kept symmetric with gpu_md5_core.cl for cores that prefer the
 * function-style entry. */
static inline void template_transform(template_state *st,
                                      const uchar *block)
{
    uint M[16];
    /* SHA1 reads message words big-endian. */
    for (int j = 0; j < 16; j++) {
        int b = j * 4;
        M[j] = ((uint)block[b]     << 24)
             | ((uint)block[b + 1] << 16)
             | ((uint)block[b + 2] << 8)
             |  (uint)block[b + 3];
    }
    sha1_block(&st->h[0], M);
}

/* template_finalize: process the tail, append the 0x80 padding marker
 * + 64-bit BE length-in-bits, and absorb. Caller passes the full
 * buffer; we run complete blocks then build the final padded block(s)
 * here. After return, st->h[0..4] holds the final SHA1 digest words
 * in BIG-ENDIAN form (sha1_block's natural output).
 *
 * Mirrors gpu_md5_core.cl rev 1.2's in-place M[16] pattern: build the
 * message words DIRECTLY from input bytes via shifts/OR-merge. The
 * difference vs MD5 is byte ordering (BE for SHA1) and length encoding
 * (BE 64-bit count, low 32 bits in M[15]).
 *
 * PERF-FIX LESSON (carried forward from gpu_md5_core.cl rev 1.2):
 *
 * Do NOT route through template_transform() here. The previous MD5
 * version built a private uchar pad[64] tail block, populated it with
 * 64+rem byte stores, then handed it to template_transform which read
 * the same 64 bytes BACK out as BE uints into a fresh M[16]. That
 * round-trip (byte-stores -> byte-loads -> uint-pack) was the 12.3%
 * wall regression vs the legacy md5_buf path on ioblade (Memo B
 * Phase B2 perf-fix, 2026-05-04).
 *
 * For full blocks the function-call boundary plus rebuilt-M[] cost is
 * non-trivial too; for the final block the extra 64-byte zeroing +
 * byte-by-byte tail copy + 4-byte length store dominates because rem
 * is small for typical wordlist inputs. We build M[] in-place and
 * call sha1_block() once per block.
 *
 * R1 mitigation preserved: single private buffer (just the M[16]
 * uints + the input data pointer), no addrspace-cast helpers, no
 * __private uchar* helper that takes a private buffer pointer. */
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
        sha1_block(&st->h[0], M);
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
        /* Length fits in this block. SHA1 BE: M[14] = high 32 bits
         * of bit count, M[15] = low 32 bits. For len < 2^29 bytes
         * (always true for our wordlist inputs), high 32 bits = 0. */
        M[14] = 0;
        M[15] = (uint)((uint)len * 8u);
        sha1_block(&st->h[0], M);
    } else {
        /* Need one extra padding-only block to hold the length. */
        sha1_block(&st->h[0], M);
        for (int j = 0; j < 16; j++) M[j] = 0;
        M[14] = 0;
        M[15] = (uint)((uint)len * 8u);
        sha1_block(&st->h[0], M);
    }
}

/* template_iterate: -i loop step. Re-encode the digest as 40-byte
 * lowercase hex ASCII and rehash. Algorithm-specific because the
 * digest geometry differs (SHA1: 5 words, 40 chars).
 *
 * Mirrors gpu_sha1_packed.cl rev 1 lines 100-108 byte-for-byte:
 *   - 5 state words -> 10 BE-encoded hex M[] words (2 words per state
 *     word, 4 hex chars each — matches sha1_to_hex_lc_p)
 *   - M[10] = 0x80000000u   (BE 0x80 padding at byte offset 40)
 *   - M[11..14] = 0
 *   - M[15] = 40 * 8 = 320 (bit count)
 *   - state reset to SHA1 IV; sha1_block. */
static inline void template_iterate(template_state *st)
{
    uint M[16];
    /* Inlined sha1_to_hex_lc_p body — keeps gpu_sha1_core self-contained
     * (gpu_sha1_packed.cl's hex_byte_be_p / sha1_to_hex_lc_p are not
     * exported through gpu_common.cl). Per-byte BE hex encoding. */
    for (int i = 0; i < 5; i++) {
        uint s = st->h[i];
        uint b0 = (s >> 24) & 0xff;
        uint b1 = (s >> 16) & 0xff;
        uint b2 = (s >> 8)  & 0xff;
        uint b3 = s & 0xff;
        /* hex_byte_be: produce two hex chars in a 16-bit BE pair. */
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
    M[10] = 0x80000000u;
    for (int j = 11; j < 15; j++) M[j] = 0u;
    M[15] = 40u * 8u;        /* 40 hex chars = 320 bits */
    /* Reinitialize state to SHA1 IV, then absorb the prepared M[]. */
    st->h[0] = 0x67452301u;
    st->h[1] = 0xEFCDAB89u;
    st->h[2] = 0x98BADCFEu;
    st->h[3] = 0x10325476u;
    st->h[4] = 0xC3D2E1F0u;
    sha1_block(&st->h[0], M);
}

/* template_digest_compare: probe the compact table with the final
 * digest. On a hit, *out_idx is set to the matched target's hash_data
 * index (mirrors probe_compact_idx semantics). Return 1 on hit, 0 on
 * miss.
 *
 * SHA1 state is stored big-endian (sha1_block's natural form). The
 * compact table is keyed on the leading 16 BYTES of the digest in
 * little-endian uint32 form (same convention as MD5 — host-side
 * compact_table_register reads the digest bytes as 4 LE uint32 to
 * form the key). We byte-swap the leading 4 words to LE before
 * calling probe_compact_idx.
 *
 * Mirrors gpu_sha1_packed.cl line 88-92's `for j: h[j] = bswap32(state[j])`
 * pattern verbatim. The 5th state word (h[4]) is NOT used by the
 * compact-table probe — only by EMIT_HIT_5 below. */
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
 * EMIT_HIT_5 (SHA1 = 5 uint32 digest words). Hits buffer convention
 * matches gpu_sha1_packed.cl: 5 LE-byteswapped digest words after the
 * widx/sidx/iter triple.
 *
 * The macro builds a private uint h[5] of LE-byteswapped state, then
 * passes it to EMIT_HIT_5 (which expects an array, not 5 individual
 * args). Same wire format as gpu_sha1_packed.cl line 98 emit. */
#define template_emit_hit(hits, hit_count, max_hits, st, widx, sidx, iter) \
    do { \
        uint _h[5]; \
        _h[0] = bswap32((st)->h[0]); \
        _h[1] = bswap32((st)->h[1]); \
        _h[2] = bswap32((st)->h[2]); \
        _h[3] = bswap32((st)->h[3]); \
        _h[4] = bswap32((st)->h[4]); \
        EMIT_HIT_5((hits), (hit_count), (max_hits), \
                   (widx), (sidx), (iter), _h) \
    } while (0)

/* B3 dedup+overflow-aware variant. See gpu_md5_core.cl for protocol
 * notes. */
#define template_emit_hit_or_overflow(hits, hit_count, max_hits, st, widx, sidx, iter, \
                                      hashes_shown, matched_idx, dedup_mask, \
                                      ovr_set, ovr_gid, lane_gid) \
    do { \
        uint _h[5]; \
        _h[0] = bswap32((st)->h[0]); \
        _h[1] = bswap32((st)->h[1]); \
        _h[2] = bswap32((st)->h[2]); \
        _h[3] = bswap32((st)->h[3]); \
        _h[4] = bswap32((st)->h[4]); \
        EMIT_HIT_5_DEDUP_OR_OVERFLOW((hits), (hit_count), (max_hits), \
                   (widx), (sidx), (iter), _h, \
                   (hashes_shown), (matched_idx), (dedup_mask), \
                   (ovr_set), (ovr_gid), (lane_gid)); \
    } while (0)
