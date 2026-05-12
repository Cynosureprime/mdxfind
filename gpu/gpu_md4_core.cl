/*
 * $Revision: 1.3 $
 * $Log: gpu_md4_core.cl,v $
 * Revision 1.3  2026/05/11 05:22:01  dlr
 * Backfill $Revision/$Log RCS keyword stanzas per feedback_rcs_keyword_stanzas.md. Passive 4-line comment block at top of file; no behavioral change. Hand-authored .cl file was missing required stanzas (per memory: all hand-authored .c/.h/.cl/.frag/.tmpl/.py/.sh files MUST contain $Revision/$Log keyword stanzas). Build green on .205 against the post-add files; OpenCL compile strips comments so no kernel behavior change.
 *
 */
/* gpu_md4_core.cl — MD4 algorithm extension functions for the generic
 * dispatch template (Memo B Phase B4 fan-out).
 *
 * Mirrors gpu_md5_core.cl rev 1.2 structure exactly. The template body
 * (gpu_template.cl) is algorithm-agnostic; this file provides the
 * MD4-specific extension functions:
 *
 *   HASH_WORDS         — digest size in 32-bit words (4 for MD4 — same
 *                        as MD5)
 *   HASH_BLOCK_BYTES   — compress-block size (64 — same as MD5)
 *   template_state     — opaque per-lane state struct (4 uint32 chaining)
 *   template_init      — install MD4 IV (same 4 standard values as MD5)
 *   template_transform — absorb one 64-byte block (LITTLE-ENDIAN word load)
 *   template_finalize  — pad and finalize (in-place M[16] pattern)
 *   template_iterate   — re-hash digest as 32-byte hex_lc (-i loop)
 *   template_digest_compare — probe compact table with leading 16 bytes
 *   template_emit_hit       — EMIT_HIT_4 wrapper (4 LE digest words)
 *
 * KEY DIFFERENCES FROM MD5:
 *
 * 1. Compression function. MD4 uses 3 rounds × 16 steps (vs MD5's 4 ×
 *    16). Round functions: F = (X & Y) | (~X & Z); G = (X & Y) | (X & Z) |
 *    (Y & Z); H = X ^ Y ^ Z. Round constants: 0, 0x5A827999, 0x6ED9EBA1.
 *    Inlined here as md4_compress (mirrors gpu_md4_packed.cl rev 1's
 *    md4_compress_p byte-for-byte; gpu_common.cl does NOT export an
 *    md4_block primitive).
 *
 * 2. Endianness same as MD5: LITTLE-ENDIAN message words and LE 64-bit
 *    length encoding (M[14] = low 32 bits of bit count, M[15] = high
 *    bits = 0).
 *
 * 3. IV same 4 values as MD5: 0x67452301, 0xEFCDAB89, 0x98BADCFE,
 *    0x10325476 (RFC 1320 §3.3 / RFC 1321 §3.3 — both algorithms share
 *    these). State is 4 uint32.
 *
 * 4. Hex iter loop same as MD5: 32 hex chars = 16 bytes = quarter of a
 *    block; pad + length fits in one 64-byte block. md5_to_hex_lc from
 *    gpu_common.cl is reusable (LE 4-word hex format is algorithm-
 *    independent — both MD4 and MD5 produce 4 LE uint32 digest words).
 *
 * 5. EMIT_HIT_4 same as MD5.
 *
 * Source order at compile time (mirrors MD5 build):
 *
 *   [ gpu_common_str, gpu_md5_rules_str, gpu_md4_core_str, gpu_template_str ]
 *
 * gpu_common_str provides md5_to_hex_lc, EMIT_HIT_4, OCLParams, and
 * probe_compact_idx. gpu_md5_rules_str provides apply_rule (the rules
 * walker is algorithm-agnostic). gpu_md4_core_str (this file) provides
 * the per-algorithm hooks INCLUDING the inline md4_compress (since
 * gpu_common.cl does not export an md4_block primitive — gpu_md4_-
 * packed.cl carries its own).
 *
 * Bytecast invariants:
 *   - Final digest h[0..3] (after template_finalize) is in NATIVE
 *     order — for MD4 that's little-endian per word (md4_compress
 *     stores LE chaining values). gpu_md4_packed.cl matches this.
 *   - template_digest_compare reads st->h[0..3] directly — no bswap
 *     needed because MD4's natural state order is already LE.
 *     (Contrast with SHA1/SHA256 which store BE and bswap before probe.)
 *   - template_emit_hit writes the 4 LE state words directly (no
 *     bswap — same as MD5).
 *
 * R1 mitigation: single private buffer pattern. The inline
 * md4_compress takes uint pointers to chaining values + a uint M[16];
 * no __private uchar* helpers; no addrspace-cast ternaries.
 *
 * R2 (register pressure): MD4's M[16] schedule is just the 16 message
 * words (no W[80] expansion like SHA1, no W[64] like SHA256) — smaller
 * compression-time footprint than either SHA family. Same overall
 * priv_mem_size as MD5 expected (dominated by the RULE_BUF_MAX = 40960
 * byte buf[]).
 *
 * PERF-FIX LESSON (from gpu_md5_core.cl rev 1.2): template_finalize
 * builds M[16] DIRECTLY from input bytes via shifts/OR-merge — NOT via
 * a uchar pad[64] tail-block round-trip. Same pattern applied here.
 */

/* Per-algorithm geometry. The kernel-cache key (R3 fix) hashes the
 * defines_str ("HASH_WORDS=4,HASH_BLOCK_BYTES=64") — note this is
 * IDENTICAL to MD5's defines_str. Distinct cache entries are guaranteed
 * because the SOURCE TEXT differs (this file vs gpu_md5_core.cl); the
 * kernel-cache hash includes the concatenated source text, not just
 * the defines. */
#ifndef HASH_WORDS
#define HASH_WORDS 4
#endif
#ifndef HASH_BLOCK_BYTES
#define HASH_BLOCK_BYTES 64
#endif

/* Per-lane state struct: MD4 carries 4 uint32 chaining values. */
typedef struct {
    uint h[HASH_WORDS];
} template_state;

/* MD4 compression — fully unrolled, 3 rounds × 16 steps. Mirrors
 * gpu_md4_packed.cl rev 1's md4_compress_p byte-for-byte. The macros
 * use rotate() (OpenCL builtin) for left rotation. State chaining is
 * little-endian per word (same as MD5). */
static inline void md4_compress(uint *hx, uint *hy, uint *hz, uint *hw, uint *M) {
    uint a = *hx, b = *hy, c = *hz, d = *hw;
#define MD4_F(x,y,z) (((x)&(y)) | ((~(x))&(z)))
#define MD4_G(x,y,z) (((x)&(y)) | ((x)&(z)) | ((y)&(z)))
#define MD4_H(x,y,z) ((x)^(y)^(z))
#define MD4_R1(a,b,c,d,k,s) a = rotate(a + MD4_F(b,c,d) + M[k], (uint)(s))
#define MD4_R2(a,b,c,d,k,s) a = rotate(a + MD4_G(b,c,d) + M[k] + 0x5A827999u, (uint)(s))
#define MD4_R3(a,b,c,d,k,s) a = rotate(a + MD4_H(b,c,d) + M[k] + 0x6ED9EBA1u, (uint)(s))
    MD4_R1(a,b,c,d, 0, 3); MD4_R1(d,a,b,c, 1, 7); MD4_R1(c,d,a,b, 2,11); MD4_R1(b,c,d,a, 3,19);
    MD4_R1(a,b,c,d, 4, 3); MD4_R1(d,a,b,c, 5, 7); MD4_R1(c,d,a,b, 6,11); MD4_R1(b,c,d,a, 7,19);
    MD4_R1(a,b,c,d, 8, 3); MD4_R1(d,a,b,c, 9, 7); MD4_R1(c,d,a,b,10,11); MD4_R1(b,c,d,a,11,19);
    MD4_R1(a,b,c,d,12, 3); MD4_R1(d,a,b,c,13, 7); MD4_R1(c,d,a,b,14,11); MD4_R1(b,c,d,a,15,19);
    MD4_R2(a,b,c,d, 0, 3); MD4_R2(d,a,b,c, 4, 5); MD4_R2(c,d,a,b, 8, 9); MD4_R2(b,c,d,a,12,13);
    MD4_R2(a,b,c,d, 1, 3); MD4_R2(d,a,b,c, 5, 5); MD4_R2(c,d,a,b, 9, 9); MD4_R2(b,c,d,a,13,13);
    MD4_R2(a,b,c,d, 2, 3); MD4_R2(d,a,b,c, 6, 5); MD4_R2(c,d,a,b,10, 9); MD4_R2(b,c,d,a,14,13);
    MD4_R2(a,b,c,d, 3, 3); MD4_R2(d,a,b,c, 7, 5); MD4_R2(c,d,a,b,11, 9); MD4_R2(b,c,d,a,15,13);
    MD4_R3(a,b,c,d, 0, 3); MD4_R3(d,a,b,c, 8, 9); MD4_R3(c,d,a,b, 4,11); MD4_R3(b,c,d,a,12,15);
    MD4_R3(a,b,c,d, 2, 3); MD4_R3(d,a,b,c,10, 9); MD4_R3(c,d,a,b, 6,11); MD4_R3(b,c,d,a,14,15);
    MD4_R3(a,b,c,d, 1, 3); MD4_R3(d,a,b,c, 9, 9); MD4_R3(c,d,a,b, 5,11); MD4_R3(b,c,d,a,13,15);
    MD4_R3(a,b,c,d, 3, 3); MD4_R3(d,a,b,c,11, 9); MD4_R3(c,d,a,b, 7,11); MD4_R3(b,c,d,a,15,15);
#undef MD4_F
#undef MD4_G
#undef MD4_H
#undef MD4_R1
#undef MD4_R2
#undef MD4_R3
    *hx = a + *hx; *hy = b + *hy; *hz = c + *hz; *hw = d + *hw;
}

/* template_init: install MD4 IV into state.
 * Same 4 values as MD5 (RFC 1320 §3.3). */
static inline void template_init(template_state *st) {
    st->h[0] = 0x67452301u;
    st->h[1] = 0xEFCDAB89u;
    st->h[2] = 0x98BADCFEu;
    st->h[3] = 0x10325476u;
}

/* template_transform: absorb one HASH_BLOCK_BYTES (64) byte block.
 * MD4 reads message words LITTLE-ENDIAN (same as MD5).
 *
 * Public extension API; the hot-path template_finalize bypasses this
 * via an in-place M[16] pattern. */
static inline void template_transform(template_state *st,
                                      const uchar *block)
{
    uint M[16];
    /* MD4 reads message words little-endian. */
    for (int j = 0; j < 16; j++) {
        int b = j * 4;
        M[j] = (uint)block[b]
             | ((uint)block[b + 1] << 8)
             | ((uint)block[b + 2] << 16)
             | ((uint)block[b + 3] << 24);
    }
    md4_compress(&st->h[0], &st->h[1], &st->h[2], &st->h[3], M);
}

/* template_finalize: process the tail, append the 0x80 padding marker
 * + 64-bit LE length-in-bits, and absorb. Mirrors gpu_md5_core.cl rev
 * 1.2's in-place M[16] pattern. The only difference vs MD5 is the
 * compression function call. */
static inline void template_finalize(template_state *st,
                                     const uchar *data, int len)
{
    uint M[16];
    int pos = 0;

    /* Process complete 64-byte blocks. Build M[] LE directly from
     * bytes (no intermediate pad[]). */
    while (len - pos >= HASH_BLOCK_BYTES) {
        for (int j = 0; j < 16; j++) {
            int b = pos + j * 4;
            M[j] = (uint)data[b]
                 | ((uint)data[b + 1] << 8)
                 | ((uint)data[b + 2] << 16)
                 | ((uint)data[b + 3] << 24);
        }
        md4_compress(&st->h[0], &st->h[1], &st->h[2], &st->h[3], M);
        pos += HASH_BLOCK_BYTES;
    }

    int rem = len - pos;  /* 0..63 */

    /* Zero scratch. */
    for (int j = 0; j < 16; j++) M[j] = 0;

    /* Tail bytes LE. */
    for (int i = 0; i < rem; i++) {
        uint v = (uint)data[pos + i];
        M[i >> 2] |= v << ((i & 3) * 8);
    }
    /* 0x80 padding marker. */
    M[rem >> 2] |= (uint)0x80 << ((rem & 3) * 8);

    if (rem < 56) {
        /* Length fits in this block. MD4 LE: M[14] = low 32 bits of
         * bit count, M[15] = high bits = 0 for our inputs. */
        M[14] = (uint)(len * 8);
        M[15] = 0;
        md4_compress(&st->h[0], &st->h[1], &st->h[2], &st->h[3], M);
    } else {
        /* Need one extra padding-only block to hold the length. */
        md4_compress(&st->h[0], &st->h[1], &st->h[2], &st->h[3], M);
        for (int j = 0; j < 16; j++) M[j] = 0;
        M[14] = (uint)(len * 8);
        M[15] = 0;
        md4_compress(&st->h[0], &st->h[1], &st->h[2], &st->h[3], M);
    }
}

/* template_iterate: -i loop step. MD4 hex output is 32 lowercase hex
 * chars (same as MD5 — both produce 16-byte digests with 4 LE uint32
 * words). md5_to_hex_lc from gpu_common.cl is reusable here because
 * the LE 4-word hex format is algorithm-independent.
 *
 * Mirrors gpu_md4_packed.cl rev 1 lines 96-103 byte-for-byte:
 *   - 4 state words -> 8 LE-encoded hex M[] words (md5_to_hex_lc)
 *   - M[8] = 0x80 (LE pad marker at byte position 32)
 *   - M[9..13] = 0
 *   - M[14] = 32 * 8 = 256, M[15] = 0
 *   - state reset to MD4 IV (same as MD5 IV); md4_compress. */
static inline void template_iterate(template_state *st)
{
    uint M[16];
    md5_to_hex_lc(st->h[0], st->h[1], st->h[2], st->h[3], M);
    M[8] = 0x80u;
    for (int j = 9; j < 14; j++) M[j] = 0u;
    M[14] = 32u * 8u;     /* 32 hex chars = 256 bits */
    M[15] = 0u;
    /* Reinitialize state to IV, then absorb the prepared M[]. */
    st->h[0] = 0x67452301u;
    st->h[1] = 0xEFCDAB89u;
    st->h[2] = 0x98BADCFEu;
    st->h[3] = 0x10325476u;
    md4_compress(&st->h[0], &st->h[1], &st->h[2], &st->h[3], M);
}

/* template_digest_compare: probe the compact table with the final
 * digest. MD4 state words are stored LITTLE-ENDIAN (md4_compress's
 * natural form, same as MD5) so NO byte-swap is needed before probing —
 * the compact table is keyed on LE uint32 words. Mirror's MD5's
 * template_digest_compare exactly. */
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
 * EMIT_HIT_4 (MD4 = 4 uint32 digest words). MD4 state is already LE so
 * no bswap; mirror MD5's pattern exactly. */
#define template_emit_hit(hits, hit_count, max_hits, st, widx, sidx, iter) \
    EMIT_HIT_4((hits), (hit_count), (max_hits), \
               (widx), (sidx), (iter), \
               (st)->h[0], (st)->h[1], (st)->h[2], (st)->h[3])

/* B3 dedup+overflow-aware variant. See gpu_md5_core.cl for protocol
 * notes. */
#define template_emit_hit_or_overflow(hits, hit_count, max_hits, st, widx, sidx, iter, \
                                      hashes_shown, matched_idx, dedup_mask, \
                                      ovr_set, ovr_gid, lane_gid) \
    EMIT_HIT_4_DEDUP_OR_OVERFLOW((hits), (hit_count), (max_hits), \
               (widx), (sidx), (iter), \
               (st)->h[0], (st)->h[1], (st)->h[2], (st)->h[3], \
               (hashes_shown), (matched_idx), (dedup_mask), \
               (ovr_set), (ovr_gid), (lane_gid))
