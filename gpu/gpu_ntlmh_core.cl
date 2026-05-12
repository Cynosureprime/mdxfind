/*
 * $Revision: 1.2 $
 * $Log: gpu_ntlmh_core.cl,v $
 * Revision 1.2  2026/05/11 05:22:01  dlr
 * Backfill $Revision/$Log RCS keyword stanzas per feedback_rcs_keyword_stanzas.md. Passive 4-line comment block at top of file; no behavioral change. Hand-authored .cl file was missing required stanzas (per memory: all hand-authored .c/.h/.cl/.frag/.tmpl/.py/.sh files MUST contain $Revision/$Log keyword stanzas). Build green on .205 against the post-add files; OpenCL compile strips comments so no kernel behavior change.
 *
 */
/* gpu_ntlmh_core.cl — NTLMH (NT password hash) algorithm extension
 * functions for the generic dispatch template (Memo B Phase B5
 * sub-batch 6, Tier B).
 *
 * NTLMH = MD4(UTF-16LE(password)). The hashcat-compatible variant uses
 * "zero-extend" UTF-16LE encoding: each input byte b -> two bytes b,0x00.
 * For ASCII inputs this is byte-identical to iconv(utf-8 -> UTF-16LE).
 * For non-ASCII (multi-byte UTF-8) inputs the zero-extend path computes
 * a different digest than the iconv path; mdxfind's CPU JOB_NTLMH tests
 * BOTH the iconv variant AND the zero-extend variant — the GPU template
 * path here implements ONLY the zero-extend variant (hashcat-compatible
 * by design), matching the existing slab kernel md4utf16_unsalted_batch
 * (gpu_md4unsalted.cl line 197+) which has the same semantic.
 *
 * The slab path's prior NTLMH validation note (mdxfind.c rev 1.394
 * commit comment): "verified byte-identical GPU vs CPU on dev1 Metal
 * and fpga OpenCL across e369 (152/152), e786 primary (100/100),
 * e786 superset (expected 30-hit iconv gap on non-ASCII, hashcat-compat
 * by design)". Same gap applies here.
 *
 * CPU reference (mdxfind.c JOB_NTLMH at line 15174):
 *
 *   if (Unicode) {
 *     MD4(cur, len, md5buf.h);            // input is already UTF-16
 *     checkhash(&md5buf, 32, 1, job);
 *   } else {
 *     // Variant 1: iconv utf-8 -> UTF-16LE -> MD4
 *     // Variant 2: zero-extend UTF-16LE -> MD4   <-- GPU implements this
 *     to_utf16le(cur, wline, len);
 *     MD4(wline, len*2, md5buf.h);
 *     checkhash(&md5buf, 32, 1, job);
 *   }
 *
 * NTLMH has NO iter loop in the CPU implementation — Maxiter is ignored.
 * For template_iterate (called only when iter < Maxiter), we provide a
 * standard MD4 hex re-feed as defensive fallback; in practice the
 * chokepoint and host pipe disable Maxiter > 1 for this algorithm.
 *
 * Block layout: per-byte LE expansion of input[i] to (input[i], 0x00).
 * For ASCII char c, MD4 word j (LE) covers UTF-16LE bytes
 * [4j..4j+3] = (input[2j], 0, input[2j+1], 0). Equivalent uint:
 *   M[j] = input[2j] | (input[2j+1] << 16)
 *
 * Length: 2 * len bytes UTF-16LE. Single MD4 block if 2*len <= 55 (so
 * len <= 27); two blocks if 2*len <= 119 (len <= 59). The chokepoint
 * limits len to a value within the rule-engine path's input bound.
 *
 * State width / byte order: MD4 carries 4 LE uint32 chaining values
 * (same as MD5; UNLIKE the SHA family).
 *
 * Cache key (R3): defines_str = "HASH_WORDS=4,HASH_BLOCK_BYTES=64" — same
 * as MD4 / MD5 / MD5RAW. Distinct cache entry guaranteed by source-text
 * hash difference.
 *
 * R1 mitigation: single private buffer; no addrspace-cast helpers.
 *
 * Source order at compile time:
 *   [ gpu_common_str, gpu_md5_rules_str, gpu_ntlmh_core_str, gpu_template_str ]
 */

#ifndef HASH_WORDS
#define HASH_WORDS 4
#endif
#ifndef HASH_BLOCK_BYTES
#define HASH_BLOCK_BYTES 64
#endif

typedef struct {
    uint h[HASH_WORDS];
} template_state;

/* MD4 compression — inlined here (renamed to md4_compress_ntlmh to avoid
 * symbol clash if gpu_md4_core_str is ever included in the same compile
 * unit). Mirrors gpu_md4_core.cl's md4_compress byte-for-byte. */
static inline void md4_compress_ntlmh(uint *hx, uint *hy, uint *hz, uint *hw, uint *M) {
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

static inline void template_init(template_state *st) {
    st->h[0] = 0x67452301u;
    st->h[1] = 0xEFCDAB89u;
    st->h[2] = 0x98BADCFEu;
    st->h[3] = 0x10325476u;
}

/* template_transform: stub for interface symmetry. NTLMH finalize handles
 * its own block packing inline (UTF-16LE expansion is interleaved with
 * the M[] build). */
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
    md4_compress_ntlmh(&st->h[0], &st->h[1], &st->h[2], &st->h[3], M);
}

/* template_finalize: NTLMH = MD4(UTF-16LE-zero-extend(input)). Build M[]
 * directly from the input bytes interleaved with zeros (the UTF-16LE
 * high-byte placeholder for ASCII chars).
 *
 * For each MD4 word j of UTF-16LE bytes:
 *   word covers UTF-16LE bytes [4j..4j+3]:
 *     UTF-16LE byte 2k   = input[k]      (low byte of UTF-16 char k)
 *     UTF-16LE byte 2k+1 = 0             (high byte = 0 for zero-extend)
 *   So byte 4j   = input[2j]
 *      byte 4j+1 = 0
 *      byte 4j+2 = input[2j+1]
 *      byte 4j+3 = 0
 * MD4 reads M LE: M[j] = byte[4j] | byte[4j+1]<<8 | byte[4j+2]<<16 | byte[4j+3]<<24
 *               = input[2j] | (input[2j+1] << 16)
 *
 * The 0x80 padding marker lands at UTF-16-byte position 2*len; if 2*len
 * is even (always true since 2*len is always even), the marker is at
 * the LOW byte of M[len/2] (when len is even) or M[(len/2)] high byte (?).
 * Actually 2*len is the count of UTF-16 bytes consumed. The pad goes at
 * byte index 2*len in the UTF-16LE stream. byte_idx = 2*len:
 *   M_word = byte_idx >> 2 = len >> 1
 *   byte_in_word = byte_idx & 3 = (2*len) & 3 = 2 * (len & 1)
 *   ⇒ byte_in_word = 0 if len even, 2 if len odd
 *
 * Bit count in UTF-16LE = 2 * len * 8 = 16 * len.
 */
static inline void template_finalize(template_state *st,
                                     const uchar *data, int len)
{
    /* Process complete UTF-16LE blocks. Each MD4 block (64 bytes) holds
     * 32 UTF-16LE chars = 32 input bytes. */
    uint M[16];
    int input_pos = 0;

    while (len - input_pos >= 32) {
        /* 32 input bytes -> 64 UTF-16LE bytes -> 16 MD4 M[] words. */
        for (int j = 0; j < 16; j++) {
            int k = input_pos + j * 2;
            M[j] = (uint)data[k] | ((uint)data[k + 1] << 16);
        }
        md4_compress_ntlmh(&st->h[0], &st->h[1], &st->h[2], &st->h[3], M);
        input_pos += 32;
    }

    /* Tail: remaining input bytes (rem in [0..31]) -> 2*rem UTF-16LE bytes. */
    int rem = len - input_pos;  /* 0..31 input bytes */
    int rem_utf16 = rem * 2;     /* 0..62 UTF-16LE bytes */

    /* Zero scratch. */
    for (int j = 0; j < 16; j++) M[j] = 0;

    /* Pack input bytes into M[] LE positions, interleaved with zeros. */
    for (int i = 0; i < rem; i++) {
        int byte_idx = i * 2;            /* low byte of UTF-16 char */
        int wi = byte_idx >> 2;
        int bi = byte_idx & 3;
        M[wi] |= ((uint)data[input_pos + i]) << (bi * 8);
        /* High byte (byte_idx + 1) is zero; no write needed. */
    }

    /* 0x80 pad marker at UTF-16 byte position rem_utf16 = 2*rem. */
    {
        int wi = rem_utf16 >> 2;
        int bi = rem_utf16 & 3;
        M[wi] |= ((uint)0x80u) << (bi * 8);
    }

    /* MD4 LE bit-count encoding: M[14] = low 32 bits, M[15] = high 32 bits.
     * Total UTF-16LE bytes = 2*len ⇒ bit count = 16 * len. */
    if (rem_utf16 < 56) {
        M[14] = (uint)((uint)len * 16u);
        M[15] = 0;
        md4_compress_ntlmh(&st->h[0], &st->h[1], &st->h[2], &st->h[3], M);
    } else {
        md4_compress_ntlmh(&st->h[0], &st->h[1], &st->h[2], &st->h[3], M);
        for (int j = 0; j < 16; j++) M[j] = 0;
        M[14] = (uint)((uint)len * 16u);
        M[15] = 0;
        md4_compress_ntlmh(&st->h[0], &st->h[1], &st->h[2], &st->h[3], M);
    }
}

/* template_iterate: defensive fallback. NTLMH has no iter loop in CPU.
 * If the host ever passes Maxiter > 1 for NTLMH (it shouldn't), this
 * computes MD4(hex_lc(prev_hash)) — same shape as MD4 template_iterate.
 * Production path (Maxiter == 1) never calls this. */
static inline void template_iterate(template_state *st)
{
    uint M[16];
    md5_to_hex_lc(st->h[0], st->h[1], st->h[2], st->h[3], M);
    M[8] = 0x80u;
    for (int j = 9; j < 14; j++) M[j] = 0u;
    M[14] = 32u * 8u;
    M[15] = 0u;
    st->h[0] = 0x67452301u;
    st->h[1] = 0xEFCDAB89u;
    st->h[2] = 0x98BADCFEu;
    st->h[3] = 0x10325476u;
    md4_compress_ntlmh(&st->h[0], &st->h[1], &st->h[2], &st->h[3], M);
}

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

#define template_emit_hit(hits, hit_count, max_hits, st, widx, sidx, iter) \
    EMIT_HIT_4((hits), (hit_count), (max_hits), \
               (widx), (sidx), (iter), \
               (st)->h[0], (st)->h[1], (st)->h[2], (st)->h[3])

#define template_emit_hit_or_overflow(hits, hit_count, max_hits, st, widx, sidx, iter, \
                                      hashes_shown, matched_idx, dedup_mask, \
                                      ovr_set, ovr_gid, lane_gid) \
    EMIT_HIT_4_DEDUP_OR_OVERFLOW((hits), (hit_count), (max_hits), \
               (widx), (sidx), (iter), \
               (st)->h[0], (st)->h[1], (st)->h[2], (st)->h[3], \
               (hashes_shown), (matched_idx), (dedup_mask), \
               (ovr_set), (ovr_gid), (lane_gid))
