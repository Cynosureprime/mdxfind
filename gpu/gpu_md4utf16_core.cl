/*
 * $Revision: 1.2 $
 * $Log: gpu_md4utf16_core.cl,v $
 * Revision 1.2  2026/05/11 05:22:01  dlr
 * Backfill $Revision/$Log RCS keyword stanzas per feedback_rcs_keyword_stanzas.md. Passive 4-line comment block at top of file; no behavioral change. Hand-authored .cl file was missing required stanzas (per memory: all hand-authored .c/.h/.cl/.frag/.tmpl/.py/.sh files MUST contain $Revision/$Log keyword stanzas). Build green on .205 against the post-add files; OpenCL compile strips comments so no kernel behavior change.
 *
 */
/* gpu_md4utf16_core.cl — MD4UTF16 (-m e496) algorithm extension functions
 * for the generic dispatch template (Memo B Phase B5 sub-batch 8).
 *
 * MD4UTF16 = MD4(UTF-16LE-zero-extend(input))  on iter == 1
 *          = MD4(UTF-16LE-zero-extend(lowercase_hex(prev_digest)))  on iter > 1
 *
 * This is a CLONE of gpu_ntlmh_core.cl's algorithm (zero-extend UTF-16LE
 * variant of MD4) with one structural difference: MD4UTF16 supports the
 * mdxfind -i / Maxiter loop, NTLMH does not. The CPU reference at
 * mdxfind.c:15040-15068 (JOB_MD4UTF16) emits one digest per iter, with
 * iter 2..Maxiter feeding back the lowercase-hex of the prior digest
 * (32 ASCII hex chars) zero-extended to UTF-16LE (64 bytes), then MD4'd.
 *
 * The hashcat-compatible "zero-extend UTF-16LE" semantic is used (each
 * input byte b -> two bytes (b, 0x00)). For ASCII inputs this is byte-
 * identical to iconv(utf-8 -> UTF-16LE) -- which is the only iconv
 * variant CPU JOB_MD4UTF16 tests (see Memo B multi-variant hook memo,
 * the SUPERSEDED notice corrects the earlier 'multi-variant' framing:
 * MD4UTF16 is in fact single-variant). For non-ASCII inputs there is a
 * documented gap with CPU iconv -- same gap as the existing slab kernel
 * md4utf16_unsalted_batch (hashcat-compatible by design).
 *
 * On the iter > 1 feedback path the input is always lowercase hex chars
 * [0-9a-f] (pure ASCII) so zero-extend == iconv and there is no gap;
 * Maxiter > 1 is byte-exact vs CPU.
 *
 * Block layout (template_finalize, iter == 1):
 *   Per-byte LE expansion of input[i] to (input[i], 0x00).
 *   For ASCII char c, MD4 word j (LE) covers UTF-16LE bytes
 *   [4j..4j+3] = (input[2j], 0, input[2j+1], 0).
 *   M[j] = input[2j] | (input[2j+1] << 16)
 *   Length: 2*len bytes UTF-16LE. Bit count = 16*len.
 *
 * Block layout (template_iterate, iter > 1):
 *   Input is fixed 32 lowercase hex chars (digest of prior iter).
 *   UTF-16LE-zero-extend yields 64 bytes -> exactly one MD4 input block.
 *   The 0x80 pad marker + length field need a SECOND MD4 block.
 *   M[j] for first block = hex[2j] | (hex[2j+1] << 16) for j in 0..15.
 *   Second block: M[0] = 0x80, M[1..13] = 0, M[14] = 64*8 = 512, M[15] = 0.
 *
 * State width / byte order: MD4 carries 4 LE uint32 chaining values
 * (same as MD5; UNLIKE the SHA family).
 *
 * Cache key (R3): defines_str = "HASH_WORDS=4,HASH_BLOCK_BYTES=64" -- same
 * as MD4 / MD5 / MD5RAW / NTLMH. Distinct cache entry guaranteed by
 * source-text hash difference (the iter step diverges from NTLMH).
 *
 * R1 mitigation: single private buffer; no addrspace-cast helpers.
 *
 * Source order at compile time:
 *   [ gpu_common_str, gpu_md5_rules_str, gpu_md4utf16_core_str, gpu_template_str ]
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

/* MD4 compression -- inlined here (renamed to md4_compress_md4utf16 to avoid
 * symbol clash if gpu_md4_core_str / gpu_ntlmh_core_str is ever included in
 * the same compile unit). Mirrors gpu_md4_core.cl's md4_compress byte-for-
 * byte; same as md4_compress_ntlmh. */
static inline void md4_compress_md4utf16(uint *hx, uint *hy, uint *hz, uint *hw, uint *M) {
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

/* template_transform: stub for interface symmetry (cloned from NTLMH).
 * The shared template body in gpu_template.cl never invokes this for
 * MD4UTF16; template_finalize handles its own block packing inline
 * (UTF-16LE expansion is interleaved with the M[] build). */
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
    md4_compress_md4utf16(&st->h[0], &st->h[1], &st->h[2], &st->h[3], M);
}

/* template_finalize: MD4UTF16 = MD4(UTF-16LE-zero-extend(input)). Build
 * M[] directly from the input bytes interleaved with zeros (the UTF-16LE
 * high-byte placeholder for ASCII chars). Cloned byte-for-byte from
 * gpu_ntlmh_core.cl template_finalize. */
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
        md4_compress_md4utf16(&st->h[0], &st->h[1], &st->h[2], &st->h[3], M);
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
     * Total UTF-16LE bytes = 2*len => bit count = 16 * len. */
    if (rem_utf16 < 56) {
        M[14] = (uint)((uint)len * 16u);
        M[15] = 0;
        md4_compress_md4utf16(&st->h[0], &st->h[1], &st->h[2], &st->h[3], M);
    } else {
        md4_compress_md4utf16(&st->h[0], &st->h[1], &st->h[2], &st->h[3], M);
        for (int j = 0; j < 16; j++) M[j] = 0;
        M[14] = (uint)((uint)len * 16u);
        M[15] = 0;
        md4_compress_md4utf16(&st->h[0], &st->h[1], &st->h[2], &st->h[3], M);
    }
}

/* template_iterate: -i loop step. CPU reference at mdxfind.c:15059-15066:
 *
 *   prmd5(md5buf.h, mdbuf, 32);     // 32 lowercase hex chars in mdbuf
 *   to_utf16le(mdbuf, linebuf, 32); // 64 UTF-16LE bytes (each ASCII byte
 *                                   //   followed by 0x00)
 *   MD4(linebuf, 64, md5buf.h);     // MD4 over the 64-byte buffer
 *
 * Implementation:
 *   1. Pack lowercase hex of state into 8 LE uint32 words via md5_to_hex_lc
 *      (defined in gpu_common.cl). Result: M_hex[0..7] each holds 4 hex
 *      bytes in LE order.
 *   2. Build the FIRST MD4 block from those 32 hex bytes via UTF-16LE
 *      zero-extend pack: M_blk[j] = hex[2j] | (hex[2j+1] << 16) for j in
 *      0..15. Read hex bytes from M_hex via byte extraction.
 *      Reset state IV, MD4-compress this first block.
 *   3. Build the SECOND MD4 block: M[0] = 0x80 pad marker (UTF-16 byte
 *      position 64 = wi=0 of block 2), M[1..13] = 0, M[14] = 64*8 = 512
 *      (bit count for 64 UTF-16LE bytes), M[15] = 0. MD4-compress. */
static inline void template_iterate(template_state *st)
{
    uint M_hex[16];
    /* Step 1: lowercase-hex of current digest -> M_hex[0..7] (32 bytes).
     * md5_to_hex_lc writes M[0..7] LE-packed; M[8..15] are not touched
     * by md5_to_hex_lc but our local M_hex was uninitialized -- ok,
     * we only read indices [0..7] below. */
    md5_to_hex_lc(st->h[0], st->h[1], st->h[2], st->h[3], M_hex);

    /* Step 2: build first MD4 block over UTF-16LE-zero-extend of the 32
     * hex chars. Hex char k is at byte position k of the linear hex
     * stream, which is at M_hex[k>>2], byte (k&3) (LE order). */
    uint M_blk[16];
    for (int j = 0; j < 16; j++) {
        int k0 = j * 2;          /* hex char index for low half of M_blk[j] */
        int k1 = j * 2 + 1;      /* hex char index for high half of M_blk[j] */
        uint c0 = (M_hex[k0 >> 2] >> ((k0 & 3) * 8)) & 0xffu;
        uint c1 = (M_hex[k1 >> 2] >> ((k1 & 3) * 8)) & 0xffu;
        M_blk[j] = c0 | (c1 << 16);
    }
    /* Reset state to MD4 IV before consuming the new input. */
    st->h[0] = 0x67452301u;
    st->h[1] = 0xEFCDAB89u;
    st->h[2] = 0x98BADCFEu;
    st->h[3] = 0x10325476u;
    md4_compress_md4utf16(&st->h[0], &st->h[1], &st->h[2], &st->h[3], M_blk);

    /* Step 3: build second MD4 block (pad + length). UTF-16 byte 64 lands
     * at wi=0, bi=0 of block 2 -- so M[0] = 0x80 (low byte of word 0). */
    M_blk[0] = 0x80u;
    for (int j = 1; j < 14; j++) M_blk[j] = 0u;
    M_blk[14] = 64u * 8u;        /* 64 UTF-16LE bytes = 512 bits */
    M_blk[15] = 0u;
    md4_compress_md4utf16(&st->h[0], &st->h[1], &st->h[2], &st->h[3], M_blk);
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
