/*
 * $Revision: 1.3 $
 * $Log: gpu_sha256raw_core.cl,v $
 * Revision 1.3  2026/08/30 14:35:34  dlr
 * Restore the RAW-family iteration rule in the GPU kernels to match mdxfind.c 1.549. These kernels were written against the post-1.290 CPU regression and validated byte-exact against it, which is why CPU-vs-GPU conformance testing never caught the error: both sides agreed on the wrong answer. The correct rule is ONE binary feed at the base, giving X(X_bin(pass)), followed by STANDARD hex iteration. The former template_iterate already implemented exactly that binary absorb, so it is renamed template_raw_refeed, moved ahead of template_finalize and called once at the end of it; template_iterate is replaced by the plain siblings hex step. For the Metal MD5 variant the algo_mode parameter is dropped since RAW has no uppercase form and the call site uses the legacy single-argument shape. Validated on fpga.local GTX 1080 via OpenCL: 15 of 15 lines identical between CPU and GPU across five types and three depths, with a bogus-hash negative control returning zero. On dev1.local Apple M1 via Metal the base is correct and matches CPU at x01 for all five types, but iterations beyond x01 are not returned; that is a coverage gap and not a wrong answer, confirmed by comm showing zero Metal lines absent from the CPU set.
 *
 * Revision 1.2  2026/05/11 05:22:02  dlr
 * Backfill $Revision/$Log RCS keyword stanzas per feedback_rcs_keyword_stanzas.md. Passive 4-line comment block at top of file; no behavioral change. Hand-authored .cl file was missing required stanzas (per memory: all hand-authored .c/.h/.cl/.frag/.tmpl/.py/.sh files MUST contain $Revision/$Log keyword stanzas). Build green on .205 against the post-add files; OpenCL compile strips comments so no kernel behavior change.
 *
 */
/* gpu_sha256raw_core.cl — SHA256RAW algorithm extension functions for
 * the generic dispatch template (Memo B Phase B5 sub-batch 6, Tier A).
 *
 * SHA256RAW differs from SHA256 ONLY in template_iterate:
 *   - SHA256    iter (gpu_sha256_core.cl): re-hash 64-byte hex_lc
 *     encoding of the digest (spans 2 blocks).
 *   - SHA256RAW iter (this file):     re-hash the 32-byte BINARY digest
 *     directly. No hex re-encoding. Single block.
 *
 * CPU reference (mdxfind.c JOB_SHA256RAW at line 27405):
 *
 *   for (x = 1; x <= Maxiter; x++) {
 *     mysha256(cur, len, md5buf.h);
 *     cur = (char *)curin.h;
 *     len = 32;                     // BINARY 32 bytes
 *     memcpy(cur, md5buf.h, len);
 *     checkhash(curin, 64, x, job);
 *   }
 *
 * Block layout for the iter step (32-byte input, SHA256 BE):
 *   M[0..7] = 32 BE digest bytes from state[0..7]
 *             (st->h[i] is the BE uint32 from sha256_block; M-load reads
 *             bytes BE — these match identically.)
 *   M[8]    = 0x80000000u  (0x80 padding marker at byte 32, BE)
 *   M[9..13]= 0
 *   M[14]   = 0  (high 32 bits of bit count)
 *   M[15]   = 32 * 8 = 256
 *
 * 32 + 1 + 8 = 41 < 56 ⇒ single block (cleaner than SHA256's hex iter
 * which spans 2 blocks at 64+1+8=73 > 56).
 *
 * State width / byte order: SHA256 carries 8 BE uint32 chaining values.
 * Final digest h[0..7] (after template_finalize) is BIG-ENDIAN — matches
 * gpu_sha256_core.cl convention. template_digest_compare bswap32's the
 * leading 4 words to LE before probing.
 *
 * All other extension functions are byte-identical to gpu_sha256_core.cl.
 *
 * Cache key (R3): defines_str = "HASH_WORDS=8,HASH_BLOCK_BYTES=64" —
 * same as SHA256. Distinct cache entry guaranteed by source-text hash
 * difference (only template_iterate differs).
 *
 * Source order at compile time:
 *   [ gpu_common_str, gpu_md5_rules_str, gpu_sha256raw_core_str, gpu_template_str ]
 */

#ifndef HASH_WORDS
#define HASH_WORDS 8
#endif
#ifndef HASH_BLOCK_BYTES
#define HASH_BLOCK_BYTES 64
#endif

typedef struct {
    uint h[HASH_WORDS];
} template_state;

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

/* template_finalize: byte-identical to gpu_sha256_core.cl. */
/* template_raw_refeed: absorb the raw digest as a fresh message. This is the
 * ONE binary feed that defines the RAW family -- the base is X(X_bin(pass)).
 * It is NOT the -i step; iteration is hex (see template_iterate below).
 * Body is verbatim the pre-2026-08-30 template_iterate, which implemented
 * exactly this absorb; only its ROLE was wrong. */
static inline void template_raw_refeed(template_state *st)
{
    uint M[16];
    M[0] = st->h[0];
    M[1] = st->h[1];
    M[2] = st->h[2];
    M[3] = st->h[3];
    M[4] = st->h[4];
    M[5] = st->h[5];
    M[6] = st->h[6];
    M[7] = st->h[7];
    M[8] = 0x80000000u;
    for (int j = 9; j < 14; j++) M[j] = 0u;
    M[14] = 0u;
    M[15] = 32u * 8u;
    /* Reset state to SHA256 IV; absorb the prepared block. */
    st->h[0] = 0x6a09e667u;
    st->h[1] = 0xbb67ae85u;
    st->h[2] = 0x3c6ef372u;
    st->h[3] = 0xa54ff53au;
    st->h[4] = 0x510e527fu;
    st->h[5] = 0x9b05688cu;
    st->h[6] = 0x1f83d9abu;
    st->h[7] = 0x5be0cd19u;
    sha256_block(&st->h[0], M);
}

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

    for (int i = 0; i < rem; i++) {
        int wi = i >> 2;
        int bi = 3 - (i & 3);
        M[wi] |= ((uint)data[pos + i]) << (bi * 8);
    }
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

    /* the one binary feed that makes this a RAW type */
    template_raw_refeed(st);
}

/* template_iterate: SHA256RAW iter — re-feed the 32-byte BINARY digest
 * (not hex-encoded) into the compression. State words are BE uint32
 * (sha256_block's natural form); M-load reads bytes BE — st->h[i]
 * directly maps to M[i] for i in 0..7.
 *
 * Layout:
 *   M[0..7]  = st->h[0..7] (32 BE digest bytes; native sha256 word order)
 *   M[8]     = 0x80000000u (0x80 BE at byte 32)
 *   M[9..13] = 0
 *   M[14]    = 0
 *   M[15]    = 32 * 8 = 256 (low 32 of bit count, BE)
 *
 * Single block (32 + 1 + 8 = 41 < 56). After absorption, h[] holds the
 * fresh digest in native SHA256 BE form.
 */
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
