/*
 * $Revision: 1.2 $
 * $Log: gpu_sha1raw_core.cl,v $
 * Revision 1.2  2026/05/11 05:22:02  dlr
 * Backfill $Revision/$Log RCS keyword stanzas per feedback_rcs_keyword_stanzas.md. Passive 4-line comment block at top of file; no behavioral change. Hand-authored .cl file was missing required stanzas (per memory: all hand-authored .c/.h/.cl/.frag/.tmpl/.py/.sh files MUST contain $Revision/$Log keyword stanzas). Build green on .205 against the post-add files; OpenCL compile strips comments so no kernel behavior change.
 *
 */
/* gpu_sha1raw_core.cl — SHA1RAW algorithm extension functions for the
 * generic dispatch template (Memo B Phase B5 sub-batch 6, Tier A).
 *
 * SHA1RAW differs from SHA1 ONLY in template_iterate:
 *   - SHA1    iter (gpu_sha1_core.cl): re-hash 40-byte hex_lc encoding.
 *   - SHA1RAW iter (this file):     re-hash the 20-byte BINARY digest
 *     directly. No hex re-encoding.
 *
 * CPU reference (mdxfind.c JOB_SHA1RAW at line 26321):
 *
 *   for (x = 1; x <= Maxiter; x++) {
 *     mysha1(cur, len, md5buf.h);
 *     cur = (char *)curin.h;
 *     len = 20;                     // BINARY 20 bytes
 *     memcpy(cur, md5buf.h, len);
 *     checkhash(md5buf, 40, x, job);
 *   }
 *
 * Block layout for the iter step (20-byte input, SHA1 BE):
 *   M[0..4] = 20 BE digest bytes from state[0..4]
 *             (st->h[i] is the BE uint32 from sha1_block; M-load reads
 *             bytes BE — these match identically.)
 *   M[5]    = 0x80000000u  (0x80 padding marker at byte 20, BE)
 *   M[6..13]= 0
 *   M[14]   = 0  (high 32 bits of bit count)
 *   M[15]   = 20 * 8 = 160
 *
 * 20 + 1 + 8 = 29 < 56 ⇒ single block.
 *
 * State width / byte order: SHA1 carries 5 BE uint32 chaining values.
 * Final digest h[0..4] (after template_finalize) is BIG-ENDIAN — matches
 * gpu_sha1_core.cl convention. template_digest_compare bswap32's the
 * leading 4 words to LE before probing (mirrors gpu_sha1_core.cl's
 * compare).
 *
 * All other extension functions are byte-identical to gpu_sha1_core.cl.
 *
 * Cache key (R3): defines_str = "HASH_WORDS=5,HASH_BLOCK_BYTES=64" —
 * same as SHA1. Distinct cache entry guaranteed by source-text hash
 * difference (only template_iterate differs).
 *
 * Source order at compile time:
 *   [ gpu_common_str, gpu_md5_rules_str, gpu_sha1raw_core_str, gpu_template_str ]
 */

#ifndef HASH_WORDS
#define HASH_WORDS 5
#endif
#ifndef HASH_BLOCK_BYTES
#define HASH_BLOCK_BYTES 64
#endif

typedef struct {
    uint h[HASH_WORDS];
} template_state;

static inline void template_init(template_state *st) {
    st->h[0] = 0x67452301u;
    st->h[1] = 0xEFCDAB89u;
    st->h[2] = 0x98BADCFEu;
    st->h[3] = 0x10325476u;
    st->h[4] = 0xC3D2E1F0u;
}

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

/* template_finalize: byte-identical to gpu_sha1_core.cl. */
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
        sha1_block(&st->h[0], M);
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
        sha1_block(&st->h[0], M);
    } else {
        sha1_block(&st->h[0], M);
        for (int j = 0; j < 16; j++) M[j] = 0;
        M[14] = 0;
        M[15] = (uint)((uint)len * 8u);
        sha1_block(&st->h[0], M);
    }
}

/* template_iterate: SHA1RAW iter — re-feed the 20-byte BINARY digest
 * (not hex-encoded) into the compression. State words are BE uint32
 * (sha1_block's natural form); M-load reads bytes BE — st->h[i]
 * directly maps to M[i] for i in 0..4.
 *
 * Layout:
 *   M[0..4]  = st->h[0..4] (20 BE digest bytes; native sha1 word order)
 *   M[5]     = 0x80000000u (0x80 BE at byte 20)
 *   M[6..13] = 0
 *   M[14]    = 0
 *   M[15]    = 20 * 8 = 160 (low 32 of bit count, BE)
 *
 * Single block (20 + 1 + 8 = 29 < 56). After absorption, h[] holds the
 * fresh digest in native SHA1 BE form.
 */
static inline void template_iterate(template_state *st)
{
    uint M[16];
    M[0] = st->h[0];
    M[1] = st->h[1];
    M[2] = st->h[2];
    M[3] = st->h[3];
    M[4] = st->h[4];
    M[5] = 0x80000000u;
    for (int j = 6; j < 14; j++) M[j] = 0u;
    M[14] = 0u;
    M[15] = 20u * 8u;
    /* Reset state to SHA1 IV; absorb the prepared block. */
    st->h[0] = 0x67452301u;
    st->h[1] = 0xEFCDAB89u;
    st->h[2] = 0x98BADCFEu;
    st->h[3] = 0x10325476u;
    st->h[4] = 0xC3D2E1F0u;
    sha1_block(&st->h[0], M);
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
        uint _h[5]; \
        _h[0] = bswap32((st)->h[0]); \
        _h[1] = bswap32((st)->h[1]); \
        _h[2] = bswap32((st)->h[2]); \
        _h[3] = bswap32((st)->h[3]); \
        _h[4] = bswap32((st)->h[4]); \
        EMIT_HIT_5((hits), (hit_count), (max_hits), \
                   (widx), (sidx), (iter), _h) \
    } while (0)

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
