/*
 * $Revision: 1.2 $
 * $Log: gpu_sha512raw_core.cl,v $
 * Revision 1.2  2026/05/11 05:22:02  dlr
 * Backfill $Revision/$Log RCS keyword stanzas per feedback_rcs_keyword_stanzas.md. Passive 4-line comment block at top of file; no behavioral change. Hand-authored .cl file was missing required stanzas (per memory: all hand-authored .c/.h/.cl/.frag/.tmpl/.py/.sh files MUST contain $Revision/$Log keyword stanzas). Build green on .205 against the post-add files; OpenCL compile strips comments so no kernel behavior change.
 *
 */
/* gpu_sha512raw_core.cl — SHA512RAW algorithm extension functions for the
 * generic dispatch template (Memo B Phase B5 sub-batch 5a, Tier 1).
 *
 * SHA512RAW differs from SHA512 ONLY in template_iterate:
 *   - SHA512  iter (gpu_sha512_core.cl): re-hash 128-byte hex_lc encoding
 *     of the digest. Bytes are ASCII '0'..'9','a'..'f'. Two blocks
 *     required (128 hex + 0x80 + 16 length = 145 > 128, so two-block).
 *   - SHA512RAW iter (this file):     re-hash the 64-byte BINARY digest
 *     directly. Single block (64 + 1 + 16 = 81 ≤ 128).
 *
 * CPU reference (mdxfind.c rev 1.394+ JOB_SHA512RAW at line 27579):
 *
 *   for (x = 1; x <= Maxiter; x++) {
 *     mysha512(...);
 *     cur = (char *)curin.h;
 *     len = 64;                         // BINARY 64 bytes
 *     memcpy(cur, md5buf.h, len);
 *     checkhash(&curin, 128, x, job);
 *   }
 *
 * Block layout for the iter step:
 *   M[0..7]  = 64 BE digest bytes from state[0..7] (each ulong holds 8
 *              bytes BE; matches the BE state ordering)
 *   M[8]     = 0x8000000000000000UL  (0x80 at byte 64 in BE position)
 *   M[9..13] = 0
 *   M[14]    = 0                 (high 64 bits of bit count)
 *   M[15]    = 64 * 8 = 512      (low 64 bits)
 *
 * All other extension functions (template_state, template_init,
 * template_transform, template_finalize, template_digest_compare,
 * template_emit_hit) are byte-identical to gpu_sha512_core.cl.
 *
 * Cache key (R3): defines_str = "HASH_WORDS=16,HASH_BLOCK_BYTES=128" —
 * same as SHA512. Distinct cache entry guaranteed by source-text hash
 * difference (only template_iterate differs).
 *
 * R1 mitigation: single private buffer pattern; no addrspace-cast helpers.
 *
 * R2 (register pressure): same compression footprint as SHA512.
 * priv_mem_size on gfx1201 expected ~ 42.5 KB, comparable to SHA512.
 *
 * Source order at compile time:
 *   [ gpu_common_str, gpu_md5_rules_str, gpu_sha512raw_core_str, gpu_template_str ]
 */

#ifndef HASH_WORDS
#define HASH_WORDS 16
#endif
#ifndef HASH_BLOCK_BYTES
#define HASH_BLOCK_BYTES 128
#endif

typedef struct {
    ulong state[8];   /* internal compression state, BIG-ENDIAN */
    uint  h[HASH_WORDS]; /* exposed digest words, LE-byteswapped uint32 (16 = full 64-byte digest) */
} template_state;

static inline void template_init(template_state *st) {
    st->state[0] = 0x6a09e667f3bcc908UL;
    st->state[1] = 0xbb67ae8584caa73bUL;
    st->state[2] = 0x3c6ef372fe94f82bUL;
    st->state[3] = 0xa54ff53a5f1d36f1UL;
    st->state[4] = 0x510e527fade682d1UL;
    st->state[5] = 0x9b05688c2b3e6c1fUL;
    st->state[6] = 0x1f83d9abfb41bd6bUL;
    st->state[7] = 0x5be0cd19137e2179UL;
}

static inline void template_state_to_h(template_state *st) {
    for (int i = 0; i < 8; i++) {
        ulong s = bswap64(st->state[i]);
        st->h[i*2]   = (uint)s;
        st->h[i*2+1] = (uint)(s >> 32);
    }
}

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

/* template_finalize: byte-identical to gpu_sha512_core.cl. SHA512
 * compression + 128-bit length + full 16-uint32 digest. */
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

/* template_iterate: SHA512RAW iter — re-feed the 64-byte BINARY digest
 * (not hex-encoded) into the compression. CPU reference at mdxfind.c
 * line 27579:
 *
 *   memcpy(cur, md5buf.h, 64);
 *   mysha512(cur, 64);
 *
 * Layout: 64 input bytes + 0x80 + zeros + 16 byte length = 81 bytes ≤ 128
 * (single block). state[0..7] hold the BE digest, mapping 1:1 onto
 * M[0..7] (sha512_block's M-load convention is BE). M[8] holds the 0x80
 * marker; M[14..15] holds the bit-count.
 */
static inline void template_iterate(template_state *st)
{
    ulong M[16];
    /* Copy the BE-packed digest directly into M[0..7]. state[i] is
     * already the BE uint64 of digest bytes [i*8..i*8+7]. */
    M[0] = st->state[0];
    M[1] = st->state[1];
    M[2] = st->state[2];
    M[3] = st->state[3];
    M[4] = st->state[4];
    M[5] = st->state[5];
    M[6] = st->state[6];
    M[7] = st->state[7];
    M[8] = 0x8000000000000000UL;   /* 0x80 BE at byte 64 */
    for (int j = 9; j < 14; j++) M[j] = 0UL;
    M[14] = 0UL;
    M[15] = 64UL * 8UL;            /* 64 binary bytes = 512 bits */

    /* Reset state to SHA512 IV; absorb the prepared block. */
    st->state[0] = 0x6a09e667f3bcc908UL;
    st->state[1] = 0xbb67ae8584caa73bUL;
    st->state[2] = 0x3c6ef372fe94f82bUL;
    st->state[3] = 0xa54ff53a5f1d36f1UL;
    st->state[4] = 0x510e527fade682d1UL;
    st->state[5] = 0x9b05688c2b3e6c1fUL;
    st->state[6] = 0x1f83d9abfb41bd6bUL;
    st->state[7] = 0x5be0cd19137e2179UL;
    sha512_block(&st->state[0], M);

    template_state_to_h(st);
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
    EMIT_HIT_16((hits), (hit_count), (max_hits), \
                (widx), (sidx), (iter), ((st)->h))

#define template_emit_hit_or_overflow(hits, hit_count, max_hits, st, widx, sidx, iter, \
                                      hashes_shown, matched_idx, dedup_mask, \
                                      ovr_set, ovr_gid, lane_gid) \
    EMIT_HIT_16_DEDUP_OR_OVERFLOW((hits), (hit_count), (max_hits), \
               (widx), (sidx), (iter), ((st)->h), \
               (hashes_shown), (matched_idx), (dedup_mask), \
               (ovr_set), (ovr_gid), (lane_gid))
