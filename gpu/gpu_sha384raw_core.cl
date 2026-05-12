/*
 * $Revision: 1.2 $
 * $Log: gpu_sha384raw_core.cl,v $
 * Revision 1.2  2026/05/11 05:22:02  dlr
 * Backfill $Revision/$Log RCS keyword stanzas per feedback_rcs_keyword_stanzas.md. Passive 4-line comment block at top of file; no behavioral change. Hand-authored .cl file was missing required stanzas (per memory: all hand-authored .c/.h/.cl/.frag/.tmpl/.py/.sh files MUST contain $Revision/$Log keyword stanzas). Build green on .205 against the post-add files; OpenCL compile strips comments so no kernel behavior change.
 *
 */
/* gpu_sha384raw_core.cl — SHA384RAW algorithm extension functions for the
 * generic dispatch template (Memo B Phase B5 sub-batch 5a, Tier 1).
 *
 * SHA384RAW differs from SHA384 ONLY in template_iterate:
 *   - SHA384  iter (gpu_sha384_core.cl): re-hash 96-byte hex_lc encoding
 *     of the digest. Bytes are ASCII '0'..'9','a'..'f'.
 *   - SHA384RAW iter (this file):     re-hash the 48-byte BINARY digest
 *     directly. No hex re-encoding.
 *
 * CPU reference (mdxfind.c rev 1.394+ JOB_SHA384RAW at line 27744):
 *
 *   for (x = 1; x <= Maxiter; x++) {
 *     sph_sha384(...);                  // hash cur[0..len)
 *     sph_sha384_close(..., md5buf.h);
 *     cur = (char *)curin.h;
 *     len = 48;                         // BINARY 48 bytes
 *     memcpy(cur, md5buf.h, len);
 *     checkhash(&curin, 96, x, job);    // hexlen=96 is for compact-table probe
 *   }
 *
 * Key invariant: at iter N, the input is the 48-byte BIG-ENDIAN binary
 * digest from iter N-1 (the standard SHA-384 output byte order). Each
 * compression block consumes 128 bytes; 48 + 1 (0x80 marker) + 16
 * (length) = 65 bytes ≤ 128. So a single block suffices for the iter
 * step.
 *
 * Block layout for the iter step:
 *   M[0..5] = 48 BE digest bytes from state[0..5] (each ulong holds 8
 *             bytes BE — same byte order as the digest output from
 *             template_finalize).
 *   M[6] = 0x8000000000000000UL  (0x80 at byte 48 in BE position)
 *   M[7..13] = 0
 *   M[14] = 0          (high 64 bits of bit count)
 *   M[15] = 48 * 8 = 384 (low 64 bits)
 *
 * All other extension functions (template_state, template_init,
 * template_transform, template_finalize, template_digest_compare,
 * template_emit_hit) are byte-identical to gpu_sha384_core.cl. We
 * duplicate the body inline rather than #include to keep the per-algo
 * source unit self-contained and the cache key clean (Memo B R3:
 * defines_str + source-text hash).
 *
 * Cache key (R3): defines_str = "HASH_WORDS=12,HASH_BLOCK_BYTES=128" —
 * same as SHA384. Distinct cache entry guaranteed by source-text hash
 * difference (only template_iterate differs; the rest is byte-identical).
 *
 * R1 mitigation: single private buffer pattern; no addrspace-cast
 * helpers. Same shape as gpu_sha384_core.cl.
 *
 * R2 (register pressure): same compression footprint as SHA512 / SHA384
 * (W[80] schedule). priv_mem_size on gfx1201 expected ~ 42.5 KB (within
 * the 64-VGPR / 256-byte hard ceiling, comparable to SHA512).
 *
 * Source order at compile time:
 *   [ gpu_common_str, gpu_md5_rules_str, gpu_sha384raw_core_str, gpu_template_str ]
 */

#ifndef HASH_WORDS
#define HASH_WORDS 12
#endif
#ifndef HASH_BLOCK_BYTES
#define HASH_BLOCK_BYTES 128
#endif

typedef struct {
    ulong state[8];   /* internal compression state, BIG-ENDIAN */
    uint  h[HASH_WORDS]; /* exposed digest words, LE-byteswapped uint32 (12 only) */
} template_state;

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

static inline void template_state_to_h(template_state *st) {
    for (int i = 0; i < 6; i++) {
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

/* template_finalize: byte-identical to gpu_sha384_core.cl. SHA384
 * compression + 128-bit length + 12-uint32 truncation. */
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

/* template_iterate: SHA384RAW iter — re-feed the 48-byte BINARY digest
 * (not hex-encoded) into the compression. CPU reference at mdxfind.c
 * line 27744:
 *
 *   memcpy(cur, md5buf.h, 48);
 *   sph_sha384(cur, 48);
 *
 * State after template_finalize: state[0..5] hold the BE-packed digest
 * (state[i] = digest bytes [i*8..i*8+7] in BE = state[i] when interpreted
 * as a BE uint64). state[6..7] have the trailing internal chain values
 * but are NOT part of the digest output.
 *
 * For the iter step, we treat the first 48 bytes of the BIG-ENDIAN
 * digest as the new input. M[0..5] = state[0..5] directly (since BE byte
 * order matches sha512's M-load convention). M[6] is the 0x80 marker;
 * M[14..15] is the bit count.
 *
 * Layout:
 *   M[0..5]  = state[0..5] (48 BE digest bytes; state ulongs are already
 *              in the BE byte order that sha512_block expects)
 *   M[6]     = 0x8000000000000000UL  (0x80 padding marker at byte 48)
 *   M[7..13] = 0
 *   M[14]    = 0          (high 64 bits of bit count)
 *   M[15]    = 48 * 8 = 384 (low 64 bits)
 *
 * Single block (48 + 1 + 16 = 65 ≤ 128). After absorption, decompose
 * state[0..5] into h[0..11].
 */
static inline void template_iterate(template_state *st)
{
    ulong M[16];
    /* Copy the BE-packed digest directly into M[0..5]. state[i] is
     * already the BE uint64 of digest bytes [i*8..i*8+7] — matches the
     * sha512_block M-load convention exactly. */
    M[0] = st->state[0];
    M[1] = st->state[1];
    M[2] = st->state[2];
    M[3] = st->state[3];
    M[4] = st->state[4];
    M[5] = st->state[5];
    M[6] = 0x8000000000000000UL;   /* 0x80 BE at byte 48 */
    M[7] = 0UL;
    for (int j = 8; j < 14; j++) M[j] = 0UL;
    M[14] = 0UL;
    M[15] = 48UL * 8UL;            /* 48 binary bytes = 384 bits */

    /* Reset state to SHA384 IV; absorb the prepared block. */
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
    EMIT_HIT_12((hits), (hit_count), (max_hits), \
                (widx), (sidx), (iter), ((st)->h))

#define template_emit_hit_or_overflow(hits, hit_count, max_hits, st, widx, sidx, iter, \
                                      hashes_shown, matched_idx, dedup_mask, \
                                      ovr_set, ovr_gid, lane_gid) \
    EMIT_HIT_12_DEDUP_OR_OVERFLOW((hits), (hit_count), (max_hits), \
               (widx), (sidx), (iter), ((st)->h), \
               (hashes_shown), (matched_idx), (dedup_mask), \
               (ovr_set), (ovr_gid), (lane_gid))
