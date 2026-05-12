/*
 * $Revision: 1.2 $
 * $Log: gpu_md5raw_core.cl,v $
 * Revision 1.2  2026/05/11 05:22:01  dlr
 * Backfill $Revision/$Log RCS keyword stanzas per feedback_rcs_keyword_stanzas.md. Passive 4-line comment block at top of file; no behavioral change. Hand-authored .cl file was missing required stanzas (per memory: all hand-authored .c/.h/.cl/.frag/.tmpl/.py/.sh files MUST contain $Revision/$Log keyword stanzas). Build green on .205 against the post-add files; OpenCL compile strips comments so no kernel behavior change.
 *
 */
/* gpu_md5raw_core.cl — MD5RAW algorithm extension functions for the
 * generic dispatch template (Memo B Phase B5 sub-batch 6, Tier A).
 *
 * MD5RAW differs from MD5 ONLY in template_iterate:
 *   - MD5    iter (gpu_md5_core.cl): re-hash 32-byte hex_lc encoding
 *     of the digest. Bytes are ASCII '0'..'9','a'..'f'.
 *   - MD5RAW iter (this file):     re-hash the 16-byte BINARY digest
 *     directly. No hex re-encoding.
 *
 * CPU reference (mdxfind.c JOB_MD5RAW at line 24237):
 *
 *   for (x = 1; x <= Maxiter; x++) {
 *     mymd5(cur, len, md5buf.h);
 *     cur = (char *)curin.h;
 *     len = 16;                     // BINARY 16 bytes
 *     memcpy(cur, md5buf.h, len);
 *     checkhash(md5buf, 32, x, job);
 *   }
 *
 * Block layout for the iter step (16-byte input, MD5 LE):
 *   M[0..3] = 16 LE digest bytes from state[0..3]
 *   M[4]    = 0x00000080u  (0x80 padding marker at byte 16, LE byte 0)
 *   M[5..13]= 0
 *   M[14]   = 16 * 8 = 128 (length in bits, LE low 32)
 *   M[15]   = 0
 *
 * 16 + 1 + 8 = 25 < 56 ⇒ single block.
 *
 * All other extension functions (template_state, template_init,
 * template_transform, template_finalize, template_digest_compare,
 * template_emit_hit) are byte-identical to gpu_md5_core.cl. Duplicated
 * inline rather than #include to keep the per-algo source unit
 * self-contained and the cache key clean (Memo B R3: defines_str +
 * source-text hash).
 *
 * Cache key (R3): defines_str = "HASH_WORDS=4,HASH_BLOCK_BYTES=64" —
 * same as MD5. Distinct cache entry guaranteed by source-text hash
 * difference (only template_iterate differs; the rest is byte-identical).
 *
 * R1 mitigation: single private buffer pattern; no addrspace-cast
 * helpers. Same shape as gpu_md5_core.cl.
 *
 * Source order at compile time:
 *   [ gpu_common_str, gpu_md5_rules_str, gpu_md5raw_core_str, gpu_template_str ]
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

static inline void template_init(template_state *st) {
    st->h[0] = 0x67452301u;
    st->h[1] = 0xEFCDAB89u;
    st->h[2] = 0x98BADCFEu;
    st->h[3] = 0x10325476u;
}

static inline void template_transform(template_state *st,
                                      const uchar *block)
{
    uint M[16];
    /* MD5 reads message words little-endian. */
    for (int j = 0; j < 16; j++) {
        int b = j * 4;
        M[j] = (uint)block[b]
             | ((uint)block[b + 1] << 8)
             | ((uint)block[b + 2] << 16)
             | ((uint)block[b + 3] << 24);
    }
    md5_block(&st->h[0], &st->h[1], &st->h[2], &st->h[3], M);
}

/* template_finalize: byte-identical to gpu_md5_core.cl. */
static inline void template_finalize(template_state *st,
                                     const uchar *data, int len)
{
    uint M[16];
    int pos = 0;

    while (len - pos >= HASH_BLOCK_BYTES) {
        for (int j = 0; j < 16; j++) {
            int b = pos + j * 4;
            M[j] = (uint)data[b]
                 | ((uint)data[b + 1] << 8)
                 | ((uint)data[b + 2] << 16)
                 | ((uint)data[b + 3] << 24);
        }
        md5_block(&st->h[0], &st->h[1], &st->h[2], &st->h[3], M);
        pos += HASH_BLOCK_BYTES;
    }

    int rem = len - pos;  /* 0..63 */

    for (int j = 0; j < 16; j++) M[j] = 0;

    for (int i = 0; i < rem; i++) {
        uint v = (uint)data[pos + i];
        M[i >> 2] |= v << ((i & 3) * 8);
    }
    M[rem >> 2] |= (uint)0x80 << ((rem & 3) * 8);

    if (rem < 56) {
        M[14] = (uint)(len * 8);
        M[15] = 0;
        md5_block(&st->h[0], &st->h[1], &st->h[2], &st->h[3], M);
    } else {
        md5_block(&st->h[0], &st->h[1], &st->h[2], &st->h[3], M);
        for (int j = 0; j < 16; j++) M[j] = 0;
        M[14] = (uint)(len * 8);
        M[15] = 0;
        md5_block(&st->h[0], &st->h[1], &st->h[2], &st->h[3], M);
    }
}

/* template_iterate: MD5RAW iter — re-feed the 16-byte BINARY digest
 * (not hex-encoded) into the compression. CPU reference at mdxfind.c
 * line 24237:
 *
 *   memcpy(cur, md5buf.h, 16);
 *   mymd5(cur, 16, md5buf.h);
 *
 * State after template_finalize: st->h[0..3] hold the 4 LE uint32 digest
 * words. They are exactly the LE-packed M[0..3] of an md5_block call
 * given the 16-byte digest as input (since md5 reads message words LE).
 *
 * Layout:
 *   M[0..3]  = st->h[0..3] (16 LE digest bytes; native md5 word order)
 *   M[4]     = 0x80u    (0x80 padding marker at byte 16, LE)
 *   M[5..13] = 0
 *   M[14]    = 16 * 8 = 128 (length in bits, LE)
 *   M[15]    = 0
 *
 * Single block (16 + 1 + 8 = 25 < 56). After absorption, h[] holds the
 * fresh digest in native MD5 LE form.
 */
static inline void template_iterate(template_state *st)
{
    uint M[16];
    M[0] = st->h[0];
    M[1] = st->h[1];
    M[2] = st->h[2];
    M[3] = st->h[3];
    M[4] = 0x80u;
    for (int j = 5; j < 14; j++) M[j] = 0u;
    M[14] = 16u * 8u;
    M[15] = 0u;
    /* Reset state to MD5 IV; absorb the prepared block. */
    st->h[0] = 0x67452301u;
    st->h[1] = 0xEFCDAB89u;
    st->h[2] = 0x98BADCFEu;
    st->h[3] = 0x10325476u;
    md5_block(&st->h[0], &st->h[1], &st->h[2], &st->h[3], M);
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
