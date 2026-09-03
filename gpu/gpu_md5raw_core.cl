/*
 * $Revision: 1.3 $
 * $Log: gpu_md5raw_core.cl,v $
 * Revision 1.3  2026/08/30 14:35:34  dlr
 * Restore the RAW-family iteration rule in the GPU kernels to match mdxfind.c 1.549. These kernels were written against the post-1.290 CPU regression and validated byte-exact against it, which is why CPU-vs-GPU conformance testing never caught the error: both sides agreed on the wrong answer. The correct rule is ONE binary feed at the base, giving X(X_bin(pass)), followed by STANDARD hex iteration. The former template_iterate already implemented exactly that binary absorb, so it is renamed template_raw_refeed, moved ahead of template_finalize and called once at the end of it; template_iterate is replaced by the plain siblings hex step. For the Metal MD5 variant the algo_mode parameter is dropped since RAW has no uppercase form and the call site uses the legacy single-argument shape. Validated on fpga.local GTX 1080 via OpenCL: 15 of 15 lines identical between CPU and GPU across five types and three depths, with a bogus-hash negative control returning zero. On dev1.local Apple M1 via Metal the base is correct and matches CPU at x01 for all five types, but iterations beyond x01 are not returned; that is a coverage gap and not a wrong answer, confirmed by comm showing zero Metal lines absent from the CPU set.
 *
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

/* template_raw_refeed: absorb the 16-byte BINARY digest as a fresh message.
 * This is the ONE binary feed that defines the RAW family: the base digest is
 * md5(md5_bin(pass)). It is NOT the -i iteration step -- see template_iterate.
 *
 * Layout: M[0..3] = 16 LE digest bytes; M[4] = 0x80 pad; M[14] = 128 bits.
 * 16 + 1 + 8 = 25 < 56, single block.
 */
static inline void template_raw_refeed(template_state *st)
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
    st->h[0] = 0x67452301u;
    st->h[1] = 0xEFCDAB89u;
    st->h[2] = 0x98BADCFEu;
    st->h[3] = 0x10325476u;
    md5_block(&st->h[0], &st->h[1], &st->h[2], &st->h[3], M);
}

/* template_finalize: plain MD5 finalize, then ONE binary re-feed. */
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
    /* the one binary feed that makes this MD5RAW rather than MD5 */
    template_raw_refeed(st);

}

/* template_iterate: -i loop step. MD5RAW iterates on the 32-byte lowercase
 * HEX encoding of the digest, exactly as plain MD5 does -- the binary feed
 * belongs to the BASE (template_raw_refeed), not to iteration.
 *
 * Corrected 2026-08-30 alongside mdxfind.c 1.549. Between mdxfind.c 1.290 and
 * 1.548 the CPU reported md5(pass) as x01 and re-fed the RAW digest each
 * round; this kernel was written against that regressed reference and matched
 * it byte-exact, which is why CPU-vs-GPU validation never caught it. The rule
 * confirmed by the gp crack corpus is: one binary feed at the base, then
 * standard hex iteration.
 */
static inline void template_iterate(template_state *st)
{
    uint M[16];
    md5_to_hex_lc(st->h[0], st->h[1], st->h[2], st->h[3], M);
    M[8] = 0x80u;
    for (int j = 9; j < 14; j++) M[j] = 0u;
    M[14] = 32u * 8u;     /* 32 hex chars = 256 bits */
    M[15] = 0u;
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
