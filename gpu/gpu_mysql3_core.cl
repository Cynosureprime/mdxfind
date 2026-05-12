/*
 * $Revision: 1.2 $
 * $Log: gpu_mysql3_core.cl,v $
 * Revision 1.2  2026/05/11 05:22:01  dlr
 * Backfill $Revision/$Log RCS keyword stanzas per feedback_rcs_keyword_stanzas.md. Passive 4-line comment block at top of file; no behavioral change. Hand-authored .cl file was missing required stanzas (per memory: all hand-authored .c/.h/.cl/.frag/.tmpl/.py/.sh files MUST contain $Revision/$Log keyword stanzas). Build green on .205 against the post-add files; OpenCL compile strips comments so no kernel behavior change.
 *
 */
/* gpu_mysql3_core.cl — MYSQL3 (-m e456) algorithm extension functions
 * for the generic dispatch template (Memo B Phase B5 sub-batch 7).
 *
 * MYSQL3 = legacy MySQL OLD_PASSWORD() hash. 64-bit output. Per-byte
 * arithmetic accumulator loop:
 *
 *   nr = 1345345333; nr2 = 0x12345671; add = 7;
 *   for each input byte b in input:
 *     if b == ' ' or b == '\t': continue;
 *     nr  ^= ((nr & 63) + add) * b + (nr << 8);
 *     nr2 += (nr2 << 8) ^ nr;
 *     add += b;
 *   final = (nr & 0x7fffffff, nr2 & 0x7fffffff)  -- 8 bytes total
 *
 * Iter step (CPU JOB_MYSQL3, mdxfind.c:25180-25186):
 *   for x = 1..Maxiter:
 *     mysql3(cur, len, linebuf);              -- linebuf = 16 ASCII hex chars
 *     get32(linebuf, curin.h, 16); checkhash(curin, 16, x, JOB_MYSQL3);
 *     cur = linebuf; len = 16;                -- next iter input is the hex
 *
 *   The hex string format is "%08x%08x" of nr & 0x7fffffff and nr2 & 0x7fffffff
 *   (lowercase). 16 ASCII hex chars total.
 *
 * Probe path (host zero-pad fix, mdxfind.c:36400-36412 rev 1.399+): the
 * compact-table HashDataBuf entries for sub-128-bit outputs are stored as
 * 16 bytes (real bytes followed by zero pad). For MYSQL3 (8-byte output),
 * the kernel emits hx = byteswap(nr), hy = byteswap(nr2), hz = hw = 0 ---
 * matches the slab kernel gpu_mysql3unsalted.cl:121-125, and matches the
 * "16-hex" CPU compact-table storage of "AAAAAAAA00000000BBBBBBBB" hex
 * patterns (where AAA = nr LE-bytes, BBB = nr2 LE-bytes; remaining 8
 * bytes zero per the host pad).
 *
 * State carried in template_state: HASH_WORDS=4 uint32. h[0]=hx, h[1]=hy,
 * h[2]=h[3]=0. We retain the byteswapped probe form so template_iterate
 * can recover nr/nr2 by reverse-byteswapping h[0]/h[1].
 *
 * Cache key (R3): defines_str = "HASH_WORDS=4,HASH_BLOCK_BYTES=64". Distinct
 * cache entry guaranteed by source-text hash difference (the per-byte
 * MYSQL3 arithmetic is unique).
 *
 * R1 mitigation: single private buffer; no addrspace-cast helpers.
 *
 * Source order at compile time:
 *   [ gpu_common_str, gpu_md5_rules_str, gpu_mysql3_core_str, gpu_template_str ]
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

/* MYSQL3 has no MD-style block. The "compress" interface here is a stub
 * for symmetry; the per-byte loop runs entirely inside template_finalize. */

static inline void template_init(template_state *st) {
    /* No work-state install needed before per-byte loop. We zero the probe
     * state slots so template_finalize and template_iterate write them
     * cleanly. */
    st->h[0] = 0u;
    st->h[1] = 0u;
    st->h[2] = 0u;
    st->h[3] = 0u;
}

/* Stub for interface symmetry (the shared template body in gpu_template.cl
 * never invokes this for MYSQL3; template_finalize handles the input
 * directly). */
static inline void template_transform(template_state *st,
                                      const uchar *block)
{
    (void)st;
    (void)block;
}

/* Inline MYSQL3 per-byte arithmetic loop. Consumes `len` input bytes from
 * `data` and produces the 8-byte digest as (nr_out, nr2_out). Skips spaces
 * and tabs (matching CPU mysql3() at mdxfind.c:3494-3513). */
static inline void mysql3_compute(const uchar *data, int len,
                                  uint *nr_out, uint *nr2_out)
{
    uint nr = 1345345333u;
    uint nr2 = 0x12345671u;
    uint add = 7u;
    for (int i = 0; i < len; i++) {
        uint c = (uint)data[i];
        if (c == 0x20u || c == 0x09u) continue;
        uint tmp = c;
        nr ^= (((nr & 63u) + add) * tmp) + (nr << 8);
        nr2 += (nr2 << 8) ^ nr;
        add += tmp;
    }
    *nr_out = nr & 0x7fffffffu;
    *nr2_out = nr2 & 0x7fffffffu;
}

/* template_finalize: runs the MYSQL3 per-byte loop on the input bytes and
 * stores the byteswapped (nr, nr2) into st->h[0..1]. h[2..3] stay zero. */
static inline void template_finalize(template_state *st,
                                     const uchar *data, int len)
{
    uint nr, nr2;
    mysql3_compute(data, len, &nr, &nr2);
    /* Probe form: byteswap nr/nr2 to match CPU compact-table LE storage of
     * the 16-hex hash representation. Slab kernel does the same at
     * gpu_mysql3unsalted.cl:121-125. */
    uint hx = ((nr  >> 24) & 0xffu) | ((nr  >> 8) & 0xff00u) |
              ((nr  << 8)  & 0xff0000u) | ((nr  << 24) & 0xff000000u);
    uint hy = ((nr2 >> 24) & 0xffu) | ((nr2 >> 8) & 0xff00u) |
              ((nr2 << 8)  & 0xff0000u) | ((nr2 << 24) & 0xff000000u);
    st->h[0] = hx;
    st->h[1] = hy;
    st->h[2] = 0u;
    st->h[3] = 0u;
}

/* template_iterate: -i loop step. CPU reference at mdxfind.c:25180-25186:
 *
 *   mysql3(cur, len, linebuf);   -- linebuf gets "%08x%08x" (16 hex chars)
 *   cur = linebuf; len = 16;     -- next iter consumes those 16 ASCII bytes
 *
 * Implementation:
 *   1. Recover nr, nr2 from the byteswapped st->h[0], st->h[1].
 *      (template_finalize stored byteswap(nr) and byteswap(nr2); reverse
 *      the byteswap to get the raw 31-bit-masked values.)
 *   2. Format 16 lowercase ASCII hex chars in a stack buffer (8 hex from nr,
 *      8 hex from nr2 -- both already & 0x7fffffff in template_finalize so
 *      "%08x" gives leading-zero-padded hex without further masking).
 *   3. Run mysql3_compute on those 16 bytes.
 *   4. Byteswap and store into st->h[0..1] (h[2..3] stay zero).
 */
static inline void template_iterate(template_state *st)
{
    /* Step 1: reverse the byteswap st->h[0/1] back to nr, nr2. */
    uint hx = st->h[0];
    uint hy = st->h[1];
    uint nr  = ((hx >> 24) & 0xffu) | ((hx >> 8) & 0xff00u) |
               ((hx << 8)  & 0xff0000u) | ((hx << 24) & 0xff000000u);
    uint nr2 = ((hy >> 24) & 0xffu) | ((hy >> 8) & 0xff00u) |
               ((hy << 8)  & 0xff0000u) | ((hy << 24) & 0xff000000u);
    /* nr and nr2 are already & 0x7fffffff (set by template_finalize). */

    /* Step 2: emit 16 lowercase hex chars. Layout matches sprintf("%08x%08x",
     * nr, nr2): char[0..7] = nr (high nibble first), char[8..15] = nr2.
     * "%08x" of nr writes nr's MSB byte (bits 24..31) as 2 hex chars first.
     * For each byte b (high to low in nr), high-nibble first then low-nibble.
     */
    uchar hexbuf[16];
    {
        /* nr -> hexbuf[0..7] */
        uint v = nr;
        for (int i = 0; i < 4; i++) {
            uint b = (v >> ((3 - i) * 8)) & 0xffu;
            uint hi = (b >> 4) & 0xfu;
            uint lo = b & 0xfu;
            hexbuf[i*2]     = (uchar)(hi + ((hi < 10u) ? '0' : ('a' - 10)));
            hexbuf[i*2 + 1] = (uchar)(lo + ((lo < 10u) ? '0' : ('a' - 10)));
        }
        /* nr2 -> hexbuf[8..15] */
        v = nr2;
        for (int i = 0; i < 4; i++) {
            uint b = (v >> ((3 - i) * 8)) & 0xffu;
            uint hi = (b >> 4) & 0xfu;
            uint lo = b & 0xfu;
            hexbuf[8 + i*2]     = (uchar)(hi + ((hi < 10u) ? '0' : ('a' - 10)));
            hexbuf[8 + i*2 + 1] = (uchar)(lo + ((lo < 10u) ? '0' : ('a' - 10)));
        }
    }

    /* Step 3+4: rerun MYSQL3 on the 16 hex bytes; byteswap and store. */
    uint nr_n, nr2_n;
    mysql3_compute(hexbuf, 16, &nr_n, &nr2_n);
    uint nhx = ((nr_n  >> 24) & 0xffu) | ((nr_n  >> 8) & 0xff00u) |
               ((nr_n  << 8)  & 0xff0000u) | ((nr_n  << 24) & 0xff000000u);
    uint nhy = ((nr2_n >> 24) & 0xffu) | ((nr2_n >> 8) & 0xff00u) |
               ((nr2_n << 8)  & 0xff0000u) | ((nr2_n << 24) & 0xff000000u);
    st->h[0] = nhx;
    st->h[1] = nhy;
    st->h[2] = 0u;
    st->h[3] = 0u;
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
