/*
 * $Revision: 1.2 $
 * $Log: gpu_sql5_core.cl,v $
 * Revision 1.2  2026/05/11 05:22:02  dlr
 * Backfill $Revision/$Log RCS keyword stanzas per feedback_rcs_keyword_stanzas.md. Passive 4-line comment block at top of file; no behavioral change. Hand-authored .cl file was missing required stanzas (per memory: all hand-authored .c/.h/.cl/.frag/.tmpl/.py/.sh files MUST contain $Revision/$Log keyword stanzas). Build green on .205 against the post-add files; OpenCL compile strips comments so no kernel behavior change.
 *
 */
/* gpu_sql5_core.cl — SQL5 (MySQL 4.1+ password) algorithm extension
 * functions for the generic dispatch template (Memo B Phase B5
 * sub-batch 6, Tier C).
 *
 * SQL5 is the MySQL 4.1+ password hash: SHA1(SHA1(password)). The mdxfind
 * implementation, however, has an unusual iter loop that distinguishes
 * SQL5 from a simple sha1(sha1(p)) primitive:
 *
 * CPU reference (mdxfind.c JOB_SQL5 at line 25301):
 *
 *   for (x = 1; x <= Maxiter; x++) {
 *     mysha1(cur, len, curin.h);          // FIRST sha1: inner = sha1(input)
 *     mysha1(curin.h, 20, md5buf.h);      // SECOND sha1: outer = sha1(inner)
 *     cur = prmd5UC(curin.h, mdbuf, 40);  // hex(UPPERCASE) of FIRST sha1
 *     len = 40;
 *     hashcnt++;
 *     checkhash(&md5buf, 40, x, job);     // probe outer == sha1(sha1(input))
 *   }
 *
 * Per-iter behavior:
 *   - At iter N, input = (N==1 ? rule-mutated buffer : UPPERCASE_HEX(40)
 *     of inner from iter N-1).
 *   - Compute inner_N = sha1(input_N). Compute outer_N = sha1(inner_N).
 *   - Probe outer_N against compact table (it's the value being looked up
 *     in MySQL's user table).
 *   - If iter < Maxiter, prepare input_{N+1} = UPPERCASE_HEX(inner_N).
 *
 * Per-algorithm state struct retains BOTH sha1 chains:
 *   - state_inner[5]: the first sha1's chain (carried across iters as
 *     the basis for the next iter's UPPERCASE-hex re-feed).
 *   - h[5]: the second sha1's chain (exposed to template_digest_compare;
 *     this is what gets matched + emitted).
 *
 * The template body's call sequence is:
 *   template_init(&st)
 *   template_finalize(&st, buf, len)   -> compute inner + outer, expose
 *                                         outer in h[]; keep inner in
 *                                         state_inner[] for iter step.
 *   for iter 1..Maxiter:
 *     digest_compare(&st)               -> probe h[]
 *     if iter < Maxiter:
 *       template_iterate(&st)           -> uppercase_hex(state_inner) ->
 *                                          sha1 -> new state_inner; then
 *                                          sha1(state_inner_BE_20) ->
 *                                          new h[].
 *
 * Cache key (R3): defines_str = "HASH_WORDS=5,HASH_BLOCK_BYTES=64" — same
 * as SHA1 / SHA1RAW. Distinct cache entry by source-text hash.
 *
 * R1 mitigation: single private buffer pattern; no addrspace-cast
 * helpers. Pattern matches gpu_sha1_core.cl.
 *
 * R2: same compression footprint as SHA1 (W[80] schedule footprint
 * doubled across two sha1_block calls per finalize/iterate). Expected
 * priv_mem_size on gfx1201 ~ identical to SHA1 (W[80] is a private
 * scalar inside sha1_block; reused across calls).
 *
 * Source order at compile time:
 *   [ gpu_common_str, gpu_md5_rules_str, gpu_sql5_core_str, gpu_template_str ]
 */

#ifndef HASH_WORDS
#define HASH_WORDS 5
#endif
#ifndef HASH_BLOCK_BYTES
#define HASH_BLOCK_BYTES 64
#endif

typedef struct {
    uint h[HASH_WORDS];       /* outer sha1 (exposed) — used by digest_compare */
    uint state_inner[5];      /* inner sha1 — basis for next iter's hex feed */
} template_state;

static inline void template_init(template_state *st) {
    /* h[] will be set by template_finalize after the outer sha1. */
    st->h[0] = 0x67452301u;
    st->h[1] = 0xEFCDAB89u;
    st->h[2] = 0x98BADCFEu;
    st->h[3] = 0x10325476u;
    st->h[4] = 0xC3D2E1F0u;
    /* state_inner is initialized inside template_finalize before the
     * inner sha1 absorb. */
    st->state_inner[0] = 0x67452301u;
    st->state_inner[1] = 0xEFCDAB89u;
    st->state_inner[2] = 0x98BADCFEu;
    st->state_inner[3] = 0x10325476u;
    st->state_inner[4] = 0xC3D2E1F0u;
}

/* template_transform: stub for interface symmetry. Not used by the SQL5
 * finalize because we handle padding inside template_finalize directly. */
static inline void template_transform(template_state *st,
                                      const uchar *block)
{
    uint M[16];
    for (int j = 0; j < 16; j++) {
        int b = j * 4;
        M[j] = ((uint)block[b]     << 24)
             | ((uint)block[b + 1] << 16)
             | ((uint)block[b + 2] << 8)
             |  (uint)block[b + 3];
    }
    sha1_block(&st->state_inner[0], M);
}

/* sql5_inner_finalize: classic SHA1 finalize on (data, len) using
 * st->state_inner as the chain. Mirrors gpu_sha1_core.cl's
 * template_finalize byte-for-byte but writes to state_inner. */
static inline void sql5_inner_finalize(template_state *st,
                                       const uchar *data, int len)
{
    uint M[16];
    int pos = 0;

    /* Reset inner state to SHA1 IV before absorbing the input. */
    st->state_inner[0] = 0x67452301u;
    st->state_inner[1] = 0xEFCDAB89u;
    st->state_inner[2] = 0x98BADCFEu;
    st->state_inner[3] = 0x10325476u;
    st->state_inner[4] = 0xC3D2E1F0u;

    while (len - pos >= HASH_BLOCK_BYTES) {
        for (int j = 0; j < 16; j++) {
            int b = pos + j * 4;
            M[j] = ((uint)data[b]     << 24)
                 | ((uint)data[b + 1] << 16)
                 | ((uint)data[b + 2] << 8)
                 |  (uint)data[b + 3];
        }
        sha1_block(&st->state_inner[0], M);
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
        sha1_block(&st->state_inner[0], M);
    } else {
        sha1_block(&st->state_inner[0], M);
        for (int j = 0; j < 16; j++) M[j] = 0;
        M[14] = 0;
        M[15] = (uint)((uint)len * 8u);
        sha1_block(&st->state_inner[0], M);
    }
}

/* sql5_outer_sha1: feed the 20-byte BE inner-sha1 digest into a fresh
 * sha1_block and store the result in st->h[]. Single block (20+1+8 = 29
 * < 56). Mirrors the SHA1RAW iter step exactly. */
static inline void sql5_outer_sha1(template_state *st)
{
    uint M[16];
    M[0] = st->state_inner[0];
    M[1] = st->state_inner[1];
    M[2] = st->state_inner[2];
    M[3] = st->state_inner[3];
    M[4] = st->state_inner[4];
    M[5] = 0x80000000u;        /* 0x80 BE at byte 20 */
    for (int j = 6; j < 14; j++) M[j] = 0u;
    M[14] = 0u;
    M[15] = 20u * 8u;          /* 160 bits */
    /* Reset h[] to SHA1 IV; absorb the prepared block. */
    st->h[0] = 0x67452301u;
    st->h[1] = 0xEFCDAB89u;
    st->h[2] = 0x98BADCFEu;
    st->h[3] = 0x10325476u;
    st->h[4] = 0xC3D2E1F0u;
    sha1_block(&st->h[0], M);
}

/* template_finalize: SQL5 = inner sha1 (over input) + outer sha1 (over
 * inner). Both states are kept in template_state for the iter step. */
static inline void template_finalize(template_state *st,
                                     const uchar *data, int len)
{
    sql5_inner_finalize(st, data, len);
    sql5_outer_sha1(st);
}

/* template_iterate: SQL5 iter. Per CPU reference:
 *   prev_inner = state_inner (in BE uint32 words)
 *   uppercase_hex_chars[40] = HEX_UC(prev_inner)
 *   new_inner = sha1(uppercase_hex_chars[40])  -> updates state_inner
 *   new_outer = sha1(new_inner_BE_20)          -> updates h[]
 *
 * The 40 uppercase hex chars fit in M[0..9]. M[10] = 0x80000000 BE
 * (padding marker at byte 40), M[14] = 0, M[15] = 320 (40 * 8). Single
 * block (40 + 1 + 8 = 49 < 56). */
static inline void template_iterate(template_state *st)
{
    uint M[16];

    /* Build M[0..9] from state_inner[0..4] as 40 BE-encoded UPPERCASE
     * hex chars. Each state word produces 8 hex chars (2 M-words BE
     * encoded). Mirrors gpu_sha1_core.cl's iter except for character
     * case: ('A' - 10) instead of ('a' - 10). */
    for (int i = 0; i < 5; i++) {
        uint s = st->state_inner[i];
        uint b0 = (s >> 24) & 0xff;
        uint b1 = (s >> 16) & 0xff;
        uint b2 = (s >> 8)  & 0xff;
        uint b3 = s & 0xff;
        uint hi0 = (b0 >> 4) & 0xf, lo0 = b0 & 0xf;
        uint hi1 = (b1 >> 4) & 0xf, lo1 = b1 & 0xf;
        uint hi2 = (b2 >> 4) & 0xf, lo2 = b2 & 0xf;
        uint hi3 = (b3 >> 4) & 0xf, lo3 = b3 & 0xf;
        uint h0 = ((hi0 + ((hi0 < 10) ? '0' : ('A' - 10))) << 8)
                |  (lo0 + ((lo0 < 10) ? '0' : ('A' - 10)));
        uint h1 = ((hi1 + ((hi1 < 10) ? '0' : ('A' - 10))) << 8)
                |  (lo1 + ((lo1 < 10) ? '0' : ('A' - 10)));
        uint h2 = ((hi2 + ((hi2 < 10) ? '0' : ('A' - 10))) << 8)
                |  (lo2 + ((lo2 < 10) ? '0' : ('A' - 10)));
        uint h3 = ((hi3 + ((hi3 < 10) ? '0' : ('A' - 10))) << 8)
                |  (lo3 + ((lo3 < 10) ? '0' : ('A' - 10)));
        M[i * 2]     = (h0 << 16) | h1;
        M[i * 2 + 1] = (h2 << 16) | h3;
    }
    M[10] = 0x80000000u;        /* 0x80 BE at byte 40 */
    for (int j = 11; j < 14; j++) M[j] = 0u;
    M[14] = 0u;
    M[15] = 40u * 8u;           /* 40 hex chars = 320 bits */

    /* Reset state_inner to SHA1 IV; absorb the prepared block to compute
     * new_inner = sha1(uppercase_hex_40). */
    st->state_inner[0] = 0x67452301u;
    st->state_inner[1] = 0xEFCDAB89u;
    st->state_inner[2] = 0x98BADCFEu;
    st->state_inner[3] = 0x10325476u;
    st->state_inner[4] = 0xC3D2E1F0u;
    sha1_block(&st->state_inner[0], M);

    /* Compute new_outer = sha1(new_inner_BE_20) into h[]. */
    sql5_outer_sha1(st);
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
    /* SHA1 state is BE; compact table is keyed on LE — bswap32 first 4. */
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
