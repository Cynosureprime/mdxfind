/*
 * $Revision: 1.2 $
 * $Log: gpu_sha1dru_core.cl,v $
 * Revision 1.2  2026/05/11 05:22:02  dlr
 * Backfill $Revision/$Log RCS keyword stanzas per feedback_rcs_keyword_stanzas.md. Passive 4-line comment block at top of file; no behavioral change. Hand-authored .cl file was missing required stanzas (per memory: all hand-authored .c/.h/.cl/.frag/.tmpl/.py/.sh files MUST contain $Revision/$Log keyword stanzas). Build green on .205 against the post-add files; OpenCL compile strips comments so no kernel behavior change.
 *
 */
/* gpu_sha1dru_core.cl — SHA1DRU (Drupal SHA1, hashcat -m 7900) algorithm
 * extension functions for the generic dispatch template (Memo B Phase B6.11).
 *
 * STATUS: B6.11 — first 1M-iteration algorithm on the unified template path.
 *
 * SHA1DRU = SHA1(pass), then 1,000,000 iterations of SHA1(hex_lc(state) || pass).
 *
 * CPU reference (mdxfind.c JOB_SHA1DRU at lines 14261-14285):
 *
 *   mysha1((char *)cur, len, curin.h);             // initial = SHA1(pass)
 *   memmove(linebuf + 40, cur, len);
 *   for (x = 1; x < 1000001; x++) {
 *     prmd5(curin.h, linebuf, 40);                 // hex_lc(state) -> linebuf[0..39]
 *     linebuf[40] = cur[0];                        // defensive overwrite of pass[0]
 *     mysha1((char *)linebuf, len + 40, curin.h);  // SHA1(hex(state) || pass)
 *   }
 *   checkhash(&curin, 40, 1, job);                 // ONE probe at end
 *
 * Key semantic property: ONLY the FINAL state (after all 1M iterations) is
 * probed. Unlike SQL5 (which probes every iter), SHA1DRU's iter loop is an
 * implementation detail of the algorithm — not user-controlled via -i.
 *
 * DESIGN: 1M-loop INSIDE template_finalize, max_iter = 1.
 * ----------------------------------------------------------
 * The host sets params.max_iter = 1 (Maxiter is the user's -i, default 1
 * for SHA1DRU; the algorithm does NOT honor -i — its 1M is hardcoded).
 *
 * The kernel's outer iter loop in template_phase0 runs ONCE:
 *
 *   for (uint iter = 1; iter <= 1; iter++) {
 *     digest_compare(&st);   // probe FINAL state from finalize
 *     // iter == max_iter, so template_iterate is NOT called
 *   }
 *
 * Therefore:
 *   - template_finalize: does the FULL chain (SHA1(pass) + 1M inner iters)
 *   - template_iterate:  STUB (never called given max_iter=1)
 *
 * This avoids 1M wasted probes against the compact table — only the
 * final state is probed, matching CPU semantics exactly.
 *
 * Why this differs from SQL5:
 *   SQL5 probes every iter (its CPU `checkhash(..., x, job)` inside the
 *   for-loop emits at every iter level). SHA1DRU's `checkhash` is OUTSIDE
 *   the loop — only the final state matters. SQL5's pattern (1M probes)
 *   would be both incorrect and 100,000x more expensive.
 *
 * Why finalize and not iterate:
 *   With max_iter = 1, the kernel's iter loop body runs once and never
 *   calls template_iterate (the `if (iter < max_iter)` guard prevents it).
 *   Putting the work in iterate would require max_iter = 2, which would
 *   then waste a probe on the initial SHA1(pass) state at iter=1.
 *
 * Source order at compile time (mirrors SHA1 build):
 *   [ gpu_common_str, gpu_md5_rules_str, gpu_sha1dru_core_str, gpu_template_str ]
 *
 * Cache key (R3): defines_str = "HASH_WORDS=5,HASH_BLOCK_BYTES=64,BASE_ALGO=sha1,
 * ITER_COUNT=1000000" — same HASH_WORDS/HASH_BLOCK_BYTES as SHA1 / SHA1RAW /
 * SQL5; the ITER_COUNT token distinguishes the cache entry. Distinct cache
 * entry by source-text hash regardless (the 1M-loop body in finalize differs).
 *
 * R1 mitigation (AMD ROCm comgr addrspace): single private buffer pattern.
 * No __private uchar* helpers; no addrspace-cast ternaries.
 *
 * R2 (register pressure): the 1M-iter body uses one M[16] scratch + the 5-word
 * state + a small private staging area for hex(state)||pass (capped to
 * RULE_BUF_LIMIT). Expected priv_mem on gfx1201 ~ comparable to or slightly
 * above SHA1 (extra hexbuf staging ~80 bytes). HARD GATE 43,024 B; SHA1
 * unsalted reading was ~41 KB so ample headroom expected.
 *
 * VALIDATION ORACLE: the existing slab kernel sha1dru_batch in gpu_sha1.cl
 * (lines 26-132 pre-B6.11). The CORE OF THIS FILE is a port of that slab
 * kernel's body (lines 80-120 of gpu_sha1.cl pre-B6.11) into the template
 * extension API. Differences from the slab kernel:
 *   - reads pass from `data` (post-rule buf passed by template) rather
 *     than from a packed wordbuf.
 *   - state stays in BE uint32 throughout; bswap to LE happens only at
 *     digest_compare / emit time (matches gpu_sha1_core.cl's convention).
 *   - the CPU's "linebuf[40] = cur[0]" defensive overwrite is unnecessary
 *     here because we build hex(state)||pass into a fresh staging buffer
 *     where bytes [40..40+plen) are written from `data` (the pass) every
 *     iter — no stale overwrite to defend against.
 */

#ifndef HASH_WORDS
#define HASH_WORDS 5
#endif
#ifndef HASH_BLOCK_BYTES
#define HASH_BLOCK_BYTES 64
#endif
#ifndef ITER_COUNT
#define ITER_COUNT 1000000
#endif

/* Per-lane state struct: SHA1 carries 5 uint32 chaining values. The template
 * only reads/writes these via template_finalize's output + template_digest_-
 * compare; it does not introspect the struct. */
typedef struct {
    uint h[HASH_WORDS];
} template_state;

/* template_init: install SHA1 IV into state. */
static inline void template_init(template_state *st) {
    st->h[0] = 0x67452301u;
    st->h[1] = 0xEFCDAB89u;
    st->h[2] = 0x98BADCFEu;
    st->h[3] = 0x10325476u;
    st->h[4] = 0xC3D2E1F0u;
}

/* template_transform: stub for interface symmetry. SHA1DRU's template_finalize
 * builds M[] in-place per block — never routes through this. */
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
    sha1_block(&st->h[0], M);
}

/* sha1dru_init_finalize: classic SHA1 finalize over (data, len) using
 * st->h as the chain (BE state). Mirrors gpu_sha1_core.cl's template_finalize
 * byte-for-byte. Used to compute the INITIAL SHA1(pass). */
static inline void sha1dru_init_finalize(template_state *st,
                                         const uchar *data, int len)
{
    uint M[16];
    int pos = 0;

    /* Reset state to SHA1 IV. */
    st->h[0] = 0x67452301u;
    st->h[1] = 0xEFCDAB89u;
    st->h[2] = 0x98BADCFEu;
    st->h[3] = 0x10325476u;
    st->h[4] = 0xC3D2E1F0u;

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

    int rem = len - pos;
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

/* SHA1DRU_MAX_PLEN: per-iter staging buffer size cap for hex(40) + pass.
 *
 * Sized at 256 bytes to comfortably hold hex(40) + plen up to ~215 bytes
 * (well above typical wordlist passwords). This matches the slab kernel's
 * 256-byte hexhash slot in gpujob.h:156 (`hexhash[GPUBATCH_MAX][256]`).
 * Passwords longer than 215 bytes would truncate at the SHA1DRU iter input
 * — same behavior as the slab path (which capped hexhash at 256 too).
 *
 * R2 (gfx1201 priv_mem): a per-iter alloc of RULE_BUF_LIMIT (~40 KB) blew
 * out the spill region (B6.11 first build measured 82,080 B per work-item
 * on gfx1201). 256 bytes here keeps us comfortably under the 43,024 B
 * HARD GATE — expected reading ~41 KB (in line with SHA1 unsalted's
 * ~41 KB baseline). */
#define SHA1DRU_MAX_PLEN  216
#define SHA1DRU_HEXBUF    (40 + SHA1DRU_MAX_PLEN)

/* sha1dru_iter_step: one iteration of the SHA1DRU inner loop:
 *   build hex_lc(state) into bytes 0..39 of a staging buffer;
 *   copy `pass` into bytes 40..40+plen-1 (capped at SHA1DRU_MAX_PLEN);
 *   total = 40 + min(plen, SHA1DRU_MAX_PLEN);
 *   reset state to SHA1 IV; absorb the staged buffer.
 *
 * Hex encoding is BIG-ENDIAN per state word, lowercase ('0'..'9','a'..'f').
 * Mirrors the CPU's prmd5 (lowercase hex) over the 20-byte BE state.
 *
 * Block layout:
 *   - For plen <= 14: total <= 54 bytes — single block (40 + plen + 1 + 8 <= 64
 *     when plen <= 14). The 0x80 marker at byte (40+plen), length BE in M[15].
 *   - For plen >= 15: total >= 55 bytes — needs two blocks (one full block
 *     of 64 bytes, then padding+length in second block).
 *   - For plen >= 24: 40 + plen >= 64 — block 1 full, second block holds
 *     remaining tail + 0x80 + zeros + length.
 *   - For plen >= 88 (total >= 128): 2 full blocks + final pad block.
 *   - For plen >= 152 (total >= 192): 3 full blocks + final pad block.
 *
 * The hexbuf staging area lives on the kernel stack (private memory). */
static inline void sha1dru_iter_step(template_state *st,
                                     const uchar *pass, int plen)
{
    __attribute__((aligned(16))) uchar hexbuf[SHA1DRU_HEXBUF];

    /* Encode 20 bytes of state[0..4] as 40 lowercase hex chars in hexbuf[0..39]. */
    for (int w = 0; w < 5; w++) {
        uint sv = st->h[w];
        for (int b = 0; b < 4; b++) {
            uint byte = (sv >> ((3 - b) * 8)) & 0xff;
            uint hi = (byte >> 4) & 0xf;
            uint lo = byte & 0xf;
            hexbuf[w * 8 + b * 2]     = (uchar)(hi + ((hi < 10u) ? '0' : ('a' - 10)));
            hexbuf[w * 8 + b * 2 + 1] = (uchar)(lo + ((lo < 10u) ? '0' : ('a' - 10)));
        }
    }

    /* Append pass at offset 40, capped at SHA1DRU_MAX_PLEN. */
    int eff_plen = (plen < SHA1DRU_MAX_PLEN) ? plen : SHA1DRU_MAX_PLEN;
    for (int i = 0; i < eff_plen; i++) {
        hexbuf[40 + i] = pass[i];
    }
    int total = 40 + eff_plen;

    /* Reset state to SHA1 IV. */
    st->h[0] = 0x67452301u;
    st->h[1] = 0xEFCDAB89u;
    st->h[2] = 0x98BADCFEu;
    st->h[3] = 0x10325476u;
    st->h[4] = 0xC3D2E1F0u;

    /* SHA1(hexbuf, total) — same finalize structure as sha1dru_init_finalize
     * but inlined to avoid a second function call across 1M iters. */
    uint M[16];
    int pos = 0;
    while (total - pos >= HASH_BLOCK_BYTES) {
        for (int j = 0; j < 16; j++) {
            int b = pos + j * 4;
            M[j] = ((uint)hexbuf[b]     << 24)
                 | ((uint)hexbuf[b + 1] << 16)
                 | ((uint)hexbuf[b + 2] << 8)
                 |  (uint)hexbuf[b + 3];
        }
        sha1_block(&st->h[0], M);
        pos += HASH_BLOCK_BYTES;
    }

    int rem = total - pos;
    for (int j = 0; j < 16; j++) M[j] = 0;
    for (int i = 0; i < rem; i++) {
        int wi = i >> 2;
        int bi = 3 - (i & 3);
        M[wi] |= ((uint)hexbuf[pos + i]) << (bi * 8);
    }
    {
        int wi = rem >> 2;
        int bi = 3 - (rem & 3);
        M[wi] |= ((uint)0x80u) << (bi * 8);
    }

    if (rem < 56) {
        M[14] = 0;
        M[15] = (uint)((uint)total * 8u);
        sha1_block(&st->h[0], M);
    } else {
        sha1_block(&st->h[0], M);
        for (int j = 0; j < 16; j++) M[j] = 0;
        M[14] = 0;
        M[15] = (uint)((uint)total * 8u);
        sha1_block(&st->h[0], M);
    }
}

/* template_finalize: the FULL SHA1DRU chain — initial SHA1(pass) followed
 * by ITER_COUNT (1,000,000) iterations of SHA1(hex_lc(state) || pass).
 *
 * After return, st->h[0..4] holds the final SHA1 digest (BIG-ENDIAN — sha1_block's
 * natural output). template_digest_compare bswap32's to LE before probing. */
static inline void template_finalize(template_state *st,
                                     const uchar *data, int len)
{
    /* Step 1: initial SHA1(pass). */
    sha1dru_init_finalize(st, data, len);

    /* Step 2: 1,000,000 iterations of SHA1(hex_lc(state) || pass). */
    for (uint iter = 0; iter < (uint)ITER_COUNT; iter++) {
        sha1dru_iter_step(st, data, len);
    }
}

/* template_iterate: STUB. With max_iter = 1 (host-set for SHA1DRU), the
 * outer iter loop in template_phase0 runs exactly once and never calls
 * template_iterate. Kept for interface symmetry with the rest of the
 * unsalted-template family — if a future variant of SHA1DRU honored -i
 * (e.g., HASH_DRUPALx2 = run two 1M-cycle SHA1DRUs back-to-back), this
 * is where the per-iter step would live. */
static inline void template_iterate(template_state *st)
{
    /* No-op — see file header comment. */
    (void)st;
}

/* template_digest_compare: probe the compact table. SHA1DRU's final digest
 * is BIG-ENDIAN; compact table is keyed LE — bswap32 first 4 words. */
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

/* template_emit_hit: emit a hit. SHA1 = 5 LE-byteswapped uint32 digest words. */
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

/* B3 dedup+overflow-aware variant. */
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
