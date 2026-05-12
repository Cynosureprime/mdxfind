/*
 * $Revision: 1.2 $
 * $Log: gpu_phpbb3_core.cl,v $
 * Revision 1.2  2026/05/11 05:22:01  dlr
 * Backfill $Revision/$Log RCS keyword stanzas per feedback_rcs_keyword_stanzas.md. Passive 4-line comment block at top of file; no behavioral change. Hand-authored .cl file was missing required stanzas (per memory: all hand-authored .c/.h/.cl/.frag/.tmpl/.py/.sh files MUST contain $Revision/$Log keyword stanzas). Build green on .205 against the post-add files; OpenCL compile strips comments so no kernel behavior change.
 *
 */
/* gpu_phpbb3_core.cl -- PHPBB3 / phpass (e455) algorithm extension
 * functions for the generic dispatch template (Memo B Phase B6 PHPBB3
 * Path A hand-written sibling of the unsalted-template family).
 *
 * STATUS: PHPBB3 fan-out 2026-05-08 -- second iterated-crypt algorithm
 * after SHA1DRU on the unified template path. Single algo_mode (0); no
 * KPASS/KSALT siblings -- JOB_PHPBB3 (e455) is a single-mode algorithm.
 *
 * PHPBB3 / phpass semantics (mirrors mdxfind.c JOB_PHPBB3 at lines
 * 13415-13628 + slab oracle gpu_phpbb3.cl):
 *
 *   salt buffer (12 bytes total): "$H$" + cost_char + 8-byte salt
 *   cost_char decodes via phpitoa64 = "./0123456789ABCDEF...xyz"
 *   iter_count = 1 << itoa64_index(salt[3])  (typically 7..30 -> 128..2^30)
 *
 *   state = MD5(salt[4..11] || pass)        -- 8-byte salt + password
 *   for (i = 0; i < iter_count; i++):
 *     state = MD5(state[16] || pass)        -- 16-byte digest + password
 *
 *   probe state once at the end (matches CPU semantics at mdxfind.c:13620
 *   which calls checkhashbb(curin, 32, s1, job) AFTER the for-loop).
 *
 * DESIGN: iter loop INSIDE template_finalize, max_iter=1.
 * --------------------------------------------------------
 * Mirrors SHA1DRU pattern (B6.11 precedent):
 *   - the iter count is INTERNAL to the algorithm (decoded from salt
 *     byte 3), NOT user-controlled via -i;
 *   - only the FINAL state is probed (CPU semantics); putting the
 *     loop in template_iterate would either waste a probe per iter
 *     OR require max_iter = iter_count (which varies per salt!), which
 *     the host iter-budget infrastructure cannot express;
 *   - host forces params.max_iter = 1 at the rules-engine pack site
 *     so the kernel's outer iter loop runs exactly once and never
 *     calls template_iterate (which is a stub).
 *
 * Salt-axis carrier: this kernel routes through the salted-template
 * scaffolding (GPU_TEMPLATE_HAS_SALT=1, SALT_POSITION=PREPEND in
 * defines_str). The salt buffer carries the FULL 12-byte "$H$<cost>
 * <8-byte salt>" prefix (mdxfind.c:40572 stores 12 bytes via
 * store_typesalt(JOB_PHPBB3, line, 12)). gpu_pack_salts is called with
 * use_hashsalt=0 (no hashsalt synthesis), so salt_buf+salt_off[i] points
 * at the raw 12-byte salt prefix and salt_lens[i] is 12.
 *
 * Inside the kernel:
 *   - salt_bytes[3]    -> cost char (phpitoa64 lookup -> log2 iter count)
 *   - salt_bytes[4..11] -> 8-byte salt for step 1
 *   - salt_bytes[0..2]  -> "$H$" prefix (unused inside the kernel; the
 *                          host already validated the format at salt-load
 *                          time via mdxfind.c:40559).
 *
 * Validation oracle: gpu_phpbb3.cl phpbb3_batch (slab kernel; this file
 * is a port of that body into the template extension API). Differences:
 *   - reads pass from `data` (post-rule buf passed by template; PRIVATE
 *     uchar *) rather than from a packed wordbuf indexed by tid (slab's
 *     `__global const uchar *pass`);
 *   - reads salt from `salt_bytes` (the global salt buffer threaded by
 *     gpu_template.cl under GPU_TEMPLATE_HAS_SALT) rather than from
 *     the slab's own salts/salt_offsets/salt_lens trio.
 *
 * The slab oracle uses M_copy_bytes (gpu_common.cl:761) which takes a
 * __global pointer; data is __private here, so we use the inline
 * byte-pack pattern (M[i>>2] |= v << ((i & 3) * 8)) instead. This is
 * the same pattern md5salt_core / md5saltpass_core use for the private
 * `data` source. Correctness invariant: identical M[] state to the
 * slab kernel's M_copy_bytes(M, off, pass, plen) call -- byte-for-byte.
 *
 * Hit replay: host calls checkhashbb(curin, 32, salt_bytes, job) where
 * salt_bytes is the FULL 12-byte "$H$<cost><8>" prefix (matches CPU
 * semantics at mdxfind.c:13620 which passes saltsnap[si].salt = the
 * full 12-byte prefix).
 *
 * R1 mitigation (AMD ROCm comgr addrspace fragility): single private
 * buffer pattern. M[16] uint32 working block + 4-uint chaining state
 * are private; salt_bytes is __global (read byte-by-byte into M[] via
 * the same inline byte-pack pattern that md5salt_core uses for both
 * private and global sources -- no addrspace casts).
 *
 * R2 (register pressure): one M[16] scratch + 4-uint state + a small
 * integer for the iter count. Comparable to MD5SALT base body minus
 * the inner-MD5 hex staging -- expected priv_mem on Pascal in the
 * 41-43 KB band shared by all unified-template dispatches.
 *
 * Source order at compile time:
 *   [ gpu_common_str, gpu_md5_rules_str, gpu_phpbb3_core_str,
 *     gpu_template_str ]
 *
 * Cache key (R3): defines_str =
 *   "HASH_WORDS=4,HASH_BLOCK_BYTES=64,HAS_SALT=1,SALT_POSITION=PREPEND,
 *    BASE_ALGO=phpbb3"
 *
 * Cache-key disambiguation:
 *   - From MD5SALT family (HASH_WORDS=4 + HAS_SALT=1 + BASE_ALGO=md5)
 *     by BASE_ALGO=phpbb3 axis.
 *   - From every other salted/unsalted template via the unique
 *     BASE_ALGO=phpbb3 token.
 */

#ifndef HASH_WORDS
#define HASH_WORDS 4
#endif
#ifndef HASH_BLOCK_BYTES
#define HASH_BLOCK_BYTES 64
#endif

/* phpitoa64 reverse-lookup table for decoding the cost character from
 * salt byte 3. Mirrors the slab kernel's phpitoa64_k constant
 * (gpu_phpbb3.cl:7). MD5-LE direct (no bswap32) -- matches the slab
 * kernel and CPU's mymd5() output convention. */
__constant uchar phpbb3_phpitoa64[] =
    "./0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";

/* Per-lane state struct: MD5 carries 4 uint32 chaining values (LE). */
typedef struct {
    uint h[HASH_WORDS];
} template_state;

/* template_init: install MD5 IV into state. Same IV as gpu_md5_core.cl. */
static inline void template_init(template_state *st) {
    st->h[0] = 0x67452301u;
    st->h[1] = 0xEFCDAB89u;
    st->h[2] = 0x98BADCFEu;
    st->h[3] = 0x10325476u;
}

/* template_transform: stub for interface symmetry. PHPBB3's
 * template_finalize builds M[] in-place per block -- never routes
 * through this. */
static inline void template_transform(template_state *st,
                                      const uchar *block)
{
    uint M[16];
    for (int j = 0; j < 16; j++) {
        int b = j * 4;
        M[j] = ((uint)block[b])
             | ((uint)block[b + 1] << 8)
             | ((uint)block[b + 2] << 16)
             | ((uint)block[b + 3] << 24);
    }
    md5_block(&st->h[0], &st->h[1], &st->h[2], &st->h[3], M);
}

/* template_finalize: full PHPBB3 chain.
 *
 * Step 1: state = MD5(salt8 || pass)
 *   - salt8 = salt_bytes[4..11] (8 bytes, packed into M[0..1])
 *   - pass  = data[0..plen-1]
 *   - total = 8 + plen
 *
 * Step 2: for each iter, state = MD5(state[16] || pass)
 *   - M[0..3] receive the previous state's 16 binary bytes (st->h[]).
 *   - M[4..] receive pass + 0x80 padding + length, computed ONCE outside
 *     the loop (pass + length doesn't change across iterations).
 *
 * Password length cap: matches the slab kernel's "pass max 39 bytes"
 * comment (gpu_phpbb3.cl:6). With plen <= 39, total = 16 + 39 = 55 bytes
 * fits in a single MD5 block (55 + 1 padding + 8 length = 64). The
 * chokepoint admit at mdxfind.c gates job->clen <= 39 so this path
 * is only entered for valid passwords. Defensive cap below clamps to
 * 39 on entry.
 *
 * algo_mode: PHPBB3 has only one mode. The arg is unused; kept for
 * interface symmetry with the salted-template signature. */
static inline void template_finalize(template_state *st,
                                     const uchar *data, int len
#ifdef GPU_TEMPLATE_HAS_SALT
                                     , __global const uchar *salt_bytes
                                     , uint salt_len
                                     , uint algo_mode
#endif
                                     )
{
#ifdef GPU_TEMPLATE_HAS_SALT
    (void)algo_mode;
    (void)salt_len;

    /* Defensive: PHPBB3 caps password length at 39 bytes (CPU at
     * mdxfind.c:13450 gates job->clen <= 39 before GPU dispatch).
     * Clamp here too so a malformed call doesn't produce out-of-bounds
     * writes to M[]. */
    int plen = len;
    if (plen > 39) plen = 39;

    /* Decode iter count from salt_bytes[3] via phpitoa64 reverse lookup.
     * Mirrors slab kernel gpu_phpbb3.cl lines 31-34 byte-for-byte. */
    uchar ic = salt_bytes[3];
    int log2count = 0;
    for (int k = 0; k < 64; k++) {
        if (phpbb3_phpitoa64[k] == ic) { log2count = k; break; }
    }
    uint count = 1u << log2count;

    /* Step 1: MD5(salt[4..11] || pass) -- single MD5 block (8 + 39 + 1
     * padding + 8 length = 56 max, well under 64).
     *
     * Build M[16] in LE byte order via the inline pack pattern (matches
     * md5salt_core's mode-0 inner-MD5 streaming loop). */
    uint M[16];
    for (int i = 0; i < 16; i++) M[i] = 0;

    /* Pack salt_bytes[4..11] (8 bytes) into M[0..1] little-endian. */
    for (int i = 0; i < 8; i++) {
        uint v = (uint)salt_bytes[4 + i];
        M[i >> 2] |= v << ((i & 3) * 8);
    }
    /* Pack pass[0..plen-1] into M starting at byte offset 8. */
    for (int i = 0; i < plen; i++) {
        int pos = 8 + i;
        uint v = (uint)data[i];
        M[pos >> 2] |= v << ((pos & 3) * 8);
    }
    int total = 8 + plen;
    /* MD5 padding marker. */
    M[total >> 2] |= (uint)0x80u << ((total & 3) * 8);
    /* Length in bits, little-endian uint32 at M[14] (low 32 bits suffice
     * for total <= 55 bytes = 440 bits). */
    M[14] = (uint)total * 8u;

    /* Reset state to MD5 IV before absorbing block 1. */
    uint hx = 0x67452301u;
    uint hy = 0xEFCDAB89u;
    uint hz = 0x98BADCFEu;
    uint hw = 0x10325476u;
    md5_block(&hx, &hy, &hz, &hw, M);

    /* Step 2: N iterations of MD5(state[16] || pass).
     *
     * Pre-pack M[] with pass + padding + length. M[0..3] will be
     * overwritten with the chaining state per iteration; M[4..] stays
     * constant across iterations.
     *
     * total = 16 + plen (max 16 + 39 = 55 -- single MD5 block). */
    total = 16 + plen;
    for (int i = 0; i < 16; i++) M[i] = 0;
    /* Pass starts at byte 16 (after the 16-byte digest slot). */
    for (int i = 0; i < plen; i++) {
        int pos = 16 + i;
        uint v = (uint)data[i];
        M[pos >> 2] |= v << ((pos & 3) * 8);
    }
    /* MD5 padding marker at byte (16 + plen). */
    M[total >> 2] |= (uint)0x80u << ((total & 3) * 8);
    /* Length in bits at M[14] (LE uint32). */
    M[14] = (uint)total * 8u;

    /* Iteration loop: only M[0..3] change per iter; the digest absorbs
     * its own previous output. Mirrors slab kernel gpu_phpbb3.cl lines
     * 55-59 byte-for-byte. */
    for (uint ic2 = 0; ic2 < count; ic2++) {
        M[0] = hx;
        M[1] = hy;
        M[2] = hz;
        M[3] = hw;
        /* Reset state to MD5 IV for each iter (the digest of the prior
         * iter is already loaded into M[0..3] via the four lines above). */
        hx = 0x67452301u;
        hy = 0xEFCDAB89u;
        hz = 0x98BADCFEu;
        hw = 0x10325476u;
        md5_block(&hx, &hy, &hz, &hw, M);
    }

    /* Install final state into template_state for digest compare + emit. */
    st->h[0] = hx;
    st->h[1] = hy;
    st->h[2] = hz;
    st->h[3] = hw;
    return;
#else
    /* Defensive fall-through for the !HAS_SALT instantiation. PHPBB3 is
     * always a salted op (the salt carries the iteration count); a no-
     * salt build would have nothing to do. Compute MD5(data) so the
     * extension interface stays well-formed. */
    uint M[16];
    int pos = 0;
    while ((len - pos) >= HASH_BLOCK_BYTES) {
        for (int j = 0; j < 16; j++) {
            int b = pos + j * 4;
            M[j] = ((uint)data[b])
                 | ((uint)data[b + 1] << 8)
                 | ((uint)data[b + 2] << 16)
                 | ((uint)data[b + 3] << 24);
        }
        md5_block(&st->h[0], &st->h[1], &st->h[2], &st->h[3], M);
        pos += HASH_BLOCK_BYTES;
    }
    int rem = len - pos;
    for (int j = 0; j < 16; j++) M[j] = 0;
    for (int i = 0; i < rem; i++) {
        int wi = i >> 2;
        int bi = (i & 3) * 8;
        M[wi] |= ((uint)data[pos + i]) << bi;
    }
    {
        int wi = rem >> 2;
        int bi = (rem & 3) * 8;
        M[wi] |= ((uint)0x80u) << bi;
    }
    if (rem < 56) {
        M[14] = (uint)((uint)len * 8u);
        md5_block(&st->h[0], &st->h[1], &st->h[2], &st->h[3], M);
    } else {
        md5_block(&st->h[0], &st->h[1], &st->h[2], &st->h[3], M);
        for (int j = 0; j < 16; j++) M[j] = 0;
        M[14] = (uint)((uint)len * 8u);
        md5_block(&st->h[0], &st->h[1], &st->h[2], &st->h[3], M);
    }
#endif
}

/* template_iterate: STUB. With max_iter = 1 (host-set for PHPBB3), the
 * outer iter loop in template_phase0 runs exactly once and never calls
 * template_iterate. Mirrors SHA1DRU pattern (gpu_sha1dru_core.cl
 * lines 313-317). */
static inline void template_iterate(template_state *st)
{
    (void)st;
}

/* template_digest_compare: probe the compact table with the 4 MD5 LE
 * digest words. Identical to gpu_md5_core.cl's template_digest_compare
 * (PHPBB3's final state is in MD5 native LE byte order). */
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

/* template_emit_hit: emit a hit. PHPBB3 = 4 LE uint32 digest words
 * (same wire format as MD5). */
#define template_emit_hit(hits, hit_count, max_hits, st, widx, sidx, iter) \
    EMIT_HIT_4((hits), (hit_count), (max_hits), \
               (widx), (sidx), (iter), \
               (st)->h[0], (st)->h[1], (st)->h[2], (st)->h[3])

/* B3 dedup+overflow-aware variant. */
#define template_emit_hit_or_overflow(hits, hit_count, max_hits, st, widx, sidx, iter, \
                                      hashes_shown, matched_idx, dedup_mask, \
                                      ovr_set, ovr_gid, lane_gid) \
    EMIT_HIT_4_DEDUP_OR_OVERFLOW((hits), (hit_count), (max_hits), \
               (widx), (sidx), (iter), \
               (st)->h[0], (st)->h[1], (st)->h[2], (st)->h[3], \
               (hashes_shown), (matched_idx), (dedup_mask), \
               (ovr_set), (ovr_gid), (lane_gid))
