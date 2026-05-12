/* gpu_md5crypt_core.cl -- MD5CRYPT (e511) algorithm extension functions
 * for the generic dispatch template (Memo B Phase Unix-crypt 1
 * MD5CRYPT Path A hand-written sibling of the unsalted-template family).
 *
 * $Revision: 1.1 $
 * $Log: gpu_md5crypt_core.cl,v $
 * Revision 1.1  2026/05/08 13:55:39  dlr
 * Initial
 *
 *
 * STATUS: MD5CRYPT fan-out 2026-05-08 -- second iterated-crypt salted
 * algorithm after PHPBB3 on the unified template path. Single algo_mode
 * (0); no KPASS/KSALT siblings -- JOB_MD5CRYPT (e511) is a single-mode
 * algorithm with a FIXED 1000-iteration count (vs PHPBB3's salt-carried
 * variable count). First Phase 1 of the Unix-crypt ladder
 * (MD5CRYPT -> SHA256CRYPT -> SHA512CRYPT -> SHA512CRYPTMD5).
 *
 * MD5CRYPT / BSD $1$ md5crypt semantics (mirrors mdxfind.c JOB_MD5CRYPT
 * at lines 13017-13117 + slab oracle gpu_md5crypt.cl):
 *
 *   salt buffer (4..12 bytes total): "$1$<salt>$" -- variable-length
 *   raw_salt = salt_bytes[3..3+saltlen-1] where saltlen <= 8
 *
 *   STEP 1: digest_b = MD5(pass || raw_salt || pass)
 *   STEP 2: digest_a = MD5(pass || "$1$" || raw_salt
 *                          || digest_b chunked-by-16 for plen bytes
 *                          || bit_byte: pass[0] if (len bit) else 0,
 *                                       repeating until len reaches 0)
 *   STEP 3: 1000 iterations
 *           for x in 0..999:
 *             buf = (x&1) ? pass : prev_state[16]
 *             if (x % 3) buf ||= raw_salt
 *             if (x % 7) buf ||= pass
 *             buf ||= (x&1) ? prev_state[16] : pass
 *             state = MD5(buf)
 *
 *   probe state once at the end (matches CPU semantics at mdxfind.c:13071
 *   which calls hybrid_check(curin.h, 16, ...) AFTER the for-loop).
 *
 * DESIGN: iter loop INSIDE template_finalize, max_iter=1.
 * --------------------------------------------------------
 * Mirrors PHPBB3 + SHA1DRU pattern (B6.11 precedent):
 *   - iteration count is INTERNAL to the algorithm (FIXED 1000 here,
 *     vs salt-carried in PHPBB3), NOT user-controlled via -i;
 *   - only the FINAL state is probed (CPU semantics);
 *   - host forces params.max_iter = 1 at the rules-engine pack site
 *     so the kernel's outer iter loop runs exactly once and never
 *     calls template_iterate (which is a stub).
 *
 * Salt-axis carrier: this kernel routes through the salted-template
 * scaffolding (GPU_TEMPLATE_HAS_SALT=1, SALT_POSITION=PREPEND in
 * defines_str). The salt buffer carries the FULL "$1$<salt>$" prefix
 * (mdxfind.c:40635 stores prefixlen bytes via store_typesalt(JOB_-
 * MD5CRYPT, line, prefixlen)). gpu_pack_salts is called with
 * use_hashsalt=0, so salt_buf+salt_off[i] points at the raw "$1$<salt>$"
 * prefix and salt_lens[i] is variable (5..12 bytes).
 *
 * Inside the kernel:
 *   - salt_bytes[0..2]  -> "$1$" prefix (used in step 2 verbatim)
 *   - salt_bytes[3..]   -> raw salt (terminated by '$' or end of buffer)
 *
 * Validation oracle: gpu_md5crypt.cl md5crypt_batch (slab kernel; this
 * file is a port of that body into the template extension API).
 * Differences:
 *   - reads pass from `data` (post-rule buf passed by template; PRIVATE
 *     uchar *) rather than from a packed wordbuf indexed by tid (slab's
 *     `__global const uchar *pass`);
 *   - reads salt from `salt_bytes` (the global salt buffer threaded by
 *     gpu_template.cl under GPU_TEMPLATE_HAS_SALT) rather than from
 *     the slab's own salts/salt_offsets/salt_lens trio.
 *
 * The slab oracle uses uchar buf[256] + md5_oneshot helper for variable-
 * length inputs that span multiple MD5 blocks. We mirror that approach
 * here -- the worst-case input is len + saltlen + len = 39 + 8 + 39 = 86
 * bytes for step 1, and similar for step 2/iter steps, so multi-block
 * MD5 absorption is required (a single MD5 block holds 55 message bytes
 * before length-encoding kicks in).
 *
 * Hit replay: host calls hybrid_check(curin.h, 16, ...) and reconstructs
 * "$1$<salt>$<22-char-phpitoa64>" via md5crypt_b64encode (existing
 * helper at gpujob_opencl.c). Mirrors the slab arm at
 * gpujob_opencl.c:1723 byte-for-byte.
 *
 * R1 mitigation (AMD ROCm comgr addrspace fragility): single private
 * buffer pattern. uchar buf[128] working buffer + 4-uint chaining state
 * are private; salt_bytes is __global (read byte-by-byte into buf via
 * inline byte loops -- no addrspace casts).
 *
 * R2 (register pressure): one uchar buf[128] + 4-uint state + a small
 * iter counter. Comparable to PHPBB3 plus the longer working buffer
 * (PHPBB3's two-block-cap kernel doesn't need a uchar buffer). Expected
 * priv_mem on Pascal in the 41-44 KB band shared by other unified-
 * template dispatches.
 *
 * Source order at compile time:
 *   [ gpu_common_str, gpu_md5_rules_str, gpu_md5crypt_core_str,
 *     gpu_template_str ]
 *
 * Cache key (R3): defines_str =
 *   "HASH_WORDS=4,HASH_BLOCK_BYTES=64,HAS_SALT=1,SALT_POSITION=PREPEND,
 *    BASE_ALGO=md5crypt"
 *
 * Cache-key disambiguation:
 *   - From MD5SALT family (HASH_WORDS=4 + HAS_SALT=1 + BASE_ALGO=md5)
 *     by BASE_ALGO=md5crypt axis.
 *   - From PHPBB3 (HASH_WORDS=4 + HAS_SALT=1 + BASE_ALGO=phpbb3)
 *     by BASE_ALGO=md5crypt axis.
 *   - From every other salted/unsalted template via the unique
 *     BASE_ALGO=md5crypt token.
 */

#ifndef HASH_WORDS
#define HASH_WORDS 4
#endif
#ifndef HASH_BLOCK_BYTES
#define HASH_BLOCK_BYTES 64
#endif

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

/* template_transform: stub for interface symmetry. MD5CRYPT's
 * template_finalize manages multi-block absorption inline -- never
 * routes through this. Provided for completeness (matches PHPBB3
 * pattern). */
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

/* md5crypt_oneshot: compute MD5(buf[0..buflen-1]) and write 4-uint LE
 * digest to out[0..3]. Handles variable-length input up to 119 bytes
 * (max two-block absorb plus length encoding; padding consumes 1 byte
 * marker plus 8 bytes length, leaving 55 + 64 = 119 bytes max).
 *
 * Mirrors slab gpu_md5crypt.cl:md5_oneshot byte-for-byte (uchar pad[]
 * staging + LE M[] pack + md5_block per absorbed block).
 *
 * Worst-case input lengths in MD5CRYPT:
 *   step 1 (pass+salt+pass):       39 + 8 + 39 = 86 bytes (2 blocks)
 *   step 2 (pass+"$1$"+salt+...): 39 + 3 + 8 + 39 + ~32 (bit bytes) = ~121 cap
 *                                  But bit_bytes = log2(plen)+1 so for
 *                                  plen <= 39 that's at most 6 extra bytes.
 *                                  Actual worst-case: 39+3+8+
 *                                  ceil(39/16)*16 + 6 = 39+3+8+48+6 = 104.
 *   step 3 iter (pass+salt+pass+pass): 39+8+39+39 = 125 bytes (3 blocks?)
 *                                  Actually: max iteration buf is pass+salt+
 *                                  pass+digest_or_pass = 39+8+39+39 = 125,
 *                                  OR digest+salt+pass+digest = 16+8+39+16 = 79.
 *
 * For a 3-block input the absorb requires absorbing two 64-byte blocks
 * before the final padded block. Max input we handle: 191 bytes
 * (3 absorbs, padding fits in 64-1-8 = 55 of the third block).
 *
 * Defensive: cap buflen at 191 on entry (kernel must not pass longer).
 */
static inline void md5crypt_oneshot(const uchar *buf, int buflen, uint *out)
{
    uint M[16];
    out[0] = 0x67452301u;
    out[1] = 0xEFCDAB89u;
    out[2] = 0x98BADCFEu;
    out[3] = 0x10325476u;

    if (buflen > 191) buflen = 191;

    int pos = 0;
    /* Process complete 64-byte blocks. */
    while (pos + 64 <= buflen) {
        for (int i = 0; i < 16; i++) {
            int b = pos + i * 4;
            M[i] = ((uint)buf[b])
                 | ((uint)buf[b + 1] << 8)
                 | ((uint)buf[b + 2] << 16)
                 | ((uint)buf[b + 3] << 24);
        }
        md5_block(&out[0], &out[1], &out[2], &out[3], M);
        pos += 64;
    }

    /* Final block(s): build a uchar pad[128] tail with the remaining
     * bytes + 0x80 marker + zeros + length. May span 1 or 2 blocks
     * depending on rem. */
    uchar pad[128];
    for (int i = 0; i < 128; i++) pad[i] = 0;
    int rem = buflen - pos;
    for (int i = 0; i < rem; i++) pad[i] = buf[pos + i];
    pad[rem] = 0x80u;

    int blocks = (rem < 56) ? 1 : 2;
    int lenoff = (blocks == 1) ? 56 : 120;
    /* Length in bits (4-byte LE; high 4 bytes stay zero -- buflen <=
     * 191 fits in 32 bits). */
    pad[lenoff]     = (uchar)((buflen * 8) & 0xff);
    pad[lenoff + 1] = (uchar)(((buflen * 8) >> 8) & 0xff);
    pad[lenoff + 2] = (uchar)(((buflen * 8) >> 16) & 0xff);
    pad[lenoff + 3] = (uchar)(((buflen * 8) >> 24) & 0xff);

    for (int b = 0; b < blocks; b++) {
        for (int i = 0; i < 16; i++) {
            int x = b * 64 + i * 4;
            M[i] = ((uint)pad[x])
                 | ((uint)pad[x + 1] << 8)
                 | ((uint)pad[x + 2] << 16)
                 | ((uint)pad[x + 3] << 24);
        }
        md5_block(&out[0], &out[1], &out[2], &out[3], M);
    }
}

/* template_finalize: full MD5CRYPT chain (3 steps + 1000-iter loop).
 *
 * Step 1: digest_b = MD5(pass || raw_salt || pass)
 * Step 2: digest_a = MD5(pass || "$1$" || raw_salt
 *                        || digest_b-chunked-by-16-for-plen-bytes
 *                        || bit-loop-bytes)
 * Step 3: 1000 iterations of variable-shape MD5(buf) per CPU at
 *         mdxfind.c:13057-13066.
 *
 * Final state in st->h[0..3] (4 LE uint32 = 16 binary bytes), probed
 * by template_digest_compare.
 *
 * Password length cap: matches CPU's handling -- typical mdxfind
 * passwords are <= 39 bytes by default (see gpu_maxlen_val derivation).
 * Defensive cap below clamps plen to 39 so the working buffer never
 * exceeds the md5crypt_oneshot 191-byte cap.
 *
 * algo_mode: MD5CRYPT has only one mode. The arg is unused; kept for
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

    /* Defensive: clamp plen at 39 (mdxfind default max) so buffer
     * lengths stay within md5crypt_oneshot's 191-byte cap. */
    int plen = len;
    if (plen > 39) plen = 39;

    /* Extract raw salt: salt_bytes points at "$1$<salt>$"; raw salt
     * starts at byte 3, ends at next '$' or salt_len. Cap at 8.
     * Mirrors slab gpu_md5crypt.cl:64-67 and CPU mdxfind.c:13039-13042. */
    int saltlen = 0;
    if (salt_len >= 4) {
        int max_scan = (int)salt_len - 3;
        if (max_scan > 8) max_scan = 8;
        for (int i = 0; i < max_scan; i++) {
            uchar c = salt_bytes[3 + i];
            if (c == 0 || c == '$') break;
            saltlen++;
        }
    }
    if (saltlen > 8) saltlen = 8;

    /* Working buffer: max input 125 bytes (step 3 iter worst case);
     * round to 192 to leave headroom for the md5crypt_oneshot cap. */
    uchar buf[192];

    /* === STEP 1: digest_b = MD5(pass || raw_salt || pass) === */
    int blen = 0;
    for (int i = 0; i < plen; i++)    buf[blen++] = data[i];
    for (int i = 0; i < saltlen; i++) buf[blen++] = salt_bytes[3 + i];
    for (int i = 0; i < plen; i++)    buf[blen++] = data[i];

    uint digest[4];
    md5crypt_oneshot(buf, blen, digest);

    /* digest_b: 16 binary bytes of step-1 digest in LE byte order. */
    uchar digest_b[16];
    for (int i = 0; i < 4; i++) {
        digest_b[i * 4]     = (uchar)(digest[i] & 0xffu);
        digest_b[i * 4 + 1] = (uchar)((digest[i] >> 8) & 0xffu);
        digest_b[i * 4 + 2] = (uchar)((digest[i] >> 16) & 0xffu);
        digest_b[i * 4 + 3] = (uchar)((digest[i] >> 24) & 0xffu);
    }

    /* === STEP 2: digest_a = MD5(pass || "$1$" || raw_salt
     *                            || digest_b-chunked-by-16-for-plen-bytes
     *                            || bit-loop-bytes) === */
    blen = 0;
    for (int i = 0; i < plen; i++) buf[blen++] = data[i];
    buf[blen++] = '$';
    buf[blen++] = '1';
    buf[blen++] = '$';
    for (int i = 0; i < saltlen; i++) buf[blen++] = salt_bytes[3 + i];
    /* Append digest_b in 16-byte chunks until plen bytes have been
     * appended. Mirrors CPU mdxfind.c:13050-13053 + slab :92-96. */
    for (int x = plen; x > 0; x -= 16) {
        int n = (x > 16) ? 16 : x;
        for (int i = 0; i < n; i++) buf[blen++] = digest_b[i];
    }
    /* Bit-dependent bytes: for each 1 bit in plen (from LSB), append
     * NUL; for each 0 bit, append pass[0]. Mirrors CPU mdxfind.c:13054-
     * 13055 + slab :98-99. plen <= 39 means at most 6 iterations. */
    for (int x = plen; x != 0; x >>= 1) {
        buf[blen++] = (x & 1) ? (uchar)0u : (uchar)data[0];
    }
    md5crypt_oneshot(buf, blen, digest);

    /* dig: 16 binary bytes of step-2 digest in LE byte order. Updated
     * each iter in the 1000-iter loop. */
    uchar dig[16];
    for (int i = 0; i < 4; i++) {
        dig[i * 4]     = (uchar)(digest[i] & 0xffu);
        dig[i * 4 + 1] = (uchar)((digest[i] >> 8) & 0xffu);
        dig[i * 4 + 2] = (uchar)((digest[i] >> 16) & 0xffu);
        dig[i * 4 + 3] = (uchar)((digest[i] >> 24) & 0xffu);
    }

    /* === STEP 3: 1000 iterations === */
    for (int x = 0; x < 1000; x++) {
        blen = 0;
        if (x & 1) {
            for (int i = 0; i < plen; i++) buf[blen++] = data[i];
        } else {
            for (int i = 0; i < 16; i++)   buf[blen++] = dig[i];
        }
        if (x % 3) {
            for (int i = 0; i < saltlen; i++) buf[blen++] = salt_bytes[3 + i];
        }
        if (x % 7) {
            for (int i = 0; i < plen; i++) buf[blen++] = data[i];
        }
        if (x & 1) {
            for (int i = 0; i < 16; i++)   buf[blen++] = dig[i];
        } else {
            for (int i = 0; i < plen; i++) buf[blen++] = data[i];
        }
        md5crypt_oneshot(buf, blen, digest);
        for (int i = 0; i < 4; i++) {
            dig[i * 4]     = (uchar)(digest[i] & 0xffu);
            dig[i * 4 + 1] = (uchar)((digest[i] >> 8) & 0xffu);
            dig[i * 4 + 2] = (uchar)((digest[i] >> 16) & 0xffu);
            dig[i * 4 + 3] = (uchar)((digest[i] >> 24) & 0xffu);
        }
    }

    /* Install final state into template_state for digest compare + emit. */
    st->h[0] = digest[0];
    st->h[1] = digest[1];
    st->h[2] = digest[2];
    st->h[3] = digest[3];
    return;
#else
    /* Defensive fall-through for the !HAS_SALT instantiation. MD5CRYPT
     * is always a salted op (the algorithm requires the salt for steps
     * 1+2+3); a no-salt build would have nothing to do. Compute MD5(data)
     * so the extension interface stays well-formed. */
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

/* template_iterate: STUB. With max_iter = 1 (host-set for MD5CRYPT), the
 * outer iter loop in template_phase0 runs exactly once and never calls
 * template_iterate. Mirrors PHPBB3 / SHA1DRU pattern. */
static inline void template_iterate(template_state *st)
{
    (void)st;
}

/* template_digest_compare: probe the compact table with the 4 MD5 LE
 * digest words. Identical to gpu_md5_core.cl's template_digest_compare
 * (MD5CRYPT's final state is in MD5 native LE byte order). */
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

/* template_emit_hit: emit a hit. MD5CRYPT = 4 LE uint32 digest words
 * (same wire format as MD5; host's hit-replay arm reconstructs the
 * "$1$<salt>$<22>" string via md5crypt_b64encode). */
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
