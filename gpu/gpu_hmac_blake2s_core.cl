/*
 * $Revision: 1.3 $
 * $Log: gpu_hmac_blake2s_core.cl,v $
 * Revision 1.3  2026/05/11 05:22:01  dlr
 * Backfill $Revision/$Log RCS keyword stanzas per feedback_rcs_keyword_stanzas.md. Passive 4-line comment block at top of file; no behavioral change. Hand-authored .cl file was missing required stanzas (per memory: all hand-authored .c/.h/.cl/.frag/.tmpl/.py/.sh files MUST contain $Revision/$Log keyword stanzas). Build green on .205 against the post-add files; OpenCL compile strips comments so no kernel behavior change.
 *
 */
/* gpu_hmac_blake2s_core.cl — HMAC-BLAKE2S algorithm core for the unified
 * dispatch template. Path A (hand-written) sibling of the codegen-emitted
 * salted cores.
 *
 * Family I HMAC-BLAKE2S carrier (2026-05-08): JOB_HMAC_BLAKE2S (e828) shares
 * a salted-template kernel slot. There is NO production JOB_BLAKE2SSALTPASS
 * algorithm; mode-0 of this kernel is structurally unreachable. Single
 * algo_mode (5) — there is no KPASS sibling op for HMAC-BLAKE2S in mdxfind.
 *
 * Hand-written rather than codegen because the BLAKE2 family carries
 * counter+flag in the per-algo state struct (RFC 7693) and uses a 4-arg
 * b2s_compress(uint *h, const uchar *block, ulong counter, int last) primitive
 * unlike the SHA / RIPEMD families that codegen targets. The codegen tool's
 * fragment library targets MD-style "absorb 64-byte block + length-pad-tail"
 * shapes; BLAKE2's counter-driven finalization doesn't fit that mold without
 * a dedicated fragment family. Path A authoring mirrors the unsalted
 * gpu_blake2s256_core.cl and inserts the HMAC body inside template_finalize
 * gated on algo_mode == 5u.
 *
 * HMAC-BLAKE2S semantics (mirrors mdxfind.c JOB_HMAC_BLAKE2S at line 30341
 * + slab oracle gpu_hmac_blake2s.cl):
 *   key = $pass (the candidate word being tested)
 *   msg = $salt (the per-hash salt)
 *   if (klen > 64) K = BLAKE2S(pass)            ; 32 bytes, zero-padded to 64
 *   else            K = pass                    ; zero-padded to 64 bytes
 *   inner = BLAKE2S((K XOR ipad) || msg)        ; 32-byte digest
 *   outer = BLAKE2S((K XOR opad) || inner)      ; 32-byte digest = HMAC output
 *
 * Hit-replay: host calls checkhashsalt(curin, 64, salt_bytes, salt_len, 0, job)
 * (iter=0 sentinel suppresses xNN suffix; mirrors KPASS hit-replay arms in
 * Families G/H). Output label is "HMAC-BLAKE2S" (no -KPASS suffix because the
 * op is named JOB_HMAC_BLAKE2S, not JOB_HMAC_BLAKE2S_KPASS — the algorithm is
 * KPASS-shape but mdxfind never named a KSALT sibling for this op).
 *
 * BLAKE2S-256 geometry (carried over from gpu_blake2s256_core.cl):
 *   - 64-byte block, 8 x uint32 chaining state, LITTLE-ENDIAN
 *   - 10-round G-function compression (b2s_compress in gpu_common.cl)
 *   - parameter block init: IV[0] ^ 0x01010020
 *     (digest_length=32 << 0 | key_length=0 << 8 | fanout=1 << 16 |
 *      depth=1 << 24)
 *   - 32-byte digest = 8 uint32 LE
 *
 *   HASH_WORDS         — digest size in 32-bit words (8 for BLAKE2S-256)
 *   HASH_BLOCK_BYTES   — compress-block size (64)
 *   template_state     — h[8] digest + t[2] counter + f[2] flag
 *   template_init      — install IV ^ parameter_block; zero counter+flag
 *   template_transform — absorb one 64-byte block (NON-FINAL)
 *   template_finalize  — branches on algo_mode:
 *                          == 5u  : HMAC body (early return after writing
 *                                   the 32-byte HMAC digest into st->h[])
 *                          else   : plain BLAKE2S(data) main body (mirror of
 *                                   gpu_blake2s256_core.cl; structurally
 *                                   unreachable in production but kept for
 *                                   correctness if any future op routes here
 *                                   in mode 0)
 *   template_iterate        — re-hash 64-char hex of digest (single-block)
 *   template_digest_compare — probe leading 16 bytes (h[0..3]) LE-direct
 *   template_emit_hit       — EMIT_HIT_8 (8 uint32 LE = 32 bytes)
 *
 * Source order at compile time (via gpu_opencl_template_compile_hmac_blake2s):
 *   [ gpu_common_str, gpu_md5_rules_str, gpu_hmac_blake2s_core_str,
 *     gpu_template_str ]
 *
 * Defines string for cache key:
 *   "HASH_WORDS=8,HASH_BLOCK_BYTES=64,HAS_SALT=1,SALT_POSITION=PREPEND,
 *    BASE_ALGO=blake2s,HMAC_KPASS=1"
 *
 * Cache-key disambiguation from BLAKE2S256 unsalted core
 * (gpu_blake2s256_core_str): HAS_SALT=1 + SALT_POSITION=PREPEND axes are
 * absent in the unsalted defines string. From every other salted-template
 * core (sha1saltpass, rmd160saltpass, etc.): BASE_ALGO=blake2s axis is
 * unique. From any potential future BLAKE2SSALTPASS unsalted-msg variant:
 * HMAC_KPASS=1 axis distinguishes (no other salted core uses HMAC_KPASS=1).
 *
 * Bytecast invariants (carried over):
 *   - Final st->h[0..7] is LITTLE-ENDIAN per uint32 — direct probe + emit,
 *     no bswap32.
 *   - Wire format: 8 LE uint32 words written via EMIT_HIT_8.
 *   - Probe key uses st->h[0..3] (leading 16 bytes LE).
 *
 * R1 mitigation (AMD ROCm comgr addrspace fragility): single-private-buffer
 * pattern. State struct is private; b2s_compress takes __private uchar*
 * (existing primitive in gpu_common.cl). HMAC body uses three 64-byte
 * private uchar buffers (key_block / ipad_block / opad_block) plus a small
 * inner-digest staging buffer — all __private, no addrspace casts.
 *
 * R2 (register pressure): HMAC body adds key_block[64] + ipad_block[64] +
 * opad_block[64] + scratch + temporary state[8] = ~256 B over the unsalted
 * BLAKE2S finalize. Comparable to other HMAC carrier kernels (Family G:
 * RMD160, Family H: RMD320). Expected priv_mem on gfx1201 within the
 * documented 41-43 KB band for B5 sub-batch siblings.
 */

/* Per-algorithm geometry. Cache key (R3 fix): defines_str carries
 * HASH_WORDS=8, HASH_BLOCK_BYTES=64, BASE_ALGO=blake2s plus HAS_SALT=1
 * and HMAC_KPASS=1 — pairwise distinct from all other template cores. */
#ifndef HASH_WORDS
#define HASH_WORDS 8
#endif
#ifndef HASH_BLOCK_BYTES
#define HASH_BLOCK_BYTES 64
#endif

/* Per-lane state struct. Identical layout to gpu_blake2s256_core.cl's
 * template_state — same h/t/f triple. The shared template body
 * (gpu_template.cl) only reads st.h[]; t[] and f[] are touched only by
 * the per-algo init/transform/finalize/iterate functions below. */
typedef struct {
    uint h[HASH_WORDS];   /* digest chaining state, LE-per-uint32 */
    uint t[2];            /* byte counter (low, high) */
    uint f[2];            /* finalization flag (0 = mid-stream,
                             0xFFFFFFFFu = last block) */
} template_state;

/* template_init: install BLAKE2S256 IV XOR'd with parameter block.
 * Identical to gpu_blake2s256_core.cl's template_init. */
static inline void template_init(template_state *st) {
    for (int i = 0; i < 8; i++) st->h[i] = B2S_IV[i];
    st->h[0] ^= 0x01010020u;   /* digest_length=32, key=0, fanout=1, depth=1 */
    st->t[0] = 0u; st->t[1] = 0u;
    st->f[0] = 0u; st->f[1] = 0u;
}

/* template_transform: absorb one full 64-byte block as a NON-FINAL block.
 * Public extension API. The HMAC body in template_finalize bypasses this
 * and inlines b2s_compress directly to avoid the function-call boundary
 * cost (matches the unsalted sibling's perf-fix lesson). */
static inline void template_transform(template_state *st,
                                      const uchar *block)
{
    uint t0 = st->t[0] + (uint)HASH_BLOCK_BYTES;
    uint carry = (t0 < st->t[0]) ? 1u : 0u;
    st->t[0] = t0;
    st->t[1] += carry;
    ulong counter = ((ulong)st->t[0]) | (((ulong)st->t[1]) << 32);
    b2s_compress(&st->h[0], block, counter, 0);
}

/* hmac_blake2s_run: helper that runs a complete BLAKE2S(buf[0..len-1])
 * and writes the 8 LE uint32 digest words into out[0..7].
 *
 * Used by the HMAC body to compute three independent BLAKE2S hashes:
 *   1. K = BLAKE2S(pass) when klen > 64 (rarely taken at GPU dispatch
 *      sizes; password lengths beyond 64 bytes are uncommon in our
 *      wordlist + rule range)
 *   2. inner = BLAKE2S((K^ipad) || msg)
 *   3. outer = BLAKE2S((K^opad) || inner)
 *
 * Matches the slab oracle's blake2s_hash() helper at gpu_hmac_blake2s.cl
 * line 10 and the CPU's blake2s() call signature at mdxfind.c:30367. */
static inline void hmac_blake2s_run(uint *out,
                                    const uchar *data, int datalen)
{
    uint h[8];
    for (int i = 0; i < 8; i++) h[i] = B2S_IV[i];
    h[0] ^= 0x01010020u;

    uchar buf[HASH_BLOCK_BYTES];
    int pos = 0;
    int rem = datalen;
    int doff = 0;
    ulong counter = 0;

    while (rem > 0) {
        if (pos == HASH_BLOCK_BYTES) {
            counter += HASH_BLOCK_BYTES;
            b2s_compress(h, buf, counter, 0);
            pos = 0;
        }
        int take = HASH_BLOCK_BYTES - pos;
        if (take > rem) take = rem;
        for (int i = 0; i < take; i++) buf[pos + i] = data[doff + i];
        pos  += take;
        doff += take;
        rem  -= take;
    }
    counter += (ulong)pos;
    for (int i = pos; i < HASH_BLOCK_BYTES; i++) buf[i] = 0;
    b2s_compress(h, buf, counter, 1);

    for (int i = 0; i < 8; i++) out[i] = h[i];
}

/* template_finalize: dual-mode entry.
 *
 * algo_mode == 5u (Family I HMAC-BLAKE2S production path):
 *   compute HMAC(key=data, msg=salt) and write 32-byte digest to st->h[].
 *
 *   The host passes:
 *     data  = the candidate password word (HMAC key)
 *     len   = password length
 *   The salt comes from the salt buffer bound at SETARG (args 17/18/19);
 *   the kernel reads it via the standard salted-template salt_buf /
 *   salt_off / salt_lens trio. SALT_POSITION=PREPEND in defines_str;
 *   HAS_SALT=1 unlocks the salt-axis arg block. The shared template body
 *   threads salt bytes into a per-algo finalize via #ifdef GPU_TEMPLATE_HAS_SALT
 *   wiring at the call site.
 *
 * algo_mode != 5u (defensive plain BLAKE2S(data) path):
 *   matches gpu_blake2s256_core.cl's template_finalize for the unsalted
 *   case. Structurally unreachable in production (mode-0 of this kernel
 *   is dead code per the carrier-kernel design) but kept for correctness
 *   parity with the unsalted sibling.
 *
 * Salt bytes for the HMAC msg come via the standard salted-template
 * trio uploaded by gpu_opencl_set_salts(): __global salt_buf (raw salt
 * bytes), __global salt_off (per-salt byte offset), __global salt_lens
 * (per-salt length). gpu_template.cl threads `salt_buf + salt_off[i]` and
 * `salt_lens[i]` into template_finalize per the GPU_TEMPLATE_HAS_SALT
 * call signature at line 387 (gpu_template.cl rev 1.6+). */
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
    if (algo_mode == 5u) {
        /* HMAC-BLAKE2S(key=data, msg=salt). Mirrors mdxfind.c:30341-30400
         * and the slab oracle gpu_hmac_blake2s.cl lines 38-97. */

        /* Step 1: build K (key block, 64 bytes). */
        uchar K[HASH_BLOCK_BYTES];
        if (len > HASH_BLOCK_BYTES) {
            /* key longer than block — hash it down to 32 bytes, zero-pad. */
            uint kh[8];
            hmac_blake2s_run(kh, data, len);
            /* Spill 8 LE uint32 into K[0..31] byte-image. */
            for (int i = 0; i < 8; i++) {
                uint w = kh[i];
                K[i*4 + 0] = (uchar)(w        & 0xFFu);
                K[i*4 + 1] = (uchar)((w >>  8) & 0xFFu);
                K[i*4 + 2] = (uchar)((w >> 16) & 0xFFu);
                K[i*4 + 3] = (uchar)((w >> 24) & 0xFFu);
            }
            for (int i = 32; i < HASH_BLOCK_BYTES; i++) K[i] = 0;
        } else {
            for (int i = 0; i < len; i++) K[i] = data[i];
            for (int i = len; i < HASH_BLOCK_BYTES; i++) K[i] = 0;
        }

        /* Step 2: inner = BLAKE2S((K XOR ipad) || msg).
         *
         * Salt length cap: mdxfind's MAXLINE / salt-pack geometry caps salt
         * length well under 256 bytes; the slab oracle's ibuf[192] bound
         * (gpu_hmac_blake2s.cl:76) covers our actual workload. We allocate
         * 64 + 256 bytes for headroom on the ipad-buffer concat — same shape
         * as the slab oracle plus 64 B extra. The salt buffer's actual length
         * (salt_len) is bounded by salt_lens (uint16_t max via SETARG arg 19),
         * so a 256-byte cap is safe at the salted-template salt-pack layer.
         *
         * salt_bytes is __global; copy salt bytes into private staging
         * before hashing (the slab oracle does the same — gpu_hmac_blake2s.cl
         * line 78 copies into a private uchar buffer).
         *
         * R1 mitigation (AMD ROCm comgr addrspace fragility): no
         * __private/__global addrspace mixing inside b2s_compress; the
         * staged ibuf[] is private, matching every other HMAC body in the
         * template family. */
        uint inner[8];
        {
            uchar ibuf[64 + 256];
            for (int i = 0; i < HASH_BLOCK_BYTES; i++) ibuf[i] = K[i] ^ 0x36u;
            uint slen = salt_len;
            if (slen > 256u) slen = 256u;
            for (uint i = 0; i < slen; i++) ibuf[64 + i] = salt_bytes[i];
            hmac_blake2s_run(inner, ibuf, 64 + (int)slen);
        }

        /* Step 3: outer = BLAKE2S((K XOR opad) || inner). 64 + 32 = 96 bytes,
         * fits in two 64-byte blocks. */
        uint outer[8];
        {
            uchar obuf[64 + 32];
            for (int i = 0; i < HASH_BLOCK_BYTES; i++) obuf[i] = K[i] ^ 0x5Cu;
            for (int i = 0; i < 8; i++) {
                uint w = inner[i];
                obuf[64 + i*4 + 0] = (uchar)(w        & 0xFFu);
                obuf[64 + i*4 + 1] = (uchar)((w >>  8) & 0xFFu);
                obuf[64 + i*4 + 2] = (uchar)((w >> 16) & 0xFFu);
                obuf[64 + i*4 + 3] = (uchar)((w >> 24) & 0xFFu);
            }
            hmac_blake2s_run(outer, obuf, 96);
        }

        /* Install HMAC digest into st->h[] for probe + emit. LE-direct
         * (matches BLAKE2S' native byte order). Counter+flag updates are
         * irrelevant after this point (template_iterate reinitializes them
         * on -i loops); leave them as set by template_init. */
        for (int i = 0; i < 8; i++) st->h[i] = outer[i];
        return;
    }
    /* Defensive fall-through to plain BLAKE2S(data) for non-HMAC algo_modes.
     * Production never reaches this in mode 0 because the carrier kernel
     * has no JOB_BLAKE2SSALTPASS dispatch path. The salt_bytes/salt_len
     * args are unused in this path; OpenCL's __global pointer cannot be
     * cast to (void) cleanly across all compilers, so we let the optimizer
     * elide them. */
#endif

    /* Plain BLAKE2S(data) main body. Identical to gpu_blake2s256_core.cl's
     * template_finalize. Reachable only as defensive fallback. */
    int pos = 0;
    while ((len - pos) > HASH_BLOCK_BYTES) {
        uint t0 = st->t[0] + (uint)HASH_BLOCK_BYTES;
        uint carry = (t0 < st->t[0]) ? 1u : 0u;
        st->t[0] = t0;
        st->t[1] += carry;
        ulong counter = ((ulong)st->t[0]) | (((ulong)st->t[1]) << 32);
        b2s_compress(&st->h[0], data + pos, counter, 0);
        pos += HASH_BLOCK_BYTES;
    }

    int rem = len - pos;
    uchar buf[HASH_BLOCK_BYTES];
    for (int i = 0; i < rem; i++) buf[i] = data[pos + i];
    for (int i = rem; i < HASH_BLOCK_BYTES; i++) buf[i] = 0;

    uint t0 = st->t[0] + (uint)rem;
    uint carry = (t0 < st->t[0]) ? 1u : 0u;
    st->t[0] = t0;
    st->t[1] += carry;
    ulong counter = ((ulong)st->t[0]) | (((ulong)st->t[1]) << 32);

    st->f[0] = 0xFFFFFFFFu;
    b2s_compress(&st->h[0], buf, counter, 1);
}

/* template_iterate: -i loop step. Re-encode the 32-byte BLAKE2s256 digest
 * as 64 lowercase hex chars and rehash. Identical to gpu_blake2s256_core.cl's
 * template_iterate.
 *
 * For HMAC-BLAKE2S (e828) the host forces max_iter=1 (CPU semantics at
 * mdxfind.c:30391 do not iter; checkhashsalt is called with iter=0). So in
 * production this function runs zero times for Family I — kept for
 * structural parity with the unsalted sibling and future-proofing if any
 * future HMAC-BLAKE2S iter variant is added. */
static inline void template_iterate(template_state *st)
{
    uint snap[8];
    for (int i = 0; i < 8; i++) snap[i] = st->h[i];

    uchar buf[HASH_BLOCK_BYTES];
    for (int i = 0; i < 8; i++) {
        uint s = snap[i];
        for (int b = 0; b < 4; b++) {
            uchar by = (uchar)((s >> (b * 8)) & 0xFFu);
            uchar hi = by >> 4, lo = by & 0xFu;
            buf[i * 8 + b * 2 + 0] = (hi < 10) ? (uchar)('0' + hi) : (uchar)('a' + (hi - 10));
            buf[i * 8 + b * 2 + 1] = (lo < 10) ? (uchar)('0' + lo) : (uchar)('a' + (lo - 10));
        }
    }

    for (int i = 0; i < 8; i++) st->h[i] = B2S_IV[i];
    st->h[0] ^= 0x01010020u;
    st->t[0] = 64u; st->t[1] = 0u;
    st->f[0] = 0xFFFFFFFFu; st->f[1] = 0u;

    b2s_compress(&st->h[0], buf, 64UL, 1);
}

/* template_digest_compare: probe leading 16 bytes (h[0..3]). LE-direct
 * (no bswap32). Identical to gpu_blake2s256_core.cl. */
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

/* template_emit_hit: emit a hit. Wraps EMIT_HIT_8 (32-byte digest =
 * 8 LE uint32 words; matches HMAC-BLAKE2S' 32-byte output, hexlen=64). */
#define template_emit_hit(hits, hit_count, max_hits, st, widx, sidx, iter) \
    EMIT_HIT_8((hits), (hit_count), (max_hits), \
               (widx), (sidx), (iter), ((st)->h))

/* B3 dedup+overflow-aware variant. */
#define template_emit_hit_or_overflow(hits, hit_count, max_hits, st, widx, sidx, iter, \
                                      hashes_shown, matched_idx, dedup_mask, \
                                      ovr_set, ovr_gid, lane_gid) \
    EMIT_HIT_8_DEDUP_OR_OVERFLOW((hits), (hit_count), (max_hits), \
               (widx), (sidx), (iter), ((st)->h), \
               (hashes_shown), (matched_idx), (dedup_mask), \
               (ovr_set), (ovr_gid), (lane_gid))
