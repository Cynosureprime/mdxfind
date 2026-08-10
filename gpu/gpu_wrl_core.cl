/*
 * $Revision: 1.3 $
 * $Log: gpu_wrl_core.cl,v $
 * Revision 1.3  2026/08/10 14:50:25  dlr
 * Remove duplicate WRL_RC, WRL_SBOX and WRL_OP definitions. gpu_common.cl rev 1.28 lifted the wrl_block primitive in from librhash for the WRLMD5PASS family helper and supplies all three; this file is only ever concatenated after gpu_common.cl in gpu_opencl_template_compile_wrl, so the second copy made every WRL template build fail with CL_BUILD_PROGRAM_FAILURE, redefinition of WRL_RC and WRL_SBOX. The effect was not merely loss of GPU acceleration: resolve_kernel returned NULL and the work was never re-routed to CPU, so WRL found zero hashes on any OpenCL device from 2026-05-27 onward while the run exited normally. Tables verified value-identical, 10 of 10 round constants and 2048 of 2048 S-box entries; the two WRL_OP bodies differ only in parenthesisation of the src argument, immaterial at the K[m] and state[m] call sites. Header note corrected: it reasoned only about gpu_wrlunsalted.cl being a separate cl_program and overlooked gpu_common.cl being in the same one. Validated on fpga GTX 1080 at iter 1 and iter 3, GPU and CPU byte-identical, 15 of 15.
 *
 * Revision 1.2  2026/05/11 05:22:02  dlr
 * Backfill $Revision/$Log RCS keyword stanzas per feedback_rcs_keyword_stanzas.md. Passive 4-line comment block at top of file; no behavioral change. Hand-authored .cl file was missing required stanzas (per memory: all hand-authored .c/.h/.cl/.frag/.tmpl/.py/.sh files MUST contain $Revision/$Log keyword stanzas). Build green on .205 against the post-add files; OpenCL compile strips comments so no kernel behavior change.
 *
 */
/* gpu_wrl_core.cl — Whirlpool (WRL, -m e5) algorithm extension functions
 * for the generic dispatch template (Memo B Phase B5 sub-batch 6.5).
 *
 * WRL = Whirlpool 512-bit hash. 64-byte digest. The compression function
 * is Miyaguchi-Preneel applied to a 64-byte BIG-ENDIAN block:
 *
 *   hash = old_hash ^ state ^ block
 *
 * where state = 10-round AES-like permutation of (block ^ old_hash) keyed
 * by old_hash + round constants. Single-block IV is all-zeros.
 *
 * Dual-purpose ship/diagnostic (ref: brief Path A 2026-05-05):
 *
 *   1. SHIP — first wired Whirlpool template; expands the unified template
 *      family from 29 to 30 algorithms.
 *   2. DIAGNOSTIC — Streebog (B5 sub-batch 5b, 2026-05-05) revealed RDNA4
 *      gfx1201 incompatibility with the rules-walker template path: 4/100
 *      lanes pass per wavefront (positions 1, 33, 65, 97 — wavefront-stride
 *      lane 0). The hypothesis is that the 16 KB __constant SBOB_SL64 table
 *      addressed via the SBOG_LPS uchar-from-ulong pattern triggers RDNA4
 *      codegen breakage. WRL has a 16 KB __constant WRL_SBOX of identical
 *      shape ([8][256] ulong) but a DIFFERENT access pattern — direct ulong
 *      indexing with shift-then-mask (`(src)[i] >> shift & 0xff`), no
 *      uchar-from-ulong reinterpret cast.
 *
 *      Decision tree:
 *        - WRL gfx1201 100/100 PASS → bug is SBOG_LPS-specific (uchar/ulong
 *          aliasing), NOT 16 KB __constant capacity.
 *        - WRL gfx1201 < 100/100, lanes 1/33/65/97 only → generic 16 KB
 *          __constant RDNA4 issue. Both algos need __global mitigation.
 *        - WRL gfx1201 different lane pattern → distinct bug class.
 *
 * Iter step (CPU JOB_WRL, mdxfind.c:28019-28024):
 *   for x = 1..Maxiter:
 *     WHIRLPOOL(cur, len, curin.h);              // 64-byte digest
 *     checkhash(curin, len, x, JOB_WRL);
 *     cur = prmd5(curin.h, mdbuf, 128);          // 128 lowercase hex chars
 *     len = 128;                                 // next iter consumes hex
 *
 *   Hex feedback: 64-byte digest -> 128 ASCII hex chars. Whirlpool then
 *   processes those 128 bytes via 2 full blocks + 1 padding-only block
 *   (128 + 1 (0x80) + 32 (256-bit length) > 128, so the length spills
 *   into a third block).
 *
 * State carried in template_state:
 *   ulong state[8]    — internal compression state (BIG-ENDIAN, ulong)
 *   uint  h[16]       — exposed digest words, LE-byteswapped uint32
 *   (probe path uses h[0..3] = leading 16 bytes LE; same convention as
 *   the slab kernel gpu_wrlunsalted.cl:765 and SHA-512 template.)
 *
 * Cache key (R3): defines_str = "HASH_WORDS=16,HASH_BLOCK_BYTES=64".
 * Distinct cache entry guaranteed by source-text hash difference.
 *
 * R1 mitigation (AMD ROCm comgr addrspace fragility): single private
 * buffer pattern. No __private uchar* helpers; no addrspace-cast
 * ternaries. ulong M[8] is private scratch only.
 *
 * Source order at compile time:
 *   [ gpu_common_str, gpu_md5_rules_str, gpu_wrl_core_str, gpu_template_str ]
 *
 * NOTE: WRL_SBOX, WRL_RC and WRL_OP are NOT defined here - they are
 * supplied by gpu_common_str, the first source in the list above. Only
 * whirlpool_compress* and the template_* specialisations are local to
 * this translation unit. The slab kernel gpu_wrlunsalted.cl carries its
 * own copies, but that is a separate cl_program, so it does not collide.
 *
 * The earlier version of this note reasoned only about gpu_wrlunsalted.cl
 * and overlooked gpu_common.cl, which is in the SAME program - that is
 * precisely how the 2026-05-27 redefinition regression went unnoticed.
 */

#ifndef HASH_WORDS
#define HASH_WORDS 16
#endif
#ifndef HASH_BLOCK_BYTES
#define HASH_BLOCK_BYTES 64
#endif

/* Whirlpool round constants, S-box and the WRL_OP macro are deliberately
 * NOT defined here. gpu_common.cl is sources[0] of the WRL template build
 * (gpu_opencl.c gpu_opencl_template_compile_wrl) and has supplied WRL_RC,
 * WRL_SBOX and WRL_OP since gpu_common.cl rev 1.28 (2026-05-27, sub-phase
 * 5b2a1 lifted wrl_block in from librhash for the WRLMD5PASS family
 * helper). Defining them a second time here made every WRL template build
 * fail with CL_BUILD_PROGRAM_FAILURE - redefinition of WRL_RC and
 * WRL_SBOX - which silently dropped WRL to CPU-only on every OpenCL
 * device from that date until this revision. The values are identical:
 * 10 of 10 round constants and 2048 of 2048 S-box entries match. The two
 * WRL_OP bodies differ only in parenthesisation of the src argument,
 * which is immaterial at the K[m] and state[m] call sites below. This
 * file is only ever concatenated after gpu_common.cl, so the shared
 * definitions are always in scope here. */

/* Whirlpool compress with zero IV (single-block fast path).
 *   hash = Miyaguchi-Preneel(0, block) = state_final ^ block
 * Used by template_finalize for the first block when input fits in one
 * block (input_len <= 31 bytes; 0x80 padding + 32-byte BE length fits).
 *
 * Mirrors gpu_wrlunsalted.cl:571-608 byte-for-byte. */
static inline void whirlpool_compress_zero_iv(ulong *hash, ulong *block) {
    ulong K[2][8];
    ulong state[2][8];
    int m = 0;

    for (int i = 0; i < 8; i++) {
        K[0][i] = 0;
        state[0][i] = block[i];
    }

    for (int r = 0; r < 10; r++) {
        K[m^1][0] = WRL_OP(K[m], 0) ^ WRL_RC[r];
        K[m^1][1] = WRL_OP(K[m], 1);
        K[m^1][2] = WRL_OP(K[m], 2);
        K[m^1][3] = WRL_OP(K[m], 3);
        K[m^1][4] = WRL_OP(K[m], 4);
        K[m^1][5] = WRL_OP(K[m], 5);
        K[m^1][6] = WRL_OP(K[m], 6);
        K[m^1][7] = WRL_OP(K[m], 7);

        state[m^1][0] = WRL_OP(state[m], 0) ^ K[m^1][0];
        state[m^1][1] = WRL_OP(state[m], 1) ^ K[m^1][1];
        state[m^1][2] = WRL_OP(state[m], 2) ^ K[m^1][2];
        state[m^1][3] = WRL_OP(state[m], 3) ^ K[m^1][3];
        state[m^1][4] = WRL_OP(state[m], 4) ^ K[m^1][4];
        state[m^1][5] = WRL_OP(state[m], 5) ^ K[m^1][5];
        state[m^1][6] = WRL_OP(state[m], 6) ^ K[m^1][6];
        state[m^1][7] = WRL_OP(state[m], 7) ^ K[m^1][7];

        m ^= 1;
    }

    for (int i = 0; i < 8; i++)
        hash[i] = state[0][i] ^ block[i];
}

/* General Whirlpool compress: hash = Miyaguchi-Preneel(cv, block) =
 *                                    cv ^ state_final ^ block.
 * Used for second-and-subsequent blocks. Mirrors gpu_wrlunsalted.cl:611-645. */
static inline void whirlpool_compress_cv(ulong *hash, ulong *block, ulong *cv) {
    ulong K[2][8];
    ulong state[2][8];
    int m = 0;

    for (int i = 0; i < 8; i++) {
        K[0][i] = cv[i];
        state[0][i] = block[i] ^ cv[i];
    }

    for (int r = 0; r < 10; r++) {
        K[m^1][0] = WRL_OP(K[m], 0) ^ WRL_RC[r];
        K[m^1][1] = WRL_OP(K[m], 1);
        K[m^1][2] = WRL_OP(K[m], 2);
        K[m^1][3] = WRL_OP(K[m], 3);
        K[m^1][4] = WRL_OP(K[m], 4);
        K[m^1][5] = WRL_OP(K[m], 5);
        K[m^1][6] = WRL_OP(K[m], 6);
        K[m^1][7] = WRL_OP(K[m], 7);

        state[m^1][0] = WRL_OP(state[m], 0) ^ K[m^1][0];
        state[m^1][1] = WRL_OP(state[m], 1) ^ K[m^1][1];
        state[m^1][2] = WRL_OP(state[m], 2) ^ K[m^1][2];
        state[m^1][3] = WRL_OP(state[m], 3) ^ K[m^1][3];
        state[m^1][4] = WRL_OP(state[m], 4) ^ K[m^1][4];
        state[m^1][5] = WRL_OP(state[m], 5) ^ K[m^1][5];
        state[m^1][6] = WRL_OP(state[m], 6) ^ K[m^1][6];
        state[m^1][7] = WRL_OP(state[m], 7) ^ K[m^1][7];

        m ^= 1;
    }

    for (int i = 0; i < 8; i++)
        hash[i] = cv[i] ^ state[0][i] ^ block[i];
}

/* Per-lane state struct. WRL carries 8 × uint64 chaining BIG-ENDIAN
 * INTERNALLY (state[]). The template body needs uint32 access to the
 * digest for digests_out[] and EMIT_HIT_16 — provide that via h[16]
 * populated at the end of each compression (template_finalize /
 * template_iterate). h[0..15] mirrors state[0..7] in LE-byteswapped
 * uint32 pairs:
 *   h[2i]   = (uint)bswap64(state[i])
 *   h[2i+1] = (uint)(bswap64(state[i]) >> 32)
 *
 * This matches gpu_wrlunsalted.cl:754-759's split convention exactly. */
typedef struct {
    ulong state[8];   /* internal compression state, BIG-ENDIAN ulong */
    uint  h[HASH_WORDS]; /* exposed digest words, LE-byteswapped uint32 */
} template_state;

static inline void template_init(template_state *st) {
    /* Whirlpool IV is all-zeros. */
    for (int i = 0; i < 8; i++) st->state[i] = 0UL;
    /* h[] is populated only after at least one compression in
     * template_finalize / template_iterate. */
}

/* Internal helper: decompose state[8] (BE ulong) into h[16] (LE uint32).
 * Mirrors gpu_wrlunsalted.cl:754-759 exactly. */
static inline void template_state_to_h(template_state *st) {
    for (int i = 0; i < 8; i++) {
        ulong s = bswap64(st->state[i]);
        st->h[i*2]   = (uint)s;
        st->h[i*2+1] = (uint)(s >> 32);
    }
}

/* Stub for template interface symmetry; the shared template body invokes
 * template_finalize directly with the full input buffer. */
static inline void template_transform(template_state *st,
                                      const uchar *block)
{
    (void)st;
    (void)block;
}

/* template_finalize: process the input bytes and write the final digest.
 *
 * Whirlpool padding spec:
 *   - block size = 64 bytes
 *   - 0x80 marker after the last input byte
 *   - 256-bit BIG-ENDIAN bit count in last 32 bytes of the FINAL block
 *
 * Single-block fits when (rem + 1 + 32 <= 64), i.e., rem <= 31.
 * Otherwise need an extra padding-only final block.
 *
 * For our chokepoint maxlen=27, all primary-input cases fit single-block.
 * The iterate path's 128-byte hex string requires multi-block (handled in
 * template_iterate). This finalize path supports general len, anchoring
 * compatibility for any future maxlen > 31 changes.
 *
 * Length encoding: bit count fits in low 64 bits of the 256-bit field
 * (M[7]); high 192 bits (M[4..6]) are zero. Matches the slab kernel's
 * `M64[6] = 0; M64[7] = bitlen;` pattern (gpu_wrlunsalted.cl:746-747).
 *
 * R1 mitigation: single private buffer (just ulong M[8] + ulong cv[8]
 * for multi-block + the input data pointer). No private uchar* helper
 * indirection. */
static inline void template_finalize(template_state *st,
                                     const uchar *data, int len)
{
    ulong M[8];
    int pos = 0;
    int first = 1;
    ulong cv[8];

    /* Process complete 64-byte blocks BIG-ENDIAN ulong load. */
    while (len - pos >= HASH_BLOCK_BYTES) {
        for (int j = 0; j < 8; j++) {
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
        if (first) {
            whirlpool_compress_zero_iv(st->state, M);
            first = 0;
        } else {
            for (int i = 0; i < 8; i++) cv[i] = st->state[i];
            whirlpool_compress_cv(st->state, M, cv);
        }
        pos += HASH_BLOCK_BYTES;
    }

    int rem = len - pos;  /* 0..63 */

    /* Build final padded block(s). Zero scratch, copy remaining tail
     * bytes BIG-ENDIAN, append 0x80 marker. */
    for (int j = 0; j < 8; j++) M[j] = 0UL;
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

    if (rem <= 31) {
        /* Length fits in this block. M[4..6] = 0; M[7] = total_bits.
         * total_bits = (input_total_bytes * 8). For multi-block path
         * (this branch is reached after len was split), total_bits is
         * still len*8 (same input). */
        M[4] = 0UL;
        M[5] = 0UL;
        M[6] = 0UL;
        M[7] = (ulong)((ulong)len * 8UL);
        if (first) {
            whirlpool_compress_zero_iv(st->state, M);
        } else {
            for (int i = 0; i < 8; i++) cv[i] = st->state[i];
            whirlpool_compress_cv(st->state, M, cv);
        }
    } else {
        /* Need one extra padding-only block to hold the length.
         * Compress this partial block (with 0x80 but no length), then
         * issue a length-only block. */
        if (first) {
            whirlpool_compress_zero_iv(st->state, M);
            first = 0;
        } else {
            for (int i = 0; i < 8; i++) cv[i] = st->state[i];
            whirlpool_compress_cv(st->state, M, cv);
        }
        for (int j = 0; j < 8; j++) M[j] = 0UL;
        M[7] = (ulong)((ulong)len * 8UL);
        for (int i = 0; i < 8; i++) cv[i] = st->state[i];
        whirlpool_compress_cv(st->state, M, cv);
    }

    /* Decompose final state into h[16] LE uint32 for digest emit. */
    template_state_to_h(st);
}

/* template_iterate: -i loop step. Re-encode the digest as 128-byte
 * lowercase hex ASCII and rehash with a fresh zero IV. WRL hex output
 * is 128 chars = 2 full 64-byte blocks. Then add a padding-only block
 * (0x80 at byte 0 + 32-byte length 128*8=1024 in bits) for total of
 * 3 compressions.
 *
 * Mirrors gpu_wrlunsalted.cl:771-833 byte-for-byte semantically. The
 * slab uses the LE h[] pre-computation; here we emit hex bytes and feed
 * them BIG-ENDIAN-loaded into the compression. Both paths produce the
 * same digest because Whirlpool is byte-stream defined; the ulong load
 * order is just an implementation detail.
 *
 * R1 mitigation preserved: only ulong scratch (M[8], cv[8]) + small
 * private uchar hex_buf[128]. */
static inline void template_iterate(template_state *st)
{
    ulong M[8];
    ulong cv[8];
    uchar hex_buf[128];

    /* Emit 128 lowercase hex chars from h[0..15] (LE uint32 digest).
     * h[i] is one uint32 LE; bytes b0=lo, b1=mid-lo, b2=mid-hi, b3=hi. */
    for (int i = 0; i < 16; i++) {
        uint s = st->h[i];
        uint b0 = s & 0xff, b1 = (s >> 8) & 0xff;
        uint b2 = (s >> 16) & 0xff, b3 = (s >> 24) & 0xff;
        uint hi0 = (b0 >> 4) & 0xf, lo0 = b0 & 0xf;
        uint hi1 = (b1 >> 4) & 0xf, lo1 = b1 & 0xf;
        uint hi2 = (b2 >> 4) & 0xf, lo2 = b2 & 0xf;
        uint hi3 = (b3 >> 4) & 0xf, lo3 = b3 & 0xf;
        hex_buf[i*8 + 0] = (uchar)(hi0 + ((hi0 < 10u) ? '0' : ('a' - 10)));
        hex_buf[i*8 + 1] = (uchar)(lo0 + ((lo0 < 10u) ? '0' : ('a' - 10)));
        hex_buf[i*8 + 2] = (uchar)(hi1 + ((hi1 < 10u) ? '0' : ('a' - 10)));
        hex_buf[i*8 + 3] = (uchar)(lo1 + ((lo1 < 10u) ? '0' : ('a' - 10)));
        hex_buf[i*8 + 4] = (uchar)(hi2 + ((hi2 < 10u) ? '0' : ('a' - 10)));
        hex_buf[i*8 + 5] = (uchar)(lo2 + ((lo2 < 10u) ? '0' : ('a' - 10)));
        hex_buf[i*8 + 6] = (uchar)(hi3 + ((hi3 < 10u) ? '0' : ('a' - 10)));
        hex_buf[i*8 + 7] = (uchar)(lo3 + ((lo3 < 10u) ? '0' : ('a' - 10)));
    }

    /* Reset state to all-zeros IV. */
    for (int i = 0; i < 8; i++) st->state[i] = 0UL;

    /* Block 1: hex_buf[0..63] BE-loaded into M[0..7]. */
    for (int j = 0; j < 8; j++) {
        int off = j * 8;
        M[j] = ((ulong)hex_buf[off]     << 56)
             | ((ulong)hex_buf[off + 1] << 48)
             | ((ulong)hex_buf[off + 2] << 40)
             | ((ulong)hex_buf[off + 3] << 32)
             | ((ulong)hex_buf[off + 4] << 24)
             | ((ulong)hex_buf[off + 5] << 16)
             | ((ulong)hex_buf[off + 6] <<  8)
             |  (ulong)hex_buf[off + 7];
    }
    whirlpool_compress_zero_iv(st->state, M);

    /* Block 2: hex_buf[64..127]. */
    for (int j = 0; j < 8; j++) {
        int off = 64 + j * 8;
        M[j] = ((ulong)hex_buf[off]     << 56)
             | ((ulong)hex_buf[off + 1] << 48)
             | ((ulong)hex_buf[off + 2] << 40)
             | ((ulong)hex_buf[off + 3] << 32)
             | ((ulong)hex_buf[off + 4] << 24)
             | ((ulong)hex_buf[off + 5] << 16)
             | ((ulong)hex_buf[off + 6] <<  8)
             |  (ulong)hex_buf[off + 7];
    }
    for (int i = 0; i < 8; i++) cv[i] = st->state[i];
    whirlpool_compress_cv(st->state, M, cv);

    /* Block 3: padding-only. 0x80 at byte 0 (M[0] high byte BE),
     * zeros, 256-bit BE bit count = 128 * 8 = 1024 in low 64 bits (M[7]).
     * Whirlpool stores 256-bit BE bit count in last 32 bytes (M[4..7]).
     * For 128 bytes input total, bit count fits in M[7] = 1024. */
    for (int j = 0; j < 8; j++) M[j] = 0UL;
    M[0] = 0x8000000000000000UL;   /* 0x80 in first byte BE */
    M[7] = 1024UL;                  /* bit count */
    for (int i = 0; i < 8; i++) cv[i] = st->state[i];
    whirlpool_compress_cv(st->state, M, cv);

    /* Decompose updated state into h[16]. */
    template_state_to_h(st);
}

/* template_digest_compare: probe the compact table with the leading
 * 16 bytes of the final digest. h[0..3] holds the 4 LE uint32 leading
 * words (populated by template_state_to_h via finalize/iterate).
 *
 * Mirrors gpu_wrlunsalted.cl:765-769 exactly: probe with h[0], h[1],
 * h[2], h[3]. */
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

/* template_emit_hit: emit a hit. Wraps EMIT_HIT_16 (WRL = 16 uint32
 * digest words, populated LE-byteswapped in st->h by finalize/iterate).
 * Wire format matches gpu_wrlunsalted.cl emit. */
#define template_emit_hit(hits, hit_count, max_hits, st, widx, sidx, iter) \
    EMIT_HIT_16((hits), (hit_count), (max_hits), \
                (widx), (sidx), (iter), ((st)->h))

/* B3 dedup+overflow-aware variant. */
#define template_emit_hit_or_overflow(hits, hit_count, max_hits, st, widx, sidx, iter, \
                                      hashes_shown, matched_idx, dedup_mask, \
                                      ovr_set, ovr_gid, lane_gid) \
    EMIT_HIT_16_DEDUP_OR_OVERFLOW((hits), (hit_count), (max_hits), \
               (widx), (sidx), (iter), ((st)->h), \
               (hashes_shown), (matched_idx), (dedup_mask), \
               (ovr_set), (ovr_gid), (lane_gid))
