/*
 * $Revision: 1.2 $
 * $Log: gpu_sha512_core.cl,v $
 * Revision 1.2  2026/05/11 05:22:02  dlr
 * Backfill $Revision/$Log RCS keyword stanzas per feedback_rcs_keyword_stanzas.md. Passive 4-line comment block at top of file; no behavioral change. Hand-authored .cl file was missing required stanzas (per memory: all hand-authored .c/.h/.cl/.frag/.tmpl/.py/.sh files MUST contain $Revision/$Log keyword stanzas). Build green on .205 against the post-add files; OpenCL compile strips comments so no kernel behavior change.
 *
 */
/* gpu_sha512_core.cl — SHA512 algorithm extension functions for the
 * generic dispatch template (Memo B Phase B5 sub-batch 1).
 *
 * Mirrors gpu_sha256_core.cl rev 1.2 structure. SHA512 is the FIRST
 * 64-bit-state algorithm in the family AND the FIRST with 128-bit length
 * encoding (vs MD5/SHA1/SHA256's 64-bit). Internal compression carries
 * 8 × uint64 chaining values; the final digest is exposed to the
 * template as 16 × uint32 LE-byteswapped (HASH_WORDS = 16) so the
 * template body's `digests_out[gid * HASH_WORDS + i] = st.h[i]` and
 * EMIT_HIT_16-style emit work without 64-bit indirection.
 *
 *   HASH_WORDS         — digest size in 32-bit words (16 for SHA512)
 *   HASH_BLOCK_BYTES   — compress-block size (128 — first deviation
 *                        from MD5/SHA1/SHA256's 64-byte block)
 *   template_state     — opaque per-lane state struct: ulong state[8]
 *                        for compression + uint h[16] for digest emit
 *   template_init      — install SHA-512 H[0..7] init constants (uint64)
 *   template_transform — absorb one 128-byte block (BIG-ENDIAN ulong load)
 *   template_finalize  — pad and finalize (in-place ulong M[16] pattern;
 *                        128-bit length encoded in M[14..15])
 *   template_iterate   — re-hash digest as 128-byte hex_lc (-i loop;
 *                        128 hex chars + 0x80 + 16-byte length needs
 *                        a SECOND block — same two-block pattern as
 *                        SHA256 iter)
 *   template_digest_compare — bswap64 first 2 ulong of state, decompose
 *                             into 4 LE uint32 (probe key)
 *   template_emit_hit       — EMIT_HIT_16 wrapper (16 LE uint32 words
 *                             from 8 ulong BE state)
 *
 * KEY POINTS vs SHA256 core:
 *
 * 1. Block size 128 bytes (16 ulong) instead of 64 bytes (16 uint).
 *    template_finalize iterates `len - pos >= 128`.
 *
 * 2. Length encoding 128-bit (M[14] high 64, M[15] low 64). For our
 *    wordlist inputs len < 2^57 bytes always, so M[14] = 0 and
 *    M[15] = len * 8.
 *
 * 3. 80-round compression with K512[80] uint64 constants and 64-bit
 *    Sigma/sigma functions (rotr64). sha512_block (gpu_common.cl
 *    line 756) handles the inner loop.
 *
 * 4. State width: 8 × uint64 internally; emit-time decompose into
 *    16 × uint32 LE matches gpu_sha512_packed.cl convention exactly.
 *    h[0] = (uint)bswap64(state[0]); h[1] = (uint)(bswap64(state[0]) >> 32).
 *    Compact-table probe key uses h[0..3] (leading 16 bytes LE) —
 *    same convention as MD5/SHA1/SHA256.
 *
 * 5. Iter loop. SHA512 hex output is 128 lowercase hex chars (64 digest
 *    bytes × 2). 128 + 1 (0x80) + 16 (length) = 145 bytes — exceeds one
 *    HASH_BLOCK_BYTES (128) block. Two-block pattern: block 1 holds the
 *    128 hex bytes BE (M[0..15]); block 2 has 0x80 marker at byte 128
 *    (M[0] = 0x8000000000000000UL BE), zeros M[1..14], M[15] = 128 * 8
 *    = 1024 (bit count low 64; high 64 in M[14] = 0).
 *
 * Source order at compile time:
 *
 *   [ gpu_common_str, gpu_md5_rules_str, gpu_sha512_core_str, gpu_template_str ]
 *
 * gpu_common_str provides sha512_block(), bswap64, sha512_to_hex_lc,
 * EMIT_HIT_16{,_DEDUP_OR_OVERFLOW}, and probe_compact_idx.
 *
 * Bytecast invariants:
 *   - Final state[0..7] (after compression) is BIG-ENDIAN — sha512_block's
 *     natural output. gpu_sha512_packed.cl matches.
 *   - st->h[0..15] holds 16 × uint32 LE-byteswapped digest words —
 *     populated by template_finalize / template_iterate at the end of
 *     each compression. Template body reads h[] directly.
 *   - template_emit_hit writes 16 LE-byteswapped uint32 to the hits
 *     buffer. Wire format matches gpu_sha512_packed.cl line 81's emit.
 *
 * R1 mitigation (AMD ROCm comgr addrspace fragility): single private
 * buffer pattern. No __private uchar* helpers; no addrspace-cast
 * ternaries. ulong M[16] is private scratch only.
 *
 * R2 (register pressure): SHA512's W[80] schedule (80 × 8 = 640 bytes
 * private scratch) is the largest in the family so far. Combined with
 * the 8-ulong state and 16-uint h[], expected priv_mem_size on gfx1201
 * MAY push past the 41,332-byte SHA224/SHA256 reading. Memo B §3 R2
 * flagged the SHA-512 family as the gfx1201 register-pressure risk;
 * production gate is the byte-exact CPU↔GPU smoke. Document the reading
 * but proceed unless the kernel fails to build or actively crashes.
 */

/* Per-algorithm geometry. Cache key (R3 fix) hashes the defines_str
 * "HASH_WORDS=16,HASH_BLOCK_BYTES=128" alongside source text so this
 * instantiation gets a distinct cache entry from MD5/SHA1/SHA256/SHA224/
 * MD4. */
#ifndef HASH_WORDS
#define HASH_WORDS 16
#endif
#ifndef HASH_BLOCK_BYTES
#define HASH_BLOCK_BYTES 128
#endif

/* Per-lane state struct. SHA512 carries 8 × uint64 chaining INTERNALLY
 * (state[]). The template body needs uint32 access to the digest for
 * digests_out[] and EMIT_HIT_16 — provide that via h[16] populated at
 * the end of each compression (template_finalize / template_iterate).
 * h[0..15] mirrors state[0..7] in LE-byteswapped uint32 pairs:
 *   h[2i]   = (uint)bswap64(state[i])
 *   h[2i+1] = (uint)(bswap64(state[i]) >> 32)
 *
 * This matches gpu_sha512_packed.cl line 67-73's split convention exactly.
 *
 * 16-byte alignment helps AMD spill geometry (cl_ulong required 8-byte;
 * declaring aligned(16) lets the compiler vectorize the M[] load). */
typedef struct {
    ulong state[8];   /* internal compression state, BIG-ENDIAN */
    uint  h[HASH_WORDS]; /* exposed digest words, LE-byteswapped uint32 */
} template_state;

/* template_init: install SHA512 IV into state.
 * Standard SHA-512 initial hash values (FIPS 180-4 §5.3.5). h[] is left
 * untouched; it is populated only after template_finalize / iterate
 * compress at least one block. */
static inline void template_init(template_state *st) {
    st->state[0] = 0x6a09e667f3bcc908UL;
    st->state[1] = 0xbb67ae8584caa73bUL;
    st->state[2] = 0x3c6ef372fe94f82bUL;
    st->state[3] = 0xa54ff53a5f1d36f1UL;
    st->state[4] = 0x510e527fade682d1UL;
    st->state[5] = 0x9b05688c2b3e6c1fUL;
    st->state[6] = 0x1f83d9abfb41bd6bUL;
    st->state[7] = 0x5be0cd19137e2179UL;
}

/* Internal helper: decompose state[8] (BE ulong) into h[16] (LE uint32).
 * Mirrors gpu_sha512_packed.cl line 67-73 exactly. */
static inline void template_state_to_h(template_state *st) {
    for (int i = 0; i < 8; i++) {
        ulong s = bswap64(st->state[i]);
        st->h[i*2]   = (uint)s;
        st->h[i*2+1] = (uint)(s >> 32);
    }
}

/* template_transform: absorb one HASH_BLOCK_BYTES (128) byte block.
 * SHA512 reads message words BIG-ENDIAN, 64-bit (8 bytes per word).
 *
 * Public extension API; the hot-path template_finalize bypasses this
 * via an in-place ulong M[16] pattern (perf-fix lesson, see file head). */
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

/* template_finalize: process the tail, append the 0x80 padding marker
 * + 128-bit BE length-in-bits, and absorb. Caller passes the full
 * buffer; we run complete 128-byte blocks then build the final padded
 * block(s) here. After return, st->state[0..7] holds the final SHA512
 * digest words BE; st->h[0..15] holds the same in LE uint32 pairs.
 *
 * Mirrors gpu_sha256_core.cl rev 1.2's in-place M[16] pattern, scaled
 * to ulong + 128-byte blocks + 128-bit length. SHA512 length encoding:
 * M[14] = high 64 bits of bit count; M[15] = low 64 bits. For our
 * wordlist inputs len < 2^57 bytes (always), high 64 bits = 0. The
 * 0x80 marker straddles the 112-byte (= 56*2) boundary check: if
 * rem < 112, length fits in this block; else need an extra block.
 *
 * R1 mitigation preserved: single private buffer (just ulong M[16]
 * + the input data pointer). */
static inline void template_finalize(template_state *st,
                                     const uchar *data, int len)
{
    ulong M[16];
    int pos = 0;

    /* Process complete 128-byte blocks, BE ulong load. */
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

    /* Zero scratch. */
    for (int j = 0; j < 16; j++) M[j] = 0UL;

    /* Copy remaining tail bytes BIG-ENDIAN. Each byte's position within
     * its 64-bit word is determined by 7-(byte_idx & 7) shift. */
    for (int i = 0; i < rem; i++) {
        int wi = i >> 3;
        int bi = 7 - (i & 7);
        M[wi] |= ((ulong)data[pos + i]) << (bi * 8);
    }
    /* 0x80 padding marker, BE byte position. */
    {
        int wi = rem >> 3;
        int bi = 7 - (rem & 7);
        M[wi] |= ((ulong)0x80UL) << (bi * 8);
    }

    if (rem < 112) {
        /* Length fits in this block. M[14] = high 64 bits = 0;
         * M[15] = low 64 bits = len * 8. */
        M[14] = 0UL;
        M[15] = (ulong)((ulong)len * 8UL);
        sha512_block(&st->state[0], M);
    } else {
        /* Need one extra padding-only block to hold the length. */
        sha512_block(&st->state[0], M);
        for (int j = 0; j < 16; j++) M[j] = 0UL;
        M[14] = 0UL;
        M[15] = (ulong)((ulong)len * 8UL);
        sha512_block(&st->state[0], M);
    }

    /* Decompose final state into h[16] LE uint32 for digest emit. */
    template_state_to_h(st);
}

/* template_iterate: -i loop step. Re-encode the digest as 128-byte
 * lowercase hex ASCII and rehash. SHA512's hex output is 128 chars,
 * which exactly fills one HASH_BLOCK_BYTES (128) block — but the
 * 0x80 marker + 16-byte length need a SECOND block.
 *
 * Mirrors gpu_sha512_packed.cl rev 1 lines 84-93 byte-for-byte:
 *   - sha512_to_hex_lc(state, M) : 8 ulong BE state -> 16 ulong BE
 *     hex M[] (each ulong holds 8 hex chars BE)
 *   - state reset to SHA512 IV; sha512_block(state, M)
 *   - second block: M[0] = 0x8000000000000000UL BE (0x80 at byte 128),
 *     M[1..14] = 0, M[15] = 128*8 = 1024
 *   - sha512_block(state, M) again
 *   - decompose state -> h[16]. */
static inline void template_iterate(template_state *st)
{
    ulong M[16];
    /* sha512_to_hex_lc (gpu_common.cl line 780) writes 16 ulong BE
     * hex words into M[] from state[]. */
    sha512_to_hex_lc(&st->state[0], M);
    /* Reset state to SHA512 IV; absorb the prepared hex block. */
    st->state[0] = 0x6a09e667f3bcc908UL;
    st->state[1] = 0xbb67ae8584caa73bUL;
    st->state[2] = 0x3c6ef372fe94f82bUL;
    st->state[3] = 0xa54ff53a5f1d36f1UL;
    st->state[4] = 0x510e527fade682d1UL;
    st->state[5] = 0x9b05688c2b3e6c1fUL;
    st->state[6] = 0x1f83d9abfb41bd6bUL;
    st->state[7] = 0x5be0cd19137e2179UL;
    sha512_block(&st->state[0], M);

    /* Second block: pad + length only. */
    M[0] = 0x8000000000000000UL;     /* 0x80 BE at byte position 128 */
    for (int j = 1; j < 15; j++) M[j] = 0UL;
    M[14] = 0UL;
    M[15] = 128UL * 8UL;             /* 128 hex chars = 1024 bits */
    sha512_block(&st->state[0], M);

    /* Decompose updated state into h[16]. */
    template_state_to_h(st);
}

/* template_digest_compare: probe the compact table with the leading
 * 16 bytes of the final digest. h[0..3] already holds the 4 LE uint32
 * leading words (populated by template_state_to_h via finalize/iterate).
 *
 * Mirrors gpu_sha512_packed.cl line 75-80 exactly: probe_compact[_idx]
 * called with h[0], h[1], h[2], h[3]. */
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

/* template_emit_hit: emit a hit. Wraps EMIT_HIT_16 (SHA512 = 16 uint32
 * digest words, populated LE-byteswapped in st->h by finalize/iterate).
 * Wire format matches gpu_sha512_packed.cl emit. */
#define template_emit_hit(hits, hit_count, max_hits, st, widx, sidx, iter) \
    EMIT_HIT_16((hits), (hit_count), (max_hits), \
                (widx), (sidx), (iter), ((st)->h))

/* B3 dedup+overflow-aware variant. See gpu_md5_core.cl for protocol
 * notes. */
#define template_emit_hit_or_overflow(hits, hit_count, max_hits, st, widx, sidx, iter, \
                                      hashes_shown, matched_idx, dedup_mask, \
                                      ovr_set, ovr_gid, lane_gid) \
    EMIT_HIT_16_DEDUP_OR_OVERFLOW((hits), (hit_count), (max_hits), \
               (widx), (sidx), (iter), ((st)->h), \
               (hashes_shown), (matched_idx), (dedup_mask), \
               (ovr_set), (ovr_gid), (lane_gid))
