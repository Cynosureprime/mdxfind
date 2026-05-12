/*
 * $Revision: 1.2 $
 * $Log: gpu_blake2b512_core.cl,v $
 * Revision 1.2  2026/05/11 05:22:01  dlr
 * Backfill $Revision/$Log RCS keyword stanzas per feedback_rcs_keyword_stanzas.md. Passive 4-line comment block at top of file; no behavioral change. Hand-authored .cl file was missing required stanzas (per memory: all hand-authored .c/.h/.cl/.frag/.tmpl/.py/.sh files MUST contain $Revision/$Log keyword stanzas). Build green on .205 against the post-add files; OpenCL compile strips comments so no kernel behavior change.
 *
 */
/* gpu_blake2b512_core.cl — BLAKE2B-512 algorithm extension functions for
 * the generic dispatch template (Memo B Phase B5 sub-batch 3).
 *
 * BLAKE2B-512 is BLAKE2b with digest_length=64 (full 512-bit output, all
 * 8 ulong of chaining state). The compression function is identical to
 * BLAKE2B-256; only the parameter-block init and output-length (=> the
 * iter hex re-encoding length and EMIT_HIT macro choice) differ.
 *
 * Geometry / byte ordering:
 *   - 128-byte block, 8 × uint64 chaining state, LITTLE-ENDIAN
 *   - 12-round G-function compression (b2b_compress in gpu_common.cl)
 *   - parameter block init: IV[0] ^ 0x01010040
 *     (digest_length=64 << 0 | key_length=0 << 8 | fanout=1 << 16 |
 *      depth=1 << 24 — high 32 bits zero)
 *   - 64-byte output = 8 ulong = 16 uint32 LE (full digest, no truncation)
 *
 *   HASH_WORDS         — digest size in 32-bit words (16 for BLAKE2B-512)
 *   HASH_BLOCK_BYTES   — compress-block size (128)
 *   template_state     — h_ulong[8] internal + t[2] + f[2] + h[16] exposed
 *   template_init      — install IV ^ parameter_block (digest_length=64)
 *   template_transform — absorb one 128-byte block (NON-FINAL)
 *   template_finalize  — process complete blocks, zero tail, run final
 *                        compression with last==1; populate h[16] = all
 *                        8 ulong as 16 uint32 LE
 *   template_iterate   — re-hash digest as 128-byte hex_lc input (64-byte
 *                        binary -> 128 lc hex chars). Per BLAKE2 strict-
 *                        greater rule: 128 hex chars exactly == HASH_-
 *                        BLOCK_BYTES, so the WHOLE hex string IS the final
 *                        block (zero complete-block iterations); single
 *                        compression with last=1, counter=128.
 *   template_digest_compare — probe leading 16 bytes (h[0..3]); LE state.
 *   template_emit_hit       — EMIT_HIT_16 wrapper (16 uint32 LE = full
 *                             64-byte BLAKE2B-512 output)
 *
 * Source order at compile time:
 *
 *   [ gpu_common_str, gpu_md5_rules_str, gpu_blake2b512_core_str, gpu_template_str ]
 *
 * gpu_common_str provides B2B_IV[8], B2B_SIGMA[12][16], b2b_compress(),
 * EMIT_HIT_16{,_DEDUP_OR_OVERFLOW}, and probe_compact_idx.
 *
 * Bytecast invariants:
 *   - h_ulong[0..7] is LE per ulong; mdxfind.c blake2b_hash() at line 2446
 *     uses memcpy(out, h, outlen) directly — direct ulong byte image.
 *   - Decomposition to st->h[0..15] uint32 LE pairs follows the same
 *     pattern as BLAKE2B-256 but spans all 8 ulong → 16 uint32:
 *       st->h[2i]   = (uint)h_ulong[i]
 *       st->h[2i+1] = (uint)(h_ulong[i] >> 32)    for i = 0..7
 *   - Wire format matches gpu_sha512_packed.cl convention exactly: 16 LE
 *     uint32 words (state ulong split into low/high uint32 halves). The
 *     hit-replay path in gpujob_opencl.c reads the same 64-byte binary
 *     digest regardless of which 16-uint32 algorithm produced it.
 *
 * R1 mitigation: single-private-buffer pattern. R2 (register pressure):
 * BLAKE2b uses 8 ulong state + 16 ulong v[] + 16 ulong m[] = ~80 ulong-
 * equivalent. Compared to SHA512's W[80] schedule (640 bytes private
 * scratch), BLAKE2b's 16 m[] + 16 v[] is 256 bytes — substantially less.
 * Expected priv_mem_size on gfx1201 may be LOWER than SHA-512's
 * 42,520 B baseline. Per-phase clGetKernelWorkGroupInfo probe.
 */

/* Per-algorithm geometry. Cache key (R3 fix): defines_str
 * "HASH_WORDS=16,HASH_BLOCK_BYTES=128" — same width/block as SHA-512 but
 * distinct source text (b2b_compress vs sha512_block; different state
 * struct), so cache key is unique per gpu_kernel_cache.c rev 1.5+. */
#ifndef HASH_WORDS
#define HASH_WORDS 16
#endif
#ifndef HASH_BLOCK_BYTES
#define HASH_BLOCK_BYTES 128
#endif

typedef struct {
    ulong h_ulong[8];     /* internal chaining state, LE per ulong */
    ulong t[2];           /* 128-bit byte counter */
    ulong f[2];           /* finalization flag pair */
    uint  h[HASH_WORDS];  /* exposed digest words, 16 uint32 LE */
} template_state;

/* Decompose all 8 ulong of internal state into 16 LE uint32 for the
 * exposed h[]. Same convention as gpu_sha512_core.cl template_state_to_h
 * but WITHOUT bswap64 — BLAKE2b state is naturally LE per ulong (the
 * memcpy(out, h, outlen) byte image gives the canonical hash output). */
static inline void template_state_to_h_b512(template_state *st) {
    for (int i = 0; i < 8; i++) {
        ulong v = st->h_ulong[i];
        st->h[i * 2]     = (uint)v;
        st->h[i * 2 + 1] = (uint)(v >> 32);
    }
}

/* template_init: install BLAKE2B-512 IV XOR'd with parameter block.
 * Param block low ulong = 0x01010040 (digest_length=64). Matches mdxfind.c
 * blake2b_hash (line 2431) `0x01010000 ^ outlen` exactly when outlen=64
 * (0x01010000 ^ 0x40 = 0x01010040). */
static inline void template_init(template_state *st) {
    for (int i = 0; i < 8; i++) st->h_ulong[i] = B2B_IV[i];
    st->h_ulong[0] ^= 0x01010040UL;   /* digest_length=64, key=0, fanout=1, depth=1 */
    st->t[0] = 0UL; st->t[1] = 0UL;
    st->f[0] = 0UL; st->f[1] = 0UL;
}

/* template_transform: absorb one full 128-byte block as NON-FINAL. */
static inline void template_transform(template_state *st,
                                      const uchar *block)
{
    ulong t0 = st->t[0] + (ulong)HASH_BLOCK_BYTES;
    ulong carry = (t0 < st->t[0]) ? 1UL : 0UL;
    st->t[0] = t0;
    st->t[1] += carry;
    b2b_compress(&st->h_ulong[0], block, st->t[0], st->t[1], 0);
}

/* template_finalize: process all input start-to-end, advance counter,
 * zero-pad final block, run final compression with last=1.
 *
 * Same shape as gpu_blake2b256_core.cl but populates the FULL 16-uint32
 * exposed digest via template_state_to_h_b512 (8 ulong → 16 uint32 LE).
 * Matches mdxfind.c JOB_BLAKE2B512 (line 30722-30730) and the host
 * blake2b_hash(out, 64, ...) byte image. */
static inline void template_finalize(template_state *st,
                                     const uchar *data, int len)
{
    int pos = 0;
    while ((len - pos) > HASH_BLOCK_BYTES) {
        ulong t0 = st->t[0] + (ulong)HASH_BLOCK_BYTES;
        ulong carry = (t0 < st->t[0]) ? 1UL : 0UL;
        st->t[0] = t0;
        st->t[1] += carry;
        b2b_compress(&st->h_ulong[0], data + pos, st->t[0], st->t[1], 0);
        pos += HASH_BLOCK_BYTES;
    }

    int rem = len - pos;
    uchar buf[HASH_BLOCK_BYTES];
    for (int i = 0; i < rem; i++) buf[i] = data[pos + i];
    for (int i = rem; i < HASH_BLOCK_BYTES; i++) buf[i] = 0;

    ulong t0 = st->t[0] + (ulong)rem;
    ulong carry = (t0 < st->t[0]) ? 1UL : 0UL;
    st->t[0] = t0;
    st->t[1] += carry;

    st->f[0] = 0xFFFFFFFFFFFFFFFFUL;
    b2b_compress(&st->h_ulong[0], buf, st->t[0], st->t[1], 1);

    template_state_to_h_b512(st);
}

/* template_iterate: re-hash 64-byte digest as 128-byte hex_lc and BLAKE2B-512
 * the hex string. Mirrors mdxfind.c JOB_BLAKE2B512 (line 30722-30729):
 * blake2b_hash(curin.h, 64, ...); len = 128; prmd5(curin.h, mdbuf, 128)
 * writes 128 lowercase hex chars; next iter blake2b's those 128 bytes.
 *
 * 128 hex chars exactly equals HASH_BLOCK_BYTES (128). Per the strict-
 * greater "while ((len - pos) > HASH_BLOCK_BYTES)" rule in finalize, the
 * 128-byte hex string is the FINAL block directly — single compression
 * with last=1, counter=128, no preceding complete-block iterations. */
static inline void template_iterate(template_state *st)
{
    uint snap[HASH_WORDS];
    for (int i = 0; i < HASH_WORDS; i++) snap[i] = st->h[i];

    uchar buf[HASH_BLOCK_BYTES];
    /* Hex-encode 16 LE uint32 -> 128 lowercase hex chars. */
    for (int i = 0; i < 16; i++) {
        uint s = snap[i];
        for (int b = 0; b < 4; b++) {
            uchar by = (uchar)((s >> (b * 8)) & 0xFFu);
            uchar hi = by >> 4, lo = by & 0xFu;
            buf[i * 8 + b * 2 + 0] = (hi < 10) ? (uchar)('0' + hi) : (uchar)('a' + (hi - 10));
            buf[i * 8 + b * 2 + 1] = (lo < 10) ? (uchar)('0' + lo) : (uchar)('a' + (lo - 10));
        }
    }
    /* No zero-padding needed: 128 hex chars exactly fill the 128-byte block. */

    /* Reset state for the new compression. */
    for (int i = 0; i < 8; i++) st->h_ulong[i] = B2B_IV[i];
    st->h_ulong[0] ^= 0x01010040UL;
    st->t[0] = 128UL; st->t[1] = 0UL;
    st->f[0] = 0xFFFFFFFFFFFFFFFFUL; st->f[1] = 0UL;

    b2b_compress(&st->h_ulong[0], buf, 128UL, 0UL, 1);

    template_state_to_h_b512(st);
}

/* template_digest_compare: probe leading 16 bytes (h[0..3]). LE state
 * convention; no bswap32. */
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

/* template_emit_hit: EMIT_HIT_16 wrapper (BLAKE2B-512 = 16 uint32 LE). */
#define template_emit_hit(hits, hit_count, max_hits, st, widx, sidx, iter) \
    EMIT_HIT_16((hits), (hit_count), (max_hits), \
                (widx), (sidx), (iter), ((st)->h))

#define template_emit_hit_or_overflow(hits, hit_count, max_hits, st, widx, sidx, iter, \
                                      hashes_shown, matched_idx, dedup_mask, \
                                      ovr_set, ovr_gid, lane_gid) \
    EMIT_HIT_16_DEDUP_OR_OVERFLOW((hits), (hit_count), (max_hits), \
               (widx), (sidx), (iter), ((st)->h), \
               (hashes_shown), (matched_idx), (dedup_mask), \
               (ovr_set), (ovr_gid), (lane_gid))
