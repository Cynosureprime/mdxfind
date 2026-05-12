/*
 * $Revision: 1.2 $
 * $Log: gpu_blake2b256_core.cl,v $
 * Revision 1.2  2026/05/11 05:22:01  dlr
 * Backfill $Revision/$Log RCS keyword stanzas per feedback_rcs_keyword_stanzas.md. Passive 4-line comment block at top of file; no behavioral change. Hand-authored .cl file was missing required stanzas (per memory: all hand-authored .c/.h/.cl/.frag/.tmpl/.py/.sh files MUST contain $Revision/$Log keyword stanzas). Build green on .205 against the post-add files; OpenCL compile strips comments so no kernel behavior change.
 *
 */
/* gpu_blake2b256_core.cl — BLAKE2B-256 algorithm extension functions for
 * the generic dispatch template (Memo B Phase B5 sub-batch 3).
 *
 * BLAKE2B-256 is BLAKE2b with digest_length=32 (256 bits). The compression
 * function is identical to BLAKE2B-512; only the parameter-block init and
 * output truncation differ.
 *
 * Geometry / byte ordering:
 *   - 128-byte block (vs BLAKE2S-256's 64-byte)
 *   - 8 × uint64 chaining state, LITTLE-ENDIAN
 *   - 12-round G-function compression (b2b_compress in gpu_common.cl)
 *   - parameter block init: IV[0] ^ 0x01010020
 *     (digest_length=32 << 0 | key_length=0 << 8 | fanout=1 << 16 |
 *      depth=1 << 24 — high 32 bits zero)
 *   - 32-byte output = first 4 ulong = 8 uint32 LE (truncated from full
 *     64-byte 8-ulong digest)
 *
 *   HASH_WORDS         — digest size in 32-bit words (8 for BLAKE2B-256)
 *   HASH_BLOCK_BYTES   — compress-block size (128)
 *   template_state     — h[8] ulong digest + t[2] ulong counter + f[2]
 *                        ulong flag, plus exposed h_uint[HASH_WORDS] for
 *                        template body's digests_out and EMIT_HIT_8
 *
 * The state struct INTERNALLY carries 8 ulong h_ulong[8] (BLAKE2b chaining)
 * but the template body needs uint32 access via .h[]. This mirrors
 * gpu_sha512_core.cl rev 1.1 exactly: the per-uint64 state is decomposed
 * into a uint h[HASH_WORDS] array at the end of finalize/iterate so the
 * template body and EMIT_HIT macros work without 64-bit indirection.
 *
 *   template_init      — install IV ^ parameter_block (h_ulong); zero
 *                        counter+flag
 *   template_transform — absorb one 128-byte block (NON-FINAL)
 *   template_finalize  — process complete blocks, zero tail, run final
 *                        compression with last==1; populate h[8] = first
 *                        4 ulong as 8 uint32 LE
 *   template_iterate   — re-hash digest as 64-byte hex_lc input (32-byte
 *                        binary -> 64 lc hex chars). 64 < 128 so the hex
 *                        string fits in one BLAKE2b block; single
 *                        compression with last=1, counter=64
 *   template_digest_compare — probe leading 16 bytes (h[0..3]) directly;
 *                             LE state convention.
 *   template_emit_hit       — EMIT_HIT_8 wrapper (8 uint32 LE; first 4
 *                             ulong of BLAKE2B output decomposed)
 *
 * Source order at compile time:
 *
 *   [ gpu_common_str, gpu_md5_rules_str, gpu_blake2b256_core_str, gpu_template_str ]
 *
 * gpu_common_str provides B2B_IV[8], B2B_SIGMA[12][16], b2b_compress(),
 * EMIT_HIT_8{,_DEDUP_OR_OVERFLOW}, and probe_compact_idx.
 *
 * Bytecast invariants:
 *   - Internal h_ulong[0..7] is LE per ulong; output bytes laid out
 *     little-endian (matches mdxfind.c blake2b_hash() at line 2446 which
 *     uses memcpy(out, h, outlen) — direct ulong byte image).
 *   - Decomposition to st->h[0..7] uint32 LE pairs:
 *       st->h[2i]   = (uint)h_ulong[i]
 *       st->h[2i+1] = (uint)(h_ulong[i] >> 32)
 *     This is the LE byte image of each ulong split into two uint32
 *     numerical halves. Probe key st->h[0..3] therefore corresponds to
 *     the leading 16 hash bytes — same convention as BLAKE2s256, MD5,
 *     and the host compact-table format.
 *   - Wire format matches gpu_blake2s256unsalted.cl emit (8 uint32 LE)
 *     so the hit-replay path in gpujob_opencl.c reads the same 32-byte
 *     binary digest regardless of which algorithm produced it.
 *
 * R1 mitigation: single-private-buffer pattern. b2b_compress takes
 * __private uchar* + __private ulong* (existing primitives in gpu_common.cl).
 *
 * R2 (register pressure): BLAKE2B's 8 ulong chaining + 16 ulong v[] +
 * 16 ulong m[] in b2b_compress = ~80 64-bit registers ~= 160 32-bit
 * registers. May approach the gfx1201 VGPR ceiling. Memo §3 R2 flagged
 * SHA-512 family as the boundary case (~50 VGPR); BLAKE2b is similar.
 * Per-phase clGetKernelWorkGroupInfo(CL_KERNEL_PRIVATE_MEM_SIZE) probe
 * provides the actual reading; SHA-512's 42,520 B on gfx1201 is the
 * comparison baseline.
 */

/* Per-algorithm geometry. */
#ifndef HASH_WORDS
#define HASH_WORDS 8
#endif
#ifndef HASH_BLOCK_BYTES
#define HASH_BLOCK_BYTES 128
#endif

/* Per-lane state struct. h_ulong[] is the 8-ulong BLAKE2b chaining state
 * used internally by b2b_compress; h[] is the LE-uint32-decomposed view
 * exposed to the template body and EMIT_HIT macros.
 *
 * Counter t[] and flag f[] are kept as ulong pairs to mirror RFC 7693
 * BLAKE2b's spec layout (BLAKE2b uses 128-bit counter / flag pair; for
 * our wordlist inputs t[1] always = 0). */
typedef struct {
    ulong h_ulong[8];     /* internal chaining state, LE per ulong */
    ulong t[2];           /* 128-bit byte counter (low, high) */
    ulong f[2];           /* finalization flag pair */
    uint  h[HASH_WORDS];  /* exposed digest words (template body reads h[]) */
} template_state;

/* Internal helper: decompose first 4 ulong of internal state into 8 LE
 * uint32 for the exposed h[] (truncated to digest_length=32 bytes). */
static inline void template_state_to_h_b256(template_state *st) {
    for (int i = 0; i < 4; i++) {
        ulong v = st->h_ulong[i];
        st->h[i * 2]     = (uint)v;
        st->h[i * 2 + 1] = (uint)(v >> 32);
    }
}

/* template_init: install BLAKE2B-256 IV XOR'd with parameter block.
 *
 * Parameter block layout (RFC 7693 §2.5, BLAKE2b version):
 *   bytes 0:    digest_length = 32
 *   byte 1:     key_length = 0
 *   byte 2:     fanout = 1
 *   byte 3:     depth = 1
 *   bytes 4..7: leaf_length = 0
 *   bytes 8..15: node_offset = 0  (BLAKE2b widens to 64-bit here)
 *   byte 16:    node_depth = 0
 *   byte 17:    inner_length = 0
 *   bytes 18..31: reserved = 0
 *   bytes 32..47: salt = 0
 *   bytes 48..63: personalization = 0
 *
 * Param block as 8 ulong LE: P[0] = 0x0000000001010020 (digest=32, key=0,
 * fanout=1, depth=1, leaf=0); P[1..7] = 0. So h_ulong[0] = IV[0] ^
 * 0x01010020; h_ulong[1..7] = IV[1..7]. Matches mdxfind.c blake2b_hash
 * (line 2431) parameter-block formula `0x01010000 ^ outlen` exactly when
 * outlen=32 (0x01010000 ^ 0x20 = 0x01010020). */
static inline void template_init(template_state *st) {
    for (int i = 0; i < 8; i++) st->h_ulong[i] = B2B_IV[i];
    st->h_ulong[0] ^= 0x01010020UL;   /* digest_length=32, key=0, fanout=1, depth=1 */
    st->t[0] = 0UL; st->t[1] = 0UL;
    st->f[0] = 0UL; st->f[1] = 0UL;
}

/* template_transform: absorb one full 128-byte block as NON-FINAL.
 * Public extension API; template_finalize hot path bypasses for perf. */
static inline void template_transform(template_state *st,
                                      const uchar *block)
{
    /* Advance 128-bit counter by full block size. */
    ulong t0 = st->t[0] + (ulong)HASH_BLOCK_BYTES;
    ulong carry = (t0 < st->t[0]) ? 1UL : 0UL;
    st->t[0] = t0;
    st->t[1] += carry;
    b2b_compress(&st->h_ulong[0], block, st->t[0], st->t[1], 0);
}

/* template_finalize: process all input start-to-end, advance counter,
 * zero-pad final block, run final compression with last=1.
 *
 * Same shape as gpu_blake2s256_core.cl template_finalize but with 128-byte
 * blocks and ulong counter. The "len > HASH_BLOCK_BYTES strictly greater"
 * loop preserves the BLAKE2 invariant: the final block runs with last=1
 * even when len is an exact multiple of 128.
 *
 * After return, h_ulong[0..7] holds the final BLAKE2B-256 chaining state;
 * the first 4 ulong contain the 32-byte output. h[0..7] (8 uint32 LE) is
 * populated by template_state_to_h_b256 for emit. */
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

    /* Decompose first 4 ulong → 8 uint32 LE for template body / emit. */
    template_state_to_h_b256(st);
}

/* template_iterate: re-hash 32-byte digest as 64-byte hex_lc and BLAKE2B-256
 * the hex string. Mirrors mdxfind.c JOB_BLAKE2B256 (line 30821-30828):
 * blake2b_hash(curin.h, 32, ...); len = 64; prmd5(curin.h, mdbuf, 64)
 * writes 64 lowercase hex chars; next iter blake2b's those 64 bytes.
 *
 * 64 hex chars < HASH_BLOCK_BYTES (128), so the hex string fits one
 * partial 128-byte block with 64 zero-padding bytes. Single compression
 * with last=1, counter=64.
 *
 * Snapshot the 8 LE uint32 from h[] BEFORE the IV reset. */
static inline void template_iterate(template_state *st)
{
    uint snap[HASH_WORDS];
    for (int i = 0; i < HASH_WORDS; i++) snap[i] = st->h[i];

    uchar buf[HASH_BLOCK_BYTES];
    /* Hex-encode 8 LE uint32 -> 64 lowercase hex chars (matches prmd5
     * byte order: byte 0 of each uint32 is hex'd first). */
    for (int i = 0; i < 8; i++) {
        uint s = snap[i];
        for (int b = 0; b < 4; b++) {
            uchar by = (uchar)((s >> (b * 8)) & 0xFFu);
            uchar hi = by >> 4, lo = by & 0xFu;
            buf[i * 8 + b * 2 + 0] = (hi < 10) ? (uchar)('0' + hi) : (uchar)('a' + (hi - 10));
            buf[i * 8 + b * 2 + 1] = (lo < 10) ? (uchar)('0' + lo) : (uchar)('a' + (lo - 10));
        }
    }
    /* Zero the rest of the 128-byte block (64..127). */
    for (int i = 64; i < HASH_BLOCK_BYTES; i++) buf[i] = 0;

    /* Reset state for the new compression. */
    for (int i = 0; i < 8; i++) st->h_ulong[i] = B2B_IV[i];
    st->h_ulong[0] ^= 0x01010020UL;
    st->t[0] = 64UL; st->t[1] = 0UL;
    st->f[0] = 0xFFFFFFFFFFFFFFFFUL; st->f[1] = 0UL;

    b2b_compress(&st->h_ulong[0], buf, 64UL, 0UL, 1);

    template_state_to_h_b256(st);
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

/* template_emit_hit: EMIT_HIT_8 wrapper (BLAKE2B-256 truncated digest =
 * 8 uint32 LE words, populated by template_state_to_h_b256). */
#define template_emit_hit(hits, hit_count, max_hits, st, widx, sidx, iter) \
    EMIT_HIT_8((hits), (hit_count), (max_hits), \
               (widx), (sidx), (iter), ((st)->h))

#define template_emit_hit_or_overflow(hits, hit_count, max_hits, st, widx, sidx, iter, \
                                      hashes_shown, matched_idx, dedup_mask, \
                                      ovr_set, ovr_gid, lane_gid) \
    EMIT_HIT_8_DEDUP_OR_OVERFLOW((hits), (hit_count), (max_hits), \
               (widx), (sidx), (iter), ((st)->h), \
               (hashes_shown), (matched_idx), (dedup_mask), \
               (ovr_set), (ovr_gid), (lane_gid))
