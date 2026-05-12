/*
 * $Revision: 1.2 $
 * $Log: gpu_keccak256_core.cl,v $
 * Revision 1.2  2026/05/11 05:22:01  dlr
 * Backfill $Revision/$Log RCS keyword stanzas per feedback_rcs_keyword_stanzas.md. Passive 4-line comment block at top of file; no behavioral change. Hand-authored .cl file was missing required stanzas (per memory: all hand-authored .c/.h/.cl/.frag/.tmpl/.py/.sh files MUST contain $Revision/$Log keyword stanzas). Build green on .205 against the post-add files; OpenCL compile strips comments so no kernel behavior change.
 *
 */
/* gpu_keccak256_core.cl - Keccak-256 algorithm extension functions for the
 * generic dispatch template (Memo B Phase B5 sub-batch 4, 2026-05-03).
 *
 * Sponge construction with rate=136 bytes (= 17 ulong), output=32 bytes,
 * domain suffix=0x01 (plain Keccak; SHA3 uses 0x06). HASH_WORDS=8 = 8 uint32
 * LE per slab-kernel convention; HASH_BLOCK_BYTES=136 = rate (the unit of
 * data the kernel absorbs per Keccak-f[1600] call).
 *
 * Geometry / byte ordering:
 *   - 1600-bit state (5x5 = 25 ulong), naturally LITTLE-ENDIAN per ulong
 *   - 24-round Keccak-f[1600] permutation (keccakf1600 in gpu_common.cl)
 *   - Padding: append suffix (0x01 / 0x06) at offset (len % rate), then 0x80
 *     at offset (rate-1), then absorb final block.
 *   - Output: first HASH_WORDS uint32 of state, decomposed as
 *       h[2i]   = (uint)sp[i]
 *       h[2i+1] = (uint)(sp[i] >> 32)
 *     Same byte image as mdxfind.c sph_keccak256_close + memcpy(curin.h, ..., 32).
 *
 * Source order at compile time (from gpu_opencl.c
 * gpu_opencl_template_compile_keccak256):
 *
 *   [ gpu_common_str, gpu_md5_rules_str, gpu_keccak256_core_str, gpu_template_str ]
 *
 * gpu_common_str provides keccakf1600() + KECCAK_RC[24] + KECCAK_ROTC[25]
 * (rev 1.14+) plus EMIT_HIT_8 + probe_compact_idx.
 *
 * Bytecast invariants:
 *   - sp[0..24] is LE per ulong; the first 4 ulong (32 bytes) hold the
 *     digest. mdxfind.c JOB_KECCAK256 (line 28791) copies 32 bytes from
 *     sph_keccak256_close - these are the LE byte image of sp[0..3].
 *   - Hit-replay (gpujob_opencl.c) reads gpu_hash_words(JOB_KECCAK256)*4 = 32
 *     bytes from hits[]; gpu_hash_words(JOB_KECCAK256) returns 8 (already in
 *     gpujob_opencl.c rev 1.85). No host-side change needed for this op.
 *
 * R1 mitigation: single-private-buffer pattern. R2 (register pressure):
 * Keccak's 25-ulong state (200 bytes) plus 25-ulong B[] scratch in keccakf
 * is dominant; expected priv_mem_size on gfx1201 around 41-43 KB
 * (compare against SHA-512 baseline 42,520 B).
 */

/* Per-algorithm geometry. Cache key: defines_str
 * "HASH_WORDS=8,HASH_BLOCK_BYTES=136" - distinct source text from BLAKE2B-256
 * (also HASH_WORDS=8, but HASH_BLOCK_BYTES=128) so cache key is unique. */
#ifndef HASH_WORDS
#define HASH_WORDS 8
#endif
#ifndef HASH_BLOCK_BYTES
#define HASH_BLOCK_BYTES 136
#endif

/* Per-algo constants for Keccak-256. */
#define KECCAK_RATE       136
#define KECCAK_RATE_WORDS  17   /* 136/8 */
#define KECCAK_OUT_BYTES   32
#define KECCAK_DOMAIN_PAD  0x01

typedef struct {
    ulong sp[25];          /* sponge state, LE per ulong */
    uint  h[HASH_WORDS];   /* exposed digest words, 8 uint32 LE */
} template_state;

/* Decompose state to exposed digest. KECCAK_OUT_BYTES / 4 = HASH_WORDS uint32. */
static inline void template_state_to_h_keccak256(template_state *st) {
    for (int i = 0; i < HASH_WORDS / 2; i++) {
        ulong v = st->sp[i];
        st->h[i * 2]     = (uint)v;
        st->h[i * 2 + 1] = (uint)(v >> 32);
    }
}

/* Absorb the final (possibly partial) block plus padding.
 * data points to the partial-block slice (0..rem-1 bytes, where rem == len % rate).
 * Caller has already absorbed all complete rate-aligned blocks. */
static inline void keccak256_absorb_pad(ulong *sp, const uchar *data, int rem) {
    /* Build the final block in private memory: copy partial bytes, zero the
     * rest, then OR in suffix at offset rem and 0x80 at offset rate-1.
     * Note: positions rem and rate-1 may collide when rem == rate-1; in that
     * case the byte ends up suffix | 0x80 (per Keccak/SHA3 spec). */
    uchar block[KECCAK_RATE];
    for (int i = 0; i < rem; i++) block[i] = data[i];
    for (int i = rem; i < KECCAK_RATE; i++) block[i] = 0;
    block[rem]            ^= (uchar)KECCAK_DOMAIN_PAD;
    block[KECCAK_RATE - 1] ^= (uchar)0x80;
    /* XOR block into state as LE ulong words. */
    for (int i = 0; i < KECCAK_RATE_WORDS; i++) {
        ulong w = 0;
        int b = i * 8;
        w |= (ulong)block[b];
        w |= (ulong)block[b + 1] << 8;
        w |= (ulong)block[b + 2] << 16;
        w |= (ulong)block[b + 3] << 24;
        w |= (ulong)block[b + 4] << 32;
        w |= (ulong)block[b + 5] << 40;
        w |= (ulong)block[b + 6] << 48;
        w |= (ulong)block[b + 7] << 56;
        sp[i] ^= w;
    }
    keccakf1600(sp);
}

/* Absorb one complete rate-byte block (no padding). */
static inline void keccak256_absorb_full(ulong *sp, const uchar *data) {
    for (int i = 0; i < KECCAK_RATE_WORDS; i++) {
        ulong w = 0;
        int b = i * 8;
        w |= (ulong)data[b];
        w |= (ulong)data[b + 1] << 8;
        w |= (ulong)data[b + 2] << 16;
        w |= (ulong)data[b + 3] << 24;
        w |= (ulong)data[b + 4] << 32;
        w |= (ulong)data[b + 5] << 40;
        w |= (ulong)data[b + 6] << 48;
        w |= (ulong)data[b + 7] << 56;
        sp[i] ^= w;
    }
    keccakf1600(sp);
}

/* template_init: zero the 25-ulong sponge state. */
static inline void template_init(template_state *st) {
    for (int i = 0; i < 25; i++) st->sp[i] = 0UL;
}

/* template_transform: NOT USED on the sponge path - the template only calls
 * template_init / template_finalize / template_iterate. Provided as a stub
 * for interface symmetry with MD-style cores (call-site coverage check). */
static inline void template_transform(template_state *st, const uchar *block) {
    keccak256_absorb_full(&st->sp[0], block);
}

/* template_finalize: process all rate-aligned chunks via absorb-and-permute,
 * then pad the final partial block with the algorithm's domain suffix and
 * absorb. Extract h[] = first HASH_WORDS uint32 LE of state. */
static inline void template_finalize(template_state *st,
                                     const uchar *data, int len)
{
    int pos = 0;
    while ((len - pos) >= KECCAK_RATE) {
        keccak256_absorb_full(&st->sp[0], data + pos);
        pos += KECCAK_RATE;
    }
    int rem = len - pos;
    keccak256_absorb_pad(&st->sp[0], data + pos, rem);
    template_state_to_h_keccak256(st);
}

/* template_iterate: re-hash digest as 2*KECCAK_OUT_BYTES lowercase hex chars.
 * For Keccak-256 / SHA3-256: 64 hex chars; less than rate=136 so single
 * partial-block absorb suffices. To keep the helper general (e.g. shared
 * with longer-output variants where 2*out > rate), we run the same
 * complete-block absorb loop then pad-and-absorb the remainder. */
static inline void template_iterate(template_state *st)
{
    uint snap[HASH_WORDS];
    for (int i = 0; i < HASH_WORDS; i++) snap[i] = st->h[i];

    /* hex_lc encode HASH_WORDS uint32 LE -> 2*KECCAK_OUT_BYTES chars. */
    int hex_chars = 2 * KECCAK_OUT_BYTES;
    uchar hex[2 * KECCAK_OUT_BYTES];
    for (int i = 0; i < HASH_WORDS; i++) {
        uint s = snap[i];
        for (int b = 0; b < 4; b++) {
            uchar by = (uchar)((s >> (b * 8)) & 0xFFu);
            uchar hi = by >> 4, lo = by & 0xFu;
            hex[i * 8 + b * 2 + 0] = (hi < 10) ? (uchar)('0' + hi) : (uchar)('a' + (hi - 10));
            hex[i * 8 + b * 2 + 1] = (lo < 10) ? (uchar)('0' + lo) : (uchar)('a' + (lo - 10));
        }
    }

    /* Re-init sponge and absorb hex_chars with padding. */
    for (int i = 0; i < 25; i++) st->sp[i] = 0UL;
    int pos = 0;
    while ((hex_chars - pos) >= KECCAK_RATE) {
        keccak256_absorb_full(&st->sp[0], hex + pos);
        pos += KECCAK_RATE;
    }
    int rem = hex_chars - pos;
    keccak256_absorb_pad(&st->sp[0], hex + pos, rem);
    template_state_to_h_keccak256(st);
}

/* template_digest_compare: probe leading 16 bytes (h[0..3]). LE state. */
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

/* template_emit_hit: EMIT_HIT_8 wrapper (Keccak-256 = 8 uint32 LE). */
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
