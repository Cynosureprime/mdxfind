/*
 * $Revision: 1.2 $
 * $Log: gpu_sha3_256_core.cl,v $
 * Revision 1.2  2026/05/11 05:22:02  dlr
 * Backfill $Revision/$Log RCS keyword stanzas per feedback_rcs_keyword_stanzas.md. Passive 4-line comment block at top of file; no behavioral change. Hand-authored .cl file was missing required stanzas (per memory: all hand-authored .c/.h/.cl/.frag/.tmpl/.py/.sh files MUST contain $Revision/$Log keyword stanzas). Build green on .205 against the post-add files; OpenCL compile strips comments so no kernel behavior change.
 *
 */
/* gpu_sha3_256_core.cl - SHA3-256 algorithm extension functions for the
 * generic dispatch template (Memo B Phase B5 sub-batch 4, 2026-05-03).
 *
 * Sponge construction with rate=136 bytes (= 17 ulong), output=32 bytes,
 * domain suffix=0x06 (NIST FIPS 202 SHA3 family). Distinct from plain
 * Keccak-256 ONLY in the suffix byte. HASH_WORDS=8, HASH_BLOCK_BYTES=136.
 *
 * Source order: [ gpu_common_str, gpu_md5_rules_str, gpu_sha3_256_core_str,
 *   gpu_template_str ].
 *
 * Bytecast invariant: sp[0..3] LE byte image == rhash_sha3 output (mdxfind.c
 * JOB_SHA3_256 line 28779). gpu_hash_words(JOB_SHA3_256) returns 8.
 */

#ifndef HASH_WORDS
#define HASH_WORDS 8
#endif
#ifndef HASH_BLOCK_BYTES
#define HASH_BLOCK_BYTES 136
#endif

#define KECCAK_RATE       136
#define KECCAK_RATE_WORDS  17
#define KECCAK_OUT_BYTES   32
#define KECCAK_DOMAIN_PAD  0x06

typedef struct {
    ulong sp[25];
    uint  h[HASH_WORDS];
} template_state;

static inline void template_state_to_h_sha3_256(template_state *st) {
    for (int i = 0; i < HASH_WORDS / 2; i++) {
        ulong v = st->sp[i];
        st->h[i * 2]     = (uint)v;
        st->h[i * 2 + 1] = (uint)(v >> 32);
    }
}

static inline void sha3_256_absorb_pad(ulong *sp, const uchar *data, int rem) {
    uchar block[KECCAK_RATE];
    for (int i = 0; i < rem; i++) block[i] = data[i];
    for (int i = rem; i < KECCAK_RATE; i++) block[i] = 0;
    block[rem]            ^= (uchar)KECCAK_DOMAIN_PAD;
    block[KECCAK_RATE - 1] ^= (uchar)0x80;
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

static inline void sha3_256_absorb_full(ulong *sp, const uchar *data) {
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

static inline void template_init(template_state *st) {
    for (int i = 0; i < 25; i++) st->sp[i] = 0UL;
}

static inline void template_transform(template_state *st, const uchar *block) {
    sha3_256_absorb_full(&st->sp[0], block);
}

static inline void template_finalize(template_state *st,
                                     const uchar *data, int len)
{
    int pos = 0;
    while ((len - pos) >= KECCAK_RATE) {
        sha3_256_absorb_full(&st->sp[0], data + pos);
        pos += KECCAK_RATE;
    }
    int rem = len - pos;
    sha3_256_absorb_pad(&st->sp[0], data + pos, rem);
    template_state_to_h_sha3_256(st);
}

static inline void template_iterate(template_state *st)
{
    uint snap[HASH_WORDS];
    for (int i = 0; i < HASH_WORDS; i++) snap[i] = st->h[i];

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

    for (int i = 0; i < 25; i++) st->sp[i] = 0UL;
    int pos = 0;
    while ((hex_chars - pos) >= KECCAK_RATE) {
        sha3_256_absorb_full(&st->sp[0], hex + pos);
        pos += KECCAK_RATE;
    }
    int rem = hex_chars - pos;
    sha3_256_absorb_pad(&st->sp[0], hex + pos, rem);
    template_state_to_h_sha3_256(st);
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
    return probe_compact_idx(
        st->h[0], st->h[1], st->h[2], st->h[3],
        compact_fp, compact_idx,
        compact_mask, max_probe, hash_data_count,
        hash_data_buf, hash_data_off,
        overflow_keys, overflow_hashes, overflow_offsets, overflow_count,
        out_idx);
}

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
