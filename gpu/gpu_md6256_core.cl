/*
 * $Revision: 1.2 $
 * $Log: gpu_md6256_core.cl,v $
 * Revision 1.2  2026/05/11 05:22:01  dlr
 * Backfill $Revision/$Log RCS keyword stanzas per feedback_rcs_keyword_stanzas.md. Passive 4-line comment block at top of file; no behavioral change. Hand-authored .cl file was missing required stanzas (per memory: all hand-authored .c/.h/.cl/.frag/.tmpl/.py/.sh files MUST contain $Revision/$Log keyword stanzas). Build green on .205 against the post-add files; OpenCL compile strips comments so no kernel behavior change.
 *
 */
/* gpu_md6256_core.cl — MD6-256 (hashcat -m 17800, JOB_MD6256=29) algorithm
 * extension functions for the generic dispatch template (Memo B Phase B7.7b).
 *
 * STATUS: B7.7b — final M5 closure (renders gpu_md6256unsalted.cl retireable
 * after B7.8 mask cap widening). MD6 is the algorithmically-largest single-
 * compression unsalted algo on the unified template path: 89-ulong N input,
 * 1753-ulong A working array (= 14,024 bytes per work-item), 104 rounds × 16
 * words = 1,664 main-loop steps per compression.
 *
 * MD6-256 = md6_hash(d=256, data, len*8). For lengths fitting one leaf block
 * (b=64 ulong = 512 bytes; the slab path's maxpasslen=511 covers all in-scope
 * inputs), this is a SINGLE compression call followed by digest extraction
 * from C[12..15] (last 4 ulong = 32 bytes = 256 bits).
 *
 * CPU reference (mdxfind.c JOB_MD6256 at lines 25836-25855):
 *
 *   case JOB_MD6256: i = 256; goto MD6_start;
 *   ...
 *   MD6_start:
 *     for (x = 1; x <= Maxiter; x++) {
 *       md6_hash(i, (unsigned char *) cur, len * 8, curin.h);
 *       len = i / 4;                            // 64 hex chars for d=256
 *       cur = prmd5(curin.h, mdbuf, len);       // lowercase hex
 *       hashcnt++;
 *       checkhash(&curin, len, x, job);
 *     }
 *
 * Per-iter behavior:
 *   - At iter N, input = (N==1 ? rule-mutated buf : LOWERCASE_HEX(64) of
 *     prior iter's 32-byte digest).
 *   - Compute md6_256_N = md6_hash(input). Probe md6_256_N against compact
 *     table.
 *   - If iter < Maxiter, prepare input_{N+1} = LOWERCASE_HEX(md6_256_N).
 *
 * DESIGN: per-iter probe like SQL5 — template_finalize does the FIRST md6
 * compression; template_iterate does the hex-feedback then the next md6
 * compression. The kernel's outer iter loop in template_phase0 calls
 * template_digest_compare after every iter and template_iterate before the
 * next. This is the SQL5 pattern (gpu_sql5_core.cl) — distinct from the
 * SHA1DRU pattern (1M loop inside template_finalize, max_iter=1) because
 * MD6256 honors user `-i` semantics.
 *
 * Per-algorithm state struct holds the 32-byte md6_256 digest as 4 ulong
 * (BE words, LE-byte-emit). Probe converts to 8 uint LE for the compact
 * table compare.
 *
 * The compression scratch (1753-ulong working array A) lives on the
 * function-local stack in template_finalize / template_iterate — NOT in
 * template_state (which would balloon every algo's per-thread state).
 *
 * Cache key (R3): defines_str = "HASH_WORDS=8,HASH_BLOCK_BYTES=64,
 * BASE_ALGO=md6". HASH_WORDS=8 (256-bit digest = 8 uint32). HASH_BLOCK_BYTES
 * is largely advisory for MD6 (the kernel processes the WHOLE input as one
 * 64-ulong = 512-byte leaf block); 64 chosen to align with default. Distinct
 * cache entry by source-text hash regardless.
 *
 * KNOWN ACCEPTED RISK (per architect §6 of project_b76plus_mask_iter_-
 * closure.md): the 1753-ulong A working array (14 KB) plus the standard
 * RULE_BUF_MAX (~40 KB) is expected to bust gfx1201's 43,024 B HARD GATE.
 * Architect predicted +800 to +1500 B over the SHA1DRU baseline of 41,328 B;
 * the actual cost may be substantially higher because of the 14 KB A[1753]
 * stack (the SHA1DRU baseline doesn't have it). User OPTION A directs
 * compile-only ship; integrated post-B7.9 validation will reveal whether
 * gfx1201 loses this kernel. Fall-back if so: leave gpu_md6256unsalted.cl
 * in tree as gfx1201-only kernel routed via slab path.
 *
 * R1 mitigation (AMD ROCm comgr addrspace): single private buffer pattern.
 * No __private uchar* helpers; no addrspace-cast ternaries. The A[] array
 * is a function-local automatic; OpenCL spec promises private addrspace
 * for stack locals (no special handling needed).
 *
 * VALIDATION ORACLE: existing slab kernel md6_256_unsalted_batch in
 * gpu_md6256unsalted.cl. The compression body of THIS file is a port of
 * that slab kernel's md6_compress_loop (lines 44-78 pre-B7.7b) plus the
 * N[89] packer (lines 99-110, 241-256 pre-B7.7b) into the template
 * extension API. Differences from the slab kernel:
 *   - reads pass from `data` (post-rule buf passed by template) rather
 *     than from a pre-packed N[89] block built by the host.
 *   - the iter feedback hex encoding of the digest happens in template_-
 *     iterate (vs. inline in the slab kernel's iter loop).
 *   - compact-table probe uses 8 LE uint32 (HASH_WORDS=8) — was hand-rolled
 *     EMIT_HIT_8 in the slab path; template path uses the standard
 *     probe_compact_idx + EMIT_HIT macros.
 *
 * Source order at compile time (mirrors other unsalted cores):
 *   [ gpu_common_str, gpu_md5_rules_str, gpu_md6256_core_str, gpu_template_str ]
 */

#ifndef HASH_WORDS
#define HASH_WORDS 8
#endif
#ifndef HASH_BLOCK_BYTES
#define HASH_BLOCK_BYTES 64
#endif

/* MD6 compression constants (mirrors gpu_md6256unsalted.cl). */
__constant ulong MD6CORE_QC[15] = {
    0x7311c2812425cfa0UL, 0x6432286434aac8e7UL, 0xb60450e9ef68b7c1UL,
    0xe8fb23908d9f06f1UL, 0xdd2e76cba691e5bfUL, 0x0cd0d63b2c30bc41UL,
    0x1f8ccf6823058f8aUL, 0x54e5ed5b88e3775dUL, 0x4ad12aae0a6d6031UL,
    0x3e7f16bb88222e0dUL, 0x8af8671d3fb50c2cUL, 0x995ad1178bd25c31UL,
    0xc878c1dd04c4b633UL, 0x3b72066c7a1552acUL, 0x0d6f3522631effcbUL
};

#define MD6CORE_n  89
#define MD6CORE_c  16
#define MD6CORE_r  104

#define MD6CORE_t0  17
#define MD6CORE_t1  18
#define MD6CORE_t2  21
#define MD6CORE_t3  31
#define MD6CORE_t4  67
#define MD6CORE_t5  89

__constant ulong MD6CORE_S0    = 0x0123456789abcdefUL;
__constant ulong MD6CORE_Smask = 0x7311c2812425cfa0UL;

#define MD6CORE_STEP(rs, ls, step) \
    x  = S;                                         \
    x ^= A[i + step - MD6CORE_t5];                  \
    x ^= A[i + step - MD6CORE_t0];                  \
    x ^= (A[i + step - MD6CORE_t1] & A[i + step - MD6CORE_t2]); \
    x ^= (A[i + step - MD6CORE_t3] & A[i + step - MD6CORE_t4]); \
    x ^= (x >> rs);                                 \
    A[i + step] = x ^ (x << ls);

/* MD6 main compression loop on a 1753-ulong working array A. The first 89
 * ulong (A[0..88]) hold N (the packed compression input); the remaining
 * r*c=1664 slots are filled by the loop. Output C[16] is at A[1737..1752];
 * the d=256-bit hash is the LAST 4 ulong of C, i.e. A[1749..1752]. */
static inline void md6core_compress_loop(ulong *A) {
    ulong x;
    ulong S = MD6CORE_S0;
    int i = MD6CORE_n;
    for (int j = 0; j < MD6CORE_r * MD6CORE_c; j += MD6CORE_c) {
        MD6CORE_STEP(10, 11,  0)
        MD6CORE_STEP( 5, 24,  1)
        MD6CORE_STEP(13,  9,  2)
        MD6CORE_STEP(10, 16,  3)
        MD6CORE_STEP(11, 15,  4)
        MD6CORE_STEP(12,  9,  5)
        MD6CORE_STEP( 2, 27,  6)
        MD6CORE_STEP( 7, 15,  7)
        MD6CORE_STEP(14,  6,  8)
        MD6CORE_STEP(15,  2,  9)
        MD6CORE_STEP( 7, 29, 10)
        MD6CORE_STEP(13,  8, 11)
        MD6CORE_STEP(11, 15, 12)
        MD6CORE_STEP( 7,  5, 13)
        MD6CORE_STEP( 6, 31, 14)
        MD6CORE_STEP(12,  9, 15)
        S = (S << 1) ^ (S >> 63) ^ (S & MD6CORE_Smask);
        i += 16;
    }
}

/* md6core_pack_and_compress: build the N[89] packed input from `data`
 * bytes of length `len`, run one compression, store the 4-ulong digest
 * (BE within each ulong, MD6's natural state output) into st->h[] as
 * 8 uint32 (BE pair-wise — the bswap to LE happens at digest_compare /
 * emit time).
 *
 * Mirrors mdxfind.c:9361-9385 (host pack) + gpu_md6256unsalted.cl:99-185
 * (slab kernel pack + compress). The single-leaf single-compression path
 * is correct for any len <= 64*8 = 512 bytes (b=64 ulong leaf block). The
 * caller is responsible for clamping `len` to <= 512. */
static inline void md6core_pack_and_compress(uint *digest_words,
                                             const uchar *data, int len)
{
    /* Working array A[1753] = N[89] head + r*c=1664 trailing slots.
     * 14,024 bytes on the kernel stack — same as slab kernel.
     * KNOWN ACCEPTED RISK: gfx1201 priv_mem may bust HARD GATE (architect
     * §6); compile-only ship per user OPTION A. */
    ulong A[1753];

    /* Clamp: leaf block holds b*w = 64 * 64 = 4096 bits = 512 bytes. */
    if (len < 0)   len = 0;
    if (len > 512) len = 512;

    /* N[0..14] = Q (15 fixed constants). */
    for (int i = 0; i < 15; i++) A[i] = MD6CORE_QC[i];
    /* N[15..22] = K = 0 (8 ulong key, unused). */
    for (int i = 15; i < 23; i++) A[i] = 0;
    /* N[23] = U = nodeID(ell=1, i=0) = 1<<56. */
    A[23] = (ulong)1 << 56;
    /* N[24] = V = control_word(r=104, L=64, z=1, p=4096-len*8, keylen=0,
     * d=256). Layout per md6_make_control_word at md6_compress.c:312:
     *   r << 48, L << 40, z << 36, p << 20, keylen << 12, d. */
    int p_bits = 64 * 64 - len * 8;
    A[24] = ((ulong)104 << 48) | ((ulong)64 << 40) | ((ulong)1 << 36)
          | ((ulong)p_bits << 20) | (ulong)256;

    /* N[25..88] = B[64] data block — pack `data` bytes into LE ulong slots
     * starting at byte offset 0 within B. The host's slab packer writes
     * bytes via memcpy() into slot[200..200+len-1] which on a little-endian
     * host translates to LE-packed ulong[]; the slab kernel then bswap64's
     * to MD6's expected BE (per md6_reverse_little_endian at md6_mode.c:
     * 822). We do the same here: build B as LE, then bswap64 in-place. */
    for (int i = 25; i < 89; i++) A[i] = 0;
    for (int i = 0; i < len; i++) {
        int wi = 25 + (i >> 3);
        int bi = (i & 7) << 3;
        A[wi] |= (ulong)data[i] << bi;
    }
    /* Convert B from LE byte order to BE — MD6 absorbs words in BE. */
    for (int i = 25; i < 89; i++) A[i] = bswap64(A[i]);

    /* Run main compression loop: 104 rounds × 16 steps. */
    md6core_compress_loop(A);

    /* Extract C[12..15] = A[1749..1752] (last 4 ulong = 256-bit digest).
     * MD6 output is BE ulong; we expand to 8 uint32 LE for compact-table
     * probe consistency (matches slab kernel's bswap64-then-split-to-uint
     * pattern at gpu_md6256unsalted.cl:198-202). */
    ulong C0 = A[1749], C1 = A[1750], C2 = A[1751], C3 = A[1752];
    ulong s0 = bswap64(C0); digest_words[0] = (uint)s0; digest_words[1] = (uint)(s0 >> 32);
    ulong s1 = bswap64(C1); digest_words[2] = (uint)s1; digest_words[3] = (uint)(s1 >> 32);
    ulong s2 = bswap64(C2); digest_words[4] = (uint)s2; digest_words[5] = (uint)(s2 >> 32);
    ulong s3 = bswap64(C3); digest_words[6] = (uint)s3; digest_words[7] = (uint)(s3 >> 32);
}

/* Per-lane state struct: 32-byte MD6-256 digest exposed as 8 LE uint32.
 * The template only reads/writes via template_finalize's output +
 * template_digest_compare; it does not introspect the struct. */
typedef struct {
    uint h[HASH_WORDS];   /* HASH_WORDS=8 (256-bit) */
} template_state;

static inline void template_init(template_state *st) {
    /* No persistent IV for MD6 (the IV is the Q/K/U/V prefix of N[],
     * which md6core_pack_and_compress builds from scratch every call).
     * Zero h[] for hygiene; template_finalize fully overwrites. */
    for (int i = 0; i < HASH_WORDS; i++) st->h[i] = 0u;
}

/* template_transform: stub for interface symmetry. MD6's template_finalize
 * builds the 89-ulong N[] in-place — never routes through this. */
static inline void template_transform(template_state *st,
                                      const uchar *block)
{
    (void)st;
    (void)block;
}

/* template_finalize: compute the FIRST iter's md6_256(data, len) and store
 * it in st->h[] as 8 LE uint32. */
static inline void template_finalize(template_state *st,
                                     const uchar *data, int len)
{
    md6core_pack_and_compress(&st->h[0], data, len);
}

/* template_iterate: per-iter step.
 *   prev_digest = st->h[0..7] (already LE)
 *   hexbuf[64] = LOWERCASE_HEX(prev_digest)
 *   new_digest = md6_256(hexbuf, 64)
 *   st->h[0..7] = new_digest (LE)
 *
 * Mirrors the CPU iter step at mdxfind.c:25849-25852 (md6_hash + prmd5).
 * The 64-byte hex buffer fits comfortably in a single MD6 leaf block (b=512
 * bytes). */
static inline void template_iterate(template_state *st)
{
    /* Build 64-byte LOWERCASE-hex of the prior digest. h[i] is LE uint32:
     * hex order is byte 0 first, byte 3 last (low byte → high byte).
     * Mirrors prmd5() at mdxfind.c which encodes bytes in little-endian
     * memory order from the MD-style state. */
    __attribute__((aligned(16))) uchar hexbuf[64];
    for (int w = 0; w < 8; w++) {
        uint sv = st->h[w];
        for (int b = 0; b < 4; b++) {
            uint byte = (sv >> (b * 8)) & 0xffu;
            uint hi = (byte >> 4) & 0xfu;
            uint lo = byte & 0xfu;
            hexbuf[w * 8 + b * 2]     = (uchar)(hi + ((hi < 10u) ? '0' : ('a' - 10)));
            hexbuf[w * 8 + b * 2 + 1] = (uchar)(lo + ((lo < 10u) ? '0' : ('a' - 10)));
        }
    }
    /* Run md6_256 over the 64-char hex buffer. */
    md6core_pack_and_compress(&st->h[0], hexbuf, 64);
}

/* template_digest_compare: probe the compact table. h[0..3] is LE; the
 * compact table is keyed on the first 4 LE uint32 — no swap needed. */
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

/* template_emit_hit: emit a hit. MD6256 = 8 LE uint32 digest words. */
#define template_emit_hit(hits, hit_count, max_hits, st, widx, sidx, iter) \
    do { \
        EMIT_HIT_8((hits), (hit_count), (max_hits), \
                   (widx), (sidx), (iter), ((st)->h)) \
    } while (0)

/* B3 dedup+overflow-aware variant. */
#define template_emit_hit_or_overflow(hits, hit_count, max_hits, st, widx, sidx, iter, \
                                      hashes_shown, matched_idx, dedup_mask, \
                                      ovr_set, ovr_gid, lane_gid) \
    do { \
        EMIT_HIT_8_DEDUP_OR_OVERFLOW((hits), (hit_count), (max_hits), \
                   (widx), (sidx), (iter), ((st)->h), \
                   (hashes_shown), (matched_idx), (dedup_mask), \
                   (ovr_set), (ovr_gid), (lane_gid)); \
    } while (0)
