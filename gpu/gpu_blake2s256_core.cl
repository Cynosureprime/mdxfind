/*
 * $Revision: 1.2 $
 * $Log: gpu_blake2s256_core.cl,v $
 * Revision 1.2  2026/05/11 05:22:01  dlr
 * Backfill $Revision/$Log RCS keyword stanzas per feedback_rcs_keyword_stanzas.md. Passive 4-line comment block at top of file; no behavioral change. Hand-authored .cl file was missing required stanzas (per memory: all hand-authored .c/.h/.cl/.frag/.tmpl/.py/.sh files MUST contain $Revision/$Log keyword stanzas). Build green on .205 against the post-add files; OpenCL compile strips comments so no kernel behavior change.
 *
 */
/* gpu_blake2s256_core.cl — BLAKE2S-256 algorithm extension functions for
 * the generic dispatch template (Memo B Phase B5 sub-batch 3).
 *
 * BLAKE2 is structurally distinct from MD-style hashes:
 *
 *   1. State carries digest h[8] PLUS a byte-counter t[2] PLUS a
 *      finalization flag f[2]. Per Memo B brief §B5 sub-batch 3, we keep
 *      counter+flag INSIDE the per-algo state struct rather than extend
 *      template_transform's signature (which would force changes across
 *      all 9 existing cores). The shared template body never reads the
 *      counter/flag fields — only template_init / _finalize / _iterate
 *      manipulate them, and those are per-algo.
 *
 *   2. NO 0x80 / length-encoding-in-tail padding: BLAKE2 zeros the tail
 *      of the final block and signals end-of-input via the f[0] flag and
 *      the byte counter. Final block runs compression with last==1.
 *
 *   3. Single-block fast path: any password <= 55 bytes (the slab
 *      kernel's typical max) is one compression with last==1, counter ==
 *      total_len. The template_finalize streams arbitrary-length input
 *      through BLAKE2s blocks generically — same correctness, no
 *      single-block special case.
 *
 * BLAKE2S-256 geometry:
 *   - 64-byte block, 8 × uint32 chaining state, LITTLE-ENDIAN
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
 *   template_transform — absorb one 64-byte block (NON-FINAL; advances
 *                        counter, runs compression with last==0)
 *   template_finalize  — process complete blocks, zero tail, run final
 *                        compression with last==1
 *   template_iterate   — re-hash digest as 64-byte hex_lc input. The hex
 *                        string of the 32-byte BLAKE2s256 digest is 64
 *                        chars, fitting one 64-byte block exactly. Counter
 *                        = 64; last=1.
 *   template_digest_compare — probe leading 16 bytes (h[0..3]) directly
 *                             (LE state, no bswap32; same as MD5 / RMD).
 *   template_emit_hit       — EMIT_HIT_8 wrapper (BLAKE2S-256 = 8 uint32
 *                             LE digest words; matches gpu_blake2s256-
 *                             unsalted.cl line 122 emit format exactly).
 *
 * Source order at compile time:
 *
 *   [ gpu_common_str, gpu_md5_rules_str, gpu_blake2s256_core_str, gpu_template_str ]
 *
 * gpu_common_str provides B2S_IV[8], B2S_SIGMA[10][16], b2s_compress(),
 * EMIT_HIT_8{,_DEDUP_OR_OVERFLOW}, and probe_compact_idx.
 *
 * Bytecast invariants:
 *   - Final st->h[0..7] (after template_finalize/iterate) is LITTLE-ENDIAN
 *     per uint32 — direct probe + direct emit, no bswap32.
 *   - Wire format matches gpu_blake2s256unsalted.cl emit: 8 LE uint32
 *     words written via EMIT_HIT_8.
 *   - Probe key uses st->h[0..3] (leading 16 bytes LE), same convention
 *     as MD5/MD4/RIPEMD-160/RIPEMD-320.
 *
 * R1 mitigation (AMD ROCm comgr addrspace fragility): single-private-buffer
 * pattern. State struct is private; b2s_compress takes __private uchar*
 * (existing primitive in gpu_common.cl rev 1.12, no addrspace-cast).
 *
 * R2 (register pressure): BLAKE2s carries 8 uint32 chaining + 16 uint32
 * working state in b2s_compress = comparable to RIPEMD-320 (which also
 * has 10 chaining + 10 working). Expected priv_mem_size on gfx1201
 * within the 41-43 KB band already documented for B5 sub-batch 1+2 algos.
 *
 * Iteration policy: BLAKE2 iterated mdxfind output is the 32-byte binary
 * digest re-encoded as 64 lowercase hex chars then re-hashed; matches
 * mdxfind.c's JOB_BLAKE2S256 case (line 30799) where len=64 between
 * iterations and prmd5() writes lowercase hex.
 */

/* Per-algorithm geometry. Cache key (R3 fix): defines_str
 * "HASH_WORDS=8,HASH_BLOCK_BYTES=64" — same width as SHA256 but distinct
 * source text (different block compress + state struct), so cache key is
 * unique. */
#ifndef HASH_WORDS
#define HASH_WORDS 8
#endif
#ifndef HASH_BLOCK_BYTES
#define HASH_BLOCK_BYTES 64
#endif

/* Per-lane state struct. Fields beyond h[] are BLAKE2-specific:
 *   t[0..1] : 64-bit byte counter split into low/high uint32 pair (matches
 *             RFC 7693 BLAKE2s "t" parameter; for our wordlist inputs t[1]
 *             is always 0, but we keep it for spec compliance + reuse).
 *   f[0..1] : finalization flag pair. Only f[0] is set by single-pass
 *             flows (no last-node-of-tree semantics); kept as a pair to
 *             mirror RFC 7693's data layout for documentation consistency.
 *
 * The shared template body (gpu_template.cl) only reads st.h[]; it never
 * touches t[] or f[]. So the new fields cost zero in the shared path.
 *
 * 16-byte alignment helps AMD spill geometry (uint32 requires 4-byte;
 * declaring aligned(16) lets the compiler vectorize the M[16] internal
 * load inside b2s_compress). */
typedef struct {
    uint h[HASH_WORDS];   /* digest chaining state, LE-per-uint32 */
    uint t[2];            /* byte counter (low, high) — t[1] unused for
                             our input sizes but kept for spec parity */
    uint f[2];            /* finalization flag (0 = mid-stream, 0xFFFFFFFFu
                             = last block); f[1] unused (last-tree-node) */
} template_state;

/* template_init: install BLAKE2s256 IV XOR'd with parameter block.
 *
 * Parameter block (RFC 7693 §2.5):
 *   bytes 0:    digest_length = 32 (BLAKE2S-256)
 *   byte 1:     key_length = 0 (no key)
 *   byte 2:     fanout = 1 (sequential)
 *   byte 3:     depth = 1 (sequential)
 *   bytes 4..7: leaf_length = 0
 *   bytes 8..13: node_offset = 0
 *   byte 14:    node_depth = 0
 *   byte 15:    inner_length = 0
 *   bytes 16..23: salt = 0 (no salt)
 *   bytes 24..31: personalization = 0
 *
 * Param block as 8 uint32 LE: P[0] = 0x01010020 (depth<<24 | fanout<<16 |
 * keylen<<8 | digestlen), P[1..7] = 0. So h[0] = IV[0] ^ 0x01010020;
 * h[1..7] = IV[1..7]. Matches mdxfind.c blake2s() (line 2334-2336) and
 * gpu_blake2s256unsalted.cl (line 105-106) exactly. */
static inline void template_init(template_state *st) {
    for (int i = 0; i < 8; i++) st->h[i] = B2S_IV[i];
    st->h[0] ^= 0x01010020u;   /* digest_length=32, key=0, fanout=1, depth=1 */
    st->t[0] = 0u; st->t[1] = 0u;
    st->f[0] = 0u; st->f[1] = 0u;
}

/* template_transform: absorb one full 64-byte block as a NON-FINAL block.
 *
 * Public extension API. BLAKE2's per-block contract differs from MD-style:
 * the host kernel (template) calls template_transform only for COMPLETE
 * non-final blocks. The hot path inside template_finalize bypasses this
 * via a direct b2s_compress call to avoid the function-call boundary
 * cost (same perf-fix lesson as MD5/SHA cores).
 *
 * For BLAKE2 the "partial-fill held over" semantic that mdxfind.c uses
 * (blake2s() at line 2336-2351) doesn't apply here because template_-
 * finalize streams the whole input start-to-end in one call. */
static inline void template_transform(template_state *st,
                                      const uchar *block)
{
    /* Advance byte counter by full block size. */
    uint t0 = st->t[0] + (uint)HASH_BLOCK_BYTES;
    uint carry = (t0 < st->t[0]) ? 1u : 0u;
    st->t[0] = t0;
    st->t[1] += carry;
    /* counter as ulong: lo | hi<<32 — LE on the OpenCL device side. */
    ulong counter = ((ulong)st->t[0]) | (((ulong)st->t[1]) << 32);
    b2s_compress(&st->h[0], block, counter, 0);
}

/* template_finalize: process all input start-to-end, advance counter,
 * zero-pad final block, run final compression with last=1.
 *
 * Mirrors mdxfind.c blake2s() (line 2329-2362) and the BLAKE2s spec.
 * For len in {0..55} bytes (our typical wordlist + rule range): single
 * compression with counter=len, last=1, tail bytes zero — matches
 * gpu_blake2s256unsalted.cl line 109-112 exactly.
 *
 * For longer inputs (len >= 64), processes complete 64-byte blocks first
 * (last=0, counter=cumulative bytes after the block), then the tail
 * (last=1).
 *
 * Edge case: len exactly == HASH_BLOCK_BYTES (== 64). Per RFC 7693 the
 * 64th byte cannot be processed in the last block of a NON-FINAL flow
 * (because then the final compression has zero new bytes, which is only
 * valid if the input is empty). The BLAKE2 spec resolves this by:
 *   - process exactly HASH_BLOCK_BYTES-1 bytes? No — the actual spec is:
 *     "If 0 < dd <= HASH_BLOCK_BYTES, do final compression with the
 *      remaining dd bytes as a partial block padded with zeros."
 *   - In practice, when (len > 0 && len % 64 == 0), the last 64 bytes are
 *     the FINAL block (last=1), counter=len. The earlier complete blocks
 *     run as last=0. So for len==64 with content: ZERO complete-block
 *     iterations; the 64-byte input becomes the final block directly.
 *
 * Implementation: while (len - pos > HASH_BLOCK_BYTES) — strictly greater,
 * not >=. This leaves the last 1..64 bytes for the final compression.
 *
 * R1 mitigation preserved: only private uchar buffer + private state +
 * b2s_compress (existing __private uchar* primitive). */
static inline void template_finalize(template_state *st,
                                     const uchar *data, int len)
{
    int pos = 0;
    /* Process all complete blocks EXCEPT the last (which becomes the
     * final block even if exactly HASH_BLOCK_BYTES bytes). Strictly
     * greater preserves the "len bytes go through with last=1 at the
     * end" invariant. */
    while ((len - pos) > HASH_BLOCK_BYTES) {
        uint t0 = st->t[0] + (uint)HASH_BLOCK_BYTES;
        uint carry = (t0 < st->t[0]) ? 1u : 0u;
        st->t[0] = t0;
        st->t[1] += carry;
        ulong counter = ((ulong)st->t[0]) | (((ulong)st->t[1]) << 32);
        b2s_compress(&st->h[0], data + pos, counter, 0);
        pos += HASH_BLOCK_BYTES;
    }

    /* Final block: copy remainder into a 64-byte zero-padded buffer.
     * rem in [0..64]; rem==0 only for empty-input edge case. */
    int rem = len - pos;
    uchar buf[HASH_BLOCK_BYTES];
    for (int i = 0; i < rem; i++) buf[i] = data[pos + i];
    for (int i = rem; i < HASH_BLOCK_BYTES; i++) buf[i] = 0;

    /* Advance counter by remaining (rem) bytes. */
    uint t0 = st->t[0] + (uint)rem;
    uint carry = (t0 < st->t[0]) ? 1u : 0u;
    st->t[0] = t0;
    st->t[1] += carry;
    ulong counter = ((ulong)st->t[0]) | (((ulong)st->t[1]) << 32);

    /* Final compression with last=1. */
    st->f[0] = 0xFFFFFFFFu;
    b2s_compress(&st->h[0], buf, counter, 1);
}

/* template_iterate: -i loop step. Re-encode the 32-byte BLAKE2s256 digest
 * as 64 lowercase hex chars and rehash. Matches mdxfind.c JOB_BLAKE2S256
 * (line 30799-30806): blake2s(curin.h, 32, ...); len = 64; prmd5(...)
 * writes 64 lc-hex chars; next iter blake2s those 64 bytes.
 *
 * 64 hex chars fit one HASH_BLOCK_BYTES (64) block exactly. Single
 * compression with last=1, counter=64.
 *
 * Snapshot the 8 LE state words BEFORE the IV reset so we can hex-encode
 * them into the next-iter input block. */
static inline void template_iterate(template_state *st)
{
    uint snap[8];
    for (int i = 0; i < 8; i++) snap[i] = st->h[i];

    uchar buf[HASH_BLOCK_BYTES];
    /* Hex-encode 8 LE uint32 -> 64 lowercase hex chars. Byte order
     * matches prmd5(): the LE byte image of each uint32 is hex'd byte-
     * by-byte (b0, b1, b2, b3 → "xxxx xxxx") so the hex string equals
     * mdxfind's prmd5 output. */
    for (int i = 0; i < 8; i++) {
        uint s = snap[i];
        for (int b = 0; b < 4; b++) {
            uchar by = (uchar)((s >> (b * 8)) & 0xFFu);
            uchar hi = by >> 4, lo = by & 0xFu;
            buf[i * 8 + b * 2 + 0] = (hi < 10) ? (uchar)('0' + hi) : (uchar)('a' + (hi - 10));
            buf[i * 8 + b * 2 + 1] = (lo < 10) ? (uchar)('0' + lo) : (uchar)('a' + (lo - 10));
        }
    }

    /* Reset state to BLAKE2s256 init (IV ^ parameter block); reset counter
     * and flag for the new compression. */
    for (int i = 0; i < 8; i++) st->h[i] = B2S_IV[i];
    st->h[0] ^= 0x01010020u;
    st->t[0] = 64u; st->t[1] = 0u;
    st->f[0] = 0xFFFFFFFFu; st->f[1] = 0u;

    b2s_compress(&st->h[0], buf, 64UL, 1);
}

/* template_digest_compare: probe leading 16 bytes (h[0..3]). LE state
 * convention; no bswap32 (same as MD5 / RIPEMD-160 / RIPEMD-320 / matches
 * gpu_blake2s256unsalted.cl probe at line 118). */
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

/* template_emit_hit: emit a hit. Wraps EMIT_HIT_8 (BLAKE2S-256 = 8 uint32
 * LE digest words; matches gpu_blake2s256unsalted.cl line 122 emit). */
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
