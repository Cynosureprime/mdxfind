/*
 * $Revision: 1.2 $
 * $Log: gpu_ripemd160_core.cl,v $
 * Revision 1.2  2026/05/11 05:22:02  dlr
 * Backfill $Revision/$Log RCS keyword stanzas per feedback_rcs_keyword_stanzas.md. Passive 4-line comment block at top of file; no behavioral change. Hand-authored .cl file was missing required stanzas (per memory: all hand-authored .c/.h/.cl/.frag/.tmpl/.py/.sh files MUST contain $Revision/$Log keyword stanzas). Build green on .205 against the post-add files; OpenCL compile strips comments so no kernel behavior change.
 *
 */
/* gpu_ripemd160_core.cl — RIPEMD-160 algorithm extension functions for the
 * generic dispatch template (Memo B Phase B5 sub-batch 2).
 *
 * RIPEMD-160 is structurally between MD5 and SHA1:
 *   - LITTLE-ENDIAN message-word layout (like MD5; UNLIKE the SHA family)
 *   - 5 × uint32 state (like SHA1)
 *   - 64-byte block, 64-bit LE length encoding (like MD5)
 *   - 80-round dual-pipeline compression (left line + right line combined
 *     into a single 5-word state at the end)
 *
 * Mirrors gpu_sha1_core.cl rev 1.2 in dispatch shape (HASH_WORDS=5,
 * EMIT_HIT_5{,_DEDUP_OR_OVERFLOW}) and gpu_md5_core.cl rev 1.3 in byte
 * ordering (LE in template_finalize / template_iterate). The combination
 * gives us the 80-round dual-pipeline character of RIPEMD-160 over the
 * familiar MD-style 64-byte block.
 *
 *   HASH_WORDS         — digest size in 32-bit words (5 for RIPEMD-160)
 *   HASH_BLOCK_BYTES   — compress-block size (64)
 *   template_state     — 5 × uint32 chaining
 *   template_init      — install RIPEMD-160 IV
 *   template_transform — absorb one 64-byte block (LE word load,
 *                        rmd160_block from gpu_common.cl)
 *   template_finalize  — pad and finalize (LE pattern, in-place M[16];
 *                        same perf-fix lesson as MD5 / SHA1)
 *   template_iterate   — re-hash digest as 40-byte hex_lc (LE packing,
 *                        mirrors gpu_rmd160unsalted.cl iter loop)
 *   template_digest_compare — probe leading 16 bytes (h[0..3]) directly,
 *                             NO bswap32 (RIPEMD output is LE per uint32,
 *                             same convention as MD5 — the host-side
 *                             compact_table_register reads LE uint32 pairs;
 *                             matches gpu_rmd160unsalted.cl's
 *                             probe_compact(state[0..3]) call)
 *   template_emit_hit       — EMIT_HIT_5 wrapper, NO bswap32 (LE state
 *                             matches MD5 convention; see compare note)
 *
 * KEY DIFFERENCE FROM gpu_sha1_core.cl:
 *
 *   SHA-1 stores state words BIG-ENDIAN (sha1_block's natural output);
 *   the SHA-1 template byte-swaps state[0..3] to LE before probing the
 *   compact table and writes LE-byteswapped state into the hits buffer.
 *
 *   RIPEMD-160 stores state words LITTLE-ENDIAN (rmd160_block's natural
 *   output — confirmed by gpu_rmd160unsalted.cl line 113 which calls
 *   probe_compact(state[0], state[1], state[2], state[3], ...) without
 *   any bswap32). The compact-table probe and hit emit pass state words
 *   directly to the macros — same convention as MD5 / MD4.
 *
 * Source order at compile time:
 *
 *   [ gpu_common_str, gpu_md5_rules_str, gpu_ripemd160_core_str, gpu_template_str ]
 *
 * gpu_common_str provides rmd160_block(), the RMD_F1..RMD_F5 + L1..L5 +
 * R1..R5 macros, EMIT_HIT_5{,_DEDUP_OR_OVERFLOW}, and probe_compact_idx.
 * gpu_md5_rules_str provides apply_rule (algorithm-agnostic). This file
 * provides the per-algorithm hooks. gpu_template_str ties them together.
 *
 * Bytecast invariants:
 *   - Final state[0..4] (after rmd160_block) is LITTLE-ENDIAN per uint32.
 *     Direct probe + direct emit; no bswap32.
 *   - apply_rule + buf in/out is identical to gpu_md5_rules.cl rev 1.28+.
 *   - template_finalize() builds M[16] DIRECTLY from input bytes via
 *     in-place LE shifts/OR-merge. Same perf-fix lesson as MD5/SHA1.
 *
 * R1 mitigation (AMD ROCm comgr addrspace fragility): single-private-
 * buffer pattern preserved.
 *
 * R2 (register pressure): rmd160_block keeps two parallel 5-word lines
 * (A,B,C,D,E + a1,b1,c1,d1,e1) plus the 80 message-word references
 * (constant indices into M[16]). The dual-pipeline carries more live
 * state than MD5's single line; comparable to SHA1's W[80] schedule
 * but with smaller working storage (160 bits state vs SHA1's 160 bits
 * + 80-word schedule). Expected priv_mem_size on gfx1201 ~ comparable
 * to MD5/SHA1 readings (within +/-200 bytes).
 *
 * PERF-FIX LESSON (carried from gpu_md5_core.cl rev 1.2):
 *
 * template_finalize MUST build M[16] DIRECTLY from input bytes — not
 * via a private uchar pad[64] buffer that is then read back as message
 * words. The byte-store / byte-load round-trip cost was the 12.3% wall
 * regression in B2's first-cut MD5 template path. RIPEMD-160 follows
 * the same in-place pattern (LE byte ordering, LE 64-bit length in
 * M[14]).
 */

/* Per-algorithm geometry. Cache key (R3 fix) hashes the defines_str
 * "HASH_WORDS=5,HASH_BLOCK_BYTES=64" alongside source text. Same
 * defines as SHA1; distinct source text guarantees distinct cache
 * entry per gpu_kernel_cache.c rev 1.5+. */
#ifndef HASH_WORDS
#define HASH_WORDS 5
#endif
#ifndef HASH_BLOCK_BYTES
#define HASH_BLOCK_BYTES 64
#endif

/* Per-lane state struct: RIPEMD-160 carries 5 uint32 chaining values. */
typedef struct {
    uint h[HASH_WORDS];
} template_state;

/* template_init: install RIPEMD-160 IV.
 * Standard RIPEMD-160 initial hash values (RIPEMD-160 spec §1; same
 * leading 4 words as MD5/SHA1 plus a fifth 0xC3D2E1F0). */
static inline void template_init(template_state *st) {
    st->h[0] = 0x67452301u;
    st->h[1] = 0xEFCDAB89u;
    st->h[2] = 0x98BADCFEu;
    st->h[3] = 0x10325476u;
    st->h[4] = 0xC3D2E1F0u;
}

/* template_transform: absorb one HASH_BLOCK_BYTES (64) byte block.
 * `block` points into the working buffer at the absorb position;
 * the caller must guarantee >=64 bytes are readable. RIPEMD-160 reads
 * message words LITTLE-ENDIAN (like MD5).
 *
 * Public extension API; the hot-path template_finalize bypasses this
 * via an in-place M[16] pattern. Kept symmetric with the rest of the
 * core family for consumers preferring the function-style entry. */
static inline void template_transform(template_state *st,
                                      const uchar *block)
{
    uint M[16];
    /* RIPEMD-160 reads message words little-endian (same as MD5). */
    for (int j = 0; j < 16; j++) {
        int b = j * 4;
        M[j] = (uint)block[b]
             | ((uint)block[b + 1] << 8)
             | ((uint)block[b + 2] << 16)
             | ((uint)block[b + 3] << 24);
    }
    rmd160_block(&st->h[0], M);
}

/* template_finalize: process the tail, append 0x80 padding marker +
 * 64-bit LE length-in-bits, and absorb. Caller passes the full buffer;
 * we run complete blocks then build the final padded block(s) here.
 * After return, st->h[0..4] holds the final RIPEMD-160 digest words
 * in LITTLE-ENDIAN form (rmd160_block's natural output).
 *
 * Mirrors gpu_md5_core.cl rev 1.2's in-place M[16] pattern (same LE
 * byte ordering and 64-bit LE length encoding). The 5th state word is
 * carried by rmd160_block transparently — the finalize logic is
 * structurally identical to MD5's; only the compression call differs. */
static inline void template_finalize(template_state *st,
                                     const uchar *data, int len)
{
    uint M[16];
    int pos = 0;

    /* Process complete 64-byte blocks. Build M[] LITTLE-ENDIAN
     * directly from bytes (no intermediate pad[]). */
    while (len - pos >= HASH_BLOCK_BYTES) {
        for (int j = 0; j < 16; j++) {
            int b = pos + j * 4;
            M[j] = (uint)data[b]
                 | ((uint)data[b + 1] << 8)
                 | ((uint)data[b + 2] << 16)
                 | ((uint)data[b + 3] << 24);
        }
        rmd160_block(&st->h[0], M);
        pos += HASH_BLOCK_BYTES;
    }

    /* Build final block(s): tail bytes + 0x80 marker + zeros + 64-bit
     * LE length. Direct-into-M[] approach. */
    int rem = len - pos;  /* 0..63 */

    /* Zero scratch. */
    for (int j = 0; j < 16; j++) M[j] = 0;

    /* Copy remaining tail bytes into M[] little-endian. */
    for (int i = 0; i < rem; i++) {
        uint v = (uint)data[pos + i];
        M[i >> 2] |= v << ((i & 3) * 8);
    }
    /* 0x80 padding marker (LE byte position). */
    M[rem >> 2] |= (uint)0x80u << ((rem & 3) * 8);

    if (rem < 56) {
        /* Length fits in this block. RIPEMD-160 LE: M[14] = low 32
         * bits of bit count, M[15] = high 32 bits. For len < 2^29 bytes
         * (always true for our wordlist inputs), high 32 bits = 0. */
        M[14] = (uint)((uint)len * 8u);
        M[15] = 0u;
        rmd160_block(&st->h[0], M);
    } else {
        /* Need one extra padding-only block to hold the length. */
        rmd160_block(&st->h[0], M);
        for (int j = 0; j < 16; j++) M[j] = 0;
        M[14] = (uint)((uint)len * 8u);
        M[15] = 0u;
        rmd160_block(&st->h[0], M);
    }
}

/* template_iterate: -i loop step. Re-encode digest as 40-byte lowercase
 * hex ASCII and rehash. RIPEMD-160 hex output is 40 chars (5 words ×
 * 8 chars = 40 hex chars) — same width as SHA1 — but the byte ordering
 * is LE.
 *
 * Mirrors gpu_rmd160unsalted.cl lines 119-133 byte-for-byte:
 *   - 5 LE state words → 10 LE-encoded hex M[] words (2 words per
 *     state word, 4 hex chars each)
 *   - M[10] = 0x80 (LE 0x80 padding at byte offset 40)
 *   - M[11..13] = 0
 *   - M[14] = 40 * 8 = 320 (low 32 bits of bit count)
 *   - M[15] = 0
 *   - state reset to RIPEMD-160 IV; rmd160_block. */
static inline void template_iterate(template_state *st)
{
    uint M[16];
    /* Hex-encode 5 LE state words into M[0..9] (40 hex chars, LE
     * packing). hex_byte_lc helper from gpu_common.cl produces a 16-bit
     * LE pair (low byte = low hex digit, high byte = high hex digit;
     * because we store the result into M[] LE this corresponds to the
     * correct ASCII byte order in the message stream). */
    for (int i = 0; i < 5; i++) {
        uint s = st->h[i];
        uint b0 = s & 0xff,        b1 = (s >> 8) & 0xff;
        uint b2 = (s >> 16) & 0xff, b3 = (s >> 24) & 0xff;
        M[i*2]     = hex_byte_lc(b0) | (hex_byte_lc(b1) << 16);
        M[i*2 + 1] = hex_byte_lc(b2) | (hex_byte_lc(b3) << 16);
    }
    M[10] = 0x80u;                   /* LE 0x80 at byte offset 40 */
    for (int j = 11; j < 14; j++) M[j] = 0u;
    M[14] = 40u * 8u;                /* 40 hex chars = 320 bits */
    M[15] = 0u;
    /* Reinitialize state to RIPEMD-160 IV; absorb the prepared M[]. */
    st->h[0] = 0x67452301u;
    st->h[1] = 0xEFCDAB89u;
    st->h[2] = 0x98BADCFEu;
    st->h[3] = 0x10325476u;
    st->h[4] = 0xC3D2E1F0u;
    rmd160_block(&st->h[0], M);
}

/* template_digest_compare: probe the compact table with the final
 * digest. RIPEMD-160 state is stored LITTLE-ENDIAN (rmd160_block's
 * natural form — confirmed by gpu_rmd160unsalted.cl line 113 calling
 * probe_compact(state[0..3]) without any byte-swap). The compact table
 * is keyed on the leading 16 BYTES of the digest in little-endian
 * uint32 form (host-side compact_table_register reads digest bytes as
 * 4 LE uint32). State words flow directly into the probe — same
 * convention as MD5 / MD4.
 *
 * The 5th state word (h[4]) is NOT used by the compact-table probe —
 * only by EMIT_HIT_5 below. */
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

/* template_emit_hit: emit a hit into the global hits buffer. Wraps
 * EMIT_HIT_5 (RIPEMD-160 = 5 uint32 digest words). Hits buffer
 * convention matches gpu_rmd160unsalted.cl: state words go directly,
 * no bswap32 (LE state matches the LE hits-buffer convention used by
 * the host hit-replay path for LE-native algorithms). */
#define template_emit_hit(hits, hit_count, max_hits, st, widx, sidx, iter) \
    EMIT_HIT_5((hits), (hit_count), (max_hits), \
               (widx), (sidx), (iter), ((st)->h))

/* B3 dedup+overflow-aware variant. The template body uses this entry;
 * hashes_shown bit is rolled back on overflow per gpu_common.cl §B3
 * protocol. */
#define template_emit_hit_or_overflow(hits, hit_count, max_hits, st, widx, sidx, iter, \
                                      hashes_shown, matched_idx, dedup_mask, \
                                      ovr_set, ovr_gid, lane_gid) \
    EMIT_HIT_5_DEDUP_OR_OVERFLOW((hits), (hit_count), (max_hits), \
               (widx), (sidx), (iter), ((st)->h), \
               (hashes_shown), (matched_idx), (dedup_mask), \
               (ovr_set), (ovr_gid), (lane_gid))
