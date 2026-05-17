/* template_finalize: compute {{BASE_ALGO}}(pass || salt).
 *
 * BIG-ENDIAN 64-bit-state APPEND variant for the SHA-384/512 family.
 * Sibling of finalize_prepend_be64.cl.frag (B6.9): same width-bearing
 * geometry (ulong M[16], 128-byte block, 128-bit BE length field) but
 * with pass-FIRST salt-SECOND ordering at the byte source. Authored as
 * a separate file from the 32-bit BE APPEND fragment
 * (finalize_append_be.cl.frag) for the same width-bearing reasons:
 * the 32-bit fragment hardcodes uint M[16], 56-byte tail-fits threshold,
 * and a 32-bit length field — none of which can be #if-tangled cleanly
 * into a SHA-512-compatible body without obscuring both algorithms.
 *
 * KEY GEOMETRY DELTAS vs finalize_append_be.cl.frag (32-bit BE APPEND):
 *   - Message word width: ulong (8 bytes) vs uint (4 bytes).
 *   - Block size: HASH_BLOCK_BYTES=128 vs 64.
 *   - Length field: 128-bit BE in M[14..15] (each ulong = 64 bits).
 *     M[14] holds high 64 bits, M[15] holds low 64 bits. For our
 *     wordlist+salt inputs total_len < 2^57 bytes always, so M[14] = 0.
 *   - Tail-fits-in-block threshold: rem < 112 (vs 56 for 64-byte
 *     blocks). 112 = 128 - 16 (16 = 2 ulongs reserved for the length
 *     field). When rem >= 112 we run an extra padding-only block.
 *   - Per-byte BE position math: byte i goes into word (i >> 3) at
 *     shift (7 - (i & 7)) * 8. (vs (i >> 2) at shift (3 - (i & 3)) * 8
 *     in the 32-bit fragment.)
 *
 * KEY DELTAS vs finalize_prepend_be64.cl.frag (B6.9 PREPEND sibling):
 *   - Salt ordering at byte source: pass FIRST (offset 0..len-1),
 *     salt SECOND (offset len..len+slen-1). The sibling fragment puts
 *     salt FIRST. Mirrors the 32-bit BE APPEND/PREPEND pair (same
 *     conditional swap at the byte-source level).
 *
 * Layout in M[]: password bytes first (offset 0..plen-1), salt bytes
 * second (offset plen..plen+slen-1), then 0x80 padding marker, then
 * zeros, then 128-bit BE length-in-bits in M[14..15]. This is the
 * inverse of the BE PREPEND 64-bit fragment (finalize_prepend_be64.cl.frag,
 * salt-first), and the 64-bit-state counterpart of the 32-bit BE APPEND
 * fragment (finalize_append_be.cl.frag, same pass-first ordering but
 * 32-bit message-word width).
 *
 * st->state[0..7] is the 8 × ulong SHA-384/512 chaining state. The
 * caller's template_state struct also carries `uint h[16]` for digest
 * emit; we populate that via template_state_to_h() at the end.
 *
 * R1 (AMD ROCm comgr addrspace fragility): single private buffer
 * pattern. salt fetch is INLINE inside this finalize body — no
 * helper takes `__global const uchar *salt_buf`.
 *
 * R2 (register pressure on gfx1201): SHA-512's W[80] schedule is
 * 80 × 8 = 640 bytes private scratch per lane (already the largest
 * in the family). Adding salt-APPEND fixup costs only the local
 * ulong M[16] (128 B, same as unsalted) plus the one extra index
 * variable; no W[] duplication. Expected priv_mem delta vs the
 * salted-PREPEND sibling: ~0 B (same M[16] scratch + same per-byte
 * loop body — only the byte-source branch order swaps). gfx1201
 * sibling reading was 42,032 B; this fragment is expected to land
 * within 200 B of that, well under the 43,024 B 3080 spill-region
 * ceiling.
 *
 * Per feedback_codegen_fragment_width_bugs.md: this fragment does NOT
 * carry a defensive state-IV reinstall. template_init() (in the per-
 * algorithm core, not in this fragment) is the canonical state
 * initializer; reinstalling here would shadow it with width-incorrect
 * constants for any non-SHA512 algo. Width-bearing constants
 * (HASH_BLOCK_BYTES=128, 112-byte threshold, 128-bit length field)
 * live IN this fragment because they ARE the width axis the fragment
 * is dedicated to — see codegen-reconsideration memo's "width-bearing
 * constants belong in templates not fragments" rule, with the
 * narrowing that fragments which ARE the width axis (vs fragments
 * agnostic-to-width) are the right home for those constants.
 */
static inline void template_finalize(template_state *st,
                                     const uchar *data, int len,
                                     __global const uchar *salt_buf,
                                     uint slen,
                                     uint algo_mode)
{
    (void)algo_mode;  /* unused in this fragment; reserved for variant flags */
    ulong M[16];
    int total_len = len + (int)slen;
    int pos = 0;

    /* Process complete 128-byte blocks. Build M[] BIG-ENDIAN directly
     * from bytes; password bytes for p < len, salt bytes for p >= len.
     * Per-byte BE position: byte i into word (i >> 3) at shift
     * (7 - (i & 7)) * 8. */
    while (total_len - pos >= HASH_BLOCK_BYTES) {
        for (int j = 0; j < 16; j++) M[j] = 0UL;
        for (int i = 0; i < HASH_BLOCK_BYTES; i++) {
            int p = pos + i;
            uchar c;
            if (p < len) {
                c = data[p];
            } else {
                c = salt_buf[p - len];
            }
            M[i >> 3] |= ((ulong)c) << ((7 - (i & 7)) * 8);
        }
        {{BASE_ALGO}}_block(&st->state[0], M);
        pos += HASH_BLOCK_BYTES;
    }

    /* Build final block(s): tail bytes + 0x80 marker + zeros + 128-bit
     * BE length. */
    int rem = total_len - pos;  /* 0..127 */
    for (int j = 0; j < 16; j++) M[j] = 0UL;
    for (int i = 0; i < rem; i++) {
        int p = pos + i;
        uchar c;
        if (p < len) {
            c = data[p];
        } else {
            c = salt_buf[p - len];
        }
        M[i >> 3] |= ((ulong)c) << ((7 - (i & 7)) * 8);
    }
    /* 0x80 padding marker, BE byte position. */
    {
        int wi = rem >> 3;
        int bi = 7 - (rem & 7);
        M[wi] |= ((ulong)0x80UL) << (bi * 8);
    }

    if (rem < 112) {
        /* Length fits in this block. M[14] = high 64 bits of bit count
         * (= 0 for total_len < 2^57 bytes), M[15] = low 64 bits. */
        M[14] = 0UL;
        M[15] = (ulong)((ulong)total_len * 8UL);
        {{BASE_ALGO}}_block(&st->state[0], M);
    } else {
        /* Need one extra padding-only block to hold the length. */
        {{BASE_ALGO}}_block(&st->state[0], M);
        for (int j = 0; j < 16; j++) M[j] = 0UL;
        M[14] = 0UL;
        M[15] = (ulong)((ulong)total_len * 8UL);
        {{BASE_ALGO}}_block(&st->state[0], M);
    }

    /* Decompose final state into h[16] LE uint32 for digest emit. */
    template_state_to_h(st);
}
