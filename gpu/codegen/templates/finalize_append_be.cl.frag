/* template_finalize: compute {{BASE_ALGO}}(pass || salt).
 *
 * BIG-ENDIAN variant for SHA-family algorithms (SHA1/SHA224/SHA256 —
 * SHA1 ships first as B6.5 SHA1PASSSALT; SHA256PASSSALT reuses this
 * fragment with HASH_WORDS / IV adjustments via the spec). SHA384/512
 * use a 64-bit-state sibling fragment (different block size / length
 * encoding) — out of scope for this fragment.
 *
 * Layout in M[]: password bytes first (offset 0..plen-1), salt bytes
 * second (offset plen..plen+slen-1), then 0x80 padding marker, then
 * zeros, then 64-bit BE length-in-bits in M[14..15] (M[14] = high 32
 * bits, M[15] = low 32 bits). This is the inverse of the BE PREPEND
 * fragment (finalize_prepend_be.cl.frag), which puts salt first; and
 * the BE counterpart of the LE APPEND fragment (finalize_append.cl.frag)
 * which uses the same pass-then-salt order but LE byte placement and
 * LE length encoding.
 *
 * Block boundary handled the same way as the BE PREPEND sibling: if
 * the tail (after 0x80) extends past byte 55 we run a second block
 * whose last 8 bytes hold the length.
 *
 * Each byte's position within its 32-bit word is determined by
 * 3-(byte_idx & 3) shift (BE: byte 0 is high octet of word). This
 * matches gpu_{{BASE_ALGO}}_core.cl's BE M[] build for the unsalted
 * path — only the source of the bytes (pass|salt instead of just data)
 * and the ordering (pass-first vs salt-first) differ.
 *
 * First APPEND-shape SHA-family salted variant on the codegen path:
 * ships SHA1PASSSALT (B6.5 / hashcat -m 100). Future SHA256PASSSALT
 * (hashcat -m 1410's APPEND counterpart, hashcat -m 1420's PREPEND is
 * SHA256SALTPASS) reuses this fragment with no further authoring —
 * the spec drives HASH_WORDS / hash_block_bytes / IV constants.
 *
 * Per feedback_codegen_fragment_width_bugs.md: this fragment does NOT
 * carry a defensive state-IV reinstall. template_init() (in the per-
 * algorithm core, not in this fragment) is the canonical state
 * initializer; reinstalling here would shadow it with width-incorrect
 * constants for non-SHA1 SHA-family algos.
 */
static inline void template_finalize(template_state *st,
                                     const uchar *data, int len,
                                     __global const uchar *salt_buf,
                                     uint slen,
                                     uint algo_mode)
{
    (void)algo_mode;  /* unused in this fragment; reserved for variant flags */
    uint M[16];
    int total_len = len + (int)slen;
    int pos = 0;

    /* IV install is the responsibility of template_init() — called by
     * template_phase0 BEFORE template_finalize. Per-algorithm IV constants
     * (SHA1 vs SHA256, 5 vs 8 words) live in the per-core file, not in
     * this fragment. (See feedback_codegen_fragment_width_bugs.md — a
     * defensive SHA1 IV reinstall in the BE PREPEND sibling silently
     * corrupted SHA256 state h[5..7] in B6.2 fan-out; this fragment
     * inherits the lesson and never reinstalls.) */

    /* Process complete 64-byte blocks. Build M[] BIG-ENDIAN directly
     * from bytes; password bytes for p < len, salt bytes for p >= len. */
    while (total_len - pos >= HASH_BLOCK_BYTES) {
        for (int j = 0; j < 16; j++) M[j] = 0;
        for (int i = 0; i < HASH_BLOCK_BYTES; i++) {
            int p = pos + i;
            uchar c;
            if (p < len) {
                c = data[p];
            } else {
                c = salt_buf[p - len];
            }
            M[i >> 2] |= ((uint)c) << ((3 - (i & 3)) * 8);
        }
        {{BASE_ALGO}}_block(&st->h[0], M);
        pos += HASH_BLOCK_BYTES;
    }

    /* Build final block(s): tail bytes + 0x80 marker + zeros + 64-bit
     * BE length. */
    int rem = total_len - pos;  /* 0..63 */
    for (int j = 0; j < 16; j++) M[j] = 0;
    for (int i = 0; i < rem; i++) {
        int p = pos + i;
        uchar c;
        if (p < len) {
            c = data[p];
        } else {
            c = salt_buf[p - len];
        }
        M[i >> 2] |= ((uint)c) << ((3 - (i & 3)) * 8);
    }
    /* 0x80 padding marker, BE byte position. */
    M[rem >> 2] |= ((uint)0x80u) << ((3 - (rem & 3)) * 8);

    if (rem < 56) {
        /* Length fits in this block. SHA-family BE: M[14] = high 32
         * bits of bit count, M[15] = low 32 bits. For total_len < 2^29
         * bytes (always true for our wordlist+salt inputs), high 32
         * bits = 0. */
        M[14] = 0;
        M[15] = (uint)((uint)total_len * 8u);
        {{BASE_ALGO}}_block(&st->h[0], M);
    } else {
        /* Need one extra padding-only block to hold the length. */
        {{BASE_ALGO}}_block(&st->h[0], M);
        for (int j = 0; j < 16; j++) M[j] = 0;
        M[14] = 0;
        M[15] = (uint)((uint)total_len * 8u);
        {{BASE_ALGO}}_block(&st->h[0], M);
    }
}
