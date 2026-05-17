/* template_finalize: compute {{BASE_ALGO}}(pass || salt).
 *
 * LITTLE-ENDIAN variant for MD-family algorithms. APPEND salt position:
 * the password bytes occupy offsets 0..plen-1 in M[], the salt bytes
 * occupy offsets plen..plen+slen-1. This is the inverse of the LE
 * PREPEND fragment (finalize_prepend.cl.frag), which puts salt first.
 *
 * Layout in M[]: password bytes first (offset 0..plen-1), salt bytes
 * second (offset plen..plen+slen-1), then 0x80 padding marker, then
 * zeros, then 64-bit LE length-in-bits in M[14..15] (M[14] = low 32
 * bits, M[15] = high 32 bits — opposite of BE which puts the high half
 * in M[14]).
 *
 * Each byte's position within its 32-bit word is determined by
 * (byte_idx & 3) shift (LE: byte 0 is low octet of word). This matches
 * gpu_{{BASE_ALGO}}_core.cl's LE M[] build for the unsalted path —
 * only the source of the bytes (pass|salt instead of just data) and
 * the ordering (pass-first vs salt-first) differ.
 *
 * Block boundary handled the same way as the PREPEND sibling: if the
 * tail (after 0x80) extends past byte 55 we run a second block whose
 * last 8 bytes hold the length.
 *
 * First APPEND-shape salted variant on the codegen path: ships
 * MD5PASSSALT (B6.4 / hashcat -m 10). Future SHA1PASSSALT and
 * SHA256PASSSALT are SHA-family APPEND variants and use a sibling BE
 * fragment (finalize_append_be.cl.frag) — the byte-position math + length
 * encoding differ for BE just as in the PREPEND family.
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

    st->h[0] = 0x67452301u;
    st->h[1] = 0xEFCDAB89u;
    st->h[2] = 0x98BADCFEu;
    st->h[3] = 0x10325476u;

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
            M[i >> 2] |= ((uint)c) << ((i & 3) * 8);
        }
        {{BASE_ALGO}}_block(&st->h[0], &st->h[1], &st->h[2], &st->h[3], M);
        pos += HASH_BLOCK_BYTES;
    }

    int rem = total_len - pos;
    for (int j = 0; j < 16; j++) M[j] = 0;
    for (int i = 0; i < rem; i++) {
        int p = pos + i;
        uchar c;
        if (p < len) {
            c = data[p];
        } else {
            c = salt_buf[p - len];
        }
        M[i >> 2] |= ((uint)c) << ((i & 3) * 8);
    }
    M[rem >> 2] |= (uint)0x80 << ((rem & 3) * 8);

    if (rem < 56) {
        M[14] = (uint)(total_len * 8);
        M[15] = 0;
        {{BASE_ALGO}}_block(&st->h[0], &st->h[1], &st->h[2], &st->h[3], M);
    } else {
        {{BASE_ALGO}}_block(&st->h[0], &st->h[1], &st->h[2], &st->h[3], M);
        for (int j = 0; j < 16; j++) M[j] = 0;
        M[14] = (uint)(total_len * 8);
        M[15] = 0;
        {{BASE_ALGO}}_block(&st->h[0], &st->h[1], &st->h[2], &st->h[3], M);
    }
}
