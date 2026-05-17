/* template_finalize: compute MD5(salt || pass).
 *
 * Layout in M[]: salt bytes first (offset 0..slen-1), password bytes
 * second (offset slen..slen+plen-1), then 0x80 padding marker, then
 * zeros, then length-in-bits. Length = (slen + plen) * 8. Block boundary
 * handled the same way as gpu_{{BASE_ALGO}}_core.cl template_finalize: if the
 * tail (after 0x80) extends past byte 55 we run a second block whose
 * last 8 bytes hold the length. */
static inline void template_finalize(template_state *st,
                                     const uchar *data, int len,
                                     __global const uchar *salt_buf,
                                     uint slen,
                                     uint algo_mode)
{
    (void)algo_mode;  /* unused in this fragment; reserved for variant flags */
    uint M[16];
    int total_len = (int)slen + len;
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
            if (p < (int)slen) {
                c = salt_buf[p];
            } else {
                c = data[p - (int)slen];
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
        if (p < (int)slen) {
            c = salt_buf[p];
        } else {
            c = data[p - (int)slen];
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
