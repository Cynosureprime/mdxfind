/* M_copy_bytes, M_set_byte provided by gpu_common.cl */

/* ---- MD5(salt + password) kernel ----
 *
 * Input: raw password bytes in hexhashes buffer, length in hexlens.
 * GPU constructs salt + password, computes MD5, checks compact table.
 * Handles 1-block (total <= 55) and 2-block (55 < total <= 119) dynamically.
 * Hit stride is 7 (includes iteration number).
 *
 * Uses byte-pointer cast (uchar *)M for message construction instead of
 * M_copy_bytes() shift-and-OR.  This matches the md5salt_batch approach
 * and avoids register-spill corruption observed with the OR-based path
 * when md5_block is inlined at high register pressure (plen >= 32).
 */
__kernel void md5saltpass_batch(
    __global const uchar *hexhashes, __global const ushort *hexlens,
    __global const uchar *salts, __global const uint *salt_offsets, __global const ushort *salt_lens,
    __global const uint *compact_fp, __global const uint *compact_idx,
    __global const OCLParams *params_buf,
    __global const uchar *hash_data_buf, __global const ulong *hash_data_off,
    __global const ushort *hash_data_len,
    __global uint *hits, __global volatile uint *hit_count,
    __global const ulong *overflow_keys, __global const uchar *overflow_hashes,
    __global const uint *overflow_offsets, __global const ushort *overflow_lengths)
{
    OCLParams params = *params_buf;
    uint tid = get_global_id(0);
    uint word_idx = tid / params.num_salts;
    uint salt_idx = params.salt_start + (tid % params.num_salts);
    if (word_idx >= params.num_words) return;

    /* Read password bytes and length */
    int plen = hexlens[word_idx];
    __global const uchar *pass = hexhashes + word_idx * 256;

    /* Read salt offset and length */
    uint soff = salt_offsets[salt_idx];
    int slen = salt_lens[salt_idx];
    int total_len = slen + plen;

    /* Build message = salt + password, compute MD5 */
    uint hx = 0x67452301, hy = 0xEFCDAB89, hz = 0x98BADCFE, hw = 0x10325476;

    { uint M[16];
      for (int i = 0; i < 16; i++) M[i] = 0;
      uchar *mb = (uchar *)M;

      if (total_len <= 55) {
        { __global const uchar *sp = salts + soff;
          for (int i = 0; i < slen; i++) mb[i] = sp[i];
        }
        for (int i = 0; i < plen; i++) mb[slen + i] = pass[i];
        mb[total_len] = 0x80;
        M[14] = total_len * 8;
        md5_block(&hx, &hy, &hz, &hw, M);
      } else {
        /* Two blocks: fill first 64 bytes from salt+pass */
        int salt_b1 = (slen < 64) ? slen : 64;
        { __global const uchar *sp = salts + soff;
          for (int i = 0; i < salt_b1; i++) mb[i] = sp[i];
        }
        int pass_b1 = 64 - salt_b1;
        if (pass_b1 > plen) pass_b1 = plen;
        if (pass_b1 > 0)
            for (int i = 0; i < pass_b1; i++) mb[salt_b1 + i] = pass[i];
        if (total_len < 64)
            mb[total_len] = 0x80;
        md5_block(&hx, &hy, &hz, &hw, M);
        /* Second block */
        for (int i = 0; i < 16; i++) M[i] = 0;
        int pos2 = 0;
        int salt_b2 = slen - salt_b1;
        if (salt_b2 > 0) {
            __global const uchar *sp = salts + soff + salt_b1;
            for (int i = 0; i < salt_b2; i++) mb[pos2++] = sp[i];
        }
        int pass_b2 = plen - pass_b1;
        if (pass_b2 > 0) {
            for (int i = 0; i < pass_b2; i++) mb[pos2++] = pass[pass_b1 + i];
        }
        if (total_len >= 64)
            mb[pos2] = 0x80;
        M[14] = total_len * 8;
        md5_block(&hx, &hy, &hz, &hw, M);
      }
    }

    if (probe_compact(hx, hy, hz, hw, compact_fp, compact_idx,
                      params.compact_mask, params.max_probe, params.hash_data_count,
                      hash_data_buf, hash_data_off,
                      overflow_keys, overflow_hashes, overflow_offsets, params.overflow_count)) {
        EMIT_HIT_4(hits, hit_count, params.max_hits, word_idx, salt_idx, 1, hx, hy, hz, hw)
    }
}

/* ---- MD5(password + salt) kernel ---- */
__kernel void md5passsalt_batch(
    __global const uchar *hexhashes, __global const ushort *hexlens,
    __global const uchar *salts, __global const uint *salt_offsets, __global const ushort *salt_lens,
    __global const uint *compact_fp, __global const uint *compact_idx,
    __global const OCLParams *params_buf,
    __global const uchar *hash_data_buf, __global const ulong *hash_data_off,
    __global const ushort *hash_data_len,
    __global uint *hits, __global volatile uint *hit_count,
    __global const ulong *overflow_keys, __global const uchar *overflow_hashes,
    __global const uint *overflow_offsets, __global const ushort *overflow_lengths)
{
    OCLParams params = *params_buf;
    uint tid = get_global_id(0);
    uint word_idx = tid / params.num_salts;
    uint salt_idx = params.salt_start + (tid % params.num_salts);
    if (word_idx >= params.num_words) return;

    int plen = hexlens[word_idx];
    __global const uchar *pass = hexhashes + word_idx * 256;
    uint soff = salt_offsets[salt_idx];
    int slen = salt_lens[salt_idx];
    int total_len = plen + slen;

    uint hx = 0x67452301, hy = 0xEFCDAB89, hz = 0x98BADCFE, hw = 0x10325476;
    { uint M[16];
      for (int i = 0; i < 16; i++) M[i] = 0;
      uchar *mb = (uchar *)M;

      if (total_len <= 55) {
        for (int i = 0; i < plen; i++) mb[i] = pass[i];
        { __global const uchar *sp = salts + soff;
          for (int i = 0; i < slen; i++) mb[plen + i] = sp[i];
        }
        mb[total_len] = 0x80;
        M[14] = total_len * 8;
        md5_block(&hx, &hy, &hz, &hw, M);
      } else {
        /* Two blocks: fill first 64 bytes from pass+salt */
        int pass_b1 = (plen < 64) ? plen : 64;
        for (int i = 0; i < pass_b1; i++) mb[i] = pass[i];
        int salt_b1 = 64 - pass_b1;
        if (salt_b1 > slen) salt_b1 = slen;
        if (salt_b1 > 0) {
            __global const uchar *sp = salts + soff;
            for (int i = 0; i < salt_b1; i++) mb[pass_b1 + i] = sp[i];
        }
        if (total_len < 64)
            mb[total_len] = 0x80;
        md5_block(&hx, &hy, &hz, &hw, M);
        /* Second block: remaining data + padding */
        for (int i = 0; i < 16; i++) M[i] = 0;
        int pos2 = 0;
        int pass_b2 = plen - pass_b1;
        if (pass_b2 > 0) {
            for (int i = 0; i < pass_b2; i++) mb[pos2++] = pass[pass_b1 + i];
        }
        int salt_b2 = slen - salt_b1;
        if (salt_b2 > 0) {
            __global const uchar *sp = salts + soff + salt_b1;
            for (int i = 0; i < salt_b2; i++) mb[pos2++] = sp[i];
        }
        if (total_len >= 64)
            mb[pos2] = 0x80;
        M[14] = total_len * 8;
        md5_block(&hx, &hy, &hz, &hw, M);
      }
    }

    if (probe_compact(hx, hy, hz, hw, compact_fp, compact_idx,
                      params.compact_mask, params.max_probe, params.hash_data_count,
                      hash_data_buf, hash_data_off,
                      overflow_keys, overflow_hashes, overflow_offsets, params.overflow_count)) {
        EMIT_HIT_4(hits, hit_count, params.max_hits, word_idx, salt_idx, 1, hx, hy, hz, hw)
    }
}
