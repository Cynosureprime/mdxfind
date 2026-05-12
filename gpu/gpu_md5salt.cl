/*
 * $Revision: 1.7 $
 * $Log: gpu_md5salt.cl,v $
 * Revision 1.7  2026/05/11 05:22:01  dlr
 * Backfill $Revision/$Log RCS keyword stanzas per feedback_rcs_keyword_stanzas.md. Passive 4-line comment block at top of file; no behavioral change. Hand-authored .cl file was missing required stanzas (per memory: all hand-authored .c/.h/.cl/.frag/.tmpl/.py/.sh files MUST contain $Revision/$Log keyword stanzas). Build green on .205 against the post-add files; OpenCL compile strips comments so no kernel behavior change.
 *
 */
/* B8 retirement (2026-05-06): md5salt_batch and md5salt_sub8_24 kernels
 * removed — JOB_MD5SALT, JOB_MD5UCSALT, JOB_MD5revMD5SALT, JOB_MD5sub8_24SALT
 * routed through unified template path (gpu_template.cl + gpu_md5salt_core.cl)
 * via params.algo_mode at B6/B6.6. RCS history retains the slab kernels
 * (rev <= 1.x) for reference.
 *
 * 2026-05-07: md5salt_iter kernel retired — JOB_MD5SALT iter coverage
 * routes through the unified template path's per-iter loop (gpu_template.cl
 * template_phase0 → template_iterate(&st) using gpu_md5salt_core.cl).
 * The probe_max_dispatch capacity probe now exercises hmac_md5_ksalt_batch
 * (same 17-arg slab signature). Only HMAC-MD5 kernels remain live in
 * this file. */

/* hex_byte_lc/uc, md5_to_hex_lc/uc provided by gpu_common.cl */

/* gpu_hmac_md5.cl — HMAC-MD5 GPU kernels
 *
 * hmac_md5_ksalt_batch: HMAC-MD5 key=$salt (e214, hashcat 60)
 *   Key = salt (from Typeuser), Message = password
 *
 * hmac_md5_kpass_batch: HMAC-MD5 key=$pass (e792, hashcat 50)
 *   Key = password, Message = salt (from Typesalt)
 *
 * HMAC(K, M) = H((K ^ opad) || H((K ^ ipad) || M))
 * Block size = 64 bytes, ipad = 0x36, opad = 0x5c
 * MD5 is little-endian — no bswap needed for key/ipad/opad blocks.
 *
 * Hit stride: 7 (word_idx, salt_idx, iter, hx, hy, hz, hw)
 */

/* HMAC-MD5 key=$salt: key=salt, message=password.
 * Dispatch: num_words x num_salts threads. */
__kernel void hmac_md5_ksalt_batch(
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
    int klen = salt_lens[salt_idx];
    __global const uchar *key = salts + soff;

    /* Step 1: Prepare key block (LE uint32 words).
     * If key > 64 bytes, hash it first. */
    uint key_block[16];
    for (int i = 0; i < 16; i++) key_block[i] = 0;

    if (klen > 64) {
        /* key = MD5(original_key) */
        uint M[16];
        for (int i = 0; i < 16; i++) M[i] = 0;
        uchar *mb = (uchar *)M;
        int copy1 = (klen < 64) ? klen : 64;
        for (int i = 0; i < copy1; i++) mb[i] = key[i];
        if (klen <= 55) { mb[klen] = 0x80; M[14] = klen * 8; }
        uint kx = 0x67452301u, ky = 0xEFCDAB89u, kz = 0x98BADCFEu, kw = 0x10325476u;
        md5_block(&kx, &ky, &kz, &kw, M);
        if (klen > 55) {
            for (int i = 0; i < 16; i++) M[i] = 0;
            int rem = klen - 64;
            mb = (uchar *)M;
            if (rem > 0) for (int i = 0; i < rem; i++) mb[i] = key[64 + i];
            mb[rem] = 0x80;
            M[14] = klen * 8;
            md5_block(&kx, &ky, &kz, &kw, M);
        }
        /* MD5 output is already LE — store directly */
        key_block[0] = kx; key_block[1] = ky;
        key_block[2] = kz; key_block[3] = kw;
        klen = 16;
    } else {
        /* Copy key bytes into key_block as LE uint32 */
        uchar *kb = (uchar *)key_block;
        for (int i = 0; i < klen; i++) kb[i] = key[i];
    }

    /* Step 2: Inner hash = MD5((key ^ ipad) || message) */
    uint ipad[16], M[16];
    for (int i = 0; i < 16; i++)
        ipad[i] = key_block[i] ^ 0x36363636u;

    /* MD5 is LE — ipad goes directly into M[], no bswap */
    for (int i = 0; i < 16; i++) M[i] = ipad[i];

    uint ihx = 0x67452301u, ihy = 0xEFCDAB89u, ihz = 0x98BADCFEu, ihw = 0x10325476u;
    md5_block(&ihx, &ihy, &ihz, &ihw, M);  /* Process ipad block (64 bytes) */

    /* Continue with message (password) + padding */
    for (int i = 0; i < 16; i++) M[i] = 0;
    int mlen = plen;
    uchar *mb = (uchar *)M;
    if (mlen <= 55) {
        for (int i = 0; i < mlen; i++) mb[i] = pass[i];
        mb[mlen] = 0x80;
        M[14] = (64 + mlen) * 8;  /* total = ipad_block + message */
        md5_block(&ihx, &ihy, &ihz, &ihw, M);
    } else {
        int copy1 = (mlen < 64) ? mlen : 64;
        for (int i = 0; i < copy1; i++) mb[i] = pass[i];
        if (mlen < 64) mb[mlen] = 0x80;
        md5_block(&ihx, &ihy, &ihz, &ihw, M);
        for (int i = 0; i < 16; i++) M[i] = 0;
        mb = (uchar *)M;
        int rem = mlen - 64;
        if (rem > 0) for (int i = 0; i < rem; i++) mb[i] = pass[64 + i];
        if (mlen >= 64) mb[(rem > 0) ? rem : 0] = 0x80;
        M[14] = (64 + mlen) * 8;
        md5_block(&ihx, &ihy, &ihz, &ihw, M);
    }

    /* Inner hash result: ihx, ihy, ihz, ihw (LE) */

    /* Step 3: Outer hash = MD5((key ^ opad) || inner_hash) */
    uint opad_block[16];
    for (int i = 0; i < 16; i++)
        opad_block[i] = key_block[i] ^ 0x5c5c5c5cu;

    for (int i = 0; i < 16; i++) M[i] = opad_block[i];

    uint ohx = 0x67452301u, ohy = 0xEFCDAB89u, ohz = 0x98BADCFEu, ohw = 0x10325476u;
    md5_block(&ohx, &ohy, &ohz, &ohw, M);  /* Process opad block (64 bytes) */

    /* Continue with inner_hash (16 bytes LE) + padding */
    for (int i = 0; i < 16; i++) M[i] = 0;
    M[0] = ihx; M[1] = ihy; M[2] = ihz; M[3] = ihw;  /* 16 bytes LE */
    M[4] = 0x80;  /* 0x80 at byte 16 = word 4, LE byte 0 */
    M[14] = (64 + 16) * 8;  /* 640 bits */
    md5_block(&ohx, &ohy, &ohz, &ohw, M);

    /* Step 4: Check result — MD5 is LE, no bswap needed */
    if (probe_compact(ohx, ohy, ohz, ohw, compact_fp, compact_idx,
                      params.compact_mask, params.max_probe, params.hash_data_count,
                      hash_data_buf, hash_data_off,
                      overflow_keys, overflow_hashes, overflow_offsets, params.overflow_count)) {
        EMIT_HIT_4(hits, hit_count, params.max_hits, word_idx, salt_idx, 1, ohx, ohy, ohz, ohw)
    }
}

/* HMAC-MD5 key=$pass: key=password, message=salt.
 * Dispatch: num_words x num_salts threads. */
__kernel void hmac_md5_kpass_batch(
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

    int klen = hexlens[word_idx];  /* password = key */
    __global const uchar *key = hexhashes + word_idx * 256;
    uint soff = salt_offsets[salt_idx];
    int mlen = salt_lens[salt_idx];  /* salt = message */
    __global const uchar *msg = salts + soff;

    /* Step 1: Prepare key block from password */
    uint key_block[16];
    for (int i = 0; i < 16; i++) key_block[i] = 0;

    if (klen > 64) {
        uint M[16];
        for (int i = 0; i < 16; i++) M[i] = 0;
        uchar *mb = (uchar *)M;
        int copy1 = (klen < 64) ? klen : 64;
        for (int i = 0; i < copy1; i++) mb[i] = key[i];
        if (klen <= 55) { mb[klen] = 0x80; M[14] = klen * 8; }
        uint kx = 0x67452301u, ky = 0xEFCDAB89u, kz = 0x98BADCFEu, kw = 0x10325476u;
        md5_block(&kx, &ky, &kz, &kw, M);
        if (klen > 55) {
            for (int i = 0; i < 16; i++) M[i] = 0;
            int rem = klen - 64;
            mb = (uchar *)M;
            if (rem > 0) for (int i = 0; i < rem; i++) mb[i] = key[64 + i];
            mb[rem] = 0x80;
            M[14] = klen * 8;
            md5_block(&kx, &ky, &kz, &kw, M);
        }
        key_block[0] = kx; key_block[1] = ky;
        key_block[2] = kz; key_block[3] = kw;
        klen = 16;
    } else {
        uchar *kb = (uchar *)key_block;
        for (int i = 0; i < klen; i++) kb[i] = key[i];
    }

    /* Step 2: Inner hash = MD5((key ^ ipad) || salt) */
    uint ipad[16], M[16];
    for (int i = 0; i < 16; i++)
        ipad[i] = key_block[i] ^ 0x36363636u;

    for (int i = 0; i < 16; i++) M[i] = ipad[i];

    uint ihx = 0x67452301u, ihy = 0xEFCDAB89u, ihz = 0x98BADCFEu, ihw = 0x10325476u;
    md5_block(&ihx, &ihy, &ihz, &ihw, M);

    for (int i = 0; i < 16; i++) M[i] = 0;
    uchar *mb = (uchar *)M;
    if (mlen <= 55) {
        for (int i = 0; i < mlen; i++) mb[i] = msg[i];
        mb[mlen] = 0x80;
        M[14] = (64 + mlen) * 8;
        md5_block(&ihx, &ihy, &ihz, &ihw, M);
    } else {
        int copy1 = (mlen < 64) ? mlen : 64;
        for (int i = 0; i < copy1; i++) mb[i] = msg[i];
        if (mlen < 64) mb[mlen] = 0x80;
        md5_block(&ihx, &ihy, &ihz, &ihw, M);
        for (int i = 0; i < 16; i++) M[i] = 0;
        mb = (uchar *)M;
        int rem = mlen - 64;
        if (rem > 0) for (int i = 0; i < rem; i++) mb[i] = msg[64 + i];
        if (mlen >= 64) mb[(rem > 0) ? rem : 0] = 0x80;
        M[14] = (64 + mlen) * 8;
        md5_block(&ihx, &ihy, &ihz, &ihw, M);
    }

    /* Step 3: Outer hash */
    uint opad_block[16];
    for (int i = 0; i < 16; i++)
        opad_block[i] = key_block[i] ^ 0x5c5c5c5cu;

    for (int i = 0; i < 16; i++) M[i] = opad_block[i];

    uint ohx = 0x67452301u, ohy = 0xEFCDAB89u, ohz = 0x98BADCFEu, ohw = 0x10325476u;
    md5_block(&ohx, &ohy, &ohz, &ohw, M);

    for (int i = 0; i < 16; i++) M[i] = 0;
    M[0] = ihx; M[1] = ihy; M[2] = ihz; M[3] = ihw;
    M[4] = 0x80;
    M[14] = (64 + 16) * 8;  /* 640 bits */
    md5_block(&ohx, &ohy, &ohz, &ohw, M);

    if (probe_compact(ohx, ohy, ohz, ohw, compact_fp, compact_idx,
                      params.compact_mask, params.max_probe, params.hash_data_count,
                      hash_data_buf, hash_data_off,
                      overflow_keys, overflow_hashes, overflow_offsets, params.overflow_count)) {
        EMIT_HIT_4(hits, hit_count, params.max_hits, word_idx, salt_idx, 1, ohx, ohy, ohz, ohw)
    }
}
