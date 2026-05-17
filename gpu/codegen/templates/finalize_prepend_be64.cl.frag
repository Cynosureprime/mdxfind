/* template_finalize: compute {{BASE_ALGO}}(salt || pass).
 *
 * BIG-ENDIAN 64-bit-state variant for the SHA-384/512 family. First
 * salted fragment in the codegen tool that uses ulong message words
 * (vs the 32-bit BE fragment finalize_prepend_be.cl.frag used by
 * SHA-1/SHA-224/SHA-256 with HASH_BLOCK_BYTES=64).
 *
 * KEY GEOMETRY DELTAS vs finalize_prepend_be.cl.frag (32-bit BE):
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
 * Layout in M[]: salt bytes first (offset 0..slen-1), password bytes
 * second (offset slen..slen+plen-1), then 0x80 padding marker, then
 * zeros, then 128-bit BE length-in-bits in M[14..15].
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
 * in the family). Adding salt-PREPEND fixup costs only the local
 * ulong M[16] (128 B, same as unsalted) plus the one extra index
 * variable; no W[] duplication. Expected priv_mem delta vs unsalted
 * SHA-512 finalize: < 50 bytes. Headline gate is gfx1201
 * private_mem_size <= 43,024 B (3080 spill-region ceiling from B5
 * sub-batch 4 fatal). Unsalted reading was 42,520 B; this addition
 * should land at 42,500 - 42,800 B. If it crosses 43,024 the path
 * is gated and the agent reports a structural mitigation. */
static inline void template_finalize(template_state *st,
                                     const uchar *data, int len,
                                     __global const uchar *salt_buf,
                                     uint slen,
                                     uint algo_mode)
{
    /* Family E (2026-05-08): HMAC-SHA384 branch. Modes 5 (KSALT, e543)
     * and 6 (KPASS, e796) compute HMAC-SHA384 and return early — no
     * SHA384(salt||pass) concatenation applies. Mirrors gpu_hmac_sha512.cl
     * hmac_sha384_ksalt_batch (lines 267-367) / hmac_sha384_kpass_batch
     * (lines 369-467) byte-for-byte (slab oracle).
     *
     *   Mode 5 (HMAC_SHA384_KSALT, e543): key = salt_buf[0..slen),
     *                                     msg = data[0..len).
     *   Mode 6 (HMAC_SHA384_KPASS, e796): key = data[0..len),
     *                                     msg = salt_buf[0..slen).
     *
     * HMAC(K, M) = SHA384((K^opad) || SHA384((K^ipad) || M)). Block size
     * = 128 bytes; ipad = 0x36, opad = 0x5c. SHA-384 uses sha512_block
     * compression (8-ulong state), digest truncated to 6 ulong = 48
     * bytes. Outer block is 128 + 48 = 176 bytes (2 sha512_block calls):
     * the first holds K^opad, the second holds the 48-byte inner digest
     * + 0x80 padding marker + 128-bit BE length (1408 bits). M[6] =
     * 0x8000000000000000UL (0x80 at byte 48 = high octet of word 6);
     * M[15] = 1408 = (128 + 48) * 8.
     *
     * GATED on HASH_WORDS == 12. The branch is structurally distinct
     * from any future HASH_WORDS == 16 SHA-512 HMAC branch (different
     * IV, output truncation, outer-block length). Both branches share
     * the algo_mode 5/6 encoding (KSALT/KPASS), and both substitute
     * BASE_ALGO=sha512 here — but only one branch is ever live in any
     * generated core (HASH_WORDS is fixed per spec at codegen time).
     *
     * R1 (AMD ROCm comgr addrspace fragility): all helper buffers are
     * private (key_block, ipad, opad_block, hm, kstate, istate, ostate);
     * salt_buf reads happen inline.
     *
     * R2 (gfx1201 register pressure): adds ~+450-600 bytes per lane on
     * top of the unsalted SHA-384 finalize — well under the 43,024 B
     * gfx1201 spill-region ceiling. */
    if (HASH_WORDS == 12 && algo_mode >= 5u) {
        /* Resolve key + message per mode. Build a 128-byte key_block
         * (ulong LE words) for the ipad/opad XOR. The slab oracle
         * (gpu_hmac_sha512.cl lines 290-311 / 392-413) uses the same
         * ulong LE byte-pack pattern — we mirror it byte-for-byte.
         *
         * SHA-384 IVs (FIPS 180-4 §5.3.4):
         *   0xcbbb9d5dc1059ed8, 0x629a292a367cd507, 0x9159015a3070dd17,
         *   0x152fecd8f70e5939, 0x67332667ffc00b31, 0x8eb44a8768581511,
         *   0xdb0c2e0d64f98fa7, 0x47b5481dbefa4fa4. */
        ulong key_block[16];
        for (int i = 0; i < 16; i++) key_block[i] = 0UL;

        int klen;
        int mlen;
        /* Step 1: prepare 128-byte key_block. If key > 128 bytes, replace
         * with SHA384(key) (48 bytes). Else pad to 128 with zeros. */
        if (algo_mode == 5u) {
            /* KSALT: key = salt (global). Slab oracle line 292-311. */
            klen = (int)slen;
            mlen = len;
            if (klen > 128) {
                ulong kstate[8] = {
                    0xcbbb9d5dc1059ed8UL, 0x629a292a367cd507UL,
                    0x9159015a3070dd17UL, 0x152fecd8f70e5939UL,
                    0x67332667ffc00b31UL, 0x8eb44a8768581511UL,
                    0xdb0c2e0d64f98fa7UL, 0x47b5481dbefa4fa4UL };
                ulong kM[16];
                int copy1 = (klen < 128) ? klen : 128;
                for (int i = 0; i < 16; i++) kM[i] = 0UL;
                /* SHA-384 = BIG-ENDIAN 64-bit: byte i shift =
                 * (7 - (i & 7)) * 8, into M[i >> 3]. */
                for (int i = 0; i < copy1; i++) {
                    ulong v = (ulong)salt_buf[i];
                    kM[i >> 3] |= v << ((7 - (i & 7)) * 8);
                }
                if (klen <= 111) {
                    int wi = klen >> 3;
                    int bi = 7 - (klen & 7);
                    kM[wi] |= ((ulong)0x80UL) << (bi * 8);
                    kM[15] = (ulong)klen * 8UL;
                }
                {{BASE_ALGO}}_block(&kstate[0], kM);
                if (klen > 111) {
                    for (int i = 0; i < 16; i++) kM[i] = 0UL;
                    int rem = klen - 128;
                    if (rem > 0) {
                        for (int i = 0; i < rem; i++) {
                            ulong v = (ulong)salt_buf[128 + i];
                            kM[i >> 3] |= v << ((7 - (i & 7)) * 8);
                        }
                    }
                    {
                        int p = (rem > 0) ? rem : 0;
                        int wi = p >> 3;
                        int bi = 7 - (p & 7);
                        kM[wi] |= ((ulong)0x80UL) << (bi * 8);
                    }
                    kM[15] = (ulong)klen * 8UL;
                    {{BASE_ALGO}}_block(&kstate[0], kM);
                }
                /* Store hashed key as LE ulong words (slab oracle line
                 * 306: `key_block[i] = bswap64(kst[i])`). 6 words for
                 * 48 bytes of SHA-384 digest. */
                for (int i = 0; i < 6; i++) key_block[i] = bswap64(kstate[i]);
                klen = 48;
            } else {
                /* Pack key bytes into key_block[0..15] ulong LE words.
                 * Slab oracle line 309-310. */
                for (int i = 0; i < klen; i++) {
                    ulong v = (ulong)salt_buf[i];
                    key_block[i >> 3] |= v << ((i & 7) * 8);
                }
            }
        } else {
            /* Mode 6 KPASS: key = data (private = post-rule password).
             * Slab oracle line 392-413. */
            klen = len;
            mlen = (int)slen;
            if (klen > 128) {
                ulong kstate[8] = {
                    0xcbbb9d5dc1059ed8UL, 0x629a292a367cd507UL,
                    0x9159015a3070dd17UL, 0x152fecd8f70e5939UL,
                    0x67332667ffc00b31UL, 0x8eb44a8768581511UL,
                    0xdb0c2e0d64f98fa7UL, 0x47b5481dbefa4fa4UL };
                ulong kM[16];
                int copy1 = (klen < 128) ? klen : 128;
                for (int i = 0; i < 16; i++) kM[i] = 0UL;
                for (int i = 0; i < copy1; i++) {
                    ulong v = (ulong)data[i];
                    kM[i >> 3] |= v << ((7 - (i & 7)) * 8);
                }
                if (klen <= 111) {
                    int wi = klen >> 3;
                    int bi = 7 - (klen & 7);
                    kM[wi] |= ((ulong)0x80UL) << (bi * 8);
                    kM[15] = (ulong)klen * 8UL;
                }
                {{BASE_ALGO}}_block(&kstate[0], kM);
                if (klen > 111) {
                    for (int i = 0; i < 16; i++) kM[i] = 0UL;
                    int rem = klen - 128;
                    if (rem > 0) {
                        for (int i = 0; i < rem; i++) {
                            ulong v = (ulong)data[128 + i];
                            kM[i >> 3] |= v << ((7 - (i & 7)) * 8);
                        }
                    }
                    {
                        int p = (rem > 0) ? rem : 0;
                        int wi = p >> 3;
                        int bi = 7 - (p & 7);
                        kM[wi] |= ((ulong)0x80UL) << (bi * 8);
                    }
                    kM[15] = (ulong)klen * 8UL;
                    {{BASE_ALGO}}_block(&kstate[0], kM);
                }
                for (int i = 0; i < 6; i++) key_block[i] = bswap64(kstate[i]);
                klen = 48;
            } else {
                for (int i = 0; i < klen; i++) {
                    ulong v = (ulong)data[i];
                    key_block[i >> 3] |= v << ((i & 7) * 8);
                }
            }
        }

        /* Step 2: inner = SHA384((key ^ ipad) || message). The key_block
         * is ulong LE; XOR with 0x36... then bswap64 to BE for sha512_-
         * block. Slab oracle line 313-318 / 415-420. */
        ulong ipad[16];
        ulong hm[16];
        for (int i = 0; i < 16; i++)
            ipad[i] = key_block[i] ^ 0x3636363636363636UL;
        for (int i = 0; i < 16; i++)
            hm[i] = bswap64(ipad[i]);

        ulong istate[8] = {
            0xcbbb9d5dc1059ed8UL, 0x629a292a367cd507UL,
            0x9159015a3070dd17UL, 0x152fecd8f70e5939UL,
            0x67332667ffc00b31UL, 0x8eb44a8768581511UL,
            0xdb0c2e0d64f98fa7UL, 0x47b5481dbefa4fa4UL };
        {{BASE_ALGO}}_block(&istate[0], hm);

        /* Continue with message bytes + 0x80 + length; total bits =
         * (128 + mlen) * 8. SHA-384 BE byte-pack at 64-bit width. */
        if (algo_mode == 5u) {
            /* msg = data (private). Slab oracle line 320-336. */
            if (mlen <= 111) {
                for (int i = 0; i < 16; i++) hm[i] = 0UL;
                for (int i = 0; i < mlen; i++) {
                    ulong v = (ulong)data[i];
                    hm[i >> 3] |= v << ((7 - (i & 7)) * 8);
                }
                {
                    int wi = mlen >> 3;
                    int bi = 7 - (mlen & 7);
                    hm[wi] |= ((ulong)0x80UL) << (bi * 8);
                }
                hm[15] = (ulong)(128 + mlen) * 8UL;
                {{BASE_ALGO}}_block(&istate[0], hm);
            } else {
                int copy1 = (mlen < 128) ? mlen : 128;
                for (int i = 0; i < 16; i++) hm[i] = 0UL;
                for (int i = 0; i < copy1; i++) {
                    ulong v = (ulong)data[i];
                    hm[i >> 3] |= v << ((7 - (i & 7)) * 8);
                }
                if (mlen < 128) {
                    int wi = mlen >> 3;
                    int bi = 7 - (mlen & 7);
                    hm[wi] |= ((ulong)0x80UL) << (bi * 8);
                }
                {{BASE_ALGO}}_block(&istate[0], hm);
                int rem = mlen - 128;
                for (int i = 0; i < 16; i++) hm[i] = 0UL;
                if (rem > 0) {
                    for (int i = 0; i < rem; i++) {
                        ulong v = (ulong)data[128 + i];
                        hm[i >> 3] |= v << ((7 - (i & 7)) * 8);
                    }
                }
                if (mlen >= 128) {
                    int p = (rem > 0) ? rem : 0;
                    int wi = p >> 3;
                    int bi = 7 - (p & 7);
                    hm[wi] |= ((ulong)0x80UL) << (bi * 8);
                }
                hm[15] = (ulong)(128 + mlen) * 8UL;
                {{BASE_ALGO}}_block(&istate[0], hm);
            }
        } else {
            /* Mode 6 KPASS: msg = salt_buf (global). Slab oracle line
             * 422-438. */
            if (mlen <= 111) {
                for (int i = 0; i < 16; i++) hm[i] = 0UL;
                for (int i = 0; i < mlen; i++) {
                    ulong v = (ulong)salt_buf[i];
                    hm[i >> 3] |= v << ((7 - (i & 7)) * 8);
                }
                {
                    int wi = mlen >> 3;
                    int bi = 7 - (mlen & 7);
                    hm[wi] |= ((ulong)0x80UL) << (bi * 8);
                }
                hm[15] = (ulong)(128 + mlen) * 8UL;
                {{BASE_ALGO}}_block(&istate[0], hm);
            } else {
                int copy1 = (mlen < 128) ? mlen : 128;
                for (int i = 0; i < 16; i++) hm[i] = 0UL;
                for (int i = 0; i < copy1; i++) {
                    ulong v = (ulong)salt_buf[i];
                    hm[i >> 3] |= v << ((7 - (i & 7)) * 8);
                }
                if (mlen < 128) {
                    int wi = mlen >> 3;
                    int bi = 7 - (mlen & 7);
                    hm[wi] |= ((ulong)0x80UL) << (bi * 8);
                }
                {{BASE_ALGO}}_block(&istate[0], hm);
                int rem = mlen - 128;
                for (int i = 0; i < 16; i++) hm[i] = 0UL;
                if (rem > 0) {
                    for (int i = 0; i < rem; i++) {
                        ulong v = (ulong)salt_buf[128 + i];
                        hm[i >> 3] |= v << ((7 - (i & 7)) * 8);
                    }
                }
                if (mlen >= 128) {
                    int p = (rem > 0) ? rem : 0;
                    int wi = p >> 3;
                    int bi = 7 - (p & 7);
                    hm[wi] |= ((ulong)0x80UL) << (bi * 8);
                }
                hm[15] = (ulong)(128 + mlen) * 8UL;
                {{BASE_ALGO}}_block(&istate[0], hm);
            }
        }

        /* Step 3: outer = SHA384((key ^ opad) || inner_hash). istate[8]
         * is BE; first 6 words copied directly into M[0..5]. M[6] =
         * 0x8000000000000000UL (0x80 BE at byte 48 = high octet of word
         * 6). M[15] = (128 + 48) * 8 = 1408. Slab oracle line 338-351
         * / 440-451. */
        ulong opad_block[16];
        for (int i = 0; i < 16; i++)
            opad_block[i] = key_block[i] ^ 0x5c5c5c5c5c5c5c5cUL;
        for (int i = 0; i < 16; i++)
            hm[i] = bswap64(opad_block[i]);

        ulong ostate[8] = {
            0xcbbb9d5dc1059ed8UL, 0x629a292a367cd507UL,
            0x9159015a3070dd17UL, 0x152fecd8f70e5939UL,
            0x67332667ffc00b31UL, 0x8eb44a8768581511UL,
            0xdb0c2e0d64f98fa7UL, 0x47b5481dbefa4fa4UL };
        {{BASE_ALGO}}_block(&ostate[0], hm);

        for (int i = 0; i < 6; i++) hm[i] = istate[i];
        hm[6] = 0x8000000000000000UL;   /* 0x80 BE at byte 48 */
        for (int i = 7; i < 15; i++) hm[i] = 0UL;
        hm[14] = 0UL;
        hm[15] = (ulong)(128 + 48) * 8UL;   /* 1408 bits */
        {{BASE_ALGO}}_block(&ostate[0], hm);

        /* Write final BE state into st->state[0..5]; template_state_to_h
         * decomposes into st->h[0..11] (12 LE uint32 words = 48 bytes).
         * state[6..7] are not used by SHA-384 truncation. */
        for (int i = 0; i < 6; i++) st->state[i] = ostate[i];
        template_state_to_h(st);
        return;
    }

    /* Family F (2026-05-08): HMAC-SHA512 branch. Modes 5 (KSALT, e218)
     * and 6 (KPASS, e797) compute HMAC-SHA512 and return early — no
     * SHA512(salt||pass) concatenation applies. Mirrors gpu_hmac_sha512.cl
     * hmac_sha512_ksalt_batch (lines 26-143) / hmac_sha512_kpass_batch
     * (lines 146-254) byte-for-byte (slab oracle).
     *
     *   Mode 5 (HMAC_SHA512_KSALT, e218): key = salt_buf[0..slen),
     *                                     msg = data[0..len).
     *   Mode 6 (HMAC_SHA512_KPASS, e797): key = data[0..len),
     *                                     msg = salt_buf[0..slen).
     *
     * HMAC(K, M) = SHA512((K^opad) || SHA512((K^ipad) || M)). Block size
     * = 128 bytes; ipad = 0x36, opad = 0x5c. SHA-512 uses sha512_block
     * compression (8-ulong state), digest is the FULL 8 ulong = 64 bytes
     * (NO truncation, vs SHA-384 which truncates to 6 ulong = 48 bytes).
     * Outer block is 128 + 64 = 192 bytes (2 sha512_block calls): the
     * first holds K^opad, the second holds the 64-byte inner digest +
     * 0x80 padding marker + 128-bit BE length (1536 bits). M[8] =
     * 0x8000000000000000UL (0x80 at byte 64 = high octet of word 8);
     * M[15] = 1536 = (128 + 64) * 8.
     *
     * GATED on HASH_WORDS == 16. The branch is structurally distinct
     * from the HASH_WORDS == 12 SHA-384 HMAC branch above (different
     * IV per FIPS 180-4 §5.3.5 vs §5.3.4, no output truncation, outer-
     * block geometry differs at the 0x80 marker position and length).
     * Both branches share the algo_mode 5/6 encoding (KSALT/KPASS), and
     * both substitute BASE_ALGO=sha512 here — but only one branch is
     * ever live in any generated core (HASH_WORDS is fixed per spec at
     * codegen time).
     *
     * R1 (AMD ROCm comgr addrspace fragility): all helper buffers are
     * private (key_block, ipad, opad_block, hm, kstate, istate, ostate);
     * salt_buf reads happen inline.
     *
     * R2 (gfx1201 register pressure): adds ~+450-600 bytes per lane on
     * top of the unsalted SHA-512 finalize — well under the 43,024 B
     * gfx1201 spill-region ceiling. */
    if (HASH_WORDS == 16 && algo_mode >= 5u) {
        /* Resolve key + message per mode. Build a 128-byte key_block
         * (ulong LE words) for the ipad/opad XOR. The slab oracle
         * (gpu_hmac_sha512.cl lines 50-78 / 169-196) uses the same
         * ulong LE byte-pack pattern — we mirror it byte-for-byte.
         *
         * SHA-512 IVs (FIPS 180-4 §5.3.5):
         *   0x6a09e667f3bcc908, 0xbb67ae8584caa73b, 0x3c6ef372fe94f82b,
         *   0xa54ff53a5f1d36f1, 0x510e527fade682d1, 0x9b05688c2b3e6c1f,
         *   0x1f83d9abfb41bd6b, 0x5be0cd19137e2179. */
        ulong key_block[16];
        for (int i = 0; i < 16; i++) key_block[i] = 0UL;

        int klen;
        int mlen;
        /* Step 1: prepare 128-byte key_block. If key > 128 bytes, replace
         * with SHA512(key) (64 bytes). Else pad to 128 with zeros. */
        if (algo_mode == 5u) {
            /* KSALT: key = salt (global). Slab oracle line 53-78. */
            klen = (int)slen;
            mlen = len;
            if (klen > 128) {
                ulong kstate[8] = {
                    0x6a09e667f3bcc908UL, 0xbb67ae8584caa73bUL,
                    0x3c6ef372fe94f82bUL, 0xa54ff53a5f1d36f1UL,
                    0x510e527fade682d1UL, 0x9b05688c2b3e6c1fUL,
                    0x1f83d9abfb41bd6bUL, 0x5be0cd19137e2179UL };
                ulong kM[16];
                int copy1 = (klen < 128) ? klen : 128;
                for (int i = 0; i < 16; i++) kM[i] = 0UL;
                /* SHA-512 = BIG-ENDIAN 64-bit: byte i shift =
                 * (7 - (i & 7)) * 8, into M[i >> 3]. */
                for (int i = 0; i < copy1; i++) {
                    ulong v = (ulong)salt_buf[i];
                    kM[i >> 3] |= v << ((7 - (i & 7)) * 8);
                }
                if (klen <= 111) {
                    int wi = klen >> 3;
                    int bi = 7 - (klen & 7);
                    kM[wi] |= ((ulong)0x80UL) << (bi * 8);
                    kM[15] = (ulong)klen * 8UL;
                }
                {{BASE_ALGO}}_block(&kstate[0], kM);
                if (klen > 111) {
                    for (int i = 0; i < 16; i++) kM[i] = 0UL;
                    int rem = klen - 128;
                    if (rem > 0) {
                        for (int i = 0; i < rem; i++) {
                            ulong v = (ulong)salt_buf[128 + i];
                            kM[i >> 3] |= v << ((7 - (i & 7)) * 8);
                        }
                    }
                    {
                        int p = (rem > 0) ? rem : 0;
                        int wi = p >> 3;
                        int bi = 7 - (p & 7);
                        kM[wi] |= ((ulong)0x80UL) << (bi * 8);
                    }
                    kM[15] = (ulong)klen * 8UL;
                    {{BASE_ALGO}}_block(&kstate[0], kM);
                }
                /* Store hashed key as LE ulong words (slab oracle line
                 * 70: `key_block[i] = bswap64(kst[i])`). 8 words for 64
                 * bytes of SHA-512 digest (vs 6 for SHA-384). */
                for (int i = 0; i < 8; i++) key_block[i] = bswap64(kstate[i]);
                klen = 64;
            } else {
                /* Pack key bytes into key_block[0..15] ulong LE words.
                 * Slab oracle line 73-77. */
                for (int i = 0; i < klen; i++) {
                    ulong v = (ulong)salt_buf[i];
                    key_block[i >> 3] |= v << ((i & 7) * 8);
                }
            }
        } else {
            /* Mode 6 KPASS: key = data (private = post-rule password).
             * Slab oracle line 172-196. */
            klen = len;
            mlen = (int)slen;
            if (klen > 128) {
                ulong kstate[8] = {
                    0x6a09e667f3bcc908UL, 0xbb67ae8584caa73bUL,
                    0x3c6ef372fe94f82bUL, 0xa54ff53a5f1d36f1UL,
                    0x510e527fade682d1UL, 0x9b05688c2b3e6c1fUL,
                    0x1f83d9abfb41bd6bUL, 0x5be0cd19137e2179UL };
                ulong kM[16];
                int copy1 = (klen < 128) ? klen : 128;
                for (int i = 0; i < 16; i++) kM[i] = 0UL;
                for (int i = 0; i < copy1; i++) {
                    ulong v = (ulong)data[i];
                    kM[i >> 3] |= v << ((7 - (i & 7)) * 8);
                }
                if (klen <= 111) {
                    int wi = klen >> 3;
                    int bi = 7 - (klen & 7);
                    kM[wi] |= ((ulong)0x80UL) << (bi * 8);
                    kM[15] = (ulong)klen * 8UL;
                }
                {{BASE_ALGO}}_block(&kstate[0], kM);
                if (klen > 111) {
                    for (int i = 0; i < 16; i++) kM[i] = 0UL;
                    int rem = klen - 128;
                    if (rem > 0) {
                        for (int i = 0; i < rem; i++) {
                            ulong v = (ulong)data[128 + i];
                            kM[i >> 3] |= v << ((7 - (i & 7)) * 8);
                        }
                    }
                    {
                        int p = (rem > 0) ? rem : 0;
                        int wi = p >> 3;
                        int bi = 7 - (p & 7);
                        kM[wi] |= ((ulong)0x80UL) << (bi * 8);
                    }
                    kM[15] = (ulong)klen * 8UL;
                    {{BASE_ALGO}}_block(&kstate[0], kM);
                }
                for (int i = 0; i < 8; i++) key_block[i] = bswap64(kstate[i]);
                klen = 64;
            } else {
                for (int i = 0; i < klen; i++) {
                    ulong v = (ulong)data[i];
                    key_block[i >> 3] |= v << ((i & 7) * 8);
                }
            }
        }

        /* Step 2: inner = SHA512((key ^ ipad) || message). The key_block
         * is ulong LE; XOR with 0x36... then bswap64 to BE for sha512_-
         * block. Slab oracle line 81-90 / 198-205. */
        ulong ipad[16];
        ulong hm[16];
        for (int i = 0; i < 16; i++)
            ipad[i] = key_block[i] ^ 0x3636363636363636UL;
        for (int i = 0; i < 16; i++)
            hm[i] = bswap64(ipad[i]);

        ulong istate[8] = {
            0x6a09e667f3bcc908UL, 0xbb67ae8584caa73bUL,
            0x3c6ef372fe94f82bUL, 0xa54ff53a5f1d36f1UL,
            0x510e527fade682d1UL, 0x9b05688c2b3e6c1fUL,
            0x1f83d9abfb41bd6bUL, 0x5be0cd19137e2179UL };
        {{BASE_ALGO}}_block(&istate[0], hm);

        /* Continue with message bytes + 0x80 + length; total bits =
         * (128 + mlen) * 8. SHA-512 BE byte-pack at 64-bit width. */
        if (algo_mode == 5u) {
            /* msg = data (private). Slab oracle line 92-109. */
            if (mlen <= 111) {
                for (int i = 0; i < 16; i++) hm[i] = 0UL;
                for (int i = 0; i < mlen; i++) {
                    ulong v = (ulong)data[i];
                    hm[i >> 3] |= v << ((7 - (i & 7)) * 8);
                }
                {
                    int wi = mlen >> 3;
                    int bi = 7 - (mlen & 7);
                    hm[wi] |= ((ulong)0x80UL) << (bi * 8);
                }
                hm[15] = (ulong)(128 + mlen) * 8UL;
                {{BASE_ALGO}}_block(&istate[0], hm);
            } else {
                int copy1 = (mlen < 128) ? mlen : 128;
                for (int i = 0; i < 16; i++) hm[i] = 0UL;
                for (int i = 0; i < copy1; i++) {
                    ulong v = (ulong)data[i];
                    hm[i >> 3] |= v << ((7 - (i & 7)) * 8);
                }
                if (mlen < 128) {
                    int wi = mlen >> 3;
                    int bi = 7 - (mlen & 7);
                    hm[wi] |= ((ulong)0x80UL) << (bi * 8);
                }
                {{BASE_ALGO}}_block(&istate[0], hm);
                int rem = mlen - 128;
                for (int i = 0; i < 16; i++) hm[i] = 0UL;
                if (rem > 0) {
                    for (int i = 0; i < rem; i++) {
                        ulong v = (ulong)data[128 + i];
                        hm[i >> 3] |= v << ((7 - (i & 7)) * 8);
                    }
                }
                if (mlen >= 128) {
                    int p = (rem > 0) ? rem : 0;
                    int wi = p >> 3;
                    int bi = 7 - (p & 7);
                    hm[wi] |= ((ulong)0x80UL) << (bi * 8);
                }
                hm[15] = (ulong)(128 + mlen) * 8UL;
                {{BASE_ALGO}}_block(&istate[0], hm);
            }
        } else {
            /* Mode 6 KPASS: msg = salt_buf (global). Slab oracle line
             * 207-223. */
            if (mlen <= 111) {
                for (int i = 0; i < 16; i++) hm[i] = 0UL;
                for (int i = 0; i < mlen; i++) {
                    ulong v = (ulong)salt_buf[i];
                    hm[i >> 3] |= v << ((7 - (i & 7)) * 8);
                }
                {
                    int wi = mlen >> 3;
                    int bi = 7 - (mlen & 7);
                    hm[wi] |= ((ulong)0x80UL) << (bi * 8);
                }
                hm[15] = (ulong)(128 + mlen) * 8UL;
                {{BASE_ALGO}}_block(&istate[0], hm);
            } else {
                int copy1 = (mlen < 128) ? mlen : 128;
                for (int i = 0; i < 16; i++) hm[i] = 0UL;
                for (int i = 0; i < copy1; i++) {
                    ulong v = (ulong)salt_buf[i];
                    hm[i >> 3] |= v << ((7 - (i & 7)) * 8);
                }
                if (mlen < 128) {
                    int wi = mlen >> 3;
                    int bi = 7 - (mlen & 7);
                    hm[wi] |= ((ulong)0x80UL) << (bi * 8);
                }
                {{BASE_ALGO}}_block(&istate[0], hm);
                int rem = mlen - 128;
                for (int i = 0; i < 16; i++) hm[i] = 0UL;
                if (rem > 0) {
                    for (int i = 0; i < rem; i++) {
                        ulong v = (ulong)salt_buf[128 + i];
                        hm[i >> 3] |= v << ((7 - (i & 7)) * 8);
                    }
                }
                if (mlen >= 128) {
                    int p = (rem > 0) ? rem : 0;
                    int wi = p >> 3;
                    int bi = 7 - (p & 7);
                    hm[wi] |= ((ulong)0x80UL) << (bi * 8);
                }
                hm[15] = (ulong)(128 + mlen) * 8UL;
                {{BASE_ALGO}}_block(&istate[0], hm);
            }
        }

        /* Step 3: outer = SHA512((key ^ opad) || inner_hash). istate[8]
         * is BE; all 8 words copied directly into M[0..7] (no truncation,
         * vs SHA-384's 6-word truncation). M[8] = 0x8000000000000000UL
         * (0x80 BE at byte 64 = high octet of word 8); M[15] = (128 + 64)
         * * 8 = 1536. Slab oracle line 111-127 / 225-238. */
        ulong opad_block[16];
        for (int i = 0; i < 16; i++)
            opad_block[i] = key_block[i] ^ 0x5c5c5c5c5c5c5c5cUL;
        for (int i = 0; i < 16; i++)
            hm[i] = bswap64(opad_block[i]);

        ulong ostate[8] = {
            0x6a09e667f3bcc908UL, 0xbb67ae8584caa73bUL,
            0x3c6ef372fe94f82bUL, 0xa54ff53a5f1d36f1UL,
            0x510e527fade682d1UL, 0x9b05688c2b3e6c1fUL,
            0x1f83d9abfb41bd6bUL, 0x5be0cd19137e2179UL };
        {{BASE_ALGO}}_block(&ostate[0], hm);

        for (int i = 0; i < 8; i++) hm[i] = istate[i];
        hm[8] = 0x8000000000000000UL;   /* 0x80 BE at byte 64 */
        for (int i = 9; i < 15; i++) hm[i] = 0UL;
        hm[14] = 0UL;
        hm[15] = (ulong)(128 + 64) * 8UL;   /* 1536 bits */
        {{BASE_ALGO}}_block(&ostate[0], hm);

        /* Write final BE state into st->state[0..7]; template_state_to_h
         * decomposes into st->h[0..15] (16 LE uint32 words = 64 bytes).
         * No truncation (vs SHA-384's 6-word state). */
        for (int i = 0; i < 8; i++) st->state[i] = ostate[i];
        template_state_to_h(st);
        return;
    }

    /* Family E carrier-only suppression: when this fragment is generated
     * for the SHA-384 spec (HASH_WORDS=12), modes < 5 fall through to the
     * SHA384(salt||pass) main body below. No JOB_SHA384SALTPASS algorithm
     * exists in mdxfind; the host never sets algo_mode<5 for this kernel
     * in production. The body below executes correctly should a future
     * SHA384SALTPASS algorithm be added. Family F (HASH_WORDS=16): modes
     * < 5 fall through to the production SHA512(salt||pass) main body
     * (JOB_SHA512SALTPASS, algo_mode=0). */
    (void)algo_mode;  /* Family E/F: actively used above; no-op for modes 0..4 */
    ulong M[16];
    int total_len = (int)slen + len;
    int pos = 0;

    /* Process complete 128-byte blocks. Build M[] BIG-ENDIAN directly
     * from bytes; salt bytes for p < slen, password bytes for p >= slen.
     * Per-byte BE position: byte i into word (i >> 3) at shift
     * (7 - (i & 7)) * 8. */
    while (total_len - pos >= HASH_BLOCK_BYTES) {
        for (int j = 0; j < 16; j++) M[j] = 0UL;
        for (int i = 0; i < HASH_BLOCK_BYTES; i++) {
            int p = pos + i;
            uchar c;
            if (p < (int)slen) {
                c = salt_buf[p];
            } else {
                c = data[p - (int)slen];
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
        if (p < (int)slen) {
            c = salt_buf[p];
        } else {
            c = data[p - (int)slen];
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
