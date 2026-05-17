/* template_finalize: compute {{BASE_ALGO}}(salt || pass).
 *
 * BIG-ENDIAN variant for SHA-family algorithms (SHA1/SHA224/SHA256/
 * SHA384/SHA512 — only SHA1 ships first; the others reuse this fragment
 * with HASH_WORDS / IV adjustments via the spec).
 *
 * Layout in M[]: salt bytes first (offset 0..slen-1), password bytes
 * second (offset slen..slen+plen-1), then 0x80 padding marker, then
 * zeros, then 64-bit BE length-in-bits in M[14..15] (M[14] = high 32
 * bits, M[15] = low 32 bits). Block boundary handled the same way as
 * gpu_{{BASE_ALGO}}_core.cl template_finalize: if the tail (after 0x80)
 * extends past byte 55 we run a second block whose last 8 bytes hold
 * the length.
 *
 * Each byte's position within its 32-bit word is determined by
 * 3-(byte_idx & 3) shift (BE: byte 0 is high octet of word). This
 * matches gpu_{{BASE_ALGO}}_core.cl's BE M[] build for the unsalted path
 * — only the source of the bytes (salt|pass instead of just data)
 * differs. */
static inline void template_finalize(template_state *st,
                                     const uchar *data, int len,
                                     __global const uchar *salt_buf,
                                     uint slen,
                                     uint algo_mode)
{
    uint M[16];
    int total_len = (int)slen + len;
    int pos = 0;

    /* IV install is the responsibility of template_init() — called by
     * template_phase0 BEFORE template_finalize. Per-algorithm IV constants
     * (SHA1 vs SHA256, 5 vs 8 words) live in the per-core file, not in
     * this fragment. (The earlier SHA1 sibling carried a defensive
     * SHA1-IV reinstall here; harmless on SHA1 but wrong-width on SHA256.
     * Removed in B6.2 SHA256 fan-out 2026-05-06.) */

    /* Family B (2026-05-07): HMAC-SHA1 branch. Modes 5 (KSALT, e215) and
     * 6 (KPASS, e793) compute HMAC-SHA1 and return early — no SHA1(salt||pass)
     * concatenation applies. Mirrors gpu_sha1.cl hmac_sha1_ksalt_batch /
     * hmac_sha1_kpass_batch (slab oracle).
     *
     *   Mode 5 (HMAC_SHA1_KSALT, e215): key = salt_buf[0..slen),
     *                                   msg = data[0..len).
     *   Mode 6 (HMAC_SHA1_KPASS, e793): key = data[0..len),
     *                                   msg = salt_buf[0..slen).
     *
     * HMAC(K, M) = SHA1((K^opad) || SHA1((K^ipad) || M)). Block size = 64
     * bytes; ipad = 0x36, opad = 0x5c. SHA1 reads message words BIG-ENDIAN
     * — bswap32 needed for ipad/opad blocks (key_block stored LE).
     *
     * Family C (2026-05-07): HMAC-SHA224 branch lives in a sibling
     * `if (HASH_WORDS == 7 && algo_mode >= 5u)` block below.
     * Family D (2026-05-08): HMAC-SHA256 branch lives in a sibling
     * `if (HASH_WORDS == 8 && algo_mode >= 5u)` block below. All three
     * branches share the same algo_mode encoding (5=KSALT, 6=KPASS) and
     * differ only by IV / state width / digest truncation / outer-block
     * length. The three branches are mutually exclusive at runtime
     * (HASH_WORDS is fixed per generated core; the conditions cannot
     * both be true).
     *
     * GATED on HASH_WORDS == 5: only SHA1 (5 BE words) gets the SHA1
     * HMAC branch. SHA224 (HASH_WORDS=7) and SHA256 (HASH_WORDS=8) cores
     * substitute {{BASE_ALGO}}=sha256 in this fragment and use their own
     * gated branches. The HASH_WORDS==5 gate is defense-in-depth so a
     * future algo_mode-bearing JOB doesn't accidentally run SHA1 IV/state
     * width through SHA224/SHA256 cores. */
    if (HASH_WORDS == 5 && algo_mode >= 5u) {
        /* Resolve key + message per mode. Build a 64-byte key_block
         * (uint LE words) for the ipad/opad XOR. The slab oracle
         * (gpu_sha1.cl hmac_sha1_ksalt_batch line 73-99 and hmac_sha1_-
         * kpass_batch line 190-213) uses the same uint LE byte-pack
         * pattern — we mirror it byte-for-byte. */
        uint key_block[16];
        for (int i = 0; i < 16; i++) key_block[i] = 0;

        int klen;
        int mlen;
        /* Step 1: prepare 64-byte key_block. If key > 64 bytes, replace
         * with SHA1(key) (20 bytes). Else pad to 64 with zeros. */
        if (algo_mode == 5u) {
            /* KSALT: key = salt (global). */
            klen = (int)slen;
            mlen = len;
            if (klen > 64) {
                /* Hash global key into key_block[0..4] (5 BE words bswapped
                 * to LE for storage in key_block). Slab oracle line 76-90. */
                uint kstate[5] = { 0x67452301u, 0xEFCDAB89u, 0x98BADCFEu,
                                   0x10325476u, 0xC3D2E1F0u };
                uint kM[16];
                int copy1 = (klen < 64) ? klen : 64;
                for (int i = 0; i < 16; i++) kM[i] = 0;
                /* SHA1 = BIG-ENDIAN: byte i shift = (3 - (i & 3)) * 8. */
                for (int i = 0; i < copy1; i++) {
                    uint v = (uint)salt_buf[i];
                    kM[i >> 2] |= v << ((3 - (i & 3)) * 8);
                }
                {{BASE_ALGO}}_block(&kstate[0], kM);
                for (int i = 0; i < 16; i++) kM[i] = 0;
                int rem = klen - 64;
                if (rem > 0) {
                    for (int i = 0; i < rem; i++) {
                        uint v = (uint)salt_buf[64 + i];
                        kM[i >> 2] |= v << ((3 - (i & 3)) * 8);
                    }
                }
                /* 0x80 marker, BE position. */
                {
                    int p = (rem > 0) ? rem : 0;
                    kM[p >> 2] |= ((uint)0x80u) << ((3 - (p & 3)) * 8);
                }
                /* BE length: M[14]=high, M[15]=low. */
                kM[14] = 0;
                kM[15] = (uint)klen * 8u;
                {{BASE_ALGO}}_block(&kstate[0], kM);
                /* Store hashed key as LE uint32 words (slab oracle line 90:
                 * `key_block[i] = bswap32(kstate[i])`). */
                for (int i = 0; i < 5; i++) key_block[i] = bswap32(kstate[i]);
                klen = 20;
            } else {
                /* Pack key bytes into key_block[0..15] uint LE words.
                 * Slab oracle line 93-98. */
                for (int i = 0; i < klen; i++) {
                    uint v = (uint)salt_buf[i];
                    key_block[i >> 2] |= v << ((i & 3) * 8);
                }
            }
        } else {
            /* Mode 6 KPASS: key = data (private = post-rule password). */
            klen = len;
            mlen = (int)slen;
            if (klen > 64) {
                uint kstate[5] = { 0x67452301u, 0xEFCDAB89u, 0x98BADCFEu,
                                   0x10325476u, 0xC3D2E1F0u };
                uint kM[16];
                int copy1 = (klen < 64) ? klen : 64;
                for (int i = 0; i < 16; i++) kM[i] = 0;
                for (int i = 0; i < copy1; i++) {
                    uint v = (uint)data[i];
                    kM[i >> 2] |= v << ((3 - (i & 3)) * 8);
                }
                {{BASE_ALGO}}_block(&kstate[0], kM);
                for (int i = 0; i < 16; i++) kM[i] = 0;
                int rem = klen - 64;
                if (rem > 0) {
                    for (int i = 0; i < rem; i++) {
                        uint v = (uint)data[64 + i];
                        kM[i >> 2] |= v << ((3 - (i & 3)) * 8);
                    }
                }
                {
                    int p = (rem > 0) ? rem : 0;
                    kM[p >> 2] |= ((uint)0x80u) << ((3 - (p & 3)) * 8);
                }
                kM[14] = 0;
                kM[15] = (uint)klen * 8u;
                {{BASE_ALGO}}_block(&kstate[0], kM);
                for (int i = 0; i < 5; i++) key_block[i] = bswap32(kstate[i]);
                klen = 20;
            } else {
                for (int i = 0; i < klen; i++) {
                    uint v = (uint)data[i];
                    key_block[i >> 2] |= v << ((i & 3) * 8);
                }
            }
        }

        /* Step 2: inner = SHA1((key ^ ipad) || message). The key_block is
         * uint LE; XOR with 0x36363636u then bswap32 to BE for sha1_block.
         * Slab oracle line 102-110 / 215-223. */
        uint ipad[16];
        uint hm[16];
        for (int i = 0; i < 16; i++)
            ipad[i] = key_block[i] ^ 0x36363636u;
        for (int i = 0; i < 16; i++)
            hm[i] = bswap32(ipad[i]);

        uint istate[5] = { 0x67452301u, 0xEFCDAB89u, 0x98BADCFEu,
                           0x10325476u, 0xC3D2E1F0u };
        {{BASE_ALGO}}_block(&istate[0], hm);

        /* Continue with message bytes + 0x80 + length; total = 64 (ipad)
         * + mlen. SHA1 BE byte-pack. */
        if (algo_mode == 5u) {
            /* msg = data (private). Slab oracle line 113-130. */
            if (mlen <= 55) {
                for (int i = 0; i < 16; i++) hm[i] = 0;
                for (int i = 0; i < mlen; i++) {
                    uint v = (uint)data[i];
                    hm[i >> 2] |= v << ((3 - (i & 3)) * 8);
                }
                hm[mlen >> 2] |= ((uint)0x80u) << ((3 - (mlen & 3)) * 8);
                hm[14] = 0;
                hm[15] = (uint)(64 + mlen) * 8u;
                {{BASE_ALGO}}_block(&istate[0], hm);
            } else {
                int copy1 = (mlen < 64) ? mlen : 64;
                for (int i = 0; i < 16; i++) hm[i] = 0;
                for (int i = 0; i < copy1; i++) {
                    uint v = (uint)data[i];
                    hm[i >> 2] |= v << ((3 - (i & 3)) * 8);
                }
                if (mlen < 64)
                    hm[mlen >> 2] |= ((uint)0x80u) << ((3 - (mlen & 3)) * 8);
                {{BASE_ALGO}}_block(&istate[0], hm);
                int rem = mlen - 64;
                for (int i = 0; i < 16; i++) hm[i] = 0;
                if (rem > 0) {
                    for (int i = 0; i < rem; i++) {
                        uint v = (uint)data[64 + i];
                        hm[i >> 2] |= v << ((3 - (i & 3)) * 8);
                    }
                }
                if (mlen >= 64) {
                    int p = (rem > 0) ? rem : 0;
                    hm[p >> 2] |= ((uint)0x80u) << ((3 - (p & 3)) * 8);
                }
                hm[14] = 0;
                hm[15] = (uint)(64 + mlen) * 8u;
                {{BASE_ALGO}}_block(&istate[0], hm);
            }
        } else {
            /* Mode 6 KPASS: msg = salt_buf (global). Slab oracle line 226-241. */
            if (mlen <= 55) {
                for (int i = 0; i < 16; i++) hm[i] = 0;
                for (int i = 0; i < mlen; i++) {
                    uint v = (uint)salt_buf[i];
                    hm[i >> 2] |= v << ((3 - (i & 3)) * 8);
                }
                hm[mlen >> 2] |= ((uint)0x80u) << ((3 - (mlen & 3)) * 8);
                hm[14] = 0;
                hm[15] = (uint)(64 + mlen) * 8u;
                {{BASE_ALGO}}_block(&istate[0], hm);
            } else {
                int copy1 = (mlen < 64) ? mlen : 64;
                for (int i = 0; i < 16; i++) hm[i] = 0;
                for (int i = 0; i < copy1; i++) {
                    uint v = (uint)salt_buf[i];
                    hm[i >> 2] |= v << ((3 - (i & 3)) * 8);
                }
                if (mlen < 64)
                    hm[mlen >> 2] |= ((uint)0x80u) << ((3 - (mlen & 3)) * 8);
                {{BASE_ALGO}}_block(&istate[0], hm);
                int rem = mlen - 64;
                for (int i = 0; i < 16; i++) hm[i] = 0;
                if (rem > 0) {
                    for (int i = 0; i < rem; i++) {
                        uint v = (uint)salt_buf[64 + i];
                        hm[i >> 2] |= v << ((3 - (i & 3)) * 8);
                    }
                }
                if (mlen >= 64) {
                    int p = (rem > 0) ? rem : 0;
                    hm[p >> 2] |= ((uint)0x80u) << ((3 - (p & 3)) * 8);
                }
                hm[14] = 0;
                hm[15] = (uint)(64 + mlen) * 8u;
                {{BASE_ALGO}}_block(&istate[0], hm);
            }
        }

        /* Step 3: outer = SHA1((key ^ opad) || inner_hash). istate[5] is
         * BE; copied directly into M[0..4]. M[5] = 0x80000000u (BE 0x80 at
         * byte 20 = high octet of word 5). M[14]=0, M[15]=672 bits.
         * Slab oracle line 134-150 / 244-257. */
        uint opad_block[16];
        for (int i = 0; i < 16; i++)
            opad_block[i] = key_block[i] ^ 0x5c5c5c5cu;
        for (int i = 0; i < 16; i++)
            hm[i] = bswap32(opad_block[i]);

        uint ostate[5] = { 0x67452301u, 0xEFCDAB89u, 0x98BADCFEu,
                           0x10325476u, 0xC3D2E1F0u };
        {{BASE_ALGO}}_block(&ostate[0], hm);

        for (int i = 0; i < 5; i++) hm[i] = istate[i];
        hm[5] = 0x80000000u;
        for (int i = 6; i < 15; i++) hm[i] = 0;
        hm[14] = 0;
        hm[15] = (64u + 20u) * 8u;     /* 672 bits */
        {{BASE_ALGO}}_block(&ostate[0], hm);

        /* Write final BE state into st->h[0..4] (template_digest_compare
         * bswaps 4 leading words to LE for compact-table probe; emit_hit
         * bswaps 5 words for hits buffer). */
        for (int i = 0; i < 5; i++) st->h[i] = ostate[i];
        return;
    }

    /* Family C (2026-05-07): HMAC-SHA224 branch. Modes 5 (KSALT, e216)
     * and 6 (KPASS, e794) compute HMAC-SHA224 and return early — no
     * SHA224(salt||pass) concatenation applies. Mirrors gpu_sha256.cl
     * hmac_sha224_ksalt_batch / hmac_sha224_kpass_batch (slab oracle).
     *
     *   Mode 5 (HMAC_SHA224_KSALT, e216): key = salt_buf[0..slen),
     *                                     msg = data[0..len).
     *   Mode 6 (HMAC_SHA224_KPASS, e794): key = data[0..len),
     *                                     msg = salt_buf[0..slen).
     *
     * HMAC(K, M) = SHA224((K^opad) || SHA224((K^ipad) || M)). Block size
     * = 64 bytes; ipad = 0x36, opad = 0x5c. SHA224 uses sha256_block
     * compression (8-word state), digest truncated to 7 words = 28 bytes.
     * Outer block is 64 + 28 = 92 bytes (2 sha256_block calls): the first
     * holds K^opad, the second holds the 28-byte inner digest + 0x80
     * padding marker + 64-bit BE length (736 bits). M[7] = 0x80000000u
     * (0x80 at byte 28 = high octet of word 7); M[15] = 736.
     *
     * GATED on HASH_WORDS == 7. The branch is structurally distinct from
     * the SHA1 branch (HASH_WORDS==5) above and the SHA256 branch
     * (HASH_WORDS==8) below — different IV, state width, digest width,
     * and outer-block length. All three branches share the algo_mode 5/6
     * encoding (KSALT/KPASS); SHA224 and SHA256 branches share BASE_ALGO=
     * sha256_block here, while SHA1 substitutes sha1_block. Only one
     * branch is ever live in any generated core (HASH_WORDS is fixed
     * per spec at codegen time). */
    if (HASH_WORDS == 7 && algo_mode >= 5u) {
        /* Resolve key + message per mode. Build a 64-byte key_block
         * (uint LE words) for the ipad/opad XOR. The slab oracle
         * (gpu_sha256.cl hmac_sha224_ksalt_batch line 295-316 and hmac_-
         * sha224_kpass_batch line 391-412) uses the same uint LE byte-
         * pack pattern — we mirror it byte-for-byte. */
        uint key_block[16];
        for (int i = 0; i < 16; i++) key_block[i] = 0;

        int klen;
        int mlen;
        /* Step 1: prepare 64-byte key_block. If key > 64 bytes, replace
         * with SHA224(key) (28 bytes). Else pad to 64 with zeros. */
        if (algo_mode == 5u) {
            /* KSALT: key = salt (global). */
            klen = (int)slen;
            mlen = len;
            if (klen > 64) {
                /* Hash global key into key_block[0..6] (7 BE words bswapped
                 * to LE for storage in key_block). Slab oracle line 297-
                 * 312. State is 8 words but only 7 are stored. */
                uint kstate[8] = { 0xc1059ed8u, 0x367cd507u, 0x3070dd17u,
                                   0xf70e5939u, 0xffc00b31u, 0x68581511u,
                                   0x64f98fa7u, 0xbefa4fa4u };
                uint kM[16];
                int copy1 = (klen < 64) ? klen : 64;
                for (int i = 0; i < 16; i++) kM[i] = 0;
                /* SHA224 = BIG-ENDIAN: byte i shift = (3 - (i & 3)) * 8. */
                for (int i = 0; i < copy1; i++) {
                    uint v = (uint)salt_buf[i];
                    kM[i >> 2] |= v << ((3 - (i & 3)) * 8);
                }
                if (klen <= 55) {
                    kM[klen >> 2] |= ((uint)0x80u) << ((3 - (klen & 3)) * 8);
                    kM[14] = 0;
                    kM[15] = (uint)klen * 8u;
                }
                {{BASE_ALGO}}_block(&kstate[0], kM);
                if (klen > 55) {
                    for (int i = 0; i < 16; i++) kM[i] = 0;
                    int rem = klen - 64;
                    if (rem > 0) {
                        for (int i = 0; i < rem; i++) {
                            uint v = (uint)salt_buf[64 + i];
                            kM[i >> 2] |= v << ((3 - (i & 3)) * 8);
                        }
                    }
                    {
                        int p = (rem > 0) ? rem : 0;
                        kM[p >> 2] |= ((uint)0x80u) << ((3 - (p & 3)) * 8);
                    }
                    kM[14] = 0;
                    kM[15] = (uint)klen * 8u;
                    {{BASE_ALGO}}_block(&kstate[0], kM);
                }
                /* Store hashed key as LE uint32 words (slab oracle line
                 * 311: `key_block[i] = bswap32(kstate[i])`). 7 words. */
                for (int i = 0; i < 7; i++) key_block[i] = bswap32(kstate[i]);
                klen = 28;
            } else {
                /* Pack key bytes into key_block[0..15] uint LE words.
                 * Slab oracle line 314-316. */
                for (int i = 0; i < klen; i++) {
                    uint v = (uint)salt_buf[i];
                    key_block[i >> 2] |= v << ((i & 3) * 8);
                }
            }
        } else {
            /* Mode 6 KPASS: key = data (private = post-rule password). */
            klen = len;
            mlen = (int)slen;
            if (klen > 64) {
                uint kstate[8] = { 0xc1059ed8u, 0x367cd507u, 0x3070dd17u,
                                   0xf70e5939u, 0xffc00b31u, 0x68581511u,
                                   0x64f98fa7u, 0xbefa4fa4u };
                uint kM[16];
                int copy1 = (klen < 64) ? klen : 64;
                for (int i = 0; i < 16; i++) kM[i] = 0;
                for (int i = 0; i < copy1; i++) {
                    uint v = (uint)data[i];
                    kM[i >> 2] |= v << ((3 - (i & 3)) * 8);
                }
                if (klen <= 55) {
                    kM[klen >> 2] |= ((uint)0x80u) << ((3 - (klen & 3)) * 8);
                    kM[14] = 0;
                    kM[15] = (uint)klen * 8u;
                }
                {{BASE_ALGO}}_block(&kstate[0], kM);
                if (klen > 55) {
                    for (int i = 0; i < 16; i++) kM[i] = 0;
                    int rem = klen - 64;
                    if (rem > 0) {
                        for (int i = 0; i < rem; i++) {
                            uint v = (uint)data[64 + i];
                            kM[i >> 2] |= v << ((3 - (i & 3)) * 8);
                        }
                    }
                    {
                        int p = (rem > 0) ? rem : 0;
                        kM[p >> 2] |= ((uint)0x80u) << ((3 - (p & 3)) * 8);
                    }
                    kM[14] = 0;
                    kM[15] = (uint)klen * 8u;
                    {{BASE_ALGO}}_block(&kstate[0], kM);
                }
                for (int i = 0; i < 7; i++) key_block[i] = bswap32(kstate[i]);
                klen = 28;
            } else {
                for (int i = 0; i < klen; i++) {
                    uint v = (uint)data[i];
                    key_block[i >> 2] |= v << ((i & 3) * 8);
                }
            }
        }

        /* Step 2: inner = SHA224((key ^ ipad) || message). The key_block
         * is uint LE; XOR with 0x36363636u then bswap32 to BE for sha256_-
         * block. Slab oracle line 318-323 / 414-419. */
        uint ipad[16];
        uint hm[16];
        for (int i = 0; i < 16; i++)
            ipad[i] = key_block[i] ^ 0x36363636u;
        for (int i = 0; i < 16; i++)
            hm[i] = bswap32(ipad[i]);

        uint istate[8] = { 0xc1059ed8u, 0x367cd507u, 0x3070dd17u,
                           0xf70e5939u, 0xffc00b31u, 0x68581511u,
                           0x64f98fa7u, 0xbefa4fa4u };
        {{BASE_ALGO}}_block(&istate[0], hm);

        /* Continue with message bytes + 0x80 + length; total = 64 (ipad)
         * + mlen. SHA224 BE byte-pack. */
        if (algo_mode == 5u) {
            /* msg = data (private). Slab oracle line 325-341. */
            if (mlen <= 55) {
                for (int i = 0; i < 16; i++) hm[i] = 0;
                for (int i = 0; i < mlen; i++) {
                    uint v = (uint)data[i];
                    hm[i >> 2] |= v << ((3 - (i & 3)) * 8);
                }
                hm[mlen >> 2] |= ((uint)0x80u) << ((3 - (mlen & 3)) * 8);
                hm[14] = 0;
                hm[15] = (uint)(64 + mlen) * 8u;
                {{BASE_ALGO}}_block(&istate[0], hm);
            } else {
                int copy1 = (mlen < 64) ? mlen : 64;
                for (int i = 0; i < 16; i++) hm[i] = 0;
                for (int i = 0; i < copy1; i++) {
                    uint v = (uint)data[i];
                    hm[i >> 2] |= v << ((3 - (i & 3)) * 8);
                }
                if (mlen < 64)
                    hm[mlen >> 2] |= ((uint)0x80u) << ((3 - (mlen & 3)) * 8);
                {{BASE_ALGO}}_block(&istate[0], hm);
                int rem = mlen - 64;
                for (int i = 0; i < 16; i++) hm[i] = 0;
                if (rem > 0) {
                    for (int i = 0; i < rem; i++) {
                        uint v = (uint)data[64 + i];
                        hm[i >> 2] |= v << ((3 - (i & 3)) * 8);
                    }
                }
                if (mlen >= 64) {
                    int p = (rem > 0) ? rem : 0;
                    hm[p >> 2] |= ((uint)0x80u) << ((3 - (p & 3)) * 8);
                }
                hm[14] = 0;
                hm[15] = (uint)(64 + mlen) * 8u;
                {{BASE_ALGO}}_block(&istate[0], hm);
            }
        } else {
            /* Mode 6 KPASS: msg = salt_buf (global). Slab oracle line
             * 421-437. */
            if (mlen <= 55) {
                for (int i = 0; i < 16; i++) hm[i] = 0;
                for (int i = 0; i < mlen; i++) {
                    uint v = (uint)salt_buf[i];
                    hm[i >> 2] |= v << ((3 - (i & 3)) * 8);
                }
                hm[mlen >> 2] |= ((uint)0x80u) << ((3 - (mlen & 3)) * 8);
                hm[14] = 0;
                hm[15] = (uint)(64 + mlen) * 8u;
                {{BASE_ALGO}}_block(&istate[0], hm);
            } else {
                int copy1 = (mlen < 64) ? mlen : 64;
                for (int i = 0; i < 16; i++) hm[i] = 0;
                for (int i = 0; i < copy1; i++) {
                    uint v = (uint)salt_buf[i];
                    hm[i >> 2] |= v << ((3 - (i & 3)) * 8);
                }
                if (mlen < 64)
                    hm[mlen >> 2] |= ((uint)0x80u) << ((3 - (mlen & 3)) * 8);
                {{BASE_ALGO}}_block(&istate[0], hm);
                int rem = mlen - 64;
                for (int i = 0; i < 16; i++) hm[i] = 0;
                if (rem > 0) {
                    for (int i = 0; i < rem; i++) {
                        uint v = (uint)salt_buf[64 + i];
                        hm[i >> 2] |= v << ((3 - (i & 3)) * 8);
                    }
                }
                if (mlen >= 64) {
                    int p = (rem > 0) ? rem : 0;
                    hm[p >> 2] |= ((uint)0x80u) << ((3 - (p & 3)) * 8);
                }
                hm[14] = 0;
                hm[15] = (uint)(64 + mlen) * 8u;
                {{BASE_ALGO}}_block(&istate[0], hm);
            }
        }

        /* Step 3: outer = SHA224((key ^ opad) || inner_hash). istate[7]
         * is BE; first 7 words copied directly into M[0..6]. M[7] =
         * 0x80000000u (BE 0x80 at byte 28 = high octet of word 7).
         * M[14]=0, M[15] = 736 bits (= 92 bytes * 8). Slab oracle line
         * 343-355 / 439-450. */
        uint opad_block[16];
        for (int i = 0; i < 16; i++)
            opad_block[i] = key_block[i] ^ 0x5c5c5c5cu;
        for (int i = 0; i < 16; i++)
            hm[i] = bswap32(opad_block[i]);

        uint ostate[8] = { 0xc1059ed8u, 0x367cd507u, 0x3070dd17u,
                           0xf70e5939u, 0xffc00b31u, 0x68581511u,
                           0x64f98fa7u, 0xbefa4fa4u };
        {{BASE_ALGO}}_block(&ostate[0], hm);

        for (int i = 0; i < 7; i++) hm[i] = istate[i];
        hm[7] = 0x80000000u;
        for (int i = 8; i < 15; i++) hm[i] = 0;
        hm[14] = 0;
        hm[15] = (64u + 28u) * 8u;     /* 736 bits */
        {{BASE_ALGO}}_block(&ostate[0], hm);

        /* Write final BE state into st->h[0..6] (template_digest_compare
         * bswaps 4 leading words to LE for compact-table probe; emit_hit
         * bswaps 7 words for hits buffer). h[7] is left untouched —
         * SHA224 truncates to 7 words. */
        for (int i = 0; i < 7; i++) st->h[i] = ostate[i];
        return;
    }

    /* Family D HMAC-SHA256 (e217 + e795), HASH_WORDS == 8.
     *
     * CRITICAL: This MUST be a runtime `if (HASH_WORDS == 8 && ...)`,
     * NEVER `#if HASH_WORDS == 8`. Rev 1.7 (2026-05-07) converted to
     * #if and produced 0 hits on Pascal NVIDIA at packed_count > 1.
     * NVCC's IR-lowering / register-allocator / spill scheduler emits
     * different PTX for single-body (preprocessor-stripped) source
     * vs three-bodies-with-DCE source. The single-body PTX is broken
     * on Pascal at HASH_WORDS=8 specifically. Diagnostic agent
     * a97e0c9ac7747151e (2026-05-08) validated the runtime form
     * produces 100/100 cracks on the same fixture rev 1.7 failed at
     * 0/100. DO NOT optimize this gate to preprocessor form. The
     * runtime gate is the convention shared with Families B, C, E,
     * F, G, H, J, K — all use `if (HASH_WORDS == N && algo_mode >=
     * 5u)`. The Family D ABORT (2026-05-07) and re-ship (2026-05-08)
     * difference is exactly this gate form: runtime ships, #if breaks
     * Pascal. Mirrors gpu_sha256.cl hmac_sha256_ksalt_batch +
     * hmac_sha256_kpass_batch (slab oracle).
     *
     *   Mode 5 (HMAC_SHA256_KSALT, e217): key = salt_buf[0..slen),
     *                                     msg = data[0..len).
     *   Mode 6 (HMAC_SHA256_KPASS, e795): key = data[0..len),
     *                                     msg = salt_buf[0..slen).
     *
     * HMAC(K, M) = SHA256((K^opad) || SHA256((K^ipad) || M)). Block size
     * = 64 bytes; ipad = 0x36, opad = 0x5c. SHA256 uses sha256_block
     * compression (8-word state); digest is full 32 bytes (no truncation).
     * Outer block is 64 + 32 = 96 bytes (2 sha256_block calls): the first
     * holds K^opad, the second holds the 32-byte inner digest + 0x80
     * padding marker + 64-bit BE length (768 bits). M[8] = 0x80000000u
     * (0x80 at byte 32 = high octet of word 8); M[15] = 768. */
    if (HASH_WORDS == 8 && algo_mode >= 5u) {
        /* Resolve key + message per mode. Build a 64-byte key_block
         * (uint LE words) for the ipad/opad XOR. The slab oracle
         * (gpu_sha256.cl hmac_sha256_ksalt_batch line 50-80 and
         * hmac_sha256_kpass_batch line 173-200) uses the same uint LE
         * byte-pack pattern — we mirror it byte-for-byte. */
        uint key_block[16];
        for (int i = 0; i < 16; i++) key_block[i] = 0;

        int klen;
        int mlen;
        /* Step 1: prepare 64-byte key_block. If key > 64 bytes, replace
         * with SHA256(key) (32 bytes). Else pad to 64 with zeros. */
        if (algo_mode == 5u) {
            /* KSALT: key = salt (global). */
            klen = (int)slen;
            mlen = len;
            if (klen > 64) {
                /* Hash global key into key_block[0..7] (8 BE words bswapped
                 * to LE for storage in key_block). SHA256 IV (FIPS 180-4
                 * §5.3.3). */
                uint kstate[8] = { 0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u,
                                   0xa54ff53au, 0x510e527fu, 0x9b05688cu,
                                   0x1f83d9abu, 0x5be0cd19u };
                uint kM[16];
                int copy1 = (klen < 64) ? klen : 64;
                for (int i = 0; i < 16; i++) kM[i] = 0;
                /* SHA256 = BIG-ENDIAN: byte i shift = (3 - (i & 3)) * 8. */
                for (int i = 0; i < copy1; i++) {
                    uint v = (uint)salt_buf[i];
                    kM[i >> 2] |= v << ((3 - (i & 3)) * 8);
                }
                if (klen <= 55) {
                    kM[klen >> 2] |= ((uint)0x80u) << ((3 - (klen & 3)) * 8);
                    kM[14] = 0;
                    kM[15] = (uint)klen * 8u;
                }
                {{BASE_ALGO}}_block(&kstate[0], kM);
                if (klen > 55) {
                    for (int i = 0; i < 16; i++) kM[i] = 0;
                    int rem = klen - 64;
                    if (rem > 0) {
                        for (int i = 0; i < rem; i++) {
                            uint v = (uint)salt_buf[64 + i];
                            kM[i >> 2] |= v << ((3 - (i & 3)) * 8);
                        }
                    }
                    {
                        int p = (rem > 0) ? rem : 0;
                        kM[p >> 2] |= ((uint)0x80u) << ((3 - (p & 3)) * 8);
                    }
                    kM[14] = 0;
                    kM[15] = (uint)klen * 8u;
                    {{BASE_ALGO}}_block(&kstate[0], kM);
                }
                /* Store hashed key as LE uint32 words (slab oracle line
                 * 71: `key_block[i] = bswap32(kstate[i])`). All 8 words. */
                for (int i = 0; i < 8; i++) key_block[i] = bswap32(kstate[i]);
                klen = 32;
            } else {
                /* Pack key bytes into key_block[0..15] uint LE words.
                 * Slab oracle line 75-79. */
                for (int i = 0; i < klen; i++) {
                    uint v = (uint)salt_buf[i];
                    key_block[i >> 2] |= v << ((i & 3) * 8);
                }
            }
        } else {
            /* Mode 6 KPASS: key = data (private = post-rule password). */
            klen = len;
            mlen = (int)slen;
            if (klen > 64) {
                uint kstate[8] = { 0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u,
                                   0xa54ff53au, 0x510e527fu, 0x9b05688cu,
                                   0x1f83d9abu, 0x5be0cd19u };
                uint kM[16];
                int copy1 = (klen < 64) ? klen : 64;
                for (int i = 0; i < 16; i++) kM[i] = 0;
                for (int i = 0; i < copy1; i++) {
                    uint v = (uint)data[i];
                    kM[i >> 2] |= v << ((3 - (i & 3)) * 8);
                }
                if (klen <= 55) {
                    kM[klen >> 2] |= ((uint)0x80u) << ((3 - (klen & 3)) * 8);
                    kM[14] = 0;
                    kM[15] = (uint)klen * 8u;
                }
                {{BASE_ALGO}}_block(&kstate[0], kM);
                if (klen > 55) {
                    for (int i = 0; i < 16; i++) kM[i] = 0;
                    int rem = klen - 64;
                    if (rem > 0) {
                        for (int i = 0; i < rem; i++) {
                            uint v = (uint)data[64 + i];
                            kM[i >> 2] |= v << ((3 - (i & 3)) * 8);
                        }
                    }
                    {
                        int p = (rem > 0) ? rem : 0;
                        kM[p >> 2] |= ((uint)0x80u) << ((3 - (p & 3)) * 8);
                    }
                    kM[14] = 0;
                    kM[15] = (uint)klen * 8u;
                    {{BASE_ALGO}}_block(&kstate[0], kM);
                }
                for (int i = 0; i < 8; i++) key_block[i] = bswap32(kstate[i]);
                klen = 32;
            } else {
                for (int i = 0; i < klen; i++) {
                    uint v = (uint)data[i];
                    key_block[i >> 2] |= v << ((i & 3) * 8);
                }
            }
        }

        /* Step 2: inner = SHA256((key ^ ipad) || message). The key_block
         * is uint LE; XOR with 0x36363636u then bswap32 to BE for sha256_-
         * block. Slab oracle line 83-92 / 203-211. */
        uint ipad[16];
        uint hm[16];
        for (int i = 0; i < 16; i++)
            ipad[i] = key_block[i] ^ 0x36363636u;
        for (int i = 0; i < 16; i++)
            hm[i] = bswap32(ipad[i]);

        uint istate[8] = { 0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u,
                           0xa54ff53au, 0x510e527fu, 0x9b05688cu,
                           0x1f83d9abu, 0x5be0cd19u };
        {{BASE_ALGO}}_block(&istate[0], hm);

        /* Continue with message bytes + 0x80 + length; total = 64 (ipad)
         * + mlen. SHA256 BE byte-pack. */
        if (algo_mode == 5u) {
            /* msg = data (private). Slab oracle line 95-112. */
            if (mlen <= 55) {
                for (int i = 0; i < 16; i++) hm[i] = 0;
                for (int i = 0; i < mlen; i++) {
                    uint v = (uint)data[i];
                    hm[i >> 2] |= v << ((3 - (i & 3)) * 8);
                }
                hm[mlen >> 2] |= ((uint)0x80u) << ((3 - (mlen & 3)) * 8);
                hm[14] = 0;
                hm[15] = (uint)(64 + mlen) * 8u;
                {{BASE_ALGO}}_block(&istate[0], hm);
            } else {
                int copy1 = (mlen < 64) ? mlen : 64;
                for (int i = 0; i < 16; i++) hm[i] = 0;
                for (int i = 0; i < copy1; i++) {
                    uint v = (uint)data[i];
                    hm[i >> 2] |= v << ((3 - (i & 3)) * 8);
                }
                if (mlen < 64)
                    hm[mlen >> 2] |= ((uint)0x80u) << ((3 - (mlen & 3)) * 8);
                {{BASE_ALGO}}_block(&istate[0], hm);
                int rem = mlen - 64;
                for (int i = 0; i < 16; i++) hm[i] = 0;
                if (rem > 0) {
                    for (int i = 0; i < rem; i++) {
                        uint v = (uint)data[64 + i];
                        hm[i >> 2] |= v << ((3 - (i & 3)) * 8);
                    }
                }
                if (mlen >= 64) {
                    int p = (rem > 0) ? rem : 0;
                    hm[p >> 2] |= ((uint)0x80u) << ((3 - (p & 3)) * 8);
                }
                hm[14] = 0;
                hm[15] = (uint)(64 + mlen) * 8u;
                {{BASE_ALGO}}_block(&istate[0], hm);
            }
        } else {
            /* Mode 6 KPASS: msg = salt_buf (global). Slab oracle line
             * 213-229. */
            if (mlen <= 55) {
                for (int i = 0; i < 16; i++) hm[i] = 0;
                for (int i = 0; i < mlen; i++) {
                    uint v = (uint)salt_buf[i];
                    hm[i >> 2] |= v << ((3 - (i & 3)) * 8);
                }
                hm[mlen >> 2] |= ((uint)0x80u) << ((3 - (mlen & 3)) * 8);
                hm[14] = 0;
                hm[15] = (uint)(64 + mlen) * 8u;
                {{BASE_ALGO}}_block(&istate[0], hm);
            } else {
                int copy1 = (mlen < 64) ? mlen : 64;
                for (int i = 0; i < 16; i++) hm[i] = 0;
                for (int i = 0; i < copy1; i++) {
                    uint v = (uint)salt_buf[i];
                    hm[i >> 2] |= v << ((3 - (i & 3)) * 8);
                }
                if (mlen < 64)
                    hm[mlen >> 2] |= ((uint)0x80u) << ((3 - (mlen & 3)) * 8);
                {{BASE_ALGO}}_block(&istate[0], hm);
                int rem = mlen - 64;
                for (int i = 0; i < 16; i++) hm[i] = 0;
                if (rem > 0) {
                    for (int i = 0; i < rem; i++) {
                        uint v = (uint)salt_buf[64 + i];
                        hm[i >> 2] |= v << ((3 - (i & 3)) * 8);
                    }
                }
                if (mlen >= 64) {
                    int p = (rem > 0) ? rem : 0;
                    hm[p >> 2] |= ((uint)0x80u) << ((3 - (p & 3)) * 8);
                }
                hm[14] = 0;
                hm[15] = (uint)(64 + mlen) * 8u;
                {{BASE_ALGO}}_block(&istate[0], hm);
            }
        }

        /* Step 3: outer = SHA256((key ^ opad) || inner_hash). istate[8]
         * is BE; all 8 words copied directly into M[0..7]. M[8] =
         * 0x80000000u (BE 0x80 at byte 32 = high octet of word 8).
         * M[14]=0, M[15] = 768 bits (= 96 bytes * 8). Slab oracle line
         * 117-133 / 232-246. */
        uint opad_block[16];
        for (int i = 0; i < 16; i++)
            opad_block[i] = key_block[i] ^ 0x5c5c5c5cu;
        for (int i = 0; i < 16; i++)
            hm[i] = bswap32(opad_block[i]);

        uint ostate[8] = { 0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u,
                           0xa54ff53au, 0x510e527fu, 0x9b05688cu,
                           0x1f83d9abu, 0x5be0cd19u };
        {{BASE_ALGO}}_block(&ostate[0], hm);

        for (int i = 0; i < 8; i++) hm[i] = istate[i];
        hm[8] = 0x80000000u;
        for (int i = 9; i < 15; i++) hm[i] = 0;
        hm[14] = 0;
        hm[15] = (64u + 32u) * 8u;     /* 768 bits */
        {{BASE_ALGO}}_block(&ostate[0], hm);

        /* Write final BE state into st->h[0..7] (template_digest_compare
         * bswaps 4 leading words to LE for compact-table probe; emit_hit
         * bswaps all 8 words for hits buffer). SHA256 emits the full
         * 8-word digest — no truncation (unlike SHA224 which drops h[7]). */
        for (int i = 0; i < 8; i++) st->h[i] = ostate[i];
        return;
    }

    /* Process complete 64-byte blocks. Build M[] BIG-ENDIAN directly
     * from bytes; salt bytes for p < slen, password bytes for p >= slen. */
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
        if (p < (int)slen) {
            c = salt_buf[p];
        } else {
            c = data[p - (int)slen];
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
