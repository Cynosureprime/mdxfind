/* template_finalize: three-step double-MD5 chain with runtime variant.
 *
 *   1. Inner MD5: compute MD5(buf[0..len)) into a local digest1 (4 uints).
 *   2. Hex-encode digest1 into M[] using params.algo_mode to select variant:
 *      - 0 (MD5SALT, e31):       lowercase hex, forward order, 32 chars
 *      - 1 (MD5UCSALT, e350):    UPPERCASE hex, forward order, 32 chars
 *      - 2 (MD5revMD5SALT, e541): lowercase hex, REVERSED character order, 32 chars
 *      - 3 (MD5sub8_24SALT, e542): lowercase hex, slice [8:24], 16 chars
 *      - 4 (MD5_MD5SALTMD5PASS, e367): pass-hex (lowercase, 32 chars) at M[8..15]
 *           PLUS pre-computed salt-hex (32 bytes from salt_buf) at M[0..7].
 *           inner_len=64; salt-append loop is overridden to slen=0 (the
 *           salt is part of the 64-byte intermediate, not an outer salt).
 *      - 5 (HMAC_MD5_KSALT, e214): HMAC-MD5 with key=salt, message=password.
 *           Computes HMAC(K, M) = MD5((K^opad) || MD5((K^ipad) || M)).
 *           Branches at TOP of template_finalize and writes result to
 *           st->h[0..3] then returns; bypasses the modes-0-4 double-MD5
 *           code path entirely.
 *      - 6 (HMAC_MD5_KPASS, e792): HMAC-MD5 with key=password, message=salt.
 *           Same HMAC math as mode 5 but key/msg swapped. CPU semantics:
 *           mdxfind.c:29250 (KSALT via mhash_hmac with key=salt) and
 *           mdxfind.c:29423 (KPASS via mhash_hmac with key=pass).
 *           Slab oracle: gpu_md5salt.cl hmac_md5_ksalt_batch (line 33) +
 *           hmac_md5_kpass_batch (line 154).
 *   3. Append salt bytes at byte offset inner_len in M[], add 0x80 padding,
 *      set M[14] = (inner_len+slen)*8 length bits, run a fresh-IV md5_block
 *      to produce the outer digest in st->h[0..3].
 *
 * Salt is fetched INLINE from `__global const uchar *salt_buf` (no helper).
 * For slen > (56 - inner_len) we extend into a second outer block.
 *
 * SHARED-KERNEL NOTE (B6.6, 2026-05-06): JOB_MD5SALT, JOB_MD5UCSALT,
 * JOB_MD5revMD5SALT, JOB_MD5sub8_24SALT all dispatch to the SAME GPU kernel
 * via GPU_TEMPLATE_MD5SALT (=33). Host sets params.algo_mode based on op.
 * No new GPU_TEMPLATE_* enum values; no new specs.py entries; the kernel
 * variant logic is local to this fragment. CPU reference: mdxfind.c:22055-
 * 22072 (the same 4-way switch on job->op around prmd5/prmd5UC/prmd5REV).
 *
 * SHARED-KERNEL NOTE (B6.8, 2026-05-06): JOB_MD5_MD5SALTMD5PASS (e367) joins
 * the SAME GPU kernel via algo_mode=4. Host packs the pre-computed salt-hex
 * (saltsnap[si].hashsalt, 32 chars) into salt_buf instead of raw salt bytes
 * (gpujob_opencl.c gpu_pack_salts already does this when use_hashsalt=1 for
 * JOB_MD5_MD5SALTMD5PASS). CPU reference: mdxfind.c:17027-17075 — outer
 * MD5 over (salt_hex_32 || pass_hex_32) = 64-byte intermediate.
 *
 * SHARED-KERNEL NOTE (Family A, 2026-05-07): JOB_HMAC_MD5 (e214) and
 * JOB_HMAC_MD5_KPASS (e792) join the SAME GPU kernel via algo_mode 5/6.
 * The HMAC body is at the TOP of template_finalize and returns early to
 * bypass the double-MD5 chain entirely (HMAC has no double-hash structure
 * like e31/e367). Slab kernels hmac_md5_ksalt_batch + hmac_md5_kpass_batch
 * are LEFT IN PLACE this commit as the probe_max_dispatch capacity-probe
 * anchor (gpu_opencl.c:1672); they become structurally unreachable for
 * these ops post-category-move (GPU_CAT_SALTPASS -> GPU_CAT_MASK) but the
 * slab dispatcher's gpu_op_category gate prevents any actual entry.
 *
 * Note: params.algo_mode is read DURING template_finalize. Variants other
 * than mode==0 (the legacy default) require GPU_TEMPLATE_HAS_SALT to be
 * defined in defines_str AND the kernel to be a salted variant — modes 1-6
 * are only meaningful for the MD5SALT-family + HMAC-MD5 JOB_*. Other
 * algorithms can reuse the algo_mode slot for their own per-variant flags. */
static inline void template_finalize(template_state *st,
                                     const uchar *data, int len,
                                     __global const uchar *salt_buf,
                                     uint slen,
                                     uint algo_mode)
{
    uint M[16];
    int pos = 0;
    uint h0 = 0x67452301u, h1 = 0xEFCDAB89u,
         h2 = 0x98BADCFEu, h3 = 0x10325476u;

    /* Family A (Family A, 2026-05-07): HMAC-MD5 branch. Modes 5 (KSALT) and
     * 6 (KPASS) compute HMAC-MD5 and return early — no double-MD5 chain
     * applies. Mirrors gpu_md5salt.cl hmac_md5_ksalt_batch / hmac_md5_kpass_-
     * batch (slab oracle).
     *
     *   Mode 5 (HMAC_MD5_KSALT, e214): key = salt_buf[0..slen),
     *                                  msg = data[0..len).
     *   Mode 6 (HMAC_MD5_KPASS, e792): key = data[0..len),
     *                                  msg = salt_buf[0..slen).
     *
     * HMAC(K, M) = MD5((K^opad) || MD5((K^ipad) || M)). Block size = 64
     * bytes; ipad = 0x36, opad = 0x5c. MD5 is little-endian — no bswap
     * needed for key/ipad/opad blocks. */
    if (algo_mode >= 5u) {
        /* Resolve key and message per mode, then build a 64-byte key_block
         * (uint LE words) for the ipad/opad XOR. The earlier draft used
         * uchar-pointer cast `(uchar *)key_block` to write key bytes; on
         * NVIDIA this produced a positional bug where only word_idx == 0
         * computed the correct HMAC. Switching to canonical M[i>>2] |= v
         * << ((i & 3) * 8) byte-pack pattern (same as mode 0's inner-MD5
         * streaming loop) avoids the cross-addrspace alignment risk and
         * matches mode 0's known-correct semantics on all GPUs. */
        uint key_block[16];
        for (int i = 0; i < 16; i++) key_block[i] = 0;

        int klen;
        int mlen;
        /* Step 1: prepare 64-byte key_block. If key > 64 bytes, replace
         * with MD5(key) (16 bytes). Else pad to 64 with zeros. */
        if (algo_mode == 5u) {
            /* KSALT: key = salt (global). */
            klen = (int)slen;
            mlen = len;
            if (klen > 64) {
                /* Hash global key into key_block[0..3] (4 uint LE words). */
                uint kM[16];
                int copy1 = (klen < 64) ? klen : 64;
                for (int i = 0; i < 16; i++) kM[i] = 0;
                for (int i = 0; i < copy1; i++) {
                    uint v = (uint)salt_buf[i];
                    kM[i >> 2] |= v << ((i & 3) * 8);
                }
                if (klen <= 55) {
                    kM[klen >> 2] |= (uint)0x80u << ((klen & 3) * 8);
                    kM[14] = (uint)klen * 8u;
                }
                uint kx = 0x67452301u, ky = 0xEFCDAB89u,
                     kz = 0x98BADCFEu, kw = 0x10325476u;
                {{BASE_ALGO}}_block(&kx, &ky, &kz, &kw, kM);
                if (klen > 55) {
                    int rem = klen - 64;
                    for (int i = 0; i < 16; i++) kM[i] = 0;
                    if (rem > 0) {
                        for (int i = 0; i < rem; i++) {
                            uint v = (uint)salt_buf[64 + i];
                            kM[i >> 2] |= v << ((i & 3) * 8);
                        }
                    }
                    {
                        int p = (rem > 0) ? rem : 0;
                        kM[p >> 2] |= (uint)0x80u << ((p & 3) * 8);
                    }
                    kM[14] = (uint)klen * 8u;
                    {{BASE_ALGO}}_block(&kx, &ky, &kz, &kw, kM);
                }
                key_block[0] = kx; key_block[1] = ky;
                key_block[2] = kz; key_block[3] = kw;
                klen = 16;
            } else {
                /* Pack key bytes into key_block[0..15] uint LE words. */
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
                uint kM[16];
                int copy1 = (klen < 64) ? klen : 64;
                for (int i = 0; i < 16; i++) kM[i] = 0;
                for (int i = 0; i < copy1; i++) {
                    uint v = (uint)data[i];
                    kM[i >> 2] |= v << ((i & 3) * 8);
                }
                if (klen <= 55) {
                    kM[klen >> 2] |= (uint)0x80u << ((klen & 3) * 8);
                    kM[14] = (uint)klen * 8u;
                }
                uint kx = 0x67452301u, ky = 0xEFCDAB89u,
                     kz = 0x98BADCFEu, kw = 0x10325476u;
                {{BASE_ALGO}}_block(&kx, &ky, &kz, &kw, kM);
                if (klen > 55) {
                    int rem = klen - 64;
                    for (int i = 0; i < 16; i++) kM[i] = 0;
                    if (rem > 0) {
                        for (int i = 0; i < rem; i++) {
                            uint v = (uint)data[64 + i];
                            kM[i >> 2] |= v << ((i & 3) * 8);
                        }
                    }
                    {
                        int p = (rem > 0) ? rem : 0;
                        kM[p >> 2] |= (uint)0x80u << ((p & 3) * 8);
                    }
                    kM[14] = (uint)klen * 8u;
                    {{BASE_ALGO}}_block(&kx, &ky, &kz, &kw, kM);
                }
                key_block[0] = kx; key_block[1] = ky;
                key_block[2] = kz; key_block[3] = kw;
                klen = 16;
            } else {
                for (int i = 0; i < klen; i++) {
                    uint v = (uint)data[i];
                    key_block[i >> 2] |= v << ((i & 3) * 8);
                }
            }
        }

        /* Step 2: inner = MD5((key ^ ipad) || message). The ipad block
         * uses key_block (uint LE words) XOR 0x36363636u directly — uniform
         * uint arithmetic, no byte-cast aliasing. The message block reads
         * bytes from `data` (mode 5) or `salt_buf` (mode 6) and packs them
         * into M[i>>2] |= v << ((i & 3) * 8) — same canonical LE byte-pack
         * pattern as mode 0's inner-MD5 streaming loop above (which is
         * known byte-exact correct on all GPUs). The earlier uchar-pointer
         * cast version produced positional bugs on NVIDIA where only
         * word_idx == 0 hashed correctly; uniform uint accumulation avoids
         * the cross-addrspace alignment risk. */
        uint hm[16];
        for (int i = 0; i < 16; i++)
            hm[i] = key_block[i] ^ 0x36363636u;
        uint ihx = 0x67452301u, ihy = 0xEFCDAB89u,
             ihz = 0x98BADCFEu, ihw = 0x10325476u;
        {{BASE_ALGO}}_block(&ihx, &ihy, &ihz, &ihw, hm);

        /* Continue with message + 0x80 + length; total = 64 (ipad block)
         * + mlen. mlen may exceed 64 — slab oracle handles only up to
         * the second message block; we mirror that exactly. */
        if (algo_mode == 5u) {
            /* msg = data (private). */
            if (mlen <= 55) {
                for (int i = 0; i < 16; i++) hm[i] = 0;
                for (int i = 0; i < mlen; i++) {
                    uint v = (uint)data[i];
                    hm[i >> 2] |= v << ((i & 3) * 8);
                }
                hm[mlen >> 2] |= (uint)0x80u << ((mlen & 3) * 8);
                hm[14] = (uint)(64 + mlen) * 8u;
                {{BASE_ALGO}}_block(&ihx, &ihy, &ihz, &ihw, hm);
            } else {
                int copy1 = (mlen < 64) ? mlen : 64;
                for (int i = 0; i < 16; i++) hm[i] = 0;
                for (int i = 0; i < copy1; i++) {
                    uint v = (uint)data[i];
                    hm[i >> 2] |= v << ((i & 3) * 8);
                }
                if (mlen < 64) hm[mlen >> 2] |= (uint)0x80u << ((mlen & 3) * 8);
                {{BASE_ALGO}}_block(&ihx, &ihy, &ihz, &ihw, hm);
                int rem = mlen - 64;
                for (int i = 0; i < 16; i++) hm[i] = 0;
                if (rem > 0) {
                    for (int i = 0; i < rem; i++) {
                        uint v = (uint)data[64 + i];
                        hm[i >> 2] |= v << ((i & 3) * 8);
                    }
                }
                if (mlen >= 64) {
                    int p = (rem > 0) ? rem : 0;
                    hm[p >> 2] |= (uint)0x80u << ((p & 3) * 8);
                }
                hm[14] = (uint)(64 + mlen) * 8u;
                {{BASE_ALGO}}_block(&ihx, &ihy, &ihz, &ihw, hm);
            }
        } else {
            /* Mode 6 KPASS: msg = salt_buf (global). Same M[i>>2] |= v
             * pattern; only the source pointer changes. */
            if (mlen <= 55) {
                for (int i = 0; i < 16; i++) hm[i] = 0;
                for (int i = 0; i < mlen; i++) {
                    uint v = (uint)salt_buf[i];
                    hm[i >> 2] |= v << ((i & 3) * 8);
                }
                hm[mlen >> 2] |= (uint)0x80u << ((mlen & 3) * 8);
                hm[14] = (uint)(64 + mlen) * 8u;
                {{BASE_ALGO}}_block(&ihx, &ihy, &ihz, &ihw, hm);
            } else {
                int copy1 = (mlen < 64) ? mlen : 64;
                for (int i = 0; i < 16; i++) hm[i] = 0;
                for (int i = 0; i < copy1; i++) {
                    uint v = (uint)salt_buf[i];
                    hm[i >> 2] |= v << ((i & 3) * 8);
                }
                if (mlen < 64) hm[mlen >> 2] |= (uint)0x80u << ((mlen & 3) * 8);
                {{BASE_ALGO}}_block(&ihx, &ihy, &ihz, &ihw, hm);
                int rem = mlen - 64;
                for (int i = 0; i < 16; i++) hm[i] = 0;
                if (rem > 0) {
                    for (int i = 0; i < rem; i++) {
                        uint v = (uint)salt_buf[64 + i];
                        hm[i >> 2] |= v << ((i & 3) * 8);
                    }
                }
                if (mlen >= 64) {
                    int p = (rem > 0) ? rem : 0;
                    hm[p >> 2] |= (uint)0x80u << ((p & 3) * 8);
                }
                hm[14] = (uint)(64 + mlen) * 8u;
                {{BASE_ALGO}}_block(&ihx, &ihy, &ihz, &ihw, hm);
            }
        }

        /* Step 3: outer = MD5((key ^ opad) || inner_hash). The inner_hash
         * is 16 bytes (4 uints) at hm[0..3] semantics; we use a fresh hop[]
         * to avoid byte-cast aliasing (same rationale as Step 2). */
        uint hop[16];
        for (int i = 0; i < 16; i++)
            hop[i] = key_block[i] ^ 0x5c5c5c5cu;
        uint ohx = 0x67452301u, ohy = 0xEFCDAB89u,
             ohz = 0x98BADCFEu, ohw = 0x10325476u;
        {{BASE_ALGO}}_block(&ohx, &ohy, &ohz, &ohw, hop);

        for (int i = 0; i < 16; i++) hop[i] = 0;
        hop[0] = ihx; hop[1] = ihy; hop[2] = ihz; hop[3] = ihw;
        hop[4] = 0x80u;          /* 0x80 at byte 16 = LE byte 0 of word 4 */
        hop[14] = (64u + 16u) * 8u; /* 640 bits */
        {{BASE_ALGO}}_block(&ohx, &ohy, &ohz, &ohw, hop);

        st->h[0] = ohx;
        st->h[1] = ohy;
        st->h[2] = ohz;
        st->h[3] = ohw;
        return;
    }

    while (len - pos >= HASH_BLOCK_BYTES) {
        for (int j = 0; j < 16; j++) {
            int b = pos + j * 4;
            M[j] = (uint)data[b]
                 | ((uint)data[b + 1] << 8)
                 | ((uint)data[b + 2] << 16)
                 | ((uint)data[b + 3] << 24);
        }
        {{BASE_ALGO}}_block(&h0, &h1, &h2, &h3, M);
        pos += HASH_BLOCK_BYTES;
    }

    int rem = len - pos;
    for (int j = 0; j < 16; j++) M[j] = 0;
    for (int i = 0; i < rem; i++) {
        uint v = (uint)data[pos + i];
        M[i >> 2] |= v << ((i & 3) * 8);
    }
    M[rem >> 2] |= (uint)0x80 << ((rem & 3) * 8);

    if (rem < 56) {
        M[14] = (uint)(len * 8);
        M[15] = 0;
        {{BASE_ALGO}}_block(&h0, &h1, &h2, &h3, M);
    } else {
        {{BASE_ALGO}}_block(&h0, &h1, &h2, &h3, M);
        for (int j = 0; j < 16; j++) M[j] = 0;
        M[14] = (uint)(len * 8);
        M[15] = 0;
        {{BASE_ALGO}}_block(&h0, &h1, &h2, &h3, M);
    }

    /* Step 2: hex-encode digest1 into M[] per params.algo_mode. Reuses the
     * canonical {{BASE_ALGO}}_to_hex_{lc,uc} helpers from gpu_common.cl —
     * those are byte-exact-validated by the existing e31 path. Modes 2/3
     * post-process the bytes after the canonical encode, so they inherit
     * the same byte-order convention.
     * Result length (`inner_len`) is 32 chars for modes 0/1/2; 16 chars
     * for mode 3; 64 chars for mode 4 (the full 64-byte intermediate is
     * built here from salt-hex + pass-hex). */
    for (int j = 0; j < 16; j++) M[j] = 0;
    uint inner_len;
    if (algo_mode == 1u) {
        /* e350 MD5UCSALT: uppercase hex, forward order. */
        {{BASE_ALGO}}_to_hex_uc(h0, h1, h2, h3, M);
        inner_len = 32u;
    } else if (algo_mode == 2u) {
        /* e541 MD5revMD5SALT: lowercase hex, REVERSED character order across the
         * full 32-char string. CPU prmd5REV at mdxfind.c:4160-4170 reverses
         * the entire 32-char output relative to forward prmd5. Strategy:
         * canonical encode then bytewise reverse M[0..31]. */
        {{BASE_ALGO}}_to_hex_lc(h0, h1, h2, h3, M);
        uchar *mb = (uchar *)M;
        for (int i = 0; i < 16; i++) {
            uchar t = mb[i];
            mb[i] = mb[31 - i];
            mb[31 - i] = t;
        }
        inner_len = 32u;
    } else if (algo_mode == 3u) {
        /* e542 MD5sub8_24SALT: lowercase hex, slice [8:24] = middle 16 chars.
         * CPU at mdxfind.c:22055-22059: prmd5(...,32) then memcpy(mdbuf, mdbuf+8, 16).
         * Strategy: canonical encode → copy bytes [8:24] down to [0:16] →
         * zero bytes [16:32] (the post-encode tail) so the subsequent salt
         * append + 0x80 marker land in clean memory. Without the zero-fill,
         * mb[16..31] retains original hex chars 16..31 from the canonical
         * encode and contaminates the outer MD5 input past the salt+0x80. */
        {{BASE_ALGO}}_to_hex_lc(h0, h1, h2, h3, M);
        uchar *mb = (uchar *)M;
        for (int i = 0; i < 16; i++) mb[i] = mb[8 + i];
        M[4] = 0u; M[5] = 0u; M[6] = 0u; M[7] = 0u;
        inner_len = 16u;
    } else if (algo_mode == 4u) {
        /* e367 MD5_MD5SALTMD5PASS: outer MD5 over the 64-byte intermediate
         * (hex32(MD5(salt)) || hex32(MD5(pass))). The CPU path at
         * mdxfind.c:17027-17075 builds linebuf[0..63] with salt-hex at
         * [0..31] and pass-hex at [32..63], then mymd5(linebuf, 64).
         *
         * Strategy:
         *   1. Pass-hex: lowercase canonical encode of digest1 into M[8..15]
         *      (bytes [32..63] of the message buffer).
         *   2. Salt-hex: pre-computed by host as saltsnap[si].hashsalt and
         *      packed into salt_buf by gpujob_opencl.c gpu_pack_salts when
         *      use_hashsalt=1 (already wired for JOB_MD5_MD5SALTMD5PASS).
         *      Copy the 32 bytes from salt_buf[0..31] into M[0..7]
         *      (bytes [0..31] of the message buffer).
         *   3. inner_len = 64; the host packed salt slot has the hashsalt
         *      at offset 0 (slen field will be set to 32 by gpu_pack_salts
         *      because hashlen=32), but the salt has ALREADY been consumed
         *      into M[0..7] above — the step-3 outer block must NOT append
         *      the salt a second time. We override slen=0 below; this
         *      forces total_len=64 (>= 56) into the else branch, where
         *      first_chunk=64-64=0 and the salt-append loop is a no-op. */
        {{BASE_ALGO}}_to_hex_lc(h0, h1, h2, h3, &M[8]);
        uchar *mb = (uchar *)M;
        for (int i = 0; i < 32; i++) mb[i] = salt_buf[i];
        inner_len = 64u;
    } else {
        /* mode 0 (default): e31 MD5SALT, lowercase hex, forward order. */
        {{BASE_ALGO}}_to_hex_lc(h0, h1, h2, h3, M);
        inner_len = 32u;
    }

    /* Step 3: outer MD5 over (hex_inner || salt). */
    /* B6.8 (2026-05-06): mode 4 (e367) consumed the salt-hex into M[0..7]
     * during step 2 — the message buffer is the COMPLETE 64-byte input,
     * no outer salt append. Override slen=0 to short-circuit the
     * salt-append loop in the else branch below: total_len=64+0=64
     * (>=56) routes to the else branch; first_chunk = 64-64 = 0; both
     * salt-append loops become no-ops; the first md5_block hashes M[]
     * exactly as packed; the second block writes 0x80 at M[0] + length
     * 512 bits at M[14]. Modes 0/1/2/3 are unaffected (slen unchanged). */
    if (algo_mode == 4u) slen = 0u;
    uchar *mbytes = (uchar *)M;
    uint total_len = inner_len + slen;
    if (total_len < 56u) {
        for (uint i = 0; i < slen; i++) {
            mbytes[inner_len + i] = salt_buf[i];
        }
        mbytes[total_len] = 0x80u;
        M[14] = total_len * 8u;
        M[15] = 0u;
        st->h[0] = 0x67452301u;
        st->h[1] = 0xEFCDAB89u;
        st->h[2] = 0x98BADCFEu;
        st->h[3] = 0x10325476u;
        {{BASE_ALGO}}_block(&st->h[0], &st->h[1], &st->h[2], &st->h[3], M);
    } else {
        uint first_chunk = 64u - inner_len;
        if (first_chunk > slen) first_chunk = slen;
        for (uint i = 0; i < first_chunk; i++) {
            mbytes[inner_len + i] = salt_buf[i];
        }
        st->h[0] = 0x67452301u;
        st->h[1] = 0xEFCDAB89u;
        st->h[2] = 0x98BADCFEu;
        st->h[3] = 0x10325476u;
        {{BASE_ALGO}}_block(&st->h[0], &st->h[1], &st->h[2], &st->h[3], M);

        for (int j = 0; j < 16; j++) M[j] = 0;
        uint rem_salt = slen - first_chunk;
        for (uint i = 0; i < rem_salt; i++) {
            uchar c = salt_buf[first_chunk + i];
            M[i >> 2] |= ((uint)c) << ((i & 3) * 8);
        }
        M[rem_salt >> 2] |= ((uint)0x80u) << ((rem_salt & 3) * 8);
        if (rem_salt < 56u) {
            M[14] = total_len * 8u;
            M[15] = 0u;
            {{BASE_ALGO}}_block(&st->h[0], &st->h[1], &st->h[2], &st->h[3], M);
        } else {
            {{BASE_ALGO}}_block(&st->h[0], &st->h[1], &st->h[2], &st->h[3], M);
            for (int j = 0; j < 16; j++) M[j] = 0;
            M[14] = total_len * 8u;
            M[15] = 0u;
            {{BASE_ALGO}}_block(&st->h[0], &st->h[1], &st->h[2], &st->h[3], M);
        }
    }
}
