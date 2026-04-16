/* gpu_md6256unsalted.cl — Pre-packed unsalted MD6-256 with mask expansion
 *
 * Input: pre-packed N[89] (712-byte) blocks in passbuf.
 *   Host packs: Q[15] + K[8]=0 + U[1] + V[1] + B[64] with message in LE.
 *   Kernel byte-swaps B[64] portion to big-endian before compression.
 *   Mask positions fill B portion at doubled-and-swapped byte offsets.
 *
 * MD6-256: r=104 rounds, c=16, n=89, d=256.
 * Compression loop: 104*16 = 1664 steps.
 * Output: last 4 words of C[16] (= last d/w = 4 × uint64).
 *
 * Dispatch: num_words × num_masks threads.
 * Hit stride: 6 (word_idx, mask_idx, hx, hy, hz, hw)
 */


/* bswap64 provided by gpu_common.cl */

/* MD6 Q constants for iteration re-hash */
__constant ulong MD6_QC[15] = {
    0x7311c2812425cfa0UL, 0x6432286434aac8e7UL, 0xb60450e9ef68b7c1UL,
    0xe8fb23908d9f06f1UL, 0xdd2e76cba691e5bfUL, 0x0cd0d63b2c30bc41UL,
    0x1f8ccf6823058f8aUL, 0x54e5ed5b88e3775dUL, 0x4ad12aae0a6d6031UL,
    0x3e7f16bb88222e0dUL, 0x8af8671d3fb50c2cUL, 0x995ad1178bd25c31UL,
    0xc878c1dd04c4b633UL, 0x3b72066c7a1552acUL, 0x0d6f3522631effcbUL
};

/* MD6 compression loop constants */
#define MD6_n  89
#define MD6_c  16
#define MD6_r  104

/* Tap positions for feedback shift register */
#define MD6_t0  17
#define MD6_t1  18
#define MD6_t2  21
#define MD6_t3  31
#define MD6_t4  67
#define MD6_t5  89

__constant ulong MD6_S0 = 0x0123456789abcdefUL;
__constant ulong MD6_Smask = 0x7311c2812425cfa0UL;

#define MD6_STEP(rs, ls, step) \
    x  = S;                               \
    x ^= A[i + step - MD6_t5];            \
    x ^= A[i + step - MD6_t0];            \
    x ^= (A[i + step - MD6_t1] & A[i + step - MD6_t2]); \
    x ^= (A[i + step - MD6_t3] & A[i + step - MD6_t4]); \
    x ^= (x >> rs);                       \
    A[i + step] = x ^ (x << ls);

/* MD6 main compression loop — operates on array A of length r*c+n */
void md6_compress_loop(ulong *A) {
    ulong x;
    ulong S = MD6_S0;
    int i = MD6_n;
    for (int j = 0; j < MD6_r * MD6_c; j += MD6_c) {
        MD6_STEP(10, 11,  0)
        MD6_STEP( 5, 24,  1)
        MD6_STEP(13,  9,  2)
        MD6_STEP(10, 16,  3)
        MD6_STEP(11, 15,  4)
        MD6_STEP(12,  9,  5)
        MD6_STEP( 2, 27,  6)
        MD6_STEP( 7, 15,  7)
        MD6_STEP(14,  6,  8)
        MD6_STEP(15,  2,  9)
        MD6_STEP( 7, 29, 10)
        MD6_STEP(13,  8, 11)
        MD6_STEP(11, 15, 12)
        MD6_STEP( 7,  5, 13)
        MD6_STEP( 6, 31, 14)
        MD6_STEP(12,  9, 15)
        S = (S << 1) ^ (S >> 63) ^ (S & MD6_Smask);
        i += 16;
    }
}

__kernel void md6_256_unsalted_batch(
    __global const uchar *words,         /* pre-packed N[89] blocks */
    __global const ushort *unused_lens,
    __global const uchar *mask_desc,
    __global const uint *unused1, __global const ushort *unused2,
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
    uint word_idx = tid / params.num_masks;
    ulong mask_idx = params.mask_start + (tid % params.num_masks);
    if (word_idx >= params.num_words) return;

    /* Load pre-packed N[89] block.
     * Host packs: N[0..24] = Q+K+U+V (fixed), N[25..88] = B (message in LE).
     * word_stride stored in jobg, passed as upload size. */
    __global const ulong *src = (__global const ulong *)(words + word_idx * 712);

    /* We need a working array A of size r*c+n = 104*16+89 = 1753 words. */
    ulong A[1753];
    for (int i = 0; i < 89; i++) A[i] = src[i];

    /* Fill mask positions into B portion (A[25..88]).
     * B is in LE uint64 from host packer; mask fills at LE byte positions.
     * We byte-swap B after mask fill. */
    uint n_pre = params.n_prepend;
    uint n_app = params.n_append;

    if (n_pre > 0 || n_app > 0) {
        uint n_total_m = n_pre + n_app;

        if (n_pre > 0) {
            uint append_combos = 1;
            for (uint i = 0; i < n_app; i++)
                append_combos *= mask_desc[n_pre + i];
            uint prepend_idx = (uint)(mask_idx / append_combos);
            uint append_idx = (uint)(mask_idx % append_combos);

            uint pidx = prepend_idx;
            for (int i = (int)n_pre - 1; i >= 0; i--) {
                uint sz = mask_desc[i];
                uchar ch = mask_desc[n_total_m + i * 256 + (pidx % sz)];
                pidx /= sz;
                int wi = 25 + (i >> 3);
                int bi = (i & 7) << 3;
                A[wi] = (A[wi] & ~(0xFFUL << bi)) | ((ulong)ch << bi);
            }

            if (n_app > 0) {
                ulong V = A[24];
                uint p_bits = (uint)((V >> 20) & 0xFFFFUL);
                int total_len = (64*64 - (int)p_bits) / 8;
                int app_start = total_len - (int)n_app;
                uint aidx = append_idx;
                for (int i = (int)n_app - 1; i >= 0; i--) {
                    int pos_idx = n_pre + i;
                    uint sz = mask_desc[pos_idx];
                    uchar ch = mask_desc[n_total_m + pos_idx * 256 + (aidx % sz)];
                    aidx /= sz;
                    int pos = app_start + i;
                    int wi = 25 + (pos >> 3);
                    int bi = (pos & 7) << 3;
                    A[wi] = (A[wi] & ~(0xFFUL << bi)) | ((ulong)ch << bi);
                }
            }
        } else {
            /* Append-only (brute-force): host pre-decomposes mask_start into
             * per-position base offsets. Kernel does fast uint32 local
             * decomposition and adds to base with carry. */
            ulong V = A[24];
            uint p_bits = (uint)((V >> 20) & 0xFFFFUL);
            int total_len = (64*64 - (int)p_bits) / 8;
            int app_start = total_len - (int)n_app;
            uint local_idx = tid % params.num_masks;
            uint aidx = local_idx;
            uint carry = 0;
            for (int i = (int)n_app - 1; i >= 0; i--) {
                uint sz = mask_desc[i];
                uint local_digit = aidx % sz;
                aidx /= sz;
                uint base_digit = (i < 8)
                    ? (uint)((params.mask_base0 >> (i * 8)) & 0xFF)
                    : (uint)((params.mask_base1 >> ((i - 8) * 8)) & 0xFF);
                uint sum = base_digit + local_digit + carry;
                carry = sum / sz;
                uint final_digit = sum % sz;
                uchar ch = mask_desc[n_total_m + i * 256 + final_digit];
                int pos = app_start + i;
                int wi = 25 + (pos >> 3);
                int bi = (pos & 7) << 3;
                A[wi] = (A[wi] & ~(0xFFUL << bi)) | ((ulong)ch << bi);
            }
        }
    }

    /* Byte-swap B portion (A[25..88]) from LE to BE for MD6 compression */
    for (int i = 25; i < 89; i++) A[i] = bswap64(A[i]);

    /* Run MD6 compression */
    md6_compress_loop(A);

    /* Extract output: last c=16 words of A, then take last d/w=4 words.
     * C[0..15] = A[r*c+n-c .. r*c+n-1] = A[1737..1752]
     * Hash = C[12..15] (last 4 uint64 words = last 256 bits = 8 uint32) */
    uint max_iter = params.max_iter;
    uint iter = 1;

    for (;;) {
        /* Extract output: C[12..15] = A[1749..1752] */
        ulong C0 = A[1749], C1 = A[1750], C2 = A[1751], C3 = A[1752];

        /* Byte-swap to LE, split into 8 uint32 for probe */
        uint h[8];
        ulong s0 = bswap64(C0); h[0] = (uint)s0; h[1] = (uint)(s0 >> 32);
        ulong s1 = bswap64(C1); h[2] = (uint)s1; h[3] = (uint)(s1 >> 32);
        ulong s2 = bswap64(C2); h[4] = (uint)s2; h[5] = (uint)(s2 >> 32);
        ulong s3 = bswap64(C3); h[6] = (uint)s3; h[7] = (uint)(s3 >> 32);

        if (probe_compact(h[0], h[1], h[2], h[3], compact_fp, compact_idx,
                          params.compact_mask, params.max_probe, params.hash_data_count,
                          hash_data_buf, hash_data_off,
                          overflow_keys, overflow_hashes, overflow_offsets, params.overflow_count)) {
            EMIT_HIT_8(hits, hit_count, params.max_hits, word_idx, mask_idx, iter, h)
        }

        if (iter >= max_iter) break;
        iter++;

        /* Hex-encode 32-byte hash into 64-char hex string */
        uchar buf[64];
        for (int i = 0; i < 8; i++) {
            uchar byte_val = (uchar)(C0 >> (56 - i * 8));
            uchar hi_n = (byte_val >> 4), lo_n = (byte_val & 0xf);
            buf[i*2]     = hi_n + (hi_n < 10 ? '0' : 'a' - 10);
            buf[i*2 + 1] = lo_n + (lo_n < 10 ? '0' : 'a' - 10);
        }
        for (int i = 0; i < 8; i++) {
            uchar byte_val = (uchar)(C1 >> (56 - i * 8));
            uchar hi_n = (byte_val >> 4), lo_n = (byte_val & 0xf);
            buf[16 + i*2]     = hi_n + (hi_n < 10 ? '0' : 'a' - 10);
            buf[16 + i*2 + 1] = lo_n + (lo_n < 10 ? '0' : 'a' - 10);
        }
        for (int i = 0; i < 8; i++) {
            uchar byte_val = (uchar)(C2 >> (56 - i * 8));
            uchar hi_n = (byte_val >> 4), lo_n = (byte_val & 0xf);
            buf[32 + i*2]     = hi_n + (hi_n < 10 ? '0' : 'a' - 10);
            buf[32 + i*2 + 1] = lo_n + (lo_n < 10 ? '0' : 'a' - 10);
        }
        for (int i = 0; i < 8; i++) {
            uchar byte_val = (uchar)(C3 >> (56 - i * 8));
            uchar hi_n = (byte_val >> 4), lo_n = (byte_val & 0xf);
            buf[48 + i*2]     = hi_n + (hi_n < 10 ? '0' : 'a' - 10);
            buf[48 + i*2 + 1] = lo_n + (lo_n < 10 ? '0' : 'a' - 10);
        }

        /* Rebuild N[89]: Q[15] + K[8]=0 + U[1] + V[1] + B[64] */
        for (int i = 0; i < 15; i++) A[i] = MD6_QC[i];
        for (int i = 15; i < 23; i++) A[i] = 0;
        A[23] = (ulong)1 << 56;
        /* V: r=104, L=64, z=1, p=3584, keylen=0, d=256 */
        A[24] = ((ulong)104 << 48) | ((ulong)64 << 40) | ((ulong)1 << 36)
              | ((ulong)3584 << 20) | (ulong)256;
        /* Pack buf into B[64] as LE uint64, then byte-swap to BE */
        for (int i = 0; i < 64; i++) {
            int wi = 25 + (i >> 3);
            int bi = (i & 7) << 3;
            if ((i & 7) == 0) A[wi] = 0;
            A[wi] |= (ulong)buf[i] << bi;
        }
        for (int i = 33; i < 89; i++) A[i] = 0;
        for (int i = 25; i < 89; i++) A[i] = bswap64(A[i]);

        md6_compress_loop(A);
    }
}
