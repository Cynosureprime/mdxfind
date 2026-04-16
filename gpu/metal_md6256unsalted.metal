/* metal_md6256unsalted.metal — Pre-packed unsalted MD6-256 with mask expansion
 *
 * MD6-256: r=104 rounds, c=16, n=89, d=256.
 * Input: pre-packed N[89] (712-byte) blocks.
 * Hit stride: 6 (word_idx, mask_idx, hx, hy, hz, hw)
 */

/* bswap64 provided by metal_common.metal */

#define MD6_n  89
#define MD6_c  16
#define MD6_r  104
#define MD6_t0  17
#define MD6_t1  18
#define MD6_t2  21
#define MD6_t3  31
#define MD6_t4  67
#define MD6_t5  89

constant int RS[16] = {10, 5,13,10,11,12, 2, 7,14,15, 7,13,11, 7, 6,12};
constant int LS[16] = {11,24, 9,16,15, 9,27,15, 6, 2,29, 8,15, 5,31, 9};

constant ulong MD6_S0 = 0x0123456789abcdefUL;
constant ulong MD6_Smask = 0x7311c2812425cfa0UL;

constant ulong MD6_QC[15] = {
    0x7311c2812425cfa0UL, 0x6432286434aac8e7UL, 0xb60450e9ef68b7c1UL,
    0xe8fb23908d9f06f1UL, 0xdd2e76cba691e5bfUL, 0x0cd0d63b2c30bc41UL,
    0x1f8ccf6823058f8aUL, 0x54e5ed5b88e3775dUL, 0x4ad12aae0a6d6031UL,
    0x3e7f16bb88222e0dUL, 0x8af8671d3fb50c2cUL, 0x995ad1178bd25c31UL,
    0xc878c1dd04c4b633UL, 0x3b72066c7a1552acUL, 0x0d6f3522631effcbUL
};

static void md6_compress_loop(thread ulong *A) {
    ulong S = MD6_S0;
    int i = MD6_n;
    for (int j = 0; j < MD6_r * MD6_c; j += MD6_c) {
        for (int step = 0; step < 16; step++) {
            ulong x = S;
            x ^= A[i + step - MD6_t5];
            x ^= A[i + step - MD6_t0];
            x ^= (A[i + step - MD6_t1] & A[i + step - MD6_t2]);
            x ^= (A[i + step - MD6_t3] & A[i + step - MD6_t4]);
            x ^= (x >> RS[step]);
            A[i + step] = x ^ (x << LS[step]);
        }
        S = (S << 1) ^ (S >> 63) ^ (S & MD6_Smask);
        i += 16;
    }
}

kernel void md6_256_unsalted_batch(
    device const uint8_t    *words       [[buffer(0)]],
    device const ushort     *unused_lens [[buffer(1)]],
    device const ushort     *unused2     [[buffer(2)]],
    device const uint8_t    *mask_desc   [[buffer(3)]],
    device const uint       *unused3     [[buffer(4)]],
    device const ushort     *unused4     [[buffer(5)]],
    device const uint       *compact_fp  [[buffer(6)]],
    device const uint       *compact_idx [[buffer(7)]],
    constant MetalParams    &params      [[buffer(8)]],
    device const uint8_t    *hash_data_buf [[buffer(9)]],
    device const uint64_t   *hash_data_off [[buffer(10)]],
    device const ushort     *hash_data_len [[buffer(11)]],
    device uint             *hits         [[buffer(12)]],
    device atomic_uint      *hit_count    [[buffer(13)]],
    device const uint64_t   *overflow_keys   [[buffer(14)]],
    device const uint8_t    *overflow_hashes [[buffer(15)]],
    device const uint       *overflow_offsets [[buffer(16)]],
    device const ushort     *overflow_lengths [[buffer(17)]],
    uint                     tid          [[thread_position_in_grid]],
    uint                     lid          [[thread_position_in_threadgroup]],
    uint                     tgsize       [[threads_per_threadgroup]])
{
    uint word_idx = tid / params.num_masks;
    ulong mask_idx = params.mask_start + (tid % params.num_masks);
    if (word_idx >= params.num_words) return;

    device const ulong *src = (device const ulong *)(words + word_idx * 712);
    ulong A[1753];
    for (int i = 0; i < 89; i++) A[i] = src[i];

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
             * per-position base offsets in mask_base0/mask_base1. Kernel does
             * fast uint32 local decomposition and adds to base with carry. */
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

    /* Byte-swap B portion (A[25..88]) from LE to BE */
    for (int i = 25; i < 89; i++) A[i] = bswap64(A[i]);

    md6_compress_loop(A);

    uint max_iter = params.max_iter;

    for (uint iter = 1; iter <= max_iter; iter++) {
        ulong C[4];
        C[0] = A[1749]; C[1] = A[1750]; C[2] = A[1751]; C[3] = A[1752];

        uint h[8];
        for (int i = 0; i < 4; i++) {
            ulong s = bswap64(C[i]);
            h[i*2]   = (uint)s;
            h[i*2+1] = (uint)(s >> 32);
        }

        /* Probe compact hash table */
        ulong key = (ulong(h[1]) << 32) | h[0];
        uint fp = uint(key >> 32); if (fp == 0) fp = 1;
        ulong pos = (key ^ (key >> 32)) & params.compact_mask;
        bool found = false;
        for (uint p = 0; p < params.max_probe && !found; p++) {
            uint cfp = compact_fp[pos]; if (cfp == 0) break;
            if (cfp == fp) { uint idx = compact_idx[pos];
                if (idx < params.hash_data_count) {
                    ulong off = hash_data_off[idx];
                    device const uint *ref = (device const uint *)(hash_data_buf + off);
                    if (h[0] == ref[0] && h[1] == ref[1] && h[2] == ref[2] && h[3] == ref[3])
                        found = true;
                } }
            pos = (pos + 1) & params.compact_mask;
        }
        if (!found && params.overflow_count > 0) {
            int lo = 0, hi2 = int(params.overflow_count) - 1;
            while (lo <= hi2 && !found) {
                int mid = (lo + hi2) / 2; ulong mkey = overflow_keys[mid];
                if (key < mkey) hi2 = mid - 1;
                else if (key > mkey) lo = mid + 1;
                else {
                    for (int d = mid; d >= 0 && overflow_keys[d] == key && !found; d--) {
                        device const uint *oref = (device const uint *)(overflow_hashes + overflow_offsets[d]);
                        if (h[0] == oref[0] && h[1] == oref[1] && h[2] == oref[2] && h[3] == oref[3])
                            found = true; }
                    for (int d = mid+1; d < int(params.overflow_count) && overflow_keys[d] == key && !found; d++) {
                        device const uint *oref = (device const uint *)(overflow_hashes + overflow_offsets[d]);
                        if (h[0] == oref[0] && h[1] == oref[1] && h[2] == oref[2] && h[3] == oref[3])
                            found = true; }
                    break; } } }
        if (found) {
            uint slot = atomic_fetch_add_explicit(hit_count, 1, memory_order_relaxed);
            if (slot < params.max_hits) {
                uint base = slot * HIT_STRIDE;
                hits[base] = word_idx; hits[base+1] = uint(mask_idx); hits[base+2] = iter;
                for (int i = 0; i < 8; i++) hits[base+3+i] = h[i];
                for (uint _z = 11; _z < HIT_STRIDE; _z++) hits[base+_z] = 0;
            }
        }
        if (iter < max_iter) {
            /* Hex-encode 32-byte hash into 64-char hex string */
            uchar buf[64];
            for (int i = 0; i < 4; i++) {
                ulong w = C[i];
                for (int b = 0; b < 8; b++) {
                    uchar byte = (uchar)(w >> (56 - b * 8));
                    uchar hi = (byte >> 4), lo = (byte & 0xf);
                    buf[i*16 + b*2]     = hi + (hi < 10 ? '0' : 'a' - 10);
                    buf[i*16 + b*2 + 1] = lo + (lo < 10 ? '0' : 'a' - 10);
                }
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
}
