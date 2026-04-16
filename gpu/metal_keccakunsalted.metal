/* metal_keccakunsalted.metal — Keccak/SHA3 unsalted with mask expansion
 *
 * Keccak-f[1600] permutation. 8 entry points:
 * Keccak-224/256/384/512 (pad 0x01) and SHA3-224/256/384/512 (pad 0x06).
 * Natively little-endian -- no byte-swap needed.
 *
 * Hit stride: 6 (word_idx, mask_idx, hx, hy, hz, hw)
 */

constant ulong RC[24] = {
    0x0000000000000001UL, 0x0000000000008082UL, 0x800000000000808aUL,
    0x8000000080008000UL, 0x000000000000808bUL, 0x0000000080000001UL,
    0x8000000080008081UL, 0x8000000000008009UL, 0x000000000000008aUL,
    0x0000000000000088UL, 0x0000000080008009UL, 0x000000008000000aUL,
    0x000000008000808bUL, 0x800000000000008bUL, 0x8000000000008089UL,
    0x8000000000008003UL, 0x8000000000008002UL, 0x8000000000000080UL,
    0x000000000000800aUL, 0x800000008000000aUL, 0x8000000080008081UL,
    0x8000000000008080UL, 0x0000000080000001UL, 0x8000000080008008UL
};

constant uint ROTC[25] = {
     0,  1, 62, 28, 27,
    36, 44,  6, 55, 20,
     3, 10, 43, 25, 39,
    41, 45, 15, 21,  8,
    18,  2, 61, 56, 14
};

static void keccak_f1600(thread ulong *st) {
    for (int round = 0; round < 24; round++) {
        ulong C[5], D[5];
        for (int x = 0; x < 5; x++)
            C[x] = st[x] ^ st[x+5] ^ st[x+10] ^ st[x+15] ^ st[x+20];
        for (int x = 0; x < 5; x++) {
            D[x] = C[(x+4) % 5] ^ rotate(C[(x+1) % 5], (ulong)1);
            for (int y = 0; y < 25; y += 5)
                st[x+y] ^= D[x];
        }
        ulong B[25];
        for (int x = 0; x < 5; x++)
            for (int y = 0; y < 5; y++)
                B[x + 5 * ((2*y + 3*x) % 5)] = rotate(st[x*5+y], (ulong)ROTC[x*5+y]);
        for (int x = 0; x < 5; x++)
            for (int y = 0; y < 25; y += 5)
                st[x+y] = B[x+y] ^ (~B[((x+1)%5)+y] & B[((x+2)%5)+y]);
        st[0] ^= RC[round];
    }
}

/* Fill masks into LE uint64 block */
static void keccak_fill_masks(thread ulong *block, uint n_pre, uint n_app, uint mask_idx,
                               device const uint8_t *mask_desc, int rate,
                               uint num_masks, ulong mask_base0, ulong mask_base1) {
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
            int wi = i >> 3;
            int bi = (i & 7) << 3;
            block[wi] = (block[wi] & ~(0xFFUL << bi)) | ((ulong)ch << bi);
        }

        if (n_app > 0) {
            thread uchar *bp = (thread uchar *)block;
            int total_len = bp[rate] | (bp[rate+1] << 8);
            int app_start = total_len - (int)n_app;
            uint aidx = append_idx;
            for (int i = (int)n_app - 1; i >= 0; i--) {
                int pos_idx = n_pre + i;
                uint sz = mask_desc[pos_idx];
                uchar ch = mask_desc[n_total_m + pos_idx * 256 + (aidx % sz)];
                aidx /= sz;
                int pos = app_start + i;
                int wi = pos >> 3;
                int bi = (pos & 7) << 3;
                block[wi] = (block[wi] & ~(0xFFUL << bi)) | ((ulong)ch << bi);
            }
        }
    } else {
        /* Append-only (brute-force): host pre-decomposes mask_start into
         * per-position base offsets in mask_base0/mask_base1. Kernel does
         * fast uint32 local decomposition and adds to base with carry. */
        thread uchar *bp = (thread uchar *)block;
        int total_len = bp[rate] | (bp[rate+1] << 8);
        int app_start = total_len - (int)n_app;
        uint local_idx = mask_idx % num_masks;
        uint aidx = local_idx;
        uint carry = 0;
        for (int i = (int)n_app - 1; i >= 0; i--) {
            uint sz = mask_desc[i];
            uint local_digit = aidx % sz;
            aidx /= sz;
            uint base_digit = (i < 8)
                ? (uint)((mask_base0 >> (i * 8)) & 0xFF)
                : (uint)((mask_base1 >> ((i - 8) * 8)) & 0xFF);
            uint sum = base_digit + local_digit + carry;
            carry = sum / sz;
            uint final_digit = sum % sz;
            uchar ch = mask_desc[n_total_m + i * 256 + final_digit];
            int pos = app_start + i;
            int wi = pos >> 3;
            int bi = (pos & 7) << 3;
            block[wi] = (block[wi] & ~(0xFFUL << bi)) | ((ulong)ch << bi);
        }
    }
}

/* Macro-expanded Keccak kernels for each rate */

#define KECCAK_METAL_KERNEL(name, rate, rate_words, hash_words, hash_ulongs, pad_byte) \
kernel void name( \
    device const uint8_t    *words       [[buffer(0)]], \
    device const ushort     *unused_lens [[buffer(1)]], \
    device const ushort     *unused2     [[buffer(2)]], \
    device const uint8_t    *mask_desc   [[buffer(3)]], \
    device const uint       *unused3     [[buffer(4)]], \
    device const ushort     *unused4     [[buffer(5)]], \
    device const uint       *compact_fp  [[buffer(6)]], \
    device const uint       *compact_idx [[buffer(7)]], \
    constant MetalParams    &params      [[buffer(8)]], \
    device const uint8_t    *hash_data_buf [[buffer(9)]], \
    device const uint64_t   *hash_data_off [[buffer(10)]], \
    device const ushort     *hash_data_len [[buffer(11)]], \
    device uint             *hits         [[buffer(12)]], \
    device atomic_uint      *hit_count    [[buffer(13)]], \
    device const uint64_t   *overflow_keys   [[buffer(14)]], \
    device const uint8_t    *overflow_hashes [[buffer(15)]], \
    device const uint       *overflow_offsets [[buffer(16)]], \
    device const ushort     *overflow_lengths [[buffer(17)]], \
    uint                     tid          [[thread_position_in_grid]], \
    uint                     lid          [[thread_position_in_threadgroup]], \
    uint                     tgsize       [[threads_per_threadgroup]]) \
{ \
    uint word_idx = tid / params.num_masks; \
    ulong mask_idx = params.mask_start + (tid % params.num_masks); \
    if (word_idx >= params.num_words) return; \
    \
    int stride = rate + 8; \
    device const ulong *src = (device const ulong *)(words + word_idx * stride); \
    ulong block[rate_words]; \
    for (int i = 0; i < rate_words; i++) block[i] = src[i]; \
    \
    if (params.n_prepend > 0 || params.n_append > 0) \
        keccak_fill_masks(block, params.n_prepend, params.n_append, \
                          mask_idx, mask_desc, rate, \
                          params.num_masks, params.mask_base0, params.mask_base1); \
    \
    ulong st[25]; \
    for (int i = 0; i < 25; i++) st[i] = 0; \
    for (int i = 0; i < rate_words; i++) st[i] = block[i]; \
    keccak_f1600(st); \
    \
    uint max_iter = params.max_iter; \
    for (uint iter = 1; iter <= max_iter; iter++) { \
        /* Extract all hash words from state (LE, no swap needed) */ \
        uint h[hash_words]; \
        for (int i = 0; i < hash_ulongs; i++) { \
            h[i*2]   = (uint)st[i]; \
            if (i*2+1 < hash_words) h[i*2+1] = (uint)(st[i] >> 32); \
        } \
        \
        ulong key = (ulong(h[1]) << 32) | h[0]; \
        uint fp = uint(key >> 32); if (fp == 0) fp = 1; \
        ulong pos = (key ^ (key >> 32)) & params.compact_mask; \
        bool found = false; \
        for (uint p = 0; p < params.max_probe && !found; p++) { \
            uint cfp = compact_fp[pos]; if (cfp == 0) break; \
            if (cfp == fp) { uint idx = compact_idx[pos]; \
                if (idx < params.hash_data_count) { \
                    ulong off = hash_data_off[idx]; \
                    device const uint *ref = (device const uint *)(hash_data_buf + off); \
                    if (h[0] == ref[0] && h[1] == ref[1] && h[2] == ref[2] && h[3] == ref[3]) \
                        found = true; \
                } } \
            pos = (pos + 1) & params.compact_mask; \
        } \
        if (!found && params.overflow_count > 0) { \
            int lo = 0, hi2 = int(params.overflow_count) - 1; \
            while (lo <= hi2 && !found) { \
                int mid = (lo + hi2) / 2; ulong mkey = overflow_keys[mid]; \
                if (key < mkey) hi2 = mid - 1; \
                else if (key > mkey) lo = mid + 1; \
                else { \
                    for (int d = mid; d >= 0 && overflow_keys[d] == key && !found; d--) { \
                        device const uint *oref = (device const uint *)(overflow_hashes + overflow_offsets[d]); \
                        if (h[0] == oref[0] && h[1] == oref[1] && h[2] == oref[2] && h[3] == oref[3]) \
                            found = true; } \
                    for (int d = mid+1; d < int(params.overflow_count) && overflow_keys[d] == key && !found; d++) { \
                        device const uint *oref = (device const uint *)(overflow_hashes + overflow_offsets[d]); \
                        if (h[0] == oref[0] && h[1] == oref[1] && h[2] == oref[2] && h[3] == oref[3]) \
                            found = true; } \
                    break; } } } \
        if (found) { \
            uint slot = atomic_fetch_add_explicit(hit_count, 1, memory_order_relaxed); \
            if (slot < params.max_hits) { \
                uint base = slot * HIT_STRIDE; \
                hits[base] = word_idx; hits[base+1] = uint(mask_idx); hits[base+2] = iter; \
                for (int i = 0; i < hash_words; i++) hits[base+3+i] = h[i]; \
                for (uint _z = 3+hash_words; _z < HIT_STRIDE; _z++) hits[base+_z] = 0; } } \
        if (iter < max_iter) { \
            /* Hex-encode hash output (LE bytes) into sponge input */ \
            int hex_bytes = hash_ulongs * 8; \
            if (hash_words == 7) hex_bytes = 28; \
            if (hash_words == 12) hex_bytes = 48; \
            int hex_chars = hex_bytes * 2; \
            uchar hex_str[256]; \
            for (int i = 0; i < hex_bytes; i++) { \
                uchar b = (uchar)(st[i/8] >> ((i%8)*8)); \
                uchar hi_n = (b >> 4), lo_n = (b & 0xf); \
                hex_str[i*2]   = hi_n + (hi_n < 10 ? '0' : 'a' - 10); \
                hex_str[i*2+1] = lo_n + (lo_n < 10 ? '0' : 'a' - 10); \
            } \
            /* Absorb hex string with sponge padding into fresh state */ \
            for (int i = 0; i < 25; i++) st[i] = 0; \
            for (int i = 0; i < hex_chars && i < rate; i++) { \
                int wi = i / 8; \
                int bi = (i % 8) * 8; \
                st[wi] ^= (ulong)hex_str[i] << bi; \
            } \
            if (hex_chars < rate) { \
                int p2 = hex_chars; \
                st[p2/8] ^= (ulong)pad_byte << ((p2%8)*8); \
                st[(rate-1)/8] ^= 0x80UL << (((rate-1)%8)*8); \
                keccak_f1600(st); \
            } else { \
                keccak_f1600(st); \
                for (int i = rate; i < hex_chars; i++) { \
                    int wi = (i - rate) / 8; \
                    int bi = ((i - rate) % 8) * 8; \
                    st[wi] ^= (ulong)hex_str[i] << bi; \
                } \
                int p2 = hex_chars - rate; \
                st[p2/8] ^= (ulong)pad_byte << ((p2%8)*8); \
                st[(rate-1)/8] ^= 0x80UL << (((rate-1)%8)*8); \
                keccak_f1600(st); \
            } \
        } \
    } \
}

/* Keccak variants (pad byte 0x01) */
KECCAK_METAL_KERNEL(keccak224_unsalted_batch, 144, 18, 7, 4, 0x01)
KECCAK_METAL_KERNEL(keccak256_unsalted_batch, 136, 17, 8, 4, 0x01)
KECCAK_METAL_KERNEL(keccak384_unsalted_batch, 104, 13, 12, 6, 0x01)
KECCAK_METAL_KERNEL(keccak512_unsalted_batch, 72, 9, 16, 8, 0x01)

/* SHA3 variants (pad byte 0x06) */
KECCAK_METAL_KERNEL(sha3_224_unsalted_batch, 144, 18, 7, 4, 0x06)
KECCAK_METAL_KERNEL(sha3_256_unsalted_batch, 136, 17, 8, 4, 0x06)
KECCAK_METAL_KERNEL(sha3_384_unsalted_batch, 104, 13, 12, 6, 0x06)
KECCAK_METAL_KERNEL(sha3_512_unsalted_batch, 72, 9, 16, 8, 0x06)
