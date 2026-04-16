/* metal_blake2s256unsalted.metal — Pre-padded unsalted BLAKE2S-256 with mask expansion
 *
 * Input: 4096 pre-padded 64-byte blocks in passbuf (word_stride=64).
 *   Packed by gpu_try_pack_unsalted(): password bytes at offset n_prepend,
 *   total_len stored as bitlen at M[14] (extract byte len = M[14]>>3).
 *   BLAKE2S uses its own padding (no 0x80/bitlen), so the packer just stores
 *   the password and length — the kernel handles BLAKE2S-specific finalization.
 *
 * Dispatch: num_words x num_masks threads.
 * Hit stride: 6 (word_idx, mask_idx, h0, h1, h2, h3)
 */

/* B2S_IV, B2S_SIGMA, b2s_compress all provided by metal_common.metal */

kernel void blake2s256_unsalted_batch(
    device const uint       *words       [[buffer(0)]],
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

    /* Load pre-packed block (16 uint32 = 64 bytes).
     * Password bytes start at n_prepend. Total length at M[14] bits [7:0] and [15:8]. */
    device const uint *src = words + word_idx * 16;
    uchar buf[64];
    for (int i = 0; i < 64; i++)
        buf[i] = ((device const uchar *)src)[i];

    /* Fill mask positions */
    uint n_pre = params.n_prepend;
    uint n_app = params.n_append;

    /* Read total_len from packed M[14] (stored as bitlen by packer, extract byte len) */
    uint M14 = src[14];
    int total_len = M14 >> 3;

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
                buf[i] = ch;
            }

            if (n_app > 0) {
                int app_start = total_len - (int)n_app;
                uint aidx = append_idx;
                for (int i = (int)n_app - 1; i >= 0; i--) {
                    int pos_idx = n_pre + i;
                    uint sz = mask_desc[pos_idx];
                    uchar ch = mask_desc[n_total_m + pos_idx * 256 + (aidx % sz)];
                    aidx /= sz;
                    buf[app_start + i] = ch;
                }
            }
        } else {
            /* Append-only (brute-force): host pre-decomposes mask_start into
             * per-position base offsets in mask_base0/mask_base1. Kernel does
             * fast uint32 local decomposition and adds to base with carry. */
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
                buf[app_start + i] = ch;
            }
        }
    }

    /* BLAKE2S-256: single-block hash (password <= 55 bytes fits in one 64-byte block).
     * Init: IV ^ parameter block (outlen=32, keylen=0, fanout=1, depth=1). */
    uint h[8];
    for (int i = 0; i < 8; i++) h[i] = B2S_IV[i];
    h[0] ^= 0x01010020u;

    /* Zero bytes after password */
    for (int i = total_len; i < 64; i++) buf[i] = 0;

    /* Single compress with last=1, counter=total_len */
    b2s_compress(h, buf, (ulong)total_len, 1);

    uint max_iter = params.max_iter;

    for (uint iter = 1; iter <= max_iter; iter++) {
        /* Probe compact table */
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
                for (uint _z = 11; _z < HIT_STRIDE; _z++) hits[base+_z] = 0; } }
        if (iter < max_iter) {
            /* Hex-encode 8 LE hash words into buf[0..63] (64 hex chars) */
            for (int i = 0; i < 8; i++) {
                uint s = h[i];
                uint b0 = s & 0xff, b1 = (s >> 8) & 0xff;
                uint b2 = (s >> 16) & 0xff, b3 = (s >> 24) & 0xff;
                uint hi0 = (b0 >> 4) & 0xf, lo0 = b0 & 0xf;
                uint hi1 = (b1 >> 4) & 0xf, lo1 = b1 & 0xf;
                uint hi2 = (b2 >> 4) & 0xf, lo2 = b2 & 0xf;
                uint hi3 = (b3 >> 4) & 0xf, lo3 = b3 & 0xf;
                buf[i*8+0] = hi0 + ((hi0 < 10) ? '0' : ('a'-10));
                buf[i*8+1] = lo0 + ((lo0 < 10) ? '0' : ('a'-10));
                buf[i*8+2] = hi1 + ((hi1 < 10) ? '0' : ('a'-10));
                buf[i*8+3] = lo1 + ((lo1 < 10) ? '0' : ('a'-10));
                buf[i*8+4] = hi2 + ((hi2 < 10) ? '0' : ('a'-10));
                buf[i*8+5] = lo2 + ((lo2 < 10) ? '0' : ('a'-10));
                buf[i*8+6] = hi3 + ((hi3 < 10) ? '0' : ('a'-10));
                buf[i*8+7] = lo3 + ((lo3 < 10) ? '0' : ('a'-10));
            }
            /* Re-init BLAKE2S and compress 64-byte hex block */
            for (int i = 0; i < 8; i++) h[i] = B2S_IV[i];
            h[0] ^= 0x01010020u;
            b2s_compress(h, buf, 64UL, 1);
        }
    }
}
