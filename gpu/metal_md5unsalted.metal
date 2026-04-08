/* metal_md5unsalted.metal — Pre-padded unsalted MD5 with mask expansion
 *
 * Input: 4096 pre-padded 64-byte M[] blocks (word_stride=64).
 * Dispatch: num_words × num_masks threads.
 * Hit stride: 6 (word_idx, mask_idx, hx, hy, hz, hw)
 */

constant uchar cs_d[] = "0123456789";
constant uchar cs_l[] = "abcdefghijklmnopqrstuvwxyz";
constant uchar cs_u[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
constant uchar cs_s[] = " !\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~";

static inline uint charset_size(uchar id) {
    switch (id) {
        case 0: return 10;  case 1: return 26;  case 2: return 26;
        case 3: return 33;  case 4: return 95;  case 5: return 256;
        default: return 10;
    }
}

static inline uchar charset_char(uchar id, uint idx) {
    switch (id) {
        case 0: return cs_d[idx];  case 1: return cs_l[idx];
        case 2: return cs_u[idx];  case 3: return cs_s[idx];
        case 4: return (uchar)(0x20 + idx);
        case 5: return (uchar)idx;
        default: return cs_d[idx];
    }
}

static void md5_compress(thread uint *hx, thread uint *hy, thread uint *hz, thread uint *hw, thread uint *M) {
    uint a = *hx, b = *hy, c = *hz, d = *hw;
    a += ((b&c)|(~b&d)) + M[ 0] + 0xd76aa478u; a = b + ((a<< 7)|(a>>25));
    d += ((a&b)|(~a&c)) + M[ 1] + 0xe8c7b756u; d = a + ((d<<12)|(d>>20));
    c += ((d&a)|(~d&b)) + M[ 2] + 0x242070dbu; c = d + ((c<<17)|(c>>15));
    b += ((c&d)|(~c&a)) + M[ 3] + 0xc1bdceeeu; b = c + ((b<<22)|(b>>10));
    a += ((b&c)|(~b&d)) + M[ 4] + 0xf57c0fafu; a = b + ((a<< 7)|(a>>25));
    d += ((a&b)|(~a&c)) + M[ 5] + 0x4787c62au; d = a + ((d<<12)|(d>>20));
    c += ((d&a)|(~d&b)) + M[ 6] + 0xa8304613u; c = d + ((c<<17)|(c>>15));
    b += ((c&d)|(~c&a)) + M[ 7] + 0xfd469501u; b = c + ((b<<22)|(b>>10));
    a += ((b&c)|(~b&d)) + M[ 8] + 0x698098d8u; a = b + ((a<< 7)|(a>>25));
    d += ((a&b)|(~a&c)) + M[ 9] + 0x8b44f7afu; d = a + ((d<<12)|(d>>20));
    c += ((d&a)|(~d&b)) + M[10] + 0xffff5bb1u; c = d + ((c<<17)|(c>>15));
    b += ((c&d)|(~c&a)) + M[11] + 0x895cd7beu; b = c + ((b<<22)|(b>>10));
    a += ((b&c)|(~b&d)) + M[12] + 0x6b901122u; a = b + ((a<< 7)|(a>>25));
    d += ((a&b)|(~a&c)) + M[13] + 0xfd987193u; d = a + ((d<<12)|(d>>20));
    c += ((d&a)|(~d&b)) + M[14] + 0xa679438eu; c = d + ((c<<17)|(c>>15));
    b += ((c&d)|(~c&a)) + M[15] + 0x49b40821u; b = c + ((b<<22)|(b>>10));
    a += ((d&b)|(~d&c)) + M[ 1] + 0xf61e2562u; a = b + ((a<< 5)|(a>>27));
    d += ((c&a)|(~c&b)) + M[ 6] + 0xc040b340u; d = a + ((d<< 9)|(d>>23));
    c += ((b&d)|(~b&a)) + M[11] + 0x265e5a51u; c = d + ((c<<14)|(c>>18));
    b += ((a&c)|(~a&d)) + M[ 0] + 0xe9b6c7aau; b = c + ((b<<20)|(b>>12));
    a += ((d&b)|(~d&c)) + M[ 5] + 0xd62f105du; a = b + ((a<< 5)|(a>>27));
    d += ((c&a)|(~c&b)) + M[10] + 0x02441453u; d = a + ((d<< 9)|(d>>23));
    c += ((b&d)|(~b&a)) + M[15] + 0xd8a1e681u; c = d + ((c<<14)|(c>>18));
    b += ((a&c)|(~a&d)) + M[ 4] + 0xe7d3fbc8u; b = c + ((b<<20)|(b>>12));
    a += ((d&b)|(~d&c)) + M[ 9] + 0x21e1cde6u; a = b + ((a<< 5)|(a>>27));
    d += ((c&a)|(~c&b)) + M[14] + 0xc33707d6u; d = a + ((d<< 9)|(d>>23));
    c += ((b&d)|(~b&a)) + M[ 3] + 0xf4d50d87u; c = d + ((c<<14)|(c>>18));
    b += ((a&c)|(~a&d)) + M[ 8] + 0x455a14edu; b = c + ((b<<20)|(b>>12));
    a += ((d&b)|(~d&c)) + M[13] + 0xa9e3e905u; a = b + ((a<< 5)|(a>>27));
    d += ((c&a)|(~c&b)) + M[ 2] + 0xfcefa3f8u; d = a + ((d<< 9)|(d>>23));
    c += ((b&d)|(~b&a)) + M[ 7] + 0x676f02d9u; c = d + ((c<<14)|(c>>18));
    b += ((a&c)|(~a&d)) + M[12] + 0x8d2a4c8au; b = c + ((b<<20)|(b>>12));
    a += (b^c^d) + M[ 5] + 0xfffa3942u; a = b + ((a<< 4)|(a>>28));
    d += (a^b^c) + M[ 8] + 0x8771f681u; d = a + ((d<<11)|(d>>21));
    c += (d^a^b) + M[11] + 0x6d9d6122u; c = d + ((c<<16)|(c>>16));
    b += (c^d^a) + M[14] + 0xfde5380cu; b = c + ((b<<23)|(b>> 9));
    a += (b^c^d) + M[ 1] + 0xa4beea44u; a = b + ((a<< 4)|(a>>28));
    d += (a^b^c) + M[ 4] + 0x4bdecfa9u; d = a + ((d<<11)|(d>>21));
    c += (d^a^b) + M[ 7] + 0xf6bb4b60u; c = d + ((c<<16)|(c>>16));
    b += (c^d^a) + M[10] + 0xbebfbc70u; b = c + ((b<<23)|(b>> 9));
    a += (b^c^d) + M[13] + 0x289b7ec6u; a = b + ((a<< 4)|(a>>28));
    d += (a^b^c) + M[ 0] + 0xeaa127fau; d = a + ((d<<11)|(d>>21));
    c += (d^a^b) + M[ 3] + 0xd4ef3085u; c = d + ((c<<16)|(c>>16));
    b += (c^d^a) + M[ 6] + 0x04881d05u; b = c + ((b<<23)|(b>> 9));
    a += (b^c^d) + M[ 9] + 0xd9d4d039u; a = b + ((a<< 4)|(a>>28));
    d += (a^b^c) + M[12] + 0xe6db99e5u; d = a + ((d<<11)|(d>>21));
    c += (d^a^b) + M[15] + 0x1fa27cf8u; c = d + ((c<<16)|(c>>16));
    b += (c^d^a) + M[ 2] + 0xc4ac5665u; b = c + ((b<<23)|(b>> 9));
    a += (c^(~d|b)) + M[ 0] + 0xf4292244u; a = b + ((a<< 6)|(a>>26));
    d += (b^(~c|a)) + M[ 7] + 0x432aff97u; d = a + ((d<<10)|(d>>22));
    c += (a^(~b|d)) + M[14] + 0xab9423a7u; c = d + ((c<<15)|(c>>17));
    b += (d^(~a|c)) + M[ 5] + 0xfc93a039u; b = c + ((b<<21)|(b>>11));
    a += (c^(~d|b)) + M[12] + 0x655b59c3u; a = b + ((a<< 6)|(a>>26));
    d += (b^(~c|a)) + M[ 3] + 0x8f0ccc92u; d = a + ((d<<10)|(d>>22));
    c += (a^(~b|d)) + M[10] + 0xffeff47du; c = d + ((c<<15)|(c>>17));
    b += (d^(~a|c)) + M[ 1] + 0x85845dd1u; b = c + ((b<<21)|(b>>11));
    a += (c^(~d|b)) + M[ 8] + 0x6fa87e4fu; a = b + ((a<< 6)|(a>>26));
    d += (b^(~c|a)) + M[15] + 0xfe2ce6e0u; d = a + ((d<<10)|(d>>22));
    c += (a^(~b|d)) + M[ 6] + 0xa3014314u; c = d + ((c<<15)|(c>>17));
    b += (d^(~a|c)) + M[13] + 0x4e0811a1u; b = c + ((b<<21)|(b>>11));
    a += (c^(~d|b)) + M[ 4] + 0xf7537e82u; a = b + ((a<< 6)|(a>>26));
    d += (b^(~c|a)) + M[11] + 0xbd3af235u; d = a + ((d<<10)|(d>>22));
    c += (a^(~b|d)) + M[ 2] + 0x2ad7d2bbu; c = d + ((c<<15)|(c>>17));
    b += (d^(~a|c)) + M[ 9] + 0xeb86d391u; b = c + ((b<<21)|(b>>11));
    *hx += a; *hy += b; *hz += c; *hw += d;
}

kernel void md5_unsalted_batch(
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
    uint mask_idx = params.mask_start + (tid % params.num_masks);
    if (word_idx >= params.num_words) return;

    device const uint *src = words + word_idx * 16;
    uint M[16];
    for (int i = 0; i < 16; i++) M[i] = src[i];

    uint n_pre = params.n_prepend;
    uint n_app = params.n_append;

    if (n_pre > 0 || n_app > 0) {
        uint append_combos = 1;
        for (uint i = 0; i < n_app; i++)
            append_combos *= charset_size(mask_desc[n_pre + i]);

        uint prepend_idx = mask_idx / append_combos;
        uint append_idx = mask_idx % append_combos;

        if (n_pre > 0) {
            uint pidx = prepend_idx;
            for (int i = (int)n_pre - 1; i >= 0; i--) {
                uint sz = charset_size(mask_desc[i]);
                uchar ch = charset_char(mask_desc[i], pidx % sz);
                pidx /= sz;
                M[i >> 2] = (M[i >> 2] & ~(0xFFu << ((i & 3) << 3)))
                           | ((uint)ch << ((i & 3) << 3));
            }
        }

        if (n_app > 0) {
            int total_len = M[14] >> 3;
            int app_start = total_len - (int)n_app;
            uint aidx = append_idx;
            for (int i = (int)n_app - 1; i >= 0; i--) {
                uchar csid = mask_desc[n_pre + i];
                uint sz = charset_size(csid);
                uchar ch = charset_char(csid, aidx % sz);
                aidx /= sz;
                int pos = app_start + i;
                M[pos >> 2] = (M[pos >> 2] & ~(0xFFu << ((pos & 3) << 3)))
                             | ((uint)ch << ((pos & 3) << 3));
            }
        }
    }

    uint hx = 0x67452301u, hy = 0xEFCDAB89u, hz = 0x98BADCFEu, hw = 0x10325476u;
    md5_compress(&hx, &hy, &hz, &hw, M);

    uint4 h = uint4(hx, hy, hz, hw);
    ulong key = (ulong(h.y) << 32) | h.x;
    uint fp = uint(key >> 32);
    if (fp == 0) fp = 1;
    ulong pos = (key ^ (key >> 32)) & params.compact_mask;
    for (uint p = 0; p < params.max_probe; p++) {
        uint cfp = compact_fp[pos];
        if (cfp == 0) break;
        if (cfp == fp) {
            uint idx = compact_idx[pos];
            if (idx < params.hash_data_count) {
                ulong off = hash_data_off[idx];
                device const uint *ref = (device const uint *)(hash_data_buf + off);
                if (h.x == ref[0] && h.y == ref[1] && h.z == ref[2] && h.w == ref[3]) {
                    uint slot = atomic_fetch_add_explicit(hit_count, 1, memory_order_relaxed);
                    if (slot < params.max_hits) {
                        uint base = slot * 6;
                        hits[base] = word_idx; hits[base+1] = mask_idx;
                        hits[base+2] = h.x; hits[base+3] = h.y;
                        hits[base+4] = h.z; hits[base+5] = h.w;
                    }
                    return;
                }
            }
        }
        pos = (pos + 1) & params.compact_mask;
    }
    if (params.overflow_count > 0) {
        int lo = 0, hi = int(params.overflow_count) - 1;
        while (lo <= hi) {
            int mid = (lo + hi) / 2;
            ulong mkey = overflow_keys[mid];
            if (key < mkey) hi = mid - 1;
            else if (key > mkey) lo = mid + 1;
            else {
                uint ooff = overflow_offsets[mid];
                device const uint *oref = (device const uint *)(overflow_hashes + ooff);
                if (h.x == oref[0] && h.y == oref[1] && h.z == oref[2] && h.w == oref[3]) {
                    uint slot = atomic_fetch_add_explicit(hit_count, 1, memory_order_relaxed);
                    if (slot < params.max_hits) {
                        uint base = slot * 6;
                        hits[base] = word_idx; hits[base+1] = mask_idx;
                        hits[base+2] = h.x; hits[base+3] = h.y;
                        hits[base+4] = h.z; hits[base+5] = h.w;
                    }
                    return;
                }
                for (int d = mid-1; d >= 0 && overflow_keys[d] == key; d--) {
                    oref = (device const uint *)(overflow_hashes + overflow_offsets[d]);
                    if (h.x == oref[0] && h.y == oref[1] && h.z == oref[2] && h.w == oref[3]) {
                        uint slot = atomic_fetch_add_explicit(hit_count, 1, memory_order_relaxed);
                        if (slot < params.max_hits) {
                            uint base = slot * 6;
                            hits[base] = word_idx; hits[base+1] = mask_idx;
                            hits[base+2] = h.x; hits[base+3] = h.y;
                            hits[base+4] = h.z; hits[base+5] = h.w;
                        }
                        return;
                    }
                }
                for (int d = mid+1; d < int(params.overflow_count) && overflow_keys[d] == key; d++) {
                    oref = (device const uint *)(overflow_hashes + overflow_offsets[d]);
                    if (h.x == oref[0] && h.y == oref[1] && h.z == oref[2] && h.w == oref[3]) {
                        uint slot = atomic_fetch_add_explicit(hit_count, 1, memory_order_relaxed);
                        if (slot < params.max_hits) {
                            uint base = slot * 6;
                            hits[base] = word_idx; hits[base+1] = mask_idx;
                            hits[base+2] = h.x; hits[base+3] = h.y;
                            hits[base+4] = h.z; hits[base+5] = h.w;
                        }
                        return;
                    }
                }
                break;
            }
        }
    }
}
