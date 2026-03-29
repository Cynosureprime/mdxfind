/* md5salt.cl — OpenCL kernels for mdxfind MD5SALT GPU acceleration */

typedef struct {
    ulong compact_mask;
    uint num_words;
    uint num_salts;
    uint max_probe;
    uint hash_data_count;
    uint max_hits;
    uint overflow_count;
    uint max_iter;
} OCLParams;

__constant uint K[64] = {
    0xd76aa478,0xe8c7b756,0x242070db,0xc1bdceee,0xf57c0faf,0x4787c62a,0xa8304613,0xfd469501,
    0x698098d8,0x8b44f7af,0xffff5bb1,0x895cd7be,0x6b901122,0xfd987193,0xa679438e,0x49b40821,
    0xf61e2562,0xc040b340,0x265e5a51,0xe9b6c7aa,0xd62f105d,0x02441453,0xd8a1e681,0xe7d3fbc8,
    0x21e1cde6,0xc33707d6,0xf4d50d87,0x455a14ed,0xa9e3e905,0xfcefa3f8,0x676f02d9,0x8d2a4c8a,
    0xfffa3942,0x8771f681,0x6d9d6122,0xfde5380c,0xa4beea44,0x4bdecfa9,0xf6bb4b60,0xbebfbc70,
    0x289b7ec6,0xeaa127fa,0xd4ef3085,0x04881d05,0xd9d4d039,0xe6db99e5,0x1fa27cf8,0xc4ac5665,
    0xf4292244,0x432aff97,0xab9423a7,0xfc93a039,0x655b59c3,0x8f0ccc92,0xffeff47d,0x85845dd1,
    0x6fa87e4f,0xfe2ce6e0,0xa3014314,0x4e0811a1,0xf7537e82,0xbd3af235,0x2ad7d2bb,0xeb86d391
};

#define FF(a,b,c,d,m,s,k) { a += ((b&c)|(~b&d)) + m + k; a = b + rotate(a,s); }
#define GG(a,b,c,d,m,s,k) { a += ((d&b)|(~d&c)) + m + k; a = b + rotate(a,s); }
#define HH(a,b,c,d,m,s,k) { a += (b^c^d) + m + k; a = b + rotate(a,s); }
#define II(a,b,c,d,m,s,k) { a += (c^(~d|b)) + m + k; a = b + rotate(a,s); }

void md5_block(uint *h0, uint *h1, uint *h2, uint *h3, uint *M) {
    uint a = *h0, b = *h1, c = *h2, d = *h3;
    FF(a,b,c,d,M[0],(uint)7,0xd76aa478u);  FF(d,a,b,c,M[1],(uint)12,0xe8c7b756u);
    FF(c,d,a,b,M[2],(uint)17,0x242070dbu);  FF(b,c,d,a,M[3],(uint)22,0xc1bdceeeu);
    FF(a,b,c,d,M[4],(uint)7,0xf57c0fafu);   FF(d,a,b,c,M[5],(uint)12,0x4787c62au);
    FF(c,d,a,b,M[6],(uint)17,0xa8304613u);  FF(b,c,d,a,M[7],(uint)22,0xfd469501u);
    FF(a,b,c,d,M[8],(uint)7,0x698098d8u);   FF(d,a,b,c,M[9],(uint)12,0x8b44f7afu);
    FF(c,d,a,b,M[10],(uint)17,0xffff5bb1u); FF(b,c,d,a,M[11],(uint)22,0x895cd7beu);
    FF(a,b,c,d,M[12],(uint)7,0x6b901122u);  FF(d,a,b,c,M[13],(uint)12,0xfd987193u);
    FF(c,d,a,b,M[14],(uint)17,0xa679438eu); FF(b,c,d,a,M[15],(uint)22,0x49b40821u);
    GG(a,b,c,d,M[1],(uint)5,0xf61e2562u);   GG(d,a,b,c,M[6],(uint)9,0xc040b340u);
    GG(c,d,a,b,M[11],(uint)14,0x265e5a51u); GG(b,c,d,a,M[0],(uint)20,0xe9b6c7aau);
    GG(a,b,c,d,M[5],(uint)5,0xd62f105du);   GG(d,a,b,c,M[10],(uint)9,0x02441453u);
    GG(c,d,a,b,M[15],(uint)14,0xd8a1e681u); GG(b,c,d,a,M[4],(uint)20,0xe7d3fbc8u);
    GG(a,b,c,d,M[9],(uint)5,0x21e1cde6u);   GG(d,a,b,c,M[14],(uint)9,0xc33707d6u);
    GG(c,d,a,b,M[3],(uint)14,0xf4d50d87u);  GG(b,c,d,a,M[8],(uint)20,0x455a14edu);
    GG(a,b,c,d,M[13],(uint)5,0xa9e3e905u);  GG(d,a,b,c,M[2],(uint)9,0xfcefa3f8u);
    GG(c,d,a,b,M[7],(uint)14,0x676f02d9u);  GG(b,c,d,a,M[12],(uint)20,0x8d2a4c8au);
    HH(a,b,c,d,M[5],(uint)4,0xfffa3942u);   HH(d,a,b,c,M[8],(uint)11,0x8771f681u);
    HH(c,d,a,b,M[11],(uint)16,0x6d9d6122u); HH(b,c,d,a,M[14],(uint)23,0xfde5380cu);
    HH(a,b,c,d,M[1],(uint)4,0xa4beea44u);   HH(d,a,b,c,M[4],(uint)11,0x4bdecfa9u);
    HH(c,d,a,b,M[7],(uint)16,0xf6bb4b60u);  HH(b,c,d,a,M[10],(uint)23,0xbebfbc70u);
    HH(a,b,c,d,M[13],(uint)4,0x289b7ec6u);  HH(d,a,b,c,M[0],(uint)11,0xeaa127fau);
    HH(c,d,a,b,M[3],(uint)16,0xd4ef3085u);  HH(b,c,d,a,M[6],(uint)23,0x04881d05u);
    HH(a,b,c,d,M[9],(uint)4,0xd9d4d039u);   HH(d,a,b,c,M[12],(uint)11,0xe6db99e5u);
    HH(c,d,a,b,M[15],(uint)16,0x1fa27cf8u); HH(b,c,d,a,M[2],(uint)23,0xc4ac5665u);
    II(a,b,c,d,M[0],(uint)6,0xf4292244u);   II(d,a,b,c,M[7],(uint)10,0x432aff97u);
    II(c,d,a,b,M[14],(uint)15,0xab9423a7u); II(b,c,d,a,M[5],(uint)21,0xfc93a039u);
    II(a,b,c,d,M[12],(uint)6,0x655b59c3u);  II(d,a,b,c,M[3],(uint)10,0x8f0ccc92u);
    II(c,d,a,b,M[10],(uint)15,0xffeff47du); II(b,c,d,a,M[1],(uint)21,0x85845dd1u);
    II(a,b,c,d,M[8],(uint)6,0x6fa87e4fu);   II(d,a,b,c,M[15],(uint)10,0xfe2ce6e0u);
    II(c,d,a,b,M[6],(uint)15,0xa3014314u);  II(b,c,d,a,M[13],(uint)21,0x4e0811a1u);
    II(a,b,c,d,M[4],(uint)6,0xf7537e82u);   II(d,a,b,c,M[11],(uint)10,0xbd3af235u);
    II(c,d,a,b,M[2],(uint)15,0x2ad7d2bbu);  II(b,c,d,a,M[9],(uint)21,0xeb86d391u);
    *h0 += a; *h1 += b; *h2 += c; *h3 += d;
}

ulong compact_mix(ulong k) {
    return k ^ (k >> 32);
}

int probe_compact(uint hx, uint hy, uint hz, uint hw,
    __global const uint *compact_fp, __global const uint *compact_idx,
    ulong compact_mask, uint max_probe, uint hash_data_count,
    __global const uchar *hash_data_buf, __global const ulong *hash_data_off,
    __global const ulong *overflow_keys, __global const uchar *overflow_hashes,
    __global const uint *overflow_offsets, uint overflow_count)
{
    ulong key = ((ulong)hy << 32) | hx;
    uint fp = (uint)(key >> 32);
    if (fp == 0) fp = 1;
    ulong pos = compact_mix(key) & compact_mask;
    for (int p = 0; p < (int)max_probe; p++) {
        uint cfp = compact_fp[pos];
        if (cfp == 0) break;
        if (cfp == fp) {
            uint idx = compact_idx[pos];
            if (idx < hash_data_count) {
                ulong off = hash_data_off[idx];
                __global const uint *ref = (__global const uint *)(hash_data_buf + off);
                if (hx == ref[0] && hy == ref[1] && hz == ref[2] && hw == ref[3])
                    return 1;
            }
        }
        pos = (pos + 1) & compact_mask;
    }
    if (overflow_count > 0) {
        int lo = 0, hi = (int)overflow_count - 1;
        while (lo <= hi) {
            int mid = (lo + hi) / 2;
            ulong mkey = overflow_keys[mid];
            if (key < mkey) hi = mid - 1;
            else if (key > mkey) lo = mid + 1;
            else {
                uint ooff = overflow_offsets[mid];
                __global const uint *oref = (__global const uint *)(overflow_hashes + ooff);
                if (hx == oref[0] && hy == oref[1] && hz == oref[2] && hw == oref[3]) return 1;
                for (int d = mid-1; d >= 0 && overflow_keys[d] == key; d--) {
                    oref = (__global const uint *)(overflow_hashes + overflow_offsets[d]);
                    if (hx == oref[0] && hy == oref[1] && hz == oref[2] && hw == oref[3]) return 1;
                }
                for (int d = mid+1; d < (int)overflow_count && overflow_keys[d] == key; d++) {
                    oref = (__global const uint *)(overflow_hashes + overflow_offsets[d]);
                    if (hx == oref[0] && hy == oref[1] && hz == oref[2] && hw == oref[3]) return 1;
                }
                break;
            }
        }
    }
    return 0;
}

__kernel void md5salt_batch(
    __global const uchar *hexhashes, __global const ushort *hexlens,
    __global const uchar *salts, __global const uint *salt_offsets, __global const ushort *salt_lens,
    __global const uint *compact_fp, __global const uint *compact_idx,
    __global const OCLParams *params_buf,
    __global const uchar *hash_data_buf, __global const ulong *hash_data_off, __global const ushort *hash_data_len,
    __global uint *hits, __global volatile uint *hit_count,
    __global const ulong *overflow_keys, __global const uchar *overflow_hashes,
    __global const uint *overflow_offsets, __global const ushort *overflow_lengths)
{
    OCLParams params = *params_buf;
    uint tid = get_global_id(0);
    uint word_idx = tid / params.num_salts;
    uint salt_idx = tid % params.num_salts;
    if (word_idx >= params.num_words) return;

    uint M[16];
    __global const uint *mwords = (__global const uint *)(hexhashes + word_idx * 256);
    for (int i = 0; i < 8; i++) M[i] = mwords[i];
    for (int i = 8; i < 16; i++) M[i] = 0;

    uint soff = salt_offsets[salt_idx];
    int slen = salt_lens[salt_idx];
    int total_len = 32 + slen;
    uchar *mbytes = (uchar *)M;
    for (int i = 0; i < slen; i++)
        mbytes[32 + i] = salts[soff + i];
    mbytes[total_len] = 0x80;
    M[14] = total_len * 8;

    uint hx = 0x67452301, hy = 0xEFCDAB89, hz = 0x98BADCFE, hw = 0x10325476;
    md5_block(&hx, &hy, &hz, &hw, M);


    if (probe_compact(hx, hy, hz, hw, compact_fp, compact_idx,
                      params.compact_mask, params.max_probe, params.hash_data_count,
                      hash_data_buf, hash_data_off,
                      overflow_keys, overflow_hashes, overflow_offsets, params.overflow_count)) {
        uint slot = atomic_add(hit_count, 1u);
        if (slot < params.max_hits) {
            uint base = slot * 6;
            hits[base] = word_idx; hits[base+1] = salt_idx;
            hits[base+2] = hx; hits[base+3] = hy;
            hits[base+4] = hz; hits[base+5] = hw;
        }
    }
}

__kernel void md5salt_sub8_24(
    __global const uchar *hexhashes, __global const ushort *hexlens,
    __global const uchar *salts, __global const uint *salt_offsets, __global const ushort *salt_lens,
    __global const uint *compact_fp, __global const uint *compact_idx,
    __global const OCLParams *params_buf,
    __global const uchar *hash_data_buf, __global const ulong *hash_data_off, __global const ushort *hash_data_len,
    __global uint *hits, __global volatile uint *hit_count,
    __global const ulong *overflow_keys, __global const uchar *overflow_hashes,
    __global const uint *overflow_offsets, __global const ushort *overflow_lengths)
{
    OCLParams params = *params_buf;
    uint tid = get_global_id(0);
    uint word_idx = tid / params.num_salts;
    uint salt_idx = tid % params.num_salts;
    if (word_idx >= params.num_words) return;

    uint M[16];
    __global const uint *mwords = (__global const uint *)(hexhashes + word_idx * 256);
    for (int i = 0; i < 4; i++) M[i] = mwords[i];
    for (int i = 4; i < 16; i++) M[i] = 0;

    uint soff = salt_offsets[salt_idx];
    int slen = salt_lens[salt_idx];
    int total_len = 16 + slen;
    uchar *mbytes = (uchar *)M;
    for (int i = 0; i < slen; i++) mbytes[16 + i] = salts[soff + i];
    mbytes[total_len] = 0x80;
    M[14] = total_len * 8;

    uint hx = 0x67452301, hy = 0xEFCDAB89, hz = 0x98BADCFE, hw = 0x10325476;
    md5_block(&hx, &hy, &hz, &hw, M);

    if (probe_compact(hx, hy, hz, hw, compact_fp, compact_idx,
                      params.compact_mask, params.max_probe, params.hash_data_count,
                      hash_data_buf, hash_data_off,
                      overflow_keys, overflow_hashes, overflow_offsets, params.overflow_count)) {
        uint slot = atomic_add(hit_count, 1u);
        if (slot < params.max_hits) {
            uint base = slot * 6;
            hits[base] = word_idx; hits[base+1] = salt_idx;
            hits[base+2] = hx; hits[base+3] = hy;
            hits[base+4] = hz; hits[base+5] = hw;
        }
    }
}

__kernel void md5salt_iter(
    __global const uchar *hexhashes, __global const ushort *hexlens,
    __global const uchar *salts, __global const uint *salt_offsets, __global const ushort *salt_lens,
    __global const uint *compact_fp, __global const uint *compact_idx,
    __global const OCLParams *params_buf,
    __global const uchar *hash_data_buf, __global const ulong *hash_data_off, __global const ushort *hash_data_len,
    __global uint *hits, __global volatile uint *hit_count,
    __global const ulong *overflow_keys, __global const uchar *overflow_hashes,
    __global const uint *overflow_offsets, __global const ushort *overflow_lengths)
{
    OCLParams params = *params_buf;
    uint tid = get_global_id(0);
    uint word_idx = tid / params.num_salts;
    uint salt_idx = tid % params.num_salts;
    if (word_idx >= params.num_words) return;

    uint M[16];
    __global const uint *mwords = (__global const uint *)(hexhashes + word_idx * 256);
    for (int i = 0; i < 8; i++) M[i] = mwords[i];
    for (int i = 8; i < 16; i++) M[i] = 0;

    uint soff = salt_offsets[salt_idx];
    int slen = salt_lens[salt_idx];
    int total_len = 32 + slen;
    uchar *mbytes = (uchar *)M;
    for (int i = 0; i < slen; i++) mbytes[32 + i] = salts[soff + i];
    mbytes[total_len] = 0x80;
    M[14] = total_len * 8;

    uint hx = 0x67452301, hy = 0xEFCDAB89, hz = 0x98BADCFE, hw = 0x10325476;
    md5_block(&hx, &hy, &hz, &hw, M);

    for (uint iter = 0; iter < params.max_iter; iter++) {
        if (iter > 0) {
            uint hwords[4]; hwords[0]=hx; hwords[1]=hy; hwords[2]=hz; hwords[3]=hw;
            uchar *hb = (uchar *)hwords;
            uint Mi[16];
            uchar *mb = (uchar *)Mi;
            for (int i = 0; i < 16; i++) {
                uchar hi = hb[i] >> 4, lo = hb[i] & 0xf;
                mb[i*2]   = hi + (hi < 10 ? '0' : 'a' - 10);
                mb[i*2+1] = lo + (lo < 10 ? '0' : 'a' - 10);
            }
            Mi[8] = 0x80; Mi[9]=0; Mi[10]=0; Mi[11]=0;
            Mi[12]=0; Mi[13]=0; Mi[14]=256; Mi[15]=0;
            hx = 0x67452301; hy = 0xEFCDAB89; hz = 0x98BADCFE; hw = 0x10325476;
            md5_block(&hx, &hy, &hz, &hw, Mi);
        }
        if (probe_compact(hx, hy, hz, hw, compact_fp, compact_idx,
                          params.compact_mask, params.max_probe, params.hash_data_count,
                          hash_data_buf, hash_data_off,
                          overflow_keys, overflow_hashes, overflow_offsets, params.overflow_count)) {
            uint slot = atomic_add(hit_count, 1u);
            if (slot < params.max_hits) {
                uint base = slot * 7;
                hits[base] = word_idx; hits[base+1] = salt_idx;
                hits[base+2] = iter + 1;
                hits[base+3] = hx; hits[base+4] = hy;
                hits[base+5] = hz; hits[base+6] = hw;
            }
        }
    }
}

/* ---- Hex conversion for MD5 iteration ---- */

/* Convert one byte to 2 packed hex chars (lowercase) */
uint hex_byte_lc(uint b) {
    uint hi = (b >> 4) & 0xf;
    uint lo = b & 0xf;
    uint hc = hi + ((hi < 10) ? '0' : ('a' - 10));
    uint lc = lo + ((lo < 10) ? '0' : ('a' - 10));
    return hc | (lc << 8);
}

/* Convert one byte to 2 packed hex chars (uppercase) */
uint hex_byte_uc(uint b) {
    uint hi = (b >> 4) & 0xf;
    uint lo = b & 0xf;
    uint hc = hi + ((hi < 10) ? '0' : ('A' - 10));
    uint lc = lo + ((lo < 10) ? '0' : ('A' - 10));
    return hc | (lc << 8);
}

/* Convert 4 MD5 uint32s to 8 M[] words of hex text (lowercase) */
void md5_to_hex_lc(uint hx, uint hy, uint hz, uint hw, uint *M) {
    uint v[4]; v[0]=hx; v[1]=hy; v[2]=hz; v[3]=hw;
    for (int i = 0; i < 4; i++) {
        uint b0 = v[i] & 0xff, b1 = (v[i]>>8) & 0xff;
        uint b2 = (v[i]>>16) & 0xff, b3 = (v[i]>>24) & 0xff;
        M[i*2]   = hex_byte_lc(b0) | (hex_byte_lc(b1) << 16);
        M[i*2+1] = hex_byte_lc(b2) | (hex_byte_lc(b3) << 16);
    }
}

/* Convert 4 MD5 uint32s to 8 M[] words of hex text (uppercase) */
void md5_to_hex_uc(uint hx, uint hy, uint hz, uint hw, uint *M) {
    uint v[4]; v[0]=hx; v[1]=hy; v[2]=hz; v[3]=hw;
    for (int i = 0; i < 4; i++) {
        uint b0 = v[i] & 0xff, b1 = (v[i]>>8) & 0xff;
        uint b2 = (v[i]>>16) & 0xff, b3 = (v[i]>>24) & 0xff;
        M[i*2]   = hex_byte_uc(b0) | (hex_byte_uc(b1) << 16);
        M[i*2+1] = hex_byte_uc(b2) | (hex_byte_uc(b3) << 16);
    }
}

/* ---- Bare MD5 iteration kernel (e1 lowercase, e2 uppercase) ----
 *
 * Input: hex(MD5(password)) pre-packed by CPU into M[] words.
 * GPU iterates: MD5(hex) -> check -> hex(result) -> MD5(hex) -> ...
 * One work item per word (no salt dimension).
 * Hit stride is always 7 (includes iteration number).
 */
__kernel void md5_iter_lc(
    __global const uchar *hexhashes, __global const ushort *hexlens,
    __global const uchar *salts_unused, __global const uint *salt_off_unused,
    __global const ushort *salt_len_unused,
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
    if (tid >= params.num_words) return;

    uint M[16];
    __global const uint *mwords = (__global const uint *)(hexhashes + tid * 256);
    for (int i = 0; i < 8; i++) M[i] = mwords[i];

    /* MD5 padding for 32-byte input -- constant across all iterations */
    for (int i = 8; i < 16; i++) M[i] = 0;
    M[8] = 0x80;
    M[14] = 32 * 8;

    for (uint iter = 1; iter <= params.max_iter; iter++) {
        uint hx = 0x67452301, hy = 0xEFCDAB89, hz = 0x98BADCFE, hw = 0x10325476;
        md5_block(&hx, &hy, &hz, &hw, M);

        if (probe_compact(hx, hy, hz, hw, compact_fp, compact_idx,
                          params.compact_mask, params.max_probe, params.hash_data_count,
                          hash_data_buf, hash_data_off,
                          overflow_keys, overflow_hashes, overflow_offsets, params.overflow_count)) {
            uint slot = atomic_add(hit_count, 1u);
            if (slot < params.max_hits) {
                uint base = slot * 7;
                hits[base] = tid; hits[base+1] = 0;
                hits[base+2] = iter + 1; /* +1: GPU iter 1 = mdxfind iter 2 */
                hits[base+3] = hx; hits[base+4] = hy;
                hits[base+5] = hz; hits[base+6] = hw;
            }
        }

        if (iter < params.max_iter) {
            md5_to_hex_lc(hx, hy, hz, hw, M);
            /* M[8..15] unchanged: padding for 32-byte input */
        }
    }
}

__kernel void md5_iter_uc(
    __global const uchar *hexhashes, __global const ushort *hexlens,
    __global const uchar *salts_unused, __global const uint *salt_off_unused,
    __global const ushort *salt_len_unused,
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
    if (tid >= params.num_words) return;

    uint M[16];
    __global const uint *mwords = (__global const uint *)(hexhashes + tid * 256);
    for (int i = 0; i < 8; i++) M[i] = mwords[i];

    for (int i = 8; i < 16; i++) M[i] = 0;
    M[8] = 0x80;
    M[14] = 32 * 8;

    for (uint iter = 1; iter <= params.max_iter; iter++) {
        uint hx = 0x67452301, hy = 0xEFCDAB89, hz = 0x98BADCFE, hw = 0x10325476;
        md5_block(&hx, &hy, &hz, &hw, M);

        if (probe_compact(hx, hy, hz, hw, compact_fp, compact_idx,
                          params.compact_mask, params.max_probe, params.hash_data_count,
                          hash_data_buf, hash_data_off,
                          overflow_keys, overflow_hashes, overflow_offsets, params.overflow_count)) {
            uint slot = atomic_add(hit_count, 1u);
            if (slot < params.max_hits) {
                uint base = slot * 7;
                hits[base] = tid; hits[base+1] = 0;
                hits[base+2] = iter + 1; /* +1: GPU iter 1 = mdxfind iter 2 */
                hits[base+3] = hx; hits[base+4] = hy;
                hits[base+5] = hz; hits[base+6] = hw;
            }
        }

        if (iter < params.max_iter) {
            md5_to_hex_uc(hx, hy, hz, hw, M);
        }
    }
}
