/* metal_sha512unsalted.metal — Pre-padded unsalted SHA512/SHA384 with mask expansion
 *
 * SELF-CONTAINED: This file includes its own Metal preamble, MetalParams,
 * and SHA512 functions. It is NOT concatenated with metal_common.metal
 * because the combined source exceeds the Metal JIT compiler's memory limits.
 */
#include <metal_stdlib>
using namespace metal;

constant uint HIT_STRIDE = 19;

struct MetalParams {
    /* 8-byte fields (offset 0-31) */
    uint64_t compact_mask;    /*  0: hash table mask */
    uint64_t mask_start;      /*  8: mask keyspace offset */
    uint64_t mask_base0;      /* 16: pre-decomposed positions 0-7 */
    uint64_t mask_base1;      /* 24: pre-decomposed positions 8-15 */
    /* 4-byte fields (offset 32-79) */
    uint     num_words;       /* 32: words in batch */
    uint     num_salts;       /* 36: salts for dispatch */
    uint     salt_start;      /* 40: starting salt index */
    uint     max_probe;       /* 44: compact table probe depth */
    uint     hash_data_count; /* 48: hash_data entries */
    uint     max_hits;        /* 52: hit buffer capacity */
    uint     overflow_count;  /* 56: overflow table entries */
    uint     max_iter;        /* 60: iteration count (-i) */
    uint     num_masks;       /* 64: mask combinations per chunk */
    uint     n_prepend;       /* 68: prepend mask positions (-N) */
    uint     n_append;        /* 72: append mask positions (-n) */
    uint     iter_count;      /* 76: per-dispatch iteration (PHPBB3) */
    uint     reserved32[4];   /* 80-95: reserved */
    uint64_t reserved64[4];   /* 96-127: reserved */
};

static inline ulong bswap64(ulong x) {
    return ((x >> 56) & 0xffUL) | ((x >> 40) & 0xff00UL) |
           ((x >> 24) & 0xff0000UL) | ((x >> 8) & 0xff000000UL) |
           ((x << 8) & 0xff00000000UL) | ((x << 24) & 0xff0000000000UL) |
           ((x << 40) & 0xff000000000000UL) | ((x << 56) & 0xff00000000000000UL);
}

static inline ulong rotr64(ulong x, uint n) { return (x >> n) | (x << (64 - n)); }

static ulong hex_byte_be64(uint b) {
    uint lo = b & 0xf, hi = b >> 4;
    return ((ulong)(hi < 10 ? hi + '0' : hi - 10 + 'a') << 8)
         | (lo < 10 ? lo + '0' : lo - 10 + 'a');
}

constant ulong K512[80] = {
    0x428a2f98d728ae22UL,0x7137449123ef65cdUL,0xb5c0fbcfec4d3b2fUL,0xe9b5dba58189dbbcUL,
    0x3956c25bf348b538UL,0x59f111f1b605d019UL,0x923f82a4af194f9bUL,0xab1c5ed5da6d8118UL,
    0xd807aa98a3030242UL,0x12835b0145706fbeUL,0x243185be4ee4b28cUL,0x550c7dc3d5ffb4e2UL,
    0x72be5d74f27b896fUL,0x80deb1fe3b1696b1UL,0x9bdc06a725c71235UL,0xc19bf174cf692694UL,
    0xe49b69c19ef14ad2UL,0xefbe4786384f25e3UL,0x0fc19dc68b8cd5b5UL,0x240ca1cc77ac9c65UL,
    0x2de92c6f592b0275UL,0x4a7484aa6ea6e483UL,0x5cb0a9dcbd41fbd4UL,0x76f988da831153b5UL,
    0x983e5152ee66dfabUL,0xa831c66d2db43210UL,0xb00327c898fb213fUL,0xbf597fc7beef0ee4UL,
    0xc6e00bf33da88fc2UL,0xd5a79147930aa725UL,0x06ca6351e003826fUL,0x142929670a0e6e70UL,
    0x27b70a8546d22ffcUL,0x2e1b21385c26c926UL,0x4d2c6dfc5ac42aedUL,0x53380d139d95b3dfUL,
    0x650a73548baf63deUL,0x766a0abb3c77b2a8UL,0x81c2c92e47edaee6UL,0x92722c851482353bUL,
    0xa2bfe8a14cf10364UL,0xa81a664bbc423001UL,0xc24b8b70d0f89791UL,0xc76c51a30654be30UL,
    0xd192e819d6ef5218UL,0xd69906245565a910UL,0xf40e35855771202aUL,0x106aa07032bbd1b8UL,
    0x19a4c116b8d2d0c8UL,0x1e376c085141ab53UL,0x2748774cdf8eeb99UL,0x34b0bcb5e19b48a8UL,
    0x391c0cb3c5c95a63UL,0x4ed8aa4ae3418acbUL,0x5b9cca4f7763e373UL,0x682e6ff3d6b2b8a3UL,
    0x748f82ee5defb2fcUL,0x78a5636f43172f60UL,0x84c87814a1f0ab72UL,0x8cc702081a6439ecUL,
    0x90befffa23631e28UL,0xa4506cebde82bde9UL,0xbef9a3f7b2c67915UL,0xc67178f2e372532bUL,
    0xca273eceea26619cUL,0xd186b8c721c0c207UL,0xeada7dd6cde0eb1eUL,0xf57d4f7fee6ed178UL,
    0x06f067aa72176fbaUL,0x0a637dc5a2c898a6UL,0x113f9804bef90daeUL,0x1b710b35131c471bUL,
    0x28db77f523047d84UL,0x32caab7b40c72493UL,0x3c9ebe0a15c9bebcUL,0x431d67c49c100d4cUL,
    0x4cc5d4becb3e42b6UL,0x597f299cfc657e2aUL,0x5fcb6fab3ad6faecUL,0x6c44198c4a475817UL
};

static void __attribute__((noinline)) sha512_compress(thread ulong *state, thread ulong *M) {
    ulong W[16];
    for (int i=0;i<16;i++) W[i]=M[i];
    ulong a=state[0],b=state[1],c=state[2],d=state[3],
          e=state[4],f=state[5],g=state[6],h=state[7];
    #define S5R(i,wi) { \
        ulong S1=rotr64(e,14)^rotr64(e,18)^rotr64(e,41); \
        ulong ch=(e&f)^(~e&g); ulong t1=h+S1+ch+K512[i]+(wi); \
        ulong S0=rotr64(a,28)^rotr64(a,34)^rotr64(a,39); \
        ulong maj=(a&b)^(a&c)^(b&c); ulong t2=S0+maj; \
        h=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2; }
    #define S5W(i) { int j=(i)&15; \
        ulong x=W[((i)-15)&15]; ulong s0=rotr64(x,1)^rotr64(x,8)^(x>>7); \
        x=W[((i)-2)&15]; ulong s1=rotr64(x,19)^rotr64(x,61)^(x>>6); \
        W[j]=W[j]+s0+W[((i)-7)&15]+s1; }
    S5R(0,W[0]);S5R(1,W[1]);S5R(2,W[2]);S5R(3,W[3]);
    S5R(4,W[4]);S5R(5,W[5]);S5R(6,W[6]);S5R(7,W[7]);
    S5R(8,W[8]);S5R(9,W[9]);S5R(10,W[10]);S5R(11,W[11]);
    S5R(12,W[12]);S5R(13,W[13]);S5R(14,W[14]);S5R(15,W[15]);
    S5W(16);S5R(16,W[0]);S5W(17);S5R(17,W[1]);S5W(18);S5R(18,W[2]);S5W(19);S5R(19,W[3]);
    S5W(20);S5R(20,W[4]);S5W(21);S5R(21,W[5]);S5W(22);S5R(22,W[6]);S5W(23);S5R(23,W[7]);
    S5W(24);S5R(24,W[8]);S5W(25);S5R(25,W[9]);S5W(26);S5R(26,W[10]);S5W(27);S5R(27,W[11]);
    S5W(28);S5R(28,W[12]);S5W(29);S5R(29,W[13]);S5W(30);S5R(30,W[14]);S5W(31);S5R(31,W[15]);
    S5W(32);S5R(32,W[0]);S5W(33);S5R(33,W[1]);S5W(34);S5R(34,W[2]);S5W(35);S5R(35,W[3]);
    S5W(36);S5R(36,W[4]);S5W(37);S5R(37,W[5]);S5W(38);S5R(38,W[6]);S5W(39);S5R(39,W[7]);
    S5W(40);S5R(40,W[8]);S5W(41);S5R(41,W[9]);S5W(42);S5R(42,W[10]);S5W(43);S5R(43,W[11]);
    S5W(44);S5R(44,W[12]);S5W(45);S5R(45,W[13]);S5W(46);S5R(46,W[14]);S5W(47);S5R(47,W[15]);
    S5W(48);S5R(48,W[0]);S5W(49);S5R(49,W[1]);S5W(50);S5R(50,W[2]);S5W(51);S5R(51,W[3]);
    S5W(52);S5R(52,W[4]);S5W(53);S5R(53,W[5]);S5W(54);S5R(54,W[6]);S5W(55);S5R(55,W[7]);
    S5W(56);S5R(56,W[8]);S5W(57);S5R(57,W[9]);S5W(58);S5R(58,W[10]);S5W(59);S5R(59,W[11]);
    S5W(60);S5R(60,W[12]);S5W(61);S5R(61,W[13]);S5W(62);S5R(62,W[14]);S5W(63);S5R(63,W[15]);
    S5W(64);S5R(64,W[0]);S5W(65);S5R(65,W[1]);S5W(66);S5R(66,W[2]);S5W(67);S5R(67,W[3]);
    S5W(68);S5R(68,W[4]);S5W(69);S5R(69,W[5]);S5W(70);S5R(70,W[6]);S5W(71);S5R(71,W[7]);
    S5W(72);S5R(72,W[8]);S5W(73);S5R(73,W[9]);S5W(74);S5R(74,W[10]);S5W(75);S5R(75,W[11]);
    S5W(76);S5R(76,W[12]);S5W(77);S5R(77,W[13]);S5W(78);S5R(78,W[14]);S5W(79);S5R(79,W[15]);
    #undef S5R
    #undef S5W
    state[0]+=a;state[1]+=b;state[2]+=c;state[3]+=d;
    state[4]+=e;state[5]+=f;state[6]+=g;state[7]+=h;
}

static void sha512_to_hex_lc(thread ulong *state, thread ulong *M) {
    for (int i = 0; i < 8; i++) {
        ulong s = state[i];
        uint b0=(s>>56)&0xff,b1=(s>>48)&0xff,b2=(s>>40)&0xff,b3=(s>>32)&0xff;
        uint b4=(s>>24)&0xff,b5=(s>>16)&0xff,b6=(s>>8)&0xff,b7=s&0xff;
        M[i*2]=(hex_byte_be64(b0)<<48)|(hex_byte_be64(b1)<<32)|(hex_byte_be64(b2)<<16)|hex_byte_be64(b3);
        M[i*2+1]=(hex_byte_be64(b4)<<48)|(hex_byte_be64(b5)<<32)|(hex_byte_be64(b6)<<16)|hex_byte_be64(b7);
    }
}

/* Compact probe-and-emit for SHA512 family. Probes compact table with first
 * 128 bits, writes full hash (nwords uint32 from state) on hit. */
static void __attribute__((noinline)) sha512_probe_emit(
    thread ulong *state, int nwords, uint word_idx, uint mask_idx,
    uint iter, uint max_iter,
    device const uint *compact_fp, device const uint *compact_idx,
    uint64_t compact_mask, uint max_probe, uint hash_data_count, uint max_hits,
    device const uint8_t *hash_data_buf, device const uint64_t *hash_data_off,
    device const uint64_t *overflow_keys, device const uint8_t *overflow_hashes,
    device const uint *overflow_offsets, uint overflow_count,
    device uint *hits, device atomic_uint *hit_count)
{
    ulong s0 = bswap64(state[0]), s1 = bswap64(state[1]);
    uint hx = (uint)s0, hy = (uint)(s0 >> 32);
    uint hz = (uint)s1, hw = (uint)(s1 >> 32);
    ulong key = (ulong(hy) << 32) | hx;
    uint fp = uint(key >> 32); if (fp == 0) fp = 1;
    ulong pos = (key ^ (key >> 32)) & compact_mask;
    bool found = false;
    for (uint p = 0; p < max_probe && !found; p++) {
        uint cfp = compact_fp[pos]; if (cfp == 0) break;
        if (cfp == fp) { uint idx = compact_idx[pos];
            if (idx < hash_data_count) {
                ulong off = hash_data_off[idx];
                device const uint *ref = (device const uint *)(hash_data_buf + off);
                if (hx == ref[0] && hy == ref[1] && hz == ref[2] && hw == ref[3]) found = true;
            } }
        pos = (pos + 1) & compact_mask;
    }
    if (!found && overflow_count > 0) {
        int lo = 0, hi2 = int(overflow_count) - 1;
        while (lo <= hi2 && !found) {
            int mid = (lo + hi2) / 2; ulong mkey = overflow_keys[mid];
            if (key < mkey) hi2 = mid - 1; else if (key > mkey) lo = mid + 1;
            else {
                for (int d = mid; d >= 0 && overflow_keys[d] == key && !found; d--) {
                    device const uint *r = (device const uint *)(overflow_hashes + overflow_offsets[d]);
                    if (hx == r[0] && hy == r[1] && hz == r[2] && hw == r[3]) found = true; }
                for (int d = mid+1; d < int(overflow_count) && overflow_keys[d] == key && !found; d++) {
                    device const uint *r = (device const uint *)(overflow_hashes + overflow_offsets[d]);
                    if (hx == r[0] && hy == r[1] && hz == r[2] && hw == r[3]) found = true; }
                break; } } }
    if (found) {
        uint slot = atomic_fetch_add_explicit(hit_count, 1u, memory_order_relaxed);
        if (slot < max_hits) {
            uint base = slot * HIT_STRIDE;
            hits[base] = word_idx; hits[base+1] = mask_idx; hits[base+2] = iter;
            for (int i = 0; i < nwords / 2; i++) {
                ulong sw = bswap64(state[i]);
                hits[base+3+i*2] = (uint)sw;
                hits[base+3+i*2+1] = (uint)(sw >> 32);
            }
            for (uint _z = 3 + nwords; _z < HIT_STRIDE; _z++) hits[base+_z] = 0;
        }
    }
}

kernel void sha512_unsalted_batch(
    device const uchar      *words       [[buffer(0)]],
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

    /* Load pre-padded block (16 uint64 = 128 bytes, little-endian from host) */
    device const ulong *src = (device const ulong *)(words + word_idx * 128);
    ulong M[16];
    for (int i = 0; i < 16; i++) M[i] = src[i];

    /* Fill mask positions -- LE bytes within uint64 words */
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
                int wi = i >> 3;
                int bi = (i & 7) << 3;
                M[wi] = (M[wi] & ~(0xFFUL << bi)) | ((ulong)ch << bi);
            }

            if (n_app > 0) {
                int total_len = (int)(M[15] & 0xFFFFFFFFUL);
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
                    M[wi] = (M[wi] & ~(0xFFUL << bi)) | ((ulong)ch << bi);
                }
            }
        } else {
            /* Append-only (brute-force): host pre-decomposes mask_start into
             * per-position base offsets in mask_base0/mask_base1. Kernel does
             * fast uint32 local decomposition and adds to base with carry. */
            int total_len = (int)(M[15] & 0xFFFFFFFFUL);
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
                int wi = pos >> 3;
                int bi = (pos & 7) << 3;
                M[wi] = (M[wi] & ~(0xFFUL << bi)) | ((ulong)ch << bi);
            }
        }
    }

    /* Save bit-length before byte-swap.
     * Host stores bitlen as uint32 at byte offset 120 = low 32 bits of M[15] (LE). */
    uint bitlen = (uint)(M[15] & 0xFFFFFFFFUL);

    /* Convert M[] from little-endian to big-endian for SHA512 */
    for (int i = 0; i < 16; i++) M[i] = bswap64(M[i]);

    /* Fix up padding: SHA512 stores 128-bit big-endian bit count at M[14..15] */
    M[14] = 0;
    M[15] = (ulong)bitlen;

    /* SHA512 compress */
    ulong state[8] = {
        0x6a09e667f3bcc908UL, 0xbb67ae8584caa73bUL,
        0x3c6ef372fe94f82bUL, 0xa54ff53a5f1d36f1UL,
        0x510e527fade682d1UL, 0x9b05688c2b3e6c1fUL,
        0x1f83d9abfb41bd6bUL, 0x5be0cd19137e2179UL
    };
    sha512_compress(state, M);

    uint max_iter = params.max_iter;

    for (uint iter = 1; iter <= max_iter; iter++) {
        sha512_probe_emit(state, 16, word_idx, mask_idx, iter, max_iter,
            compact_fp, compact_idx, params.compact_mask, params.max_probe,
            params.hash_data_count, params.max_hits,
            hash_data_buf, hash_data_off, overflow_keys, overflow_hashes,
            overflow_offsets, params.overflow_count, hits, hit_count);
        if (iter < max_iter) {
            /* Hex-encode 8 BE state words into M[0..15] (128 hex chars = full block) */
            sha512_to_hex_lc(state, M);
            /* First compress: data block */
            state[0] = 0x6a09e667f3bcc908UL; state[1] = 0xbb67ae8584caa73bUL;
            state[2] = 0x3c6ef372fe94f82bUL; state[3] = 0xa54ff53a5f1d36f1UL;
            state[4] = 0x510e527fade682d1UL; state[5] = 0x9b05688c2b3e6c1fUL;
            state[6] = 0x1f83d9abfb41bd6bUL; state[7] = 0x5be0cd19137e2179UL;
            sha512_compress(state, M);
            /* Second compress: padding block */
            M[0] = 0x8000000000000000UL;
            for (int i = 1; i < 15; i++) M[i] = 0;
            M[15] = 128 * 8;  /* 128 hex bytes = 1024 bits */
            sha512_compress(state, M);
        }
    }
}

/* SHA384 -- same compress as SHA512, different IV, truncated output (48 bytes) */
kernel void sha384_unsalted_batch(
    device const uchar      *words       [[buffer(0)]],
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

    device const ulong *src = (device const ulong *)(words + word_idx * 128);
    ulong M[16];
    for (int i = 0; i < 16; i++) M[i] = src[i];

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
                int wi = i >> 3;
                int bi = (i & 7) << 3;
                M[wi] = (M[wi] & ~(0xFFUL << bi)) | ((ulong)ch << bi);
            }
            if (n_app > 0) {
                int total_len = (int)(M[15] & 0xFFFFFFFFUL);
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
                    M[wi] = (M[wi] & ~(0xFFUL << bi)) | ((ulong)ch << bi);
                }
            }
        } else {
            /* Append-only (brute-force): host pre-decomposes mask_start into
             * per-position base offsets in mask_base0/mask_base1. Kernel does
             * fast uint32 local decomposition and adds to base with carry. */
            int total_len = (int)(M[15] & 0xFFFFFFFFUL);
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
                int wi = pos >> 3;
                int bi = (pos & 7) << 3;
                M[wi] = (M[wi] & ~(0xFFUL << bi)) | ((ulong)ch << bi);
            }
        }
    }

    uint bitlen = (uint)(M[15] & 0xFFFFFFFFUL);
    for (int i = 0; i < 16; i++) M[i] = bswap64(M[i]);
    M[14] = 0;
    M[15] = (ulong)bitlen;

    /* SHA384 IV */
    ulong state[8] = {
        0xcbbb9d5dc1059ed8UL, 0x629a292a367cd507UL,
        0x9159015a3070dd17UL, 0x152fecd8f70e5939UL,
        0x67332667ffc00b31UL, 0x8eb44a8768581511UL,
        0xdb0c2e0d64f98fa7UL, 0x47b5481dbefa4fa4UL
    };
    sha512_compress(state, M);

    uint max_iter = params.max_iter;

    for (uint iter = 1; iter <= max_iter; iter++) {
        sha512_probe_emit(state, 12, word_idx, mask_idx, iter, max_iter,
            compact_fp, compact_idx, params.compact_mask, params.max_probe,
            params.hash_data_count, params.max_hits,
            hash_data_buf, hash_data_off, overflow_keys, overflow_hashes,
            overflow_offsets, params.overflow_count, hits, hit_count);
        if (iter < max_iter) {
            /* Hex-encode 6 BE state words into M[0..11] (96 hex chars).
             * 96 bytes fits in one SHA512 block (< 112 byte limit). */
            for (int i = 0; i < 6; i++) {
                ulong s = state[i];
                uint b0 = (s >> 56) & 0xff, b1 = (s >> 48) & 0xff;
                uint b2 = (s >> 40) & 0xff, b3 = (s >> 32) & 0xff;
                uint b4 = (s >> 24) & 0xff, b5 = (s >> 16) & 0xff;
                uint b6 = (s >> 8)  & 0xff, b7 = s & 0xff;
                M[i*2]   = (hex_byte_be64(b0) << 48) | (hex_byte_be64(b1) << 32)
                          | (hex_byte_be64(b2) << 16) | hex_byte_be64(b3);
                M[i*2+1] = (hex_byte_be64(b4) << 48) | (hex_byte_be64(b5) << 32)
                          | (hex_byte_be64(b6) << 16) | hex_byte_be64(b7);
            }
            /* Padding: 0x80 at byte 96, zeros, length at M[14..15] */
            M[12] = 0x8000000000000000UL;
            M[13] = 0;
            M[14] = 0;
            M[15] = 96 * 8;  /* 96 hex bytes = 768 bits */
            /* Reset to SHA384 IV */
            state[0] = 0xcbbb9d5dc1059ed8UL; state[1] = 0x629a292a367cd507UL;
            state[2] = 0x9159015a3070dd17UL; state[3] = 0x152fecd8f70e5939UL;
            state[4] = 0x67332667ffc00b31UL; state[5] = 0x8eb44a8768581511UL;
            state[6] = 0xdb0c2e0d64f98fa7UL; state[7] = 0x47b5481dbefa4fa4UL;
            sha512_compress(state, M);
        }
    }
}
