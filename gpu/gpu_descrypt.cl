/* gpu_descrypt.cl — DES crypt GPU kernel
 *
 * Uses precomputed SPtrans[8][64] tables (combined S-box + P-permutation)
 * copied to __local memory for fast divergent access on NVIDIA.
 * Key schedule uses bit extraction (PC-1/PC-2 permutation tables).
 * Compact table stores pre-FP (IP-applied) hash values.
 */

/* Combined S-box + P-permutation tables */
__constant uint SPtrans[8][64] = {
    { 0x00808200u,0x00000000u,0x00008000u,0x00808202u,0x00808002u,0x00008202u,0x00000002u,0x00008000u,
      0x00000200u,0x00808200u,0x00808202u,0x00000200u,0x00800202u,0x00808002u,0x00800000u,0x00000002u,
      0x00000202u,0x00800200u,0x00800200u,0x00008200u,0x00008200u,0x00808000u,0x00808000u,0x00800202u,
      0x00008002u,0x00800002u,0x00800002u,0x00008002u,0x00000000u,0x00000202u,0x00008202u,0x00800000u,
      0x00008000u,0x00808202u,0x00000002u,0x00808000u,0x00808200u,0x00800000u,0x00800000u,0x00000200u,
      0x00808002u,0x00008000u,0x00008200u,0x00800002u,0x00000200u,0x00000002u,0x00800202u,0x00008202u,
      0x00808202u,0x00008002u,0x00808000u,0x00800202u,0x00800002u,0x00000202u,0x00008202u,0x00808200u,
      0x00000202u,0x00800200u,0x00800200u,0x00000000u,0x00008002u,0x00008200u,0x00000000u,0x00808002u },
    { 0x40084010u,0x40004000u,0x00004000u,0x00084010u,0x00080000u,0x00000010u,0x40080010u,0x40004010u,
      0x40000010u,0x40084010u,0x40084000u,0x40000000u,0x40004000u,0x00080000u,0x00000010u,0x40080010u,
      0x00084000u,0x00080010u,0x40004010u,0x00000000u,0x40000000u,0x00004000u,0x00084010u,0x40080000u,
      0x00080010u,0x40000010u,0x00000000u,0x00084000u,0x00004010u,0x40084000u,0x40080000u,0x00004010u,
      0x00000000u,0x00084010u,0x40080010u,0x00080000u,0x40004010u,0x40080000u,0x40084000u,0x00004000u,
      0x40080000u,0x40004000u,0x00000010u,0x40084010u,0x00084010u,0x00000010u,0x00004000u,0x40000000u,
      0x00004010u,0x40084000u,0x00080000u,0x40000010u,0x00080010u,0x40004010u,0x40000010u,0x00080010u,
      0x00084000u,0x00000000u,0x40004000u,0x00004010u,0x40000000u,0x40080010u,0x40084010u,0x00084000u },
    { 0x00000104u,0x04010100u,0x00000000u,0x04010004u,0x04000100u,0x00000000u,0x00010104u,0x04000100u,
      0x00010004u,0x04000004u,0x04000004u,0x00010000u,0x04010104u,0x00010004u,0x04010000u,0x00000104u,
      0x04000000u,0x00000004u,0x04010100u,0x00000100u,0x00010100u,0x04010000u,0x04010004u,0x00010104u,
      0x04000104u,0x00010100u,0x00010000u,0x04000104u,0x00000004u,0x04010104u,0x00000100u,0x04000000u,
      0x04010100u,0x04000000u,0x00010004u,0x00000104u,0x00010000u,0x04010100u,0x04000100u,0x00000000u,
      0x00000100u,0x00010004u,0x04010104u,0x04000100u,0x04000004u,0x00000100u,0x00000000u,0x04010004u,
      0x04000104u,0x00010000u,0x04000000u,0x04010104u,0x00000004u,0x00010104u,0x00010100u,0x04000004u,
      0x04010000u,0x04000104u,0x00000104u,0x04010000u,0x00010104u,0x00000004u,0x04010004u,0x00010100u },
    { 0x80401000u,0x80001040u,0x80001040u,0x00000040u,0x00401040u,0x80400040u,0x80400000u,0x80001000u,
      0x00000000u,0x00401000u,0x00401000u,0x80401040u,0x80000040u,0x00000000u,0x00400040u,0x80400000u,
      0x80000000u,0x00001000u,0x00400000u,0x80401000u,0x00000040u,0x00400000u,0x80001000u,0x00001040u,
      0x80400040u,0x80000000u,0x00001040u,0x00400040u,0x00001000u,0x00401040u,0x80401040u,0x80000040u,
      0x00400040u,0x80400000u,0x00401000u,0x80401040u,0x80000040u,0x00000000u,0x00000000u,0x00401000u,
      0x00001040u,0x00400040u,0x80400040u,0x80000000u,0x80401000u,0x80001040u,0x80001040u,0x00000040u,
      0x80401040u,0x80000040u,0x80000000u,0x00001000u,0x80400000u,0x80001000u,0x00401040u,0x80400040u,
      0x80001000u,0x00001040u,0x00400000u,0x80401000u,0x00000040u,0x00400000u,0x00001000u,0x00401040u },
    { 0x00000080u,0x01040080u,0x01040000u,0x21000080u,0x00040000u,0x00000080u,0x20000000u,0x01040000u,
      0x20040080u,0x00040000u,0x01000080u,0x20040080u,0x21000080u,0x21040000u,0x00040080u,0x20000000u,
      0x01000000u,0x20040000u,0x20040000u,0x00000000u,0x20000080u,0x21040080u,0x21040080u,0x01000080u,
      0x21040000u,0x20000080u,0x00000000u,0x21000000u,0x01040080u,0x01000000u,0x21000000u,0x00040080u,
      0x00040000u,0x21000080u,0x00000080u,0x01000000u,0x20000000u,0x01040000u,0x21000080u,0x20040080u,
      0x01000080u,0x20000000u,0x21040000u,0x01040080u,0x20040080u,0x00000080u,0x01000000u,0x21040000u,
      0x21040080u,0x00040080u,0x21000000u,0x21040080u,0x01040000u,0x00000000u,0x20040000u,0x21000000u,
      0x00040080u,0x01000080u,0x20000080u,0x00040000u,0x00000000u,0x20040000u,0x01040080u,0x20000080u },
    { 0x10000008u,0x10200000u,0x00002000u,0x10202008u,0x10200000u,0x00000008u,0x10202008u,0x00200000u,
      0x10002000u,0x00202008u,0x00200000u,0x10000008u,0x00200008u,0x10002000u,0x10000000u,0x00002008u,
      0x00000000u,0x00200008u,0x10002008u,0x00002000u,0x00202000u,0x10002008u,0x00000008u,0x10200008u,
      0x10200008u,0x00000000u,0x00202008u,0x10202000u,0x00002008u,0x00202000u,0x10202000u,0x10000000u,
      0x10002000u,0x00000008u,0x10200008u,0x00202000u,0x10202008u,0x00200000u,0x00002008u,0x10000008u,
      0x00200000u,0x10002000u,0x10000000u,0x00002008u,0x10000008u,0x10202008u,0x00202000u,0x10200000u,
      0x00202008u,0x10202000u,0x00000000u,0x10200008u,0x00000008u,0x00002000u,0x10200000u,0x00202008u,
      0x00002000u,0x00200008u,0x10002008u,0x00000000u,0x10202000u,0x10000000u,0x00200008u,0x10002008u },
    { 0x00100000u,0x02100001u,0x02000401u,0x00000000u,0x00000400u,0x02000401u,0x00100401u,0x02100400u,
      0x02100401u,0x00100000u,0x00000000u,0x02000001u,0x00000001u,0x02000000u,0x02100001u,0x00000401u,
      0x02000400u,0x00100401u,0x00100001u,0x02000400u,0x02000001u,0x02100000u,0x02100400u,0x00100001u,
      0x02100000u,0x00000400u,0x00000401u,0x02100401u,0x00100400u,0x00000001u,0x02000000u,0x00100400u,
      0x02000000u,0x00100400u,0x00100000u,0x02000401u,0x02000401u,0x02100001u,0x02100001u,0x00000001u,
      0x00100001u,0x02000000u,0x02000400u,0x00100000u,0x02100400u,0x00000401u,0x00100401u,0x02100400u,
      0x00000401u,0x02000001u,0x02100401u,0x02100000u,0x00100400u,0x00000000u,0x00000001u,0x02100401u,
      0x00000000u,0x00100401u,0x02100000u,0x00000400u,0x02000001u,0x02000400u,0x00000400u,0x00100001u },
    { 0x08000820u,0x00000800u,0x00020000u,0x08020820u,0x08000000u,0x08000820u,0x00000020u,0x08000000u,
      0x00020020u,0x08020000u,0x08020820u,0x00020800u,0x08020800u,0x00020820u,0x00000800u,0x00000020u,
      0x08020000u,0x08000020u,0x08000800u,0x00000820u,0x00020800u,0x00020020u,0x08020020u,0x08020800u,
      0x00000820u,0x00000000u,0x00000000u,0x08020020u,0x08000020u,0x08000800u,0x00020820u,0x00020000u,
      0x00020820u,0x00020000u,0x08020800u,0x00000800u,0x00000020u,0x08020020u,0x00000800u,0x00020820u,
      0x08000800u,0x00000020u,0x08000020u,0x08020000u,0x08020020u,0x08000000u,0x00020000u,0x08000820u,
      0x00000000u,0x08020820u,0x00020020u,0x08000020u,0x08020000u,0x08000800u,0x08000820u,0x00000000u,
      0x08020820u,0x00020800u,0x00020800u,0x00000820u,0x00000820u,0x00020020u,0x08000000u,0x08020800u }
};

/* PC-1/PC-2/key schedule tables */
__constant uchar pc1_c[28] = {
    57,49,41,33,25,17, 9, 1,58,50,42,34,26,18,
    10, 2,59,51,43,35,27,19,11, 3,60,52,44,36 };
__constant uchar pc1_d[28] = {
    63,55,47,39,31,23,15, 7,62,54,46,38,30,22,
    14, 6,61,53,45,37,29,21,13, 5,28,20,12, 4 };
__constant uchar pc2[48] = {
    14,17,11,24, 1, 5, 3,28,15, 6,21,10,23,19,12, 4,26, 8,16, 7,27,20,13, 2,
    41,52,31,37,47,55,30,40,51,45,33,48,44,49,39,56,34,53,46,42,50,36,29,32 };
__constant uchar key_shifts[16] = {1,1,2,2,2,2,2,2,1,2,2,2,2,2,2,1};

uint gb64(uint hi, uint lo, uint b) {
    return (b <= 32) ? ((hi >> (32 - b)) & 1u) : ((lo >> (64 - b)) & 1u);
}
uint gb28(uint v, uint b) { return (v >> (28 - b)) & 1u; }
uint a2b(uint ch) {
    if (ch >= 'a') return ch - 'a' + 38;
    if (ch >= 'A') return ch - 'A' + 12;
    if (ch >= '.') return ch - '.';
    return 0;
}
uint compute_saltbits(uint salt) {
    uint sb = 0;
    for (int i = 0; i < 12; i++) sb |= ((salt >> i) & 1u) << (23 - i);
    return sb;
}

void des_key_schedule(uint khi, uint klo, uint *ekl, uint *ekr) {
    uint c = 0, d = 0;
    for (int i = 0; i < 28; i++) {
        c |= gb64(khi, klo, pc1_c[i]) << (27 - i);
        d |= gb64(khi, klo, pc1_d[i]) << (27 - i);
    }
    uint total_shift = 0;
    for (int rnd = 0; rnd < 16; rnd++) {
        total_shift += key_shifts[rnd];
        uint tc = ((c << total_shift) | (c >> (28 - total_shift))) & 0x0FFFFFFFu;
        uint td = ((d << total_shift) | (d >> (28 - total_shift))) & 0x0FFFFFFFu;
        uint kl = 0, kr = 0;
        for (int i = 0; i < 24; i++) {
            uint b = pc2[i];
            kl |= ((b <= 28) ? gb28(tc, b) : gb28(td, b - 28)) << (23 - i);
        }
        for (int i = 0; i < 24; i++) {
            uint b = pc2[24 + i];
            kr |= ((b <= 28) ? gb28(tc, b) : gb28(td, b - 28)) << (23 - i);
        }
        ekl[rnd] = kl; ekr[rnd] = kr;
    }
}

/* Feistel round using __local SPtrans */
uint des_f(__local uint (*s_SP)[64], uint r, uint kl, uint kr, uint saltbits) {
    uint r48l = ((r & 0x00000001u) << 23) | ((r & 0xf8000000u) >> 9) |
                ((r & 0x1f800000u) >> 11) | ((r & 0x01f80000u) >> 13) | ((r & 0x001f8000u) >> 15);
    uint r48r = ((r & 0x0001f800u) << 7) | ((r & 0x00001f80u) << 5) |
                ((r & 0x000001f8u) << 3) | ((r & 0x0000001fu) << 1) | ((r & 0x80000000u) >> 31);
    uint f = (r48l ^ r48r) & saltbits;
    r48l ^= f ^ kl; r48r ^= f ^ kr;
    return s_SP[0][(r48l >> 18) & 0x3fu] | s_SP[1][(r48l >> 12) & 0x3fu]
         | s_SP[2][(r48l >>  6) & 0x3fu] | s_SP[3][ r48l        & 0x3fu]
         | s_SP[4][(r48r >> 18) & 0x3fu] | s_SP[5][(r48r >> 12) & 0x3fu]
         | s_SP[6][(r48r >>  6) & 0x3fu] | s_SP[7][ r48r        & 0x3fu];
}

__kernel void descrypt_batch(
    __global const uchar *hexhashes, __global const ushort *hexlens,
    __global const uchar *salts, __global const uint *salt_offsets, __global const ushort *salt_lens,
    __global const uint *compact_fp, __global const uint *compact_idx,
    __global const OCLParams *params_buf,
    __global const uchar *hash_data_buf, __global const ulong *hash_data_off,
    __global const ushort *hash_data_len,
    __global uint *hits, __global volatile uint *hit_count,
    __global const ulong *overflow_keys, __global const uchar *overflow_hashes,
    __global const uint *overflow_offsets, __global const ushort *overflow_lengths)
{
    OCLParams params = *params_buf;
    uint gid = get_global_id(0);
    uint lid = get_local_id(0);
    uint lsz = get_local_size(0);
    /* Copy SPtrans to __local — must happen before any early return
     * so all threads in the workgroup participate in the barrier */
    __local uint s_SP[8][64];
    for (uint i = lid; i < 512; i += lsz) {
        uint box = i >> 6, idx = i & 63;
        s_SP[box][idx] = SPtrans[box][idx];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    uint word_idx = gid / params.num_salts;
    uint salt_idx = params.salt_start + (gid % params.num_salts);
    if (word_idx >= params.num_words) return;

    int plen = hexlens[word_idx];
    if (plen > 8) plen = 8;
    __global const uchar *pass = hexhashes + word_idx * 256;
    uchar kb[8];
    for (int i = 0; i < 8; i++) kb[i] = (i < plen) ? (pass[i] << 1) : 0;
    uint khi = ((uint)kb[0] << 24) | ((uint)kb[1] << 16) | ((uint)kb[2] << 8) | kb[3];
    uint klo = ((uint)kb[4] << 24) | ((uint)kb[5] << 16) | ((uint)kb[6] << 8) | kb[7];

    uint ekl[16], ekr[16];
    des_key_schedule(khi, klo, ekl, ekr);

    uint soff = salt_offsets[salt_idx];
    uint salt = a2b((uint)salts[soff]) | (a2b((uint)salts[soff + 1]) << 6);
    uint saltbits = compute_saltbits(salt);

    uint l = 0, r = 0;
    for (int iter = 0; iter < 25; iter++) {
        uint fv;
        fv = des_f(s_SP, r, ekl[ 0], ekr[ 0], saltbits) ^ l; l = r; r = fv;
        fv = des_f(s_SP, r, ekl[ 1], ekr[ 1], saltbits) ^ l; l = r; r = fv;
        fv = des_f(s_SP, r, ekl[ 2], ekr[ 2], saltbits) ^ l; l = r; r = fv;
        fv = des_f(s_SP, r, ekl[ 3], ekr[ 3], saltbits) ^ l; l = r; r = fv;
        fv = des_f(s_SP, r, ekl[ 4], ekr[ 4], saltbits) ^ l; l = r; r = fv;
        fv = des_f(s_SP, r, ekl[ 5], ekr[ 5], saltbits) ^ l; l = r; r = fv;
        fv = des_f(s_SP, r, ekl[ 6], ekr[ 6], saltbits) ^ l; l = r; r = fv;
        fv = des_f(s_SP, r, ekl[ 7], ekr[ 7], saltbits) ^ l; l = r; r = fv;
        fv = des_f(s_SP, r, ekl[ 8], ekr[ 8], saltbits) ^ l; l = r; r = fv;
        fv = des_f(s_SP, r, ekl[ 9], ekr[ 9], saltbits) ^ l; l = r; r = fv;
        fv = des_f(s_SP, r, ekl[10], ekr[10], saltbits) ^ l; l = r; r = fv;
        fv = des_f(s_SP, r, ekl[11], ekr[11], saltbits) ^ l; l = r; r = fv;
        fv = des_f(s_SP, r, ekl[12], ekr[12], saltbits) ^ l; l = r; r = fv;
        fv = des_f(s_SP, r, ekl[13], ekr[13], saltbits) ^ l; l = r; r = fv;
        fv = des_f(s_SP, r, ekl[14], ekr[14], saltbits) ^ l; l = r; r = fv;
        fv = des_f(s_SP, r, ekl[15], ekr[15], saltbits) ^ l; l = r; r = fv;
        uint tmp = l; l = r; r = tmp;
    }

    if (probe_compact(l, r, 0, 0, compact_fp, compact_idx,
                      params.compact_mask, params.max_probe, params.hash_data_count,
                      hash_data_buf, hash_data_off,
                      overflow_keys, overflow_hashes, overflow_offsets, params.overflow_count)) {
        uint slot = atomic_add(hit_count, 1u);
        if (slot < params.max_hits) {
            uint base = slot * 7;
            hits[base]   = word_idx;
            hits[base+1] = salt_idx;
            hits[base+2] = 1;
            hits[base+3] = l;
            hits[base+4] = r;
            hits[base+5] = 0;
            hits[base+6] = 0;
            mem_fence(CLK_GLOBAL_MEM_FENCE);
        }
    }
}
