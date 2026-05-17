/*
 * $Revision: $
 * $Log: $
 *
 * Hand-port of gpu/gpu_descrypt_core.cl (RCS rev 1.1) to Metal.
 *
 * Phase 2d.9a (2026-05-16): cl2metal.py marked UNSUITABLE for DESCRYPT
 * per architect Task #293 Option A. The OpenCL source uses a 520-LOC
 * __constant DES SP-table + bespoke per-thread Feistel that requires
 * hand-curated address-space tags; the translator's autotransforms
 * would either misclassify the SPtrans table or emit wrong qualifiers
 * on the descrypt_des_f / descrypt_des_key_schedule helpers (whose
 * args carry only primitive scalars + thread-local arrays). Manually
 * porting is cheaper than extending the translator for a single op.
 *
 * Hand-port discipline (mirrors metal_md5crypt_core.metal byte for
 * byte except where DES algorithm requires divergence):
 *   __constant -> constant
 *   __global   -> device
 *   __private  -> thread
 *   typedef struct ... T;  ->  struct T { ... };
 *   T *st      -> thread T &st
 *   uint *ek_l (private array) -> thread uint *ek_l
 *
 * The SP-table contents are copied verbatim from gpu/gpu_descrypt_core.cl
 * (rev 1.1) without modification -- only the address-space tag changes
 * (__constant -> constant). Same for the PC-1/PC-2/key-shift constants.
 *
 * NOTE: This is NOT cl2metal codegen output. There is no companion YAML
 * overlay. Future updates to gpu_descrypt_core.cl require parallel hand-
 * edits here (and the rev-1.1 source comment block above should be
 * updated to the new OpenCL rev number when that happens).
 *
 * DESCRYPT / Unix DES crypt(3) "old-style" semantics (see OpenCL twin
 * for the full design memo). Single algo_mode (7); HASH_WORDS=4 with
 * h[0..1] = pre-FP (l, r) and h[2..3] = zero-padded to match the host
 * compact-table 16-byte layout (4 il + 4 ir + 8 zero pad).
 *
 * Hit replay: gpu/gpujob_metal.m's JOB_DESCRYPT arm calls
 * metal_des_reconstruct(curin.i[0], curin.i[1], salt_bytes, desbuf) to
 * reconstruct the 13-char crypt(3) hash; probes JudyJ[JOB_DESCRYPT] for
 * the line and emits via prfound. Mirrors gpu/gpujob_opencl.c
 * JOB_DESCRYPT arm at line 2204-2232 byte-for-byte.
 */

#ifndef HASH_WORDS
#define HASH_WORDS 4
#endif
#ifndef HASH_BLOCK_BYTES
#define HASH_BLOCK_BYTES 64
#endif

/* DES SP-tables (combined S-box + P-permutation), copied verbatim from
 * gpu_descrypt_core.cl rev 1.1 (only address-space tag changed:
 * __constant -> constant). Metal does NOT support workgroup-shared
 * __local memory in the shared template scaffold (would require
 * barrier inside template_finalize); reading from `constant` directly
 * is the Metal idiom for per-thread divergent table access. */
constant uint METAL_DESCRYPT_SPtrans[8][64] = {
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

/* PC-1/PC-2/key-shift tables (copied verbatim from gpu_descrypt_core.cl
 * rev 1.1; address-space tag is the only change). */
constant uchar METAL_DESCRYPT_pc1_c[28] = {
    57,49,41,33,25,17, 9, 1,58,50,42,34,26,18,
    10, 2,59,51,43,35,27,19,11, 3,60,52,44,36 };
constant uchar METAL_DESCRYPT_pc1_d[28] = {
    63,55,47,39,31,23,15, 7,62,54,46,38,30,22,
    14, 6,61,53,45,37,29,21,13, 5,28,20,12, 4 };
constant uchar METAL_DESCRYPT_pc2[48] = {
    14,17,11,24, 1, 5, 3,28,15, 6,21,10,23,19,12, 4,26, 8,16, 7,27,20,13, 2,
    41,52,31,37,47,55,30,40,51,45,33,48,44,49,39,56,34,53,46,42,50,36,29,32 };
constant uchar METAL_DESCRYPT_key_shifts[16] = {1,1,2,2,2,2,2,2,1,2,2,2,2,2,2,1};

/* Bit extraction helpers (ported byte-for-byte from gpu_descrypt_core.cl
 * rev 1.1). The METAL_DESCRYPT_ prefix avoids any symbol collision with
 * future Metal cores that might want similarly-named helpers. */
static inline uint metal_descrypt_gb64(uint hi, uint lo, uint b) {
    return (b <= 32) ? ((hi >> (32 - b)) & 1u) : ((lo >> (64 - b)) & 1u);
}
static inline uint metal_descrypt_gb28(uint v, uint b) { return (v >> (28 - b)) & 1u; }
static inline uint metal_descrypt_a2b(uint ch) {
    if (ch >= 'a') return ch - 'a' + 38;
    if (ch >= 'A') return ch - 'A' + 12;
    if (ch >= '.') return ch - '.';
    return 0;
}
static inline uint metal_descrypt_compute_saltbits(uint salt) {
    uint sb = 0;
    for (int i = 0; i < 12; i++) sb |= ((salt >> i) & 1u) << (23 - i);
    return sb;
}

/* Build 16 round keys (ek_l[0..15], ek_r[0..15]) from the 64-bit key
 * (khi, klo). Mirrors gpu_descrypt_core.cl descrypt_des_key_schedule
 * byte-for-byte (PC-1, 16 left-rotations with cumulative shift counts,
 * PC-2). All array params are thread-local (the caller passes private
 * uint[16] arrays). */
static inline void metal_descrypt_des_key_schedule(uint khi, uint klo,
                                                   thread uint *ek_l,
                                                   thread uint *ek_r)
{
    uint c = 0, d = 0;
    for (int i = 0; i < 28; i++) {
        c |= metal_descrypt_gb64(khi, klo, METAL_DESCRYPT_pc1_c[i]) << (27 - i);
        d |= metal_descrypt_gb64(khi, klo, METAL_DESCRYPT_pc1_d[i]) << (27 - i);
    }
    uint total_shift = 0;
    for (int rnd = 0; rnd < 16; rnd++) {
        total_shift += METAL_DESCRYPT_key_shifts[rnd];
        uint tc = ((c << total_shift) | (c >> (28 - total_shift))) & 0x0FFFFFFFu;
        uint td = ((d << total_shift) | (d >> (28 - total_shift))) & 0x0FFFFFFFu;
        uint kl = 0, kr = 0;
        for (int i = 0; i < 24; i++) {
            uint b = METAL_DESCRYPT_pc2[i];
            kl |= ((b <= 28) ? metal_descrypt_gb28(tc, b)
                             : metal_descrypt_gb28(td, b - 28)) << (23 - i);
        }
        for (int i = 0; i < 24; i++) {
            uint b = METAL_DESCRYPT_pc2[24 + i];
            kr |= ((b <= 28) ? metal_descrypt_gb28(tc, b)
                             : metal_descrypt_gb28(td, b - 28)) << (23 - i);
        }
        ek_l[rnd] = kl;
        ek_r[rnd] = kr;
    }
}

/* DES Feistel round. r' = E(r) salted XOR with key, then 8-way SP-table
 * lookup. Reads from `constant METAL_DESCRYPT_SPtrans` (the slab gpu_
 * descrypt.cl uses __local cache; the template path skips the workgroup-
 * shared cache to keep the shared template scaffold barrier-free). */
static inline uint metal_descrypt_des_f(uint r, uint kl, uint kr, uint saltbits) {
    uint r48l = ((r & 0x00000001u) << 23) | ((r & 0xf8000000u) >> 9) |
                ((r & 0x1f800000u) >> 11) | ((r & 0x01f80000u) >> 13) |
                ((r & 0x001f8000u) >> 15);
    uint r48r = ((r & 0x0001f800u) <<  7) | ((r & 0x00001f80u) <<  5) |
                ((r & 0x000001f8u) <<  3) | ((r & 0x0000001fu) <<  1) |
                ((r & 0x80000000u) >> 31);
    uint f = (r48l ^ r48r) & saltbits;
    r48l ^= f ^ kl;
    r48r ^= f ^ kr;
    return METAL_DESCRYPT_SPtrans[0][(r48l >> 18) & 0x3fu]
         | METAL_DESCRYPT_SPtrans[1][(r48l >> 12) & 0x3fu]
         | METAL_DESCRYPT_SPtrans[2][(r48l >>  6) & 0x3fu]
         | METAL_DESCRYPT_SPtrans[3][ r48l        & 0x3fu]
         | METAL_DESCRYPT_SPtrans[4][(r48r >> 18) & 0x3fu]
         | METAL_DESCRYPT_SPtrans[5][(r48r >> 12) & 0x3fu]
         | METAL_DESCRYPT_SPtrans[6][(r48r >>  6) & 0x3fu]
         | METAL_DESCRYPT_SPtrans[7][ r48r        & 0x3fu];
}

/* Per-lane state struct. DES emits a pre-FP (l, r) pair = 64 bits. We
 * carry it in h[0..1]; h[2..3] are zero-padded so probe_compact_idx
 * sees the same 16-byte layout the host's compact-table loader stores
 * (mdxfind.c:40433-40435: 4 il + 4 ir + 8 zero pad). HASH_WORDS=4 stays
 * the canonical width for this template instantiation. */
struct template_state {
    uint h[HASH_WORDS];
};

/* template_init: zero the state. DES has no IV; (l, r) start at 0 in the
 * 25-iter Feistel loop. */
static inline void template_init(thread template_state &st) {
    st.h[0] = 0u;
    st.h[1] = 0u;
    st.h[2] = 0u;
    st.h[3] = 0u;
}

/* template_transform: stub for interface symmetry. DESCRYPT's
 * template_finalize manages the full DES state inline -- never routes
 * through this. Provided for completeness (matches PHPBB3 / MD5CRYPT
 * pattern). */
static inline void template_transform(thread template_state &st,
                                      thread const uchar *block)
{
    (void)st;
    (void)block;
}

/* template_finalize: full DESCRYPT chain.
 *
 * Step 1: build 8-byte DES key from data[0..min(plen,8)) with KEY left-
 *   shift (mirrors crypt-des.c:626-630 byte-for-byte; bytes past first
 *   NUL pad with zero, but mdxfind's input is fixed-length post-rule so
 *   we just use min(plen, 8) bytes shifted left by 1 and zero-pad the
 *   rest).
 *
 * Step 2: decode 2-char phpitoa64 salt from salt_bytes[0..2) into a
 *   12-bit salt, then expand via metal_descrypt_compute_saltbits to
 *   24-bit saltbits.
 *
 * Step 3: run 25 DES iterations, each with 16 Feistel rounds + final
 *   swap (mirrors gpu_descrypt_core.cl rev 1.1 byte-for-byte).
 *
 * Step 4: install pre-FP (l, r) into st.h[0..1]; zero h[2..3] for the
 *   compact-table probe.
 *
 * algo_mode: DESCRYPT has only one mode (7). The arg is unused; kept
 * for interface symmetry with the salted-template signature. */
static inline void template_finalize(thread template_state &st,
                                     device const uchar *data,
                                     int len,
                                     device const uchar *salt_bytes,
                                     uint salt_len,
                                     uint algo_mode)
{
    (void)algo_mode;
    (void)salt_len;

    /* Defensive cap: standard DES uses only the first 8 bytes of the
     * key (host-side rules-engine pack site already clamps for the
     * synthetic no-rule pass; clamp here too so masked / rule-extended
     * outputs bigger than 8 bytes silently truncate). */
    int plen = len;
    if (plen > 8) plen = 8;

    /* Step 1: build 8-byte DES key buffer. byte = (data[i] << 1)
     * with bytes past min(plen,8) zero-padded. */
    uchar kb[8];
    for (int i = 0; i < 8; i++) {
        kb[i] = (i < plen) ? (uchar)((uint)data[i] << 1) : (uchar)0u;
    }

    /* Pack to two 32-bit halves (BE) for des_key_schedule. */
    uint khi = ((uint)kb[0] << 24) | ((uint)kb[1] << 16)
             | ((uint)kb[2] <<  8) |  (uint)kb[3];
    uint klo = ((uint)kb[4] << 24) | ((uint)kb[5] << 16)
             | ((uint)kb[6] <<  8) |  (uint)kb[7];

    /* Step 2: 16 round keys via PC-1, left-rotations, PC-2. */
    uint ek_l[16], ek_r[16];
    metal_descrypt_des_key_schedule(khi, klo, ek_l, ek_r);

    /* Step 3a: decode 2-char phpitoa64 salt + expand to 24-bit saltbits.
     * Defensive: assume salt_len == 2 (guaranteed by the host's
     * gpu_pack_salts filter which skips saltlen != 2 for JOB_DESCRYPT). */
    uint salt = metal_descrypt_a2b((uint)salt_bytes[0])
              | (metal_descrypt_a2b((uint)salt_bytes[1]) << 6);
    uint saltbits = metal_descrypt_compute_saltbits(salt);

    /* Step 3b: 25 DES iterations of (l, r) -> 16 rounds + final swap.
     * Mirrors gpu_descrypt_core.cl rev 1.1 byte-for-byte. */
    uint l = 0u, r = 0u;
    for (int iter = 0; iter < 25; iter++) {
        uint fv;
        fv = metal_descrypt_des_f(r, ek_l[ 0], ek_r[ 0], saltbits) ^ l; l = r; r = fv;
        fv = metal_descrypt_des_f(r, ek_l[ 1], ek_r[ 1], saltbits) ^ l; l = r; r = fv;
        fv = metal_descrypt_des_f(r, ek_l[ 2], ek_r[ 2], saltbits) ^ l; l = r; r = fv;
        fv = metal_descrypt_des_f(r, ek_l[ 3], ek_r[ 3], saltbits) ^ l; l = r; r = fv;
        fv = metal_descrypt_des_f(r, ek_l[ 4], ek_r[ 4], saltbits) ^ l; l = r; r = fv;
        fv = metal_descrypt_des_f(r, ek_l[ 5], ek_r[ 5], saltbits) ^ l; l = r; r = fv;
        fv = metal_descrypt_des_f(r, ek_l[ 6], ek_r[ 6], saltbits) ^ l; l = r; r = fv;
        fv = metal_descrypt_des_f(r, ek_l[ 7], ek_r[ 7], saltbits) ^ l; l = r; r = fv;
        fv = metal_descrypt_des_f(r, ek_l[ 8], ek_r[ 8], saltbits) ^ l; l = r; r = fv;
        fv = metal_descrypt_des_f(r, ek_l[ 9], ek_r[ 9], saltbits) ^ l; l = r; r = fv;
        fv = metal_descrypt_des_f(r, ek_l[10], ek_r[10], saltbits) ^ l; l = r; r = fv;
        fv = metal_descrypt_des_f(r, ek_l[11], ek_r[11], saltbits) ^ l; l = r; r = fv;
        fv = metal_descrypt_des_f(r, ek_l[12], ek_r[12], saltbits) ^ l; l = r; r = fv;
        fv = metal_descrypt_des_f(r, ek_l[13], ek_r[13], saltbits) ^ l; l = r; r = fv;
        fv = metal_descrypt_des_f(r, ek_l[14], ek_r[14], saltbits) ^ l; l = r; r = fv;
        fv = metal_descrypt_des_f(r, ek_l[15], ek_r[15], saltbits) ^ l; l = r; r = fv;
        uint tmp = l; l = r; r = tmp;
    }

    /* Step 4: install pre-FP (l, r) into compact-table-probe state.
     * h[2..3] zero-padded to match the host's compact-table layout
     * (mdxfind.c:40433-40435 stores 4 il + 4 ir + 8 zero pad = 16 B). */
    st.h[0] = l;
    st.h[1] = r;
    st.h[2] = 0u;
    st.h[3] = 0u;
    return;
}

/* template_iterate: STUB. With max_iter = 1 (host-set for DESCRYPT), the
 * outer iter loop in template_phase0 runs exactly once and never calls
 * template_iterate. Mirrors PHPBB3 / MD5CRYPT pattern. */
static inline void template_iterate(thread template_state &st)
{
    (void)st;
}

/* template_digest_compare: probe the compact table with (l, r, 0, 0).
 * The host's compact-table loader (mdxfind.c:40402-40436) applies the
 * inverse FP permutation to the 13-char crypt hash and stores the
 * resulting (il, ir) at byte offsets 0..7, then zero-pads bytes 8..15.
 * Our state's (h[0], h[1], 0, 0) matches that layout byte-for-byte. */
static inline int template_digest_compare(thread const template_state &st,
                                device const uint *compact_fp,
                                device const uint *compact_idx,
                                ulong compact_mask,
                                uint max_probe,
                                uint hash_data_count,
                                device const uchar *hash_data_buf,
                                device const ulong *hash_data_off,
                                device const ulong *overflow_keys,
                                device const uchar *overflow_hashes,
                                device const uint *overflow_offsets,
                                uint overflow_count,
                                thread uint *out_idx)
{
    return probe_compact_idx(
        st.h[0], st.h[1], st.h[2], st.h[3],
        compact_fp, compact_idx,
        compact_mask, max_probe, hash_data_count,
        hash_data_buf, hash_data_off,
        overflow_keys, overflow_hashes, overflow_offsets, overflow_count,
        out_idx);
}


/* B3 dedup+overflow-aware variant. Mirrors metal_md5crypt_core.metal +
 * metal_phpbb3_core.metal -- only the OR_OVERFLOW variant is emitted;
 * the plain template_emit_hit macro is not referenced from the Metal
 * template scaffold. */
#define template_emit_hit_or_overflow(hits, hit_count, max_hits, st, widx, sidx, iter, \
                                      hashes_shown, matched_idx, dedup_mask, \
                                      ovr_set, ovr_gid, lane_gid) \
    EMIT_HIT_4_DEDUP_OR_OVERFLOW((hits), (hit_count), (max_hits), \
               (widx), (sidx), (iter), \
               (st).h[0], (st).h[1], (st).h[2], (st).h[3], \
               (hashes_shown), (matched_idx), (dedup_mask), \
               (ovr_set), (ovr_gid), (lane_gid))
