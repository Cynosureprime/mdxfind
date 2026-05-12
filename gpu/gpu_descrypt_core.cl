/* gpu_descrypt_core.cl -- DESCRYPT (e500) algorithm extension functions
 * for the generic dispatch template (Unix-crypt Phase 5: hand-written
 * Path A salted-template kernel; ports the slab gpu_descrypt.cl algorithm
 * body into the unified template extension API).
 *
 * $Revision: 1.1 $
 * $Log: gpu_descrypt_core.cl,v $
 * Revision 1.1  2026/05/08 21:43:19  dlr
 * DESCRYPT (e500) Phase 5: hand-written Path A core kernel for unified template path. HASH_WORDS=4 (l, r, 0, 0 emit pattern). 25 DES Feistel iters with 12-bit salt expansion (compute_saltbits ported from retired slab gpu_descrypt.cl). algo_mode=7. Standard DES only; extended _CCCCSSSS stays CPU. NO iter loop (template_iterate stub). 8-byte truncation implicit via DES key schedule (post-rule outputs > 8 bytes silently drop bytes 9+, matching CPU bsd_crypt_des semantics).
 *
 *
 * STATUS: DESCRYPT Phase 5 (2026-05-08) -- last Unix-crypt op to migrate
 * from slab to template path. Phases 1-4 (MD5CRYPT, SHA256CRYPT,
 * SHA512CRYPT, SHA512CRYPTMD5) preceded; the slab gpu_descrypt.cl is
 * RETIRED in this same commit. Single algo_mode (7); bespoke kernel that
 * will NOT share with BCRYPT (BCRYPT will need its own algo_modes for
 * future BCRYPT variants like BCRYPTMD5).
 *
 * DESCRYPT / Unix DES crypt(3) "old-style" semantics (mirrors mdxfind.c
 * JOB_DESCRYPT at lines 23636-23722 + crypt-des.c bsd_crypt_des +
 * slab oracle gpu_descrypt.cl):
 *
 *   salt buffer (2 bytes total): 2-char phpitoa64 salt
 *   salt = phpitoa64(salt[0]) | (phpitoa64(salt[1]) << 6)  (12-bit)
 *   key  = pass[0..min(plen,8)) << 1, zero-padded to 8 bytes
 *
 *   25 iterations of DES Feistel:
 *     for iter in 0..24:
 *       16 rounds of (l, r) -> (r, l XOR DES_F(r, ek[round], saltbits))
 *       swap(l, r)        // matches CPU do_des()'s tail r=l, l=f swap
 *
 *   probe (l, r, 0, 0) once at end. Compact table format: pre-FP form
 *   (host-side at mdxfind.c:40402-40436 applies inverse FP permutation
 *   to the stored 13-char crypt hash, recovering pre-FP (il, ir) for the
 *   compact-table layout 4 il + 4 ir + 8 zero pad = 16 bytes).
 *
 * DESIGN: 25-iter Feistel loop INSIDE template_finalize, max_iter=1.
 * --------------------------------------------------------------------
 * Mirrors PHPBB3 / MD5CRYPT / SHA256CRYPT / SHA512CRYPT pattern
 * (Unix-crypt ladder Phases 1-4 precedent):
 *   - the 25-iteration count is INTERNAL to the algorithm and FIXED;
 *     NOT user-controlled via -i;
 *   - only the FINAL state is probed (CPU semantics at mdxfind.c:23673
 *     calls JSLG once after the for-loop in bsd_crypt_des);
 *   - host forces params.max_iter = 1 at the rules-engine pack site
 *     so the kernel's outer iter loop runs exactly once and never
 *     calls template_iterate (which is a stub).
 *
 * Truncation strategy (HYBRID host+kernel, per architect §4):
 *   - HOST-side at the rules-engine pack site (mdxfind.c:11021-11026):
 *     when job->op == JOB_DESCRYPT && len > 8, clamp pack_len = 8 before
 *     the (char)len write + memcpy. Honors user's "prior to dispatching"
 *     intent for the no-rule synthetic pass.
 *   - KERNEL-side IMPLICIT via DES key schedule: the standard 8-byte DES
 *     key buffer below fills with zero-padding past first NUL so bytes
 *     9+ of post-rule outputs are ignored automatically. Defensive
 *     `if (plen > 8) plen = 8;` clamp below makes this explicit.
 *   - Net: GPU and CPU produce IDENTICAL results for any input length.
 *     Two distinct rule outputs that differ only in bytes 9+ collide on
 *     the same DES output (acceptable -- same as CPU).
 *
 * Salt-axis carrier: this kernel routes through the salted-template
 * scaffolding (GPU_TEMPLATE_HAS_SALT=1, SALT_POSITION=PREPEND in
 * defines_str). The salt buffer carries the 2-byte phpitoa64 salt
 * directly (mdxfind.c:44649 stores 2 bytes via JSLI(JudyJ[JOB_DESCRYPT],
 * line) and store_typesalt(JOB_DESCRYPT, line, 2)). gpu_pack_salts is
 * called with use_hashsalt=0 (no hashsalt synthesis).
 *
 * Extended DES (`_CCCCSSSS` 9-char salt, mdxfind.c:40395 second arm) is
 * OUT OF SCOPE for this Phase 5 kernel. The salt-pack filter at
 * gpujob_opencl.c gpu_pack_salts (~line 485-503) skips saltlen != 2
 * for JOB_DESCRYPT, so extended-DES salts CPU-fallback through
 * bsd_crypt_des unchanged.
 *
 * Inside the kernel:
 *   - salt_bytes[0..1] -> 2-char phpitoa64 salt
 *
 * Validation oracle: gpu_descrypt.cl descrypt_batch (slab kernel; this
 * file is a port of that body into the template extension API).
 * Differences from slab:
 *   - reads pass from `data` (post-rule buf passed by template; PRIVATE
 *     uchar *) rather than from a hexhashes buffer indexed by word_idx;
 *   - reads salt from `salt_bytes` (the global salt buffer threaded by
 *     gpu_template.cl under GPU_TEMPLATE_HAS_SALT) rather than from
 *     the slab's own salts/salt_offsets/salt_lens trio;
 *   - probes via template_digest_compare's probe_compact_idx (matches
 *     the existing 16-byte compact-table format: 4 il + 4 ir + 8 zero
 *     pad) using HASH_WORDS=4 with zero-pad upper two words.
 *
 * The slab kernel uses __local cached SPtrans for fast divergent S-box
 * access on NVIDIA. We REPLICATE that pattern here: the workgroup-shared
 * 8x64-uint __local s_SP[8][64] is initialized BEFORE the kernel's
 * per-lane work begins (must happen before any early return so all
 * threads in the workgroup participate in the barrier). Since
 * template_finalize is called from inside template_phase0's per-lane
 * code, we cannot use __local memory there; we use the __constant
 * SPtrans[][] directly. (Lower performance vs slab __local cache, but
 * structurally clean -- no addrspace casts, no workgroup barriers in
 * the shared template scaffold.)
 *
 * Hit replay: host calls des_reconstruct(curin.i[0], curin.i[1],
 * salt_bytes, desbuf) to reconstruct the 13-char crypt hash, probes
 * JudyJ[JOB_DESCRYPT] for the line, applies CAS dedup, and emits via
 * prfound. The display password is CLAMPED TO 8 BYTES per CPU parity
 * (mirrors mdxfind.c:23676-23677 `i = min(len, 8)` for non-extended
 * salts). Q1 user decision 2026-05-08.
 *
 * R1 mitigation (AMD ROCm comgr addrspace fragility): single private
 * buffer pattern. uint kb[8] working buffer + 16 ek_l/ek_r uint pairs
 * for the round keys are private; salt_bytes is __global (read 2 bytes
 * inline, no addrspace casts).
 *
 * R2 (register pressure): 16+16 = 32 uint round-key arrays + 8-uint key
 * buffer + 2 uint (l, r) state. Comparable to PHPBB3 plus the round-key
 * storage (PHPBB3 only has 4-uint MD5 state). Expected priv_mem on
 * Pascal in the 41-43 KB band shared by other unified-template
 * dispatches (DES algorithm state adds ~256 B over PHPBB3's baseline).
 *
 * Source order at compile time:
 *   [ gpu_common_str, gpu_md5_rules_str, gpu_descrypt_core_str,
 *     gpu_template_str ]
 *
 * Cache key (R3): defines_str =
 *   "HASH_WORDS=4,HASH_BLOCK_BYTES=64,HAS_SALT=1,SALT_POSITION=PREPEND,
 *    BASE_ALGO=descrypt"
 *
 * Cache-key disambiguation:
 *   - From MD5SALT family (HASH_WORDS=4 + HAS_SALT=1 + BASE_ALGO=md5)
 *     by BASE_ALGO=descrypt axis.
 *   - From PHPBB3 (HASH_WORDS=4 + HAS_SALT=1 + BASE_ALGO=phpbb3)
 *     by BASE_ALGO=descrypt axis.
 *   - From MD5CRYPT (HASH_WORDS=4 + HAS_SALT=1 + BASE_ALGO=md5crypt)
 *     by BASE_ALGO=descrypt axis.
 *   - From every other salted/unsalted template via the unique
 *     BASE_ALGO=descrypt token.
 *
 * algo_mode: DESCRYPT uses algo_mode=7 (next free after 0..6). Bespoke
 * kernel; will NOT share with BCRYPT or any other algo. The `(void)
 * algo_mode;` cast below documents that the kernel ignores the value
 * (single-mode algorithm). The host sets pparams->algo_mode = 7u for
 * cache-key consistency with the defines_str BASE_ALGO=descrypt.
 */

#ifndef HASH_WORDS
#define HASH_WORDS 4
#endif
#ifndef HASH_BLOCK_BYTES
#define HASH_BLOCK_BYTES 64
#endif

/* DES SP-tables (combined S-box + P-permutation), ported byte-for-byte
 * from gpu_descrypt.cl. The slab kernel caches these in __local for
 * fast divergent access; the template path reads from __constant
 * directly (no workgroup barrier in the shared template scaffold). */
__constant uint DESCRYPT_SPtrans[8][64] = {
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

/* PC-1/PC-2/key-shift tables (ported from gpu_descrypt.cl). */
__constant uchar DESCRYPT_pc1_c[28] = {
    57,49,41,33,25,17, 9, 1,58,50,42,34,26,18,
    10, 2,59,51,43,35,27,19,11, 3,60,52,44,36 };
__constant uchar DESCRYPT_pc1_d[28] = {
    63,55,47,39,31,23,15, 7,62,54,46,38,30,22,
    14, 6,61,53,45,37,29,21,13, 5,28,20,12, 4 };
__constant uchar DESCRYPT_pc2[48] = {
    14,17,11,24, 1, 5, 3,28,15, 6,21,10,23,19,12, 4,26, 8,16, 7,27,20,13, 2,
    41,52,31,37,47,55,30,40,51,45,33,48,44,49,39,56,34,53,46,42,50,36,29,32 };
__constant uchar DESCRYPT_key_shifts[16] = {1,1,2,2,2,2,2,2,1,2,2,2,2,2,2,1};

/* Bit extraction helpers (ported from gpu_descrypt.cl). */
static inline uint descrypt_gb64(uint hi, uint lo, uint b) {
    return (b <= 32) ? ((hi >> (32 - b)) & 1u) : ((lo >> (64 - b)) & 1u);
}
static inline uint descrypt_gb28(uint v, uint b) { return (v >> (28 - b)) & 1u; }
static inline uint descrypt_a2b(uint ch) {
    if (ch >= 'a') return ch - 'a' + 38;
    if (ch >= 'A') return ch - 'A' + 12;
    if (ch >= '.') return ch - '.';
    return 0;
}
static inline uint descrypt_compute_saltbits(uint salt) {
    uint sb = 0;
    for (int i = 0; i < 12; i++) sb |= ((salt >> i) & 1u) << (23 - i);
    return sb;
}

/* Build 16 round keys (ek_l[0..15], ek_r[0..15]) from the 64-bit key
 * (khi, klo). Mirrors slab gpu_descrypt.cl:des_key_schedule byte-for-
 * byte (PC-1, 16 left-rotations with cumulative shift counts, PC-2). */
static inline void descrypt_des_key_schedule(uint khi, uint klo,
                                             uint *ek_l, uint *ek_r)
{
    uint c = 0, d = 0;
    for (int i = 0; i < 28; i++) {
        c |= descrypt_gb64(khi, klo, DESCRYPT_pc1_c[i]) << (27 - i);
        d |= descrypt_gb64(khi, klo, DESCRYPT_pc1_d[i]) << (27 - i);
    }
    uint total_shift = 0;
    for (int rnd = 0; rnd < 16; rnd++) {
        total_shift += DESCRYPT_key_shifts[rnd];
        uint tc = ((c << total_shift) | (c >> (28 - total_shift))) & 0x0FFFFFFFu;
        uint td = ((d << total_shift) | (d >> (28 - total_shift))) & 0x0FFFFFFFu;
        uint kl = 0, kr = 0;
        for (int i = 0; i < 24; i++) {
            uint b = DESCRYPT_pc2[i];
            kl |= ((b <= 28) ? descrypt_gb28(tc, b)
                             : descrypt_gb28(td, b - 28)) << (23 - i);
        }
        for (int i = 0; i < 24; i++) {
            uint b = DESCRYPT_pc2[24 + i];
            kr |= ((b <= 28) ? descrypt_gb28(tc, b)
                             : descrypt_gb28(td, b - 28)) << (23 - i);
        }
        ek_l[rnd] = kl;
        ek_r[rnd] = kr;
    }
}

/* DES Feistel round. r' = E(r) salted XOR with key, then 8-way SP-table
 * lookup. Reads from __constant DESCRYPT_SPtrans (slab uses __local
 * cache; template path skips the workgroup-shared cache to keep the
 * shared template scaffold barrier-free). */
static inline uint descrypt_des_f(uint r, uint kl, uint kr, uint saltbits) {
    uint r48l = ((r & 0x00000001u) << 23) | ((r & 0xf8000000u) >> 9) |
                ((r & 0x1f800000u) >> 11) | ((r & 0x01f80000u) >> 13) |
                ((r & 0x001f8000u) >> 15);
    uint r48r = ((r & 0x0001f800u) <<  7) | ((r & 0x00001f80u) <<  5) |
                ((r & 0x000001f8u) <<  3) | ((r & 0x0000001fu) <<  1) |
                ((r & 0x80000000u) >> 31);
    uint f = (r48l ^ r48r) & saltbits;
    r48l ^= f ^ kl;
    r48r ^= f ^ kr;
    return DESCRYPT_SPtrans[0][(r48l >> 18) & 0x3fu]
         | DESCRYPT_SPtrans[1][(r48l >> 12) & 0x3fu]
         | DESCRYPT_SPtrans[2][(r48l >>  6) & 0x3fu]
         | DESCRYPT_SPtrans[3][ r48l        & 0x3fu]
         | DESCRYPT_SPtrans[4][(r48r >> 18) & 0x3fu]
         | DESCRYPT_SPtrans[5][(r48r >> 12) & 0x3fu]
         | DESCRYPT_SPtrans[6][(r48r >>  6) & 0x3fu]
         | DESCRYPT_SPtrans[7][ r48r        & 0x3fu];
}

/* Per-lane state struct. DES emits a pre-FP (l, r) pair = 64 bits. We
 * carry it in h[0..1]; h[2..3] are zero-padded so probe_compact_idx
 * sees the same 16-byte layout the host's compact-table loader stores
 * (mdxfind.c:40433-40435: 4 il + 4 ir + 8 zero pad). HASH_WORDS=4 stays
 * the canonical width for this template instantiation. */
typedef struct {
    uint h[HASH_WORDS];
} template_state;

/* template_init: zero the state. DES has no IV; (l, r) start at 0 in the
 * 25-iter Feistel loop (slab gpu_descrypt.cl:186 `uint l = 0, r = 0;`). */
static inline void template_init(template_state *st) {
    st->h[0] = 0u;
    st->h[1] = 0u;
    st->h[2] = 0u;
    st->h[3] = 0u;
}

/* template_transform: stub for interface symmetry. DESCRYPT's
 * template_finalize manages the full DES state inline -- never routes
 * through this. Provided for completeness (matches PHPBB3 / MD5CRYPT
 * pattern). */
static inline void template_transform(template_state *st,
                                      const uchar *block)
{
    (void)st;
    (void)block;
}

/* template_finalize: full DESCRYPT chain.
 *
 * Step 1: build 8-byte DES key from data[0..min(plen,8)) with KEY left-
 *   shift (mirrors crypt-des.c:626-630 `*q++ = *key << 1; if (*key !=
 *   '\0') key++;` -- bytes past first NUL pad with zero, BUT mdxfind's
 *   input is fixed-length post-rule so we just use min(plen, 8) bytes
 *   shifted left by 1 and zero-pad the rest).
 *
 * Step 2: decode 2-char phpitoa64 salt from salt_bytes[0..2) into a
 *   12-bit salt, then expand via descrypt_compute_saltbits to 24-bit
 *   saltbits (mirrors slab :184).
 *
 * Step 3: run 25 DES iterations, each with 16 Feistel rounds + final
 *   swap (mirrors slab :187-205 byte-for-byte; slab oracle is the
 *   correctness reference).
 *
 * Step 4: install pre-FP (l, r) into st->h[0..1]; zero h[2..3] for the
 *   compact-table probe.
 *
 * algo_mode: DESCRYPT has only one mode (7). The arg is unused; kept
 * for interface symmetry with the salted-template signature. */
static inline void template_finalize(template_state *st,
                                     const uchar *data, int len
#ifdef GPU_TEMPLATE_HAS_SALT
                                     , __global const uchar *salt_bytes
                                     , uint salt_len
                                     , uint algo_mode
#endif
                                     )
{
#ifdef GPU_TEMPLATE_HAS_SALT
    (void)algo_mode;
    (void)salt_len;

    /* Defensive cap: standard DES uses only the first 8 bytes of the
     * key (host-side rules-engine pack site already clamps for the
     * synthetic no-rule pass; clamp here too so masked / rule-extended
     * outputs bigger than 8 bytes silently truncate -- matches CPU
     * crypt-des.c:626 NUL-terminate-on-first-NUL semantics). */
    int plen = len;
    if (plen > 8) plen = 8;

    /* Step 1: build 8-byte DES key buffer. byte = (data[i] << 1)
     * with bytes past min(plen,8) zero-padded. Mirrors slab :174-175. */
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
    descrypt_des_key_schedule(khi, klo, ek_l, ek_r);

    /* Step 3a: decode 2-char phpitoa64 salt + expand to 24-bit saltbits.
     * Defensive: assume salt_len == 2 (guaranteed by salt-pack filter
     * at gpujob_opencl.c gpu_pack_salts which skips saltlen != 2 for
     * JOB_DESCRYPT). Reading salt_bytes[0..1] with no bounds check
     * matches slab :182-184 byte-for-byte. */
    uint salt = descrypt_a2b((uint)salt_bytes[0])
              | (descrypt_a2b((uint)salt_bytes[1]) << 6);
    uint saltbits = descrypt_compute_saltbits(salt);

    /* Step 3b: 25 DES iterations of (l, r) -> 16 rounds + final swap.
     * Mirrors slab :186-206 byte-for-byte. */
    uint l = 0u, r = 0u;
    for (int iter = 0; iter < 25; iter++) {
        uint fv;
        fv = descrypt_des_f(r, ek_l[ 0], ek_r[ 0], saltbits) ^ l; l = r; r = fv;
        fv = descrypt_des_f(r, ek_l[ 1], ek_r[ 1], saltbits) ^ l; l = r; r = fv;
        fv = descrypt_des_f(r, ek_l[ 2], ek_r[ 2], saltbits) ^ l; l = r; r = fv;
        fv = descrypt_des_f(r, ek_l[ 3], ek_r[ 3], saltbits) ^ l; l = r; r = fv;
        fv = descrypt_des_f(r, ek_l[ 4], ek_r[ 4], saltbits) ^ l; l = r; r = fv;
        fv = descrypt_des_f(r, ek_l[ 5], ek_r[ 5], saltbits) ^ l; l = r; r = fv;
        fv = descrypt_des_f(r, ek_l[ 6], ek_r[ 6], saltbits) ^ l; l = r; r = fv;
        fv = descrypt_des_f(r, ek_l[ 7], ek_r[ 7], saltbits) ^ l; l = r; r = fv;
        fv = descrypt_des_f(r, ek_l[ 8], ek_r[ 8], saltbits) ^ l; l = r; r = fv;
        fv = descrypt_des_f(r, ek_l[ 9], ek_r[ 9], saltbits) ^ l; l = r; r = fv;
        fv = descrypt_des_f(r, ek_l[10], ek_r[10], saltbits) ^ l; l = r; r = fv;
        fv = descrypt_des_f(r, ek_l[11], ek_r[11], saltbits) ^ l; l = r; r = fv;
        fv = descrypt_des_f(r, ek_l[12], ek_r[12], saltbits) ^ l; l = r; r = fv;
        fv = descrypt_des_f(r, ek_l[13], ek_r[13], saltbits) ^ l; l = r; r = fv;
        fv = descrypt_des_f(r, ek_l[14], ek_r[14], saltbits) ^ l; l = r; r = fv;
        fv = descrypt_des_f(r, ek_l[15], ek_r[15], saltbits) ^ l; l = r; r = fv;
        uint tmp = l; l = r; r = tmp;
    }

    /* Step 4: install pre-FP (l, r) into compact-table-probe state.
     * h[2..3] zero-padded to match the host's compact-table layout
     * (mdxfind.c:40433-40435 stores 4 il + 4 ir + 8 zero pad = 16 B). */
    st->h[0] = l;
    st->h[1] = r;
    st->h[2] = 0u;
    st->h[3] = 0u;
    return;
#else
    /* Defensive fall-through for !HAS_SALT. DESCRYPT is always salted
     * (the algorithm requires the 2-char phpitoa64 salt for the F-round
     * E-expansion mix); a no-salt build would have nothing to do. Set
     * state to zero and return. */
    (void)data;
    (void)len;
    st->h[0] = 0u;
    st->h[1] = 0u;
    st->h[2] = 0u;
    st->h[3] = 0u;
#endif
}

/* template_iterate: STUB. With max_iter = 1 (host-set for DESCRYPT), the
 * outer iter loop in template_phase0 runs exactly once and never calls
 * template_iterate. Mirrors PHPBB3 / MD5CRYPT / SHA256CRYPT / SHA512CRYPT
 * pattern. */
static inline void template_iterate(template_state *st)
{
    (void)st;
}

/* template_digest_compare: probe the compact table with (l, r, 0, 0).
 * The host's compact-table loader (mdxfind.c:40402-40436) applies the
 * inverse FP permutation to the 13-char crypt hash and stores the
 * resulting (il, ir) at byte offsets 0..7, then zero-pads bytes 8..15.
 * Our state's (h[0], h[1], 0, 0) matches that layout byte-for-byte. */
static inline int template_digest_compare(
    const template_state *st,
    __global const uint *compact_fp, __global const uint *compact_idx,
    ulong compact_mask, uint max_probe, uint hash_data_count,
    __global const uchar *hash_data_buf, __global const ulong *hash_data_off,
    __global const ulong *overflow_keys, __global const uchar *overflow_hashes,
    __global const uint *overflow_offsets, uint overflow_count,
    uint *out_idx)
{
    return probe_compact_idx(
        st->h[0], st->h[1], st->h[2], st->h[3],
        compact_fp, compact_idx,
        compact_mask, max_probe, hash_data_count,
        hash_data_buf, hash_data_off,
        overflow_keys, overflow_hashes, overflow_offsets, overflow_count,
        out_idx);
}

/* template_emit_hit: emit a hit. DESCRYPT = pre-FP (l, r) = 2 LE uint32
 * + 2 zero-pad words. Same EMIT_HIT_4 wire format as MD5/PHPBB3/MD5CRYPT
 * (host's hit-replay arm reconstructs the 13-char crypt hash via
 * des_reconstruct in gpujob_opencl.c). */
#define template_emit_hit(hits, hit_count, max_hits, st, widx, sidx, iter) \
    EMIT_HIT_4((hits), (hit_count), (max_hits), \
               (widx), (sidx), (iter), \
               (st)->h[0], (st)->h[1], (st)->h[2], (st)->h[3])

/* B3 dedup+overflow-aware variant. */
#define template_emit_hit_or_overflow(hits, hit_count, max_hits, st, widx, sidx, iter, \
                                      hashes_shown, matched_idx, dedup_mask, \
                                      ovr_set, ovr_gid, lane_gid) \
    EMIT_HIT_4_DEDUP_OR_OVERFLOW((hits), (hit_count), (max_hits), \
               (widx), (sidx), (iter), \
               (st)->h[0], (st)->h[1], (st)->h[2], (st)->h[3], \
               (hashes_shown), (matched_idx), (dedup_mask), \
               (ovr_set), (ovr_gid), (lane_gid))
