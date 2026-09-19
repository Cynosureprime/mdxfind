/* gpu_descrypt_core.cl -- DESCRYPT (e500) algorithm extension functions
 * for the generic dispatch template (Unix-crypt Phase 5: hand-written
 * Path A salted-template kernel; ports the slab gpu_descrypt.cl algorithm
 * body into the unified template extension API).
 *
 * $Revision: 1.2 $
 * $Log: gpu_descrypt_core.cl,v $
 * Revision 1.2  2026/09/18 20:45:00  dlr
 * Phase 0a DESCRYPT e500 GPU speedup, both backends. Three increments to the existing scalar carrier, no new kernel and no new family: one, the SP tables staged into local and threadgroup memory via a new sibling hook GPU_TEMPLATE_HAS_SHARED_LOCAL, one shared 2 KB workgroup copy rather than the per-lane writable slab of HAS_LOCAL_BUFFER, which pins the workgroup to 8 and does not compose with HAS_PRE_SALT; two, a nibble-table key setup built on device from the same DESCRYPT_pc2 walk the bit loop used; three, GPU_TEMPLATE_HAS_PRE_SALT on the scalar kernel, hoisting the key schedule out of the salt loop where it had been recomputed for every one of 4,096 salts. The two backends ship at different levels because the measurements disagree about which increment pays, and the reason is structural. OpenCL ships level 1: increment 1 alone takes a GTX 1080 from 13.5 to 146.1 M crypt per second at 3,471 salts, 10.79 times, and 1.40 times ahead of hashcat -a 0 measured on the same non-cracking fixture the same hour at 104.6. Increment 2 is neutral there at minus 0.04 percent and increment 3 costs 75.9 percent at 16 salts because it divides the NDRange by SALT_BATCH. Metal ships level 3 with the key tables off: increment 1 gives only 10 to 11 percent because Apple has no constant-cache broadcast penalty to relieve, increment 2 is a loss of 9.5 to 11.2 percent, and increment 3 is the whole win at plus 32.8 percent on an M1 and plus 44.2 percent on an M2 Max, because the Metal generic-family grid is num_words alone with rule and mask and salt as inner loops, so the hoist removes work without removing a grid axis. Net shipped gain over pre-0a at 3,471 salts: 10.79 times on the 1080, 35.8 percent on M1, 42.3 percent on M2 Max. Both Apple GPUs independently agree level 3 is fastest at every salt count, so no per-GPU selection is warranted. DESCRYPT_SALT_BATCH is its own constant at 16 and deliberately not dynsize_compile_time_N, whose single shared 64 belongs to the MD5SALT servo and would make two algorithms re-JIT each other. All knobs are compile-time ifndef; no environment variable is read. Byte identity of the shared templates proven rather than assumed: 111 OpenCL program variants and 162 Metal variants preprocessed against the prior revision with zero lines differing, the template additions being pure insertions inside the new ifdef. Correctness validated against glibc and Darwin crypt 3, an implementation by different authors rather than mdxfind own crypt-des.c, over 8 passwords by all 4,096 salts, 32,768 of 32,768 recovered with zero differences on both backends and zero CPU versus GPU divergence, plus CPU equals GPU on the real 47,366-hash list, the five adjacent DES-family types unchanged, the 1.591 salt-compaction bounds fix not regressed, and -z mode unchanged. Two brief assumptions were measured wrong and are recorded here: the key schedule does not become 25 to 35 percent of the kernel once the tables move, and increment 1 returns no constant-bank headroom because the constant table is the source the local copy initialises from, with dot-const totalling 21,324 bytes before and after.
 *
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

/* ----------------------------------------------------------------------
 * Phase 0a (2026-09-18) -- three increments over the scalar kernel, all
 * gated on DESCRYPT_PHASE0A so each can be priced on its own:
 *
 *   >= 1  the 2 KB SP table is staged into ONE workgroup-shared __local
 *         copy (read-only and identical for every lane, so this is NOT
 *         BCRYPT's per-lane partition) via GPU_TEMPLATE_HAS_SHARED_LOCAL.
 *         NVIDIA's constant cache broadcasts one address per request, so
 *         a warp's ~32 distinct S-box indices serialise there; __local is
 *         banked and services them in parallel.
 *   >= 2  PC-2 becomes four nibble-indexed lookup tables (448 more uints
 *         of the same __local block) instead of 768 single-bit
 *         extractions per key.  The tables are BUILT ON DEVICE from the
 *         very same DESCRYPT_pc2[] walk the bit loop used, so the table
 *         and the loop cannot drift apart.
 *   >= 3  the key schedule moves out of the salt loop entirely
 *         (GPU_TEMPLATE_HAS_PRE_SALT): it depends only on the candidate,
 *         and the OpenCL salt axis is in the gid, so today it is redone
 *         for every one of 4,096 salts.
 *
 * The __constant SPtrans stays -- it is the source the __local copy is
 * initialised from, so the 2 KB is NOT returned to NVIDIA's constant
 * bank.  (OpenCL 1.2 has no other program-scope address space.)
 * ---------------------------------------------------------------------- */
#ifndef DESCRYPT_PHASE0A
#define DESCRYPT_PHASE0A 3
#endif

/* Increment 2 has its own name so it can be switched off while 1 and 3
 * stay on.  That combination is not a shipping configuration -- it exists
 * because "does the table-driven key setup still pay once the schedule is
 * hoisted out of the salt loop?" is a question only a measurement can
 * answer, and answering it needs L1+L3-without-L2 as a data point. */
#ifndef DESCRYPT_KS_TABLE
#if DESCRYPT_PHASE0A >= 2
#define DESCRYPT_KS_TABLE 1
#else
#define DESCRYPT_KS_TABLE 0
#endif
#endif

#if DESCRYPT_PHASE0A >= 1
#define GPU_TEMPLATE_HAS_SHARED_LOCAL 1
/* SP table occupies uints [0, 512): row r, column c at (r << 6) | c. */
#define DESCRYPT_SP_UINTS   512u
#if DESCRYPT_KS_TABLE
/* Key-schedule nibble tables occupy uints [512, 960).  Four sub-tables of
 * 7 groups x 16 values, in this order:
 *   +  0  TCkl[g][v]   contribution of tc nibble g to the LEFT  24 bits
 *   +112  TCkr[g][v]   contribution of tc nibble g to the RIGHT 24 bits
 *   +224  TDkl[g][v]   contribution of td nibble g to the LEFT  24 bits
 *   +336  TDkr[g][v]   contribution of td nibble g to the RIGHT 24 bits
 * All four are built generically.  In the standard PC-2 the left 24 bits
 * come only from C and the right 24 only from D, so TCkr and TDkl are in
 * fact all-zero and the lookup could be halved to 14 per round -- that
 * specialisation is deliberately NOT taken here, because it would bake a
 * property of the table into the code that reads it. */
#define DESCRYPT_KS_OFF     512u
#define DESCRYPT_KS_UINTS   448u
#define GPU_TEMPLATE_SHARED_LOCAL_UINTS (DESCRYPT_SP_UINTS + DESCRYPT_KS_UINTS)
#else
#define GPU_TEMPLATE_SHARED_LOCAL_UINTS (DESCRYPT_SP_UINTS)
#endif
#endif

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

#ifdef GPU_TEMPLATE_HAS_SHARED_LOCAL
/* template_shared_local_init: cooperative fill of the workgroup-shared
 * __local block.  The template calls this from kernel scope BEFORE its
 * `gid >= total` early return and follows it with a barrier -- a barrier
 * placed after that return is unreachable for the tail lanes of the last
 * workgroup and is undefined behaviour.  Strided by lsz, hashcat's
 * m01500_a0-pure.cl:37-58 shape.
 *
 * Every table here is derived from the SAME __constant arrays the inline
 * bit loops read, so an error can only be in this function, not a drift
 * between two independent encodings of PC-2. */
static inline void template_shared_local_init(__local uint *tbl,
                                              uint lid, uint lsz)
{
    for (uint i = lid; i < DESCRYPT_SP_UINTS; i += lsz) {
        tbl[i] = DESCRYPT_SPtrans[i >> 6][i & 63u];
    }
#if DESCRYPT_KS_TABLE
    /* 112 entries = 7 nibble groups x 16 nibble values.  Bit b of a 28-bit
     * C/D word sits at bit position (28 - b); group g covers b in
     * [4g+1, 4g+4], so the group's nibble is (word >> (24 - 4g)) & 15 with
     * b == 4g+1 as its MOST significant bit. */
    for (uint e = lid; e < 112u; e += lsz) {
        uint g = e >> 4;
        uint v = e & 15u;
        uint tckl = 0u, tckr = 0u, tdkl = 0u, tdkr = 0u;
        for (uint i = 0u; i < 24u; i++) {
            uint bl = (uint)DESCRYPT_pc2[i];
            uint br = (uint)DESCRYPT_pc2[24u + i];
            if (bl <= 28u) {
                if (((bl - 1u) >> 2) == g)
                    tckl |= ((v >> (3u - ((bl - 1u) & 3u))) & 1u) << (23u - i);
            } else {
                uint b = bl - 28u;
                if (((b - 1u) >> 2) == g)
                    tdkl |= ((v >> (3u - ((b - 1u) & 3u))) & 1u) << (23u - i);
            }
            if (br <= 28u) {
                if (((br - 1u) >> 2) == g)
                    tckr |= ((v >> (3u - ((br - 1u) & 3u))) & 1u) << (23u - i);
            } else {
                uint b = br - 28u;
                if (((b - 1u) >> 2) == g)
                    tdkr |= ((v >> (3u - ((b - 1u) & 3u))) & 1u) << (23u - i);
            }
        }
        tbl[DESCRYPT_KS_OFF +   0u + e] = tckl;
        tbl[DESCRYPT_KS_OFF + 112u + e] = tckr;
        tbl[DESCRYPT_KS_OFF + 224u + e] = tdkl;
        tbl[DESCRYPT_KS_OFF + 336u + e] = tdkr;
    }
#endif
}
#endif /* GPU_TEMPLATE_HAS_SHARED_LOCAL */

/* Build 16 round keys (ek_l[0..15], ek_r[0..15]) from the 64-bit key
 * (khi, klo). Mirrors slab gpu_descrypt.cl:des_key_schedule byte-for-
 * byte (PC-1, 16 left-rotations with cumulative shift counts, PC-2). */
static inline void descrypt_des_key_schedule(uint khi, uint klo,
                                             uint *ek_l, uint *ek_r
#if DESCRYPT_KS_TABLE
                                             , __local const uint *ks
#endif
                                             )
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
#if DESCRYPT_KS_TABLE
        for (uint g = 0u; g < 7u; g++) {
            uint base = g << 4;
            uint nc = (tc >> (24u - (g << 2))) & 15u;
            uint nd = (td >> (24u - (g << 2))) & 15u;
            kl |= ks[         base + nc] | ks[224u + base + nd];
            kr |= ks[112u  + base + nc] | ks[336u + base + nd];
        }
#else
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
#endif
        ek_l[rnd] = kl;
        ek_r[rnd] = kr;
    }
}

/* DES Feistel round. r' = E(r) salted XOR with key, then 8-way SP-table
 * lookup. Phase 0a >= 1 reads the workgroup-shared __local copy; below
 * that it reads __constant DESCRYPT_SPtrans directly. */
static inline uint descrypt_des_f(uint r, uint kl, uint kr, uint saltbits
#ifdef GPU_TEMPLATE_HAS_SHARED_LOCAL
                                  , __local const uint *s_sp
#endif
                                  ) {
    uint r48l = ((r & 0x00000001u) << 23) | ((r & 0xf8000000u) >> 9) |
                ((r & 0x1f800000u) >> 11) | ((r & 0x01f80000u) >> 13) |
                ((r & 0x001f8000u) >> 15);
    uint r48r = ((r & 0x0001f800u) <<  7) | ((r & 0x00001f80u) <<  5) |
                ((r & 0x000001f8u) <<  3) | ((r & 0x0000001fu) <<  1) |
                ((r & 0x80000000u) >> 31);
    uint f = (r48l ^ r48r) & saltbits;
    r48l ^= f ^ kl;
    r48r ^= f ^ kr;
#ifdef GPU_TEMPLATE_HAS_SHARED_LOCAL
    return s_sp[  0u + ((r48l >> 18) & 0x3fu)]
         | s_sp[ 64u + ((r48l >> 12) & 0x3fu)]
         | s_sp[128u + ((r48l >>  6) & 0x3fu)]
         | s_sp[192u + ( r48l        & 0x3fu)]
         | s_sp[256u + ((r48r >> 18) & 0x3fu)]
         | s_sp[320u + ((r48r >> 12) & 0x3fu)]
         | s_sp[384u + ((r48r >>  6) & 0x3fu)]
         | s_sp[448u + ( r48r        & 0x3fu)];
#else
    return DESCRYPT_SPtrans[0][(r48l >> 18) & 0x3fu]
         | DESCRYPT_SPtrans[1][(r48l >> 12) & 0x3fu]
         | DESCRYPT_SPtrans[2][(r48l >>  6) & 0x3fu]
         | DESCRYPT_SPtrans[3][ r48l        & 0x3fu]
         | DESCRYPT_SPtrans[4][(r48r >> 18) & 0x3fu]
         | DESCRYPT_SPtrans[5][(r48r >> 12) & 0x3fu]
         | DESCRYPT_SPtrans[6][(r48r >>  6) & 0x3fu]
         | DESCRYPT_SPtrans[7][ r48r        & 0x3fu];
#endif
}

/* Per-lane state struct. DES emits a pre-FP (l, r) pair = 64 bits. We
 * carry it in h[0..1]; h[2..3] are zero-padded so probe_compact_idx
 * sees the same 16-byte layout the host's compact-table loader stores
 * (mdxfind.c:48283-48327: 4 il + 4 ir + 8 zero pad). HASH_WORDS=4 stays
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

/* descrypt_build_key: 8-byte DES key from data[0..min(len,8)) with the
 * crypt-des.c:626-630 left-shift-by-1, zero padded.  The 8-byte clamp is
 * the CPU's truncation semantics (bytes 9+ of a rule output are dropped);
 * the host clamps too, at mdxfind.c:14261. */
static inline void descrypt_build_key(const uchar *data, int len,
                                      uint *khi, uint *klo)
{
    int plen = len;
    if (plen > 8) plen = 8;
    uchar kb[8];
    for (int i = 0; i < 8; i++) {
        kb[i] = (i < plen) ? (uchar)((uint)data[i] << 1) : (uchar)0u;
    }
    *khi = ((uint)kb[0] << 24) | ((uint)kb[1] << 16)
         | ((uint)kb[2] <<  8) |  (uint)kb[3];
    *klo = ((uint)kb[4] << 24) | ((uint)kb[5] << 16)
         | ((uint)kb[6] <<  8) |  (uint)kb[7];
}

/* descrypt_run: the salt-dependent half -- decode the 2-char phpitoa64
 * salt, expand to 24 saltbits, run 25 x 16 Feistel rounds from a zero
 * block, install the pre-FP (l, r).  Byte-for-byte the Step 3/4 body of
 * the pre-Phase-0a template_finalize. */
static inline void descrypt_run(template_state *st,
                                const uint *ek_l, const uint *ek_r,
                                __global const uchar *salt_bytes
#ifdef GPU_TEMPLATE_HAS_SHARED_LOCAL
                                , __local const uint *s_sp
#endif
                                )
{
#ifdef GPU_TEMPLATE_HAS_SHARED_LOCAL
#define DESF(rr, kk) descrypt_des_f((rr), ek_l[kk], ek_r[kk], saltbits, s_sp)
#else
#define DESF(rr, kk) descrypt_des_f((rr), ek_l[kk], ek_r[kk], saltbits)
#endif
    uint salt = descrypt_a2b((uint)salt_bytes[0])
              | (descrypt_a2b((uint)salt_bytes[1]) << 6);
    uint saltbits = descrypt_compute_saltbits(salt);

    uint l = 0u, r = 0u;
    for (int iter = 0; iter < 25; iter++) {
        uint fv;
        fv = DESF(r,  0) ^ l; l = r; r = fv;
        fv = DESF(r,  1) ^ l; l = r; r = fv;
        fv = DESF(r,  2) ^ l; l = r; r = fv;
        fv = DESF(r,  3) ^ l; l = r; r = fv;
        fv = DESF(r,  4) ^ l; l = r; r = fv;
        fv = DESF(r,  5) ^ l; l = r; r = fv;
        fv = DESF(r,  6) ^ l; l = r; r = fv;
        fv = DESF(r,  7) ^ l; l = r; r = fv;
        fv = DESF(r,  8) ^ l; l = r; r = fv;
        fv = DESF(r,  9) ^ l; l = r; r = fv;
        fv = DESF(r, 10) ^ l; l = r; r = fv;
        fv = DESF(r, 11) ^ l; l = r; r = fv;
        fv = DESF(r, 12) ^ l; l = r; r = fv;
        fv = DESF(r, 13) ^ l; l = r; r = fv;
        fv = DESF(r, 14) ^ l; l = r; r = fv;
        fv = DESF(r, 15) ^ l; l = r; r = fv;
        uint tmp = l; l = r; r = tmp;
    }
    st->h[0] = l;
    st->h[1] = r;
    st->h[2] = 0u;
    st->h[3] = 0u;
#undef DESF
}

#if defined(GPU_TEMPLATE_HAS_PRE_SALT) && defined(GPU_TEMPLATE_HAS_SALT)
/* ----------------------------------------------------------------------
 * Phase 0a increment 3: the key schedule depends ONLY on the candidate,
 * but the OpenCL salt axis rides in the gid (gpu_template.cl), so before
 * this it was recomputed for every one of 4,096 salts.  Under
 * GPU_TEMPLATE_HAS_PRE_SALT the template replaces the salt axis with a
 * salt_chunk axis of SALT_BATCH salts and calls template_pre_salt once
 * per chunk.  There is no sentinel/fallback arm: DESCRYPT has a single
 * algo_mode and the schedule is ALWAYS hoistable.
 * ---------------------------------------------------------------------- */
typedef struct {
    uint ek_l[16];
    uint ek_r[16];
} template_pre_salt_state;

static inline void template_pre_salt(const uchar *data, int len,
                                     uint algo_mode,
                                     template_pre_salt_state *pre
#ifdef GPU_TEMPLATE_HAS_SHARED_LOCAL
                                     , __local const uint *shared_tbl
#endif
                                     )
{
    (void)algo_mode;
    uint khi, klo;
    descrypt_build_key(data, len, &khi, &klo);
#if DESCRYPT_KS_TABLE
    descrypt_des_key_schedule(khi, klo, pre->ek_l, pre->ek_r,
                              shared_tbl + DESCRYPT_KS_OFF);
#else
    descrypt_des_key_schedule(khi, klo, pre->ek_l, pre->ek_r);
#ifdef GPU_TEMPLATE_HAS_SHARED_LOCAL
    (void)shared_tbl;
#endif
#endif
}

static inline void template_finalize_post(template_state *st,
                                          const template_pre_salt_state *pre,
                                          const uchar *data, int len,
                                          __global const uchar *salt_bytes,
                                          uint salt_len,
                                          uint algo_mode
#ifdef GPU_TEMPLATE_HAS_SHARED_LOCAL
                                          , __local const uint *shared_tbl
#endif
                                          )
{
    (void)data;
    (void)len;
    (void)salt_len;
    (void)algo_mode;
#ifdef GPU_TEMPLATE_HAS_SHARED_LOCAL
    descrypt_run(st, pre->ek_l, pre->ek_r, salt_bytes, shared_tbl);
#else
    descrypt_run(st, pre->ek_l, pre->ek_r, salt_bytes);
#endif
}
#endif /* GPU_TEMPLATE_HAS_PRE_SALT && GPU_TEMPLATE_HAS_SALT */

/* template_finalize: full DESCRYPT chain (key build -> schedule -> 25
 * DES iterations -> pre-FP (l, r) install).  Under
 * GPU_TEMPLATE_HAS_PRE_SALT the template does not call this -- it calls
 * template_pre_salt + template_finalize_post above -- but it stays
 * compiled and correct so the two paths can be diffed. */
static inline void template_finalize(template_state *st,
                                     const uchar *data, int len
#ifdef GPU_TEMPLATE_HAS_SALT
                                     , __global const uchar *salt_bytes
                                     , uint salt_len
                                     , uint algo_mode
#endif
#ifdef GPU_TEMPLATE_HAS_SHARED_LOCAL
                                     , __local const uint *shared_tbl
#endif
                                     )
{
#ifdef GPU_TEMPLATE_HAS_SALT
    (void)algo_mode;
    (void)salt_len;

    uint khi, klo;
    descrypt_build_key(data, len, &khi, &klo);

    uint ek_l[16], ek_r[16];
#if DESCRYPT_KS_TABLE
    descrypt_des_key_schedule(khi, klo, ek_l, ek_r,
                              shared_tbl + DESCRYPT_KS_OFF);
#else
    descrypt_des_key_schedule(khi, klo, ek_l, ek_r);
#endif

#ifdef GPU_TEMPLATE_HAS_SHARED_LOCAL
    descrypt_run(st, ek_l, ek_r, salt_bytes, shared_tbl);
#else
    descrypt_run(st, ek_l, ek_r, salt_bytes);
#endif
    return;
#else
    /* Defensive fall-through for !HAS_SALT. DESCRYPT is always salted
     * (the algorithm requires the 2-char phpitoa64 salt for the F-round
     * E-expansion mix); a no-salt build would have nothing to do. Set
     * state to zero and return. */
    (void)data;
    (void)len;
#ifdef GPU_TEMPLATE_HAS_SHARED_LOCAL
    (void)shared_tbl;
#endif
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
