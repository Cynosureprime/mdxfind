/* gpu_bcrypt_core.cl -- BCRYPT (e450) algorithm extension functions for
 * the generic dispatch template (Unix-crypt Phase 6: hand-written Path A
 * salted-template kernel with workgroup-shared __local Eksblowfish state;
 * ports the slab gpu_bcrypt.cl algorithm body into the unified template
 * extension API).
 *
 * $Revision: 1.1 $
 * $Log: gpu_bcrypt_core.cl,v $
 * Revision 1.1  2026/05/09 05:01:40  dlr
 * BCRYPT (e450) Phase 6: hand-written Path A core kernel for unified template path. HASH_WORDS=6 (24-byte digest, first 4 LE words probe + all 6 emit). algo_mode=8 (reserves 8-15 for BCRYPT family). 8-arg template_finalize signature (gated on GPU_TEMPLATE_HAS_LOCAL_BUFFER): receives __local *sbox_pool + lid for Eksblowfish state. Per-lane 1024-uint slab partition (4*256 S-boxes) at sbox_pool + lid*1024. Per-salt-string cost factor parsing (slab pattern; SIMT divergence accepted). 2^cost Eksblowfish loop INSIDE template_finalize; max_iter=1 forced host-side. BCRYPT_INIT_P + BCRYPT_INIT_S0..S3 inlined (~267 LOC of constants). Standard variants [abxy]; k unsupported (slen!=29 reject). 72-byte key truncation defensive cap. Final algorithm in Phase 6 slab-retirement ladder.
 *
 *
 * STATUS: BCRYPT Phase 6 (2026-05-08) -- final slab algo to migrate from
 * slab to template path. Phases 1-5 preceded (MD5CRYPT / SHA256CRYPT /
 * SHA512CRYPT / SHA512CRYPTMD5 / DESCRYPT). Bespoke kernel; algo_mode=8u
 * for ALL FOUR ops in this carrier (BCRYPT/BCRYPTMD5/BCRYPTSHA1/
 * BCRYPTSHA512 -- compound siblings host-preprocess to 32/40/128-char hex
 * then fall through to bcrypt's BF_set_key cycling). algo_mode 8-15 is
 * RESERVED for future BCRYPT-family kernel-side variants.
 *
 * BCRYPT semantics (mirrors mdxfind.c JOB_BCRYPT at lines 14169-14230 +
 * the slab oracle gpu_bcrypt.cl):
 *
 *   salt buffer (28 or 29 bytes): "$2[abkxy]$NN$<22-base64-chars>" or
 *                                 "$2k$NN$<21-base64-chars>"
 *   cost = 2-digit decimal at offset 4..5 (clamped 4..31)
 *   raw_salt = bf_decode_salt(salt+7) -> 16 raw bytes
 *
 *   Eksblowfish key schedule + 2^cost iterations + final encrypt of
 *   "OrpheanBeholderScryDoubt" 64 times yields 24 bytes of output =
 *   6 BE uint32. Swap to LE for compact-table probe (HASH_WORDS=6,
 *   first 4 LE words probed = 16 bytes).
 *
 *   Truncation: BCRYPT silently truncates passwords > 72 bytes via
 *   BF_set_key key-cycling natural drop semantics (matches CPU
 *   crypt_blowfish/wrapper.c:288 "chars after 72 are ignored").
 *
 * DESIGN: 2^cost iter loop INSIDE template_finalize, max_iter=1.
 * --------------------------------------------------------------------
 * Mirrors PHPBB3 / MD5CRYPT / SHA256CRYPT / SHA512CRYPT / DESCRYPT
 * pattern (Unix-crypt ladder Phases 1-5 precedent):
 *   - the 2^cost iteration count is INTERNAL to the algorithm (parsed
 *     per-salt-string); NOT user-controlled via -i;
 *   - only the FINAL state is probed (CPU semantics: bcrypt yields a
 *     single 24-byte digest);
 *   - host forces params.max_iter = 1 at the rules-engine pack site so
 *     the kernel's outer iter loop runs exactly once and never calls
 *     template_iterate (which is a stub).
 *
 * Truncation strategy (HYBRID host+kernel, per architect §1):
 *   - HOST-side at the rules-engine pack site (mdxfind.c ~11082): when
 *     family is BCRYPT and pack_len > 72, clamp pack_len = 72 before
 *     the (char)len write + memcpy. Honors user's "prior to dispatching"
 *     intent for the no-rule synthetic pass.
 *   - KERNEL-side IMPLICIT via BF_set_key cycling: post-rule outputs >
 *     72 bytes get silently dropped because the inner cycling pointer
 *     reads pw[0..keylen-1] cyclically, where keylen is clamped <= 72
 *     by the defensive `if (len > 72) len = 72;` below.
 *   - Net: GPU and CPU produce IDENTICAL results for any input length.
 *     Two distinct rule outputs that differ only in bytes 73+ collide
 *     on the same bcrypt output (acceptable -- same as CPU).
 *
 * Salt-axis carrier: this kernel routes through the salted-template
 * scaffolding (GPU_TEMPLATE_HAS_SALT=1, SALT_POSITION=PREPEND in
 * defines_str). The salt buffer carries the FULL 28- or 29-byte
 * "$2[abkxy]$NN$<base64>..." string; per-string parsing (cost decode,
 * variant prefix recognition, base64 decode) happens INSIDE the kernel
 * (mirrors slab gpu_bcrypt.cl lines 459-487 byte-for-byte).
 *
 * Local memory layout (R1 mitigation -- Eksblowfish state is ~4KB/lane):
 * --------------------------------------------------------------------
 * The S-boxes (4 x 256 uint = 1024 uints = 4 KB per lane) live in
 * workgroup-shared __local memory at offset (lid * 1024). Each lane
 * owns a private 1024-uint partition; no inter-lane synchronization
 * required (init copies from __constant per-lane; no shared write).
 * The total __local footprint is BCRYPT_WG_SIZE (8) * 1024 uint = 32 KB
 * per workgroup, which fits Pascal (48 KB), RDNA (64 KB), and Mali-T860
 * (32 KB cap exactly).
 *
 * The __local *sbox_pool argument is supplied by the SCAFFOLD edit in
 * gpu_template.cl (declares __local at kernel-function scope per
 * OpenCL 1.2 §6.5.3, which forbids __local in non-kernel functions).
 * The 8-arg template_finalize signature is gated behind
 * GPU_TEMPLATE_HAS_LOCAL_BUFFER which only the BCRYPT carrier compile
 * defines via build_opts (ifdef-elision keeps non-BCRYPT instantiations
 * byte-identical to pre-Phase-6 -- R-S1 verified 2026-05-08).
 *
 * P-array stays in private memory (18 uints = 72 bytes, fits registers,
 * accessed sequentially not randomly). Comparable to the slab's
 * register-resident P-array.
 *
 * Inside the kernel:
 *   - data[0..len) = post-rule password (truncated to <= 72)
 *   - salt_bytes[0..salt_len) = full "$2[abkxy]$NN$..." salt string
 *   - sbox_pool[lid*1024 .. lid*1024+1024) = per-lane S0/S1/S2/S3 partition
 *
 * Validation oracle: gpu_bcrypt.cl bcrypt_batch (slab kernel; this file
 * is a port of that body into the template extension API).
 * Differences from slab:
 *   - reads pass from `data` (post-rule buf passed by template; PRIVATE
 *     uchar *) rather than from a hexhashes buffer indexed by word_idx;
 *   - reads salt from `salt_bytes` (the global salt buffer threaded by
 *     gpu_template.cl under GPU_TEMPLATE_HAS_SALT) rather than from the
 *     slab's own salts/salt_offsets/salt_lens trio;
 *   - probes via template_digest_compare's probe_compact_idx using
 *     HASH_WORDS=6 with all 6 LE words available in st->h.
 *
 * Hit replay: host calls bf_encode_23 (NEW helper in gpujob_opencl.c,
 * authored by B2) to reconstruct the 60-char $2b$ crypt hash, probes
 * JudyJ[JOB_BCRYPT] for the line, applies CAS dedup, and emits via
 * prfound. Display password is FULL post-rule plaintext (CPU does NOT
 * clamp display for BCRYPT -- the truncation is INSIDE BF_set_key,
 * not at display). Q1 user decision 2026-05-08.
 *
 * R1 mitigation (AMD ROCm comgr addrspace fragility): all S-box reads
 * use the BF_ROUND/BF_ENCRYPT macros parameterized over __local
 * pointers; no addrspace casts. The expanded_key array stays private.
 * The salt_w[4] decoded raw-salt array stays private.
 *
 * R2 (register pressure): 18 uint P-array + 18 uint expanded_key + 4
 * uint salt_w + 6 uint ctext + 16 uint raw-salt buffer (max). Total
 * private: ~62 uints = 248 bytes. Comparable to PHPBB3/MD5CRYPT.
 * Expected priv_mem on Pascal in the 41-43 KB band shared by other
 * unified-template dispatches (BCRYPT adds ~2-4 KB over baseline due
 * to the 18-element P-array round-key storage).
 *
 * Source order at compile time:
 *   [ gpu_common_str, gpu_md5_rules_str, gpu_bcrypt_core_str,
 *     gpu_template_str ]
 *
 * Cache key (R3): defines_str =
 *   "HASH_WORDS=6,HASH_BLOCK_BYTES=64,HAS_SALT=1,SALT_POSITION=PREPEND,
 *    BASE_ALGO=bcrypt,GPU_TEMPLATE_HAS_LOCAL_BUFFER=1,
 *    GPU_TEMPLATE_LOCAL_BUFFER_PER_LANE=1024,BCRYPT_WG_SIZE=8"
 *
 * Cache-key disambiguation: BASE_ALGO=bcrypt + HASH_WORDS=6 uniquely
 * identifies this template instantiation. No other algo uses
 * HASH_WORDS=6 + HAS_SALT=1.
 *
 * algo_mode: BCRYPT family uses algo_mode=8 (next free after DESCRYPT's
 * 7). All FOUR ops (BCRYPT/BCRYPTMD5/BCRYPTSHA1/BCRYPTSHA512) share
 * algo_mode=8 because the kernel logic is IDENTICAL -- the host preprocesses
 * the input (MD5/SHA1/SHA512 hashing -> hex string) before the rules-engine
 * pack site, so by the time the kernel sees the data buffer, all four ops
 * are bcrypt(hex_string_or_password). Reserved range 8-15 for future
 * BCRYPT-family variants requiring kernel-side preprocessing.
 */

#ifndef HASH_WORDS
#define HASH_WORDS 6
#endif
#ifndef HASH_BLOCK_BYTES
#define HASH_BLOCK_BYTES 64
#endif

/* ---- Blowfish init state from Pi digits (ported from slab gpu_bcrypt.cl) ---- */

__constant uint BCRYPT_INIT_P[18] = {
    0x243f6a88, 0x85a308d3, 0x13198a2e, 0x03707344,
    0xa4093822, 0x299f31d0, 0x082efa98, 0xec4e6c89,
    0x452821e6, 0x38d01377, 0xbe5466cf, 0x34e90c6c,
    0xc0ac29b7, 0xc97c50dd, 0x3f84d5b5, 0xb5470917,
    0x9216d5d9, 0x8979fb1b
};

__constant uint BCRYPT_INIT_S0[256] = {
    0xd1310ba6, 0x98dfb5ac, 0x2ffd72db, 0xd01adfb7,
    0xb8e1afed, 0x6a267e96, 0xba7c9045, 0xf12c7f99,
    0x24a19947, 0xb3916cf7, 0x0801f2e2, 0x858efc16,
    0x636920d8, 0x71574e69, 0xa458fea3, 0xf4933d7e,
    0x0d95748f, 0x728eb658, 0x718bcd58, 0x82154aee,
    0x7b54a41d, 0xc25a59b5, 0x9c30d539, 0x2af26013,
    0xc5d1b023, 0x286085f0, 0xca417918, 0xb8db38ef,
    0x8e79dcb0, 0x603a180e, 0x6c9e0e8b, 0xb01e8a3e,
    0xd71577c1, 0xbd314b27, 0x78af2fda, 0x55605c60,
    0xe65525f3, 0xaa55ab94, 0x57489862, 0x63e81440,
    0x55ca396a, 0x2aab10b6, 0xb4cc5c34, 0x1141e8ce,
    0xa15486af, 0x7c72e993, 0xb3ee1411, 0x636fbc2a,
    0x2ba9c55d, 0x741831f6, 0xce5c3e16, 0x9b87931e,
    0xafd6ba33, 0x6c24cf5c, 0x7a325381, 0x28958677,
    0x3b8f4898, 0x6b4bb9af, 0xc4bfe81b, 0x66282193,
    0x61d809cc, 0xfb21a991, 0x487cac60, 0x5dec8032,
    0xef845d5d, 0xe98575b1, 0xdc262302, 0xeb651b88,
    0x23893e81, 0xd396acc5, 0x0f6d6ff3, 0x83f44239,
    0x2e0b4482, 0xa4842004, 0x69c8f04a, 0x9e1f9b5e,
    0x21c66842, 0xf6e96c9a, 0x670c9c61, 0xabd388f0,
    0x6a51a0d2, 0xd8542f68, 0x960fa728, 0xab5133a3,
    0x6eef0b6c, 0x137a3be4, 0xba3bf050, 0x7efb2a98,
    0xa1f1651d, 0x39af0176, 0x66ca593e, 0x82430e88,
    0x8cee8619, 0x456f9fb4, 0x7d84a5c3, 0x3b8b5ebe,
    0xe06f75d8, 0x85c12073, 0x401a449f, 0x56c16aa6,
    0x4ed3aa62, 0x363f7706, 0x1bfedf72, 0x429b023d,
    0x37d0d724, 0xd00a1248, 0xdb0fead3, 0x49f1c09b,
    0x075372c9, 0x80991b7b, 0x25d479d8, 0xf6e8def7,
    0xe3fe501a, 0xb6794c3b, 0x976ce0bd, 0x04c006ba,
    0xc1a94fb6, 0x409f60c4, 0x5e5c9ec2, 0x196a2463,
    0x68fb6faf, 0x3e6c53b5, 0x1339b2eb, 0x3b52ec6f,
    0x6dfc511f, 0x9b30952c, 0xcc814544, 0xaf5ebd09,
    0xbee3d004, 0xde334afd, 0x660f2807, 0x192e4bb3,
    0xc0cba857, 0x45c8740f, 0xd20b5f39, 0xb9d3fbdb,
    0x5579c0bd, 0x1a60320a, 0xd6a100c6, 0x402c7279,
    0x679f25fe, 0xfb1fa3cc, 0x8ea5e9f8, 0xdb3222f8,
    0x3c7516df, 0xfd616b15, 0x2f501ec8, 0xad0552ab,
    0x323db5fa, 0xfd238760, 0x53317b48, 0x3e00df82,
    0x9e5c57bb, 0xca6f8ca0, 0x1a87562e, 0xdf1769db,
    0xd542a8f6, 0x287effc3, 0xac6732c6, 0x8c4f5573,
    0x695b27b0, 0xbbca58c8, 0xe1ffa35d, 0xb8f011a0,
    0x10fa3d98, 0xfd2183b8, 0x4afcb56c, 0x2dd1d35b,
    0x9a53e479, 0xb6f84565, 0xd28e49bc, 0x4bfb9790,
    0xe1ddf2da, 0xa4cb7e33, 0x62fb1341, 0xcee4c6e8,
    0xef20cada, 0x36774c01, 0xd07e9efe, 0x2bf11fb4,
    0x95dbda4d, 0xae909198, 0xeaad8e71, 0x6b93d5a0,
    0xd08ed1d0, 0xafc725e0, 0x8e3c5b2f, 0x8e7594b7,
    0x8ff6e2fb, 0xf2122b64, 0x8888b812, 0x900df01c,
    0x4fad5ea0, 0x688fc31c, 0xd1cff191, 0xb3a8c1ad,
    0x2f2f2218, 0xbe0e1777, 0xea752dfe, 0x8b021fa1,
    0xe5a0cc0f, 0xb56f74e8, 0x18acf3d6, 0xce89e299,
    0xb4a84fe0, 0xfd13e0b7, 0x7cc43b81, 0xd2ada8d9,
    0x165fa266, 0x80957705, 0x93cc7314, 0x211a1477,
    0xe6ad2065, 0x77b5fa86, 0xc75442f5, 0xfb9d35cf,
    0xebcdaf0c, 0x7b3e89a0, 0xd6411bd3, 0xae1e7e49,
    0x00250e2d, 0x2071b35e, 0x226800bb, 0x57b8e0af,
    0x2464369b, 0xf009b91e, 0x5563911d, 0x59dfa6aa,
    0x78c14389, 0xd95a537f, 0x207d5ba2, 0x02e5b9c5,
    0x83260376, 0x6295cfa9, 0x11c81968, 0x4e734a41,
    0xb3472dca, 0x7b14a94a, 0x1b510052, 0x9a532915,
    0xd60f573f, 0xbc9bc6e4, 0x2b60a476, 0x81e67400,
    0x08ba6fb5, 0x571be91f, 0xf296ec6b, 0x2a0dd915,
    0xb6636521, 0xe7b9f9b6, 0xff34052e, 0xc5855664,
    0x53b02d5d, 0xa99f8fa1, 0x08ba4799, 0x6e85076a
};

__constant uint BCRYPT_INIT_S1[256] = {
    0x4b7a70e9, 0xb5b32944, 0xdb75092e, 0xc4192623,
    0xad6ea6b0, 0x49a7df7d, 0x9cee60b8, 0x8fedb266,
    0xecaa8c71, 0x699a17ff, 0x5664526c, 0xc2b19ee1,
    0x193602a5, 0x75094c29, 0xa0591340, 0xe4183a3e,
    0x3f54989a, 0x5b429d65, 0x6b8fe4d6, 0x99f73fd6,
    0xa1d29c07, 0xefe830f5, 0x4d2d38e6, 0xf0255dc1,
    0x4cdd2086, 0x8470eb26, 0x6382e9c6, 0x021ecc5e,
    0x09686b3f, 0x3ebaefc9, 0x3c971814, 0x6b6a70a1,
    0x687f3584, 0x52a0e286, 0xb79c5305, 0xaa500737,
    0x3e07841c, 0x7fdeae5c, 0x8e7d44ec, 0x5716f2b8,
    0xb03ada37, 0xf0500c0d, 0xf01c1f04, 0x0200b3ff,
    0xae0cf51a, 0x3cb574b2, 0x25837a58, 0xdc0921bd,
    0xd19113f9, 0x7ca92ff6, 0x94324773, 0x22f54701,
    0x3ae5e581, 0x37c2dadc, 0xc8b57634, 0x9af3dda7,
    0xa9446146, 0x0fd0030e, 0xecc8c73e, 0xa4751e41,
    0xe238cd99, 0x3bea0e2f, 0x3280bba1, 0x183eb331,
    0x4e548b38, 0x4f6db908, 0x6f420d03, 0xf60a04bf,
    0x2cb81290, 0x24977c79, 0x5679b072, 0xbcaf89af,
    0xde9a771f, 0xd9930810, 0xb38bae12, 0xdccf3f2e,
    0x5512721f, 0x2e6b7124, 0x501adde6, 0x9f84cd87,
    0x7a584718, 0x7408da17, 0xbc9f9abc, 0xe94b7d8c,
    0xec7aec3a, 0xdb851dfa, 0x63094366, 0xc464c3d2,
    0xef1c1847, 0x3215d908, 0xdd433b37, 0x24c2ba16,
    0x12a14d43, 0x2a65c451, 0x50940002, 0x133ae4dd,
    0x71dff89e, 0x10314e55, 0x81ac77d6, 0x5f11199b,
    0x043556f1, 0xd7a3c76b, 0x3c11183b, 0x5924a509,
    0xf28fe6ed, 0x97f1fbfa, 0x9ebabf2c, 0x1e153c6e,
    0x86e34570, 0xeae96fb1, 0x860e5e0a, 0x5a3e2ab3,
    0x771fe71c, 0x4e3d06fa, 0x2965dcb9, 0x99e71d0f,
    0x803e89d6, 0x5266c825, 0x2e4cc978, 0x9c10b36a,
    0xc6150eba, 0x94e2ea78, 0xa5fc3c53, 0x1e0a2df4,
    0xf2f74ea7, 0x361d2b3d, 0x1939260f, 0x19c27960,
    0x5223a708, 0xf71312b6, 0xebadfe6e, 0xeac31f66,
    0xe3bc4595, 0xa67bc883, 0xb17f37d1, 0x018cff28,
    0xc332ddef, 0xbe6c5aa5, 0x65582185, 0x68ab9802,
    0xeecea50f, 0xdb2f953b, 0x2aef7dad, 0x5b6e2f84,
    0x1521b628, 0x29076170, 0xecdd4775, 0x619f1510,
    0x13cca830, 0xeb61bd96, 0x0334fe1e, 0xaa0363cf,
    0xb5735c90, 0x4c70a239, 0xd59e9e0b, 0xcbaade14,
    0xeecc86bc, 0x60622ca7, 0x9cab5cab, 0xb2f3846e,
    0x648b1eaf, 0x19bdf0ca, 0xa02369b9, 0x655abb50,
    0x40685a32, 0x3c2ab4b3, 0x319ee9d5, 0xc021b8f7,
    0x9b540b19, 0x875fa099, 0x95f7997e, 0x623d7da8,
    0xf837889a, 0x97e32d77, 0x11ed935f, 0x16681281,
    0x0e358829, 0xc7e61fd6, 0x96dedfa1, 0x7858ba99,
    0x57f584a5, 0x1b227263, 0x9b83c3ff, 0x1ac24696,
    0xcdb30aeb, 0x532e3054, 0x8fd948e4, 0x6dbc3128,
    0x58ebf2ef, 0x34c6ffea, 0xfe28ed61, 0xee7c3c73,
    0x5d4a14d9, 0xe864b7e3, 0x42105d14, 0x203e13e0,
    0x45eee2b6, 0xa3aaabea, 0xdb6c4f15, 0xfacb4fd0,
    0xc742f442, 0xef6abbb5, 0x654f3b1d, 0x41cd2105,
    0xd81e799e, 0x86854dc7, 0xe44b476a, 0x3d816250,
    0xcf62a1f2, 0x5b8d2646, 0xfc8883a0, 0xc1c7b6a3,
    0x7f1524c3, 0x69cb7492, 0x47848a0b, 0x5692b285,
    0x095bbf00, 0xad19489d, 0x1462b174, 0x23820e00,
    0x58428d2a, 0x0c55f5ea, 0x1dadf43e, 0x233f7061,
    0x3372f092, 0x8d937e41, 0xd65fecf1, 0x6c223bdb,
    0x7cde3759, 0xcbee7460, 0x4085f2a7, 0xce77326e,
    0xa6078084, 0x19f8509e, 0xe8efd855, 0x61d99735,
    0xa969a7aa, 0xc50c06c2, 0x5a04abfc, 0x800bcadc,
    0x9e447a2e, 0xc3453484, 0xfdd56705, 0x0e1e9ec9,
    0xdb73dbd3, 0x105588cd, 0x675fda79, 0xe3674340,
    0xc5c43465, 0x713e38d8, 0x3d28f89e, 0xf16dff20,
    0x153e21e7, 0x8fb03d4a, 0xe6e39f2b, 0xdb83adf7
};

__constant uint BCRYPT_INIT_S2[256] = {
    0xe93d5a68, 0x948140f7, 0xf64c261c, 0x94692934,
    0x411520f7, 0x7602d4f7, 0xbcf46b2e, 0xd4a20068,
    0xd4082471, 0x3320f46a, 0x43b7d4b7, 0x500061af,
    0x1e39f62e, 0x97244546, 0x14214f74, 0xbf8b8840,
    0x4d95fc1d, 0x96b591af, 0x70f4ddd3, 0x66a02f45,
    0xbfbc09ec, 0x03bd9785, 0x7fac6dd0, 0x31cb8504,
    0x96eb27b3, 0x55fd3941, 0xda2547e6, 0xabca0a9a,
    0x28507825, 0x530429f4, 0x0a2c86da, 0xe9b66dfb,
    0x68dc1462, 0xd7486900, 0x680ec0a4, 0x27a18dee,
    0x4f3ffea2, 0xe887ad8c, 0xb58ce006, 0x7af4d6b6,
    0xaace1e7c, 0xd3375fec, 0xce78a399, 0x406b2a42,
    0x20fe9e35, 0xd9f385b9, 0xee39d7ab, 0x3b124e8b,
    0x1dc9faf7, 0x4b6d1856, 0x26a36631, 0xeae397b2,
    0x3a6efa74, 0xdd5b4332, 0x6841e7f7, 0xca7820fb,
    0xfb0af54e, 0xd8feb397, 0x454056ac, 0xba489527,
    0x55533a3a, 0x20838d87, 0xfe6ba9b7, 0xd096954b,
    0x55a867bc, 0xa1159a58, 0xcca92963, 0x99e1db33,
    0xa62a4a56, 0x3f3125f9, 0x5ef47e1c, 0x9029317c,
    0xfdf8e802, 0x04272f70, 0x80bb155c, 0x05282ce3,
    0x95c11548, 0xe4c66d22, 0x48c1133f, 0xc70f86dc,
    0x07f9c9ee, 0x41041f0f, 0x404779a4, 0x5d886e17,
    0x325f51eb, 0xd59bc0d1, 0xf2bcc18f, 0x41113564,
    0x257b7834, 0x602a9c60, 0xdff8e8a3, 0x1f636c1b,
    0x0e12b4c2, 0x02e1329e, 0xaf664fd1, 0xcad18115,
    0x6b2395e0, 0x333e92e1, 0x3b240b62, 0xeebeb922,
    0x85b2a20e, 0xe6ba0d99, 0xde720c8c, 0x2da2f728,
    0xd0127845, 0x95b794fd, 0x647d0862, 0xe7ccf5f0,
    0x5449a36f, 0x877d48fa, 0xc39dfd27, 0xf33e8d1e,
    0x0a476341, 0x992eff74, 0x3a6f6eab, 0xf4f8fd37,
    0xa812dc60, 0xa1ebddf8, 0x991be14c, 0xdb6e6b0d,
    0xc67b5510, 0x6d672c37, 0x2765d43b, 0xdcd0e804,
    0xf1290dc7, 0xcc00ffa3, 0xb5390f92, 0x690fed0b,
    0x667b9ffb, 0xcedb7d9c, 0xa091cf0b, 0xd9155ea3,
    0xbb132f88, 0x515bad24, 0x7b9479bf, 0x763bd6eb,
    0x37392eb3, 0xcc115979, 0x8026e297, 0xf42e312d,
    0x6842ada7, 0xc66a2b3b, 0x12754ccc, 0x782ef11c,
    0x6a124237, 0xb79251e7, 0x06a1bbe6, 0x4bfb6350,
    0x1a6b1018, 0x11caedfa, 0x3d25bdd8, 0xe2e1c3c9,
    0x44421659, 0x0a121386, 0xd90cec6e, 0xd5abea2a,
    0x64af674e, 0xda86a85f, 0xbebfe988, 0x64e4c3fe,
    0x9dbc8057, 0xf0f7c086, 0x60787bf8, 0x6003604d,
    0xd1fd8346, 0xf6381fb0, 0x7745ae04, 0xd736fccc,
    0x83426b33, 0xf01eab71, 0xb0804187, 0x3c005e5f,
    0x77a057be, 0xbde8ae24, 0x55464299, 0xbf582e61,
    0x4e58f48f, 0xf2ddfda2, 0xf474ef38, 0x8789bdc2,
    0x5366f9c3, 0xc8b38e74, 0xb475f255, 0x46fcd9b9,
    0x7aeb2661, 0x8b1ddf84, 0x846a0e79, 0x915f95e2,
    0x466e598e, 0x20b45770, 0x8cd55591, 0xc902de4c,
    0xb90bace1, 0xbb8205d0, 0x11a86248, 0x7574a99e,
    0xb77f19b6, 0xe0a9dc09, 0x662d09a1, 0xc4324633,
    0xe85a1f02, 0x09f0be8c, 0x4a99a025, 0x1d6efe10,
    0x1ab93d1d, 0x0ba5a4df, 0xa186f20f, 0x2868f169,
    0xdcb7da83, 0x573906fe, 0xa1e2ce9b, 0x4fcd7f52,
    0x50115e01, 0xa70683fa, 0xa002b5c4, 0x0de6d027,
    0x9af88c27, 0x773f8641, 0xc3604c06, 0x61a806b5,
    0xf0177a28, 0xc0f586e0, 0x006058aa, 0x30dc7d62,
    0x11e69ed7, 0x2338ea63, 0x53c2dd94, 0xc2c21634,
    0xbbcbee56, 0x90bcb6de, 0xebfc7da1, 0xce591d76,
    0x6f05e409, 0x4b7c0188, 0x39720a3d, 0x7c927c24,
    0x86e3725f, 0x724d9db9, 0x1ac15bb4, 0xd39eb8fc,
    0xed545578, 0x08fca5b5, 0xd83d7cd3, 0x4dad0fc4,
    0x1e50ef5e, 0xb161e6f8, 0xa28514d9, 0x6c51133c,
    0x6fd5c7e7, 0x56e14ec4, 0x362abfce, 0xddc6c837,
    0xd79a3234, 0x92638212, 0x670efa8e, 0x406000e0
};

__constant uint BCRYPT_INIT_S3[256] = {
    0x3a39ce37, 0xd3faf5cf, 0xabc27737, 0x5ac52d1b,
    0x5cb0679e, 0x4fa33742, 0xd3822740, 0x99bc9bbe,
    0xd5118e9d, 0xbf0f7315, 0xd62d1c7e, 0xc700c47b,
    0xb78c1b6b, 0x21a19045, 0xb26eb1be, 0x6a366eb4,
    0x5748ab2f, 0xbc946e79, 0xc6a376d2, 0x6549c2c8,
    0x530ff8ee, 0x468dde7d, 0xd5730a1d, 0x4cd04dc6,
    0x2939bbdb, 0xa9ba4650, 0xac9526e8, 0xbe5ee304,
    0xa1fad5f0, 0x6a2d519a, 0x63ef8ce2, 0x9a86ee22,
    0xc089c2b8, 0x43242ef6, 0xa51e03aa, 0x9cf2d0a4,
    0x83c061ba, 0x9be96a4d, 0x8fe51550, 0xba645bd6,
    0x2826a2f9, 0xa73a3ae1, 0x4ba99586, 0xef5562e9,
    0xc72fefd3, 0xf752f7da, 0x3f046f69, 0x77fa0a59,
    0x80e4a915, 0x87b08601, 0x9b09e6ad, 0x3b3ee593,
    0xe990fd5a, 0x9e34d797, 0x2cf0b7d9, 0x022b8b51,
    0x96d5ac3a, 0x017da67d, 0xd1cf3ed6, 0x7c7d2d28,
    0x1f9f25cf, 0xadf2b89b, 0x5ad6b472, 0x5a88f54c,
    0xe029ac71, 0xe019a5e6, 0x47b0acfd, 0xed93fa9b,
    0xe8d3c48d, 0x283b57cc, 0xf8d56629, 0x79132e28,
    0x785f0191, 0xed756055, 0xf7960e44, 0xe3d35e8c,
    0x15056dd4, 0x88f46dba, 0x03a16125, 0x0564f0bd,
    0xc3eb9e15, 0x3c9057a2, 0x97271aec, 0xa93a072a,
    0x1b3f6d9b, 0x1e6321f5, 0xf59c66fb, 0x26dcf319,
    0x7533d928, 0xb155fdf5, 0x03563482, 0x8aba3cbb,
    0x28517711, 0xc20ad9f8, 0xabcc5167, 0xccad925f,
    0x4de81751, 0x3830dc8e, 0x379d5862, 0x9320f991,
    0xea7a90c2, 0xfb3e7bce, 0x5121ce64, 0x774fbe32,
    0xa8b6e37e, 0xc3293d46, 0x48de5369, 0x6413e680,
    0xa2ae0810, 0xdd6db224, 0x69852dfd, 0x09072166,
    0xb39a460a, 0x6445c0dd, 0x586cdecf, 0x1c20c8ae,
    0x5bbef7dd, 0x1b588d40, 0xccd2017f, 0x6bb4e3bb,
    0xdda26a7e, 0x3a59ff45, 0x3e350a44, 0xbcb4cdd5,
    0x72eacea8, 0xfa6484bb, 0x8d6612ae, 0xbf3c6f47,
    0xd29be463, 0x542f5d9e, 0xaec2771b, 0xf64e6370,
    0x740e0d8d, 0xe75b1357, 0xf8721671, 0xaf537d5d,
    0x4040cb08, 0x4eb4e2cc, 0x34d2466a, 0x0115af84,
    0xe1b00428, 0x95983a1d, 0x06b89fb4, 0xce6ea048,
    0x6f3f3b82, 0x3520ab82, 0x011a1d4b, 0x277227f8,
    0x611560b1, 0xe7933fdc, 0xbb3a792b, 0x344525bd,
    0xa08839e1, 0x51ce794b, 0x2f32c9b7, 0xa01fbac9,
    0xe01cc87e, 0xbcc7d1f6, 0xcf0111c3, 0xa1e8aac7,
    0x1a908749, 0xd44fbd9a, 0xd0dadecb, 0xd50ada38,
    0x0339c32a, 0xc6913667, 0x8df9317c, 0xe0b12b4f,
    0xf79e59b7, 0x43f5bb3a, 0xf2d519ff, 0x27d9459c,
    0xbf97222c, 0x15e6fc2a, 0x0f91fc71, 0x9b941525,
    0xfae59361, 0xceb69ceb, 0xc2a86459, 0x12baa8d1,
    0xb6c1075e, 0xe3056a0c, 0x10d25065, 0xcb03a442,
    0xe0ec6e0e, 0x1698db3b, 0x4c98a0be, 0x3278e964,
    0x9f1f9532, 0xe0d392df, 0xd3a0342b, 0x8971f21e,
    0x1b0a7441, 0x4ba3348c, 0xc5be7120, 0xc37632d8,
    0xdf359f8d, 0x9b992f2e, 0xe60b6f47, 0x0fe3f11d,
    0xe54cda54, 0x1edad891, 0xce6279cf, 0xcd3e7e6f,
    0x1618b166, 0xfd2c1d05, 0x848fd2c5, 0xf6fb2299,
    0xf523f357, 0xa6327623, 0x93a83531, 0x56cccd02,
    0xacf08162, 0x5a75ebb5, 0x6e163697, 0x88d273cc,
    0xde966292, 0x81b949d0, 0x4c50901b, 0x71c65614,
    0xe6c6c7bd, 0x327a140a, 0x45e1d006, 0xc3f27b9a,
    0xc9aa53fd, 0x62a80f00, 0xbb25bfe2, 0x35bdd2f6,
    0x71126905, 0xb2040222, 0xb6cbcf7c, 0xcd769c2b,
    0x53113ec0, 0x1640e3d3, 0x38abbd60, 0x2547adf0,
    0xba38209c, 0xf746ce76, 0x77afa1c5, 0x20756060,
    0x85cbfe4e, 0x8ae88dd8, 0x7aaaf9b0, 0x4cf9aa7e,
    0x1948c25c, 0x02fb8a8c, 0x01c36ae4, 0xd6ebe1f9,
    0x90d4f869, 0xa65cdea0, 0x3f09252d, 0xc208e69f,
    0xb74e6132, 0xce77e25b, 0x578fdfe3, 0x3ac372e6
};

/* ---- Blowfish primitives (ported from slab gpu_bcrypt.cl) ---- */

/* BCRYPT_BF_ROUND: F(x) = ((S0[a] + S1[b]) ^ S2[c]) + S3[d]
 * where x = (a:8 || b:8 || c:8 || d:8) big-endian. S* are __local
 * pointers (per-lane partition of sbox_pool); P is private. */
#define BCRYPT_BF_ROUND(L, R, N, S0, S1, S2, S3, P) \
{ \
    uint _bf_tmp = S0[(L >> 24) & 0xffu]; \
    _bf_tmp += S1[(L >> 16) & 0xffu]; \
    _bf_tmp ^= S2[(L >> 8) & 0xffu]; \
    _bf_tmp += S3[L & 0xffu]; \
    R ^= P[N + 1] ^ _bf_tmp; \
}

/* BCRYPT_BF_ENCRYPT: 16-round Blowfish encrypt (L,R) in place.
 * After the macro: L holds the left ciphertext word, R holds the right
 * ciphertext word. Matches reference swap at end: tmp=R; R=L;
 * L=tmp^P[17]. */
#define BCRYPT_BF_ENCRYPT(L, R, S0, S1, S2, S3, P) \
{ \
    L ^= P[0]; \
    BCRYPT_BF_ROUND(L, R,  0, S0, S1, S2, S3, P); \
    BCRYPT_BF_ROUND(R, L,  1, S0, S1, S2, S3, P); \
    BCRYPT_BF_ROUND(L, R,  2, S0, S1, S2, S3, P); \
    BCRYPT_BF_ROUND(R, L,  3, S0, S1, S2, S3, P); \
    BCRYPT_BF_ROUND(L, R,  4, S0, S1, S2, S3, P); \
    BCRYPT_BF_ROUND(R, L,  5, S0, S1, S2, S3, P); \
    BCRYPT_BF_ROUND(L, R,  6, S0, S1, S2, S3, P); \
    BCRYPT_BF_ROUND(R, L,  7, S0, S1, S2, S3, P); \
    BCRYPT_BF_ROUND(L, R,  8, S0, S1, S2, S3, P); \
    BCRYPT_BF_ROUND(R, L,  9, S0, S1, S2, S3, P); \
    BCRYPT_BF_ROUND(L, R, 10, S0, S1, S2, S3, P); \
    BCRYPT_BF_ROUND(R, L, 11, S0, S1, S2, S3, P); \
    BCRYPT_BF_ROUND(L, R, 12, S0, S1, S2, S3, P); \
    BCRYPT_BF_ROUND(R, L, 13, S0, S1, S2, S3, P); \
    BCRYPT_BF_ROUND(L, R, 14, S0, S1, S2, S3, P); \
    BCRYPT_BF_ROUND(R, L, 15, S0, S1, S2, S3, P); \
    uint _bf_tmp4 = R; \
    R = L; \
    L = _bf_tmp4 ^ P[17]; \
}

/* bcrypt_bf_body: encrypt (0,0) through all P entries, then all S
 * entries. Exact port of slab gpu_bcrypt.cl bf_body() function. */
static void bcrypt_bf_body(__local uint *S0, __local uint *S1,
                           __local uint *S2, __local uint *S3, uint *P)
{
    uint L = 0, R = 0;
    for (int i = 0; i < 18; i += 2) {
        BCRYPT_BF_ENCRYPT(L, R, S0, S1, S2, S3, P);
        P[i] = L;
        P[i + 1] = R;
    }
    for (int i = 0; i < 256; i += 2) {
        BCRYPT_BF_ENCRYPT(L, R, S0, S1, S2, S3, P);
        S0[i] = L;
        S0[i + 1] = R;
    }
    for (int i = 0; i < 256; i += 2) {
        BCRYPT_BF_ENCRYPT(L, R, S0, S1, S2, S3, P);
        S1[i] = L;
        S1[i + 1] = R;
    }
    for (int i = 0; i < 256; i += 2) {
        BCRYPT_BF_ENCRYPT(L, R, S0, S1, S2, S3, P);
        S2[i] = L;
        S2[i + 1] = R;
    }
    for (int i = 0; i < 256; i += 2) {
        BCRYPT_BF_ENCRYPT(L, R, S0, S1, S2, S3, P);
        S3[i] = L;
        S3[i + 1] = R;
    }
}

/* ---- bcrypt base64 decode (./A-Za-z0-9) ---- */

/* bcrypt base64 decode table: ASCII 0x2E ('.') through 0x7A ('z')
 * ./0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz = 0..63
 * Ported from slab gpu_bcrypt.cl bf_atoi64_tbl. */
__constant uchar BCRYPT_atoi64_tbl[77] = {
    /*  .   /   0   1   2   3   4   5   6   7   8   9   :   ;   <   = */
         0,  1, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 64, 64, 64,
    /*  >   ?   @   A   B   C   D   E   F   G   H   I   J   K   L   M */
        64, 64, 64,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14,
    /*  N   O   P   Q   R   S   T   U   V   W   X   Y   Z   [   \   ] */
        15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 64, 64, 64,
    /*  ^   _   `   a   b   c   d   e   f   g   h   i   j   k   l   m */
        64, 64, 64, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
    /*  n   o   p   q   r   s   t   u   v   w   x   y   z */
        41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53
};

#define BCRYPT_B64(c) BCRYPT_atoi64_tbl[(c) - 0x2E]

/* Decode 22 base64 chars -> 16 bytes of raw salt.
 * bcrypt base64: each group of 4 chars -> 3 bytes.
 * 22 chars = 5 full groups (15 bytes) + 2 remaining chars (1 byte).
 * Ported from slab gpu_bcrypt.cl bf_decode_salt. */
static void bcrypt_decode_salt(uchar *dst, __global const uchar *src)
{
    uint c1, c2, c3, c4;
    int si = 0, di = 0;

    /* 5 full groups of 4 chars -> 15 bytes */
    for (int g = 0; g < 5; g++) {
        c1 = BCRYPT_B64(src[si++]);
        c2 = BCRYPT_B64(src[si++]);
        c3 = BCRYPT_B64(src[si++]);
        c4 = BCRYPT_B64(src[si++]);
        dst[di++] = (uchar)((c1 << 2) | ((c2 & 0x30) >> 4));
        dst[di++] = (uchar)(((c2 & 0x0f) << 4) | ((c3 & 0x3c) >> 2));
        dst[di++] = (uchar)(((c3 & 0x03) << 6) | c4);
    }

    /* Last 2 chars -> 1 byte */
    c1 = BCRYPT_B64(src[si++]);
    c2 = BCRYPT_B64(src[si++]);
    dst[di] = (uchar)((c1 << 2) | ((c2 & 0x30) >> 4));
}

/* Per-lane state struct. BCRYPT emits 24 bytes = 6 LE uint32 words.
 * HASH_WORDS=6 stays the canonical width; first 4 words probe the
 * compact table (16 byte fingerprint), all 6 emit on hit. */
typedef struct {
    uint h[HASH_WORDS];
} template_state;

/* template_init: zero the state. BCRYPT initializes (L,R) inline in
 * template_finalize (matches slab gpu_bcrypt.cl pattern); the state
 * struct is filled at the end of template_finalize. */
static inline void template_init(template_state *st) {
    for (int i = 0; i < HASH_WORDS; i++) st->h[i] = 0u;
}

/* template_transform: stub for interface symmetry. BCRYPT's
 * template_finalize manages the full Eksblowfish state inline -- never
 * routes through this. Mirrors PHPBB3 / DESCRYPT pattern. */
static inline void template_transform(template_state *st, const uchar *block)
{
    (void)st;
    (void)block;
}

/* template_finalize: full bcrypt chain.
 *
 * 8-arg salted variant with workgroup-shared __local sbox_pool: the
 * scaffold (gpu_template.cl) declares the pool at kernel-function scope
 * (OpenCL 1.2 §6.5.3 forbids __local in non-kernel functions); each
 * lane claims its 1024-uint partition at sbox_pool + lid*1024.
 *
 * Step 1: defensive cap len <= 72 (matches CPU crypt_blowfish/wrapper.c
 *   :288 "chars after 72 are ignored"; host-side rules-engine pack site
 *   already clamps for the synthetic no-rule pass; clamp here too so
 *   masked / rule-extended outputs > 72 silently truncate via BF_set_key
 *   key-cycling natural drop semantics).
 *
 * Step 2: copy data[0..len) to private pw[72] buffer. bcrypt key
 *   includes the NUL terminator (matches slab :454-457 + reference
 *   crypt_blowfish.c BF_set_key); append NUL within 72-byte cap.
 *
 * Step 3: parse cost from salt_bytes[4..5] (2-digit decimal); decode
 *   16-byte raw salt from salt_bytes[7..28] via bcrypt_decode_salt
 *   (mirrors slab :459-487 byte-for-byte).
 *
 *   $2k$ variant has 21-byte base64 salt at offset 7 (slen_full=28
 *   instead of 29) but the kernel ignores variant prefix differences
 *   -- it decodes 22 bytes starting at offset 7 regardless. For $2k$
 *   the 22nd char would be the FIRST char of the hash, which is
 *   harmless because bcrypt host loader is responsible for ensuring
 *   correct salt-string format (variant prefix recognition is at
 *   compact-table-load time in mdxfind.c, NOT kernel time). NOTE:
 *   slab oracle accepts $2k$ rejects via slen_full < 29 guard at
 *   :475; we mirror that defensive check below.
 *
 * Step 4: BF_set_key with key cycling (mirrors slab :489-505); pack
 *   18 BE uint32 words from pw[0..keylen) cyclically via modular
 *   pointer increment.
 *
 * Step 5: claim per-lane __local S-box partition (4 x 256 uints =
 *   1024 uints); copy BCRYPT_INIT_S0..S3 from __constant via vstore4
 *   (mirrors slab :511-534).
 *
 * Step 6: P = BCRYPT_INIT_P ^ expanded_key; Eksblowfish setup with
 *   salt XOR (mirrors slab :536-576).
 *
 * Step 7: 2^cost main iteration loop: alternating BF_set_key with
 *   expanded_key and salt (mirrors slab :583-602).
 *
 * Step 8: encrypt "OrpheanBeholderScryDoubt" 64 times; convert 6 BE
 *   uint32 -> 6 LE uint32 (mirrors slab :605-627); install into st->h.
 *
 * algo_mode: BCRYPT family uses algo_mode=8 (single mode for all 4
 * compound siblings). Future BCRYPT-family variants requiring
 * kernel-side preprocessing claim 9-15. For the current 4-op set,
 * the kernel ignores the value (host preprocesses inputs before pack). */
static inline void template_finalize(template_state *st,
                                     const uchar *data, int len
#ifdef GPU_TEMPLATE_HAS_SALT
                                     , __global const uchar *salt_bytes
                                     , uint salt_len
                                     , uint algo_mode
#endif
#ifdef GPU_TEMPLATE_HAS_LOCAL_BUFFER
                                     , __local uint *sbox_pool
                                     , uint lid
#endif
                                     )
{
#if defined(GPU_TEMPLATE_HAS_SALT) && defined(GPU_TEMPLATE_HAS_LOCAL_BUFFER)
    /* Algo_mode gate: BCRYPT family uses algo_mode=8. Kernels compiled
     * with this core but invoked at a different algo_mode return early
     * with zero state (no false-positive probes). algo_mode == 8u is
     * the only valid setting; all 4 ops (BCRYPT/BCRYPTMD5/BCRYPTSHA1/
     * BCRYPTSHA512) host-preprocess inputs and share kernel logic. */
    if (algo_mode != 8u) {
        for (int i = 0; i < HASH_WORDS; i++) st->h[i] = 0u;
        return;
    }

    /* Step 1: defensive cap on key length. */
    int plen = len;
    if (plen > 72) plen = 72;
    if (plen < 0) plen = 0;

    /* Step 2: copy data to private pw buffer; append NUL within cap.
     * bcrypt key includes the NUL terminator (matches slab :453-457). */
    uchar pw[72];
    int keylen = plen;
    for (int i = 0; i < keylen; i++) pw[i] = data[i];
    if (keylen < 72) {
        pw[keylen] = 0;
        keylen++;
    }

    /* Step 3a: parse cost from salt_bytes[4..5] (defensive check on
     * salt_len -- slab :466-468 + :475 guard slen_full >= 7 + >= 29). */
    uint cost = 0;
    if (salt_len >= 7u) {
        cost = (uint)(salt_bytes[4] - '0') * 10u + (uint)(salt_bytes[5] - '0');
    }
    if (cost < 4u) cost = 4u;
    if (cost > 31u) cost = 31u;
    uint count = 1u << cost;

    /* Step 3b: decode 22-char base64 salt at offset 7 -> 16 raw bytes.
     * Defensive: require salt_len >= 29 ($2[abxy]$NN$ + 22 = 29 bytes,
     * matches slab :475 guard). $2k$ variant (slen_full=28) is rejected
     * here -- host loader can opt to accept by upgrading the salt
     * string to $2b$ form before pack. */
    if (salt_len < 29u) {
        for (int i = 0; i < HASH_WORDS; i++) st->h[i] = 0u;
        return;
    }
    uchar raw_salt[16];
    bcrypt_decode_salt(raw_salt, salt_bytes + 7);

    /* Convert 16 raw bytes to 4 BE uint32 (BF_swap equivalent). */
    uint salt_w[4];
    salt_w[0] = ((uint)raw_salt[0]  << 24) | ((uint)raw_salt[1]  << 16) |
                ((uint)raw_salt[2]  << 8)  |  (uint)raw_salt[3];
    salt_w[1] = ((uint)raw_salt[4]  << 24) | ((uint)raw_salt[5]  << 16) |
                ((uint)raw_salt[6]  << 8)  |  (uint)raw_salt[7];
    salt_w[2] = ((uint)raw_salt[8]  << 24) | ((uint)raw_salt[9]  << 16) |
                ((uint)raw_salt[10] << 8)  |  (uint)raw_salt[11];
    salt_w[3] = ((uint)raw_salt[12] << 24) | ((uint)raw_salt[13] << 16) |
                ((uint)raw_salt[14] << 8)  |  (uint)raw_salt[15];

    /* Step 4: BF_set_key -- pack password bytes into 18 BE uint32, cycling.
     * $2b$ / $2y$ path (bug=0, safety=0). Key bytes including NUL terminator
     * are packed big-endian, cycling. Mirrors slab :492-505. */
    uint expanded_key[18];
    {
        int ptr = 0;
        for (int i = 0; i < 18; i++) {
            uint tmp = 0;
            for (int j = 0; j < 4; j++) {
                tmp <<= 8;
                tmp |= (uint)pw[ptr];
                ptr++;
                if (ptr >= keylen) ptr = 0;
            }
            expanded_key[i] = tmp;
        }
    }

    /* Step 5: claim per-lane S-box partition; copy init data from
     * __constant via vstore4 (no barrier needed -- per-lane writes).
     * Mirrors slab :511-534. */
    __local uint *S0 = sbox_pool + lid * 1024u;
    __local uint *S1 = S0 + 256;
    __local uint *S2 = S1 + 256;
    __local uint *S3 = S2 + 256;

    /* Step 6: P = BCRYPT_INIT_P ^ expanded_key; init S-boxes. */
    uint P[18];
    for (int i = 0; i < 18; i++)
        P[i] = BCRYPT_INIT_P[i] ^ expanded_key[i];

    /* Copy S-box init data via vstore4. 256 uints = 64 x uint4 per S-box.
     * Mirrors slab :527-534. */
    for (int i = 0; i < 64; i++)
        vstore4(vload4(i, (__constant uint *)BCRYPT_INIT_S0), 0, S0 + i * 4);
    for (int i = 0; i < 64; i++)
        vstore4(vload4(i, (__constant uint *)BCRYPT_INIT_S1), 0, S1 + i * 4);
    for (int i = 0; i < 64; i++)
        vstore4(vload4(i, (__constant uint *)BCRYPT_INIT_S2), 0, S2 + i * 4);
    for (int i = 0; i < 64; i++)
        vstore4(vload4(i, (__constant uint *)BCRYPT_INIT_S3), 0, S3 + i * 4);

    /* Eksblowfish setup: encrypt with salt XOR to fill P and S.
     * Mirrors slab :536-576. */
    {
        uint L = 0, R = 0;

        /* Fill P-array: 9 pairs, salt alternates [0,1] and [2,3] via (i & 2). */
        for (int i = 0; i < 18; i += 2) {
            L ^= salt_w[i & 2];
            R ^= salt_w[(i & 2) + 1];
            BCRYPT_BF_ENCRYPT(L, R, S0, S1, S2, S3, P);
            P[i] = L;
            P[i + 1] = R;
        }

        /* Fill S-boxes: continue salt alternation. Each S-box has 128 pairs;
         * within each pair-of-pairs (j+=4), first pair XORs salt[2],salt[3],
         * second pair XORs salt[0],salt[1]. */
        for (int box = 0; box < 4; box++) {
            __local uint *S;
            if      (box == 0) S = S0;
            else if (box == 1) S = S1;
            else if (box == 2) S = S2;
            else               S = S3;

            for (int j = 0; j < 256; j += 4) {
                L ^= salt_w[2];
                R ^= salt_w[3];
                BCRYPT_BF_ENCRYPT(L, R, S0, S1, S2, S3, P);
                S[j]     = L;
                S[j + 1] = R;

                L ^= salt_w[0];
                R ^= salt_w[1];
                BCRYPT_BF_ENCRYPT(L, R, S0, S1, S2, S3, P);
                S[j + 2] = L;
                S[j + 3] = R;
            }
        }
    }

    /* Step 7: main loop -- 2^cost iterations of:
     *   1. XOR expanded_key into P[0..17], then bcrypt_bf_body
     *   2. XOR salt into P[0..17] (alternating pairs), then bcrypt_bf_body
     * Mirrors slab :583-602. */
    for (uint iter = 0; iter < count; iter++) {
        for (int i = 0; i < 18; i++)
            P[i] ^= expanded_key[i];
        bcrypt_bf_body(S0, S1, S2, S3, P);

        P[0]  ^= salt_w[0]; P[1]  ^= salt_w[1];
        P[2]  ^= salt_w[2]; P[3]  ^= salt_w[3];
        P[4]  ^= salt_w[0]; P[5]  ^= salt_w[1];
        P[6]  ^= salt_w[2]; P[7]  ^= salt_w[3];
        P[8]  ^= salt_w[0]; P[9]  ^= salt_w[1];
        P[10] ^= salt_w[2]; P[11] ^= salt_w[3];
        P[12] ^= salt_w[0]; P[13] ^= salt_w[1];
        P[14] ^= salt_w[2]; P[15] ^= salt_w[3];
        P[16] ^= salt_w[0]; P[17] ^= salt_w[1];
        bcrypt_bf_body(S0, S1, S2, S3, P);
    }

    /* Step 8: encrypt "OrpheanBeholderScryDoubt" 64 times.
     * 24 bytes = 6 BE uint32 words, encrypted as 3 pairs.
     * Mirrors slab :607-618. */
    uint ctext[6];
    ctext[0] = 0x4f727068u; /* "Orph" */
    ctext[1] = 0x65616e42u; /* "eanB" */
    ctext[2] = 0x65686f6cu; /* "ehol" */
    ctext[3] = 0x64657253u; /* "derS" */
    ctext[4] = 0x63727944u; /* "cryD" */
    ctext[5] = 0x6f756274u; /* "oubt" */

    for (int r = 0; r < 64; r++) {
        BCRYPT_BF_ENCRYPT(ctext[0], ctext[1], S0, S1, S2, S3, P);
        BCRYPT_BF_ENCRYPT(ctext[2], ctext[3], S0, S1, S2, S3, P);
        BCRYPT_BF_ENCRYPT(ctext[4], ctext[5], S0, S1, S2, S3, P);
    }

    /* Convert 6 BE words to 6 LE words for compact-table probe.
     * Mirrors slab :620-627. */
    for (int i = 0; i < 6; i++) {
        uchar4 b = as_uchar4(ctext[i]);
        st->h[i] = as_uint((uchar4)(b.w, b.z, b.y, b.x));
    }
    return;
#else
    /* Defensive fall-through for !HAS_SALT or !HAS_LOCAL_BUFFER.
     * BCRYPT requires both: salt_bytes for cost+raw_salt, and the
     * __local sbox_pool for Eksblowfish state. A build with either
     * undefined would have nothing functional to do. */
    (void)data;
    (void)len;
    for (int i = 0; i < HASH_WORDS; i++) st->h[i] = 0u;
#endif
}

/* template_iterate: STUB. With max_iter = 1 (host-set for BCRYPT), the
 * outer iter loop in template_phase0 runs exactly once and never calls
 * template_iterate. Mirrors PHPBB3 / DESCRYPT pattern. */
static inline void template_iterate(template_state *st)
{
    (void)st;
}

/* template_digest_compare: probe the compact table with first 4 LE
 * words (16 bytes) of the 24-byte bcrypt output. Mirrors slab
 * :629-635 (probe_compact uses out[0..3]). */
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

/* template_emit_hit: emit a hit. BCRYPT = 6 LE uint32 words. Uses the
 * NEW EMIT_HIT_6 wire format (6 hash words at hits[base+3..base+8]).
 * Host's hit-replay arm reconstructs the 60-char $2b$ crypt hash via
 * bf_encode_23 in gpujob_opencl.c (B2 helper). */
#define template_emit_hit(hits, hit_count, max_hits, st, widx, sidx, iter) \
    EMIT_HIT_6((hits), (hit_count), (max_hits), \
               (widx), (sidx), (iter), (st)->h)

/* B3 dedup+overflow-aware variant. NEW macro
 * EMIT_HIT_6_DEDUP_OR_OVERFLOW added in this same change to
 * gpu_common.cl (Goal 3 of B1; mirrors EMIT_HIT_5/7 pattern). */
#define template_emit_hit_or_overflow(hits, hit_count, max_hits, st, widx, sidx, iter, \
                                      hashes_shown, matched_idx, dedup_mask, \
                                      ovr_set, ovr_gid, lane_gid) \
    EMIT_HIT_6_DEDUP_OR_OVERFLOW((hits), (hit_count), (max_hits), \
               (widx), (sidx), (iter), (st)->h, \
               (hashes_shown), (matched_idx), (dedup_mask), \
               (ovr_set), (ovr_gid), (lane_gid))
