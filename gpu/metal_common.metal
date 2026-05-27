/*
 * $Revision: 1.26 $
 * $Log: metal_common.metal,v $
 * Revision 1.26  2026/05/27 17:46:13  dlr
 * sub-phase 5b1b2 Metal twin lift rmd128_block primitive into metal_common.metal mirror of gpu_common.cl rev 1.27 rmd128_block byte-for-byte 4-uint state dual pipeline left line F1 F2 F3 F4 right line F4 F3 F2 F1 per Bosselaers Table 4 R2 inline comments preserved reuses RMD_F1 through F4 round functions defined above for rmd160_block RMD_F5 unused defines local RMD128_STEP_METAL 4-arg variant without E and without C-rotation matching rmd128.h FF GG HH II macro shape LL1M through LL4M left line round-K macros RR1M through RR4M right line round-K macros M suffix avoids collision with rmd160 LL1 LL5 R1 R5 names pointer-state convention thread uint hash thread const uint X matches rmd160_block sha1_block sha256_block static inline Pattern 3 R6 not applicable XOR AND OR NOT only no bitselect R7 no nested block comments inserted between rmd160_block and rmd320_block keeping RMD family clustered used by emit_outer_rmd128_concat_then_hash_metal 5b1b3 family helper
 *
 * Revision 1.25  2026/05/27 16:58:18  dlr
 * sub-phase 5b1a2 Metal twin lift md2_block primitive into metal_common.metal mirror of gpu_common.cl rev 1.26 md2_block byte-for-byte same RFC 1319 with errata MD2_PI 256-byte S-box namespaced MTL_MD2_PI per Pattern 2 thread uchar state plus thread uchar checksum plus thread const uchar data plus int update_checksum signature 18-round state transform plus per-data-block checksum update update_checksum flag skips on final checksum block per RFC errata R3 byte-identical S-box no retype R5 static inline Pattern 3 R7 no nested block comments R6 not applicable XOR only no bitselect family helper outer_md2_concat_then_hash_metal in 5b1a3 will call md2_block via thread-local state plus checksum scratch
 *
 * Revision 1.24  2026/05/23 05:23:01  dlr
 * sub-phase 5a.4 lift md4_block primitive into metal_common.metal mirror of gpu_common.cl md4_block byte-for-byte Metal twin pointer-state convention thread uint pointer to state plus thread const uint pointer to M same MD4_F G H round functions same round-2 constant 0x5A827999u round-3 constant 0x6ED9EBA1u namespaced via MTL_ prefix per Pattern 2 LE-schedule no byte-swap output sibling to existing rmd160_block sha1_block sha256_block sha512_block validated PASS 8 of 8 on Apple M2 Max Metal byte-exact for e122 MD4MD5PASS family member
 *
 * Revision 1.23  2026/05/21 12:41:00  dlr
 * Phase 1 sub-phase 1a.2 D9.1.b rename overflow_first_rule to num_rules at offset 108 OCLParams field repurposed for kernel A1 A3 source rule count B3 path stops using slot host writes 1 when not applicable update prose comments accordingly
 *
 * Revision 1.22  2026/05/21 04:56:55  dlr
 * Phase 1a sub-phase 1a.1b-continued: claimed reserved32 slots as base_word_idx (80) and packed_size (84) named fields. Added typedef MetalParams OCLParams bridge for translator-driven Metal sources.
 *
 * Revision 1.21  2026/05/19 13:48:43  dlr
 * Fix MTL_SHA512_F0o and F1o: use arithmetic Ch/Maj forms (no bitselect) since Metal MSL lacks bitselect for scalars. Xcode 26 toolchain on dev1 rejects bitselect for ulong; Metal select takes bool not bitmask. Arithmetic forms are semantically identical.
 *
 * Revision 1.20  2026/05/19 13:36:35  dlr
 * Step A Metal port: sha512_block flat-unrolled 80-step scalar body with MTL_SHA512_STEP_S and MTL_SHA512_EXPAND_S macros. Replaces W[80] array plus for-loop form; eliminates 640-byte stack allocation. Uses bitselect for Ch/Maj (Metal MSL native). Metal inlines and benefits -- no noinline attr. Mirrors gpu_common.cl rev 1.22 Step A pattern.
 *
 * Revision 1.19  2026/05/17 01:32:23  dlr
 * Phase 2d.9b BCRYPT prep. Add EMIT_HIT_6_DEDUP_OR_OVERFLOW macro for 6-word (24-byte) digest emit. Sibling of EMIT_HIT_4/5/7/8/10/12/16. Hits buffer write loop iterates 0..5; tail-zero loop starts at _z=9u. FIRST 6-word Metal family is JOB_BCRYPT. Semantics IDENTICAL to OpenCL EMIT_HIT_6_DEDUP_OR_OVERFLOW.
 *
 * Revision 1.18  2026/05/16 18:51:24  dlr
 * Phase 2d.7b Keccak/SHA-3 sponge family prep -- add keccakf1600 (thread ulong pointer-state, 25-ulong sponge state, 24 rounds) plus MTL_KECCAK_RC[24] plus MTL_KECCAK_ROTC[25] constants. Single shared sponge primitive for the 8 Keccak/SHA-3 ops (KECCAK-{224,256,384,512} plus SHA3-{224,256,384,512}); per-algo distinguishers are rate and domain-pad and output-bytes baked into per-core .cl files. Constants are MTL_-prefixed per Pattern 2 (sibling to MTL_SHA256_K and MTL_SHA512_K and MTL_B2S_IV and MTL_B2B_SIGMA); cl2metal.py rev 1.7 rewrites bare KECCAK_RC plus KECCAK_ROTC token references in generated core sources to the MTL_-prefixed names. Pointer-state signature (thread ulong state) matches sha512_block plus rmd160_block convention. keccakf1600 takes no uchar block arg so no dual address-space overload is required (compare b2s_compress and b2b_compress); the per-core absorb helpers (keccak256_absorb_full plus keccak256_absorb_pad etc.) ARE called with device-const slices and thread-const slices and DO need dual overloads -- handled by the cl2metal.py rev 1.7 dual_addr_space_helpers translator extension. rotl64 helper reused from Phase 2d.7a Blake2. CPU plus GPU md5sum parity verified on dev1 M1 for all 8 ops (op 84-91).
 *
 * Revision 1.17  2026/05/16 17:50:20  dlr
 * Phase 2d.7a BLAKE2 family prep -- add b2s_compress plus b2b_compress pointer-state compressions plus MTL_B2S_IV plus MTL_B2S_SIGMA plus MTL_B2B_IV plus MTL_B2B_SIGMA constants plus rotl64 64-bit rotate-left helper plus MTL_B2B_G G-mixing macro. Each compress provides TWO address-space overloads (thread const uchar block plus device const uchar block) via shared MTL_B2S_COMPRESS_BODY and MTL_B2B_COMPRESS_BODY textual macros; Metal forbids generic address-space pointers in function signatures so two overloads are required because template_finalize hot path passes data plus pos as device const uchar from buf_scratch_pool while template_iterate uses a thread-local buf[64 or 128]. Constants are MTL_-prefixed per Pattern 2 (sibling to MTL_SHA256_K and MTL_SHA512_K and MTL_MD5_S); cl2metal.py rev 1.6 rewrites bare B2S_IV plus B2S_SIGMA plus B2B_IV plus B2B_SIGMA token references in generated core sources to the MTL_-prefixed names. Pointer-state signatures (thread uint h and thread const uchar block; thread ulong h and thread const uchar block) match sha512_block plus rmd160_block convention -- cl2metal.py rev 1.6 preserves call-site addr-of via _BLOCK_HELPERS_POINTER_STATE. Mirrors gpu_common.cl rev with byte-exact compression bodies. CPU plus GPU md5sum parity verified on dev1 M1 for BLAKE2S256 (op=844), BLAKE2B256 (op=845), BLAKE2B512 (op=841). Architect Phase 2d.7a roadmap sub-phase.
 *
 * Revision 1.16  2026/05/16 16:42:07  dlr
 * Phase 2d.6 RIPEMD family prep -- add rmd160_block plus rmd320_block plus RMD_F1 through F5 plus RMD_STEP plus L1 through L5 plus R1 through R5 round-macro helpers; add EMIT_HIT_10_DEDUP_OR_OVERFLOW for 10-uint32 LE digest (RIPEMD-320 = 320 bits). Pointer-state signatures (thread uint hash, thread const uint X) match sha1_block/sha256_block convention -- cl2metal.py rev 1.5 preserves call-site addr-of via _BLOCK_HELPERS_POINTER_STATE. RMD_STEP uses rotl32 (Metal substitute for OpenCL rotate). Mirrors gpu_common.cl rev with byte-exact compression bodies; CPU/GPU md5sum parity verified on dev1 M1 for both RMD160 (op=17) and RMD320 (op=816). Architect Phase 2d roadmap section 9 ripemd-160/320 wave.
 *
 * Revision 1.15  2026/05/16 04:34:29  dlr
 * Phase 2d.5.3 sha384 prep: add EMIT_HIT_12_DEDUP_OR_OVERFLOW for 12-word digest emit. Sibling of EMIT_HIT_5/7/8/16. SHA-384 truncates SHA-512 to state[0..5]; digest spans hits[base+3..+14] with 4 trailing slots zeroed (HIT_STRIDE=19).
 *
 * Revision 1.14  2026/05/16 03:56:14  dlr
 * Phase 2d.5.1 sha512 family on Metal -- FIRST 64-bit-state family. Adds bswap64 + rotr64 + sha512_block (thread ulong pointer-state, W[80] = 640 bytes private scratch) + sha512_to_hex_lc + EMIT_HIT_16_DEDUP_OR_OVERFLOW. EMIT_HIT_16 fills hits[base+3..+18] (3 metadata + 16 digest = 19 = HIT_STRIDE exactly; tail-zero loop body never enters). MTL_SHA512_K[80] constants namespaced per Pattern 2; MTL_S512_* helper macros similarly namespaced. SHA-512 W[80] is the largest in the family (vs SHA-1 W[80] uint = 320 B, SHA-256 W[64] = 256 B). Verified compile-clean on iMac Intel + dev1 ARM64 (Apple M1) -- no exceeds-temporary-registers errors despite architect §5 R2 risk callout. Companion to gpu_metal.m / .h / gpujob_metal.m.
 *
 * Revision 1.13  2026/05/16 02:49:08  dlr
 * Phase 2d.4.3 sha224 prep: add EMIT_HIT_7_DEDUP_OR_OVERFLOW for 7-word digest emit. Sibling of EMIT_HIT_5 + EMIT_HIT_8. Drops the 8th internal state word (SHA-224 truncates SHA-256 to 7 uint32 output). Tail-zero loop starts at _z=10u.
 *
 * Revision 1.12  2026/05/16 02:23:43  dlr
 * *** empty log message ***
 *
 * Revision 1.11  2026/05/16 00:05:13  dlr
 * *** empty log message ***
 *
 * Revision 1.10  2026/05/12 13:35:49  dlr
 * Phase 1 Metal port fresh start (replaces retired Phase 0 design). Mirrors gpu/gpu_common.cl: MetalParams 128-byte struct (byte-identical to OCLParams; static_asserts via __builtin_offsetof on size + algo_mode/num_words/max_iter/overflow_first_word offsets), HIT_STRIDE=19, RULE_BUF_MAX=40960, md5_block (single-block compress), md5_to_hex_lc/_uc (pre-declared for Phase 2 iter), probe_compact_idx (compact+overflow table lookup), EMIT_HIT_4_DEDUP_OR_OVERFLOW macro (Metal atomics: atomic_fetch_or_explicit etc.). Patterns 1/2/3/6 design rules documented at top + enforced by metal_jit_harness.m --check-patterns + offline xcrun metal. MD5 round constants namespaced MTL_MD5_K/S/G (was bare K/S/G in retired source — collision risk). All helpers static inline. Verified on dev1.local (Apple M1): build_metallib.sh + metal_jit_harness pass; pattern check reports 0 violations.
 *
 */
/* metal_common.metal — minimum shared primitives for the Metal template
 * path (Phase 1). Mirrors a subset of gpu/gpu_common.cl. NO `kernel void`
 * declarations in this file.
 *
 * Phase 1 scope (mirrors gpu_md5_core.cl + template_phase0 minimums):
 *   - MetalParams struct (byte-identical to OCLParams; 128 bytes).
 *   - HIT_STRIDE = 19; RULE_BUF_MAX / RULE_BUF_LIMIT host-wire constants.
 *   - md5_block (single-block MD5 compress function).
 *   - md5_to_hex_lc / md5_to_hex_uc (pre-declared for Phase 2 iter loop).
 *   - probe_compact_idx (compact-table + overflow-table hash lookup).
 *   - EMIT_HIT_4_DEDUP_OR_OVERFLOW (single emit macro Phase 1 uses).
 *
 * --- DESIGN RULES (Phase 0.5 patterns) ---
 *
 * These six rules came out of the 2026-04 Metal JIT failure spike. They
 * are *encoded* here (not just commented): the JIT harness in
 * metal_jit_harness.m --check-patterns greps for violations; offline
 * `xcrun metal` builds in build_metallib.sh fail loudly on most of them.
 * Reviewers MUST observe these when adding primitives.
 *
 *   Pattern 1: ADDRESS-SPACE QUALIFIERS ON ALL POINTERS.
 *     Every function pointer parameter has an explicit `device`,
 *     `threadgroup`, `constant`, or `thread` qualifier. Metal forbids
 *     `auto` / generic address spaces in function signatures (unlike
 *     OpenCL's __generic). Helpers that touch the working buffer take
 *     `thread const uint *` or `thread uint *`. Helpers reading global
 *     state take `device const T *`. Bare-pointer casts without a space
 *     qualifier (e.g. cast-to-uchar-pointer with no `thread`/`device`)
 *     are rejected by both the harness and the offline compiler — the
 *     correct form is `(thread uchar *)x` or `(device uchar *)x`.
 *
 *   Pattern 2: NAMESPACE PER-ALGORITHM CONSTANTS.
 *     MD5 round tables are MTL_MD5_K, MTL_MD5_S, MTL_MD5_G. The retired
 *     metal_common.metal used bare K/S/G which collided when a future
 *     SHA family file used the same names. New cores MUST use a family
 *     prefix.
 *
 *   Pattern 3: HELPERS ARE `static` OR `static inline`.
 *     Every non-kernel function in this file is declared `static
 *     inline`. Multi-TU metallib link (build_metallib.sh) treats each
 *     family TU independently; bare `void md5_block(...)` would collide
 *     at link time with another family TU's md5_block. Grep guard:
 *     `^(void|int|uint|ulong|float|double)\s+\w` (i.e. a return type
 *     followed by an identifier at column 0, without `static`) is a
 *     violation.
 *
 *   Pattern 4: ONE TU PER FAMILY IN build_metallib.sh.
 *     Each family compiles separately (xcrun metal -c). build_metallib.sh
 *     enforces this — never merge two families into one TU.
 *
 *   Pattern 5: INITIALIZE THREADGROUP BEFORE READ.
 *     Any `threadgroup` declaration MUST be initialized before its
 *     first read, with a `threadgroup_barrier(mem_flags::mem_threadgroup)`
 *     between init and read. Phase 1 doesn't use threadgroup memory; the
 *     rule is pre-stated for Phase 2 algorithms that will.
 *
 *   Pattern 6: MULTI-LINE MACROS — REVIEW IN HARNESS.
 *     Multi-line `#define` with backslash continuations are valid but
 *     have caught the offline compiler with hidden whitespace.
 *     metal_jit_harness.m --check-patterns greps `\\\\\\s+\\n` for
 *     trailing-whitespace-after-backslash. Keep multi-line macros to
 *     a minimum and prefer `static inline` helpers when possible.
 *
 * --- WIRE FORMAT INVARIANTS ---
 *
 *   sizeof(MetalParams) == 128       (mirrored from OCLParams).
 *   offsetof(MetalParams, algo_mode) == 120.
 *   HIT_STRIDE == 19                  (gpu_common.cl line 70).
 *   RULE_BUF_MAX == 40960             (gpu_md5_rules.cl line 118).
 *   RULE_BUF_LIMIT == 40959.
 *
 * Phase 1 unused but pre-declared for Phase 2+:
 *   md5_to_hex_lc / md5_to_hex_uc — iter loop hex re-encode.
 */

#include <metal_stdlib>
using namespace metal;

/* --- Wire-format constants (host-mirror invariants) --- */

#ifndef HIT_STRIDE
#define HIT_STRIDE 19u
#endif

#ifndef RULE_BUF_MAX
#define RULE_BUF_MAX 40960
#endif

#ifndef RULE_BUF_LIMIT
#define RULE_BUF_LIMIT (RULE_BUF_MAX - 1)
#endif

/* --- MetalParams: 128-byte uniform API. BYTE-IDENTICAL to OCLParams in
 * gpu/gpu_common.cl. The layout MUST match because gpujob_opencl.c host
 * code populates the same struct and the Metal kernel reads it as the
 * payload prefix. See gpu_common.cl line ~30 for the OpenCL twin.
 *
 * static_assert verifies the byte layout at compile time. If anyone
 * changes OCLParams without also updating MetalParams (or vice versa)
 * the Metal kernel build trips here, surfacing the divergence before
 * silent miscompute. */
struct MetalParams {
    ulong compact_mask;        /*   0: hash table mask */
    ulong mask_start;          /*   8: mask keyspace offset */
    ulong mask_base0;          /*  16: pre-decomposed positions 0-7 */
    ulong mask_base1;          /*  24: pre-decomposed positions 8-15 */
    uint  num_words;           /*  32: words in batch */
    uint  num_salts;           /*  36: salts for dispatch */
    uint  salt_start;          /*  40: starting salt index */
    uint  max_probe;           /*  44: compact table probe depth */
    uint  hash_data_count;     /*  48: hash_data entries */
    uint  max_hits;            /*  52: hit buffer capacity */
    uint  overflow_count;      /*  56: overflow table entries */
    uint  max_iter;            /*  60: iteration count (-i) */
    uint  num_masks;           /*  64: mask combinations per chunk */
    uint  n_prepend;           /*  68: prepend mask positions (-N) */
    uint  n_append;            /*  72: append mask positions (-n) */
    uint  iter_count;          /*  76: per-dispatch iteration (PHPBB3) */
    /* Phase 1a sub-phase 1a.1b-continued (2026-05-20): claim reserved32[2]
     * as named fields for the two-kernel pipeline. Mirrors OpenCL
     * gpu/gpu_common.cl rev 1.23. Standard dispatcher leaves these zero;
     * kernel A1 reads packed_size as output buffer capacity (bytes) and
     * base_word_idx as kernel A source word index (hit-attribution).
     * Per feedback_rename_reserved_slots.md: rename in same commit. */
    uint  base_word_idx;       /*  80-83: kernel A source word index */
    uint  packed_size;         /*  84-87: kernel A output buffer capacity */
    uint  input_cursor_start;  /*  88: B3 input cursor */
    uint  rule_cursor_start;   /*  92: B3 rule cursor */
    uint  inner_iter;          /*  96: BF inner iteration count */
    uint  overflow_first_set;  /* 100: B3 first-overflow flag */
    uint  overflow_first_word; /* 104: B3 word_idx CAS-min target */
    uint  num_rules;           /* 108: source rule count for kernel A1/A3;
                                *      reads as 1 when not applicable */
    ulong num_salts_per_page;  /* 112: B6 salt-axis paging */
    uint  algo_mode;           /* 120: B6.6 per-algorithm runtime variant */
    uint  mask_offset_per_word;/* 124: BF chunk word stride; 0 == not BF */
};

/* Phase 1a sub-phase 1a.1b-continued (2026-05-20): OCLParams typedef
 * bridge for translator-driven sources. cl2metal.py emits the source-
 * level type name `OCLParams` (matches the upstream OpenCL .cl source);
 * Metal's hand-written shared infrastructure uses MetalParams. The two
 * structs are byte-identical (enforced by static_asserts below + host-
 * side _Static_asserts in gpu_metal.m). Translator-driven .metal files
 * include metal_common (which provides MetalParams + this typedef) so
 * `OCLParams params; params.field;` works without source rewrites. */
typedef MetalParams OCLParams;

/* Note: Metal does NOT expose `offsetof` from <cstddef> in shader sources;
 * we use the compiler builtin `__builtin_offsetof` which Apple's metal
 * frontend supports. Verified on dev1.local (Apple metal 32023.864). */
static_assert(sizeof(MetalParams) == 128, "MetalParams MUST be 128 bytes (OCLParams parity)");
static_assert(__builtin_offsetof(MetalParams, algo_mode) == 120, "MetalParams.algo_mode offset MUST be 120 (B6.6 wire format)");
static_assert(__builtin_offsetof(MetalParams, num_words) == 32, "MetalParams.num_words offset MUST be 32");
static_assert(__builtin_offsetof(MetalParams, max_iter) == 60, "MetalParams.max_iter offset MUST be 60");
static_assert(__builtin_offsetof(MetalParams, overflow_first_word) == 104, "MetalParams.overflow_first_word offset MUST be 104");

/* --- MD5 round constants (namespaced per Pattern 2) ---
 *
 * Bare names K[], S[], G[] in the retired metal_common.metal collided
 * with other family files when build_metallib.sh tried to consolidate.
 * Phase 1 namespacing: MTL_MD5_* makes future SHA / BLAKE family
 * constants safe to coexist in the same metallib.
 *
 * G[] is the MD5 message-index permutation (round 2-4). Not needed by
 * the inline FF/GG/HH/II macros below (they hard-code message indices),
 * but kept for Phase 2 utilities that may want a table-driven form. */

constant uint MTL_MD5_S[64] = {
    7,12,17,22, 7,12,17,22, 7,12,17,22, 7,12,17,22,
    5, 9,14,20, 5, 9,14,20, 5, 9,14,20, 5, 9,14,20,
    4,11,16,23, 4,11,16,23, 4,11,16,23, 4,11,16,23,
    6,10,15,21, 6,10,15,21, 6,10,15,21, 6,10,15,21
};

/* --- Helper: 32-bit rotate-left. ---
 * Metal's std::rotate works on uint; we use the explicit form for
 * clarity at FF/GG/HH/II callsites. Pattern 3: static inline. */
static inline uint rotl32(uint x, uint n) {
    return (x << n) | (x >> (32u - n));
}

/* --- MD5 round macros (mirrors gpu_common.cl lines 542-545) ---
 * Pattern 6: multi-line macros are minimised. These four are simple
 * one-liners with trailing-semicolons inside braces — no continuations.
 * The macro hygiene is identical to OpenCL.  */
#define MTL_MD5_FF(a,b,c,d,m,s,k) { a += ((b&c)|(~b&d)) + m + k; a = b + rotl32(a, (uint)s); }
#define MTL_MD5_GG(a,b,c,d,m,s,k) { a += ((d&b)|(~d&c)) + m + k; a = b + rotl32(a, (uint)s); }
#define MTL_MD5_HH(a,b,c,d,m,s,k) { a += (b^c^d) + m + k;        a = b + rotl32(a, (uint)s); }
#define MTL_MD5_II(a,b,c,d,m,s,k) { a += (c^(~d|b)) + m + k;     a = b + rotl32(a, (uint)s); }

/* --- MTL_MD2_PI[256]: MD2 S-box (RFC 1319 Table T; copy-paste-no-retype
 *      from md2/md2.c lines 17-34). MTL_ prefix per Pattern 2 (collision
 *      avoidance across TUs). Used by md2_block below.
 *      R3 mitigation: byte-identical to gpu_common.cl MD2_PI[256]. */
constant uchar MTL_MD2_PI[256] = {
    41, 46, 67, 201, 162, 216, 124, 1, 61, 54, 84, 161, 236, 240, 6,
    19, 98, 167, 5, 243, 192, 199, 115, 140, 152, 147, 43, 217, 188,
    76, 130, 202, 30, 155, 87, 60, 253, 212, 224, 22, 103, 66, 111, 24,
    138, 23, 229, 18, 190, 78, 196, 214, 218, 158, 222, 73, 160, 251,
    245, 142, 187, 47, 238, 122, 169, 104, 121, 145, 21, 178, 7, 63,
    148, 194, 16, 137, 11, 34, 95, 33, 128, 127, 93, 154, 90, 144, 50,
    39, 53, 62, 204, 231, 191, 247, 151, 3, 255, 25, 48, 179, 72, 165,
    181, 209, 215, 94, 146, 42, 172, 86, 170, 198, 79, 184, 56, 210,
    150, 164, 125, 182, 118, 252, 107, 226, 156, 116, 4, 241, 69, 157,
    112, 89, 100, 113, 135, 32, 134, 91, 207, 101, 230, 45, 168, 2, 27,
    96, 37, 173, 174, 176, 185, 246, 28, 70, 97, 105, 52, 64, 126, 15,
    85, 71, 163, 35, 221, 81, 175, 58, 195, 92, 249, 206, 186, 197,
    234, 38, 44, 83, 13, 110, 133, 40, 132, 9, 211, 223, 205, 244, 65,
    129, 77, 82, 106, 220, 55, 200, 108, 193, 171, 250, 36, 225, 123,
    8, 12, 189, 177, 74, 120, 136, 149, 139, 227, 99, 232, 109, 233,
    203, 213, 254, 59, 0, 29, 57, 242, 239, 183, 14, 102, 88, 208, 228,
    166, 119, 114, 248, 235, 117, 75, 10, 49, 68, 80, 180, 143, 237,
    31, 26, 219, 153, 141, 51, 159, 17, 131, 20
};

/* --- md2_block: single 16-byte MD2 compress block (RFC 1319 with errata).
 *
 * Lifted 2026-05-27 from md2/md2.c md2_transform for the hx codegen
 * sub-phase 5b.1a e120 MD2MD5PASS family emit. Mirrors
 * gpu_common.cl::md2_block byte-for-byte (S-box matches; 18-round
 * structure matches; update_checksum flag matches).
 *
 * Signature: state (thread uchar* 48 bytes), checksum (thread uchar*
 * 16 bytes), data (thread const uchar* 16 bytes). update_checksum
 * selects per-data-block checksum update (1) or skip (0) -- the final
 * checksum-block call uses 0 per RFC errata.
 *
 * R7 NO nested block comments in donor port.
 * Pattern 3: static inline (per-TU; no link collision). */
static inline void md2_block(thread uchar *state, thread uchar *checksum,
                             thread const uchar *data,
                             int update_checksum)
{
    int j, k;
    uint t;

    // Spec step 1: copy block into state[16..31]; xor into state[32..47].
    for (j = 0; j < 16; j++) {
        state[j + 16] = data[j];
        state[j + 32] = (uchar)(state[j + 16] ^ state[j]);
    }

    // Spec step 2: 18 rounds of state transform.
    t = 0;
    for (j = 0; j < 18; j++) {
        for (k = 0; k < 48; k++) {
            state[k] = (uchar)(state[k] ^ MTL_MD2_PI[t]);
            t = state[k];
        }
        t = (t + (uint)j) & 0xFFu;
    }

    // Spec step 3: per-data-block checksum update (skipped on final
    // checksum-block call per RFC errata).
    if (update_checksum) {
        t = checksum[15];
        for (j = 0; j < 16; j++) {
            checksum[j] = (uchar)(checksum[j] ^ MTL_MD2_PI[data[j] ^ t]);
            t = checksum[j];
        }
    }
}

/* --- md4_block: single 64-byte MD4 compress block (RFC 1320).
 *
 * Lifted 2026-05-23 from metal_md4_core.metal::md4_compress for the
 * hx codegen sub-phase 5a.4 e122 MD4MD5PASS family emit. Mirrors
 * gpu_common.cl::md4_block byte-for-byte (constants match; round
 * order matches). Same IV as MD5; message schedule is 16 uint32 LE
 * words (caller packs the 64-byte block into M[0..15] little-endian).
 * Three rounds of 16 steps each (F/G/H), round-2 constant
 * 0x5A827999u, round-3 constant 0x6ED9EBA1u, round-1 constant 0.
 *
 * Signature uses pointer-state convention `thread uint *state`
 * matching sha1_block / sha256_block (cl2metal.py rev preserves
 * call-site addr-of via _BLOCK_HELPERS_POINTER_STATE). The OpenCL
 * twin uses 4 separate `uint *h0..h3` args; this Metal version takes
 * a single 4-uint state pointer for register-pressure parity with
 * other BE-family blocks. The hx family emit body (Metal) calls
 * md4_block via the pointer-state convention.
 *
 * Output is LE; NO byte-swap needed before compact_fp probe. */
static inline void md4_block(thread uint *state, thread const uint *M)
{
    uint a = state[0], b = state[1], c = state[2], d = state[3];
#define MTL_MD4_F(x,y,z) (((x)&(y)) | ((~(x))&(z)))
#define MTL_MD4_G(x,y,z) (((x)&(y)) | ((x)&(z)) | ((y)&(z)))
#define MTL_MD4_H(x,y,z) ((x)^(y)^(z))
#define MTL_MD4_R1(a,b,c,d,k,s) a = rotl32(a + MTL_MD4_F(b,c,d) + M[k], (uint)(s))
#define MTL_MD4_R2(a,b,c,d,k,s) a = rotl32(a + MTL_MD4_G(b,c,d) + M[k] + 0x5A827999u, (uint)(s))
#define MTL_MD4_R3(a,b,c,d,k,s) a = rotl32(a + MTL_MD4_H(b,c,d) + M[k] + 0x6ED9EBA1u, (uint)(s))
    MTL_MD4_R1(a,b,c,d, 0, 3); MTL_MD4_R1(d,a,b,c, 1, 7); MTL_MD4_R1(c,d,a,b, 2,11); MTL_MD4_R1(b,c,d,a, 3,19);
    MTL_MD4_R1(a,b,c,d, 4, 3); MTL_MD4_R1(d,a,b,c, 5, 7); MTL_MD4_R1(c,d,a,b, 6,11); MTL_MD4_R1(b,c,d,a, 7,19);
    MTL_MD4_R1(a,b,c,d, 8, 3); MTL_MD4_R1(d,a,b,c, 9, 7); MTL_MD4_R1(c,d,a,b,10,11); MTL_MD4_R1(b,c,d,a,11,19);
    MTL_MD4_R1(a,b,c,d,12, 3); MTL_MD4_R1(d,a,b,c,13, 7); MTL_MD4_R1(c,d,a,b,14,11); MTL_MD4_R1(b,c,d,a,15,19);
    MTL_MD4_R2(a,b,c,d, 0, 3); MTL_MD4_R2(d,a,b,c, 4, 5); MTL_MD4_R2(c,d,a,b, 8, 9); MTL_MD4_R2(b,c,d,a,12,13);
    MTL_MD4_R2(a,b,c,d, 1, 3); MTL_MD4_R2(d,a,b,c, 5, 5); MTL_MD4_R2(c,d,a,b, 9, 9); MTL_MD4_R2(b,c,d,a,13,13);
    MTL_MD4_R2(a,b,c,d, 2, 3); MTL_MD4_R2(d,a,b,c, 6, 5); MTL_MD4_R2(c,d,a,b,10, 9); MTL_MD4_R2(b,c,d,a,14,13);
    MTL_MD4_R2(a,b,c,d, 3, 3); MTL_MD4_R2(d,a,b,c, 7, 5); MTL_MD4_R2(c,d,a,b,11, 9); MTL_MD4_R2(b,c,d,a,15,13);
    MTL_MD4_R3(a,b,c,d, 0, 3); MTL_MD4_R3(d,a,b,c, 8, 9); MTL_MD4_R3(c,d,a,b, 4,11); MTL_MD4_R3(b,c,d,a,12,15);
    MTL_MD4_R3(a,b,c,d, 2, 3); MTL_MD4_R3(d,a,b,c,10, 9); MTL_MD4_R3(c,d,a,b, 6,11); MTL_MD4_R3(b,c,d,a,14,15);
    MTL_MD4_R3(a,b,c,d, 1, 3); MTL_MD4_R3(d,a,b,c, 9, 9); MTL_MD4_R3(c,d,a,b, 5,11); MTL_MD4_R3(b,c,d,a,13,15);
    MTL_MD4_R3(a,b,c,d, 3, 3); MTL_MD4_R3(d,a,b,c,11, 9); MTL_MD4_R3(c,d,a,b, 7,11); MTL_MD4_R3(b,c,d,a,15,15);
#undef MTL_MD4_F
#undef MTL_MD4_G
#undef MTL_MD4_H
#undef MTL_MD4_R1
#undef MTL_MD4_R2
#undef MTL_MD4_R3
    state[0] += a; state[1] += b; state[2] += c; state[3] += d;
}

/* --- md5_block: single 64-byte MD5 compress block.
 *
 * Mirrors gpu_common.cl::md5_block byte-for-byte (constants match;
 * round order matches). M[] is `thread const uint *` (Pattern 1):
 * the only call site lives inside template_finalize which builds
 * M[] on the per-lane thread stack.
 *
 * Pattern 3: static inline. */
static inline void md5_block(thread uint &h0, thread uint &h1,
                             thread uint &h2, thread uint &h3,
                             thread const uint *M)
{
    uint a = h0, b = h1, c = h2, d = h3;
    MTL_MD5_FF(a,b,c,d,M[0], 7,0xd76aa478u);  MTL_MD5_FF(d,a,b,c,M[1],12,0xe8c7b756u);
    MTL_MD5_FF(c,d,a,b,M[2],17,0x242070dbu);  MTL_MD5_FF(b,c,d,a,M[3],22,0xc1bdceeeu);
    MTL_MD5_FF(a,b,c,d,M[4], 7,0xf57c0fafu);  MTL_MD5_FF(d,a,b,c,M[5],12,0x4787c62au);
    MTL_MD5_FF(c,d,a,b,M[6],17,0xa8304613u);  MTL_MD5_FF(b,c,d,a,M[7],22,0xfd469501u);
    MTL_MD5_FF(a,b,c,d,M[8], 7,0x698098d8u);  MTL_MD5_FF(d,a,b,c,M[9],12,0x8b44f7afu);
    MTL_MD5_FF(c,d,a,b,M[10],17,0xffff5bb1u); MTL_MD5_FF(b,c,d,a,M[11],22,0x895cd7beu);
    MTL_MD5_FF(a,b,c,d,M[12], 7,0x6b901122u); MTL_MD5_FF(d,a,b,c,M[13],12,0xfd987193u);
    MTL_MD5_FF(c,d,a,b,M[14],17,0xa679438eu); MTL_MD5_FF(b,c,d,a,M[15],22,0x49b40821u);
    MTL_MD5_GG(a,b,c,d,M[1], 5,0xf61e2562u);  MTL_MD5_GG(d,a,b,c,M[6], 9,0xc040b340u);
    MTL_MD5_GG(c,d,a,b,M[11],14,0x265e5a51u); MTL_MD5_GG(b,c,d,a,M[0],20,0xe9b6c7aau);
    MTL_MD5_GG(a,b,c,d,M[5], 5,0xd62f105du);  MTL_MD5_GG(d,a,b,c,M[10], 9,0x02441453u);
    MTL_MD5_GG(c,d,a,b,M[15],14,0xd8a1e681u); MTL_MD5_GG(b,c,d,a,M[4],20,0xe7d3fbc8u);
    MTL_MD5_GG(a,b,c,d,M[9], 5,0x21e1cde6u);  MTL_MD5_GG(d,a,b,c,M[14], 9,0xc33707d6u);
    MTL_MD5_GG(c,d,a,b,M[3],14,0xf4d50d87u);  MTL_MD5_GG(b,c,d,a,M[8],20,0x455a14edu);
    MTL_MD5_GG(a,b,c,d,M[13], 5,0xa9e3e905u); MTL_MD5_GG(d,a,b,c,M[2], 9,0xfcefa3f8u);
    MTL_MD5_GG(c,d,a,b,M[7],14,0x676f02d9u);  MTL_MD5_GG(b,c,d,a,M[12],20,0x8d2a4c8au);
    MTL_MD5_HH(a,b,c,d,M[5], 4,0xfffa3942u);  MTL_MD5_HH(d,a,b,c,M[8],11,0x8771f681u);
    MTL_MD5_HH(c,d,a,b,M[11],16,0x6d9d6122u); MTL_MD5_HH(b,c,d,a,M[14],23,0xfde5380cu);
    MTL_MD5_HH(a,b,c,d,M[1], 4,0xa4beea44u);  MTL_MD5_HH(d,a,b,c,M[4],11,0x4bdecfa9u);
    MTL_MD5_HH(c,d,a,b,M[7],16,0xf6bb4b60u);  MTL_MD5_HH(b,c,d,a,M[10],23,0xbebfbc70u);
    MTL_MD5_HH(a,b,c,d,M[13], 4,0x289b7ec6u); MTL_MD5_HH(d,a,b,c,M[0],11,0xeaa127fau);
    MTL_MD5_HH(c,d,a,b,M[3],16,0xd4ef3085u);  MTL_MD5_HH(b,c,d,a,M[6],23,0x04881d05u);
    MTL_MD5_HH(a,b,c,d,M[9], 4,0xd9d4d039u);  MTL_MD5_HH(d,a,b,c,M[12],11,0xe6db99e5u);
    MTL_MD5_HH(c,d,a,b,M[15],16,0x1fa27cf8u); MTL_MD5_HH(b,c,d,a,M[2],23,0xc4ac5665u);
    MTL_MD5_II(a,b,c,d,M[0], 6,0xf4292244u);  MTL_MD5_II(d,a,b,c,M[7],10,0x432aff97u);
    MTL_MD5_II(c,d,a,b,M[14],15,0xab9423a7u); MTL_MD5_II(b,c,d,a,M[5],21,0xfc93a039u);
    MTL_MD5_II(a,b,c,d,M[12], 6,0x655b59c3u); MTL_MD5_II(d,a,b,c,M[3],10,0x8f0ccc92u);
    MTL_MD5_II(c,d,a,b,M[10],15,0xffeff47du); MTL_MD5_II(b,c,d,a,M[1],21,0x85845dd1u);
    MTL_MD5_II(a,b,c,d,M[8], 6,0x6fa87e4fu);  MTL_MD5_II(d,a,b,c,M[15],10,0xfe2ce6e0u);
    MTL_MD5_II(c,d,a,b,M[6],15,0xa3014314u);  MTL_MD5_II(b,c,d,a,M[13],21,0x4e0811a1u);
    MTL_MD5_II(a,b,c,d,M[4], 6,0xf7537e82u);  MTL_MD5_II(d,a,b,c,M[11],10,0xbd3af235u);
    MTL_MD5_II(c,d,a,b,M[2],15,0x2ad7d2bbu);  MTL_MD5_II(b,c,d,a,M[9],21,0xeb86d391u);
    h0 += a; h1 += b; h2 += c; h3 += d;
}

/* --- bswap32: 32-bit byte swap (mirrors gpu_common.cl line 755).
 *
 * Used by SHA-family digest_compare to convert big-endian state words to
 * little-endian for the compact-table probe key. Pattern 3: static inline. */
static inline uint bswap32(uint x) {
    return ((x >> 24) & 0xffu) | ((x >> 8) & 0xff00u) |
           ((x << 8) & 0xff0000u) | ((x << 24) & 0xff000000u);
}

/* --- sha1_block: single 64-byte SHA-1 compress block.
 *
 * Mirrors gpu_common.cl::sha1_block (RFC 3174) byte-for-byte: round
 * constants 0x5A827999u/0x6ED9EBA1u/0x8F1BBCDCu/0xCA62C1D6u, 4×20-step
 * loop structure, in-place schedule W[80] built from the input message
 * words.
 *
 * Signature (Phase 2d.3.1 SHA-1 canary): keeps the POINTER-state shape
 * `thread uint *state, thread const uint *M`. Unlike md5_block (which
 * splits the 4-word state into 4 separate `thread uint &` args for
 * scalar-ref efficiency), SHA-1's 5-word state stays as a single
 * pointer. Rationale: 5 separate scalar-ref args bloat the call
 * boilerplate without measurable Apple GPU codegen benefit, and
 * pointer-state matches the OpenCL twin verbatim (translator cl2metal.py
 * rev 1.3+ preserves `&st.h[0]` at call sites via the discovered-
 * scalar-state-helpers split). State writes happen in-place via the
 * `state[0] += a; ...; state[4] += e;` epilogue.
 *
 * Pattern 1: both pointer args explicitly thread-qualified. Pattern 3:
 * static inline (per-TU; no link collision with sha1_block in other
 * family TUs).
 *
 * R2 register pressure (gfx1201 / Apple M-series): W[80] = 80 uint32 =
 * 320 bytes private stack per lane. Phase 2d.3.1 dev3 M2 Max smoke
 * measures priv_mem_size ~ same band as MD5's 64-byte M[16] + 4-word
 * state (M[] alone amortises smaller; W[] is a one-shot allocation
 * inside this fn so the lifetime is short). */
static inline void sha1_block(thread uint *state, thread const uint *M)
{
    uint W[80];
    for (int i = 0; i < 16; i++) W[i] = M[i];
    for (int i = 16; i < 80; i++)
        W[i] = rotl32(W[i-3] ^ W[i-8] ^ W[i-14] ^ W[i-16], (uint)1);

    uint a = state[0], b = state[1], c = state[2], d = state[3], e = state[4];
    uint t;
    for (int i = 0; i < 20; i++) {
        t = rotl32(a, (uint)5) + ((b & c) | (~b & d)) + e + 0x5A827999u + W[i];
        e = d; d = c; c = rotl32(b, (uint)30); b = a; a = t;
    }
    for (int i = 20; i < 40; i++) {
        t = rotl32(a, (uint)5) + (b ^ c ^ d) + e + 0x6ED9EBA1u + W[i];
        e = d; d = c; c = rotl32(b, (uint)30); b = a; a = t;
    }
    for (int i = 40; i < 60; i++) {
        t = rotl32(a, (uint)5) + ((b & c) | (b & d) | (c & d)) + e + 0x8F1BBCDCu + W[i];
        e = d; d = c; c = rotl32(b, (uint)30); b = a; a = t;
    }
    for (int i = 60; i < 80; i++) {
        t = rotl32(a, (uint)5) + (b ^ c ^ d) + e + 0xCA62C1D6u + W[i];
        e = d; d = c; c = rotl32(b, (uint)30); b = a; a = t;
    }
    state[0] += a; state[1] += b; state[2] += c; state[3] += d; state[4] += e;
}

/* --- SHA-256 round constants + helper macros (mirrors gpu_common.cl
 * lines 890-907). Constants namespaced MTL_SHA256_K per Pattern 2 to
 * avoid bare K[] collision with MD5 / SHA1 / SHA512 constants in the
 * same metallib TU when build_metallib.sh consolidates a multi-family
 * TU (it currently doesn't, but the namespace discipline is structural
 * insurance per Pattern 2). The OpenCL twin uses bare `SHA256_K` which
 * is permissible only because gpu_common.cl is the ONE compilation unit
 * carrying it; on the Metal side every family TU could include
 * metal_common, so we namespace. */
constant uint MTL_SHA256_K[64] = {
    0x428a2f98u,0x71374491u,0xb5c0fbcfu,0xe9b5dba5u,0x3956c25bu,0x59f111f1u,0x923f82a4u,0xab1c5ed5u,
    0xd807aa98u,0x12835b01u,0x243185beu,0x550c7dc3u,0x72be5d74u,0x80deb1feu,0x9bdc06a7u,0xc19bf174u,
    0xe49b69c1u,0xefbe4786u,0x0fc19dc6u,0x240ca1ccu,0x2de92c6fu,0x4a7484aau,0x5cb0a9dcu,0x76f988dau,
    0x983e5152u,0xa831c66du,0xb00327c8u,0xbf597fc7u,0xc6e00bf3u,0xd5a79147u,0x06ca6351u,0x14292967u,
    0x27b70a85u,0x2e1b2138u,0x4d2c6dfcu,0x53380d13u,0x650a7354u,0x766a0abbu,0x81c2c92eu,0x92722c85u,
    0xa2bfe8a1u,0xa81a664bu,0xc24b8b70u,0xc76c51a3u,0xd192e819u,0xd6990624u,0xf40e3585u,0x106aa070u,
    0x19a4c116u,0x1e376c08u,0x2748774cu,0x34b0bcb5u,0x391c0cb3u,0x4ed8aa4au,0x5b9cca4fu,0x682e6ff3u,
    0x748f82eeu,0x78a5636fu,0x84c87814u,0x8cc70208u,0x90befffau,0xa4506cebu,0xbef9a3f7u,0xc67178f2u
};

/* SHA-256 helper macros. The OpenCL twin uses bare names S256_ROTR etc.;
 * Metal namespaces them MTL_S256_* per Pattern 2 (avoid future collision
 * with SHA-512 helpers that may share similar S512_ROTR/EP/SIG shapes).
 * Use the `rotate` builtin via metal::rotate. Mirrors gpu_common.cl
 * lines 901-907 byte-for-byte. */
#define MTL_S256_ROTR(x,n) rotate((uint)(x), (uint)(32u-(n)))
#define MTL_S256_CH(x,y,z)  ((x & y) ^ (~x & z))
#define MTL_S256_MAJ(x,y,z) ((x & y) ^ (x & z) ^ (y & z))
#define MTL_S256_EP0(x)  (MTL_S256_ROTR(x,2u)  ^ MTL_S256_ROTR(x,13u) ^ MTL_S256_ROTR(x,22u))
#define MTL_S256_EP1(x)  (MTL_S256_ROTR(x,6u)  ^ MTL_S256_ROTR(x,11u) ^ MTL_S256_ROTR(x,25u))
#define MTL_S256_SIG0(x) (MTL_S256_ROTR(x,7u)  ^ MTL_S256_ROTR(x,18u) ^ (x >> 3))
#define MTL_S256_SIG1(x) (MTL_S256_ROTR(x,17u) ^ MTL_S256_ROTR(x,19u) ^ (x >> 10))

/* --- sha256_block: single 64-byte SHA-256 compress block.
 *
 * Mirrors gpu_common.cl::sha256_block (FIPS 180-4 §6.2) byte-for-byte:
 * SHA256_K round constants, 64-round single loop with S256_EP1/EP0/CH/MAJ,
 * message schedule W[64] built from M[0..15] with S256_SIG0/SIG1.
 *
 * Signature (Phase 2d.4.1 SHA-2/256 canary): POINTER-state shape
 * `thread uint *state, thread const uint *M`, matching sha1_block. The
 * 8-word chaining state stays as a single pointer (vs MD5_block's
 * 4-scalar-by-ref shape). State writes happen in-place via the
 * `state[0] += a; ...; state[7] += h;` epilogue.
 *
 * Pattern 1: both pointer args explicitly thread-qualified.
 * Pattern 3: static inline (per-TU; no link collision).
 *
 * R2 register pressure: W[64] = 64 uint32 = 256 bytes private stack per
 * lane. Smaller than SHA-1's W[80] = 320 bytes. Expected priv_mem_size
 * on Apple Silicon comparable to SHA-1 (rules walker buf[] dominates).
 * Watch this for the SHA-512 family which uses ulong W[80] = 640 bytes. */
static inline void sha256_block(thread uint *state, thread const uint *M)
{
    uint W[64];
    for (int i = 0; i < 16; i++) W[i] = M[i];
    for (int i = 16; i < 64; i++)
        W[i] = MTL_S256_SIG1(W[i-2]) + W[i-7] + MTL_S256_SIG0(W[i-15]) + W[i-16];

    uint a = state[0], b = state[1], c = state[2], d = state[3];
    uint e = state[4], f = state[5], g = state[6], h = state[7];

    for (int i = 0; i < 64; i++) {
        uint t1 = h + MTL_S256_EP1(e) + MTL_S256_CH(e,f,g) + MTL_SHA256_K[i] + W[i];
        uint t2 = MTL_S256_EP0(a) + MTL_S256_MAJ(a,b,c);
        h = g; g = f; f = e; e = d + t1;
        d = c; c = b; b = a; a = t1 + t2;
    }

    state[0] += a; state[1] += b; state[2] += c; state[3] += d;
    state[4] += e; state[5] += f; state[6] += g; state[7] += h;
}

/* --- bswap64: 64-bit byte swap (mirrors gpu_common.cl line 760).
 *
 * Phase 2d.5.1 SHA-512 canary -- FIRST 64-bit-state family on Metal.
 * Used by sha512_core's template_state_to_h() to convert big-endian
 * ulong state words to LE before decomposing into uint32 pairs for
 * digest emit + compact-table probe key. Pattern 3: static inline. */
static inline ulong bswap64(ulong x) {
    return ((x >> 56) & 0xffUL)              | ((x >> 40) & 0xff00UL) |
           ((x >> 24) & 0xff0000UL)          | ((x >> 8)  & 0xff000000UL) |
           ((x << 8)  & 0xff00000000UL)      | ((x << 24) & 0xff0000000000UL) |
           ((x << 40) & 0xff000000000000UL)  | ((x << 56) & 0xff00000000000000UL);
}

/* --- rotr64: 64-bit rotate-right (mirrors gpu_common.cl line 767).
 *
 * Phase 2d.5.1: used by sha512_block's S0/S1/sigma round functions.
 * Pattern 3: static inline. */
static inline ulong rotr64(ulong x, uint n) {
    return (x >> n) | (x << (64u - n));
}

/* --- SHA-512 round constants + helper macros (mirrors gpu_common.cl
 * lines 931-952, with the K512 80-entry table).
 *
 * Constants namespaced MTL_SHA512_K per Pattern 2 -- avoids collision
 * with K512 in any future TU consolidation. The OpenCL twin uses bare
 * `K512` which is permissible because gpu_common.cl is ONE compilation
 * unit; on the Metal side every family TU could include metal_common,
 * so we namespace. SHA-512 helper macros similarly MTL_S512_*. */
constant ulong MTL_SHA512_K[80] = {
    0x428a2f98d728ae22UL, 0x7137449123ef65cdUL, 0xb5c0fbcfec4d3b2fUL, 0xe9b5dba58189dbbcUL,
    0x3956c25bf348b538UL, 0x59f111f1b605d019UL, 0x923f82a4af194f9bUL, 0xab1c5ed5da6d8118UL,
    0xd807aa98a3030242UL, 0x12835b0145706fbeUL, 0x243185be4ee4b28cUL, 0x550c7dc3d5ffb4e2UL,
    0x72be5d74f27b896fUL, 0x80deb1fe3b1696b1UL, 0x9bdc06a725c71235UL, 0xc19bf174cf692694UL,
    0xe49b69c19ef14ad2UL, 0xefbe4786384f25e3UL, 0x0fc19dc68b8cd5b5UL, 0x240ca1cc77ac9c65UL,
    0x2de92c6f592b0275UL, 0x4a7484aa6ea6e483UL, 0x5cb0a9dcbd41fbd4UL, 0x76f988da831153b5UL,
    0x983e5152ee66dfabUL, 0xa831c66d2db43210UL, 0xb00327c898fb213fUL, 0xbf597fc7beef0ee4UL,
    0xc6e00bf33da88fc2UL, 0xd5a79147930aa725UL, 0x06ca6351e003826fUL, 0x142929670a0e6e70UL,
    0x27b70a8546d22ffcUL, 0x2e1b21385c26c926UL, 0x4d2c6dfc5ac42aedUL, 0x53380d139d95b3dfUL,
    0x650a73548baf63deUL, 0x766a0abb3c77b2a8UL, 0x81c2c92e47edaee6UL, 0x92722c851482353bUL,
    0xa2bfe8a14cf10364UL, 0xa81a664bbc423001UL, 0xc24b8b70d0f89791UL, 0xc76c51a30654be30UL,
    0xd192e819d6ef5218UL, 0xd69906245565a910UL, 0xf40e35855771202aUL, 0x106aa07032bbd1b8UL,
    0x19a4c116b8d2d0c8UL, 0x1e376c085141ab53UL, 0x2748774cdf8eeb99UL, 0x34b0bcb5e19b48a8UL,
    0x391c0cb3c5c95a63UL, 0x4ed8aa4ae3418acbUL, 0x5b9cca4f7763e373UL, 0x682e6ff3d6b2b8a3UL,
    0x748f82ee5defb2fcUL, 0x78a5636f43172f60UL, 0x84c87814a1f0ab72UL, 0x8cc702081a6439ecUL,
    0x90befffa23631e28UL, 0xa4506cebde82bde9UL, 0xbef9a3f7b2c67915UL, 0xc67178f2e372532bUL,
    0xca273eceea26619cUL, 0xd186b8c721c0c207UL, 0xeada7dd6cde0eb1eUL, 0xf57d4f7fee6ed178UL,
    0x06f067aa72176fbaUL, 0x0a637dc5a2c898a6UL, 0x113f9804bef90daeUL, 0x1b710b35131c471bUL,
    0x28db77f523047d84UL, 0x32caab7b40c72493UL, 0x3c9ebe0a15c9bebcUL, 0x431d67c49c100d4cUL,
    0x4cc5d4becb3e42b6UL, 0x597f299cfc657e2aUL, 0x5fcb6fab3ad6faecUL, 0x6c44198c4a475817UL
};

/* --- SHA-512 scalar helper macros for flat-unrolled sha512_block body.
 * Step A Metal port (mirrors gpu_common.cl rev 1.22 MDX_SHA512_STEP_S pattern).
 * Avoids W[80] array spill on GPU; Metal compiler benefits from inlining
 * (opposite of Pascal NVIDIA noinline discipline -- per
 * feedback_md5_block_noinline_pascal.md, Metal inlines and benefits).
 *
 * MTL_SHA512_S0_S/S1_S: big-sigma (compression). S2_S/S3_S: small-sigma (schedule).
 * MTL_SHA512_F0o: Ch = (z)^((x)&((y)^(z))) -- arithmetic form (Metal lacks bitselect).
 * MTL_SHA512_F1o: Maj = ((x)&(y))|((z)&((x)^(y))) -- arithmetic Maj.
 * MTL_SHA512_STEP_S: one round; caller rotates arg order, not register names.
 * MTL_SHA512_EXPAND_S: w[i] for i>=16 from four prior words. */
#define MTL_SHA512_S0_S(x) (rotr64((x), 28u) ^ rotr64((x), 34u) ^ rotr64((x), 39u))
#define MTL_SHA512_S1_S(x) (rotr64((x), 14u) ^ rotr64((x), 18u) ^ rotr64((x), 41u))
#define MTL_SHA512_S2_S(x) (rotr64((x),  1u) ^ rotr64((x),  8u) ^ ((x) >> 7))
#define MTL_SHA512_S3_S(x) (rotr64((x), 19u) ^ rotr64((x), 61u) ^ ((x) >> 6))
#define MTL_SHA512_F0o(x,y,z) ((z) ^ ((x) & ((y) ^ (z))))
#define MTL_SHA512_F1o(x,y,z) (((x) & (y)) | ((z) & ((x) ^ (y))))
#define MTL_SHA512_STEP_S(a,b,c,d,e,f,g,h,x,K) \
{ \
    (h) += (K); \
    (h) += (x); \
    (h) += MTL_SHA512_S1_S(e); \
    (h) += MTL_SHA512_F0o((e),(f),(g)); \
    (d) += (h); \
    (h) += MTL_SHA512_S0_S(a); \
    (h) += MTL_SHA512_F1o((a),(b),(c)); \
}
#define MTL_SHA512_EXPAND_S(x,y,z,w) \
    (MTL_SHA512_S3_S(x) + (y) + MTL_SHA512_S2_S(z) + (w))

/* --- sha512_block: single 128-byte SHA-512 compress block.
 *
 * Step A Metal port: flat-unrolled 80-step scalar body replacing the
 * W[80] array + for-loop form. 16 scalar w0_t..wf_t hold the message
 * schedule; STEP_S + EXPAND_S macros inline each round. Eliminates
 * the 640-byte W[] private-stack allocation entirely.
 *
 * Metal compiler inlining policy is the OPPOSITE of Pascal NVIDIA
 * (per feedback_md5_block_noinline_pascal.md): Apple Metal benefits
 * from inlining sha512_block. KEEP static inline; NO noinline attr.
 * R2 register pressure on M-series is lower than Pascal because
 * Apple Metal scheduler handles the flat scalar form well.
 *
 * Signature unchanged: thread ulong *state, thread const ulong *M.
 * Pattern 1: both pointer args thread-qualified. Pattern 3: static inline. */
static inline void sha512_block(thread ulong *state, thread const ulong *M)
{
    ulong w0_t = M[0];
    ulong w1_t = M[1];
    ulong w2_t = M[2];
    ulong w3_t = M[3];
    ulong w4_t = M[4];
    ulong w5_t = M[5];
    ulong w6_t = M[6];
    ulong w7_t = M[7];
    ulong w8_t = M[8];
    ulong w9_t = M[9];
    ulong wa_t = M[10];
    ulong wb_t = M[11];
    ulong wc_t = M[12];
    ulong wd_t = M[13];
    ulong we_t = M[14];
    ulong wf_t = M[15];
    ulong a = state[0], b = state[1], c = state[2], d = state[3];
    ulong e = state[4], f = state[5], g = state[6], h = state[7];
    MTL_SHA512_STEP_S(a, b, c, d, e, f, g, h, w0_t, MTL_SHA512_K[0]);
    MTL_SHA512_STEP_S(h, a, b, c, d, e, f, g, w1_t, MTL_SHA512_K[1]);
    MTL_SHA512_STEP_S(g, h, a, b, c, d, e, f, w2_t, MTL_SHA512_K[2]);
    MTL_SHA512_STEP_S(f, g, h, a, b, c, d, e, w3_t, MTL_SHA512_K[3]);
    MTL_SHA512_STEP_S(e, f, g, h, a, b, c, d, w4_t, MTL_SHA512_K[4]);
    MTL_SHA512_STEP_S(d, e, f, g, h, a, b, c, w5_t, MTL_SHA512_K[5]);
    MTL_SHA512_STEP_S(c, d, e, f, g, h, a, b, w6_t, MTL_SHA512_K[6]);
    MTL_SHA512_STEP_S(b, c, d, e, f, g, h, a, w7_t, MTL_SHA512_K[7]);
    MTL_SHA512_STEP_S(a, b, c, d, e, f, g, h, w8_t, MTL_SHA512_K[8]);
    MTL_SHA512_STEP_S(h, a, b, c, d, e, f, g, w9_t, MTL_SHA512_K[9]);
    MTL_SHA512_STEP_S(g, h, a, b, c, d, e, f, wa_t, MTL_SHA512_K[10]);
    MTL_SHA512_STEP_S(f, g, h, a, b, c, d, e, wb_t, MTL_SHA512_K[11]);
    MTL_SHA512_STEP_S(e, f, g, h, a, b, c, d, wc_t, MTL_SHA512_K[12]);
    MTL_SHA512_STEP_S(d, e, f, g, h, a, b, c, wd_t, MTL_SHA512_K[13]);
    MTL_SHA512_STEP_S(c, d, e, f, g, h, a, b, we_t, MTL_SHA512_K[14]);
    MTL_SHA512_STEP_S(b, c, d, e, f, g, h, a, wf_t, MTL_SHA512_K[15]);
    w0_t = MTL_SHA512_EXPAND_S(we_t, w9_t, w1_t, w0_t);
    MTL_SHA512_STEP_S(a, b, c, d, e, f, g, h, w0_t, MTL_SHA512_K[16]);
    w1_t = MTL_SHA512_EXPAND_S(wf_t, wa_t, w2_t, w1_t);
    MTL_SHA512_STEP_S(h, a, b, c, d, e, f, g, w1_t, MTL_SHA512_K[17]);
    w2_t = MTL_SHA512_EXPAND_S(w0_t, wb_t, w3_t, w2_t);
    MTL_SHA512_STEP_S(g, h, a, b, c, d, e, f, w2_t, MTL_SHA512_K[18]);
    w3_t = MTL_SHA512_EXPAND_S(w1_t, wc_t, w4_t, w3_t);
    MTL_SHA512_STEP_S(f, g, h, a, b, c, d, e, w3_t, MTL_SHA512_K[19]);
    w4_t = MTL_SHA512_EXPAND_S(w2_t, wd_t, w5_t, w4_t);
    MTL_SHA512_STEP_S(e, f, g, h, a, b, c, d, w4_t, MTL_SHA512_K[20]);
    w5_t = MTL_SHA512_EXPAND_S(w3_t, we_t, w6_t, w5_t);
    MTL_SHA512_STEP_S(d, e, f, g, h, a, b, c, w5_t, MTL_SHA512_K[21]);
    w6_t = MTL_SHA512_EXPAND_S(w4_t, wf_t, w7_t, w6_t);
    MTL_SHA512_STEP_S(c, d, e, f, g, h, a, b, w6_t, MTL_SHA512_K[22]);
    w7_t = MTL_SHA512_EXPAND_S(w5_t, w0_t, w8_t, w7_t);
    MTL_SHA512_STEP_S(b, c, d, e, f, g, h, a, w7_t, MTL_SHA512_K[23]);
    w8_t = MTL_SHA512_EXPAND_S(w6_t, w1_t, w9_t, w8_t);
    MTL_SHA512_STEP_S(a, b, c, d, e, f, g, h, w8_t, MTL_SHA512_K[24]);
    w9_t = MTL_SHA512_EXPAND_S(w7_t, w2_t, wa_t, w9_t);
    MTL_SHA512_STEP_S(h, a, b, c, d, e, f, g, w9_t, MTL_SHA512_K[25]);
    wa_t = MTL_SHA512_EXPAND_S(w8_t, w3_t, wb_t, wa_t);
    MTL_SHA512_STEP_S(g, h, a, b, c, d, e, f, wa_t, MTL_SHA512_K[26]);
    wb_t = MTL_SHA512_EXPAND_S(w9_t, w4_t, wc_t, wb_t);
    MTL_SHA512_STEP_S(f, g, h, a, b, c, d, e, wb_t, MTL_SHA512_K[27]);
    wc_t = MTL_SHA512_EXPAND_S(wa_t, w5_t, wd_t, wc_t);
    MTL_SHA512_STEP_S(e, f, g, h, a, b, c, d, wc_t, MTL_SHA512_K[28]);
    wd_t = MTL_SHA512_EXPAND_S(wb_t, w6_t, we_t, wd_t);
    MTL_SHA512_STEP_S(d, e, f, g, h, a, b, c, wd_t, MTL_SHA512_K[29]);
    we_t = MTL_SHA512_EXPAND_S(wc_t, w7_t, wf_t, we_t);
    MTL_SHA512_STEP_S(c, d, e, f, g, h, a, b, we_t, MTL_SHA512_K[30]);
    wf_t = MTL_SHA512_EXPAND_S(wd_t, w8_t, w0_t, wf_t);
    MTL_SHA512_STEP_S(b, c, d, e, f, g, h, a, wf_t, MTL_SHA512_K[31]);
    w0_t = MTL_SHA512_EXPAND_S(we_t, w9_t, w1_t, w0_t);
    MTL_SHA512_STEP_S(a, b, c, d, e, f, g, h, w0_t, MTL_SHA512_K[32]);
    w1_t = MTL_SHA512_EXPAND_S(wf_t, wa_t, w2_t, w1_t);
    MTL_SHA512_STEP_S(h, a, b, c, d, e, f, g, w1_t, MTL_SHA512_K[33]);
    w2_t = MTL_SHA512_EXPAND_S(w0_t, wb_t, w3_t, w2_t);
    MTL_SHA512_STEP_S(g, h, a, b, c, d, e, f, w2_t, MTL_SHA512_K[34]);
    w3_t = MTL_SHA512_EXPAND_S(w1_t, wc_t, w4_t, w3_t);
    MTL_SHA512_STEP_S(f, g, h, a, b, c, d, e, w3_t, MTL_SHA512_K[35]);
    w4_t = MTL_SHA512_EXPAND_S(w2_t, wd_t, w5_t, w4_t);
    MTL_SHA512_STEP_S(e, f, g, h, a, b, c, d, w4_t, MTL_SHA512_K[36]);
    w5_t = MTL_SHA512_EXPAND_S(w3_t, we_t, w6_t, w5_t);
    MTL_SHA512_STEP_S(d, e, f, g, h, a, b, c, w5_t, MTL_SHA512_K[37]);
    w6_t = MTL_SHA512_EXPAND_S(w4_t, wf_t, w7_t, w6_t);
    MTL_SHA512_STEP_S(c, d, e, f, g, h, a, b, w6_t, MTL_SHA512_K[38]);
    w7_t = MTL_SHA512_EXPAND_S(w5_t, w0_t, w8_t, w7_t);
    MTL_SHA512_STEP_S(b, c, d, e, f, g, h, a, w7_t, MTL_SHA512_K[39]);
    w8_t = MTL_SHA512_EXPAND_S(w6_t, w1_t, w9_t, w8_t);
    MTL_SHA512_STEP_S(a, b, c, d, e, f, g, h, w8_t, MTL_SHA512_K[40]);
    w9_t = MTL_SHA512_EXPAND_S(w7_t, w2_t, wa_t, w9_t);
    MTL_SHA512_STEP_S(h, a, b, c, d, e, f, g, w9_t, MTL_SHA512_K[41]);
    wa_t = MTL_SHA512_EXPAND_S(w8_t, w3_t, wb_t, wa_t);
    MTL_SHA512_STEP_S(g, h, a, b, c, d, e, f, wa_t, MTL_SHA512_K[42]);
    wb_t = MTL_SHA512_EXPAND_S(w9_t, w4_t, wc_t, wb_t);
    MTL_SHA512_STEP_S(f, g, h, a, b, c, d, e, wb_t, MTL_SHA512_K[43]);
    wc_t = MTL_SHA512_EXPAND_S(wa_t, w5_t, wd_t, wc_t);
    MTL_SHA512_STEP_S(e, f, g, h, a, b, c, d, wc_t, MTL_SHA512_K[44]);
    wd_t = MTL_SHA512_EXPAND_S(wb_t, w6_t, we_t, wd_t);
    MTL_SHA512_STEP_S(d, e, f, g, h, a, b, c, wd_t, MTL_SHA512_K[45]);
    we_t = MTL_SHA512_EXPAND_S(wc_t, w7_t, wf_t, we_t);
    MTL_SHA512_STEP_S(c, d, e, f, g, h, a, b, we_t, MTL_SHA512_K[46]);
    wf_t = MTL_SHA512_EXPAND_S(wd_t, w8_t, w0_t, wf_t);
    MTL_SHA512_STEP_S(b, c, d, e, f, g, h, a, wf_t, MTL_SHA512_K[47]);
    w0_t = MTL_SHA512_EXPAND_S(we_t, w9_t, w1_t, w0_t);
    MTL_SHA512_STEP_S(a, b, c, d, e, f, g, h, w0_t, MTL_SHA512_K[48]);
    w1_t = MTL_SHA512_EXPAND_S(wf_t, wa_t, w2_t, w1_t);
    MTL_SHA512_STEP_S(h, a, b, c, d, e, f, g, w1_t, MTL_SHA512_K[49]);
    w2_t = MTL_SHA512_EXPAND_S(w0_t, wb_t, w3_t, w2_t);
    MTL_SHA512_STEP_S(g, h, a, b, c, d, e, f, w2_t, MTL_SHA512_K[50]);
    w3_t = MTL_SHA512_EXPAND_S(w1_t, wc_t, w4_t, w3_t);
    MTL_SHA512_STEP_S(f, g, h, a, b, c, d, e, w3_t, MTL_SHA512_K[51]);
    w4_t = MTL_SHA512_EXPAND_S(w2_t, wd_t, w5_t, w4_t);
    MTL_SHA512_STEP_S(e, f, g, h, a, b, c, d, w4_t, MTL_SHA512_K[52]);
    w5_t = MTL_SHA512_EXPAND_S(w3_t, we_t, w6_t, w5_t);
    MTL_SHA512_STEP_S(d, e, f, g, h, a, b, c, w5_t, MTL_SHA512_K[53]);
    w6_t = MTL_SHA512_EXPAND_S(w4_t, wf_t, w7_t, w6_t);
    MTL_SHA512_STEP_S(c, d, e, f, g, h, a, b, w6_t, MTL_SHA512_K[54]);
    w7_t = MTL_SHA512_EXPAND_S(w5_t, w0_t, w8_t, w7_t);
    MTL_SHA512_STEP_S(b, c, d, e, f, g, h, a, w7_t, MTL_SHA512_K[55]);
    w8_t = MTL_SHA512_EXPAND_S(w6_t, w1_t, w9_t, w8_t);
    MTL_SHA512_STEP_S(a, b, c, d, e, f, g, h, w8_t, MTL_SHA512_K[56]);
    w9_t = MTL_SHA512_EXPAND_S(w7_t, w2_t, wa_t, w9_t);
    MTL_SHA512_STEP_S(h, a, b, c, d, e, f, g, w9_t, MTL_SHA512_K[57]);
    wa_t = MTL_SHA512_EXPAND_S(w8_t, w3_t, wb_t, wa_t);
    MTL_SHA512_STEP_S(g, h, a, b, c, d, e, f, wa_t, MTL_SHA512_K[58]);
    wb_t = MTL_SHA512_EXPAND_S(w9_t, w4_t, wc_t, wb_t);
    MTL_SHA512_STEP_S(f, g, h, a, b, c, d, e, wb_t, MTL_SHA512_K[59]);
    wc_t = MTL_SHA512_EXPAND_S(wa_t, w5_t, wd_t, wc_t);
    MTL_SHA512_STEP_S(e, f, g, h, a, b, c, d, wc_t, MTL_SHA512_K[60]);
    wd_t = MTL_SHA512_EXPAND_S(wb_t, w6_t, we_t, wd_t);
    MTL_SHA512_STEP_S(d, e, f, g, h, a, b, c, wd_t, MTL_SHA512_K[61]);
    we_t = MTL_SHA512_EXPAND_S(wc_t, w7_t, wf_t, we_t);
    MTL_SHA512_STEP_S(c, d, e, f, g, h, a, b, we_t, MTL_SHA512_K[62]);
    wf_t = MTL_SHA512_EXPAND_S(wd_t, w8_t, w0_t, wf_t);
    MTL_SHA512_STEP_S(b, c, d, e, f, g, h, a, wf_t, MTL_SHA512_K[63]);
    w0_t = MTL_SHA512_EXPAND_S(we_t, w9_t, w1_t, w0_t);
    MTL_SHA512_STEP_S(a, b, c, d, e, f, g, h, w0_t, MTL_SHA512_K[64]);
    w1_t = MTL_SHA512_EXPAND_S(wf_t, wa_t, w2_t, w1_t);
    MTL_SHA512_STEP_S(h, a, b, c, d, e, f, g, w1_t, MTL_SHA512_K[65]);
    w2_t = MTL_SHA512_EXPAND_S(w0_t, wb_t, w3_t, w2_t);
    MTL_SHA512_STEP_S(g, h, a, b, c, d, e, f, w2_t, MTL_SHA512_K[66]);
    w3_t = MTL_SHA512_EXPAND_S(w1_t, wc_t, w4_t, w3_t);
    MTL_SHA512_STEP_S(f, g, h, a, b, c, d, e, w3_t, MTL_SHA512_K[67]);
    w4_t = MTL_SHA512_EXPAND_S(w2_t, wd_t, w5_t, w4_t);
    MTL_SHA512_STEP_S(e, f, g, h, a, b, c, d, w4_t, MTL_SHA512_K[68]);
    w5_t = MTL_SHA512_EXPAND_S(w3_t, we_t, w6_t, w5_t);
    MTL_SHA512_STEP_S(d, e, f, g, h, a, b, c, w5_t, MTL_SHA512_K[69]);
    w6_t = MTL_SHA512_EXPAND_S(w4_t, wf_t, w7_t, w6_t);
    MTL_SHA512_STEP_S(c, d, e, f, g, h, a, b, w6_t, MTL_SHA512_K[70]);
    w7_t = MTL_SHA512_EXPAND_S(w5_t, w0_t, w8_t, w7_t);
    MTL_SHA512_STEP_S(b, c, d, e, f, g, h, a, w7_t, MTL_SHA512_K[71]);
    w8_t = MTL_SHA512_EXPAND_S(w6_t, w1_t, w9_t, w8_t);
    MTL_SHA512_STEP_S(a, b, c, d, e, f, g, h, w8_t, MTL_SHA512_K[72]);
    w9_t = MTL_SHA512_EXPAND_S(w7_t, w2_t, wa_t, w9_t);
    MTL_SHA512_STEP_S(h, a, b, c, d, e, f, g, w9_t, MTL_SHA512_K[73]);
    wa_t = MTL_SHA512_EXPAND_S(w8_t, w3_t, wb_t, wa_t);
    MTL_SHA512_STEP_S(g, h, a, b, c, d, e, f, wa_t, MTL_SHA512_K[74]);
    wb_t = MTL_SHA512_EXPAND_S(w9_t, w4_t, wc_t, wb_t);
    MTL_SHA512_STEP_S(f, g, h, a, b, c, d, e, wb_t, MTL_SHA512_K[75]);
    wc_t = MTL_SHA512_EXPAND_S(wa_t, w5_t, wd_t, wc_t);
    MTL_SHA512_STEP_S(e, f, g, h, a, b, c, d, wc_t, MTL_SHA512_K[76]);
    wd_t = MTL_SHA512_EXPAND_S(wb_t, w6_t, we_t, wd_t);
    MTL_SHA512_STEP_S(d, e, f, g, h, a, b, c, wd_t, MTL_SHA512_K[77]);
    we_t = MTL_SHA512_EXPAND_S(wc_t, w7_t, wf_t, we_t);
    MTL_SHA512_STEP_S(c, d, e, f, g, h, a, b, we_t, MTL_SHA512_K[78]);
    wf_t = MTL_SHA512_EXPAND_S(wd_t, w8_t, w0_t, wf_t);
    MTL_SHA512_STEP_S(b, c, d, e, f, g, h, a, wf_t, MTL_SHA512_K[79]);
    state[0] += a; state[1] += b; state[2] += c; state[3] += d;
    state[4] += e; state[5] += f; state[6] += g; state[7] += h;
}

/* --- RIPEMD-160 / RIPEMD-320 round helper macros + compression blocks.
 *
 * Phase 2d.6 RIPEMD family Metal port. Mirrors gpu_common.cl lines
 * 994-1267 byte-for-byte. Both compress functions take pointer-state
 * (`thread uint *hash, thread const uint *X`) matching sha1_block /
 * sha256_block convention; cl2metal.py registers them in
 * _BLOCK_HELPERS_POINTER_STATE so call sites `&st.h[0]` translate
 * verbatim.
 *
 * Naming discipline (Pattern 2): bare RMD_F1..F5 + RMD_STEP names are
 * preserved (the OpenCL twin uses them; no naming collisions inside
 * metal_common because no other family defines these tokens). The L1..L5
 * and R1..R5 macros are also preserved — distinct names from any SHA /
 * MD round helper. The dual-pipeline body in rmd160_block uses them to
 * keep the source visually parallel to the OpenCL twin, which simplifies
 * cross-platform diff review.
 *
 * Metal substitution: OpenCL `rotate((A), (uint)(S))` becomes
 * `rotl32((A), (uint)(S))`. The rotation magnitude S is a small
 * compile-time integer in every call site — Metal's `metal::rotate`
 * builtin also works but rotl32 keeps the in-file calling convention
 * consistent with sha1_block / sha256_block / md5_block. */
#define RMD_F1(x, y, z) ((x) ^ (y) ^ (z))
#define RMD_F2(x, y, z) ((((y) ^ (z)) & (x)) ^ (z))
#define RMD_F3(x, y, z) (((x) | ~(y)) ^ (z))
#define RMD_F4(x, y, z) ((((x) ^ (y)) & (z)) ^ (y))
#define RMD_F5(x, y, z) ((x) ^ ((y) | ~(z)))

#define RMD_STEP(FUNC, A, B, C, D, E, X, S, K) \
    (A) += FUNC((B), (C), (D)) + (X) + K; \
    (A) = rotl32((A), (uint)(S)) + (E); \
    (C) = rotl32((C), (uint)10);

#define L1(A,B,C,D,E,X,S) RMD_STEP(RMD_F1,A,B,C,D,E,X,S,0u)
#define L2(A,B,C,D,E,X,S) RMD_STEP(RMD_F2,A,B,C,D,E,X,S,0x5a827999u)
#define L3(A,B,C,D,E,X,S) RMD_STEP(RMD_F3,A,B,C,D,E,X,S,0x6ed9eba1u)
#define L4(A,B,C,D,E,X,S) RMD_STEP(RMD_F4,A,B,C,D,E,X,S,0x8f1bbcdcu)
#define L5(A,B,C,D,E,X,S) RMD_STEP(RMD_F5,A,B,C,D,E,X,S,0xa953fd4eu)
#define R1(A,B,C,D,E,X,S) RMD_STEP(RMD_F5,A,B,C,D,E,X,S,0x50a28be6u)
#define R2(A,B,C,D,E,X,S) RMD_STEP(RMD_F4,A,B,C,D,E,X,S,0x5c4dd124u)
#define R3(A,B,C,D,E,X,S) RMD_STEP(RMD_F3,A,B,C,D,E,X,S,0x6d703ef3u)
#define R4(A,B,C,D,E,X,S) RMD_STEP(RMD_F2,A,B,C,D,E,X,S,0x7a6d76e9u)
#define R5(A,B,C,D,E,X,S) RMD_STEP(RMD_F1,A,B,C,D,E,X,S,0u)

/* --- rmd160_block: single 64-byte RIPEMD-160 compress block.
 *
 * Mirrors gpu_common.cl::rmd160_block byte-for-byte. Dual-pipeline
 * (left line L1..L5, right line R1..R5) processing of 80 steps each,
 * combined into a single 5-word state at the end. Pointer-state form
 * matches sha1_block / sha256_block (cl2metal.py preserves &st.h[0]
 * call-site form). Pattern 1: both pointer args thread-qualified.
 * Pattern 3: static inline.
 *
 * R2 register pressure (architect §5 callout): the dual-pipeline body
 * carries A,B,C,D,E + a1,b1,c1,d1,e1 = 10 live uint32 + 16-word M[]
 * input. Comparable to sha1_block's working set in size; no W[80]
 * schedule (rmd160 reads M[] directly via the round macros). */
static inline void rmd160_block(thread uint *hash, thread const uint *X) {
    uint A = hash[0], B = hash[1], C = hash[2], D = hash[3], E = hash[4];
    uint a1, b1, c1, d1, e1;
    L1(A,B,C,D,E,X[0],11);L1(E,A,B,C,D,X[1],14);L1(D,E,A,B,C,X[2],15);L1(C,D,E,A,B,X[3],12);
    L1(B,C,D,E,A,X[4],5);L1(A,B,C,D,E,X[5],8);L1(E,A,B,C,D,X[6],7);L1(D,E,A,B,C,X[7],9);
    L1(C,D,E,A,B,X[8],11);L1(B,C,D,E,A,X[9],13);L1(A,B,C,D,E,X[10],14);L1(E,A,B,C,D,X[11],15);
    L1(D,E,A,B,C,X[12],6);L1(C,D,E,A,B,X[13],7);L1(B,C,D,E,A,X[14],9);L1(A,B,C,D,E,X[15],8);
    L2(E,A,B,C,D,X[7],7);L2(D,E,A,B,C,X[4],6);L2(C,D,E,A,B,X[13],8);L2(B,C,D,E,A,X[1],13);
    L2(A,B,C,D,E,X[10],11);L2(E,A,B,C,D,X[6],9);L2(D,E,A,B,C,X[15],7);L2(C,D,E,A,B,X[3],15);
    L2(B,C,D,E,A,X[12],7);L2(A,B,C,D,E,X[0],12);L2(E,A,B,C,D,X[9],15);L2(D,E,A,B,C,X[5],9);
    L2(C,D,E,A,B,X[2],11);L2(B,C,D,E,A,X[14],7);L2(A,B,C,D,E,X[11],13);L2(E,A,B,C,D,X[8],12);
    L3(D,E,A,B,C,X[3],11);L3(C,D,E,A,B,X[10],13);L3(B,C,D,E,A,X[14],6);L3(A,B,C,D,E,X[4],7);
    L3(E,A,B,C,D,X[9],14);L3(D,E,A,B,C,X[15],9);L3(C,D,E,A,B,X[8],13);L3(B,C,D,E,A,X[1],15);
    L3(A,B,C,D,E,X[2],14);L3(E,A,B,C,D,X[7],8);L3(D,E,A,B,C,X[0],13);L3(C,D,E,A,B,X[6],6);
    L3(B,C,D,E,A,X[13],5);L3(A,B,C,D,E,X[11],12);L3(E,A,B,C,D,X[5],7);L3(D,E,A,B,C,X[12],5);
    L4(C,D,E,A,B,X[1],11);L4(B,C,D,E,A,X[9],12);L4(A,B,C,D,E,X[11],14);L4(E,A,B,C,D,X[10],15);
    L4(D,E,A,B,C,X[0],14);L4(C,D,E,A,B,X[8],15);L4(B,C,D,E,A,X[12],9);L4(A,B,C,D,E,X[4],8);
    L4(E,A,B,C,D,X[13],9);L4(D,E,A,B,C,X[3],14);L4(C,D,E,A,B,X[7],5);L4(B,C,D,E,A,X[15],6);
    L4(A,B,C,D,E,X[14],8);L4(E,A,B,C,D,X[5],6);L4(D,E,A,B,C,X[6],5);L4(C,D,E,A,B,X[2],12);
    L5(B,C,D,E,A,X[4],9);L5(A,B,C,D,E,X[0],15);L5(E,A,B,C,D,X[5],5);L5(D,E,A,B,C,X[9],11);
    L5(C,D,E,A,B,X[7],6);L5(B,C,D,E,A,X[12],8);L5(A,B,C,D,E,X[2],13);L5(E,A,B,C,D,X[10],12);
    L5(D,E,A,B,C,X[14],5);L5(C,D,E,A,B,X[1],12);L5(B,C,D,E,A,X[3],13);L5(A,B,C,D,E,X[8],14);
    L5(E,A,B,C,D,X[11],11);L5(D,E,A,B,C,X[6],8);L5(C,D,E,A,B,X[15],5);L5(B,C,D,E,A,X[13],6);
    a1 = A; b1 = B; c1 = C; d1 = D; e1 = E;
    A = hash[0]; B = hash[1]; C = hash[2]; D = hash[3]; E = hash[4];
    R1(A,B,C,D,E,X[5],8);R1(E,A,B,C,D,X[14],9);R1(D,E,A,B,C,X[7],9);R1(C,D,E,A,B,X[0],11);
    R1(B,C,D,E,A,X[9],13);R1(A,B,C,D,E,X[2],15);R1(E,A,B,C,D,X[11],15);R1(D,E,A,B,C,X[4],5);
    R1(C,D,E,A,B,X[13],7);R1(B,C,D,E,A,X[6],7);R1(A,B,C,D,E,X[15],8);R1(E,A,B,C,D,X[8],11);
    R1(D,E,A,B,C,X[1],14);R1(C,D,E,A,B,X[10],14);R1(B,C,D,E,A,X[3],12);R1(A,B,C,D,E,X[12],6);
    R2(E,A,B,C,D,X[6],9);R2(D,E,A,B,C,X[11],13);R2(C,D,E,A,B,X[3],15);R2(B,C,D,E,A,X[7],7);
    R2(A,B,C,D,E,X[0],12);R2(E,A,B,C,D,X[13],8);R2(D,E,A,B,C,X[5],9);R2(C,D,E,A,B,X[10],11);
    R2(B,C,D,E,A,X[14],7);R2(A,B,C,D,E,X[15],7);R2(E,A,B,C,D,X[8],12);R2(D,E,A,B,C,X[12],7);
    R2(C,D,E,A,B,X[4],6);R2(B,C,D,E,A,X[9],15);R2(A,B,C,D,E,X[1],13);R2(E,A,B,C,D,X[2],11);
    R3(D,E,A,B,C,X[15],9);R3(C,D,E,A,B,X[5],7);R3(B,C,D,E,A,X[1],15);R3(A,B,C,D,E,X[3],11);
    R3(E,A,B,C,D,X[7],8);R3(D,E,A,B,C,X[14],6);R3(C,D,E,A,B,X[6],6);R3(B,C,D,E,A,X[9],14);
    R3(A,B,C,D,E,X[11],12);R3(E,A,B,C,D,X[8],13);R3(D,E,A,B,C,X[12],5);R3(C,D,E,A,B,X[2],14);
    R3(B,C,D,E,A,X[10],13);R3(A,B,C,D,E,X[0],13);R3(E,A,B,C,D,X[4],7);R3(D,E,A,B,C,X[13],5);
    R4(C,D,E,A,B,X[8],15);R4(B,C,D,E,A,X[6],5);R4(A,B,C,D,E,X[4],8);R4(E,A,B,C,D,X[1],11);
    R4(D,E,A,B,C,X[3],14);R4(C,D,E,A,B,X[11],14);R4(B,C,D,E,A,X[15],6);R4(A,B,C,D,E,X[0],14);
    R4(E,A,B,C,D,X[5],6);R4(D,E,A,B,C,X[12],9);R4(C,D,E,A,B,X[2],12);R4(B,C,D,E,A,X[13],9);
    R4(A,B,C,D,E,X[9],12);R4(E,A,B,C,D,X[7],5);R4(D,E,A,B,C,X[10],15);R4(C,D,E,A,B,X[14],8);
    R5(B,C,D,E,A,X[12],8);R5(A,B,C,D,E,X[15],5);R5(E,A,B,C,D,X[10],12);R5(D,E,A,B,C,X[4],9);
    R5(C,D,E,A,B,X[1],12);R5(B,C,D,E,A,X[5],5);R5(A,B,C,D,E,X[8],14);R5(E,A,B,C,D,X[7],6);
    R5(D,E,A,B,C,X[6],8);R5(C,D,E,A,B,X[2],13);R5(B,C,D,E,A,X[13],6);R5(A,B,C,D,E,X[14],5);
    R5(E,A,B,C,D,X[0],15);R5(D,E,A,B,C,X[3],13);R5(C,D,E,A,B,X[9],11);R5(B,C,D,E,A,X[11],11);
    D += c1 + hash[1]; hash[1] = hash[2] + d1 + E; hash[2] = hash[3] + e1 + A;
    hash[3] = hash[4] + a1 + B; hash[4] = hash[0] + b1 + C; hash[0] = D;
}

/* --- rmd128_block: single 64-byte RIPEMD-128 compress block.
 *
 * Mirrors gpu_common.cl::rmd128_block byte-for-byte. 4-uint state.
 * Dual pipeline: left line F1->F2->F3->F4; right line F4->F3->F2->F1
 * per Bosselaers Table 4 (RMD-128 right line is F4 F3 F2 F1, NOT
 * F5 F4 F3 F2 F1 as in RMD-160) -- R2 callout. Reuses RMD_F1..RMD_F4
 * round-functions defined above for rmd160_block; RMD_F5 is unused
 * by RMD-128. Defines local RMD128_STEP_METAL (4-arg variant without
 * the +E and without C-rotation that RMD_STEP carries) so this body
 * matches rmd128.h's FF/GG/HH/II macro shape directly.
 *
 * R6 (Metal scalar bitselect failure mode): not applicable here --
 * RMD_F1..F4 are XOR/AND/OR/NOT compositions; no bitselect used.
 *
 * Pointer-state convention (thread uint *hash, thread const uint *X)
 * matches rmd160_block / sha1_block / sha256_block. */
#define RMD128_STEP_METAL(FUNC, A, B, C, D, X, S, K) \
    (A) += FUNC((B), (C), (D)) + (X) + K; \
    (A) = rotl32((A), (uint)(S));

#define LL1M(A,B,C,D,X,S) RMD128_STEP_METAL(RMD_F1,A,B,C,D,X,S,0u)
#define LL2M(A,B,C,D,X,S) RMD128_STEP_METAL(RMD_F2,A,B,C,D,X,S,0x5a827999u)
#define LL3M(A,B,C,D,X,S) RMD128_STEP_METAL(RMD_F3,A,B,C,D,X,S,0x6ed9eba1u)
#define LL4M(A,B,C,D,X,S) RMD128_STEP_METAL(RMD_F4,A,B,C,D,X,S,0x8f1bbcdcu)
/* Right line: F4 F3 F2 F1 ordering -- Bosselaers Table 4. */
#define RR1M(A,B,C,D,X,S) RMD128_STEP_METAL(RMD_F4,A,B,C,D,X,S,0x50a28be6u)
#define RR2M(A,B,C,D,X,S) RMD128_STEP_METAL(RMD_F3,A,B,C,D,X,S,0x5c4dd124u)
#define RR3M(A,B,C,D,X,S) RMD128_STEP_METAL(RMD_F2,A,B,C,D,X,S,0x6d703ef3u)
#define RR4M(A,B,C,D,X,S) RMD128_STEP_METAL(RMD_F1,A,B,C,D,X,S,0u)

static inline void rmd128_block(thread uint *hash, thread const uint *X) {
    uint A = hash[0], B = hash[1], C = hash[2], D = hash[3];
    uint a1, b1, c1, d1;
    /* left round 1 (F1) */
    LL1M(A,B,C,D,X[ 0],11); LL1M(D,A,B,C,X[ 1],14); LL1M(C,D,A,B,X[ 2],15); LL1M(B,C,D,A,X[ 3],12);
    LL1M(A,B,C,D,X[ 4], 5); LL1M(D,A,B,C,X[ 5], 8); LL1M(C,D,A,B,X[ 6], 7); LL1M(B,C,D,A,X[ 7], 9);
    LL1M(A,B,C,D,X[ 8],11); LL1M(D,A,B,C,X[ 9],13); LL1M(C,D,A,B,X[10],14); LL1M(B,C,D,A,X[11],15);
    LL1M(A,B,C,D,X[12], 6); LL1M(D,A,B,C,X[13], 7); LL1M(C,D,A,B,X[14], 9); LL1M(B,C,D,A,X[15], 8);
    /* left round 2 (F2) */
    LL2M(A,B,C,D,X[ 7], 7); LL2M(D,A,B,C,X[ 4], 6); LL2M(C,D,A,B,X[13], 8); LL2M(B,C,D,A,X[ 1],13);
    LL2M(A,B,C,D,X[10],11); LL2M(D,A,B,C,X[ 6], 9); LL2M(C,D,A,B,X[15], 7); LL2M(B,C,D,A,X[ 3],15);
    LL2M(A,B,C,D,X[12], 7); LL2M(D,A,B,C,X[ 0],12); LL2M(C,D,A,B,X[ 9],15); LL2M(B,C,D,A,X[ 5], 9);
    LL2M(A,B,C,D,X[ 2],11); LL2M(D,A,B,C,X[14], 7); LL2M(C,D,A,B,X[11],13); LL2M(B,C,D,A,X[ 8],12);
    /* left round 3 (F3) */
    LL3M(A,B,C,D,X[ 3],11); LL3M(D,A,B,C,X[10],13); LL3M(C,D,A,B,X[14], 6); LL3M(B,C,D,A,X[ 4], 7);
    LL3M(A,B,C,D,X[ 9],14); LL3M(D,A,B,C,X[15], 9); LL3M(C,D,A,B,X[ 8],13); LL3M(B,C,D,A,X[ 1],15);
    LL3M(A,B,C,D,X[ 2],14); LL3M(D,A,B,C,X[ 7], 8); LL3M(C,D,A,B,X[ 0],13); LL3M(B,C,D,A,X[ 6], 6);
    LL3M(A,B,C,D,X[13], 5); LL3M(D,A,B,C,X[11],12); LL3M(C,D,A,B,X[ 5], 7); LL3M(B,C,D,A,X[12], 5);
    /* left round 4 (F4) */
    LL4M(A,B,C,D,X[ 1],11); LL4M(D,A,B,C,X[ 9],12); LL4M(C,D,A,B,X[11],14); LL4M(B,C,D,A,X[10],15);
    LL4M(A,B,C,D,X[ 0],14); LL4M(D,A,B,C,X[ 8],15); LL4M(C,D,A,B,X[12], 9); LL4M(B,C,D,A,X[ 4], 8);
    LL4M(A,B,C,D,X[13], 9); LL4M(D,A,B,C,X[ 3],14); LL4M(C,D,A,B,X[ 7], 5); LL4M(B,C,D,A,X[15], 6);
    LL4M(A,B,C,D,X[14], 8); LL4M(D,A,B,C,X[ 5], 6); LL4M(C,D,A,B,X[ 6], 5); LL4M(B,C,D,A,X[ 2],12);
    /* save left line, restart with IV for right line */
    a1 = A; b1 = B; c1 = C; d1 = D;
    A = hash[0]; B = hash[1]; C = hash[2]; D = hash[3];
    /* right round 1 (F4) -- Bosselaers Table 4 RMD-128 right line F4 F3 F2 F1 */
    RR1M(A,B,C,D,X[ 5], 8); RR1M(D,A,B,C,X[14], 9); RR1M(C,D,A,B,X[ 7], 9); RR1M(B,C,D,A,X[ 0],11);
    RR1M(A,B,C,D,X[ 9],13); RR1M(D,A,B,C,X[ 2],15); RR1M(C,D,A,B,X[11],15); RR1M(B,C,D,A,X[ 4], 5);
    RR1M(A,B,C,D,X[13], 7); RR1M(D,A,B,C,X[ 6], 7); RR1M(C,D,A,B,X[15], 8); RR1M(B,C,D,A,X[ 8],11);
    RR1M(A,B,C,D,X[ 1],14); RR1M(D,A,B,C,X[10],14); RR1M(C,D,A,B,X[ 3],12); RR1M(B,C,D,A,X[12], 6);
    /* right round 2 (F3) */
    RR2M(A,B,C,D,X[ 6], 9); RR2M(D,A,B,C,X[11],13); RR2M(C,D,A,B,X[ 3],15); RR2M(B,C,D,A,X[ 7], 7);
    RR2M(A,B,C,D,X[ 0],12); RR2M(D,A,B,C,X[13], 8); RR2M(C,D,A,B,X[ 5], 9); RR2M(B,C,D,A,X[10],11);
    RR2M(A,B,C,D,X[14], 7); RR2M(D,A,B,C,X[15], 7); RR2M(C,D,A,B,X[ 8],12); RR2M(B,C,D,A,X[12], 7);
    RR2M(A,B,C,D,X[ 4], 6); RR2M(D,A,B,C,X[ 9],15); RR2M(C,D,A,B,X[ 1],13); RR2M(B,C,D,A,X[ 2],11);
    /* right round 3 (F2) */
    RR3M(A,B,C,D,X[15], 9); RR3M(D,A,B,C,X[ 5], 7); RR3M(C,D,A,B,X[ 1],15); RR3M(B,C,D,A,X[ 3],11);
    RR3M(A,B,C,D,X[ 7], 8); RR3M(D,A,B,C,X[14], 6); RR3M(C,D,A,B,X[ 6], 6); RR3M(B,C,D,A,X[ 9],14);
    RR3M(A,B,C,D,X[11],12); RR3M(D,A,B,C,X[ 8],13); RR3M(C,D,A,B,X[12], 5); RR3M(B,C,D,A,X[ 2],14);
    RR3M(A,B,C,D,X[10],13); RR3M(D,A,B,C,X[ 0],13); RR3M(C,D,A,B,X[ 4], 7); RR3M(B,C,D,A,X[13], 5);
    /* right round 4 (F1) */
    RR4M(A,B,C,D,X[ 8],15); RR4M(D,A,B,C,X[ 6], 5); RR4M(C,D,A,B,X[ 4], 8); RR4M(B,C,D,A,X[ 1],11);
    RR4M(A,B,C,D,X[ 3],14); RR4M(D,A,B,C,X[11],14); RR4M(C,D,A,B,X[15], 6); RR4M(B,C,D,A,X[ 0],14);
    RR4M(A,B,C,D,X[ 5], 6); RR4M(D,A,B,C,X[12], 9); RR4M(C,D,A,B,X[ 2],12); RR4M(B,C,D,A,X[13], 9);
    RR4M(A,B,C,D,X[ 9],12); RR4M(D,A,B,C,X[ 7], 5); RR4M(C,D,A,B,X[10],15); RR4M(B,C,D,A,X[14], 8);
    /* combine: mirrors rmd128.c lines 188-193. Write hash[1..3] before
     * overwriting hash[0]; temporaries cover the ordering dependency. */
    D += c1 + hash[1];
    hash[1] = hash[2] + d1 + A;
    hash[2] = hash[3] + a1 + B;
    hash[3] = hash[0] + b1 + C;
    hash[0] = D;
}

/* --- rmd320_block: single 64-byte RIPEMD-320 compress block.
 *
 * Mirrors gpu_common.cl::rmd320_block byte-for-byte. Dual-pipeline
 * compression with cross-swap between rounds (one A/B/C/D/E register
 * swaps with the corresponding AA/BB/CC/DD/EE register after each
 * 16-step round). Unlike rmd160_block the two pipelines do NOT merge —
 * the result is added back into hash[0..9] with the cross-mixed
 * accumulation noted in the OpenCL twin source. */
static inline void rmd320_block(thread uint *hash, thread const uint *X) {
    uint A = hash[0], B = hash[1], C = hash[2], D = hash[3], E = hash[4];
    uint AA = hash[5], BB = hash[6], CC = hash[7], DD = hash[8], EE = hash[9];

    /* j=0..15 */
    RMD_STEP(RMD_F1, A, B, C, D, E, X[0], 11, 0x00000000u);
    RMD_STEP(RMD_F1, E, A, B, C, D, X[1], 14, 0x00000000u);
    RMD_STEP(RMD_F1, D, E, A, B, C, X[2], 15, 0x00000000u);
    RMD_STEP(RMD_F1, C, D, E, A, B, X[3], 12, 0x00000000u);
    RMD_STEP(RMD_F1, B, C, D, E, A, X[4], 5, 0x00000000u);
    RMD_STEP(RMD_F1, A, B, C, D, E, X[5], 8, 0x00000000u);
    RMD_STEP(RMD_F1, E, A, B, C, D, X[6], 7, 0x00000000u);
    RMD_STEP(RMD_F1, D, E, A, B, C, X[7], 9, 0x00000000u);
    RMD_STEP(RMD_F1, C, D, E, A, B, X[8], 11, 0x00000000u);
    RMD_STEP(RMD_F1, B, C, D, E, A, X[9], 13, 0x00000000u);
    RMD_STEP(RMD_F1, A, B, C, D, E, X[10], 14, 0x00000000u);
    RMD_STEP(RMD_F1, E, A, B, C, D, X[11], 15, 0x00000000u);
    RMD_STEP(RMD_F1, D, E, A, B, C, X[12], 6, 0x00000000u);
    RMD_STEP(RMD_F1, C, D, E, A, B, X[13], 7, 0x00000000u);
    RMD_STEP(RMD_F1, B, C, D, E, A, X[14], 9, 0x00000000u);
    RMD_STEP(RMD_F1, A, B, C, D, E, X[15], 8, 0x00000000u);
    RMD_STEP(RMD_F5, AA, BB, CC, DD, EE, X[5], 8, 0x50A28BE6u);
    RMD_STEP(RMD_F5, EE, AA, BB, CC, DD, X[14], 9, 0x50A28BE6u);
    RMD_STEP(RMD_F5, DD, EE, AA, BB, CC, X[7], 9, 0x50A28BE6u);
    RMD_STEP(RMD_F5, CC, DD, EE, AA, BB, X[0], 11, 0x50A28BE6u);
    RMD_STEP(RMD_F5, BB, CC, DD, EE, AA, X[9], 13, 0x50A28BE6u);
    RMD_STEP(RMD_F5, AA, BB, CC, DD, EE, X[2], 15, 0x50A28BE6u);
    RMD_STEP(RMD_F5, EE, AA, BB, CC, DD, X[11], 15, 0x50A28BE6u);
    RMD_STEP(RMD_F5, DD, EE, AA, BB, CC, X[4], 5, 0x50A28BE6u);
    RMD_STEP(RMD_F5, CC, DD, EE, AA, BB, X[13], 7, 0x50A28BE6u);
    RMD_STEP(RMD_F5, BB, CC, DD, EE, AA, X[6], 7, 0x50A28BE6u);
    RMD_STEP(RMD_F5, AA, BB, CC, DD, EE, X[15], 8, 0x50A28BE6u);
    RMD_STEP(RMD_F5, EE, AA, BB, CC, DD, X[8], 11, 0x50A28BE6u);
    RMD_STEP(RMD_F5, DD, EE, AA, BB, CC, X[1], 14, 0x50A28BE6u);
    RMD_STEP(RMD_F5, CC, DD, EE, AA, BB, X[10], 14, 0x50A28BE6u);
    RMD_STEP(RMD_F5, BB, CC, DD, EE, AA, X[3], 12, 0x50A28BE6u);
    RMD_STEP(RMD_F5, AA, BB, CC, DD, EE, X[12], 6, 0x50A28BE6u);
    { uint T = A; A = AA; AA = T; }
    /* j=16..31 */
    RMD_STEP(RMD_F2, E, A, B, C, D, X[7], 7, 0x5A827999u);
    RMD_STEP(RMD_F2, D, E, A, B, C, X[4], 6, 0x5A827999u);
    RMD_STEP(RMD_F2, C, D, E, A, B, X[13], 8, 0x5A827999u);
    RMD_STEP(RMD_F2, B, C, D, E, A, X[1], 13, 0x5A827999u);
    RMD_STEP(RMD_F2, A, B, C, D, E, X[10], 11, 0x5A827999u);
    RMD_STEP(RMD_F2, E, A, B, C, D, X[6], 9, 0x5A827999u);
    RMD_STEP(RMD_F2, D, E, A, B, C, X[15], 7, 0x5A827999u);
    RMD_STEP(RMD_F2, C, D, E, A, B, X[3], 15, 0x5A827999u);
    RMD_STEP(RMD_F2, B, C, D, E, A, X[12], 7, 0x5A827999u);
    RMD_STEP(RMD_F2, A, B, C, D, E, X[0], 12, 0x5A827999u);
    RMD_STEP(RMD_F2, E, A, B, C, D, X[9], 15, 0x5A827999u);
    RMD_STEP(RMD_F2, D, E, A, B, C, X[5], 9, 0x5A827999u);
    RMD_STEP(RMD_F2, C, D, E, A, B, X[2], 11, 0x5A827999u);
    RMD_STEP(RMD_F2, B, C, D, E, A, X[14], 7, 0x5A827999u);
    RMD_STEP(RMD_F2, A, B, C, D, E, X[11], 13, 0x5A827999u);
    RMD_STEP(RMD_F2, E, A, B, C, D, X[8], 12, 0x5A827999u);
    RMD_STEP(RMD_F4, EE, AA, BB, CC, DD, X[6], 9, 0x5C4DD124u);
    RMD_STEP(RMD_F4, DD, EE, AA, BB, CC, X[11], 13, 0x5C4DD124u);
    RMD_STEP(RMD_F4, CC, DD, EE, AA, BB, X[3], 15, 0x5C4DD124u);
    RMD_STEP(RMD_F4, BB, CC, DD, EE, AA, X[7], 7, 0x5C4DD124u);
    RMD_STEP(RMD_F4, AA, BB, CC, DD, EE, X[0], 12, 0x5C4DD124u);
    RMD_STEP(RMD_F4, EE, AA, BB, CC, DD, X[13], 8, 0x5C4DD124u);
    RMD_STEP(RMD_F4, DD, EE, AA, BB, CC, X[5], 9, 0x5C4DD124u);
    RMD_STEP(RMD_F4, CC, DD, EE, AA, BB, X[10], 11, 0x5C4DD124u);
    RMD_STEP(RMD_F4, BB, CC, DD, EE, AA, X[14], 7, 0x5C4DD124u);
    RMD_STEP(RMD_F4, AA, BB, CC, DD, EE, X[15], 7, 0x5C4DD124u);
    RMD_STEP(RMD_F4, EE, AA, BB, CC, DD, X[8], 12, 0x5C4DD124u);
    RMD_STEP(RMD_F4, DD, EE, AA, BB, CC, X[12], 7, 0x5C4DD124u);
    RMD_STEP(RMD_F4, CC, DD, EE, AA, BB, X[4], 6, 0x5C4DD124u);
    RMD_STEP(RMD_F4, BB, CC, DD, EE, AA, X[9], 15, 0x5C4DD124u);
    RMD_STEP(RMD_F4, AA, BB, CC, DD, EE, X[1], 13, 0x5C4DD124u);
    RMD_STEP(RMD_F4, EE, AA, BB, CC, DD, X[2], 11, 0x5C4DD124u);
    { uint T = B; B = BB; BB = T; }
    /* j=32..47 */
    RMD_STEP(RMD_F3, D, E, A, B, C, X[3], 11, 0x6ED9EBA1u);
    RMD_STEP(RMD_F3, C, D, E, A, B, X[10], 13, 0x6ED9EBA1u);
    RMD_STEP(RMD_F3, B, C, D, E, A, X[14], 6, 0x6ED9EBA1u);
    RMD_STEP(RMD_F3, A, B, C, D, E, X[4], 7, 0x6ED9EBA1u);
    RMD_STEP(RMD_F3, E, A, B, C, D, X[9], 14, 0x6ED9EBA1u);
    RMD_STEP(RMD_F3, D, E, A, B, C, X[15], 9, 0x6ED9EBA1u);
    RMD_STEP(RMD_F3, C, D, E, A, B, X[8], 13, 0x6ED9EBA1u);
    RMD_STEP(RMD_F3, B, C, D, E, A, X[1], 15, 0x6ED9EBA1u);
    RMD_STEP(RMD_F3, A, B, C, D, E, X[2], 14, 0x6ED9EBA1u);
    RMD_STEP(RMD_F3, E, A, B, C, D, X[7], 8, 0x6ED9EBA1u);
    RMD_STEP(RMD_F3, D, E, A, B, C, X[0], 13, 0x6ED9EBA1u);
    RMD_STEP(RMD_F3, C, D, E, A, B, X[6], 6, 0x6ED9EBA1u);
    RMD_STEP(RMD_F3, B, C, D, E, A, X[13], 5, 0x6ED9EBA1u);
    RMD_STEP(RMD_F3, A, B, C, D, E, X[11], 12, 0x6ED9EBA1u);
    RMD_STEP(RMD_F3, E, A, B, C, D, X[5], 7, 0x6ED9EBA1u);
    RMD_STEP(RMD_F3, D, E, A, B, C, X[12], 5, 0x6ED9EBA1u);
    RMD_STEP(RMD_F3, DD, EE, AA, BB, CC, X[15], 9, 0x6D703EF3u);
    RMD_STEP(RMD_F3, CC, DD, EE, AA, BB, X[5], 7, 0x6D703EF3u);
    RMD_STEP(RMD_F3, BB, CC, DD, EE, AA, X[1], 15, 0x6D703EF3u);
    RMD_STEP(RMD_F3, AA, BB, CC, DD, EE, X[3], 11, 0x6D703EF3u);
    RMD_STEP(RMD_F3, EE, AA, BB, CC, DD, X[7], 8, 0x6D703EF3u);
    RMD_STEP(RMD_F3, DD, EE, AA, BB, CC, X[14], 6, 0x6D703EF3u);
    RMD_STEP(RMD_F3, CC, DD, EE, AA, BB, X[6], 6, 0x6D703EF3u);
    RMD_STEP(RMD_F3, BB, CC, DD, EE, AA, X[9], 14, 0x6D703EF3u);
    RMD_STEP(RMD_F3, AA, BB, CC, DD, EE, X[11], 12, 0x6D703EF3u);
    RMD_STEP(RMD_F3, EE, AA, BB, CC, DD, X[8], 13, 0x6D703EF3u);
    RMD_STEP(RMD_F3, DD, EE, AA, BB, CC, X[12], 5, 0x6D703EF3u);
    RMD_STEP(RMD_F3, CC, DD, EE, AA, BB, X[2], 14, 0x6D703EF3u);
    RMD_STEP(RMD_F3, BB, CC, DD, EE, AA, X[10], 13, 0x6D703EF3u);
    RMD_STEP(RMD_F3, AA, BB, CC, DD, EE, X[0], 13, 0x6D703EF3u);
    RMD_STEP(RMD_F3, EE, AA, BB, CC, DD, X[4], 7, 0x6D703EF3u);
    RMD_STEP(RMD_F3, DD, EE, AA, BB, CC, X[13], 5, 0x6D703EF3u);
    { uint T = C; C = CC; CC = T; }
    /* j=48..63 */
    RMD_STEP(RMD_F4, C, D, E, A, B, X[1], 11, 0x8F1BBCDCu);
    RMD_STEP(RMD_F4, B, C, D, E, A, X[9], 12, 0x8F1BBCDCu);
    RMD_STEP(RMD_F4, A, B, C, D, E, X[11], 14, 0x8F1BBCDCu);
    RMD_STEP(RMD_F4, E, A, B, C, D, X[10], 15, 0x8F1BBCDCu);
    RMD_STEP(RMD_F4, D, E, A, B, C, X[0], 14, 0x8F1BBCDCu);
    RMD_STEP(RMD_F4, C, D, E, A, B, X[8], 15, 0x8F1BBCDCu);
    RMD_STEP(RMD_F4, B, C, D, E, A, X[12], 9, 0x8F1BBCDCu);
    RMD_STEP(RMD_F4, A, B, C, D, E, X[4], 8, 0x8F1BBCDCu);
    RMD_STEP(RMD_F4, E, A, B, C, D, X[13], 9, 0x8F1BBCDCu);
    RMD_STEP(RMD_F4, D, E, A, B, C, X[3], 14, 0x8F1BBCDCu);
    RMD_STEP(RMD_F4, C, D, E, A, B, X[7], 5, 0x8F1BBCDCu);
    RMD_STEP(RMD_F4, B, C, D, E, A, X[15], 6, 0x8F1BBCDCu);
    RMD_STEP(RMD_F4, A, B, C, D, E, X[14], 8, 0x8F1BBCDCu);
    RMD_STEP(RMD_F4, E, A, B, C, D, X[5], 6, 0x8F1BBCDCu);
    RMD_STEP(RMD_F4, D, E, A, B, C, X[6], 5, 0x8F1BBCDCu);
    RMD_STEP(RMD_F4, C, D, E, A, B, X[2], 12, 0x8F1BBCDCu);
    RMD_STEP(RMD_F2, CC, DD, EE, AA, BB, X[8], 15, 0x7A6D76E9u);
    RMD_STEP(RMD_F2, BB, CC, DD, EE, AA, X[6], 5, 0x7A6D76E9u);
    RMD_STEP(RMD_F2, AA, BB, CC, DD, EE, X[4], 8, 0x7A6D76E9u);
    RMD_STEP(RMD_F2, EE, AA, BB, CC, DD, X[1], 11, 0x7A6D76E9u);
    RMD_STEP(RMD_F2, DD, EE, AA, BB, CC, X[3], 14, 0x7A6D76E9u);
    RMD_STEP(RMD_F2, CC, DD, EE, AA, BB, X[11], 14, 0x7A6D76E9u);
    RMD_STEP(RMD_F2, BB, CC, DD, EE, AA, X[15], 6, 0x7A6D76E9u);
    RMD_STEP(RMD_F2, AA, BB, CC, DD, EE, X[0], 14, 0x7A6D76E9u);
    RMD_STEP(RMD_F2, EE, AA, BB, CC, DD, X[5], 6, 0x7A6D76E9u);
    RMD_STEP(RMD_F2, DD, EE, AA, BB, CC, X[12], 9, 0x7A6D76E9u);
    RMD_STEP(RMD_F2, CC, DD, EE, AA, BB, X[2], 12, 0x7A6D76E9u);
    RMD_STEP(RMD_F2, BB, CC, DD, EE, AA, X[13], 9, 0x7A6D76E9u);
    RMD_STEP(RMD_F2, AA, BB, CC, DD, EE, X[9], 12, 0x7A6D76E9u);
    RMD_STEP(RMD_F2, EE, AA, BB, CC, DD, X[7], 5, 0x7A6D76E9u);
    RMD_STEP(RMD_F2, DD, EE, AA, BB, CC, X[10], 15, 0x7A6D76E9u);
    RMD_STEP(RMD_F2, CC, DD, EE, AA, BB, X[14], 8, 0x7A6D76E9u);
    { uint T = D; D = DD; DD = T; }
    /* j=64..79 */
    RMD_STEP(RMD_F5, B, C, D, E, A, X[4], 9, 0xA953FD4Eu);
    RMD_STEP(RMD_F5, A, B, C, D, E, X[0], 15, 0xA953FD4Eu);
    RMD_STEP(RMD_F5, E, A, B, C, D, X[5], 5, 0xA953FD4Eu);
    RMD_STEP(RMD_F5, D, E, A, B, C, X[9], 11, 0xA953FD4Eu);
    RMD_STEP(RMD_F5, C, D, E, A, B, X[7], 6, 0xA953FD4Eu);
    RMD_STEP(RMD_F5, B, C, D, E, A, X[12], 8, 0xA953FD4Eu);
    RMD_STEP(RMD_F5, A, B, C, D, E, X[2], 13, 0xA953FD4Eu);
    RMD_STEP(RMD_F5, E, A, B, C, D, X[10], 12, 0xA953FD4Eu);
    RMD_STEP(RMD_F5, D, E, A, B, C, X[14], 5, 0xA953FD4Eu);
    RMD_STEP(RMD_F5, C, D, E, A, B, X[1], 12, 0xA953FD4Eu);
    RMD_STEP(RMD_F5, B, C, D, E, A, X[3], 13, 0xA953FD4Eu);
    RMD_STEP(RMD_F5, A, B, C, D, E, X[8], 14, 0xA953FD4Eu);
    RMD_STEP(RMD_F5, E, A, B, C, D, X[11], 11, 0xA953FD4Eu);
    RMD_STEP(RMD_F5, D, E, A, B, C, X[6], 8, 0xA953FD4Eu);
    RMD_STEP(RMD_F5, C, D, E, A, B, X[15], 5, 0xA953FD4Eu);
    RMD_STEP(RMD_F5, B, C, D, E, A, X[13], 6, 0xA953FD4Eu);
    RMD_STEP(RMD_F1, BB, CC, DD, EE, AA, X[12], 8, 0x00000000u);
    RMD_STEP(RMD_F1, AA, BB, CC, DD, EE, X[15], 5, 0x00000000u);
    RMD_STEP(RMD_F1, EE, AA, BB, CC, DD, X[10], 12, 0x00000000u);
    RMD_STEP(RMD_F1, DD, EE, AA, BB, CC, X[4], 9, 0x00000000u);
    RMD_STEP(RMD_F1, CC, DD, EE, AA, BB, X[1], 12, 0x00000000u);
    RMD_STEP(RMD_F1, BB, CC, DD, EE, AA, X[5], 5, 0x00000000u);
    RMD_STEP(RMD_F1, AA, BB, CC, DD, EE, X[8], 14, 0x00000000u);
    RMD_STEP(RMD_F1, EE, AA, BB, CC, DD, X[7], 6, 0x00000000u);
    RMD_STEP(RMD_F1, DD, EE, AA, BB, CC, X[6], 8, 0x00000000u);
    RMD_STEP(RMD_F1, CC, DD, EE, AA, BB, X[2], 13, 0x00000000u);
    RMD_STEP(RMD_F1, BB, CC, DD, EE, AA, X[13], 6, 0x00000000u);
    RMD_STEP(RMD_F1, AA, BB, CC, DD, EE, X[14], 5, 0x00000000u);
    RMD_STEP(RMD_F1, EE, AA, BB, CC, DD, X[0], 15, 0x00000000u);
    RMD_STEP(RMD_F1, DD, EE, AA, BB, CC, X[3], 13, 0x00000000u);
    RMD_STEP(RMD_F1, CC, DD, EE, AA, BB, X[9], 11, 0x00000000u);
    RMD_STEP(RMD_F1, BB, CC, DD, EE, AA, X[11], 11, 0x00000000u);
    { uint T = E; E = EE; EE = T; }

    /* RIPEMD-320 accumulation (Bosselaers reference): A..E into hash[0..4],
     * AA..EE into hash[5..9]. Spec note from OpenCL twin: hash[4] += E and
     * hash[9] += EE (the cross-mixed accumulation in the j=64..79 swap
     * applies to register E vs EE, so after the swap, hash[0..4] receive
     * the left-line A..D plus the post-swap E (which is the original EE)).
     * The OpenCL twin documents this carefully — see gpu_common.cl
     * rmd320_block lines 1254-1266. */
    hash[0] += A;  hash[1] += B;  hash[2] += C;  hash[3] += D;  hash[4] += E;
    hash[5] += AA; hash[6] += BB; hash[7] += CC; hash[8] += DD; hash[9] += EE;
}

/* --- rotl64: 64-bit rotate-left helper (mirrors gpu_common.cl rotate(ulong))
 *
 * Phase 2d.7a BLAKE2 family prep. Sibling of rotl32 (32-bit) and rotr64
 * (64-bit rotate-right used by SHA-512). BLAKE2 spec rotation is rotate-
 * RIGHT semantically, but the OpenCL twin expresses it as left-rotate by
 * (64 - n) using OpenCL's rotate() builtin. Keeping the same convention
 * here so b2b_compress is byte-for-byte mirror of gpu_common.cl::b2b_compress.
 * Pattern 3: static inline. */
static inline ulong rotl64(ulong x, ulong n) {
    return (x << n) | (x >> (64ul - n));
}

/* --- BLAKE2S compress (mirrors gpu_common.cl lines 1269-1318 byte-for-byte).
 *
 * Phase 2d.7a BLAKE2 family Metal port. BLAKE2S-256 carrier; same
 * compression used by future BLAKE2 variants (BLAKE2B uses the separate
 * b2b_compress below). Pointer-state signature (thread uint *h,
 * thread const uchar *block) matches sha1_block / sha256_block / rmd160_block
 * convention; cl2metal.py rev 1.6 registers b2s_compress in
 * _BLOCK_HELPERS_POINTER_STATE so call-site `&st.h[0]` is preserved.
 *
 * Constants are MTL_-prefixed per Pattern 2 (sibling to MTL_SHA256_K,
 * MTL_SHA512_K, MTL_MD5_S). Round body uses rotl32 (Metal substitute for
 * OpenCL rotate of uint).
 *
 * R2 register pressure: 8 uint32 chaining + 16 uint32 v[] + 16 uint32 m[]
 * working state. Comparable to RIPEMD-160's dual-pipeline body. */
constant uint MTL_B2S_IV[8] = {
    0x6A09E667u, 0xBB67AE85u, 0x3C6EF372u, 0xA54FF53Au,
    0x510E527Fu, 0x9B05688Cu, 0x1F83D9ABu, 0x5BE0CD19u
};

constant uchar MTL_B2S_SIGMA[10][16] = {
    { 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 },
    { 14,10,4,8,9,15,13,6,1,12,0,2,11,7,5,3 },
    { 11,8,12,0,5,2,15,13,10,14,3,6,7,1,9,4 },
    { 7,9,3,1,13,12,11,14,2,6,5,10,4,0,15,8 },
    { 9,0,5,7,2,4,10,15,14,1,11,12,6,8,3,13 },
    { 2,12,6,10,0,11,8,3,4,13,7,5,15,14,1,9 },
    { 12,5,1,15,14,13,4,10,0,7,6,3,9,2,8,11 },
    { 13,11,7,14,12,1,3,9,5,0,15,4,8,6,2,10 },
    { 6,15,14,9,11,3,0,8,12,2,13,7,1,4,10,5 },
    { 10,2,8,4,7,6,1,5,15,11,9,14,3,12,13,0 }
};

/* b2s_compress macro body shared between thread and device address-space
 * overloads. Metal requires distinct function signatures per address
 * space (no generic address-space pointers); both overloads use this
 * single body via the textual macro to keep the compression byte-exact
 * with the OpenCL twin. */
#define MTL_B2S_COMPRESS_BODY(_HPTR, _BLOCK) \
    uint v[16], m[16]; \
    for (int i = 0; i < 8; i++) { v[i] = (_HPTR)[i]; v[i+8] = MTL_B2S_IV[i]; } \
    v[12] ^= (uint)counter; \
    v[13] ^= (uint)(counter >> 32); \
    if (last) v[14] = ~v[14]; \
    for (int i = 0; i < 16; i++) \
        m[i] = ((uint)(_BLOCK)[i*4]) | ((uint)(_BLOCK)[i*4+1]<<8) | \
               ((uint)(_BLOCK)[i*4+2]<<16) | ((uint)(_BLOCK)[i*4+3]<<24); \
    for (int r = 0; r < 10; r++) { \
        constant const uchar *s = MTL_B2S_SIGMA[r]; \
        v[0]+=v[4]+m[s[0]]; v[12]=rotl32(v[12]^v[0],(uint)16); v[8]+=v[12]; v[4]=rotl32(v[4]^v[8],(uint)20); \
        v[0]+=v[4]+m[s[1]]; v[12]=rotl32(v[12]^v[0],(uint)24); v[8]+=v[12]; v[4]=rotl32(v[4]^v[8],(uint)25); \
        v[1]+=v[5]+m[s[2]]; v[13]=rotl32(v[13]^v[1],(uint)16); v[9]+=v[13]; v[5]=rotl32(v[5]^v[9],(uint)20); \
        v[1]+=v[5]+m[s[3]]; v[13]=rotl32(v[13]^v[1],(uint)24); v[9]+=v[13]; v[5]=rotl32(v[5]^v[9],(uint)25); \
        v[2]+=v[6]+m[s[4]]; v[14]=rotl32(v[14]^v[2],(uint)16); v[10]+=v[14]; v[6]=rotl32(v[6]^v[10],(uint)20); \
        v[2]+=v[6]+m[s[5]]; v[14]=rotl32(v[14]^v[2],(uint)24); v[10]+=v[14]; v[6]=rotl32(v[6]^v[10],(uint)25); \
        v[3]+=v[7]+m[s[6]]; v[15]=rotl32(v[15]^v[3],(uint)16); v[11]+=v[15]; v[7]=rotl32(v[7]^v[11],(uint)20); \
        v[3]+=v[7]+m[s[7]]; v[15]=rotl32(v[15]^v[3],(uint)24); v[11]+=v[15]; v[7]=rotl32(v[7]^v[11],(uint)25); \
        v[0]+=v[5]+m[s[8]]; v[15]=rotl32(v[15]^v[0],(uint)16); v[10]+=v[15]; v[5]=rotl32(v[5]^v[10],(uint)20); \
        v[0]+=v[5]+m[s[9]]; v[15]=rotl32(v[15]^v[0],(uint)24); v[10]+=v[15]; v[5]=rotl32(v[5]^v[10],(uint)25); \
        v[1]+=v[6]+m[s[10]]; v[12]=rotl32(v[12]^v[1],(uint)16); v[11]+=v[12]; v[6]=rotl32(v[6]^v[11],(uint)20); \
        v[1]+=v[6]+m[s[11]]; v[12]=rotl32(v[12]^v[1],(uint)24); v[11]+=v[12]; v[6]=rotl32(v[6]^v[11],(uint)25); \
        v[2]+=v[7]+m[s[12]]; v[13]=rotl32(v[13]^v[2],(uint)16); v[8]+=v[13]; v[7]=rotl32(v[7]^v[8],(uint)20); \
        v[2]+=v[7]+m[s[13]]; v[13]=rotl32(v[13]^v[2],(uint)24); v[8]+=v[13]; v[7]=rotl32(v[7]^v[8],(uint)25); \
        v[3]+=v[4]+m[s[14]]; v[14]=rotl32(v[14]^v[3],(uint)16); v[9]+=v[14]; v[4]=rotl32(v[4]^v[9],(uint)20); \
        v[3]+=v[4]+m[s[15]]; v[14]=rotl32(v[14]^v[3],(uint)24); v[9]+=v[14]; v[4]=rotl32(v[4]^v[9],(uint)25); \
    } \
    for (int i = 0; i < 8; i++) (_HPTR)[i] ^= v[i] ^ v[i+8];

/* Thread-address-space block overload. Used by template_iterate (which
 * builds the next-iter input in a thread-local buf[64]). */
static inline void b2s_compress(thread uint *h, thread const uchar *block,
                                ulong counter, int last) {
    MTL_B2S_COMPRESS_BODY(h, block)
}

/* Device-address-space block overload. Used by template_finalize hot
 * path which passes `data + pos` -- a slice of the device-side
 * buf_scratch_pool. Metal forbids generic address-space pointers in
 * function signatures, so two overloads are required. */
static inline void b2s_compress(thread uint *h, device const uchar *block,
                                ulong counter, int last) {
    MTL_B2S_COMPRESS_BODY(h, block)
}

/* --- BLAKE2B compress (mirrors gpu_common.cl lines 1320-1406 byte-for-byte).
 *
 * Phase 2d.7a BLAKE2 family Metal port. BLAKE2B-256 (32-byte truncated)
 * and BLAKE2B-512 (full 64-byte) both use this single compression.
 * 128-byte block, 64-bit lanes, 12-round G-mixing.
 *
 * Constants are MTL_-prefixed per Pattern 2. The G-mixing macro uses
 * rotl64 (Metal substitute for OpenCL rotate of ulong); the spec calls
 * for rotate-RIGHT by (32, 24, 16, 63), expressed as rotate-LEFT by
 * (64-32, 64-24, 64-16, 64-63) to mirror the OpenCL twin's formulation.
 *
 * R2 register pressure: 8 ulong chaining (h) + 16 ulong v[] + 16 ulong m[]
 * working state = 80 64-bit registers ~= 160 32-bit registers. Architect
 * §3 R2 callout treats this as the boundary case; M1 + AMD verified
 * during OpenCL twin's prior gfx1201 testing (no spill issues in
 * comparable BLAKE2b-256 dispatch). */
constant ulong MTL_B2B_IV[8] = {
    0x6A09E667F3BCC908UL, 0xBB67AE8584CAA73BUL,
    0x3C6EF372FE94F82BUL, 0xA54FF53A5F1D36F1UL,
    0x510E527FADE682D1UL, 0x9B05688C2B3E6C1FUL,
    0x1F83D9ABFB41BD6BUL, 0x5BE0CD19137E2179UL
};

/* BLAKE2b uses 12 rounds; SIGMA wraps modulo 10 (rounds 10/11 reuse 0/1)
 * per RFC 7693. Stored as 12 rows for direct round-index lookup. */
constant uchar MTL_B2B_SIGMA[12][16] = {
    {  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15 },
    { 14, 10,  4,  8,  9, 15, 13,  6,  1, 12,  0,  2, 11,  7,  5,  3 },
    { 11,  8, 12,  0,  5,  2, 15, 13, 10, 14,  3,  6,  7,  1,  9,  4 },
    {  7,  9,  3,  1, 13, 12, 11, 14,  2,  6,  5, 10,  4,  0, 15,  8 },
    {  9,  0,  5,  7,  2,  4, 10, 15, 14,  1, 11, 12,  6,  8,  3, 13 },
    {  2, 12,  6, 10,  0, 11,  8,  3,  4, 13,  7,  5, 15, 14,  1,  9 },
    { 12,  5,  1, 15, 14, 13,  4, 10,  0,  7,  6,  3,  9,  2,  8, 11 },
    { 13, 11,  7, 14, 12,  1,  3,  9,  5,  0, 15,  4,  8,  6,  2, 10 },
    {  6, 15, 14,  9, 11,  3,  0,  8, 12,  2, 13,  7,  1,  4, 10,  5 },
    { 10,  2,  8,  4,  7,  6,  1,  5, 15, 11,  9, 14,  3, 12, 13,  0 },
    {  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15 },
    { 14, 10,  4,  8,  9, 15, 13,  6,  1, 12,  0,  2, 11,  7,  5,  3 }
};

/* BLAKE2b G mixing: rotates by (32, 24, 16, 63). Mirrors gpu_common.cl
 * B2B_G macro byte-for-byte; rotl64 substitutes OpenCL's rotate(ulong). */
#define MTL_B2B_G(a, b, c, d, x, y) do { \
    a = a + b + (x); \
    d = rotl64(d ^ a, (ulong)(64 - 32)); \
    c = c + d; \
    b = rotl64(b ^ c, (ulong)(64 - 24)); \
    a = a + b + (y); \
    d = rotl64(d ^ a, (ulong)(64 - 16)); \
    c = c + d; \
    b = rotl64(b ^ c, (ulong)(64 - 63)); \
} while (0)

/* b2b_compress macro body shared between thread and device address-space
 * overloads (same rationale as MTL_B2S_COMPRESS_BODY above). */
#define MTL_B2B_COMPRESS_BODY(_HPTR, _BLOCK) \
    ulong v[16], m[16]; \
    for (int i = 0; i < 16; i++) { \
        int b = i * 8; \
        m[i] = ((ulong)(_BLOCK)[b]) \
             | ((ulong)(_BLOCK)[b + 1] << 8) \
             | ((ulong)(_BLOCK)[b + 2] << 16) \
             | ((ulong)(_BLOCK)[b + 3] << 24) \
             | ((ulong)(_BLOCK)[b + 4] << 32) \
             | ((ulong)(_BLOCK)[b + 5] << 40) \
             | ((ulong)(_BLOCK)[b + 6] << 48) \
             | ((ulong)(_BLOCK)[b + 7] << 56); \
    } \
    for (int i = 0; i < 8; i++) { v[i] = (_HPTR)[i]; v[i + 8] = MTL_B2B_IV[i]; } \
    v[12] ^= t0; \
    v[13] ^= t1; \
    if (last) v[14] = ~v[14]; \
    for (int r = 0; r < 12; r++) { \
        constant const uchar *s = MTL_B2B_SIGMA[r]; \
        MTL_B2B_G(v[ 0], v[ 4], v[ 8], v[12], m[s[ 0]], m[s[ 1]]); \
        MTL_B2B_G(v[ 1], v[ 5], v[ 9], v[13], m[s[ 2]], m[s[ 3]]); \
        MTL_B2B_G(v[ 2], v[ 6], v[10], v[14], m[s[ 4]], m[s[ 5]]); \
        MTL_B2B_G(v[ 3], v[ 7], v[11], v[15], m[s[ 6]], m[s[ 7]]); \
        MTL_B2B_G(v[ 0], v[ 5], v[10], v[15], m[s[ 8]], m[s[ 9]]); \
        MTL_B2B_G(v[ 1], v[ 6], v[11], v[12], m[s[10]], m[s[11]]); \
        MTL_B2B_G(v[ 2], v[ 7], v[ 8], v[13], m[s[12]], m[s[13]]); \
        MTL_B2B_G(v[ 3], v[ 4], v[ 9], v[14], m[s[14]], m[s[15]]); \
    } \
    for (int i = 0; i < 8; i++) (_HPTR)[i] ^= v[i] ^ v[i + 8];

/* Thread-block overload. Used by template_iterate (next-iter input in
 * thread-local buf[128]). */
static inline void b2b_compress(thread ulong *h, thread const uchar *block,
                                ulong t0, ulong t1, int last) {
    MTL_B2B_COMPRESS_BODY(h, block)
}

/* Device-block overload. Used by template_finalize hot path which passes
 * a slice of device-side buf_scratch_pool. */
static inline void b2b_compress(thread ulong *h, device const uchar *block,
                                ulong t0, ulong t1, int last) {
    MTL_B2B_COMPRESS_BODY(h, block)
}

/* --- Keccak-f[1600] permutation (Phase 2d.7b). Mirrors gpu_common.cl
 * rev 1.14+ byte-for-byte. Shared by all 8 sponge ops (Keccak-{224,256,
 * 384,512} + SHA3-{224,256,384,512}); differences are entirely per-algo
 * rate + domain-pad byte handled by the absorb helpers in each
 * metal_<algo>_core.metal. Pattern 2: constants are MTL_-prefixed per
 * the namespacing rule (sibling of MTL_SHA256_K / MTL_SHA512_K /
 * MTL_MD5_S). cl2metal.py rev 1.7 rewrites bare KECCAK_RC + KECCAK_ROTC
 * token references in generated core sources to the MTL_-prefixed
 * names. Pointer-state signature (thread ulong *st) matches sha512_block
 * / rmd160_block convention; keccakf1600 takes no uchar* arg so no
 * dual address-space overload is required (compare b2s_compress /
 * b2b_compress). The 25-ulong state is `thread`-local in every caller
 * (template_state::sp embedded as `ulong sp[25]`). Pattern 1: pointer
 * arg explicitly thread-qualified. Pattern 3: static inline. */
constant ulong MTL_KECCAK_RC[24] = {
    0x0000000000000001UL, 0x0000000000008082UL, 0x800000000000808aUL,
    0x8000000080008000UL, 0x000000000000808bUL, 0x0000000080000001UL,
    0x8000000080008081UL, 0x8000000000008009UL, 0x000000000000008aUL,
    0x0000000000000088UL, 0x0000000080008009UL, 0x000000008000000aUL,
    0x000000008000808bUL, 0x800000000000008bUL, 0x8000000000008089UL,
    0x8000000000008003UL, 0x8000000000008002UL, 0x8000000000000080UL,
    0x000000000000800aUL, 0x800000008000000aUL, 0x8000000080008081UL,
    0x8000000000008080UL, 0x0000000080000001UL, 0x8000000080008008UL
};

constant uint MTL_KECCAK_ROTC[25] = {
     0,  1, 62, 28, 27,
    36, 44,  6, 55, 20,
     3, 10, 43, 25, 39,
    41, 45, 15, 21,  8,
    18,  2, 61, 56, 14
};

/* --- rotl64 (64-bit rotate-left) is defined above near b2b_compress
 * (Phase 2d.7a). keccakf1600 reuses it for the theta and rho+pi steps. */

static inline void keccakf1600(thread ulong *st) {
    for (int round = 0; round < 24; round++) {
        ulong C[5], D[5];
        for (int x = 0; x < 5; x++)
            C[x] = st[x] ^ st[x+5] ^ st[x+10] ^ st[x+15] ^ st[x+20];
        for (int x = 0; x < 5; x++) {
            D[x] = C[(x+4) % 5] ^ rotl64(C[(x+1) % 5], (ulong)1);
            for (int y = 0; y < 25; y += 5)
                st[x+y] ^= D[x];
        }
        ulong B[25];
        for (int x = 0; x < 5; x++)
            for (int y = 0; y < 5; y++)
                B[x + 5 * ((2*y + 3*x) % 5)] = rotl64(st[x*5+y], (ulong)MTL_KECCAK_ROTC[x*5+y]);
        for (int x = 0; x < 5; x++)
            for (int y = 0; y < 25; y += 5)
                st[x+y] = B[x+y] ^ (~B[((x+1)%5)+y] & B[((x+2)%5)+y]);
        st[0] ^= MTL_KECCAK_RC[round];
    }
}

/* --- hex_byte_be64: hex-encode one byte into a BE 16-bit pair within a
 * 64-bit word slot (mirrors gpu_common.cl line 787).
 *
 * Returns the lowercase-hex-encoded representation of byte b as a 16-bit
 * value where the upper 8 bits are the high-nibble character and the
 * lower 8 bits are the low-nibble character. Sized to ulong to allow
 * shifting into BE positions inside the 8-hex-char ulong words produced
 * by sha512_to_hex_lc. Pattern 3: static inline. */
static inline ulong mtl_hex_byte_be64(uint b) {
    uint hi = (b >> 4) & 0xfu;
    uint lo = b & 0xfu;
    return ((ulong)(hi + ((hi < 10u) ? (uint)'0' : (uint)('a' - 10))) << 8)
         |  (ulong)(lo + ((lo < 10u) ? (uint)'0' : (uint)('a' - 10)));
}

/* --- sha512_to_hex_lc: encode 8-ulong BE SHA-512 state into 16-ulong BE
 * hex M[] (each ulong holds 8 hex chars BE). Mirrors gpu_common.cl
 * line 978 byte-for-byte.
 *
 * Used by sha512_core's template_iterate() -i loop step to re-hash the
 * 128-byte lowercase-hex representation of the previous digest. Pattern
 * 3: static inline. Pattern 1: both pointer args explicitly thread-
 * qualified. */
static inline void sha512_to_hex_lc(thread const ulong *state, thread ulong *M)
{
    for (int i = 0; i < 8; i++) {
        ulong s = state[i];
        uint b0 = (uint)((s >> 56) & 0xffUL), b1 = (uint)((s >> 48) & 0xffUL);
        uint b2 = (uint)((s >> 40) & 0xffUL), b3 = (uint)((s >> 32) & 0xffUL);
        uint b4 = (uint)((s >> 24) & 0xffUL), b5 = (uint)((s >> 16) & 0xffUL);
        uint b6 = (uint)((s >> 8)  & 0xffUL), b7 = (uint)(s & 0xffUL);
        M[i*2]   = (mtl_hex_byte_be64(b0) << 48) | (mtl_hex_byte_be64(b1) << 32)
                 | (mtl_hex_byte_be64(b2) << 16) |  mtl_hex_byte_be64(b3);
        M[i*2+1] = (mtl_hex_byte_be64(b4) << 48) | (mtl_hex_byte_be64(b5) << 32)
                 | (mtl_hex_byte_be64(b6) << 16) |  mtl_hex_byte_be64(b7);
    }
}

/* --- Hex encode helpers (mirrors gpu_common.cl lines 771-812).
 * Pattern 3: static inline. Pattern 1: thread-qualified `M`.
 * Pre-declared for Phase 2 iter loop (template_iterate); unused in
 * Phase 1 raw-MD5 dispatch. */
static inline uint mtl_hex_byte_lc(uint b) {
    uint hi = (b >> 4) & 0xfu;
    uint lo = b & 0xfu;
    uint hc = hi + ((hi < 10u) ? (uint)'0' : (uint)('a' - 10));
    uint lc = lo + ((lo < 10u) ? (uint)'0' : (uint)('a' - 10));
    return hc | (lc << 8);
}

static inline uint mtl_hex_byte_uc(uint b) {
    uint hi = (b >> 4) & 0xfu;
    uint lo = b & 0xfu;
    uint hc = hi + ((hi < 10u) ? (uint)'0' : (uint)('A' - 10));
    uint lc = lo + ((lo < 10u) ? (uint)'0' : (uint)('A' - 10));
    return hc | (lc << 8);
}

static inline void md5_to_hex_lc(uint hx, uint hy, uint hz, uint hw,
                                 thread uint *M)
{
    uint v[4]; v[0]=hx; v[1]=hy; v[2]=hz; v[3]=hw;
    for (int i = 0; i < 4; i++) {
        uint b0 = v[i] & 0xffu;        uint b1 = (v[i] >> 8) & 0xffu;
        uint b2 = (v[i] >> 16) & 0xffu;uint b3 = (v[i] >> 24) & 0xffu;
        M[i*2]   = mtl_hex_byte_lc(b0) | (mtl_hex_byte_lc(b1) << 16);
        M[i*2+1] = mtl_hex_byte_lc(b2) | (mtl_hex_byte_lc(b3) << 16);
    }
}

static inline void md5_to_hex_uc(uint hx, uint hy, uint hz, uint hw,
                                 thread uint *M)
{
    uint v[4]; v[0]=hx; v[1]=hy; v[2]=hz; v[3]=hw;
    for (int i = 0; i < 4; i++) {
        uint b0 = v[i] & 0xffu;        uint b1 = (v[i] >> 8) & 0xffu;
        uint b2 = (v[i] >> 16) & 0xffu;uint b3 = (v[i] >> 24) & 0xffu;
        M[i*2]   = mtl_hex_byte_uc(b0) | (mtl_hex_byte_uc(b1) << 16);
        M[i*2+1] = mtl_hex_byte_uc(b2) | (mtl_hex_byte_uc(b3) << 16);
    }
}

/* --- compact_mix: hash-table mixing (mirrors gpu_common.cl line 623).
 * Pattern 3: static inline. */
static inline ulong compact_mix(ulong k) { return k ^ (k >> 32); }

/* --- probe_compact_idx: compact hash table + overflow lookup.
 *
 * Mirrors gpu_common.cl::probe_compact_idx (line 690). Returns 1 on
 * hit (and sets *out_idx), 0 on miss.
 *
 * Pattern 1: every pointer parameter is address-space-qualified:
 *   device const T *   for global read-only tables
 *   thread uint *      for the result-write pointer (lane-local).
 *
 * The 12 buffer args mirror the OpenCL 12 __global args; semantics
 * identical (compact_fp/compact_idx pair, hash_data, overflow). */
static inline int probe_compact_idx(
    uint hx, uint hy, uint hz, uint hw,
    device const uint  *compact_fp,
    device const uint  *compact_idx,
    ulong               compact_mask,
    uint                max_probe,
    uint                hash_data_count,
    device const uchar *hash_data_buf,
    device const ulong *hash_data_off,
    device const ulong *overflow_keys,
    device const uchar *overflow_hashes,
    device const uint  *overflow_offsets,
    uint                overflow_count,
    thread uint        *out_idx)
{
    ulong key = ((ulong)hy << 32) | hx;
    uint  fp  = (uint)(key >> 32);
    if (fp == 0u) fp = 1u;
    ulong pos = compact_mix(key) & compact_mask;
    for (int p = 0; p < (int)max_probe; p++) {
        uint cfp = compact_fp[pos];
        if (cfp == 0u) break;
        if (cfp == fp) {
            uint idx = compact_idx[pos];
            if (idx < hash_data_count) {
                ulong off = hash_data_off[idx];
                device const uint *ref = (device const uint *)(hash_data_buf + off);
                if (hx == ref[0] && hy == ref[1] && hz == ref[2] && hw == ref[3]) {
                    *out_idx = idx;
                    return 1;
                }
            }
        }
        pos = (pos + 1u) & compact_mask;
    }
    if (overflow_count > 0u) {
        int lo = 0, hi = (int)overflow_count - 1;
        while (lo <= hi) {
            int mid = (lo + hi) / 2;
            ulong mkey = overflow_keys[mid];
            if (key < mkey) hi = mid - 1;
            else if (key > mkey) lo = mid + 1;
            else {
                uint ooff = overflow_offsets[mid];
                device const uint *oref = (device const uint *)(overflow_hashes + ooff);
                if (hx == oref[0] && hy == oref[1] && hz == oref[2] && hw == oref[3]) {
                    *out_idx = hash_data_count + (uint)mid;
                    return 1;
                }
                for (int d = mid - 1; d >= 0 && overflow_keys[d] == key; d--) {
                    oref = (device const uint *)(overflow_hashes + overflow_offsets[d]);
                    if (hx == oref[0] && hy == oref[1] && hz == oref[2] && hw == oref[3]) {
                        *out_idx = hash_data_count + (uint)d;
                        return 1;
                    }
                }
                for (int d = mid + 1; d < (int)overflow_count && overflow_keys[d] == key; d++) {
                    oref = (device const uint *)(overflow_hashes + overflow_offsets[d]);
                    if (hx == oref[0] && hy == oref[1] && hz == oref[2] && hw == oref[3]) {
                        *out_idx = hash_data_count + (uint)d;
                        return 1;
                    }
                }
                break;
            }
        }
    }
    return 0;
}

/* --- EMIT_HIT_4_DEDUP_OR_OVERFLOW (mirrors gpu_common.cl line 312).
 *
 * Pattern 6: ONE multi-line macro in Phase 1. Trailing-backslash
 * continuations have NO whitespace after them; offline `xcrun metal -E`
 * preprocess pass surfaces any drift; metal_jit_harness --check-patterns
 * greps for `\\ +\n` to catch the bug class.
 *
 * Pattern 1: callers MUST pass:
 *   hits         as `device uint *`
 *   hit_count    as `device atomic_uint *`
 *   hashes_shown as `device atomic_uint *`
 *   ovr_set      as `device atomic_uint *`
 *   ovr_gid      as `device atomic_uint *`
 *
 * Metal's atomics are typed (vs OpenCL's volatile-pointer-as-atomic).
 * The kernel must declare the pointers with `atomic_uint` element type;
 * the host buffer layout (offsets 100, 104, 128) is identical, but the
 * pointer view is typed.
 *
 * Semantics IDENTICAL to OpenCL EMIT_HIT_4_DEDUP_OR_OVERFLOW (see
 * gpu_common.cl §B3 protocol notes). Differences:
 *   - atomic_or       -> atomic_fetch_or_explicit(..., memory_order_relaxed)
 *   - atomic_and      -> atomic_fetch_and_explicit(..., memory_order_relaxed)
 *   - atomic_add      -> atomic_fetch_add_explicit(..., memory_order_relaxed)
 *   - atomic_cmpxchg  -> atomic_compare_exchange_weak_explicit
 *   - mem_fence(GLOBAL_MEM_FENCE) -> threadgroup_barrier(mem_flags::mem_device)
 */
#define MTL_OVR_CASMIN_GID(ovr_gid, lane_gid)                                          \
    do {                                                                               \
        uint _cur, _new = (uint)(lane_gid);                                            \
        do {                                                                           \
            _cur = atomic_load_explicit((ovr_gid), memory_order_relaxed);              \
            if (_new >= _cur) break;                                                   \
        } while (!atomic_compare_exchange_weak_explicit(                               \
                    (ovr_gid), &_cur, _new,                                            \
                    memory_order_relaxed, memory_order_relaxed));                      \
    } while (0)

#define EMIT_HIT_4_DEDUP_OR_OVERFLOW(hits, hit_count, max_hits, widx, sidx, iter,      \
                                     a, b, c, d,                                       \
                                     hashes_shown, matched_idx, dedup_mask,            \
                                     ovr_set, ovr_gid, lane_gid)                       \
    do {                                                                               \
        uint _dm = (uint)(dedup_mask);                                                 \
        uint _mi = (uint)(matched_idx);                                                \
        uint _prev = atomic_fetch_or_explicit(&(hashes_shown)[_mi], _dm,               \
                                              memory_order_relaxed);                   \
        if ((_prev & _dm) == 0u) {                                                     \
            uint _slot = atomic_fetch_add_explicit((hit_count), 1u,                    \
                                                   memory_order_relaxed);              \
            if (_slot < (max_hits)) {                                                  \
                uint _base = _slot * HIT_STRIDE;                                       \
                (hits)[_base]   = (widx);                                              \
                (hits)[_base+1] = (sidx);                                              \
                (hits)[_base+2] = (iter);                                              \
                (hits)[_base+3] = (a);                                                 \
                (hits)[_base+4] = (b);                                                 \
                (hits)[_base+5] = (c);                                                 \
                (hits)[_base+6] = (d);                                                 \
                for (uint _z = 7u; _z < HIT_STRIDE; _z++) (hits)[_base+_z] = 0u;       \
                threadgroup_barrier(mem_flags::mem_device);                            \
            } else {                                                                   \
                atomic_fetch_and_explicit(&(hashes_shown)[_mi], ~_dm,                  \
                                          memory_order_relaxed);                       \
                MTL_OVR_CASMIN_GID((ovr_gid), (lane_gid));                             \
                atomic_fetch_or_explicit((ovr_set), 1u, memory_order_relaxed);         \
            }                                                                          \
        }                                                                              \
    } while (0)

/* --- EMIT_HIT_5_DEDUP_OR_OVERFLOW (mirrors gpu_common.cl line 336).
 *
 * Phase 2d.3.1 SHA-1 canary: 5-word digest emit variant. Same dedup +
 * overflow protocol as EMIT_HIT_4_DEDUP_OR_OVERFLOW (above); the only
 * differences are:
 *   - `h` is `thread const uint *` array of 5 words (vs 4 scalar args
 *     a/b/c/d in the 4-word form), so the hits-buffer write loop is
 *     `for _i = 0..4: hits[_base+3+_i] = h[_i]` instead of 4 explicit
 *     scalar stores. Mirrors the OpenCL twin verbatim.
 *   - Tail-zero loop starts at _z = 8u (was 7u in the 4-word form), so
 *     hits[_base+8..HIT_STRIDE-1] gets zeroed.
 *
 * Pattern 6: multi-line macro; backslash continuations have NO trailing
 * whitespace (metal_jit_harness --check-patterns enforces this).
 * Pattern 1: callers MUST pass:
 *   hits         as `device uint *`
 *   hit_count    as `device atomic_uint *`
 *   hashes_shown as `device atomic_uint *`
 *   ovr_set      as `device atomic_uint *`
 *   ovr_gid      as `device atomic_uint *`
 *
 * Semantics IDENTICAL to OpenCL EMIT_HIT_5_DEDUP_OR_OVERFLOW. Atomic
 * substitution is identical to the 4-word form above. */
#define EMIT_HIT_5_DEDUP_OR_OVERFLOW(hits, hit_count, max_hits, widx, sidx, iter, h,   \
                                     hashes_shown, matched_idx, dedup_mask,            \
                                     ovr_set, ovr_gid, lane_gid)                       \
    do {                                                                               \
        uint _dm = (uint)(dedup_mask);                                                 \
        uint _mi = (uint)(matched_idx);                                                \
        uint _prev = atomic_fetch_or_explicit(&(hashes_shown)[_mi], _dm,               \
                                              memory_order_relaxed);                   \
        if ((_prev & _dm) == 0u) {                                                     \
            uint _slot = atomic_fetch_add_explicit((hit_count), 1u,                    \
                                                   memory_order_relaxed);              \
            if (_slot < (max_hits)) {                                                  \
                uint _base = _slot * HIT_STRIDE;                                       \
                (hits)[_base]   = (widx);                                              \
                (hits)[_base+1] = (sidx);                                              \
                (hits)[_base+2] = (iter);                                              \
                for (uint _i = 0u; _i < 5u; _i++) (hits)[_base+3u+_i] = (h)[_i];       \
                for (uint _z = 8u; _z < HIT_STRIDE; _z++) (hits)[_base+_z] = 0u;       \
                threadgroup_barrier(mem_flags::mem_device);                            \
            } else {                                                                   \
                atomic_fetch_and_explicit(&(hashes_shown)[_mi], ~_dm,                  \
                                          memory_order_relaxed);                       \
                MTL_OVR_CASMIN_GID((ovr_gid), (lane_gid));                             \
                atomic_fetch_or_explicit((ovr_set), 1u, memory_order_relaxed);         \
            }                                                                          \
        }                                                                              \
    } while (0)

/* --- EMIT_HIT_6_DEDUP_OR_OVERFLOW (mirrors gpu_common.cl EMIT_HIT_6).
 *
 * Phase 2d.9b BCRYPT sibling: 6-word (24-byte) digest emit variant. Same
 * dedup + overflow protocol as EMIT_HIT_4 / EMIT_HIT_5 / EMIT_HIT_7;
 * the only differences are:
 *   - `h` is `thread const uint *` array of 6 words. The hits-buffer write
 *     loop is `for _i = 0..5: hits[_base+3+_i] = h[_i]`. BCRYPT's full
 *     24-byte digest is emitted (host hit-replay reads first 23 bytes for
 *     bf_encode_23; the 24th byte is the BE->LE swap tail pad that
 *     BF_encode discards).
 *   - Tail-zero loop starts at _z = 9u (3 metadata + 6 digest = 9):
 *     hits[_base+9..HIT_STRIDE-1] gets zeroed (10 trailing slots).
 *
 * Semantics IDENTICAL to OpenCL EMIT_HIT_6_DEDUP_OR_OVERFLOW. */
#define EMIT_HIT_6_DEDUP_OR_OVERFLOW(hits, hit_count, max_hits, widx, sidx, iter, h,   \
                                     hashes_shown, matched_idx, dedup_mask,            \
                                     ovr_set, ovr_gid, lane_gid)                       \
    do {                                                                               \
        uint _dm = (uint)(dedup_mask);                                                 \
        uint _mi = (uint)(matched_idx);                                                \
        uint _prev = atomic_fetch_or_explicit(&(hashes_shown)[_mi], _dm,               \
                                              memory_order_relaxed);                   \
        if ((_prev & _dm) == 0u) {                                                     \
            uint _slot = atomic_fetch_add_explicit((hit_count), 1u,                    \
                                                   memory_order_relaxed);              \
            if (_slot < (max_hits)) {                                                  \
                uint _base = _slot * HIT_STRIDE;                                       \
                (hits)[_base]   = (widx);                                              \
                (hits)[_base+1] = (sidx);                                              \
                (hits)[_base+2] = (iter);                                              \
                for (uint _i = 0u; _i < 6u; _i++) (hits)[_base+3u+_i] = (h)[_i];       \
                for (uint _z = 9u; _z < HIT_STRIDE; _z++) (hits)[_base+_z] = 0u;       \
                threadgroup_barrier(mem_flags::mem_device);                            \
            } else {                                                                   \
                atomic_fetch_and_explicit(&(hashes_shown)[_mi], ~_dm,                  \
                                          memory_order_relaxed);                       \
                MTL_OVR_CASMIN_GID((ovr_gid), (lane_gid));                             \
                atomic_fetch_or_explicit((ovr_set), 1u, memory_order_relaxed);         \
            }                                                                          \
        }                                                                              \
    } while (0)

/* --- EMIT_HIT_7_DEDUP_OR_OVERFLOW (mirrors gpu_common.cl line 348).
 *
 * Phase 2d.4.3 SHA-224 sibling: 7-word digest emit variant. Same dedup +
 * overflow protocol as EMIT_HIT_4 / EMIT_HIT_5 / EMIT_HIT_8; differences:
 *   - `h` is `thread const uint *` array of 7 words. The hits-buffer write
 *     loop is `for _i = 0..6: hits[_base+3+_i] = h[_i]`. SHA-224's 8th
 *     internal word (h[7]) is NOT emitted (28-byte / 7-word digest).
 *   - Tail-zero loop starts at _z = 10u: hits[_base+10..HIT_STRIDE-1]
 *     gets zeroed. Mirror of the OpenCL EMIT_HIT_7_DEDUP_OR_OVERFLOW
 *     defined in gpu_common.cl.
 *
 * Semantics IDENTICAL to OpenCL EMIT_HIT_7_DEDUP_OR_OVERFLOW. */
#define EMIT_HIT_7_DEDUP_OR_OVERFLOW(hits, hit_count, max_hits, widx, sidx, iter, h,   \
                                     hashes_shown, matched_idx, dedup_mask,            \
                                     ovr_set, ovr_gid, lane_gid)                       \
    do {                                                                               \
        uint _dm = (uint)(dedup_mask);                                                 \
        uint _mi = (uint)(matched_idx);                                                \
        uint _prev = atomic_fetch_or_explicit(&(hashes_shown)[_mi], _dm,               \
                                              memory_order_relaxed);                   \
        if ((_prev & _dm) == 0u) {                                                     \
            uint _slot = atomic_fetch_add_explicit((hit_count), 1u,                    \
                                                   memory_order_relaxed);              \
            if (_slot < (max_hits)) {                                                  \
                uint _base = _slot * HIT_STRIDE;                                       \
                (hits)[_base]   = (widx);                                              \
                (hits)[_base+1] = (sidx);                                              \
                (hits)[_base+2] = (iter);                                              \
                for (uint _i = 0u; _i < 7u; _i++) (hits)[_base+3u+_i] = (h)[_i];       \
                for (uint _z = 10u; _z < HIT_STRIDE; _z++) (hits)[_base+_z] = 0u;      \
                threadgroup_barrier(mem_flags::mem_device);                            \
            } else {                                                                   \
                atomic_fetch_and_explicit(&(hashes_shown)[_mi], ~_dm,                  \
                                          memory_order_relaxed);                       \
                MTL_OVR_CASMIN_GID((ovr_gid), (lane_gid));                             \
                atomic_fetch_or_explicit((ovr_set), 1u, memory_order_relaxed);         \
            }                                                                          \
        }                                                                              \
    } while (0)

/* --- EMIT_HIT_8_DEDUP_OR_OVERFLOW (mirrors gpu_common.cl line 360).
 *
 * Phase 2d.4.1 SHA-2/256 canary: 8-word digest emit variant. Same dedup +
 * overflow protocol as EMIT_HIT_4 / EMIT_HIT_5; the only differences
 * from EMIT_HIT_5 are:
 *   - `h` is `thread const uint *` array of 8 words (vs 5 for SHA-1),
 *     so the hits-buffer write loop is
 *     `for _i = 0..7: hits[_base+3+_i] = h[_i]`.
 *   - Tail-zero loop starts at _z = 11u (was 8u in 5-word, 7u in 4-word):
 *     hits[_base+11..HIT_STRIDE-1] gets zeroed. HIT_STRIDE = 19 means
 *     8 trailing slots are zeroed when digest occupies hits[_base+3..+10].
 *
 * Pattern 6: multi-line macro; backslash continuations have NO trailing
 * whitespace. Pattern 1: callers MUST pass typed atomic_uint pointers per
 * EMIT_HIT_4_DEDUP_OR_OVERFLOW protocol.
 *
 * Semantics IDENTICAL to OpenCL EMIT_HIT_8_DEDUP_OR_OVERFLOW; future
 * SHA-2 siblings (sha224 7 words, sha384 12, sha512 16) will add their
 * own EMIT_HIT_N macros following the same pattern. */
#define EMIT_HIT_8_DEDUP_OR_OVERFLOW(hits, hit_count, max_hits, widx, sidx, iter, h,   \
                                     hashes_shown, matched_idx, dedup_mask,            \
                                     ovr_set, ovr_gid, lane_gid)                       \
    do {                                                                               \
        uint _dm = (uint)(dedup_mask);                                                 \
        uint _mi = (uint)(matched_idx);                                                \
        uint _prev = atomic_fetch_or_explicit(&(hashes_shown)[_mi], _dm,               \
                                              memory_order_relaxed);                   \
        if ((_prev & _dm) == 0u) {                                                     \
            uint _slot = atomic_fetch_add_explicit((hit_count), 1u,                    \
                                                   memory_order_relaxed);              \
            if (_slot < (max_hits)) {                                                  \
                uint _base = _slot * HIT_STRIDE;                                       \
                (hits)[_base]   = (widx);                                              \
                (hits)[_base+1] = (sidx);                                              \
                (hits)[_base+2] = (iter);                                              \
                for (uint _i = 0u; _i < 8u; _i++) (hits)[_base+3u+_i] = (h)[_i];       \
                for (uint _z = 11u; _z < HIT_STRIDE; _z++) (hits)[_base+_z] = 0u;      \
                threadgroup_barrier(mem_flags::mem_device);                            \
            } else {                                                                   \
                atomic_fetch_and_explicit(&(hashes_shown)[_mi], ~_dm,                  \
                                          memory_order_relaxed);                       \
                MTL_OVR_CASMIN_GID((ovr_gid), (lane_gid));                             \
                atomic_fetch_or_explicit((ovr_set), 1u, memory_order_relaxed);         \
            }                                                                          \
        }                                                                              \
    } while (0)

/* --- EMIT_HIT_12_DEDUP_OR_OVERFLOW (mirrors gpu_common.cl 12-word emit).
 *
 * Phase 2d.5.3 SHA-384 sibling: 12-word digest emit variant. Same dedup +
 * overflow protocol as EMIT_HIT_4 / EMIT_HIT_5 / EMIT_HIT_7 / EMIT_HIT_8 /
 * EMIT_HIT_16; the differences from EMIT_HIT_8 are:
 *   - `h` is `thread const uint *` array of 12 words (vs 8 for SHA-256),
 *     so the hits-buffer write loop is
 *     `for _i = 0..11: hits[_base+3+_i] = h[_i]`.
 *   - Tail-zero loop starts at _z = 15u (= 3 metadata + 12 digest):
 *     hits[_base+15..HIT_STRIDE-1] gets zeroed (4 trailing slots).
 *
 * SHA-384 is a truncated SHA-512: the kernel computes the full 8-ulong
 * SHA-512 state with SHA-384 IVs, then template_state_to_h() decomposes
 * state[0..5] (6 ulong = 12 uint32 = 48 bytes) into st->h with the
 * SHA-384 truncation. state[6..7] are dropped.
 *
 * Pattern 6: multi-line macro; backslash continuations have NO trailing
 * whitespace. Pattern 1: callers MUST pass typed atomic_uint pointers per
 * EMIT_HIT_4_DEDUP_OR_OVERFLOW protocol.
 *
 * Semantics IDENTICAL to OpenCL EMIT_HIT_12_DEDUP_OR_OVERFLOW. */
#define EMIT_HIT_12_DEDUP_OR_OVERFLOW(hits, hit_count, max_hits, widx, sidx, iter, h,  \
                                      hashes_shown, matched_idx, dedup_mask,           \
                                      ovr_set, ovr_gid, lane_gid)                      \
    do {                                                                               \
        uint _dm = (uint)(dedup_mask);                                                 \
        uint _mi = (uint)(matched_idx);                                                \
        uint _prev = atomic_fetch_or_explicit(&(hashes_shown)[_mi], _dm,               \
                                              memory_order_relaxed);                   \
        if ((_prev & _dm) == 0u) {                                                     \
            uint _slot = atomic_fetch_add_explicit((hit_count), 1u,                    \
                                                   memory_order_relaxed);              \
            if (_slot < (max_hits)) {                                                  \
                uint _base = _slot * HIT_STRIDE;                                       \
                (hits)[_base]   = (widx);                                              \
                (hits)[_base+1] = (sidx);                                              \
                (hits)[_base+2] = (iter);                                              \
                for (uint _i = 0u; _i < 12u; _i++) (hits)[_base+3u+_i] = (h)[_i];      \
                for (uint _z = 15u; _z < HIT_STRIDE; _z++) (hits)[_base+_z] = 0u;      \
                threadgroup_barrier(mem_flags::mem_device);                            \
            } else {                                                                   \
                atomic_fetch_and_explicit(&(hashes_shown)[_mi], ~_dm,                  \
                                          memory_order_relaxed);                       \
                MTL_OVR_CASMIN_GID((ovr_gid), (lane_gid));                             \
                atomic_fetch_or_explicit((ovr_set), 1u, memory_order_relaxed);         \
            }                                                                          \
        }                                                                              \
    } while (0)

/* --- EMIT_HIT_16_DEDUP_OR_OVERFLOW (mirrors gpu_common.cl 16-word emit).
 *
 * Phase 2d.5.1 SHA-2/512 canary: 16-word digest emit variant -- LARGEST
 * digest width supported by mdxfind. Same dedup + overflow protocol as
 * EMIT_HIT_4 / EMIT_HIT_5 / EMIT_HIT_7 / EMIT_HIT_8; the differences
 * from EMIT_HIT_8 are:
 *   - `h` is `thread const uint *` array of 16 words (vs 8 for SHA-256),
 *     so the hits-buffer write loop is
 *     `for _i = 0..15: hits[_base+3+_i] = h[_i]`.
 *   - Tail-zero loop starts at _z = 19u (= HIT_STRIDE). Because the 16
 *     digest words exactly fill the 16 trailing slots of the 19-uint32
 *     HIT_STRIDE slot (3 metadata + 16 digest = 19), the tail-zero loop
 *     body is never entered. The loop is kept for textual symmetry with
 *     the smaller EMIT_HIT_N macros + as a future-proofing harness if
 *     HIT_STRIDE ever grows.
 *
 * Pattern 6: multi-line macro; backslash continuations have NO trailing
 * whitespace. Pattern 1: callers MUST pass typed atomic_uint pointers per
 * EMIT_HIT_4_DEDUP_OR_OVERFLOW protocol.
 *
 * Semantics IDENTICAL to OpenCL EMIT_HIT_16_DEDUP_OR_OVERFLOW. */
#define EMIT_HIT_16_DEDUP_OR_OVERFLOW(hits, hit_count, max_hits, widx, sidx, iter, h,  \
                                      hashes_shown, matched_idx, dedup_mask,           \
                                      ovr_set, ovr_gid, lane_gid)                      \
    do {                                                                               \
        uint _dm = (uint)(dedup_mask);                                                 \
        uint _mi = (uint)(matched_idx);                                                \
        uint _prev = atomic_fetch_or_explicit(&(hashes_shown)[_mi], _dm,               \
                                              memory_order_relaxed);                   \
        if ((_prev & _dm) == 0u) {                                                     \
            uint _slot = atomic_fetch_add_explicit((hit_count), 1u,                    \
                                                   memory_order_relaxed);              \
            if (_slot < (max_hits)) {                                                  \
                uint _base = _slot * HIT_STRIDE;                                       \
                (hits)[_base]   = (widx);                                              \
                (hits)[_base+1] = (sidx);                                              \
                (hits)[_base+2] = (iter);                                              \
                for (uint _i = 0u; _i < 16u; _i++) (hits)[_base+3u+_i] = (h)[_i];      \
                for (uint _z = 19u; _z < HIT_STRIDE; _z++) (hits)[_base+_z] = 0u;      \
                threadgroup_barrier(mem_flags::mem_device);                            \
            } else {                                                                   \
                atomic_fetch_and_explicit(&(hashes_shown)[_mi], ~_dm,                  \
                                          memory_order_relaxed);                       \
                MTL_OVR_CASMIN_GID((ovr_gid), (lane_gid));                             \
                atomic_fetch_or_explicit((ovr_set), 1u, memory_order_relaxed);         \
            }                                                                          \
        }                                                                              \
    } while (0)

/* --- EMIT_HIT_10_DEDUP_OR_OVERFLOW (mirrors gpu_common.cl line 429).
 *
 * Phase 2d.6 RIPEMD-320 sibling: 10-word digest emit variant. Same dedup +
 * overflow protocol as EMIT_HIT_5 / EMIT_HIT_7 / EMIT_HIT_8 / EMIT_HIT_12 /
 * EMIT_HIT_16. Differences from EMIT_HIT_8:
 *   - `h` is `thread const uint *` array of 10 words (RIPEMD-320 = 10
 *     uint32 LE digest), so the hits-buffer write loop is
 *     `for _i = 0..9: hits[_base+3+_i] = h[_i]`.
 *   - Tail-zero loop starts at _z = 13u (= 3 metadata + 10 digest):
 *     hits[_base+13..HIT_STRIDE-1] gets zeroed (6 trailing slots).
 *
 * Pattern 6: multi-line macro; backslash continuations have NO trailing
 * whitespace. Pattern 1: callers MUST pass typed atomic_uint pointers per
 * EMIT_HIT_4_DEDUP_OR_OVERFLOW protocol.
 *
 * Semantics IDENTICAL to OpenCL EMIT_HIT_10_DEDUP_OR_OVERFLOW. */
#define EMIT_HIT_10_DEDUP_OR_OVERFLOW(hits, hit_count, max_hits, widx, sidx, iter, h,  \
                                      hashes_shown, matched_idx, dedup_mask,           \
                                      ovr_set, ovr_gid, lane_gid)                      \
    do {                                                                               \
        uint _dm = (uint)(dedup_mask);                                                 \
        uint _mi = (uint)(matched_idx);                                                \
        uint _prev = atomic_fetch_or_explicit(&(hashes_shown)[_mi], _dm,               \
                                              memory_order_relaxed);                   \
        if ((_prev & _dm) == 0u) {                                                     \
            uint _slot = atomic_fetch_add_explicit((hit_count), 1u,                    \
                                                   memory_order_relaxed);              \
            if (_slot < (max_hits)) {                                                  \
                uint _base = _slot * HIT_STRIDE;                                       \
                (hits)[_base]   = (widx);                                              \
                (hits)[_base+1] = (sidx);                                              \
                (hits)[_base+2] = (iter);                                              \
                for (uint _i = 0u; _i < 10u; _i++) (hits)[_base+3u+_i] = (h)[_i];      \
                for (uint _z = 13u; _z < HIT_STRIDE; _z++) (hits)[_base+_z] = 0u;      \
                threadgroup_barrier(mem_flags::mem_device);                            \
            } else {                                                                   \
                atomic_fetch_and_explicit(&(hashes_shown)[_mi], ~_dm,                  \
                                          memory_order_relaxed);                       \
                MTL_OVR_CASMIN_GID((ovr_gid), (lane_gid));                             \
                atomic_fetch_or_explicit((ovr_set), 1u, memory_order_relaxed);         \
            }                                                                          \
        }                                                                              \
    } while (0)
