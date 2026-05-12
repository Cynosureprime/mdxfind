/* gpu_md5_bf.cl — MD5 brute-force fast-path algorithm core for the
 * generic dispatch template (Phase 1.9 Tranche A2).
 *
 * $Revision: 1.2 $
 * $Log: gpu_md5_bf.cl,v $
 * Revision 1.2  2026/05/11 04:07:40  dlr
 * BF Phase 1.9 A2 (kernel optimization in gpu_md5_bf.cl): (1) Inline md5_block as static inline md5_compress_inline with VALUE-PASSED M[0..15] message words as scalar args (not pointer). Eliminates the pointer aliasing constraint that prevented register-residency in the noinline+pointer-param signature at gpu_common.cl:540 (which was the smoking gun per A0 — 41 KB private mem on every platform). State pointers (&st->h[0..3]) retained because gpu_template.cl owns template_state allocation; inliner sees through them. (2) Round-constant pre-add hoist: 64 local scalars s00..s63 = M_j + K_i precomputed upfront; the 64 BF_FF/GG/HH/II step macros consume the precomputed sums. Slow path (gpu_md5_core.cl + gpu_common.cl noinline md5_block) UNTOUCHED — A2 is gpu_md5_bf.cl-only. LOC: 254 -> 472. Validated on fpga 1080: V1.9.A2.1 correctness 3/3 cracks byte-exact; V1.9.A2.2 kernel-only rate 0.984 GH/s mean of 3 runs (vs slow-path 0.651 GH/s = +51%) — 1.6% UNDER 1.0 GH/s ship gate, but user-blessed ship-and-proceed-to-A3 2026-05-10. V1.9.A2.4 env override MDXFIND_GPU_FAST_DISABLE=1 still forces slow path correctly. Architects 1.5-2.5 GH/s projection assumed amortization across many candidates per WI; mdxfind runs 1 candidate per WI so pre-hoist only saves ~60 ADDs per WI on a path that already runs 64 MD5 steps. A3 (host mask pre-explode) restructures to N-candidates-per-WI which should amortize the pre-hoist properly.
 *
 * Revision 1.2  2026/05/11 04:00:00  dlr
 * Tranche A2: inline md5_block (value-passed M[16]) + round-constant
 * pre-add hoist. Removes the noinline pointer-param boundary from
 * gpu_common.cl:540 for the BF-fast path. State + message words kept
 * in registers across the round chain; (M_j + K_round) sums hoisted
 * outside the round body so the round step is a single F/G/H/I plus
 * rotate+add. Slow path (gpu_md5_core.cl) unchanged.
 *
 * Revision 1.1  2026/05/11 03:45:43  dlr
 * Initial revision — A1 skeleton (verbatim of gpu_md5_core.cl 1.2).
 *
 * Phase 1.9 architecture: side-by-side fast path for the hottest BF case
 * (unsalted JOB_MD5, no-rules, append-mask npre==0 / napp in [1,8]).
 * Selected at gpu_template_resolve_kernel time when bf_fast_eligible is
 * set on the dispatch; everything else continues to use gpu_md5_core.cl
 * (the slow generic path). See project_bf_phase19_kernel_parity.md.
 *
 * TRANCHE A2 SCOPE (this revision):
 *
 *  (A2.1) Inline MD5 compressor. A new `md5_compress_inline` function
 *         is defined locally in this file with `static inline` and a
 *         signature that takes M[0..15] BY VALUE (16 uint args) instead
 *         of via a __private uint* parameter. The four call sites that
 *         previously invoked `md5_block(&h0, &h1, &h2, &h3, M)` now
 *         pass the message words directly: `md5_compress_inline(&h0,
 *         &h1, &h2, &h3, M[0], M[1], ..., M[15])`. The state pointers
 *         remain because they reference st->h[0..3]; the inliner sees
 *         through them at the call site since `template_state st` is
 *         a stack-local with no escapes. The gpu_common.cl `md5_block`
 *         is left intact (still used by other cores via the slow path).
 *
 *         Rationale: the gpu_common.cl `md5_block` declaration is
 *         `__attribute__((noinline)) void md5_block(uint *h0..h3, uint
 *         *M)` (gpu_common.cl:540). Three pathologies in one site:
 *           1. `noinline` forbids the compiler from inlining.
 *           2. Pointer params force state to round-trip through
 *              __private memory across the call boundary.
 *           3. `__private uint *M` often spills to local under Pascal
 *              register pressure.
 *         A0 measured 41,040 B priv_mem on every platform — that is
 *         the smoking gun. Value-passing M[] + inline gives the
 *         compiler unambiguous license to keep state + M in registers.
 *
 *  (A2.2) Round-constant pre-add hoist. Each MD5 step has the shape
 *         `a = b + rotate(a + F(b,c,d) + M_j + K_i, s)`. The two
 *         operands `M_j + K_i` are independent of (a,b,c,d) and can
 *         be summed once before the round body. The compiler is free
 *         to do this anyway when M is a constant; but when M was
 *         pointer-aliased and noinline prevented inlining, the
 *         compiler could not prove the values constant. With (A2.1)
 *         already inlining the function and passing M by value, we
 *         get this fold for free at -O3 — but we make it explicit
 *         here for: (a) clarity, (b) better PTX/AMDIL scheduling
 *         (independent adds become a tree the scheduler can issue
 *         in parallel), and (c) future tranches A3-A4 where the
 *         60 pre-adds for an entire candidate batch could be
 *         hoisted outside an inner per-candidate loop.
 *
 *         Implementation: the function body precomputes 64 local
 *         scalars `s0..s63` = `M_j + K_i` upfront (one per MD5 round
 *         step, in round order), then the 64 STEP invocations use
 *         the precomputed scalars. The K_i values are inlined as
 *         literals at each step (compiler folds at constant prop).
 *
 *  NOT IN A2: host-side mask pre-explosion (A3), bitselect / vendor
 *  intrinsics (A4). These are separate tranches.
 *
 * Mali compatibility: NOT a constraint per user decision 2026-05-10.
 * (Mali support is being retired in a separate task.)
 *
 * Source-order at compile time (gpu_opencl_template_md5_bf_compile):
 *
 *   [ gpu_common_str, gpu_md5_rules_str, gpu_md5_bf_str, gpu_template_str ]
 *
 * gpu_md5_rules_str provides apply_rule() for the rules walker; this
 * file provides the per-algorithm hooks; gpu_template_str ties them
 * into the template_phase0 kernel (same kernel name as the slow path —
 * the algorithm core is selected at source-array compose time, not at
 * runtime; the cache key disambiguates via defines_str "BF_FAST_MD5=1"
 * plus the source-text hash).
 *
 * Bytecast invariants (carry over from gpu_md5_core.cl 1.2):
 *   - apply_rule + buf in/out is identical to gpu_md5_rules.cl rev 1.28+.
 *   - md5 padding + multi-block logic is identical to md5_buf in
 *     gpu_md5_rules.cl (we do not call md5_buf directly to keep this
 *     file self-contained for future algorithm forks; the bytewise
 *     equivalence is the byte-exact gate).
 *   - template_finalize() builds the M[16] message words directly
 *     from the input bytes (mirrors md5_buf) and calls the inline
 *     compressor once per block. It does NOT route through
 *     template_transform() — that wrapper exists for the public API
 *     but is bypassed in the hot path because the byte-buffer
 *     round-trip cost would dominate.
 *
 * R1 mitigation (AMD ROCm comgr addrspace fragility): the template
 * passes a `__private uchar *buf` and `__private template_state *st`.
 * No addrspace-cast ternary pointers, no helper that takes a generic
 * pointer to private state. Pattern matches the safe single-private-
 * buffer model from gpu_md5_rules.cl r28.
 */

/* Per-algorithm geometry. The template uses these as structural
 * compile-time constants; the kernel-cache key (gpu_kernel_cache R3
 * fix) hashes the defines_str ("HASH_WORDS=4,HASH_BLOCK_BYTES=64,
 * BF_FAST_MD5=1") alongside the source set so distinct instantiations
 * get distinct cache entries even though source text would otherwise
 * be identical. */
#ifndef HASH_WORDS
#define HASH_WORDS 4
#endif
#ifndef HASH_BLOCK_BYTES
#define HASH_BLOCK_BYTES 64
#endif

/* Per-lane state struct: MD5 carries 4 uint32 chaining values.
 * SHA1 will carry 5; SHA256 8; SHA384/SHA512 8 ulong. The template
 * only reads/writes the digest words via template_finalize's output
 * + template_digest_compare; it does not introspect the struct. */
typedef struct {
    uint h[HASH_WORDS];
} template_state;

/* ============================================================== */
/* A2: local inline MD5 compressor with value-passed M[16] and
 * pre-hoisted (M_j + K_i) round-constant sums.
 *
 * The macros mirror gpu_common.cl FF/GG/HH/II semantically. Their
 * signatures take a SINGLE precomputed scalar `mk = M_j + K_i`
 * instead of separate (m, k) operands. Functionally identical:
 *   a += F(b,c,d) + M_j + K_i; a = b + rotate(a, s)
 * is the same as
 *   a += F(b,c,d) + mk;        a = b + rotate(a, s)
 * for `mk = M_j + K_i` (addition is associative on uint32). */
#define BF_FF(a,b,c,d,mk,s) { a += ((b&c)|(~b&d)) + (mk); a = b + rotate(a,(uint)(s)); }
#define BF_GG(a,b,c,d,mk,s) { a += ((d&b)|(~d&c)) + (mk); a = b + rotate(a,(uint)(s)); }
#define BF_HH(a,b,c,d,mk,s) { a += (b^c^d) + (mk);        a = b + rotate(a,(uint)(s)); }
#define BF_II(a,b,c,d,mk,s) { a += (c^(~d|b)) + (mk);     a = b + rotate(a,(uint)(s)); }

/* Single inline compressor used by every call site in this file.
 * All 16 message words are value-passed; the compiler is free to
 * keep them in registers across the round chain. The 60 round-
 * constant pre-adds are computed once upfront; the remaining
 * dependency chain is F/G/H/I -> rotate -> add (3 ops per step). */
static inline void md5_compress_inline(
    uint *h0, uint *h1, uint *h2, uint *h3,
    uint M0,  uint M1,  uint M2,  uint M3,
    uint M4,  uint M5,  uint M6,  uint M7,
    uint M8,  uint M9,  uint M10, uint M11,
    uint M12, uint M13, uint M14, uint M15)
{
    /* State copy into scalars — compiler keeps these in registers. */
    uint a = *h0, b = *h1, c = *h2, d = *h3;

    /* Round 1 (FF): 16 steps, message index = step index. */
    uint s00 = M0  + 0xd76aa478u;
    uint s01 = M1  + 0xe8c7b756u;
    uint s02 = M2  + 0x242070dbu;
    uint s03 = M3  + 0xc1bdceeeu;
    uint s04 = M4  + 0xf57c0fafu;
    uint s05 = M5  + 0x4787c62au;
    uint s06 = M6  + 0xa8304613u;
    uint s07 = M7  + 0xfd469501u;
    uint s08 = M8  + 0x698098d8u;
    uint s09 = M9  + 0x8b44f7afu;
    uint s10 = M10 + 0xffff5bb1u;
    uint s11 = M11 + 0x895cd7beu;
    uint s12 = M12 + 0x6b901122u;
    uint s13 = M13 + 0xfd987193u;
    uint s14 = M14 + 0xa679438eu;
    uint s15 = M15 + 0x49b40821u;

    /* Round 2 (GG): message index permutation = (1+5k) mod 16. */
    uint s16 = M1  + 0xf61e2562u;
    uint s17 = M6  + 0xc040b340u;
    uint s18 = M11 + 0x265e5a51u;
    uint s19 = M0  + 0xe9b6c7aau;
    uint s20 = M5  + 0xd62f105du;
    uint s21 = M10 + 0x02441453u;
    uint s22 = M15 + 0xd8a1e681u;
    uint s23 = M4  + 0xe7d3fbc8u;
    uint s24 = M9  + 0x21e1cde6u;
    uint s25 = M14 + 0xc33707d6u;
    uint s26 = M3  + 0xf4d50d87u;
    uint s27 = M8  + 0x455a14edu;
    uint s28 = M13 + 0xa9e3e905u;
    uint s29 = M2  + 0xfcefa3f8u;
    uint s30 = M7  + 0x676f02d9u;
    uint s31 = M12 + 0x8d2a4c8au;

    /* Round 3 (HH): message index permutation = (5+3k) mod 16. */
    uint s32 = M5  + 0xfffa3942u;
    uint s33 = M8  + 0x8771f681u;
    uint s34 = M11 + 0x6d9d6122u;
    uint s35 = M14 + 0xfde5380cu;
    uint s36 = M1  + 0xa4beea44u;
    uint s37 = M4  + 0x4bdecfa9u;
    uint s38 = M7  + 0xf6bb4b60u;
    uint s39 = M10 + 0xbebfbc70u;
    uint s40 = M13 + 0x289b7ec6u;
    uint s41 = M0  + 0xeaa127fau;
    uint s42 = M3  + 0xd4ef3085u;
    uint s43 = M6  + 0x04881d05u;
    uint s44 = M9  + 0xd9d4d039u;
    uint s45 = M12 + 0xe6db99e5u;
    uint s46 = M15 + 0x1fa27cf8u;
    uint s47 = M2  + 0xc4ac5665u;

    /* Round 4 (II): message index permutation = 7k mod 16. */
    uint s48 = M0  + 0xf4292244u;
    uint s49 = M7  + 0x432aff97u;
    uint s50 = M14 + 0xab9423a7u;
    uint s51 = M5  + 0xfc93a039u;
    uint s52 = M12 + 0x655b59c3u;
    uint s53 = M3  + 0x8f0ccc92u;
    uint s54 = M10 + 0xffeff47du;
    uint s55 = M1  + 0x85845dd1u;
    uint s56 = M8  + 0x6fa87e4fu;
    uint s57 = M15 + 0xfe2ce6e0u;
    uint s58 = M6  + 0xa3014314u;
    uint s59 = M13 + 0x4e0811a1u;
    uint s60 = M4  + 0xf7537e82u;
    uint s61 = M11 + 0xbd3af235u;
    uint s62 = M2  + 0x2ad7d2bbu;
    uint s63 = M9  + 0xeb86d391u;

    /* Round 1 — F(b,c,d) = (b&c)|(~b&d). Rotations: 7,12,17,22. */
    BF_FF(a,b,c,d, s00, 7);   BF_FF(d,a,b,c, s01,12);
    BF_FF(c,d,a,b, s02,17);   BF_FF(b,c,d,a, s03,22);
    BF_FF(a,b,c,d, s04, 7);   BF_FF(d,a,b,c, s05,12);
    BF_FF(c,d,a,b, s06,17);   BF_FF(b,c,d,a, s07,22);
    BF_FF(a,b,c,d, s08, 7);   BF_FF(d,a,b,c, s09,12);
    BF_FF(c,d,a,b, s10,17);   BF_FF(b,c,d,a, s11,22);
    BF_FF(a,b,c,d, s12, 7);   BF_FF(d,a,b,c, s13,12);
    BF_FF(c,d,a,b, s14,17);   BF_FF(b,c,d,a, s15,22);

    /* Round 2 — G(b,c,d) = (d&b)|(~d&c). Rotations: 5,9,14,20. */
    BF_GG(a,b,c,d, s16, 5);   BF_GG(d,a,b,c, s17, 9);
    BF_GG(c,d,a,b, s18,14);   BF_GG(b,c,d,a, s19,20);
    BF_GG(a,b,c,d, s20, 5);   BF_GG(d,a,b,c, s21, 9);
    BF_GG(c,d,a,b, s22,14);   BF_GG(b,c,d,a, s23,20);
    BF_GG(a,b,c,d, s24, 5);   BF_GG(d,a,b,c, s25, 9);
    BF_GG(c,d,a,b, s26,14);   BF_GG(b,c,d,a, s27,20);
    BF_GG(a,b,c,d, s28, 5);   BF_GG(d,a,b,c, s29, 9);
    BF_GG(c,d,a,b, s30,14);   BF_GG(b,c,d,a, s31,20);

    /* Round 3 — H(b,c,d) = b^c^d. Rotations: 4,11,16,23. */
    BF_HH(a,b,c,d, s32, 4);   BF_HH(d,a,b,c, s33,11);
    BF_HH(c,d,a,b, s34,16);   BF_HH(b,c,d,a, s35,23);
    BF_HH(a,b,c,d, s36, 4);   BF_HH(d,a,b,c, s37,11);
    BF_HH(c,d,a,b, s38,16);   BF_HH(b,c,d,a, s39,23);
    BF_HH(a,b,c,d, s40, 4);   BF_HH(d,a,b,c, s41,11);
    BF_HH(c,d,a,b, s42,16);   BF_HH(b,c,d,a, s43,23);
    BF_HH(a,b,c,d, s44, 4);   BF_HH(d,a,b,c, s45,11);
    BF_HH(c,d,a,b, s46,16);   BF_HH(b,c,d,a, s47,23);

    /* Round 4 — I(b,c,d) = c ^ (~d | b). Rotations: 6,10,15,21. */
    BF_II(a,b,c,d, s48, 6);   BF_II(d,a,b,c, s49,10);
    BF_II(c,d,a,b, s50,15);   BF_II(b,c,d,a, s51,21);
    BF_II(a,b,c,d, s52, 6);   BF_II(d,a,b,c, s53,10);
    BF_II(c,d,a,b, s54,15);   BF_II(b,c,d,a, s55,21);
    BF_II(a,b,c,d, s56, 6);   BF_II(d,a,b,c, s57,10);
    BF_II(c,d,a,b, s58,15);   BF_II(b,c,d,a, s59,21);
    BF_II(a,b,c,d, s60, 6);   BF_II(d,a,b,c, s61,10);
    BF_II(c,d,a,b, s62,15);   BF_II(b,c,d,a, s63,21);

    /* Add into chaining state. */
    *h0 += a; *h1 += b; *h2 += c; *h3 += d;
}

/* template_init: install algorithm IV into state. */
static inline void template_init(template_state *st) {
    st->h[0] = 0x67452301u;
    st->h[1] = 0xEFCDAB89u;
    st->h[2] = 0x98BADCFEu;
    st->h[3] = 0x10325476u;
}

/* template_transform: absorb one HASH_BLOCK_BYTES (64) byte block.
 * `block` points into the working buffer at the absorb position;
 * the caller must guarantee >=64 bytes are readable. */
static inline void template_transform(template_state *st,
                                      const uchar *block)
{
    uint M[16];
    /* MD5 reads message words little-endian. */
    for (int j = 0; j < 16; j++) {
        int b = j * 4;
        M[j] = (uint)block[b]
             | ((uint)block[b + 1] << 8)
             | ((uint)block[b + 2] << 16)
             | ((uint)block[b + 3] << 24);
    }
    md5_compress_inline(&st->h[0], &st->h[1], &st->h[2], &st->h[3],
                        M[0],  M[1],  M[2],  M[3],
                        M[4],  M[5],  M[6],  M[7],
                        M[8],  M[9],  M[10], M[11],
                        M[12], M[13], M[14], M[15]);
}

/* template_finalize: process the tail, append the 0x80 padding marker
 * + length-in-bits, and absorb. Caller passes the full buffer; we run
 * complete blocks then build the final padded block(s) here. After
 * return, st->h[0..HASH_WORDS-1] holds the final digest words.
 *
 * A2 (this rev): each call to md5_block has been replaced by
 * md5_compress_inline(...) with the 16 message words passed BY VALUE.
 * Byte-exact behavior with A1 (verified via the BF correctness gate
 * V1.9.A2.1). The compiler is now free to keep state + M in registers
 * across the round chain instead of round-tripping through __private
 * memory at the noinline call boundary that the prior gpu_common.cl
 * md5_block imposed. */
static inline void template_finalize(template_state *st,
                                     const uchar *data, int len)
{
    uint M[16];
    int pos = 0;

    /* Process complete 64-byte blocks. Build M[] directly from
     * bytes (no intermediate pad[]). Identical to md5_buf's main loop. */
    while (len - pos >= HASH_BLOCK_BYTES) {
        for (int j = 0; j < 16; j++) {
            int b = pos + j * 4;
            M[j] = (uint)data[b]
                 | ((uint)data[b + 1] << 8)
                 | ((uint)data[b + 2] << 16)
                 | ((uint)data[b + 3] << 24);
        }
        md5_compress_inline(&st->h[0], &st->h[1], &st->h[2], &st->h[3],
                            M[0],  M[1],  M[2],  M[3],
                            M[4],  M[5],  M[6],  M[7],
                            M[8],  M[9],  M[10], M[11],
                            M[12], M[13], M[14], M[15]);
        pos += HASH_BLOCK_BYTES;
    }

    /* Build final block(s): tail bytes + 0x80 marker + zeros + length.
     * Same direct-into-M[] approach as md5_buf — no intermediate uchar
     * pad[]. */
    int rem = len - pos;  /* 0..63 */

    /* Zero scratch. */
    for (int j = 0; j < 16; j++) M[j] = 0;

    /* Copy remaining tail bytes into M[] little-endian. */
    for (int i = 0; i < rem; i++) {
        uint v = (uint)data[pos + i];
        M[i >> 2] |= v << ((i & 3) * 8);
    }
    /* 0x80 padding marker. */
    M[rem >> 2] |= (uint)0x80 << ((rem & 3) * 8);

    if (rem < 56) {
        /* Length fits in this block. */
        M[14] = (uint)(len * 8);
        M[15] = 0;
        md5_compress_inline(&st->h[0], &st->h[1], &st->h[2], &st->h[3],
                            M[0],  M[1],  M[2],  M[3],
                            M[4],  M[5],  M[6],  M[7],
                            M[8],  M[9],  M[10], M[11],
                            M[12], M[13], M[14], M[15]);
    } else {
        /* Need one extra padding-only block to hold the length. */
        md5_compress_inline(&st->h[0], &st->h[1], &st->h[2], &st->h[3],
                            M[0],  M[1],  M[2],  M[3],
                            M[4],  M[5],  M[6],  M[7],
                            M[8],  M[9],  M[10], M[11],
                            M[12], M[13], M[14], M[15]);
        for (int j = 0; j < 16; j++) M[j] = 0;
        M[14] = (uint)(len * 8);
        M[15] = 0;
        md5_compress_inline(&st->h[0], &st->h[1], &st->h[2], &st->h[3],
                            M[0],  M[1],  M[2],  M[3],
                            M[4],  M[5],  M[6],  M[7],
                            M[8],  M[9],  M[10], M[11],
                            M[12], M[13], M[14], M[15]);
    }
}

/* template_iterate: -i loop step. Re-encode the digest as 32-byte
 * hex ASCII and rehash. Algorithm-specific because the
 * digest geometry differs. Mirrors gpu_md5_rules.cl rev 1.28's iter
 * loop exactly so byte-exact.
 *
 * B7.7a (2026-05-07): MD5UC variant via algo_mode. The CPU iter loop
 * (mdxfind.c MDstart) selects prmd5UC vs prmd5 based on
 * job->op == JOB_MD5UC at every iter step, so the inter-iter hex
 * encoding is uppercase for JOB_MD5UC. Iter=1 is byte-exact between
 * MD5 and MD5UC (the UC path only fires inter-iter). The
 * GPU_TEMPLATE_ITERATE_HAS_ALGO_MODE define (set ONLY in this core
 * and gpu_md5_core.cl) extends the signature to accept algo_mode;
 * gpu_template.cl uses an #ifdef at the call site to invoke the
 * right shape. Other cores' template_iterate stay at the legacy
 * `(st)` signature.
 *
 * algo_mode == 0: lowercase hex (JOB_MD5 — default).
 * algo_mode == 1: uppercase hex (JOB_MD5UC). */
#define GPU_TEMPLATE_ITERATE_HAS_ALGO_MODE 1
static inline void template_iterate(template_state *st, uint algo_mode)
{
    uint M[16];
    if (algo_mode == 1u) {
        md5_to_hex_uc(st->h[0], st->h[1], st->h[2], st->h[3], M);
    } else {
        md5_to_hex_lc(st->h[0], st->h[1], st->h[2], st->h[3], M);
    }
    M[8] = 0x80u;
    for (int j = 9; j < 14; j++) M[j] = 0u;
    M[14] = 32u * 8u;     /* 32 hex chars = 256 bits */
    M[15] = 0u;
    /* Reinitialize state to IV, then absorb the prepared M[]. */
    st->h[0] = 0x67452301u;
    st->h[1] = 0xEFCDAB89u;
    st->h[2] = 0x98BADCFEu;
    st->h[3] = 0x10325476u;
    md5_compress_inline(&st->h[0], &st->h[1], &st->h[2], &st->h[3],
                        M[0],  M[1],  M[2],  M[3],
                        M[4],  M[5],  M[6],  M[7],
                        M[8],  M[9],  M[10], M[11],
                        M[12], M[13], M[14], M[15]);
}

/* template_digest_compare: probe the compact table with the final
 * digest. On a hit, *out_idx is set to the matched target's hash_data
 * index (mirrors probe_compact_idx semantics). Return 1 on hit, 0 on
 * miss. Identical to gpu_md5_core.cl's probe wrapper. */
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

/* template_emit_hit: emit a hit into the global hits buffer. Wraps
 * EMIT_HIT_4 (MD5 = 4 uint32 digest words). The template body invokes
 * this through the algorithm core so HASH_WORDS-specific EMIT_HIT_N
 * macros stay encapsulated. Input order matches md5_rules_phase0's
 * EMIT_HIT_4 call site. */
#define template_emit_hit(hits, hit_count, max_hits, st, widx, sidx, iter) \
    EMIT_HIT_4((hits), (hit_count), (max_hits), \
               (widx), (sidx), (iter), \
               (st)->h[0], (st)->h[1], (st)->h[2], (st)->h[3])

/* B3 dedup+overflow-aware variant. Wraps EMIT_HIT_4_DEDUP_OR_OVERFLOW
 * to keep the hashes_shown[] dedup bit consistent across overflow
 * re-issues (see gpu_common.cl §B3 protocol notes). The macro takes
 * the dedup state (hashes_shown ptr, matched_idx, dedup_mask) PLUS
 * the overflow channel (ovr_set, ovr_gid, lane_gid). The kernel must
 * NOT pre-set the dedup bit; the macro does that atomically. */
#define template_emit_hit_or_overflow(hits, hit_count, max_hits, st, widx, sidx, iter, \
                                      hashes_shown, matched_idx, dedup_mask, \
                                      ovr_set, ovr_gid, lane_gid) \
    EMIT_HIT_4_DEDUP_OR_OVERFLOW((hits), (hit_count), (max_hits), \
               (widx), (sidx), (iter), \
               (st)->h[0], (st)->h[1], (st)->h[2], (st)->h[3], \
               (hashes_shown), (matched_idx), (dedup_mask), \
               (ovr_set), (ovr_gid), (lane_gid))
