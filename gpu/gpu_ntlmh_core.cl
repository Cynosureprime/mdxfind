/*
 * $Revision: 1.3 $
 * $Log: gpu_ntlmh_core.cl,v $
 * Revision 1.3  2026/09/17 03:37:43  dlr
 * NTLMH as a dual-variant type: variant 2 is the hashcat zero-extend UTF-16LE (primary, probed first), variant 1 the real iconv UTF-8 to UTF-16LE conversion, with have_alt suppressing the second compare when the candidate is all-ASCII and the two coincide. Uses the ALT_DIGEST hook added to gpu_template.cl. Paired with the Metal core.
 *
 * Revision 1.2  2026/05/11 05:22:01  dlr
 * Backfill $Revision/$Log RCS keyword stanzas per feedback_rcs_keyword_stanzas.md. Passive 4-line comment block at top of file; no behavioral change. Hand-authored .cl file was missing required stanzas (per memory: all hand-authored .c/.h/.cl/.frag/.tmpl/.py/.sh files MUST contain $Revision/$Log keyword stanzas). Build green on .205 against the post-add files; OpenCL compile strips comments so no kernel behavior change.
 *
 */
/* gpu_ntlmh_core.cl — NTLMH (NT password hash) algorithm extension
 * functions for the generic dispatch template (Memo B Phase B5
 * sub-batch 6, Tier B).
 *
 * NTLMH = MD4(UTF-16LE(password)), and mdxfind's CPU JOB_NTLMH computes
 * TWO digests per candidate (mdxfind.c:19254-19277), both of which count:
 *
 *   variant 1  MD4(iconv("UTF-16LE//IGNORE","UTF-8")(password))
 *   variant 2  MD4(zero-extend(password))  -- each byte b -> b, 0x00
 *
 * For an ALL-ASCII password the two are the same bytes and the same
 * digest.  They diverge as soon as any byte is >= 0x80.
 *
 * 2026-09-15 CORRECTNESS FIX.  This core used to compute variant 2 only,
 * documented as a "hashcat-compat gap, by design" (and the prior slab
 * validation note recorded it as an "expected 30-hit iconv gap on
 * non-ASCII").  The gap is not benign: on a non-ASCII candidate the GPU
 * claims the batch, the CPU never redoes the work, and a variant-1 target
 * is reported as not found -- a clean exit 0 and "None found, sorry!"
 * with nothing to distinguish it from an honest negative.  Measured on
 * fpga.local (GTX 1080, driver 535.183.01) with a known answer: CPU
 * found it, GPU found nothing.
 *
 * Both variants are now computed and both are probed.  st->h keeps
 * variant 2, so the primary probe and its emitted digest are BIT-
 * IDENTICAL to the pre-fix kernel; st->ha carries variant 1 and is
 * probed through the GPU_TEMPLATE_HAS_ALT_DIGEST hook in gpu_template.cl,
 * which is compiled out of all the other template programs.  The second
 * MD4 and the second probe are skipped entirely when the candidate is
 * all-ASCII, which is the case where they would be redundant, so the
 * ASCII-only workload pays one scan of the candidate and nothing else.
 *
 * CPU reference (mdxfind.c JOB_NTLMH at line 15174):
 *
 *   if (Unicode) {
 *     MD4(cur, len, md5buf.h);            // input is already UTF-16
 *     checkhash(&md5buf, 32, 1, job);
 *   } else {
 *     // Variant 1: iconv utf-8 -> UTF-16LE -> MD4
 *     // Variant 2: zero-extend UTF-16LE -> MD4   <-- GPU implements this
 *     to_utf16le(cur, wline, len);
 *     MD4(wline, len*2, md5buf.h);
 *     checkhash(&md5buf, 32, 1, job);
 *   }
 *
 * NTLMH has NO iter loop in the CPU implementation — Maxiter is ignored.
 * For template_iterate (called only when iter < Maxiter), we provide a
 * standard MD4 hex re-feed as defensive fallback; in practice the
 * chokepoint and host pipe disable Maxiter > 1 for this algorithm.
 *
 * Block layout, variant 2: per-byte LE expansion of input[i] to
 * (input[i], 0x00). For ASCII char c, MD4 word j (LE) covers UTF-16LE
 * bytes [4j..4j+3] = (input[2j], 0, input[2j+1], 0). Equivalent uint:
 *   M[j] = input[2j] | (input[2j+1] << 16)
 * Block layout, variant 1: UTF-16 code unit k occupies UTF-16LE bytes 2k
 * and 2k+1, so it lands in MD4 word (k mod 32) >> 1 at shift 16 * (k & 1);
 * a block closes every 32 code units.  See md4_over_utf16le_ntlmh.
 *
 * Length: 2 * len bytes UTF-16LE. Single MD4 block if 2*len <= 55 (so
 * len <= 27); two blocks if 2*len <= 119 (len <= 59). The chokepoint
 * limits len to a value within the rule-engine path's input bound.
 *
 * State width / byte order: MD4 carries 4 LE uint32 chaining values
 * (same as MD5; UNLIKE the SHA family).
 *
 * Cache key (R3): defines_str = "HASH_WORDS=4,HASH_BLOCK_BYTES=64" — same
 * as MD4 / MD5 / MD5RAW. Distinct cache entry guaranteed by source-text
 * hash difference.
 *
 * R1 mitigation: single private buffer; no addrspace-cast helpers.
 *
 * Source order at compile time:
 *   [ gpu_common_str, gpu_md5_rules_str, gpu_ntlmh_core_str, gpu_template_str ]
 */

#ifndef HASH_WORDS
#define HASH_WORDS 4
#endif
#ifndef HASH_BLOCK_BYTES
#define HASH_BLOCK_BYTES 64
#endif

/* Tell gpu_template.cl to compile its second-digest probe/emit block.
 * The macro is defined HERE, in the core, because the core source string is
 * concatenated ahead of the template source string in gpu_template_sources()
 * (gpu/gpu_opencl.c) -- so no host-side -D and no build_opts change is
 * needed, and every other template program's preprocessed text is
 * unaffected. */
#define GPU_TEMPLATE_HAS_ALT_DIGEST 1

typedef struct {
    uint h[HASH_WORDS];    /* variant 2: zero-extend (primary, probed first) */
    uint ha[HASH_WORDS];   /* variant 1: real iconv UTF-8 -> UTF-16LE        */
    uint have_alt;         /* 0 when the candidate is all-ASCII and ha == h  */
} template_state;

/* MD4 compression — inlined here (renamed to md4_compress_ntlmh to avoid
 * symbol clash if gpu_md4_core_str is ever included in the same compile
 * unit). Mirrors gpu_md4_core.cl's md4_compress byte-for-byte. */
static inline void md4_compress_ntlmh(uint *hx, uint *hy, uint *hz, uint *hw, uint *M) {
    uint a = *hx, b = *hy, c = *hz, d = *hw;
#define MD4_F(x,y,z) (((x)&(y)) | ((~(x))&(z)))
#define MD4_G(x,y,z) (((x)&(y)) | ((x)&(z)) | ((y)&(z)))
#define MD4_H(x,y,z) ((x)^(y)^(z))
#define MD4_R1(a,b,c,d,k,s) a = rotate(a + MD4_F(b,c,d) + M[k], (uint)(s))
#define MD4_R2(a,b,c,d,k,s) a = rotate(a + MD4_G(b,c,d) + M[k] + 0x5A827999u, (uint)(s))
#define MD4_R3(a,b,c,d,k,s) a = rotate(a + MD4_H(b,c,d) + M[k] + 0x6ED9EBA1u, (uint)(s))
    MD4_R1(a,b,c,d, 0, 3); MD4_R1(d,a,b,c, 1, 7); MD4_R1(c,d,a,b, 2,11); MD4_R1(b,c,d,a, 3,19);
    MD4_R1(a,b,c,d, 4, 3); MD4_R1(d,a,b,c, 5, 7); MD4_R1(c,d,a,b, 6,11); MD4_R1(b,c,d,a, 7,19);
    MD4_R1(a,b,c,d, 8, 3); MD4_R1(d,a,b,c, 9, 7); MD4_R1(c,d,a,b,10,11); MD4_R1(b,c,d,a,11,19);
    MD4_R1(a,b,c,d,12, 3); MD4_R1(d,a,b,c,13, 7); MD4_R1(c,d,a,b,14,11); MD4_R1(b,c,d,a,15,19);
    MD4_R2(a,b,c,d, 0, 3); MD4_R2(d,a,b,c, 4, 5); MD4_R2(c,d,a,b, 8, 9); MD4_R2(b,c,d,a,12,13);
    MD4_R2(a,b,c,d, 1, 3); MD4_R2(d,a,b,c, 5, 5); MD4_R2(c,d,a,b, 9, 9); MD4_R2(b,c,d,a,13,13);
    MD4_R2(a,b,c,d, 2, 3); MD4_R2(d,a,b,c, 6, 5); MD4_R2(c,d,a,b,10, 9); MD4_R2(b,c,d,a,14,13);
    MD4_R2(a,b,c,d, 3, 3); MD4_R2(d,a,b,c, 7, 5); MD4_R2(c,d,a,b,11, 9); MD4_R2(b,c,d,a,15,13);
    MD4_R3(a,b,c,d, 0, 3); MD4_R3(d,a,b,c, 8, 9); MD4_R3(c,d,a,b, 4,11); MD4_R3(b,c,d,a,12,15);
    MD4_R3(a,b,c,d, 2, 3); MD4_R3(d,a,b,c,10, 9); MD4_R3(c,d,a,b, 6,11); MD4_R3(b,c,d,a,14,15);
    MD4_R3(a,b,c,d, 1, 3); MD4_R3(d,a,b,c, 9, 9); MD4_R3(c,d,a,b, 5,11); MD4_R3(b,c,d,a,13,15);
    MD4_R3(a,b,c,d, 3, 3); MD4_R3(d,a,b,c,11, 9); MD4_R3(c,d,a,b, 7,11); MD4_R3(b,c,d,a,15,15);
#undef MD4_F
#undef MD4_G
#undef MD4_H
#undef MD4_R1
#undef MD4_R2
#undef MD4_R3
    *hx = a + *hx; *hy = b + *hy; *hz = c + *hz; *hw = d + *hw;
}

static inline void template_init(template_state *st) {
    st->h[0] = 0x67452301u;
    st->h[1] = 0xEFCDAB89u;
    st->h[2] = 0x98BADCFEu;
    st->h[3] = 0x10325476u;
    st->have_alt = 0u;
}

/* template_transform: stub for interface symmetry. NTLMH finalize handles
 * its own block packing inline (UTF-16LE expansion is interleaved with
 * the M[] build). */
static inline void template_transform(template_state *st,
                                      const uchar *block)
{
    uint M[16];
    for (int j = 0; j < 16; j++) {
        int b = j * 4;
        M[j] = (uint)block[b]
             | ((uint)block[b + 1] << 8)
             | ((uint)block[b + 2] << 16)
             | ((uint)block[b + 3] << 24);
    }
    md4_compress_ntlmh(&st->h[0], &st->h[1], &st->h[2], &st->h[3], M);
}

/* template_finalize: NTLMH = MD4(UTF-16LE-zero-extend(input)). Build M[]
 * directly from the input bytes interleaved with zeros (the UTF-16LE
 * high-byte placeholder for ASCII chars).
 *
 * For each MD4 word j of UTF-16LE bytes:
 *   word covers UTF-16LE bytes [4j..4j+3]:
 *     UTF-16LE byte 2k   = input[k]      (low byte of UTF-16 char k)
 *     UTF-16LE byte 2k+1 = 0             (high byte = 0 for zero-extend)
 *   So byte 4j   = input[2j]
 *      byte 4j+1 = 0
 *      byte 4j+2 = input[2j+1]
 *      byte 4j+3 = 0
 * MD4 reads M LE: M[j] = byte[4j] | byte[4j+1]<<8 | byte[4j+2]<<16 | byte[4j+3]<<24
 *               = input[2j] | (input[2j+1] << 16)
 *
 * The 0x80 padding marker lands at UTF-16-byte position 2*len; if 2*len
 * is even (always true since 2*len is always even), the marker is at
 * the LOW byte of M[len/2] (when len is even) or M[(len/2)] high byte (?).
 * Actually 2*len is the count of UTF-16 bytes consumed. The pad goes at
 * byte index 2*len in the UTF-16LE stream. byte_idx = 2*len:
 *   M_word = byte_idx >> 2 = len >> 1
 *   byte_in_word = byte_idx & 3 = (2*len) & 3 = 2 * (len & 1)
 *   ⇒ byte_in_word = 0 if len even, 2 if len odd
 *
 * Bit count in UTF-16LE = 2 * len * 8 = 16 * len.
 */
/* Variant 2 (zero-extend).  Body unchanged from the pre-2026-09-15 core
 * apart from taking the chaining state by pointer instead of reaching into
 * template_state, so it can be pointed at st->h or anywhere else.  h4 must
 * already hold the MD4 IV. */
static inline void md4_zeroext_ntlmh(uint *h4, const uchar *data, int len)
{
    /* Process complete UTF-16LE blocks. Each MD4 block (64 bytes) holds
     * 32 UTF-16LE chars = 32 input bytes. */
    uint M[16];
    int input_pos = 0;

    while (len - input_pos >= 32) {
        /* 32 input bytes -> 64 UTF-16LE bytes -> 16 MD4 M[] words. */
        for (int j = 0; j < 16; j++) {
            int k = input_pos + j * 2;
            M[j] = (uint)data[k] | ((uint)data[k + 1] << 16);
        }
        md4_compress_ntlmh(&h4[0], &h4[1], &h4[2], &h4[3], M);
        input_pos += 32;
    }

    /* Tail: remaining input bytes (rem in [0..31]) -> 2*rem UTF-16LE bytes. */
    int rem = len - input_pos;  /* 0..31 input bytes */
    int rem_utf16 = rem * 2;     /* 0..62 UTF-16LE bytes */

    /* Zero scratch. */
    for (int j = 0; j < 16; j++) M[j] = 0;

    /* Pack input bytes into M[] LE positions, interleaved with zeros. */
    for (int i = 0; i < rem; i++) {
        int byte_idx = i * 2;            /* low byte of UTF-16 char */
        int wi = byte_idx >> 2;
        int bi = byte_idx & 3;
        M[wi] |= ((uint)data[input_pos + i]) << (bi * 8);
        /* High byte (byte_idx + 1) is zero; no write needed. */
    }

    /* 0x80 pad marker at UTF-16 byte position rem_utf16 = 2*rem. */
    {
        int wi = rem_utf16 >> 2;
        int bi = rem_utf16 & 3;
        M[wi] |= ((uint)0x80u) << (bi * 8);
    }

    /* MD4 LE bit-count encoding: M[14] = low 32 bits, M[15] = high 32 bits.
     * Total UTF-16LE bytes = 2*len => bit count = 16 * len. */
    if (rem_utf16 < 56) {
        M[14] = (uint)((uint)len * 16u);
        M[15] = 0;
        md4_compress_ntlmh(&h4[0], &h4[1], &h4[2], &h4[3], M);
    } else {
        md4_compress_ntlmh(&h4[0], &h4[1], &h4[2], &h4[3], M);
        for (int j = 0; j < 16; j++) M[j] = 0;
        M[14] = (uint)((uint)len * 16u);
        M[15] = 0;
        md4_compress_ntlmh(&h4[0], &h4[1], &h4[2], &h4[3], M);
    }
}

/* Variant 1 (the real conversion).  MD4 of the UTF-16LE form of a UTF-8
 * input, assembled one 64-byte block at a time so no UTF-16 scratch buffer
 * is needed.  h4 must already hold the MD4 IV.
 *
 * The decoder mirrors iconv_open("UTF-16LE//IGNORE", "UTF-8")
 * (mdxfind.c:12593), which is the converter CPU JOB_NTLMH uses at
 * mdxfind.c:19262-19270, and matches glibc BYTE-FOR-BYTE including on
 * malformed input.  Proven, not asserted: a host transliteration of this
 * exact loop was run against glibc iconv on .205 over 18,802,400 inputs
 * (every 1-, 2- and 3-byte string exhaustively, a structured 4-byte
 * sweep, ASCII lengths 0..63, 2/3/4-byte characters at every block
 * alignment, and 1.5M pseudo-random strings) with ZERO differences.
 *
 * Why //IGNORE falls out of the code rather than being special-cased:
 * glibc skips the bytes it validated and resynchronises, and a
 * continuation byte 0x80..0xBF is never a valid lead, so dropping ONE
 * byte and re-entering the loop reaches the same resynchronisation point
 * with the same output.  A sequence truncated by end-of-input produces no
 * output in either implementation.
 *
 * Rejected as glibc rejects them: 0xC0/0xC1 and any other overlong form,
 * the surrogate range U+D800..U+DFFF, and anything above U+10FFFF.
 * Astral codepoints become a correct surrogate PAIR.
 *
 * KEPT IN STEP BY HAND with the identical function in
 * gpu_md4utf16_core.cl.  They are duplicated rather than shared through
 * gpu_common.cl on purpose: gpu_common.cl is concatenated into all 57
 * template programs, and putting a function there rebuilds and re-JITs
 * every one of them for the benefit of two.
 */
static inline void md4_over_utf16le_ntlmh(uint *h4, const uchar *data, int len,
                                          int *dropped)
{
    uint M[16];
    for (int j = 0; j < 16; j++) M[j] = 0u;

    uint nunits = 0u;   /* UTF-16 code units emitted so far */
    int  i = 0;
    *dropped = 0;

    while (i < len) {
        uint c = (uint)data[i];
        uint cp;
        int  adv;

        if (c < 0x80u) {
            cp = c; adv = 1;
        } else if (c >= 0xC2u && c <= 0xDFu && i + 1 < len &&
                   (data[i + 1] & 0xC0u) == 0x80u) {
            cp  = ((c & 0x1Fu) << 6) | (uint)(data[i + 1] & 0x3Fu);
            adv = 2;
        } else if (c >= 0xE0u && c <= 0xEFu && i + 2 < len &&
                   (data[i + 1] & 0xC0u) == 0x80u &&
                   (data[i + 2] & 0xC0u) == 0x80u) {
            cp  = ((c & 0x0Fu) << 12) |
                  ((uint)(data[i + 1] & 0x3Fu) << 6) |
                   (uint)(data[i + 2] & 0x3Fu);
            adv = 3;
            /* overlong, or a lone surrogate -- glibc rejects both */
            if (cp < 0x800u || (cp >= 0xD800u && cp <= 0xDFFFu)) { i += 1; *dropped = 1; continue; }
        } else if (c >= 0xF0u && c <= 0xF4u && i + 3 < len &&
                   (data[i + 1] & 0xC0u) == 0x80u &&
                   (data[i + 2] & 0xC0u) == 0x80u &&
                   (data[i + 3] & 0xC0u) == 0x80u) {
            cp  = ((c & 0x07u) << 18) |
                  ((uint)(data[i + 1] & 0x3Fu) << 12) |
                  ((uint)(data[i + 2] & 0x3Fu) << 6) |
                   (uint)(data[i + 3] & 0x3Fu);
            adv = 4;
            if (cp < 0x10000u || cp > 0x10FFFFu) { i += 1; *dropped = 1; continue; }
        } else {
            i += 1;      /* //IGNORE: drop one byte and resynchronise */
            *dropped = 1;
            continue;
        }
        i += adv;

        uint u0, u1; int nu;
        if (cp < 0x10000u) {
            u0 = cp; u1 = 0u; nu = 1;
        } else {
            uint v = cp - 0x10000u;
            u0 = 0xD800u + (v >> 10);
            u1 = 0xDC00u + (v & 0x3FFu);
            nu = 2;
        }
        for (int e = 0; e < nu; e++) {
            uint u = (e == 0) ? u0 : u1;
            M[(nunits & 31u) >> 1] |= u << ((nunits & 1u) * 16u);
            nunits++;
            if ((nunits & 31u) == 0u) {
                md4_compress_ntlmh(&h4[0], &h4[1], &h4[2], &h4[3], M);
                for (int j = 0; j < 16; j++) M[j] = 0u;
            }
        }
    }

    uint r = nunits & 31u;
    M[r >> 1] |= 0x80u << ((r & 1u) * 16u);
    if (r * 2u < 56u) {
        M[14] = nunits << 4;
        M[15] = nunits >> 28;
        md4_compress_ntlmh(&h4[0], &h4[1], &h4[2], &h4[3], M);
    } else {
        md4_compress_ntlmh(&h4[0], &h4[1], &h4[2], &h4[3], M);
        for (int j = 0; j < 16; j++) M[j] = 0u;
        M[14] = nunits << 4;
        M[15] = nunits >> 28;
        md4_compress_ntlmh(&h4[0], &h4[1], &h4[2], &h4[3], M);
    }
}

/* template_finalize: compute BOTH CPU variants.  st->h gets variant 2
 * (bit-identical to the pre-fix kernel); st->ha gets variant 1.
 *
 * TWO conditions gate the alt digest, and BOTH matter.
 *
 * 1. The candidate must carry a byte >= 0x80.  For an all-ASCII candidate
 *    the two variants are the same digest, so the second probe would be
 *    pure cost -- and this is the common case, so the ASCII-only workload
 *    pays one scan of the candidate and nothing else.
 *
 * 2. No byte may have been dropped by the decoder.  This one is NOT an
 *    optimisation, it is the CPU's semantics.  The CPU guard is
 *
 *        x = iconv(cd,&icin,&ic_inleft,&icout,&ic_outleft);
 *        if (x >= 0) { MD4(wline, MAXLINE - ic_outleft, ...); checkhash(); }
 *
 *    (mdxfind.c:19262-19270), and glibc's iconv with //IGNORE returns -1
 *    -- EILSEQ when it skipped a byte, EINVAL when the input ended
 *    mid-sequence -- EVEN THOUGH it still wrote the surviving output.
 *    Measured on .205: "a\x80b" gives ret=-1/EILSEQ with 4 output bytes,
 *    "a\xc3" gives ret=-1/EINVAL with 2.  So for ANY candidate that is
 *    not wholly valid UTF-8, CPU JOB_NTLMH computes the zero-extend
 *    variant ONLY, and a kernel that probed a variant-1 digest there
 *    would be probing a digest the CPU never computes.  Measured on the
 *    volume fixture: 168 of 539 words (910 of 1078 possible digests).
 *
 *    JOB_MD4UTF16 (e496) tests `ic_outleft == MAXLINE*2` instead and so
 *    DOES use a partial conversion -- the two types genuinely differ
 *    here, which is why the predicate is not shared between the cores. */
static inline void template_finalize(template_state *st,
                                     const uchar *data, int len)
{
    md4_zeroext_ntlmh(st->h, data, len);

    int non_ascii = 0;
    for (int i = 0; i < len; i++) {
        if (data[i] & 0x80u) { non_ascii = 1; break; }
    }
    st->have_alt = 0u;
    if (non_ascii) {
        int dropped = 0;
        st->ha[0] = 0x67452301u;
        st->ha[1] = 0xEFCDAB89u;
        st->ha[2] = 0x98BADCFEu;
        st->ha[3] = 0x10325476u;
        md4_over_utf16le_ntlmh(st->ha, data, len, &dropped);
        if (!dropped) st->have_alt = 1u;
    }
}

/* template_iterate: defensive fallback. NTLMH has no iter loop in CPU.
 * If the host ever passes Maxiter > 1 for NTLMH (it shouldn't), this
 * computes MD4(hex_lc(prev_hash)) — same shape as MD4 template_iterate.
 * Production path (Maxiter == 1) never calls this. */
static inline void template_iterate(template_state *st)
{
    uint M[16];
    /* The hex re-feed is pure ASCII, so there is no second variant past
     * iter 1.  Clear have_alt so the alt probe cannot re-read a digest
     * that belongs to the previous iteration. */
    st->have_alt = 0u;
    md5_to_hex_lc(st->h[0], st->h[1], st->h[2], st->h[3], M);
    M[8] = 0x80u;
    for (int j = 9; j < 14; j++) M[j] = 0u;
    M[14] = 32u * 8u;
    M[15] = 0u;
    st->h[0] = 0x67452301u;
    st->h[1] = 0xEFCDAB89u;
    st->h[2] = 0x98BADCFEu;
    st->h[3] = 0x10325476u;
    md4_compress_ntlmh(&st->h[0], &st->h[1], &st->h[2], &st->h[3], M);
}

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

/* GPU_TEMPLATE_HAS_ALT_DIGEST hooks.  template_digest_compare_alt returns
 * 0 without touching memory when the candidate was all-ASCII, so the
 * ASCII-only case issues exactly one probe, as before the fix. */
static inline int template_digest_compare_alt(
    const template_state *st,
    __global const uint *compact_fp, __global const uint *compact_idx,
    ulong compact_mask, uint max_probe, uint hash_data_count,
    __global const uchar *hash_data_buf, __global const ulong *hash_data_off,
    __global const ulong *overflow_keys, __global const uchar *overflow_hashes,
    __global const uint *overflow_offsets, uint overflow_count,
    uint *out_idx)
{
    if (!st->have_alt) return 0;
    return probe_compact_idx(
        st->ha[0], st->ha[1], st->ha[2], st->ha[3],
        compact_fp, compact_idx,
        compact_mask, max_probe, hash_data_count,
        hash_data_buf, hash_data_off,
        overflow_keys, overflow_hashes, overflow_offsets, overflow_count,
        out_idx);
}

#define template_emit_hit(hits, hit_count, max_hits, st, widx, sidx, iter) \
    EMIT_HIT_4((hits), (hit_count), (max_hits), \
               (widx), (sidx), (iter), \
               (st)->h[0], (st)->h[1], (st)->h[2], (st)->h[3])

#define template_emit_hit_or_overflow(hits, hit_count, max_hits, st, widx, sidx, iter, \
                                      hashes_shown, matched_idx, dedup_mask, \
                                      ovr_set, ovr_gid, lane_gid) \
    EMIT_HIT_4_DEDUP_OR_OVERFLOW((hits), (hit_count), (max_hits), \
               (widx), (sidx), (iter), \
               (st)->h[0], (st)->h[1], (st)->h[2], (st)->h[3], \
               (hashes_shown), (matched_idx), (dedup_mask), \
               (ovr_set), (ovr_gid), (lane_gid))

/* Same emit, carrying the ALT digest.  The host reads the digest straight
 * out of the hit record (gpujob_opencl.c, "Decode the candidate hash from
 * the hit entry") and passes it to checkhash without recomputing, so a
 * variant-1 hit reports the variant-1 digest and the plaintext that
 * produced it. */
#define template_emit_hit_alt_or_overflow(hits, hit_count, max_hits, st, widx, sidx, iter, \
                                      hashes_shown, matched_idx, dedup_mask, \
                                      ovr_set, ovr_gid, lane_gid) \
    EMIT_HIT_4_DEDUP_OR_OVERFLOW((hits), (hit_count), (max_hits), \
               (widx), (sidx), (iter), \
               (st)->ha[0], (st)->ha[1], (st)->ha[2], (st)->ha[3], \
               (hashes_shown), (matched_idx), (dedup_mask), \
               (ovr_set), (ovr_gid), (lane_gid))
