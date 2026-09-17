/*
 * classify_utf8 -- reference implementation.
 *
 * Tier 0  any byte >= 0x80?           in-bounds vector/word/byte cascade
 * Tier 1  an always-invalid lead?     folded into tier 2's first step
 * Tier 2  full structural validation  scalar, runs only when tier 0 fired
 *
 * SAFETY NOTE, and it is the whole reason tier 0 looks the way it does: a
 * reusable classifier may NOT read past p + n.  The obvious fast path -- one
 * unaligned 16-byte load masked to n -- over-reads by up to 15 bytes.  That is
 * fine inside mdxfind, whose readbuf has slack, and unacceptable in a library:
 * it can fault on a page boundary and it trips ASan/valgrind on every call.
 *
 * The aligned-downward trick findeol() uses does NOT solve this.  Aligning down
 * avoids page FAULTS, but it still reads bytes before p, which are outside the
 * object -- undefined behaviour, and AddressSanitizer flags it on the first
 * call.  That was measured, not reasoned about: the first version of this file
 * did exactly that and ASan aborted immediately on an exact-sized allocation.
 *
 * So tier 0 uses only IN-BOUNDS accesses: full 16-byte vectors while at least
 * 16 bytes remain, then 8-byte words, then single bytes.  A 14-byte password
 * therefore takes one 8-byte word plus six byte tests, with no vector setup and
 * no over-read.
 *
 * $Revision: $
 * $Log: $
 */
#include "classify_utf8.h"

#if !defined(U8C_FORCE_PORTABLE)
#if defined(__SSE2__) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 2)
#  include <emmintrin.h>
#  define U8C_SSE2 1
#endif
#if defined(__ARM_NEON) || defined(__ARM_NEON__) || defined(__aarch64__)
#  include <arm_neon.h>
#  define U8C_NEON 1
/* vmaxvq_u8, the horizontal "max across the vector", is an AArch64 instruction.
 * 32-bit ARM has NEON and defines __ARM_NEON, but has no such instruction, so
 * naming it there is a compile error rather than a slow path: the ARMv7 build
 * of this file never linked.  ARMv7 reaches the same answer with the pairwise
 * maximum, three steps to fold eight lanes into one. */
#  if defined(__aarch64__) || defined(_M_ARM64)
#    define u8c_maxv(v) vmaxvq_u8(v)
#  else
static inline unsigned char u8c_maxv(uint8x16_t v)
{
    uint8x8_t m = vpmax_u8(vget_low_u8(v), vget_high_u8(v));  /* 16 -> 8 */
    m = vpmax_u8(m, m);                                       /*  8 -> 4 */
    m = vpmax_u8(m, m);                                       /*  4 -> 2 */
    m = vpmax_u8(m, m);                                       /*  2 -> 1 */
    return vget_lane_u8(m, 0);
}
#  endif
#endif
#if defined(__ALTIVEC__) || defined(__VSX__)
#  include <altivec.h>
#  define U8C_VSX 1
#endif
#endif /* !U8C_FORCE_PORTABLE */

#include <stdint.h>
#include <string.h>

/* ---- tier 0: does any byte in p[0..n-1] have the high bit set? ---- */

#if defined(U8C_SSE2)
static int u8c_has_high(const unsigned char *p, size_t n)
{
    const uint64_t M = (uint64_t)0x8080808080808080ULL;
    size_t i = 0;
    if (n >= 16) {
        /* Complete vectors, then ONE overlapping vector anchored at the end.
         * Both are wholly inside [p, p+n): the tail load starts at p+n-16, so
         * it re-reads bytes already seen rather than reading past the object.
         * That is what keeps a 17-byte line at two loads instead of a vector
         * plus a byte loop. */
        for (; i + 16 <= n; i += 16) {
            __m128i v = _mm_loadu_si128((const __m128i *)(p + i));
            if (_mm_movemask_epi8(v)) return 1;   /* >= 0x80, one instruction */
        }
        if (i < n) {
            __m128i v = _mm_loadu_si128((const __m128i *)(p + n - 16));
            if (_mm_movemask_epi8(v)) return 1;
        }
        return 0;
    }
    if (n >= 8) {
        /* Two overlapping 8-byte words cover any 8..15 byte line, both in
         * bounds.  A 14-byte password costs exactly two loads and one test. */
        uint64_t w1, w2;
        __builtin_memcpy(&w1, p, 8);
        __builtin_memcpy(&w2, p + n - 8, 8);
        return ((w1 | w2) & M) ? 1 : 0;
    }
    /* n < 8: at most seven byte tests, and no load can be widened safely. */
    while (i < n) { if (p[i] & 0x80u) return 1; i++; }
    return 0;
}
#elif defined(U8C_NEON)
static int u8c_has_high(const unsigned char *p, size_t n)
{
    /* NEON has no pmovmskb.  vmaxvq_u8 answers "any high byte" in one insn
     * and gives no positions -- which is all tier 0 needs.  Complete blocks
     * only, same in-bounds rule as the SSE2 path; the byte tail is left scalar
     * lengths.  The tail MUST use the same overlapping-word form as the SSE2
     * path: an earlier version left it byte-at-a-time and measured 0.63x
     * against portable C-99 on both arm64 hosts, because a 13-byte password is
     * entirely tail. */
    const uint64_t M = (uint64_t)0x8080808080808080ULL;
    size_t i = 0;
    if (n >= 16) {
        for (; i + 16 <= n; i += 16) {
            uint8x16_t v = vld1q_u8(p + i);
            if (u8c_maxv(v) >= 0x80) return 1;
        }
        if (i < n) {                 /* overlapping final vector, in bounds */
            uint8x16_t v = vld1q_u8(p + n - 16);
            if (u8c_maxv(v) >= 0x80) return 1;
        }
        return 0;
    }
    if (n >= 8) {                    /* two overlapping words cover 8..15 */
        uint64_t w1, w2;
        memcpy(&w1, p, 8);
        memcpy(&w2, p + n - 8, 8);
        return ((w1 | w2) & M) ? 1 : 0;
    }
    while (i < n) { if (p[i] & 0x80u) return 1; i++; }
    return 0;
}
#elif defined(U8C_VSX)
static int u8c_has_high(const unsigned char *p, size_t n)
{
    /* POWER AltiVec/VSX.  vec_any_ge against a 0x80 splat is the direct
     * "any byte >= 0x80" predicate, which is all tier 0 needs.
     *
     * vec_vsx_ld is REQUIRED, not merely faster: plain vec_ld truncates its
     * address to a 16-byte boundary, which would read bytes before p -- the
     * same out-of-object access that ASan caught in the first x86 attempt.
     * Same overlapping in-bounds tail as every other rung. */
    const uint64_t M = (uint64_t)0x8080808080808080ULL;
    const __vector unsigned char thr = vec_splats((unsigned char)0x80);
    size_t i = 0;
    if (n >= 16) {
        for (; i + 16 <= n; i += 16) {
            __vector unsigned char v = vec_vsx_ld(0, p + i);
            if (vec_any_ge(v, thr)) return 1;
        }
        if (i < n) {
            __vector unsigned char v = vec_vsx_ld(0, p + n - 16);
            if (vec_any_ge(v, thr)) return 1;
        }
        return 0;
    }
    if (n >= 8) {
        uint64_t w1, w2;
        memcpy(&w1, p, 8);
        memcpy(&w2, p + n - 8, 8);
        return ((w1 | w2) & M) ? 1 : 0;
    }
    while (i < n) { if (p[i] & 0x80u) return 1; i++; }
    return 0;
}
#else
static int u8c_has_high(const unsigned char *p, size_t n)
{
    /* Pure C-99: word-at-a-time on the high-bit mask, with the same
     * overlapping in-bounds tail as the vector paths.  No intrinsics, no
     * platform predicates, no over-read. */
    const unsigned long long M = 0x8080808080808080ULL;
    size_t i = 0;
    if (n >= 8) {
        for (; i + 8 <= n; i += 8) {
            unsigned long long w;
            memcpy(&w, p + i, 8);
            if (w & M) return 1;
        }
        if (i < n) {                       /* overlapping final word */
            unsigned long long w;
            memcpy(&w, p + n - 8, 8);
            if (w & M) return 1;
        }
        return 0;
    }
    while (i < n) { if (p[i] & 0x80u) return 1; i++; }
    return 0;
}
#endif

/* ---- tier 2: full structural validation, RFC 3629 ---- */
static int u8c_valid(const unsigned char *p, size_t n)
{
    size_t i = 0;
    int nmulti = 0;
    while (i < n) {
        unsigned c = p[i];
        if (c < 0x80u) { i++; continue; }          /* ASCII, incl. embedded NUL */
        unsigned need;
        if (c >= 0xC2u && c <= 0xDFu)      need = 1;
        else if (c >= 0xE0u && c <= 0xEFu) need = 2;
        else if (c >= 0xF0u && c <= 0xF4u) need = 3;
        else return 0;      /* C0,C1 overlong lead; F5..FF out of range; 80..BF stray */
        if (i + need >= n) return 0;               /* truncated */
        for (unsigned k = 1; k <= need; k++)
            if ((p[i + k] & 0xC0u) != 0x80u) return 0;
        /* second-byte range restrictions: overlongs and surrogates */
        if (need == 2) {
            if (c == 0xE0u && p[i+1] < 0xA0u) return 0;   /* overlong 3-byte  */
            if (c == 0xEDu && p[i+1] > 0x9Fu) return 0;   /* UTF-16 surrogate */
        } else if (need == 3) {
            if (c == 0xF0u && p[i+1] < 0x90u) return 0;   /* overlong 4-byte  */
            if (c == 0xF4u && p[i+1] > 0x8Fu) return 0;   /* > U+10FFFF       */
        }
        nmulti++;
        i += need + 1;
    }
    return nmulti > 0;   /* "at least one valid multibyte sequence" */
}

unsigned classify_utf8(const void *pv, size_t n)
{
    const unsigned char *p = (const unsigned char *)pv;
    if (p == NULL || n == 0) return U8C_NONE;
    if (!u8c_has_high(p, n)) return U8C_ASCII;
    /* High byte present: the walk must COMPLETE -- a line carrying good
     * multibyte text and one corrupt byte is not UTF-8, so an early exit on
     * the first valid sequence would misclassify it. */
    return u8c_valid(p, n) ? (U8C_HI | U8C_UTF8) : U8C_HI;
}

/* ---- optional: padded variant, for callers that guarantee slack ----
 *
 * Operator 2026-09-13: inside mdxfind there is ample readable space BEFORE and
 * AFTER every candidate word by construction, so a pre-read or over-read cannot
 * fault there.  This variant exploits that: for any line of 16 bytes or fewer it
 * is ONE unaligned load plus one mask, with no tail cascade at all.
 *
 * PRECONDITION, and it is the caller's to keep: at least 16 bytes must be
 * readable at p + n.  Calling this on an exact-sized allocation is undefined
 * and AddressSanitizer will abort.  classify_utf8() is the safe entry point and
 * requires nothing.
 */
#if defined(U8C_SSE2) && !defined(U8C_FORCE_PORTABLE)
static int u8c_has_high_padded(const unsigned char *p, size_t n)
{
    size_t i = 0;
    if (n <= 16) {
        __m128i v = _mm_loadu_si128((const __m128i *)p);   /* reads slack */
        unsigned m = (unsigned)_mm_movemask_epi8(v);
        if (n < 16) m &= (1u << n) - 1u;
        return m ? 1 : 0;
    }
    for (; i + 16 <= n; i += 16) {
        __m128i v = _mm_loadu_si128((const __m128i *)(p + i));
        if (_mm_movemask_epi8(v)) return 1;
    }
    if (i < n) {
        __m128i v = _mm_loadu_si128((const __m128i *)(p + i));
        unsigned m = (unsigned)_mm_movemask_epi8(v) & ((1u << (n - i)) - 1u);
        if (m) return 1;
    }
    return 0;
}
unsigned classify_utf8_padded(const void *pv, size_t n)
{
    const unsigned char *p = (const unsigned char *)pv;
    if (p == NULL || n == 0) return U8C_NONE;
    if (!u8c_has_high_padded(p, n)) return U8C_ASCII;
    return u8c_valid(p, n) ? (U8C_HI | U8C_UTF8) : U8C_HI;
}
#elif defined(U8C_NEON) && !defined(U8C_FORCE_PORTABLE)
static int u8c_has_high_padded(const unsigned char *p, size_t n)
{
    const uint8_t idx[16] = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15};
    size_t i = 0;
    if (n <= 16) {
        uint8x16_t v = vld1q_u8(p);                       /* reads slack */
        uint8x16_t lim = vdupq_n_u8((uint8_t)(n > 15 ? 15 : n));
        uint8x16_t keep = vcltq_u8(vld1q_u8(idx), lim);   /* lanes < n */
        if (n >= 16) keep = vdupq_n_u8(0xff);
        return u8c_maxv(vandq_u8(v, keep)) >= 0x80;
    }
    for (; i + 16 <= n; i += 16)
        if (u8c_maxv(vld1q_u8(p + i)) >= 0x80) return 1;
    if (i < n) {
        uint8x16_t v = vld1q_u8(p + i);
        uint8x16_t lim = vdupq_n_u8((uint8_t)(n - i));
        uint8x16_t keep = vcltq_u8(vld1q_u8(idx), lim);
        if (u8c_maxv(vandq_u8(v, keep)) >= 0x80) return 1;
    }
    return 0;
}
unsigned classify_utf8_padded(const void *pv, size_t n)
{
    const unsigned char *p = (const unsigned char *)pv;
    if (p == NULL || n == 0) return U8C_NONE;
    if (!u8c_has_high_padded(p, n)) return U8C_ASCII;
    return u8c_valid(p, n) ? (U8C_HI | U8C_UTF8) : U8C_HI;
}
#else
unsigned classify_utf8_padded(const void *pv, size_t n)
{ return classify_utf8(pv, n); }   /* no padded win available on this target */
#endif
