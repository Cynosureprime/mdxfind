#include <stdio.h>

#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#include <limits.h>
#include <time.h>
#include <wctype.h>
#include <sys/types.h>
#include <errno.h>
#include <stdint.h>

#ifdef POWERPC
#define NOTINTEL 1
#if defined(__VSX__) || defined(__ALTIVEC__)
#include <altivec.h>
#endif
#endif
#ifdef ARM
#define NOTINTEL 1
#if ARM > 6
#include <arm_neon.h>
extern int Neon;
#endif
#endif
#ifdef SPARC
#define NOTINTEL 1
#endif
#ifdef AIX
#define NOTINTEL 1
#endif

#ifndef NOTINTEL
#include <emmintrin.h>
#include <xmmintrin.h>
#include <cpuid.h>
int IntelSSE;
int HasSSSE3;
#endif


#include "mdxfind.h"

extern unsigned char trhex[];
extern int b64_encode(char *clrstr, char *b64dst, int inlen);

static char *Version = "$Header: /Users/dlr/src/mdfind/RCS/ruleproc.c,v 1.29 2026/05/03 16:14:57 dlr Exp dlr $";
/*
 * $Log: ruleproc.c,v $
 * Revision 1.29  2026/05/03 16:14:57  dlr
 * h/H bug fix in CPU rule walker: applyrule's c=='H' test is now also c==RULE_OP_HEX_UPPER (0xc3) — the post-packrules opcode that the switch routes through. Previously H mode silently used lowercase Hextab because the post-switch test never matched. GPU walker (gpu_md5_rules.cl 1.25) now consistent with this fix.
 *
 * Revision 1.28  2026/05/03 02:46:40  dlr
 * Drop M/4/6/X/Q acceptance from gpu_rule_safe_phase0 — slice-4
 * path B classifier revert. Lockstep with gpu/gpu_md5_rules.cl 1.24:
 * the kernel no longer implements memory ops, so the classifier
 * rejects them and routes those rules to CPU. HashMob × rockyou
 * unaffected (no memory ops in HashMob per task #73 audit).
 *
 * Revision 1.27  2026/05/02 13:51:39  dlr
 * Mechanical: rename 104 bare-hex case labels in applyrule()
 * to named RULE_OP_* constants. Pure refactor -- no behavioral
 * change (each named constant is #defined to its hex value).
 * Re-applies user's 22 in-flight substitutions overwritten by
 * #79 (rev 1.26) and extends the rename to all 60 rule-op
 * opcodes. 0xff/0xfe cases unchanged (no #defined name).
 *
 * Revision 1.26  2026/05/02 13:39:16  dlr
 * Slices 2+3 classifier: gpu_rule_safe_phase0 admits rejection ops (_ < > ! / ( ) — RULE_OP_REJ_LEN_NE/GE/LE/HAS/NHAS/FIRST/LAST) and hex ops (H h — RULE_OP_HEX_UPPER/LOWER) per kernel rev 1.22. Q (RULE_OP_MEM_REJ) stays in default-reject — memory ops deferred to a later slice. Validator gate green on ioblade RTX 4070 Ti.
 *
 * Revision 1.25  2026/05/01 23:15:49  dlr
 * packrules emit-site remap to high-bit opcodes (0xc1..0xfd); gpu_rule_safe_phase0 classifier remap; format_op_for_error helper. Bytecode contract validated byte-exact via #65 harness on AMD gfx1201.
 *
 * Revision 1.24  2026/05/01 16:47:58  dlr
 * Add 0x80-0xfd opcode aliases to applyrule (FAST + slow paths) +
 * env-gated MDXFIND_RULE_VALIDATOR printf. Pure additive: production
 * paths byte-identical when env unset. Step 1+2 of unified GPU rule
 * walker plan; packrules + GPU walker remain on ASCII opcodes until
 * atomic swap in next session.
 *
 * Revision 1.23  2026/04/29 00:34:17  dlr
 * Two-pool jobg slot allocator: rules-engine and legacy chokepoint pools are now disjoint. struct jobg gains slot_kind, packed_buf_size, word_offset_entries fields; fill sites read sizes from the slot. New gpujob_get_free_rules entry point. gpujob_init hard-stops if rule count exceeds compile-time ceiling. Fixes SEGV from cross-path slot reuse with mismatched buffer caps. Bundle includes ruleproc '3' (TOGGLE_AT_SEP) op support and packrules NEED_BYTES truncation guards. Validated on mmt: 21,289 cracks for HashMob.100k.rule x rockyou.txt in 617s, exact match with pre-fix baseline.
 *
 * Revision 1.22  2026/04/28 01:39:27  dlr
 * GPU rule classifier: accept y and Y as 2-byte position-arg ops. Closes 99% of remaining HashMob coverage gap (these were the dominant CPU-only ops). Removed h/H/y/Y from the deferred list (only h/H remain as deferred for hex output).
 *
 * Revision 1.21  2026/04/28 01:26:28  dlr
 * GPU rule classifier: accept Phase 1 batch 4. New singles: d f q { } k K. New 2-byte (10): + - L R . , @ Z z p. New 3-byte (3): * x O. Updated NOT-supported list to call out the deferred classes (reject ops, memory ops, hex output, mdxfind-specific S, slowrule escape) so future maintainers know the line. With this addition the classifier accepts the bulk of single-input transforming ops in the rule language; the remaining gap is reject-ops + stateful memory + hex.
 *
 * Revision 1.20  2026/04/28 01:02:03  dlr
 * GPU rule classifier: accept Phase 1 batch 3 ops (s/i/o three-byte) plus the variable-length 0xff (multi-char append) and 0xfe (multi-char prepend). For 0xff/0xfe the classifier reads the N byte then skips N stored bytes, rejecting any rule with N==0 or a stored NUL (both indicate malformed bytecode). With this addition, classify_rules will route a much larger fraction of HashMob rule sets to GPU rather than CPU.
 *
 * Revision 1.19  2026/04/28 00:29:03  dlr
 * GPU rule classifier: accept Phase 1 batch 2 ops. Single-byte additions: [ ]. Two-byte additions: $ ^ ' D (each consumes one parameter byte; positiontranslate-encoded for ' and D, literal byte for $ and ^). Multi-char 0xff/0xfe append/prepend bytecodes still rejected — kernel doesn't handle them yet, those rules route to CPU.
 *
 * Revision 1.18  2026/04/28 00:14:54  dlr
 * GPU rule classifier: accept Phase 1 batch 1 ops (c, C, t, T, E, e) plus parameter-byte skip for two-byte T and e. Function name kept as gpu_rule_safe_phase0 for now to minimize call-site churn — semantically it now classifies any kernel-supported op set, not strictly Phase 0; rename deferred to a later cleanup pass.
 *
 * Revision 1.17  2026/04/27 21:53:26  dlr
 * GPU rule engine Phase 0 classifier (project_gpu_rule_engine_design.md rev 3, §6). Adds gpu_rule_safe_phase0() — single-stage op-based predicate accepting only Tier-1 ops {l, u, r, :, space, tab} in the post-packrules bytecode — and classify_rules() — partitions a rule array into full / gpu / cpu lists preserving original order. struct rule_lists declared in mdxfind.h alongside applyrule. Verified against synthetic mixed input (7 GPU + 5 CPU partition correct). Empirical note: HashMob.{100,1k,5k,100k}.rule classify as 0% GPU-eligible at Phase 0 — they all use ops beyond l/u/r — so Phase 0 validation will need a synthetic test fixture, not HashMob, to exercise the GPU path.
 *
 * Revision 1.16  2026/04/27 01:04:10  dlr
 * ruleproc.c: cross-platform SIMD for lfastcmp and the GPU-pack zeroing site. Adds ARM NEON (vceqq_u8 + 64-bit lane reduction in lfastcmp; vst1q_u8 in the zeroing block) and PowerPC VSX (vec_xl unaligned load + vec_all_eq in lfastcmp; vec_xst in zeroing) paths. Pure Altivec without VSX picks up vec_st in the zeroing block (slot is 16-byte aligned per the existing comment). Both paths converge to a byte-precise tail loop. The earlier rev (1.15) was Intel-only; this completes the platform matrix. Add altivec.h include for POWERPC builds. NEON intrinsic gating uses __ARM_NEON; VSX gating uses __VSX__; pure Altivec uses __ALTIVEC__. Local x86_64 build clean; full -z test matrix unchanged from rev 1.15.
 *
 * Revision 1.15  2026/04/27 00:51:21  dlr
 * ruleproc.c: fix lfastcmp over-read on short inputs. The previous unsigned-long implementation rounded the byte count UP to whole longs, comparing 8 bytes (on 64-bit) regardless of the requested length. For rule outputs shorter than 8 bytes, this read past the buffer end and returned false-positive 'different' even when the actual bytes matched — which prevented the auto-skip detection at line 2169 (return -2 when rule output equals input) from ever firing for short inputs. Replace with a byte-precise compare: SSE2 16-byte parallel for the bulk on Intel, plain byte loop for the tail and for non-x86 (NOTINTEL). Single call site (line 2169), no API change. Verified: 'u' rule on already-uppercase 'ABC' now correctly returns -2; auto-skip works for short rule outputs across CPU and SSE-batch JOB_MD5 paths.
 *
 * Revision 1.14  2026/04/22 22:02:53  dlr
 * struct rule_workspace in mdxfind.h with extern applyrule, remove duplicate declarations
 *
 * Revision 1.13  2026/04/22 18:23:53  dlr
 * applyrule workspace parameter, rule_error diagnostic with caret position
 *
 * Revision 1.12  2026/03/23 17:48:54  dlr
 * Runtime SSE2/SSSE3 dispatch for get32(), remove SSSE3 requirement. Add HasSSSE3 global, SHA1 C fallback for SSE2-only CPUs.
 *
 * Revision 1.11  2025/11/28 18:24:48  dlr
 * replace local memory copy with memcpy, will revisit.
 * Add control-b rule for base64 conversion
 *
 * Revision 1.10  2025/11/10 21:11:09  dlr
 * Fix potential "start of buffer" overwrite when processing multiple ^ rules
 * This does not affect most hashes, but can cause a problem with parallel
 * processing on  MD5 and others.
 *
 * Revision 1.9  2025/10/21 18:11:28  dlr
 * Fix dup line
 *
 * Revision 1.8  2025/10/21 16:19:00  dlr
 * Make v rule a tiny bit faster
 *
 * Revision 1.7  2025/10/16 14:30:10  dlr
 * change rule 9 to v, change order from char, count to count, char
 *
 * Revision 1.6  2025/10/10 19:42:48  dlr
 * Add 9, h and H rules
 *
 * Revision 1.5  2025/08/11 14:19:41  dlr
 * add parserules()
 *
 * Revision 1.4  2020/03/11 02:49:29  dlr
 * SSSE modifications complete.  About to start on fastrule
 *
 * Revision 1.3  2020/03/08 07:12:31  dlr
 * Improve rule processing
 *
 */

extern char *Rulepos;


#ifdef NOTDEF
void print128(char *s,__m128i v)
{
    unsigned char *z = (unsigned char *)&v;
    int x;
    fprintf(stderr,"%s",s);
    for (x=0; x < 16; x++)
       fprintf(stderr,"%02x",z[x]);
    fprintf(stderr,"\n");
}
#endif

/*
 * rule_error — report a rule parse error with context.
 * Shows the full rule line with a caret (^) pointing to the
 * position of the error, similar to a compiler diagnostic.
 *
 *   Rule: d ] ] ] 31e eE 31s
 *                            ^
 *   Error: Invalid replace in rule
 */
static void rule_error(const char *msg, const char *orule,
                       const char *rule)
{
	int pos = (int)(rule - orule);
	int len = (int)strlen(orule);
	int i;

	/* trim trailing newline for display */
	if (len > 0 && (orule[len-1] == '\n' || orule[len-1] == '\r'))
		len--;

	fprintf(stderr, "  Rule: %.*s\n", len, orule);
	fprintf(stderr, "        ");
	for (i = 0; i < pos && i < len; i++)
		fputc(' ', stderr);
	fprintf(stderr, "^\n");
	fprintf(stderr, "  Error: %s\n", msg);
}

static inline unsigned char positiontranslate(char c) {
   char *res;
   res = strchr(Rulepos,c);
   if (!res) {
       fprintf(stderr,"Invalid position %c in rules",c);
       return(1);
    }
   return(((res - Rulepos) & 0xff)+1);
}

#ifdef SPARC
#define NOUNALIGN 1
static inline int lfastcmp(void *dest,void *src,int len) {
  unsigned char *d = (unsigned char *) dest;
  unsigned char *s = (unsigned char *) src;
  while (len--) {
    if (*s++ != *d++)
      return(1);
  }
  return(0);
}
#else
#ifdef AIX
#define NOUNALIGN 1
static inline int lfastcmp(void *dest,void *src,int len) {
  unsigned char *d = (unsigned char *) dest;
  unsigned char *s = (unsigned char *) src;
  while (len--) {
    if (*s++ != *d++)
      return(1);
  }
  return(0);
}
#else

static inline int lfastcmp(void *dest, void *src, int len) {
  /* Byte-precise compare; returns 0 on match, 1 on difference.
   *
   * The previous implementation cast to unsigned long * and rounded the
   * count UP to whole longs. That over-read past the buffer end and
   * returned false-positive "different" for lengths not a multiple of
   * sizeof(unsigned long) — which broke the applyrule auto-skip
   * detection at the bottom of the function (line ~2169) for short
   * rule outputs (len < 8 on 64-bit always reported "different" even
   * when the actual bytes matched).
   *
   * Per-platform 16-byte parallel compare:
   *   - x86 SSE2: _mm_cmpeq_epi8 + _mm_movemask_epi8
   *   - ARM NEON (AArch64 / v7 with NEON): vceqq_u8 + 64-bit lane reduction
   *     (vminvq_u8 is AArch64-only, so the lane-OR reduction works on both)
   *   - PowerPC VSX (POWER8+): vec_xl unaligned load + vec_all_eq
   *   - Pure Altivec without VSX: falls through to byte loop (16-byte
   *     unaligned-load via vec_perm + vec_lvsl is correct but rarely
   *     buys throughput on the short strings this function sees).
   *
   * All paths converge to a byte-precise tail loop, which is also the
   * complete path for ARMv6, SPARC, and any platform without one of
   * the SIMD predicates above. */
  unsigned char *d = (unsigned char *) dest;
  unsigned char *s = (unsigned char *) src;
#if defined(__SSE2__) || (defined(_MSC_VER) && (defined(_M_X64) || defined(_M_AMD64)))
  while (len >= 16) {
    __m128i a = _mm_loadu_si128((const __m128i *) d);
    __m128i b = _mm_loadu_si128((const __m128i *) s);
    if (_mm_movemask_epi8(_mm_cmpeq_epi8(a, b)) != 0xFFFF)
      return (1);
    d += 16; s += 16; len -= 16;
  }
#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
  while (len >= 16) {
    uint8x16_t a   = vld1q_u8(d);
    uint8x16_t b   = vld1q_u8(s);
    /* vceqq_u8 yields 0xFF per byte where equal, 0x00 where different.
     * Invert and OR-reduce two 64-bit lanes — any non-zero bit means a
     * mismatch. Works on both AArch64 and ARMv7-with-NEON. */
    uint8x16_t neq = vmvnq_u8(vceqq_u8(a, b));
    uint64x2_t r   = vreinterpretq_u64_u8(neq);
    if (vgetq_lane_u64(r, 0) | vgetq_lane_u64(r, 1))
      return (1);
    d += 16; s += 16; len -= 16;
  }
#elif defined(__VSX__)
  while (len >= 16) {
    /* VSX vec_xl is the unaligned load (lxv on POWER8+).
     * vec_all_eq returns 1 when every element matches. */
    __vector unsigned char a = vec_xl(0, d);
    __vector unsigned char b = vec_xl(0, s);
    if (!vec_all_eq(a, b))
      return (1);
    d += 16; s += 16; len -= 16;
  }
#endif
  while (len--) {
    if (*s++ != *d++)
      return (1);
  }
  return (0);
}

#endif
#endif


void getcpuinfo() {
    int a,b,c,d;
#ifndef NOTINTEL
    IntelSSE = HasSSSE3 = a = b = c = d = 0;
    __cpuid(1,a,b,c,d);
    if (c & bit_SSE3)
	IntelSSE = 30;
    if (c & bit_SSSE3)
	HasSSSE3 = 1;
    if (c & bit_SSE4_1)
    	IntelSSE = 41;
    if (c & bit_SSE4_2)
    	IntelSSE = 42;
#endif
}

#define PARSEHEX \
	c1 = *t++;\
	if (c1 && c1 == '\\') {\
	    c1 = *t++;\
	    switch (c1) {\
		case '3':\
		case '2':\
		case '1':\
		case '0':\
		    if (*t != 'x' && *t != 'X') {\
			c1 -= '0';\
			while (*t >= '0' && *t <= '7') {\
			    c1 = c1 << 3;\
			    c1 |= (*t - '0');\
			    t++;\
			}\
			lbuf[x++] = c1;\
			break;\
		    }\
		    /* fall through */\
\
		case 'x':\
		case 'X':\
		    c1 = (char)trhex[*(unsigned char *)t++];\
		    c1 = (c1 << 4) + (char)trhex[*(unsigned char *)t++];\
		    lbuf[x++] = c1;\
		    break;\
		case '\\':\
		    lbuf[x++] = c1;\
		    break;\
\
		case '\000':\
		    lbuf[x++] = '\\';\
		    break;\
		default:\
		    t--;\
		    lbuf[x++] = '\\';\
		    break;\
	    }\
	} else { lbuf[x++]=c1;}

/* Truncation guards. Each multi-byte op needs N more bytes after `c`
 * (the op byte already consumed). If those bytes aren't there — i.e.,
 * the rule line ends mid-op — bail cleanly instead of reading past the
 * buffer's NUL terminator into adjacent memory.
 *
 * Pre-fix behavior: the bare *s++ reads kept advancing `s` past the
 * NUL into whatever was in memory after the rule line. When packrules
 * was called over a buffer of consecutive rules (mdxfind's typical
 * case), this overflowed into the NEXT rule's bytes, both producing
 * garbage bytecode for the truncated rule AND corrupting the next
 * rule's input slot. Diagnosed via gpu_rule_coverage 2026-04-28:
 * HashMob.100k.rule rule #2 (`d ] ] ] 31e eE 31s` — bare 's' at end,
 * needs 2 args) overflowed into rule #3 (`r o5~ o5t r`), turning
 * rule #3's compiled bytecode into the trailing 3 bytes of rule #2's
 * runaway processing. */
#define NEED_BYTES(n_) do { \
    for (int _i = 0; _i < (n_); _i++) { \
        if (s[_i] == 0) { \
            char _msg[64]; \
            snprintf(_msg, sizeof(_msg), \
                "Truncated rule: op '%c' needs %d more byte%s", \
                c, (n_), (n_) == 1 ? "" : "s"); \
            rule_error(_msg, line, s); \
            rulefail++; \
            goto pack_op_done; \
        } \
    } \
} while (0)

/* High-bit opcode mapping (rev 1.25+ — bytecode contract).
 * Range: 0xc1..0xfd, packed from 0xfd downward by HashMob.100k.rule freq.
 * 0xfe = multi-^ prepend, 0xff = multi-$ append (variable-length, 2+N bytes).
 * packrules() emits these high-bit forms; applyrule() consumes them.
 *
 *   0xfd = 'i'  insert at pos              (12.95% HashMob.100k.rule)
 *   0xfc = 'o'  overwrite at pos           ( 5.42%)
 *   0xfb = 'T'  toggle case at pos         ( 3.19%)
 *   0xfa = '+'  increment byte at pos      ( ~1.85%)
 *   0xf9 = '-'  decrement byte at pos      ( ~1.85%)
 *   0xf8 = '\'' truncate to len            ( 2.73%)
 *   0xf7 = ']'  drop last char             ( 2.65%)
 *   0xf6 = 's'  substitute X with Y        ( 2.57%)
 *   0xf5 = 'l'  lowercase                  whole-string class (combined ~7.13%)
 *   0xf4 = 'u'  uppercase
 *   0xf3 = 'c'  capitalize
 *   0xf2 = 'C'  inverse capitalize
 *   0xf1 = 'r'  reverse
 *   0xf0 = 't'  toggle case (whole)
 *   0xef = 'E'  title-case at space
 *   0xee = 'e'  title-case at sep X
 *   0xed = 'd'  duplicate
 *   0xec = 'f'  reflect (append reverse)
 *   0xeb = 'q'  duplicate each char
 *   0xea = '{'  rotate left
 *   0xe9 = '}'  rotate right
 *   0xe8 = 'k'  swap first two
 *   0xe7 = 'K'  swap last two
 *   0xe6 = '['  drop first char
 *   0xe5 = '$'  append byte
 *   0xe4 = '^'  prepend byte
 *   0xe3 = 'D'  delete at pos
 *   0xe2 = 'L'  bit-shift left at pos
 *   0xe1 = 'R'  bit-shift right at pos
 *   0xe0 = '.'  replace pos with next
 *   0xdf = ','  replace pos with prev
 *   0xde = '@'  purge char X
 *   0xdd = 'Z'  duplicate last N times
 *   0xdc = 'z'  duplicate first N times
 *   0xdb = 'p'  repeat input N+1 times
 *   0xda = 'y'  duplicate first N at start
 *   0xd9 = 'Y'  duplicate last N at end
 *   0xd8 = '*'  swap two positions
 *   0xd7 = 'x'  extract substring
 *   0xd6 = 'O'  omit substring
 *   0xd5 = '3'  toggle case after Nth sep
 *   0xd4 = ':'  no-op pass-through
 *   0xd3 = ' '  no-op pass-through (space)
 *   0xd2 = '\t' no-op pass-through (tab)
 *   0xd1 = 'M'  memorize current pass
 *   0xd0 = '4'  append memory
 *   0xcf = '6'  prepend memory
 *   0xce = 'Q'  reject if equals memory
 *   0xcd = 'X'  insert memory substr
 *   0xcc = '_'  reject if len != N
 *   0xcb = '<'  reject if len >= N
 *   0xca = '>'  reject if len <= N
 *   0xc9 = '!'  reject if contains X
 *   0xc8 = '/'  reject if not contains X
 *   0xc7 = '('  reject if first != X
 *   0xc6 = ')'  reject if last != X
 *   0xc5 = 'S'  special: a/A -> 0x0a
 *   0xc4 = '#'  early exit (success)
 *   0xc3 = 'H'  hex encode (uppercase)
 *   0xc2 = 'h'  hex encode (lowercase)
 *   0xc1 = 'v'  divide-and-insert
 */
#define RULE_OP_INSERT      0xfd
#define RULE_OP_OVERWRITE   0xfc
#define RULE_OP_TOGGLE_AT   0xfb
#define RULE_OP_INC         0xfa
#define RULE_OP_DEC         0xf9
#define RULE_OP_TRUNC       0xf8
#define RULE_OP_DROP_LAST   0xf7
#define RULE_OP_SUB         0xf6
#define RULE_OP_LOWER       0xf5
#define RULE_OP_UPPER       0xf4
#define RULE_OP_CAP         0xf3
#define RULE_OP_CAP_INV     0xf2
#define RULE_OP_REVERSE     0xf1
#define RULE_OP_TOGGLE      0xf0
#define RULE_OP_TITLE_SP    0xef
#define RULE_OP_TITLE_SEP   0xee
#define RULE_OP_DUP         0xed
#define RULE_OP_REFLECT     0xec
#define RULE_OP_DUP_EACH    0xeb
#define RULE_OP_ROT_L       0xea
#define RULE_OP_ROT_R       0xe9
#define RULE_OP_SWAP_FRONT  0xe8
#define RULE_OP_SWAP_BACK   0xe7
#define RULE_OP_DROP_FIRST  0xe6
#define RULE_OP_APPEND      0xe5
#define RULE_OP_PREPEND     0xe4
#define RULE_OP_DEL_AT      0xe3
#define RULE_OP_BIT_SHL     0xe2
#define RULE_OP_BIT_SHR     0xe1
#define RULE_OP_REPL_NEXT   0xe0
#define RULE_OP_REPL_PREV   0xdf
#define RULE_OP_PURGE       0xde
#define RULE_OP_DUP_LAST    0xdd
#define RULE_OP_DUP_FIRST   0xdc
#define RULE_OP_REPEAT      0xdb
#define RULE_OP_DUP_PREFIX  0xda
#define RULE_OP_DUP_SUFFIX  0xd9
#define RULE_OP_SWAP_AT     0xd8
#define RULE_OP_EXTRACT     0xd7
#define RULE_OP_OMIT        0xd6
#define RULE_OP_TOGGLE_SEP  0xd5
#define RULE_OP_NOOP        0xd4
#define RULE_OP_NOOP_SP     0xd3
#define RULE_OP_NOOP_TAB    0xd2
#define RULE_OP_MEM_STORE   0xd1
#define RULE_OP_MEM_APP     0xd0
#define RULE_OP_MEM_PRE     0xcf
#define RULE_OP_MEM_REJ     0xce
#define RULE_OP_MEM_INSERT  0xcd
#define RULE_OP_REJ_LEN_NE  0xcc
#define RULE_OP_REJ_LEN_GE  0xcb
#define RULE_OP_REJ_LEN_LE  0xca
#define RULE_OP_REJ_HAS     0xc9
#define RULE_OP_REJ_NHAS    0xc8
#define RULE_OP_REJ_FIRST   0xc7
#define RULE_OP_REJ_LAST    0xc6
#define RULE_OP_S_SPECIAL   0xc5
#define RULE_OP_HASH_EXIT   0xc4
#define RULE_OP_HEX_UPPER   0xc3
#define RULE_OP_HEX_LOWER   0xc2
#define RULE_OP_DIV_INSERT  0xc1

/* format_op_for_error: invert the high-bit opcode mapping for stderr.
 * After packrules() emits high-bit bytecode (rev 1.25+), error messages
 * that print the offending byte via "%c" would otherwise show 0x80+
 * mojibake. This helper returns the source-form ASCII character for any
 * known opcode, or the byte itself for non-opcode bytes (literals,
 * positions). Cosmetic only — does not affect bytecode semantics. */
static char format_op_for_error(unsigned char b) {
    switch (b) {
        case RULE_OP_INSERT:     return 'i';
        case RULE_OP_OVERWRITE:  return 'o';
        case RULE_OP_TOGGLE_AT:  return 'T';
        case RULE_OP_INC:        return '+';
        case RULE_OP_DEC:        return '-';
        case RULE_OP_TRUNC:      return '\'';
        case RULE_OP_DROP_LAST:  return ']';
        case RULE_OP_SUB:        return 's';
        case RULE_OP_LOWER:      return 'l';
        case RULE_OP_UPPER:      return 'u';
        case RULE_OP_CAP:        return 'c';
        case RULE_OP_CAP_INV:    return 'C';
        case RULE_OP_REVERSE:    return 'r';
        case RULE_OP_TOGGLE:     return 't';
        case RULE_OP_TITLE_SP:   return 'E';
        case RULE_OP_TITLE_SEP:  return 'e';
        case RULE_OP_DUP:        return 'd';
        case RULE_OP_REFLECT:    return 'f';
        case RULE_OP_DUP_EACH:   return 'q';
        case RULE_OP_ROT_L:      return '{';
        case RULE_OP_ROT_R:      return '}';
        case RULE_OP_SWAP_FRONT: return 'k';
        case RULE_OP_SWAP_BACK:  return 'K';
        case RULE_OP_DROP_FIRST: return '[';
        case RULE_OP_APPEND:     return '$';
        case RULE_OP_PREPEND:    return '^';
        case RULE_OP_DEL_AT:     return 'D';
        case RULE_OP_BIT_SHL:    return 'L';
        case RULE_OP_BIT_SHR:    return 'R';
        case RULE_OP_REPL_NEXT:  return '.';
        case RULE_OP_REPL_PREV:  return ',';
        case RULE_OP_PURGE:      return '@';
        case RULE_OP_DUP_LAST:   return 'Z';
        case RULE_OP_DUP_FIRST:  return 'z';
        case RULE_OP_REPEAT:     return 'p';
        case RULE_OP_DUP_PREFIX: return 'y';
        case RULE_OP_DUP_SUFFIX: return 'Y';
        case RULE_OP_SWAP_AT:    return '*';
        case RULE_OP_EXTRACT:    return 'x';
        case RULE_OP_OMIT:       return 'O';
        case RULE_OP_TOGGLE_SEP: return '3';
        case RULE_OP_NOOP:       return ':';
        case RULE_OP_NOOP_SP:    return ' ';
        case RULE_OP_NOOP_TAB:   return '\t';
        case RULE_OP_MEM_STORE:  return 'M';
        case RULE_OP_MEM_APP:    return '4';
        case RULE_OP_MEM_PRE:    return '6';
        case RULE_OP_MEM_REJ:    return 'Q';
        case RULE_OP_MEM_INSERT: return 'X';
        case RULE_OP_REJ_LEN_NE: return '_';
        case RULE_OP_REJ_LEN_GE: return '<';
        case RULE_OP_REJ_LEN_LE: return '>';
        case RULE_OP_REJ_HAS:    return '!';
        case RULE_OP_REJ_NHAS:   return '/';
        case RULE_OP_REJ_FIRST:  return '(';
        case RULE_OP_REJ_LAST:   return ')';
        case RULE_OP_S_SPECIAL:  return 'S';
        case RULE_OP_HASH_EXIT:  return '#';
        case RULE_OP_HEX_UPPER:  return 'H';
        case RULE_OP_HEX_LOWER:  return 'h';
        case RULE_OP_DIV_INSERT: return 'v';
        default:                 return (char)b;
    }
}

int packrules(char *line) {
  char *t, *s, *d, c, lbuf[10240], n,c1;
  int x, y, rulefail = 0;


  s = d = line;
  while ((c = *s++)) {
    if (c == '#' || c == '\r' || c == '\n')
      break;
    if (c <= 0 || c > 126) {rulefail=1; break;}
    switch (c) {

      case '[':
        if (s[0] == '^') {
	   /* `[^X` extension — needs s[0]='^' (already checked) plus
	    * s[1]=char. If s[1] is NUL the rule was truncated. */
	   NEED_BYTES(2);
	   *d++ = RULE_OP_OVERWRITE;
	   *d++ = 1;
	   *d++ = s[1];
	   s += 2;
	} else {
	   *d++ = RULE_OP_DROP_FIRST;
	}
	break;
      case '$':
      case '^':
        /* Both '$' and '^' need at least 1 byte after the op (the
         * char to append/prepend). PARSEHEX is too permissive — if the
         * next byte is NUL it silently writes NUL into lbuf and counts
         * it as a "char", advancing t past the NUL into adjacent
         * memory. Pre-check explicitly. */
        NEED_BYTES(1);
        t = s;
        x = 0;
	PARSEHEX;
        while (*t) {
          if (*t == c && t[1] && x < 254) {
	    t++;
	    PARSEHEX;
            continue;
          }
          if (*t == ' ' || *t == '\t' || *t == ':') {
            t++;
            continue;
          }
          break;
        }
        switch (x) {
          case 0:
	    *d++ = (c == '$') ? RULE_OP_APPEND : RULE_OP_PREPEND;
	    *d++ = c1;
            break;

          case 1:
            *d++ = (c == '$') ? RULE_OP_APPEND : RULE_OP_PREPEND;
            *d++ = lbuf[0];
            s = t;
            break;

          default:
            switch (c) {
              case '^':
                *d++ = 0xfe;
		*d++ = (unsigned char) x;
                for (y = x - 1; y >= 0; y--)
                  *d++ = lbuf[y];
                break;
              case '$':
                *d++ = 0xff;
		*d++ = (unsigned char) x;
                for (y = 0; y < x; y++)
                  *d++ = lbuf[y];
                break;
              default:
                fprintf(stderr, "impossible\n");
                exit(1);
                break;
            }
            s = t;
            break;
        }
        break;


      case '@':
      case 'e':
      case '!':
        NEED_BYTES(1);
        switch (c) {
          case '@': *d++ = RULE_OP_PURGE;        break;
          case 'e': *d++ = RULE_OP_TITLE_SEP;    break;
          case '!': *d++ = RULE_OP_REJ_HAS;      break;
        }
        *d++ = *s++;
        break;

      case 'D':
      case '\'':
      case 'Z':
      case 'z':
      case '/':
      case '(':
      case ')':
      case '_':
      case '<':
      case '>':
      case '+':
      case '-':
      case '.':
      case ',':
      case 'T':
      case 'L':
      case 'R':
      case 'y':
      case 'Y':
      case 'p':
	NEED_BYTES(1);
	switch (c) {
	  case 'D':  *d++ = RULE_OP_DEL_AT;     break;
	  case '\'': *d++ = RULE_OP_TRUNC;      break;
	  case 'Z':  *d++ = RULE_OP_DUP_LAST;   break;
	  case 'z':  *d++ = RULE_OP_DUP_FIRST;  break;
	  case '/':  *d++ = RULE_OP_REJ_NHAS;   break;
	  case '(':  *d++ = RULE_OP_REJ_FIRST;  break;
	  case ')':  *d++ = RULE_OP_REJ_LAST;   break;
	  case '_':  *d++ = RULE_OP_REJ_LEN_NE; break;
	  case '<':  *d++ = RULE_OP_REJ_LEN_GE; break;
	  case '>':  *d++ = RULE_OP_REJ_LEN_LE; break;
	  case '+':  *d++ = RULE_OP_INC;        break;
	  case '-':  *d++ = RULE_OP_DEC;        break;
	  case '.':  *d++ = RULE_OP_REPL_NEXT;  break;
	  case ',':  *d++ = RULE_OP_REPL_PREV;  break;
	  case 'T':  *d++ = RULE_OP_TOGGLE_AT;  break;
	  case 'L':  *d++ = RULE_OP_BIT_SHL;    break;
	  case 'R':  *d++ = RULE_OP_BIT_SHR;    break;
	  case 'y':  *d++ = RULE_OP_DUP_PREFIX; break;
	  case 'Y':  *d++ = RULE_OP_DUP_SUFFIX; break;
	  case 'p':  *d++ = RULE_OP_REPEAT;     break;
	}
	n = *s++;
        if ((n < '0') || (n > '9' && n < 'A') || (n > 'Z' && n < 'a') ||
	   (n > 'z') ) {
	  fprintf(stderr, "Invalid position %c for %c\n", n, c);
          rulefail++;
        }
        *d++ = positiontranslate(n);
	break;


      case 'v':
        NEED_BYTES(2);
        *d++ = RULE_OP_DIV_INSERT;
	n = *s++;
        if ((n < '1') || (n > '9' && n < 'A') || (n > 'Z' && n < 'a') ||
	   (n > 'z') ) {
	  fprintf(stderr, "Invalid position %c for %c\n", n, c);
          rulefail++;
        }
        *d++ = positiontranslate(n)-1;
        *d++ = *s++;
	break;

      case 's':
        NEED_BYTES(2);
        *d++ = RULE_OP_SUB;
        *d++ = *s++;
        *d++ = *s++;
	break;

      case '=':
      case '%':
        /* No applyrule case exists for '=' / '%' — they pack as legacy
         * ASCII bytes and applyrule's default-case silently skips. No
         * RULE_OP_* mapping needed; preserved verbatim. */
        NEED_BYTES(2);
        *d++ = c;
	n = *s++;
        if ((n < '0') || (n > '9' && n < 'A') || (n > 'Z' && n < 'a') ||
	   (n > 'z') ) {
          { char _msg[64]; snprintf(_msg, sizeof(_msg),
            "Invalid position '%c' for command '%c'", n, c);
            rule_error(_msg, line, s - 1); }
          rulefail++;
        }
        *d++ = positiontranslate(n);
        *d++ = *s++;
        break;

      case 'i':
      case 'o':
        NEED_BYTES(2);
        *d++ = (c == 'i') ? RULE_OP_INSERT : RULE_OP_OVERWRITE;
        n = *s++;
        if ((n < '0') || (n > '9' && n < 'A') || (n > 'Z' && n < 'a') ||
	   (n > 'z') ) {
          { char _msg[64]; snprintf(_msg, sizeof(_msg),
            "Invalid position '%c' for command '%c'", n, c);
            rule_error(_msg, line, s - 1); }
          rulefail++;
        }
        *d++ = positiontranslate(n);
        *d++ = *s++;
        break;

      case '3':
        /* Hashcat RULE_OP_MANGLE_TOGGLE_AT_SEP: `3 N C`
         * Walk the string; count occurrences of separator C; toggle
         * the case of the first alphabetic byte AFTER the Nth
         * occurrence. 3-byte op (op + position + literal-separator),
         * same wire shape as 'i'/'o'. */
        NEED_BYTES(2);
        *d++ = RULE_OP_TOGGLE_SEP;
        n = *s++;
        if ((n < '0') || (n > '9' && n < 'A') || (n > 'Z' && n < 'a') ||
	   (n > 'z') ) {
          { char _msg[64]; snprintf(_msg, sizeof(_msg),
            "Invalid position '%c' for command '%c'", n, c);
            rule_error(_msg, line, s - 1); }
          rulefail++;
        }
        *d++ = positiontranslate(n);
        *d++ = *s++;
        break;

      case 'O':
      case 'x':
      case '*':
        NEED_BYTES(2);
        switch (c) {
          case 'O': *d++ = RULE_OP_OMIT;    break;
          case 'x': *d++ = RULE_OP_EXTRACT; break;
          case '*': *d++ = RULE_OP_SWAP_AT; break;
        }
        n = *s++;
        if ((n < '0') || (n > '9' && n < 'A') || (n > 'Z' && n < 'a') ||
	   (n > 'z') ) {
          { char _msg[64]; snprintf(_msg, sizeof(_msg),
            "Invalid position '%c' for command '%c'", n, c);
            rule_error(_msg, line, s - 1); }
          rulefail++;
        }
        *d++ = positiontranslate(n);
        n = *s++;
        if ((n < '0') || (n > '9' && n < 'A') || (n > 'Z' && n < 'a') ||
	   (n > 'z') ) {
          { char _msg[64]; snprintf(_msg, sizeof(_msg),
            "Invalid position '%c' for command '%c'", n, c);
            rule_error(_msg, line, s - 1); }
          rulefail++;
        }
        *d++ = positiontranslate(n);
	break;

      case 'X':
        /* RULE_OP_MEM_INSERT in applyrule (case 0xcd / 'X'): 3-arg
         * memory-substring insert. Emit high-bit opcode. */
        NEED_BYTES(3);
        *d++ = RULE_OP_MEM_INSERT;
        n = *s++;
        if ((n < '0') || (n > '9' && n < 'A') || (n > 'Z' && n < 'a') ||
	   (n > 'z') ) {
          { char _msg[64]; snprintf(_msg, sizeof(_msg),
            "Invalid position '%c' for command '%c'", n, c);
            rule_error(_msg, line, s - 1); }
          rulefail++;
        }
        *d++ = positiontranslate(n);
        n = *s++;
        if ((n < '0') || (n > '9' && n < 'A') || (n > 'Z' && n < 'a') ||
	   (n > 'z') ) {
          { char _msg[64]; snprintf(_msg, sizeof(_msg),
            "Invalid position '%c' for command '%c'", n, c);
            rule_error(_msg, line, s - 1); }
          rulefail++;
        }
        *d++ = positiontranslate(n);
        n = *s++;
        if ((n < '0') || (n > '9' && n < 'A') || (n > 'Z' && n < 'a') ||
	   (n > 'z') ) {
          { char _msg[64]; snprintf(_msg, sizeof(_msg),
            "Invalid position '%c' for command '%c'", n, c);
            rule_error(_msg, line, s - 1); }
          rulefail++;
        }
        *d++ = positiontranslate(n);
	break;

      case ' ':
      case '\t':
      case ':':
        break;


      default:
        /* Map remaining single-byte ASCII opcodes to their high-bit
         * counterparts so applyrule sees only 0x80+ bytes after rev 1.25
         * packrules. Includes whole-string ops (l u r c C t E d f q { } k K)
         * and the isolated control bytes (M 4 6 Q S # H h). Anything else
         * passes through verbatim (preserves legacy default-case no-op). */
        switch (c) {
          case 'l': *d++ = RULE_OP_LOWER;       break;
          case 'u': *d++ = RULE_OP_UPPER;       break;
          case 'c': *d++ = RULE_OP_CAP;         break;
          case 'C': *d++ = RULE_OP_CAP_INV;     break;
          case 't': *d++ = RULE_OP_TOGGLE;      break;
          case 'r': *d++ = RULE_OP_REVERSE;     break;
          case 'E': *d++ = RULE_OP_TITLE_SP;    break;
          case 'd': *d++ = RULE_OP_DUP;         break;
          case 'f': *d++ = RULE_OP_REFLECT;     break;
          case 'q': *d++ = RULE_OP_DUP_EACH;    break;
          case '{': *d++ = RULE_OP_ROT_L;       break;
          case '}': *d++ = RULE_OP_ROT_R;       break;
          case 'k': *d++ = RULE_OP_SWAP_FRONT;  break;
          case 'K': *d++ = RULE_OP_SWAP_BACK;   break;
          case ']': *d++ = RULE_OP_DROP_LAST;   break;
          case 'M': *d++ = RULE_OP_MEM_STORE;   break;
          case '4': *d++ = RULE_OP_MEM_APP;     break;
          case '6': *d++ = RULE_OP_MEM_PRE;     break;
          case 'Q': *d++ = RULE_OP_MEM_REJ;     break;
          case 'S': *d++ = RULE_OP_S_SPECIAL;   break;
          case '#': *d++ = RULE_OP_HASH_EXIT;   break;
          case 'H': *d++ = RULE_OP_HEX_UPPER;   break;
          case 'h': *d++ = RULE_OP_HEX_LOWER;   break;
          default:  *d++ = c;                   break;
        }
        break;
    }
pack_op_done:
    if (rulefail) break;   /* halt the outer rule walk on any error so
                            * we don't keep reading past truncated input */
  }
  *d++ = 0;
  return (rulefail);
}
#undef NEED_BYTES

char * parserules(char *line) {
  char *t, *s, c, lbuf[10240], n,c1;
  char *lastvalid;
  int x, y, rulefail = 0;


  lastvalid = s = line;
  while ((c = *s)) {
    if (c == '#' || c == '\r' || c == '\n')
      break;
    if (c <= 0 || c > 126) {rulefail=1; break;}
    if (c == ' ' || c == '\t' || c == ':') {
        s++;
        continue;
    }
    lastvalid = s++;
    switch (c) {

      case '[':
        if (s[0] == '^') {
	   s += 2;
	} else {
	}
	break;
      case '$':
      case '^':
        t = s;
        x = 0;
	PARSEHEX;
        while (*t) {
          if (*t == c && t[1] && x < 254) {
 	    lastvalid = t;
	    t++;
	    PARSEHEX;
            continue;
          }
          if (*t == ' ' || *t == '\t' || *t == ':') {
            t++;
            continue;
          }
          break;
        }
        switch (x) {
          case 0:
            break;

          case 1:
            s = t;
            break;

          default:
            switch (c) {
              case '^':
                break;
              case '$':
                break;
              default:
                fprintf(stderr, "impossible\n");
                exit(1);
                break;
            }
            s = t;
            break;
        }
        break;


      case '@':
      case 'e':
      case '!':
        s++;
        break;

      case 'D':
      case '\'':
      case 'Z':
      case 'z':
      case '/':
      case '(':
      case ')':
      case '_':
      case '<':
      case '>':
      case '+':
      case '-':
      case '.':
      case ',':
      case 'T':
      case 'L':
      case 'R':
      case 'y':
      case 'Y':
      case 'p':
	n = *s++;
        if ((n < '0') || (n > '9' && n < 'A') || (n > 'Z' && n < 'a') ||
	   (n > 'z') ) {
          { char _msg[64]; snprintf(_msg, sizeof(_msg),
            "Invalid position '%c' for command '%c'", n, c);
            rule_error(_msg, line, s - 1); }
          rulefail++;
        }
	break;


      case 'v':
        n = *s++;
        if ((n < '1') || (n > '9' && n < 'A') || (n > 'Z' && n < 'a') ||
	   (n > 'z') ) {
	  fprintf(stderr, "Invalid position %c for %c\n", n, c);
          rulefail++;
        }
	n = *s++;
	break;

      case 's':
	break;

      case '=':
      case '%':
	n = *s++;
        if ((n < '0') || (n > '9' && n < 'A') || (n > 'Z' && n < 'a') ||
	   (n > 'z') ) {
          { char _msg[64]; snprintf(_msg, sizeof(_msg),
            "Invalid position '%c' for command '%c'", n, c);
            rule_error(_msg, line, s - 1); }
          rulefail++;
        }
        s++;
        break;

      case 'i':
      case 'o':
        n = *s++;
        if ((n < '0') || (n > '9' && n < 'A') || (n > 'Z' && n < 'a') ||
	   (n > 'z') ) {
          { char _msg[64]; snprintf(_msg, sizeof(_msg),
            "Invalid position '%c' for command '%c'", n, c);
            rule_error(_msg, line, s - 1); }
          rulefail++;
        }
        s++;
        break;

      case '3':
        /* Hashcat TOGGLE_AT_SEP: `3 N C` — same shape as 'i'/'o'. */
        n = *s++;
        if ((n < '0') || (n > '9' && n < 'A') || (n > 'Z' && n < 'a') ||
	   (n > 'z') ) {
          { char _msg[64]; snprintf(_msg, sizeof(_msg),
            "Invalid position '%c' for command '%c'", n, c);
            rule_error(_msg, line, s - 1); }
          rulefail++;
        }
        s++;
        break;

      case 'O':
      case 'x':
      case '*':
        n = *s++;
        if ((n < '0') || (n > '9' && n < 'A') || (n > 'Z' && n < 'a') ||
	   (n > 'z') ) {
          { char _msg[64]; snprintf(_msg, sizeof(_msg),
            "Invalid position '%c' for command '%c'", n, c);
            rule_error(_msg, line, s - 1); }
          rulefail++;
        }
        s++;
        if ((n < '0') || (n > '9' && n < 'A') || (n > 'Z' && n < 'a') ||
	   (n > 'z') ) {
          { char _msg[64]; snprintf(_msg, sizeof(_msg),
            "Invalid position '%c' for command '%c'", n, c);
            rule_error(_msg, line, s - 1); }
          rulefail++;
        }
	break;

      case 'X':
        n = *s++;
        if ((n < '0') || (n > '9' && n < 'A') || (n > 'Z' && n < 'a') ||
	   (n > 'z') ) {
          { char _msg[64]; snprintf(_msg, sizeof(_msg),
            "Invalid position '%c' for command '%c'", n, c);
            rule_error(_msg, line, s - 1); }
          rulefail++;
        }
        s++;
        if ((n < '0') || (n > '9' && n < 'A') || (n > 'Z' && n < 'a') ||
	   (n > 'z') ) {
          { char _msg[64]; snprintf(_msg, sizeof(_msg),
            "Invalid position '%c' for command '%c'", n, c);
            rule_error(_msg, line, s - 1); }
          rulefail++;
        }
        s++;
        if ((n < '0') || (n > '9' && n < 'A') || (n > 'Z' && n < 'a') ||
	   (n > 'z') ) {
          { char _msg[64]; snprintf(_msg, sizeof(_msg),
            "Invalid position '%c' for command '%c'", n, c);
            rule_error(_msg, line, s - 1); }
          rulefail++;
        }
	break;

      case ' ':
      case '\t':
      case ':':
        break;


      default:
        break;
    }
  }
  if (rulefail) return(NULL);
  return (lastvalid);
}


/* GPU rule engine — kernel-supported op classifier.
 *
 * Walk a single packed-rule bytecode (post-packrules) and return 1 if
 * every byte is a Tier-1 op acceptable to the GPU kernel. Any other
 * byte — 0x02 slowrule escape, mdxfind-specific S/#/v/=/% ops, the
 * memory-op family (M 4 6 X Q), and so on — fails eligibility and the
 * rule routes to the CPU list. As of slice 3 (kernel rev 1.22) the
 * reject and hex-output families are GPU-eligible. Memory ops were
 * tried in slice 4 (kernel rev 1.24) and reverted (path B): the
 * mem[40K] private buffer doubled per-thread private memory to ~80K
 * and FATAL'd CL_OUT_OF_HOST_MEMORY on RTX 3080 — kept CPU-only.
 *
 * Tier-1 op set grows incrementally per phase batch. Each new op
 * promotion adds its byte to this switch AND the kernel's per-op
 * interpreter (gpu/gpu_md5_rules.cl apply_rule_op) together.
 *
 * Currently supported (Phase 0 + Phase 1 batches 1-4 + slices 2/3):
 *   single-byte: l u r c C t E [ ] d f q { } k K
 *                H h (hex output; slice 3 — kernel rev 1.22)
 *                _ < > ! / ( ) (reject ops; slice 2 — kernel rev 1.22)
 *                (plus pass-through ' ' '\t' ':')
 *   two-byte:    T (position)
 *                e (literal delim byte)
 *                $ (literal append char, single-char form only)
 *                ^ (literal prepend char, single-char form only)
 *                ' (truncation length)
 *                D (delete-at position)
 *                + - L R . , (per-position arithmetic / shift / copy)
 *                @ (literal byte to purge)
 *                Z z (append/prepend N copies of last/first char)
 *                p   (extra copies appended)
 *                y Y (duplicate first/last N chars at start/end)
 *   three-byte:  s (find byte, replace byte)
 *                i (position, insert byte)
 *                o (position, overwrite byte)
 *                * (position A, position B; swap)
 *                x (start, length; extract substring)
 *                O (start, count; omit chars)
 *
 *   variable:    0xff (multi-char append for $X$Y$Z chains, 2+N bytes)
 *                0xfe (multi-char prepend for ^X^Y^Z chains, 2+N bytes)
 *
 * NOT supported (rule routes to CPU):
 *   - M 4 6 X Q                     — memory ops (slice-4 path B revert;
 *                                     mem[40K] OOM'd on 3080)
 *   - S # v = %                     — mdxfind-specific / placeholder ops
 *   - 0x02 (slowrule escape: base64-encode-word)
 *   - Anything else
 *
 * For multi-byte ops we must consume the parameter bytes too, otherwise
 * the next iteration would misinterpret them as ops.
 */
static int gpu_rule_safe_phase0(const char *packed_rule) {
    unsigned char c;
    while ((c = (unsigned char)*packed_rule++)) {
        switch (c) {
            case RULE_OP_LOWER: case RULE_OP_UPPER: case RULE_OP_REVERSE:
            case RULE_OP_CAP: case RULE_OP_CAP_INV: case RULE_OP_TOGGLE:
            case RULE_OP_TITLE_SP:
            case RULE_OP_DROP_FIRST: case RULE_OP_DROP_LAST:
            case RULE_OP_DUP: case RULE_OP_REFLECT: case RULE_OP_DUP_EACH:
            case RULE_OP_ROT_L: case RULE_OP_ROT_R:
            case RULE_OP_SWAP_FRONT: case RULE_OP_SWAP_BACK:
            case RULE_OP_NOOP:
            case RULE_OP_NOOP_SP: case RULE_OP_NOOP_TAB:
                continue;
            case RULE_OP_TOGGLE_AT: case RULE_OP_TITLE_SEP:
            case RULE_OP_APPEND: case RULE_OP_PREPEND:
            case RULE_OP_TRUNC: case RULE_OP_DEL_AT:
            case RULE_OP_INC: case RULE_OP_DEC:
            case RULE_OP_BIT_SHL: case RULE_OP_BIT_SHR:
            case RULE_OP_REPL_NEXT: case RULE_OP_REPL_PREV:
            case RULE_OP_PURGE:
            case RULE_OP_DUP_LAST: case RULE_OP_DUP_FIRST:
            case RULE_OP_REPEAT:
            case RULE_OP_DUP_PREFIX: case RULE_OP_DUP_SUFFIX:
            /* Rejection ops (rev 1.26): kernel rev 1.22 implements
             * `_ < > ! / ( )` with byte-exact applyrule semantics and
             * returns -1 to signal rejection. The four md5_rules kernels
             * (production, test, test_iter, validate) honor the sentinel.
             * Memory-op rejection Q (RULE_OP_MEM_REJ) stays in default-reject
             * — the M/4/6/X/Q family is CPU-only (slice-4 path B reverted
             * the kernel impl, see gpu/gpu_md5_rules.cl rev 1.24). */
            case RULE_OP_REJ_LEN_NE: case RULE_OP_REJ_LEN_GE:
            case RULE_OP_REJ_LEN_LE: case RULE_OP_REJ_HAS:
            case RULE_OP_REJ_NHAS: case RULE_OP_REJ_FIRST:
            case RULE_OP_REJ_LAST:
                /* Two-byte op: consume one parameter byte. */
                if (!*packed_rule++) return 0;
                continue;
            /* Hex emit ops (rev 1.26): kernel rev 1.22 implements `H` and
             * `h` (RULE_OP_HEX_UPPER / RULE_OP_HEX_LOWER) with byte-exact
             * applyrule semantics. Single-byte ops, no parameter to skip. */
            case RULE_OP_HEX_UPPER: case RULE_OP_HEX_LOWER:
                continue;
            case RULE_OP_SUB:
            case RULE_OP_INSERT: case RULE_OP_OVERWRITE:
            case RULE_OP_SWAP_AT: case RULE_OP_EXTRACT: case RULE_OP_OMIT:
            case RULE_OP_TOGGLE_SEP:
                /* Three-byte op: consume two parameter bytes.
                 * RULE_OP_TOGGLE_SEP is hashcat RULE_OP_MANGLE_TOGGLE_AT_SEP —
                 * same wire shape (op + position-byte + literal-byte) as
                 * 'i'/'o', so it joins this group. */
                if (!*packed_rule++) return 0;
                if (!*packed_rule++) return 0;
                continue;
            case 0xff: case 0xfe: {
                /* Multi-char append/prepend: 2+N bytes total. */
                unsigned char N = (unsigned char)*packed_rule++;
                if (N == 0) return 0;
                for (unsigned int j = 0; j < N; j++) {
                    if (!*packed_rule++) return 0;
                }
                continue;
            }
            default:
                return 0;
        }
    }
    return 1;
}

/* classify_rules — partition a packed rule set into GPU-eligible and
 * CPU-only lists per Phase 0 design (project_gpu_rule_engine_design.md
 * rev 3, §6). The full[] array is aliased — caller continues to own
 * the rule strings; gpu[] and cpu[] are malloc'd index arrays whose
 * pointers are stable for the lifetime of full[]. Both partitions
 * preserve original input order.
 *
 * Returns: ngpu (number of GPU-eligible rules; 0..nrules). */
int classify_rules(char **rules, int nrules, struct rule_lists *out) {
    if (!out) return -1;
    out->full  = rules;
    out->nfull = nrules;
    out->gpu   = (char **)malloc((size_t)nrules * sizeof(char *));
    out->cpu   = (char **)malloc((size_t)nrules * sizeof(char *));
    if (!out->gpu || !out->cpu) {
        free(out->gpu); free(out->cpu);
        out->gpu = out->cpu = NULL;
        out->ngpu = out->ncpu = 0;
        return -1;
    }
    out->ngpu = 0;
    out->ncpu = 0;
    for (int i = 0; i < nrules; i++) {
        if (gpu_rule_safe_phase0(rules[i])) {
            out->gpu[out->ngpu++] = rules[i];
        } else {
            out->cpu[out->ncpu++] = rules[i];
        }
    }
    return out->ngpu;
}

/* Free the gpu[] and cpu[] arrays. The rule strings themselves are NOT
 * freed — those are owned by the caller. The full[] alias is also left
 * alone (caller owns). */
void rule_lists_free(struct rule_lists *rl) {
    if (!rl) return;
    free(rl->gpu); rl->gpu = NULL; rl->ngpu = 0;
    free(rl->cpu); rl->cpu = NULL; rl->ncpu = 0;
    rl->full = NULL; rl->nfull = 0;
}


/* Apply rules to current word.  Basic error checking only.
   line points to the original input word - no touching this.
   pass points to the word we will be altering.
   len is the length of the line.  You cannot assume null terminates the line.
   rule points to the input rule, null terminated.
*/
#define FASTLEN 32

int applyrule(char *line, char *pass, int len, char *rule,
              struct rule_workspace *ws) {
    char *s, *d, *t, r, *cpass;
    unsigned char c, c1;
    char *orule = rule;
    unsigned long *s1, *d1;
    int x, y, z, clen, tlen;
#ifndef NOTINTEL
    __m128i *p128,*q128, a128,b128,c128,d128;
#endif
    char *Memory = ws->Memory;
    char *Base64buf = ws->Base64buf;
    static char *hextab = "0123456789abcdef";
    static char *Hextab = "0123456789ABCDEF";
    int memlen;
    int _retval = -2;
    /* MDXFIND_RULE_VALIDATOR=<anything>: emit one VALIDATE line per call
     * to applyrule on stderr.  Cached at first call to keep getenv() out
     * of the hot path (~100M invocations on HashMob.100k x rockyou). */
    static int _validate_cached = -1;
    int validate;
    if (_validate_cached == -1)
        _validate_cached = (getenv("MDXFIND_RULE_VALIDATOR") != NULL) ? 1 : 0;
    validate = _validate_cached;

    memlen = 0;
    Memory[0] = 0;
  if (len > MAXLINE) 
     { _retval = (-2); goto _validate_exit; }

if (len < FASTLEN) {

  cpass = pass+512;
  memcpy(cpass,line,len);

  cpass[len] = 0;
  clen = len;
  rule = orule;

  while ((c = *rule++)) {
    if (cpass < (pass+FASTLEN)) goto slowrule;
    /* fprintf(stderr,"rule=%c%s len=%d curpass=%s\n",c,rule,clen,cpass);   */
    switch (c) {
      case 0x02: /* control B */
	goto slowrule;

      default:
        /*
	      { char _msg[64]; snprintf(_msg, sizeof(_msg),
	        "Unknown rule command '%c'", c);
	        rule_error(_msg, orule, rule - 1); }
        { _retval = (-1); goto _validate_exit; }
        */
        break;
      case RULE_OP_HEX_LOWER:
      case 'h':
      case RULE_OP_HEX_UPPER:
      case 'H':
	  goto slowrule;
	  break;
      case 0xff:
	x = *rule++ & 0xff;
	s = rule;
	rule += x;
	if ((clen + x) > FASTLEN)
	   goto slowrule;
        memcpy(cpass+clen,s,x);
	clen += x;
        break;

      case 0xfe:
	x = *rule++ & 0xff;
	t = rule + x;
	if ((x+clen) > FASTLEN)
	   goto slowrule;
	cpass -= x;
	for (y=0; y < x; y++)
	    cpass[y] = rule[y];
        clen += x;
        rule = t;
        break;


      case RULE_OP_MEM_STORE:
      case 'M':
	memcpy(Memory,cpass,clen);
        memlen = clen;
	break;

      case RULE_OP_MEM_APP:
      case '4':
	y = memlen;
	if ((clen + memlen) > FASTLEN)
	   goto slowrule;
	if (y < 0)
	   y = 0;
	if (y == 0) break;
	memcpy(cpass+clen,Memory,y);
	clen += y;
	break;

    case RULE_OP_MEM_PRE:
    case '6':
	y = memlen;
	if ((clen + memlen) > FASTLEN)
	   goto slowrule;
	if (y < 0)
	   y = 0;
	if (y == 0) break;
	cpass -= y;
	memcpy(cpass,Memory,y);
	clen += y;
	break;

    case RULE_OP_MEM_REJ:
    case 'Q':
        if (memlen == clen && memcmp(cpass,Memory,memlen) == 0)
	    { _retval = (-1); goto _validate_exit; }
	break;

    case RULE_OP_MEM_INSERT:
    case 'X':
        y = *rule++ - 1;
	tlen = *rule++ - 1;
	z = *rule++ - 1;
	if ((clen + tlen) > FASTLEN) 
	    goto slowrule;
	if (tlen > memlen)
	    tlen = memlen;
	for (x=clen; x >= z; x--)
	   cpass[x+tlen] = cpass[x];
	for (x=0; x < tlen; x++) 
	   cpass[x+z] = Memory[x];
	clen += tlen;
	break;

        


      case RULE_OP_REJ_LEN_NE:
      case '_':
        y = *rule++ - 1;
	if (y != len)
	    { _retval = (-1); goto _validate_exit; }
	break;
      case RULE_OP_REJ_LEN_GE:
      case '<':
        y = *rule++ - 1; 
        if (clen < y)
          { _retval = (-1); goto _validate_exit; }
        break;
      case RULE_OP_REJ_LEN_LE:
      case '>':
        y = *rule++ - 1; 
        if (clen > y)
          { _retval = (-1); goto _validate_exit; }
        break;

      case RULE_OP_REJ_HAS:
      case '!':
        c = *rule++;
	for (x=0; x < clen; x++)
	    if (cpass[x] == c) { _retval = (-1); goto _validate_exit; }
        break;

      case RULE_OP_REJ_NHAS:
      case '/':
        c = *rule++;
	for (x=0; x < clen; x++)
	   if (cpass[x] == c) break;
        if (x >= clen )
          { _retval = (-1); goto _validate_exit; }
        break;

      case RULE_OP_REJ_FIRST:
      case '(':
        c = *rule++;
        if (clen > 0 && cpass[0] != c)
          { _retval = (-1); goto _validate_exit; }
        break;
      case RULE_OP_REJ_LAST:
      case ')':
        c = *rule++;
        if (clen > 0 && cpass[clen - 1] != c)
          { _retval = (-1); goto _validate_exit; }
        break;


      case RULE_OP_S_SPECIAL:
      case 'S':
        for (x = 0; x < clen; x++) {
          if (cpass[x] == 'a' || cpass[x] == 'A')
            cpass[x] = 0xa;
        }
        break;

      case RULE_OP_HASH_EXIT:
      case '#':
        goto fast_exit;
        break;

      case RULE_OP_NOOP:
      case ':':
      case RULE_OP_NOOP_SP:
      case ' ':
      case RULE_OP_NOOP_TAB:
      case '\t':
        break;

      case RULE_OP_LOWER:
      case 'l':
#ifdef NOTINTEL
        for (x = 0; x < clen; x++) {
          c = cpass[x];
          if (c >= 'A' && c <= 'Z')
            cpass[x] = c ^ 0x20;
        }
#else
	for (t=cpass,x=0; ((unsigned long)t & 15)  && x < clen; x++, t++) {
	   c = *t;
	   if (c >= 'A' && c <= 'Z')
	       *t = c ^ 0x20;
	}
	p128 = (__m128i *)t;
	for (; x < clen; x += 16) {
	    d128 = *p128;
	    a128 = _mm_sub_epi8(d128, _mm_set1_epi8((char)('A'+128)));
	    b128 = _mm_cmpgt_epi8(a128,_mm_set1_epi8((char)(-128+'Z'-'A')));
	    c128 = _mm_andnot_si128(b128,_mm_set1_epi8(0x20));
	    *p128++ = _mm_xor_si128(d128,c128);
	}
#endif
        break;

      case RULE_OP_UPPER:
      case 'u':
#ifdef NOTINTEL
        for (x = 0; x < clen; x++) {
          c = cpass[x];
          if (c >= 'a' && c <= 'z')
            cpass[x] = c ^ 0x20;
        }
#else
	for (t=cpass,x=0; ((unsigned long) t & 15) && x < clen; x++, t++) {
	   c = *t;
	   if (c >= 'a' && c <= 'z')
	       *t = c ^ 0x20;
	}
	p128 = (__m128i *)t;
	for (; x < clen; x += 16) {
	    d128 = *p128;
	    a128 = _mm_sub_epi8(d128, _mm_set1_epi8((char)('a'+128)));

	    b128 = _mm_cmpgt_epi8(a128,_mm_set1_epi8((char)(-128+'z'-'a')));
	    c128 = _mm_andnot_si128(b128,_mm_set1_epi8(0x20));
	    *p128++ = _mm_xor_si128(d128,c128);
	}
#endif
        break;

      case RULE_OP_CAP:
      case 'c':
#ifdef NOTINTEL
        for (z = x = 0; x < clen; x++) {
          c = cpass[x];
          if (z == 0 && ((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z'))) {
            if (c >= 'a' && c <= 'z')
              cpass[x] = c - 0x20;
            z = 1;
            continue;
          }
          if (c >= 'A' && c <= 'Z')
            cpass[x] = c + 0x20;
        }
#else
	for (t=cpass,x=0; ((unsigned long) t & 15) && x < clen; x++, t++) {
	   c = *t;
	   if (c >= 'A' && c <= 'Z')
	       *t = c ^ 0x20;
	}
	p128 = (__m128i *)t;
	for (; x < clen; x += 16) {
	    d128 = *p128;
	    a128 = _mm_sub_epi8(d128, _mm_set1_epi8((char)('A'+128)));
	    b128 = _mm_cmpgt_epi8(a128,_mm_set1_epi8((char)(-128+'Z'-'A')));
	    c128 = _mm_andnot_si128(b128,_mm_set1_epi8(0x20));
	    *p128++ = _mm_xor_si128(d128,c128);
	}
	for (x=0; x < clen; x++) {
	    c = cpass[x];
	    if (c >= 'a' && c <= 'z') {
	        cpass[x] = c ^ 0x20;
		break;
	    }
	}
#endif
        break;

      case RULE_OP_CAP_INV:
      case 'C':
#ifdef NOTINTEL
        for (z = x = 0; x < clen; x++) {
          c = cpass[x];
          if (z == 0 && ((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z'))) {
            if (c >= 'A' && c <= 'Z')
              cpass[x] = c + 0x20;
            z = 1;
            continue;
          }
          if (c >= 'a' && c <= 'z')
            cpass[x] = c - 0x20;
        }
#else
	for (t=cpass,x=0; ((unsigned long) t & 15) && x < clen; x++, t++) {
	   c = *t;
	   if (c >= 'a' && c <= 'z')
	       *t = c ^ 0x20;
	}
	p128 = (__m128i *)t;
	for (; x < clen; x += 16) {
	    d128 = *p128;
	    a128 = _mm_sub_epi8(d128, _mm_set1_epi8((char)('a'+128)));
	    b128 = _mm_cmpgt_epi8(a128,_mm_set1_epi8((char)(-128+'Z'-'A')));
	    c128 = _mm_andnot_si128(b128,_mm_set1_epi8(0x20));
	    *p128++ = _mm_xor_si128(d128,c128);
	}
	for (x=0; x < clen; x++) {
	    c = cpass[x];
	    if (c >= 'A' && c <= 'Z') {
	        cpass[x] = c ^ 0x20;
		break;
	    }
	}
#endif
        break;

      case RULE_OP_TOGGLE:
      case 't':
#ifdef NOTINTEL
        for (x = 0; x < clen; x++) {
          c = cpass[x];
          if ((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z'))
            cpass[x] = c ^ 0x20;
        }
#else
	for (t=cpass,x=0; ((unsigned long) t & 15) && x < clen; x++, t++) {
	   c = *t;
	   if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z'))
	       *t = c ^ 0x20;
	}
	p128 = (__m128i *)t;
	for (; x < clen; x += 16) {
	    d128 = *p128;
	    a128 = _mm_sub_epi8(d128, _mm_set1_epi8((char)('a'+128)));
	    b128 = _mm_cmpgt_epi8(a128,_mm_set1_epi8((char)(-128+'z'-'a')));
	    a128 = _mm_sub_epi8(d128, _mm_set1_epi8((char)('A'+128)));
	    c128 = _mm_cmpgt_epi8(a128,_mm_set1_epi8((char)(-128+'Z'-'A')));
	    b128 = _mm_and_si128(b128,c128);
	    c128 = _mm_andnot_si128(b128,_mm_set1_epi8(0x20));
	    *p128++ = _mm_xor_si128(d128,c128);
	}
#endif
        break;

      case RULE_OP_TOGGLE_AT:
      case 'T':
        y = *rule++ - 1;
        c = cpass[y];
        if ((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z'))
          cpass[y] = c ^ 0x20;
        break;

      case RULE_OP_REVERSE:
      case 'r':
        for (x = 0; x < clen / 2; x++) {
          c = cpass[x];
          cpass[x] = cpass[clen - x - 1];
          cpass[clen - x - 1] = c;
        }
        break;

      case RULE_OP_DUP:
      case 'd':
        tlen = clen;
        if ((tlen + clen) > FASTLEN)
          goto slowrule;
        if (tlen > 0) {
	  memcpy(cpass+clen,cpass,tlen);
          clen += tlen;
	}
        break;

      case RULE_OP_REFLECT:
      case 'f':
        tlen = clen;
        if ((tlen + clen) > FASTLEN)
          goto slowrule;
        if (tlen < 0)
          tlen = 0;
        for (x = 0; x < tlen; x++)
          cpass[clen + tlen - x - 1] = cpass[x];
        clen += tlen;
        break;

      case RULE_OP_ROT_L:
      case '{':
        if (clen > 0) {
          y = 1;
          while (*rule == '{' && y < clen) {
            y++;
            rule++;
          }
          for (x = 0; x < y; x++)
            cpass[x + clen] = cpass[x];
          for (; x < (clen + y); x++)
            cpass[x - y] = cpass[x];
        }
        break;

      case RULE_OP_ROT_R:
      case '}':
        if (clen > 0) {
          y = 1;
          while (*rule == '}' && y < clen) {
            y++;
            rule++;
          }
          for (x = clen - 1; x >= 0; x--)
            cpass[x + y] = cpass[x];
          for (x = 0; x < y; x++)
            cpass[x] = cpass[x + clen];
        }
        break;

      case RULE_OP_APPEND:
      case '$':
        c = *rule++;
        if (!c) {
          fprintf(stderr, "Out of rules in append at %s\n", orule);
          { _retval = (-3); goto _validate_exit; }
        }
	if ((clen+1) < FASTLEN) 
	  cpass[clen++] = c;
	else
	  goto slowrule;
        break;

      case RULE_OP_PREPEND:
      case '^':
        c = *rule++;
        if (!c) {
          fprintf(stderr, "Out of rules in insert at %s\n", orule);
          { _retval = (-3); goto _validate_exit; }
        }
	if ((clen+1) > FASTLEN)
	  goto slowrule;
	cpass--;
	cpass[0] = c;
	clen++;
        break;

      case RULE_OP_DROP_FIRST:
      case '[':
        if (clen > 0) {
          y = 1;
          while (*rule == '[' && y < clen) {
            y++;
            rule++;
          }
	  cpass += y;
          clen -= y;
        }
        break;

      case RULE_OP_DROP_LAST:
      case ']':
        if (clen > 0) {
          y = 1;
          while (*rule == ']' && y < clen) {
            y++;
            rule++;
          }
          clen -= y;
        }
        break;

      case RULE_OP_DEL_AT:
      case 'D':
        y = *rule++ - 1;
        if (y < clen) {
          for (x = y + 1; x < clen; x++)
            cpass[x - 1] = cpass[x];

          clen--;
        }
        break;

      case RULE_OP_EXTRACT:
      case 'x':
        y = *rule++ - 1;
        z = *rule++ - 1;
        if (clen > y) {
          for (x = 0; x < z && ((y + x) < clen); x++) {
            cpass[x] = cpass[y + x];
          }
          clen = x;
          if (clen < 0)
            clen = 0;
        }
        break;
      case RULE_OP_OMIT:
      case 'O':
        y = *rule++ - 1;
        z = *rule++ - 1;
        if (clen > y && (y + z) <= clen) {
          for (x = y; x < clen && (x + z) < clen; x++) {
            cpass[x] = cpass[x + z];
          }
          clen = x;
          if (clen < 0)
            clen = 0;
        }
        break;
      case RULE_OP_INSERT:
      case 'i':
        y = *rule++ - 1;
        c = *rule++;
        if (!c) {
          fprintf(stderr, "Invalid insert character in rule %s\n", orule);
          { _retval = (-3); goto _validate_exit; }
        }
        if (clen > y) {
	  if ((clen+1) > FASTLEN)
	      goto slowrule;
	   for (x = clen; x >= y && x > 0; x--)
	      cpass[x] = cpass[x - 1];
	    clen++;
	    cpass[y] = c;
        }
        break;
      case RULE_OP_OVERWRITE:
      case 'o':
        y = *rule++ - 1;
        c = *rule++;
	if (c == 0) {
	    fprintf(stderr,"Invalid character in o rule: %x\n",c);
	    { _retval = (-3); goto _validate_exit; }
	}
        if (y < clen)
          cpass[y] = c;
	if (y == 0 && clen == 0) {
	   cpass[0] = c; clen++;
	}
        break;
      case RULE_OP_TRUNC:
      case '\'':
        y = *rule++ - 1;
        if (y < clen)
          clen = y;
        break;

      case RULE_OP_DIV_INSERT:
      case 'v':
	x = *rule++;
	c1 = *rule++;
	if (x <=0) {
	  fprintf(stderr,"Invalid count %d in rule: %c\n",x,format_op_for_error((unsigned char)c));
	  { _retval = (-3); goto _validate_exit; }
	}
	if (clen < x) break;
        if ((clen + clen/x) >= FASTLEN) goto slowrule;
        y = clen / x;
	s = &cpass[clen-1];
	d = s + y;
        for (y = clen; y > 0; y--) {
	  if ((y%x) == 0) {
 	    *d-- = c1;
	    if (s == d) break;
	  }
	  *d-- = *s--;
        }
	clen += clen / x;
	cpass[clen] = 0;
	break;

      case RULE_OP_SUB:
      case 's':
	c = *rule++;
        r = *rule++;
        if (!c || !r) {
          rule_error("'s' (substitute) requires two characters: sXY",
                     orule, rule - (c ? 2 : 1));
          { _retval = (-3); goto _validate_exit; }
        }
#ifdef NOTINTEL
        for (x = 0; x < clen; x++) {
          if (cpass[x] == c)
            cpass[x] = r;
        }
#else
	for (t=cpass,x=0; ((unsigned long) t & 15) && x < clen; x++, t++) {
	    if (*t == c)
	        *t = r;
	}
	p128 = (__m128i *)t;
	for (; x < clen; x += 16) {
	    d128 = *p128;
	    a128 = _mm_cmpeq_epi8(d128,_mm_set1_epi8((char)c));
	    b128 = _mm_and_si128(a128,_mm_set1_epi8((char)(c^r)));
	    *p128++ = _mm_xor_si128(d128,b128);
	}
#endif
        break;

      case RULE_OP_PURGE:
      case '@':
        c = *rule++;

        if (!c) {
          fprintf(stderr, "Invalid purge in rule %s\n", orule);
          { _retval = (-3); goto _validate_exit; }
        }
        d = cpass;
        s = cpass;
        for (x = 0; x < clen; x++) {
          if (*s != c)
            *d++ = *s;
          s++;
        }
        clen -= (s - d);
	if (clen < 0)
	  clen = 0;
        break;

      case RULE_OP_DUP_FIRST:
      case 'z':
        y = *rule++ - 1;
        if (clen > 0) {
	  if ((clen+y) > FASTLEN)
	    goto slowrule;
          for (x = clen - 1; x > 0; x--)
            cpass[x + y] = cpass[x];
          for (x = 1; x <= y; x++)
            cpass[x] = cpass[0];
          clen += y;
        }
        break;

      case RULE_OP_DUP_LAST:
      case 'Z':
        y = *rule++ - 1;
        if (clen > 0) {
	  if ((y + clen) > FASTLEN)
	    goto slowrule;
          for (x = 0; x < y; x++)
            cpass[x + clen] = cpass[clen - 1];
          clen += y;
        }
        break;

      case RULE_OP_DUP_EACH:
      case 'q':
        tlen = clen * 2;
        if (tlen > FASTLEN)
          goto slowrule;
        for (x = clen * 2; x > 0; x -= 2) {
          cpass[x - 1] = cpass[x / 2 - 1];
          cpass[x - 2] = cpass[x / 2 - 1];
        }
        clen += clen;
        break;

      case RULE_OP_REPEAT:
      case 'p':
        y = *rule++ - 1;
        if (clen > 0 && y > 0) {
          d = &cpass[clen];
          z = y;
          tlen = clen;
          for (; y; y--) {
            if ((clen + tlen) > FASTLEN)
              goto slowrule;
            for (x = 0; x < tlen; x++)
              *d++ = cpass[x];
            clen += tlen;
          }
        }
        break;

      case RULE_OP_SWAP_FRONT:
      case 'k':
        if (clen >1) {
	   c = cpass[0];
	   cpass[0] = cpass[1];
	   cpass[1] = c;
	}
	break;

      case RULE_OP_SWAP_BACK:
      case 'K':
        if (clen > 1) {
          c = cpass[clen - 2];
          cpass[clen - 2] = cpass[clen - 1];
          cpass[clen - 1] = c;
        }
        break;

      case RULE_OP_SWAP_AT:
      case '*':
        y = *rule++ - 1;
        z = *rule++ - 1;
        if (y < clen && z < clen) {
          c = cpass[y];
          cpass[y] = cpass[z];
          cpass[z] = c;
        }
        break;

      case RULE_OP_BIT_SHL:
      case 'L':
        y = *rule++ - 1;
        if (y < clen)
          cpass[y] = cpass[y] << 1;
        break;

      case RULE_OP_BIT_SHR:
      case 'R':
        y = *rule++ - 1;
        if (y < clen)
          cpass[y] = cpass[y] >> 1;
        break;

      case RULE_OP_INC:
      case '+':
        y = *rule++ - 1;
        if (y < clen)
          cpass[y]++;
        break;

      case RULE_OP_DEC:
      case '-':
        y = *rule++ - 1;
        if (y < clen)
          cpass[y]--;
        break;

      case RULE_OP_REPL_NEXT:
      case '.':
        y = *rule++ - 1;
        if (y < clen)
          cpass[y] = cpass[y + 1];
        break;

      case RULE_OP_REPL_PREV:
      case ',':
        y = *rule++ - 1;
        if (y < clen && y > 0)
          cpass[y] = cpass[y - 1];
        break;

      case RULE_OP_DUP_PREFIX:
      case 'y':
        y = *rule++ - 1;
        if (clen > 0 && y <= clen) {
	  if ((clen+y) > FASTLEN)
	     goto slowrule;
          memmove(cpass + y, cpass, clen);
          clen += y;
        }
        break;
      
      case RULE_OP_DUP_SUFFIX:
      case 'Y':
        y = *rule++ - 1;
        if (clen > 0 && y <= clen) {
	  if ((clen+y) > FASTLEN)
	    goto slowrule;
          memmove(cpass + clen, cpass + (clen - y), y);
          clen += y;
        }
        break;

      case RULE_OP_TITLE_SP:
      case 'E':
        for (z = x = 0; x < clen; x++) {
          c = cpass[x];
          if (c == ' ')
            z = 0;
          else if (z == 0 && (c >= 'a' && c <= 'z')) {
            z = 1;
            cpass[x] = c ^ 0x20;
          } else if (c >= 'A' && c <= 'Z')
            cpass[x] = c ^ 0x20;
        }
        break;
      case RULE_OP_TITLE_SEP:
      case 'e':
	c1 = *rule++;
        for (z = x = 0; x < clen; x++) {
          c = cpass[x];
          if (c == c1)
            z = 0;
          else if (z == 0 && (c >= 'a' && c <= 'z')) {
            z = 1;
            cpass[x] = c ^ 0x20;
          } else if (c >= 'A' && c <= 'Z')
            cpass[x] = c ^ 0x20;
        }
        break;

      case RULE_OP_TOGGLE_SEP:
      case '3':
        /* Hashcat RULE_OP_MANGLE_TOGGLE_AT_SEP: walk cpass, count
         * occurrences of separator c1; after the y-th occurrence,
         * toggle case of the first alphabetic char and stop. */
        y = *rule++ - 1;          /* upos */
        c1 = *rule++;             /* separator */
        {
          int toggle_next = 0;
          int occurrence  = 0;
          for (x = 0; x < clen; x++) {
            c = cpass[x];
            if (c == c1) {
              if (occurrence == y) {
                toggle_next = 1;
              } else {
                occurrence++;
              }
              continue;
            }
            if (toggle_next) {
              if ((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z'))
                cpass[x] = c ^ 0x20;
              break;
            }
          }
        }
        break;
    }
  }
fast_exit:
  memmove(pass,cpass,clen);
  goto app_exit;
 
}
slowrule:
  memcpy(pass,line,len);
  pass[len] = 0;

  clen = len;
  rule = orule;

  while ((c = *rule++)) {
    /* printf("rule=%c%s len=%d curpass=%s\n",c,rule,clen,pass);   */
    switch (c) {
      default:
        /*
	      { char _msg[64]; snprintf(_msg, sizeof(_msg),
	        "Unknown rule command '%c'", c);
	        rule_error(_msg, orule, rule - 1); }
        { _retval = (-1); goto _validate_exit; }
        */
        break;
      case 0x02: /* Control B */
	if (clen > (MAXLINE*4/3)) break;
	clen = b64_encode(pass, Base64buf, clen);
	memcpy(pass,Base64buf,clen); pass[clen] = 0;
	break;

      case RULE_OP_HEX_LOWER:
      case 'h':
      case RULE_OP_HEX_UPPER:
      case 'H':
        d = hextab;
	if (c == 'H' || c == RULE_OP_HEX_UPPER) d = Hextab;
        x = clen;
        if ((clen +x) > MAXLINE)
          x = MAXLINE - clen;
	clen = clen + x;
        for (x--; x >=0; x--) {
          c = pass[x];
	  pass[x*2] = d[(c>>4)&0xf];
	  pass[(x*2)+1] = d[c & 0xf];
	}
	pass[clen] = 0;
	break;
	  
 
           
      case 0xff:
	x = *rule++ & 0xff;
	s = rule;
	rule += x;
	if ((clen + x) > MAXLINE)
	   x = MAXLINE - clen;
	memcpy(pass+clen,s,x);
	clen += x;
        break;

      case 0xfe:
	x = *rule++ & 0xff;
	t = rule + x;
	if ((x+clen) > MAXLINE)
	   x = MAXLINE-clen;
	memmove(pass+x,pass,clen);
	for (y=0; y < x; y++)
	    pass[y] = rule[y];
        clen += x;
        rule = t;
        break;


      case RULE_OP_MEM_STORE:
      case 'M':
	memcpy(Memory,pass,clen);
        memlen = clen;
	break;

      case RULE_OP_MEM_APP:
      case '4':
	y = memlen;
	if ((clen + memlen) > MAXLINE)
	   y = MAXLINE - clen;
	if (y < 0)
	   y = 0;
	if (y == 0) break;
	memcpy(pass+clen,Memory,y);
	clen += y;
	break;

    case RULE_OP_MEM_PRE:
    case '6':
	y = memlen;
	if ((clen + memlen) > MAXLINE)
	   y = MAXLINE - clen;
	if (y < 0)
	   y = 0;
	if (y == 0) break;
	memmove(pass+y,pass,clen);
	memcpy(pass,Memory,y);
	clen += y;
	break;

    case RULE_OP_MEM_REJ:
    case 'Q':
        if (memlen == clen && memcmp(pass,Memory,memlen) == 0)
	    { _retval = (-1); goto _validate_exit; }
	break;

    case RULE_OP_MEM_INSERT:
    case 'X':
        y = *rule++ - 1;
	tlen = *rule++ - 1;
	z = *rule++ - 1;
	if ((clen + tlen) > MAXLINE) 
	    tlen = MAXLINE - clen;
	if (tlen > memlen)
	    tlen = memlen;
	for (x=clen; x >= z; x--)
	   pass[x+tlen] = pass[x];
	for (x=0; x < tlen; x++) 
	   pass[x+z] = Memory[x];
	clen += tlen;
	break;

        


      case RULE_OP_REJ_LEN_NE:
      case '_':
        y = *rule++ - 1;
	if (y != len)
	    { _retval = (-1); goto _validate_exit; }
	break;
      case RULE_OP_REJ_LEN_GE:
      case '<':
        y = *rule++ - 1; 
        if (clen < y)
          { _retval = (-1); goto _validate_exit; }
        break;
      case RULE_OP_REJ_LEN_LE:
      case '>':
        y = *rule++ - 1; 
        if (clen > y)
          { _retval = (-1); goto _validate_exit; }
        break;

      case RULE_OP_REJ_HAS:
      case '!':
        c = *rule++;
	for (x=0; x < clen; x++)
	    if (pass[x] == c) { _retval = (-1); goto _validate_exit; }
        break;

      case RULE_OP_REJ_NHAS:
      case '/':
        c = *rule++;
	for (x=0; x < clen; x++)
	   if (pass[x] == c) break;
        if (x >= clen )
          { _retval = (-1); goto _validate_exit; }
        break;

      case RULE_OP_REJ_FIRST:
      case '(':
        c = *rule++;
        if (clen > 0 && pass[0] != c)
          { _retval = (-1); goto _validate_exit; }
        break;
      case RULE_OP_REJ_LAST:
      case ')':
        c = *rule++;
        if (clen > 0 && pass[clen - 1] != c)
          { _retval = (-1); goto _validate_exit; }
        break;


      case RULE_OP_S_SPECIAL:
      case 'S':
        for (x = 0; x < clen; x++) {
          if (pass[x] == 'a' || pass[x] == 'A')
            pass[x] = 0xa;
        }
        break;

      case RULE_OP_HASH_EXIT:
      case '#':
        goto app_exit;
        break;

      case RULE_OP_NOOP:
      case ':':
      case RULE_OP_NOOP_SP:
      case ' ':
      case RULE_OP_NOOP_TAB:
      case '\t':
        break;

      case RULE_OP_LOWER:
      case 'l':
#ifdef NOTINTEL
        for (x = 0; x < clen; x++) {
          c = pass[x];
          if (c >= 'A' && c <= 'Z')
            pass[x] = c ^ 0x20;
        }
#else
	for (t=pass,x=0; ((unsigned long)t & 15)  && x < clen; x++, t++) {
	   c = *t;
	   if (c >= 'A' && c <= 'Z')
	       *t = c ^ 0x20;
	}
	p128 = (__m128i *)t;
	for (; x < clen; x += 16) {
	    d128 = *p128;
	    a128 = _mm_sub_epi8(d128, _mm_set1_epi8((char)('A'+128)));
	    b128 = _mm_cmpgt_epi8(a128,_mm_set1_epi8((char)(-128+'Z'-'A')));
	    c128 = _mm_andnot_si128(b128,_mm_set1_epi8(0x20));
	    *p128++ = _mm_xor_si128(d128,c128);
	}
#endif
        break;

      case RULE_OP_UPPER:
      case 'u':
#ifdef NOTINTEL
        for (x = 0; x < clen; x++) {
          c = pass[x];
          if (c >= 'a' && c <= 'z')
            pass[x] = c ^ 0x20;
        }
#else
	for (t=pass,x=0; ((unsigned long) t & 15) && x < clen; x++, t++) {
	   c = *t;
	   if (c >= 'a' && c <= 'z')
	       *t = c ^ 0x20;
	}
	p128 = (__m128i *)t;
	for (; x < clen; x += 16) {
	    d128 = *p128;
	    a128 = _mm_sub_epi8(d128, _mm_set1_epi8((char)('a'+128)));

	    b128 = _mm_cmpgt_epi8(a128,_mm_set1_epi8((char)(-128+'z'-'a')));
	    c128 = _mm_andnot_si128(b128,_mm_set1_epi8(0x20));
	    *p128++ = _mm_xor_si128(d128,c128);
	}
#endif
        break;

      case RULE_OP_CAP:
      case 'c':
#ifdef NOTINTEL
        for (z = x = 0; x < clen; x++) {
          c = pass[x];
          if (z == 0 && ((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z'))) {
            if (c >= 'a' && c <= 'z')
              pass[x] = c - 0x20;
            z = 1;
            continue;
          }
          if (c >= 'A' && c <= 'Z')
            pass[x] = c + 0x20;
        }
#else
	for (t=pass,x=0; ((unsigned long) t & 15) && x < clen; x++, t++) {
	   c = *t;
	   if (c >= 'A' && c <= 'Z')
	       *t = c ^ 0x20;
	}
	p128 = (__m128i *)t;
	for (; x < clen; x += 16) {
	    d128 = *p128;
	    a128 = _mm_sub_epi8(d128, _mm_set1_epi8((char)('A'+128)));
	    b128 = _mm_cmpgt_epi8(a128,_mm_set1_epi8((char)(-128+'Z'-'A')));
	    c128 = _mm_andnot_si128(b128,_mm_set1_epi8(0x20));
	    *p128++ = _mm_xor_si128(d128,c128);
	}
	for (x=0; x < clen; x++) {
	    c = pass[x];
	    if (c >= 'a' && c <= 'z') {
	        pass[x] = c ^ 0x20;
		break;
	    }
	}
#endif
        break;

      case RULE_OP_CAP_INV:
      case 'C':
#ifdef NOTINTEL
        for (z = x = 0; x < clen; x++) {
          c = pass[x];
          if (z == 0 && ((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z'))) {
            if (c >= 'A' && c <= 'Z')
              pass[x] = c + 0x20;
            z = 1;
            continue;
          }
          if (c >= 'a' && c <= 'z')
            pass[x] = c - 0x20;
        }
#else
	for (t=pass,x=0; ((unsigned long) t & 15) && x < clen; x++, t++) {
	   c = *t;
	   if (c >= 'a' && c <= 'z')
	       *t = c ^ 0x20;
	}
	p128 = (__m128i *)t;
	for (; x < clen; x += 16) {
	    d128 = *p128;
	    a128 = _mm_sub_epi8(d128, _mm_set1_epi8((char)('a'+128)));
	    b128 = _mm_cmpgt_epi8(a128,_mm_set1_epi8((char)(-128+'Z'-'A')));
	    c128 = _mm_andnot_si128(b128,_mm_set1_epi8(0x20));
	    *p128++ = _mm_xor_si128(d128,c128);
	}
	for (x=0; x < clen; x++) {
	    c = pass[x];
	    if (c >= 'A' && c <= 'Z') {
	        pass[x] = c ^ 0x20;
		break;
	    }
	}
#endif
        break;

      case RULE_OP_TOGGLE:
      case 't':
#ifdef NOTINTEL
        for (x = 0; x < clen; x++) {
          c = pass[x];
          if ((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z'))
            pass[x] = c ^ 0x20;
        }
#else
	for (t=pass,x=0; ((unsigned long) t & 15) && x < clen; x++, t++) {
	   c = *t;
	   if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z'))
	       *t = c ^ 0x20;
	}
	p128 = (__m128i *)t;
	for (; x < clen; x += 16) {
	    d128 = *p128;
	    a128 = _mm_sub_epi8(d128, _mm_set1_epi8((char)('a'+128)));
	    b128 = _mm_cmpgt_epi8(a128,_mm_set1_epi8((char)(-128+'z'-'a')));
	    a128 = _mm_sub_epi8(d128, _mm_set1_epi8((char)('A'+128)));
	    c128 = _mm_cmpgt_epi8(a128,_mm_set1_epi8((char)(-128+'Z'-'A')));
	    b128 = _mm_and_si128(b128,c128);
	    c128 = _mm_andnot_si128(b128,_mm_set1_epi8(0x20));
	    *p128++ = _mm_xor_si128(d128,c128);
	}
#endif
        break;

      case RULE_OP_TOGGLE_AT:
      case 'T':
        y = *rule++ - 1;
        c = pass[y];
        if ((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z'))
          pass[y] = c ^ 0x20;
        break;

      case RULE_OP_REVERSE:
      case 'r':
        for (x = 0; x < clen / 2; x++) {
          c = pass[x];
          pass[x] = pass[clen - x - 1];
          pass[clen - x - 1] = c;
        }
        break;

      case RULE_OP_DUP:
      case 'd':
        tlen = clen;
        if ((tlen + clen) > MAXLINE)
          tlen = MAXLINE - clen;
        if (tlen > 0) {
	  memcpy(pass+clen,pass,tlen);
	  clen += tlen;
	}
        break;

      case RULE_OP_REFLECT:
      case 'f':
        tlen = clen;
        if ((tlen + clen) > MAXLINE)
          tlen = MAXLINE - clen;
        if (tlen < 0)
          tlen = 0;
        for (x = 0; x < tlen; x++)
          pass[clen + tlen - x - 1] = pass[x];
        clen += tlen;
        break;

      case RULE_OP_ROT_L:
      case '{':
        if (clen > 0) {
          y = 1;
          while (*rule == '{' && y < clen) {
            y++;
            rule++;
          }
          for (x = 0; x < y; x++)
            pass[x + clen] = pass[x];
          for (; x < (clen + y); x++)
            pass[x - y] = pass[x];
        }
        break;

      case RULE_OP_ROT_R:
      case '}':
        if (clen > 0) {
          y = 1;
          while (*rule == '}' && y < clen) {
            y++;
            rule++;
          }
          for (x = clen - 1; x >= 0; x--)
            pass[x + y] = pass[x];
          for (x = 0; x < y; x++)
            pass[x] = pass[x + clen];
        }
        break;

      case RULE_OP_APPEND:
      case '$':
        c = *rule++;
        if (!c) {
          fprintf(stderr, "Out of rules in append at %s\n", orule);
          { _retval = (-3); goto _validate_exit; }
        }
	if ((clen+1) < MAXLINE) 
	  pass[clen++] = c;
        break;

      case RULE_OP_PREPEND:
      case '^':
        c = *rule++;
        if (!c) {
          fprintf(stderr, "Out of rules in insert at %s\n", orule);
          { _retval = (-3); goto _validate_exit; }
        }
	if ((clen+1) < MAXLINE) {
	  memmove(pass+1,pass,clen);
	  pass[0] = c;
	  clen++;
	}
        break;

      case RULE_OP_DROP_FIRST:
      case '[':
        if (clen > 0) {
          y = 1;
          while (*rule == '[' && y < clen) {
            y++;
            rule++;
          }
	  memmove(pass,pass+y,clen);
          clen -= y;
        }
        break;

      case RULE_OP_DROP_LAST:
      case ']':
        if (clen > 0) {
          y = 1;
          while (*rule == ']' && y < clen) {
            y++;
            rule++;
          }
          clen -= y;
        }
        break;

      case RULE_OP_DEL_AT:
      case 'D':
        y = *rule++ - 1;
        if (y < clen) {
          for (x = y + 1; x < clen; x++)
            pass[x - 1] = pass[x];

          clen--;
        }
        break;

      case RULE_OP_EXTRACT:
      case 'x':
        y = *rule++ - 1;
        z = *rule++ - 1;
        if (clen > y) {
          for (x = 0; x < z && ((y + x) < clen); x++) {
            pass[x] = pass[y + x];
          }
          clen = x;
          if (clen < 0)
            clen = 0;
        }
        break;
      case RULE_OP_OMIT:
      case 'O':
        y = *rule++ - 1;
        z = *rule++ - 1;
        if (clen > y && (y + z) <= clen) {
          for (x = y; x < clen && (x + z) < clen; x++) {
            pass[x] = pass[x + z];
          }
          clen = x;
          if (clen < 0)
            clen = 0;
        }
        break;
      case RULE_OP_INSERT:
      case 'i':
        y = *rule++ - 1;
        c = *rule++;
        if (!c) {
          fprintf(stderr, "Invalid insert character in rule %s\n", orule);
          { _retval = (-3); goto _validate_exit; }
        }
        if (clen > y) {
	  if ((clen+1) < MAXLINE) {
	    for (x = clen; x >= y && x > 0; x--)
	      pass[x] = pass[x - 1];
	    clen++;
	    pass[y] = c;
	  }
        }
        break;
      case RULE_OP_OVERWRITE:
      case 'o':
        y = *rule++ - 1;
        c = *rule++;
	if (c == 0) {
	    fprintf(stderr,"Invalid character in o rule: %x\n",c);
	    { _retval = (-3); goto _validate_exit; }
	}
        if (y < clen)
          pass[y] = c;
	if (y == 0 && clen == 0) {
	   pass[0] = c; clen++;
	}
        break;
      case RULE_OP_TRUNC:
      case '\'':
        y = *rule++ - 1;
        if (y < clen)
          clen = y;
        break;

      case RULE_OP_DIV_INSERT:
      case 'v':
	x = *rule++;
	c1 = *rule++;
	if (x <=0) {
	  fprintf(stderr,"Invalid count %d in rule: %c\n",x,format_op_for_error((unsigned char)c));
	  { _retval = (-3); goto _validate_exit; }
	}
        y = clen / x;
	s = &pass[clen-1];
	d = s + y;
        for (y = clen; y > 0; y--) {
	  if ((y%x) == 0) {
 	    *d-- = c1;
	    if (s == d) break;
	  }
	  *d-- = *s--;
        }
	clen += clen / x;
	pass[clen] = 0;
	break;
	
      case RULE_OP_SUB:
      case 's':
        c = *rule++;
        r = *rule++;
        if (!c || !r) {
          rule_error("'s' (substitute) requires two characters: sXY",
                     orule, rule - (c ? 2 : 1));
          { _retval = (-3); goto _validate_exit; }
        }
#ifdef NOTINTEL
        for (x = 0; x < clen; x++) {
          if (pass[x] == c)
            pass[x] = r;
        }
#else
	for (t=pass,x=0; ((unsigned long) t & 15) && x < clen; x++, t++) {
	    if (*t == c)
	        *t = r;
	}
	p128 = (__m128i *)t;
	for (; x < clen; x += 16) {
	    d128 = *p128;
	    a128 = _mm_cmpeq_epi8(d128,_mm_set1_epi8((char)c));
	    b128 = _mm_and_si128(a128,_mm_set1_epi8((char)(c^r)));
	    *p128++ = _mm_xor_si128(d128,b128);
	}
#endif
        break;

      case RULE_OP_PURGE:
      case '@':
        c = *rule++;

        if (!c) {
          fprintf(stderr, "Invalid purge in rule %s\n", orule);
          { _retval = (-3); goto _validate_exit; }
        }
        d = pass;
        s = pass;
        for (x = 0; x < clen; x++) {
          if (*s != c)
            *d++ = *s;
          s++;
        }
        clen -= (s - d);
	if (clen < 0)
	  clen = 0;
        break;

      case RULE_OP_DUP_FIRST:
      case 'z':
        y = *rule++ - 1;
        if (clen > 0) {
	  if ((clen+y) > MAXLINE)
	    y = MAXLINE - clen;
          for (x = clen - 1; x > 0; x--)
            pass[x + y] = pass[x];
          for (x = 1; x <= y; x++)
            pass[x] = pass[0];
          clen += y;
        }
        break;

      case RULE_OP_DUP_LAST:
      case 'Z':
        y = *rule++ - 1;
        if (clen > 0) {
	  if ((y + clen) > MAXLINE)
	    y = MAXLINE - clen;
          for (x = 0; x < y; x++)
            pass[x + clen] = pass[clen - 1];
          clen += y;
        }
        break;

      case RULE_OP_DUP_EACH:
      case 'q':
        tlen = clen * 2;
        if (tlen > MAXLINE)
          break;
        for (x = clen * 2; x > 0; x -= 2) {
          pass[x - 1] = pass[x / 2 - 1];
          pass[x - 2] = pass[x / 2 - 1];
        }
        clen += clen;
        break;

      case RULE_OP_REPEAT:
      case 'p':
        y = *rule++ - 1;
        if (clen > 0 && y > 0) {
          d = &pass[clen];
          z = y;
          tlen = clen;
          for (; y; y--) {
            if ((clen + tlen) > MAXLINE)
              break;
            for (x = 0; x < tlen; x++)
              *d++ = pass[x];
            clen += tlen;
          }
        }
        break;

      case RULE_OP_SWAP_FRONT:
      case 'k':
        if (clen >1) {
	   c = pass[0];
	   pass[0] = pass[1];
	   pass[1] = c;
	}
	break;

      case RULE_OP_SWAP_BACK:
      case 'K':
        if (clen > 1) {
          c = pass[clen - 2];
          pass[clen - 2] = pass[clen - 1];
          pass[clen - 1] = c;
        }
        break;

      case RULE_OP_SWAP_AT:
      case '*':
        y = *rule++ - 1;
        z = *rule++ - 1;
        if (y < clen && z < clen) {
          c = pass[y];
          pass[y] = pass[z];
          pass[z] = c;
        }
        break;

      case RULE_OP_BIT_SHL:
      case 'L':
        y = *rule++ - 1;
        if (y < clen)
          pass[y] = pass[y] << 1;
        break;

      case RULE_OP_BIT_SHR:
      case 'R':
        y = *rule++ - 1;
        if (y < clen)
          pass[y] = pass[y] >> 1;
        break;

      case RULE_OP_INC:
      case '+':
        y = *rule++ - 1;
        if (y < clen)
          pass[y]++;
        break;

      case RULE_OP_DEC:
      case '-':
        y = *rule++ - 1;
        if (y < clen)
          pass[y]--;
        break;

      case RULE_OP_REPL_NEXT:
      case '.':
        y = *rule++ - 1;
        if (y < clen)
          pass[y] = pass[y + 1];
        break;

      case RULE_OP_REPL_PREV:
      case ',':
        y = *rule++ - 1;
        if (y < clen && y > 0)
          pass[y] = pass[y - 1];
        break;

      case RULE_OP_DUP_PREFIX:
      case 'y':
        y = *rule++ - 1;
        if (clen > 0 && y <= clen) {
	  if ((clen+y) > MAXLINE)
	     y = MAXLINE - clen;
          memmove(pass + y, pass, clen);
          clen += y;
        }
        break;
      
      case RULE_OP_DUP_SUFFIX:
      case 'Y':
        y = *rule++ - 1;
        if (clen > 0 && y <= clen) {
	  if ((clen+y) > MAXLINE)
	    y = MAXLINE - clen;
          memmove(pass + clen, pass + (clen - y), y);
          clen += y;
        }
        break;

      case RULE_OP_TITLE_SP:
      case 'E':
        for (z = x = 0; x < clen; x++) {
          c = pass[x];
          if (c == ' ')
            z = 0;
          else if (z == 0 && (c >= 'a' && c <= 'z')) {
            z = 1;
            pass[x] = c ^ 0x20;
          } else if (c >= 'A' && c <= 'Z')
            pass[x] = c ^ 0x20;
        }
        break;
      case RULE_OP_TITLE_SEP:
      case 'e':
	c1 = *rule++;
        for (z = x = 0; x < clen; x++) {
          c = pass[x];
          if (c == c1)
            z = 0;
          else if (z == 0 && (c >= 'a' && c <= 'z')) {
            z = 1;
            pass[x] = c ^ 0x20;
          } else if (c >= 'A' && c <= 'Z')
            pass[x] = c ^ 0x20;
        }
        break;

      case RULE_OP_TOGGLE_SEP:
      case '3':
        /* Hashcat RULE_OP_MANGLE_TOGGLE_AT_SEP — slow-path mirror
         * of the fast-path implementation. */
        y = *rule++ - 1;
        c1 = *rule++;
        {
          int toggle_next = 0;
          int occurrence  = 0;
          for (x = 0; x < clen; x++) {
            c = pass[x];
            if (c == c1) {
              if (occurrence == y) {
                toggle_next = 1;
              } else {
                occurrence++;
              }
              continue;
            }
            if (toggle_next) {
              if ((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z'))
                pass[x] = c ^ 0x20;
              break;
            }
          }
        }
        break;
    }
  }
app_exit:
  if (clen < 0)
    { _retval = (-1); goto _validate_exit; }
  pass[clen] = 0;
  /* fprintf(stderr,"final rule=%s len=%d pass=%s\n",orule,clen,pass);  */
  if (len != clen || lfastcmp(line, pass, clen) != 0)
    { _retval = (clen); goto _validate_exit; }
  _retval = -2;
  /* fall through to validator */

_validate_exit:
  if (validate) {
    int _i;
    int _rulen = (int) strlen(orule);
    int _outlen = (_retval >= 0) ? _retval : 0;
    fprintf(stderr, "VALIDATE word=");
    for (_i = 0; _i < len; _i++) fprintf(stderr, "%02x", (unsigned char) line[_i]);
    fprintf(stderr, " rulebytes=");
    for (_i = 0; _i < _rulen; _i++) fprintf(stderr, "%02x", (unsigned char) orule[_i]);
    fprintf(stderr, " retlen=%d outlen=%d output=", _retval, _outlen);
    for (_i = 0; _i < _outlen; _i++) fprintf(stderr, "%02x", (unsigned char) pass[_i]);
    fprintf(stderr, "\n");
  }
  return _retval;
}

/*
 * applyrules_gpu_pack — Apply rules in bulk and pack results directly
 * into GPU raw buffer slots (pre-padded hash blocks).
 *
 * Parameters:
 *   line       - original word (read-only)
 *   len        - original word length
 *   rules      - packed rule stream (concatenated: [uint16 len][packed][0x00] ...)
 *   nrules     - number of rules to process from the stream
 *   raw        - GPU raw buffer (stride * maxcount bytes)
 *   stride     - bytes per slot (64 for MD5/SHA1/SHA256, 128 for SHA384/SHA512)
 *   startidx   - first slot index to fill
 *   maxcount   - max slots available in raw buffer
 *   passlen    - per-slot password length array (for GPU hit reconstruction)
 *   ruleindex  - per-slot rule index array (for Ruleindex in GPU hits)
 *   passbuf    - scratch buffer for applyrule() (MAXLINE*3 bytes)
 *   cpu_needed - set to 1 if any rule produced a valid candidate that was
 *                too long for GPU. Caller should re-process the entire word
 *                through the CPU SIMD path to catch these.
 *   rules_used - set to number of rules consumed from the stream.
 *                Caller advances rule pointer by this count.
 *
 * Returns: number of GPU slots filled.
 *
 * Slots are pre-padded with 0x80 and bit-length for the target hash.
 * Rules that reject the word or produce unchanged output are silently
 * skipped — no per-rule tracking. If ANY valid candidate exceeds the
 * GPU length limit, *cpu_needed is set so the caller can re-process
 * the word on CPU (at negligible cost: the GPU-found hashes will have
 * their PV already decremented, so CPU re-finds are no-ops).
 */
int applyrules_gpu_pack(char *line, int len, char *rules, int nrules,
                        char *raw, int stride, int startidx, int maxcount,
                        uint16_t *passlen, int *ruleindex, char *passbuf,
                        int *cpu_needed, int *rules_used,
                        struct rule_workspace *ws)
{
    int i, idx, count = 0;
    char *rule = rules;
    /* Max password length per stride: MD5/SHA1/SHA256 (stride 64) = 55,
     * SHA384/SHA512 (stride 128) = 111. Formula: stride - 1(0x80) - 8(bitlen) */
    int maxpasslen = stride - 9;
    if (stride >= 128) maxpasslen = stride - 17;  /* 64-bit bitlen field */

    idx = startidx;
    for (i = 0; i < nrules && idx < maxcount; i++) {
        unsigned short rsize = *((unsigned short *)rule);
        if (rsize == 0) break;
        char *packed = rule + 2;

        int clen = applyrule(line, passbuf, len, packed, ws);
        rule += rsize + 2;

        if (clen <= 0)
            continue;  /* rejected or unchanged — skip silently */

        if (clen > maxpasslen) {
            *cpu_needed = 1;  /* valid but too long — flag for CPU re-process */
            continue;
        }

        /* Pack directly into GPU slot: aligned SIMD zero, copy, pad, bitlen.
         * raw buffer is 16-byte aligned (jobg struct layout guarantees this).
         *
         * 4× 16-byte stores cover the 64-byte slot; 8× covers the 128-byte
         * slot used by SHA-512 / SHA-384 / MD6 packing. Each backend uses
         * its native byte-vector type and explicit store intrinsics so
         * codegen does not depend on compiler treatment of typed-pointer
         * vector assignment. */
        char *slot = raw + (idx * stride);
#if defined(__SSE2__) || (defined(_MSC_VER) && (defined(_M_X64) || defined(_M_AMD64)))
        { __m128i z = _mm_setzero_si128();
          _mm_store_si128((__m128i *)(slot +  0), z);
          _mm_store_si128((__m128i *)(slot + 16), z);
          _mm_store_si128((__m128i *)(slot + 32), z);
          _mm_store_si128((__m128i *)(slot + 48), z);
          if (stride >= 128) {
            _mm_store_si128((__m128i *)(slot +  64), z);
            _mm_store_si128((__m128i *)(slot +  80), z);
            _mm_store_si128((__m128i *)(slot +  96), z);
            _mm_store_si128((__m128i *)(slot + 112), z);
          }
        }
#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
        { uint8x16_t z = vdupq_n_u8(0);
          vst1q_u8((uint8_t *)(slot +  0), z);
          vst1q_u8((uint8_t *)(slot + 16), z);
          vst1q_u8((uint8_t *)(slot + 32), z);
          vst1q_u8((uint8_t *)(slot + 48), z);
          if (stride >= 128) {
            vst1q_u8((uint8_t *)(slot +  64), z);
            vst1q_u8((uint8_t *)(slot +  80), z);
            vst1q_u8((uint8_t *)(slot +  96), z);
            vst1q_u8((uint8_t *)(slot + 112), z);
          }
        }
#elif defined(__VSX__)
        { __vector unsigned char z = vec_splats((unsigned char)0);
          vec_xst(z,   0, (unsigned char *)slot);
          vec_xst(z,  16, (unsigned char *)slot);
          vec_xst(z,  32, (unsigned char *)slot);
          vec_xst(z,  48, (unsigned char *)slot);
          if (stride >= 128) {
            vec_xst(z,  64, (unsigned char *)slot);
            vec_xst(z,  80, (unsigned char *)slot);
            vec_xst(z,  96, (unsigned char *)slot);
            vec_xst(z, 112, (unsigned char *)slot);
          }
        }
#elif defined(__ALTIVEC__)
        /* Pure Altivec without VSX: vec_st requires 16-byte alignment,
         * which the caller guarantees on this slot. */
        { __vector unsigned char z = vec_splats((unsigned char)0);
          vec_st(z,   0, (unsigned char *)slot);
          vec_st(z,  16, (unsigned char *)slot);
          vec_st(z,  32, (unsigned char *)slot);
          vec_st(z,  48, (unsigned char *)slot);
          if (stride >= 128) {
            vec_st(z,  64, (unsigned char *)slot);
            vec_st(z,  80, (unsigned char *)slot);
            vec_st(z,  96, (unsigned char *)slot);
            vec_st(z, 112, (unsigned char *)slot);
          }
        }
#else
        memset(slot, 0, stride);
#endif
        memcpy(slot, passbuf, clen);
        slot[clen] = (char)0x80;
        if (stride >= 128)
            ((uint32_t *)slot)[30] = clen * 8;
        else
            ((uint32_t *)slot)[14] = clen * 8;

        passlen[idx] = (uint16_t)clen;
        ruleindex[idx] = i;
        idx++;
        count++;
    }

    *rules_used = i;
    return count;
}

