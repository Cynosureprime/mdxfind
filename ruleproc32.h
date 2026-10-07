/* $Revision: 1.6 $
 *
 * $Log: ruleproc32.h,v $
 * Revision 1.6  2026/10/05 03:43:33  dlr
 * Declare the encoding conversions added to ruleproc32.c, with the contract and the reasons in one place: a length or a negative RULE32_ERR_, never a partial result; the caller owns the UTF-32 scratch because these run per candidate and a MAXLINE local is not an option; and the three defects of asking iconv instead -- silent dropping under //IGNORE, a return value glibc and macOS libiconv disagree about, and a static link that dlopens gconv modules and segfaults on a host with a different glibc. Documents that utf16_to_utf8 refuses an unpaired surrogate rather than emitting CESU-8, and treats a dropped codepoint from utf32_to_utf8 as an error, since a drop there would mean the caller conversion was unsound.
 *
 * Revision 1.5  2026/09/28 11:37:15  dlr
 * Move the large thread-local rule scratch buffers from static TLS to the heap. A static __thread array does not get a buffer off the thread stack, which is what the comments at these sites believed: glibc carves the static TLS block out of the same allocation as the thread stack and replicates it into every thread, including threads created inside a dlopen library. mdxfind carried 6,881,648 bytes of .tbss, 95 percent of it eight uint32_t arrays of RULE32_MAXCP at 819,200 bytes each, and AMD fglrx 1573.4 clCreateCommandQueue hung forever on gp1 because its internal helper thread could not be created; a ballast harness bisects the threshold to between 131,072 and 262,144 bytes, and hashcat and the gbench harness ran on the same card minutes apart because their TLS is small or absent. Each buffer now keeps a thread-local pointer and allocates once per thread on first use through the new RULE32_TLS_SCRATCH in ruleproc32.h, using calloc so the zeroing matches .tbss semantics of zeroed once per thread rather than per call, never freed because worker threads live for the run exactly as the TLS block did, and a loud exit naming file and line on allocation failure. Measured .tbss falls from 6,881,648 to 41,256 bytes on the Linux GPU build and thread_bss from 5,120,216 to 41,152 on the non-GPU build; the residue is cached, deliberately left as an array because it is under the threshold and carries sizeof uses. mdxfind now initialises the Tahiti GPU in 2.0 seconds and cracks on it, including a two-digit mask matching the CPU exactly, where every earlier build hung. procrule -8 against mdxfind -8 gives 10,946 of 10,947 candidates both before and after, the one difference being the known Turkish dotless-i locale variant.
 *
 * Revision 1.4  2026/09/05 01:10:04  dlr
 * Opcodes for the string forms of insert-every-N, toggle-after-separator and title-case-on-separator.
 *
 * Revision 1.3  2026/09/05 00:48:01  dlr
 * Opcodes and the operand contract for quoted string operands.
 *
 * Revision 1.2  2026/09/04 23:04:01  dlr
 * Fix the include guard, add the collapsed append and prepend opcodes. The ifndef around the fallback MAXLINE define had its endif fifty lines later, so it swallowed the packrule32 and applyrule32 declarations. The standalone test tools never define MAXLINE and always compiled the block, so it was invisible until procrule, which does define it, failed to see the declarations at all. The multi-append and multi-prepend opcodes live here rather than in rule_ops.h because that header is shared verbatim with ruleproc.c and is proven token-for-token identical between the two engines, while the packed stream is this engine own.
 *
 * Revision 1.1  2026/09/04 20:54:43  dlr
 * Initial revision
 *
 *
 * ruleproc32.h -- UTF-32 rule engine for procrule.
 *
 * A SIBLING of ruleproc.c, not a replacement. The byte engine stays exactly as
 * it is: some formats need byte semantics, and the $HEX[] round trip already
 * works. This path is selected by a flag and runs the WHOLE pipeline in
 * codepoint space -- input assumed UTF-8, decoded to UTF-32, rules applied over
 * uint32_t, re-encoded to UTF-8 on output.
 *
 * Why a separate engine rather than widening applyrule(): that function is 956
 * lines with 42 SSE intrinsic uses, and those vectorise 16 BYTES per
 * instruction. Widening them to uint32_t is not a retrofit, it is a different
 * program, and the byte path would be put at risk for no gain.
 */

#ifndef RULEPROC32_H
#define RULEPROC32_H

#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>

/* ---- buffer sizing ----
 *
 * MAXLINE is 40KB and must agree with mdxfind.h. It already disagreed once:
 * procrule.c held 20KB while ruleproc.c bounded against mdxfind.h's 40KB, so
 * applyrule() wrote up to 40KB into a 20KB buffer and corrupted the heap
 * (procrule.c 1.19). The sizes below are derived from MAXLINE rather than
 * written out, so that cannot recur here.
 */
#ifndef MAXLINE
#define MAXLINE (40*1024)
#endif
/* ---- packed rule stream ----
 *
 * A uint32_t stream: opcode, then its operands, each one word. The byte engine
 * packs an operand into a single byte, which cannot hold a codepoint -- that
 * difference is the whole point of this engine, and is why the two packed
 * layouts are NOT interchangeable even though the opcode VALUES are shared.
 *
 * One consequence is the feature today's evidence asked for: `$X` carries its
 * operand as a codepoint, so appending an emoji is ONE rule rather than four
 * chained byte appends, and it composes with the duplication verbs.
 */
#define RULE32_END 0u          /* terminates a packed rule */
#define RULE32_OP_SUB_STR   0x1002u  /* substitute one STRING for another */
#define RULE32_OP_PURGE_STR 0x1003u  /* remove every occurrence of a STRING */
#define RULE32_OP_INSERT_STR 0x1004u /* insert a STRING at a cluster position */
#define RULE32_OP_OVERW_STR  0x1005u /* replace a cluster with a STRING */
#define RULE32_OP_REJ_HAS_STR   0x1006u
#define RULE32_OP_REJ_NHAS_STR  0x1007u
#define RULE32_OP_REJ_FIRST_STR 0x1008u
#define RULE32_OP_REJ_LAST_STR  0x1009u
#define RULE32_OP_DIV_INS_STR   0x100Au /* insert a STRING every N clusters */
#define RULE32_OP_TOGGLE_SEP_STR 0x100Bu /* toggle after the Nth STRING */
#define RULE32_OP_TITLE_SEP_STR 0x100Cu /* title-case after each STRING */
#define RULE32_QUOTE 0x0022u  /* the operand delimiter */

/* Collapsed append/prepend runs. `$1$2$3` is ONE op with a count and three
 * codepoints, mirroring what the byte engine does with its 0xff/0xfe opcodes.
 * Without it the UTF-32 engine scaled linearly in the length of an append
 * chain while the byte engine stayed flat -- 24 appends per rule cost 3.6x
 * the byte engine where a single append cost 2.1x, and chains of year and
 * suffix appends are everywhere in real rule files.
 *
 * These live HERE and not in rule_ops.h on purpose: that header is shared
 * verbatim with ruleproc.c and is proven token-for-token identical between
 * the two engines. The packed stream, by contrast, is this engine's own, so
 * the values only have to avoid the byte engine's opcode range. */
#define RULE32_OP_APPEND_MULTI  0x1000u
#define RULE32_OP_PREPEND_MULTI 0x1001u

/* Compile one rule line (already decoded to UTF-32) into packed form.
 * Returns the number of words written, or RULE32_ERR_INVALID if the rule is
 * malformed or uses a verb not yet implemented. */
/* Quoted operands. An operand may be written as a quoted STRING rather than a
 * single codepoint: $"123" appends three characters in one op, s"X""Y"
 * substitutes one string for another, and an operand can be something with no
 * single-codepoint spelling at all -- a Devanagari conjunct, a Yoruba vowel
 * carrying two marks, an emoji joiner sequence. A doubled quote is one literal.
 *
 * Always on in this engine. A double quote is a legitimate literal operand in
 * byte-engine rules -- 21 times in 100,000 real ones -- so a few of those read
 * differently here. That is deliberate and in keeping with the rest of the
 * mode, which already differs on shifts, on what a position counts, and on
 * uppercasing an eszett. An unterminated quote is a rule error, reported.
 */
int packrule32(const uint32_t *line, int linelen, uint32_t *out, int outmax);

/* Apply a packed rule to a candidate.
 *
 * Returns the new length in codepoints, or negative on error. `out` must have
 * room for RULE32_MAXCP.
 *
 * VARIANTS. Some case mappings are locale-ambiguous, so one rule/input pair can
 * have more than one correct answer. Turkish and Azeri uppercase i to U+0130
 * and lowercase I to U+0131, where every other locale gives I and i. There is
 * no way to know from the candidate which system stored the password, and
 * picking one silently loses the other -- so the engine emits BOTH, and the
 * caller runs the rule once per variant.
 *
 *   int nv;
 *   for (int v = 0; ; v++) {
 *       int len = applyrule32(rule, in, inlen, out, outmax, v, &nv);
 *       ... emit ...
 *       if (v + 1 >= nv) break;
 *   }
 *
 * Pass variant 0 first; *nvariants is then the number of distinct outputs for
 * THIS rule and THIS input. It is 1 unless an ambiguous mapping was actually
 * reached, so an ASCII word does not double its output for nothing.
 *
 * Locale is a property of the storing SYSTEM, not of a character, so a Turkish
 * reading applies to every i in the word at once. That keeps the count at 2
 * rather than 2^n for a word with n ambiguous characters -- the distinction
 * matters, because per-character permutation would explode on real input.
 *
 * *nvariants may be NULL if the caller only wants variant 0. */
int applyrule32(const uint32_t *rule, const uint32_t *in, int inlen,
                uint32_t *out, int outmax, int variant, int *nvariants);


/* Working buffer, in CODEPOINTS. A MAXLINE-byte UTF-8 line decodes to at most
 * MAXLINE codepoints, since no codepoint occupies less than one byte -- so the
 * 5x is not for the decode, it is headroom for rules that GROW the buffer:
 * d and Z duplicate, p repeats, reflect doubles. Operator's call, 2026-09-04. */
#define RULE32_MAXCP    (5 * MAXLINE)

/* Thread-local scratch that lives on the HEAP, not in the static TLS block.
 *
 * A `static __thread T x[N]` does NOT get the array off the thread stack, which
 * is what the comments at these sites believed. glibc carves the static TLS
 * block out of the SAME allocation as the thread stack, and replicates the whole
 * PT_TLS into every thread -- including threads created inside dlopen'd
 * libraries. mdxfind carried 6,881,648 bytes of .tbss this way, 95% of it eight
 * uint32_t[RULE32_MAXCP] arrays, and fglrx 1573.4's clCreateCommandQueue
 * deadlocked because its internal helper thread could not be created above a
 * threshold measured between 128 KB and 256 KB (gp1, 2026-09-27).
 *
 * Keep the POINTER in TLS (8 bytes) and the storage on the heap. Allocated once
 * per thread on first use and deliberately never freed: worker threads live for
 * the run, exactly as the TLS block did, so the lifetime is unchanged. calloc
 * also matches .tbss semantics -- zeroed once per thread, not per call. */
#define RULE32_TLS_SCRATCH(ptr, nelem)                                         \
    do {                                                                       \
        if (__builtin_expect((ptr) == NULL, 0)) {                              \
            (ptr) = calloc((size_t)(nelem), sizeof(*(ptr)));                    \
            if ((ptr) == NULL) {                                               \
                fprintf(stderr, "%s:%d: calloc of %zu bytes for " #ptr         \
                        " failed\n", __FILE__, __LINE__,                       \
                        (size_t)(nelem) * sizeof(*(ptr)));                     \
                exit(1);                                                       \
            }                                                                  \
        }                                                                      \
    } while (0)

/* Byte buffer able to hold ANY encodable working buffer. A codepoint takes at
 * most 4 bytes in UTF-8, so this is 4x the codepoint capacity -- 800KB against
 * the working buffer's 800KB. Encoding a full working buffer into a MAXLINE
 * byte buffer would overflow it by 20x, which is precisely the 1.19 defect in
 * new clothes; size output buffers with THIS, not with MAXLINE. */
#define RULE32_MAXBYTES (4 * RULE32_MAXCP)

/* Compile-time guard on the relationship the two constants must keep. If the
 * derivation above is ever edited so a full working buffer no longer fits its
 * byte buffer, the build fails here rather than the heap failing at runtime. */
typedef char rule32_sizing_check[(RULE32_MAXBYTES >= 4 * RULE32_MAXCP) ? 1 : -1];

/* Decode UTF-8 to UTF-32.
 *
 * STRICT. Returns the codepoint count on success, or a negative code.
 *
 * The two failures are kept DISTINCT on purpose. RULE32_ERR_INVALID is a
 * property of the data and means this line dies, which is a normal outcome.
 * RULE32_ERR_NOROOM means the caller sized a buffer wrongly, which is a bug in
 * the program. Returning one value for both would let a silently truncated
 * candidate present as a rejected line -- the exact class of quiet wrong answer
 * this engine exists to remove. Invalid input dies here rather than being escaped: after a rule has
 * duplicated, reversed or truncated the buffer there is no way to identify
 * which codepoints stood for raw bytes, so a round trip cannot be honoured and
 * promising one would be worse than refusing.
 *
 * Rejects, all of which a permissive decoder would silently accept:
 *   - overlong forms (0xC0 0x80 and friends)
 *   - surrogates D800-DFFF encoded as three bytes (CESU-8)
 *   - scalars above U+10FFFF
 *   - lead bytes 0xC0, 0xC1, 0xF5-0xFF, which are never valid
 *   - truncated sequences and stray continuation bytes
 */
#define RULE32_ERR_INVALID (-1)   /* input is not well-formed UTF-8: line dies */
#define RULE32_ERR_NOROOM  (-2)   /* output buffer too small: caller bug */
#define RULE32_REJECTED    (-3)   /* a rejection rule fired: emit nothing */

int utf8_to_utf32(const unsigned char *in, int inlen, uint32_t *out, int outmax);

/* Encode UTF-32 to UTF-8.
 *
 * Returns the byte count written, or RULE32_ERR_NOROOM if the output buffer is
 * too small. There is no RULE32_ERR_INVALID here: an unencodable codepoint is
 * dropped rather than failing the line.
 *
 * Ordinary verbs cannot produce an unencodable value: with valid input and no
 * escaping every codepoint in the buffer is already a valid scalar. The only
 * source of one is a verb doing ARITHMETIC on a codepoint -- BIT_SHL, BIT_SHR,
 * INC, DEC, SUB -- pushing it past U+10FFFF or into the surrogate range. Those
 * codepoints are DROPPED and the rest of the line is emitted, per the operator's
 * rule: convert what is possible, and emit something for every line unless the
 * result is empty. *ndropped receives the count so a caller can report it; pass
 * NULL if not wanted.
 */
int utf32_to_utf8(const uint32_t *in, int inlen, unsigned char *out, int outmax,
                  int *ndropped);

/* ---- Encoding conversions ---------------------------------------------
 *
 * These exist so that nothing outside this file has to decide what
 * "well-formed UTF-8" means.  mdxfind previously asked iconv, which was a
 * mistake in three ways: //IGNORE silently DROPS what it cannot convert, so a
 * malformed candidate was shortened and hashed rather than refused; glibc and
 * macOS libiconv disagree about what //IGNORE returns, so the same candidate
 * was accepted on one platform and skipped on the other; and linking glibc's
 * iconv statically dlopens gconv modules, which segfaults on a host with a
 * different glibc while passing every test on the build host.
 *
 * Every function here returns a LENGTH or a negative RULE32_ERR_*.  Never a
 * partial result.  The decode is utf8_to_utf32() verbatim, so these agree with
 * the -8 rule engine on every boundary condition by construction rather than
 * by inspection: lone continuations, 0xC0/0xC1, 0xF5-0xFF, truncation,
 * overlongs, surrogates and anything above U+10FFFF.
 *
 * The caller owns the uint32_t scratch: these run per candidate, so a
 * MAXLINE-sized local is not an option.  It must hold at least as many entries
 * as the input could yield codepoints.
 */

#define UTF16_LE 0
#define UTF16_BE 1

/* UTF-8 -> UTF-16.  Returns the byte count written (2 per BMP codepoint, 4 per
 * astral one, which becomes a surrogate pair), RULE32_ERR_INVALID for
 * ill-formed input, or RULE32_ERR_NOROOM. */
int utf8_to_utf16(const unsigned char *in, int inlen,
                  unsigned char *out, int outmax,
                  uint32_t *scratch, int scratchmax, int big_endian);

/* UTF-16 -> UTF-8.  Pairs surrogates; an unpaired one is RULE32_ERR_INVALID
 * rather than CESU-8.  An odd input length is RULE32_ERR_INVALID.  Unlike
 * utf32_to_utf8 this NEVER drops: a dropped codepoint here would mean the
 * caller's own conversion was unsound, so it is reported. */
int utf16_to_utf8(const unsigned char *in, int inlen,
                  unsigned char *out, int outmax,
                  uint32_t *scratch, int scratchmax, int big_endian);

/* Single-byte code page -> UTF-16.  Every byte yields exactly one code unit,
 * so the output is exactly 2*inlen bytes and there is no error but NOROOM.
 * Windows mapped the user's code page into UTF-16LE through its own tables; it
 * did NOT zero-extend, and the two agree only where the code points happen to
 * equal the byte values. */
#define CP_1251 1251
#define CP_1252 1252
int cp_to_utf16(const unsigned char *in, int inlen,
                unsigned char *out, int outmax, int codepage, int big_endian);

/* UTF-8 -> UTF-7 (RFC 2152), reproducing byte for byte what SHA1UTF7 has
 * always hashed.  Two details of that are not free choices:
 *
 *  - Set O -- ! " # $ % & * ; < = > @ [ ] ^ _ ` { | } -- is ENCODED, not passed
 *    through, which RFC 2152 permits either way.
 *  - a base64 run is ALWAYS closed with '-'.  The caller used to append an 'X'
 *    to force iconv to flush its shift state, and since X is itself a base64
 *    character the terminator was always emitted into what got hashed.  The
 *    'X' trick is no longer needed: this flushes because it is told to.
 *
 * Consecutive non-direct codepoints share one run, astral ones go in as a
 * UTF-16 surrogate pair, and a literal '+' becomes "+-".
 * Returns the byte count, or a negative RULE32_ERR_*.
 */
int utf8_to_utf7(const unsigned char *in, int inlen, char *out, int outmax,
                 uint32_t *scratch, int scratchmax);

/* Do two code pages map every byte of this buffer to the same codepoint?  The
 * UTF-16 form is a pure per-byte map, so this is exactly the condition "the two
 * conversions would compare equal" -- answered without performing the second
 * one, so a redundant conversion and hash are never done rather than done and
 * discarded.  CP1251 and CP1252 agree on 159 of 256 byte values. */
int cp_tables_agree(const unsigned char *in, int inlen, int cpa, int cpb);

/* True if cp is a value UTF-8 can represent: a scalar, not a surrogate. */
#define UTF32_ENCODABLE(cp) ((cp) <= 0x10FFFFu && ((cp) < 0xD800u || (cp) > 0xDFFFu))

#endif
