/* $Revision: 1.4 $
 *
 * $Log: ruleproc32.h,v $
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

/* True if cp is a value UTF-8 can represent: a scalar, not a surrogate. */
#define UTF32_ENCODABLE(cp) ((cp) <= 0x10FFFFu && ((cp) < 0xD800u || (cp) > 0xDFFFu))

#endif
