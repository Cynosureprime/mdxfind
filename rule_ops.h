/* $Revision: 1.3 $
 *
 * $Log: rule_ops.h,v $
 * Revision 1.3  2026/09/13 02:38:38  dlr
 * Make ruleproc32 agree with ruleproc, and give procrule the means to prove it.
 *
 * The UTF-32 engine had drifted from the byte engine across many of the 2026-09-11
 * extensions. Bringing them together needed a way to SEE a divergence first, which
 * procrule could not do: generate mode drops a rejected rule, an empty candidate
 * and a candidate equal to its input, and each of those hides a real difference.
 *
 * procrule gains -A, audit mode: one line per word-and-rule pair with nothing
 * suppressed, rejected rules marked :REJECT, unchanged ones :SAME, empty ones
 * :EMPTY and malformed ones :BADRULE. The markers cannot collide with a candidate
 * because any candidate containing a colon is HEX-wrapped, which is also why -A
 * forces hex encoding on. Match mode is untouched. Note the option had to be added
 * to BOTH getopt strings: the file has a getopt call for AIX and a getopt_long call
 * for everyone else, and patching only the first one leaves the flag rejected at
 * runtime on every platform that matters.
 *
 * The UTF-32 bridge in procrule now speaks applyrule's return contract. The two
 * engines number their negatives INVERSELY -- byte -1 rejected, -2 output equals
 * input, -3 bad rule, against UTF-32 -1 bad input, -2 no room, -3 rejected -- and
 * the bridge collapsed all of them to -3. That is harmless where it is used, since
 * generate mode skips every negative alike and the rule-validity gates bypass the
 * bridge through validrule_u32, but it makes the engines impossible to compare: a
 * divergence report cannot tell a rejection from a malformed rule. The bridge also
 * now returns -2 for an unchanged candidate, which the UTF-32 engine has no code
 * for, so the same no-op rule no longer reads as two different outcomes.
 *
 * Shared, in rule_ops.h, which is included by exactly the two engines: the 15-entry
 * class membership table, the byte and codepoint match macros, rule_class_byte and
 * rule_verb_opcodes. All four were local to ruleproc.c. A class that means one
 * thing in one engine and another in the other is a silent wrong answer in
 * whichever one the caller did not test. The GPU kernels keep their own generated
 * copy of the table, which no kernel includes this header to get.
 *
 * THREE BUGS in ruleproc32, each a fix the byte engine had already made:
 *
 *   Extract guarded only the start position, so an overrunning count extracted
 *   whatever was available -- john behaviour, superseded by ruling. x14 on aB3 gave
 *   B3 where the oracle leaves the word alone. Omit, immediately below it, always
 *   had the full guard and a comment stating the requirement.
 *
 *   Memory-insert discarded its offset operand, which is the same bug ruleproc.c
 *   fixed on 2026-09-11, and it clamped the count instead of rejecting. MX042 on
 *   aB3 gave aBaB33 where the oracle rejects because offset plus count exceeds what
 *   memory holds, and X042 with no memory stored no-opped where the oracle rejects.
 *
 *   Insert tested greater-than where the oracle tests greater-or-equal, so
 *   inserting at a position equal to the length silently did nothing instead of
 *   appending. i1?d on the one-cluster word x gave xx against the oracle's
 *   x-question-x-question.
 *
 * CHARACTER CLASSES implemented in ruleproc32: all ten opcodes, both syntaxes, the
 * nine class-capable verbs and the tilde prefix, including the separate opcode for
 * hashcat title-with-class, which is a different algorithm rather than john's with
 * a class test substituted. Membership comes from the one shared table. Every class
 * rule now compiles to byte-identical bytecode in the two engines.
 *
 * POSITION RANGE: pos_of accepted only 0-9 and A-Z, stopping at 35, while
 * ruleproc.c has 62 positions. That refused EVERY rule with a lowercase position
 * operand across every positional verb, not just the 3NX case that surfaced it.
 * Note this is an mdxfind extension past hashcat, which rejects Ta outright; four
 * lines of best64 use one.
 *
 * UNKNOWN VERBS are now refused by the byte engine, per operator ruling: match
 * hcrule, which rejects. The old default arm emitted an unrecognised verb as a
 * literal byte, so M2 packed as memory-store followed by a bare 0x32 that the
 * walker then read as an opcode, and 0x32 is not one. hashcat reports it as an
 * invalid or unsupported rule.
 *
 * VALIDATION. Both engines now accept and refuse the same rules and produce the
 * same output: 80-rule sweep, 616 lines each, zero divergences, with both refusing
 * exactly B, M2 and the class form of the length reject. All 13 john class forms and
 * every tilde form agree. The byte engine is unmoved: its own audit output is
 * identical before and after the shared-header move, hashcat-oracle conformance
 * holds at 284 of 284 against hashcat master's own rule engine, mixguard and nulfix
 * pass, and best64, top_500 and the HashMob files all compile to the same counts as
 * before, so the tightening drops nothing real.
 *
 * NON-ASCII is characterised rather than equalised, because the engines are meant
 * to differ there: 158 of 264 pairs agree. Reverse differs on every word, the byte
 * engine emitting a reversed byte pair that is not valid UTF-8; truncation splits
 * characters mid-sequence; case mapping works only under -u, where eszett becomes
 * SS and Greek uppercases correctly; and length rejects count clusters rather than
 * bytes. Two results confirm decisions rather than design: the class verbs agree on
 * every non-ASCII word because membership is deliberately ASCII-bounded in both,
 * and duplication agrees byte for byte including keeping emoji sequences whole.
 *
 * KNOWN LIMITS, both recorded at the code: class membership above 0xff matches only
 * the ALL and HIGH classes, so purging lowercase does not touch Greek or Cyrillic
 * under -u, and the class verbs toggle case in ASCII only. Making either
 * Unicode-general would move -u away from the oracle rather than towards it, so both
 * are decisions to take separately. The compiled streams still differ in operand
 * encoding -- positions 1-based in the byte engine against 0-based here, the
 * multi-append opcodes, and prepend operand order -- which is the next piece.
 *
 * Revision 1.2  2026/09/11 19:47:25  dlr
 * Rule-engine parity block, character classes on GPU, 2-byte packed-word length, and two dispatch guards.
 *
 * Byte engine brought to parity with the documented john and hashcat feature sets. Character classes in both syntaxes: inline ?C uses the john table with complement-by-case-toggle, a ~ prefix selects the hashcat table. Nine verbs take a class, opcodes 0x80 to 0x89. ~e?C needs its own opcode 0x89 because hashcat class title-case is a different algorithm from the john form, not the same one with a class test substituted. The ?s class is hashcat class_sym in BOTH tables and john user classes ?0 to ?9 are not implemented, both per operator ruling. ?? is the literal-? escape, so purging a literal ? is now written @?? and 8 lines across the shipped rule files stop loading, every one of them a rule john also rejects.
 *
 * Other parity fixes in the same block: c C E and e act on position 0, not on the first alphabetic character, which john and hashcat agree on and mdxfind did not; x and X follow hashcat when out of range, superseding an earlier ruling for john; a candidate a rule empties is kept, following hashcat; T is bounds-checked as john does; X honours the memory offset N, which was read and then ignored so every offset gave the same answer; B is added from hashcat master. ruleproc32 refuses a class loudly rather than silently reinterpreting it, and accepts the ?? escape identically, closing a byte-versus-utf32 divergence where the same rule text produced different candidates in each engine with no diagnostic.
 *
 * All ten class opcodes plus =NX and %NX promoted from CPU-only to GPU-eligible and implemented in all six rules kernels, sharing one 480-byte constant-address-space membership table in gpu_common.cl and metal_common.metal so the six cannot drift. Three real rule files now have zero CPU-only rules.
 *
 * Packed-word length widened from one byte to two, little-endian, written and read byte by byte so it depends on neither host alignment nor host endianness. The admission gate allows GPU_RULES_MAX_INPUT_LEN 40959 while the wire field held 255, so any word of 256 to 40959 bytes was hashed as its first len mod 256 bytes: 285 as its first 29, 300 as its first 44. The emitted plaintext matched the emitted digest so nothing downstream could detect it. 11 kernel read sites across 7 files, 3 host writers, 2 host hit decoders and the buffer-full check move together.
 *
 * Two dispatch guards. C2.3 now tests the per-word word_packed_by_rules_engine rather than the thread-persistent my_jobg_rules, which differ whenever a word is walked but not packed. FastRule is disabled when this iteration is not the no-rule pass, making entered-at-the-rule-stream-origin a precondition rather than an assumption: the SIMD walker was re-draining the remaining rules from the first CPU-only rule output, so exactly one CPU-partition candidate survived per word.
 *
 * Validation: john class sweep 66 of 75 with all 9 differences the ruled empty-word policy, hashcat class sweep 21 of 21, X 22 of 22, x 16 of 16, c C E e 50 of 50 against john, rules32 conformance 165 of 165, zero silent byte-versus-utf32 divergences, wire round trip proven for all 40960 admitted lengths. Five GPU fixtures CPU equals GPU on a GTX 1080 and an M2 Max, including 113 of 113 on the fixture that measured 84 of 113 before the FastRule guard, and 126 of 126 on words from 1 to 4096 bytes. Regression on shipped rule files: Hash-IT_Crazy_Rules 6828885 rules, all_gj 208010 and T0XlC byte-identical.
 *
 * Also included in mdxfind.c and authored by Waffle, not by me: the AIX cmiyc challenge-3 algorithm validation comment block, recording the ppcemu emulated-oracle confirmation and the 504-hash corpus confirmation at the live parameters.
 *
 * Revision 1.1  2026/09/04 20:54:54  dlr
 * Initial revision
 *
 *
 * rule_ops.h -- the packed-rule opcode set, shared by both rule engines.
 *
 * Extracted verbatim from ruleproc.c so that ruleproc.c (bytes) and
 * ruleproc32.c (codepoints) cannot drift apart. Two copies of a 61-entry verb
 * table WILL diverge, and a divergence here is silent: a rule compiles under one
 * engine and means something else under the other. This session has twice paid
 * for exactly that shape of drift elsewhere in the tree.
 *
 * Values are unchanged and must stay unchanged: they are written into the
 * packed rule stream.
 *
 * NOTE the operand width differs between the engines even though the opcodes do
 * not. The byte engine stores an operand as one byte; the UTF-32 engine stores a
 * codepoint, which does not fit in one. The opcode VALUES are shared; the packed
 * stream layouts are not interchangeable.
 */

#ifndef RULE_OPS_H
#define RULE_OPS_H

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

/* hashcat 'B' RULE_OP_MANGLE_CHR_ADD: add the byte value of X to the byte at
 * position N, wrapping.  Implemented in hashcat master (rp_cpu.c
 * mangle_chr_add, include/types.h, and the device-conversion path in rp.c)
 * but NOT in the 6.2.5 release, and absent from John entirely.  Added
 * 2026-09-11 on the operator's ruling.  0xc0 was the only free value below
 * the existing 0xc1-0xfd band. */
#define RULE_OP_CHR_ADD     0xc0

/* ---- Character classes (D6, 2026-09-11) --------------------------------
 *
 * Nine verbs can take a character CLASS instead of a literal character.  Two
 * syntaxes are accepted and they are disjoint at the first byte of the verb,
 * so both references can be run verbatim:
 *
 *   John      inline    @?d   e?d   !?d   /?d   (?d   )?d   s?dX  =N?d  %N?d
 *   hashcat   ~ prefix  ~@?d  ~e?d  ~!?d  ~/?d  ~(?d  ~)?d  ~s?dX ~=N?d ~%N?d
 *
 * The PREFIX SELECTS THE CLASS TABLE as well as the syntax.  Unprefixed uses
 * John's letters with complement-by-case-toggle (?D is "not a digit");
 * ~-prefixed uses hashcat's six letters with no complement, so ~...?H is
 * uppercase hex where an unprefixed ?H would be a complement.  That single
 * collision is the reason for the split.
 *
 * `??` is John's escape for a literal `?`.  This means `@?` -- purge the
 * literal character `?` -- must now be written `@??`.  Operator's ruling,
 * 2026-09-11.  Note also the ruling that `?s` is hashcat's class_sym() in
 * BOTH tables (space plus every printable non-alphanumeric), not John's
 * narrower CHARS_SPECIALS.  John's user classes ?0-?9 are NOT implemented:
 * they would need a config file, which the safety spine forbids.
 *
 * Packed operand layouts (byte engine).  The class byte is never 0, so it
 * cannot be mistaken for the stream terminator:
 *
 *   0x80  op, class, Y        0x83..0x86  op, class
 *   0x81  op, class           0x87, 0x88  op, position, class
 *   0x82  op, class
 *
 * 0x80-0x88 were free: opcodes otherwise occupy raw printable ASCII and the
 * 0xc0-0xfd band, with 0xfe/0xff the multi-byte append/prepend.
 */
#define RULE_OP_SUB_CLASS       0x80   /* s?CY  replace class C with Y      */
#define RULE_OP_PURGE_CLASS     0x81   /* @?C   purge class C               */
#define RULE_OP_TITLE_CLASS     0x82   /* e?C   title case, class C as sep  */
#define RULE_OP_REJ_HAS_CLASS   0x83   /* !?C   reject if contains class C  */
#define RULE_OP_REJ_NHAS_CLASS  0x84   /* /?C   reject unless contains C    */
#define RULE_OP_REJ_FIRST_CLASS 0x85   /* (?C   reject unless first in C    */
#define RULE_OP_REJ_LAST_CLASS  0x86   /* )?C   reject unless last in C     */
#define RULE_OP_REJ_AT_CLASS    0x87   /* =N?C  reject unless pos N in C    */
#define RULE_OP_REJ_CNT_CLASS   0x88   /* %N?C  reject unless N or more in C*/
#define RULE_OP_TITLE_CLASS_HC  0x89   /* ~e?C  hashcat's title-with-class  */

/* Why `e?C` needs TWO opcodes.  hashcat's class form is not its literal form
 * with a class test substituted -- mangle_title_sep_class_l() in rp_cpu.c is
 * a different algorithm.  It case-normalises the separator itself (lowercases
 * every position, then uppercases the ones that follow a class member),
 * whereas the literal mangle_title_sep() leaves the separator alone and ends
 * with an unconditional uppercase of position 0.  John's `e?C` follows the
 * literal shape.  Measured on six words: John and mdxfind agree on all six
 * for `e?l`, hashcat differs on four.  So `e?C` is John and `~e?C` is
 * hashcat -- the prefix selects the reference, not merely the class table.
 */

/* Class ids, as stored in the packed stream.  Dense 1..15 so the bitmap
 * table indexes as (byte & RULE_CLASS_MASK) - 1.  RULE_CLASS_NOT is the
 * complement bit; ids start at 1 so a class operand is never a NUL byte. */
#define RULE_CLASS_NOT   0x80
#define RULE_CLASS_MASK  0x7f
#define RULE_CLASS_LOWER  1    /* ?l  a-z                                  */
#define RULE_CLASS_UPPER  2    /* ?u  A-Z                                  */
#define RULE_CLASS_DIGIT  3    /* ?d  0-9                                  */
#define RULE_CLASS_SYM    4    /* ?s  hashcat class_sym(): ' ' + specials  */
#define RULE_CLASS_LHEX   5    /* ~?h 0-9a-f        (hashcat table only)   */
#define RULE_CLASS_UHEX   6    /* ~?H 0-9A-F        (hashcat table only)   */
#define RULE_CLASS_VOWEL  7    /* ?v  aeiouAEIOU    (John table only)      */
#define RULE_CLASS_CONS   8    /* ?c  consonants    (John table only)      */
#define RULE_CLASS_WS     9    /* ?w  space, tab    (John table only)      */
#define RULE_CLASS_PUNCT 10    /* ?p  .,:;'"?!`     (John table only)      */
#define RULE_CLASS_ALPHA 11    /* ?a  a-zA-Z        (John table only)      */
#define RULE_CLASS_ALNUM 12    /* ?x  a-zA-Z0-9     (John table only)      */
#define RULE_CLASS_CTRL  13    /* ?o  control bytes (John table only)      */
#define RULE_CLASS_HIGH  14    /* ?b  0x80-0xff     (John table only)      */
#define RULE_CLASS_ALL   15    /* ?z, ?y            (John table only)      */
#define RULE_CLASS_COUNT 15

/* Class membership, shared by BOTH rule engines.
 *
 * This table and its test macros were local to ruleproc.c.  They are here so
 * the byte engine and the UTF-32 engine cannot drift: a class that means one
 * thing in ruleproc.c and another in ruleproc32.c is a silent wrong answer in
 * whichever one the caller did not test.  rule_ops.h is included by exactly
 * those two translation units, so `static const` costs one copy each and no
 * other TU sees an unused object.
 *
 * The GPU kernels carry their own 480-byte copy in the constant address space
 * of gpu_common.cl / metal_common.metal.  That one is generated to be
 * byte-identical and is validated by the classfix fixture; it is deliberately
 * not shared through this header, which no kernel includes.
 *
 * The complement bit is applied by XOR at test time, so there is no second
 * table and a complemented class costs nothing extra.
 */
static const unsigned char rule_class_bits[RULE_CLASS_COUNT][32] = {
  /* 1  LOWER  26 members */ {0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0xfe,0xff,0xff,0x07,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00},
  /* 2  UPPER  26 members */ {0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0xfe,0xff,0xff,0x07,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00},
  /* 3  DIGIT  10 members */ {0x00,0x00,0x00,0x00,0x00,0x00,0xff,0x03,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00},
  /* 4  SYM    33 members */ {0x00,0x00,0x00,0x00,0xff,0xff,0x00,0xfc,0x01,0x00,0x00,0xf8,0x01,0x00,0x00,0x78,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00},
  /* 5  LHEX   16 members */ {0x00,0x00,0x00,0x00,0x00,0x00,0xff,0x03,0x00,0x00,0x00,0x00,0x7e,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00},
  /* 6  UHEX   16 members */ {0x00,0x00,0x00,0x00,0x00,0x00,0xff,0x03,0x7e,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00},
  /* 7  VOWEL  10 members */ {0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x22,0x82,0x20,0x00,0x22,0x82,0x20,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00},
  /* 8  CONS   42 members */ {0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0xdc,0x7d,0xdf,0x07,0xdc,0x7d,0xdf,0x07,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00},
  /* 9  WS      2 members */ {0x00,0x02,0x00,0x00,0x01,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00},
  /* 10 PUNCT   9 members */ {0x00,0x00,0x00,0x00,0x86,0x50,0x00,0x8c,0x00,0x00,0x00,0x00,0x01,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00},
  /* 11 ALPHA  52 members */ {0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0xfe,0xff,0xff,0x07,0xfe,0xff,0xff,0x07,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00},
  /* 12 ALNUM  62 members */ {0x00,0x00,0x00,0x00,0x00,0x00,0xff,0x03,0xfe,0xff,0xff,0x07,0xfe,0xff,0xff,0x07,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00},
  /* 13 CTRL   47 members */ {0xfe,0xfd,0xff,0xff,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x80,0x30,0xe1,0xc1,0xfd,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00},
  /* 14 HIGH  128 members */ {0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff},
  /* 15 ALL   256 members */ {0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff},
};

/* Does byte CH belong to the class named by packed class byte CB? */
#define RULE_CLASS_MATCH(cb, ch) \
  ((((rule_class_bits[((cb) & RULE_CLASS_MASK) - 1][(unsigned char)(ch) >> 3] \
      >> ((ch) & 7)) ^ ((cb) >> 7)) & 1) != 0)

/* Does CODEPOINT cp belong to the class named by packed class byte CB?
 *
 * For cp <= 0xff this is exactly the byte test above, which is what keeps the
 * two engines in agreement wherever the byte engine has meaningful behaviour
 * at all -- it sees UTF-8 BYTES, so a class test on a multi-byte character is
 * a test on its individual bytes.
 *
 * Above 0xff the byte engine has no behaviour to match, so the table's own
 * intent is followed: ALL matches everything, and HIGH means "not ASCII" and
 * so matches too.  Every other class is ASCII-defined and does not match.
 * CONSEQUENCE, and it is a real limitation rather than an oversight: `@?l`
 * does not purge Greek or Cyrillic lowercase under -u.  Making the classes
 * Unicode-general is a separate decision from engine parity, and would move
 * -u AWAY from the byte engine rather than towards it.
 */
/* Map a class letter to its packed class byte, or 0 if the letter names no
 * class in the requested table.  Shared by both engines.
 *   table 0 -- John: inline ?C, complement by case-toggling the letter.
 *   table 1 -- hashcat: ~-prefixed, six classes, no complement (so ?H is
 *              uppercase hex here and "not hex" there -- the one collision
 *              that forces the tables apart).
 */
static inline unsigned char rule_class_byte(char letter, int table) {
  unsigned char id = 0, neg = 0;
  char lc = letter;

  if (table) {
    switch (letter) {
      case 'l': return RULE_CLASS_LOWER;
      case 'u': return RULE_CLASS_UPPER;
      case 'd': return RULE_CLASS_DIGIT;
      case 's': return RULE_CLASS_SYM;
      case 'h': return RULE_CLASS_LHEX;
      case 'H': return RULE_CLASS_UHEX;
      default:  return 0;
    }
  }

  if (lc >= 'A' && lc <= 'Z') { lc = (char)(lc | 0x20); neg = RULE_CLASS_NOT; }
  switch (lc) {
    case 'l': id = RULE_CLASS_LOWER; break;
    case 'u': id = RULE_CLASS_UPPER; break;
    case 'd': id = RULE_CLASS_DIGIT; break;
    case 's': id = RULE_CLASS_SYM;   break;
    case 'v': id = RULE_CLASS_VOWEL; break;
    case 'c': id = RULE_CLASS_CONS;  break;
    case 'w': id = RULE_CLASS_WS;    break;
    case 'p': id = RULE_CLASS_PUNCT; break;
    case 'a': id = RULE_CLASS_ALPHA; break;
    case 'x': id = RULE_CLASS_ALNUM; break;
    case 'o': id = RULE_CLASS_CTRL;  break;
    case 'b': id = RULE_CLASS_HIGH;  break;
    /* John registers ?Z as the empty set, and ?Y as empty when no codepage
     * is configured, so their lowercase complements are both "everything".
     * The byte engine has no codepage, so that is the branch we match. */
    case 'z': id = RULE_CLASS_ALL;   break;
    case 'y': id = RULE_CLASS_ALL;   break;
    default:  return 0;
  }
  return (unsigned char)(id | neg);
}

static inline int rule_verb_opcodes(char v, unsigned char *plain, unsigned char *cls)
{
  /* NOTE: 'e' is overridden by the caller for the ~ form -- see
   * RULE_OP_TITLE_CLASS_HC in rule_ops.h for why the two differ. */
  switch (v) {
    case '@': *plain = RULE_OP_PURGE;      *cls = RULE_OP_PURGE_CLASS;     return 1;
    case 'e': *plain = RULE_OP_TITLE_SEP;  *cls = RULE_OP_TITLE_CLASS;     return 1;
    case '!': *plain = RULE_OP_REJ_HAS;    *cls = RULE_OP_REJ_HAS_CLASS;   return 1;
    case '/': *plain = RULE_OP_REJ_NHAS;   *cls = RULE_OP_REJ_NHAS_CLASS;  return 1;
    case '(': *plain = RULE_OP_REJ_FIRST;  *cls = RULE_OP_REJ_FIRST_CLASS; return 1;
    case ')': *plain = RULE_OP_REJ_LAST;   *cls = RULE_OP_REJ_LAST_CLASS;  return 1;
    case 's': *plain = RULE_OP_SUB;        *cls = RULE_OP_SUB_CLASS;       return 1;
    case '=': *plain = (unsigned char)'='; *cls = RULE_OP_REJ_AT_CLASS;    return 1;
    case '%': *plain = (unsigned char)'%'; *cls = RULE_OP_REJ_CNT_CLASS;   return 1;
    default:  return 0;
  }
}

#define RULE_CLASS_MATCH_CP(cb, cp)                                         \
  ( ( ( ((uint32_t)(cp) <= 0xffu)                                           \
        ? (unsigned)((rule_class_bits[((cb) & RULE_CLASS_MASK) - 1]         \
                                     [(unsigned)(cp) >> 3]                  \
                      >> ((cp) & 7)) & 1u)                                  \
        : (unsigned)(((((cb) & RULE_CLASS_MASK) == RULE_CLASS_ALL) ||       \
                      (((cb) & RULE_CLASS_MASK) == RULE_CLASS_HIGH))        \
                     ? 1u : 0u)                                             \
      ) ^ (unsigned)((cb) >> 7) ) & 1u )

#endif
