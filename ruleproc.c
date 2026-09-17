#include <stdio.h>

#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <sys/types.h>
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

static char *Version __attribute__((unused)) = "$Header: /Users/dlr/src/mdfind/RCS/ruleproc.c,v 1.46 2026/09/17 05:23:51 dlr Exp dlr $";
/*
 * $Log: ruleproc.c,v $
 * Revision 1.46  2026/09/17 05:23:51  dlr
 * Drop the MDXFIND_RULE_VALIDATOR gate; the rule validator is never enabled. This file is shared with procrule, which was rebuilt and checked: both rule engines still behave, the byte engine reversing a non-ASCII word producing mojibake and the UTF-32 engine producing droewssap with correct Unicode case mapping.
 *
 * Revision 1.45  2026/09/14 13:03:29  dlr
 * Make R a LOGICAL right shift, and remove L undefined behaviour. R shifted a plain char, which is signed on x86 and on Apple ARM64, so a bare >> 1 sign-extended: byte 0x83 became 0xc1 rather than 0x41. These are BYTE strings and a 1-bit must not appear at the top of a byte that had none. Operator ruling 2026-09-14: logical is correct, and hashcat is wrong here rather than mdxfind being incompatible -- hashcat rp_cpu.c mangle_chr_shiftr is arr[upos] >>= 1 on char arr[], the same defect, but it never exercises it because it works in ASCII where the sign bit is never set. Do not restore compatibility. Found by disagreement with the OpenCL rules kernel, which shifts an unsigned type and was right all along: on 2,009 encoding-diverse words the two engines differed on 182 of 28,440 candidates, every one from the single rule R0 and every one with a first byte at or above 0x80, and on a hash list carrying the device digest mdxfind emitted a hash:plaintext pair that did not verify -- a false positive, because the device computed one candidate and the host replay reconstructed another. The old code also disagreed with itself across the fleet, since plain char is unsigned by default on Linux aarch64. The unsigned cast states the intent instead of inheriting it from the ABI. L gets the same cast for a different reason: << 1 on a negative signed char is undefined behaviour, and the value is unchanged on every platform we build, so that half is a UB fix with no semantic effect. The UTF-32 engine needs no change -- it shifts a uint32_t codepoint, so it was already logical, and a value pushed past U+10FFFF or into the surrogate block is dropped by utf32_to_utf8, which the operator confirms is the acceptable outcome for an operator at this level. tools/ruletests/hcnorm could not have caught this: it has zero R operators, so the one fixture whose oracle is hashcat own code does not exercise the opcode.
 *
 * Revision 1.44  2026/09/13 23:50:40  dlr
 * Gate the third compiler diagnostic site with Rulequiet. positiontranslate printed "Invalid position %c in rules" ungated -- the other two live in packrules_len and the rest route through rule_error, both already gated in 1.43. Under -8 packrules is run on every rule purely to discover which rules it refuses, because that refusal IS the UTF-32-REQUIRED tag, so this site has to be as quiet as the others. It surfaced once procrule 1.34 restored the line terminator packrules expects: that moved where the compiler gives up on a quoted operand from a gated site to this one, and the dispatch8 fixture went from 8 of 8 to a failure on the -R unreachable count. This is also the only diagnostic in the file with no trailing newline, so when it did print it glued itself onto the following line and broke a count read anchored at line start. Byte mode is unaffected -- Rulequiet is only ever set while probing under -8 -- and is byte-identical to 1.31 on 29,756,528 candidates.
 *
 * Revision 1.43  2026/09/13 21:06:37  dlr
 * Add Rulequiet: suppress COMPILER diagnostics while probing rule validity.
 *
 * procrule under -8 now runs the byte compiler on every rule precisely to discover
 * which rules it refuses, because that refusal is the UTF-32-REQUIRED tag. Those
 * refusals are expected answers, not faults, so their diagnostics are noise: a
 * 100k-rule file with 2 percent UTF-8-extension rules printed two thousand caret
 * blocks, which made -8 unusable on a real rule file.
 *
 * Rulequiet gates the two compiler diagnostic sites -- rule_error() and
 * packrules_len()s position checks -- and nothing else. It never suppresses
 * applyrule()s runtime diagnostics, which report a different class of problem.
 * Defaults to 0, so mdxfind and every existing caller are unaffected.
 *
 * procrule sets it only while the probe runs, and only when the UTF-32 compiler
 * already accepted the rule. If BOTH compilers refuse, the rule really is bad and
 * the caret line is the only thing that says why, so it stays audible. Verified:
 * rules x, i and ~~~ are still refused under -8 WITH their caret diagnostic, while
 * a quoted-operand rule is tagged silently.
 *
 * Revision 1.42  2026/09/13 12:17:26  dlr
 * -Z prints retained rule SOURCE TEXT; the bytecode decoder is deleted.
 *
 * Completes the -Z work begun in 1.586, which fixed attribution. This fixes what
 * -Z PRINTS, and removes an out-of-bounds read.
 *
 * THE DEFECT. The -Z histogram decoded packed bytecode back into a rule line by
 * switching on ASCII opcode letters. But packrules emits 0x80 to 0xfd for verbs
 * and stores OPERANDS as raw bytes, so an operand could collide with an opcode
 * case. Rule $z packs to e5 7a 00: the 7a matched the position-operand case, the
 * entry terminator was consumed as that operand, Rulepos was indexed at -1, and
 * the walk continued past the entry into the next rule. Measured before this
 * change on a 33-rule file whose operands cover every colliding verb: the output
 * carried 34 bytes of 0xe5 and 35 NULs, one row printed two rules joined, and the
 * byte census showed raw bytecode on stderr. Packed bytecode is simply not
 * decompilable, so no version of that decoder can be made correct.
 *
 * THE FIX. Retain the source text instead. struct rule_ent gains srcoff and
 * srclen, struct rulestore gains a source slab, and rs_add_src carries the text;
 * rs_add is now that with a NULL source, so procrule is unaffected. The -r/-R
 * load site copies the line BEFORE packrules_len compiles it in place, since
 * s = d = line destroys it. The flatten copies the text out of the store, indexed
 * 1..Numrules to line up with RuleCnt and Ruleindex, because rs_reset frees the
 * store once Rules is built. Retention is unconditional, not gated on -Z:
 * options are processed in argv order, so -r foo -Z loads rules before -Z is
 * seen, and gating would need an argv pre-scan and create a flag-order-sensitive
 * invisible mode. struct RuleHist gains an explicit length because the slab is
 * packed rather than NUL-terminated. 122 lines of decoder deleted.
 *
 * -R PROVENANCE. rs_product joins the two parents source text with a space,
 * which is itself a runnable rule line, since mdxfind applies the ops on a line
 * left to right. Materialised at product time rather than kept as parent
 * pointers because rs_product already materialises the full dst x mul BYTECODE
 * product, so the text is the same order of growth and not a new one. A rule
 * deduplicated away contributes no orphan text, and the FIRST spelling of a
 * duplicated rule is the one -Z reports, which is input-file order.
 *
 * VALIDATED on three engines, readable text, counts matching the documented
 * per-rule truth exactly.
 *   mixguard, 7 of 10 GPU-eligible, the mixed partition: No rule 6, l 3, u 6,
 *   c 4, $1 6, v2- 6, ^a 6, v3x 5, r 6, v4_ 5, d 6. CPU equals OpenCL on a
 *   GTX 1080 equals Metal on an Apple M1, 59 of 59 found on each.
 *   33 colliding operands, every verb whose letter an operand can collide with,
 *   at 33 of 33 eligible so every rule crosses the GPU: all 33 render correctly
 *   and the byte census is clean on all three, against 0xe5 x34 and NUL x35
 *   before. 33 of 33 found.
 *   New 2x2 -R product, which no existing fixture covered: 4 rules named
 *   runnably as l $1, l $2, u $1, u $2 in left-major order, 5 of 5 found,
 *   identical on all three engines.
 *
 * procrule builds and runs unchanged, as rs_add keeps its signature.
 *
 * BUILD NOTE for the next person shipping these files by hand: ruleproc.c needs
 * rule_ops.h at 1.3 or later. Shipping ruleproc.c without it fails with
 * rule_class_byte, rule_verb_opcodes and RULE_CLASS_MATCH undeclared. Both GPU
 * build hosts had a stale copy.
 *
 * Revision 1.41  2026/09/13 02:38:37  dlr
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
 * Revision 1.40  2026/09/12 16:14:23  dlr
 * Rule length reaches the classifier and the GPU program: NUL-bearing rules are
 * now GPU-eligible and transferred intact, and the O-of-n-squared pack scan is gone.
 *
 * Stage 2 of the rule input restructure. Stage 1 made the length first-class in the
 * host store; this carries it into classification and into the device transfer.
 *
 * gpu_rule_safe_phase0 is length-bounded. It used to walk the bytecode as a C string
 * and test every operand byte with a not-zero check, so a 0x00 OPERAND read as
 * end-of-rule and the rule was reported NOT GPU-safe. Any rule carrying one silently
 * stayed CPU-only: correct results, no GPU, no diagnostic. The walk is now bounded by
 * the true packed length, so an operand may be any byte, and a truncated rule is
 * rejected because its operands run past the length rather than because one of them
 * happens to be zero. Every fall-through label group is preserved exactly, including
 * the four-byte X arm that sits AFTER the three-byte body which nine labels reach by
 * falling through - moving it in front of them once took d3ad0ne from 37 CPU-only
 * rules to 14,958.
 *
 * classify_rules now carries per-rule lengths through to both partitions, and records
 * each entry's ORIGINAL index. struct rule_lists gains fulllen, gpulen, cpulen, gpuidx
 * and cpuidx; rule_lists_free frees the new arrays.
 *
 * Both GPU pack sites consequently change in two ways. They size and copy each rule
 * with its true length instead of strlen, which used to stop at a 0x00 operand and
 * hand the device a truncated rule whose operand count still said N - the kernel then
 * satisfied the missing operands from the next rule's bytes. And they take the original
 * index directly from gpuidx instead of recovering it by restarting a pointer scan at
 * zero for every entry, although classify_rules preserves order.
 *
 * That scan was estimated at roughly 5e9 pointer compares at HashMob.100k scale, an
 * arithmetic figure that had never been measured. Measured now, on HashMob.100k.rule
 * via mdxfind's own T-plus instrumentation: startup falls from 2.840s to 0.652s, a 4.3x
 * reduction saving 2.19 seconds.
 *
 * MEASURED AGAINST THE PRE-CHANGE BINARY. A four-line probe whose middle two rules
 * differ only after a NUL, run on the untouched 1.583 build, reports "4 rules read"
 * then "3 total rules in use" - one rule silently DROPPED - and "GPU rule engine: 2/3
 * rules eligible", the survivor refused as not GPU-safe. The same probe on stage 1 plus
 * 2 reports 4 read, 4 in use, 4 of 4 eligible.
 *
 * VALIDATION. New nulfix fixture, 10 lines yielding 9 rules, 5 words, 45 unique
 * candidates of which 25 contain a NUL byte: CPU equals GPU exactly on OpenCL against
 * a GTX 1080 and on Metal against an M2 Max, 45 of 45, NUL-bearing 25 of 25 on both
 * sides, 9 of 9 rules eligible. The eight existing fixtures all pass on both backends
 * at their exact counts - classfix 232, gpufix 144, longfix 111, mixfix 145, gatefix
 * 84, memfix 100, mixguard 59 with its CPU-partition check at 16 of 16 and its mixed
 * partition confirmed at 7 of 10 eligible, and divfix reproducing its asserted
 * divergence of CPU 30 against GPU 19. Rule counts are unchanged from the 1.583
 * reference on all seven shipped files. Throughput on fpga against rockyou by best64
 * is 4.69s wall and 1.055 hash Gh per second against a 4.68s and 1.047 baseline, with
 * peak VRAM at 325 MiB exactly as before, 1000 of 1000 found on every repeat.
 *
 * NOT a regression and out of scope: a single-char NUL form, either prepend or append
 * alone, is rejected by the loader's applyrule validity probe as a bad line. The
 * untouched 1.583 build rejects it identically and procrule rejects its equivalent, so
 * the two engines agree; only a multi-char form carries a NUL through today. The nulfix
 * fixture keeps such a line deliberately to exercise that path, which is why it yields
 * 9 rules from 10 lines.
 *
 * Still ahead, and the reason the format work is separate: the GPU program remains
 * NUL-terminated, so the kernel walk still depends on every op advancing by exactly its
 * operand count in all six kernels across OpenCL and Metal. Stage 3 replaces that with
 * a two-byte length prefix.
 *
 * Revision 1.39  2026/09/12 15:00:57  dlr
 * Rule length is now first-class: NUL-clean rules, input-file order, left-major -R.
 *
 * Stage 1 of the rule input restructure, per operator directive 2026-09-12: accept
 * NULs in rules and pass the length, which rules out Judy string de-duplication and
 * forces a hash or literal compare.
 *
 * THE DEFECT. The packed bytecode is not a C string. An operand byte may legitimately
 * be 0x00 via PARSEHEX, and 0x00 is never an opcode, so deriving the length with
 * strlen truncated a rule at its first NUL operand. Both applyrule and all six GPU
 * kernel walkers were always NUL-clean, because each consumes operands by explicit
 * count, so the corruption happened entirely in storage, before either engine saw the
 * bytes. procrule was correct throughout for the same reason: it never put the packed
 * form in a string-keyed container.
 *
 * Two separate wrong answers followed, both silent:
 *
 *   Truncation. The rule was stored short while its operand count still said N, so
 *   applyrule read the missing operands out of whatever followed. Measured: the rule
 *   that appends NUL then B, applied to ABC, produced the hex for ABC NUL NUL instead
 *   of ABC NUL B, and the bad byte tracked the NEXT rule in the file - the low byte of
 *   that entry's length prefix. For the last rule in the buffer the read goes past the
 *   end of the Rules allocation. The emitted plaintext hashed to the emitted digest,
 *   so nothing downstream could detect it.
 *
 *   Collapse, which is worse. Appending NUL then B and appending NUL then C both
 *   stringified to the same two bytes, so JudySL treated them as ONE key and silently
 *   DROPPED one of the two rules. A rule file could hold N rules and mdxfind would run
 *   N-1, with the load receipt reporting the deduplicated count as though nothing had
 *   happened.
 *
 * WHAT REPLACES IT. packrules_len yields the true packed length; packrules is retained
 * as a wrapper so the callers in procrule.c, gpu_rules_test.c, rule-bench.c and pr.c
 * are untouched. A new rule store replaces the RuleArray and NRuleArray JudySL pair:
 * an append-only bytecode slab plus a linear entry table, deduplicated by FNV-1a-64
 * with open addressing and every hash hit confirmed by an exact memcmp, so a false
 * dedup - which would silently drop a rule - is impossible by construction rather than
 * improbable. Dedup earns little, 0 to 0.13 percent on the shipped files, and is kept
 * only because Numrules feeds the ETA; nothing was traded away for it.
 *
 * Two properties the JudySL could not provide. Index order is now INPUT-FILE order
 * across multiple -r files, where JSLF and JSLN iterated in lexicographic order of the
 * bytecode - rules executed in bytecode order, not the order written. And -R is now
 * LEFT-MAJOR per operator ruling, first file varying slowest, which required the store
 * to be instantiable and the -R file to be buffered before the product is formed: a
 * streaming read can only produce right-major.
 *
 * Rules[] entry layout is deliberately UNCHANGED - two-byte length equal to bytecode
 * length plus one, then the bytecode, then a NUL - so the rule_ptrs walk, applyrule
 * and the classifier are all untouched. Only the copy becomes correct, memcpy at the
 * true length in place of strcpy, and only the order changes.
 *
 * ValidRules deleted. It had exactly three references: a declaration, one malloc of
 * MemSize plus four, and one memmove from Rules. It was never read anywhere in the
 * tree, and at Hash-IT_Crazy_Rules scale it held roughly 150 MB to no purpose.
 *
 * VALIDATION. All four NUL-bearing forms now agree byte for byte with procrule, and
 * the two rules that used to collapse both load and both fire. Note that two of the
 * four only ever LOOKED correct: the missing operand was 0x00 and the byte read past
 * the truncation was the NUL terminator, also 0x00, so they were right by coincidence
 * while still overreading. Seven fixtures pass on the CPU path - classfix 232, gpufix
 * 144, longfix 111, mixfix 145, gatefix 84, memfix 100, mixguard 59 - and divfix gives
 * its expected CPU 30. Rule counts are identical to the pre-change 1.583 binary on
 * every shipped rule file: best64 77, top_500 499, HashMob 1k 999, 5k 4997, 10k 9997,
 * 100k 99995, t.rule 1. Content equivalence checked across platforms as well as
 * versions: best64 against the first 100000 lines of rockyou with 1000 target hashes
 * gives a found-set md5 of 9cdd812e918311b5b0988fee36e47fb9 from both the pre-change
 * 1.583 build on Linux and this build on macOS, from inputs verified identical by md5.
 * Execution order confirmed to follow the rule file, and the -R product confirmed
 * left-major and complete. Scale: 100000 rules load in 1.07 seconds at 86 MB peak RSS,
 * and a 200000-rule -R product in 1.09 seconds at 100 MB.
 *
 * NOT yet addressed, and staged deliberately. The GPU program packer still sizes and
 * copies each rule with strlen, so a NUL-bearing rule is still truncated on its way to
 * the device; the classifier still treats a NUL operand as end-of-rule and marks such a
 * rule not GPU-safe, so it would stay CPU-only regardless. Both are stage 3, which
 * also moves the GPU program to a two-byte length prefix across OpenCL and Metal.
 *
 * Revision 1.38  2026/09/12 12:54:47  dlr
 * A bare hash character is a COMMENT, per Waffle. Correct the 1.37 wording.
 *
 * 1.37 called it a rule terminator and said explicitly that it is not a line
 * comment. That was wrong. A bare hash -- one standing where a verb is expected --
 * comments out the rest of the line and nothing after it is examined, so c then
 * hash then dollar-one means capitalize, then a comment. packrules says this
 * directly by testing the character in the SAME condition as carriage return and
 * newline: a bare hash is treated as end of line.
 *
 * The operand case does not make it something other than a comment. That is how
 * comment characters behave everywhere, the shell included: dollar-hash appends the
 * character, at-hash purges it, s-a-hash substitutes it, because there the
 * character is a verb operand and therefore data. A whole-line comment is the
 * degenerate case with the hash first.
 *
 * 1.37 also framed a trailing comment as a silent truncation to be grepped for.
 * Removed: there is nothing to diagnose in a comment working as designed.
 *
 * Unchanged from 1.36 and 1.37: 0xc4 is unreachable, the case that would emit it is
 * dead code, the character cannot force a rule onto the CPU, and the CPU-only set
 * is S 0xc5, v 0xc1 and 0x02.
 *
 * Revision 1.37  2026/09/12 12:51:51  dlr
 * Tighten the 0xc4 note: a hash character terminates a rule only in OPCODE
 * position, and is a literal as an operand. It is not a line comment.
 *
 * The 1.36 note said packrules breaks on the hash character anywhere in the line.
 * That is wrong in a way that misleads: the test sits at the TOP OF THE LOOP, which
 * is reached only where an opcode is expected, because operand bytes are consumed
 * by the pointer increment inside each case and never reach it. Measured: dollar-
 * hash appends the character, at-hash purges it, s-a-hash substitutes it. What does
 * truncate is a hash in opcode position, so dollar-one hash dollar-two silently
 * compiles to dollar-one with no diagnostic.
 *
 * A line starting with the character compiles to nothing and is dropped, which is
 * why it serves for comment lines and why it is easy to mistake for one. The
 * conclusion of 1.36 is unchanged: 0xc4 is unreachable, the case that would emit it
 * is dead code, and the CPU-only set is S 0xc5, v 0xc1 and 0x02.
 *
 * Revision 1.36  2026/09/12 12:32:43  dlr
 * Correct the CPU-only opcode count in the classifier comment: three, not four.
 *
 * RULE_OP_HASH_EXIT 0xc4 is not reachable and must not be counted as a way to
 * force a rule onto the CPU. packrules breaks out of its packing loop on a hash
 * character anywhere in the line, so the hash truncates the rule at COMPILE time
 * and everything after it is discarded; the case that would emit 0xc4 sits in the
 * same function after that break and is dead code. Measured: rule c then hash then
 * dollar-one applied to pass yields Pass, while c then dollar-one yields Pass1, and
 * the first of those classifies as GPU-eligible because it compiles to a bare c.
 *
 * So the CPU-only set is S 0xc5, v 0xc1 and 0x02. With 0x02 out of GPU scope by
 * ruling and S pending the A2 restructure, v is in practice the only usable one for
 * building the mixed GPU and CPU partition that the FastRule precondition guard
 * needs in order to be exercised. Found during the documentation pass for 1.583,
 * which measured it rather than taking the comment at its word.
 *
 * Revision 1.35  2026/09/12 12:12:51  dlr
 * Input-word gate at 1024 bytes, walker buffer at 2048, and the memory family on GPU.
 *
 * The admission gate and the walker buffer size were one constant doing two jobs.
 * They are now separate: GPU_RULES_MAX_INPUT_LEN 1024 gates the INPUT WORD only,
 * GPU_RULES_WALKER_BUF_ELEMS is twice that, and GPU_RULES_WALKER_BUF_SLACK 15 backs
 * the usable limit off to 2033 to cover n+1 boundary conditions such as a
 * terminating NUL.
 *
 * Operator ruling that sets the policy: output length is never checked. It is not
 * possible to bound what a rule can produce, because for any limit, on either
 * engine, a rule that overflows it can always be written. So a limit is SET,
 * arbitrarily, and a rule that exceeds it does what it can and stops; whatever has
 * been generated becomes the candidate. The check is strictly and only on the input
 * word size. A word over 1024 bytes goes to the CPU, and if the CPU engine then
 * exceeds MAXLINE it stops the same way and the hash is computed over whatever was
 * generated. There are no environment variable overrides, this is compile time
 * only, and there is no retry to CPU when a rule line exceeds the GPU limit. That
 * deliberately permits divergence between the two engines; -G none is the remedy.
 *
 * Memory family M 4 6 Q X promoted from CPU-only to GPU-eligible and implemented in
 * all six rules kernels against a second RULE_BUF_MAX buffer. The earlier attempt
 * was reverted because at 40960 bytes a second buffer doubled per-thread private
 * memory and aborted with CL_OUT_OF_HOST_MEMORY on an RTX 3080; at 2048 the pair
 * costs 4 KB per thread, a tenth of what ONE buffer cost before, so the resource
 * objection is gone rather than worked around. applyrule sets memlen to 0 at the
 * top of every call, so memory state never crosses a word-and-rule boundary and
 * there is no cross-rule state for an independent work-item to reproduce.
 *
 * RULE_BUF_MAX reaches the kernels by -D through exactly one funnel per backend.
 * OpenCL: gpu_kernel_cache.c, 67 build sites, with the define folded into
 * defines_str so it enters compute_key and a stale cache binary built at a
 * different size cannot load. Metal: a new metal_compile_opts helper in
 * gpu_metal.m, 9 call sites converted. All six kernels carry a matching ifndef
 * RULE_BUF_MAX default of 2048 so they still compile standalone. In
 * metal_common.metal the class membership table, its helper and the class opcode
 * defines moved OUT of the ifndef RULE_BUF_MAX block where they had been trapped:
 * they are needed whenever the file is compiled, and with the -D live that block
 * is skipped, which produced 20 undeclared-identifier errors.
 *
 * Two ruleproc.c comments corrected against their own code. The header still
 * listed M 4 6 X Q and also = % as CPU-only after both had been promoted, and the
 * rejection-ops comment still claimed Q stays in default-reject. Only four opcodes
 * now sit outside the whitelist, S 0xc5, hash 0xc4, v 0xc1 and 0x02, and they are
 * the only remaining way to build a mixed GPU and CPU partition.
 *
 * Validation. Seven fixtures, CPU equals GPU on a GTX 1080 and an M2 Max: classfix
 * 232, gpufix 144, longfix 111, mixfix 145, gatefix 84 exercising the
 * never-before-executed over-gate routing path, memfix 100, and divfix, which
 * asserts divergence and is the functional proof the -D reached the device by
 * finding the within-limit 19 of 30 rather than all 30. pretoday 113. A new
 * mixguard fixture covers the FastRule precondition guard at mdxfind.c 13764, which
 * no other fixture reaches any more now that the memory family is eligible: with
 * the guard removed its CPU-partition recovery collapses to exactly 6 of 16, one
 * per word. gatefix by contrast still scores 84 of 84 with the guard removed, so it
 * never covered that guard at all.
 *
 * Benchmark: rockyou 14341564 lines by best64.rule, 77 rules, -m e1, 1000 hashes,
 * median of 3 runs. fpga GTX 1080 peak VRAM 1769 MiB on the deployed 1.581 against
 * 325 MiB here, an 81.6 percent reduction from the buffer resize, with throughput
 * up from 0.990 to 1.047 hash Gh per second. Compiling the memory family back out
 * saves a further 76 MiB and moves throughput by -0.57 percent, inside noise, so
 * the family stays. mmt 72 cores with -G none: 51.13s on the deployed 1.245 against
 * 3.14s here.
 *
 * Revision 1.34  2026/09/11 19:47:24  dlr
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
 * Revision 1.33  2026/09/11 04:15:58  dlr
 * Fix the `.` rule op at the last position, in the byte engine and in all six GPU rule kernels. The op replaces the character at position N with the character at N+1, and at the boundary the three implementations gave three different answers.
 *
 * The byte engine validated N and then read N+1: `if (y < clen) cpass[y] = cpass[y+1]` admitted y == clen-1 and read one byte past the end of the candidate, picking up whatever the shared workspace held from a previous, longer one. Output therefore depended on wordlist order and on the thread split - `K $1 .9 $0` applied to Jay020171 gave Jay020117Z0 after a run of Zs, Jay020117a0 after a run of as, and Jay0201170 with a clean buffer. Two sites, the len < FASTLEN fast path and the slow path, both now guard y+1, which is the byte actually read. The GPU kernels bounds-checked correctly but substituted a zero byte when pos+1 was out of range, embedding a NUL in the candidate - a third answer again, and a CPU/GPU divergence: the two paths could produce different candidates for the same rule, which breaks the hit-set parity the Phase 4 work established. All six now no-op instead. ruleproc32.c was already correct and is untouched.
 *
 * hashcat settles which answer is right, rather than preference: an out-of-range `.N` is a no-op and the candidate is still emitted. Verified with hashcat --stdout on abcdefghij - `.8` gives abcdefghjj, `.9` and `.A` give abcdefghij unchanged - then swept over every position 0-B against every word length 1-12 for both `.` and `,`, 288 cases, zero mismatches. `,` needed no change: its existing y > 0 clause already guards the y-1 read and hashcat agrees that `,0` is a no-op while `,1` applies. Note the byte engine returns -2 where hashcat emits the unchanged word, which is correct for mdxfind because the plain is already covered by the implicit no-rule pass.
 *
 * The corruption was live in real output, not only at the boundary of a synthetic case. Against all_gj.rule, 208,010 rules by 20 words, the pre-fix binary emitted 43 NUL bytes and the post-fix binary emits none. Both failure modes were present: truncation, which emptied 7 lines entirely and turned 0ek20171 into 0ek2017, and residue substitution, which turned 171 into 172 by reading a 2 left behind by an earlier candidate.
 *
 * Validated on real hardware, not by inspection. The regression fixture is built around a hash only the buggy kernel can produce - md5 of abcdefghi followed by a NUL, b2eb01e4089a69425bf845937127e68f - alongside md5 of a bare abcdefghi for the old CPU truncation, so a stale header or a build that missed the change fails positively rather than silently passing. CPU, OpenCL on fpga.local (GTX 1080) and Metal on dev3.local all found exactly the two correct hashes and neither poison hash, with the GPU rule engine confirmed active from the run output rather than assumed. Broader sweep over all 10,140 rules in all_gj.rule containing `.` or `,`: 60 words gave 453,004 unique candidates, CPU and OpenCL both 453,004 of 453,004 with an empty hash-set diff; 3 words gave 27,258 and Metal matched CPU exactly.
 *
 * Two things worth recording about the generated headers. gpu/metal_md5_rules_str.h was absent from the working tree entirely and the copy on dev1 was stale from May 13, while gpu_metal.m includes it at line 199 and concatenates it into the JIT MTLLibrary at 809 and 2398 - and per that file the rules variant is ALWAYS JIT, never the precompiled metallib, so the header is load-bearing. Without regenerating it the Metal build would have compiled the pre-fix kernel and reported success. It is regenerated here with metal2str.py, which is the correct generator for .metal sources; cl2str.py happens to produce identical output but relying on that would be relying on a coincidence. Separately, gpu_md5_rules_str.h was checked in with kv keyword substitution, which is wrong for a generated string header: it embeds the .cl source's own $Revision and $Log markers inside its string literals, so a kv check-in would rewrite the revision and inject unquoted log text after the $Log line, breaking the C syntax of the embedded kernel. Set to -ko before this check-in, per the standing rule for gpu_*_str.h files. The stored 1.26 predates the stanza and is not itself damaged.
 *
 * Flagged, not addressed here: the Metal rules-engine hit buffer is far smaller than the OpenCL packed path and has no overflow-reissue loop, so at 453,004 candidates it drops cracks - it does self-report the loss with a MISMATCH audit line. Worth an architect look on its own.
 *
 * Revision 1.32  2026/09/06 15:05:48  dlr
 * Move the 61 RULE_OP_* opcode defines into rule_ops.h and include it.
 *
 * The byte engine and the UTF-32 engine both need the opcode numbering, and a second copy of 61 defines is a second place for them to drift. rule_ops.h is the single definition; ruleproc.c now includes it. No functional change: the defines are identical and the file is otherwise untouched.
 *
 * Revision 1.31  2026/05/18 05:22:32  dlr
 * ruleproc.c: remove unused includes (limits.h, wctype.h, errno.h) flagged by clangd; verified zero references in file.
 *
 * Revision 1.30  2026/05/18 05:14:07  dlr
 * Phase 3 ubuntu22 warning sweep: mark Version static unused. parserules: mark lbuf (written by PARSEHEX macro, never read) and y unused. applyrule: remove genuinely unused s1 d1 q128 locals.
 *
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
/* Suppress COMPILER diagnostics (this function and packrules_len's position
 * checks).  Set by a caller that is deliberately probing whether a rule
 * compiles and for which an refusal is an expected answer rather than an
 * error -- procrule under -8 tries the byte compiler on every rule precisely
 * to discover which rules it refuses, and those refusals are the UTF-32
 * REQUIRED tag, not faults.  Defaults off, so mdxfind and every existing
 * caller are unaffected.  It never suppresses applyrule()'s runtime
 * diagnostics, which report a different class of problem. */
int Rulequiet = 0;

static void rule_error(const char *msg, const char *orule,
                       const char *rule)
{
	int pos = (int)(rule - orule);
	int len = (int)strlen(orule);
	int i;

	if (Rulequiet) return;

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
       /* The THIRD compiler diagnostic site, and the one Rulequiet missed: the
        * other two are in packrules_len and the rest route through
        * rule_error().  Under -8 packrules is run on every rule purely to
        * discover which rules it refuses -- that refusal IS the tag -- so this
        * has to be as quiet as the others.  It is also the only one with no
        * trailing newline, so when it did print it GLUED itself onto the next
        * line and broke a fixture that reads a count anchored at line start. */
       if (!Rulequiet) fprintf(stderr,"Invalid position %c in rules",c);
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
#include "rule_ops.h"

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
        case RULE_OP_CHR_ADD:    return 'B';
        case RULE_OP_DIV_INSERT: return 'v';
        /* Class forms report as their base verb.  The caret in the error
         * already points at the operand, which is where the class sits. */
        case RULE_OP_SUB_CLASS:        return 's';
        case RULE_OP_PURGE_CLASS:      return '@';
        case RULE_OP_TITLE_CLASS:      return 'e';
        case RULE_OP_TITLE_CLASS_HC:   return 'e';
        case RULE_OP_REJ_HAS_CLASS:    return '!';
        case RULE_OP_REJ_NHAS_CLASS:   return '/';
        case RULE_OP_REJ_FIRST_CLASS:  return '(';
        case RULE_OP_REJ_LAST_CLASS:   return ')';
        case RULE_OP_REJ_AT_CLASS:     return '=';
        case RULE_OP_REJ_CNT_CLASS:    return '%';
        default:                 return (char)b;
    }
}

/* ---- character classes (D6) -------------------------------------------
 *
 * Membership bitmap, one 32-byte row per class, in class-id order.  Generated
 * from the reference definitions -- John's CHARS_* macros (rules.c:166-190)
 * and hashcat's class_*() predicates (src/rp.c) -- not hand-typed.  ?s is
 * hashcat's class_sym() in both tables per the operator's ruling.
 *
 * The complement bit is applied by XOR at test time, so there is no second
 * table and a complemented class costs nothing extra.
 */
/* rule_class_bits and RULE_CLASS_MATCH now live in rule_ops.h so that this
 * engine and ruleproc32.c share ONE definition rather than two that can
 * drift.  rule_ops.h is included by exactly these two files. */
/* Map a class letter to its packed class byte, or 0 if the letter names no
 * class in the requested table.
 *   table 0 -- John: inline ?C, complement by case-toggling the letter.
 *   table 1 -- hashcat: ~-prefixed, six classes, no complement (so ?H is
 *              uppercase hex here and "not hex" there -- the one collision
 *              that forces the tables apart).
 */
/* rule_class_byte() moved to rule_ops.h: the UTF-32 engine needs the same
 * letter-to-class mapping, and two copies of a 15-entry table keyed on
 * single letters is exactly the kind of thing that drifts unnoticed. */

/* Parse a character-or-class operand at *sp.
 *
 * Returns  0  literal character: *litp set, *sp advanced 1 (or 2 for `??`)
 *         >0  packed class byte, *sp advanced 2
 *         -1  malformed: truncated, or a letter that names no class
 *
 * `??` is John's escape for a literal `?`, which is why `@?` -- purge the
 * literal `?` -- must now be written `@??`.  Under table 1 (~-prefixed) a
 * class is mandatory: hashcat defines no literal form there.
 */
static int rule_parse_class_operand(char **sp, int table, char *litp) {
  char *s = *sp;
  unsigned char cb;

  if (table) {
    /* hashcat requires the `?`; `??` is its literal-`?` escape, verified
     * against its own engine (rp_cpu.c RULE_OP_CLASS_BASED, the `case '?'`
     * arm under each of ~s / ~@ / ~e). */
    if (s[0] != '?' || !s[1]) return (-1);
    if (s[1] == '?') { *litp = '?'; *sp = s + 2; return 0; }
    if (!(cb = rule_class_byte(s[1], 1))) return (-1);
    *sp = s + 2;
    return (int)cb;
  }
  if (!s[0]) return (-1);
  if (s[0] != '?') { *litp = s[0]; *sp = s + 1; return 0; }
  if (!s[1]) return (-1);
  if (s[1] == '?') { *litp = '?'; *sp = s + 2; return 0; }
  if (!(cb = rule_class_byte(s[1], 0))) return (-1);
  *sp = s + 2;
  return (int)cb;
}


/* Opcode pair for the nine class-capable verbs: the plain form and the class
 * form.  Returns 0 if the verb takes no class.  `=` and `%` pack as their
 * literal ASCII bytes in the plain form -- a historical exception to the
 * high-bit opcode range, kept because that encoding is already in the stream.
 */
/* rule_verb_opcodes() moved to rule_ops.h -- ruleproc32.c needs the same
 * verb-to-opcode-pair mapping for the `~` prefix. */

/* packrules_len -- as packrules, but also yields the TRUE packed length.
 *
 * The packed bytecode is NOT a C string: an operand byte may legitimately be
 * 0x00 (`$\x00` via PARSEHEX), and 0x00 is never an opcode.  Deriving the
 * length with strlen() therefore truncates the rule at its first NUL operand,
 * and callers that then copy or key on it with string semantics store a short
 * rule whose operand count still says N -- applyrule and the GPU walkers both
 * skip operands by explicit count, so they read the missing operands out of
 * whatever follows.  Measured before this change: `$\x00$\x42` and `$\x00$\x43`
 * both produced $HEX[4142430000] from ABC, and the bad byte tracked the NEXT
 * rule in the file (0x06 when that rule packed to 6 bytes -- its length-prefix
 * low byte).  For the last rule in the buffer the read goes past the end.
 *
 * *packedlen receives the byte count EXCLUDING the trailing NUL that is still
 * written for the benefit of callers that have not yet been converted.
 * packrules() is retained as a wrapper so the existing callers in procrule.c,
 * gpu_rules_test.c, rule-bench.c and pr.c are untouched. */
int packrules_len(char *line, int *packedlen) {
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
      case '/':          /* character operand -- see the note in pass 1 */
      case '(':
      case ')':
        /* Each of these takes a literal character OR an inline John class
         * `?C`.  `??` is the escape for a literal `?`. */
        NEED_BYTES(1);
        {
          unsigned char _plain = 0, _cls = 0;
          char _lit = 0;
          int _cb;
          rule_verb_opcodes(c, &_plain, &_cls);
          _cb = rule_parse_class_operand(&s, 0, &_lit);
          if (_cb < 0) {
            char _msg[72]; snprintf(_msg, sizeof(_msg),
              "Invalid character class for command '%c'", c);
            rule_error(_msg, line, s);
            rulefail++;
            goto pack_op_done;
          }
          *d++ = (char)(_cb ? _cls : _plain);
          *d++ = _cb ? (char)(unsigned char)_cb : _lit;
        }
        break;

      case 'D':
      case '\'':
      case 'Z':
      case 'z':
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
	  if (!Rulequiet) fprintf(stderr, "Invalid position %c for %c\n", n, c);
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
	  if (!Rulequiet) fprintf(stderr, "Invalid position %c for %c\n", n, c);
          rulefail++;
        }
        *d++ = positiontranslate(n)-1;
        *d++ = *s++;
	break;

      case 's':
        /* `sXY` literal, or `s?CY` -- replace every character of class C
         * with Y. */
        NEED_BYTES(2);
        {
          char _lit = 0;
          int _cb = rule_parse_class_operand(&s, 0, &_lit);
          if (_cb < 0 || !*s) {
            rule_error("'s' needs two operands: sXY or s?CY", line, s);
            rulefail++;
            goto pack_op_done;
          }
          *d++ = (char)(_cb ? RULE_OP_SUB_CLASS : RULE_OP_SUB);
          *d++ = _cb ? (char)(unsigned char)_cb : _lit;
          *d++ = *s++;
        }
	break;

      case '=':
      case '%':
        /* '=' and '%' pack as their literal ASCII bytes (0x3d, 0x25) rather
         * than a RULE_OP_* value -- a historical exception to the 0xc1-0xfd
         * opcode range, kept because that encoding is already in the stream.
         * applyrule DOES now implement both; it previously did not, and the
         * default case skipped only the opcode so the operands executed as
         * instructions. */
        NEED_BYTES(2);
	n = *s++;
        if ((n < '0') || (n > '9' && n < 'A') || (n > 'Z' && n < 'a') ||
	   (n > 'z') ) {
          { char _msg[64]; snprintf(_msg, sizeof(_msg),
            "Invalid position '%c' for command '%c'", n, c);
            rule_error(_msg, line, s - 1); }
          rulefail++;
        }
        /* The final operand is a literal character or an inline `?C` class.
         * The opcode is chosen after the operand is parsed, so it is emitted
         * here rather than above. */
        {
          char _lit = 0;
          int _cb = rule_parse_class_operand(&s, 0, &_lit);
          if (_cb < 0) {
            char _msg[72]; snprintf(_msg, sizeof(_msg),
              "Invalid character class for command '%c'", c);
            rule_error(_msg, line, s);
            rulefail++;
            goto pack_op_done;
          }
          *d++ = (char)(_cb ? (c == '=' ? RULE_OP_REJ_AT_CLASS
                                        : RULE_OP_REJ_CNT_CLASS)
                            : (unsigned char)c);
          *d++ = positiontranslate(n);
          *d++ = _cb ? (char)(unsigned char)_cb : _lit;
        }
        break;

      case 'i':
      case 'o':
      case 'B':          /* BNX -- add byte value of X to the byte at N */
        NEED_BYTES(2);
        *d++ = (c == 'i') ? RULE_OP_INSERT
             : (c == 'o') ? RULE_OP_OVERWRITE : RULE_OP_CHR_ADD;
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

      case '~':
        /* hashcat's class-based rule prefix.  The verb that follows takes a
         * MANDATORY `?C` operand drawn from hashcat's six-class table, which
         * is what makes the two syntaxes able to coexist: `?H` is uppercase
         * hex here and the complement of hex in John's inline form.
         *
         * hashcat documents six of the nine (~! ~/ ~( ~) ~= ~%) as -j/-k only
         * because its device rule engine cannot reject a candidate.  mdxfind
         * has no host/device split in the rule language and implements all
         * nine on both paths. */
        NEED_BYTES(3);
        {
          char _v = *s++, _lit = 0;
          unsigned char _plain = 0, _cls = 0;
          int _cb;
          if (!rule_verb_opcodes(_v, &_plain, &_cls)) {
            char _msg[72]; snprintf(_msg, sizeof(_msg),
              "'~' takes no character class for command '%c'", _v);
            rule_error(_msg, line, s - 1);
            rulefail++;
            goto pack_op_done;
          }
          if (_v == '=' || _v == '%') {
            n = *s++;
            if ((n < '0') || (n > '9' && n < 'A') || (n > 'Z' && n < 'a') ||
                (n > 'z') ) {
              char _msg[72]; snprintf(_msg, sizeof(_msg),
                "Invalid position '%c' for command '~%c'", n, _v);
              rule_error(_msg, line, s - 1);
              rulefail++;
              goto pack_op_done;
            }
          }
          _cb = rule_parse_class_operand(&s, 1, &_lit);
          if (_cb < 0) {
            char _msg[80]; snprintf(_msg, sizeof(_msg),
              "'~%c' needs a hashcat class: ?l ?u ?d ?s ?h ?H", _v);
            rule_error(_msg, line, s);
            rulefail++;
            goto pack_op_done;
          }
          /* hashcat's ~e?C is a different algorithm from John's e?C. */
          if (_v == 'e' && _cb) _cls = RULE_OP_TITLE_CLASS_HC;
          /* `~@??` and `@??` are the same rule and pack identically. */
          *d++ = (char)(_cb ? _cls : _plain);
          if (_v == '=' || _v == '%')
            *d++ = positiontranslate(n);
          *d++ = _cb ? (char)(unsigned char)_cb : _lit;
          if (_v == 's') {
            if (!*s) {
              rule_error("'~s?CY' needs the replacement character Y", line, s);
              rulefail++;
              goto pack_op_done;
            }
            *d++ = *s++;
          }
        }
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
          default:
            /* Refuse an unrecognised verb rather than emitting it as a literal
             * byte.  Operator ruling 2026-09-12: match hcrule, which rejects.
             * The old passthrough turned a typo into malformed bytecode -- `M2`
             * packed as MEM_STORE followed by a bare 0x32, which the walker then
             * read as an OPCODE, and 0x32 is not one.  hashcat 6.2.5 reports
             * "Skipping invalid or unsupported rule" for the same input. */
            {
              char _msg[64];
              snprintf(_msg, sizeof(_msg), "Unknown rule command '%c'",
                       (c >= ' ' && c < 127) ? c : '?');
              rule_error(_msg, line, s - 1);
              rulefail++;
            }
            break;
        }
        break;
    }
pack_op_done:
    if (rulefail) break;   /* halt the outer rule walk on any error so
                            * we don't keep reading past truncated input */
  }
  *d++ = 0;
  if (packedlen)
    *packedlen = (int)(d - line - 1);   /* exclude the trailing NUL */
  return (rulefail);
}

int packrules(char *line) {
  int ignored;
  return packrules_len(line, &ignored);
}

/* ---------------------------------------------------------------------------
 * Rule store: length-carrying, input-order, hash-deduplicated.
 *
 * Replaces the JudySL (JSLI) rule arrays.  A JudySL key is a NUL-terminated
 * string, which cannot represent a packed rule containing a 0x00 OPERAND, so
 * the old store truncated such rules at load; see packrules_len above for the
 * measured consequence.  Operator directive 2026-09-12: accept NULs in rules
 * and pass the length, which rules out Judy's string de-duplication and forces
 * a hash or literal compare.
 *
 * Two further properties the JudySL could not give us:
 *   - INPUT ORDER.  JSLF/JSLN iterate in lexicographic order of the bytecode,
 *     so rules executed in bytecode order, not the order of the rule file.
 *     rs_ent[] is linear and append-only, so index order IS input order,
 *     across multiple -r files.
 *   - A stable ordinal per rule, which -Z and -R provenance hang off later.
 *
 * Dedup is by FNV-1a-64 over (bytes,len) with linear probing, and every hash
 * hit is confirmed by an exact memcmp.  A false dedup silently DROPS a rule,
 * which is a correctness bug, so equality is never inferred from the hash
 * alone.  Dedup earns little -- 0 to 0.13% on the shipped rule files and
 * EXACTLY ZERO on Hash-IT_Crazy_Rules (6,828,885 in, 6,828,885 out) -- it is
 * kept because Numrules feeds the ETA, and nothing is traded away for it.
 * --------------------------------------------------------------------------- */

/* One instance per store.  -R needs two live at once: the accumulated product
 * and the file being multiplied in, and left-major order (operator ruling,
 * 2026-09-12: A1B1, A1B2, A1B3, A2B1...) requires the second file to be fully
 * buffered before the product is formed, because the first file must vary
 * slowest.  A singleton could not express either. */

static uint64_t rs_hash(const char *b, int len)
{
    uint64_t h = 1469598103934665603ULL;          /* FNV-1a-64 offset basis */
    for (int i = 0; i < len; i++) {
        h ^= (unsigned char)b[i];
        h *= 1099511628211ULL;
    }
    return h;
}

static int rs_grow_htab(struct rulestore *rs)
{
    size_t newmask = rs->hmask ? ((rs->hmask + 1) * 2 - 1) : 1023;
    uint32_t *nt = calloc(newmask + 1, sizeof(uint32_t));
    if (!nt) return -1;
    for (int i = 0; i < rs->n; i++) {
        size_t j = rs_hash(rs->slab + rs->ent[i].off, rs->ent[i].len) & newmask;
        while (nt[j]) j = (j + 1) & newmask;
        nt[j] = (uint32_t)(i + 1);
    }
    free(rs->htab);
    rs->htab  = nt;
    rs->hmask = newmask;
    return 0;
}

/* rs_add -- append a packed rule unless an identical one is already present.
 * Returns the slot index (>= 0) for a new rule, -1 for an exact duplicate,
 * -2 on allocation failure or an out-of-range length.
 *
 * A length of 0 is accepted and stored: it is what `:` packs to.  The LOADER
 * filters those out rather than this function, because the implicit no-rule
 * pass is foundational and an explicit `:` must not add a second one. */
int rs_add(struct rulestore *rs, const char *bytes, int len)
{
    return rs_add_src(rs, bytes, len, NULL, 0);
}

/* rs_add_src -- rs_add, plus retention of the rule's ORIGINAL SOURCE TEXT.
 * The source is appended only when the rule is actually stored, so a rule
 * deduplicated away does not leave orphan text and the FIRST spelling of a
 * duplicated rule is the one -Z reports -- which is input-file order. */
int rs_add_src(struct rulestore *rs, const char *bytes, int len,
               const char *src, int srclen)
{
    if (!rs || len < 0 || len > 65535) return -2;   /* 16-bit length field  */
    if (srclen < 0 || srclen > 65535) return -2;
    if (!src) srclen = 0;

    if (!rs->htab && rs_grow_htab(rs) < 0) return -2;
    if ((size_t)rs->n * 10 >= (rs->hmask + 1) * 7 && rs_grow_htab(rs) < 0) return -2;

    uint64_t h = rs_hash(bytes, len);
    size_t   j = h & rs->hmask;
    while (rs->htab[j]) {
        struct rule_ent *e = &rs->ent[rs->htab[j] - 1];
        if (e->len == len && memcmp(rs->slab + e->off, bytes, (size_t)len) == 0)
            return -1;                              /* exact duplicate      */
        j = (j + 1) & rs->hmask;                    /* hash collision only  */
    }

    if (rs->used + (size_t)len > rs->cap) {
        size_t nc = rs->cap ? rs->cap * 2 : (1 << 20);
        while (nc < rs->used + (size_t)len) nc *= 2;
        char *ns = realloc(rs->slab, nc);
        if (!ns) return -2;
        rs->slab = ns;
        rs->cap  = nc;
    }
    if (rs->n >= rs->entcap) {
        int ne = rs->entcap ? rs->entcap * 2 : 4096;
        struct rule_ent *nn = realloc(rs->ent, (size_t)ne * sizeof(*nn));
        if (!nn) return -2;
        rs->ent    = nn;
        rs->entcap = ne;
    }

    if (srclen) {
        if (rs->srcused + (size_t)srclen > rs->srccap) {
            size_t nc = rs->srccap ? rs->srccap * 2 : (1 << 20);
            while (nc < rs->srcused + (size_t)srclen) nc *= 2;
            char *nsrc = realloc(rs->src, nc);
            if (!nsrc) return -2;
            rs->src    = nsrc;
            rs->srccap = nc;
        }
    }

    memcpy(rs->slab + rs->used, bytes, (size_t)len);
    rs->ent[rs->n].off = (uint32_t)rs->used;
    rs->ent[rs->n].len = (unsigned short)len;
    rs->used += (size_t)len;
    if (srclen) {
        memcpy(rs->src + rs->srcused, src, (size_t)srclen);
        rs->ent[rs->n].srcoff = (uint32_t)rs->srcused;
        rs->ent[rs->n].srclen = (unsigned short)srclen;
        rs->srcused += (size_t)srclen;
    } else {
        rs->ent[rs->n].srcoff = 0;
        rs->ent[rs->n].srclen = 0;
    }
    rs->htab[j] = (uint32_t)(rs->n + 1);
    return rs->n++;
}

void rs_reset(struct rulestore *rs)
{
    if (!rs) return;
    free(rs->slab); free(rs->ent); free(rs->htab); free(rs->src);
    memset(rs, 0, sizeof(*rs));
}

/* rs_product -- replace `dst` with the LEFT-MAJOR concatenation of dst x mul.
 * Left-major: dst varies slowest, so the emitted order is
 * dst0.mul0, dst0.mul1, ... dst0.mulN, dst1.mul0, ...
 * A pair whose combined length would exceed `maxlen` is skipped, matching the
 * previous loader's `(curlen + len) < MAXLINE` guard.  Returns 0, or -1 on
 * allocation failure, in which case dst is left untouched. */
int rs_product(struct rulestore *dst, const struct rulestore *mul, int maxlen)
{
    struct rulestore out;
    char *tmp, *stmp;
    int i, k;

    if (!dst || !mul) return -1;
    memset(&out, 0, sizeof(out));
    tmp = malloc((size_t)maxlen + 2);
    if (!tmp) return -1;
    /* Source text for a product rule is the two parents' text joined by a
     * space, which is itself a runnable rule line -- mdxfind applies the ops
     * on a line left to right, so "l" x "$1" really is "l $1".  Materialised
     * here rather than kept as parent pointers because this function already
     * materialises the full |dst| x |mul| BYTECODE product, so the text is the
     * same order of growth and not a new one. */
    stmp = malloc((size_t)maxlen * 2 + 4);
    if (!stmp) { free(tmp); return -1; }

    for (i = 0; i < dst->n; i++) {
        for (k = 0; k < mul->n; k++) {
            int la = dst->ent[i].len, lb = mul->ent[k].len;
            if (la + lb >= maxlen) continue;
            int sa = dst->ent[i].srclen, sb = mul->ent[k].srclen, sl = 0;
            memcpy(tmp,      dst->slab + dst->ent[i].off, (size_t)la);
            memcpy(tmp + la, mul->slab + mul->ent[k].off, (size_t)lb);
            if (sa) { memcpy(stmp, dst->src + dst->ent[i].srcoff, (size_t)sa); sl = sa; }
            if (sa && sb) stmp[sl++] = ' ';
            if (sb) { memcpy(stmp + sl, mul->src + mul->ent[k].srcoff, (size_t)sb); sl += sb; }
            if (rs_add_src(&out, tmp, la + lb, sl ? stmp : NULL, sl) == -2) {
                free(tmp); free(stmp); rs_reset(&out); return -1;
            }
        }
    }
    free(tmp);
    free(stmp);
    rs_reset(dst);
    *dst = out;
    return 0;
}
#undef NEED_BYTES

char * parserules(char *line) {
  /* lbuf is written by the PARSEHEX macro (line 311) but never read inside
   * parserules — historical from when parserules emitted hex-decoded byte
   * sequences via lbuf. Kept (and silenced) to preserve PARSEHEX's macro
   * shape and parserules' parsing invariants. */
  char *t, *s, c, lbuf[10240] __attribute__((unused)), n,c1;
  char *lastvalid;
  int x, y __attribute__((unused)), rulefail = 0;


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
      /* `/`, `(` and `)` take a CHARACTER operand, not a position.  They were
       * grouped with the position verbs below, so packrules ran
       * positiontranslate() over a literal character: `/s` stored 0x37, the
       * index of 's' in Rulepos plus one, instead of 0x73.  The executors are
       * correct; the compiler mangled the operand, so all three always
       * rejected.  John and hashcat both document them as character tests:
       * "/X reject the word unless it contains character X", "(X ... unless
       * its first character is X", ")X ... unless its last character is X".
       * Moving them here also correctly relaxes validation -- any character
       * is a legal operand, where the position group refused anything outside
       * [0-9A-Za-z]. */
      case '/':
      case '(':
      case ')':
        {
          char _lit = 0;
          if (rule_parse_class_operand(&s, 0, &_lit) < 0) {
            char _msg[72]; snprintf(_msg, sizeof(_msg),
              "Invalid character class for command '%c'", c);
            rule_error(_msg, line, s);
            rulefail++;
          }
        }
        break;

      case 'D':
      case '\'':
      case 'Z':
      case 'z':
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
	  if (!Rulequiet) fprintf(stderr, "Invalid position %c for %c\n", n, c);
          rulefail++;
        }
	n = *s++;
	break;

      case 's':
        /* This arm consumed NOTHING before the class work, so `s`'s two
         * operands were re-entered as verbs: `sxy` validated 'x' as the
         * two-position extract op, read 'y' as one position and NUL as the
         * other, and reported a bogus "Invalid position".  It now consumes
         * its own operands. */
        {
          char _lit = 0;
          int _cb = rule_parse_class_operand(&s, 0, &_lit);
          if (_cb < 0 || !*s) {
            rule_error("'s' needs two operands: sXY or s?CY", line, s);
            rulefail++;
          } else {
            s++;
          }
        }
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
        {
          char _lit = 0;
          if (rule_parse_class_operand(&s, 0, &_lit) < 0) {
            char _msg[72]; snprintf(_msg, sizeof(_msg),
              "Invalid character class for command '%c'", c);
            rule_error(_msg, line, s);
            rulefail++;
          }
        }
        break;

      case 'i':
      case 'o':
      case 'B':
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

      case '~':
        /* hashcat class prefix -- validate the same shape packrules packs. */
        {
          char _v = *s++, _lit = 0;
          unsigned char _plain = 0, _cls = 0;
          if (!rule_verb_opcodes(_v, &_plain, &_cls)) {
            char _msg[72]; snprintf(_msg, sizeof(_msg),
              "'~' takes no character class for command '%c'", _v);
            rule_error(_msg, line, s - 1);
            rulefail++;
            break;
          }
          if (_v == '=' || _v == '%') {
            n = *s++;
            if ((n < '0') || (n > '9' && n < 'A') || (n > 'Z' && n < 'a') ||
                (n > 'z') ) {
              char _msg[72]; snprintf(_msg, sizeof(_msg),
                "Invalid position '%c' for command '~%c'", n, _v);
              rule_error(_msg, line, s - 1);
              rulefail++;
              break;
            }
          }
          if (rule_parse_class_operand(&s, 1, &_lit) < 0) {
            char _msg[80]; snprintf(_msg, sizeof(_msg),
              "'~%c' needs a hashcat class: ?l ?u ?d ?s ?h ?H", _v);
            rule_error(_msg, line, s);
            rulefail++;
            break;
          }
          if (_v == 's') {
            if (!*s) {
              rule_error("'~s?CY' needs the replacement character Y", line, s);
              rulefail++;
              break;
            }
            s++;
          }
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
 * NOT supported (rule routes to CPU).  After the 2026-09-11 promotions this
 * is only four opcodes; M 4 6 X Q and = % are now GPU-eligible and are listed
 * in the switch below, not here:
 *   - S  (0xc5)                     — mdxfind-specific; pending the A2
 *                                     restructure to john CONV_SOURCE/CONV_SHIFT
 *   - v  (0xc1)                     — divide-insert
 *   - 0x02 (slowrule escape: base64-encode-word) — out of GPU scope by ruling
 *   - Anything else
 *
 * THREE, not four: RULE_OP_HASH_EXIT 0xc4 is NOT reachable and must not be
 * counted here.  '#' IS A COMMENT.  A bare '#' -- one standing where a verb is
 * expected -- comments out the rest of the line, and nothing after it is
 * examined; packrules() says so directly by testing it in the SAME condition as
 * '\r' and '\n', so a bare '#' is treated as end of line.  c#$1 therefore means
 * capitalize, then a comment.  As a verb's OPERAND, '#' is ordinary data, which
 * is how comment characters behave everywhere: $# appends '#', @# purges it, sa#
 * substitutes it.  A whole-line comment is just the degenerate case with the '#'
 * first.
 *
 * Consequence for this switch: '#' never reaches the packed bytecode, so 0xc4
 * cannot appear here, the 'case' that would emit it is dead code, and '#' CANNOT
 * be used to force a rule onto the CPU.  c#$1 classifies as GPU-eligible because
 * it compiles to a bare c.
 *
 * Those three are the ONLY way to build a mixed GPU/CPU partition now, which is
 * what the FastRule precondition guard at mdxfind.c:13764 needs in order to be
 * exercised at all.  With 0x02 out of scope by ruling and S pending the A2
 * restructure, v is in practice the only usable one.  See the mixguard fixture.
 *
 * For multi-byte ops we must consume the parameter bytes too, otherwise
 * the next iteration would misinterpret them as ops.
 */
/* Length-bounded since 2026-09-12.  It used to walk the bytecode as a C string,
 * `while ((c = *packed_rule++))`, and every multi-byte arm tested its operand
 * bytes with `if (!*packed_rule++) return 0;`.  That treated a 0x00 OPERAND as
 * end-of-rule, so any rule carrying one -- `$\x00` via PARSEHEX -- was reported
 * NOT GPU-safe and silently stayed CPU-only: correct results, no GPU, no
 * diagnostic.  The walk is now bounded by the true packed length, so an operand
 * may be any byte including 0x00, and a truncated rule is rejected because its
 * operands run past `len` rather than because one of them happens to be zero. */
static int gpu_rule_safe_phase0(const char *packed_rule, int len) {
    int k = 0;
    while (k < len) {
        unsigned char c = (unsigned char)packed_rule[k++];
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
            /* Memory family, GPU-enabled 2026-09-11 (operator ruling).  All
             * six rules kernels now implement M 4 6 Q X against a second
             * RULE_BUF_MAX buffer.  The 2026 attempt was reverted because at
             * 40960 bytes a second buffer doubled per-thread private memory
             * and FATAL'd CL_OUT_OF_HOST_MEMORY on an RTX 3080; at 2048 the
             * pair costs 4 KB per thread, a tenth of what ONE buffer cost
             * before, so the resource objection is gone.
             * memlen resets per applyrule call on the CPU, so there is no
             * cross-rule state for an independent work-item to reproduce. */
            case RULE_OP_MEM_STORE: case RULE_OP_MEM_APP:
            case RULE_OP_MEM_PRE:   case RULE_OP_MEM_REJ:
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
             * Q (RULE_OP_MEM_REJ) is NO LONGER in default-reject: it moved up
             * into the memory-family group above on 2026-09-11 along with
             * M 4 6 X, which superseded the slice-4 path B revert described
             * there (that revert is why gpu/gpu_md5_rules.cl rev 1.24 dropped
             * the kernel impl; RULE_BUF_MAX 2048 removed the resource cost). */
            case RULE_OP_REJ_LEN_NE: case RULE_OP_REJ_LEN_GE:
            /* Character-class forms, promoted to GPU 2026-09-11: the six
             * rules kernels implement all ten, sharing one 480-byte
             * membership table in the constant address space of the common
             * source.  Promoting them matters beyond speed -- every class
             * rule used to force a mixed GPU/CPU partition, and a
             * partitioned run has to union two result sets. */
            case RULE_OP_PURGE_CLASS:     case RULE_OP_TITLE_CLASS:
            case RULE_OP_TITLE_CLASS_HC:  case RULE_OP_REJ_HAS_CLASS:
            case RULE_OP_REJ_NHAS_CLASS:  case RULE_OP_REJ_FIRST_CLASS:
            case RULE_OP_REJ_LAST_CLASS:
            case RULE_OP_REJ_LEN_LE: case RULE_OP_REJ_HAS:
            case RULE_OP_REJ_NHAS: case RULE_OP_REJ_FIRST:
            case RULE_OP_REJ_LAST:
                /* Two-byte op: one operand byte, any value. */
                if (k + 1 > len) return 0;
                k += 1;
                continue;
            /* Hex emit ops (rev 1.26): kernel rev 1.22 implements `H` and
             * `h` (RULE_OP_HEX_UPPER / RULE_OP_HEX_LOWER) with byte-exact
             * applyrule semantics. Single-byte ops, no parameter to skip. */
            case RULE_OP_HEX_UPPER: case RULE_OP_HEX_LOWER:
                continue;
            case RULE_OP_SUB:
            case RULE_OP_CHR_ADD:
            case RULE_OP_INSERT: case RULE_OP_OVERWRITE:
            /* Three-byte class forms, plus `=NX`/`%NX` which pack as their
             * literal ASCII bytes and were CPU-only for no reason other than
             * that this switch never listed them. */
            case RULE_OP_SUB_CLASS: case RULE_OP_REJ_AT_CLASS:
            case RULE_OP_REJ_CNT_CLASS:
            case 0x3d: case 0x25:
            case RULE_OP_SWAP_AT: case RULE_OP_EXTRACT: case RULE_OP_OMIT:
            case RULE_OP_TOGGLE_SEP:
                /* Three-byte op: two operand bytes, any value.
                 * RULE_OP_TOGGLE_SEP is hashcat RULE_OP_MANGLE_TOGGLE_AT_SEP —
                 * same wire shape (op + position-byte + literal-byte) as
                 * 'i'/'o', so it joins this group. */
                if (k + 2 > len) return 0;
                k += 2;
                continue;
            /* `X N M I` is the only FOUR-byte op: opcode plus three
             * position-encoded operands.  Placed AFTER the three-byte group's
             * body on purpose -- the labels above it (SUB, CHR_ADD, INSERT,
             * OVERWRITE, the three-byte class forms, `=`, `%`) have no body of
             * their own and FALL THROUGH to it.  Inserting a four-byte body in
             * front of them made all nine consume three operand bytes instead
             * of two, desynchronising the walk: d3ad0ne.rule went from 37
             * CPU-only rules to 14,958. */
            case RULE_OP_MEM_INSERT:
                if (k + 3 > len) return 0;
                k += 3;
                continue;
            case 0xff: case 0xfe: {
                /* Multi-char append/prepend: 2+N bytes total.  N operand bytes
                 * of ANY value -- this arm is where the old NUL test did the
                 * most damage, because a multi-char append is exactly how a
                 * 0x00 reaches a rule. */
                unsigned char N;
                if (k + 1 > len) return 0;
                N = (unsigned char)packed_rule[k++];
                if (N == 0) return 0;
                if (k + N > len) return 0;
                k += N;
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
 * lens[] is REQUIRED and parallel to rules[]: it carries each rule's true
 * packed length, because the bytecode is not a C string -- an operand byte may
 * be 0x00.  gpulen[]/cpulen[] carry the lengths through to the partitions, and
 * gpuidx[]/cpuidx[] record each entry's ORIGINAL index so a caller never has to
 * recover it by searching for the pointer.  The GPU pack sites used to do
 * exactly that, restarting the scan at 0 for every entry although this function
 * preserves order -- roughly 5e9 pointer compares at HashMob.100k scale.
 *
 * Returns: ngpu (number of GPU-eligible rules; 0..nrules). */
int classify_rules(char **rules, const unsigned short *lens, int nrules,
                   struct rule_lists *out) {
    if (!out || !lens) return -1;
    memset(out, 0, sizeof(*out));
    out->full    = rules;
    out->fulllen = lens;
    out->nfull   = nrules;
    out->gpu     = (char **)malloc((size_t)nrules * sizeof(char *));
    out->cpu     = (char **)malloc((size_t)nrules * sizeof(char *));
    out->gpulen  = (unsigned short *)malloc((size_t)nrules * sizeof(unsigned short));
    out->cpulen  = (unsigned short *)malloc((size_t)nrules * sizeof(unsigned short));
    out->gpuidx  = (int *)malloc((size_t)nrules * sizeof(int));
    out->cpuidx  = (int *)malloc((size_t)nrules * sizeof(int));
    if (!out->gpu || !out->cpu || !out->gpulen || !out->cpulen ||
        !out->gpuidx || !out->cpuidx) {
        rule_lists_free(out);
        return -1;
    }
    for (int i = 0; i < nrules; i++) {
        if (gpu_rule_safe_phase0(rules[i], (int)lens[i])) {
            out->gpulen[out->ngpu] = lens[i];
            out->gpuidx[out->ngpu] = i;          /* original Rules[] index */
            out->gpu[out->ngpu++]  = rules[i];
        } else {
            out->cpulen[out->ncpu] = lens[i];
            out->cpuidx[out->ncpu] = i;
            out->cpu[out->ncpu++]  = rules[i];
        }
    }
    return out->ngpu;
}

/* Free the gpu[] and cpu[] arrays. The rule strings themselves are NOT
 * freed — those are owned by the caller. The full[] alias is also left
 * alone (caller owns). */
void rule_lists_free(struct rule_lists *rl) {
    if (!rl) return;
    free(rl->gpu);    free(rl->cpu);
    free(rl->gpulen); free(rl->cpulen);
    free(rl->gpuidx); free(rl->cpuidx);
    memset(rl, 0, sizeof(*rl));   /* full[]/fulllen[] are aliases, caller-owned */
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
    int x, y, z, clen, tlen;
#ifndef NOTINTEL
    __m128i *p128, a128,b128,c128,d128;
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
        _validate_cached = 0;   /* was MDXFIND_RULE_VALIDATOR */
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

      /* `=NX` reject unless the character at position N is X.
       * `%NX` reject unless X occurs at least N times.
       *
       * Documented and AGREEING in both references -- John: "reject the word
       * unless character in position N is equal to X" / "unless it contains
       * at least N instances of X"; hashcat: "reject plains that do not
       * contain char X at pos N" / "that contain char X less than N times".
       *
       * packrules already emitted the right shape (opcode, then the
       * position/count position-translated, then the literal character) but
       * no executor existed, and the default case skipped only the OPCODE
       * byte.  The two operand bytes were then executed as instructions, so
       * `=0p $Z` yielded `passe` instead of `passerZ` -- silent
       * misexecution, not the documented no-op.  198 rules in
       * HashMob.100k.rule and 83,494 in rules/Hash-IT_Crazy_Rules.rule
       * contain one of these verbs.  Verified against john.  Both are
       * CPU-only: classify_rules does not list 0x3d/0x25, so such rules fall
       * to its default and route away from the GPU, which is correct. */
      case '=':
        y = *rule++ - 1;
        c = *rule++;
        if (y >= clen || (unsigned char)cpass[y] != c)
          { _retval = (-1); goto _validate_exit; }
        break;

      case '%':
        y = *rule++ - 1;
        c = *rule++;
        { int _cnt = 0;
          for (x = 0; x < clen; x++)
            if ((unsigned char)cpass[x] == c) _cnt++;
          if (_cnt < y) { _retval = (-1); goto _validate_exit; } }
        break;

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
        /* `X N M I`: insert M characters of the memory buffer, starting at
         * OFFSET N within it, at position I of the candidate.
         *
         * N was read and then ignored -- the copy was Memory[x] rather than
         * Memory[y + x] -- so every offset produced the same result:
         * MX034/MX134/MX234 on "abcdef" all gave "abcdabcef" where john and
         * hashcat give abcdabcef / abcdbcdef / abcdcdeef.  Both references
         * agree on that, so it was simply wrong.  Fixed 2026-09-11.
         *
         * OUT-OF-RANGE FOLLOWS HASHCAT (operator ruling, 2026-09-11; this
         * reverses an earlier reading that clamped like john).  hashcat's
         * mangle_insert_multi() rejects rather than clamping, on exactly
         * these conditions:
         *   no memory stored           mem_len < 1
         *   insert position past end   I > clen
         *   offset past memory         N > memlen
         *   range overruns memory      N + M > memlen
         *   zero count                 M < 1
         *   result too long            clen + M > buffer
         * john instead clamps I to the end and M to what memory holds, so
         * MX049 on "abcdef" is "abcdefabcd" in john and a reject here.
         *
         * NOTE hashcat's own implementation also MUTATES its memory buffer
         * while reading it (the memmove in mangle_insert_multi), so a second
         * X in the same rule sees shifted memory.  That is not replicated:
         * it reads as a defect in hashcat, not as semantics.  Reported. */
        y = *rule++ - 1;
	tlen = *rule++ - 1;
	z = *rule++ - 1;
	if (memlen < 1 || tlen < 1 || z > clen || y > memlen ||
	    (y + tlen) > memlen)
	    { _retval = (-1); goto _validate_exit; }
	if ((clen + tlen) > FASTLEN) 
	    goto slowrule;
	for (x=clen; x >= z; x--)
	   cpass[x+tlen] = cpass[x];
	for (x=0; x < tlen; x++) 
	   cpass[x+z] = Memory[y+x];
	clen += tlen;
	break;

        


      case RULE_OP_REJ_LEN_NE:
      case '_':
        /* John: "_N reject the word unless it is N characters long" -- the
         * CURRENT length at this point in the rule, not the original word's.
         * This tested `len`, so `$a _5` on a 4-char word rejected even though
         * the candidate is 5 characters by then, and `$a _4` wrongly kept it.
         * `<` and `>` alongside already use clen. */
        y = *rule++ - 1;
	if (y != clen)
	    { _retval = (-1); goto _validate_exit; }
	break;
      case RULE_OP_REJ_LEN_GE:
      case '<':
        /* John: "<N reject the word unless it is less than N characters
         * long" -- so reject when clen >= N.  This tested clen < y, i.e. it
         * kept clen >= N, the exact complement.  Operator ruling 2026-09-11:
         * match John, which is both documented and runnable; hashcat's doc
         * says "reject plains of length greater than N" (inclusive) and its
         * release will not run the verb at all. */
        y = *rule++ - 1;
        if (clen >= y)
          { _retval = (-1); goto _validate_exit; }
        break;
      case RULE_OP_REJ_LEN_LE:
      case '>':
        /* John: ">N reject the word unless it is greater than N characters
         * long" -- so reject when clen <= N.  Complement of the old test.
         * Same ruling as `<` above. */
        y = *rule++ - 1;
        if (clen <= y)
          { _retval = (-1); goto _validate_exit; }
        break;

      /* ---- character-class forms (D6, 2026-09-11) --------------------
       * Each mirrors its literal-operand sibling with RULE_CLASS_MATCH() in
       * place of the equality test.  The complement lives in the high bit of
       * the class byte and is applied by XOR inside the macro, so `?D` costs
       * exactly what `?d` costs.
       *
       * CPU-only: classify_rules does not admit 0x80-0x88, so a rule using
       * one routes to the CPU list.  That is correct, not a gap -- the same
       * arrangement as `=` and `%`.
       */
      case RULE_OP_SUB_CLASS: {
        unsigned char _cb = (unsigned char)*rule++;
        c = *rule++;
        for (x = 0; x < clen; x++)
          if (RULE_CLASS_MATCH(_cb, cpass[x]))
            cpass[x] = c;
        break; }

      case RULE_OP_PURGE_CLASS: {
        unsigned char _cb = (unsigned char)*rule++;
        d = cpass;
        s = cpass;
        for (x = 0; x < clen; x++) {
          if (!RULE_CLASS_MATCH(_cb, *s))
            *d++ = *s;
          s++;
        }
        clen -= (s - d);
        if (clen < 0)
          clen = 0;
        break; }

      case RULE_OP_TITLE_CLASS: {
        /* John's e?C -- the literal `e` shape with a class separator test.
         * Positional word start, as for `e`. */
        unsigned char _cb = (unsigned char)*rule++;
        for (z = 0, x = 0; x < clen; x++) {
          c = cpass[x];
          if (RULE_CLASS_MATCH(_cb, c)) { z = 0; continue; }
          if (z == 0) {
            z = 1;
            if (c >= 'a' && c <= 'z') cpass[x] = c ^ 0x20;
          } else {
            if (c >= 'A' && c <= 'Z') cpass[x] = c ^ 0x20;
          }
        }
        break; }

      case RULE_OP_TITLE_CLASS_HC: {
        /* hashcat mangle_title_sep_class_*: lowercase every position, then
         * uppercase position 0 and every position whose PREDECESSOR was in
         * the class.  The class test reads the byte before it is modified,
         * and the separator is itself case-normalised -- both differences
         * from John's e?C above. */
        unsigned char _cb = (unsigned char)*rule++;
        int _up = 1;
        for (x = 0; x < clen; x++) {
          int _this = _up;
          c = cpass[x];
          _up = RULE_CLASS_MATCH(_cb, c) ? 1 : 0;
          if (c >= 'A' && c <= 'Z') { c ^= 0x20; cpass[x] = c; }
          if (_this && c >= 'a' && c <= 'z')
            cpass[x] = c ^ 0x20;
        }
        break; }

      case RULE_OP_REJ_HAS_CLASS: {
        unsigned char _cb = (unsigned char)*rule++;
        for (x = 0; x < clen; x++)
          if (RULE_CLASS_MATCH(_cb, cpass[x]))
            { _retval = (-1); goto _validate_exit; }
        break; }

      case RULE_OP_REJ_NHAS_CLASS: {
        unsigned char _cb = (unsigned char)*rule++;
        for (x = 0; x < clen; x++)
          if (RULE_CLASS_MATCH(_cb, cpass[x]))
            break;
        if (x >= clen)
          { _retval = (-1); goto _validate_exit; }
        break; }

      case RULE_OP_REJ_FIRST_CLASS: {
        unsigned char _cb = (unsigned char)*rule++;
        if (clen > 0 && !RULE_CLASS_MATCH(_cb, cpass[0]))
          { _retval = (-1); goto _validate_exit; }
        break; }

      case RULE_OP_REJ_LAST_CLASS: {
        unsigned char _cb = (unsigned char)*rule++;
        if (clen > 0 && !RULE_CLASS_MATCH(_cb, cpass[clen - 1]))
          { _retval = (-1); goto _validate_exit; }
        break; }

      case RULE_OP_REJ_AT_CLASS: {
        unsigned char _cb;
        y = *rule++ - 1;
        _cb = (unsigned char)*rule++;
        if (y >= clen || !RULE_CLASS_MATCH(_cb, cpass[y]))
          { _retval = (-1); goto _validate_exit; }
        break; }

      case RULE_OP_REJ_CNT_CLASS: {
        unsigned char _cb;
        int _cnt = 0;
        y = *rule++ - 1;
        _cb = (unsigned char)*rule++;
        for (x = 0; x < clen; x++)
          if (RULE_CLASS_MATCH(_cb, cpass[x]))
            _cnt++;
        if (_cnt < y)
          { _retval = (-1); goto _validate_exit; }
        break; }

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
        /* John and hashcat both act on POSITION 0, not on the first
         * alphabetic character: `c` on "!bang" is "!bang" in both, where
         * mdxfind gave "!Bang".  Same for `C`: "!BANG" in both, "!bANG"
         * here.  Verified word-by-word against john --stdout and against
         * hashcat's own _old_apply_rule().  Fixed 2026-09-11. */
#ifdef NOTINTEL
        for (x = 0; x < clen; x++) {
          c = cpass[x];
          if (c >= 'A' && c <= 'Z')
            cpass[x] = c + 0x20;
        }
        if (clen > 0) {
          c = cpass[0];
          if (c >= 'a' && c <= 'z')
            cpass[0] = c - 0x20;
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
	if (clen > 0) {
	    c = cpass[0];
	    if (c >= 'a' && c <= 'z')
	        cpass[0] = c ^ 0x20;
	}
#endif
        break;

      case RULE_OP_CAP_INV:
      case 'C':
#ifdef NOTINTEL
        for (x = 0; x < clen; x++) {
          c = cpass[x];
          if (c >= 'a' && c <= 'z')
            cpass[x] = c - 0x20;
        }
        if (clen > 0) {
          c = cpass[0];
          if (c >= 'A' && c <= 'Z')
            cpass[0] = c + 0x20;
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
	if (clen > 0) {
	    c = cpass[0];
	    if (c >= 'A' && c <= 'Z')
	        cpass[0] = c ^ 0x20;
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
        /* Bounds check: john guards this explicitly (rules.c case 'T':
         * "if (pos < length)").  Without it a position up to 61 reads --
         * and can write -- shared workspace past the candidate, the same
         * shape as the `.N` one-past-the-end read.  Not observable today
         * because the bytes there are NUL, so nothing toggles and clen is
         * unchanged; all three engines agree on T3/T9/TZ for "abc".  The
         * guard is one compare and removes the hazard. */
        y = *rule++ - 1;
        if (y < clen) {
          c = cpass[y];
          if ((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z'))
            cpass[y] = c ^ 0x20;
        }
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
        /* `x N M`: extract M characters from position N.
         *
         * FOLLOWS HASHCAT (operator ruling 2026-09-11, superseding an earlier
         * ruling for john).  hashcat's mangle_extract() no-ops on BOTH
         * out-of-range conditions and never extracts a partial run:
         *     if (upos >= arr_len)          return arr_len;
         *     if ((upos + ulen) > arr_len)  return arr_len;
         * john instead rejects an out-of-range START and extracts as much as
         * is available when the COUNT overruns, so on "abc":
         *     x22   john "c"     hashcat "abc"
         *     x23   john "c"     hashcat "abc"
         *     x90   john REJECT  hashcat "abc"
         * A zero count with an in-range start still yields an EMPTY
         * candidate in both, which john then rejects under its empty-word
         * policy -- also ruled to hashcat, so it is kept here. */
        y = *rule++ - 1;
        z = *rule++ - 1;
        if (y < clen && (y + z) <= clen) {
          for (x = 0; x < z; x++)
            cpass[x] = cpass[y + x];
          clen = z;
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
        /* `>=`, not `>`: inserting at position == length is an APPEND, and
         * both references do it -- `i1a` on "a" gives "aa" in john and in
         * hashcat 6.2.5, where this gave "a" unchanged.  The shift loop
         * degenerates correctly at y == clen.  Position BEYOND the length
         * stays a no-op, which matches hashcat; john appends there instead
         * and the two references disagree, so that case is left alone. */
        if (clen >= y) {
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
        /* Unsigned cast for the same reason as SHR below, plus one of its own:
         * `<< 1` on a NEGATIVE signed char is undefined behaviour in C.  The
         * value produced is unchanged on every platform we build -- the top bit
         * is discarded either way -- so this is a UB fix, not a semantic one. */
          cpass[y] = (char)(((unsigned char)cpass[y]) << 1);
        break;

      case RULE_OP_BIT_SHR:
      case 'R':
        y = *rule++ - 1;
        if (y < clen)
        /* LOGICAL shift, not arithmetic.  `pass` is `char *`, which is SIGNED
         * on x86 and on Apple ARM64, so a bare `>> 1` sign-extends: byte 0x83
         * became 0xc1 rather than 0x41.  These are BYTE strings; a 1-bit must
         * not appear at the top of a byte that had none.  The unsigned cast
         * makes the intent explicit rather than inheriting it from the ABI --
         * plain char is UNSIGNED by default on Linux aarch64, so the old code
         * also disagreed with itself across the fleet.
         *
         * hashcat's rp_cpu.c has the same bug (`char arr[]`, `arr[upos] >>= 1`)
         * but never exercises it, because it works in ASCII where the sign bit
         * is never set.  Operator ruling 2026-09-14: logical is correct and
         * hashcat is wrong here; do NOT "restore" compatibility.
         *
         * Measured cost of the old behaviour: on 2,009 encoding-diverse words
         * the CPU and the GPU kernel disagreed on 182 of 28,440 candidates from
         * the single rule R0, and on a hash list containing the device's digest
         * mdxfind emitted a hash:plaintext pair that did not verify.
         */
          cpass[y] = (char)(((unsigned char)cpass[y]) >> 1);
        break;

      /* hashcat `BNX`: add the byte value of X to the byte at position N,
       * wrapping.  Out of range is a no-op, matching hashcat's
       * mangle_chr_add().  hashcat-only; John has no `B`. */
      case RULE_OP_CHR_ADD:
      case 'B':
        y = *rule++ - 1;
        c = *rule++;
        if (y < clen)
          cpass[y] = (unsigned char)(cpass[y] + c);
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
        /* Guard y+1, which is the byte actually READ, not just y.  `y < clen`
         * admitted y == clen-1 and then read cpass[clen] -- one past the end
         * of the candidate, picking up whatever the shared workspace held
         * from a previous, longer candidate.  The output therefore depended
         * on wordlist order and thread split: `K $1 .9 $0` on Jay020171 gave
         * Jay020117Z0 after a run of Zs, Jay020117a0 after a run of as, and
         * Jay0201170 with a clean buffer.
         *
         * hashcat settles what the right answer is: an out-of-range `.N` is a
         * NO-OP and the candidate is still emitted.  Verified with
         * `hashcat --stdout` on abcdefghij: `.8` gives abcdefghjj, `.9` and
         * `.A` both give abcdefghij unchanged.  ruleproc32.c already did this;
         * the GPU kernels wrote a NUL instead, which is a third answer again
         * and is fixed in the same revision.
         *
         * `,` below needs no change: its `y > 0` clause already guards the
         * y-1 read, and hashcat agrees -- `,0` is a no-op, `,1` applies. */
        y = *rule++ - 1;
        if (y + 1 < clen)
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
        /* `E` is `e` with a space separator -- see the note on
         * RULE_OP_TITLE_SEP below for why the word start is positional. */
        for (z = 0, x = 0; x < clen; x++) {
          c = cpass[x];
          if (c == ' ') { z = 0; continue; }
          if (z == 0) {
            z = 1;
            if (c >= 'a' && c <= 'z') cpass[x] = c ^ 0x20;
          } else {
            if (c >= 'A' && c <= 'Z') cpass[x] = c ^ 0x20;
          }
        }
        break;
      case RULE_OP_TITLE_SEP:
      case 'e':
	c1 = *rule++;
        /* Word start is POSITIONAL: the first character after a
         * separator (or position 0), whether or not it is a letter.  The
         * old code only consumed the word start when it SAW a letter, so a
         * leading non-letter left it pending and the next letter was
         * capitalised instead: "!bang" gave "!Bang" where john and hashcat
         * both give "!bang", and "9nine" gave "9Nine" for "9nine".
         *
         * hashcat additionally force-uppercases position 0 at the end
         * (mangle_title_sep's trailing MANGLE_UPPER_AT(arr,0)), so `ea` on
         * "apple" is "APple" there and "aPple" in john.  We follow john --
         * that is also what mdxfind has always done -- and the difference
         * only shows when the separator itself sits at position 0. */
        for (z = 0, x = 0; x < clen; x++) {
          c = cpass[x];
          if (c == c1) { z = 0; continue; }   /* next char starts a word */
          if (z == 0) {
            z = 1;
            if (c >= 'a' && c <= 'z') cpass[x] = c ^ 0x20;
          } else {
            if (c >= 'A' && c <= 'Z') cpass[x] = c ^ 0x20;
          }
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
      /* `=NX` reject unless the character at position N is X.
       * `%NX` reject unless X occurs at least N times.
       *
       * Documented and AGREEING in both references -- John: "reject the word
       * unless character in position N is equal to X" / "unless it contains
       * at least N instances of X"; hashcat: "reject plains that do not
       * contain char X at pos N" / "that contain char X less than N times".
       *
       * packrules already emitted the right shape (opcode, then the
       * position/count position-translated, then the literal character) but
       * no executor existed, and the default case skipped only the OPCODE
       * byte.  The two operand bytes were then executed as instructions, so
       * `=0p $Z` yielded `passe` instead of `passerZ` -- silent
       * misexecution, not the documented no-op.  198 rules in
       * HashMob.100k.rule and 83,494 in rules/Hash-IT_Crazy_Rules.rule
       * contain one of these verbs.  Verified against john.  Both are
       * CPU-only: classify_rules does not list 0x3d/0x25, so such rules fall
       * to its default and route away from the GPU, which is correct. */
      case '=':
        y = *rule++ - 1;
        c = *rule++;
        if (y >= clen || (unsigned char)pass[y] != c)
          { _retval = (-1); goto _validate_exit; }
        break;

      case '%':
        y = *rule++ - 1;
        c = *rule++;
        { int _cnt = 0;
          for (x = 0; x < clen; x++)
            if ((unsigned char)pass[x] == c) _cnt++;
          if (_cnt < y) { _retval = (-1); goto _validate_exit; } }
        break;

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
        /* `X N M I`: insert M characters of the memory buffer, starting at
         * OFFSET N within it, at position I of the candidate.
         *
         * N was read and then ignored -- the copy was Memory[x] rather than
         * Memory[y + x] -- so every offset produced the same result:
         * MX034/MX134/MX234 on "abcdef" all gave "abcdabcef" where john and
         * hashcat give abcdabcef / abcdbcdef / abcdcdeef.  Both references
         * agree on that, so it was simply wrong.  Fixed 2026-09-11.
         *
         * OUT-OF-RANGE FOLLOWS HASHCAT (operator ruling, 2026-09-11; this
         * reverses an earlier reading that clamped like john).  hashcat's
         * mangle_insert_multi() rejects rather than clamping, on exactly
         * these conditions:
         *   no memory stored           mem_len < 1
         *   insert position past end   I > clen
         *   offset past memory         N > memlen
         *   range overruns memory      N + M > memlen
         *   zero count                 M < 1
         *   result too long            clen + M > buffer
         * john instead clamps I to the end and M to what memory holds, so
         * MX049 on "abcdef" is "abcdefabcd" in john and a reject here.
         *
         * NOTE hashcat's own implementation also MUTATES its memory buffer
         * while reading it (the memmove in mangle_insert_multi), so a second
         * X in the same rule sees shifted memory.  That is not replicated:
         * it reads as a defect in hashcat, not as semantics.  Reported. */
        y = *rule++ - 1;
	tlen = *rule++ - 1;
	z = *rule++ - 1;
	if (memlen < 1 || tlen < 1 || z > clen || y > memlen ||
	    (y + tlen) > memlen)
	    { _retval = (-1); goto _validate_exit; }
	if ((clen + tlen) > MAXLINE) 
	    tlen = MAXLINE - clen;
	for (x=clen; x >= z; x--)
	   pass[x+tlen] = pass[x];
	for (x=0; x < tlen; x++) 
	   pass[x+z] = Memory[y+x];
	clen += tlen;
	break;

        


      case RULE_OP_REJ_LEN_NE:
      case '_':
        /* John: "_N reject the word unless it is N characters long" -- the
         * CURRENT length at this point in the rule, not the original word's.
         * This tested `len`, so `$a _5` on a 4-char word rejected even though
         * the candidate is 5 characters by then, and `$a _4` wrongly kept it.
         * `<` and `>` alongside already use clen. */
        y = *rule++ - 1;
	if (y != clen)
	    { _retval = (-1); goto _validate_exit; }
	break;
      case RULE_OP_REJ_LEN_GE:
      case '<':
        /* John: "<N reject the word unless it is less than N characters
         * long" -- so reject when clen >= N.  This tested clen < y, i.e. it
         * kept clen >= N, the exact complement.  Operator ruling 2026-09-11:
         * match John, which is both documented and runnable; hashcat's doc
         * says "reject plains of length greater than N" (inclusive) and its
         * release will not run the verb at all. */
        y = *rule++ - 1;
        if (clen >= y)
          { _retval = (-1); goto _validate_exit; }
        break;
      case RULE_OP_REJ_LEN_LE:
      case '>':
        /* John: ">N reject the word unless it is greater than N characters
         * long" -- so reject when clen <= N.  Complement of the old test.
         * Same ruling as `<` above. */
        y = *rule++ - 1;
        if (clen <= y)
          { _retval = (-1); goto _validate_exit; }
        break;

      /* ---- character-class forms (D6, 2026-09-11) --------------------
       * Each mirrors its literal-operand sibling with RULE_CLASS_MATCH() in
       * place of the equality test.  The complement lives in the high bit of
       * the class byte and is applied by XOR inside the macro, so `?D` costs
       * exactly what `?d` costs.
       *
       * CPU-only: classify_rules does not admit 0x80-0x88, so a rule using
       * one routes to the CPU list.  That is correct, not a gap -- the same
       * arrangement as `=` and `%`.
       */
      case RULE_OP_SUB_CLASS: {
        unsigned char _cb = (unsigned char)*rule++;
        c = *rule++;
        for (x = 0; x < clen; x++)
          if (RULE_CLASS_MATCH(_cb, pass[x]))
            pass[x] = c;
        break; }

      case RULE_OP_PURGE_CLASS: {
        unsigned char _cb = (unsigned char)*rule++;
        d = pass;
        s = pass;
        for (x = 0; x < clen; x++) {
          if (!RULE_CLASS_MATCH(_cb, *s))
            *d++ = *s;
          s++;
        }
        clen -= (s - d);
        if (clen < 0)
          clen = 0;
        break; }

      case RULE_OP_TITLE_CLASS: {
        /* John's e?C -- the literal `e` shape with a class separator test.
         * Positional word start, as for `e`. */
        unsigned char _cb = (unsigned char)*rule++;
        for (z = 0, x = 0; x < clen; x++) {
          c = pass[x];
          if (RULE_CLASS_MATCH(_cb, c)) { z = 0; continue; }
          if (z == 0) {
            z = 1;
            if (c >= 'a' && c <= 'z') pass[x] = c ^ 0x20;
          } else {
            if (c >= 'A' && c <= 'Z') pass[x] = c ^ 0x20;
          }
        }
        break; }

      case RULE_OP_TITLE_CLASS_HC: {
        /* hashcat mangle_title_sep_class_*: lowercase every position, then
         * uppercase position 0 and every position whose PREDECESSOR was in
         * the class.  The class test reads the byte before it is modified,
         * and the separator is itself case-normalised -- both differences
         * from John's e?C above. */
        unsigned char _cb = (unsigned char)*rule++;
        int _up = 1;
        for (x = 0; x < clen; x++) {
          int _this = _up;
          c = pass[x];
          _up = RULE_CLASS_MATCH(_cb, c) ? 1 : 0;
          if (c >= 'A' && c <= 'Z') { c ^= 0x20; pass[x] = c; }
          if (_this && c >= 'a' && c <= 'z')
            pass[x] = c ^ 0x20;
        }
        break; }

      case RULE_OP_REJ_HAS_CLASS: {
        unsigned char _cb = (unsigned char)*rule++;
        for (x = 0; x < clen; x++)
          if (RULE_CLASS_MATCH(_cb, pass[x]))
            { _retval = (-1); goto _validate_exit; }
        break; }

      case RULE_OP_REJ_NHAS_CLASS: {
        unsigned char _cb = (unsigned char)*rule++;
        for (x = 0; x < clen; x++)
          if (RULE_CLASS_MATCH(_cb, pass[x]))
            break;
        if (x >= clen)
          { _retval = (-1); goto _validate_exit; }
        break; }

      case RULE_OP_REJ_FIRST_CLASS: {
        unsigned char _cb = (unsigned char)*rule++;
        if (clen > 0 && !RULE_CLASS_MATCH(_cb, pass[0]))
          { _retval = (-1); goto _validate_exit; }
        break; }

      case RULE_OP_REJ_LAST_CLASS: {
        unsigned char _cb = (unsigned char)*rule++;
        if (clen > 0 && !RULE_CLASS_MATCH(_cb, pass[clen - 1]))
          { _retval = (-1); goto _validate_exit; }
        break; }

      case RULE_OP_REJ_AT_CLASS: {
        unsigned char _cb;
        y = *rule++ - 1;
        _cb = (unsigned char)*rule++;
        if (y >= clen || !RULE_CLASS_MATCH(_cb, pass[y]))
          { _retval = (-1); goto _validate_exit; }
        break; }

      case RULE_OP_REJ_CNT_CLASS: {
        unsigned char _cb;
        int _cnt = 0;
        y = *rule++ - 1;
        _cb = (unsigned char)*rule++;
        for (x = 0; x < clen; x++)
          if (RULE_CLASS_MATCH(_cb, pass[x]))
            _cnt++;
        if (_cnt < y)
          { _retval = (-1); goto _validate_exit; }
        break; }

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
        /* John and hashcat both act on POSITION 0, not on the first
         * alphabetic character: `c` on "!bang" is "!bang" in both, where
         * mdxfind gave "!Bang".  Same for `C`: "!BANG" in both, "!bANG"
         * here.  Verified word-by-word against john --stdout and against
         * hashcat's own _old_apply_rule().  Fixed 2026-09-11. */
#ifdef NOTINTEL
        for (x = 0; x < clen; x++) {
          c = pass[x];
          if (c >= 'A' && c <= 'Z')
            pass[x] = c + 0x20;
        }
        if (clen > 0) {
          c = pass[0];
          if (c >= 'a' && c <= 'z')
            pass[0] = c - 0x20;
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
	if (clen > 0) {
	    c = pass[0];
	    if (c >= 'a' && c <= 'z')
	        pass[0] = c ^ 0x20;
	}
#endif
        break;

      case RULE_OP_CAP_INV:
      case 'C':
#ifdef NOTINTEL
        for (x = 0; x < clen; x++) {
          c = pass[x];
          if (c >= 'a' && c <= 'z')
            pass[x] = c - 0x20;
        }
        if (clen > 0) {
          c = pass[0];
          if (c >= 'A' && c <= 'Z')
            pass[0] = c + 0x20;
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
	if (clen > 0) {
	    c = pass[0];
	    if (c >= 'A' && c <= 'Z')
	        pass[0] = c ^ 0x20;
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
        /* Bounds check: john guards this explicitly (rules.c case 'T':
         * "if (pos < length)").  Without it a position up to 61 reads --
         * and can write -- shared workspace past the candidate, the same
         * shape as the `.N` one-past-the-end read.  Not observable today
         * because the bytes there are NUL, so nothing toggles and clen is
         * unchanged; all three engines agree on T3/T9/TZ for "abc".  The
         * guard is one compare and removes the hazard. */
        y = *rule++ - 1;
        if (y < clen) {
          c = pass[y];
          if ((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z'))
            pass[y] = c ^ 0x20;
        }
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
        /* `x N M`: extract M characters from position N.
         *
         * FOLLOWS HASHCAT (operator ruling 2026-09-11, superseding an earlier
         * ruling for john).  hashcat's mangle_extract() no-ops on BOTH
         * out-of-range conditions and never extracts a partial run:
         *     if (upos >= arr_len)          return arr_len;
         *     if ((upos + ulen) > arr_len)  return arr_len;
         * john instead rejects an out-of-range START and extracts as much as
         * is available when the COUNT overruns, so on "abc":
         *     x22   john "c"     hashcat "abc"
         *     x23   john "c"     hashcat "abc"
         *     x90   john REJECT  hashcat "abc"
         * A zero count with an in-range start still yields an EMPTY
         * candidate in both, which john then rejects under its empty-word
         * policy -- also ruled to hashcat, so it is kept here. */
        y = *rule++ - 1;
        z = *rule++ - 1;
        if (y < clen && (y + z) <= clen) {
          for (x = 0; x < z; x++)
            pass[x] = pass[y + x];
          clen = z;
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
        /* `>=`, not `>`: see the note in the FASTLEN path above -- inserting
         * at position == length is an append, and both references do it. */
        if (clen >= y) {
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
        /* Unsigned cast for the same reason as SHR below, plus one of its own:
         * `<< 1` on a NEGATIVE signed char is undefined behaviour in C.  The
         * value produced is unchanged on every platform we build -- the top bit
         * is discarded either way -- so this is a UB fix, not a semantic one. */
          pass[y] = (char)(((unsigned char)pass[y]) << 1);
        break;

      case RULE_OP_BIT_SHR:
      case 'R':
        y = *rule++ - 1;
        if (y < clen)
        /* LOGICAL shift, not arithmetic.  `pass` is `char *`, which is SIGNED
         * on x86 and on Apple ARM64, so a bare `>> 1` sign-extends: byte 0x83
         * became 0xc1 rather than 0x41.  These are BYTE strings; a 1-bit must
         * not appear at the top of a byte that had none.  The unsigned cast
         * makes the intent explicit rather than inheriting it from the ABI --
         * plain char is UNSIGNED by default on Linux aarch64, so the old code
         * also disagreed with itself across the fleet.
         *
         * hashcat's rp_cpu.c has the same bug (`char arr[]`, `arr[upos] >>= 1`)
         * but never exercises it, because it works in ASCII where the sign bit
         * is never set.  Operator ruling 2026-09-14: logical is correct and
         * hashcat is wrong here; do NOT "restore" compatibility.
         *
         * Measured cost of the old behaviour: on 2,009 encoding-diverse words
         * the CPU and the GPU kernel disagreed on 182 of 28,440 candidates from
         * the single rule R0, and on a hash list containing the device's digest
         * mdxfind emitted a hash:plaintext pair that did not verify.
         */
          pass[y] = (char)(((unsigned char)pass[y]) >> 1);
        break;

      /* hashcat `BNX`: add the byte value of X to the byte at position N,
       * wrapping.  Out of range is a no-op, matching hashcat's
       * mangle_chr_add().  hashcat-only; John has no `B`. */
      case RULE_OP_CHR_ADD:
      case 'B':
        y = *rule++ - 1;
        c = *rule++;
        if (y < clen)
          pass[y] = (unsigned char)(pass[y] + c);
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
        /* Same one-past-the-end read as the FASTLEN path above, same fix.
         * Guard y+1, which is the byte read; hashcat no-ops out of range. */
        y = *rule++ - 1;
        if (y + 1 < clen)
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
        /* `E` is `e` with a space separator -- see the note on
         * RULE_OP_TITLE_SEP below for why the word start is positional. */
        for (z = 0, x = 0; x < clen; x++) {
          c = pass[x];
          if (c == ' ') { z = 0; continue; }
          if (z == 0) {
            z = 1;
            if (c >= 'a' && c <= 'z') pass[x] = c ^ 0x20;
          } else {
            if (c >= 'A' && c <= 'Z') pass[x] = c ^ 0x20;
          }
        }
        break;
      case RULE_OP_TITLE_SEP:
      case 'e':
	c1 = *rule++;
        /* Word start is POSITIONAL: the first character after a
         * separator (or position 0), whether or not it is a letter.  The
         * old code only consumed the word start when it SAW a letter, so a
         * leading non-letter left it pending and the next letter was
         * capitalised instead: "!bang" gave "!Bang" where john and hashcat
         * both give "!bang", and "9nine" gave "9Nine" for "9nine".
         *
         * hashcat additionally force-uppercases position 0 at the end
         * (mangle_title_sep's trailing MANGLE_UPPER_AT(arr,0)), so `ea` on
         * "apple" is "APple" there and "aPple" in john.  We follow john --
         * that is also what mdxfind has always done -- and the difference
         * only shows when the separator itself sits at position 0. */
        for (z = 0, x = 0; x < clen; x++) {
          c = pass[x];
          if (c == c1) { z = 0; continue; }   /* next char starts a word */
          if (z == 0) {
            z = 1;
            if (c >= 'a' && c <= 'z') pass[x] = c ^ 0x20;
          } else {
            if (c >= 'A' && c <= 'Z') pass[x] = c ^ 0x20;
          }
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

