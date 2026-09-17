/* $Revision: 1.16 $
 *
 * $Log: ruleproc32.c,v $
 * Revision 1.16  2026/09/15 06:05:39  dlr
 * Fold `c` cased-scan into its lowercase pass, and CORRECT the figures stated in 1.15.
 *
 * CORRECTION TO 1.15 FIRST. That revision says single-pass upper_all is worth "4-5% on a real rule file" without naming a host, and the number came from mdxfind "N seconds hashing", which is QUANTIZED TO ONE SECOND -- across 185 logs every value ends .02/.03/.04. On a nine-second cell that quantum is +-11% and it is bimodal, not noisy, so repeats do not average it out. Re-measured on the GPU device summary busy= field, 10ms resolution, three readings per cell, byte mode carried as an invariant control:
 *
 *   verb          fpga GTX 1080        mmt RTX 4070 Ti SUPER
 *   u             17.02 -> 8.94 s      2.16 -> 1.42 s
 *                 1.904x               1.521x
 *   C             20.84 -> 12.66 s     2.80 -> 1.83 s
 *                 1.646x               1.530x
 *   HashMob.100k  75.74 -> 73.18 s     8.12 -> 8.03 s
 *                 1.035x               1.012x
 *   byte control  14.90 -> 14.90 s     3.55 -> 3.55 s
 *                 1.000x               1.000x
 *
 * So 1.15 is right about u, 1.904x against the 1.89x it claims. Its real-rule-file figure is right FOR fpga, 2.5 to 4.9 percent across five rulesets, and wrong as a general claim: mmt is a third of that at 1.1 to 1.2 percent. The gain is DEVICE-DEPENDENT and future figures should name the host.
 *
 * Two things 1.15 did not know. `C` gains as much as `u` -- 1.646x on fpga, 1.530x on mmt -- because CAP_INV calls upper_all too; it was never measured before this run. And on fpga EVERY UTF-32 verb gained 2 to 6 percent, including verbs that touch neither change (l 1.049x, r 1.056x, t 1.057x, x02 1.058x, T0 1.055x, D1 1.047x, sab 1.032x, none 1.020x), because deleting the two-pass slow path shrinks the kernel. The byte control at exactly 1.000x is what proves that is real rather than drift or run-order bias: byte mode runs LAST in the script and did not move. mmt shows none of it, so a larger card absorbs the instruction-footprint effect.
 *
 * AND ONE REGRESSION, which the same control makes real: `d` got 2.3 percent SLOWER on fpga, 11.80 -> 12.08 s, identical across all three readings, while flat on mmt. Duplicate touches neither change. Code layout shifted and one verb landed worse. Not understood; recorded rather than rounded away.
 *
 * NOW THIS CHANGE. `c` lowercased the whole buffer and then walked it AGAIN hunting the first cased codepoint. That second loop breaks at the first cased codepoint, which sounds cheap and is not: MEASURED on UTF8test2.txt, 63.20 percent of words carry NO cased codepoint at all, so the break never fires and the scan runs the whole word -- 33.8 percent of a full extra pass overall. Cost on this kernel tracks the NUMBER OF PASSES over the codepoint buffer, so that was a third of a verb. The hunt now rides along with the lowercase pass: same number of cp_is_cased calls, one fewer traversal. fc survives greek_final_sigma, which runs between the two, because sigma and final sigma are both cased so it can neither move nor create the first cased position, and buf[fc] is read after that pass exactly as the old loop saw it.
 *
 * Worth 1.201x on fpga (12.84 -> 10.69 s) and 1.064x on mmt, and nothing measurable on a real rule file -- `c` is a small fraction of real rules -- so it is banked for the verb, not for the headline.
 *
 * Validated against the PREVIOUS implementation as an independent oracle, because the host twin compares the two engines and both changed identically: 713,488 output lines over eight verb combinations (c, C, c l, c r, c u, l c, E, c T0) across UTF8test2, UTF8test1, NONUTF8test1, the expanding-word fixture and cluster-words, all byte-identical after sorting -- sorted because procrule is multithreaded and its output order varies, which I first misread as a divergence. Greek checked separately for the sigma path: SOFOS -> Sofos with final sigma, both arms identical. Host twin 164,540,744 pairs DIFF=0.
 *
 * Revision 1.15  2026/09/15 03:19:27  dlr
 * Make upper_all single-pass, which is worth 1.89x on the u verb and 4-5% on a real rule file. It detected expansion in a pre-scan and then mapped, so u traversed the buffer twice where l traverses it once -- and MEASURED on an fpga.local GTX 1080 over 38,370 distinct rules x 55,000 UTF8test2 words, that is exactly what it cost: 123.9 Mh/s against l 233.7, a ratio of 1.886 against a predicted 2. Single-pass takes u to 234.0, landing on l 233.7 and t 234.0, the other one-traversal verbs. Whole-ruleset effect is smaller and is the honest figure: cluster-free 105.3 to 110.9, HashMob.100k 72.3 to 75.3, cluster-sensitive 44.9 to 45.8. Rule sets with no case verb also gain, 300.2 to 350.0, because the two-pass slow path becomes dead code and the kernel shrinks. Expansion is handled inline instead: cp_upper_full already returns 1 for every codepoint that does not grow, all but 0.0127 percent of a multi-script corpus, and the tail shift runs only for the rest. Growth cannot cascade, since no expansion output is itself an expanding codepoint, so i skips what was just written exactly as the backwards write did. The route here was a failure worth recording: a block bitmap derived from cp_expand, which removed 91.7 percent of the pre-scan binary searches and was correctness-clean, measured exactly 1.00x on the GPU and 1.065x on the CPU. The search was never the cost; the traversal was. Cost on this kernel tracks the NUMBER OF PASSES over the codepoint buffer and almost nothing else -- count passes before proposing anything here. Validated with the previous implementation as an independent oracle, since the host twin compares the two engines and both changed identically: 590,070 output lines over six uppercase verbs on UTF8test2, UTF8test1, NONUTF8test1 and the expanding-word fixture, plus byte mode, all byte-identical after sorting; host twin 164,540,744 pairs DIFF=0.
 *
 * Revision 1.14  2026/09/15 00:54:09  dlr
 * Record the measured outcome of tightening cl_all_plain, which was tried and reverted. The gate answers "can cluster index differ from codepoint index in this buffer" and is a bare >= U+0300 threshold: correct, and conservative. On UTF8test2.txt 66.34 percent of the walks it triggers turn out unnecessary, so asking the precise question cl_next asks -- extend, ZWJ, regional indicator, conjoining Hangul -- removes exactly those. It was byte-exact on 164,540,744 word-rule pairs and 1.5 to 1.6x SLOWER on an fpga.local GTX 1080: cluster-sensitive 45.84 to 28.12 Mh/s, cluster-free 112.85 to 75.26, byte mode unchanged as the control. Both arms were one tree differing in one object file. The precise test is not cheap, since cp_is_extend reaches cp_is_mark, a binary search over 290 ranges, in place of one compare, to save a walk whose own fast path is two compares per codepoint. The cluster-FREE half slowed by the same factor, which no per-call cost can explain. Comment only, no code change. The CPU gate is kept in step with the device deliberately: the two must stage identically or the host twin stops being an oracle.
 *
 * Revision 1.13  2026/09/13 02:38:37  dlr
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
 * Revision 1.12  2026/09/11 19:47:24  dlr
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
 * Revision 1.11  2026/09/11 02:04:40  dlr
 * Two fixes to the UTF-32 rule compiler, both found by measurement rather than by test failure, and both invisible in procrule output.
 *
 * First, the run collapser was defeated by whitespace. packrule32 detected a run of $ or ^ with a strict stride-2 scan, so any separator broke it, while ruleproc.c swallows space, tab and colon before its own run detection. Real rule files are space-separated, so the collapse fired on almost nothing. The old comment recorded the decision and its reasoning - stopping the run at a separator merely emits a second op and the output is identical, so this takes the version with less to get wrong - which was right about the output and never measured the cost. The cost is linear in run length, because each uncollapsed prepend shifts the whole buffer, which is the quadratic the collapse exists to remove. Measured on applyrule32 directly, space-separated prepends on a 28-codepoint word, 1M iterations: length 2 went 46.0ns to 28.3ns, length 5 went 93.7ns to 28.7ns, length 10 went 191.2ns to 31.4ns, length 18 went 398.8ns to 33.3ns and length 24 went 577.4ns to 34.7ns. Linear before, flat after. Published rulesets run short - 85.5 percent of HashMob.100k prepend runs are length 2 to 5, so 1.6x to 3.3x there - but long hand-written runs are used regularly and that is where the 6x to 16.6x lands, and where the byte engine's original optimisation was being forfeited. New skip_run_seps takes space, tab and colon; it is deliberately separate from skip_seps, whose other caller would parse differently because this engine emitted an op for a standalone colon. Two passes over the source rather than one into a buffer, since the run length must be emitted before the operands and a local array here would be a stack buffer. Capped at 254 to match ruleproc.c, whose run length is a single byte, so a longer run emits a second op exactly as it does there and the two packed forms stay structurally convertible. Whole-pipeline effect in procrule is nil and slightly negative in noise, which the arithmetic explains: 40ns saved on 13 percent of rules against roughly 1.3us of CPU per candidate is 0.4 percent. It should matter more in mdxfind on a fast unsalted type, where the CPU rule engine rather than the hashing is the bottleneck, but that is unmeasured and no claim is made.
 *
 * Second, a standalone colon now emits nothing, matching ruleproc.c. This is an mdxfind invariant rather than a style choice: mdxfind always tries the unmodified plain before any rule is applied, hashcat does not, so hashcat-derived rulesets conventionally carry a colon as their first rule, and under mdxfind that rule duplicates the implicit no-rule pass and is discarded at load time as no effect - a rule that packs to nothing has mystrlen 0 and never reaches RuleArray. Emitting RULE_OP_NOOP defeated that discard: the rule packed to a non-empty program, survived the loader, and produced a candidate identical to the plain, so the no-rule pass would be hashed twice and the duplicate credited to a rule in the -Z histogram.
 *
 * Neither defect is observable through procrule, which suppresses a candidate equal to the word it came from, so both engines look silent; both were found with a harness calling applyrule and applyrule32 directly. That harness also shows the colon case is the special case of a general gap that is NOT fixed here: applyrule ends with an explicit output-equals-input test and returns -2 for the auto-skip, applyrule32 has no such test, so any rule that is a no-op on a given word still yields a duplicate of the no-rule pass. That cannot be fixed in this file - -2 collides with RULE32_ERR_NOROOM and renumbering it would break a published contract - so it belongs in the eventual applyrule-convention bridge, which can compare codepoint arrays before encoding and synthesise -2 in the byte engine's return space.
 *
 * Recorded for whoever writes that bridge or a unified loader: packrule32_pass ends with EMIT(RULE32_END) and the terminator counts toward the returned length, so an empty program is length 0 in the byte form and length 1 in this one. A discard test written as length equal to zero would keep every colon rule alive and reintroduce exactly the duplicate this change removes.
 *
 * Gates: regress/rules32_test.py 165 of 165 with the driver rebuilt from source, since it has gone stale and reported false failures before; warning-clean under -Wall -Wextra as well as the Makefile flags; and output byte-identical to the pre-change binary across all_gj.rule and HashMob.100k.rule, 308,010 rules against 20 words, 12.4M candidates over both engines. The unreachable space and tab cases in the verb switch are left alone, the loop head having already skipped both.
 *
 * Revision 1.10  2026/09/07 03:22:12  dlr
 * Implement the backslash operand escape for $ and ^ in the UTF-32 engine, closing the last undeliberate divergence from the byte engine. ruleproc.c runs PARSEHEX on exactly two verbs, $ and ^ (sites 566/570 and 900/904); everywhere else the operand is a raw byte, including s, whose 20,638 corpus lines of the s9\ and s\1 shapes must keep meaning a literal backslash. So the escape goes in the $/^ case only, not into read_operand. Before this, ^\x32p1 meant prepend 2 then duplicate under the byte engine and, under -u, prepend a backslash and then re-read x32 as the extract verb: 2pass2pass against ssss. $\\ gave one backslash there and nothing here. Handles \xNN, \XNN, octal \0 to \3, \\ and a trailing backslash; anything else leaves the backslash as the operand and re-reads the next character as a token, which is what both engines already did. A value of 0x80 or above is refused with RULE32_ERR_INVALID rather than read as U+00NN: the byte engine inserts a raw byte there, a byte is not a codepoint, and this engine invents an interpretation nowhere else. The refusal is loud, printing Invalid rule line. Ignored. and naming the rule, and is the same return the packer already uses in about 30 places for a rule it cannot express; the quoted form $"e-acute" says it unambiguously and is already supported. Gate over the 20,433 backslash-bearing lines of the local 8.77M-line corpus, four words including Arabic and a Latin-1 supplement character: byte engine byte-identical, and under -u exactly one rule changes, ^\x32p1, changing to agree with the byte engine. rules32 conformance 165 of 165 before and after; the driver at /tmp/utf32_rule_drive is built from this file and reports 106 stale failures if it is not rebuilt.
 *
 * Revision 1.9  2026/09/06 14:52:29  dlr
 * Accept a literal double quote by re-parsing the whole line, not by heuristic.
 *
 * A double quote is both the operand delimiter and an ordinary character to append, insert or substitute. '$"' is unambiguous to a reader and, to read_operand, an operand that opens and never closes. 138 of the 17,320 catalog rules are exactly that shape -- $" ^" @" e" i0"..i9" o0"..o9" and 62 substitutions -- and every one was refused. That was the whole of the divergence from the byte engine: no rule without a quote differed.
 *
 * The line survives packing, so the resolution is to ask the parser twice rather than to guess from quote counts: once with quoted operands enabled and, only if that returns RULE32_ERR_INVALID, once with quotes as plain characters. RULE32_ERR_NOROOM is never retried, since that means the caller sized a buffer wrongly and retrying would hide it.
 *
 * Strictly additive by construction, and verified so: all 17,182 previously-valid rules produce byte-identical output before and after, and the byte engine is untouched. Catalog validity under -u goes 17,182 -> 17,320, matching the byte engine exactly. Quoted operands still parse as strings ($"123" -> pass123, s"a" "XY" -> pXYss, doubled quote still one literal), and unterminated rules with trailing text are still refused, because their remainder is not valid rule syntax under either reading.
 *
 * End to end: discovering rules for targets pass" and "hello found nothing before and now finds $", ^" and i0", each confirmed by the audit trail.
 *
 * Revision 1.8  2026/09/05 01:40:43  dlr
 * Make .N deterministic at the last position. It copies the NEXT codepoint over position N, and at the last position there is no next one. ruleproc.c reads its NUL terminator there and writes a NUL, which is an artefact of a NUL-terminated buffer rather than a decision. Mirroring it read one past the live length of a buffer that is static and REUSED, so the answer depended on the candidate processed before: the same word gave 1234567Z after one word and 1234567Q after another. Output that depends on processing order is not usable, and this is the buffer-reuse class that gets fixed here rather than replayed. The character is now left alone at the last position. Found by the documentation pass while checking corner cases, not by the gates -- none of them varied the preceding candidate.
 *
 * Revision 1.7  2026/09/05 01:14:39  dlr
 * Make f cluster-aware, which r already was. Reflect appended a codepoint-reversed copy, so it put an Arabic fatha on the wrong letter, detached a Hebrew word from its points, began the copy with a bare skin-tone modifier, and turned the US flag into SU -- a DIFFERENT flag rather than a mangled one. Same two-pass shape as the reversal. Checked z, Z, y, Y and q at the same time on multi-codepoint clusters: those were already correct and are now recorded as such. Control-B, the base64 rule, stays unimplemented here and a rule using it is refused: it is absent from rule_ops.h so it was never part of the verb contract, it appears zero times in 111,000 real rules, and it is undocumented.
 *
 * Revision 1.6  2026/09/05 01:10:04  dlr
 * Finish the quoted-operand sweep: v, 3 and e now take a string too. vNS inserts a string after every N clusters, 3NS toggles after the Nth occurrence of a string separator, and eS title-cases on one. A one-character separator and a string separator are genuinely different rules, which the corpus now records: 31 with a hyphen and 31 with a double hyphen pick different letters in the same word. Every verb in the engine that takes a literal operand now accepts a quoted string; the only raw literal left is the chained prepend, where each operand is one character by construction.
 *
 * Revision 1.5  2026/09/05 00:48:01  dlr
 * Quoted operands: an operand may be a STRING, not a single codepoint. Written as $"123" or s"abc" "XY", which makes two things possible that were not. An operand can now be something with no single-codepoint spelling at all -- a Devanagari conjunct, a Yoruba vowel carrying two marks, a joiner sequence -- where before such a rule could not be written and was refused. And an operand can be longer than one character, so a suffix is one op rather than a chain, and substitution and purge work on strings of differing length. A doubled quote inside is one literal. Between two quoted operands a separator is allowed and skipped; it is NOT skipped after an unquoted one, because a space is a perfectly good operand and four rules in the first thousand of a real file substitute something for one. Divergence from the byte engine is confined to rules using a literal double quote: 7 in 10,000, of which 5 are refused loudly as unterminated rather than reparsed. That is deliberate and in keeping with a mode that already differs on shifts, on what a position counts, and on uppercasing an eszett.
 *
 * Revision 1.4  2026/09/05 00:08:03  dlr
 * Treat Khmer coeng as a conjunct linker. A frequency-ordered audit of the 43 most-spoken languages, 26 scripts, found reversing Khmer left a trailing coeng: U+17D2 is defined as combining with the consonant that FOLLOWS to form a subscript, so it cannot end a word. Tamil was flagged by the same check and is correct as it stands: its pulli is always visible and a consonant carrying one is a normal standalone letter, which is why Unicode excludes Tamil from Indic_Conjunct_Break. The audit is otherwise clean across all 43: no stranded marks, no divergence between the two normal forms, no disagreement with the independent model.
 *
 * Revision 1.3  2026/09/05 00:03:01  dlr
 * Cluster layer: a rule now moves and removes whole grapheme clusters, not codepoints. Every position and length operand counts clusters, so a rule that says delete character three means the third thing a reader sees. Combining marks travel with their base, which fixes Arabic harakat, Hebrew niqqud, Thai vowel signs and NFD Latin: reversing meem-fatha-reh by codepoint put the fatha on the reh, and dropping the leading codepoint left a mark with no base, which is malformed text rather than a shorter word. Emoji join too: a joiner sequence, a regional-indicator pair and a skin-tone modifier are each one cluster. Korean conjoining jamo form one syllable per UAX 29 GB6 through GB8, which matters because macOS stores filenames decomposed. Devanagari consonant plus virama plus consonant is one conjunct per GB9c, so a drop no longer leaves a dangling halant, and the non-joiner extends rather than starting a word of its own. Halfwidth katakana sound marks are filed as modifier LETTERS but are Extend in UAX 29, so they needed naming. Reversal is two in-place passes and no scratch buffer. Validated against an independent model: 116 rules over 208 words across ten scripts, zero disagreements, and the ASCII gate is unchanged.
 *
 * Revision 1.2  2026/09/04 23:04:01  dlr
 * Complete the verb set at all 61, and widen the case layer well past Latin. The last seven verbs are S, hash-exit, hex upper and lower, div-insert, toggle-at-separator and the two whitespace no-ops, each derived from ruleproc.c rather than inferred. Hex is the one verb where both engines agree for ALL input, not just ASCII: the byte engine hexes raw bytes and the input is UTF-8, so hexing the UTF-8 serialisation reproduces it exactly. Fixed the capitalise verbs, which had been written to hashcat semantics by inference: ruleproc.c capitalises the first ALPHABETIC codepoint, skipping leading digits, so 4aD4aD4a gives 4Ad4ad4a and not 4ad4ad4a. Collapsed runs of append and prepend into single ops, mirroring the byte engine: uncollapsed prepends are quadratic because each shifts the whole buffer, and 24 of them ran 9.1 times the byte engine before and 3.1 after. Case coverage goes from 3 script blocks to 23, dispatched by binary search, plus a 102-entry table for mappings whose uppercase GROWS: the twelve ligatures, the eszett and 84 Greek Extended forms a 1:1 table cannot hold. Greek carries a positional final sigma. All 3232 table entries and all 102 expansions verified against an independent source.
 *
 * Revision 1.1  2026/09/04 20:54:53  dlr
 * Initial revision
 *
 *
 * ruleproc32.c -- UTF-32 rule engine for procrule. Conversion layer.
 *
 * See ruleproc32.h for why this is a separate engine rather than a widening of
 * applyrule().
 */

#include <stdint.h>
#include <string.h>
#include "ruleproc32.h"
#include "latin_case.h"
#include "combining.h"

/* Minimum scalar legally representable at each sequence length. A form that
 * encodes a scalar below its length's minimum is "overlong": 0xC0 0x80 is
 * NUL written in two bytes. Overlongs matter beyond tidiness -- they were a
 * real path for smuggling '/' and NUL past byte-oriented filters, which is why
 * the standard makes them ill-formed rather than merely redundant. */
static const uint32_t utf8_min_for_len[5] = { 0, 0, 0x80u, 0x800u, 0x10000u };

int utf8_to_utf32(const unsigned char *in, int inlen, uint32_t *out, int outmax)
{
    int i = 0, n = 0;

    if (!in || !out || outmax <= 0) return RULE32_ERR_NOROOM;

    while (i < inlen) {
        unsigned char c = in[i];
        int len;
        uint32_t cp;

        if (c < 0x80u) {            /* ASCII: the overwhelmingly common case */
            if (n >= outmax) return RULE32_ERR_NOROOM;
            out[n++] = c;
            i++;
            continue;
        }

        /* Lead-byte classification. 0x80-0xBF is a continuation with no lead,
         * 0xC0/0xC1 could only ever begin an overlong, and 0xF5-0xFF would
         * encode above U+10FFFF. None can start a sequence. */
        if      (c >= 0xC2u && c <= 0xDFu) { len = 2; cp = c & 0x1Fu; }
        else if (c >= 0xE0u && c <= 0xEFu) { len = 3; cp = c & 0x0Fu; }
        else if (c >= 0xF0u && c <= 0xF4u) { len = 4; cp = c & 0x07u; }
        else return RULE32_ERR_INVALID;

        if (i + len > inlen) return RULE32_ERR_INVALID;   /* truncated */

        for (int k = 1; k < len; k++) {
            unsigned char cc = in[i + k];
            if ((cc & 0xC0u) != 0x80u) return RULE32_ERR_INVALID;  /* not a continuation */
            cp = (cp << 6) | (uint32_t)(cc & 0x3Fu);
        }

        if (cp < utf8_min_for_len[len]) return RULE32_ERR_INVALID;  /* overlong */
        if (!UTF32_ENCODABLE(cp)) return RULE32_ERR_INVALID; /* surrogate or >10FFFF */

        if (n >= outmax) return RULE32_ERR_NOROOM;
        out[n++] = cp;
        i += len;
    }
    return n;
}

int utf32_to_utf8(const uint32_t *in, int inlen, unsigned char *out, int outmax,
                  int *ndropped)
{
    int i, n = 0, dropped = 0;

    if (!in || !out || outmax < 0) return RULE32_ERR_NOROOM;

    for (i = 0; i < inlen; i++) {
        uint32_t cp = in[i];

        /* Arithmetic verbs (BIT_SHL, BIT_SHR, INC, DEC, SUB) can carry a
         * codepoint out of range. Drop it and keep going: the operator's rule
         * is to emit what is possible rather than lose the whole line. */
        if (!UTF32_ENCODABLE(cp)) { dropped++; continue; }

        if (cp < 0x80u) {
            if (n + 1 > outmax) return RULE32_ERR_NOROOM;
            out[n++] = (unsigned char)cp;
        } else if (cp < 0x800u) {
            if (n + 2 > outmax) return RULE32_ERR_NOROOM;
            out[n++] = (unsigned char)(0xC0u | (cp >> 6));
            out[n++] = (unsigned char)(0x80u | (cp & 0x3Fu));
        } else if (cp < 0x10000u) {
            if (n + 3 > outmax) return RULE32_ERR_NOROOM;
            out[n++] = (unsigned char)(0xE0u | (cp >> 12));
            out[n++] = (unsigned char)(0x80u | ((cp >> 6) & 0x3Fu));
            out[n++] = (unsigned char)(0x80u | (cp & 0x3Fu));
        } else {
            if (n + 4 > outmax) return RULE32_ERR_NOROOM;
            out[n++] = (unsigned char)(0xF0u | (cp >> 18));
            out[n++] = (unsigned char)(0x80u | ((cp >> 12) & 0x3Fu));
            out[n++] = (unsigned char)(0x80u | ((cp >> 6) & 0x3Fu));
            out[n++] = (unsigned char)(0x80u | (cp & 0x3Fu));
        }
    }
    if (ndropped) *ndropped = dropped;
    return n;
}

/* ------------------------------------------------------------------ *
 * Rule compilation and application, codepoint-native.
 * ------------------------------------------------------------------ */

#include "rule_ops.h"

/* Position operands in hashcat/JtR rules are a single base-36-ish character:
 * 0-9 then A-Z. Returns -1 if not a position character. */
static int pos_of(uint32_t c)
{
    if (c >= '0' && c <= '9') return (int)(c - '0');
    if (c >= 'A' && c <= 'Z') return (int)(c - 'A') + 10;
    /* a-z are positions 36-61.  ruleproc.c's Rulepos is
     * "0123456789A-Za-z", 62 positions, and this engine stopped at 35 -- so it
     * refused EVERY rule with a lowercase position operand, across every
     * positional verb, not just the `3NX` case that surfaced it.
     * NOTE this is an mdxfind extension beyond hashcat, which rejects a
     * lowercase position (`Ta` -> REJECT in rp_cpu.c).  Matching ruleproc.c is
     * the goal here; whether ruleproc.c should itself narrow to hashcat's 36 is
     * a separate question about the oracle. */
    if (c >= 'a' && c <= 'z') return (int)(c - 'a') + 36;
    return -1;
}

/* ASCII-only case mapping, deliberately, for this tranche. Unicode case is a
 * separate step: it needs a real 1:1 table for Latin-1 and Latin Extended-A,
 * and the length-changing cases (eszett -> SS) must be excluded because they
 * break every position-indexed op that follows them in the same rule. */
/* Locale-ambiguous pairs. Only the Turkish/Azeri dotted-i family qualifies
 * today: every other mapping in Latin-1 and Latin Extended-A is 1:1 across
 * locales. `amb` is set when one is reached, which is what tells the caller a
 * second variant exists. */
#define CP_LATIN_SMALL_I        0x0069u   /* i */
#define CP_LATIN_CAPITAL_I      0x0049u   /* I */
#define CP_LATIN_CAPITAL_I_DOT  0x0130u   /* I with dot above */
#define CP_LATIN_SMALL_DOTLESS  0x0131u   /* dotless i */
#define CP_GREEK_SMALL_SIGMA   0x03C3u   /* medial sigma */
#define CP_GREEK_SMALL_FINAL    0x03C2u   /* word-final sigma */
#define CP_LATIN_SHARP_S        0x00DFu   /* eszett */

/* Locate a codepoint in the generated case tables. The table covers 23 script
 * blocks scattered from U+00A0 to U+1E943, so a chain of range comparisons
 * would cost up to 23 branches on every character of every candidate. The
 * ranges are emitted sorted, so this is a binary search: five comparisons at
 * the current size, and adding a script costs log time, not linear.
 * Returns a pointer into the requested table, or NULL if there is no mapping. */
/* True if c is a combining mark, so it belongs to the base before it rather
 * than standing on its own. */
static int cp_is_mark(uint32_t c)
{
    int lo = 0, hi = CP_MARK_N - 1;

    if (c < CP_MARK_LO || c > CP_MARK_HI) return 0;   /* all ASCII, in one test */
    while (lo <= hi) {
        int mid = (lo + hi) >> 1;
        if (c < cp_marks[mid].lo)      hi = mid - 1;
        else if (c > cp_marks[mid].hi) lo = mid + 1;
        else return 1;
    }
    return 0;
}

/* Codepoints that continue a cluster but are not combining marks.
 *
 * The Mn/Mc/Me test covers accents, harakat and niqqud, and it already catches
 * variation selectors and the enclosing keycap, so a keycap emoji clustered
 * correctly by accident. It does NOT catch the three that make emoji work:
 * a skin-tone modifier is Sk, a zero-width joiner is Cf, and a regional
 * indicator is plain So. Each needs naming. */
#define CP_ZWJ          0x200Du
#define CP_EMOJI_MOD_LO 0x1F3FBu   /* Fitzpatrick type-1 */
#define CP_EMOJI_MOD_HI 0x1F3FFu   /* Fitzpatrick type-6 */
#define CP_RI_LO        0x1F1E6u   /* regional indicator A */
#define CP_RI_HI        0x1F1FFu   /* regional indicator Z */
#define CP_TAG_LO       0xE0020u   /* tag characters, for subdivision flags */
#define CP_TAG_HI       0xE007Fu
#define CP_ZWNJ         0x200Cu   /* zero-width NON-joiner */
#define CP_HW_VOICED     0xFF9Eu  /* halfwidth katakana dakuten */
#define CP_HW_SEMIVOICED 0xFF9Fu  /* halfwidth katakana handakuten */

static int cp_is_ri(uint32_t c) { return c >= CP_RI_LO && c <= CP_RI_HI; }

/* Hangul. A Korean syllable can be written as ONE precomposed codepoint or as
 * its conjoining jamo -- leading consonant, vowel, optional trailing consonant
 * -- stored separately. Both spell the same word, and macOS stores filenames
 * decomposed, so harvested Korean arrives either way.
 *
 * The jamo are category Lo, not marks, so nothing above catches them: a
 * decomposed syllable came apart into two or three pieces, and a rule that
 * dropped one left a bare vowel or a consonant with nothing to attach to.
 * These are the UAX #29 rules GB6 to GB8. */
#define HG_NONE 0
#define HG_L    1   /* choseong, leading consonant */
#define HG_V    2   /* jungseong, vowel */
#define HG_T    3   /* jongseong, trailing consonant */
#define HG_LV   4   /* precomposed syllable with no trailing consonant */
#define HG_LVT  5   /* precomposed syllable with one */

static int hangul_class(uint32_t c)
{
    if (c >= 0x1100u && c <= 0x115Fu) return HG_L;
    if (c >= 0x1160u && c <= 0x11A7u) return HG_V;
    if (c >= 0x11A8u && c <= 0x11FFu) return HG_T;
    if (c >= 0xA960u && c <= 0xA97Cu) return HG_L;      /* Extended-A */
    if (c >= 0xD7B0u && c <= 0xD7C6u) return HG_V;      /* Extended-B */
    if (c >= 0xD7CBu && c <= 0xD7FBu) return HG_T;
    if (c >= 0xAC00u && c <= 0xD7A3u)
        return ((c - 0xAC00u) % 28u) ? HG_LVT : HG_LV;
    return HG_NONE;
}

/* May b continue the syllable that a is part of? */
static int hangul_join(int a, int b)
{
    switch (a) {
        case HG_L:   return b == HG_L || b == HG_V || b == HG_LV || b == HG_LVT;
        case HG_LV:
        case HG_V:   return b == HG_V || b == HG_T;
        case HG_LVT:
        case HG_T:   return b == HG_T;
        default:     return 0;
    }
}

/* Continues the cluster it follows, without joining anything after it. */
static int cp_is_extend(uint32_t c)
{
    if (c >= CP_EMOJI_MOD_LO && c <= CP_EMOJI_MOD_HI) return 1;
    if (c >= CP_TAG_LO && c <= CP_TAG_HI) return 1;
    /* The NON-joiner extends the cluster it follows, exactly as the joiner
     * does. In Devanagari it is what SUPPRESSES a conjunct, so it is written
     * deliberately between two consonants; leaving it to start a cluster of
     * its own produced a word beginning with a bare non-joiner. */
    if (c == CP_ZWNJ) return 1;
    /* Halfwidth katakana voiced and semi-voiced sound marks. These are the
     * only characters in UAX #29's Extend set that are neither marks nor
     * anything else named above -- Unicode files them as modifier LETTERS, so
     * the Mn/Mc/Me test walks straight past them. Halfwidth katakana is still
     * everywhere in Japanese banking and legacy systems, and without this a
     * voiced kana came apart into a bare letter and a floating mark. */
    if (c == CP_HW_VOICED || c == CP_HW_SEMIVOICED) return 1;
    return cp_is_mark(c);
}

/* ---- Indic conjuncts (UAX #29 GB9c) ----
 *
 * A consonant, a virama, and another consonant render as ONE conjunct
 * ligature: Devanagari ksha is ka plus virama plus ssa, and a reader sees one
 * character. Splitting on codepoints left a DANGLING VIRAMA -- dropping the
 * last codepoint of hindi gave a word ending in a halant, which is malformed
 * text, not a shorter word. That is the same failure the Arabic marks had.
 *
 * The linker set is the virama of each script for which Unicode defines the
 * Indic_Conjunct_Break property. The consonant test is approximated as "a
 * letter of the same script", which over-joins only where a virama is followed
 * by an independent vowel -- rare, and far less wrong than a dangling halant.
 */
static int cp_indic_script(uint32_t c)
{
    if (c >= 0x0900u && c <= 0x097Fu) return 1;   /* Devanagari */
    if (c >= 0x0980u && c <= 0x09FFu) return 2;   /* Bengali */
    if (c >= 0x0A80u && c <= 0x0AFFu) return 3;   /* Gujarati */
    if (c >= 0x0B00u && c <= 0x0B7Fu) return 4;   /* Oriya */
    if (c >= 0x0C00u && c <= 0x0C7Fu) return 5;   /* Telugu */
    if (c >= 0x0D00u && c <= 0x0D7Fu) return 6;   /* Malayalam */
    if (c >= 0x1780u && c <= 0x17FFu) return 7;   /* Khmer */
    return 0;
}

static int cp_is_linker(uint32_t c)
{
    /* The six viramas for which Unicode defines Indic_Conjunct_Break, plus
     * Khmer coeng. Coeng is not part of that property, but it is a stronger
     * case than any of them: it is DEFINED as combining with the consonant
     * that follows to form a subscript, so it cannot end a word. Reversing
     * Khmer left one trailing, which is malformed text.
     *
     * Tamil, Kannada, Gurmukhi, Sinhala, Myanmar and Tibetan viramas are
     * deliberately NOT here. Unicode leaves them out, and for Tamil that is
     * plainly right: its pulli is always visible and a consonant carrying one
     * is a normal standalone letter, so joining would merge units a reader
     * sees as separate. The others are a judgement call left at the standard's
     * default rather than decided here. */
    return c == 0x094Du || c == 0x09CDu || c == 0x0ACDu ||
           c == 0x0B4Du || c == 0x0C4Du || c == 0x0D4Du ||
           c == 0x17D2u;
}

/* ---- cluster indexing ----
 *
 * Every position and length operand in this engine counts GRAPHEME CLUSTERS,
 * not codepoints. A rule that says "delete character 3" means the third thing
 * a reader sees, and on vocalised Arabic or pointed Hebrew the third codepoint
 * is routinely a vowel mark hanging off the second letter. Indexing by
 * codepoint cut clusters in half and stranded marks with no base.
 *
 * The translation happens at the top of each verb, so the codepoint logic
 * underneath -- including the clamping quirks derived from ruleproc.c -- is
 * untouched. On input with no combining marks, cl_start is the identity and
 * cl_count returns the length, so ASCII behaviour is bit-for-bit what it was.
 */

/* Codepoint index where cluster k begins. Returns n when k is at or past the
 * end, so it doubles as an exclusive bound. */
/* Advance from i past exactly ONE cluster. Every other cluster routine is
 * written in terms of this, so the definition lives in one place.
 *
 * Two shapes beyond base-plus-marks. A pair of regional indicators is one
 * cluster, which is how a flag is encoded. And a zero-width joiner pulls in
 * whatever follows it, which is what makes a profession emoji or a family a
 * single unit rather than two or three people who come apart under a reverse.
 * The joiner rule is applied unconditionally rather than only between
 * pictographs: a joiner anywhere else is vanishingly rare, and joining is the
 * answer that cannot produce a string starting with a dangling joiner. */
static inline int cl_next(const uint32_t *b, int n, int i)
{
    if (i >= n) return n;

    /* Fast path, and it carries almost all real traffic. Nothing below the
     * LOWEST mark at U+0300 can continue a cluster -- the joiner is U+200D,
     * skin tones and tags are astral -- and nothing below U+1F1E6 is a regional
     * indicator. So a base under that with a successor under the joiner is a
     * one-codepoint cluster, settled in two comparisons. Every ASCII candidate
     * takes this path; without it, position-indexed rules ran 63% slower than
     * the mark-only walk this replaced. */
    if (b[i] < CP_RI_LO && (i + 1 >= n || b[i + 1] < CP_MARK_LO)) return i + 1;

    if (cp_is_ri(b[i])) {
        i++;
        if (i < n && cp_is_ri(b[i])) i++;       /* the pair is one flag */
    } else {
        int h = hangul_class(b[i]);
        i++;                                    /* the base */
        if (h) {                                /* and the rest of the syllable */
            while (i < n) {
                int h2 = hangul_class(b[i]);
                if (!h2 || !hangul_join(h, h2)) break;
                h = h2; i++;
            }
        }
    }
    for (;;) {
        int linker = 0;
        while (i < n && cp_is_extend(b[i])) {
            if (cp_is_linker(b[i])) linker = 1;
            i++;
        }
        /* GB9c: a run of extends containing a virama joins the consonant that
         * follows, and a three-consonant conjunct just loops again. */
        if (linker && i < n && cp_indic_script(b[i])) { i++; continue; }
        if (i < n && b[i] == CP_ZWJ) {
            i++;                                /* the joiner */
            if (i < n) i++;                     /* and what it joins */
            continue;
        }
        break;
    }
    return i;
}

/* True when nothing in the buffer can join or extend, so cluster index and
 * codepoint index are the same. One comparison per codepoint answers it, and
 * it lets the two routines below skip the per-cluster walk entirely -- which
 * is what every ASCII candidate does.
 *
 * DO NOT "TIGHTEN" THIS TEST -- see the long note at u32_cl_all_plain() in
 * gpu/gpu_u32_walker.inc. Asking the precise question cl_next() asks was
 * byte-exact on 164.5M pairs and 1.5-1.6x SLOWER on the GPU, in BOTH the
 * cluster-sensitive and the cluster-free half. The CPU side is kept in step
 * with the device deliberately: the two must stage identically or the host
 * twin stops being an oracle. */
static inline int cl_all_plain(const uint32_t *b, int n)
{
    int i;
    for (i = 0; i < n; i++) if (b[i] >= CP_MARK_LO) return 0;
    return 1;
}

static int cl_start(const uint32_t *b, int n, int k)
{
    int i = 0;
    if (cl_all_plain(b, n)) return k < n ? k : n;
    while (i < n && k > 0) { i = cl_next(b, n, i); k--; }
    return i;
}

/* How many clusters the buffer holds. */
/* Reverse the half-open span [i,j). Used to swap two clusters that are not
 * the same length: reversing each span and then the whole run turns A mid B
 * into B mid A without a scratch buffer. */
static void rev_span(uint32_t *b, int i, int j)
{
    for (j--; i < j; i++, j--) { uint32_t t = b[i]; b[i] = b[j]; b[j] = t; }
}

/* Swap the clusters starting at cluster indices k1 and k2 (k1 < k2). */
static void cl_swap(uint32_t *b, int n, int k1, int k2)
{
    int a0 = cl_start(b, n, k1), a1 = cl_start(b, n, k1 + 1);
    int b0 = cl_start(b, n, k2), b1 = cl_start(b, n, k2 + 1);
    if (a1 > b0) return;
    rev_span(b, a0, a1); rev_span(b, a1, b0); rev_span(b, b0, b1);
    rev_span(b, a0, b1);
}

static int cl_count(const uint32_t *b, int n)
{
    int i = 0, c = 0;
    if (cl_all_plain(b, n)) return n;
    while (i < n) { i = cl_next(b, n, i); c++; }
    return c;
}

static const uint32_t *cp_case_lookup(uint32_t c, const uint32_t *data)
{
    int lo = 0, hi = CP_CASE_NRANGES - 1;

    /* Reject anything outside the covered span before searching. Without this
     * every ASCII digit and punctuation character -- which the caller's
     * letter-only fast path does not catch -- paid a full binary search, and
     * case-heavy rules over an alphanumeric wordlist ran 19% slower than the
     * three-range version this replaced. One comparison buys it all back. */
    if (c < cp_case_ranges[0].lo || c > cp_case_ranges[CP_CASE_NRANGES - 1].hi)
        return NULL;

    while (lo <= hi) {
        int mid = (lo + hi) >> 1;
        if (c < cp_case_ranges[mid].lo)      hi = mid - 1;
        else if (c > cp_case_ranges[mid].hi) lo = mid + 1;
        else return &data[cp_case_ranges[mid].off + (c - cp_case_ranges[mid].lo)];
    }
    return NULL;
}

static uint32_t cp_upper(uint32_t c, int turkish, int *amb)
{
    if (c == CP_LATIN_SMALL_I) { if (amb) *amb = 1;
        return turkish ? CP_LATIN_CAPITAL_I_DOT : CP_LATIN_CAPITAL_I; }
    if (c == CP_LATIN_SMALL_DOTLESS) return CP_LATIN_CAPITAL_I;   /* unambiguous */
    if (c >= 'a' && c <= 'z') return c - 32;
    /* Every cased script block the generator covers, via one binary search.
     * A codepoint the table leaves unmapped is either caseless or excluded as
     * length-changing (eszett, U+0149), and is correctly left alone. */
    { const uint32_t *p = cp_case_lookup(c, cp_case_up); if (p) return *p; }
    return c;
}

static uint32_t cp_lower(uint32_t c, int turkish, int *amb)
{
    if (c == CP_LATIN_CAPITAL_I) { if (amb) *amb = 1;
        return turkish ? CP_LATIN_SMALL_DOTLESS : CP_LATIN_SMALL_I; }
    /* U+0130 lowercases to i + COMBINING DOT ABOVE in full Unicode, which is
     * length-changing and therefore excluded from the table. The simple 1:1
     * answer is plain i, which is what the policy calls for. */
    if (c == CP_LATIN_CAPITAL_I_DOT) return CP_LATIN_SMALL_I;
    if (c >= 'A' && c <= 'Z') return c + 32;
    { const uint32_t *p = cp_case_lookup(c, cp_case_lo); if (p) return *p; }
    return c;
}

/* Read one operand starting at line[i], which is the codepoint AFTER the verb.
 * Writes the codepoints to buf and returns how many, or -1 if malformed.
 * *used receives how many codepoints of the rule line were consumed.
 *
 * Unquoted, that is exactly one codepoint and one consumed -- which is why
 * every existing rule packs identically. Quoted, it is the text between the
 * delimiters, with a doubled quote standing for one literal. */
/* Skip separators between two operands of the same verb. Needed because a
 * doubled quote already means one literal quote, so `s"ab""cd"` reads as the
 * single operand ab"cd rather than as two -- write `s"ab" "cd"` and the
 * reading is unambiguous. Without a separator the second operand is missing
 * and the rule is refused, which is loud rather than silently wrong. */
static int skip_seps(const uint32_t *line, int linelen, int i)
{
    while (i < linelen && (line[i] == ' ' || line[i] == '\t')) i++;
    return i;
}

/* Separator set for the `$`/`^` run gatherer, which is NOT the same set as
 * skip_seps().  ruleproc.c's gatherer swallows space, tab AND colon between
 * run elements (ruleproc.c:574), because in the byte engine those three are
 * pure separators that emit nothing at all (ruleproc.c:819-822).  This engine
 * emits RULE_OP_NOOP for a standalone `:`, which is why the colon cannot join
 * skip_seps() -- there it would change what the other caller parses.  Dropping
 * a NOOP inside a run is safe by definition. */
static int skip_run_seps(const uint32_t *line, int linelen, int i)
{
    while (i < linelen &&
           (line[i] == ' ' || line[i] == '\t' || line[i] == ':')) i++;
    return i;
}

/* Set for the second parsing pass, where a quote is an ordinary character
 * rather than the operand delimiter.  See packrule32(). */
static __thread int Rule32LiteralQuotes = 0;

/* One hex digit, or -1.  Used only by the `$`/`^` escape path. */
static int hexval32(uint32_t c)
{
    if (c >= '0' && c <= '9') return (int)(c - '0');
    if (c >= 'a' && c <= 'f') return (int)(c - 'a') + 10;
    if (c >= 'A' && c <= 'F') return (int)(c - 'A') + 10;
    return -1;
}

static int read_operand(const uint32_t *line, int linelen, int i,
                        uint32_t *buf, int bufmax, int *used)
{
    int n = 0, j;

    if (i >= linelen) return -1;
    if (line[i] != RULE32_QUOTE || Rule32LiteralQuotes) {
        if (bufmax < 1) return -1;
        buf[0] = line[i]; *used = 1; return 1;
    }
    for (j = i + 1; j < linelen; j++) {
        if (line[j] == RULE32_QUOTE) {
            if (j + 1 < linelen && line[j + 1] == RULE32_QUOTE) {
                if (n >= bufmax) return -1;
                buf[n++] = RULE32_QUOTE; j++; continue;   /* doubled: literal */
            }
            *used = (j - i) + 1;                          /* through the close */
            return n;
        }
        if (n >= bufmax) return -1;
        buf[n++] = line[j];
    }
    return -1;                                            /* never closed */
}

/* A class-capable operand slot.
 *
 * ruleproc.c reads `?C` here as a character CLASS and `??` as a literal `?`.
 * This engine implements neither, and the one outcome that must not happen is
 * SILENTLY MEANING SOMETHING ELSE.  Before this guard, on "Pass1 x":
 *
 *     @?d    byte engine "Pass x"   -u "Pass1 xPass1 x"   (purge ?, then d)
 *     @?l    byte engine "P1"       -u "pass1 x"
 *     e?d    byte engine identity   -u "Pass1 xPass1 x"
 *     =N?C %N?C /?C (?C )?C        -u silently rejected everything
 *
 * Plausible output on both sides, no diagnostic, from the same rule text --
 * the shape that reads as a clean run and is not one.
 *
 * So: `??` yields the same literal `?` ruleproc.c yields, and a class is
 * REFUSED, which procrule reports as "Invalid rule line. Ignored." naming the
 * rule.  Class support under -u is a later pass; until it lands, a rule using
 * one fails loudly instead of meaning something different in each engine.
 */
static int read_class_operand(const uint32_t *line, int linelen, int i,
                              uint32_t *buf, int bufmax, int *used,
                              unsigned char *clsp, int table)
{
    if (clsp) *clsp = 0;
    /* NOT gated on Rule32LiteralQuotes.  `?` is not a quote, and packrule32
     * retries the whole pass with that flag set after any RULE32_ERR_INVALID,
     * so gating here let the retry sail straight past the guard and compile
     * the old reinterpretation -- which is exactly the silent divergence this
     * function exists to prevent. */
    if (table) {
        /* hashcat's ~-prefixed form: the `?` is mandatory and `??` is its
         * literal-`?` escape, exactly as ruleproc.c reads it. */
        if (i >= linelen || line[i] != (uint32_t)'?' || i + 1 >= linelen)
            return -1;
        if (line[i + 1] == (uint32_t)'?') {
            if (bufmax < 1) return -1;
            buf[0] = (uint32_t)'?'; *used = 2; return 1;
        }
        {   unsigned char cb = rule_class_byte((char)line[i + 1], 1);
            if (!cb) return -1;
            if (clsp) *clsp = cb;
            *used = 2; return 1; }
    }
    if (i < linelen && line[i] == (uint32_t)'?') {
        if (i + 1 < linelen && line[i + 1] == (uint32_t)'?') {
            if (bufmax < 1) return -1;
            buf[0] = (uint32_t)'?';
            *used = 2;
            return 1;                      /* the escape: a literal '?' */
        }
        /* An inline John class.  Previously refused outright so that a class
         * could not mean two different things in the two engines; now it means
         * the same thing, because both consult rule_class_byte() and the one
         * shared membership table in rule_ops.h. */
        if (i + 1 >= linelen) return -1;
        {   unsigned char cb = rule_class_byte((char)line[i + 1], 0);
            if (!cb) return -1;
            if (clsp) *clsp = cb;
            *used = 2; return 1; }
    }
    return read_operand(line, linelen, i, buf, bufmax, used);
}

static int packrule32_pass(const uint32_t *line, int linelen,
                           uint32_t *out, int outmax)
{
    int i = 0, n = 0;

#define EMIT(v)   do { if (n >= outmax) return RULE32_ERR_NOROOM; out[n++] = (uint32_t)(v); } while (0)
#define NEED(k)   do { if (i + (k) >= linelen) return RULE32_ERR_INVALID; } while (0)
#define POSARG    do { NEED(1); { int _p = pos_of(line[i+1]); \
                      if (_p < 0) return RULE32_ERR_INVALID; EMIT(_p); i += 2; } } while (0)

    while (i < linelen) {
        uint32_t c = line[i];

        if (c == '#') break;                       /* comment to end of line */
        if (c == ' ' || c == '\t') { i++; continue; }

        switch (c) {
            /* `:` emits NOTHING, matching ruleproc.c:819-822, and the reason
             * is an mdxfind invariant rather than a style choice: mdxfind
             * ALWAYS tries the unmodified plain before any rule is applied.
             * hashcat does not, so hashcat-derived rulesets conventionally
             * carry `:` as their first rule -- and under mdxfind that rule is
             * a duplicate of the implicit no-rule pass, which the loader
             * discards at load time as "no effect" (a rule that packs to
             * nothing has mystrlen 0 and is never inserted into RuleArray,
             * mdxfind.c:52260).
             *
             * Emitting RULE_OP_NOOP defeated that: the rule packed to a
             * non-empty program, survived the loader, and produced a
             * candidate identical to the plain -- so the no-rule pass would
             * be hashed twice and the duplicate credited to a rule in the -Z
             * histogram. Verified with a direct applyrule/applyrule32 harness
             * (procrule cannot show it: it suppresses a candidate equal to
             * the word it came from, so both engines look silent).
             *
             * The ' ' and '\t' cases below are unreachable -- the loop head
             * skips both before the switch -- and are left alone rather than
             * removed, since the byte engine also carries their opcodes for
             * consumption. */
            case ':':                           i++; break;
            case 'l': EMIT(RULE_OP_LOWER);      i++; break;
            case 'u': EMIT(RULE_OP_UPPER);      i++; break;
            case 'c': EMIT(RULE_OP_CAP);        i++; break;
            case 'C': EMIT(RULE_OP_CAP_INV);    i++; break;
            case 't': EMIT(RULE_OP_TOGGLE);     i++; break;
            case 'r': EMIT(RULE_OP_REVERSE);    i++; break;
            case 'd': EMIT(RULE_OP_DUP);        i++; break;
            case 'f': EMIT(RULE_OP_REFLECT);    i++; break;
            case '{': EMIT(RULE_OP_ROT_L);      i++; break;
            case '}': EMIT(RULE_OP_ROT_R);      i++; break;
            case '[': EMIT(RULE_OP_DROP_FIRST); i++; break;
            case ']': EMIT(RULE_OP_DROP_LAST);  i++; break;
            case 'k': EMIT(RULE_OP_SWAP_FRONT); i++; break;
            case 'K': EMIT(RULE_OP_SWAP_BACK);  i++; break;
            case 'S': EMIT(RULE_OP_S_SPECIAL); i++; break;
            case 'H': EMIT(RULE_OP_HEX_UPPER); i++; break;
            case 'h': EMIT(RULE_OP_HEX_LOWER); i++; break;
            case '#': EMIT(RULE_OP_HASH_EXIT); i++; break;
            case ' ':  EMIT(RULE_OP_NOOP_SP);  i++; break;
            case '\t': EMIT(RULE_OP_NOOP_TAB); i++; break;

            /* vNX -- insert X after every N codepoints. ruleproc.c stores
             * positiontranslate(N)-1, and positiontranslate is 1-BASED
             * (it returns index+1), so the stored value is the plain 0-based
             * index -- which is the count itself: `v2` inserts every 2. */
            case 'v': {
                uint32_t ob[512]; int used = 0, on, k, p;
                NEED(2);
                p = pos_of(line[i+1]);
                if (p < 0) return RULE32_ERR_INVALID;
                on = read_operand(line, linelen, i+2, ob, 512, &used);
                if (on <= 0) return RULE32_ERR_INVALID;
                if (on == 1) { EMIT(RULE_OP_DIV_INSERT); EMIT(p); EMIT(ob[0]); }
                else { EMIT(RULE32_OP_DIV_INS_STR); EMIT(p); EMIT(on);
                       for (k = 0; k < on; k++) EMIT(ob[k]); }
                i += 2 + used; break;
            }

            /* 3NX -- toggle the first cased codepoint after the Nth X.
             * ruleproc.c is inconsistent here: it stores positiontranslate(N)
             * WITHOUT the -1 every other position verb applies, and subtracts
             * one at execution instead. Net effect is the 0-based index, so
             * this format stores that directly and the executor does not
             * subtract. The packed form is ours, not ruleproc.c's wire shape. */
            case '3': {
                uint32_t ob[512]; int used = 0, on, k, p;
                NEED(2);
                p = pos_of(line[i+1]);
                if (p < 0) return RULE32_ERR_INVALID;
                on = read_operand(line, linelen, i+2, ob, 512, &used);
                if (on <= 0) return RULE32_ERR_INVALID;
                if (on == 1) { EMIT(RULE_OP_TOGGLE_SEP); EMIT(p); EMIT(ob[0]); }
                else { EMIT(RULE32_OP_TOGGLE_SEP_STR); EMIT(p); EMIT(on);
                       for (k = 0; k < on; k++) EMIT(ob[k]); }
                i += 2 + used; break;
            }

            /* Operand is a CODEPOINT, not a byte. `$` plus an emoji is one
             * rule here; the byte engine would need four chained appends and
             * could not compose them with the duplication verbs. */
            /* Collapse a run of appends or prepends into one op, swallowing
             * the same separators ruleproc.c swallows: space, tab, colon.
             *
             * This used to stop the run at any separator, on the reasoning
             * that emitting a second op gives identical output.  The output IS
             * identical -- and it is 17x slower, because each uncollapsed
             * prepend shifts the whole buffer, which is the quadratic the
             * collapse exists to remove.  Measured on applyrule32 directly,
             * 24 ops on a 28-codepoint word: the adjacent spelling `^a^b...`
             * ran 34.1ns and the space-separated `^a ^b ...` ran 578.7ns.
             * Real rule files are space-separated, so the optimisation was
             * firing on almost nothing: 13.1% of HashMob.100k.rule carries a
             * prepend run of two or more, longest 18.
             *
             * Two passes over the source rather than one pass into a buffer:
             * the run length has to be emitted before the operands, and a
             * local array here would be a stack buffer.
             *
             * The 254 cap matches ruleproc.c, whose run length is a single
             * byte (ruleproc.c:567, `x < 254`).  Keeping the same cap keeps
             * the two packed forms structurally convertible, and a longer run
             * simply emits a second op exactly as it does there. */
            case '$':
            case '^': {
                uint32_t verb = line[i];
                int k, cnt = 0;
                NEED(1);
                /* Quoted form: the operand is a STRING, appended or prepended
                 * as written. Note the chained form `^a^b^c` yields cba,
                 * because each prepend pushes in front of the last, whereas
                 * `^"abc"` must yield abc -- the author wrote a string, not a
                 * sequence of operations. The executor reverses a multi
                 * prepend, so the quoted operand is stored reversed to cancel
                 * that out and come back in the order it was written. */
                if (line[i+1] == RULE32_QUOTE) {
                    uint32_t opbuf[RULE32_MAXCP > 4096 ? 4096 : RULE32_MAXCP];
                    int used = 0, on;
                    on = read_operand(line, linelen, i + 1, opbuf,
                                      (int)(sizeof opbuf / sizeof opbuf[0]), &used);
                    if (on < 0) return RULE32_ERR_INVALID;   /* unterminated */
                    if (on == 0) { i += 1 + used; break; }   /* empty: no-op */
                    if (on == 1) {
                        EMIT(verb == '$' ? RULE_OP_APPEND : RULE_OP_PREPEND);
                        EMIT(opbuf[0]);
                    } else {
                        EMIT(verb == '$' ? RULE32_OP_APPEND_MULTI
                                         : RULE32_OP_PREPEND_MULTI);
                        EMIT(on);
                        if (verb == '$')
                            for (k = 0; k < on; k++) EMIT(opbuf[k]);
                        else
                            for (k = on - 1; k >= 0; k--) EMIT(opbuf[k]);
                    }
                    i += 1 + used;
                    break;
                }
                /* Backslash escapes, for `$` and `^` ONLY -- the two verbs
                 * where ruleproc.c runs PARSEHEX.  Every other verb takes a
                 * raw operand there and must keep taking one here: in the
                 * local rule corpus 20,638 lines use a literal backslash as a
                 * substitution operand (`s9\` and `s\1` shapes) and 11,523
                 * append or prepend one, so moving this into read_operand()
                 * would silently change the meaning of all of them.
                 *
                 * An ASCII escape resolves to the same character the byte
                 * engine produces, which is the point of doing this at all.
                 * A value of 0x80 or above is REFUSED rather than read as
                 * U+00NN: there the byte engine inserts a raw byte, a byte is
                 * not a codepoint, and inventing an interpretation is the one
                 * thing this engine does not do anywhere else.  The quoted
                 * form says it without ambiguity -- $"é" -- and is already
                 * supported.
                 *
                 * Before this, `$\\` gave one backslash under the byte engine
                 * and nothing here, and `$\x41` gave `A` there and a literal
                 * backslash followed by a misparse here.  That was the only
                 * undeliberate divergence between the two engines. */
                if (line[i+1] == '\\') {
                    uint32_t v = '\\';
                    int adv = 2;                     /* verb + backslash */
                    if (i + 2 < linelen) {
                        uint32_t c2 = line[i+2];
                        if (c2 == 'x' || c2 == 'X') {
                            int d1 = -1, d2 = -1;
                            if (i + 4 < linelen) {
                                d1 = hexval32(line[i+3]);
                                d2 = hexval32(line[i+4]);
                            }
                            if (d1 < 0 || d2 < 0) return RULE32_ERR_INVALID;
                            v = (uint32_t)((d1 << 4) | d2);
                            if (v > 0x7F) return RULE32_ERR_INVALID;
                            adv = 5;
                        } else if (c2 >= '0' && c2 <= '3'
                                   && !(i + 3 < linelen
                                        && (line[i+3] == 'x' || line[i+3] == 'X'))) {
                            uint32_t acc = c2 - '0';
                            int j2 = i + 3;
                            while (j2 < linelen && line[j2] >= '0' && line[j2] <= '7') {
                                acc = (acc << 3) | (line[j2] - '0');
                                if (acc > 0x10FFFF) return RULE32_ERR_INVALID;
                                j2++;
                            }
                            if (acc > 0x7F) return RULE32_ERR_INVALID;
                            v = acc;
                            adv = j2 - i;
                        } else if (c2 == '\\') {
                            v = '\\';
                            adv = 3;
                        }
                        /* anything else: the operand is the backslash itself
                         * and line[i+2] is parsed as the next rule token,
                         * which is what both engines already did. */
                    }
                    EMIT(verb == '$' ? RULE_OP_APPEND : RULE_OP_PREPEND);
                    EMIT(v);
                    i += adv;
                    break;
                }
                {
                    int j = i, endi;
                    while (j + 1 < linelen && line[j] == verb && cnt < 254) {
                        cnt++;
                        j = skip_run_seps(line, linelen, j + 2);
                    }
                    endi = j;
                    if (cnt == 1) {
                        EMIT(verb == '$' ? RULE_OP_APPEND : RULE_OP_PREPEND);
                        EMIT(line[i+1]);
                        i += 2;
                        break;
                    }
                    EMIT(verb == '$' ? RULE32_OP_APPEND_MULTI
                                     : RULE32_OP_PREPEND_MULTI);
                    EMIT(cnt);
                    for (j = i, k = 0; k < cnt; k++) {
                        EMIT(line[j+1]);
                        j = skip_run_seps(line, linelen, j + 2);
                    }
                    i = endi;
                    break;
                }
            }

            /* Rejection verbs. These decide whether a candidate is emitted at
             * all, so they are the most common verbs in real rule files and
             * have to be present before the engine is useful on one. */
            case '_': EMIT(RULE_OP_REJ_LEN_NE); POSARG; break;
            case '<': EMIT(RULE_OP_REJ_LEN_GE); POSARG; break;
            case '>': EMIT(RULE_OP_REJ_LEN_LE); POSARG; break;

            /* `=NX` and `%NX`.  packrule32 refused both outright, so any
             * rule using them died as an invalid rule while the byte engine
             * accepted and misexecuted them.  Both are documented and in
             * agreement across john and hashcat.  They pack as their literal
             * ASCII opcode (0x3d, 0x25) to match ruleproc.c's stream shape.
             * Only a single codepoint operand is accepted, which is all
             * either reference defines; the class forms (`=N?C`, `%N?C`)
             * are a separate feature neither engine has. */
            case '=':
            case '%': {
                uint32_t ob[4]; int used = 0, on, p; unsigned char cb = 0;
                NEED(2);
                p = pos_of(line[i+1]);
                if (p < 0) return RULE32_ERR_INVALID;
                on = read_class_operand(line, linelen, i+2, ob, 4, &used, &cb, 0);
                if (on != 1) return RULE32_ERR_INVALID;
                /* `=N?C` / `%N?C`: the class forms.  The comment above said
                 * neither engine had them; ruleproc.c gained both with the
                 * class work, so they are here too. */
                if (cb) { EMIT(line[i] == '=' ? RULE_OP_REJ_AT_CLASS
                                              : RULE_OP_REJ_CNT_CLASS);
                          EMIT(p); EMIT(cb); }
                else { EMIT(line[i] == '=' ? (uint32_t)'=' : (uint32_t)'%');
                       EMIT(p); EMIT(ob[0]); }
                i += 2 + used; break;
            }
            case '!': {
                uint32_t ob[512]; int used = 0, on, k; unsigned char cb = 0;
                NEED(1);
                on = read_class_operand(line, linelen, i+1, ob, 512, &used, &cb, 0);
                if (on <= 0) return RULE32_ERR_INVALID;
                if (cb) { EMIT(RULE_OP_REJ_HAS_CLASS); EMIT(cb); }
                else if (on == 1) { EMIT(RULE_OP_REJ_HAS); EMIT(ob[0]); }
                else { EMIT(RULE32_OP_REJ_HAS_STR); EMIT(on);
                       for (k = 0; k < on; k++) EMIT(ob[k]); }
                i += 1 + used; break;
            }
            case '/': {
                uint32_t ob[512]; int used = 0, on, k; unsigned char cb = 0;
                NEED(1);
                on = read_class_operand(line, linelen, i+1, ob, 512, &used, &cb, 0);
                if (on <= 0) return RULE32_ERR_INVALID;
                if (cb) { EMIT(RULE_OP_REJ_NHAS_CLASS); EMIT(cb); }
                else if (on == 1) { EMIT(RULE_OP_REJ_NHAS); EMIT(ob[0]); }
                else { EMIT(RULE32_OP_REJ_NHAS_STR); EMIT(on);
                       for (k = 0; k < on; k++) EMIT(ob[k]); }
                i += 1 + used; break;
            }
            case '(': {
                uint32_t ob[512]; int used = 0, on, k; unsigned char cb = 0;
                NEED(1);
                on = read_class_operand(line, linelen, i+1, ob, 512, &used, &cb, 0);
                if (on <= 0) return RULE32_ERR_INVALID;
                if (cb) { EMIT(RULE_OP_REJ_FIRST_CLASS); EMIT(cb); }
                else if (on == 1) { EMIT(RULE_OP_REJ_FIRST); EMIT(ob[0]); }
                else { EMIT(RULE32_OP_REJ_FIRST_STR); EMIT(on);
                       for (k = 0; k < on; k++) EMIT(ob[k]); }
                i += 1 + used; break;
            }
            case ')': {
                uint32_t ob[512]; int used = 0, on, k; unsigned char cb = 0;
                NEED(1);
                on = read_class_operand(line, linelen, i+1, ob, 512, &used, &cb, 0);
                if (on <= 0) return RULE32_ERR_INVALID;
                if (cb) { EMIT(RULE_OP_REJ_LAST_CLASS); EMIT(cb); }
                else if (on == 1) { EMIT(RULE_OP_REJ_LAST); EMIT(ob[0]); }
                else { EMIT(RULE32_OP_REJ_LAST_STR); EMIT(on);
                       for (k = 0; k < on; k++) EMIT(ob[k]); }
                i += 1 + used; break;
            }
            case '@': {           /* @X -- remove every X, or every string X */
                uint32_t ob[512]; int used = 0, on, k; unsigned char cb = 0;
                NEED(1);
                on = read_class_operand(line, linelen, i+1, ob, 512, &used, &cb, 0);
                if (on <= 0) return RULE32_ERR_INVALID;      /* empty needle */
                if (cb) { EMIT(RULE_OP_PURGE_CLASS); EMIT(cb); }
                else if (on == 1) { EMIT(RULE_OP_PURGE); EMIT(ob[0]); }
                else { EMIT(RULE32_OP_PURGE_STR); EMIT(on);
                       for (k = 0; k < on; k++) EMIT(ob[k]); }
                i += 1 + used; break;
            }
            /* hashcat's `~` character-class prefix.  Refused outright,
             * for the reason in read_class_operand().  Explicit so the
             * diagnostic names `~` and not whichever verb follows it. */
            /* hashcat's `~` class prefix.  The verb that follows takes a
             * MANDATORY `?C` operand from hashcat's six-class table, which is
             * what lets the two syntaxes coexist: `?H` is uppercase hex there
             * and the complement of hex in john's inline form.  Mirrors
             * ruleproc.c, including the emission ORDER -- opcode, then the
             * position for `=`/`%`, then the class, then the replacement for
             * `s` -- and the `~e?C` override to its own opcode, because
             * hashcat's title-with-class is a different algorithm rather than
             * john's with a class test substituted. */
            case '~': {
                uint32_t ob[4]; int used = 0, on, pp = -1;
                unsigned char cb = 0, plain = 0, cls = 0;
                uint32_t v;
                NEED(2);
                v = line[i+1];
                if (!rule_verb_opcodes((char)v, &plain, &cls))
                    return RULE32_ERR_INVALID;
                if (v == '=' || v == '%') {
                    NEED(3);
                    pp = pos_of(line[i+2]);
                    if (pp < 0) return RULE32_ERR_INVALID;
                    on = read_class_operand(line, linelen, i+3, ob, 4, &used, &cb, 1);
                } else {
                    on = read_class_operand(line, linelen, i+2, ob, 4, &used, &cb, 1);
                }
                if (on <= 0) return RULE32_ERR_INVALID;
                if (v == 'e' && cb) cls = RULE_OP_TITLE_CLASS_HC;
                EMIT(cb ? (uint32_t)cls : (uint32_t)plain);
                if (v == '=' || v == '%') EMIT((uint32_t)pp);
                EMIT(cb ? (uint32_t)cb : ob[0]);
                if (v == 's') {
                    int q = (v == '=' || v == '%') ? i + 3 + used : i + 2 + used;
                    if (q >= linelen) return RULE32_ERR_INVALID;
                    EMIT(line[q]);
                    used += 1;
                }
                i += ((v == '=' || v == '%') ? 3 : 2) + used;
                break; }

            case 'q': EMIT(RULE_OP_DUP_EACH);  i++; break;
            case '+': EMIT(RULE_OP_INC);       POSARG; break;
            case '-': EMIT(RULE_OP_DEC);       POSARG; break;
            case '.': EMIT(RULE_OP_REPL_NEXT); POSARG; break;
            case ',': EMIT(RULE_OP_REPL_PREV); POSARG; break;
            case 'L': EMIT(RULE_OP_BIT_SHL);   POSARG; break;
            case 'R': EMIT(RULE_OP_BIT_SHR);   POSARG; break;
            case 'M': EMIT(RULE_OP_MEM_STORE); i++; break;
            case '4': EMIT(RULE_OP_MEM_APP);   i++; break;
            case '6': EMIT(RULE_OP_MEM_PRE);   i++; break;
            case 'Q': EMIT(RULE_OP_MEM_REJ);   i++; break;

            case 'X':   /* XNMI -- N ignored, M from memory, at position I */
                NEED(3); EMIT(RULE_OP_MEM_INSERT);
                { int a = pos_of(line[i+1]), b = pos_of(line[i+2]), cc = pos_of(line[i+3]);
                  if (a < 0 || b < 0 || cc < 0) return RULE32_ERR_INVALID;
                  EMIT(a); EMIT(b); EMIT(cc); }
                i += 4; break;
            case 'E': EMIT(RULE_OP_TITLE_SP); i++; break;
            case 'e': {
                uint32_t ob[512]; int used = 0, on, k; unsigned char cb = 0;
                NEED(1);
                on = read_class_operand(line, linelen, i+1, ob, 512, &used, &cb, 0);
                if (on <= 0) return RULE32_ERR_INVALID;
                if (cb) { EMIT(RULE_OP_TITLE_CLASS); EMIT(cb); }
                else if (on == 1) { EMIT(RULE_OP_TITLE_SEP); EMIT(ob[0]); }
                else { EMIT(RULE32_OP_TITLE_SEP_STR); EMIT(on);
                       for (k = 0; k < on; k++) EMIT(ob[k]); }
                i += 1 + used; break;
            }

            /* sXY -- substitute every X with Y. Either operand may be a
             * quoted string, and they need not be the same length, so the
             * string form is a separate op with a rebuild rather than the
             * in-place codepoint swap. */
            case 's': {
                uint32_t fb[512], tb[512];
                int fu = 0, tu = 0, fn, tn, k; unsigned char fcb = 0;
                NEED(1);
                fn = read_class_operand(line, linelen, i+1, fb, 512, &fu, &fcb, 0);
                if (fn < 0) return RULE32_ERR_INVALID;
                { int q0 = i + 1 + fu, q = q0;
                  /* Skip separators ONLY between two QUOTED operands. A space
                   * is a perfectly good unquoted operand -- `s_ ` substitutes
                   * an underscore for a space, and four such rules live in the
                   * first thousand of a real rule file. Skipping there ate the
                   * operand and refused the rule. */
                  if (fu > 1) {
                      int q2 = skip_seps(line, linelen, q0);
                      if (q2 < linelen && line[q2] == RULE32_QUOTE) q = q2;
                  }
                  if (q >= linelen) return RULE32_ERR_INVALID;
                  tn = read_operand(line, linelen, q, tb, 512, &tu);
                  if (tn < 0) return RULE32_ERR_INVALID;
                  tu += q - q0; }                  /* count any separators */
                /* `s?CY` -- substitute every member of class C with Y.  Only
                 * the FROM operand may be a class; the replacement is always a
                 * character, as in ruleproc.c. */
                if (fcb) {
                    if (tn != 1) return RULE32_ERR_INVALID;
                    EMIT(RULE_OP_SUB_CLASS); EMIT(fcb); EMIT(tb[0]);
                } else if (fn == 1 && tn == 1) {
                    EMIT(RULE_OP_SUB); EMIT(fb[0]); EMIT(tb[0]);
                } else {
                    if (fn == 0) return RULE32_ERR_INVALID;   /* empty needle */
                    EMIT(RULE32_OP_SUB_STR);
                    EMIT(fn); for (k = 0; k < fn; k++) EMIT(fb[k]);
                    EMIT(tn); for (k = 0; k < tn; k++) EMIT(tb[k]);
                }
                i += 1 + fu + tu; break;
            }

            case '*':   /* *NM -- swap the codepoints at N and M */
                NEED(2); EMIT(RULE_OP_SWAP_AT);
                { int p = pos_of(line[i+1]), q = pos_of(line[i+2]);
                  if (p < 0 || q < 0) return RULE32_ERR_INVALID; EMIT(p); EMIT(q); }
                i += 3; break;

            case 'y':   /* yN -- duplicate the first N to the front */
                EMIT(RULE_OP_DUP_PREFIX); POSARG; break;
            case 'Y':   /* YN -- duplicate the last N to the end */
                EMIT(RULE_OP_DUP_SUFFIX); POSARG; break;

            case 'T': EMIT(RULE_OP_TOGGLE_AT); POSARG; break;
            case 'D': EMIT(RULE_OP_DEL_AT);    POSARG; break;
            case '\'': EMIT(RULE_OP_TRUNC);    POSARG; break;
            case 'z': EMIT(RULE_OP_DUP_FIRST); POSARG; break;
            case 'Z': EMIT(RULE_OP_DUP_LAST);  POSARG; break;
            case 'p': EMIT(RULE_OP_REPEAT);    POSARG; break;

            case 'i': {  /* iNX -- insert X at cluster position N */
                uint32_t ob[512]; int used = 0, on, k, p;
                NEED(2);
                p = pos_of(line[i+1]);
                if (p < 0) return RULE32_ERR_INVALID;
                on = read_operand(line, linelen, i+2, ob, 512, &used);
                if (on <= 0) return RULE32_ERR_INVALID;
                if (on == 1) { EMIT(RULE_OP_INSERT); EMIT(p); EMIT(ob[0]); }
                else { EMIT(RULE32_OP_INSERT_STR); EMIT(p); EMIT(on);
                       for (k = 0; k < on; k++) EMIT(ob[k]); }
                i += 2 + used; break;
            }

            case 'o': {  /* oNX -- replace cluster N with X */
                uint32_t ob[512]; int used = 0, on, k, p;
                NEED(2);
                p = pos_of(line[i+1]);
                if (p < 0) return RULE32_ERR_INVALID;
                on = read_operand(line, linelen, i+2, ob, 512, &used);
                if (on <= 0) return RULE32_ERR_INVALID;
                if (on == 1) { EMIT(RULE_OP_OVERWRITE); EMIT(p); EMIT(ob[0]); }
                else { EMIT(RULE32_OP_OVERW_STR); EMIT(p); EMIT(on);
                       for (k = 0; k < on; k++) EMIT(ob[k]); }
                i += 2 + used; break;
            }

            case 'x':   /* xNM -- extract M codepoints from N */
                NEED(2); EMIT(RULE_OP_EXTRACT);
                { int p = pos_of(line[i+1]), q = pos_of(line[i+2]);
                  if (p < 0 || q < 0) return RULE32_ERR_INVALID; EMIT(p); EMIT(q); }
                i += 3; break;

            case 'O':   /* ONM -- omit M codepoints from N */
                NEED(2); EMIT(RULE_OP_OMIT);
                { int p = pos_of(line[i+1]), q = pos_of(line[i+2]);
                  if (p < 0 || q < 0) return RULE32_ERR_INVALID; EMIT(p); EMIT(q); }
                i += 3; break;

            default:
                /* Not yet implemented in this tranche. Refuse the rule rather
                 * than silently dropping the verb -- a rule that half-applies
                 * is a wrong answer that looks like a right one. */
                return RULE32_ERR_INVALID;
        }
    }
    EMIT(RULE32_END);
    return n;
#undef EMIT
#undef NEED
#undef POSARG
}

/*
 * packrule32 - compile a rule line, retrying with quotes taken literally
 *
 * A double quote is both the operand delimiter and a perfectly ordinary
 * character to append, insert or substitute.  `$"` is unambiguous to a reader
 * and, to the parser, an operand that opens and never closes: 138 of the
 * 17,320 catalog rules are exactly that shape and every one was refused,
 * which is the whole of the divergence from the byte engine.
 *
 * The line survives packing, so the resolution is to ask the parser twice --
 * once with quoted operands enabled, and, only if that fails, once with
 * quotes as plain characters.  No heuristic on quote counts is needed: the
 * question "does this line parse" already answers it.
 *
 * A line that compiles today keeps its present meaning, because the second
 * pass runs only after the first has failed.  A line that compiles under
 * neither reading is still refused.  RULE32_ERR_NOROOM is never retried: it
 * means the caller sized a buffer wrongly, and retrying would hide that.
 */
int packrule32(const uint32_t *line, int linelen, uint32_t *out, int outmax)
{
    int rc = packrule32_pass(line, linelen, out, outmax);

    if (rc != RULE32_ERR_INVALID) return rc;

    Rule32LiteralQuotes = 1;
    rc = packrule32_pass(line, linelen, out, outmax);
    Rule32LiteralQuotes = 0;
    return rc;
}

/* "Alphabetic" for the two capitalize verbs. ruleproc.c tests [A-Za-z];
 * here the test is whether the codepoint has a case mapping at all, so the
 * verbs reach Latin-1, Extended-A and Cyrillic the same way every other case
 * verb in this file does. Probes pass NULL for amb: asking whether a
 * codepoint is cased must not mark the word as locale-ambiguous. */
static const struct cp_expand_ent *cp_expand_find(uint32_t c);

static int cp_is_cased(uint32_t c, int turkish)
{
    /* A ligature has no 1:1 mapping in EITHER direction -- its uppercase grows
     * and its lowercase is itself -- so asking cp_upper alone would call it
     * caseless and `c` would walk straight past one looking for a letter. */
    if (cp_expand_find(c)) return 1;
    return cp_upper(c, turkish, NULL) != c || cp_lower(c, turkish, NULL) != c;
}

/* Greek sigma is the one case mapping that depends on WHERE the character sits
 * rather than what it is: a sigma ending a word is U+03C2, anywhere else
 * U+03C3. The table cannot express that, so it maps capital sigma to the
 * medial form and this runs afterwards over a whole-word lowercase.
 *
 * "End of word" is end of buffer, or a following codepoint that is not a
 * cased letter -- so a rule that lowercases "ΟΔΟΣ ΜΟΥ" gets both sigmas right,
 * and one that lowercases "ΟΔΟΣΜΟΥ" correctly leaves the medial form. Only the
 * whole-string verbs call this; a single-position toggle has no word to be at
 * the end of. */
/* Uppercase that is allowed to GROW. German eszett uppercases to SS, which is
 * the one mapping in the covered ranges that changes length; the generated
 * table excludes it precisely because a table cannot hold it, so it lands
 * here instead. Writes 1 or 2 codepoints and returns the count.
 *
 * Only the whole-string verbs use this. A position-indexed verb keeps the 1:1
 * cp_upper, because growing a character underneath an index would move every
 * position the rest of the rule refers to. So `u` on strasse-with-eszett gives
 * STRASSE, while a toggle AT a position leaves the eszett alone. */
static const struct cp_expand_ent *cp_expand_find(uint32_t c)
{
    int lo = 0, hi = CP_EXPAND_N - 1;

    /* Bounds first. Every ASCII character fails this in one comparison, which
     * matters because the counting pass in upper_all asks about every
     * character of every candidate. */
    if (c < CP_EXPAND_LO || c > CP_EXPAND_HI) return NULL;
    while (lo <= hi) {
        int mid = (lo + hi) >> 1;
        if (c < cp_expand[mid].cp)      hi = mid - 1;
        else if (c > cp_expand[mid].cp) lo = mid + 1;
        else return &cp_expand[mid];
    }
    return NULL;
}

static int cp_upper_full(uint32_t c, int turkish, int *amb, uint32_t *out)
{
    const struct cp_expand_ent *e = cp_expand_find(c);
    int k;

    if (e) { for (k = 0; k < (int)e->n; k++) out[k] = e->out[k]; return (int)e->n; }
    out[0] = cp_upper(c, turkish, amb);
    return 1;
}

/* Uppercase the whole buffer in place, growing it where a mapping expands.
 * ONE pass over the buffer, which is the whole point.
 *
 * It used to detect expansion in a pre-scan and then map, so `u` traversed the
 * buffer twice where `l` traverses it once -- and MEASURED on an fpga.local
 * GTX 1080, that is exactly what `u` cost: 123.9 Mh/s against `l`'s 233.7, a
 * ratio of 1.886. Deleting the pre-scan outright took `u` to 262.8. The pass is
 * the cost, not the work inside it: making the pre-scan's test cheap, via a
 * derived block bitmap that removed 91.7% of its binary searches, bought
 * exactly 1.00x on the GPU. See gpu/CLAUDE.md.
 *
 * So expansion is handled inline instead. cp_upper_full returns 1 for every
 * codepoint that does not grow, which is all but 0.0127% of a multi-script
 * corpus, and the shift below runs only for the rest. Growth cannot cascade --
 * no expansion output is itself an expanding codepoint -- so `i` skips the
 * codepoints just written, exactly as the backwards write did. */
static int upper_all(uint32_t *b, int n, int turkish, int *amb, int maxcp)
{
    uint32_t tmp[CP_EXPAND_MAX];
    int i, j, k, len = n;

    for (i = 0; i < len; i++) {
        k = cp_upper_full(b[i], turkish, amb, tmp);
        if (k == 1) { b[i] = tmp[0]; continue; }
        if (len + k - 1 > maxcp) return -1;
        for (j = len - 1; j > i; j--) b[j + k - 1] = b[j];
        for (j = 0; j < k; j++)      b[i + j] = tmp[j];
        len += k - 1;
        i   += k - 1;
    }
    return len;
}

static void greek_final_sigma(uint32_t *b, int n, int turkish)
{
    int i;
    for (i = 0; i < n; i++) {
        if (b[i] != CP_GREEK_SMALL_SIGMA) continue;
        if (i + 1 == n || !cp_is_cased(b[i + 1], turkish))
            b[i] = CP_GREEK_SMALL_FINAL;
    }
}

int applyrule32(const uint32_t *rule, const uint32_t *in, int inlen,
                uint32_t *out, int outmax, int variant, int *nvariants)
{
    /* Thread-local rather than automatic. Each of these is 800KB at
     * RULE32_MAXCP, and two of them on the stack is 1.6MB per worker -- past
     * the default thread stack on several of the build targets. The byte
     * engine has the same pair (cpass and Memory) and keeps them out of frame
     * for the same reason. */
    static __thread uint32_t buf[RULE32_MAXCP];
    static __thread uint32_t mem[RULE32_MAXCP];
    /* Scratch for the string substitute, which cannot be done in place: the
     * replacement may be longer than what it replaces. Separate from mem,
     * which belongs to the memory verbs and must survive across ops. */
    static __thread uint32_t mem2[RULE32_MAXCP];
    int len, i, j;
    int memlen = 0;      /* memory is per-invocation, exactly as ruleproc.c resets it */
    int turkish = (variant == 1);   /* variant 0 = default locale, 1 = tr/az */
    int amb = 0;                    /* set when an ambiguous mapping is reached */

    if (nvariants) *nvariants = 1;

    if (!rule || !in || !out) return RULE32_ERR_NOROOM;
    if (inlen < 0 || inlen > RULE32_MAXCP) return RULE32_ERR_NOROOM;
    len = inlen;
    for (i = 0; i < len; i++) buf[i] = in[i];

/* Every verb that grows the buffer checks against RULE32_MAXCP. The byte engine
 * once wrote 40KB into a 20KB buffer (procrule.c 1.19) because a bound lived in
 * one file and the buffer in another; here the bound and the buffer are the same
 * constant, and every growth path goes through this macro. */
#define ROOM(need) do { if ((need) > RULE32_MAXCP) return RULE32_ERR_NOROOM; } while (0)

    while (*rule != RULE32_END) {
        uint32_t op = *rule++;

        switch (op) {
            case RULE_OP_NOOP: break;

            /* Collapsed runs. Appending is a straight copy to the tail.
             * Prepending must reproduce what N separate prepends do, and each
             * one pushes in front of the last, so `^1^2^3` yields 321 -- the
             * operands land in REVERSE rule order. */
            case RULE32_OP_APPEND_MULTI: {
                int cnt = (int)*rule++, k;
                ROOM(len + cnt);
                for (k = 0; k < cnt; k++) buf[len + k] = rule[k];
                rule += cnt;
                len += cnt;
                break;
            }

            case RULE32_OP_PREPEND_MULTI: {
                int cnt = (int)*rule++, k;
                ROOM(len + cnt);
                for (k = len - 1; k >= 0; k--) buf[k + cnt] = buf[k];
                for (k = 0; k < cnt; k++) buf[k] = rule[cnt - 1 - k];
                rule += cnt;
                len += cnt;
                break;
            }

            /* Memory group. The buffer does NOT persist between calls:
             * ruleproc.c clears memlen at entry, so a rule that reads memory
             * without storing first sees nothing. */
            /* Arithmetic on a CODEPOINT, not a byte. This is the deliberate
             * semantic change: `+`, `-`, `L` and `R` operate on the full
             * codepoint, so `L` on U+0041 gives U+0082 where the byte engine
             * gave 0x82. Waffle, 2026-09-04: "bit shift becomes 32-bit shift,
             * not 8-bit. Yes, this is a change. No, it won't work for
             * everything. Yes, we will have to document it."
             *
             * On ASCII the two agree for every value that stays in range,
             * which is what keeps the identity gate green. These verbs are
             * also the ONLY way to produce a codepoint UTF-8 cannot encode --
             * past U+10FFFF or inside the surrogate block -- and utf32_to_utf8
             * drops exactly those, emitting the rest of the line. */
            case RULE_OP_INC: { int y = (int)*rule++;
                if (y < len) buf[y]++; break; }
            case RULE_OP_DEC: { int y = (int)*rule++;
                if (y < len) buf[y]--; break; }
            case RULE_OP_BIT_SHL: { int y = (int)*rule++;
                if (y < len) buf[y] = buf[y] << 1; break; }
            case RULE_OP_BIT_SHR: { int y = (int)*rule++;
                if (y < len) buf[y] = buf[y] >> 1; break; }

            /* Note the asymmetry, which is ruleproc.c's and is mirrored: `.`
             * reads buf[y+1] with only `y < len` checked, so at the last
             * position it reads one past the end; `,` additionally requires
             * y > 0. Kept identical rather than tidied -- the byte engine's
             * output is the reference, and buf is oversized so the read is
             * in-bounds here. */
            /* `.N` copies the NEXT codepoint over position N. At the last
             * position there is no next one. ruleproc.c reads cpass[clen],
             * its NUL terminator, so it writes a NUL -- an artefact of a
             * NUL-terminated buffer, not a decision. Mirroring that read one
             * past the live length of a buffer that is static and REUSED, so
             * the answer depended on the previous candidate: the same word
             * gave 1234567Z after one word and 1234567Q after another. A rule
             * whose output depends on processing order is not a rule. At the
             * last position the character is now left alone. */
            case RULE_OP_REPL_NEXT: { int y = (int)*rule++;
                if (y < len - 1) buf[y] = buf[y + 1]; break; }
            case RULE_OP_REPL_PREV: { int y = (int)*rule++;
                if (y < len && y > 0) buf[y] = buf[y - 1]; break; }

            case RULE_OP_MEM_STORE:
                for (i = 0; i < len; i++) mem[i] = buf[i];
                memlen = len; break;

            case RULE_OP_MEM_APP:
                if (memlen > 0) { ROOM(len + memlen);
                    for (i = 0; i < memlen; i++) buf[len + i] = mem[i];
                    len += memlen; }
                break;

            case RULE_OP_MEM_PRE:
                if (memlen > 0) { ROOM(len + memlen);
                    for (i = len - 1; i >= 0; i--) buf[i + memlen] = buf[i];
                    for (i = 0; i < memlen; i++) buf[i] = mem[i];
                    len += memlen; }
                break;

            case RULE_OP_MEM_REJ:
                if (memlen == len) {
                    for (i = 0; i < len; i++) if (buf[i] != mem[i]) break;
                    if (i == len) return RULE32_REJECTED;
                }
                break;

            /* XNMI. Three operands are read but ruleproc.c uses only the last
             * two: N is hashcat's offset INTO memory and is silently ignored
             * there. Mirrored rather than corrected -- byte-for-byte agreement
             * on ASCII is the gate, and quietly changing a verb's meaning
             * between the two engines is the drift this design exists to
             * avoid. Worth revisiting deliberately, not as a side effect. */
            /* `X N M I` -- insert M of memory, from OFFSET N within it, at
             * position I.  Brought to ruleproc.c, the oracle.  Two divergences
             * were measured here with procrule -A:
             *
             *  1. N was read and DISCARDED, so every offset gave the same
             *     answer.  That was the byte engine's bug too and it was fixed
             *     there on 2026-09-11; this engine still carried it.
             *  2. It CLAMPED the count to what memory held and no-opped when the
             *     position was past the end.  ruleproc.c follows hashcat's
             *     mangle_insert_multi() and REJECTS on each of:
             *         no memory stored        memlen < 1
             *         zero count              M < 1
             *         position past end       I > length
             *         offset past memory      N > memlen
             *         range overruns memory   N + M > memlen
             *     Measured: MX042 on "aB3" gave "aBaB33" by clamping where
             *     ruleproc.c rejects (N+M = 4 > memlen 3), and X042 with no
             *     memory no-opped where ruleproc.c rejects.  Only the
             *     result-too-long case clamps, exactly as the oracle does.
             *
             * Memory is counted in codepoints, as in ruleproc.c, while the
             * insert position is a CLUSTER index like every other position
             * operand in this engine. */
            case RULE_OP_MEM_INSERT: {
                int off, cnt, at, nc, a;
                off = (int)*rule++;
                cnt = (int)*rule++;
                at  = (int)*rule++;
                nc  = cl_count(buf, len);
                if (memlen < 1 || cnt < 1 || at > nc ||
                    off > memlen || (off + cnt) > memlen)
                    return RULE32_REJECTED;
                if (len + cnt > RULE32_MAXCP) cnt = RULE32_MAXCP - len;
                if (cnt < 1) break;
                a = cl_start(buf, len, at);
                for (i = len - 1; i >= a; i--) buf[i + cnt] = buf[i];
                for (i = 0; i < cnt; i++)      buf[a + i] = mem[off + i];
                len += cnt;
                break; }

            /* Rejection. `_` compares against the ORIGINAL input length, not
             * the current one -- ruleproc.c tests `y != len`, the parameter,
             * while the others test clen. Using the running length here would
             * change which candidates survive a multi-verb rule. */
            /* Length is a count of CLUSTERS, so a rule asking for words of
             * eight characters gets eight as a reader counts them, not eight
             * codepoints of which three might be vowel marks. */
            /* CURRENT length, not the original.  This read cl_count(in,
             * inlen), so `$a _5` on a 4-cluster word rejected although the
             * candidate is 5 clusters by then.  Matches ruleproc.c's fix and
             * john: "_N reject the word unless it is N characters long". */
            case RULE_OP_REJ_LEN_NE: { int y = (int)*rule++;
                if (y != cl_count(buf, len)) return RULE32_REJECTED; break; }
            /* john: "<N reject unless it is less than N characters long",
             * ">N ... unless greater than N".  These tested the complement,
             * so `<5` kept clusters >= 5.  Operator ruling 2026-09-11: match
             * john.  Same fix as ruleproc.c. */
            case RULE_OP_REJ_LEN_GE: { int y = (int)*rule++;
                if (cl_count(buf, len) >= y) return RULE32_REJECTED; break; }
            case RULE_OP_REJ_LEN_LE: { int y = (int)*rule++;
                if (cl_count(buf, len) <= y) return RULE32_REJECTED; break; }
            case '=': { int y = (int)*rule++; uint32_t cp = *rule++;
                if (y >= len || buf[y] != cp) return RULE32_REJECTED; break; }
            case '%': { int y = (int)*rule++; uint32_t cp = *rule++;
                int cnt = 0;
                for (i = 0; i < len; i++) if (buf[i] == cp) cnt++;
                if (cnt < y) return RULE32_REJECTED; break; }
            case RULE_OP_REJ_HAS: { uint32_t cp = *rule++;
                for (i = 0; i < len; i++) if (buf[i] == cp) return RULE32_REJECTED;
                break; }
            case RULE_OP_REJ_NHAS: { uint32_t cp = *rule++;
                for (i = 0; i < len; i++) if (buf[i] == cp) break;
                if (i >= len) return RULE32_REJECTED; break; }
            case RULE_OP_REJ_FIRST: { uint32_t cp = *rule++;
                if (len > 0 && buf[0] != cp) return RULE32_REJECTED; break; }
            case RULE_OP_REJ_LAST: { uint32_t cp = *rule++;
                if (len > 0 && buf[len-1] != cp) return RULE32_REJECTED; break; }

            case RULE_OP_PURGE: { uint32_t cp = *rule++; int d = 0;
                for (i = 0; i < len; i++) if (buf[i] != cp) buf[d++] = buf[i];
                len = d; break; }

            case RULE_OP_SUB: { uint32_t from = *rule++, to = *rule++;
                for (i = 0; i < len; i++) if (buf[i] == from) buf[i] = to;
                break; }

            /* String substitute. Left to right, and it does NOT rescan what it
             * just wrote, so a replacement containing its own needle cannot
             * loop. Built forwards into a scratch buffer because the result
             * may be longer or shorter than the input. */
            case RULE32_OP_SUB_STR: {
                int fn = (int)*rule++;
                const uint32_t *fp = rule;
                int tn, w = 0, k;
                rule += fn;
                tn = (int)*rule++;
                { const uint32_t *tp = rule;
                  rule += tn;
                  for (i = 0; i < len; ) {
                      if (i + fn <= len) {
                          for (k = 0; k < fn && buf[i+k] == fp[k]; k++) ;
                          if (k == fn) {
                              if (w + tn > RULE32_MAXCP) return RULE32_ERR_NOROOM;
                              for (k = 0; k < tn; k++) mem2[w++] = tp[k];
                              i += fn; continue;
                          }
                      }
                      if (w >= RULE32_MAXCP) return RULE32_ERR_NOROOM;
                      mem2[w++] = buf[i++];
                  }
                }
                for (i = 0; i < w; i++) buf[i] = mem2[i];
                len = w; break;
            }

            /* A string occurs in the buffer starting at position q? */
            case RULE32_OP_REJ_HAS_STR:
            case RULE32_OP_REJ_NHAS_STR:
            case RULE32_OP_REJ_FIRST_STR:
            case RULE32_OP_REJ_LAST_STR: {
                int fn = (int)*rule++;
                const uint32_t *fp = rule;
                int k, hit = 0, start;
                rule += fn;
                if (op == RULE32_OP_REJ_FIRST_STR) {
                    if (fn <= len) {
                        for (k = 0; k < fn && buf[k] == fp[k]; k++) ;
                        hit = (k == fn);
                    }
                } else if (op == RULE32_OP_REJ_LAST_STR) {
                    if (fn <= len) {
                        start = len - fn;
                        for (k = 0; k < fn && buf[start+k] == fp[k]; k++) ;
                        hit = (k == fn);
                    }
                } else {
                    for (i = 0; i + fn <= len && !hit; i++) {
                        for (k = 0; k < fn && buf[i+k] == fp[k]; k++) ;
                        if (k == fn) hit = 1;
                    }
                }
                if (op == RULE32_OP_REJ_NHAS_STR || op == RULE32_OP_REJ_FIRST_STR ||
                    op == RULE32_OP_REJ_LAST_STR) {
                    if (!hit) return RULE32_REJECTED;
                } else if (hit) return RULE32_REJECTED;
                break;
            }

            case RULE32_OP_INSERT_STR: {
                int k = (int)*rule++, sn = (int)*rule++;
                const uint32_t *sp = rule;
                int at, w;
                rule += sn;
                at = cl_start(buf, len, k);
                if (cl_count(buf, len) > k) {
                    ROOM(len + sn);
                    for (w = len - 1; w >= at; w--) buf[w + sn] = buf[w];
                    for (w = 0; w < sn; w++) buf[at + w] = sp[w];
                    len += sn;
                }
                break;
            }

            case RULE32_OP_OVERW_STR: {
                int k = (int)*rule++, sn = (int)*rule++;
                const uint32_t *sp = rule;
                int a, b, w, d;
                rule += sn;
                if (k < cl_count(buf, len)) {
                    a = cl_start(buf, len, k);
                    b = cl_start(buf, len, k + 1);
                    d = sn - (b - a);
                    ROOM(len + (d > 0 ? d : 0));
                    if (d > 0) for (w = len - 1; w >= b; w--) buf[w + d] = buf[w];
                    else if (d < 0) for (w = b; w < len; w++) buf[w + d] = buf[w];
                    for (w = 0; w < sn; w++) buf[a + w] = sp[w];
                    len += d;
                }
                break;
            }

            /* vNS -- insert the string S after every N clusters. */
            case RULE32_OP_DIV_INS_STR: {
                int x = (int)*rule++, sn = (int)*rule++;
                const uint32_t *sp = rule;
                int nc, total, w, k, c2, a, b;
                rule += sn;
                if (x <= 0) return RULE32_ERR_INVALID;
                nc = cl_count(buf, len);
                if (nc < x) break;
                total = len + (nc / x) * sn;
                ROOM(total);
                w = total;
                for (c2 = nc - 1; c2 >= 0; c2--) {
                    if (((c2 + 1) % x) == 0)
                        for (k = sn - 1; k >= 0; k--) buf[--w] = sp[k];
                    a = cl_start(buf, len, c2); b = cl_start(buf, len, c2 + 1);
                    for (k = b - 1; k >= a; k--) buf[--w] = buf[k];
                }
                len = total;
                break;
            }

            /* 3NS -- toggle the first cased codepoint after the Nth S. */
            case RULE32_OP_TOGGLE_SEP_STR: {
                int y = (int)*rule++, sn = (int)*rule++;
                const uint32_t *sp = rule;
                int toggle_next = 0, occurrence = 0, k;
                rule += sn;
                for (i = 0; i < len; ) {
                    if (i + sn <= len) {
                        for (k = 0; k < sn && buf[i+k] == sp[k]; k++) ;
                        if (k == sn) {
                            if (occurrence == y) toggle_next = 1;
                            else occurrence++;
                            i += sn; continue;
                        }
                    }
                    if (toggle_next) {
                        uint32_t u = cp_upper(buf[i], turkish, &amb);
                        buf[i] = (u != buf[i]) ? u : cp_lower(buf[i], turkish, &amb);
                        break;
                    }
                    i++;
                }
                break;
            }

            /* eS -- title-case: raise the first cased codepoint, and the first
             * after each occurrence of the separator string, lower the rest. */
            case RULE32_OP_TITLE_SEP_STR: {
                int sn = (int)*rule++;
                const uint32_t *sp = rule;
                int z = 0, k;
                rule += sn;
                for (i = 0; i < len; ) {
                    if (i + sn <= len) {
                        for (k = 0; k < sn && buf[i+k] == sp[k]; k++) ;
                        if (k == sn) { z = 0; i += sn; continue; }
                    }
                    { uint32_t ch = buf[i];
                      if (z == 0 && cp_upper(ch, turkish, NULL) != ch) {
                          z = 1; buf[i] = cp_upper(ch, turkish, &amb);
                      } else if (cp_lower(ch, turkish, NULL) != ch) {
                          buf[i] = cp_lower(ch, turkish, &amb);
                      } }
                    i++;
                }
                break;
            }

            case RULE32_OP_PURGE_STR: {
                int fn = (int)*rule++;
                const uint32_t *fp = rule;
                int w = 0, k;
                rule += fn;
                for (i = 0; i < len; ) {
                    if (i + fn <= len) {
                        for (k = 0; k < fn && buf[i+k] == fp[k]; k++) ;
                        if (k == fn) { i += fn; continue; }
                    }
                    buf[w++] = buf[i++];
                }
                len = w; break;
            }

            case RULE_OP_SWAP_AT: { int p = (int)*rule++, q = (int)*rule++;
                int nc = cl_count(buf, len);
                if (p < nc && q < nc && p != q)
                    cl_swap(buf, len, p < q ? p : q, p < q ? q : p);
                break; }

            /* q -- double every CLUSTER, so a pointed letter is doubled with
             * its points rather than the mark being doubled on its own. */
            case RULE_OP_DUP_EACH: {
                int nc = cl_count(buf, len), w, a, b, c2;
                ROOM(len * 2);
                w = len * 2;
                for (c2 = nc - 1; c2 >= 0; c2--) {
                    a = cl_start(buf, len, c2); b = cl_start(buf, len, c2 + 1);
                    for (j = 0; j < 2; j++) { int t2;
                        for (t2 = b - 1; t2 >= a; t2--) buf[--w] = buf[t2]; }
                }
                len *= 2; break; }

            case RULE_OP_DUP_PREFIX: { int nrep = (int)*rule++;
                int cpn = cl_start(buf, len, nrep);
                if (nrep > 0 && cl_count(buf, len) >= nrep) { ROOM(len + cpn);
                    for (i = len - 1; i >= 0; i--) buf[i + cpn] = buf[i];
                    for (i = 0; i < cpn; i++) buf[i] = buf[cpn + i];
                    len += cpn; }
                break; }

            case RULE_OP_DUP_SUFFIX: { int nrep = (int)*rule++;
                int nc = cl_count(buf, len);
                int cpn = len - cl_start(buf, len, nc - nrep);
                if (nrep > 0 && nc >= nrep) { ROOM(len + cpn);
                    for (i = 0; i < cpn; i++) buf[len + i] = buf[len - cpn + i];
                    len += cpn; }
                break; }

            /* E and e: title case. Subtler than "upper the first, lower the
             * rest", and I had it wrong three ways before reading ruleproc.c:
             *   - the "start of word" flag is set ONLY by a lowercase LETTER,
             *     so a leading digit does not consume it ("3password" -> "3Password")
             *   - characters that are not letters are left entirely alone
             *   - any OTHER uppercase letter is lowered
             * The byte engine tests a-z and A-Z directly; the codepoint form
             * asks the case table instead, which generalises it to every script
             * the table covers while staying identical on ASCII. */
            case RULE_OP_TITLE_SP:
            case RULE_OP_TITLE_SEP: {
                uint32_t sep = (op == RULE_OP_TITLE_SEP) ? *rule++ : (uint32_t)' ';
                int z = 0;
                for (i = 0; i < len; i++) {
                    uint32_t ch = buf[i];
                    /* An already-uppercase letter at a word start IS the
                     * capital: keep it and mark the word started.  The old
                     * form lowercased it and left z at 0, so the NEXT
                     * lowercase letter was capitalised instead -- Hello1
                     * became hEllo1, "abc Def" became "Abc dEf".  Same defect
                     * and same fix as ruleproc.c's four TITLE sites; hashcat
                     * 6.2.5 and john both give Hello1 / Abc Def. */
                    if (ch == sep) { z = 0; }
                    else if (cp_lower(ch, turkish, NULL) != ch) {
                        if (z == 0) z = 1;          /* already capitalised */
                        else buf[i] = cp_lower(ch, turkish, &amb);
                    } else if (cp_upper(ch, turkish, NULL) != ch) {
                        if (z == 0) { z = 1; buf[i] = cp_upper(ch, turkish, &amb); }
                    }
                }
                break; }

            case RULE_OP_LOWER: {
                /* The sigma pass is skipped unless one was actually produced.
                 * Folding the test into the mapping loop costs one compare per
                 * character instead of a second walk over every candidate. */
                int sigma = 0;
                for (i = 0; i < len; i++) {
                    buf[i] = cp_lower(buf[i], turkish, &amb);
                    if (buf[i] == CP_GREEK_SMALL_SIGMA) sigma = 1;
                }
                if (sigma) greek_final_sigma(buf, len, turkish);
                break;
            }
            case RULE_OP_UPPER: {
                int nl = upper_all(buf, len, turkish, &amb, RULE32_MAXCP);
                if (nl < 0) return RULE32_ERR_NOROOM;
                len = nl;
                break;
            }

            /* NOT hashcat's `c`. ruleproc.c capitalizes the first ALPHABETIC
             * codepoint, skipping any leading digits or punctuation, and
             * lowercases the rest -- hashcat uppercases position 0 whatever it
             * is. The two agree on every word that starts with a letter, which
             * is why this only surfaced on a word starting with a digit:
             * `4aD4aD4a` gives `4Ad4ad4a` here and `4ad4ad4a` under hashcat.
             * ruleproc.c's INTEL and NOTINTEL paths differ in shape but agree
             * in result; the lower-all-then-raise-the-first form below is the
             * INTEL one. `C` is the exact mirror. */
            case RULE_OP_CAP:
                /* Uppercase the first cased codepoint, expanding if it is one
                 * of the growing forms -- `c` is a whole-word verb, so it
                 * follows the same rule as `u`. Note this is uppercase, not
                 * Unicode TITLE case: the fi ligature gives FInd here, where
                 * title case would give Find. Title case is a third mapping
                 * this engine does not carry.
                 *
                 * The hunt for that codepoint rides along with the lowercase
                 * pass rather than walking the buffer again. It used to be a
                 * second loop, and although that loop breaks at the first
                 * cased codepoint it is not therefore cheap: MEASURED on
                 * UTF8test2.txt, 63.20% of words carry NO cased codepoint at
                 * all, so the break never fires and the scan runs the whole
                 * word -- 33.8% of a full extra pass overall. Traversals are
                 * what cost on the device (see u32_upper_all in
                 * gpu/gpu_u32_walker.inc), so that was a third of a verb for
                 * the same number of cp_is_cased calls.
                 *
                 * fc survives greek_final_sigma, which runs between the two:
                 * it maps sigma to final sigma and both are cased, so it can
                 * neither move nor create the first cased position. buf[fc] is
                 * read AFTER that pass, exactly as the old second loop saw it. */
                { int sigma = 0, fc = -1;
                  for (i = 0; i < len; i++) {
                      buf[i] = cp_lower(buf[i], turkish, &amb);
                      if (buf[i] == CP_GREEK_SMALL_SIGMA) sigma = 1;
                      if (fc < 0 && cp_is_cased(buf[i], turkish)) fc = i;
                  }
                  if (sigma) greek_final_sigma(buf, len, turkish);
                  if (fc >= 0) {
                      uint32_t ex[CP_EXPAND_MAX];
                      int n2, j;
                      n2 = cp_upper_full(buf[fc], turkish, &amb, ex);
                      if (n2 > 1) {
                          ROOM(len + n2 - 1);
                          for (j = len - 1; j > fc; j--) buf[j + n2 - 1] = buf[j];
                          len += n2 - 1;
                      }
                      for (j = 0; j < n2; j++) buf[fc + j] = ex[j];
                  } }
                break;

            case RULE_OP_CAP_INV: {
                int nl = upper_all(buf, len, turkish, &amb, RULE32_MAXCP);
                if (nl < 0) return RULE32_ERR_NOROOM;
                len = nl;
                }
                for (i = 0; i < len; i++)
                    if (cp_is_cased(buf[i], turkish)) {
                        buf[i] = cp_lower(buf[i], turkish, &amb); break;
                    }
                break;

            case RULE_OP_TOGGLE:
                for (i = 0; i < len; i++) {
                    uint32_t u = cp_upper(buf[i], turkish, &amb);
                    buf[i] = (u != buf[i]) ? u : cp_lower(buf[i], turkish, &amb);
                }
                break;

            case RULE_OP_TOGGLE_AT: {
                int k = (int)*rule++;
                int p = cl_start(buf, len, k);      /* the cluster's BASE */
                if (k < cl_count(buf, len)) {
                    uint32_t u = cp_upper(buf[p], turkish, &amb);
                    buf[p] = (u != buf[p]) ? u : cp_lower(buf[p], turkish, &amb);
                }
                break;
            }

            /* Reversal is the headline fix: this reverses CODEPOINTS. The byte
             * engine reverses bytes and mangles every multi-byte character,
             * which cost real contest points. It does NOT reverse grapheme
             * clusters -- a base plus combining mark still comes apart, and
             * that is a known, accepted limit of the codepoint layer. */
            /* Reverse by GRAPHEME CLUSTER, not by codepoint, so a combining
             * mark stays with the base it belongs to. Reversing Arabic
             * meem-fatha-reh by codepoint put the fatha on the reh, which is a
             * different word; Hebrew niqqud came away from its letter the same
             * way, and an NFD e-plus-acute came apart entirely.
             *
             * Done in two in-place passes and no scratch buffer, which matters
             * because the working buffers here are already 800KB apiece.
             * Reverse every codepoint, which leaves each cluster's marks
             * sitting BEFORE its base; then reverse each run of marks-then-base
             * back, which restores the cluster while leaving the clusters
             * themselves in their new order.
             *
             * A word with no marks takes the first pass and a scan that finds
             * nothing to do, so ASCII, CJK and unpointed Arabic or Hebrew are
             * unchanged from the codepoint reversal they had before. */
            /* Reverse each cluster in place, then reverse the whole buffer.
             * The clusters come out in the opposite order with their contents
             * restored, and it needs no scratch buffer -- which matters, as the
             * working buffers here are 800KB apiece.
             *
             * The earlier version reversed everything first and then repaired
             * runs of marks-then-base. That only worked because a combining
             * mark FOLLOWS its base; a joiner sequence has bases on both sides
             * of the joiner, so the repair pass could not see the group. This
             * form makes no assumption about what a cluster contains. */
            case RULE_OP_REVERSE:
                for (i = 0; i < len; ) {
                    int e = cl_next(buf, len, i);
                    rev_span(buf, i, e);
                    i = (e > i) ? e : i + 1;
                }
                rev_span(buf, 0, len);
                break;

            case RULE_OP_DUP:
                ROOM(len * 2);
                for (i = 0; i < len; i++) buf[len + i] = buf[i];
                len *= 2;
                break;

            /* Reflect: append the reversed word. Cluster-aware for the same
             * reason `r` is -- reflecting by codepoint put an Arabic fatha on
             * the wrong letter, detached a skin-tone modifier so the copy began
             * with a bare one, and turned the US flag into SU, which is a
             * DIFFERENT flag rather than a mangled one. Same two-pass shape as
             * the reversal: copy forwards, reverse each cluster of the copy in
             * place, then reverse the copy as a whole. */
            case RULE_OP_REFLECT: {
                int e;
                ROOM(len * 2);
                for (i = 0; i < len; i++) buf[len + i] = buf[i];
                for (i = len; i < len * 2; ) {
                    e = cl_next(buf, len * 2, i);
                    rev_span(buf, i, e);
                    i = (e > i) ? e : i + 1;
                }
                rev_span(buf, len, len * 2);
                len *= 2;
                break;
            }

            /* Rotate by one CLUSTER: the leading letter and its marks move to
             * the end together. */
            case RULE_OP_ROT_L:
                if (cl_count(buf, len) > 1) {
                    int c1 = cl_start(buf, len, 1);
                    rev_span(buf, 0, c1); rev_span(buf, c1, len); rev_span(buf, 0, len);
                }
                break;

            case RULE_OP_ROT_R:
                if (cl_count(buf, len) > 1) {
                    int a = cl_start(buf, len, cl_count(buf, len) - 1);
                    rev_span(buf, 0, a); rev_span(buf, a, len); rev_span(buf, 0, len);
                }
                break;

            /* Drop a whole CLUSTER, not a codepoint. Dropping the leading
             * codepoint of a vocalised Arabic or Hebrew word left its
             * combining mark behind with nothing to attach to -- a string
             * that begins with a bare fatha or qamats is not a word, it is
             * malformed. Dropping the trailing codepoint was the mirror
             * defect: it took the mark off the last letter and left the
             * letter, so the word still ended in a letter that had lost its
             * vowel. Neither is what the rule means.
             *
             * These two verbs take no position operand, so cluster semantics
             * here are unambiguous: there is no index for a caller to have
             * meant in codepoints. */
            case RULE_OP_DROP_FIRST: {
                int e;
                if (!len) break;
                e = cl_next(buf, len, 0);
                for (i = 0; i + e < len; i++) buf[i] = buf[i + e];
                len -= e;
                break;
            }

            case RULE_OP_DROP_LAST: {
                int nc;
                if (!len) break;
                nc = cl_count(buf, len);
                len = cl_start(buf, len, nc - 1);
                break;
            }

            case RULE_OP_SWAP_FRONT:
                if (cl_count(buf, len) > 1) cl_swap(buf, len, 0, 1);
                break;

            case RULE_OP_SWAP_BACK: {
                int nc = cl_count(buf, len);
                if (nc > 1) cl_swap(buf, len, nc - 2, nc - 1);
                break;
            }

            /* One codepoint, one rule. */
            case RULE_OP_APPEND: {
                uint32_t cp = *rule++;
                ROOM(len + 1); buf[len++] = cp; break;
            }

            case RULE_OP_PREPEND: {
                uint32_t cp = *rule++;
                ROOM(len + 1);
                for (i = len; i > 0; i--) buf[i] = buf[i - 1];
                buf[0] = cp; len++; break;
            }

            case RULE_OP_DEL_AT: {
                int k = (int)*rule++, a, b, w;
                if (k >= cl_count(buf, len)) break;
                a = cl_start(buf, len, k);
                b = cl_start(buf, len, k + 1);
                for (w = a; w + (b - a) < len; w++) buf[w] = buf[w + (b - a)];
                len -= (b - a);
                break;
            }

            case RULE_OP_TRUNC: {
                int k = (int)*rule++;
                if (k < cl_count(buf, len)) len = cl_start(buf, len, k);
                break;
            }

            /* ruleproc.c guards with `clen > y`, STRICTLY greater: inserting at
             * the position one past the end is a no-op, not an append. I had
             * `p <= len`, which appended and produced an extra candidate on
             * eight rules in the corpus. */
            /* `>=`, not `>`: inserting at position == length is an APPEND, and
             * both references do it -- ruleproc.c states exactly that at its own
             * RULE_OP_INSERT, having fixed the same off-by-one.  Measured with
             * procrule -A: `i1?d` on the one-cluster word "x" gave "xx" here,
             * the insert having silently done nothing, against "x?x?" from the
             * oracle. */
            case RULE_OP_INSERT: {
                int k = (int)*rule++; uint32_t cp = *rule++;
                int p = cl_start(buf, len, k);
                if (cl_count(buf, len) >= k) { ROOM(len + 1);
                    for (i = len; i > p; i--) buf[i] = buf[i - 1];
                    buf[p] = cp; len++; }
                break;
            }

            /* Overwrite replaces the WHOLE cluster, not just its base: leaving
             * the marks behind would attach them to a letter they were never
             * written for. */
            case RULE_OP_OVERWRITE: {
                int k = (int)*rule++; uint32_t cp = *rule++;
                int p = k, a, b, w;
                if (k < cl_count(buf, len)) {
                    a = cl_start(buf, len, k);
                    b = cl_start(buf, len, k + 1);
                    buf[a] = cp;
                    for (w = a + 1; w + (b - a - 1) < len; w++) buf[w] = buf[w + (b - a - 1)];
                    len -= (b - a - 1);
                }
                /* Special case carried over from ruleproc.c: overwriting
                 * position 0 of an EMPTY buffer appends instead of doing
                 * nothing, so `o0X` on an emptied candidate still yields X. */
                if (p == 0 && len == 0) { ROOM(1); buf[0] = cp; len = 1; }
                break;
            }

            case RULE_OP_DUP_FIRST: {
                int nrep = (int)*rule++;
                int c1 = cl_start(buf, len, 1);          /* first cluster width */
                if (len) { ROOM(len + nrep * c1);
                    for (i = len - 1; i >= 0; i--) buf[i + nrep * c1] = buf[i];
                    for (i = 0; i < nrep; i++)
                        for (j = 0; j < c1; j++) buf[i * c1 + j] = buf[nrep * c1 + j];
                    len += nrep * c1; }
                break;
            }

            case RULE_OP_DUP_LAST: {
                int nrep = (int)*rule++;
                int a = cl_start(buf, len, cl_count(buf, len) - 1), cw = len - a;
                if (len) { ROOM(len + nrep * cw);
                    for (i = 0; i < nrep; i++)
                        for (j = 0; j < cw; j++) buf[len + i * cw + j] = buf[a + j];
                    len += nrep * cw; }
                break;
            }

            case RULE_OP_REPEAT: {
                int nrep = (int)*rule++;
                if (nrep > 0) { ROOM(len * (nrep + 1));
                    for (j = 1; j <= nrep; j++)
                        for (i = 0; i < len; i++) buf[j * len + i] = buf[i];
                    len *= (nrep + 1); }
                break;
            }

            /* Clamping derived from ruleproc.c, not inferred. When the start
             * is past the end the word is left UNCHANGED -- it is not
             * truncated and not emptied. Copy stops at the shorter of the
             * requested count and what remains, and the new length is whatever
             * was actually copied. Guessing this produced a word-destroying
             * difference on real rules. */
            /* `x N M` -- extract M clusters from N.  Matches ruleproc.c, which
             * is the oracle here (validated candidate-set-identical against
             * hashcat master's rp_cpu.c).  hashcat's mangle_extract() no-ops on
             * BOTH out-of-range conditions and never extracts a partial run:
             *     if (upos >= arr_len)         return arr_len;
             * a   if ((upos + ulen) > arr_len) return arr_len;
             * The (k + m) <= nc half of the guard was missing here, so a count
             * that overran extracted as much as was available -- john's
             * behaviour, superseded by operator ruling 2026-09-11.  Measured
             * divergence: `x14` on "aB3" gave "B3" where ruleproc.c leaves the
             * word unchanged.  RULE_OP_OMIT immediately below always had the
             * full guard, and its comment states the same requirement. */
            case RULE_OP_EXTRACT: {
                int k = (int)*rule++, m = (int)*rule++, x;
                int nc = cl_count(buf, len);
                int a = cl_start(buf, len, k), b = cl_start(buf, len, k + m);
                if (nc > k && (k + m) <= nc) {
                    for (x = 0; x < b - a; x++) buf[x] = buf[a + x];
                    len = x;
                }
                break;
            }

            /* Also from ruleproc.c. Note the guard requires the whole range to
             * fit -- (p + m) <= len -- and leaves the word untouched otherwise,
             * rather than omitting as much as it can. The final length is where
             * the shift loop stopped, which is NOT len - m at the boundary. */
            case RULE_OP_OMIT: {
                int k = (int)*rule++, m = (int)*rule++, x;
                int nc = cl_count(buf, len);
                int a = cl_start(buf, len, k), b = cl_start(buf, len, k + m);
                if (nc > k && (k + m) <= nc) {
                    for (x = a; x + (b - a) < len; x++) buf[x] = buf[x + (b - a)];
                    len = x;
                }
                break;
            }

            /* Whitespace and the explicit no-ops. ruleproc.c folds ' ' and
             * '\t' into the same arm as ':', so a rule file with padding
             * between verbs processes rather than failing. */
            case RULE_OP_NOOP_SP:
            case RULE_OP_NOOP_TAB:
                break;

            /* `#` stops rule processing and emits the buffer as it stands.
             * ruleproc.c reaches this by `goto fast_exit`, which skips the
             * remainder of the rule but still copies the word out -- it is an
             * early return, not a rejection. */
            case RULE_OP_HASH_EXIT:
                goto rule_done;

            /* `S`. Not hashcat's; an mdxfind extension that replaces every
             * 'a'/'A' with a literal newline (0x0A). ASCII-only in ruleproc.c
             * and deliberately left ASCII-only here: the verb names a byte
             * value, so widening it to every codepoint that looks like an `a`
             * would change which words it fires on. */
            case RULE_OP_S_SPECIAL:
                for (i = 0; i < len; i++)
                    if (buf[i] == 'a' || buf[i] == 'A') buf[i] = 0x0a;
                break;

            /* `vXc` -- insert c after every X codepoints. ruleproc.c writes
             * this as a backwards in-place shuffle that breaks as soon as the
             * read and write pointers meet; the form below is the same output
             * built forwards, verified against that loop for both the exact
             * (len % X == 0) and short-tail cases. A word shorter than X is
             * left alone rather than gaining a trailing separator. */
            case RULE_OP_DIV_INSERT: {
                int x = (int)*rule++;
                uint32_t c1 = *rule++;
                int total, w;
                if (x <= 0) return RULE32_ERR_INVALID;   /* malformed rule */
                if (len < x) break;
                total = len + len / x;
                ROOM(total);
                w = total;
                for (i = len - 1; i >= 0; i--) {
                    if (((i + 1) % x) == 0) buf[--w] = c1;
                    buf[--w] = buf[i];
                }
                len = total;
                break;
            }

            /* `3Nc` -- toggle the first cased codepoint following the N-th
             * occurrence of separator c, then stop. The operand is stored
             * 1-based and ruleproc.c subtracts one on read; mirrored, because
             * the compiler is shared and emits the same byte.
             *
             * ruleproc.c restricts the toggle to [A-Za-z] via `c ^ 0x20`.
             * Here it goes through the same case pair every other case verb
             * uses, so the separator can be any codepoint and the toggled
             * character can be Latin-1, Extended-A or Cyrillic. That is the
             * point of the engine; on ASCII the two agree exactly. */
            case RULE_OP_TOGGLE_SEP: {
                int y = (int)*rule++;      /* already 0-based; see packrule32 */
                uint32_t c1 = *rule++;
                int toggle_next = 0, occurrence = 0;
                for (i = 0; i < len; i++) {
                    if (buf[i] == c1) {
                        if (occurrence == y) toggle_next = 1;
                        else occurrence++;
                        continue;
                    }
                    if (toggle_next) {
                        uint32_t u = cp_upper(buf[i], turkish, &amb);
                        buf[i] = (u != buf[i]) ? u
                               : cp_lower(buf[i], turkish, &amb);
                        break;
                    }
                }
                break;
            }

            /* `H` / `h` -- hex-encode the word. This is the one verb where the
             * byte engine and the codepoint engine agree for ALL input, not
             * just ASCII: ruleproc.c hexes the raw bytes, the input to procrule
             * is UTF-8, so hexing the UTF-8 serialisation of the codepoints
             * reproduces it exactly. Hexing the 32-bit scalars instead would
             * give eight digits per character and disagree on every word.
             *
             * Written backwards in place, as ruleproc.c does. That is safe for
             * the same reason: every codepoint expands to at least two hex
             * digits, so the write index never falls behind the read index.
             *
             * Divergence, deliberate: ruleproc.c clamps at MAXLINE by encoding
             * as many leading bytes as fit and leaving the remaining
             * characters raw, producing a half-hexed hybrid. Here an
             * over-length result is NOROOM. RULE32_MAXCP is 5*MAXLINE so no
             * real line reaches it, and a caller that does has a bug worth
             * hearing about rather than a silently mangled word. */
            case RULE_OP_HEX_LOWER:
            case RULE_OP_HEX_UPPER: {
                static const char lc[] = "0123456789abcdef";
                static const char uc[] = "0123456789ABCDEF";
                const char *d = (op == RULE_OP_HEX_UPPER) ? uc : lc;
                unsigned char tmp[4];
                int total = 0, w, k, b;

                for (i = 0; i < len; i++) {
                    k = utf32_to_utf8(&buf[i], 1, tmp, (int)sizeof(tmp), NULL);
                    if (k > 0) total += 2 * k;      /* unencodable: dropped */
                }
                ROOM(total);
                w = total;
                for (i = len - 1; i >= 0; i--) {
                    k = utf32_to_utf8(&buf[i], 1, tmp, (int)sizeof(tmp), NULL);
                    if (k <= 0) continue;
                    for (b = k - 1; b >= 0; b--) {
                        buf[--w] = (uint32_t)(unsigned char)d[tmp[b] & 0xf];
                        buf[--w] = (uint32_t)(unsigned char)d[(tmp[b] >> 4) & 0xf];
                    }
                }
                len = total;
                break;
            }

            /* ---- character classes, opcodes 0x80-0x89 -------------------
             * Mirrors ruleproc.c, which is the oracle: membership comes from
             * the ONE shared table in rule_ops.h via RULE_CLASS_MATCH_CP, so a
             * class cannot mean two things in the two engines.
             *
             * Case toggling below is ASCII-only, exactly as ruleproc.c does it.
             * That is deliberate: this change is about PARITY, and making the
             * class verbs Unicode-aware would move -u away from the oracle
             * rather than towards it.  Same rationale as the >0xff membership
             * note in rule_ops.h.  Both are decisions to take separately.
             *
             * Positions are CLUSTER indices, as everywhere else in this engine,
             * so a position operand is resolved through cl_start(). */
            case RULE_OP_SUB_CLASS: {
                unsigned char cb = (unsigned char)*rule++;
                uint32_t to = *rule++;
                for (i = 0; i < len; i++)
                    if (RULE_CLASS_MATCH_CP(cb, buf[i])) buf[i] = to;
                break; }

            case RULE_OP_PURGE_CLASS: {
                unsigned char cb = (unsigned char)*rule++;
                int w = 0;
                for (i = 0; i < len; i++)
                    if (!RULE_CLASS_MATCH_CP(cb, buf[i])) buf[w++] = buf[i];
                len = w;
                break; }

            /* john's `e?C`: the class is a SEPARATOR.  A member resets the
             * "next letter starts a word" state; the letter after it is upper
             * and the rest of a word is lowered. */
            case RULE_OP_TITLE_CLASS: {
                unsigned char cb = (unsigned char)*rule++;
                int st = 0;
                for (i = 0; i < len; i++) {
                    uint32_t c = buf[i];
                    if (RULE_CLASS_MATCH_CP(cb, c)) { st = 0; continue; }
                    if (st == 0) { st = 1;
                        if (c >= 'a' && c <= 'z') buf[i] = c ^ 0x20; }
                    else { if (c >= 'A' && c <= 'Z') buf[i] = c ^ 0x20; }
                }
                break; }

            /* hashcat's `~e?C` is a DIFFERENT algorithm, not the same one with
             * a class test substituted -- it lowercases every character first
             * and decides the next position's case from THIS one, so the class
             * is tested pre-modification.  That is why it has its own opcode. */
            case RULE_OP_TITLE_CLASS_HC: {
                unsigned char cb = (unsigned char)*rule++;
                int up = 1;
                for (i = 0; i < len; i++) {
                    uint32_t c = buf[i];
                    int this_up = up;
                    up = RULE_CLASS_MATCH_CP(cb, c) ? 1 : 0;
                    if (c >= 'A' && c <= 'Z') { c ^= 0x20; buf[i] = c; }
                    if (this_up && c >= 'a' && c <= 'z') buf[i] = c ^ 0x20;
                }
                break; }

            case RULE_OP_REJ_HAS_CLASS: {
                unsigned char cb = (unsigned char)*rule++;
                for (i = 0; i < len; i++)
                    if (RULE_CLASS_MATCH_CP(cb, buf[i])) return RULE32_REJECTED;
                break; }

            case RULE_OP_REJ_NHAS_CLASS: {
                unsigned char cb = (unsigned char)*rule++;
                for (i = 0; i < len; i++)
                    if (RULE_CLASS_MATCH_CP(cb, buf[i])) break;
                if (i >= len) return RULE32_REJECTED;
                break; }

            case RULE_OP_REJ_FIRST_CLASS: {
                unsigned char cb = (unsigned char)*rule++;
                if (len > 0 && !RULE_CLASS_MATCH_CP(cb, buf[0]))
                    return RULE32_REJECTED;
                break; }

            case RULE_OP_REJ_LAST_CLASS: {
                unsigned char cb = (unsigned char)*rule++;
                if (len > 0) {
                    int a = cl_start(buf, len, cl_count(buf, len) - 1);
                    if (!RULE_CLASS_MATCH_CP(cb, buf[a])) return RULE32_REJECTED;
                }
                break; }

            case RULE_OP_REJ_AT_CLASS: {
                int y = (int)*rule++;
                unsigned char cb = (unsigned char)*rule++;
                if (y >= cl_count(buf, len)) return RULE32_REJECTED;
                if (!RULE_CLASS_MATCH_CP(cb, buf[cl_start(buf, len, y)]))
                    return RULE32_REJECTED;
                break; }

            case RULE_OP_REJ_CNT_CLASS: {
                int y = (int)*rule++, cnt = 0;
                unsigned char cb = (unsigned char)*rule++;
                for (i = 0; i < len; i++)
                    if (RULE_CLASS_MATCH_CP(cb, buf[i])) cnt++;
                if (cnt < y) return RULE32_REJECTED;
                break; }

            default:
                /* All 61 verbs are implemented; this is now unreachable for any
                 * rule the shared compiler produced, and catches corruption. */
                return RULE32_ERR_INVALID;
        }
    }
rule_done:
#undef ROOM

    /* Report the second variant only when one was actually reached. An ASCII
     * word, or any word a case op never touched ambiguously, stays at 1. */
    if (nvariants && amb) *nvariants = 2;

    if (len > outmax) return RULE32_ERR_NOROOM;
    for (i = 0; i < len; i++) out[i] = buf[i];
    return len;
}
