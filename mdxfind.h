/*
 * $Log: mdxfind.h,v $
 * Revision 1.30  2026/09/17 06:11:54  dlr
 * Correct the bf_fast_eligible note: the condition no longer includes an environment variable. MDXFIND_GPU_FAST_DISABLE could veto the BF-fast MD5 template kernel and was removed 2026-09-16.
 *
 * Revision 1.29  2026/09/17 03:22:25  dlr
 * Remove JOBFLAG_IP with the -n i IPv4 append mode. Waffle: its purpose has been served and externally generating IP address ranges is trivial. Bit 8 is free; the three unbuilt unversioned copies mdxnew.c, mdxfindsparc64.c and mdxfindnew.c still reference it and are unaffected by the product build. rotcheck.c carries its own define and does not include this header.
 *
 * Revision 1.28  2026/09/13 12:17:26  dlr
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
 * Revision 1.27  2026/09/12 16:14:23  dlr
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
 * Revision 1.26  2026/09/12 15:00:57  dlr
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
 * Revision 1.25  2026/08/12 01:22:09  dlr
 * Add MYSHA256 streaming workspace struct and mysha256_begin/add/end plus mysha256_cpu_detect prototypes. Caller-owned fixed-size struct carved from the existing per-thread work buffers like any other procjob workspace, so there is no allocation and nothing to free.
 *
 * Revision 1.24  2026/05/19 01:20:01  dlr
 * word-retirement ETA: move struct Linehints to mdxfind.h + add retired_line field + InflightLines global. CPU retirement updates at procjob mid-job checkpoint and job-return site. InflightLines increment at dispatch. Chunk-reset clears retired_line. 15s tick aggregator computes RetiredLines_rate. Display arm uses retirement-based ETA with bootstrap fallback to hash-rate until first tick fires.
 *
 * Revision 1.23  2026/05/11 03:48:16  dlr
 * BF Phase 1.9 A1: struct job +unsigned char bf_fast_eligible after bf_inner_iter (~line 127). Host BF chunk producer populates; procjob short-circuit copies to jobg.
 *
 * Revision 1.22  2026/05/10 21:20:25  dlr
 * BF Phase 1.8: struct job +unsigned int bf_inner_iter at 117-123, adjacent to bf_offset_per_word/bf_num_masks. Main-thread BF chunk producer populates; procjob short-circuit copies to jobg.
 *
 * Revision 1.21  2026/05/10 14:03:58  dlr
 * BF Tranche 3 plumbing: add bf_offset_per_word + bf_num_masks (uint) fields to struct job. Together with existing MaskIndex (carries bf_mask_start) and MaskCount (carries chunk_total), procjob short-circuit translates these to jobg fields at chokepoint entry. Default 0 = not a BF chunk.
 *
 * Revision 1.20  2026/05/10 05:11:16  dlr
 * BF Tranche 1 plumbing: define JOBFLAG_BF_CHUNK (0x80) for the chunk-as-job migration. Set on jobg slots produced by main thread BF chunk producer; procjob short-circuits at chokepoint entry and submits to gpujob queue. Inert until Tranche 3 sets the flag.
 *
 * Revision 1.19  2026/05/04 14:32:59  dlr
 * Add extern decls for tsfprintf() + tsfprintf_pin_summary() — startup-phase diagnostic instrumentation. Implementation in mdxfind.c above malloc_pinned. Used by gpu_opencl.c (forward-declared inline there since it doesn't include mdxfind.h) and gpujob_opencl.c (which does). Always-on with a single static lock for stderr serialization.
 *
 * Revision 1.18  2026/05/03 21:46:54  dlr
 * Widening + cache-poison fix.
 *
 * Widening (forward-looking; Shooter's hashmob.net.found.v7 is 3.16B lines, fits in 32-bit but no headroom):
 * - struct job::startline,numline: unsigned int -> unsigned long long; reordered (8-byte first, 4-byte after); prefix[] aligned(16) to keep buffer alignment after the +8 bytes
 * - struct Linehints::curline,numline: widened + reordered
 * - MDXlowest_line: 'volatile unsigned int = 0xFFFFFFFF' -> 'atomic_ullong = ULLONG_MAX'. C11 atomics auto-handle platform conditionalization (x86-64/ARM64: plain mov/ldr/str; i386: LOCK CMPXCHG8B; ARMv7: LDREXD/STREXD). All 4 read/write sites use atomic_load_explicit/atomic_store_explicit with memory_order_relaxed. lowest_line/restart locals widened to match. Sentinels migrated to ULLONG_MAX.
 * - cacheline()::nextline static + curline local widened (the actual chunk-position counter)
 * - procjob locals (curline,numline,sl) widened
 * - read-loop locals widened
 * - numline = UINT_MAX -> ULLONG_MAX (lineswanted stays 32-bit — per-job target)
 * - struct jobg fully reorganized (8-byte first / 4-byte / 1-byte slot_kind / 4-byte arrays / 2-byte arrays / aligned union last); 'unsigned int line_num' deleted (vestigial — never compared, only assigned)
 * - gpujob_get_free* dropped 'startline' parameter (the parameter was already discarded with '(void)startline;')
 * - 7 dead 'g->line_num = job->startline;' lines deleted
 * - 5 callers updated to drop startline arg
 *
 * Cache-poison fix (root cause of Shooter's wrong-ETA bug 2026-05-03: cache held 12M lines for hashmob.net.found.v7 vs 3.16B actual):
 * - Fix 1: linecount_file() gains 'int *complete' out-param. Default *complete=0; set to 1 only via natural fall-through after read-loop runs to EOF. linecount_thread() now skips cache store when complete=0 — partial counts from HashWaiting early-bail no longer poison future sessions.
 * - Fix 2a: New atomic_ullong CurfileBytesRead/CurfileBytesTotal globals. cacheline() accumulates readlen into CurfileBytesRead per chunk. Main read loop sets CurfileBytesTotal from sb.st_size at file open; resets CurfileBytesRead.
 * - Fix 2b: After 'Fileline += Linecount' in main read loop, if Fileline > AutoCountTotalLines (cache was wrong), project new estimate from byte-position ratio (Fileline * total_bytes / read_bytes) and CAS-loop-bump AutoCountTotalLines monotonically. ReportStats's ETA self-corrects without code change to ReportStats.
 * - Fix 2c: New linecount_cache_finalize(filepath, size, mtime, actual_lines) function. Called after gzclose per wordlist (skipping stdin + Fileline==0). Opens own SQLite connection, INSERT OR REPLACE under WAL+busy_timeout, PRAGMA wal_checkpoint(TRUNCATE), close — overwrites poisoned partial entry with the authoritative count discovered at EOF.
 * - comfort line: 'line N' position now appears in the percentage branch too (not just the no-estimate fallback).
 *
 * Revision 1.17  2026/04/27 21:53:26  dlr
 * GPU rule engine Phase 0 classifier (project_gpu_rule_engine_design.md rev 3, §6). Adds gpu_rule_safe_phase0() — single-stage op-based predicate accepting only Tier-1 ops {l, u, r, :, space, tab} in the post-packrules bytecode — and classify_rules() — partitions a rule array into full / gpu / cpu lists preserving original order. struct rule_lists declared in mdxfind.h alongside applyrule. Verified against synthetic mixed input (7 GPU + 5 CPU partition correct). Empirical note: HashMob.{100,1k,5k,100k}.rule classify as 0% GPU-eligible at Phase 0 — they all use ops beyond l/u/r — so Phase 0 validation will need a synthetic test fixture, not HashMob, to exercise the GPU path.
 *
 * Revision 1.16  2026/04/22 22:02:53  dlr
 * struct rule_workspace and extern applyrule in header
 *
 * Revision 1.15  2026/04/22 18:23:53  dlr
 * Add struct rule_workspace for heap-allocated applyrule buffers
 *
 * Revision 1.14  2026/04/14 04:46:11  dlr
 * GPU brute-force: timing probe, per-chunk dispatch, uint64 mask_start, base-offset decomposition, immediate hit processing, MD5SHA256SHA256 (e996)
 *
 * Revision 1.13  2026/04/05 03:55:52  dlr
 * Include emmintrin.h under NOTINTEL guard, MAXCHUNK 50MB for Apple Silicon (not embedded ARM)
 *
 * Revision 1.12  2026/04/04 18:53:45  dlr
 * Per-algorithm dispatch with linehints: rate-based lineswanted from bench_rates.h, EMA feedback in ReportStats, per-algorithm curline tracking, GPU lineswanted=UINT_MAX for ordering, Lowline from min(curline), struct job reorder + fileno + JOBFLAG_GPU, FAM enum moved to gpujob.h
 *
 * Revision 1.11  2026/03/25 23:11:05  dlr
 * Move Hashchain struct to header
 *
 * Revision 1.10  2026/03/23 02:51:54  dlr
 * Replace -n digit hack with mask-based hybrid attack: -n "?l?d" append, -N prepend, ?[0-9a-f] custom classes
 *
 * Revision 1.9  2025/08/24 22:08:56  dlr
 * changes for atomic
 *
 * Revision 1.8  2025/08/23 22:26:25  dlr
 * Move to new outbuf
 *
 * Revision 1.7  2020/03/11 02:49:29  dlr
 * SSSE modifications complete.  About to start on fastrule
 *
 * Revision 1.6  2017/10/19 03:38:44  dlr
 * Add rule counter
 *
 * Revision 1.5  2017/08/25 05:09:54  dlr
 * minor change for ARM6
 *
 * Revision 1.4  2017/08/25 04:16:03  dlr
 * Porting for ARM/POWERPC.  Fix SQL5
 *
 * Revision 1.3  2017/06/30 13:35:32  dlr
 * fix for ARM
 *
 * Revision 1.2  2017/06/30 13:23:13  dlr
 * Added SVAL
 *
 * Revision 1.1  2017/06/29 14:09:29  dlr
 * Initial revision
 *
 *
 */
#if ARM > 6
#include <arm_neon.h>
#endif
#ifndef NOTINTEL
#include <emmintrin.h>
#endif

#define MAXLINE (40*1024)
struct job {
    /* 8-byte fields grouped first */
    struct job *next;
    char *readbuf,*outbuf,*pass;
    unsigned int *found;
    struct LineInfo *readindex;
    char *filename;
    int *doneprint;
    unsigned long long Numbers;
    unsigned long long MaskIndex;
    unsigned long long MaskCount;
    unsigned long long startline, numline;   /* widened from unsigned int — wordlist line positions can exceed 4.29B */
    /* 4-byte fields */
    int op,len,clen,flags;
    int Ruleindex,digits,outlen,fileno;
    /* BF chunk-as-job (Tranche 3, 2026-05-09): when JOBFLAG_BF_CHUNK is set,
     * MaskIndex carries bf_mask_start (chunk's base cursor in the global
     * keyspace), MaskCount carries chunk_total (candidates in this chunk),
     * and these two fields carry the per-word stride / mask range. The
     * procjob short-circuit translates these into jobg fields at chokepoint
     * entry. Default 0 = not a BF chunk. */
    unsigned int bf_offset_per_word;
    unsigned int bf_num_masks;
    /* BF Phase 1.8 (2026-05-10): kernel inner iteration count for this chunk.
     * 0 or 1 = today's behavior (bit-identical). Cap=16. Set by
     * adaptive_bf_chunk_size servo; procjob short-circuit copies into
     * jobg.bf_inner_iter. Unsalted BF only; servo forces 1 on salted ops. */
    unsigned int bf_inner_iter;
    /* Phase 1.9 Tranche A1 (2026-05-10): when 1, the chunk producer has
     * pre-qualified this BF chunk for the BF-fast MD5 template kernel
     * (gpu_md5_bf.cl). Conditions: op==JOB_MD5, Numrules<=1, unsalted,
     * append-only mask (npre==0, napp in [1,8]).  An
     * MDXFIND_GPU_FAST_DISABLE env var could also veto it; that was
     * removed 2026-09-16.  Procjob short-circuit copies
     * this into jobg.bf_fast_eligible. Default 0 = slow template path.
     * Wider eligibility (multi-rule, prepend, salted) is intentionally
     * out of A1 scope; A2-A4 do not widen this gate. */
    unsigned int bf_fast_eligible;
    /* Buffers — explicit 16-byte alignment for SIMD; widening startline+numline
     * pushed prefix off natural 16-alignment, so we mark it explicitly. */
    char prefix[MAXLINE] __attribute__((aligned(16)));
    char line[MAXLINE+MAXLINE];
};
#define JOBFLAG_PRINT 1
#define JOBFLAG_HEX 2
#define JOBFLAG_NUMBERS 4
#define JOBFLAG_PREPEND 16
#define JOBFLAG_GPU 32
#define JOBFLAG_BRUTEFORCE 64
#define JOBFLAG_BF_CHUNK 128  /* BF chunk-as-job: produced by main thread for procjob short-circuit fill */

union HashU {
    unsigned char h[256];
    uint32_t i[64];
    unsigned long long v[32];
#ifndef NOTINTEL
    __m128i x[16];
#endif
#if ARM > 6
    uint32x4_t x[16];
#endif
#ifdef POWERPC
    vector unsigned int x[16];
#endif
};

struct Hashchain {
    struct Hashchain *next;
    unsigned short int flags, len;
    unsigned char hash[1];
};

/* Per-algorithm dispatch hints (rate, EMA, line tracking, retirement).
 * Defined here so gpujob_opencl.c can update retired_line on GPU completion. */
struct Linehints {
    /* 8-byte fields */
    long long rate;
    volatile long long hashes_accum;  /* per-algorithm hash counter for EMA feedback */
    unsigned long long curline, numline;  /* widened from unsigned int -- wordlist line positions can exceed 4.29B */
    volatile unsigned long long retired_line;  /* word-retirement ETA: monotonic per-op, reset at chunk start */
    /* 4-byte fields */
    unsigned int lineswanted, gpu;
};

#ifdef ARM
union sse_value {
#if ARM > 6
    uint32x4_t sse;
#else
    uint64_t sse,sse1;
#endif
    uint64_t longs[2];
    uint32_t words[4];
    uint8_t raw8[16];
} __attribute__((aligned(16)));
typedef union sse_value SVAL;
#endif
#ifdef POWERPC
union sse_value {
   vector unsigned int sse;
    uint64_t longs[2];
    uint32_t words[4];
    uint8_t raw8[16];
} __attribute__((aligned(16)));
typedef union sse_value SVAL;
#endif

#ifdef SPARC
union sse_value {
   uint64_t sse,sse1;
    uint64_t longs[2];
    uint32_t words[4];
    uint8_t raw8[16];
} __attribute__((aligned(16)));
typedef union sse_value SVAL;
#endif

#ifndef NOTINTEL
union sse_value {
    __m128i sse;
    uint64_t longs[2];
    uint32_t words[4];
    uint8_t raw8[16];
} __attribute__((aligned(16)));
typedef union sse_value SVAL;
#endif

/* Rule processing workspace — defined here before any includes,
 * so it's available for both mdxfind and procrule builds. */
#define RULE_WORKSPACE_SIZE ((40*1024) + 16)
struct rule_workspace {
    char Memory[RULE_WORKSPACE_SIZE];
    char Base64buf[RULE_WORKSPACE_SIZE];
};

extern int applyrule(char *line, char *pass, int len, char *rule, struct rule_workspace *ws);
extern int packrules(char *line);
/* As packrules, but yields the true packed length (excluding the trailing
 * NUL).  Required by any caller that copies or keys on the bytecode: a
 * packed rule may contain a 0x00 OPERAND, so strlen() truncates it. */
extern int packrules_len(char *line, int *packedlen);

/* Rule store -- length-carrying, input-order, hash-deduplicated.  Replaces the
 * JudySL rule arrays, which keyed on a NUL-terminated string and so could not
 * hold a packed rule containing a 0x00 operand, and which iterated in
 * bytecode order rather than input-file order. */
/* srcoff/srclen locate this rule's ORIGINAL SOURCE TEXT in rulestore.src.
 * packrules compiles in place (s = d = line) and destroys the source, so the
 * text must be captured BEFORE packing and carried here -- the packed bytecode
 * cannot be decompiled back to a rule line, and the -Z decoder that tried to do
 * so read out of bounds.  srclen == 0 means no source was supplied. */
struct rule_ent { uint32_t off; unsigned short len;
                  uint32_t srcoff; unsigned short srclen; };
struct rulestore {
    char            *slab;     /* packed bytecode, back to back          */
    size_t           used, cap;
    struct rule_ent *ent;      /* linear -- index order IS input order   */
    int              n, entcap;
    uint32_t        *htab;     /* slot+1 into ent; 0 = empty             */
    size_t           hmask;
    char            *src;      /* rule SOURCE TEXT, back to back          */
    size_t           srcused, srccap;
};
extern int  rs_add(struct rulestore *rs, const char *bytes, int len);
/* As rs_add, but also retains the rule's source text for -Z.  rs_add is this
 * with src == NULL, so existing callers (procrule) are unaffected. */
extern int  rs_add_src(struct rulestore *rs, const char *bytes, int len,
                       const char *src, int srclen);
extern void rs_reset(struct rulestore *rs);
extern int  rs_product(struct rulestore *dst, const struct rulestore *mul,
                       int maxlen);

/* GPU rule engine — three-list partition (Phase 0 design memo).
 * `full` is the original (caller-owned) array; gpu and cpu are owned
 * pointer arrays into the same packed-rule strings — no copy. */
struct rule_lists {
    char **full;     int nfull;
    char **gpu;      int ngpu;
    char **cpu;      int ncpu;
    /* Parallel length arrays.  The packed bytecode is not a C string -- an
     * operand byte may be 0x00 -- so a length must travel with every rule.
     * fulllen[] is an alias of the caller's array; gpulen[]/cpulen[] are
     * malloc'd alongside gpu[]/cpu[].  gpuidx[]/cpuidx[] give each entry's
     * ORIGINAL index, so no caller needs to search for it. */
    const unsigned short *fulllen;
    unsigned short *gpulen;
    unsigned short *cpulen;
    int            *gpuidx;
    int            *cpuidx;
};

extern int classify_rules(char **rules, const unsigned short *lens,
                          int nrules, struct rule_lists *out);
extern void rule_lists_free(struct rule_lists *rl);

/* Startup-phase diagnostic instrumentation (Shooter 12-GPU rig).
 * Always-on. Prefixes a "[T+ S.SSSs] " stamp to the formatted message,
 * thread-safe via a private lock so the prefix and message stay on a
 * single line under multi-thread emission. tsfprintf_pin_summary()
 * emits a consolidated end-of-init pin tally (per-(reason, size_class)
 * malloc_pinned() outcomes). MDXFIND_PIN_TRACE=1 in the environment
 * additionally enables per-attempt tsfprintf() lines. Defined in
 * mdxfind.c above malloc_pinned. */
#include <stdio.h>
extern void tsfprintf(FILE *fp, const char *fmt, ...)
    __attribute__((format(printf, 2, 3)));
extern void tsfprintf_pin_summary(void);

#define BCRYPT_HASHSIZE 64
#define MAXVECSIZE 2000000  /* Maximum test vector size */

#define MAXTHREADS 8

#define LDAP_MAX_UTF8_LEN  ( sizeof(wchar_t) * 3/2 )
#define FLOOR_LOG2(x) (31 - __builtin_clz((x) | 1))
static inline int log2i(uint64_t n) {
#define S(k) if (n >= ((uint64_t)1 << k)) { i += k; n >>= k; }
    int i = -(n == 0); S(32); S(16); S(8); S(4); S(2); S(1); return i;
#undef S
}

/* MAXCHUNK sets the maximum amount of memory used for each chunk.
   As I write this, typical hard drive speeds are 100 Mbytes/sec, so
   100M represents about 1 seconds of data.  Increase as appropriate.
*/
#if defined(ARM) && !defined(MACOSX)
/* INPUTCHUNK - maximum number of hashes to process at once from stdin */
#define INPUTCHUNK (100000)
#define MAXCHUNK (5*1024*1024)
#else
/* INPUTCHUNK - maximum number of hashes to process at once from stdin */
#define INPUTCHUNK (10000000)
#define MAXCHUNK (50*1024*1024)
#endif

#define MAXLINEPERCHUNK (MAXCHUNK/2/8)

#define MAXLJOB (32)

/* word-retirement ETA: linehints array (indexed by op) and inflight counter */
extern struct Linehints *linehints;
extern int linehints_count;
extern volatile unsigned long long InflightLines;


/* Streaming SHA-256 workspace.
 *
 * Caller-owned fixed-size struct, carved from the existing per-thread work
 * buffers like any other procjob workspace -- no allocation, no teardown,
 * nothing to free. Thread-safe because every mutable byte is the caller's;
 * the only shared state is the dispatch pointer set once by arm_ce_detect()
 * before any worker thread starts. Implementation and the per-processor
 * gating both live in mymd5.c with the rest of the primitives.
 *
 * mysha256() (one-shot) remains the right interface for fixed-size buffers.
 * These three exist for types that must feed a digest incrementally and
 * previously had to escape to sph_sha256 or OpenSSL EVP to do it. */
typedef struct {
  unsigned int  h[8];
  unsigned long long bits;
  unsigned int  nbuf;
  unsigned char buf[64];
} MYSHA256;

extern void mysha256_begin(MYSHA256 *s);
extern void mysha256_add(MYSHA256 *s, const void *p, size_t len);
extern void mysha256_end(MYSHA256 *s, unsigned char *out);
extern void mysha256_cpu_detect(void);
