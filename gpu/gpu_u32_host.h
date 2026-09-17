/*
 * $Revision: 1.1 $
 * $Log: gpu_u32_host.h,v $
 * Revision 1.1  2026/09/14 19:30:05  dlr
 * Initial revision.
 *
 *
 */
/* gpu_u32_host.h -- host side of the UTF-32 rule path.  HEADER-ONLY.
 *
 * Included by gpu/gpu_opencl.c (and, when the Metal kernel bodies land, by
 * gpu/gpu_metal.m).  Header-only and all-static ON PURPOSE: a separate .c
 * would need a rule added to the Makefile of every build host, and each of
 * those Makefiles is that host's only copy under its own local RCS.  Nothing
 * here needs its own translation unit, so nothing here gets one.
 *
 * It also needs no edit to mdxfind.c.  Everything it reads already exists
 * there as a global, and the UTF-32 rule stream is built HERE, lazily, from
 * the rule SOURCE TEXT that mdxfind already retains per rule.
 *
 * ---- WHAT IT DOES ------------------------------------------------------
 *
 *  1. gpu_u32_build_program()  -- appends the packrule32 uint32 streams to the
 *     byte program and widens the offset table to 2 * n_rules, carrying the
 *     per-rule capability in the SECOND half's high bits.
 *  2. gpu_u32_tag_word_offsets() -- classifies each packed word with
 *     classify_utf8() and ORs the 2-bit class into word_offset bits 30-31.
 *  3. gpu_u32_partition_word_offsets() -- stable partition of one batch's
 *     offset array by class, for the lane-partition variants.
 *  4. gpu_u32_active() -- one predicate, no configuration.
 *
 * ---- WHY THE WORD CLASS IS RE-DERIVED HERE -----------------------------
 *
 * The obvious wiring is for mdxfind.c to OR the class in at the pack site,
 * where `struct LineInfo`'s `enc` field is already to hand.  This does NOT do
 * that, and the reason is worth stating: the packed word IS the byte string
 * mdxfind classified -- classify_utf8() runs after the $HEX[] unwrap and the
 * pack site writes exactly those bytes -- so re-deriving the class from the
 * packed buffer gives the same answer from the same function, needs no edit to
 * a file another agent is actively editing, and gives a free cross-check: if
 * the two ever disagree, one of them is wrong about what got packed.
 *
 * The cost is one classify_utf8() pass over at most
 * GPU_RULES_MAX_WORDS_PER_BATCH = 16384 words per dispatch -- roughly 160 KB
 * of byte scanning against a dispatch of up to 1.6 billion lanes.  It is
 * per-BATCH, not per-lane.
 */

#ifndef GPU_U32_HOST_H
#define GPU_U32_HOST_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include "../ruleproc32.h"
#include "../classify_utf8.h"
#include "../gpujob.h"

/* C linkage, and it is not optional.  gpu_metal.m is compiled as Objective-C++
 * on dev1 (gpujob_metal.m is plain Objective-C, which is why only one of the
 * two showed it), so without this the Metal backend's calls mangle and the
 * link fails with four undefined symbols and clang's own hint:
 *   "found '_gpu_u32_active' in gpu_u32_host.o, declaration possibly missing
 *    extern \"C\"".
 * Wrapping the whole file rather than the declarations alone is deliberate:
 * if the implementation TU is ever built by a C++ compiler the DEFINITIONS
 * need the same linkage, and a half-wrapped header would then fail the same
 * way in the other direction. */
#ifdef __cplusplus
extern "C" {
#endif

/* gpujob.h's whole body is gated on a GPU backend being configured
 * ((__APPLE__ && METAL_GPU) || CUDA_GPU || OPENCL_GPU), so an out-of-band
 * compile -- gpu_u32_wire_hosttest.c, a standalone syntax check -- sees none
 * of its constants.  Fallbacks here, in the same shape the kernels use for
 * RULE_BUF_MAX, so those compiles work without pretending a backend is
 * configured.  A real build always has the gpujob.h values. */
#ifndef GPU_RULES_MAX_WORDS_PER_BATCH
#define GPU_RULES_MAX_WORDS_PER_BATCH 16384
#endif
#ifndef GPU_RULES_WALKER_BUF_ELEMS
#define GPU_RULES_WALKER_BUF_ELEMS 2048
#endif

/* ---- the wire format, mirrored from gpu_md5_rules32.cl ----
 * If you change one, change the other; there is no header shared with the
 * kernel because the kernel source is a string literal at runtime. */
#define GPU_U32_R_HASFORM 0x80000000u
#define GPU_U32_R_BYTEOK  0x40000000u
#define GPU_U32_R_OFFMASK 0x3fffffffu

#define GPU_U32_CLASS_A   0u   /* ASCII or empty -- decodable, not wide */
#define GPU_U32_CLASS_W   1u   /* HI|UTF8        -- decodable, wide     */
#define GPU_U32_CLASS_I   2u   /* HI alone       -- NOT decodable       */
#define GPU_U32_W_SHIFT   30
#define GPU_U32_W_OFFMASK 0x3fffffffu

/* ---- globals owned by mdxfind.c, read-only here ----
 * Declared rather than included because mdxfind.h does not export them and
 * this header must not force a dependency on mdxfind.c's internals beyond
 * these seven names.  Each is cited so a rename is findable. */
extern int            Utf32Rules;      /* mdxfind.c, `case '8':` sets it     */
extern unsigned char *Rule_u32ok;      /* UTF-32 compiler accepts rule i     */
extern unsigned char *Rule_byteok;     /* byte compiler accepts rule i       */
extern char          *Rule_u32src;     /* retained rule SOURCE text          */
extern uint32_t      *Rule_u32srcoff;  /* 1-based, indexed by Ruleindex      */
extern unsigned int   Numrules;
extern int           *gpu_rule_slot;   /* gpu rule idx -> 1-BASED Rules[]
                                        * slot; 0 = the synthetic `:` pass   */
extern int            gpu_legacy_slot_unused; /* 1 = rl.ncpu == 0 after the
                                        * n_u32only correction; gates the
                                        * `currule = NULL` short-circuit that
                                        * $Log 1563 measured at ~half the wall
                                        * time.  Read here only to REPORT it,
                                        * because a flag worth half the wall
                                        * time should be asserted and not
                                        * inferred from a timing. */
extern unsigned char *gpu_rule_membership; /* 1 = rule i is device-walked,
                                        * 0-based by Rules[] order.  WRITTEN
                                        * here for the tier-3 clawback below */

/* ==== the module's interface =========================================
 *
 * THE BODIES ARE NOT IN THIS HEADER.  They live in gpu_u32_host.c, which is
 * this header with GPU_U32_HOST_IMPLEMENTATION defined -- the single-header
 * pattern -- and that file is compiled into EVERY configuration.
 *
 * WHY IT MOVED.  The bodies used to be `static` here with one exported wrapper
 * in gpu_opencl.c, co-located with the lazy so writer and reader shared one
 * static cache by construction.  That reasoning was right and it stopped being
 * sufficient the moment mdxfind.c called the pre-filter from its METAL block
 * too: gpu_opencl.c is not compiled on a METAL_GPU build, so the call was an
 * undefined symbol at link and NOTHING had linked a Metal-only build since.
 *
 * One backend-neutral TU is strictly better than the co-location it replaces.
 * The cache is still one instance -- there is only one TU that can hold it --
 * and now BOTH backends' host files read the same one, which is the property
 * the OpenCL-only version had by accident of there being one backend.
 */
/* Declared by the BACKEND at init: "I have the UTF-32 kernel bodies, not just
 * the generated walker."  Default is 0, so a backend that carries the walker
 * and nothing else cannot be mistaken for capable merely because it compiles.
 * gpu_u32_active() requires it. */
void     gpu_u32_set_backend_capable(int yes);

int      gpu_u32_active(void);

/* Hand the caller the built program instead of exposing the cache.  Returns 1
 * if it has been built, 0 otherwise.  The cache is private to
 * gpu_u32_host.c ON PURPOSE: there is exactly one copy, and a backend that
 * reached into it would be able to disagree with the one that filled it. */
int      gpu_u32_get_program(const unsigned char **prog, uint32_t *len,
                             const uint32_t **offs, int *n_rules);

int      gpu_u32_build_program(
             const unsigned char *byte_prog, uint32_t byte_len,
             const uint32_t *byte_offs, int n_rules,
             unsigned char **out_prog, uint32_t *out_len, uint32_t **out_offs,
             int *n_u32_out, int *n_u32only_out);

/* Build the device UTF-32 stream EARLY and clear gpu_rule_membership[] for any
 * rule the device walker cannot run.  Call from each rule-pack block AFTER the
 * membership fill loop and before any word is read; see the comment on the
 * definition for what goes wrong at any other point.  Returns the number of
 * rules whose membership it cleared, or -1 if it could not apply. */
int      gpu_u32_membership_prefilter(const unsigned char *byte_prog,
             uint32_t byte_len, const uint32_t *byte_offs, int n_rules);

void     gpu_u32_tag_word_offsets(uint32_t *woff, uint32_t n_words,
             const unsigned char *packed, uint32_t counts[3]);
uint32_t gpu_u32_partition_word_offsets(uint32_t *woff, uint32_t n_words,
             uint32_t *scratch);

/* Returned when the DEVICE used the byte engine for this pair, so the caller
 * must run applyrule() itself.  See the definition for why the byte arm is not
 * called from inside this module. */
#define GPU_U32_REPLAY_BYTE (-100)
int      gpu_u32_replay_apply(const char *word, int wlen, int ridx,
             char *out, int outmax);

/* The device's own program and offset table, published by the builder so the
 * hit replay reproduces the DEVICE's engine choice rather than recomputing it. */
extern const unsigned char *gpu_u32_dev_prog;
extern const uint32_t      *gpu_u32_dev_offs;
extern int                  gpu_u32_dev_nrules;

#ifdef GPU_U32_HOST_IMPLEMENTATION

/* Set by the backend at init.  0 until a backend claims it, so a backend that
 * carries the generated walker but not the kernel bodies -- Metal, today --
 * cannot be treated as capable merely because the walker compiles. */
static int gpu_u32_backend_capable = 0;

void gpu_u32_set_backend_capable(int yes) { gpu_u32_backend_capable = yes ? 1 : 0; }

int gpu_u32_active(void)
{
    if (!Utf32Rules) return 0;               /* no -8: nothing to do        */
    if (!gpu_u32_backend_capable) return 0;  /* this backend cannot run it  */
    /* Degenerate: a rule file whose every line is `:`.  The store drops them,
     * Numrules is 0, there is no rule to apply -- so no engine choice exists
     * and the byte walker's synthetic pass is the complete answer. */
    if (!Rule_u32src && !Rule_u32srcoff && Numrules == 0) return 0;
    return 1;
}

/* ==== IS THE UTF-32 DEVICE PATH ACTIVE? ==============================
 *
 * NO ENVIRONMENT VARIABLES.  `MDXFIND_CACHE` is the only environment input
 * mdxfind takes, because a tool whose behaviour changes from the environment
 * changes it invisibly.  Four knobs existed here while the design was being
 * chosen -- MDXFIND_GPU_U32_MODE, _CONV, _BUF_ELEMS and _PROBE -- and every
 * one of them now has an answer, so there is nothing left to select:
 *
 *   MODE      1 is the answer.  2 was WITHDRAWN by measurement (14,687 against
 *             the CPU's 15,950 with the sets DISJOINT: the partition is applied
 *             to the payload copy while the hit replay reads the caller's
 *             array).  3 and 4 were never built.
 *   CONV      C1, decode in the lane.  C2's ceiling was measured at ~10% (the
 *             in-lane decode is 8.5-10.2% of the pair; the walk is the rest)
 *             against a 4x wider staging read, an extra dispatch and 128 MB of
 *             VRAM per batch.  The pre-pass kernel is deleted, not disabled.
 *   BUF_ELEMS 2048, and it is now a compile-time constant injected from the one
 *             -D funnel.  It is the knob with a real consequence -- 6.0-6.5x
 *             private memory -- which is precisely why it should not be a
 *             runtime dial that changes what a binary does.
 *   PROBE     a diagnostic.  It earned its keep by finding the no-rule dedup
 *             bug and it belongs in the test harness, not in a shipped
 *             binary's environment.
 *
 * REMOVING THE MODE KNOB ALSO REMOVES A CLASS OF WRONG ANSWER, which is the
 * better half of this.  MODE=0 was the DEFAULT, so `-8` on a GPU build ran the
 * device BYTE walker unless the operator opted in -- measured at 2 recovered
 * against the CPU's 15,950 on semantics.md5.  With no knob, `-8` either
 * engages the UTF-32 path or the backend cannot, and the second case is a
 * visible capability gap rather than a quiet default.
 *
 * So the predicate is now three facts and no configuration:
 *   -8 was given, AND there are rules with retained source, AND this backend
 *   can actually run the UTF-32 walker.
 *
 * The third is set by the backend at init -- gpu_u32_set_backend_capable() --
 * because it is the one thing this module cannot know about itself.  A backend
 * that has the walker but not the kernel bodies must NOT claim it.
 */
/* ==== 1. build the combined rule program ==============================
 *
 * Input is the byte program and offset table that the existing
 * classify_rules + pack block already produced.  Output is a NEW program with
 * the uint32 packrule32 streams appended after a 4-byte alignment pad, and a
 * NEW offset table of 2 * n_rules entries.
 *
 * `n_rules` INCLUDES the synthetic `:` no-rule pass, which is the last entry
 * and whose gpu_rule_slot[] is 0.  It gets no UTF-32 form: it is the no-rule
 * pass, its byte program is a single NUL, and the byte walker returns the
 * input unchanged, which is what the CPU does for it too.
 *
 * Returns 0 on success.  On any failure it returns -1 having freed nothing
 * that it did not allocate -- the caller keeps the byte-only program and the
 * run proceeds with the byte walker, which is a correct configuration.
 */
int gpu_u32_build_program(
    const unsigned char *byte_prog, uint32_t byte_len,
    const uint32_t *byte_offs, int n_rules,
    unsigned char **out_prog, uint32_t *out_len, uint32_t **out_offs,
    int *n_u32_out, int *n_u32only_out)
{
    uint32_t pad, base, pos, cap;
    unsigned char *prog = NULL;
    uint32_t *offs = NULL;
    uint32_t *stream = NULL, *src32 = NULL;
    int i, n_u32 = 0, n_u32only = 0;

    if (!byte_prog || !byte_offs || n_rules <= 0 || !out_prog || !out_len ||
        !out_offs) return -1;
    if (!Rule_u32src || !Rule_u32srcoff || !Rule_u32ok || !gpu_rule_slot) {
        fprintf(stderr,
            "OpenCL: UTF-32 rule path requested but mdxfind retained no rule "
            "source text (Rule_u32src=%p Rule_u32srcoff=%p Rule_u32ok=%p "
            "gpu_rule_slot=%p) -- staying on the byte walker\n",
            (void *)Rule_u32src, (void *)Rule_u32srcoff,
            (void *)Rule_u32ok, (void *)gpu_rule_slot);
        return -1;
    }

    /* 4-align the UTF-32 region: the kernel casts a uchar* into a uint*, and
     * an unaligned cast is undefined on every backend and faults on some. */
    pad  = (4u - (byte_len & 3u)) & 3u;
    base = byte_len + pad;

    /* Capacity: each rule's stream is at most a few words per source
     * character, and packrule32's own output bound is the operand count.
     * 8 words per source byte plus a terminator is generous and is checked
     * per rule below anyway. */
    cap = base;
    for (i = 0; i < n_rules; i++) {
        int slot = gpu_rule_slot[i];
        if (slot >= 1 && slot <= (int)Numrules && Rule_u32ok[slot - 1]) {
            const char *src = Rule_u32src + Rule_u32srcoff[slot];
            cap += (uint32_t)(strlen(src) + 2) * 8u * 4u;
        }
    }
    if (cap > (GPU_U32_R_OFFMASK + 1u)) {
        fprintf(stderr,
            "OpenCL: UTF-32 rule program would be %u bytes, past the 2^30 "
            "the offset field can address -- staying on the byte walker\n", cap);
        return -1;
    }

    prog   = (unsigned char *)malloc(cap);
    offs   = (uint32_t *)malloc((size_t)n_rules * 2u * sizeof(uint32_t));
    stream = (uint32_t *)malloc(RULE32_MAXCP * sizeof(uint32_t));
    src32  = (uint32_t *)malloc((MAXLINE + 16) * sizeof(uint32_t));
    if (!prog || !offs || !stream || !src32) {
        free(prog); free(offs); free(stream); free(src32);
        fprintf(stderr, "OpenCL: malloc failed building the UTF-32 rule "
                        "program -- staying on the byte walker\n");
        return -1;
    }

    memcpy(prog, byte_prog, byte_len);
    for (i = 0; i < (int)pad; i++) prog[byte_len + i] = 0;
    pos = base;

    for (i = 0; i < n_rules; i++) {
        int slot = gpu_rule_slot[i];
        int bok  = 1, nw, k;
        const char *src;

        offs[i] = byte_offs[i];          /* first half: UNTAGGED, unchanged */
        offs[n_rules + i] = 0;           /* no form, no byteok -- filled below */

        if (slot < 1 || slot > (int)Numrules) {
            /* The synthetic `:` no-rule pass.  Byte engine, always. */
            offs[n_rules + i] = GPU_U32_R_BYTEOK;
            continue;
        }
        bok = Rule_byteok ? (Rule_byteok[slot - 1] != 0) : 1;
        if (bok) offs[n_rules + i] |= GPU_U32_R_BYTEOK;
        else     n_u32only++;            /* the byte compiler refused it:
                                          * a U32-REQUIRED rule, admitted to
                                          * Rules[] with the 0x00 placeholder */

        if (!Rule_u32ok[slot - 1]) continue;   /* BYTE-ONLY rule: no form */

        src = Rule_u32src + Rule_u32srcoff[slot];
        nw = utf8_to_utf32((const unsigned char *)src, (int)strlen(src),
                           src32, MAXLINE + 16);
        if (nw < 0) continue;            /* rule text not UTF-8: no form */
        nw = packrule32(src32, nw, stream, RULE32_MAXCP);
        if (nw < 0) continue;            /* compiler refused: no form */

        /* THE LENGTH IS packrule32's RETURN VALUE.  Do not scan for
         * RULE32_END.
         *
         * RULE32_END is 0 and an OPERAND is very often 0 -- every rule with a
         * position operand of 0 has one: `+0`, `D0`, `L0`, `o0Q`, `*01`.  A
         * scan stops at the operand, truncates the stream before its real
         * terminator, and the device walker then runs off the end of the rule
         * into the NEXT rule's words.  Measured on encoding-rules/same.rule x
         * words.txt: 56 wrong candidates out of 504 pairs, and `*01` came back
         * as a rejection.
         *
         * This is the SAME defect class the byte program block in mdxfind.c
         * already carries a comment about -- "strlen would stop at a 0x00
         * operand and hand the device a truncated rule whose operand count
         * still said N, which the kernel would then satisfy from the next
         * rule's bytes".  It was reproduced here in the uint32 domain by a
         * scan that looked obviously safe.  packrule32_pass ends
         * `EMIT(RULE32_END); return n;` so the count it returns INCLUDES the
         * terminator and is the only authority.
         *
         * The sibling walker test cannot catch this: it hands apply_rule32 the
         * whole packrule32 output buffer and never computes a length. Only the
         * wire test, which builds the program the way the device receives it,
         * exercises it. */
        k = nw;

        /* Refuse what the DEVICE walker does not implement -- the twelve
         * string-operand opcodes.  Measured: 4 occurrences in 338,310 real
         * rule lines, so this costs ~0.001% of real-rule coverage and buys a
         * walker with no variable-length operands except the two affix runs.
         * A rule refused here keeps HASFORM clear and its pairs are walked on
         * whichever engine pick_engine chooses, exactly as if the UTF-32
         * compiler had refused it. */
        {
            /* EVERY word is examined, with NO opcode/operand skipping.
             *
             * The obvious form walks the stream properly, skipping each
             * opcode's operands -- and that form has a false-ACCEPT path: if
             * the walk ever desynchronises (an operand whose VALUE is 0x1000,
             * and its `2 + count` skip then jumps over a real string opcode)
             * the rule is admitted and the device walker hits its
             * `default: return U32_ERR_INVALID` on a rule that should have
             * been refused.  A false accept is a wrong ANSWER.
             *
             * Scanning every word cannot miss one.  Its only error is the
             * other direction: a rule whose OPERAND is a codepoint in
             * U+1002..U+100C (Tibetan) is refused although it uses no string
             * op.  That costs coverage -- the pair falls back to whatever
             * engine pick_engine chooses -- and never costs correctness.
             * Coverage is the cheap currency here and a wrong answer is not.
             *
             * It also needs no operand-count table, which would be a FOURTH
             * copy of one (applyrule32, the device walker, and the host-test
             * harness already carry it). */
            int bad = 0, j;
            for (j = 0; j < k; j++)
                if (stream[j] >= 0x1002u && stream[j] <= 0x100Cu) { bad = 1; break; }
            if (bad) continue;
        }

        if (pos + (uint32_t)k * 4u > cap) {
            fprintf(stderr,
                "OpenCL: UTF-32 rule program capacity exceeded at rule %d "
                "(pos=%u need=%u cap=%u) -- staying on the byte walker\n",
                i, pos, (uint32_t)k * 4u, cap);
            free(prog); free(offs); free(stream); free(src32);
            return -1;
        }
        memcpy(prog + pos, stream, (size_t)k * 4u);
        offs[n_rules + i] |= GPU_U32_R_HASFORM | (pos & GPU_U32_R_OFFMASK);
        pos += (uint32_t)k * 4u;
        n_u32++;
    }

    free(stream); free(src32);
    *out_prog = prog;
    *out_len  = pos;
    *out_offs = offs;
    if (n_u32_out)      *n_u32_out = n_u32;
    if (n_u32only_out)  *n_u32only_out  = n_u32only;
    return 0;
}

/* ==== 1b. THE MEMBERSHIP PRE-FILTER, and WHY it cannot wait ============
 *
 * MEASURED BUG, GTX 1080, 2026-09-14.  With a rule file mixing 27
 * byte-eligible rules and the 14 U32-only rules of
 * tools/ruletests/encoding-rules/u32-ops.rule, `syntax.md5` recovered
 * **9,011 against the CPU's 13,435** -- 4,424 lost, about a third, and 6 of
 * the 14 is about a third.
 *
 * Cause: `gpu_u32_build_program` clears `gpu_rule_membership[]` for the six
 * tier-3 string-operand rules it cannot walk (the clawback below), but it runs
 * at FIRST DISPATCH.  A rules batch holds up to
 * GPU_RULES_MAX_WORDS_PER_BATCH = 16384 words, so on a 2,009-word list the
 * first dispatch is the END-OF-JOB FLUSH -- every word has already been walked
 * by then, each one consulting membership at the C2.3 gate and skipping those
 * six rules because membership still said 1.  The device skipped them too
 * (neither capability bit), so the pairs were walked by nobody.
 *
 * Clearing membership is therefore useless unless it happens BEFORE the first
 * word is walked.  This function does the whole build early, caches it, and
 * the dispatch-time lazy reuses the cached result rather than repeating it.
 *
 * It must be called from mdxfind.c immediately after `gpu_rule_slot = slot;`
 * in each of the two rule-pack blocks -- that is the first moment the
 * gpu-index -> Rules[] map exists, and it is still before any word is read.
 * ONE LINE each.  It is a no-op unless the UTF-32 path is active.
 *
 * Returns the number of rules whose membership it cleared.
 */
int gpu_u32_build_program(
    const unsigned char *byte_prog, uint32_t byte_len,
    const uint32_t *byte_offs, int n_rules,
    unsigned char **out_prog, uint32_t *out_len, uint32_t **out_offs,
    int *n_u32_out, int *n_u32only_out);

/* The one cached build, shared by the pre-filter and the dispatch-time lazy. */
static unsigned char *gpu_u32_cached_prog = NULL;
static uint32_t       gpu_u32_cached_len  = 0;
static uint32_t      *gpu_u32_cached_offs = NULL;
static int            gpu_u32_cached_n    = 0;
static int            gpu_u32_cached_done = 0;

/* The pre-filter's implementation.  STATIC, and header-only like everything
 * else here.
 *
 * The EXPORTED `gpu_u32_membership_prefilter` that mdxfind.c calls is a
 * one-line wrapper in gpu_opencl.c, declared in gpu_opencl.h.  It has to be
 * that way round for two reasons, and the second one is not obvious:
 *
 *  1. LINKAGE.  This header is included by gpu_opencl.c AND gpujob_opencl.c.
 *     The `#define GPU_U32_HOST_H` guard stops double inclusion within a TU
 *     and does nothing across TUs, so a NON-static function definition here is
 *     one definition per including TU -- a duplicate symbol at link.  It had
 *     never been compiled when this was found (`nm` showed the symbol absent
 *     from all three object files), so "it links today" was not evidence of
 *     anything.
 *
 *  2. THE CACHE MUST HAVE EXACTLY ONE INSTANCE.  gpu_u32_cached_* are static,
 *     so each including TU gets its own copy.  The pre-filter WRITES the cache
 *     and gpu_opencl_u32_rules_lazy READS it; if the exported entry point lived
 *     anywhere but gpu_opencl.c, those two would be looking at different
 *     caches, the lazy would miss the early build, and it would silently do the
 *     packrule32 pass a second time -- losing the whole point, which is that
 *     the membership clawback has to happen before the first word is walked.
 *     Putting the wrapper in gpu_opencl.c makes them the same cache by
 *     construction.
 *
 * Idempotent on `gpu_u32_cached_done`, so the pack-time call and the
 * first-dispatch lazy cannot disagree: whichever runs first builds, the other
 * reuses. */
int gpu_u32_membership_prefilter(const unsigned char *byte_prog,
                                  uint32_t byte_len,
                                  const uint32_t *byte_offs, int n_rules)
{
    int n_u32 = 0, n_u32only = 0, cleared = 0, i;

    /* THE ONE MESSAGE THAT STAYS REAL.  With the mode knob gone there is no
     * silent byte-semantics DEFAULT left to warn about -- but `-8` on a build
     * whose backend cannot run the UTF-32 walker is a genuine capability gap
     * and must say so once, rather than quietly producing the byte answer.
     * MEASURED, so the number is not rhetorical: on semantics.md5 the byte
     * answer is 2 recovered where the UTF-32 answer is 15,950. */
    if (Utf32Rules && !gpu_u32_backend_capable && Numrules > 0) {
        static int said = 0;
        if (!said) {
            said = 1;
            fprintf(stderr,
                "-8: this GPU backend cannot run the UTF-32 rule walker, so the "
                "device is applying BYTE rule semantics and -8 reaches only the "
                "CPU-walked residue. Results will differ from `-G none`, which "
                "gives the UTF-32 answer.\n");
        }
        return 0;
    }
    if (!gpu_u32_active()) return 0;
    if (!byte_prog || !byte_offs || n_rules <= 0) return 0;
    /* Already built: skip the build, still (re)apply the clawback. */
    if (gpu_u32_cached_done) goto apply_clawback;

    if (gpu_u32_build_program(byte_prog, byte_len, byte_offs, n_rules,
                              &gpu_u32_cached_prog, &gpu_u32_cached_len,
                              &gpu_u32_cached_offs, &n_u32, &n_u32only) != 0)
        return 0;                 /* builder already explained itself */
    gpu_u32_cached_n    = n_rules;
    gpu_u32_cached_done = 1;

apply_clawback:
    /* ---- TIER-3 MEMBERSHIP CLAWBACK -----------------------------------
     *
     * A rule can be admitted to the device list and then turn out to have NO
     * UTF-32 form -- the twelve RULE32_OP_*_STR string-operand opcodes are
     * refused by the builder, as is a rule whose own text does not decode.
     * For a rule that ALSO has no byte form (a U32-only rule) that leaves
     * BOTH capability bits clear: u32_pick_engine returns -1 for every word
     * class so the device emits nothing, while gpu_rule_membership[] tells
     * the CPU walker it need not walk it either.  **The rule then produces
     * nothing at all, silently.**
     *
     * Applied HERE rather than inside the builder, and re-applied on every
     * call rather than once, because of an ordering trap that has now bitten
     * twice.  The builder runs once and caches; membership is allocated by
     * mdxfind.c AFTER the first sensible call site.  With the clawback inside
     * the builder it saw `gpu_rule_membership == NULL`, skipped silently, and
     * the mixed fixture stayed at 9,011 against the CPU's 13,435 -- the
     * pre-filter reported success and changed nothing.
     *
     * So: the build is cached and the clawback is not.  Call this as many
     * times as you like; the expensive half happens once. */
    if (!gpu_rule_membership) {
        /* LOUD, because this is the failure that hides.  Not fatal: the
         * caller may legitimately call early and again later. */
        static int warned = 0;
        if (!warned) {
            warned = 1;
            fprintf(stderr,
                "GPU rule engine: WARNING gpu_u32_membership_prefilter() ran "
                "before gpu_rule_membership was allocated, so the tier-3 "
                "clawback could not be applied. Call it AFTER the "
                "`gpu_rule_membership[slot[i] - 1] = 1;` fill loop and still "
                "before any word is read, or rules the device cannot walk are "
                "walked by nobody.\n");
        }
        return -1;
    }
    for (i = 0; i < gpu_u32_cached_n; i++) {
        int slot = gpu_rule_slot[i];
        uint32_t r32 = gpu_u32_cached_offs[gpu_u32_cached_n + i];
        if (slot < 1 || slot > (int)Numrules) continue;      /* synthetic */
        if (r32 & (GPU_U32_R_HASFORM | GPU_U32_R_BYTEOK)) continue;
        if (!gpu_rule_membership[slot - 1]) continue;         /* already clear */
        gpu_rule_membership[slot - 1] = 0;
        cleared++;
    }
    if (cleared)
        fprintf(stderr,
            "GPU rule engine: %d rule(s) were admitted to the device but the "
            "UTF-32 walker cannot run them (string-operand opcodes) and they "
            "have no byte form — membership cleared, they are walked on the "
            "CPU\n", cleared);
    return cleared;
}

/* ==== 2. tag the per-word class into word_offset bits 30-31 ============
 *
 * Operates on the PAYLOAD COPY of the offset array, never on the caller's --
 * the jobg slot's word_offset[] is reused across dispatches and is read back
 * by the host hit replay (gpujob_opencl.c, `g->word_offset[widx]`), which
 * masks nothing.  Tagging in place there would corrupt the plaintext of every
 * emitted crack.  That is the sharpest hazard in this file.
 *
 * `packed` is the packed-words region; each word is a 2-byte little-endian
 * length followed by its bytes, exactly as the pack site writes it.
 */
void gpu_u32_tag_word_offsets(uint32_t *woff, uint32_t n_words,
                                     const unsigned char *packed,
                                     uint32_t counts[3])
{
    uint32_t i;

    counts[0] = counts[1] = counts[2] = 0;
    for (i = 0; i < n_words; i++) {
        uint32_t off = woff[i] & GPU_U32_W_OFFMASK;
        int wlen = (int)packed[off] | ((int)packed[off + 1] << 8);
        unsigned enc = classify_utf8(packed + off + 2, wlen);
        uint32_t cls;

        /* classify_utf8 guarantees exactly one of U8C_ASCII / U8C_HI for
         * n > 0, and U8C_UTF8 only in company with U8C_HI -- which is what
         * collapses the word axis to three states.  Empty words carry
         * neither and are class A, decodable and not wide, which is what
         * pick_engine does with U8C_NONE. */
        if ((enc & U8C_WIDE) == U8C_WIDE)   cls = GPU_U32_CLASS_W;
        else if (enc & U8C_HI)              cls = GPU_U32_CLASS_I;
        else                                cls = GPU_U32_CLASS_A;

        counts[cls]++;
        woff[i] = off | (cls << GPU_U32_W_SHIFT);
    }
}

/* ==== 3. stable partition of one batch by class =======================
 *
 * Reorders the PAYLOAD COPY so class A and I lanes come first and class W
 * lanes last.  The geometry is `word_idx = gid % n_words`, so consecutive
 * lanes within a warp differ in WORD -- which is exactly why permuting this
 * array makes the engine branch warp-coherent for all but one warp per class
 * boundary.
 *
 * Safe with no mapping table for one reason, and it is worth stating because
 * it is the whole basis of the variant: the kernel and the host hit replay
 * address a word ONLY through word_offset, so `widx` stays self-consistent on
 * both sides.  The permutation changes which lane handles which word and
 * nothing else.  packed_buf is untouched.
 *
 * STABLE within each class, so the run is deterministic and two runs of the
 * same batch produce hits in the same order.
 *
 * Returns the number of class-A|I entries, i.e. the lane boundary, which T3
 * will need as its NDRange split.
 */
uint32_t gpu_u32_partition_word_offsets(uint32_t *woff, uint32_t n_words,
                                               uint32_t *scratch)
{
    uint32_t i, n_ai = 0, n_w = 0;

    for (i = 0; i < n_words; i++) {
        uint32_t cls = woff[i] >> GPU_U32_W_SHIFT;
        if (cls == GPU_U32_CLASS_W) scratch[n_w++] = woff[i];
        else                        woff[n_ai++]   = woff[i];
    }
    for (i = 0; i < n_w; i++) woff[n_ai + i] = scratch[i];
    return n_ai;
}


/* ==== 4. the HIT REPLAY =================================================
 *
 * THE BUG THIS EXISTS TO FIX, measured on a GTX 1080.
 *
 * The device reports a hit as (word_idx, rule_idx) and the host re-derives the
 * PLAINTEXT by re-running the rule.  That replay called applyrule() -- the BYTE
 * engine -- unconditionally, so for every pair the device routed to the UTF-32
 * walker the emitted plaintext was the byte engine's different answer.  The
 * digest was right (it matched a loaded target) and the plaintext did not hash
 * to it:
 *
 *   reported  4dc83a6af242e3330631f9a89e21a570
 *   plaintext $HEX[90e28099c391c281...]   md5 -> e9ab169033059d1e...  WRONG
 *   -G none   $HEX[  e28099c391c281...]   md5 -> 4dc83a6af242e333.   right
 *
 * One extra leading byte, on 15,165 of 15,165 recovered lines.  mdx-architect
 * predicted this class of failure and expected digest and plaintext to AGREE
 * while both disagreed with `-G none`, which would have been undetectable.  It
 * is better than that: they disagree with EACH OTHER, so re-hashing the emitted
 * plaintext catches it, which is how it was caught.
 *
 * THE FIX USES THE DEVICE'S OWN DECISION, not a recomputation of it.
 * gpu_u32_dev_offs[] is the very offset table the kernel read, so the
 * capability bits are the device's; the word class comes from classify_utf8()
 * over the same packed bytes the kernel staged; and the UTF-32 arm is handed
 * the SAME packed stream out of the SAME combined program the device walked.
 * There is no second compile and no second opinion about which engine ran.
 *
 * Contract is applyrule()'s, because that is what the call site already
 * handles: >=0 length, -1 skip this pair, -2 output equals input.
 */
/* EXTERNAL linkage, not static, and this matters: this header is included by
 * BOTH gpu_opencl.c (which sets these when it builds the program) and
 * gpujob_opencl.c (whose replay reads them).  A `static` here would give each
 * translation unit its own copy, the replay would see NULL forever, and
 * gpu_u32_replay_apply would silently fall through to applyrule() -- which is
 * the exact bug it was written to fix, reintroduced by a storage class.
 * Storage is defined once, in the TU that sets GPU_U32_HOST_DEFINE_STORAGE. */
const unsigned char *gpu_u32_dev_prog   = NULL;
const uint32_t      *gpu_u32_dev_offs   = NULL;
int                  gpu_u32_dev_nrules = 0;

/* Returned when the DEVICE used the byte engine for this pair, so the caller
 * must run applyrule() itself.
 *
 * The byte arm is NOT called from here on purpose.  Doing so would mean
 * including ../mdxfind.h for applyrule() and struct rule_workspace, and
 * mdxfind.h has no include guard -- every consumer that already includes it
 * (gpujob_opencl.c, both host tests) then fails with a dozen redefinitions.
 * Handing the decision back is also the truer shape: the call site already has
 * the bytecode pointer and the workspace in hand, and already handles
 * applyrule's -1/-2. */

int gpu_u32_replay_apply(const char *word, int wlen, int ridx,
                                char *out, int outmax)
{
    static __thread uint32_t in32[RULE32_MAXCP];
    static __thread uint32_t out32[RULE32_MAXCP];
    uint32_t r32;
    unsigned enc;
    int u32_can, byte_can, wide, u32_must, il, ol, bl;

    if (!gpu_u32_dev_offs || ridx < 0 || ridx >= gpu_u32_dev_nrules)
        return GPU_U32_REPLAY_BYTE;

    r32      = gpu_u32_dev_offs[gpu_u32_dev_nrules + ridx];
    u32_can  = (r32 & GPU_U32_R_HASFORM) != 0;
    byte_can = (r32 & GPU_U32_R_BYTEOK)  != 0;

    /* pick_engine, on the SAME inputs the kernel used.  Kept in the shape
     * mdxfind.c and procrule.c write it, because those two are token-for-token
     * identical on purpose and a paraphrase here would be a fifth dialect. */
    enc      = classify_utf8(word, (size_t)wlen);
    wide     = ((enc & U8C_WIDE) == U8C_WIDE);
    u32_must = (u32_can && !byte_can);

    if ((wide || u32_must) && u32_can &&
        (!(enc & U8C_HI) || (enc & U8C_UTF8))) {
        /* UTF-32 arm.  The stream is the device's own. */
        il = utf8_to_utf32((const unsigned char *)word, wlen, in32, RULE32_MAXCP);
        if (il < 0) return -1;
        ol = applyrule32((const uint32_t *)(gpu_u32_dev_prog +
                                            (r32 & GPU_U32_R_OFFMASK)),
                         in32, il, out32, RULE32_MAXCP, 0, NULL);
        if (ol < 0) return -1;              /* rejected, or no room */
        bl = utf32_to_utf8(out32, ol, (unsigned char *)out, outmax - 1, NULL);
        if (bl < 0) return -1;
        out[bl] = 0;
        /* -2 for an unchanged candidate, as applyrule reports it.  The UTF-32
         * engine has no such code, and without this the synthetic no-rule pass
         * would be credited twice and this rule credited for a candidate it
         * did not produce -- the -Z inflation the byte replay already guards
         * against at this call site. */
        if (bl == wlen && memcmp(out, word, (size_t)bl) == 0) return -2;
        return bl;
    }
    if (byte_can) return GPU_U32_REPLAY_BYTE;
    return -1;                               /* neither engine: skip the pair */
}


/* See the declaration.  Hands out const pointers deliberately: a backend able
 * to rewrite the built program could diverge from the one the replay reads,
 * which is the failure the single cache exists to prevent. */
int gpu_u32_get_program(const unsigned char **prog, uint32_t *len,
                        const uint32_t **offs, int *n_rules)
{
    if (!gpu_u32_cached_done) return 0;
    if (prog)    *prog    = gpu_u32_cached_prog;
    if (len)     *len     = gpu_u32_cached_len;
    if (offs)    *offs    = gpu_u32_cached_offs;
    if (n_rules) *n_rules = gpu_u32_cached_n;
    return 1;
}

#endif /* GPU_U32_HOST_IMPLEMENTATION */

#ifdef __cplusplus
}   /* extern "C" */
#endif

#endif /* GPU_U32_HOST_H */
