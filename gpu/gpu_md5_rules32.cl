/*
 * $Revision: $
 * $Log: $
 *
 */
/* gpu_md5_rules32.cl -- MD5 kernels driving the UTF-32 rule walker.
 *
 * Compiled AFTER gpu_common.cl, gpu_md5_rules.cl and gpu_u32_walker.cl, and
 * it depends on all three:
 *   gpu_common.cl      OCLParams, probe_compact_idx, EMIT_HIT_*, rule_class_match
 *   gpu_md5_rules.cl   md5_buf(), the RULE_OP_* opcode names, apply_rule()
 *                      (the BYTE walker -- still needed, see below)
 *   gpu_u32_walker.cl  u32_decode / u32_encode / apply_rule32 and the tables
 *
 * WHY THE BYTE WALKER IS STILL NEEDED.  pick_engine() (mdxfind.c) routes a
 * class-W word crossed with a BYTE-ONLY rule to the BYTE engine, and BYTE-ONLY
 * rules are real: 31 of HashMob.100k.rule, 1 of HashMob.10k.rule -- rules whose
 * own TEXT is not valid UTF-8, so the UTF-32 compiler refuses them.  A device
 * carrying only the UTF-32 walker would silently drop those pairs.  Both
 * walkers live here and the engine is chosen per (word, rule).
 *
 * ---- THE WIRE FORMAT, and why it needs no new kernel argument -------------
 *
 * OCLParams is FULL: 128 bytes, every offset named, and word_offset sits at a
 * hard-coded payload offset of 132 in 13 sites across 9 kernel files.  So
 * nothing new goes in it.  Instead:
 *
 *   rule_program  [ byte bytecode per rule, NUL-terminated ]
 *                 [ pad to a 4-byte boundary ]
 *                 [ uint32 packrule32 streams, RULE32_END-terminated ]
 *
 *   rule_offset[rule_idx]              byte offset of the BYTE stream.
 *                                      *** UNTAGGED AND UNCHANGED. ***
 *   rule_offset[n_rules + rule_idx]    bit 31 = a UTF-32 form exists
 *                                      bit 30 = Rule_byteok
 *                                      bits 0-29 = 4-aligned byte offset of
 *                                        the UTF-32 stream (meaningless when
 *                                        bit 31 is clear)
 *
 * BOTH capability bits live in the SECOND half, and the first half is left
 * exactly as md5_rules_phase0 already reads it.  That is not tidiness: the
 * production byte kernel masks NOTHING when it does
 * `uint rpos = rule_offset[rule_idx];`, so a tag bit in the first half would
 * send it to a wild offset.  Keeping the first half pristine means T0 (the
 * byte-only baseline) and the byte launch of T3 need no change to
 * gpu_md5_rules.cl at all -- which is what makes the comparison a comparison
 * rather than two different programs.
 *
 *   word_offset[word_idx]              byte offset into the packed words,
 *                                      bits 30-31 = per-word class:
 *                                        0 = A  ASCII / empty  (decodable)
 *                                        1 = W  HI|UTF8        (decodable)
 *                                        2 = I  HI alone       (NOT decodable)
 *
 * Both fields have room: the rules-path packed buffer is
 * GPU_RULES_MAX_WORDS_PER_BATCH * 256 = 2^22 and the rule program is well
 * under 2^30 at every real rule-file size (HashMob.100k packs to ~500 KB).
 * The host asserts prog_len <= 2^30 before it sets a tag bit.
 *
 * Carrying the class in word_offset rather than in a parallel array is what
 * makes the LANE PARTITION variants free: the kernels and both host hit
 * replays address a word ONLY through word_offset, so permuting that array
 * permutes the lanes with no change to the packed bytes and no mapping table.
 *
 * ---- THE KERNELS ---------------------------------------------------------
 *
 *   md5_rules_phase0          (in gpu_md5_rules.cl)  the BYTE walker
 *   md5_rules32_phase0        UTF-32 walker for every lane
 *   md5_rules_mixed_phase0    per-lane engine choice -- the PRODUCTION kernel
 *                             for `-8`; same 14 shared arguments and same
 *                             coalesced payload as md5_rules_phase0 plus one
 *                             of its own (the case tables), so the host swaps
 *                             the handle and changes nothing else
 *   md5_rules_mixed_validate  emits post-rule UTF-8 BYTES per (word, rule);
 *                             the oracle-diff instrument, not a hot path
 *
 * A per-batch class PARTITION and a C2 decode PRE-PASS were both built,
 * measured and removed; see the notes at the partition site in gpu_opencl.c
 * and above md5_rules_mixed_phase0 here.
 */

/* The wire-format constants (U32_R_*, U32_OFF_MASK, U32_TAG_SHIFT,
 * U32_OUT_BYTES) and u32_run_pair() moved into gpu_u32_walker.inc, so BOTH
 * backends share one copy.  They belong there rather than here because
 * u32_run_pair's signature is already expressible in the address-space macros
 * the walker uses -- `U32_GLOBAL const uchar *prog` and `U32_BUF uchar *buf`
 * are exactly what OpenCL's private-buf and Metal's device-buf spellings need
 * -- so writing it twice would have been a choice, not a necessity. */

/* The C2 PRE-PASS kernel that decoded every word once into a uint32 buffer is
 * DELETED, not disabled.
 *
 * MEASURED before removing it: the in-lane decode is 8.5-10.2% of a UTF-32
 * pair's cost (the walk is the other ~90%), so a pre-pass had a ~10% ceiling --
 * and it paid for that with a 4x wider per-lane staging read, an extra
 * dispatch against the $Log 1549 budget, and a global buffer of
 * n_words * U32_BUF_ELEMS * 4 = 16384 * 2048 * 4 = 128 MB per batch.
 *
 * Also worth stating because it is the part that is easy to get wrong: INSIDE
 * A LANE the two options are the same thing.  The geometry is one lane per
 * (word, rule), so a lane decodes exactly one word either way; "decode once
 * per word" only ever existed as a separate pass over global memory.  And the
 * ENCODE is per-lane in both variants -- the output differs per rule -- so it
 * is the half a pre-pass structurally cannot amortise.
 *
 * Deleted rather than left behind a knob: it was env-selectable, and
 * MDXFIND_CACHE is the only environment input mdxfind takes.  Dead kernel
 * source also costs JIT time and constant budget in every build.
 */

/* ==== T1 / T2 production kernel: per-lane engine choice ===============
 *
 * Signature is IDENTICAL to md5_rules_phase0 in gpu_md5_rules.cl -- same 14
 * arguments, same coalesced payload, same offsets 128 / 132 / 132+4*n_words.
 * That is deliberate: the host dispatch path can swap the cl_kernel handle
 * and change nothing else, which is the only way the T0/T1/T2 comparison is
 * measuring the kernel rather than the plumbing.
 */
__kernel
void md5_rules_mixed_phase0(
    __global uchar        *payload,
    __global const uchar  *rule_program,
    __global const uint   *rule_offset,
    __global const uint   *compact_fp,
    __global const uint   *compact_idx,
    __global const uchar  *hash_data_buf,
    __global const ulong  *hash_data_off,
    __global const ushort *unused_hash_data_len,
    __global uint         *hits,
    __global const ulong  *overflow_keys,
    __global const uchar  *overflow_hashes,
    __global const uint   *overflow_offsets,
    __global const ushort *unused_overflow_lengths,
    __global volatile uint *hashes_shown,
    /* The two 1:1 case tables, CONCATENATED: up at [0 .. U32_CASE_N-1], lo
     * above it.  A __global argument rather than __constant because 26,048
     * bytes of constant data put this program 13 KB past NVIDIA's 64 KB bank
     * -- ptxas refused the build outright.  See gpu_u32_walker.inc. */
    __global const uint   *case_tab
    )
{
    __global const OCLParams *params_buf = (__global const OCLParams *)payload;
    OCLParams params = *params_buf;
    uint n_words = params.num_words;
    uint n_rules = params.num_masks;
    uint total = n_words * n_rules;

    uint gid = get_global_id(0);
    if (gid >= total) return;

    uint word_idx = gid % n_words;
    uint rule_idx = gid / n_words;

    /* B3 two-cursor re-issue protocol, byte-for-byte as md5_rules_phase0 does
     * it: on a dispatch following a hit-buffer overflow, lanes that lex-precede
     * the cursor early-return so the hits already returned are not re-emitted. */
    if (params.input_cursor_start > 0u || params.rule_cursor_start > 0u) {
        if (rule_idx < params.rule_cursor_start) return;
        if (rule_idx == params.rule_cursor_start &&
            word_idx < params.input_cursor_start) return;
    }

    __global volatile uint *hit_count = (__global volatile uint *)(payload + 128);
    __global const uint   *word_offset = (__global const uint *)(payload + 132);
    uint pkt_off = 132u + (n_words * 4u);
    __global const uchar  *words = payload + pkt_off;
    __global volatile uint *ovr_set = (__global volatile uint *)(payload + 100);
    __global volatile uint *ovr_gid = (__global volatile uint *)(payload + 104);

    __attribute__((aligned(16))) uchar bytebuf[RULE_BUF_MAX];
    __attribute__((aligned(16))) uint  u32buf[U32_BUF_ELEMS];
    __attribute__((aligned(16))) uint  u32mem[U32_BUF_ELEMS];
    __attribute__((aligned(16))) uchar cand[U32_OUT_BYTES];

    uint raw   = word_offset[word_idx];
    uint wpos  = raw & U32_OFF_MASK;
    uint wcls  = raw >> U32_TAG_SHIFT;
    int  wlen  = (int)words[wpos] | ((int)words[wpos + 1] << 8);
    wpos += 2;
    if (wlen > RULE_BUF_LIMIT) wlen = RULE_BUF_LIMIT;

    uint byte_roff = rule_offset[rule_idx];
    uint u32_roff  = rule_offset[n_rules + rule_idx];

    /* The synthetic no-rule pass: an EMPTY byte program.  It must still run
     * MD5+probe -- it IS the no-rule pass, which mdxfind treats as
     * foundational.  It is detected on the byte stream because the synthetic
     * entry has no UTF-32 form by construction. */
    int is_no_rule = (rule_program[byte_roff] == 0);

    int engine = 0;
    int clen = u32_run_pair(words, wpos, wlen, rule_program,
                            byte_roff, u32_roff, wcls,
                            bytebuf, u32buf, u32mem,
                            cand, U32_OUT_BYTES, case_tab, &engine);
    if (clen < 0) return;

    /* No-op detection.  The byte kernel compares its walker buffer to the
     * input bytes; here the comparison is against the ENCODED output, which is
     * the same test the CPU bridge makes when it returns -2 for an unchanged
     * candidate.  Comparing CODEPOINTS instead would miss the case where the
     * rule changed the codepoint sequence but not its UTF-8 serialisation --
     * which cannot happen for valid input, but the byte comparison is free and
     * does not depend on that being true. */
    if (!is_no_rule && clen == wlen) {
        int changed = 0;
        for (int i = 0; i < wlen; i++) {
            if (cand[i] != words[wpos + i]) { changed = 1; break; }
        }
        if (!changed) return;
    }

    uint hx, hy, hz, hw;
    md5_buf(cand, clen, &hx, &hy, &hz, &hw);

    uint max_iter = params.max_iter;
    if (max_iter < 1) max_iter = 1;
    for (uint iter = 1; iter <= max_iter; iter++) {
        uint matched_idx = 0u;
        if (probe_compact_idx(hx, hy, hz, hw,
                              compact_fp, compact_idx,
                              params.compact_mask, params.max_probe,
                              params.hash_data_count,
                              hash_data_buf, hash_data_off,
                              overflow_keys, overflow_hashes,
                              overflow_offsets, params.overflow_count,
                              &matched_idx)) {
            uint mask = 1u << (iter & 31);
            /* Same B3 overflow-aware, dedup-aware emit as md5_rules_phase0.
             * The dedup-bit ROLLBACK on overflow is load-bearing: without it
             * the bit stays set across dispatches and the re-issue lane
             * silently drops the crack. */
            EMIT_HIT_4_DEDUP_OR_OVERFLOW(hits, hit_count, params.max_hits,
                       word_idx, rule_idx, iter, hx, hy, hz, hw,
                       hashes_shown, matched_idx, mask,
                       ovr_set, ovr_gid, gid);
        }
        if (iter < max_iter) {
            /* md5_to_hex_lc writes the 32 hex chars packed into M[0..7], so
             * the re-hash is assembled here rather than through md5_buf. */
            uint M[16];
            md5_to_hex_lc(hx, hy, hz, hw, M);
            M[8] = 0x80u;
            for (int j = 9; j < 14; j++) M[j] = 0u;
            M[14] = 32u * 8u;
            M[15] = 0u;
            hx = 0x67452301u; hy = 0xEFCDAB89u;
            hz = 0x98BADCFEu; hw = 0x10325476u;
            md5_block(&hx, &hy, &hz, &hw, M);
        }
    }
    (void)hashes_shown;
    (void)unused_hash_data_len;
    (void)unused_overflow_lengths;
}

/* ==== pure UTF-32 production kernel ===================================
 *
 * Every lane takes the UTF-32 arm.  This is the SECOND LAUNCH of the
 * two-dispatch variant (T3): the host partitions word_offset[] by class, then
 * launches the existing byte kernel over the class-A|I lane range and this one
 * over the class-W range.  Its register allocation is the UTF-32 walker's
 * alone, where md5_rules_mixed_phase0's is the union of both walkers even for
 * lanes that take the byte path -- which is exactly the effect T3 exists to
 * measure.
 *
 * It reads lane indices through params.input_cursor_start as a lane BASE, so
 * the host does not have to re-pack the batch: the two launches cover disjoint
 * lane ranges of the same payload.
 */
__kernel
void md5_rules32_phase0(
    __global uchar        *payload,
    __global const uchar  *rule_program,
    __global const uint   *rule_offset,
    __global const uint   *compact_fp,
    __global const uint   *compact_idx,
    __global const uchar  *hash_data_buf,
    __global const ulong  *hash_data_off,
    __global const ushort *unused_hash_data_len,
    __global uint         *hits,
    __global const ulong  *overflow_keys,
    __global const uchar  *overflow_hashes,
    __global const uint   *overflow_offsets,
    __global const ushort *unused_overflow_lengths,
    __global volatile uint *hashes_shown,
    /* The two 1:1 case tables, CONCATENATED: up at [0 .. U32_CASE_N-1], lo
     * above it.  A __global argument rather than __constant because 26,048
     * bytes of constant data put this program 13 KB past NVIDIA's 64 KB bank
     * -- ptxas refused the build outright.  See gpu_u32_walker.inc. */
    __global const uint   *case_tab
    )
{
    __global const OCLParams *params_buf = (__global const OCLParams *)payload;
    OCLParams params = *params_buf;
    uint n_words = params.num_words;
    uint n_rules = params.num_masks;
    uint total = n_words * n_rules;

    uint gid = get_global_id(0) + params.base_word_idx;   /* lane base for T3 */
    if (gid >= total) return;

    uint word_idx = gid % n_words;
    uint rule_idx = gid / n_words;

    __global volatile uint *hit_count = (__global volatile uint *)(payload + 128);
    __global const uint   *word_offset = (__global const uint *)(payload + 132);
    uint pkt_off = 132u + (n_words * 4u);
    __global const uchar  *words = payload + pkt_off;
    __global volatile uint *ovr_set = (__global volatile uint *)(payload + 100);
    __global volatile uint *ovr_gid = (__global volatile uint *)(payload + 104);

    __attribute__((aligned(16))) uint  u32buf[U32_BUF_ELEMS];
    __attribute__((aligned(16))) uint  u32mem[U32_BUF_ELEMS];
    __attribute__((aligned(16))) uchar cand[U32_OUT_BYTES];

    uint raw  = word_offset[word_idx];
    uint wpos = raw & U32_OFF_MASK;
    int  wlen = (int)words[wpos] | ((int)words[wpos + 1] << 8);
    wpos += 2;
    if (wlen > RULE_BUF_LIMIT) wlen = RULE_BUF_LIMIT;

    uint u32_roff = rule_offset[n_rules + rule_idx];
    if (!(u32_roff & U32_R_HASFORM)) return;  /* host must not route this lane */

    int n = u32_decode(words + wpos, wlen, u32buf, U32_BUF_ELEMS);
    if (n < 0) return;
    n = apply_rule32((__global const uint *)(rule_program + (u32_roff & U32_R_OFFMASK)),
                     u32buf, u32mem, n, case_tab);
    if (n < 0) return;
    int clen = u32_encode(u32buf, n, cand, U32_OUT_BYTES);
    if (clen < 0) return;

    if (clen == wlen) {
        int changed = 0;
        for (int i = 0; i < wlen; i++)
            if (cand[i] != words[wpos + i]) { changed = 1; break; }
        if (!changed) return;
    }

    uint hx, hy, hz, hw;
    md5_buf(cand, clen, &hx, &hy, &hz, &hw);

    uint max_iter = params.max_iter;
    if (max_iter < 1) max_iter = 1;
    for (uint iter = 1; iter <= max_iter; iter++) {
        uint matched_idx = 0u;
        if (probe_compact_idx(hx, hy, hz, hw,
                              compact_fp, compact_idx,
                              params.compact_mask, params.max_probe,
                              params.hash_data_count,
                              hash_data_buf, hash_data_off,
                              overflow_keys, overflow_hashes,
                              overflow_offsets, params.overflow_count,
                              &matched_idx)) {
            uint mask = 1u << (iter & 31);
            /* Same B3 overflow-aware, dedup-aware emit as md5_rules_phase0.
             * The dedup-bit ROLLBACK on overflow is load-bearing: without it
             * the bit stays set across dispatches and the re-issue lane
             * silently drops the crack. */
            EMIT_HIT_4_DEDUP_OR_OVERFLOW(hits, hit_count, params.max_hits,
                       word_idx, rule_idx, iter, hx, hy, hz, hw,
                       hashes_shown, matched_idx, mask,
                       ovr_set, ovr_gid, gid);
        }
        if (iter < max_iter) {
            /* md5_to_hex_lc writes the 32 hex chars packed into M[0..7], so
             * the re-hash is assembled here rather than through md5_buf. */
            uint M[16];
            md5_to_hex_lc(hx, hy, hz, hw, M);
            M[8] = 0x80u;
            for (int j = 9; j < 14; j++) M[j] = 0u;
            M[14] = 32u * 8u;
            M[15] = 0u;
            hx = 0x67452301u; hy = 0xEFCDAB89u;
            hz = 0x98BADCFEu; hw = 0x10325476u;
            md5_block(&hx, &hy, &hz, &hw, M);
        }
    }
    (void)hashes_shown;
    (void)unused_hash_data_len;
    (void)unused_overflow_lengths;
}

/* ==== VALIDATE kernels: the oracle-diff instruments ===================
 *
 * Wire format is md5_rules_phase0_validate's exactly, so one host parser
 * reads both:
 *   slot[0..1]  retlen, int16 LE (sign-preserving)
 *   slot[2..3]  outlen, uint16 LE
 *   slot[4..]   the candidate bytes
 *
 * These emit IDENTITY output -- a candidate equal to its input is written out
 * like any other.  That is on purpose and it is the recorded trap: procrule
 * suppresses identity output while hashcat's hcrule emits it, and a per-rule
 * TARGET test passes spuriously for identity-output rules because the implicit
 * no-rule pass already satisfies the target.  Comparing full candidate streams
 * avoids both.
 */
#define MD5_RULES32_VALIDATE_MAX_BUF   U32_OUT_BYTES
#define MD5_RULES32_VALIDATE_RECORD_SZ (4 + MD5_RULES32_VALIDATE_MAX_BUF)

__kernel
void md5_rules_mixed_validate(
    __global const uchar  *words,
    __global const uint   *word_offset,
    __global const uchar  *rule_program,
    __global const uint   *rule_offset,
    __global const OCLParams *params_buf,
    __global uchar        *records_out,
    __global const uint   *case_tab)
{
    OCLParams params = *params_buf;
    uint n_words = params.num_words;
    uint n_rules = params.num_masks;
    uint total = n_words * n_rules;
    uint gid = get_global_id(0);
    if (gid >= total) return;

    uint word_idx = gid % n_words;
    uint rule_idx = gid / n_words;

    __attribute__((aligned(16))) uchar bytebuf[RULE_BUF_MAX];
    __attribute__((aligned(16))) uint  u32buf[U32_BUF_ELEMS];
    __attribute__((aligned(16))) uint  u32mem[U32_BUF_ELEMS];
    __attribute__((aligned(16))) uchar cand[U32_OUT_BYTES];

    uint raw  = word_offset[word_idx];
    uint wpos = raw & U32_OFF_MASK;
    uint wcls = raw >> U32_TAG_SHIFT;
    int  wlen = (int)words[wpos] | ((int)words[wpos + 1] << 8);
    wpos += 2;
    if (wlen > RULE_BUF_LIMIT) wlen = RULE_BUF_LIMIT;

    int engine = 0;
    int clen = u32_run_pair(words, wpos, wlen, rule_program,
                            rule_offset[rule_idx], rule_offset[n_rules + rule_idx],
                            wcls, bytebuf, u32buf, u32mem,
                            cand, U32_OUT_BYTES, case_tab, &engine);

    int retlen = clen;
    int outlen = (retlen >= 0) ? retlen : 0;

    uint slot = word_idx * n_rules + rule_idx;
    uint base = slot * (uint)MD5_RULES32_VALIDATE_RECORD_SZ;

    short retlen16 = (short)retlen;
    records_out[base + 0] = (uchar)((uint)retlen16 & 0xffu);
    records_out[base + 1] = (uchar)(((uint)retlen16 >> 8) & 0xffu);
    ushort outlen16 = (ushort)outlen;
    records_out[base + 2] = (uchar)((uint)outlen16 & 0xffu);
    records_out[base + 3] = (uchar)(((uint)outlen16 >> 8) & 0xffu);
    for (int i = 0; i < outlen && i < MD5_RULES32_VALIDATE_MAX_BUF; i++)
        records_out[base + 4 + (uint)i] = cand[i];
}
