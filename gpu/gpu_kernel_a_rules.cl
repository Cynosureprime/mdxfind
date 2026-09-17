/*
 * $Revision: $
 * $Log: $
 *
 */
/* gpu_kernel_a_rules.cl -- Kernel A1 (rules-only) candidate producer.
 *
 * Production kernel A variant A1 per Phase 1a spec
 *   project_kernel_a_variants_phase1a_spec_2026-05-20.md.
 *
 * Produces packed candidates from input words x rule set via the
 * buffer-quadruple API (b_packed_buf, b_chunk_index, b_kernelA_state,
 * b_params). Output is consumable by any kernel B that obeys the
 * buffer-quadruple contract; in this codebase the existing kernel B
 * salt-axis dispatcher (gpu_opencl_kernelb_dispatch_proto) is the
 * primary consumer.
 *
 * Mechanical lineage: split from gpu_md5_rules.cl's md5_rules_phase0
 * production kernel. Identical apply_rule walker, identical opcode
 * contract, identical (word, rule) geometry, identical B3 cursor
 * semantics, identical no-op / rejection skip optimizations. The
 * MD5 + probe + EMIT_HIT tail is replaced by a write of [len][bytes]
 * into a packed candidate buffer with per-slot byte offset.
 *
 * Authoritative buffer contract:
 *   project_two_kernel_candidate_buffer_contract.md
 * Phase 1a A-variant spec:
 *   project_kernel_a_variants_phase1a_spec_2026-05-20.md
 *
 * Contract per buffer:
 *   b_packed_buf        - [len][bytes][len][bytes]... post-rule candidates.
 *                         len byte stored as uchar; bytes follow uncompressed.
 *                         Written at slot's reserved byte offset.
 *   b_chunk_index       - uint32 per slot. b_chunk_index[slot] = byte offset
 *                         into b_packed_buf where this candidate's len byte
 *                         lives. Pure physical byte offsets only.
 *   b_kernelA_state     - small counter buffer:
 *                           offset 0 : uint slot_counter   (atomic_inc)
 *                           offset 4 : uint byte_counter   (atomic_add for
 *                                                            variable-size
 *                                                            byte reservation)
 *                           offset 8 : uint overflow_flag  (set if either
 *                                                            counter exceeds
 *                                                            its capacity;
 *                                                            host re-issues
 *                                                            with larger
 *                                                            buffer or
 *                                                            smaller chunk)
 *
 * Walker behavior (mirrors md5_rules_phase0):
 *   1. Decode payload -> params, hit_count region (unused here but the same
 *      payload layout per the brief), word_offset, words.
 *   2. Decompose gid -> (word_idx, rule_idx) rule-major: same as the prod
 *      kernel.
 *   3. B3 cursor early-return for chunked overflow restart.
 *   4. Stage input word into private buf[].
 *   5. apply_rule(rule_program + rpos, buf, wlen) -> new_len.
 *   6. Rejection (new_len < 0): no slot reserved, no buffer write.
 *   7. No-op detection (!is_no_rule && new_len == wlen && buf == words):
 *      synthetic no-rule pass already covered this candidate; no slot.
 *      Foundational mdxfind behavior (feedback_no_rule_pass.md).
 *   8. Otherwise: reserve a slot, write [new_len][buf bytes] into
 *      b_packed_buf at the reserved byte offset, write byte offset into
 *      b_chunk_index[slot].
 *
 * Kernel B contract (consumer side, separate kernel):
 *   uint slot      = gid;
 *   uint wpos      = b_chunk_index[slot];
 *   if (wpos >= params.packed_size) return;  -- bounds guard per contract S5.2
 *   uint plen      = b_packed_buf[wpos];
 *   uchar *cand    = b_packed_buf + wpos + 1;
 *   uint widx      = params.base_word_idx + slot;
 *   -- kernel B then hashes cand[0..plen-1] and probes; EMIT_HIT_4 with
 *      widx + iter + digest. Per contract S7.1, no rule_idx sidecar is
 *      propagated; the plaintext IS the candidate and the host reads it
 *      from b_packed_buf[b_chunk_index[slot]].
 *
 * Geometry: same rule-major dispatch as md5_rules_phase0:
 *   global_size = n_words * n_rules
 *   word_idx    = gid % n_words
 *   rule_idx    = gid / n_words
 *
 * Reused primitives from gpu_common.cl: OCLParams struct only.
 *   - No md5_block / md5_buf / md5_to_hex_lc reference (kernel A pure).
 *   - No probe_compact_idx reference.
 *   - No EMIT_HIT_N reference.
 *
 * Salt-axis logic: not applicable. Salts are kernel B's concern. The
 * rules walker section of md5_rules_phase0 has no salt-axis fan-out
 * (verified by reading lines 1064-1153 of gpu_md5_rules.cl rev 1.30);
 * salts feed into the trailing MD5+probe block which kernel A omits.
 *
 * ====================================================================
 * KNOB G (2026-05-29) -- coalesced uint4 (16-byte) candidate writes.
 * --------------------------------------------------------------------
 * Build-gated by -DKNOBG_VEC_WRITE=1, which the host sets.  It used to be
 * reachable with MDXFIND_EXPERIMENT_RULES_CODEGEN_VEC_WRITE=1 plus a parent
 * flag; both env vars were removed 2026-09-16.  When the macro is
 * defined, the per-byte write loop that emits [len][bytes] into
 * b_packed_buf is replaced by uint4 (16-byte) stores from a private
 * 16-aligned staging buffer. Each slot's atomic byte-claim is rounded
 * UP to a multiple of 16 (need_aligned = (need_bytes + 15) & ~15), so
 * every slot starts on a 16-byte boundary and consumes a 16-byte
 * multiple of bytes. Pad bytes inside each slot's padded tail are
 * undefined; the consumer reads only [len] bytes (plen) and never
 * accesses padding -- bit-perfect equivalence with the legacy path.
 *
 * Invariant (proof by induction): the candidate buffer base is
 * 16-byte aligned by OpenCL guarantee; first byte_off=0 (16-aligned);
 * every subsequent claim adds a multiple of 16 -> every byte_off is a
 * multiple of 16. Aligned uint4 stores into the slot region are legal.
 *
 * Capacity: cap_packed sizing (host: cap_slots * 256) is unchanged.
 * 256 is the max need_aligned (need_bytes max = 256 = 1+255). Knob G
 * shifts ACTUAL bytes consumed per slot from emit_len+1 to
 * 16*ceil((emit_len+1)/16) ~ +3.8 B/slot on rockyou-1m mean (still
 * well under the cap; bounded halve-K retry handles the impossible-
 * by-construction overflow case unchanged).
 *
 * Authoritative spec: project_rules_codegen_knob_g_spec_2026-05-29.md.
 * ====================================================================
 */

/* ---- The private byte walker, and when it is NOT compiled ------------
 *
 * Kernel A carries its own copy of the opcode table, case_flip_mask() and
 * apply_rule().  Under `-8` the host puts gpu_md5_rules.cl in this program
 * instead -- the UTF-32 walker's byte arm calls apply_rule(), and the walker
 * must be defined between apply_rule and the kernel, which one source file
 * cannot express -- so both copies would be in the same translation unit and
 * the second would be a redefinition.
 *
 * Guarding this copy out rather than the shared one is deliberate: the shared
 * one is what the UTF-32 walker was written against.  The substitution is
 * byte-exact -- the three OpenCL copies of this span (gpu_md5_rules.cl,
 * gpu_kernel_a_rules.cl, gpu_kernel_a_rules_masks.cl) are code-identical after
 * comments are stripped; they differ only in kernel-A's own KERNELA_STATE_* /
 * GPU_MASK_* constants, which sit OUTSIDE the guarded regions, and in the
 * PROFILE_VARIANT==3 timing stub, which is a diagnostic build that does not
 * combine with `-8`.
 * -------------------------------------------------------------------- */
#ifndef GPU_U32_WALKER_PRESENT
/* ==== Opcode definitions (must match ruleproc.c RULE_OP_* exactly) ==== */
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
#define RULE_OP_CHR_ADD     0xc0

/* Rule-walker scratch in ELEMENTS; host injects -D RULE_BUF_MAX from
 * GPU_RULES_WALKER_BUF_ELEMS (gpujob.h).  Fallback only.  LIMIT is MAX-15
 * (NUL + n+1 slack), and exceeding it is defined behaviour: the rule stops
 * and what it produced becomes the candidate.  See gpu_md5_rules.cl. */
#ifndef RULE_BUF_MAX
#define RULE_BUF_MAX   2048
#endif
#ifndef RULE_BUF_LIMIT
#define RULE_BUF_LIMIT (RULE_BUF_MAX - 15)
#endif
#endif  /* !GPU_U32_WALKER_PRESENT -- see the note above region A */

/* Candidate-buffer width.  MUST sit OUTSIDE the region-A guard: under `-8` that
 * guard is false, and a KA_BUF_BYTES defined inside it is simply absent -- which
 * is how this shipped for one build, with the kernel-A program failing to
 * compile on `use of undeclared identifier KA_BUF_BYTES` and every e347 and
 * MAKE_MD5PASS crack lost.  U32_OUT_BYTES (8192) under `-8` because the UTF-32
 * arm encodes up to U32_BUF_LIMIT codepoints at 4 bytes each; RULE_BUF_MAX
 * (2048) otherwise, which is what the byte walker bounds itself to.  Sizing it
 * at 2048 under `-8` would make u32_encode return NOROOM -- the pair silently
 * dropped -- exactly where the byte arm would instead have produced a long
 * candidate and let the [len] clamp below truncate it. */
#ifdef GPU_U32_WALKER_PRESENT
#define KA_BUF_BYTES  U32_OUT_BYTES
#else
#define KA_BUF_BYTES  RULE_BUF_MAX
#endif

/* Kernel-A state buffer offsets. Single source of truth for host wiring
 * (Phase 4) to mirror via fixed-offset writes/reads. */
#define KERNELA_STATE_SLOT_COUNTER   0u
#define KERNELA_STATE_BYTE_COUNTER   4u
#define KERNELA_STATE_OVERFLOW_FLAG  8u
#define KERNELA_STATE_BYTES         12u

/* ==== Branchless case-flip helper ====================================
 *
 * case_flip_mask(c) returns 0x20 if `c` is alphabetic (A-Z or a-z),
 * else 0. Verbatim from gpu_md5_rules.cl rev 1.30.
 */
#ifndef GPU_U32_WALKER_PRESENT
static inline uchar case_flip_mask(uchar c) {
    uchar v = (uchar)((c | (uchar)0x20) - (uchar)'a');
    return (uchar)((v < (uchar)26) ? 0x20 : 0);
}

/* ==== Unified single-buffer in-place rule walker =====================
 *
 * Verbatim copy of apply_rule from gpu_md5_rules.cl rev 1.30 (lines
 * 179-954). The walker is identical bytecode contract, identical
 * rejection semantics, identical bounds discipline. Future post-prototype
 * refactor may hoist this into a shared header consumed by both kernels.
 * For Phase 2 mechanical-split scope, verbatim duplication is correct.
 */
static int apply_rule(__global const uchar *prog, uchar *buf, int len)
{
#if defined(PROFILE_VARIANT) && PROFILE_VARIANT == 3
    /* V3 (PROFILE_VARIANT=3): stub apply_rule — return input length
     * unchanged; suppress the huge opcode switch entirely. The walker
     * still runs (word read into buf), the atomic claim still happens,
     * the per-byte write still happens; only the apply_rule body is
     * stubbed. Used to isolate the cost of the rule-walker switch.
     * Compile-time: this `if`-true block is the entire function body. */
    return len;
#else
    int k = 0;
    /* ---- memory family: M 4 6 Q X (GPU-enabled 2026-09-11) ------------
     * Per-call scratch, matching ruleproc.c exactly: applyrule() sets
     * memlen = 0 at the top of EVERY call, so memory state never crosses a
     * (word, rule) boundary.  That is what makes these ops portable to a
     * walker where each work-item is independent -- there is no cross-rule
     * state to reproduce.
     *
     * A second RULE_BUF_MAX buffer used to be unaffordable: at 40960 bytes it
     * doubled per-thread private memory and FATAL'd CL_OUT_OF_HOST_MEMORY on
     * an RTX 3080 (mdxfind.c rev 1.24 attempt, reverted).  At 2048 it is 4 KB
     * per thread for both buffers together, a tenth of what one buffer cost
     * before, which is why the operator reopened this. */
    uchar mem[RULE_BUF_MAX];
    int   memlen = 0;
    int orig_len = len;

    for (int n = 0; n < 256; n++) {
        uchar op = prog[k];
        if (op == 0) break;

        switch (op) {

            /* ---- Variable-length affix ops (0xff / 0xfe) ---- */
            case 0xff: {
                int N = (int)prog[k + 1];
                int n_copy = N;
                if (len + n_copy > RULE_BUF_LIMIT) n_copy = RULE_BUF_LIMIT - len;
                if (n_copy < 0) n_copy = 0;
                for (int j = 0; j < n_copy; j++) {
                    buf[len + j] = prog[k + 2 + j];
                }
                len += n_copy;
                k += 2 + N;
                break;
            }
            case 0xfe: {
                int N = (int)prog[k + 1];
                int new_pre = N;
                if (len + new_pre > RULE_BUF_LIMIT) new_pre = RULE_BUF_LIMIT - len;
                if (new_pre < 0) new_pre = 0;
                for (int j = len - 1; j >= 0; j--) {
                    int dst = j + new_pre;
                    if (dst <= RULE_BUF_LIMIT) buf[dst] = buf[j];
                }
                for (int j = 0; j < new_pre; j++) {
                    buf[j] = prog[k + 2 + j];
                }
                len += new_pre;
                k += 2 + N;
                break;
            }

            /* ---- Insert / overwrite / toggle ---- */
            case RULE_OP_INSERT: {
                int pos = (int)prog[k + 1] - 1;
                uchar ch = prog[k + 2];
                /* pos <= len: inserting at position == length is an APPEND,
                 * which john and hashcat 6.2.5 both do. */
                if (pos >= 0 && pos <= len && len < RULE_BUF_LIMIT) {
                    for (int j = len; j > pos; j--) buf[j] = buf[j - 1];
                    buf[pos] = ch;
                    len++;
                }
                k += 3;
                break;
            }
            case RULE_OP_OVERWRITE: {
                int pos = (int)prog[k + 1] - 1;
                uchar ch = prog[k + 2];
                if (pos >= 0 && pos < len) buf[pos] = ch;
                if (pos == 0 && len == 0) { buf[0] = ch; len++; }
                k += 3;
                break;
            }
            case RULE_OP_TOGGLE_AT: {
                int pos = (int)prog[k + 1] - 1;
                if (pos >= 0 && pos < len) {
                    buf[pos] ^= case_flip_mask(buf[pos]);
                }
                k += 2;
                break;
            }

            /* ---- Per-position arithmetic (branchless) ---- */
            /* hashcat `BNX`: add the byte value of X to the byte at
             * position N, wrapping.  No-op out of range, matching
             * hashcat's mangle_chr_add() and ruleproc.c.  Branchless,
             * in the style of RULE_OP_INC below. */
            case RULE_OP_CHR_ADD: {
                int pos = (int)prog[k + 1] - 1;
                uchar add = prog[k + 2];
                int valid = ((pos >= 0) & (pos < len));
                int safe_pos = valid ? pos : 0;
                buf[safe_pos] = (uchar)(buf[safe_pos] + (valid ? add : (uchar)0));
                k += 3;
                break;
            }
            case RULE_OP_INC: {
                int pos = (int)prog[k + 1] - 1;
                int valid = ((pos >= 0) & (pos < len));
                int safe_pos = valid ? pos : 0;
                buf[safe_pos] = (uchar)(buf[safe_pos] + (uchar)valid);
                k += 2;
                break;
            }
            case RULE_OP_DEC: {
                int pos = (int)prog[k + 1] - 1;
                int valid = ((pos >= 0) & (pos < len));
                int safe_pos = valid ? pos : 0;
                buf[safe_pos] = (uchar)(buf[safe_pos] - (uchar)valid);
                k += 2;
                break;
            }

            /* ---- Length-shrink ---- */
            case RULE_OP_TRUNC: {
                int pos = (int)prog[k + 1] - 1;
                len = (pos < len) ? pos : len;
                k += 2;
                break;
            }
            case RULE_OP_DROP_LAST: {
                len -= (len > 0);
                k += 1;
                break;
            }

            /* ---- Substitute ---- */
            case RULE_OP_SUB: {
                uchar c1 = prog[k + 1], c2 = prog[k + 2];
                int j = 0;
                int vbound = len & ~15;
                uchar16 v_c1 = (uchar16)c1;
                uchar16 v_c2 = (uchar16)c2;
                for (; j < vbound; j += 16) {
                    uchar16 v = vload16(0, buf + j);
                    char16 eq = (v == v_c1);
                    uchar16 eq_mask = as_uchar16(eq);
                    uchar16 result = v ^ ((v ^ v_c2) & eq_mask);
                    vstore16(result, 0, buf + j);
                }
                for (; j < len; j++) {
                    if (buf[j] == c1) buf[j] = c2;
                }
                k += 3;
                break;
            }

            /* ---- Whole-string case ops ---- */
            case RULE_OP_LOWER: {
                int j = 0;
                int vbound = len & ~15;
                for (; j < vbound; j += 16) {
                    uchar16 v = vload16(0, buf + j);
                    char16 is_upper = (v >= (uchar16)'A') & (v <= (uchar16)'Z');
                    uchar16 mask = as_uchar16(is_upper) & (uchar16)0x20;
                    vstore16(v ^ mask, 0, buf + j);
                }
                for (; j < len; j++) {
                    uchar c = buf[j];
                    if (c >= 'A' && c <= 'Z') buf[j] = c ^ (uchar)0x20;
                }
                k += 1;
                break;
            }
            case RULE_OP_UPPER: {
                int j = 0;
                int vbound = len & ~15;
                for (; j < vbound; j += 16) {
                    uchar16 v = vload16(0, buf + j);
                    char16 is_lower = (v >= (uchar16)'a') & (v <= (uchar16)'z');
                    uchar16 mask = as_uchar16(is_lower) & (uchar16)0x20;
                    vstore16(v ^ mask, 0, buf + j);
                }
                for (; j < len; j++) {
                    uchar c = buf[j];
                    if (c >= 'a' && c <= 'z') buf[j] = c ^ (uchar)0x20;
                }
                k += 1;
                break;
            }
            case RULE_OP_CAP: {
                int j = 0;
                int vbound = len & ~15;
                for (; j < vbound; j += 16) {
                    uchar16 v = vload16(0, buf + j);
                    char16 is_upper = (v >= (uchar16)'A') & (v <= (uchar16)'Z');
                    uchar16 mask = as_uchar16(is_upper) & (uchar16)0x20;
                    vstore16(v ^ mask, 0, buf + j);
                }
                for (; j < len; j++) {
                    uchar c = buf[j];
                    if (c >= 'A' && c <= 'Z') buf[j] = c ^ (uchar)0x20;
                }
                /* john and hashcat both act on POSITION 0, not on the
                 * first alphabetic character: `c` on "!bang" gives "!bang"
                 * in both, where the find-first form gave "!Bang".  Mirrors
                 * ruleproc.c. */
                if (len > 0) {
                    uchar c = buf[0];
                    if (c >= 'a' && c <= 'z') buf[0] = c ^ (uchar)0x20;
                }
                k += 1;
                break;
            }
            case RULE_OP_CAP_INV: {
                int j = 0;
                int vbound = len & ~15;
                for (; j < vbound; j += 16) {
                    uchar16 v = vload16(0, buf + j);
                    char16 is_lower = (v >= (uchar16)'a') & (v <= (uchar16)'z');
                    uchar16 mask = as_uchar16(is_lower) & (uchar16)0x20;
                    vstore16(v ^ mask, 0, buf + j);
                }
                for (; j < len; j++) {
                    uchar c = buf[j];
                    if (c >= 'a' && c <= 'z') buf[j] = c ^ (uchar)0x20;
                }
                /* john and hashcat both act on POSITION 0, not on the
                 * first alphabetic character: `c` on "!bang" gives "!bang"
                 * in both, where the find-first form gave "!Bang".  Mirrors
                 * ruleproc.c. */
                if (len > 0) {
                    uchar c = buf[0];
                    if (c >= 'A' && c <= 'Z') buf[0] = c ^ (uchar)0x20;
                }
                k += 1;
                break;
            }
            case RULE_OP_REVERSE: {
                int i = 0, j = len - 1;
                while (i < j) {
                    uchar t = buf[i];
                    buf[i] = buf[j];
                    buf[j] = t;
                    i++;
                    j--;
                }
                k += 1;
                break;
            }
            case RULE_OP_TOGGLE: {
                int j = 0;
                int vbound = len & ~15;
                for (; j < vbound; j += 16) {
                    uchar16 v = vload16(0, buf + j);
                    uchar16 v_or = v | (uchar16)0x20;
                    uchar16 v_d  = v_or - (uchar16)'a';
                    char16 is_alpha = (v_d < (uchar16)26);
                    uchar16 mask = as_uchar16(is_alpha) & (uchar16)0x20;
                    vstore16(v ^ mask, 0, buf + j);
                }
                for (; j < len; j++) {
                    uchar c = buf[j];
                    if ((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z'))
                        buf[j] = c ^ (uchar)0x20;
                }
                k += 1;
                break;
            }

            /* ---- Title-case ---- */
            case RULE_OP_TITLE_SP: {
                int z = 0;
                for (int j = 0; j < len; j++) {
                    uchar c = buf[j];
                    /* An already-uppercase letter at a word start IS the
                     * capital: keep it and mark the word started.  The old
                     * form left z at 0 so the NEXT lowercase letter was
                     * capitalised -- Hello1 became hEllo1. */
                    /* Word start is POSITIONAL: the first character after a
                     * separator, or position 0, whether or not it is a letter.
                     * The old form only consumed the word start when it SAW a
                     * letter, so "!bang" gave "!Bang" where john and hashcat
                     * both give "!bang".  Mirrors ruleproc.c. */
                    if (c == ' ') { z = 0; continue; }
                    if (z == 0) { z = 1; if (c >= 'a' && c <= 'z') buf[j] = c ^ case_flip_mask(c); }
                    else { if (c >= 'A' && c <= 'Z') buf[j] = c ^ case_flip_mask(c); }
                }
                k += 1;
                break;
            }
            case RULE_OP_TITLE_SEP: {
                uchar delim = prog[k + 1];
                int z = 0;
                for (int j = 0; j < len; j++) {
                    uchar c = buf[j];
                    /* Word start is POSITIONAL: the first character after a
                     * separator, or position 0, whether or not it is a letter.
                     * The old form only consumed the word start when it SAW a
                     * letter, so "!bang" gave "!Bang" where john and hashcat
                     * both give "!bang".  Mirrors ruleproc.c. */
                    if (c == delim) { z = 0; continue; }
                    if (z == 0) { z = 1; if (c >= 'a' && c <= 'z') buf[j] = c ^ case_flip_mask(c); }
                    else { if (c >= 'A' && c <= 'Z') buf[j] = c ^ case_flip_mask(c); }
                }
                k += 2;
                break;
            }

            /* ---- Whole-buffer length-grow ops ---- */
            case RULE_OP_DUP: {
                int tlen = len;
                if (len + tlen <= RULE_BUF_LIMIT && tlen > 0) {
                    int j = 0;
                    int vbound = tlen & ~15;
                    for (; j < vbound; j += 16) {
                        uchar16 v = vload16(0, buf + j);
                        vstore16(v, 0, buf + len + j);
                    }
                    for (; j < tlen; j++) buf[len + j] = buf[j];
                    len += tlen;
                }
                k += 1;
                break;
            }
            case RULE_OP_REFLECT: {
                int tlen = len;
                if (len + tlen <= RULE_BUF_LIMIT && tlen > 0) {
                    for (int j = 0; j < tlen; j++)
                        buf[len + tlen - 1 - j] = buf[j];
                    len += tlen;
                }
                k += 1;
                break;
            }
            case RULE_OP_DUP_EACH: {
                int tlen = len;
                if (tlen * 2 <= RULE_BUF_LIMIT && tlen > 0) {
                    for (int j = tlen - 1; j >= 0; j--) {
                        uchar c = buf[j];
                        buf[j * 2]     = c;
                        buf[j * 2 + 1] = c;
                    }
                    len = tlen * 2;
                }
                k += 1;
                break;
            }

            /* ---- Rotation / swap ---- */
            case RULE_OP_ROT_L: {
                if (len > 0) {
                    uchar first = buf[0];
                    for (int j = 0; j < len - 1; j++) buf[j] = buf[j + 1];
                    buf[len - 1] = first;
                }
                k += 1;
                break;
            }
            case RULE_OP_ROT_R: {
                if (len > 0) {
                    uchar last = buf[len - 1];
                    for (int j = len - 1; j > 0; j--) buf[j] = buf[j - 1];
                    buf[0] = last;
                }
                k += 1;
                break;
            }
            case RULE_OP_SWAP_FRONT: {
                if (len > 1) {
                    uchar t = buf[0]; buf[0] = buf[1]; buf[1] = t;
                }
                k += 1;
                break;
            }
            case RULE_OP_SWAP_BACK: {
                if (len > 1) {
                    uchar t = buf[len - 2];
                    buf[len - 2] = buf[len - 1];
                    buf[len - 1] = t;
                }
                k += 1;
                break;
            }

            /* ---- Drop first / append / prepend ---- */
            case RULE_OP_DROP_FIRST: {
                if (len > 0) {
                    for (int j = 0; j < len - 1; j++) buf[j] = buf[j + 1];
                    len--;
                }
                k += 1;
                break;
            }
            case RULE_OP_APPEND: {
                uchar ch = prog[k + 1];
                if (len < RULE_BUF_LIMIT) buf[len++] = ch;
                k += 2;
                break;
            }
            case RULE_OP_PREPEND: {
                uchar ch = prog[k + 1];
                if (len < RULE_BUF_LIMIT) {
                    for (int j = len; j > 0; j--) buf[j] = buf[j - 1];
                    buf[0] = ch;
                    len++;
                }
                k += 2;
                break;
            }

            case RULE_OP_DEL_AT: {
                int pos = (int)prog[k + 1] - 1;
                if (pos >= 0 && pos < len) {
                    for (int j = pos; j < len - 1; j++) buf[j] = buf[j + 1];
                    len--;
                }
                k += 2;
                break;
            }

            /* ---- Per-position bit shifts / nearest-neighbor copy ---- */
            case RULE_OP_BIT_SHL: {
                int pos = (int)prog[k + 1] - 1;
                if (pos >= 0 && pos < len) buf[pos] = buf[pos] << 1;
                k += 2;
                break;
            }
            case RULE_OP_BIT_SHR: {
                int pos = (int)prog[k + 1] - 1;
                if (pos >= 0 && pos < len) buf[pos] = buf[pos] >> 1;
                k += 2;
                break;
            }
            case RULE_OP_REPL_NEXT: {
                int pos = (int)prog[k + 1] - 1;
                if (pos >= 0 && pos < len) {
                    /* NO-OP when pos+1 is out of range, matching hashcat and
                     * ruleproc.c: `hashcat --stdout` on abcdefghij gives
                     * abcdefghjj for `.8` and abcdefghij unchanged for `.9`.
                     * This wrote (uchar)0 instead, embedding a NUL in the
                     * candidate -- a third answer again, different from both
                     * the CPU engine's one-past-the-end read and hashcat's
                     * no-op, so CPU and GPU hit sets could disagree on any
                     * rule using `.` at the last position. */
                    if (pos + 1 < len) buf[pos] = buf[pos + 1];
                }
                k += 2;
                break;
            }
            case RULE_OP_REPL_PREV: {
                int pos = (int)prog[k + 1] - 1;
                if (pos > 0 && pos < len) buf[pos] = buf[pos - 1];
                k += 2;
                break;
            }

            /* ---- Purge ---- */
            case RULE_OP_PURGE: {
                uchar ch = prog[k + 1];
                int w = 0;
                for (int j = 0; j < len; j++) {
                    if (buf[j] != ch) buf[w++] = buf[j];
                }
                len = w;
                k += 2;
                break;
            }

            /* ---- Last/first-char duplicators ---- */
            case RULE_OP_DUP_LAST: {
                int n2 = (int)prog[k + 1] - 1;
                if (len > 0 && n2 > 0 && len + n2 <= RULE_BUF_LIMIT) {
                    uchar last = buf[len - 1];
                    int j = 0;
                    int vbound = n2 & ~15;
                    uchar16 v_last = (uchar16)last;
                    for (; j < vbound; j += 16) {
                        vstore16(v_last, 0, buf + len + j);
                    }
                    for (; j < n2; j++) buf[len + j] = last;
                    len += n2;
                }
                k += 2;
                break;
            }
            case RULE_OP_DUP_FIRST: {
                int n2 = (int)prog[k + 1] - 1;
                if (len > 0 && n2 > 0 && len + n2 <= RULE_BUF_LIMIT) {
                    uchar first = buf[0];
                    for (int j = len - 1; j > 0; j--) buf[j + n2] = buf[j];
                    for (int j = 1; j <= n2; j++) buf[j] = first;
                    len += n2;
                }
                k += 2;
                break;
            }
            case RULE_OP_REPEAT: {
                int n2 = (int)prog[k + 1] - 1;
                int tlen = len;
                if (tlen > 0 && n2 > 0) {
                    for (int copy = 0; copy < n2; copy++) {
                        if (len + tlen > RULE_BUF_LIMIT) break;
                        for (int j = 0; j < tlen; j++) buf[len + j] = buf[j];
                        len += tlen;
                    }
                }
                k += 2;
                break;
            }
            case RULE_OP_DUP_PREFIX: {
                int n2 = (int)prog[k + 1] - 1;
                if (len > 0 && n2 > 0 && n2 <= len && len + n2 <= RULE_BUF_LIMIT) {
                    for (int j = len - 1; j >= 0; j--) buf[j + n2] = buf[j];
                    for (int j = 0; j < n2; j++) buf[j] = buf[j + n2];
                    len += n2;
                }
                k += 2;
                break;
            }
            case RULE_OP_DUP_SUFFIX: {
                int n2 = (int)prog[k + 1] - 1;
                if (len > 0 && n2 > 0 && n2 <= len && len + n2 <= RULE_BUF_LIMIT) {
                    for (int j = 0; j < n2; j++) buf[len + j] = buf[len - n2 + j];
                    len += n2;
                }
                k += 2;
                break;
            }

            /* ---- Position-paired ops ---- */
            case RULE_OP_SWAP_AT: {
                int posA = (int)prog[k + 1] - 1;
                int posB = (int)prog[k + 2] - 1;
                if (posA >= 0 && posA < len && posB >= 0 && posB < len) {
                    uchar t = buf[posA]; buf[posA] = buf[posB]; buf[posB] = t;
                }
                k += 3;
                break;
            }

            /* ---- character classes (0x80-0x89) and =NX / %NX ----------
             * Mirrors ruleproc.c's class arms exactly.  The 480-byte
             * membership table and rule_class_match() live in the COMMON
             * source, in the constant address space -- deliberately not a
             * function-local array, which is what put the retired memory-op
             * attempt over the per-thread private-memory budget.
             *
             * Promoting these off the CPU-only list is the point: every
             * character-class rule used to force a mixed GPU/CPU partition,
             * and a partitioned run has to union two result sets.
             */
            case RULE_OP_SUB_CLASS: {
                uchar cb = prog[k + 1];
                uchar y  = prog[k + 2];
                for (int j = 0; j < len; j++)
                    if (rule_class_match(cb, buf[j])) buf[j] = y;
                k += 3; break;
            }
            case RULE_OP_PURGE_CLASS: {
                uchar cb = prog[k + 1];
                int d = 0;
                for (int j = 0; j < len; j++)
                    if (!rule_class_match(cb, buf[j])) buf[d++] = buf[j];
                len = d;
                k += 2; break;
            }
            case RULE_OP_TITLE_CLASS: {
                /* john's e?C: positional word start, separator untouched. */
                uchar cb = prog[k + 1];
                int z = 0;
                for (int j = 0; j < len; j++) {
                    uchar c = buf[j];
                    if (rule_class_match(cb, c)) { z = 0; continue; }
                    if (z == 0) {
                        z = 1;
                        if (c >= 'a' && c <= 'z') buf[j] = c ^ (uchar)0x20;
                    } else {
                        if (c >= 'A' && c <= 'Z') buf[j] = c ^ (uchar)0x20;
                    }
                }
                k += 2; break;
            }
            case RULE_OP_TITLE_CLASS_HC: {
                /* hashcat's ~e?C is a DIFFERENT algorithm from john's e?C
                 * above: it lowercases every position, uppercases position 0
                 * and every position whose PREDECESSOR was in the class, and
                 * case-normalises the separator itself.  The class test reads
                 * the pre-modification byte. */
                uchar cb = prog[k + 1];
                int up = 1;
                for (int j = 0; j < len; j++) {
                    uchar c = buf[j];
                    int this_up = up;
                    up = rule_class_match(cb, c) ? 1 : 0;
                    if (c >= 'A' && c <= 'Z') { c ^= (uchar)0x20; buf[j] = c; }
                    if (this_up && c >= 'a' && c <= 'z') buf[j] = c ^ (uchar)0x20;
                }
                k += 2; break;
            }
            case RULE_OP_REJ_HAS_CLASS: {
                uchar cb = prog[k + 1];
                for (int j = 0; j < len; j++)
                    if (rule_class_match(cb, buf[j])) return -1;
                k += 2; break;
            }
            case RULE_OP_REJ_NHAS_CLASS: {
                uchar cb = prog[k + 1];
                int found = 0;
                for (int j = 0; j < len; j++)
                    if (rule_class_match(cb, buf[j])) { found = 1; break; }
                if (!found) return -1;
                k += 2; break;
            }
            case RULE_OP_REJ_FIRST_CLASS: {
                uchar cb = prog[k + 1];
                if (len > 0 && !rule_class_match(cb, buf[0])) return -1;
                k += 2; break;
            }
            case RULE_OP_REJ_LAST_CLASS: {
                uchar cb = prog[k + 1];
                if (len > 0 && !rule_class_match(cb, buf[len - 1])) return -1;
                k += 2; break;
            }
            case RULE_OP_REJ_AT_CLASS: {
                int   y  = (int)prog[k + 1] - 1;
                uchar cb = prog[k + 2];
                if (y >= len || !rule_class_match(cb, buf[y])) return -1;
                k += 3; break;
            }
            case RULE_OP_REJ_CNT_CLASS: {
                int   y  = (int)prog[k + 1] - 1;
                uchar cb = prog[k + 2];
                int cnt = 0;
                for (int j = 0; j < len; j++)
                    if (rule_class_match(cb, buf[j])) cnt++;
                if (cnt < y) return -1;
                k += 3; break;
            }
            case RULE_OP_REJ_AT_CHR: {
                /* `=NX` reject unless the character at position N is X. */
                int   y = (int)prog[k + 1] - 1;
                uchar c = prog[k + 2];
                if (y >= len || buf[y] != c) return -1;
                k += 3; break;
            }
            case RULE_OP_REJ_CNT_CHR: {
                /* `%NX` reject unless X occurs at least N times. */
                int   y = (int)prog[k + 1] - 1;
                uchar c = prog[k + 2];
                int cnt = 0;
                for (int j = 0; j < len; j++) if (buf[j] == c) cnt++;
                if (cnt < y) return -1;
                k += 3; break;
            }

            /* ---- memory family, mirroring ruleproc.c's SLOW path ---------
             * The CPU fast path escapes to slowrule on overflow; the slow
             * path CLAMPS (y = MAXLINE - clen) rather than skipping, and that
             * is the behaviour with a real limit, so it is what we mirror at
             * RULE_BUF_LIMIT.  `6` prepends by memmove-right-then-copy
             * because a GPU buffer has no headroom before index 0, where the
             * CPU walks cpass backwards into its 512-byte slack. */
            case RULE_OP_MEM_STORE: {          /* M -- store candidate */
                for (int j = 0; j < len; j++) mem[j] = buf[j];
                memlen = len;
                k += 1; break;
            }
            case RULE_OP_MEM_APP: {            /* 4 -- append memory */
                int y = memlen;
                if (len + y > RULE_BUF_LIMIT) y = RULE_BUF_LIMIT - len;
                if (y < 0) y = 0;
                if (y > 0) {
                    for (int j = 0; j < y; j++) buf[len + j] = mem[j];
                    len += y;
                }
                k += 1; break;
            }
            case RULE_OP_MEM_PRE: {            /* 6 -- prepend memory */
                int y = memlen;
                if (len + y > RULE_BUF_LIMIT) y = RULE_BUF_LIMIT - len;
                if (y < 0) y = 0;
                if (y > 0) {
                    for (int j = len - 1; j >= 0; j--) buf[j + y] = buf[j];
                    for (int j = 0; j < y; j++) buf[j] = mem[j];
                    len += y;
                }
                k += 1; break;
            }
            case RULE_OP_MEM_REJ: {            /* Q -- reject if == memory */
                if (memlen == len) {
                    int same = 1;
                    for (int j = 0; j < len; j++) {
                        if (buf[j] != mem[j]) { same = 0; break; }
                    }
                    if (same) return -1;
                }
                k += 1; break;
            }
            case RULE_OP_MEM_INSERT: {         /* X N M I */
                /* Insert M chars of memory from OFFSET N at position I.
                 * Out-of-range REJECTS, following hashcat's
                 * mangle_insert_multi -- operator ruling 2026-09-11, and
                 * identical to the CPU arm. */
                int y    = (int)prog[k + 1] - 1;   /* offset within memory */
                int tlen = (int)prog[k + 2] - 1;   /* count              */
                int z    = (int)prog[k + 3] - 1;   /* insert position    */
                if (memlen < 1 || tlen < 1 || z > len ||
                    y > memlen || (y + tlen) > memlen) return -1;
                if (len + tlen > RULE_BUF_LIMIT) tlen = RULE_BUF_LIMIT - len;
                if (tlen > 0) {
                    for (int j = len; j >= z; j--) buf[j + tlen] = buf[j];
                    for (int j = 0; j < tlen; j++) buf[z + j] = mem[y + j];
                    len += tlen;
                }
                k += 4; break;
            }
            case RULE_OP_EXTRACT: {
                /* xAB: extract B characters from position A.  Mirrors
                 * ruleproc.c, which follows HASHCAT here (operator ruling
                 * 2026-09-11, superseding an earlier ruling for john).
                 * hashcat's mangle_extract() no-ops on BOTH out-of-range
                 * conditions and never extracts a partial run:
                 *     if (upos >= arr_len)         return arr_len;
                 *     if ((upos + ulen) > arr_len) return arr_len;
                 * On "abc": x22 and x23 are "abc" here and "c" in john;
                 * x90 is "abc" here and a reject in john.  A zero count
                 * with an in-range start yields an EMPTY candidate, which
                 * is kept -- john's empty-word rejection was also ruled to
                 * hashcat.  start==0 self-copies, as on the CPU. */
                int start = (int)prog[k + 1] - 1;
                int count = (int)prog[k + 2] - 1;
                if (start >= 0 && start < len && (start + count) <= len) {
                    for (int q = 0; q < count; q++) {
                        buf[q] = buf[start + q];
                    }
                    len = count;
                }
                k += 3;
                break;
            }
            case RULE_OP_OMIT: {
                int pos = (int)prog[k + 1] - 1;
                int count = (int)prog[k + 2] - 1;
                if (pos >= 0 && pos < len && count > 0 && pos + count <= len) {
                    for (int j = pos; j + count < len; j++) buf[j] = buf[j + count];
                    len -= count;
                }
                k += 3;
                break;
            }
            case RULE_OP_TOGGLE_SEP: {
                int upos = (int)prog[k + 1] - 1;
                uchar sep = prog[k + 2];
                int toggle_next = 0;
                int occurrence = 0;
                for (int j = 0; j < len; j++) {
                    uchar c = buf[j];
                    if (c == sep) {
                        if (occurrence == upos) toggle_next = 1;
                        else occurrence++;
                        continue;
                    }
                    if (toggle_next) {
                        if ((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z'))
                            buf[j] = c ^ case_flip_mask(c);
                        break;
                    }
                }
                k += 3;
                break;
            }

            /* ---- No-ops ---- */
            case RULE_OP_NOOP:
            case RULE_OP_NOOP_SP:
            case RULE_OP_NOOP_TAB:
                k += 1;
                break;

            /* ---- Special ops (silent no-ops) ---- */
            case RULE_OP_S_SPECIAL:  k += 1; break;
            case RULE_OP_HASH_EXIT:  k += 1; break;
            case RULE_OP_DIV_INSERT: k += 3; break;

            /* ---- Rejection ops ---- */
            case RULE_OP_REJ_LEN_NE: {
                /* CURRENT length, not the original word's. */
                int y = (int)prog[k + 1] - 1;
                if (y != len) return -1;
                k += 2;
                break;
            }
            case RULE_OP_REJ_LEN_GE: {
                /* john: "<N reject unless less than N chars" -- reject when
                 * len >= N.  This was the complement, matching ruleproc.c's
                 * old behaviour; both fixed together so CPU and GPU agree. */
                int y = (int)prog[k + 1] - 1;
                if (len >= y) return -1;
                k += 2;
                break;
            }
            case RULE_OP_REJ_LEN_LE: {
                /* john: ">N reject unless greater than N chars". */
                int y = (int)prog[k + 1] - 1;
                if (len <= y) return -1;
                k += 2;
                break;
            }
            case RULE_OP_REJ_HAS: {
                uchar c = prog[k + 1];
                for (int j = 0; j < len; j++) {
                    if (buf[j] == c) return -1;
                }
                k += 2;
                break;
            }
            case RULE_OP_REJ_NHAS: {
                uchar c = prog[k + 1];
                int found = 0;
                for (int j = 0; j < len; j++) {
                    if (buf[j] == c) { found = 1; break; }
                }
                if (!found) return -1;
                k += 2;
                break;
            }
            case RULE_OP_REJ_FIRST: {
                uchar c = prog[k + 1];
                if (len > 0 && buf[0] != c) return -1;
                k += 2;
                break;
            }
            case RULE_OP_REJ_LAST: {
                uchar c = prog[k + 1];
                if (len > 0 && buf[len - 1] != c) return -1;
                k += 2;
                break;
            }

            /* H/h hex emit: split cases (ROCm comgr addrspace fix per
             * gpu_md5_rules.cl rev 1.27). */
            case RULE_OP_HEX_UPPER: {
                const uchar uhex[16] = {
                    '0','1','2','3','4','5','6','7',
                    '8','9','A','B','C','D','E','F'
                };
                int x = len;
                if (x + len > RULE_BUF_LIMIT) x = RULE_BUF_LIMIT - len;
                if (x < 0) x = 0;
                int new_len = len + x;
                for (int i = x - 1; i >= 0; i--) {
                    uchar c = buf[i];
                    buf[i * 2]     = uhex[(c >> 4) & 0xf];
                    buf[i * 2 + 1] = uhex[c & 0xf];
                }
                len = new_len;
                k += 1;
                break;
            }
            case RULE_OP_HEX_LOWER: {
                const uchar lhex[16] = {
                    '0','1','2','3','4','5','6','7',
                    '8','9','a','b','c','d','e','f'
                };
                int x = len;
                if (x + len > RULE_BUF_LIMIT) x = RULE_BUF_LIMIT - len;
                if (x < 0) x = 0;
                int new_len = len + x;
                for (int i = x - 1; i >= 0; i--) {
                    uchar c = buf[i];
                    buf[i * 2]     = lhex[(c >> 4) & 0xf];
                    buf[i * 2 + 1] = lhex[c & 0xf];
                }
                len = new_len;
                k += 1;
                break;
            }

            default:
                return len;
        }
    }
    return len;
#endif  /* PROFILE_VARIANT == 3 stub guard */
}
#endif  /* !GPU_U32_WALKER_PRESENT -- see the note above region A */

/* ---- Kernel A1 (rules-only) production kernel --------------------
 *
 * Payload layout is identical to md5_rules_phase0 (Memo B B1
 * coalesced layout):
 *
 *   offset   0 : OCLParams params
 *   offset 128 : uint hit_count            (unused by kernel A; reserved
 *                                            for payload symmetry with
 *                                            md5_rules_phase0)
 *   offset 132 : uint word_offset[num_words]
 *   offset 132 + 4*num_words : uchar packed_words[]
 *
 * params.base_word_idx is read for hit-attribution propagation. Phase 1
 * rev 1.23 of gpu_common.cl renamed reserved32[0] -> base_word_idx with
 * this exact semantic. Kernel A does NOT write base_word_idx; the host
 * sets it before dispatch.
 *
 * params.packed_size is read as the kernel-A output buffer capacity (bytes
 * available in b_packed_buf). Overflow detection: any candidate that would
 * push byte_counter past packed_size sets overflow_flag and returns
 * without writing.
 *
 * Output buffer caps (capacity guards):
 *   b_packed_buf       capacity = params.packed_size bytes
 *   b_chunk_index      capacity = params.num_words * params.num_rules slots
 *                                  (worst case: every (word, rule) emits
 *                                   a slot, which is the rejection-free /
 *                                   no-op-free upper bound).
 *
 * On overflow the kernel signals via overflow_flag (state offset 8); host
 * retries with a smaller chunk or larger buffer. Mirrors the rules-engine
 * B3 cursor pattern.
 */

__kernel
void cand_rules_phase0(
    __global uchar         *payload,
    __global const uchar   *rule_program,
    __global const uint    *rule_offset,
    __global uchar         *b_packed_buf,
    __global uint          *b_chunk_index,
    __global volatile uint *b_kernelA_state
    /* UTF-32 path: the two 1:1 case tables, CONCATENATED.  Appended LAST so
     * every existing argument index is unchanged.  __global rather than
     * __constant because 26,048 bytes in the constant bank put the program
     * past NVIDIA's 64 KB limit and ptxas refused the build. */
#ifdef GPU_U32_WALKER_PRESENT
    , __global const uint  *case_tab
#endif
    )
{
    __global const OCLParams *params_buf = (__global const OCLParams *)payload;
    OCLParams params = *params_buf;
    uint n_words = params.num_words;
    /* RULE-AXIS CHUNKING (Option A, 2026-05-29): params.num_rules carries the
     * per-chunk rule COUNT (rule_count) and params.rule_cursor_start carries
     * the per-chunk rule BASE (rule_start) into rule_offset[]. The host loops
     * disjoint rule sub-ranges [rule_start, rule_start+rule_count) so the
     * candidate buffer for one dispatch is bounded to n_words*rule_count*256
     * <= cap. When the whole ruleset fits one chunk (e347 / small rule count)
     * the host sets rule_start=0 and rule_count=n_rules, making this decode
     * BYTE-IDENTICAL to the pre-chunking geometry (rule_idx = gid / n_words).
     * The legacy B3 overflow-cursor early-return is REMOVED from this A1 path:
     * with chunking the per-chunk buffer cannot overflow by construction
     * (host caps K), and overloading rule_cursor_start with two meanings
     * (chunk base vs overflow-restart cursor) would corrupt the chunk decode.
     * Overflow is now handled host-side by a bounded halve-K retry. */
    uint rule_start = params.rule_cursor_start;  /* chunk base into rule_offset[] */
    uint n_rules    = params.num_rules;           /* this chunk's rule_count */
    uint total      = n_words * n_rules;

    uint gid = get_global_id(0);

    if (gid >= total) return;

    /* ==== PROFILE_VARIANT scaffolding (2026-05-29, perf decomposition) =====
     * When the kernel is JIT-built with -DPROFILE_VARIANT=N (N in 1..5), the
     * per-lane kernel body is progressively stubbed to attribute kernel_a_us
     * to its components. V0 (default; macro undefined) = byte-identical to
     * the production path. ALL variants leave b_kernelA_state untouched
     * (slot_counter = 0, byte_counter = 0) so the host sees actual_slots=0
     * and skips kernel B; kernel_a_us is captured by the existing CL
     * profiling event before kernel B runs (host site
     * gpu_opencl.c:13059 `if (actual_slots == 0) ... continue` already exists).
     * Crack-parity is INTENTIONALLY NOT preserved for V1..V5; results are
     * throwaway, this is timing-only profiling infrastructure.
     *
     *   V1 = no atomic claim (per-lane deterministic offset gid*256 instead
     *        of atomic_add; per-byte write still happens; apply_rule still
     *        runs). Measures: atomic_add cost share.
     *   V2 = no candidate write (atomic_add still happens but per-byte
     *        write loop is no-op'd; index store skipped). Measures: per-byte
     *        write loop cost share (BUT note V2 lets atomic still fire so
     *        slot_counter would advance; we force it back to 0 below via
     *        an end-of-kernel reset by lane 0 to keep host-side
     *        actual_slots=0 invariant).
     *   V3 = stub apply_rule (returns wlen unchanged; rule walker walks
     *        but switch body is gone). Measures: apply_rule switch cost.
     *   V4 = walker only — read word into buf, then return. NO apply_rule,
     *        NO atomic, NO write. Measures: word-read + apply_rule + write
     *        in aggregate (delta from V5).
     *   V5 = empty kernel — return immediately. Measures: dispatch overhead
     *        + chunk-loop overhead + kernel launch latency.
     *
     * V5 must short-circuit BEFORE the word-read; place gate here. */
#if defined(PROFILE_VARIANT) && PROFILE_VARIANT == 5
    return;
#endif

    uint word_idx = gid % n_words;
    uint rule_idx = rule_start + (gid / n_words);  /* GLOBAL rule index */

    /* Deterministic sub-buffer pointers from params.num_words (identical
     * to md5_rules_phase0). The compiler hoists since they depend only on
     * params (uniform across the dispatch). hit_count + ovr_set + ovr_gid
     * regions exist for payload symmetry; kernel A does not read or write
     * them. */
    __global const uint   *word_offset = (__global const uint *)(payload + 132);
    uint pkt_off = 132u + (n_words * 4u);
    __global const uchar  *words = payload + pkt_off;

    /* Private buffer (16-byte aligned, matches md5_rules_phase0).
     * KA_BUF_BYTES, not RULE_BUF_MAX -- see the note at its definition. */
    __attribute__((aligned(16))) uchar buf[KA_BUF_BYTES];

    /* Per-lane emit state. Populated for in-range lanes only. */
    int should_emit = 0;
    uint emit_len = 0u;
    uint need_bytes = 0u;
    uint wpos = 0u;

    /* Walk the rule. */
    {
#ifdef GPU_U32_WALKER_PRESENT
        /* Bits 30-31 carry the per-word CLASS under `-8`
         * (gpu_u32_tag_word_offsets), so the offset MUST be masked.  The byte
         * arm below does not, because in byte mode the host never writes the
         * tag; failing to mask when it does cost a
         * `md5_rules ovr-state read err=-5` on the legacy kernel.
         *
         * No staging loop here: u32_run_pair reads the word out of `words`
         * itself, on both of its arms. */
        uint raw_woff = word_offset[word_idx];
        wpos = raw_woff & U32_OFF_MASK;
        uint wcls = raw_woff >> U32_TAG_SHIFT;
        int wlen = (int)words[wpos] | ((int)words[wpos + 1] << 8);
        wpos += 2;   /* 2-byte little-endian length: see gpujob.h */
        if (wlen > RULE_BUF_LIMIT) wlen = RULE_BUF_LIMIT;
#else
        wpos = word_offset[word_idx];
        int wlen = (int)words[wpos] | ((int)words[wpos + 1] << 8);
        wpos += 2;   /* 2-byte little-endian length: see gpujob.h */
        if (wlen > RULE_BUF_LIMIT) wlen = RULE_BUF_LIMIT;
        for (int i = 0; i < wlen; i++) buf[i] = words[wpos + i];
#endif

#if defined(PROFILE_VARIANT) && PROFILE_VARIANT == 4
        /* V4 (walker only): stop here. word_read into buf is done; skip
         * apply_rule, atomic claim, per-byte write, index store. The
         * compiler must not dead-strip the buf write — `buf` is private
         * stack; force a fence by reading it back into a sink. */
        if (buf[0] == 0xffu) {
            /* unreachable on real data (wlen>0 by construction); inserted
             * to keep the buf write in the live range so the compiler does
             * not strip the read into buf. */
            atomic_or(&b_kernelA_state[KERNELA_STATE_OVERFLOW_FLAG / 4u], 0u);
        }
        return;
#endif

        uint rpos = rule_offset[rule_idx];
        int is_no_rule = (rule_program[rpos] == 0);
#ifdef GPU_U32_WALKER_PRESENT
        /* UTF-32 path (`-8`).  The second half of rule_offset holds the UTF-32
         * stream offset plus the two capability bits, based at
         * params.num_masks -- which is 0 on every byte-mode kernel-A dispatch
         * and is set by the host to the FIRST-HALF ENTRY COUNT when `-8` is
         * active.  Basing it on that count rather than on this chunk's
         * params.num_rules is required: num_rules is the per-chunk rule COUNT
         * under rule-axis chunking, so a second or later chunk would index the
         * wrong half.
         *
         * `buf` is passed as both bytebuf and out; the aliasing is safe by
         * inspection of u32_run_pair (its byte arm's closing copy becomes
         * self-assignment, its UTF-32 arm never touches bytebuf). */
        uint u32_rpos = rule_offset[params.num_masks + rule_idx];
        __attribute__((aligned(16))) uint u32buf[U32_BUF_ELEMS];
        __attribute__((aligned(16))) uint u32mem[U32_BUF_ELEMS];
        int engine = 0;
        int new_len = u32_run_pair(words, wpos, wlen, rule_program,
                                   rpos, u32_rpos, wcls,
                                   buf, u32buf, u32mem,
                                   buf, KA_BUF_BYTES, case_tab, &engine);
#else
        int new_len = apply_rule(rule_program + rpos, buf, wlen);
#endif

        if (new_len >= 0) {
            /* No-op detection: synthetic ":" no-rule pass already covered
             * this candidate; skip slot emission. Foundational mdxfind
             * behavior (feedback_no_rule_pass.md). Matches md5_rules_phase0
             * exactly. */
            int suppress_noop = 0;
            if (!is_no_rule && new_len == wlen) {
                int changed = 0;
                for (int i = 0; i < wlen; i++) {
                    if (buf[i] != words[wpos + i]) { changed = 1; break; }
                }
                if (!changed) suppress_noop = 1;
            }
            if (!suppress_noop) {
                /* Clamp new_len to fit in the [len] byte. RULE_BUF_LIMIT <
                 * 65536 so caller-side capacity ensures it; defensive guard
                 * for the uchar length-byte slot. */
                uint elen = (uint)new_len;
                if (elen > 255u) elen = 255u;
                emit_len = elen;
                need_bytes = 1u + elen;   /* [len][bytes] */
                should_emit = 1;
            }
        }
    }

    /* Per-lane global atomic_add slot/byte reservation. */
    if (!should_emit) return;

#if defined(PROFILE_VARIANT) && PROFILE_VARIANT == 1
    /* V1 (no atomic claim): substitute deterministic per-lane offsets for
     * the atomic_add slot/byte reservation. Per-byte write still happens
     * (so V0 - V1 attributes the atomic_add cost). slot_counter +
     * byte_counter stay at zero, so host sees actual_slots=0 and skips
     * kernel B. The fake byte_off (gid * 256) keeps writes within the
     * pre-allocated packed buffer for any gid covered by the host's K-cap
     * (worst-case K*num_words slots, 256 B/slot = cap). */
    uint byte_off = gid * 256u;
    uint slot     = gid;
    if (byte_off + need_bytes > params.packed_size) return;
    if (slot >= total) return;
    b_packed_buf[byte_off] = (uchar)emit_len;
    for (uint i = 0; i < emit_len; i++) {
        b_packed_buf[byte_off + 1u + i] = buf[i];
    }
    b_chunk_index[slot] = byte_off;
#elif defined(PROFILE_VARIANT) && PROFILE_VARIANT == 2
    /* V2 (no candidate write): atomic claim still runs (so V0 - V2
     * attributes the per-byte write loop). Per-byte memcpy + index store
     * are removed. We INTENTIONALLY do not write slot_counter — we use a
     * private read-write of the byte_counter to keep the atomic_add live
     * (compiler must not fold; the result `byte_off` is unused). To prevent
     * the host from running kernel B with non-zero actual_slots, we leave
     * slot_counter alone (it stays 0). */
    uint byte_off = atomic_add(
        &b_kernelA_state[KERNELA_STATE_BYTE_COUNTER / 4u],
        need_bytes);
    /* Keep byte_off live so the compiler doesn't drop the atomic; cheap
     * side-effect via overflow flag (ored with 0 = no-op functionally). */
    if (byte_off == 0xffffffffu) {
        atomic_or(&b_kernelA_state[KERNELA_STATE_OVERFLOW_FLAG / 4u], 0u);
    }
    /* Slot-counter NOT incremented (intentional — host actual_slots=0 ->
     * kernel B skipped). Per-byte write loop omitted (the measured target). */
#elif defined(PROFILE_VARIANT) && PROFILE_VARIANT == 6
    /* V6 (Knob G micro-benchmark; D4 in knob_g_spec): V0 baseline with
     * the Knob G vectorized write loop FORCED ON, regardless of
     * KNOBG_VEC_WRITE. Used to measure V0-vs-V6 kernel_a_us delta in
     * the same harness that motivated Knob G -- closes the prediction-
     * verification loop. Slot/byte claim semantics + crack output are
     * IDENTICAL to V0 with KNOBG_VEC_WRITE=1; the only reason V6
     * exists separately is so the operator can toggle just this leg
     * of the cost share independently of Knob G.  Both were selected by
     * env vars until 2026-09-16; nothing selects them now. */
    {
        uint need_aligned = (need_bytes + 15u) & ~15u;
        uint byte_off = atomic_add(
            &b_kernelA_state[KERNELA_STATE_BYTE_COUNTER / 4u],
            need_aligned);
        if (byte_off + need_aligned > params.packed_size) {
            atomic_or(&b_kernelA_state[KERNELA_STATE_OVERFLOW_FLAG / 4u], 1u);
            return;
        }
        uint slot = atomic_add(
            &b_kernelA_state[KERNELA_STATE_SLOT_COUNTER / 4u], 1u);
        if (slot >= total) {
            atomic_or(&b_kernelA_state[KERNELA_STATE_OVERFLOW_FLAG / 4u], 1u);
            return;
        }
        __private uchar __attribute__((aligned(16))) stage[256];
        stage[0] = (uchar)emit_len;
        for (uint i = 0; i < emit_len; i++) stage[1u + i] = buf[i];
        uint nvec = need_aligned / 16u;
        __global uint4 *dst = (__global uint4 *)(b_packed_buf + byte_off);
        __private uint4 *src = (__private uint4 *)stage;
        for (uint v = 0; v < nvec; v++) dst[v] = src[v];
        b_chunk_index[slot] = byte_off;
    }
#else
    /* V0 (production baseline) — legacy path. KNOB G gated. */
#ifdef KNOBG_VEC_WRITE
    /* KNOB G ON: round need_bytes up to a 16-byte multiple. The atomic
     * shape is unchanged (single atomic_add); only the value is rounded.
     * Slot-start alignment proof: base ptr 16-aligned + each running sum
     * adds a multiple of 16 -> every byte_off is 16-aligned. */
    uint need_aligned = (need_bytes + 15u) & ~15u;
    uint byte_off = atomic_add(
        &b_kernelA_state[KERNELA_STATE_BYTE_COUNTER / 4u],
        need_aligned);

    /* Capacity guard: use the post-rounding byte count for the bound
     * check (otherwise the tail uint4 stores could spill past packed). */
    if (byte_off + need_aligned > params.packed_size) {
        atomic_or(&b_kernelA_state[KERNELA_STATE_OVERFLOW_FLAG / 4u], 1u);
        return;
    }
#else
    uint byte_off = atomic_add(
        &b_kernelA_state[KERNELA_STATE_BYTE_COUNTER / 4u],
        need_bytes);

    /* Capacity guard: did this lane's reservation push past packed_size?
     * If so, flag overflow and return WITHOUT incrementing slot_counter
     * (so the slot count reflects only successfully-emitted candidates). */
    if (byte_off + need_bytes > params.packed_size) {
        atomic_or(&b_kernelA_state[KERNELA_STATE_OVERFLOW_FLAG / 4u], 1u);
        return;
    }
#endif

    uint slot = atomic_add(
        &b_kernelA_state[KERNELA_STATE_SLOT_COUNTER / 4u], 1u);

    /* Slot-index capacity: bounded by num_words * num_rules (host alloc
     * size). If for some reason slot >= total (overflow paradox: lanes
     * past 'total' early-returned at the top), flag and skip. */
    if (slot >= total) {
        atomic_or(&b_kernelA_state[KERNELA_STATE_OVERFLOW_FLAG / 4u], 1u);
        return;
    }

    /* --- Write [len][bytes] into packed buf ----------------------- */
#ifdef KNOBG_VEC_WRITE
    /* KNOB G ON: stage [len][bytes] into a private 16-aligned buffer,
     * then issue need_aligned/16 uint4 stores. emit_len in [0,255] ->
     * need_aligned in [16,256] -> nvec in [1,16]. Pad bytes in
     * stage[1+emit_len .. need_aligned) are undefined; consumer reads
     * only plen bytes after the [len] header (see §3 spec). */
    {
        __private uchar __attribute__((aligned(16))) stage[256];
        stage[0] = (uchar)emit_len;
        for (uint i = 0; i < emit_len; i++) stage[1u + i] = buf[i];
        uint nvec = need_aligned / 16u;
        __global uint4 *dst = (__global uint4 *)(b_packed_buf + byte_off);
        __private uint4 *src = (__private uint4 *)stage;
        for (uint v = 0; v < nvec; v++) dst[v] = src[v];
    }
#else
    b_packed_buf[byte_off] = (uchar)emit_len;
    for (uint i = 0; i < emit_len; i++) {
        b_packed_buf[byte_off + 1u + i] = buf[i];
    }
#endif
    b_chunk_index[slot] = byte_off;
#endif  /* PROFILE_VARIANT 1/2/6/V0 selection */
    /* Per-spec invariant 1: caller (Phase 4 host) relies on in-order
     * single-queue FIFO to ensure these writes are visible to kernel B
     * before kernel B dispatches. No explicit fence; the queue boundary
     * provides the cross-kernel global-memory visibility. */
}
