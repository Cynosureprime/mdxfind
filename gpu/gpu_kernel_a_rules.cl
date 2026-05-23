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
 */

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

#define RULE_BUF_MAX   40960
#define RULE_BUF_LIMIT (RULE_BUF_MAX - 1)

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
    int k = 0;
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
                if (pos >= 0 && pos < len && len < RULE_BUF_LIMIT) {
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
                for (int q = 0; q < len; q++) {
                    uchar c = buf[q];
                    if (c >= 'a' && c <= 'z') {
                        buf[q] = c ^ (uchar)0x20;
                        break;
                    }
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
                for (int q = 0; q < len; q++) {
                    uchar c = buf[q];
                    if (c >= 'A' && c <= 'Z') {
                        buf[q] = c ^ (uchar)0x20;
                        break;
                    }
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
                    if (c == ' ') { z = 0; }
                    else if (z == 0 && c >= 'a' && c <= 'z') { z = 1; buf[j] = c ^ case_flip_mask(c); }
                    else if (c >= 'A' && c <= 'Z') { buf[j] = c ^ case_flip_mask(c); }
                }
                k += 1;
                break;
            }
            case RULE_OP_TITLE_SEP: {
                uchar delim = prog[k + 1];
                int z = 0;
                for (int j = 0; j < len; j++) {
                    uchar c = buf[j];
                    if (c == delim) { z = 0; }
                    else if (z == 0 && c >= 'a' && c <= 'z') { z = 1; buf[j] = c ^ case_flip_mask(c); }
                    else if (c >= 'A' && c <= 'Z') { buf[j] = c ^ case_flip_mask(c); }
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
                    buf[pos] = (pos + 1 < len) ? buf[pos + 1] : (uchar)0;
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
            case RULE_OP_EXTRACT: {
                int start = (int)prog[k + 1] - 1;
                int count = (int)prog[k + 2] - 1;
                if (start > 0 && start < len && count > 0) {
                    int actual = 0;
                    while (actual < count && (start + actual) < len) {
                        buf[actual] = buf[start + actual];
                        actual++;
                    }
                    len = actual;
                } else if (start == 0 && start < len && count > 0) {
                    int actual = 0;
                    while (actual < count && actual < len) {
                        actual++;
                    }
                    len = actual;
                } else if (start >= len || start < 0) {
                    /* applyrule: clen > y false branch, no change. */
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
                int y = (int)prog[k + 1] - 1;
                if (y != orig_len) return -1;
                k += 2;
                break;
            }
            case RULE_OP_REJ_LEN_GE: {
                int y = (int)prog[k + 1] - 1;
                if (len < y) return -1;
                k += 2;
                break;
            }
            case RULE_OP_REJ_LEN_LE: {
                int y = (int)prog[k + 1] - 1;
                if (len > y) return -1;
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
}

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
    )
{
    __global const OCLParams *params_buf = (__global const OCLParams *)payload;
    OCLParams params = *params_buf;
    uint n_words = params.num_words;
    uint n_rules = params.num_rules;
    uint total   = n_words * n_rules;

    uint gid = get_global_id(0);
    if (gid >= total) return;

    uint word_idx = gid % n_words;
    uint rule_idx = gid / n_words;

    /* B3 cursor check (mirrors md5_rules_phase0). On a re-issue dispatch
     * the host sets input_cursor_start + rule_cursor_start to the (word,
     * rule) coordinate of the first overflowing lane from the prior
     * dispatch; lanes whose (rule, word) lex-precedes that early-return. */
    if (params.input_cursor_start > 0u || params.rule_cursor_start > 0u) {
        if (rule_idx < params.rule_cursor_start) return;
        if (rule_idx == params.rule_cursor_start &&
            word_idx < params.input_cursor_start) return;
    }

    /* Deterministic sub-buffer pointers from params.num_words (identical
     * to md5_rules_phase0). The compiler hoists since they depend only on
     * params (uniform across the dispatch). hit_count + ovr_set + ovr_gid
     * regions exist for payload symmetry; kernel A does not read or write
     * them. */
    __global const uint   *word_offset = (__global const uint *)(payload + 132);
    uint pkt_off = 132u + (n_words * 4u);
    __global const uchar  *words = payload + pkt_off;

    /* Private buffer (16-byte aligned, matches md5_rules_phase0). */
    __attribute__((aligned(16))) uchar buf[RULE_BUF_MAX];

    uint wpos = word_offset[word_idx];
    int wlen = (int)words[wpos++];
    if (wlen > RULE_BUF_LIMIT) wlen = RULE_BUF_LIMIT;
    for (int i = 0; i < wlen; i++) buf[i] = words[wpos + i];

    uint rpos = rule_offset[rule_idx];
    int is_no_rule = (rule_program[rpos] == 0);
    int new_len = apply_rule(rule_program + rpos, buf, wlen);

    /* Rejection sentinel: no slot reserved, no buffer write. */
    if (new_len < 0) return;

    /* No-op detection: synthetic ":" no-rule pass already covered this
     * candidate; skip slot emission. Foundational mdxfind behavior
     * (feedback_no_rule_pass.md). Matches md5_rules_phase0 exactly. */
    if (!is_no_rule && new_len == wlen) {
        int changed = 0;
        for (int i = 0; i < wlen; i++) {
            if (buf[i] != words[wpos + i]) { changed = 1; break; }
        }
        if (!changed) return;
    }

    /* Clamp new_len to fit in the [len] byte. RULE_BUF_LIMIT < 65536 so
     * caller-side capacity ensures it; defensive guard for the uchar
     * length-byte slot. */
    uint emit_len = (uint)new_len;
    if (emit_len > 255u) emit_len = 255u;

    /* --- Reserve a candidate slot --------------------------------- */
    uint need_bytes = 1u + emit_len;   /* [len][bytes] */
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

    uint slot = atomic_add(
        &b_kernelA_state[KERNELA_STATE_SLOT_COUNTER / 4u], 1u);

    /* Slot-index capacity: bounded by num_words * num_rules (host alloc
     * size). If for some reason slot >= total (overflow paradox: lanes
     * past 'total' early-returned at line 1086), flag and skip. */
    if (slot >= total) {
        atomic_or(&b_kernelA_state[KERNELA_STATE_OVERFLOW_FLAG / 4u], 1u);
        return;
    }

    /* --- Write [len][bytes] into packed buf ----------------------- */
    b_packed_buf[byte_off] = (uchar)emit_len;
    for (uint i = 0; i < emit_len; i++) {
        b_packed_buf[byte_off + 1u + i] = buf[i];
    }

    /* --- Write per-slot byte offset ------------------------------- *
     * Per contract S7.1 (post-Phase-2 amendment), there is NO parallel
     * rule_idx sidecar. The post-rule plaintext IS the candidate stored
     * at b_packed_buf[byte_off]; rule attribution can be re-derived from
     * slot_idx if a future need arises. */
    b_chunk_index[slot] = byte_off;

    /* Per-spec invariant 1: caller (Phase 4 host) relies on in-order
     * single-queue FIFO to ensure these writes are visible to kernel B
     * before kernel B dispatches. No explicit fence; the queue boundary
     * provides the cross-kernel global-memory visibility. */
}
