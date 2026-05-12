/*
 * $Revision: 1.30 $
 * $Log: gpu_md5_rules.cl,v $
 * Revision 1.30  2026/05/11 05:22:01  dlr
 * Backfill $Revision/$Log RCS keyword stanzas per feedback_rcs_keyword_stanzas.md. Passive 4-line comment block at top of file; no behavioral change. Hand-authored .cl file was missing required stanzas (per memory: all hand-authored .c/.h/.cl/.frag/.tmpl/.py/.sh files MUST contain $Revision/$Log keyword stanzas). Build green on .205 against the post-add files; OpenCL compile strips comments so no kernel behavior change.
 *
 */
/* gpu_md5_rules.cl — Phase 0+ GPU rule expansion + multi-block MD5 kernel
 *
 * UNIFIED WALKER (rev 1.17+):
 *   Single-buffer in-place per-op semantics. apply_rule(prog, buf, len)
 *   replaces the previous apply_rule_op() function-call walker + ping-pong
 *   buffer model. All three kernels (md5_rules_phase0,
 *   md5_rules_phase0_test, md5_rules_phase0_validate) share the SAME walker.
 *
 *   Bytecode contract:
 *     - Opcodes are high-bit (0xc1..0xfd contiguous + 0xfe/0xff variable).
 *     - Mapping is fixed by ruleproc.c rev 1.25+ packrules() emit-sites.
 *     - applyrule() in ruleproc.c is the byte-exact CPU reference.
 *
 *   Buffer model:
 *     - Single private uchar buf[512] (NOT 64, NOT 128, NOT 256).
 *     - Length stored in private int len.
 *     - In-place modification: ops mutate buf in place.
 *     - Length grows past current `len` for append/insert/prepend ops
 *       (write into uninitialized tail; bound check `len + N <= 127`).
 *     - Length shrinks for `]`, `D`, `[`, `'`, `O`, `x`, `@` (decrement
 *       len, optionally compact).
 *
 *   Dispatch:
 *     - switch (op) on contiguous 0xc1..0xfd values + 0xff/0xfe special.
 *     - 0xff (multi-$) and 0xfe (multi-^) handled OUTSIDE the main switch
 *       (variable-length argument bytes).
 *
 *   Geometry (rule-major):
 *     global_size = n_words * n_rules
 *     gid -> (word_idx, rule_idx) by rule-major decomposition:
 *       word_idx = gid % n_words
 *       rule_idx = gid / n_words
 *     => consecutive lanes within a warp share rule_idx, varying word_idx.
 *
 *   Per work-item:
 *     1. Stage input word into buf[0..wlen-1].
 *     2. apply_rule(rule_program + rpos, buf, wlen) -> new_len.
 *     3. (production) skip MD5+probe if buf == input (no-op detection).
 *     4. (production) Multi-block MD5 of buf[0..new_len). probe_compact;
 *        emit hit if found.
 *     (test kernel) emit MD5 digest per (word, rule).
 *     (validate kernel) emit fixed-size record [retlen][outlen][bytes].
 *
 * Primitives reused from gpu_common.cl: md5_block, md5_to_hex_lc,
 * probe_compact_idx, EMIT_HIT_4, OCLParams.
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

#define RULE_BUF_MAX   40960                      /* was 512 — bumped 2026-05-02 to match
                                                   * CPU's MAXLINE=40*1024 in mdxfind.h:63.
                                                   * Tracks ruleproc.c's MAXLINE clamp so any
                                                   * CPU-producible rule output fits in the
                                                   * GPU walker without divergent clamping. */
#define RULE_BUF_LIMIT (RULE_BUF_MAX - 1)         /* now 40959 */

/* ==== Branchless case-flip helper ====================================
 *
 * case_flip_mask(c) returns 0x20 if `c` is alphabetic (A-Z or a-z),
 * else 0. Uses the unsigned-subtract trick: `(c | 0x20) - 'a' < 26u`.
 * The OR lifts uppercase into lowercase territory (or leaves lowercase
 * unchanged); the unsigned compare is true only for 'a'..'z'.
 *
 * Robust across the entire 0x00..0xff input space (verified against
 * predicate-landmine sweep in mdx-team-state #72): no false positives
 * on '@' (0x40), '[' (0x5b), '`' (0x60), '{' (0x7b), DEL (0x7f), or
 * any control / high-bit byte.
 *
 * Compiles to predicated mov on jump-table backends:
 *   NVIDIA  -> selp.b32
 *   AMD     -> v_cndmask_b32
 *   ARM/Mali-> CSEL
 * All single-cycle.
 *
 * NOTE: OpenCL select(a,b,c) on scalar uchar uses MSB-of-c as the
 * boolean predicate (NOT C-style nonzero). The unsigned compare here
 * yields 0 or 1 (MSB always 0), so select() would always pick `a`.
 * Use the explicit ternary form instead — the compiler emits the
 * correct predicated move on every backend.
 */
static inline uchar case_flip_mask(uchar c) {
    /* Promote to uchar for the unsigned 8-bit subtract: OpenCL promotes
     * uchar to int in expressions, so a naive `(c|0x20) - 'a'` becomes
     * signed and `-49 < 26u` is TRUE, misclassifying digits. Cast the
     * subtraction result back to uchar so the unsigned wrap kicks in,
     * then compare to 26. */
    uchar v = (uchar)((c | (uchar)0x20) - (uchar)'a');
    return (uchar)((v < (uchar)26) ? 0x20 : 0);
}

/* ==== Unified single-buffer in-place rule walker =====================
 *
 * apply_rule:
 *   - prog: __global pointer to NUL-terminated bytecode (offset already
 *           applied by caller; first byte is the first opcode).
 *   - buf:  private uchar[RULE_BUF_MAX] register array, pre-staged with
 *           the input word in buf[0..len-1].
 *   - len:  initial length of the staged input.
 * Returns: post-rule length (>= 0) on success, -1 if a rejection op fires.
 *          The buffer is modified in place.
 *
 * Rejection semantics (rev 1.22): rejection ops `_ < > ! / ( )` mirror
 * applyrule's `_retval = -1` early-exit. When fired, apply_rule returns
 * -1 and callers must skip MD5+probe (production), emit zero digests
 * (test kernels), or emit `retlen=-1, outlen=0, no bytes` (validate).
 *
 * Note for the `_` op: it tests length-not-equal against the ORIGINAL
 * input length (the `len` parameter at function entry), NOT the running
 * clen. We capture orig_len at entry to match applyrule byte-exact.
 */
static int apply_rule(__global const uchar *prog, uchar *buf, int len)
{
    int k = 0;
    int orig_len = len;     /* preserved for the `_ N` length-equal test */

    for (int n = 0; n < 256; n++) {
        uchar op = prog[k];
        if (op == 0) break;

        /* Main switch — contiguous high-bit opcodes 0xc1..0xff.
         * 0xff/0xfe are variable-length (op + N + N data bytes); the rest
         * are fixed-size. Folded into the switch so the compiler can emit a
         * single jump table over the contiguous range, saving ~0.4 cyc/iter
         * on average across HashMob.100k frequency mix vs the prior hoist
         * (which added 2 compares/branch on every fixed-size opcode).
         * Listed by frequency rank then by opcode value for readability. */
        switch (op) {

            /* ---- Variable-length affix ops (0xff / 0xfe) ---- */
            case 0xff: {
                /* Multi-char append: 2 + N bytes, N data bytes follow N-byte.
                 * Wave F: cap-once-then-copy. Compute the actual usable
                 * count once instead of branching on every byte. */
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
                /* Multi-char prepend: shift right by N, then write N data
                 * bytes at buf[0..N-1]. */
                int N = (int)prog[k + 1];
                int new_pre = N;
                if (len + new_pre > RULE_BUF_LIMIT) new_pre = RULE_BUF_LIMIT - len;
                if (new_pre < 0) new_pre = 0;
                /* Shift right (work backward to avoid overwrite). */
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

            /* ---- Top-frequency: insert / overwrite / toggle ---- */
            case RULE_OP_INSERT: {
                /* iNX: insert byte arg2 at position arg1-1.
                 * applyrule semantics: only fires when (clen > pos). */
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
                /* Branchless: case_flip_mask returns 0 for non-alphabetic
                 * bytes, so XOR with 0 is identity — the inner alphabetic
                 * test collapses into the helper. Bounds check on `pos`
                 * stays — required to match applyrule semantics. */
                int pos = (int)prog[k + 1] - 1;
                if (pos >= 0 && pos < len) {
                    buf[pos] ^= case_flip_mask(buf[pos]);
                }
                k += 2;
                break;
            }

            /* ---- Per-position arithmetic (Wave E branchless) ----
             *
             * Pattern: clamp `pos` into a safe slot, mask the increment
             * by the validity flag, unconditionally write. When `valid`
             * is 0 we write `buf[0] + 0` — a no-op idempotent self-write.
             * On NVIDIA / AMD this compiles to one selp/v_cndmask + one
             * predicated add, eliminating the divergent branch. */
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

            /* ---- Length-shrink (Wave E) ----
             * NOTE: the original `if (pos < len) len = pos;` matches
             * applyrule. Branchless version preserved as a ternary —
             * equivalent code shape, no behavioral change. */
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
                /* Wave D SIMD: vector compare against c1, blend c2 into
                 * matched lanes, vstore. Length-preserving — past-len
                 * writes invisible per detector + md5_buf semantics. */
                uchar c1 = prog[k + 1], c2 = prog[k + 2];
                int j = 0;
                int vbound = len & ~15;
                uchar16 v_c1 = (uchar16)c1;
                uchar16 v_c2 = (uchar16)c2;
                for (; j < vbound; j += 16) {
                    uchar16 v = vload16(0, buf + j);
                    char16 eq = (v == v_c1);
                    /* Blend: where eq is true (-1 = 0xFF), pick v_c2,
                     * else v. (v ^ ((v ^ v_c2) & eq_mask)). */
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

            /* ---- Whole-string case ops ---- *
             *
             * SIMD strategy (Wave C, mdx-team-state #75):
             *   Vector vload16/vstore16 over `(len & ~15)` bytes plus a
             *   scalar tail for `len & 15` bytes. Bytes past `len` are
             *   read+written but ignored downstream — the no-op detector
             *   at line 813-819 reads buf[0..wlen-1] only, and md5_buf
             *   reads buf[0..len-1] only. Length-preserving ops do not
             *   change `len`, so writes past `len` are invisible.
             *
             *   Predicate: same shape as case_flip_mask but vectorized.
             *     v_or = v | 0x20             (lift A-Z into a-z range)
             *     v_d  = (uchar)(v_or - 'a')  (0..25 for letters)
             *     m    = (v_d < 26) ? 0x20 : 0   per-lane
             *   For `l`: only flip when v originally was upper-case.
             *     m_l = m & ~(v & 0x20)   (mask&0x20 already 0 for upper,
             *                              so equivalent to `m if v>='A'&&v<='Z'`)
             *     simpler: `mask = ((v >= 'A') & (v <= 'Z')) & 0x20`.
             */
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
                /* Capitalize: lowercase all (vector), then upper-case
                 * the first lowercase letter (scalar pass — variable
                 * exit, intrinsically scalar). Matches ruleproc.c SSE. */
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
                /* Pass 2: scalar find-first-lower. */
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
                /* Anti-capitalize: uppercase all (vector), then lower-case
                 * the first uppercase letter (scalar pass). */
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
                /* Pass 2: scalar find-first-upper. */
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
                    /* Vectorized case_flip_mask: alphabetic detection
                     * via unsigned 8-bit subtract + compare. */
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
                /* Wave F: vector copy buf[0..tlen-1] -> buf[len..len+tlen-1].
                 * Source/dest don't overlap (offset == tlen). Scalar tail
                 * keeps writes within tlen — must NOT overshoot into the
                 * uninitialized region past 2*tlen since this is a length-
                 * grow op (output buf[0..2*tlen-1] is read downstream). */
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
                /* "abc" -> "aabbcc"; length doubles. Work from the end so
                 * we don't overwrite source bytes still being read. */
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
                /* Rotate left by 1: "abcd"->"bcda".
                 * applyrule: while (*rule == '{' && y < clen) coalesces
                 * consecutive {{; the new bytecode emits each as a separate
                 * RULE_OP_ROT_L so we step once per opcode. Functionally
                 * equivalent (chained single-step rotation == multi-step). */
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
                /* Wave F: broadcast vstore for the fill loop. */
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
                    /* Shift right by n2. */
                    for (int j = len - 1; j > 0; j--) buf[j + n2] = buf[j];
                    /* applyrule's loop is `for (x = clen - 1; x > 0; x--)
                     *    cpass[x + y] = cpass[x];` then `for (x = 1; x <= y; x++)
                     *    cpass[x] = cpass[0];` — buf[0] is preserved (it is
                     * the source we copy from), buf[1..n2] are filled with
                     * first. So buf[0] keeps its original value. */
                    for (int j = 1; j <= n2; j++) buf[j] = first;
                    len += n2;
                }
                k += 2;
                break;
            }
            case RULE_OP_REPEAT: {
                /* p N: append N copies of the input to itself (input
                 * total = (N+1)*len; applyrule reads "y = *rule++ - 1;"
                 * then loops y times. In our bytecode the arg is also
                 * positiontranslate-encoded so n2 = arg-1, and the loop
                 * runs n2 times. */
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
                /* y N: duplicate first N chars at the start.
                 * applyrule: memmove(cpass + n, cpass, clen); for (x = 0;
                 * x < n; x++) cpass[x] = cpass[x + n] is implicit in the
                 * pre-move state (the leading n bytes equal the original
                 * leading n bytes after the right-shift). Actually applyrule
                 * uses memmove only; the prepended bytes are buf[0..n-1]
                 * which are now the first n bytes of the original input. */
                int n2 = (int)prog[k + 1] - 1;
                if (len > 0 && n2 > 0 && n2 <= len && len + n2 <= RULE_BUF_LIMIT) {
                    /* memmove(cpass + n, cpass, clen) — shift right by n. */
                    for (int j = len - 1; j >= 0; j--) buf[j + n2] = buf[j];
                    /* The first n bytes of the buffer are now whatever was
                     * shifted out — we want them to be the original first n
                     * bytes. After the shift, buf[n..n+n-1] equals the
                     * original buf[0..n-1]; we just need to copy those down. */
                    for (int j = 0; j < n2; j++) buf[j] = buf[j + n2];
                    len += n2;
                }
                k += 2;
                break;
            }
            case RULE_OP_DUP_SUFFIX: {
                /* Y N: duplicate last N chars at the end. */
                int n2 = (int)prog[k + 1] - 1;
                if (len > 0 && n2 > 0 && n2 <= len && len + n2 <= RULE_BUF_LIMIT) {
                    /* memmove(cpass + clen, cpass + (clen - n), n) */
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
                /* xAB: out = in[A..A+B-1] capped at len-A.
                 * applyrule: for (x = 0; x < z && ((y + x) < clen); x++)
                 *   cpass[x] = cpass[y + x]; clen = x; */
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
                    /* start=0 case: src and dst overlap at index 0; the
                     * applyrule shape `cpass[x] = cpass[y + x]` with y=0
                     * is buf[x]=buf[x] — no change beyond the truncation. */
                    int actual = 0;
                    while (actual < count && actual < len) {
                        actual++;
                    }
                    len = actual;
                } else if (start >= len || start < 0) {
                    /* applyrule: if (clen > y) - false branch, no change. */
                }
                k += 3;
                break;
            }
            case RULE_OP_OMIT: {
                int pos = (int)prog[k + 1] - 1;
                int count = (int)prog[k + 2] - 1;
                if (pos >= 0 && pos < len && count > 0 && pos + count <= len) {
                    /* applyrule: for (x = y; x < clen && (x + z) < clen; x++)
                     *   cpass[x] = cpass[x + z]; clen = x;
                     * The loop terminates when x + z == clen, so x = clen - z,
                     * matching len - count. */
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

            /* ---- Special ops (still silent no-ops) ------------------- */
            case RULE_OP_S_SPECIAL:  k += 1; break;
            case RULE_OP_HASH_EXIT:  k += 1; break;
            case RULE_OP_DIV_INSERT: k += 3; break;  /* op + count + char */

            /* ---- Rejection ops (rev 1.22) ----------------------------
             * Match applyrule byte-exact (ruleproc.c:1410-1456). On
             * rejection we return -1 immediately; the caller is
             * responsible for skipping MD5+probe / writing zero digests
             * / emitting the validate sentinel record. The argument
             * encoding is `arg - 1` per packrules positiontranslate. */
            case RULE_OP_REJ_LEN_NE: {
                /* `_ N`: reject if orig_len != (arg - 1). applyrule
                 * tests the original input length, NOT the running clen. */
                int y = (int)prog[k + 1] - 1;
                if (y != orig_len) return -1;
                k += 2;
                break;
            }
            case RULE_OP_REJ_LEN_GE: {
                /* `< N`: reject if clen < (arg - 1). Macro name is
                 * misleading — applyrule rejects on LESS-THAN, not GE. */
                int y = (int)prog[k + 1] - 1;
                if (len < y) return -1;
                k += 2;
                break;
            }
            case RULE_OP_REJ_LEN_LE: {
                /* `> N`: reject if clen > (arg - 1). Macro name is
                 * misleading — applyrule rejects on GREATER-THAN. */
                int y = (int)prog[k + 1] - 1;
                if (len > y) return -1;
                k += 2;
                break;
            }
            case RULE_OP_REJ_HAS: {
                /* `! X`: reject if buf contains X. */
                uchar c = prog[k + 1];
                for (int j = 0; j < len; j++) {
                    if (buf[j] == c) return -1;
                }
                k += 2;
                break;
            }
            case RULE_OP_REJ_NHAS: {
                /* `/ X`: reject if buf does NOT contain X. */
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
                /* `( X`: reject if (clen > 0 && buf[0] != X). Empty buf
                 * is accepted (matches applyrule clen-guard). */
                uchar c = prog[k + 1];
                if (len > 0 && buf[0] != c) return -1;
                k += 2;
                break;
            }
            case RULE_OP_REJ_LAST: {
                /* `) X`: reject if (clen > 0 && buf[clen-1] != X). Empty
                 * buf is accepted (matches applyrule clen-guard). */
                uchar c = prog[k + 1];
                if (len > 0 && buf[len - 1] != c) return -1;
                k += 2;
                break;
            }

            /* H/h hex emit: cases split into independent bodies (rev 1.27,
             * 2026-05-04). The unified version at rev 1.25 used a ternary-
             * selected `const uchar *d` pointing at one of two private-memory
             * tables; AMD's ROCm comgr LLVM 21.1.8 bitcode linker rejected the
             * resulting addrspace(5) ↔ generic pointer aliasing with
             * "Invalid record" / "Linking bitcode failed" (gfx1201 RX 9070).
             * NVIDIA PTX path was unaffected (no IR-level link). Bisect on
             * ioblade pinned the regression to r25 vs r24. Splitting the
             * cases removes the pointer alias — each branch references its
             * own private-memory table directly, no addrspace cast. */
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
                /* Unknown opcode — bail to avoid runaway walk. */
                return len;
        }
    }
    return len;
}

/* ---- Multi-block MD5 of a private byte buffer --------------------- */

/* MD5 over (data, len) where data is private memory. Up to 5 blocks
 * (256 B + 9 padding bytes -> ceil((256+9)/64) = 5).
 *
 * Builds each 64-byte block from the byte stream, dispatches to
 * md5_block(&hx,&hy,&hz,&hw, M). The final block(s) include the 0x80
 * padding marker and the 64-bit little-endian length-in-bits at
 * M[14]/M[15]. If len%64 >= 56 the length doesn't fit in the final
 * data block and we emit one extra padding-only block.
 */
static void md5_buf(const uchar *data, int len,
                    uint *hx, uint *hy, uint *hz, uint *hw)
{
    uint M[16];
    int pos = 0;

    *hx = 0x67452301u;
    *hy = 0xEFCDAB89u;
    *hz = 0x98BADCFEu;
    *hw = 0x10325476u;

    /* Process complete 64-byte blocks. */
    while (len - pos >= 64) {
        for (int j = 0; j < 16; j++) {
            int b = pos + j * 4;
            M[j] = (uint)data[b]
                 | ((uint)data[b + 1] << 8)
                 | ((uint)data[b + 2] << 16)
                 | ((uint)data[b + 3] << 24);
        }
        md5_block(hx, hy, hz, hw, M);
        pos += 64;
    }

    /* Build final block(s): tail bytes + 0x80 marker + zeros + length. */
    int rem = len - pos;  /* 0..63 */

    /* Zero scratch. */
    for (int j = 0; j < 16; j++) M[j] = 0;

    /* Copy remaining tail bytes into M[] little-endian. */
    for (int i = 0; i < rem; i++) {
        uint v = (uint)data[pos + i];
        M[i >> 2] |= v << ((i & 3) * 8);
    }
    /* 0x80 padding marker. */
    M[rem >> 2] |= (uint)0x80 << ((rem & 3) * 8);

    if (rem < 56) {
        /* Length fits in this block. */
        M[14] = (uint)(len * 8);
        M[15] = 0;
        md5_block(hx, hy, hz, hw, M);
    } else {
        /* Need one extra padding-only block to hold the length. */
        md5_block(hx, hy, hz, hw, M);
        for (int j = 0; j < 16; j++) M[j] = 0;
        M[14] = (uint)(len * 8);
        M[15] = 0;
        md5_block(hx, hy, hz, hw, M);
    }
}

/* ---- Production kernel --------------------------------------------
 *
 * Memo B B1 (2026-05-03): coalesced dispatch payload.
 *
 * The four per-dispatch host->GPU writes (b_packed_buf + b_chunk_index
 * + b_params + b_hit_count) are now ONE write into b_dispatch_payload
 * with the layout:
 *
 *   offset   0 : OCLParams params           (sizeof(OCLParams) == 128)
 *   offset 128 : uint hit_count             (4 bytes; kernel atomic_inc)
 *   offset 132 : uint word_offset[num_words] (deterministic from params.num_words)
 *   offset 132 + 4*num_words : uchar packed_words[]  (variable; up to 4 MiB)
 *
 * Offsets are deterministic (computable from params.num_words alone),
 * so no extra fields are stored. params.num_words is the only knob.
 *
 * Per project_gpu_pcie_baseline_20260427.md, this saves the per-call
 * SUBMIT->START tax (4 x 2.3 ms = ~9.2 ms on PCIe 3.0; ~190 us on PCIe
 * 4.0) by collapsing 4 small/medium writes into 1 medium write. The
 * pinning win from malloc_pinned host source is preserved (the staging
 * buffer the host packs into is still malloc_pinned'd; see
 * gpu_opencl_dispatch_md5_rules in gpu_opencl.c).
 *
 * The test/validate kernels (md5_rules_phase0_test*, *_validate) keep
 * the multi-arg signature -- they're diagnostic / harness paths, not
 * on the production hot loop. */

__kernel
void md5_rules_phase0(
    __global uchar        *payload,         /* coalesced: params + hit_count + word_offset + packed */
    __global const uchar  *rule_program,    /* compiled rule bytecode (NUL-term) */
    __global const uint   *rule_offset,     /* byte offsets per rule */
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
    __global volatile uint *hashes_shown
    )
{
    /* Decode payload header. params struct sits at offset 0; copy to
     * private memory so subsequent field accesses stay in registers. */
    __global const OCLParams *params_buf = (__global const OCLParams *)payload;
    OCLParams params = *params_buf;
    uint n_words = params.num_words;
    uint n_rules = params.num_masks;
    uint total = n_words * n_rules;

    uint gid = get_global_id(0);
    if (gid >= total) return;

    uint word_idx = gid % n_words;
    uint rule_idx = gid / n_words;

    /* B3 cursor check (Memo B §2 two-cursor protocol).
     *
     * On a re-issue dispatch following a hit-buffer overflow, the host
     * sets input_cursor_start + rule_cursor_start to the (word, rule)
     * coordinate of the first overflowing lane from the prior dispatch.
     * Lanes whose (rule, word) lex-precedes the cursor early-return so
     * we don't re-emit the hits already returned in the prior dispatch.
     *
     * Lex order: (rule_idx, word_idx). A lane is "before the cursor" iff:
     *   rule_idx < rule_cursor_start, OR
     *   (rule_idx == rule_cursor_start AND word_idx < input_cursor_start)
     *
     * Default behavior: cursor=0 => no lanes skip, identical to pre-B3. */
    if (params.input_cursor_start > 0u || params.rule_cursor_start > 0u) {
        if (rule_idx < params.rule_cursor_start) return;
        if (rule_idx == params.rule_cursor_start &&
            word_idx < params.input_cursor_start) return;
    }

    /* Deterministic sub-buffer pointers from params.num_words.
     *   hit_count  @ offset 128
     *   word_offset @ offset 132
     *   packed_words @ offset 132 + 4*n_words
     * The compiler hoists these out of the per-lane work since they
     * depend only on params (uniform across the dispatch). */
    __global volatile uint *hit_count = (__global volatile uint *)(payload + 128);
    __global const uint   *word_offset = (__global const uint *)(payload + 132);
    uint pkt_off = 132u + (n_words * 4u);
    __global const uchar  *words = payload + pkt_off;

    /* B3: pointers to overflow_first_set (offset 100) and
     * overflow_first_word (offset 104). Used by the overflow-aware
     * EMIT_HIT_4_OR_OVERFLOW macro to record the first lane that
     * overflowed the hit buffer. The kernel writes overflow_first_word
     * as the lane gid (uint32, monotonic in (rule_idx, word_idx) lex
     * order); host re-derives word/rule via gid %/n_words and gid /
     * n_words. */
    __global volatile uint *ovr_set =
        (__global volatile uint *)(payload + 100);
    __global volatile uint *ovr_gid =
        (__global volatile uint *)(payload + 104);

    /* Single private buffer, 16-byte aligned for any future vectorized
     * op fast-paths (current walker is scalar in-place). */
    __attribute__((aligned(16))) uchar buf[RULE_BUF_MAX];

    uint wpos = word_offset[word_idx];
    int wlen = (int)words[wpos++];
    if (wlen > RULE_BUF_LIMIT) wlen = RULE_BUF_LIMIT;
    for (int i = 0; i < wlen; i++) buf[i] = words[wpos + i];

    uint rpos = rule_offset[rule_idx];
    /* Fast detect: if first byte is NUL, this is the synthetic no-rule
     * pass (k == 0 at entry). apply_rule returns wlen unmodified. We
     * MUST run MD5+probe for the synthetic pass — it IS the no-rule pass. */
    int is_no_rule = (rule_program[rpos] == 0);
    int new_len = apply_rule(rule_program + rpos, buf, wlen);

    /* Rejection sentinel: rule fired `_ < > ! / ( )` and rejected this
     * (word, rule) pair. Skip MD5+probe entirely. */
    if (new_len < 0) return;

    /* No-op detection: if at least one op was processed AND the post-rule
     * buffer matches the original input bit-for-bit, the synthetic ":"
     * no-rule pass already covered this candidate — skip MD5+probe.
     *
     * Compare buf[0..new_len-1] to words[wpos..wpos+wlen-1]. */
    if (!is_no_rule && new_len == wlen) {
        int changed = 0;
        for (int i = 0; i < wlen; i++) {
            if (buf[i] != words[wpos + i]) { changed = 1; break; }
        }
        if (!changed) return;
    }

    /* --- Multi-block MD5 of buf[0..new_len). --------------------- */
    uint hx, hy, hz, hw;
    md5_buf(buf, new_len, &hx, &hy, &hz, &hw);

    /* --- Iterated probe (mirrors gpu_md5_packed.cl). ------------- */
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
            /* B3 overflow-aware dedup-aware emit (Memo B §2). On overflow,
             * lane CAS-min's its gid into ovr_gid, signals ovr_set, AND
             * rolls back the hashes_shown bit it just set so the re-issue
             * lane can re-emit. Without the rollback, the dedup bit stays
             * set across dispatches and silently drops the crack. */
            EMIT_HIT_4_DEDUP_OR_OVERFLOW(hits, hit_count, params.max_hits,
                       word_idx, rule_idx, iter, hx, hy, hz, hw,
                       hashes_shown, matched_idx, mask,
                       ovr_set, ovr_gid, gid);
        }
        if (iter < max_iter) {
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
}

/* ---- TEST kernel: emits MD5 digest per (word, rule) -------------- */

__kernel
void md5_rules_phase0_test(
    __global const uchar  *words,
    __global const uint   *word_offset,
    __global const uchar  *rule_program,
    __global const uint   *rule_offset,
    __global const OCLParams *params_buf,
    __global uint         *digests_out)
{
    OCLParams params = *params_buf;
    uint n_words = params.num_words;
    uint n_rules = params.num_masks;
    uint total = n_words * n_rules;
    uint gid = get_global_id(0);
    if (gid >= total) return;

    uint word_idx = gid % n_words;
    uint rule_idx = gid / n_words;

    __attribute__((aligned(16))) uchar buf[RULE_BUF_MAX];
    uint wpos = word_offset[word_idx];
    int wlen = (int)words[wpos++];
    if (wlen > RULE_BUF_LIMIT) wlen = RULE_BUF_LIMIT;
    for (int i = 0; i < wlen; i++) buf[i] = words[wpos + i];

    uint rpos = rule_offset[rule_idx];
    int new_len = apply_rule(rule_program + rpos, buf, wlen);

    /* Rejection sentinel: write all-zero digest. The host test harness
     * treats four-zeros as the rejection marker for this kernel. */
    if (new_len < 0) {
        digests_out[gid * 4 + 0] = 0u;
        digests_out[gid * 4 + 1] = 0u;
        digests_out[gid * 4 + 2] = 0u;
        digests_out[gid * 4 + 3] = 0u;
        return;
    }

    uint hx, hy, hz, hw;
    md5_buf(buf, new_len, &hx, &hy, &hz, &hw);

    digests_out[gid * 4 + 0] = hx;
    digests_out[gid * 4 + 1] = hy;
    digests_out[gid * 4 + 2] = hz;
    digests_out[gid * 4 + 3] = hw;
}

/* ---- VALIDATE kernel: emits per-(word,rule) post-rule buffer state -- */

/* Validator wire format (rev 1.23, 2026-05-02 field widening):
 *   slot[0..1] = retlen as int16 little-endian (sign-preserving;
 *                values: -1 reject, -2 auto-skip, -3 fatal, >=0 length)
 *   slot[2..3] = outlen as uint16 little-endian
 *   slot[4..3+RULE_BUF_MAX] = post-rule buffer bytes
 * Record size = 4 + RULE_BUF_MAX. With RULE_BUF_MAX=40960 → 40964-byte slots.
 * Host-side mirror constants in gpu_opencl.h MUST match. */
#define MD5_RULES_VALIDATE_MAX_BUF   RULE_BUF_MAX
#define MD5_RULES_VALIDATE_RECORD_SZ (4 + MD5_RULES_VALIDATE_MAX_BUF)

__kernel
void md5_rules_phase0_validate(
    __global const uchar  *words,
    __global const uint   *word_offset,
    __global const uchar  *rule_program,
    __global const uint   *rule_offset,
    __global const OCLParams *params_buf,
    __global uchar        *records_out)
{
    OCLParams params = *params_buf;
    uint n_words = params.num_words;
    uint n_rules = params.num_masks;
    uint total = n_words * n_rules;
    uint gid = get_global_id(0);
    if (gid >= total) return;

    uint word_idx = gid % n_words;
    uint rule_idx = gid / n_words;

    __attribute__((aligned(16))) uchar buf[RULE_BUF_MAX];
    uint wpos = word_offset[word_idx];
    int wlen = (int)words[wpos++];
    if (wlen > RULE_BUF_LIMIT) wlen = RULE_BUF_LIMIT;
    for (int i = 0; i < wlen; i++) buf[i] = words[wpos + i];

    uint rpos = rule_offset[rule_idx];
    int new_len = apply_rule(rule_program + rpos, buf, wlen);

    /* Rejection sentinel: retlen=-1, outlen=0, no bytes. The host parser
     * (gpu_opencl.c rev 1.72+) reads retlen as int16 LE and outlen as
     * uint16 LE; the R2 reconciliation rule from #65 maps CPU retlen ∈
     * {-1, -3} to effective_output="" so the byte-exact diff lines up.
     *
     * Field widening rev 1.23 (2026-05-02): retlen / outlen widened from
     * 8-bit to 16-bit LE so outputs > 255 bytes are recorded unclamped.
     * The prior `if (retlen > 255) retlen = 255` clamp is removed; int16
     * range -32768..32767 covers all RULE_BUF_LIMIT values up to 40959. */
    int retlen = new_len;
    int outlen = (retlen >= 0) ? retlen : 0;

    uint slot = word_idx * n_rules + rule_idx;
    uint base = slot * (uint)MD5_RULES_VALIDATE_RECORD_SZ;

    /* retlen as int16 LE (sign-preserving via two-byte split of int16 cast). */
    short retlen16 = (short)retlen;
    records_out[base + 0] = (uchar)((uint)retlen16 & 0xffu);
    records_out[base + 1] = (uchar)(((uint)retlen16 >> 8) & 0xffu);
    /* outlen as uint16 LE. */
    ushort outlen16 = (ushort)outlen;
    records_out[base + 2] = (uchar)((uint)outlen16 & 0xffu);
    records_out[base + 3] = (uchar)(((uint)outlen16 >> 8) & 0xffu);
    for (int i = 0; i < outlen && i < MD5_RULES_VALIDATE_MAX_BUF; i++) {
        records_out[base + 4 + (uint)i] = buf[i];
    }
}

/* ---- TEST kernel (iter variant): emits per-iteration digests --------
 *
 * md5_rules_phase0_test_iter — sibling test kernel that exercises the
 * production md5_rules_phase0 iteration loop. Same (word, rule) input
 * geometry as md5_rules_phase0_test, but writes ONE digest PER iteration
 * level into digests_out[(gid * max_iter + (iter-1)) * 4 .. +3]. */
__kernel
void md5_rules_phase0_test_iter(
    __global const uchar  *words,
    __global const uint   *word_offset,
    __global const uchar  *rule_program,
    __global const uint   *rule_offset,
    __global const OCLParams *params_buf,
    __global uint         *digests_out)
{
    OCLParams params = *params_buf;
    uint n_words = params.num_words;
    uint n_rules = params.num_masks;
    uint total = n_words * n_rules;
    uint gid = get_global_id(0);
    if (gid >= total) return;

    uint word_idx = gid % n_words;
    uint rule_idx = gid / n_words;

    __attribute__((aligned(16))) uchar buf[RULE_BUF_MAX];
    uint wpos = word_offset[word_idx];
    int wlen = (int)words[wpos++];
    if (wlen > RULE_BUF_LIMIT) wlen = RULE_BUF_LIMIT;
    for (int i = 0; i < wlen; i++) buf[i] = words[wpos + i];

    uint rpos = rule_offset[rule_idx];
    int new_len = apply_rule(rule_program + rpos, buf, wlen);

    /* Rejection sentinel: zero all per-iteration slots. Same convention
     * as the non-iter test kernel — host harness treats four-zeros per
     * iteration level as the rejection marker. */
    uint max_iter_r = params.max_iter;
    if (max_iter_r < 1) max_iter_r = 1;
    if (new_len < 0) {
        for (uint iter = 1; iter <= max_iter_r; iter++) {
            uint base = (gid * max_iter_r + (iter - 1u)) * 4u;
            digests_out[base + 0] = 0u;
            digests_out[base + 1] = 0u;
            digests_out[base + 2] = 0u;
            digests_out[base + 3] = 0u;
        }
        return;
    }

    uint hx, hy, hz, hw;
    md5_buf(buf, new_len, &hx, &hy, &hz, &hw);

    uint max_iter = max_iter_r;
    for (uint iter = 1; iter <= max_iter; iter++) {
        uint base = (gid * max_iter + (iter - 1u)) * 4u;
        digests_out[base + 0] = hx;
        digests_out[base + 1] = hy;
        digests_out[base + 2] = hz;
        digests_out[base + 3] = hw;
        if (iter < max_iter) {
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
}
