/*
 * $Revision: 1.1 $
 * $Log: metal_md5_rules.metal,v $
 * Revision 1.1  2026/05/12 16:40:56  dlr
 * Initial check-in: Phase 2a Metal port of the rules-engine walker. Ports apply_rule from gpu_md5_rules.cl (1385 LOC OpenCL twin) to Metal; KERNEL BODY is NOT ported (kernel bodies live in metal_template.metal; main session extends template_phase0 with a GPU_TEMPLATE_HAS_RULES block that calls apply_rule from this TU). All 48 RULE_OP_ opcodes covered (0xc1..0xff range plus variable-length 0xff/0xfe affix ops). Rejection semantics mirror OpenCL twin: ops underscore-LT-GT-bang-slash-paren-paren return -1 to signal lane skip. Pattern 1 (every kernel-arg buffer is device qualified; every lane-local pointer is thread qualified). Pattern 3 (all helpers static inline). Scalar-only port for Phase 2a; vector heads from OpenCL twin (LOWER, UPPER, CAP, CAP_INV, TOGGLE, SUB, DUP, DUP_LAST) deferred to Phase 2c after correctness-proven landing. case_flip_mask helper local to walker (Phase 2c+ may hoist if other family kernels need it). xcrun metal compile clean on iMac AMD Radeon Pro 580X; metal_jit_harness --check-patterns reports 0 violations across the 4-file family TU (metal_common + metal_md5_core + metal_md5_rules + metal_template). 804 LOC including header + helper.
 *
 *
 */
/* metal_md5_rules.metal — Phase 2a Metal port of the rules-engine walker.
 *
 * Ports the apply_rule() function from gpu/gpu_md5_rules.cl (1385 LOC OpenCL
 * twin) to Metal. The KERNEL BODY is NOT ported here — kernel bodies live in
 * metal_template.metal (the unified template_phase0 kernel); main session
 * extends template_phase0 with a GPU_TEMPLATE_HAS_RULES block that calls
 * apply_rule() from this TU.
 *
 * Phase 2a scope (memo §3 row 2):
 *   - Single-buffer in-place per-op semantics. apply_rule(prog, buf, len)
 *     walks the bytecode and mutates buf in place.
 *   - All 48 RULE_OP_* opcodes from gpu_md5_rules.cl (0xc1..0xff range
 *     plus the variable-length 0xff/0xfe affix ops).
 *   - Rejection semantics: ops `_ < > ! / ( )` return -1 to signal lane
 *     skip (no MD5 + probe).
 *
 * Pattern enforcement (memo §12 + Phase 0.5):
 *   Pattern 1: `prog` is `device const uchar *`, `buf` is `device uchar *`
 *              (post task #250 scratch-pool migration — see file header §3),
 *              return value is plain int (lane-private).
 *   Pattern 3: function is `static inline`. The walker is large enough
 *              that the compiler may decline to inline; that is fine —
 *              the keyword still gives the linker a private symbol and
 *              keeps the kernel TU self-contained.
 *
 * Concatenation order (build_metallib.sh per-family):
 *   metal_common.metal  (MetalParams, md5_block, probe_compact_idx,
 *                        RULE_BUF_MAX / RULE_BUF_LIMIT, HIT_STRIDE)
 *   metal_md5_core.metal (template_state, template_init/_finalize/
 *                         _digest_compare/_emit_hit_or_overflow,
 *                         template_iterate stub)
 *   metal_md5_rules.metal (THIS FILE — apply_rule)
 *   metal_template.metal  (template_phase0 kernel; calls apply_rule
 *                         inside the GPU_TEMPLATE_HAS_RULES block —
 *                         extension is main-session scope per memo
 *                         §3 row 4)
 *
 * --- Byte-exactness contract ---
 *
 * apply_rule() in this file MUST produce identical output to applyrule()
 * in ruleproc.c (CPU reference) AND apply_rule() in gpu_md5_rules.cl
 * (OpenCL twin) for every bytecode sequence the host emits. The walker
 * is the single source of truth for CPU<->GPU parity on the rules axis.
 *
 * Per memo §6 validation gate 2a-1: byte-exact parity check is `-M e1
 * -r best64.rule rockyou-1k.txt against 1k-hash compact` — hit count +
 * sorted hit md5sum must be identical CPU/Metal.
 *
 * --- Differences from OpenCL twin ---
 *
 *   1. Address space: __global -> device, __private -> thread.
 *      Every `__global const uchar *prog` becomes `device const uchar *prog`;
 *      `uchar *buf` (implicit __private in OpenCL) becomes `device uchar *buf`
 *      since task #250 — see file header §3. The OpenCL twin keeps buf in
 *      __private because OpenCL drivers spill private memory to global
 *      transparently; Apple Metal does NOT spill — a 40 KB thread-local
 *      array consumes registers + temporary-register budget, and on M2
 *      Max blows the PSO-create register-allocator gate. Moving to
 *      `device uchar *buf` keeps RULE_BUF_MAX at 40 KB without truncation.
 *      The `len` parameter is plain int (pass-by-value).
 *
 *   2. Vector intrinsics: OpenCL uses uchar16 / vload16 / vstore16 / select
 *      for vectorized case ops (LOWER, UPPER, CAP, CAP_INV, TOGGLE, SUB,
 *      DUP, DUP_LAST). Metal has uchar16, vload, vstore equivalents but
 *      via different syntax. For Phase 2a we use the SCALAR fallback path
 *      from the OpenCL twin (each case op has a scalar inner loop after
 *      the vector head — Phase 2c may add vector fast-paths after JIT
 *      harness Pattern 1+3 selftest verifies them).
 *
 *      Rationale: byte-exact parity must land first. The scalar loops in
 *      the OpenCL twin are correctness-proven against ruleproc.c
 *      applyrule(); the vector heads are a perf optimization on top.
 *      Phase 2a takes correctness; Phase 2c+ adds the vector heads if
 *      profiling shows we need them.
 *
 *   3. select(): OpenCL's `select(a, b, c)` -> Metal's `(c ? b : a)` (or
 *      `select(a, b, c)` if Metal has the function — checked metal3.0;
 *      yes it does, same arg order as OpenCL). The retired Phase 0
 *      design used Metal's select() directly; Phase 2a keeps the
 *      explicit ternary form everywhere for clarity (the OpenCL twin
 *      already noted the select() MSB-predicate gotcha and uses ternaries
 *      for the same reason — we mirror that choice).
 *
 *   4. case_flip_mask: identical between OpenCL and Metal. Defined here
 *      (NOT in metal_common.metal) because the walker is the only caller
 *      Phase 2a — Phase 2c+ may hoist it if other family kernels start
 *      using it.
 */

/* ==== Opcode definitions (must match ruleproc.c RULE_OP_* + gpu_md5_rules.cl) ==== */
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

/* RULE_BUF_MAX / RULE_BUF_LIMIT live in metal_common.metal (~line 92);
 * we reference, not redefine. Mirrors OpenCL gpu_md5_rules.cl which also
 * pulls these from gpu_common.cl. */

/* ==== Branchless case-flip helper ====================================
 *
 * case_flip_mask(c) returns 0x20 if `c` is alphabetic (A-Z or a-z),
 * else 0. Uses the unsigned-subtract trick: `(c | 0x20) - 'a' < 26u`.
 * The OR lifts uppercase into lowercase territory (or leaves lowercase
 * unchanged); the unsigned compare is true only for 'a'..'z'.
 *
 * Robust across the entire 0x00..0xff input space (mirrors gpu_md5_-
 * rules.cl rev 1.30 helper at line 149).
 *
 * Cast the subtraction result back to uchar so the unsigned wrap kicks
 * in — Metal promotes uchar to int in expressions same as OpenCL.
 */
static inline uchar case_flip_mask(uchar c) {
    uchar v = (uchar)((c | (uchar)0x20) - (uchar)'a');
    return (uchar)((v < (uchar)26) ? 0x20 : 0);
}

/* ==== Unified single-buffer in-place rule walker =====================
 *
 * apply_rule:
 *   - prog: device pointer to NUL-terminated bytecode (offset already
 *           applied by caller; first byte is the first opcode).
 *   - buf:  device uchar slice of size RULE_BUF_MAX (task #250 scratch
 *           pool — gpu_metal.m buf_scratch_pool, indexed by word_idx).
 *           Pre-staged with the input word in buf[0..len-1].
 *   - len:  initial length of the staged input.
 * Returns: post-rule length (>= 0) on success, -1 if a rejection op fires.
 *          The buffer is modified in place.
 *
 * Rejection semantics mirror the OpenCL twin (rev 1.30 line 167):
 * ops `_ < > ! / ( )` return -1; the caller (template_phase0 in
 * metal_template.metal) must skip MD5+probe.
 *
 * The `_` op tests length-not-equal against the ORIGINAL input length
 * (the `len` parameter at function entry), NOT the running len. We
 * capture orig_len at entry to match applyrule byte-exact.
 *
 * The outer loop bound `for (int n = 0; n < 256; ...)` mirrors the
 * OpenCL twin: matches the per-rule opcode cap that ruleproc.c emits.
 */
static int apply_rule(device const uchar *prog, device uchar *buf, int len)
{
    int k = 0;
    int orig_len = len;     /* preserved for the `_ N` length-equal test */

    for (int n = 0; n < 256; n++) {
        uchar op = prog[k];
        if (op == 0) break;

        /* Main switch — contiguous high-bit opcodes 0xc1..0xff.
         * 0xff/0xfe are variable-length (op + N + N data bytes); the rest
         * are fixed-size. Scalar-only port for Phase 2a; vector heads from
         * the OpenCL twin (LOWER, UPPER, CAP, CAP_INV, TOGGLE, SUB, DUP,
         * DUP_LAST) are deferred — see file header §3 (scalar paths are
         * correctness-proven). */
        switch (op) {

            /* ---- Variable-length affix ops (0xff / 0xfe) ---- */
            case 0xff: {
                /* Multi-char append: 2 + N bytes, N data bytes follow N-byte. */
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

            /* ---- Per-position arithmetic ---- */
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

            /* ---- Substitute (scalar; OpenCL twin has uchar16 head) ---- */
            case RULE_OP_SUB: {
                uchar c1 = prog[k + 1], c2 = prog[k + 2];
                for (int j = 0; j < len; j++) {
                    if (buf[j] == c1) buf[j] = c2;
                }
                k += 3;
                break;
            }

            /* ---- Whole-string case ops (scalar) ---- */
            case RULE_OP_LOWER: {
                for (int j = 0; j < len; j++) {
                    uchar c = buf[j];
                    if (c >= 'A' && c <= 'Z') buf[j] = c ^ (uchar)0x20;
                }
                k += 1;
                break;
            }
            case RULE_OP_UPPER: {
                for (int j = 0; j < len; j++) {
                    uchar c = buf[j];
                    if (c >= 'a' && c <= 'z') buf[j] = c ^ (uchar)0x20;
                }
                k += 1;
                break;
            }
            case RULE_OP_CAP: {
                /* Capitalize: lowercase all, then upper-case the first
                 * lowercase letter. Matches ruleproc.c. */
                for (int j = 0; j < len; j++) {
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
                /* Anti-capitalize: uppercase all, then lower-case first
                 * uppercase letter. */
                for (int j = 0; j < len; j++) {
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
                for (int j = 0; j < len; j++) {
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
                    for (int j = 0; j < tlen; j++) buf[len + j] = buf[j];
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
                    for (int j = 0; j < n2; j++) buf[len + j] = last;
                    len += n2;
                }
                k += 2;
                break;
            }
            case RULE_OP_DUP_FIRST: {
                int n2 = (int)prog[k + 1] - 1;
                if (len > 0 && n2 > 0 && len + n2 <= RULE_BUF_LIMIT) {
                    uchar first = buf[0];
                    /* Shift right by n2 (mirror OpenCL twin's loop). */
                    for (int j = len - 1; j > 0; j--) buf[j + n2] = buf[j];
                    /* buf[0] preserved (it's the source we copy from);
                     * fill buf[1..n2] with first. */
                    for (int j = 1; j <= n2; j++) buf[j] = first;
                    len += n2;
                }
                k += 2;
                break;
            }
            case RULE_OP_REPEAT: {
                /* p N: append N copies of input to itself. */
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
                /* y N: duplicate first N chars at the start. */
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
                /* Y N: duplicate last N chars at the end. */
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
                /* xAB: out = in[A..A+B-1] capped at len-A. */
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
                /* `_ N`: reject if orig_len != (arg - 1). */
                int y = (int)prog[k + 1] - 1;
                if (y != orig_len) return -1;
                k += 2;
                break;
            }
            case RULE_OP_REJ_LEN_GE: {
                /* `< N`: reject if len < (arg - 1). */
                int y = (int)prog[k + 1] - 1;
                if (len < y) return -1;
                k += 2;
                break;
            }
            case RULE_OP_REJ_LEN_LE: {
                /* `> N`: reject if len > (arg - 1). */
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
                /* `( X`: reject if (len > 0 && buf[0] != X). */
                uchar c = prog[k + 1];
                if (len > 0 && buf[0] != c) return -1;
                k += 2;
                break;
            }
            case RULE_OP_REJ_LAST: {
                /* `) X`: reject if (len > 0 && buf[len-1] != X). */
                uchar c = prog[k + 1];
                if (len > 0 && buf[len - 1] != c) return -1;
                k += 2;
                break;
            }

            /* ---- H/h hex emit ---- */
            case RULE_OP_HEX_UPPER: {
                /* Mirror OpenCL twin's per-case private table (rev 1.27;
                 * AMD ROCm bitcode-linker issue is OpenCL-specific, but
                 * we keep the same split-cases structure for parity). */
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
                /* Unknown opcode — bail to avoid runaway walk. Mirrors
                 * OpenCL twin: returns len (NOT -1; rejection is reserved
                 * for known rejection-class opcodes). */
                return len;
        }
    }
    return len;
}
