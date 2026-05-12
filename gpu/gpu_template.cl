/*
 * $Revision: 1.16 $
 * $Log: gpu_template.cl,v $
 * Revision 1.16  2026/05/11 14:16:27  dlr
 * Phase 1.8 kernel-side REVERT (gpu_template.cl): remove the runtime-bounded `for (uint iter_idx = 0u; iter_idx < inner_iter; iter_idx++)` wrap at line ~439 (and matching close at ~701). Caused 2.19× Pascal kernel regression (NVCC register pressure spike on inner_iter=1 codegen path; whole-kernel register allocator forced worst-case allocation). Confirmed via task #239 mdx-debug spike on JOB_MD5 BF (Phase 1.5 1.18 GH/s → Phase 1.8 0.54 GH/s on fpga 1080). User observed e31 regression which traced to same root cause — every algo using gpu_template.cl (incl JOB_MD5SALT) inherited the wrap. Phase 1.8c body-duplication attempt (Phase 0 defer) confirmed whole-kernel reg-alloc scoping was the issue. Also removed iter_idx variable, `if (iter_idx > 0u) { ... }` save/restore block, and Phase 1.8 mask_idx_abs += iter_idx*mask_size shift (now mask_idx_abs = mask_idx_abs_base directly). Combined_ridx unsalted encoding reverted to Phase 1.7 form `rule_idx * mask_size + mask_idx_local`. Phase 1.5 ulong widening of mask_idx_abs PRESERVED (correctness for >4G keyspaces). LOC delta -35 (769 → 734). Salted combined_ridx unchanged (host servo guards salted with inner_iter=1). Validated on fpga 1080: V1 raw MD5 BF 1.021 GH/s (restored Phase 1.7 baseline); V2 e31 MD5SALT 4-digit 3/3 cracks 0.03s; V3 non-BF wordlist 3/3; V4 10-digit BF (>4G keyspace) 3/3 cracks at 1.897 GH/s — Phase 1.5 widening preserved.
 *
 * Revision 1.15  2026/05/11 05:22:02  dlr
 * Backfill $Revision/$Log RCS keyword stanzas per feedback_rcs_keyword_stanzas.md. Passive 4-line comment block at top of file; no behavioral change. Hand-authored .cl file was missing required stanzas (per memory: all hand-authored .c/.h/.cl/.frag/.tmpl/.py/.sh files MUST contain $Revision/$Log keyword stanzas). Build green on .205 against the post-add files; OpenCL compile strips comments so no kernel behavior change.
 *
 */
/* gpu_template.cl — generic dispatch-template kernel skeleton (Memo B
 * Phase B2).
 *
 * STATUS: B2 — STRUCTURAL PREREQUISITE, default OFF, MD5-only.
 *
 * The template body provides the SHARED infrastructure that every
 * algorithm template-instantiated under Memo B Phase B4-B7 will use:
 *
 *   - payload decode (B1 coalesced wire format from
 *     project_memo_b_dispatch_template.md §1)
 *   - geometry (gid -> (word_idx, rule_idx))
 *   - rule walker (apply_rule from gpu_md5_rules.cl, shared verbatim)
 *   - rejection + no-op detection (preserves feedback_no_rule_pass.md)
 *   - iterated probe loop (drives template_digest_compare in a
 *     loop with template_iterate steps)
 *   - cursor skeleton (read but unused in B2; B3 wires real restart
 *     logic per Memo B §2)
 *   - overflow back-pressure (today's "warn at 90%, drop" semantics
 *     only -- full grow/split/drain is B3)
 *
 * The template invokes per-algorithm extension functions defined in
 * gpu_<algo>_core.cl (this build: gpu_md5_core.cl):
 *
 *   template_state              -- algorithm-defined state struct
 *   template_init               -- algorithm-defined IV install
 *   template_transform          -- per-block compress
 *   template_finalize           -- pad + finalize
 *   template_iterate            -- -i loop step (hex_lc + rehash)
 *   template_digest_compare     -- probe wrapper
 *   template_emit_hit           -- HASH_WORDS-aware EMIT_HIT_N wrapper
 *   HASH_BLOCK_BYTES, HASH_WORDS -- compile-time geometry
 *
 * Build sources passed to clCreateProgramWithSource (in this order):
 *
 *   gpu_common_str        -- md5_block, EMIT_HIT_*, OCLParams,
 *                            probe_compact_idx, hex helpers
 *   gpu_md5_rules_str     -- apply_rule (shared rules walker)
 *   gpu_md5_core_str      -- algorithm geometry + 4 extension fns
 *   gpu_template_str      -- this file (template_phase0 kernel)
 *
 * Cache key (R3 mitigation): gpu_kernel_cache_build_program is now
 * passed defines_str = "HASH_WORDS=4,HASH_BLOCK_BYTES=64" so distinct
 * algorithm tuples receive distinct cache keys even though the
 * gpu_template.cl source text is identical.
 *
 * Side-by-side with gpu_md5_rules.cl rev 1.28+: B2 ships this kernel
 * alongside md5_rules_phase0; the production path remains the latter
 * unless MDXFIND_GPU_TEMPLATE=md5 is set in the environment. Default
 * off keeps R2 (register pressure on AMD gfx1201) from blocking
 * merge -- B5 flips the default once enough algorithms are wired
 * through the template that the side-by-side kernels become carrying
 * cost.
 *
 * R1 mitigation (AMD ROCm comgr addrspace fragility): the template
 * keeps the single-private-buffer pattern from gpu_md5_rules.cl r28.
 * No __private uchar* helper functions; no addrspace-cast ternary
 * pointers. Inline modifications only.
 *
 * Kernel signature MUST MATCH md5_rules_phase0 (gpu_md5_rules.cl) so
 * gpu_opencl_dispatch_md5_rules can swap kernels by changing only the
 * cl_kernel handle. 14 args, payload first.
 */

/* Phase 6 BCRYPT (2026-05-08): the BCRYPT carrier needs a workgroup-shared
 * __local buffer (Eksblowfish S-boxes, 4 KB per lane × BCRYPT_WG_SIZE=8
 * lanes = 32 KB per WG) — slab gpu_bcrypt.cl precedent. OpenCL 1.2 §6.5.3
 * forbids `__local` declarations inside non-kernel functions, so we
 * declare it INSIDE template_phase0 (kernel-function scope) and pass it
 * as a __local pointer + per-lane lid into template_finalize. The 8-arg
 * salted template_finalize signature is gated on
 * GPU_TEMPLATE_HAS_LOCAL_BUFFER (set only by gpu_opencl_template_compile_-
 * bcrypt's build_opts -D list). All other algorithms (60+) compile
 * with the macro UNDEFINED — the preprocessor strips the ifdef block
 * entirely so their build output is byte-identical to pre-Phase 6.
 *
 * The reqd_work_group_size attribute pins WG=BCRYPT_WG_SIZE (8) at
 * kernel-decl time — required because the __local buffer is sized
 * relative to the WG, and the dispatch-side `local = BCRYPT_WG_SIZE`
 * override (gpu_opencl.c) must match. Mirror of slab gpu_bcrypt.cl
 * line 427. */

/* B1+lane-batch MD5SALT pre-salt hoist scaffold (2026-05-09 experiment):
 * introduces an optional dispatch shape gated behind GPU_TEMPLATE_HAS_PRE_SALT.
 * Purpose: lift the password-only portion of a salted hash (e.g. the
 * inner MD5(password) of MD5SALT's MD5(MD5(password) || salt)) OUT of the
 * per-(word,rule,mask,salt) inner loop and into a per-(word,rule,mask,
 * salt_chunk) pre-pass, where salt_chunk groups SALT_BATCH consecutive
 * salts. The salt axis is replaced with a salt_chunk axis of size
 * ceil(num_salts_per_page / SALT_BATCH); each lane processes SALT_BATCH
 * salts serially after computing inner+hex once. Tunable via
 * SALT_BATCH compile-time macro (default 16; pass via -DSALT_BATCH=N).
 *
 * Mutual exclusion: GPU_TEMPLATE_HAS_PRE_SALT is mutually exclusive with
 * GPU_TEMPLATE_HAS_LOCAL_BUFFER. It REQUIRES GPU_TEMPLATE_HAS_SALT.
 *
 * Hook signatures (defined by the algorithm core, e.g. gpu_md5salt_core.cl):
 *
 *   typedef struct { uint M[16]; uint inner_len; } template_pre_salt_state;
 *
 *   void template_pre_salt(uchar *buf, int new_len, uint algo_mode,
 *                          template_pre_salt_state *pre_state);
 *       For algo_modes that benefit from inner-MD5+hex hoisting (mode 0),
 *       computes the password-only digest and stores hex-encoded result in
 *       pre_state->M with pre_state->inner_len == 32 (or whatever inner
 *       layout the algorithm uses). For algo_modes that don't benefit
 *       (modes 1-6), sets pre_state->inner_len = 0xFFFFFFFFu (sentinel)
 *       so template_finalize_post falls through to the legacy path.
 *
 *   void template_finalize_post(template_state *st,
 *                               const template_pre_salt_state *pre_state,
 *                               const uchar *buf, int new_len,
 *                               __global const uchar *salt, uint slen,
 *                               uint algo_mode);
 *       For non-sentinel pre_state: completes outer-MD5(hex32 || salt).
 *       For sentinel pre_state: calls into legacy template_finalize().
 *
 * Per-dispatch wall-clock target 250-500 ms with num_salts_per_page +
 * SALT_BATCH sized so the inner serial loop dominates without any single
 * kernel exceeding 500 ms. The host (gpu_opencl.c) computes
 * chunks_per_page = (num_salts_per_page + SALT_BATCH - 1) / SALT_BATCH
 * for the NDRange salt axis; the kernel performs the same calc to bound
 * the inner loop.
 *
 * Scaffold-only: when GPU_TEMPLATE_HAS_PRE_SALT is undefined (every
 * algo today), the preprocessor strips every HAS_PRE_SALT-gated block —
 * non-opt-in algorithms compile and execute byte-identically to pre-B1. */
#ifndef SALT_BATCH
#define SALT_BATCH 16
#endif

#ifdef GPU_TEMPLATE_HAS_LOCAL_BUFFER
__kernel __attribute__((reqd_work_group_size(BCRYPT_WG_SIZE, 1, 1)))
#else
__kernel
#endif
void template_phase0(
    __global uchar        *payload,         /* coalesced: params + hit_count + word_offset + packed */
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
    /* B7.1-B7.5: multi-position prepend+append mask charsets. Layout is
     * row-major by position index, using the SLAB-PATH CONVENTION (see
     * gpu_kernels.cl md5_mask_batch / gpu_mask_desc): positions
     * [0..n_prepend) are prepend rows, positions [n_prepend..n_prepend+
     * n_append) are append rows. Row i at offset i*256 holds the 256-byte
     * charset table for that position; only the first mask_sizes[i] bytes
     * of each row are valid. Total capacity is MASK_POS_CAP=32 rows
     * (16 prepend + 16 append max), so the buffer is 32*256=8192 bytes.
     *
     * B7.1 was single-position append (only row 0 read). B7.2 extended to
     * multi-position append (rows [0..n_append) for the n_prepend==0 case).
     * B7.3 adds single-position prepend (row 0 holds the prepend charset
     * when n_prepend==1; n_append==0). B7.4 extends to multi-position
     * prepend (rows [0..n_prepend) for the n_append==0 case). B7.5 mixes
     * both: rows [0..n_prepend) prepend, rows [n_prepend..n_prepend+
     * n_append) append (matches slab convention exactly).
     *
     * For B7.2 backward-compat (n_prepend==0): row 0 still holds append
     * position 0's charset — the layout reduces to the B7.2 packing.
     * Byte-exact backward-compatible. */
    __global const uchar  *mask_charsets,
    /* B7.1-B7.5: per-position charset sizes. mask_sizes[i] gives the count
     * of valid bytes in mask_charsets[i*256..i*256+256). Only the first
     * (n_prepend + n_append) entries are read; the rest is sentinel of 1
     * (so any stray divmod terminates safely). The kernel uses mask_sizes[]
     * to decompose mask_idx into per-position indices via successive
     * divmod (last position innermost in each section). */
    __global const uint   *mask_sizes
    /* B6 salt-axis (2026-05-06): three appended args, gated behind
     * #ifdef GPU_TEMPLATE_HAS_SALT. Layout mirrors gpu_md5salt.cl slab path:
     *   salt_buf  -- concatenated salt bytes for the WHOLE salt list (not
     *                paged); salt N starts at salt_offsets[N], length is
     *                salt_lens[N]. The host uploads the full list once via
     *                gpu_opencl_set_salts() per batch.
     *   salt_off  -- per-salt byte offset into salt_buf, uint32 array.
     *   salt_len  -- per-salt byte length, uint16 (ushort) array
     *                (matches gpu_md5salt.cl:3 + host uint16_t at
     *                gpu_opencl.c:2580-2591 — see §13.3).
     *
     * The unsalted instantiation elides these args entirely; the kernel
     * signature stays at 16 args for the 32 already-validated cores.
     * Salted variant signature is 19 args. */
#ifdef GPU_TEMPLATE_HAS_SALT
    , __global const uchar  *salt_buf
    , __global const uint   *salt_off
    , __global const ushort *salt_lens
#endif
    )
{
    /* Decode payload header. params struct sits at offset 0; copy to
     * private memory so subsequent field accesses stay in registers. */
    __global const OCLParams *params_buf = (__global const OCLParams *)payload;
    OCLParams params = *params_buf;
    uint n_words = params.num_words;
    uint n_rules = params.num_masks;
    /* B7.1-B7.5 mask geometry: third axis is mask_idx in [0, mask_size).
     * When no mask is active mask_size == 1 and the kernel collapses to
     * the pre-B7 (word, rule) behavior — one-iteration mask loop, no
     * append, no prepend.
     *
     * mask_size is threaded through OCLParams.num_salts. The salts field
     * is unused in the rules-engine / template dispatch path (the slab
     * path's salted kernels have their own kern_packed_fam[] entries),
     * so we repurpose it as mask_size — same convention as num_masks
     * being repurposed as n_rules in this dispatch path. The host sets
     * num_salts = product(mask_sizes[0..n_prepend+n_append)) when MaskCount
     * is active and the (n_prepend, n_append) configuration is in scope
     * (B7.5 scope: n_prepend in [0, MASK_POS_CAP], n_append in [0,
     * MASK_POS_CAP], n_prepend + n_append >= 1, both bounded by
     * MASK_POS_CAP=16); otherwise num_salts = 1.
     *
     * B7.1: n_append==1 only.
     * B7.2: n_prepend==0, n_append in [1, MASK_POS_CAP].
     * B7.3: n_prepend==1, n_append==0.
     * B7.4: n_prepend in [1, MASK_POS_CAP], n_append==0.
     * B7.5: n_prepend in [0, MASK_POS_CAP], n_append in [0, MASK_POS_CAP],
     *       n_prepend + n_append >= 1.
     * B7.6: same as B7.5; user-defined classes are charset-byte changes
     *       in the upload, no kernel change.
     * B7.8: cap lifted from 8 to 16 per side (32 total rows). Same
     *       divmod loop, just larger bound. */
    uint mask_active = ((params.n_prepend >= 1u) || (params.n_append >= 1u)) ? 1u : 0u;
#ifdef GPU_TEMPLATE_HAS_SALT
    /* B6 salt-axis (2026-05-06): num_salts is overloaded as
     *   num_salts = mask_size * num_salts_per_page
     * so the host can carry both axes in a single uint32 field without
     * touching the OCLParams wire format. params.num_salts_per_page is
     * the new uint64 slot at offset 112 (was reserved64[0]). When
     * num_salts_per_page == 1 (no salt paging) and num_salts == mask_size
     * the math collapses to the pre-B6 layout. */
    uint num_salts_per_page = (uint)params.num_salts_per_page;
    if (num_salts_per_page == 0u) num_salts_per_page = 1u;
    uint mask_size = mask_active
        ? (params.num_salts / num_salts_per_page)
        : 1u;
    if (mask_size == 0u) mask_size = 1u;
#ifdef GPU_TEMPLATE_HAS_PRE_SALT
    /* Lane-level salt batching (2026-05-09 experiment): salt axis is
     * replaced by salt_chunk axis of size ceil(num_salts_per_page /
     * SALT_BATCH). Each lane processes SALT_BATCH salts serially after
     * a single template_pre_salt() call, amortising the inner-MD5+hex
     * work across SALT_BATCH outer-MD5 evaluations. */
    uint chunks_per_page = (num_salts_per_page + (uint)SALT_BATCH - 1u)
                         / (uint)SALT_BATCH;
    if (chunks_per_page == 0u) chunks_per_page = 1u;
    uint total = n_words * n_rules * mask_size * chunks_per_page;
#else
    /* gid layout (innermost -> outermost): mask -> word -> rule -> salt.
     * total = n_words * n_rules * mask_size * num_salts_per_page. */
    uint total = n_words * n_rules * mask_size * num_salts_per_page;
#endif
#else
    uint mask_size = mask_active ? params.num_salts : 1u;
    if (mask_size == 0u) mask_size = 1u;
    uint total = n_words * n_rules * mask_size;
#endif

    uint gid = get_global_id(0);
    if (gid >= total) return;

    /* gid -> (mask_idx, word_idx, rule_idx[, salt_local]) with mask
     * innermost so the existing (word, rule) lex-ordering is preserved
     * at mask_size==1. mask innermost also keeps the cursor protocol's
     * lex order (rule_idx, word_idx) monotonic across overflow re-issues. */
    uint mask_idx_local = gid % mask_size;
    uint gid_wr         = gid / mask_size;
    uint word_idx       = gid_wr % n_words;
    /* BF chunk-as-job (2026-05-10 Phase 1.5): kernel emits LOCAL mask_idx
     * (within-chunk, fits uint32 since mspw < 4G) for hit emit; uses
     * ABSOLUTE mask_idx (ulong) for hash compute. Host re-adds at hit-replay.
     * Reverses Tranche 2's "kernel emits absolute" + Tranche 3's "host
     * doesn't re-add" — those interlocked to truncate above 2^32 and the
     * <=4G gate disabled mask decode entirely for >4G keyspaces.
     *
     * For non-BF (mask_start == 0, mask_offset_per_word == 0), abs == local.
     * Cost: small ulong divmod overhead in charset decompose; MD5 compute
     * remains the dominant kernel cost. Per project_bf_chunk_as_job.md Q1.
     *
     * BF Phase 1.8 inner_iter wrap REVERTED (2026-05-10): the runtime-bounded
     * for-loop introduced a 2.19× Pascal regression measured in task #239
     * (raw MD5 BF) and a parallel JOB_MD5SALT (e31) regression. Kernel is
     * back to Phase 1.7's single-pass shape. Host-side machinery
     * (params.inner_iter, adaptive_bf_chunk_size servo, jobg bf_inner_iter)
     * is left as dead-but-harmless; cleanup can happen separately. */
    ulong mask_idx_abs_base = params.mask_start
                       + (ulong)word_idx * (ulong)params.mask_offset_per_word
                       + (ulong)mask_idx_local;
    uint gid_wrr  = gid_wr / n_words;
    uint rule_idx = gid_wrr % n_rules;
#ifdef GPU_TEMPLATE_HAS_SALT
#ifdef GPU_TEMPLATE_HAS_PRE_SALT
    /* Pre-salt path: the salt axis is folded into a salt_chunk axis where
     * each chunk covers SALT_BATCH consecutive salts. The actual salt_local
     * (and salt_idx_global) are computed inside the per-salt loop below. */
    uint salt_chunk = gid_wrr / n_rules;
    uint salt_base  = salt_chunk * (uint)SALT_BATCH;
#else
    /* Salt is the outermost axis. salt_local indexes into the current
     * dispatch's page; salt_idx_global = params.salt_start + salt_local
     * indexes into the host's salt list (gpu_pack_salts → b_salt_data /
     * b_salt_off / b_salt_lens). */
    uint salt_local      = gid_wrr / n_rules;
    uint salt_idx_global = (uint)params.salt_start + salt_local;
#endif
#endif

    /* B3 cursor check (Memo B §2). On a re-issue dispatch following hit
     * buffer overflow, lanes whose (rule, word, mask) lex-precedes the
     * cursor early-return. cursor=0 == identical to pre-B3 behavior.
     * B7.1: cursor compares only on (rule, word); within a (rule, word)
     * pair, mask_idx is processed entirely before advance. The kernel's
     * EMIT macro CAS-min's on lane gid, so on overflow the host sees the
     * lex-first lane gid and re-derives (mask, word, rule) the same way. */
    if (params.input_cursor_start > 0u || params.rule_cursor_start > 0u) {
        if (rule_idx < params.rule_cursor_start) return;
        if (rule_idx == params.rule_cursor_start &&
            word_idx < params.input_cursor_start) return;
    }

    /* Deterministic sub-buffer pointers (B1 wire format). */
    __global volatile uint *hit_count = (__global volatile uint *)(payload + 128);
    __global const uint   *word_offset = (__global const uint *)(payload + 132);
    uint pkt_off = 132u + (n_words * 4u);
    __global const uchar  *words = payload + pkt_off;

    /* B3 overflow channel pointers. See gpu_common.cl §B3 protocol. */
    __global volatile uint *ovr_set =
        (__global volatile uint *)(payload + 100);
    __global volatile uint *ovr_gid =
        (__global volatile uint *)(payload + 104);

    /* Single private buffer; same RULE_BUF_MAX as gpu_md5_rules.cl
     * (the macro is defined there and visible to us via the shared
     * compile unit). 16-byte aligned for any vectorized fast paths. */
    __attribute__((aligned(16))) uchar buf[RULE_BUF_MAX];

    /* Phase 6 BCRYPT (2026-05-08): workgroup-shared __local buffer for the
     * Eksblowfish S-boxes (4 × 256 uint = 4 KB per lane × BCRYPT_WG_SIZE
     * lanes = 32 KB per WG). Declared at kernel-function scope per
     * OpenCL 1.2 §6.5.3 (which forbids __local in non-kernel functions);
     * passed by pointer to template_finalize via the 8-arg signature
     * below (gated on GPU_TEMPLATE_HAS_LOCAL_BUFFER). When the macro is
     * UNDEFINED (every algo other than BCRYPT), the preprocessor strips
     * this block entirely — non-BCRYPT instantiations are byte-identical
     * to pre-Phase 6. Mirrors slab gpu_bcrypt.cl line 511. The §6.5.3
     * spec compliance note: __local declarations after divergent control
     * flow (the gid >= total early return at line 186) is empirically
     * accepted by Pascal NVCC + AMD ROCm + Mali Rusticl per the slab's
     * ship history; Phase 4 4-cell smoke validates each vendor. */
#ifdef GPU_TEMPLATE_HAS_LOCAL_BUFFER
    __local uint sbox_pool[BCRYPT_WG_SIZE * GPU_TEMPLATE_LOCAL_BUFFER_PER_LANE];
#endif

    uint wpos = word_offset[word_idx];
    int wlen = (int)words[wpos++];
    if (wlen > RULE_BUF_LIMIT) wlen = RULE_BUF_LIMIT;
    for (int i = 0; i < wlen; i++) buf[i] = words[wpos + i];

    uint rpos = rule_offset[rule_idx];

    /* Synthetic no-rule discriminator (preserves feedback_no_rule_pass.md
     * semantics; mirrors gpu_md5_rules.cl rev 1.28). The first byte being
     * NUL means k==0 at apply_rule entry == the synthetic ":" pass. */
    int is_no_rule = (rule_program[rpos] == 0);
    int new_len = apply_rule(rule_program + rpos, buf, wlen);

    /* Rejection sentinel: apply_rule fired a `_ < > ! / ( )` op. */
    if (new_len < 0) return;

    /* No-op detection: if at least one op was processed AND the post-
     * rule buffer is bit-identical to the input, the synthetic ":" pass
     * already covered this candidate -- skip hash + probe. Per
     * feedback_no_rule_pass.md: with masks active, BOTH the no-rule pass
     * AND each rule iterate every mask, so the same (word, mask_idx)
     * coordinate is already covered by the synthetic ":" pass at this
     * mask_idx — the skip remains correct. */
    if (!is_no_rule && new_len == wlen) {
        int changed = 0;
        for (int i = 0; i < wlen; i++) {
            if (buf[i] != words[wpos + i]) { changed = 1; break; }
        }
        if (!changed) return;
    }

    /* B7.1-B7.5 mask: decompose mask_idx into prepend+append per-position
     * indices and modify buf in-place. Slab convention (matches
     * gpu_kernels.cl md5_mask_batch and gpu_mask_desc layout):
     *
     *   mask_idx = prepend_idx * append_combos + append_idx
     *   append_combos = product(mask_sizes[n_prepend..n_prepend+n_append))
     *
     * Within each section the LAST position (highest index) is innermost
     * / cycles fastest, matching the slab path's per-position decomposition
     * (`for i = n-1; i >= 0; i--` style).
     *
     * mask_charsets is row-major with the slab convention: rows
     * [0..n_prepend) hold prepend charsets, rows [n_prepend..n_prepend+
     * n_append) hold append charsets. Total kernel cap is MASK_POS_CAP=16
     * per section so positions stay below 32 rows.
     *
     * Modification order:
     *   1. (Optional) shift buf right by n_prepend bytes to make room.
     *   2. Write prepend chars into buf[0..n_prepend).
     *   3. Write append chars into buf[new_len..new_len+n_append).
     *   4. new_len += n_prepend + n_append.
     *
     * Backward-compat with B7.1+B7.2 (n_prepend==0): step 1 is a no-op
     * and step 2 is skipped; rows [0..n_append) are append rows — exact
     * same kernel-side behavior as B7.2. mask_charsets[0*256+...] reads
     * the same byte. Byte-exact backward-compatible.
     *
     * Bounds-checked against RULE_BUF_LIMIT for safety. */
    uint npre = params.n_prepend;
    uint napp = params.n_append;
    if (npre > 16u) npre = 16u;
    if (napp > 16u) napp = 16u;

    /* BF Phase 1.8 inner_iter wrap REVERTED (2026-05-10): mask_idx_abs is
     * the base value computed above; no per-iter shift. */
    ulong mask_idx_abs = mask_idx_abs_base;

    if (npre >= 1u || napp >= 1u) {
        /* Compute append_combos for the prepend/append split.
         * Phase 1.5 (2026-05-10): widened to ulong because the product can
         * exceed 2^32 for >9-position masks of size 10+ (e.g., ?d^10 = 1e10).
         * Pre-Phase-1.5 this was uint; mask_idx < 4G constrained the
         * keyspace so append_combos truncation didn't bite. With mask_idx_abs
         * now ulong and >4G keyspaces in scope, append_combos MUST be ulong
         * or the divmod produces garbage past the uint32 wrap point. */
        ulong append_combos = 1u;
        for (uint j = 0u; j < napp; j++) {
            uint sz = mask_sizes[npre + j];
            if (sz == 0u) sz = 1u;
            append_combos *= (ulong)sz;
        }
        if (append_combos == 0u) append_combos = 1u;
        /* Phase 1.5 (2026-05-10): use ABSOLUTE mask_idx (ulong) for charset
         * decompose so chunks past keyspace_offset 2^32 receive the correct
         * per-position character indices. The divmod loop below truncates
         * each per-position index to uint after `% psize` (psize <= 256),
         * so prepend_idx / append_idx are kept as ulong here. */
        ulong prepend_idx = mask_idx_abs / append_combos;
        ulong append_idx  = mask_idx_abs % append_combos;

        /* Step 1: shift buf right by npre bytes to make room at the front
         * (only when npre > 0). Iterate from high to low to avoid clobber. */
        if (npre > 0u) {
            uint shift_dst_end = (uint)new_len + npre;
            if (shift_dst_end > RULE_BUF_LIMIT) {
                /* Truncate the shift to keep within bounds. */
                if ((uint)new_len + npre > RULE_BUF_LIMIT) {
                    /* Drop trailing bytes that would exceed the limit. */
                    if (new_len > (int)(RULE_BUF_LIMIT - npre))
                        new_len = (int)(RULE_BUF_LIMIT - npre);
                }
            }
            for (int i = new_len - 1; i >= 0; i--) {
                buf[i + (int)npre] = buf[i];
            }
        }

        /* Step 2: write prepend chars at buf[0..npre).
         * Decompose prepend_idx with last position innermost (i==npre-1
         * cycles fastest). Each iteration: pidx = remaining % size_i;
         * remaining /= size_i; buf[i] = mask_charsets[i*256 + pidx]. */
        if (npre > 0u) {
            /* Phase 1.5: remaining is ulong because prepend_idx is ulong;
             * pidx fits uint (psize <= 256 enforced by host). */
            ulong remaining = prepend_idx;
            for (uint k = 0u; k < npre; k++) {
                uint i = npre - 1u - k;
                uint psize = mask_sizes[i];
                if (psize == 0u) psize = 1u;
                uint pidx = (uint)(remaining % (ulong)psize);
                remaining /= (ulong)psize;
                if (i < RULE_BUF_LIMIT) {
                    buf[i] = mask_charsets[i * 256u + pidx];
                }
            }
        }

        /* Step 3: write append chars at buf[new_len + npre .. new_len +
         * npre + napp). Same decomposition order. mask_charsets row index
         * for append position j is (npre + j). */
        if (napp > 0u) {
            uint append_base = (uint)new_len + npre;
            /* Phase 1.5: remaining is ulong because append_idx is ulong;
             * pidx fits uint (psize <= 256 enforced by host). */
            ulong remaining = append_idx;
            for (uint k = 0u; k < napp; k++) {
                uint i = napp - 1u - k;
                uint row = npre + i;
                uint psize = mask_sizes[row];
                if (psize == 0u) psize = 1u;
                uint pidx = (uint)(remaining % (ulong)psize);
                remaining /= (ulong)psize;
                uint dst = append_base + i;
                if (dst < RULE_BUF_LIMIT) {
                    buf[dst] = mask_charsets[row * 256u + pidx];
                }
            }
        }

        /* Step 4: advance new_len. Truncate at RULE_BUF_LIMIT. */
        uint new_total = (uint)new_len + npre + napp;
        if (new_total <= RULE_BUF_LIMIT) {
            new_len = (int)new_total;
        } else {
            new_len = RULE_BUF_LIMIT;
        }
    }

    /* --- Algorithm-specific hash via template extension functions. --- */
#ifdef GPU_TEMPLATE_HAS_PRE_SALT
    /* Lane-level salt batching (2026-05-09 experiment): compute the
     * password-only portion of the salted hash ONCE per (word, rule, mask,
     * salt_chunk), then loop over up to SALT_BATCH salts serially. Each
     * iteration reuses pre_state but re-initialises template_state for
     * the per-salt outer-MD5. The algorithm core (gpu_md5salt_core.cl)
     * defines template_pre_salt to populate pre_state from buf and
     * template_finalize_post to consume pre_state + salt and produce the
     * final per-salt digest in st. */
    template_pre_salt_state pre_state;
    template_pre_salt(buf, new_len, params.algo_mode, &pre_state);

    for (uint i = 0u; i < (uint)SALT_BATCH; i++) {
        uint salt_local = salt_base + i;
        if (salt_local >= num_salts_per_page) break;
        uint salt_idx_global = (uint)params.salt_start + salt_local;
        uint  s_off = salt_off[salt_idx_global];
        uint  s_len = (uint)salt_lens[salt_idx_global];

        template_state st;
        template_init(&st);
        template_finalize_post(&st, &pre_state,
                               buf, new_len,
                               salt_buf + s_off, s_len,
                               params.algo_mode);
#else
    template_state st;
    template_init(&st);
#ifdef GPU_TEMPLATE_HAS_SALT
    /* B6 salt-axis: pass salt bytes + length to the core. The core
     * computes its algorithm's salted hash inline (see gpu_md5salt_core.cl
     * for the JOB_MD5SALT double-MD5 chain, gpu_md5saltpass_core.cl for
     * the JOB_MD5SALTPASS prepend, etc.). salt_idx_global is the index
     * into the host's full salt list (b_salt_data / b_salt_off /
     * b_salt_lens uploaded once via gpu_opencl_set_salts). */
    uint  s_off = salt_off[salt_idx_global];
    uint  s_len = (uint)salt_lens[salt_idx_global];
#ifdef GPU_TEMPLATE_HAS_LOCAL_BUFFER
    /* Phase 6 BCRYPT (2026-05-08): 8-arg signature variant — passes the
     * workgroup-shared __local sbox_pool + per-lane lid so the core's
     * template_finalize can claim its 1024-uint partition (lid * 1024).
     * Gated on GPU_TEMPLATE_HAS_LOCAL_BUFFER; only the BCRYPT carrier
     * sets this define. */
    template_finalize(&st, buf, new_len, salt_buf + s_off, s_len,
                      params.algo_mode, sbox_pool, (uint)get_local_id(0));
#else
    template_finalize(&st, buf, new_len, salt_buf + s_off, s_len, params.algo_mode);
#endif
#else
    template_finalize(&st, buf, new_len);
#endif
#endif /* GPU_TEMPLATE_HAS_PRE_SALT */

    /* --- Iterated probe (algorithm-agnostic loop; per-iter step is
     *      template_iterate, which is algorithm-specific). --- */
    uint max_iter = params.max_iter;
    if (max_iter < 1) max_iter = 1;
    for (uint iter = 1; iter <= max_iter; iter++) {
        uint matched_idx = 0u;
        if (template_digest_compare(&st,
                                    compact_fp, compact_idx,
                                    params.compact_mask, params.max_probe,
                                    params.hash_data_count,
                                    hash_data_buf, hash_data_off,
                                    overflow_keys, overflow_hashes,
                                    overflow_offsets, params.overflow_count,
                                    &matched_idx))
        {
            /* B3 dedup+overflow-aware emit (Memo B §2). The macro itself
             * does the hashes_shown atomic_or; on overflow, it rolls back
             * the dedup bit so re-issue lanes can re-emit. Without the
             * rollback, dedup bits stay set across dispatches and silently
             * drop cracks. See gpu_common.cl §B3 protocol.
             *
             * B7.1: pack mask_idx into the rule_idx slot of the hit entry.
             * combined_ridx = rule_idx * mask_size + mask_idx so the host
             * can decode rule_idx = combined / mask_size and mask_idx =
             * combined % mask_size. Hit-stride is unchanged (slot[1] is
             * the second word; was just rule_idx pre-B7). When mask_size
             * == 1 (no mask), combined_ridx == rule_idx and the host
             * decodes a 0 mask_idx — bit-identical to pre-B7 behavior.
             *
             * B6 (2026-05-06): for salted variants the kernel packs
             *   combined_ridx = ((rule_idx * mask_size) + mask_idx)
             *                   * num_salts_per_page + salt_local
             * so host hit-replay can recover (salt_local, mask_idx,
             * rule_idx) via three div/mod steps. The kernel ALSO emits
             * salt_idx_global = params.salt_start + salt_local in slot[1]
             * of the hit entry would require widening EMIT_HIT_N (rejected
             * per §10 Q2 / Decision C.3). Instead the host dispatcher
             * post-processes each page's hits after readback to convert
             * salt_local -> salt_idx_global by adding salt_start. By the
             * time hits reach gpujob_opencl.c hit-replay, slot[1] always
             * encodes ((rule_idx * mask_size) + mask_idx) *
             * num_salts_per_page + SALT_IDX_GLOBAL — i.e., a uniform
             * format across all salt pages. When num_salts_per_page == 1
             * (unsalted dispatch / salt-axis disabled) the layout
             * collapses to the existing rule_idx*mask_size + mask_idx —
             * bit-identical to pre-B6. */
            /* Phase 1.5 (2026-05-10): hit emit uses LOCAL mask_idx (within-
             * chunk; bounded by mask_size which equals bf_num_masks for BF
             * chunks or gpu_mask_total for non-BF). Encoding fits uint32
             * because rule_idx * mask_size * num_salts_per_page < 2^32 in
             * all valid configurations. Host re-adds bf_mask_start +
             * widx*offset at hit-replay to recover the absolute index. */
#ifdef GPU_TEMPLATE_HAS_SALT
            /* Salted path encoding: pack rule, mask-local and salt-local into
             * a single uint per the B6 layout. */
            uint combined_ridx =
                (rule_idx * mask_size + mask_idx_local) * num_salts_per_page +
                salt_local;
#else
            /* Unsalted path (Phase 1.7 form, restored after Phase 1.8 revert):
             * combined_ridx = rule_idx * mask_size + mask_idx_local. The
             * Phase 1.8 inner-iter packing caused a 2.19× Pascal regression
             * (task #239) and a parallel JOB_MD5SALT regression; reverted. */
            uint combined_ridx =
                rule_idx * mask_size + mask_idx_local;
#endif
            uint mask = 1u << (iter & 31);
            template_emit_hit_or_overflow(hits, hit_count, params.max_hits,
                              &st, word_idx, combined_ridx, iter,
                              hashes_shown, matched_idx, mask,
                              ovr_set, ovr_gid, gid);
        }
        if (iter < max_iter) {
            /* B7.7a (2026-05-07): MD5UC algo_mode threading. Only
             * gpu_md5_core.cl defines GPU_TEMPLATE_ITERATE_HAS_ALGO_MODE
             * (its template_iterate accepts an algo_mode arg to switch
             * lowercase/uppercase hex). All other cores keep the
             * legacy `(st)` signature. Mirrors the existing
             * GPU_TEMPLATE_HAS_SALT pattern at template_finalize. */
#ifdef GPU_TEMPLATE_ITERATE_HAS_ALGO_MODE
            template_iterate(&st, params.algo_mode);
#else
            template_iterate(&st);
#endif
        }
    }
#ifdef GPU_TEMPLATE_HAS_PRE_SALT
    /* Close the SALT_BATCH inner salt loop opened above the iter loop. */
    }
#endif
}

/* template_phase0_test: byte-exact harness twin of template_phase0.
 * Mirrors md5_rules_phase0_test in gpu_md5_rules.cl: same
 * (word, rule) geometry, but emits the digest into digests_out[]
 * unconditionally instead of probing the compact table.
 *
 * Used by gpu_rules_test.c with --engine=template to validate that
 * the template path produces byte-identical digests to the legacy
 * path. Signature mirrors md5_rules_phase0_test exactly so the
 * harness can swap kernel name + recompile sources without touching
 * the dispatch wiring.
 *
 * B6 (2026-05-06): the harness only compiles for unsalted instantiations
 * — salted variants don't have a no-salt template_finalize signature,
 * and the test harness itself doesn't carry salt args. Salted byte-exact
 * validation goes through the real template_phase0 + a dedicated
 * fixture in gpu_rules_test.c (post-ship). */
#ifndef GPU_TEMPLATE_HAS_SALT
__kernel
void template_phase0_test(
    __global const uchar     *words,
    __global const uint      *word_offset,
    __global const uchar     *rule_program,
    __global const uint      *rule_offset,
    __global const OCLParams *params_buf,
    __global uint            *digests_out)
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

    /* Rejection sentinel: emit zero digest. Same convention as
     * md5_rules_phase0_test for diff-the-harness compatibility. */
    if (new_len < 0) {
        for (int i = 0; i < HASH_WORDS; i++) digests_out[gid * HASH_WORDS + i] = 0u;
        return;
    }

    template_state st;
    template_init(&st);
    template_finalize(&st, buf, new_len);

    for (int i = 0; i < HASH_WORDS; i++) digests_out[gid * HASH_WORDS + i] = st.h[i];
}
#endif /* !GPU_TEMPLATE_HAS_SALT */
