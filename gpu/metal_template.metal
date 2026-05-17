/*
 * $Revision: 1.3 $
 * $Log: metal_template.metal,v $
 * Revision 1.3  2026/05/13 03:39:11  dlr
 * Phase 2b row 1: add GPU_TEMPLATE_HAS_MASK axis to template_phase0. New gated buffer args mask_charsets (buffer 12) and mask_sizes (buffer 13). Under the macro the kernel gains a third gid axis (mask innermost: mask_idx_local = gid % mask_size; word_idx and rule_idx unchanged in lex order), then decomposes mask_idx into per-position prepend+append indices via divmod (mirror of gpu_template.cl lines 405-531) and modifies thread uchar buf[] in place AFTER the rules walker but BEFORE template_finalize. Mirrors slab convention: rows [0..n_prepend) are prepend rows, rows [n_prepend..n_prepend+n_append) are append. RULE_BUF_LIMIT bound enforced; append_combos widened to ulong (matches OpenCL Phase 1.5 widening for >4G keyspaces). Combined_ridx packing extends: HAS_MASK alone -> mask_idx_local; HAS_RULES alone -> rule_idx; both -> rule_idx*mask_size+mask_idx_local; neither -> 0. Coexists with HAS_RULES: M-alone signature has buffers 0..9,12,13 (gap at 10,11); RM dense 0..13. Apple Metal driver-verified to accept gap-numbered buffer indices via xcrun metal standalone compile + AIR inspection on iMac before this commit. axis_mask_size and axis_n_rules locals factor the gid divmod into a single shape so the four (none/R/M/RM) variants share the same arithmetic. Phase 2b non-goals: salt axis, iter loop, pre-salt hoist, BCRYPT local buffer, BF chunk-as-job (params.mask_start and params.mask_offset_per_word read 0 in 2b). Offline xcrun metal compile clean for all 4 variants on iMac AMD Radeon Pro 580X (no new warnings beyond pre-existing template_transform/template_iterate/apply_rule unused-function noise). Per Phase 2b memo rows 1+8.
 *
 * Revision 1.2  2026/05/12 17:14:30  dlr
 * Phase 2a row 4: add GPU_TEMPLATE_HAS_RULES axis to template_phase0. Under the macro the kernel gains two new buffer args (rule_program at 10, rule_offset at 11), decomposes gid into (word_idx, rule_idx) with rule innermost, invokes apply_rule from metal_md5_rules.metal (same family TU + same JIT concat), and packs combined_ridx = rule_idx for hit emit. Synthetic-no-rule discriminator + post-rule no-op skip preserve feedback_no_rule_pass.md semantics. Without the macro the kernel compiles to Phase 1 behavior byte-identical: gid -> word_idx direct, no walker. n_rules sourced from params.num_masks (mirrors OpenCL twin gpu_template.cl line 215). Offline xcrun metal compile clean for both variants on iMac (no-rules AIR 12.9KB, rules AIR 23.5KB). Per memo §3 rows 4-7 + Phase 0.5 patterns 1-6.
 *
 * Revision 1.2  2026/05/12 18:00:00  dlr
 * Phase 2a row 4: add GPU_TEMPLATE_HAS_RULES axis to template_phase0. New gated buffer args rule_program (buffer 10) and rule_offset (buffer 11); under the macro the kernel decomposes gid into (word_idx, rule_idx) with rule innermost, then invokes apply_rule from metal_md5_rules.metal (same family TU) — same shape as gpu/gpu_template.cl lines 374-403. Synthetic-no-rule discriminator (first byte NUL -> is_no_rule=1) and post-rule no-op skip preserve feedback_no_rule_pass.md byte semantics. Hit emit packs combined_ridx = rule_idx (mask_size==1 in 2a; mask axis lands in 2b). Without the macro the kernel compiles to Phase 1 behavior byte-identical: gid -> word_idx direct, no walker. n_rules sourced from params.num_masks (same overload as OpenCL twin). Per memo §3 rows 4-7 + Phase 0.5 patterns 1-6 enforced (every new arg device-qualified; no threadgroup decls added).
 *
 * Revision 1.1  2026/05/12 13:35:19  dlr
 * Initial check-in: Phase 1 Metal template kernel. Subset of gpu/gpu_template.cl: raw MD5 unsalted only, gid -> word_idx direct (no rules, no mask, no salt), single iter cycle (max_iter==1). Payload wire format byte-identical to OpenCL: MetalParams at offset 0-127, hit_count at 128, word_offset at 132, packed words after. Buffer indices 0..9 sequential. Pattern 1: every kernel arg device-or-thread-qualified. Pattern 5: no threadgroup decls (rule pre-stated for Phase 2).
 *
 */
/* metal_template.metal — generic dispatch-template kernel for Metal.
 * Mirrors a SUBSET of gpu/gpu_template.cl.
 *
 * Compile-time axes (Phase 2a row 4 + Phase 2b row 1):
 *   GPU_TEMPLATE_HAS_RULES   undef -> Phase 1 no-rules path (gid -> word).
 *                            defined -> apply_rule walker invoked per lane;
 *                            kernel signature gains rule_program (buffer 10)
 *                            and rule_offset (buffer 11).
 *   GPU_TEMPLATE_HAS_MASK    undef -> no mask axis (mask_size==1).
 *                            defined -> per-lane mask_idx decompose + in-place
 *                            buf prepend/append; kernel signature gains
 *                            mask_charsets (buffer 12) and mask_sizes
 *                            (buffer 13). Independent of HAS_RULES — the
 *                            four (none/R/M/RM) variants compile from the
 *                            same source. Buffer indices are gap-tolerant:
 *                            M-alone uses 12/13 even though 10/11 are absent.
 *                            Mirrors gpu/gpu_template.cl lines 181-188,
 *                            216-275, 405-531 (less BF and salt — not in 2b).
 *
 * Phase 1 scope (memo §5 mapping table; memo §1 non-goals — applies when
 * GPU_TEMPLATE_HAS_RULES is undefined):
 *   - Raw MD5 unsalted, NO rules, NO mask, NO salt, NO iter, NO BF,
 *     NO HMAC, NO BCRYPT, NO PRE_SALT, NO threadgroup LOCAL_BUFFER.
 *   - One word per lane (gid -> word_idx directly).
 *   - max_iter is read but only the iter==1 case is exercised (no
 *     iter loop body).
 *   - Cursor protocol disabled (input_cursor_start/rule_cursor_start
 *     populated to 0 by the host; kernel reads but takes no action).
 *
 * Phase 2a row 4: GPU_TEMPLATE_HAS_RULES axis adds the rules-engine walker
 * invocation. Mirrors gpu_template.cl lines 374-403.
 * Phase 2b row 1 (this revision): GPU_TEMPLATE_HAS_MASK axis adds in-place
 * prepend/append into the per-lane buf, mirroring gpu_template.cl lines
 * 181-188 (arg decls), 216-275 (geometry + mask_idx decompose), 405-531
 * (modification). Coexists with HAS_RULES — RM variant binds buffers
 * 10, 11, 12, 13; M-alone binds 12, 13 (gap-OK; verified by xcrun metal
 * standalone compile on iMac before this commit).
 * Phase 2c+ re-adds: salt axis, iter loop, pre-salt hoist, BCRYPT local
 * buffer, BF chunk-as-job (Metal currently lacks BF; 2b's
 * mask_offset_per_word + mask_start kernel args read 0 from params).
 *
 * Phase 2e (pre-salt hoist + SIMD lane batching):
 *   GPU_TEMPLATE_HAS_PRE_SALT  undef -> per-salt template_finalize (Phase
 *                              2c shape; the inner-MD5+hex32 is recomputed
 *                              for every salt).
 *                              defined -> template_pre_salt() runs ONCE
 *                              per (word, rule, mask); inner salt loop
 *                              iterates SALT_BATCH salts at a time via
 *                              template_finalize_post(). REQUIRES
 *                              GPU_TEMPLATE_HAS_SALT.
 *   SALT_BATCH                 compile-time macro (default 16). Tile size
 *                              for the inner salt loop under HAS_PRE_SALT.
 *                              On Metal one-thread-per-word the tile is
 *                              an unroll hint for the compiler -- the
 *                              kernel still iterates the FULL salt list,
 *                              just in stride-SALT_BATCH chunks. Per-tier
 *                              selection lives in gpu_metal.m
 *                              metal_select_salt_batch().
 *   Mirrors gpu_template.cl HAS_PRE_SALT block (lines 534-585), adapted
 *   to Metal's one-thread-per-word grid (the OpenCL twin folds salt_chunk
 *   into the outer NDRange axis; Metal keeps it as an inner-loop tile).
 *
 * --- Pattern enforcement (from memo §12 + Phase 0.5) ---
 *
 *   Pattern 1: every kernel-arg buffer is `device` or `constant`;
 *              every lane-local pointer is `thread`.
 *   Pattern 2: kernel name is `template_phase0` (matches OpenCL).
 *   Pattern 5: NO threadgroup declarations in Phase 1 (rule pre-stated
 *              for Phase 2: threadgroup decl MUST be initialized before
 *              first read; pair with threadgroup_barrier(mem_threadgroup)).
 *
 * --- Payload wire format (matches gpujob_opencl.c layout) ---
 *
 *   offset 0..127     MetalParams params (mirrors OCLParams; 128 B)
 *   offset 128..131   uint hit_count (atomic, written by kernel)
 *   offset 132..132+4*n_words-1  uint word_offset[n_words]
 *                                   per-word byte offsets into packed
 *                                   words buffer (word_offset[i] points
 *                                   at the length-prefixed word i)
 *   offset 132+4*n_words..     packed words (length byte + bytes)
 *
 *   ovr_set lives at offset 100  (inside params: overflow_first_set)
 *   ovr_gid lives at offset 104  (inside params: overflow_first_word)
 *
 * The layout is byte-identical to the OpenCL kernel's payload reads.
 * Reuses gpujob_opencl.c's existing packer.
 *
 * --- Kernel signature mapping (memo §5) ---
 *
 *   OpenCL                                 Metal
 *   __global uchar *payload                device uchar *payload [[buffer(0)]]
 *   __global const T *                     device const T *      [[buffer(N)]]
 *   __global volatile uint *hashes_shown   device atomic_uint *hashes_shown [[buffer(N)]]
 *   __global uint *hits                    device uint *hits     [[buffer(N)]]
 *   get_global_id(0)                       uint gid [[thread_position_in_grid]]
 *
 * Phase 1 args (10 buffers; subset of OpenCL's 14-arg template_phase0):
 *   payload, compact_fp, compact_idx, hash_data_buf, hash_data_off,
 *   hits, overflow_keys, overflow_hashes, overflow_offsets, hashes_shown.
 *
 * Phase 2a HAS_RULES extension: 12 buffers (adds rule_program at 10,
 * rule_offset at 11). The host (gpu_metal.m) MUST bind in the same order
 * and MUST always bind buffers 10/11 when the rules variant PSO is in
 * use. The two PSO variants live in two distinct MTLLibrary objects
 * (no-rules from embedded metallib; rules JIT-compiled with the macro
 * defined).
 *
 * Phase 2b HAS_MASK extension: 12 buffers total in the M-alone variant
 * (0..9 + 12, 13 — buffers 10 and 11 are absent in the kernel signature
 * because the rules-block is preprocessor-stripped). 14 buffers total in
 * the RM variant (0..13 dense). Apple Metal's argument-buffer validator
 * accepts gap-numbered indices (10/11 absent in M-alone is fine — the
 * other lanes of the signature still bind dense 0..9, 12, 13).
 *
 * Buffer indices:
 *   - Phase 1 (none):     0..9 dense                 + 14 (buf_scratch_pool)
 *   - Phase 2a (R only):  0..11 dense                + 14
 *   - Phase 2b (M only):  0..9, 12, 13 (gap at 10..11) + 14
 *   - Phase 2b (RM):      0..13 dense                + 14
 *
 * Task #250 buffer 14 (buf_scratch_pool) is always bound for every
 * variant — the 40 KB per-lane scratch lives in device memory rather
 * than thread-local registers to fit M2 Max's register budget. Host
 * sizes it to num_words * RULE_BUF_MAX (640 MB at 16K-word peak),
 * allocated lazily on first dispatch. See gpu_metal.m
 * metal_ensure_buf_scratch_pool() / buf_scratch_pool static.
 */

/* Phase 2e SALT_BATCH default. Tile size for the inner salt loop under
 * GPU_TEMPLATE_HAS_PRE_SALT. The host passes -DSALT_BATCH=N (per-tier
 * selection via metal_select_salt_batch in gpu_metal.m: 8 for M1, 16
 * for M2/M2 Max, 32 for M3+; env override MDXFIND_METAL_SALT_BATCH).
 * Default 16 (M2 Max sweet spot) when neither macro nor env is set. */
#ifndef SALT_BATCH
#define SALT_BATCH 16
#endif

kernel void template_phase0(
    device uchar          *payload          [[buffer(0)]],
    device const uint     *compact_fp       [[buffer(1)]],
    device const uint     *compact_idx      [[buffer(2)]],
    device const uchar    *hash_data_buf    [[buffer(3)]],
    device const ulong    *hash_data_off    [[buffer(4)]],
    device uint           *hits             [[buffer(5)]],
    device const ulong    *overflow_keys    [[buffer(6)]],
    device const uchar    *overflow_hashes  [[buffer(7)]],
    device const uint     *overflow_offsets [[buffer(8)]],
    device atomic_uint    *hashes_shown     [[buffer(9)]],
#ifdef GPU_TEMPLATE_HAS_RULES
    device const uchar    *rule_program     [[buffer(10)]],
    device const uint     *rule_offset      [[buffer(11)]],
#endif
#ifdef GPU_TEMPLATE_HAS_MASK
    /* Phase 2b row 1: mask charset table + per-position sizes. Layout mirrors
     * gpu/gpu_opencl.c b_template_mask_charsets / b_template_mask_sizes
     * exactly:
     *   mask_charsets[i*256..i*256+256)   = 256-byte charset row for position i;
     *                                       only the first mask_sizes[i] bytes
     *                                       are valid.
     *   mask_sizes[i]                     = uint32 modulus for position i.
     * Positions [0..n_prepend) are prepend rows; positions [n_prepend..
     * n_prepend+n_append) are append rows. MASK_POS_CAP = 16 per side
     * (32 rows total = 8 KB charset buffer). The host (gpu_metal_set_mask
     * in gpu_metal.m) allocates the full 32-row buffer and writes only the
     * active rows; unused rows are zero-filled + mask_sizes[unused] = 1
     * sentinel so any stray divmod terminates safely. */
    device const uchar    *mask_charsets    [[buffer(12)]],
    device const uint     *mask_sizes       [[buffer(13)]],
#endif
#ifdef GPU_TEMPLATE_HAS_SALT
    /* Phase 2c salt axis. Mirrors gpu/gpu_template.cl lines 203-207:
     *   salt_buf  -- concatenated salt bytes for the whole salt list (not
     *                paged); salt N starts at salt_off[N], length is
     *                salt_lens[N]. Host uploads the full list once via
     *                gpu_metal_set_salt (mirrors gpu_opencl_set_salts).
     *   salt_off  -- per-salt byte offset into salt_buf (uint32 array).
     *   salt_lens -- per-salt length (uint16 array).
     * Buffer indices 15/16/17 -- gap-tolerant; 14 is buf_scratch_pool,
     * 12/13 mask, 10/11 rules. Apple Metal driver-validated to accept the
     * gap layout when MS variant binds 12/13 + 15/16/17 without 10/11. */
    device const uchar    *salt_buf         [[buffer(15)]],
    device const uint     *salt_off         [[buffer(16)]],
    device const ushort   *salt_lens        [[buffer(17)]],
#endif
    /* Task #250 scratch-pool migration. The per-lane 40 KB `buf` was previously
     * `thread uchar buf[RULE_BUF_MAX]` — a lane-private register array. On
     * M1 / Apple Silicon Phase 2a the compiler accepted this; on M2 Max
     * (T6020, larger SIMD width / different register-allocator headroom)
     * the PSO-create gate rejects it: "Compute function exceeds available
     * temporary registers". RULE_BUF_MAX cannot be truncated without
     * losing rule-output coverage (per memo §1; user load-bearing).
     *
     * Solution: move buf to device storage. Host allocates one MTLBuffer
     * sized num_words * RULE_BUF_MAX (640 MB at the 16K-word peak — fits
     * comfortably in M1 16 GB / M2 Max 96+ GB unified memory). Each lane
     * owns the slice `buf_scratch_pool + word_idx * RULE_BUF_MAX`. The
     * kernel restructure (this revision) makes the outer grid one-thread-
     * per-word and folds the rule × mask axes into an inner double-loop
     * — same shape as hashcat's canonical per-word inner-loop model.
     *
     * Always bound (all four PSO variants). The OpenCL twin keeps `buf`
     * in __private because OpenCL drivers spill to global transparently;
     * Apple Metal does NOT spill, so the move is required, not optional. */
    device uchar          *buf_scratch_pool [[buffer(14)]],
    uint                   gid              [[thread_position_in_grid]]
#ifdef GPU_TEMPLATE_HAS_LOCAL_BUFFER
    ,
    /* Phase 2d.9b BCRYPT: per-lane index within the workgroup. Used to
     * partition the threadgroup-shared sbox_pool into per-lane 1024-uint
     * slots (sbox_pool + lid * 1024). Only added under HAS_LOCAL_BUFFER
     * so non-BCRYPT instantiations keep their exact pre-2d.9b signature
     * (byte-identity gate per TRAP 1 of the Phase 2d.9b brief). */
    uint                   lid              [[thread_position_in_threadgroup]]
#endif
    )
{
    /* Decode payload header. Copy params to thread (lane-private) memory
     * so subsequent field accesses stay in registers — same optimization
     * as the OpenCL twin (`OCLParams params = *params_buf;`). */
    device const MetalParams *params_buf =
        (device const MetalParams *)payload;
    MetalParams params = *params_buf;

    uint n_words = params.num_words;
#ifdef GPU_TEMPLATE_HAS_RULES
    /* params.num_masks carries the per-dispatch rule COUNT (which may
     * be smaller than the total rule program when host-side rule-axis
     * sub-batching is in effect — task #250). The synthetic no-rule
     * pass uploads n_rules==1 with a 1-byte program containing NUL.
     *
     * params.salt_start carries the per-dispatch rule BASE (offset into
     * the rule_offset[] table). The kernel loops over rule_idx in
     * [rule_base, rule_base + rule_count). combined_ridx packs the
     * absolute rule_idx so hit-replay decoding sees a stable index
     * across all sub-batches.
     *
     * Sub-batching exists to keep per-dispatch kernel runtime below
     * Apple's `kIOGPUCommandBufferCallbackErrorImpactingInteractivity`
     * watchdog (~2 second hard wall). On 100K-rule programs an unsplit
     * dispatch would inner-loop 100K rules per word lane; chunked
     * dispatches inner-loop a few hundred rules each. salt_start is
     * unused in the non-salt MD5 path so we overload it cleanly. */
    uint rule_count = params.num_masks;
    if (rule_count == 0u) rule_count = 1u;
    /* Phase 2c: read rule_base from params.rule_cursor_start (its real
     * meaning -- offset into rule_offset[] for sub-batching). Previously
     * (Phase 2b) we overloaded params.salt_start because HAS_SALT was undef
     * and params.salt_start carried no other meaning. With the salt axis
     * arriving in Phase 2c, params.salt_start returns to its OpenCL
     * semantics (salt-page start). The OCLParams field rule_cursor_start
     * at offset 92 has been claimed for real use (see
     * feedback_rename_reserved_slots.md); gpu_metal.m writes it under the
     * same name in lockstep. */
    uint rule_base  = params.rule_cursor_start;
#else
    uint rule_count = 1u;
    uint rule_base  = 0u;
#endif

#ifdef GPU_TEMPLATE_HAS_SALT
    /* Phase 2c salt-axis geometry. num_salts is overloaded as
     *   num_salts = mask_size * num_salts_per_page
     * so mask_size below is derived via division (mirrors
     * gpu_template.cl:251-256). For Phase 2c num_salts_per_page is 1
     * (single dispatch per salt batch), so the math collapses to
     * mask_size == num_salts and the unsalted Phase 2b layout is
     * preserved bit-identically. */
    uint num_salts_per_page = (uint)params.num_salts_per_page;
    if (num_salts_per_page == 0u) num_salts_per_page = 1u;
#endif

#ifdef GPU_TEMPLATE_HAS_MASK
    /* Phase 2b row 1 mask geometry. Mirrors gpu_template.cl lines 216-275.
     * mask_size derivation depends on whether the salt axis is also active:
     *   - HAS_SALT undef (2b): mask_size = num_salts.
     *   - HAS_SALT defined (2c): mask_size = num_salts / num_salts_per_page
     *     (mirrors gpu_template.cl:251-256 overload).
     *
     * mask_active is the gate: a mask is in scope only when n_prepend or
     * n_append is >= 1. With no mask, the kernel collapses to the (word,
     * rule) shape (mask_size==1). */
    uint mask_active = ((params.n_prepend >= 1u) || (params.n_append >= 1u)) ? 1u : 0u;
#ifdef GPU_TEMPLATE_HAS_SALT
    uint mask_size = mask_active
        ? (params.num_salts / num_salts_per_page)
        : 1u;
#else
    uint mask_size = mask_active ? params.num_salts : 1u;
#endif
    if (mask_size == 0u) mask_size = 1u;
#else
    uint mask_size = 1u;
#endif

    /* Task #250: kernel restructure — one thread per word; rule × mask
     * axes fold into an inner double-loop. The OpenCL twin keeps all
     * three axes on the outer grid (OpenCL drivers spill thread-private
     * arrays transparently to global so the 40 KB per-lane buf cost is
     * paid once per-PE not once per-lane). Apple Metal does NOT spill —
     * the only way to keep RULE_BUF_MAX at 40 KB is one device-buf slice
     * per lane, and the cheapest "slice per lane" indexing is to make
     * each lane own ONE word for its entire run. The inner double-loop
     * mirrors hashcat's canonical per-word inner-loop model.
     *
     * Implications:
     *   - Outer grid = n_words (was n_words * n_rules * mask_size).
     *   - Per-lane buf pool slot = buf_scratch_pool[gid * RULE_BUF_MAX].
     *     Reused across the inner double-loop iterations; the original
     *     word must be re-staged from `words[]` at the top of each rule
     *     iteration since the previous rule mutated buf in place.
     *   - combined_ridx packing is unchanged (rule_idx, mask_idx_local).
     *   - Hit dedup / overflow semantics unchanged. Hit order may differ
     *     from the prior grid layout (per-word clustering instead of
     *     interleaved), but gpujob_metal.m's hit-replay does not depend
     *     on order; CPU-side sort + dedup absorbs any reshuffling. */
    if (gid >= n_words) return;
    uint word_idx = gid;

    /* B3 cursor protocol read-only in Phase 1/2a. The kernel reads the
     * fields so the wire format stays the same; never advances past
     * gid==word_idx 0..n_words-1. Phase 2b+ wires the mask half. */
    /* (no-op: input_cursor_start / rule_cursor_start are populated 0). */

    /* --- Payload sub-buffer pointers (wire format invariant) --- */
    device atomic_uint *hit_count =
        (device atomic_uint *)(payload + 128);
    device const uint  *word_offset =
        (device const uint *)(payload + 132);
    uint pkt_off = 132u + (n_words * 4u);
    device const uchar *words = payload + pkt_off;

    /* Overflow channel pointers (offsets 100, 104 — inside params).
     * Typed as atomic_uint per Pattern 1 + Metal atomics. */
    device atomic_uint *ovr_set =
        (device atomic_uint *)(payload + 100);
    device atomic_uint *ovr_gid =
        (device atomic_uint *)(payload + 104);

    /* Per-lane work buffer slice. Lives in device memory now (task #250
     * — see signature comment for buf_scratch_pool). Each lane owns
     * RULE_BUF_MAX contiguous bytes starting at slot[word_idx]. */
    device uchar *buf =
        buf_scratch_pool + (ulong)word_idx * (ulong)RULE_BUF_MAX;

    /* Phase 2d.9b BCRYPT (2026-05-16): workgroup-shared threadgroup buffer
     * for the Eksblowfish S-boxes (4 x 256 uint = 4 KB per lane x
     * BCRYPT_WG_SIZE lanes = 32 KB per workgroup). Declared at kernel-
     * function scope (Metal requires threadgroup decls at function scope);
     * passed by pointer to template_finalize via the 8-arg signature
     * variant below (gated on GPU_TEMPLATE_HAS_LOCAL_BUFFER). When the
     * macro is UNDEFINED (every algo other than BCRYPT), the preprocessor
     * strips this block entirely -- non-BCRYPT instantiations are byte-
     * identical to pre-Phase-2d.9b. Mirrors gpu_template.cl lines 370-372
     * (the OpenCL twin's __local sbox_pool) byte-for-byte.
     *
     * Apple Silicon maxThreadgroupMemoryLength = 32 KB exactly matches
     * BCRYPT_WG_SIZE=8 lanes x 1024 uint x 4 bytes/uint = 32 KB. The
     * dispatch site in gpu_metal.m forces threadsPerThreadgroup = 8 for
     * JOB_BCRYPT specifically (per TRAP 2 of the Phase 2d.9b brief); any
     * larger workgroup would overflow Apple's per-workgroup threadgroup
     * memory cap. */
#ifdef GPU_TEMPLATE_HAS_LOCAL_BUFFER
    threadgroup uint sbox_pool[BCRYPT_WG_SIZE * GPU_TEMPLATE_LOCAL_BUFFER_PER_LANE];
#endif

    /* Cache word origin (wpos, wlen) once outside the inner loop.
     * Each rule iteration re-stages buf[0..wlen) from words[]. */
    uint wpos = word_offset[word_idx];
    int wlen_orig = (int)words[wpos++];
    if (wlen_orig > RULE_BUF_LIMIT) wlen_orig = RULE_BUF_LIMIT;

    /* Inner double-loop: rule_idx (outer) × mask_idx_local (inner). The
     * structure is identical for all four PSO variants — collapsing axes
     * (mask_size==1 or rule_count==1) just makes the loop body run once
     * on that axis. The runtime gate macros isolate the per-axis side
     * effects (rule walker, mask expansion, combined_ridx packing).
     *
     * rule_idx iterates [rule_base, rule_base + rule_count) — when host-
     * side sub-batching is active, rule_base is non-zero and rule_count
     * is smaller than the total program. The synthetic no-rule pass is
     * processed inside the same range (host populates the program at
     * absolute rule_idx==0; sub-batches that include rule_base==0 also
     * include the synthetic pass). */
    uint rule_end = rule_base + rule_count;
    for (uint rule_idx = rule_base; rule_idx < rule_end; rule_idx++) {
        (void)rule_idx;  /* silence -Wunused when neither axis defined */

        for (uint mask_idx_local = 0u; mask_idx_local < mask_size; mask_idx_local++) {
            (void)mask_idx_local;

            /* Re-stage the input word into buf[0..wlen). Previous inner-
             * loop iteration may have mutated buf via rules walker
             * and/or mask expansion. Cost = wlen bytes per (rule, mask)
             * tuple — typically 6-20 bytes (rockyou word length); the
             * rule walker + MD5 finalize dwarf this. */
            int wlen = wlen_orig;
            for (int i = 0; i < wlen; i++) buf[i] = words[wpos + i];

#ifdef GPU_TEMPLATE_HAS_RULES
            /* Rules-engine walker (OpenCL twin gpu_template.cl lines 379-403).
             * apply_rule() is declared in metal_md5_rules.metal — concatenated
             * into the same family TU by build_metallib.sh, AND prepended to
             * the JIT source by gpu_metal.m's rules-variant compile path.
             * Note: apply_rule's `buf` is `device uchar *` post task #250. */
            uint rpos = rule_offset[rule_idx];

            /* Synthetic no-rule discriminator (preserves feedback_no_rule_pass.md
             * semantics; mirrors gpu_md5_rules.cl rev 1.28). The first byte
             * being NUL means k==0 at apply_rule entry == the synthetic no-
             * rule pass. */
            int is_no_rule = (rule_program[rpos] == 0);
            int new_len = apply_rule(rule_program + rpos, buf, wlen);

            /* Rejection sentinel: apply_rule fired a `_ < > ! / ( )` op. */
            if (new_len < 0) continue;

            /* No-op detection: if at least one op was processed AND the
             * post-rule buffer is bit-identical to the input, the synthetic
             * no-rule pass already covered this candidate -- skip hash +
             * probe. Per feedback_no_rule_pass.md. */
            if (!is_no_rule && new_len == wlen) {
                int changed = 0;
                for (int i = 0; i < wlen; i++) {
                    if (buf[i] != words[wpos + i]) { changed = 1; break; }
                }
                if (!changed) continue;
            }
#else
            int new_len = wlen;
#endif

#ifdef GPU_TEMPLATE_HAS_MASK
            /* Phase 2b row 1 mask: decompose mask_idx into prepend+append
             * per-position indices and modify buf in-place. Mirrors
             * gpu_template.cl lines 405-531. Steps 1-4 identical to
             * pre-task-#250 layout — only address space of `buf` changes. */
            uint npre = params.n_prepend;
            uint napp = params.n_append;
            if (npre > 16u) npre = 16u;
            if (napp > 16u) napp = 16u;

            ulong mask_idx_abs = (ulong)params.mask_start
                               + (ulong)word_idx * (ulong)params.mask_offset_per_word
                               + (ulong)mask_idx_local;

            if (npre >= 1u || napp >= 1u) {
                ulong append_combos = 1u;
                for (uint j = 0u; j < napp; j++) {
                    uint sz = mask_sizes[npre + j];
                    if (sz == 0u) sz = 1u;
                    append_combos *= (ulong)sz;
                }
                if (append_combos == 0u) append_combos = 1u;
                ulong prepend_idx = mask_idx_abs / append_combos;
                ulong append_idx  = mask_idx_abs % append_combos;

                /* Step 1: shift buf right by npre bytes, high to low to
                 * avoid clobber. */
                if (npre > 0u) {
                    uint shift_dst_end = (uint)new_len + npre;
                    if (shift_dst_end > RULE_BUF_LIMIT) {
                        if ((uint)new_len + npre > RULE_BUF_LIMIT) {
                            if (new_len > (int)(RULE_BUF_LIMIT - npre))
                                new_len = (int)(RULE_BUF_LIMIT - npre);
                        }
                    }
                    for (int i = new_len - 1; i >= 0; i--) {
                        buf[i + (int)npre] = buf[i];
                    }
                }

                /* Step 2: write prepend chars at buf[0..npre). */
                if (npre > 0u) {
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

                /* Step 3: write append chars. */
                if (napp > 0u) {
                    uint append_base = (uint)new_len + npre;
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
#endif

            /* Phase 2c salt-inner loop. With HAS_SALT undef this loop
             * collapses to a single iteration (salt_count==1, salt_local
             * always 0) -- bit-identical to the Phase 2b layout. With
             * HAS_SALT defined the loop iterates num_salts_per_page
             * salts, each with its own template_finalize(salt) call;
             * the digest_compare + emit body runs once per (rule, mask,
             * salt) triple.
             *
             * Salt-inner (not salt-outer) keeps the post-task-#250 one-
             * lane-per-word invariant unbroken -- the 40 KB scratch buf
             * belongs to one word, and rebinding salt outside the lane
             * would waste the scratch slot. For Phase 2c with
             * num_salts_per_page=1 the loop runs once per lane (no perf
             * cost vs salt-outer); Phase 2d/2e can refactor to salt-outer
             * when num_salts_per_page > 1 becomes real. Mirrors
             * gpu_template.cl:243-256 + 562-585 + 639-644.
             *
             * Phase 2e GPU_TEMPLATE_HAS_PRE_SALT: hoist the inner
             * MD5(password) + hex32 encoding ONCE per (word, rule, mask)
             * via template_pre_salt(), then iterate the salt list calling
             * template_finalize_post() which consumes the carrier.
             * SALT_BATCH controls the inner tile size -- when the salt
             * loop body is small enough, the Metal compiler unrolls the
             * SALT_BATCH-stride inner cycle, amortising rule/mask
             * overhead across SALT_BATCH outer-MD5 evaluations. The
             * iteration still covers the full num_salts_per_page;
             * SALT_BATCH is an unroll knob, not a per-dispatch chunk
             * boundary (host outer chunking via METAL_SALT_CHUNK_SIZE
             * is orthogonal). Mirrors gpu_template.cl:534-585. */
#ifdef GPU_TEMPLATE_HAS_SALT
            uint salt_count = num_salts_per_page;
#else
            uint salt_count = 1u;
#endif
#if defined(GPU_TEMPLATE_HAS_PRE_SALT) && defined(GPU_TEMPLATE_HAS_SALT)
            /* Pre-salt hoist: compute password-only inner-MD5 + hex32
             * once per (word, rule, mask). All SALT_BATCH-tiled salt
             * iterations below reuse pre_state. */
            template_pre_salt_state pre_state;
            template_pre_salt(buf, new_len, params.algo_mode, pre_state);
#endif
            for (uint salt_base = 0u; salt_base < salt_count; salt_base += (uint)SALT_BATCH) {
                uint tile_end = salt_base + (uint)SALT_BATCH;
                if (tile_end > salt_count) tile_end = salt_count;
                for (uint salt_local = salt_base; salt_local < tile_end; salt_local++) {
                (void)salt_local;

                /* MD5(buf[0..new_len-1]) or MD5SALT(buf, salt). buf is
                 * `device const uchar *` (post task #250); the
                 * template_finalize variant is selected at TU build time:
                 *   - HAS_SALT undef: 3-arg form in metal_md5_core.metal.
                 *   - HAS_SALT defined: 6-arg form in metal_md5salt_core.metal.
                 * The two TUs are mutually exclusive within one MTLLibrary
                 * (only one of metal_md5_core.metal / metal_md5salt_core.metal
                 * is concatenated into each PSO variant's JIT source).
                 *
                 * Phase 2e HAS_PRE_SALT: template_finalize_post consumes
                 * pre_state (the hoisted inner-MD5 + hex32) and computes
                 * only the outer MD5(hex32 || salt). For algo_mode != 0
                 * the carrier is sentinel and the function falls through
                 * to the legacy 6-arg template_finalize. */
                template_state st;
                template_init(st);
#ifdef GPU_TEMPLATE_HAS_SALT
                /* salt_idx_global = salt_start + salt_local; salt_start
                 * is the per-dispatch page base (0 in Phase 2c since
                 * num_salts_per_page=1 and we cover the whole salt list
                 * in one dispatch). */
                uint salt_idx_global = (uint)params.salt_start + salt_local;
                uint  s_off = salt_off[salt_idx_global];
                uint  s_len = (uint)salt_lens[salt_idx_global];
#ifdef GPU_TEMPLATE_HAS_PRE_SALT
                template_finalize_post(st, pre_state,
                                       buf, new_len,
                                       salt_buf + s_off, s_len,
                                       params.algo_mode);
#else
#ifdef GPU_TEMPLATE_HAS_LOCAL_BUFFER
                /* Phase 2d.9b BCRYPT (2026-05-16): 8-arg signature variant.
                 * Mirrors gpu_template.cl line 577-578 byte-for-byte. The
                 * threadgroup-shared sbox_pool + lid are passed so the
                 * core's template_finalize claims its 1024-uint per-lane
                 * partition (sbox_pool + lid * 1024). Gated on
                 * GPU_TEMPLATE_HAS_LOCAL_BUFFER; only the BCRYPT carrier
                 * compile sets this define via preprocessorMacros in the
                 * Metal library loader. */
                template_finalize(st, buf, new_len,
                                  salt_buf + s_off, s_len,
                                  params.algo_mode,
                                  sbox_pool, lid);
#else
                template_finalize(st, buf, new_len,
                                  salt_buf + s_off, s_len,
                                  params.algo_mode);
#endif
#endif
#else
                template_finalize(st, buf, new_len);
#endif

                /* --- Iterated probe loop — Phase 1 fires it exactly once. --- */
                uint max_iter = params.max_iter;
                if (max_iter < 1u) max_iter = 1u;
                for (uint iter = 1u; iter <= max_iter; iter++) {
                    uint matched_idx = 0u;
                    if (template_digest_compare(st,
                                                compact_fp, compact_idx,
                                                params.compact_mask, params.max_probe,
                                                params.hash_data_count,
                                                hash_data_buf, hash_data_off,
                                                overflow_keys, overflow_hashes,
                                                overflow_offsets, params.overflow_count,
                                                &matched_idx))
                    {
                        /* combined_ridx packing -- extends Phase 2b with salt:
                         *   HAS_SALT:  combined_ridx = ((rule * mask_size +
                         *               mask_local) * num_salts_per_page) +
                         *               salt_local
                         *   no SALT:   Phase 2b packing (rule * mask_size +
                         *               mask_local) etc.
                         * Mirrors gpu_template.cl:639-651. Hit-replay
                         * divmods by num_salts_per_page (salt-axis), then
                         * mask_size (mask-axis), to recover all three
                         * indices. Host post-processes the salt_local
                         * field to salt_idx_global by adding params.salt_start
                         * (no-op for num_salts_per_page=1 / salt_start=0). */
#if defined(GPU_TEMPLATE_HAS_RULES) && defined(GPU_TEMPLATE_HAS_MASK) && defined(GPU_TEMPLATE_HAS_SALT)
                        uint combined_ridx =
                            (rule_idx * mask_size + mask_idx_local) * num_salts_per_page +
                            salt_local;
#elif defined(GPU_TEMPLATE_HAS_RULES) && defined(GPU_TEMPLATE_HAS_SALT)
                        uint combined_ridx =
                            rule_idx * num_salts_per_page + salt_local;
#elif defined(GPU_TEMPLATE_HAS_MASK) && defined(GPU_TEMPLATE_HAS_SALT)
                        uint combined_ridx =
                            mask_idx_local * num_salts_per_page + salt_local;
#elif defined(GPU_TEMPLATE_HAS_SALT)
                        uint combined_ridx = salt_local;
#elif defined(GPU_TEMPLATE_HAS_RULES) && defined(GPU_TEMPLATE_HAS_MASK)
                        uint combined_ridx = rule_idx * mask_size + mask_idx_local;
#elif defined(GPU_TEMPLATE_HAS_RULES)
                        uint combined_ridx = rule_idx;
#elif defined(GPU_TEMPLATE_HAS_MASK)
                        uint combined_ridx = mask_idx_local;
#else
                        uint combined_ridx = 0u;
#endif
                        uint mask = 1u << (iter & 31u);
                        template_emit_hit_or_overflow(hits, hit_count, params.max_hits,
                                                      st, word_idx, combined_ridx, iter,
                                                      hashes_shown, matched_idx, mask,
                                                      ovr_set, ovr_gid, gid);
                    }
                    /* Phase 1: template_iterate() is intentionally NOT called.
                     * Phase 2 re-adds it inside `if (iter < max_iter) { ... }`. */
                }
                } /* salt_local (inner tile) */
            } /* salt_base (SALT_BATCH-stride outer) */
        } /* mask_idx_local */
    } /* rule_idx */
}
