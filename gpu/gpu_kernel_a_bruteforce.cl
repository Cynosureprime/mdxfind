/*
 * $Revision: $
 * $Log: $
 *
 */
/* gpu_kernel_a_bruteforce.cl -- Kernel A4 (brute-force) candidate producer.
 *
 * Production kernel A variant A4 per Phase 1a sub-phase 1a.4 spec
 *   project_kernel_a_a4_bruteforce_spec_2026-05-21.md.
 *
 * A4 closes the Phase 1a hand-written kernel-A family (A1 rules, A2 masks,
 * A3 rules+masks, A4 brute-force). Unlike A1/A2/A3, A4 consumes NO input
 * word stream -- the candidate IS the mask expansion. Per parent spec the
 * BF chunk descriptor (bf_mask_start, bf_offset_per_word, bf_num_masks)
 * arrives on OCLParams and the host pre-sizes num_words as the synthetic
 * lane count via the procjob short-circuit (mdxfind.c:10498-10501).
 *
 * Output flows through the same buffer-quadruple API as A1/A2/A3
 * (b_packed_buf, b_chunk_index, b_kernelA_state, b_params), making any
 * kernel B consumer that obeys the buffer-quadruple contract a valid
 * downstream consumer.
 *
 * Topology (per spec sec.3):
 *   global_size = num_words * bf_num_masks
 *   synthetic_word_idx   = gid / bf_num_masks
 *   mask_offset_in_chunk = gid % bf_num_masks
 *   absolute_mask_idx    = bf_mask_start
 *                        + synthetic_word_idx * bf_offset_per_word
 *                        + mask_offset_in_chunk
 *
 * D11.1.a (pure BF only): no rule axis; no prepend mask axis. The kernel
 * defends n_prepend == 0 as a BF invariant (BF fast-path eligibility gate
 * at mdxfind.c:49032 requires MaskPrependLen == 0).
 *
 * D11.2.a (charset wire encoding reuse): identical class-encoded charset
 * table as A2; the 4 persistent mask buffers (mask_pattern_prepend, mask_-
 * pattern_append, mask_charsets, mask_class_counts) are reused verbatim
 * from gpu_opencl_kernel_a_upload_mask_buffers. Zero new persistent
 * buffers. The mask_pattern_prepend slot is bound to a zeroed buffer at
 * host dispatch time for arg-symmetry with A2; the kernel never reads it.
 *
 * D11.3.a (absolute_mask_idx integer width): ulong (uint64). mdxfind
 * supports keyspaces > 2^32 (10-digit BF = 10^10), and bf_mask_start is
 * already ulong at offset 8 of OCLParams. uint32 throughout would
 * truncate; the ladder operates in ulong via the mask_expand_into_gpu
 * helper (signature matches A2).
 *
 * D11.4.a (combined OpenCL + Metal in sub-phase 1a.4): the Metal twin
 * (metal_kernel_a_bruteforce.metal) is produced by the cl2metal.py +
 * post_kernel_a_metal.py translator chain. The translator R1-R6 rules
 * are family-generic (A1/A2/A3 proven); A4 adds zero new R-rules. R1
 * (apply_rule rename) is a no-op for A4 since A4 has no rule walker.
 *
 * Mechanical lineage:
 *  - mask_expand_into_gpu and mask_pattern_total are VERBATIM COPIES of
 *    gpu_kernel_a_masks.cl rev 1.3 mask helpers. Spec R8 acknowledges
 *    the duplication; Phase 1b will hoist into a shared .cl.frag. Until
 *    then any change to the A2 mask helpers MUST BE MIRRORED HERE
 *    MANUALLY.
 *  - The kernel function cand_bruteforce_phase0 is structurally minimal:
 *    decode params, compute (synthetic_word_idx, mask_offset_in_chunk),
 *    materialize the append mask into a private uchar[MAX_MASK_POS_GPU]
 *    buffer, atomic-reserve a slot + byte offset, write [len][bytes].
 *
 * Authoritative buffer contract:
 *   project_two_kernel_candidate_buffer_contract.md
 * Phase 1a A4-variant spec:
 *   project_kernel_a_a4_bruteforce_spec_2026-05-21.md
 *
 * Contract per buffer (identical to A1/A2/A3):
 *   b_packed_buf        - [len][bytes][len][bytes]... post-mask candidates.
 *                         len byte stored as uchar; bytes follow uncompressed.
 *                         Written at slot's reserved byte offset.
 *                         Layout: [final_len][appbuf...] -- no input word
 *                         component since BF has no input stream.
 *   b_chunk_index       - uint32 per slot. b_chunk_index[slot] = byte offset
 *                         into b_packed_buf where this candidate's len byte
 *                         lives. Pure physical byte offsets only.
 *   b_kernelA_state     - small counter buffer (same layout as A1/A2/A3):
 *                           offset 0 : uint slot_counter   (atomic_inc)
 *                           offset 4 : uint byte_counter   (atomic_add for
 *                                                            variable-size
 *                                                            byte reservation)
 *                           offset 8 : uint overflow_flag  (set if either
 *                                                            counter exceeds
 *                                                            its capacity)
 *
 * Walker behavior:
 *   1. Decode payload -> params. Read num_words, bf_num_masks, app_len,
 *      pre_len, bf_mask_start, bf_offset, packed_size.
 *   2. Compute total = num_words * bf_num_masks; guard gid >= total.
 *   3. Decompose gid -> (synthetic_word_idx, mask_offset_in_chunk).
 *   4. Compute absolute_mask_idx in ulong per topology formula.
 *   5. Materialize appbuf via mask_expand_into_gpu(absolute_mask_idx,
 *      mask_pattern_append, app_len, ...).
 *   6. Defensive guard: pre_len == 0 (BF invariant per mdxfind.c:49032);
 *      final_len in [1, 255].
 *   7. Reserve slot + byte offset (identical atomic discipline to A2);
 *      write [final_len][appbuf bytes] into b_packed_buf; record byte
 *      offset in b_chunk_index[slot].
 *
 * Geometry: BF dispatch (NEW vs A2's word-major, A3's rule-major):
 *   global_size          = num_words * bf_num_masks
 *   synthetic_word_idx   = gid / bf_num_masks
 *   mask_offset_in_chunk = gid % bf_num_masks
 *
 * Reused primitives from gpu_common.cl: OCLParams struct only.
 *   - No md5_block / md5_buf / md5_to_hex_lc reference (kernel A pure).
 *   - No probe_compact_idx reference.
 *   - No EMIT_HIT_N reference.
 *
 * OCLParams fields consumed by A4:
 *   mask_start          = bf_mask_start (absolute mask cursor at chunk start)
 *   num_words           = synthetic lane count (per-chunk; pre-sized by host)
 *   num_masks           = bf_num_masks (per-iter mask range)
 *   n_prepend           = 0 (BF invariant; A4 reads as defensive guard)
 *   n_append            = MaskAppendLen (mask byte count per candidate)
 *   base_word_idx       = host-set (for future kernel B attribution)
 *   packed_size         = bytes available in b_packed_buf
 *   mask_offset_per_word = bf_offset_per_word (per-lane mask stride)
 *   inner_iter          = read but NOT used (A4 v1 ignores inner_iter > 1;
 *                          Phase 1a.4.x deferred)
 *
 * OCLParams fields NOT consumed by A4 (set by host but ignored):
 *   num_rules           (no rule axis)
 *   input_cursor_start  (A4 v1 single-shot per dispatch)
 *   rule_cursor_start   (no rule axis)
 */

/* 2026-05-30 long-mask amendment: A4 absorbs the new descriptor wire
 * format via helper rename + 1-byte-END prebuf binding. The kernel-arg
 * signature is unchanged; the host now uploads a 320 B descriptor
 * stream into mask_pattern_append and a 1-byte END descriptor into
 * mask_pattern_prepend. params.n_prepend = 1 (just the END byte),
 * params.n_append = app descriptor byte length.
 *
 * BF invariant `pre_len == 0` translates post-amendment to "prebuf walker
 * sees END at offset 0 and returns 0 expanded bytes" -- the kernel just
 * skips the prebuf composition naturally without an explicit guard. */

/* ==== A4_PROFILE_VARIANT scaffolding (2026-05-30, BF-engine decomposition) ===
 * Per architect spec project_kernel_a_a4_profile_variant_spec_2026-05-30.md
 * D2.a (V0..V5). When kernel is JIT-built with -DA4_PROFILE_VARIANT=N
 * (N in 1..5), per-lane body is progressively stubbed. V0 (macro undefined)
 * = byte-identical to production cand_bruteforce_phase0.
 *
 * Macro NAME differs from A1's PROFILE_VARIANT so A1 + A4 programs can
 * coexist in the same JIT context without cross-contamination if both
 * env vars are set (per spec §4 D1.a).
 *
 * Variant semantics (V0-V_N = component cost-share):
 *   V0 baseline -- full cand_bruteforce_phase0 (uint4 stores by default
 *      per 2026-05-30 A4 C5 default-on refactor; see V0 site below).
 *   V1 no atomic claim (per-lane gid*256 strided offset; per-byte write
 *      still happens). Measures atomic_add cost (CAVEAT: likely
 *      CONFOUNDED on NVIDIA/AGX/RDNA per spec §9 R1 + A1 #346/#354
 *      priors -- strided offset breaks SLC write coalescing).
 *   V2 no per-byte write (atomic claim still runs; per-byte loop +
 *      chunk_index store omitted). Measures C5 per-byte write cost
 *      share. Slot counter intentionally NOT incremented so host
 *      actual_slots stays 0.
 *   V3 stub charset walk (mask_expand_run_into_gpu returns minimal
 *      fixed length, fills outbuf with 'A' without consulting descriptor
 *      stream or charset table). EXPECTED A4 DOMINANT measurement;
 *      V0-V3 = charset-walk (C3) cost share.
 *   V4 decode-only (gid decompose + absolute_mask_idx ladder, then
 *      return BEFORE mask_expand_run_into_gpu). V4 - V5 = C2 decode +
 *      index-arithmetic cost share.
 *   V5 empty kernel (return immediately BEFORE param decode). Measures
 *      C1 dispatch overhead + WG scheduling baseline.
 *
 * Variants V1/V3/V4/V5: b_kernelA_state slot_counter and byte_counter
 * untouched (or restored to 0 implicitly). V2 atomic_add still fires
 * but slot_counter stays 0. Host sees actual_slots == 0 and the
 * harness-mode return path short-circuits. Crack-parity for V1..V5 is
 * INTENTIONALLY NOT preserved (timing-only stubs).
 *
 * 2026-05-30 A4 C5 DEFAULT-ON REFACTOR: the V6 cell that previously held
 * the env-gated KNOBG variant has been REMOVED. The V0 baseline now
 * unconditionally uses uint4 (16-byte) device stores -- the Knob-G-analog
 * shape that V6 was measuring is permanently the production path. Setting
 * A4_PROFILE_VARIANT=6 falls through to V0 (the active range is 0..5).
 *
 * Host gate (per spec §4 R5): activated ONLY when KERNEL_A_VARIANT=4
 * is also active, via gpu_opencl_kernel_a4_profile_variant() accessor. */

/* Wire-format tags (must match mdxfind.c host packer). */
#define GPU_MASK_DESC_TAG_LIT   0x00u
#define GPU_MASK_DESC_TAG_VAR   0x01u
#define GPU_MASK_DESC_TAG_END   0xFFu

/* Per-side caps (must match mdxfind.c host defines). */
#define GPU_MASK_VAR_CAP        16
#define GPU_MASK_LIT_BYTES_CAP  224
#define GPU_MASK_DESC_BYTES_CAP 320
#define GPU_MASK_SIDE_EXPANDED_CAP (GPU_MASK_LIT_BYTES_CAP + GPU_MASK_VAR_CAP)

/* Deprecated alias retained for grep compatibility. */
#define MAX_MASK_POS_GPU       16
#define MASK_LITERAL_SENTINEL  0xffu

/* Bounds: final candidate length cap. The [len] uchar byte caps at 255
 * regardless. For BF the final candidate is purely the append mask
 * expansion so final_len == app_len <= MAX_MASK_POS_GPU (which is itself
 * <= 16 in practice). The 255 cap is structural -- we early-return rather
 * than corrupt the [len] slot under any pathological future scaling. */
#define MASK_FINAL_LEN_LIMIT   255u

/* Kernel-A state buffer offsets. Single source of truth for host wiring
 * to mirror via fixed-offset writes/reads. Identical to A1/A2/A3. */
#define KERNELA_STATE_SLOT_COUNTER   0u
#define KERNELA_STATE_BYTE_COUNTER   4u
#define KERNELA_STATE_OVERFLOW_FLAG  8u
#define KERNELA_STATE_BYTES         12u

/* Charset table flat-array stride. MASK_MAX_CLASSES * 256 bytes total. */
#define MASK_CHARSET_STRIDE          256u

/* ==== Mask expander helper ============================================
 *
 * VERBATIM COPY from gpu_kernel_a_masks.cl rev 1.3 (Phase 1a sub-phase
 * 1a.2). Spec R8 TODO: hoist into a shared .cl.frag in Phase 1b. Until
 * then any change to A2's helper MUST be mirrored here manually.
 *
 * Literal port of mask_expand_into from mdxfind.c lines 7646-7660:
 *
 *   for (i = patlen - 1; i >= 0; i--) {
 *       if (pattern[i].classid == MASK_LITERAL)
 *           buf[i] = pattern[i].literal;
 *       else {
 *           buf[i] = mc->chars[index % mc->count];
 *           index /= mc->count;
 *       }
 *   }
 *
 * GPU adaptations:
 *  - pattern is a flat 2-byte-per-entry uchar buffer; classid at offset
 *    i*2, literal at offset i*2+1. MASK_LITERAL_SENTINEL (0xff) replaces
 *    the CPU's signed -1.
 *  - mask_charsets is the flat 16*256 byte table; mask_class_counts is
 *    the uint32-per-class count vector.
 *  - outbuf is thread-private (the apply path passes a stack array).
 *  - index is ulong to handle absolute mask indices up to 2^63; the
 *    division ladder operates on running ulong values.
 *
 * Static inline (kernel A1 convention; no noinline needed since this
 * helper has no hash compression). */
/* 2026-05-30 amendment: walker rewritten for run-descriptor wire
 * format (Design Y). 2-pass: scan descriptor copying literals + recording
 * VAR positions; then expand VARs HIGH-TO-LOW to preserve CPU's
 * right-to-left consumption semantics. */
static int mask_expand_run_into_gpu(
    ulong idx,
    __global const uchar *desc,
    uint desc_bytes,
    __global const uchar *mask_charsets,
    __global const uint  *mask_class_counts,
    uchar *outbuf)
{
#if defined(A4_PROFILE_VARIANT) && A4_PROFILE_VARIANT == 3
    /* V3 stub: skip the EXPENSIVE part of the charset walk (the modulo
     * + global-byte-load + idx/=cc ladder per VAR), but PRESERVE the
     * output length contract so downstream host caps don't overflow.
     *
     * Compute the true expanded length cheaply (one pass scanning the
     * descriptor stream summing LIT lens + counting VARs -- same shape
     * as the V0 pass 1 walk, but WITHOUT the costly VAR pass 2). Then
     * fill outbuf with fixed 'A' to that length. V0_us - V3_us
     * approximates the VAR pass-2 cost share (C3b + C3c per spec §2);
     * the LIT scan (C3a) is shared between V0 and V3, so V3 measures
     * only the per-VAR charset-fetch + divide-ladder elimination.
     *
     * Per spec §2: V3 is the "primary A4 expected dominant" measurement.
     * The aggregate (V0-V3) IS still expected to dominate on A4 because
     * the per-VAR expansion (the actual stubbed work) is the dominant
     * compute for the brute-force engine. */
    (void)idx;
    (void)mask_charsets;
    (void)mask_class_counts;
    int stub_len = 0;
    {
        uint p = 0;
        while (p < desc_bytes) {
            uchar tag = desc[p++];
            if (tag == GPU_MASK_DESC_TAG_END) break;
            if (tag == GPU_MASK_DESC_TAG_LIT) {
                if (p + 2u > desc_bytes) break;
                uint lit_len = (uint)desc[p] | ((uint)desc[p + 1u] << 8);
                p += 2u;
                if (p + lit_len > desc_bytes) break;
                stub_len += (int)lit_len; p += lit_len;
            } else if (tag == GPU_MASK_DESC_TAG_VAR) {
                if (p + 1u > desc_bytes) break;
                p++;
                stub_len += 1;
            } else break;
        }
    }
    if (stub_len < 0) stub_len = 0;
    if (stub_len > GPU_MASK_SIDE_EXPANDED_CAP) stub_len = GPU_MASK_SIDE_EXPANDED_CAP;
    for (int i = 0; i < stub_len; i++) outbuf[i] = 'A';
    return stub_len;
#else
    uchar var_classids[GPU_MASK_VAR_CAP];
    uint  var_outpos[GPU_MASK_VAR_CAP];
    int   n_vars  = 0;
    int   out_len = 0;
    uint  p       = 0;
    while (p < desc_bytes) {
        uchar tag = desc[p++];
        if (tag == GPU_MASK_DESC_TAG_END) break;
        if (tag == GPU_MASK_DESC_TAG_LIT) {
            if (p + 2u > desc_bytes) break;
            uint lit_len = (uint)desc[p] | ((uint)desc[p + 1u] << 8);
            p += 2u;
            if (p + lit_len > desc_bytes) break;
            for (uint i = 0; i < lit_len; i++) outbuf[out_len + (int)i] = desc[p + i];
            out_len += (int)lit_len; p += lit_len;
        } else if (tag == GPU_MASK_DESC_TAG_VAR) {
            if (p + 1u > desc_bytes) break;
            uchar cid = desc[p++];
            if (n_vars >= GPU_MASK_VAR_CAP) break;
            var_classids[n_vars] = cid; var_outpos[n_vars] = (uint)out_len;
            outbuf[out_len] = 0; out_len++; n_vars++;
        } else break;
    }
    for (int i = n_vars - 1; i >= 0; i--) {
        uint cid = (uint)var_classids[i];
        uint cc  = mask_class_counts[cid]; if (cc == 0u) cc = 1u;
        outbuf[var_outpos[i]] = mask_charsets[cid * MASK_CHARSET_STRIDE
                                              + (uint)(idx % (ulong)cc)];
        idx /= (ulong)cc;
    }
    return out_len;
#endif  /* A4_PROFILE_VARIANT == 3 */
}

/* ==== Total-cardinality helper =========================================
 *
 * VERBATIM COPY from gpu_kernel_a_masks.cl rev 1.3 (Phase 1a sub-phase
 * 1a.2). Same hoist note as mask_expand_into_gpu applies.
 *
 * Compute the product of class sizes for a (pattern, patlen) pair.
 * Returns 1 when patlen == 0 (matches CPU mdxfind convention; MaskTotal
 * is 1 when the corresponding side is unused). Returns 1 also for any
 * literal-only pattern (literal positions contribute factor 1).
 *
 * NOTE: A4 does not actually CALL mask_pattern_total in the kernel body
 * (the host supplies bf_num_masks directly via OCLParams.num_masks, and
 * the absolute_mask_idx formula carries the stride explicitly). The
 * helper is kept here byte-identical to A2 for translator stability
 * (cl2metal.py overlay shape) and Phase 1b hoist consistency. */
static inline ulong mask_pattern_total(
    __global const uchar *pattern,
    uint patlen,
    __global const uint  *mask_class_counts)
{
    ulong total = 1ul;
    for (uint i = 0; i < patlen; i++) {
        uchar cid = pattern[i * 2];
        if (cid != MASK_LITERAL_SENTINEL) {
            total *= (ulong)mask_class_counts[(uint)cid];
        }
    }
    return total;
}

/* ---- Kernel A4 (brute-force) production kernel -------------------
 *
 * Payload layout (per spec sec.6 "Payload structure"):
 *
 *   offset   0 : OCLParams params         (128 bytes)
 *   offset 128 : uint hit_count           (4 bytes; reserved for payload
 *                                            symmetry; unused by kernel A)
 *
 * Total payload size = 132 bytes. A4 does NOT carry word_offset[] /
 * packed_words[] tails because BF has no input word stream -- the
 * candidate IS the mask expansion. The mask_pattern_prepend arg is
 * host-bound to a zeroed buffer for arg-symmetry with A2.
 *
 * 8-arg signature: byte-identical to A2. mask_pattern_prepend slot
 * present but unread (kernel gates on pre_len == 0 via the BF invariant).
 *
 * params.base_word_idx is read by kernel B (host attribution); A4 does
 * not consume it directly. The host sets it before dispatch.
 *
 * params.packed_size is read as the b_packed_buf capacity. Overflow
 * detection: any candidate whose reservation would push byte_counter
 * past packed_size sets overflow_flag and returns without writing.
 *
 * Output buffer caps:
 *   b_packed_buf       capacity = params.packed_size bytes
 *   b_chunk_index      capacity = params.num_words * params.num_masks slots
 *                                  (worst case: every (lane, mask) emits
 *                                   one slot -- no rejection path in A4
 *                                   under the BF invariant).
 */

__kernel
void cand_bruteforce_phase0(
    __global uchar         *payload,
    __global const uchar   *mask_pattern_prepend,
    __global const uchar   *mask_pattern_append,
    __global const uchar   *mask_charsets,
    __global const uint    *mask_class_counts,
    __global uchar         *b_packed_buf,
    __global uint          *b_chunk_index,
    __global volatile uint *b_kernelA_state
    )
{
#if defined(A4_PROFILE_VARIANT) && A4_PROFILE_VARIANT == 5
    /* V5: empty kernel; return BEFORE any payload decode. Measures C1
     * (dispatch overhead + WG scheduling + kernel launch latency). */
    return;
#endif
    __global const OCLParams *params_buf = (__global const OCLParams *)payload;
    OCLParams params = *params_buf;
    uint n_words      = params.num_words;
    uint bf_num_masks = params.num_masks;
    uint app_dbytes   = params.n_append;              /* descriptor stream byte length */
    /* n_prepend is now descriptor byte length (==1 for A4's END-only prebuf);
     * A4 never composes prebuf, so we ignore it post-amendment. */
    ulong bf_start    = params.mask_start;
    uint bf_offset    = params.mask_offset_per_word;

    uint total = n_words * bf_num_masks;
    uint gid = get_global_id(0);
    if (gid >= total) return;

    uint synthetic_word_idx   = gid / bf_num_masks;
    uint mask_offset_in_chunk = gid % bf_num_masks;

    ulong absolute_mask_idx = bf_start
                            + (ulong)synthetic_word_idx * (ulong)bf_offset
                            + (ulong)mask_offset_in_chunk;

#if defined(A4_PROFILE_VARIANT) && A4_PROFILE_VARIANT == 4
    /* V4: decode-only -- everything up through absolute_mask_idx is done
     * (C2 component); NO mask_expand, NO atomic, NO write. Sentinel
     * side-effect on unreachable bit pattern keeps absolute_mask_idx
     * live (compiler must not dead-strip the ulong ladder). V4 - V5 =
     * decode + index-arithmetic (C2) cost share; V3 - V4 = charset walk
     * (C3) isolated. */
    if (absolute_mask_idx == 0xffffffffffffffffUL) {
        atomic_or(&b_kernelA_state[KERNELA_STATE_OVERFLOW_FLAG / 4u], 0u);
    }
    return;
#endif

    /* Per-thread expanded-mask scratch (240 B for new wire format). */
    uchar appbuf[GPU_MASK_SIDE_EXPANDED_CAP];

    int applen = mask_expand_run_into_gpu(absolute_mask_idx,
                                          mask_pattern_append, app_dbytes,
                                          mask_charsets, mask_class_counts,
                                          appbuf);

    /* Final candidate length: [appbuf] only -- no prebuf, no input word. */
    uint final_len = (uint)applen;
    if (final_len == 0u || final_len > MASK_FINAL_LEN_LIMIT) return;

    /* --- Reserve a candidate slot ---------------------------------
     * Identical atomic discipline to A1/A2/A3: byte reservation first,
     * then capacity guard, then slot reservation. Overflow flag latched
     * on either overflow. */
    uint need_bytes = 1u + final_len;   /* [len][bytes] */

#if defined(A4_PROFILE_VARIANT) && A4_PROFILE_VARIANT == 1
    /* V1 (no atomic claim): substitute per-lane deterministic offsets
     * for atomic_add slot + byte reservation. Per-byte write STILL
     * HAPPENS so V0 - V1 attributes the atomic cost (CAVEAT: confounded
     * per spec §9 R1 -- strided gid*256 offset breaks SLC write
     * coalescing on NVIDIA/AGX/RDNA). slot_counter + byte_counter stay
     * at zero (no atomic = no increment) so host sees actual_slots == 0
     * and harness short-circuits.
     *
     * Capacity guards: per spec §3 V1 description -- the fake byte_off
     * = gid * 256 stays within pre-allocated packed buffer for any gid
     * covered by the host's chunk cap; defensively re-check. */
    uint byte_off = gid * 256u;
    uint slot     = gid;
    if (byte_off + need_bytes > params.packed_size) return;
    if (slot >= total) return;
    b_packed_buf[byte_off] = (uchar)final_len;
    {
        uint p = byte_off + 1u;
        for (int i = 0; i < applen; i++) b_packed_buf[p++] = appbuf[i];
    }
    b_chunk_index[slot] = byte_off;
#elif defined(A4_PROFILE_VARIANT) && A4_PROFILE_VARIANT == 2
    /* V2 (no candidate write): atomic claim still runs (so V0 - V2
     * attributes the per-byte write loop C5). Per-byte memcpy + index
     * store are removed. slot_counter intentionally NOT incremented
     * (host actual_slots stays 0 -> harness short-circuits). Use a
     * private read-write of byte_counter to keep the atomic_add live
     * so the compiler does not fold it. */
    uint byte_off = atomic_add(
        &b_kernelA_state[KERNELA_STATE_BYTE_COUNTER / 4u],
        need_bytes);
    /* Keep byte_off live without producing a visible effect (or-with-0
     * is a no-op functionally; prevents DCE of the atomic_add). */
    if (byte_off == 0xffffffffu) {
        atomic_or(&b_kernelA_state[KERNELA_STATE_OVERFLOW_FLAG / 4u], 0u);
    }
    /* Slot-counter NOT incremented. Per-byte write loop + chunk_index
     * store omitted (the measured target). */
#else
    /* V0 (production baseline). 2026-05-30 A4 C5 default-on refactor:
     * the uint4 (16-byte) device-store write path is now the ONLY write
     * path -- the env-gated KNOBG_A4_VEC_WRITE preprocessor branch and
     * the legacy per-byte loop have both been removed. Per the paired
     * Phase 0 verdicts (project_a4_c5_phase0_opencl_ptx_verdict_-
     * 2026-05-30 + project_a4_c5_phase0_metal_air_verdict_2026-05-30):
     * explicit V_uint4 stores lower to 16-byte device stores on both
     * Apple AGX (metal-llc) and NVIDIA Pascal/Maxwell (NVVM); the legacy
     * V_char per-byte loop remained scalar (st.global.u8 / store i8).
     * NVIDIA delivered -15% Pascal / -21% Maxwell wall on the production
     * fixture; Apple AGX delivered null (load-side bound; see followup).
     *
     * Round need_bytes up to a 16-byte multiple. The atomic shape is a
     * single atomic_add; only the value is rounded. Slot-start alignment
     * proof: base ptr 16-aligned + each running sum adds a multiple of
     * 16 -> every byte_off is 16-aligned. */
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

    uint slot = atomic_add(
        &b_kernelA_state[KERNELA_STATE_SLOT_COUNTER / 4u], 1u);

    /* Slot-index capacity: bounded by num_words * num_masks (uniform
     * emit-rate; no rule rejection, no length-vary). */
    if (slot >= total) {
        atomic_or(&b_kernelA_state[KERNELA_STATE_OVERFLOW_FLAG / 4u], 1u);
        return;
    }

    /* --- Write [len][bytes] into packed buf via uint4 stores ----------
     * Direct-from-appbuf uint4 stores (D5.a shape -- no stage[] copy).
     * Build the [len|bytes] payload via shifted reads from the thread-
     * private appbuf into 16-byte uint4 packets. need_aligned/16 stores;
     * tail reads past applen are safe (appbuf is 240 B private), pad
     * bytes in the dst tail are undefined but never inspected by the
     * consumer (plen byte caps the read). emit_len cap for A4 is
     * MaskAppendLen <= MAX_MASK_POS_GPU = 16; need_aligned in [16,32];
     * nvec in [1,2]. */
    {
        __global uint4 *dst = (__global uint4 *)(b_packed_buf + byte_off);
        uint hdr0 = (uint)final_len
                  | ((uint)appbuf[0]  <<  8)
                  | ((uint)appbuf[1]  << 16)
                  | ((uint)appbuf[2]  << 24);
        uint hdr1 = (uint)appbuf[3]
                  | ((uint)appbuf[4]  <<  8)
                  | ((uint)appbuf[5]  << 16)
                  | ((uint)appbuf[6]  << 24);
        uint hdr2 = (uint)appbuf[7]
                  | ((uint)appbuf[8]  <<  8)
                  | ((uint)appbuf[9]  << 16)
                  | ((uint)appbuf[10] << 24);
        uint hdr3 = (uint)appbuf[11]
                  | ((uint)appbuf[12] <<  8)
                  | ((uint)appbuf[13] << 16)
                  | ((uint)appbuf[14] << 24);
        dst[0] = (uint4)(hdr0, hdr1, hdr2, hdr3);
        uint nvec = need_aligned / 16u;
        for (uint v = 1u; v < nvec; v++) {
            uint base = v * 16u - 1u;
            uint w0 = (uint)appbuf[base + 0u]
                    | ((uint)appbuf[base + 1u] <<  8)
                    | ((uint)appbuf[base + 2u] << 16)
                    | ((uint)appbuf[base + 3u] << 24);
            uint w1 = (uint)appbuf[base + 4u]
                    | ((uint)appbuf[base + 5u] <<  8)
                    | ((uint)appbuf[base + 6u] << 16)
                    | ((uint)appbuf[base + 7u] << 24);
            uint w2 = (uint)appbuf[base + 8u]
                    | ((uint)appbuf[base + 9u] <<  8)
                    | ((uint)appbuf[base + 10u] << 16)
                    | ((uint)appbuf[base + 11u] << 24);
            uint w3 = (uint)appbuf[base + 12u]
                    | ((uint)appbuf[base + 13u] <<  8)
                    | ((uint)appbuf[base + 14u] << 16)
                    | ((uint)appbuf[base + 15u] << 24);
            dst[v] = (uint4)(w0, w1, w2, w3);
        }
    }

    /* --- Write per-slot byte offset -------------------------------
     * Per contract S7.1, no parallel mask_idx sidecar. The composed
     * plaintext IS the candidate at b_packed_buf[byte_off]; mask
     * attribution can be re-derived from slot_idx if a future need
     * arises (synthetic_word_idx and mask_offset_in_chunk are
     * recoverable from slot via the absolute_mask_idx formula). */
    b_chunk_index[slot] = byte_off;
#endif  /* A4_PROFILE_VARIANT 1 / 2 / V0 selection */

    /* Per-spec invariant 1: caller (Phase 4 host) relies on in-order
     * single-queue FIFO to ensure these writes are visible to kernel B
     * before kernel B dispatches. No explicit fence; the queue boundary
     * provides the cross-kernel global-memory visibility. */
}
