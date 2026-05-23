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

/* Mask wire encoding sentinel: classid byte == 0xff means MASK_LITERAL.
 * Host upload path translates mdxfind's signed -1 to this unsigned sentinel.
 * Picked from the [0..255] uchar space's high gap (max real classid = 15
 * = MASK_CUSTOM_0+7); 0xff leaves room to grow up to 254 future classes.
 * Decision D9.2.a in the A2 spec; A4 reuses the same encoding verbatim. */
#define MASK_LITERAL_SENTINEL  0xffu

/* Hard cap on mask positions per side. Mirrors mdxfind.c MAX_MASK_POS=16.
 * Per-thread private buffer size; kept identical so the wire format stays
 * 1:1 with the CPU pattern layout. */
#define MAX_MASK_POS_GPU       16

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
static inline void mask_expand_into_gpu(
    ulong index,
    __global const uchar *pattern,
    uint patlen,
    __global const uchar *mask_charsets,
    __global const uint  *mask_class_counts,
    uchar *outbuf)
{
    for (int i = (int)patlen - 1; i >= 0; i--) {
        uchar cid = pattern[i * 2];
        if (cid == MASK_LITERAL_SENTINEL) {
            outbuf[i] = pattern[i * 2 + 1];
        } else {
            uint cc = mask_class_counts[(uint)cid];
            outbuf[i] = mask_charsets[(uint)cid * MASK_CHARSET_STRIDE
                                      + (uint)(index % (ulong)cc)];
            index /= (ulong)cc;
        }
    }
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
    __global const OCLParams *params_buf = (__global const OCLParams *)payload;
    OCLParams params = *params_buf;
    uint n_words      = params.num_words;
    uint bf_num_masks = params.num_masks;
    uint app_len      = params.n_append;
    uint pre_len      = params.n_prepend;             /* BF invariant: expect 0 */
    ulong bf_start    = params.mask_start;
    uint bf_offset    = params.mask_offset_per_word;

    uint total = n_words * bf_num_masks;
    uint gid = get_global_id(0);
    if (gid >= total) return;

    /* Lane-major outer + mask-inner decomposition. Maximizes contiguous
     * writes to b_packed_buf (lane N writes at slot=N, byte_off ~ N *
     * (1+app_len)) and keeps atomic contention to one byte_counter add
     * plus one slot_counter add per lane. */
    uint synthetic_word_idx   = gid / bf_num_masks;
    uint mask_offset_in_chunk = gid % bf_num_masks;

    /* D11.3.a: ulong absolute_mask_idx. bf_start is already ulong (off 8);
     * synthetic_word_idx * bf_offset_per_word + mask_offset_in_chunk could
     * overflow uint32 for 10-digit+ BF keyspaces, so we widen all
     * arithmetic to ulong. Matches A2's mask_expand_into_gpu signature
     * and mdxfind.c:7656 CPU oracle. */
    ulong absolute_mask_idx = bf_start
                            + (ulong)synthetic_word_idx * (ulong)bf_offset
                            + (ulong)mask_offset_in_chunk;

    /* Per-thread private mask scratch. MAX_MASK_POS_GPU bytes. */
    uchar appbuf[MAX_MASK_POS_GPU];

    mask_expand_into_gpu(absolute_mask_idx, mask_pattern_append, app_len,
                         mask_charsets, mask_class_counts, appbuf);

    /* BF invariant defensive guard (spec R2): the BF fast-path gate at
     * mdxfind.c:49032 requires MaskPrependLen == 0. If the host
     * pathologically ships pre_len > 0, the kernel returns rather than
     * silently producing wrong output. Dispatcher should also assert
     * MaskPrependLen == 0 host-side and FATAL on violation. */
    if (pre_len != 0u) return;

    /* Final candidate length: [appbuf] only -- no prebuf, no input word.
     * Clamp via early-return rather than corrupting the [len] slot. With
     * MAX_MASK_POS_GPU=16, app_len <= 16 << 255 in practice; the cap is
     * structural for future-scaling. */
    uint final_len = app_len;
    if (final_len == 0u || final_len > MASK_FINAL_LEN_LIMIT) return;

    /* --- Reserve a candidate slot ---------------------------------
     * Identical atomic discipline to A1/A2/A3: byte reservation first,
     * then capacity guard, then slot reservation. Overflow flag latched
     * on either overflow. */
    uint need_bytes = 1u + final_len;   /* [len][bytes] */
    uint byte_off = atomic_add(
        &b_kernelA_state[KERNELA_STATE_BYTE_COUNTER / 4u],
        need_bytes);

    if (byte_off + need_bytes > params.packed_size) {
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

    /* --- Write [len][bytes] into packed buf -----------------------
     * Layout: [final_len][appbuf...]. No prebuf, no input word -- BF's
     * candidate IS the mask expansion. */
    b_packed_buf[byte_off] = (uchar)final_len;
    uint p = byte_off + 1u;
    for (uint i = 0; i < app_len; i++) b_packed_buf[p++] = appbuf[i];

    /* --- Write per-slot byte offset -------------------------------
     * Per contract S7.1, no parallel mask_idx sidecar. The composed
     * plaintext IS the candidate at b_packed_buf[byte_off]; mask
     * attribution can be re-derived from slot_idx if a future need
     * arises (synthetic_word_idx and mask_offset_in_chunk are
     * recoverable from slot via the absolute_mask_idx formula). */
    b_chunk_index[slot] = byte_off;

    /* Per-spec invariant 1: caller (Phase 4 host) relies on in-order
     * single-queue FIFO to ensure these writes are visible to kernel B
     * before kernel B dispatches. No explicit fence; the queue boundary
     * provides the cross-kernel global-memory visibility. */
}
