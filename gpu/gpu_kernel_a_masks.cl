/*
 * $Revision: $
 * $Log: $
 *
 */
/* gpu_kernel_a_masks.cl -- Kernel A2 (masks-only) candidate producer.
 *
 * Production kernel A variant A2 per Phase 1a sub-phase 1a.2 spec
 *   project_kernel_a_a2_masks_spec_2026-05-20.md
 * with 2026-05-30 long-mask amendment (D1.b Design Y wire format)
 *   project_kernel_a_a2_a3_long_mask_amendment_2026-05-30.md.
 *
 * Produces packed candidates from input words x mask combinations via
 * the buffer-quadruple API (b_packed_buf, b_chunk_index, b_kernelA_state,
 * b_params). Output is consumable by any kernel B that obeys the
 * buffer-quadruple contract; in this codebase the existing kernel B
 * salt-axis dispatcher (gpu_opencl_kernelb_dispatch_proto) is the
 * primary consumer.
 *
 * Topology (per parent spec a2 sec.10.2 decision):
 *   mask-major outer, word-parallel inner.
 *   global_size = n_words * n_masks
 *   word_idx    = gid % n_words
 *   mask_idx    = gid / n_words
 *
 * Wire format (2026-05-30 D1.b Design Y -- interleaved run-descriptor):
 *
 *   GPU_MASK_DESC_TAG_LIT (0x00) lit_len:u16 (little-endian) bytes[lit_len]
 *   GPU_MASK_DESC_TAG_VAR (0x01) classid:u8     (one placeholder)
 *   GPU_MASK_DESC_TAG_END (0xFF) terminator
 *
 * Per-side caps:
 *   GPU_MASK_VAR_CAP       = 16   placeholder positions
 *   GPU_MASK_LIT_BYTES_CAP = 224  literal bytes
 *   GPU_MASK_DESC_BYTES_CAP= 320  descriptor stream byte cap
 *
 * The walker (mask_expand_run_into_gpu) is a unified 2-pass design that
 * handles ALL CPU-parsable mask shapes (literal-only, var-only,
 * interleaved). Per amendment D4.a there is NO short-mask fast path;
 * the same walker covers pre-amendment workloads (?d?d, ?l, etc.) and
 * post-amendment long-lit workloads byte-identically.
 *
 * Pass 1: scan descriptor, copy literal runs verbatim, mark each VAR
 *         position with a (classid, outpos) entry in a per-thread table.
 * Pass 2: walk the VAR table HIGH-TO-LOW and consume idx %= cc / idx /= cc
 *         in order to preserve CPU's mask_expand_into semantics
 *         (mdxfind.c:7840-7848).
 *
 * Mechanical lineage: this kernel ports the CPU mask_expand_into helper
 * from mdxfind.c lines 7836-7849 directly into per-thread private state.
 * The descriptor walker handles the new wire format; the CPU oracle
 * still operates on the MaskPos[] layout.
 *
 * Authoritative buffer contract:
 *   project_two_kernel_candidate_buffer_contract.md
 *
 * Contract per buffer (identical to A1):
 *   b_packed_buf        - [len][bytes][len][bytes]... post-mask candidates.
 *   b_chunk_index       - uint32 per slot. b_chunk_index[slot] = byte
 *                         offset where this candidate's len byte lives.
 *   b_kernelA_state     - small counter buffer (same layout as A1):
 *                           offset 0 : uint slot_counter
 *                           offset 4 : uint byte_counter
 *                           offset 8 : uint overflow_flag
 *
 * OCLParams fields consumed by A2:
 *   num_words           = batch word count
 *   num_masks           = MaskTotal = MaskPrependTotal * MaskAppendTotal
 *   n_prepend           = descriptor stream BYTE LENGTH for prepend side
 *                         (post-amendment semantic; was position-count
 *                         pre-amendment)
 *   n_append            = descriptor stream BYTE LENGTH for append side
 *   base_word_idx       = source word index (for kernel B attribution)
 *   packed_size         = bytes available in b_packed_buf
 */

/* Wire-format tags (must match mdxfind.c host packer). */
#define GPU_MASK_DESC_TAG_LIT   0x00u
#define GPU_MASK_DESC_TAG_VAR   0x01u
#define GPU_MASK_DESC_TAG_END   0xFFu

/* Per-side caps (must match mdxfind.c host defines). */
#define GPU_MASK_VAR_CAP        16
#define GPU_MASK_LIT_BYTES_CAP  224
#define GPU_MASK_DESC_BYTES_CAP 320

/* Per-thread expanded-mask scratch capacity. lit_bytes + var positions. */
#define GPU_MASK_SIDE_EXPANDED_CAP (GPU_MASK_LIT_BYTES_CAP + GPU_MASK_VAR_CAP)

/* Final candidate length cap (uchar len byte). */
#define MASK_FINAL_LEN_LIMIT   255u

/* Kernel-A state buffer offsets. */
#define KERNELA_STATE_SLOT_COUNTER   0u
#define KERNELA_STATE_BYTE_COUNTER   4u
#define KERNELA_STATE_OVERFLOW_FLAG  8u
#define KERNELA_STATE_BYTES         12u

/* Charset table flat-array stride. MASK_MAX_CLASSES * 256 bytes total. */
#define MASK_CHARSET_STRIDE          256u

/* ==== Mask expander helper (post-amendment 2026-05-30) ===============
 *
 * Walks a run-descriptor stream and produces the expanded mask bytes
 * into the thread-private outbuf. Returns total expanded length.
 *
 * Per the D4.a single-unified-walker decision, both pre-amendment short
 * masks and post-amendment long-lit masks take this path.
 *
 * Per amendment §5 the walker uses a 2-pass design to preserve the CPU
 * mask_expand_into semantics (right-to-left variable consumption order
 * via decreasing-i loop at mdxfind.c:7840). Pass 1 lays out the
 * descriptor; Pass 2 walks variables high-to-low to consume idx.
 *
 * All lanes in a workgroup read the SAME descriptor (mask is workgroup-
 * invariant). Branch dispatch is warp-coherent; divergence is zero. */
static int mask_expand_run_into_gpu(
    ulong idx,
    __global const uchar *desc,
    uint desc_bytes,                          /* upper bound for safety */
    __global const uchar *mask_charsets,
    __global const uint  *mask_class_counts,
    uchar *outbuf)
{
    /* Pass 1: scan descriptor, copy literals, record var positions. */
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
            /* Bounded copy. lit_len <= GPU_MASK_LIT_BYTES_CAP per
             * gpu_pack_mask_descriptor()'s caller invariant. */
            for (uint i = 0; i < lit_len; i++) {
                outbuf[out_len + (int)i] = desc[p + i];
            }
            out_len += (int)lit_len;
            p += lit_len;
        } else if (tag == GPU_MASK_DESC_TAG_VAR) {
            if (p + 1u > desc_bytes) break;
            uchar cid = desc[p++];
            if (n_vars >= GPU_MASK_VAR_CAP) break;  /* defensive */
            var_classids[n_vars] = cid;
            var_outpos[n_vars]   = (uint)out_len;
            outbuf[out_len] = 0;
            out_len += 1;
            n_vars  += 1;
        } else {
            /* Unknown tag: bail. Host invariant guarantees this never
             * happens; we silently truncate to whatever was decoded. */
            break;
        }
    }

    /* Pass 2: expand variables HIGH-TO-LOW to preserve CPU's
     * mdxfind.c:7836-7849 left-to-right pattern indexing under reverse
     * iteration (the CPU's i = patlen-1; i--; loop). */
    for (int i = n_vars - 1; i >= 0; i--) {
        uint  cid = (uint)var_classids[i];
        uint  cc  = mask_class_counts[cid];
        if (cc == 0u) cc = 1u;
        outbuf[var_outpos[i]] = mask_charsets[cid * MASK_CHARSET_STRIDE
                                              + (uint)(idx % (ulong)cc)];
        idx /= (ulong)cc;
    }

    return out_len;
}

/* ==== Per-side cardinality helper ====================================
 *
 * Compute the product of class sizes (placeholder positions) in a
 * descriptor stream. Returns 1 when stream has zero VAR runs (matches
 * CPU convention; MaskTotal is 1 when the corresponding side is unused
 * or literal-only). */
static ulong mask_pattern_total_run(
    __global const uchar *desc,
    uint desc_bytes,
    __global const uint  *mask_class_counts)
{
    ulong total = 1ul;
    uint  p     = 0;
    while (p < desc_bytes) {
        uchar tag = desc[p++];
        if (tag == GPU_MASK_DESC_TAG_END) break;
        if (tag == GPU_MASK_DESC_TAG_LIT) {
            if (p + 2u > desc_bytes) break;
            uint lit_len = (uint)desc[p] | ((uint)desc[p + 1u] << 8);
            p += 2u + lit_len;
        } else if (tag == GPU_MASK_DESC_TAG_VAR) {
            if (p + 1u > desc_bytes) break;
            uchar cid = desc[p++];
            uint cc = mask_class_counts[(uint)cid];
            if (cc > 0u) total *= (ulong)cc;
        } else {
            break;
        }
    }
    return total;
}

/* ---- Kernel A2 (masks-only) production kernel --------------------
 *
 * Payload layout identical to A1 / md5_rules_phase0:
 *
 *   offset   0 : OCLParams params
 *   offset 128 : uint hit_count
 *   offset 132 : uint word_offset[num_words]
 *   offset 132 + 4*num_words : uchar packed_words[]
 *
 * Output buffer caps:
 *   b_packed_buf       capacity = params.packed_size bytes
 *   b_chunk_index      capacity = params.num_words * params.num_masks slots
 *
 * 8-arg signature is BYTE-IDENTICAL to pre-amendment. Only the semantic
 * of mask_pattern_prepend / mask_pattern_append changes: they are now
 * run-descriptor streams sized GPU_MASK_DESC_BYTES_CAP (320 B) each.
 * params.n_prepend / params.n_append carry the descriptor stream BYTE
 * LENGTH (used as an upper-bound safety guard inside the walker loop).
 */
__kernel
void cand_masks_phase0(
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
    uint n_masks      = params.num_masks;
    uint prep_dbytes  = params.n_prepend;   /* descriptor stream byte length */
    uint app_dbytes   = params.n_append;    /* descriptor stream byte length */
    uint total        = n_words * n_masks;

    uint gid = get_global_id(0);
    if (gid >= total) return;

    uint word_idx = gid % n_words;
    uint mask_idx = gid / n_words;

    /* Compute append-side cardinality from the descriptor stream.
     * Uniform across the workgroup (all lanes read the same descriptor),
     * so warp-coherent. */
    ulong append_total = mask_pattern_total_run(mask_pattern_append,
                                                app_dbytes,
                                                mask_class_counts);

    /* mdxfind procjob convention (mdxfind.c:12348-12349):
     *   append_idx  = idx % MaskAppendTotal
     *   prepend_idx = idx / MaskAppendTotal
     *
     * append_total is guaranteed >= 1 by mask_pattern_total_run's empty
     * stream semantics; the guard below mirrors pre-amendment defensive
     * code. */
    ulong append_idx;
    ulong prepend_idx;
    if (append_total > 0ul) {
        append_idx  = (ulong)mask_idx % append_total;
        prepend_idx = (ulong)mask_idx / append_total;
    } else {
        append_idx  = 0ul;
        prepend_idx = (ulong)mask_idx;
    }

    /* Per-thread expanded-mask scratch. 240 B each side; 480 B per WI;
     * 30 KB per WG at WG=64. Fits Pascal 65 KB and Apple M1 32 KB
     * threadgroup limits. */
    uchar prebuf[GPU_MASK_SIDE_EXPANDED_CAP];
    uchar appbuf[GPU_MASK_SIDE_EXPANDED_CAP];

    int prelen = mask_expand_run_into_gpu(prepend_idx,
                                          mask_pattern_prepend, prep_dbytes,
                                          mask_charsets, mask_class_counts,
                                          prebuf);
    int applen = mask_expand_run_into_gpu(append_idx,
                                          mask_pattern_append,  app_dbytes,
                                          mask_charsets, mask_class_counts,
                                          appbuf);

    /* Decode input word from payload (same layout as A1). */
    __global const uint   *word_offset = (__global const uint *)(payload + 132);
    uint pkt_off = 132u + (n_words * 4u);
    __global const uchar  *words = payload + pkt_off;

    uint wpos = word_offset[word_idx];
    uint wlen = (uint)words[wpos++];

    /* Final candidate length: [prebuf][word][appbuf]. */
    uint final_len = (uint)prelen + wlen + (uint)applen;
    if (final_len > MASK_FINAL_LEN_LIMIT) return;

    /* Reserve slot + byte offset (A1's atomic discipline). */
    uint need_bytes = 1u + final_len;
    uint byte_off = atomic_add(
        &b_kernelA_state[KERNELA_STATE_BYTE_COUNTER / 4u],
        need_bytes);

    if (byte_off + need_bytes > params.packed_size) {
        atomic_or(&b_kernelA_state[KERNELA_STATE_OVERFLOW_FLAG / 4u], 1u);
        return;
    }

    uint slot = atomic_add(
        &b_kernelA_state[KERNELA_STATE_SLOT_COUNTER / 4u], 1u);

    if (slot >= total) {
        atomic_or(&b_kernelA_state[KERNELA_STATE_OVERFLOW_FLAG / 4u], 1u);
        return;
    }

    /* Write [len][bytes] into packed buf. */
    b_packed_buf[byte_off] = (uchar)final_len;
    uint p = byte_off + 1u;
    for (int i = 0; i < prelen; i++) b_packed_buf[p++] = prebuf[i];
    for (uint i = 0; i < wlen;   i++) b_packed_buf[p++] = words[wpos + i];
    for (int i = 0; i < applen; i++) b_packed_buf[p++] = appbuf[i];

    b_chunk_index[slot] = byte_off;
}
