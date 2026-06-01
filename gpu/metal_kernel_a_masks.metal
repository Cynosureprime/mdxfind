/*
 * $Revision: $
 * $Log: $
 *
 * Hand-ported from gpu_kernel_a_masks.cl (post 2026-05-30 long-mask
 * amendment). Walker rewrite is non-trivial (per-thread var table +
 * 2-pass design) so we maintain this as a hand-written port rather
 * than depending on cl2metal.py for translation. Source stays in
 * lockstep with the OpenCL twin via review discipline; structural
 * changes to one MUST land in the other in the same commit per
 * feedback_opencl_metal_parity.
 */

/* gpu_kernel_a_masks.cl -- Kernel A2 (masks-only) candidate producer
 * (Metal twin). See gpu_kernel_a_masks.cl for the full design memo.
 *
 * Post-amendment Metal-side notes:
 *   - __global -> device
 *   - __private (implicit on per-thread arrays) -> thread
 *   - atomic_uint with atomic_fetch_add_explicit / atomic_fetch_or_explicit
 *   - kernel entry uses [[thread_position_in_grid]]
 *
 * 8-arg signature is BYTE-IDENTICAL to OpenCL twin (post-amendment).
 */

/* Wire-format tags (must match mdxfind.c host packer). */
#define GPU_MASK_DESC_TAG_LIT   0x00u
#define GPU_MASK_DESC_TAG_VAR   0x01u
#define GPU_MASK_DESC_TAG_END   0xFFu

/* Per-side caps (must match mdxfind.c host defines). */
#define GPU_MASK_VAR_CAP        16
#define GPU_MASK_LIT_BYTES_CAP  224
#define GPU_MASK_DESC_BYTES_CAP 320

/* Per-thread expanded-mask scratch capacity. */
#define GPU_MASK_SIDE_EXPANDED_CAP (GPU_MASK_LIT_BYTES_CAP + GPU_MASK_VAR_CAP)

/* Final candidate length cap (uchar len byte). */
#define MASK_FINAL_LEN_LIMIT   255u

/* Kernel-A state buffer offsets. */
#define KERNELA_STATE_SLOT_COUNTER   0u
#define KERNELA_STATE_BYTE_COUNTER   4u
#define KERNELA_STATE_OVERFLOW_FLAG  8u
#define KERNELA_STATE_BYTES         12u

/* Charset table flat-array stride. */
#define MASK_CHARSET_STRIDE          256u

/* ==== Mask expander helper (2-pass; HIGH-TO-LOW variable expand) ==== */
static int mask_expand_run_into_gpu(ulong idx,
                                    device const uchar *desc,
                                    uint desc_bytes,
                                    device const uchar *mask_charsets,
                                    device const uint  *mask_class_counts,
                                    thread uchar *outbuf)
{
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
            for (uint i = 0; i < lit_len; i++) {
                outbuf[out_len + (int)i] = desc[p + i];
            }
            out_len += (int)lit_len;
            p += lit_len;
        } else if (tag == GPU_MASK_DESC_TAG_VAR) {
            if (p + 1u > desc_bytes) break;
            uchar cid = desc[p++];
            if (n_vars >= GPU_MASK_VAR_CAP) break;
            var_classids[n_vars] = cid;
            var_outpos[n_vars]   = (uint)out_len;
            outbuf[out_len] = 0;
            out_len += 1;
            n_vars  += 1;
        } else {
            break;
        }
    }

    for (int i = n_vars - 1; i >= 0; i--) {
        uint cid = (uint)var_classids[i];
        uint cc  = mask_class_counts[cid];
        if (cc == 0u) cc = 1u;
        outbuf[var_outpos[i]] = mask_charsets[cid * MASK_CHARSET_STRIDE
                                              + (uint)(idx % (ulong)cc)];
        idx /= (ulong)cc;
    }

    return out_len;
}

/* ==== Per-side cardinality helper ==================================== */
static ulong mask_pattern_total_run(device const uchar *desc,
                                    uint desc_bytes,
                                    device const uint  *mask_class_counts)
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

/* ---- Kernel A2 (masks-only) production kernel (Metal) ------------- */
kernel
void cand_masks_phase0(device uchar         *payload                    [[buffer(0)]],
                       device const uchar   *mask_pattern_prepend       [[buffer(1)]],
                       device const uchar   *mask_pattern_append        [[buffer(2)]],
                       device const uchar   *mask_charsets              [[buffer(3)]],
                       device const uint    *mask_class_counts          [[buffer(4)]],
                       device uchar         *b_packed_buf               [[buffer(5)]],
                       device uint          *b_chunk_index              [[buffer(6)]],
                       device atomic_uint   *b_kernelA_state            [[buffer(7)]],
                       uint                  gid                        [[thread_position_in_grid]])
{
    device const OCLParams *params_buf = (device const OCLParams *)payload;
    OCLParams params = *params_buf;
    uint n_words     = params.num_words;
    uint n_masks     = params.num_masks;
    uint prep_dbytes = params.n_prepend;
    uint app_dbytes  = params.n_append;
    uint total       = n_words * n_masks;

    if (gid >= total) return;

    uint word_idx = gid % n_words;
    uint mask_idx = gid / n_words;

    ulong append_total = mask_pattern_total_run(mask_pattern_append,
                                                app_dbytes,
                                                mask_class_counts);

    ulong append_idx;
    ulong prepend_idx;
    if (append_total > 0ul) {
        append_idx  = (ulong)mask_idx % append_total;
        prepend_idx = (ulong)mask_idx / append_total;
    } else {
        append_idx  = 0ul;
        prepend_idx = (ulong)mask_idx;
    }

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

    device const uint   *word_offset = (device const uint *)(payload + 132);
    uint pkt_off = 132u + (n_words * 4u);
    device const uchar  *words = payload + pkt_off;

    uint wpos = word_offset[word_idx];
    uint wlen = (uint)words[wpos++];

    uint final_len = (uint)prelen + wlen + (uint)applen;
    if (final_len > MASK_FINAL_LEN_LIMIT) return;

    uint need_bytes = 1u + final_len;
    uint byte_off = atomic_fetch_add_explicit(
        &b_kernelA_state[KERNELA_STATE_BYTE_COUNTER / 4u],
        need_bytes, memory_order_relaxed);

    if (byte_off + need_bytes > params.packed_size) {
        atomic_fetch_or_explicit(
            &b_kernelA_state[KERNELA_STATE_OVERFLOW_FLAG / 4u],
            1u, memory_order_relaxed);
        return;
    }

    uint slot = atomic_fetch_add_explicit(
        &b_kernelA_state[KERNELA_STATE_SLOT_COUNTER / 4u],
        1u, memory_order_relaxed);

    if (slot >= total) {
        atomic_fetch_or_explicit(
            &b_kernelA_state[KERNELA_STATE_OVERFLOW_FLAG / 4u],
            1u, memory_order_relaxed);
        return;
    }

    b_packed_buf[byte_off] = (uchar)final_len;
    uint p = byte_off + 1u;
    for (int i = 0; i < prelen; i++) b_packed_buf[p++] = prebuf[i];
    for (uint i = 0; i < wlen;   i++) b_packed_buf[p++] = words[wpos + i];
    for (int i = 0; i < applen; i++) b_packed_buf[p++] = appbuf[i];

    b_chunk_index[slot] = byte_off;
}
