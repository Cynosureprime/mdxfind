/*
 * $Revision: 1.1 $
 * $Log: metal_md5_core.metal,v $
 * Revision 1.1  2026/05/12 13:35:11  dlr
 * Initial check-in: Phase 1 Metal MD5 core. Mirrors gpu/gpu_md5_core.cl line-by-line for the byte-exact MD5 hash chain (template_init IV, template_finalize pad+compress with in-place M[] OR-merge to preserve the 12.3% wall regression fix from Memo B Phase B2). template_iterate pre-declared for Phase 2 -i loop port (Phase 1 max_iter==1 only). Pattern 3 enforced: every helper is static inline. Pattern 1 enforced: every pointer arg is thread-qualified or device-qualified.
 *
 */
/* metal_md5_core.metal — MD5 algorithm extension for the generic
 * Metal dispatch template (Phase 1).
 *
 * Mirrors gpu/gpu_md5_core.cl (which is the OpenCL twin). Provides the
 * per-algorithm hooks that template_phase0 (in metal_template.metal)
 * calls into:
 *
 *   HASH_WORDS         — digest size in 32-bit words (4 for MD5)
 *   HASH_BLOCK_BYTES   — compress-block size (64 for MD5)
 *   template_state     — opaque per-lane state struct
 *   template_init      — initialize state to algorithm IV
 *   template_transform — absorb one HASH_BLOCK_BYTES block
 *   template_finalize  — pad and finalize a buffer; produces final state
 *   template_digest_compare — probe compact table from final state
 *   template_iterate   — re-hash digest as 32-byte hex_lc (-i loop)
 *   template_emit_hit_or_overflow — emit a hit (HASH_WORDS-aware)
 *
 * Phase 1 scope: raw MD5 only (no rules, no salt, no iter). Most
 * functions here ARE consumed by metal_template.metal even in Phase 1
 * (template_init/_finalize/_digest_compare/_emit_hit), but
 * template_iterate is unused in Phase 1 (max_iter==1 is the only
 * supported value). The fn is pre-declared anyway so Phase 2's iter
 * loop port is mechanical.
 *
 * Source-order at metallib build time (build_metallib.sh):
 *   1. metal_common.metal     — MetalParams, md5_block, probe_*, hex helpers, EMIT
 *   2. metal_md5_core.metal   — this file (per-algo hooks)
 *   3. metal_template.metal   — template_phase0 kernel
 *
 * Patterns 1/3 enforced: every fn is static inline (pattern 3); every
 * pointer arg is address-space-qualified (pattern 1).
 */

#ifndef HASH_WORDS
#define HASH_WORDS 4
#endif
#ifndef HASH_BLOCK_BYTES
#define HASH_BLOCK_BYTES 64
#endif

/* template_state: per-lane MD5 chaining values. Mirrors gpu_md5_core.cl
 * line 71. SHA1 will carry 5; SHA256 8; SHA384/512 8 ulong. The template
 * does NOT introspect the struct; it only reads digest words through
 * template_finalize's effect + template_digest_compare. */
struct template_state {
    uint h[HASH_WORDS];
};

/* template_init: install MD5 IV. */
static inline void template_init(thread template_state &st)
{
    st.h[0] = 0x67452301u;
    st.h[1] = 0xEFCDAB89u;
    st.h[2] = 0x98BADCFEu;
    st.h[3] = 0x10325476u;
}

/* template_transform: absorb one 64-byte block. The block lives in the
 * lane's thread-buffer (template_phase0 builds it in `thread uchar buf[]`)
 * so `block` is `thread const uchar *`. Mirrors gpu_md5_core.cl line 86. */
static inline void template_transform(thread template_state &st,
                                      thread const uchar *block)
{
    uint M[16];
    for (int j = 0; j < 16; j++) {
        int b = j * 4;
        M[j] = (uint)block[b]
             | ((uint)block[b + 1] << 8)
             | ((uint)block[b + 2] << 16)
             | ((uint)block[b + 3] << 24);
    }
    md5_block(st.h[0], st.h[1], st.h[2], st.h[3], M);
}

/* template_finalize: process tail, append 0x80 + length-in-bits, absorb.
 *
 * Mirrors gpu_md5_core.cl::template_finalize byte-for-byte (line 131).
 * Critical: do NOT route through template_transform — instead build
 * M[16] directly via in-place OR-merge of input bytes (same as md5_buf
 * in gpu_md5_rules.cl). The byte-stores -> byte-loads round-trip in the
 * transform-wrapper version was a 12.3% wall regression on ioblade
 * (Memo B Phase B2 perf-fix). Metal Phase 1 inherits this shape so we
 * stay byte-exact AND perf-parity with the OpenCL twin.
 *
 * Pattern 1: `data` is `device const uchar *` — task #250 (Metal scratch
 * pool migration) moved per-lane buf out of thread-private storage into
 * a device-side pool to fix the M2 Max PSO failure
 * (Compute function exceeds available temporary registers). Reads from
 * the device buf are higher-latency than thread, but Metal's caches on
 * Apple Silicon hide this well in practice — and the move is the only
 * way to keep RULE_BUF_MAX at 40 KB without truncating rule outputs. */
static inline void template_finalize(thread template_state &st,
                                     device const uchar *data, int len)
{
    uint M[16];
    int pos = 0;

    /* Complete 64-byte blocks. */
    while (len - pos >= HASH_BLOCK_BYTES) {
        for (int j = 0; j < 16; j++) {
            int b = pos + j * 4;
            M[j] = (uint)data[b]
                 | ((uint)data[b + 1] << 8)
                 | ((uint)data[b + 2] << 16)
                 | ((uint)data[b + 3] << 24);
        }
        md5_block(st.h[0], st.h[1], st.h[2], st.h[3], M);
        pos += HASH_BLOCK_BYTES;
    }

    /* Final block(s): tail bytes + 0x80 marker + zeros + length. */
    int rem = len - pos;  /* 0..63 */

    for (int j = 0; j < 16; j++) M[j] = 0u;

    for (int i = 0; i < rem; i++) {
        uint v = (uint)data[pos + i];
        M[i >> 2] |= v << ((i & 3) * 8);
    }
    M[rem >> 2] |= (uint)0x80u << ((rem & 3) * 8);

    if (rem < 56) {
        M[14] = (uint)(len * 8);
        M[15] = 0u;
        md5_block(st.h[0], st.h[1], st.h[2], st.h[3], M);
    } else {
        md5_block(st.h[0], st.h[1], st.h[2], st.h[3], M);
        for (int j = 0; j < 16; j++) M[j] = 0u;
        M[14] = (uint)(len * 8);
        M[15] = 0u;
        md5_block(st.h[0], st.h[1], st.h[2], st.h[3], M);
    }
}

/* template_iterate: -i loop step (re-hash digest-as-hex32).
 *
 * Mirrors gpu_md5_core.cl::template_iterate (line 201). Phase 1 does
 * NOT use this (max_iter==1 only); pre-declared for Phase 2.
 *
 * algo_mode == 0 (JOB_MD5): lowercase hex.
 * algo_mode == 1 (JOB_MD5UC): uppercase hex (B7.7a 2026-05-07). */
static inline void template_iterate(thread template_state &st, uint algo_mode)
{
    uint M[16];
    if (algo_mode == 1u) {
        md5_to_hex_uc(st.h[0], st.h[1], st.h[2], st.h[3], M);
    } else {
        md5_to_hex_lc(st.h[0], st.h[1], st.h[2], st.h[3], M);
    }
    M[8] = 0x80u;
    for (int j = 9; j < 14; j++) M[j] = 0u;
    M[14] = 32u * 8u;     /* 32 hex chars = 256 bits */
    M[15] = 0u;
    /* Reinitialize state to IV, then absorb. */
    st.h[0] = 0x67452301u;
    st.h[1] = 0xEFCDAB89u;
    st.h[2] = 0x98BADCFEu;
    st.h[3] = 0x10325476u;
    md5_block(st.h[0], st.h[1], st.h[2], st.h[3], M);
}

/* template_digest_compare: probe the compact table with state's first 4
 * digest words. Wrapper around probe_compact_idx.
 *
 * Pattern 1: all global tables `device const`; out_idx `thread`. */
static inline int template_digest_compare(
    thread const template_state &st,
    device const uint  *compact_fp,
    device const uint  *compact_idx,
    ulong               compact_mask,
    uint                max_probe,
    uint                hash_data_count,
    device const uchar *hash_data_buf,
    device const ulong *hash_data_off,
    device const ulong *overflow_keys,
    device const uchar *overflow_hashes,
    device const uint  *overflow_offsets,
    uint                overflow_count,
    thread uint        *out_idx)
{
    return probe_compact_idx(
        st.h[0], st.h[1], st.h[2], st.h[3],
        compact_fp, compact_idx,
        compact_mask, max_probe, hash_data_count,
        hash_data_buf, hash_data_off,
        overflow_keys, overflow_hashes, overflow_offsets, overflow_count,
        out_idx);
}

/* template_emit_hit_or_overflow: dedup+overflow-aware hit emit. Wraps
 * EMIT_HIT_4_DEDUP_OR_OVERFLOW (MD5 = 4 uint32 digest words). Mirrors
 * gpu_md5_core.cl line 266.
 *
 * Macro form (not inline fn) because the underlying EMIT macro takes
 * the hits/hit_count/etc. as device-atomic pointers; threading those
 * through an inline wrapper requires Metal's overloaded address-space
 * deduction which is fragile at JIT time. Keep it a macro to mirror
 * the OpenCL twin's macro form 1:1.  */
#define template_emit_hit_or_overflow(hits, hit_count, max_hits,                       \
                                      st_ref, widx, sidx, iter,                        \
                                      hashes_shown, matched_idx, dedup_mask,           \
                                      ovr_set, ovr_gid, lane_gid)                      \
    EMIT_HIT_4_DEDUP_OR_OVERFLOW((hits), (hit_count), (max_hits),                      \
                                 (widx), (sidx), (iter),                               \
                                 (st_ref).h[0], (st_ref).h[1],                         \
                                 (st_ref).h[2], (st_ref).h[3],                         \
                                 (hashes_shown), (matched_idx), (dedup_mask),          \
                                 (ovr_set), (ovr_gid), (lane_gid))
