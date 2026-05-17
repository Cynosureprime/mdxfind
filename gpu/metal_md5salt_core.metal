/*
 * $Revision: 1.1 $
 * $Log: metal_md5salt_core.metal,v $
 * Revision 1.1  2026/05/13 14:44:27  dlr
 * Initial check-in: Phase 2c-A MD5SALT Metal core.
 *
 */
/* metal_md5salt_core.metal -- MD5SALT (JOB_MD5SALT, op e31, hashcat -m 10):
 * MD5(hex32(MD5(pass)) || salt) double-MD5 chain. Metal Phase 2c port of
 * gpu/gpu_md5salt_core.cl algo_mode 0 only.
 *
 * Mirrors gpu_md5salt_core.cl byte-for-byte for the e31 inner+outer chain
 * (template_init IV, template_finalize: inner MD5(pass) -> hex32 lowercase
 * -> outer MD5(hex32 || salt)). Address-space port:
 *   __global  -> device
 *   __private -> thread
 *
 * Phase 2c scope:
 *   - algo_mode 0 ONLY (e31 lowercase hex32 + raw salt append).
 *   - Drops modes 1-6 (UC/REV/SUB8_24/MD5_MD5SALTMD5PASS/HMAC) -- those
 *     are Phase 2d+ siblings.
 *   - Drops salt_pack_uint helper (the simpler byte-by-byte append loop is
 *     used inline; OpenCL twin's uint-pack micro-opt was for Pascal RMW
 *     pathology -- Apple Metal has no equivalent and the few-cycle
 *     difference is dominated by md5_block cost).
 *   - Reuses unsalted MD5 template_iterate convention (HEX_FEEDBACK
 *     matches CPU JOB_MD5SALT iter convention per
 *     gpu_md5salt_core.cl:749).
 *
 * Phase 2e additions (pre-salt hoist + SIMD lane batching):
 *   - template_pre_salt_state struct (port of gpu_md5salt_core.cl:91-94).
 *   - template_pre_salt: computes the password-only inner MD5+hex32 once
 *     per (word, rule, mask), stores into a `thread` carrier; sentinel
 *     for modes 1-6 (which fall through to legacy template_finalize).
 *   - template_finalize_post: consumes the carrier + salt, computes
 *     outer MD5(hex32 || salt). Mirrors gpu_md5salt_core.cl:682-747.
 *   - Gated behind GPU_TEMPLATE_HAS_PRE_SALT in metal_template.metal;
 *     the SALT_BATCH lane-batch loop iterates the carrier across up to
 *     SALT_BATCH salts per template_pre_salt evaluation. Mode 0 only;
 *     modes 1-6 retain per-salt full-finalize via the sentinel path.
 *
 * Source-order in the salted JIT TU:
 *   1. metal_common.metal         (MetalParams, md5_block, md5_to_hex_lc,
 *                                  probe_compact_idx, EMIT macros)
 *   2. metal_md5_rules.metal      (apply_rule walker; concatenated for
 *                                  GPU_TEMPLATE_HAS_RULES variants)
 *   3. metal_md5salt_core.metal   (THIS file -- replaces metal_md5_core.metal
 *                                  for salted TUs; provides the 6-arg
 *                                  template_finalize)
 *   4. metal_template.metal       (the template_phase0 kernel, with
 *                                  GPU_TEMPLATE_HAS_SALT defined)
 *
 * Critical: this file and metal_md5_core.metal define the same set of
 * symbols (template_state, template_init, template_iterate,
 * template_digest_compare, template_emit_hit_or_overflow, AND
 * template_finalize). Only ONE may be linked into any given MTLLibrary.
 * The unsalted libraries (none/R/M/RM) concat metal_md5_core.metal; the
 * salted libraries (S/RS/MS/RMS) concat metal_md5salt_core.metal. The
 * call-site dispatch lives at the #ifdef GPU_TEMPLATE_HAS_SALT block in
 * metal_template.metal -- it picks the 3-arg template_finalize when
 * HAS_SALT is undef, the 6-arg form when defined.
 *
 * Patterns 1/3 enforced (per metal_jit_harness.m):
 *   - Every pointer arg is address-space-qualified (device or thread).
 *   - Every function is static inline.
 *   - No threadgroup decls (Pattern 5).
 */

#ifndef HASH_WORDS
#define HASH_WORDS 4
#endif
#ifndef HASH_BLOCK_BYTES
#define HASH_BLOCK_BYTES 64
#endif

/* template_state: per-lane MD5 chaining values. Mirrors
 * metal_md5_core.metal's struct + gpu_md5salt_core.cl line 43. */
struct template_state {
    uint h[HASH_WORDS];
};

/* template_pre_salt_state: carrier for the password-only portion of the
 * salted MD5 chain. Phase 2e pre-salt hoist (mirrors
 * gpu_md5salt_core.cl:91-94). Lifted OUT of the per-salt loop -- one
 * inner-MD5+hex32 evaluation amortises across SALT_BATCH outer-MD5
 * evaluations.
 *
 * Layout: 68 bytes per lane (16 uints + 1 uint sentinel). Lives in
 * `thread` storage; written once by template_pre_salt(), read repeatedly
 * by template_finalize_post() inside the inner salt loop.
 *
 * Sentinel: inner_len == 0xFFFFFFFFu means "fall back to legacy
 * template_finalize" (for algo_modes other than 0; Phase 2e only hoists
 * mode 0, modes 1-6 remain on the per-salt full-finalize path). */
struct template_pre_salt_state {
    uint M[16];
    uint inner_len;
};

#define TEMPLATE_PRE_SALT_SENTINEL 0xFFFFFFFFu

/* template_init: install MD5 IV. */
static inline void template_init(thread template_state &st)
{
    st.h[0] = 0x67452301u;
    st.h[1] = 0xEFCDAB89u;
    st.h[2] = 0x98BADCFEu;
    st.h[3] = 0x10325476u;
}

/* template_transform: kept for API completeness -- never called on the
 * hot path. template_finalize streams M[] directly. */
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

/* template_finalize: e31 MD5SALT chain. Mirrors gpu_md5salt_core.cl
 * template_finalize algo_mode==0 path (lines 198-606). Drops modes 1-6
 * branches and the salt_pack_uint helper.
 *
 * Steps:
 *   1. Inner MD5(buf[0..len)) -> local h0/h1/h2/h3 (4 uints).
 *   2. md5_to_hex_lc into M[0..7] (32 lowercase hex chars, byte order
 *      matches CPU prmd5).
 *   3. Outer MD5(hex32 || salt) -> st.h[0..3]:
 *      - inner_len = 32 (hex32 spans M[0..7]).
 *      - Salt bytes appended at M[8..] (byte offset 32 onward).
 *      - 0x80 marker + length-bits padding per MD5 spec.
 *      - One md5_block call when total_len < 56; otherwise a second
 *        block with the length-bits trailing.
 *
 * Pattern 1: `data` is `device const uchar *` (per task #250 buf-pool
 * migration -- the per-lane slice in metal_template.metal lives in
 * device storage). `salt_buf` is also `device const uchar *` -- the
 * host's salt MTLBuffer is uploaded via gpu_metal_set_salt and
 * referenced by offset.
 *
 * The algo_mode parameter is read but only mode 0 is meaningful in
 * Phase 2c (modes 1-6 are out of scope per file header). Callers in
 * the Phase 2c HAS_SALT block always pass params.algo_mode==0 for
 * JOB_MD5SALT. The unused-arg pattern silences -Wunused without code
 * size cost; future Phase 2d may put mode-switch logic here. */
static inline void template_finalize(thread template_state &st,
                                     device const uchar *data, int len,
                                     device const uchar *salt_buf,
                                     uint slen,
                                     uint algo_mode)
{
    (void)algo_mode;  /* Phase 2c: only mode 0; modes 1-6 deferred to 2d+ */

    uint M[16];
    int pos = 0;
    uint h0 = 0x67452301u, h1 = 0xEFCDAB89u,
         h2 = 0x98BADCFEu, h3 = 0x10325476u;

    /* Step 1: inner MD5(data[0..len)). Streams complete 64-byte blocks,
     * then the tail block(s) with 0x80 + length-bits padding. */
    while (len - pos >= HASH_BLOCK_BYTES) {
        for (int j = 0; j < 16; j++) {
            int b = pos + j * 4;
            M[j] = (uint)data[b]
                 | ((uint)data[b + 1] << 8)
                 | ((uint)data[b + 2] << 16)
                 | ((uint)data[b + 3] << 24);
        }
        md5_block(h0, h1, h2, h3, M);
        pos += HASH_BLOCK_BYTES;
    }

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
        md5_block(h0, h1, h2, h3, M);
    } else {
        md5_block(h0, h1, h2, h3, M);
        for (int j = 0; j < 16; j++) M[j] = 0u;
        M[14] = (uint)(len * 8);
        M[15] = 0u;
        md5_block(h0, h1, h2, h3, M);
    }

    /* Step 2: hex-encode digest1 into M[0..7] (32 lowercase chars =
     * 32 bytes). Zero the trailing M[8..15] so the salt-append +
     * 0x80 + length-bits land in clean memory. mirrors
     * gpu_md5salt_core.cl line 491 + 550. */
    for (int j = 8; j < 16; j++) M[j] = 0u;
    md5_to_hex_lc(h0, h1, h2, h3, M);
    uint inner_len = 32u;

    /* Step 3: outer MD5(hex32 || salt). Salt bytes append at byte
     * offset inner_len (==32) in M[]. For total_len < 56 one block
     * suffices; otherwise two blocks. Byte-wise append via shift+OR
     * mirrors the legacy `mbytes` loop in OpenCL (we drop
     * salt_pack_uint -- the uint-pack micro-opt was for Pascal byte-
     * RMW pathology; Apple Metal has no equivalent gain, and the
     * cycle delta is dwarfed by the two md5_block calls). */
    uint total_len = inner_len + slen;
    if (total_len < 56u) {
        /* Salt bytes + 0x80 marker fit in the first (and only) block. */
        for (uint i = 0u; i < slen; i++) {
            uint pos_byte = inner_len + i;
            uint v = (uint)salt_buf[i];
            M[pos_byte >> 2] |= v << ((pos_byte & 3) * 8);
        }
        {
            uint eom_pos = inner_len + slen;
            M[eom_pos >> 2] |= (uint)0x80u << ((eom_pos & 3) * 8);
        }
        M[14] = total_len * 8u;
        M[15] = 0u;
        st.h[0] = 0x67452301u;
        st.h[1] = 0xEFCDAB89u;
        st.h[2] = 0x98BADCFEu;
        st.h[3] = 0x10325476u;
        md5_block(st.h[0], st.h[1], st.h[2], st.h[3], M);
    } else {
        /* First block: salt bytes only (no 0x80 yet -- it lives in
         * the second block). first_chunk = bytes of salt that fit in
         * this block after inner_len. */
        uint first_chunk = 64u - inner_len;
        if (first_chunk > slen) first_chunk = slen;
        for (uint i = 0u; i < first_chunk; i++) {
            uint pos_byte = inner_len + i;
            uint v = (uint)salt_buf[i];
            M[pos_byte >> 2] |= v << ((pos_byte & 3) * 8);
        }
        st.h[0] = 0x67452301u;
        st.h[1] = 0xEFCDAB89u;
        st.h[2] = 0x98BADCFEu;
        st.h[3] = 0x10325476u;
        md5_block(st.h[0], st.h[1], st.h[2], st.h[3], M);

        /* Second block: remaining salt bytes + 0x80 + length-bits.
         * rem_salt < 64; if rem_salt < 56 the length-bits land in
         * this same block, otherwise one more block. */
        for (int j = 0; j < 16; j++) M[j] = 0u;
        uint rem_salt = slen - first_chunk;
        for (uint i = 0u; i < rem_salt; i++) {
            uint v = (uint)salt_buf[first_chunk + i];
            M[i >> 2] |= v << ((i & 3) * 8);
        }
        M[rem_salt >> 2] |= (uint)0x80u << ((rem_salt & 3) * 8);
        if (rem_salt < 56u) {
            M[14] = total_len * 8u;
            M[15] = 0u;
            md5_block(st.h[0], st.h[1], st.h[2], st.h[3], M);
        } else {
            md5_block(st.h[0], st.h[1], st.h[2], st.h[3], M);
            for (int j = 0; j < 16; j++) M[j] = 0u;
            M[14] = total_len * 8u;
            M[15] = 0u;
            md5_block(st.h[0], st.h[1], st.h[2], st.h[3], M);
        }
    }
}

/* template_pre_salt: Phase 2e pre-salt hoist. Compute the password-only
 * portion of the salted MD5 chain ONCE per (word, rule, mask), then
 * amortise across the inner salt loop via template_finalize_post.
 *
 * Mirrors gpu_md5salt_core.cl:615-674 verbatim for algo_mode==0 (e31).
 * Mode 0 path:
 *   1. Streaming MD5 of `data[0..len)` -> 4-uint digest1.
 *   2. md5_to_hex_lc into pre.M[0..7] (32 lowercase hex chars).
 *   3. pre.M[8..15] zeroed (downstream salt-append + 0x80 + padding lands
 *      in clean memory).
 *   4. pre.inner_len = 32u.
 *
 * Modes 1-6: write sentinel and return. template_finalize_post falls
 * through to the legacy 6-arg template_finalize for those modes.
 *
 * Pattern 1: `data` is `device const uchar *` (per task #250 buf-pool
 * migration). `pre` is `thread` (per-lane register storage). */
static inline void template_pre_salt(device const uchar *data, int len,
                                     uint algo_mode,
                                     thread template_pre_salt_state &pre)
{
    if (algo_mode != 0u) {
        pre.inner_len = TEMPLATE_PRE_SALT_SENTINEL;
        return;
    }

    uint Mi[16];
    int pos = 0;
    uint h0 = 0x67452301u, h1 = 0xEFCDAB89u,
         h2 = 0x98BADCFEu, h3 = 0x10325476u;

    while (len - pos >= HASH_BLOCK_BYTES) {
        for (int j = 0; j < 16; j++) {
            int b = pos + j * 4;
            Mi[j] = (uint)data[b]
                  | ((uint)data[b + 1] << 8)
                  | ((uint)data[b + 2] << 16)
                  | ((uint)data[b + 3] << 24);
        }
        md5_block(h0, h1, h2, h3, Mi);
        pos += HASH_BLOCK_BYTES;
    }

    int rem = len - pos;
    for (int j = 0; j < 16; j++) Mi[j] = 0u;
    for (int i = 0; i < rem; i++) {
        uint v = (uint)data[pos + i];
        Mi[i >> 2] |= v << ((i & 3) * 8);
    }
    Mi[rem >> 2] |= (uint)0x80u << ((rem & 3) * 8);

    if (rem < 56) {
        Mi[14] = (uint)(len * 8);
        Mi[15] = 0u;
        md5_block(h0, h1, h2, h3, Mi);
    } else {
        md5_block(h0, h1, h2, h3, Mi);
        for (int j = 0; j < 16; j++) Mi[j] = 0u;
        Mi[14] = (uint)(len * 8);
        Mi[15] = 0u;
        md5_block(h0, h1, h2, h3, Mi);
    }

    /* Hex-encode digest1 into pre.M[0..7]; zero pre.M[8..15] so the
     * salt-append + 0x80 + length-bits path lands in clean memory. The
     * md5_to_hex_lc helper uses plain `=` stores so pre.M[0..7] is fully
     * overwritten -- only the upper half needs explicit zeroing. */
    for (int j = 8; j < 16; j++) pre.M[j] = 0u;
    md5_to_hex_lc(h0, h1, h2, h3, pre.M);
    pre.inner_len = 32u;
}

/* template_finalize_post: Phase 2e pre-salt hoist consumer. Runs ONCE
 * per salt under the lane-batch pattern. For algo_mode 0 (the only
 * hoisted mode) consumes pre.M (the 32-char hex of MD5(password)) and
 * computes the outer MD5(hex32 || salt). For sentinel pre_state (any
 * algo_mode != 0) dispatches to the legacy 6-arg template_finalize.
 *
 * Mirrors gpu_md5salt_core.cl:682-747 verbatim. Drops salt_pack_uint
 * (same rationale as legacy template_finalize -- Apple Metal lacks
 * Pascal byte-RMW pathology). */
static inline void template_finalize_post(thread template_state &st,
                                          thread const template_pre_salt_state &pre,
                                          device const uchar *data, int len,
                                          device const uchar *salt_buf,
                                          uint slen,
                                          uint algo_mode)
{
    if (pre.inner_len == TEMPLATE_PRE_SALT_SENTINEL) {
        /* Fallback: legacy template_finalize handles modes 1-6 + the
         * full inner computation. Pass through buf+len unchanged. */
        template_finalize(st, data, len, salt_buf, slen, algo_mode);
        return;
    }

    /* Mode 0 hoisted path: pre.M[0..7] holds the 32-char lowercase hex
     * of the inner MD5. Compute outer MD5(hex32 || salt) into st.h. */
    uint M[16];
    for (int j = 0; j < 16; j++) M[j] = pre.M[j];
    uint inner_len = pre.inner_len;

    uint total_len = inner_len + slen;
    if (total_len < 56u) {
        for (uint i = 0u; i < slen; i++) {
            uint pos_byte = inner_len + i;
            uint v = (uint)salt_buf[i];
            M[pos_byte >> 2] |= v << ((pos_byte & 3) * 8);
        }
        {
            uint eom_pos = inner_len + slen;
            M[eom_pos >> 2] |= (uint)0x80u << ((eom_pos & 3) * 8);
        }
        M[14] = total_len * 8u;
        M[15] = 0u;
        st.h[0] = 0x67452301u;
        st.h[1] = 0xEFCDAB89u;
        st.h[2] = 0x98BADCFEu;
        st.h[3] = 0x10325476u;
        md5_block(st.h[0], st.h[1], st.h[2], st.h[3], M);
    } else {
        uint first_chunk = 64u - inner_len;
        if (first_chunk > slen) first_chunk = slen;
        for (uint i = 0u; i < first_chunk; i++) {
            uint pos_byte = inner_len + i;
            uint v = (uint)salt_buf[i];
            M[pos_byte >> 2] |= v << ((pos_byte & 3) * 8);
        }
        st.h[0] = 0x67452301u;
        st.h[1] = 0xEFCDAB89u;
        st.h[2] = 0x98BADCFEu;
        st.h[3] = 0x10325476u;
        md5_block(st.h[0], st.h[1], st.h[2], st.h[3], M);

        for (int j = 0; j < 16; j++) M[j] = 0u;
        uint rem_salt = slen - first_chunk;
        for (uint i = 0u; i < rem_salt; i++) {
            uint v = (uint)salt_buf[first_chunk + i];
            M[i >> 2] |= v << ((i & 3) * 8);
        }
        M[rem_salt >> 2] |= (uint)0x80u << ((rem_salt & 3) * 8);
        if (rem_salt < 56u) {
            M[14] = total_len * 8u;
            M[15] = 0u;
            md5_block(st.h[0], st.h[1], st.h[2], st.h[3], M);
        } else {
            md5_block(st.h[0], st.h[1], st.h[2], st.h[3], M);
            for (int j = 0; j < 16; j++) M[j] = 0u;
            M[14] = total_len * 8u;
            M[15] = 0u;
            md5_block(st.h[0], st.h[1], st.h[2], st.h[3], M);
        }
    }
}

/* template_iterate: -i loop step (re-hash digest-as-hex32).
 *
 * Mirrors metal_md5_core.metal template_iterate (lowercase hex) which
 * mirrors gpu_md5salt_core.cl line 750 -- HEX_FEEDBACK matches CPU
 * JOB_MD5SALT iter convention (mdxfind.c default `prmd5` at line 9706).
 * No per-iter salt re-application -- salt is consumed ONCE in
 * template_finalize step 3, then the iter loop re-hashes the digest as
 * hex32 the same way unsalted MD5 does. */
static inline void template_iterate(thread template_state &st)
{
    uint M[16];
    md5_to_hex_lc(st.h[0], st.h[1], st.h[2], st.h[3], M);
    M[8] = 0x80u;
    for (int j = 9; j < 14; j++) M[j] = 0u;
    M[14] = 32u * 8u;     /* 32 hex chars = 256 bits */
    M[15] = 0u;
    st.h[0] = 0x67452301u;
    st.h[1] = 0xEFCDAB89u;
    st.h[2] = 0x98BADCFEu;
    st.h[3] = 0x10325476u;
    md5_block(st.h[0], st.h[1], st.h[2], st.h[3], M);
}

/* template_digest_compare: probe the compact table with state's first
 * 4 digest words. Identical to metal_md5_core.metal's variant. */
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
 * EMIT_HIT_4_DEDUP_OR_OVERFLOW (MD5 = 4 uint32 digest words). Identical
 * to metal_md5_core.metal's macro -- the hit-record format is the same
 * (4 digest words after widx + sidx + iter), regardless of whether the
 * digest came from raw MD5 or MD5SALT. */
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
