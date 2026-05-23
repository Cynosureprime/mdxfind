/*
 * hx_emit_metal.c -- Metal per-backend emit helpers for the hx P4
 *                    state-machine walker.
 *
 * Sub-phase 2a.4 (2026-05-21): Metal twin of 2a.3's e347 OpenCL emitter
 * plus real implementations of the per-opcode helpers (which previously
 * returned -1 with stderr stubs). Structural mirror of hx_emit_opencl.c
 * with the OpenCL->Metal token-level translations applied per
 * post_kernel_a_metal.py R1-R6:
 *
 *   __kernel void              -> kernel void
 *   __global                   -> device
 *   __local                    -> threadgroup
 *   __private                  -> (omitted; Metal default)
 *   const __global             -> device const
 *   __constant                 -> constant
 *   get_global_id(0)           -> uint gid [[thread_position_in_grid]]
 *   atomic_inc(&x)             -> atomic_fetch_add_explicit(&x, 1u,
 *                                                           memory_order_relaxed)
 *   barrier(CLK_LOCAL_MEM_FENCE)-> threadgroup_barrier(
 *                                     mem_flags::mem_threadgroup)
 *   bitselect (scalar)         -> arithmetic Ch/Maj forms
 *   __attribute__((noinline))  -> omit (Apple Metal inlines + benefits;
 *                                 per feedback_md5_block_noinline_pascal.md)
 *
 * The emitted source is intended to be JIT-compiled via
 * gpu_metal_jit_compile_source_with_common() (added to gpu_metal.m in
 * 2a.4 Part D), which PREPENDS metal_common_str at newLibraryWithSource
 * time. That gives the emitted source access to:
 *
 *   md5_block          -- metal_common.metal (thread uint& refs +
 *                         thread const uint *M; static inline; Apple
 *                         inlines + benefits)
 *   MetalParams /
 *   OCLParams typedef  -- metal_common.metal (byte-identical to
 *                         OCLParams; 128 bytes; struct fields named
 *                         identically)
 *   HIT_STRIDE         -- metal_common.metal
 *   EMIT_HIT_4_DEDUP_OR_OVERFLOW -- metal_common.metal (typed atomic_uint*
 *                                  args vs OpenCL's volatile uint*)
 *   probe_compact_idx  -- metal_common.metal (`device const` qualifiers
 *                         on read-only tables; `thread uint *` for
 *                         result-write pointer)
 *
 * The Metal emitter must compute the SAME chain as the OpenCL emitter
 * (`md5(md5_hex(md5_hex(pass)) . salt)`), NOT the broken hand-written
 * gpu_kernelb_md5md5salt_nocache.cl's `md5(md5_bin(pass) . salt)` chain.
 * The hand-written kernel B is a STRUCTURAL reference for tp0 shape, not
 * a byte-exact oracle. 2a.5 validates against the hashpipe CPU oracle.
 *
 * Per feedback_no_nested_block_comments_in_cl.md the emitted source
 * uses // comments only (rule applies to .metal as well as .cl).
 *
 * Per feedback_metal_xcode26_bitselect_scalar.md no scalar bitselect()
 * appears in any emitted Metal source; the e347 emitter uses only
 * arithmetic ops + helper functions backed by metal_common.metal's
 * MTL_MD5_FF/GG/HH/II macros (which use ((b&c)|(~b&d)) etc.).
 *
 * Per feedback_external_failures_are_fatal.md the walker itself never
 * silently drops; allocation failures bubble up as negative returns and
 * the caller (mdxfind.c harness) treats them as fatal.
 *
 * Sub-phase 5a.3 (2026-05-22): adds hx_emit_family_md5pass_metal --
 * Metal twin of the 5a.2 OpenCL family emitter for the MAKE_MD5PASS
 * family. Same structural shape: validate code[1] callname is md5,
 * resolve outer primitive via hx_callname_for_entry(entry, 4), FATAL on
 * unknown/5b-deferred/non-SHA1 (other 5a primitives are 5a.4 scope),
 * emit shared helpers md5_buf_global_metal + state_to_hex32_bytes_metal,
 * emit outer body emit_outer_sha1_concat_then_hash_metal, emit the
 * kernel entry kernelb_hx_codegen_phase0. SHA1 outer body includes
 * BE-to-LE state byte-swap per feedback_be_state_primitives_need_-
 * byteswap_in_codegen.md (single 4-shift idiom per word, mirrors
 * OpenCL twin). Kernel signature uses 18 sequential [[buffer(N)]] args
 * mirroring the 2a.6 e347 Metal twin; salt-related args 3/4/5 are
 * IGNORED by the family body (unsalted). Atomic counters hit_count
 * and hashes_shown typed `device atomic_uint *`; ovr_set / ovr_gid
 * explicit kernel args (Metal can't cast through non-atomic pointer
 * to atomic_uint, unlike the OpenCL twin that aliases off payload).
 *
 * $Revision: 1.6 $
 * $Log: hx_emit_metal.c,v $
 * Revision 1.6  2026/05/23 05:23:35  dlr
 * sub-phase 5a.4 Metal twin fan out 6 remaining MAKE_MD5PASS family primitives md4 rmd160 sha224 sha256 sha384 sha512 outer body emit helpers each structural mirror of the OpenCL twin in hx_emit_opencl.c 1.7 Metal-specific idioms device const uchar pass thread uint state pointers state_to_hex32_bytes_metal helper md5_buf_global_metal helper md4_block from metal_common.metal 1.24 sha256_block sha512_block rmd160_block from existing metal_common.metal byte-swap idiom pure shifts and masks no scalar bitselect per feedback_metal_xcode26_bitselect_scalar emit_family_md5pass_kernel_metal switch on outer_id selects helper name per primitive declaration line plus call line built inline 7-arm dispatch hx_emit_family_md5pass_metal widened the per-primitive switch dispatch HX_PRIM_MD5 e123 outlier remains deferred cross-arch validated PASS 8 of 8 on Apple M2 Max dev3 for e122 e159 e163 e165 e167 e169 plus e161 5a.3 regression e347 production regression PASS
 *
 * Revision 1.5  2026/05/23 03:21:35  dlr
 * sub-phase 5a.3 Metal twin of OpenCL family emitter add hx_emit_family_md5pass_metal entry point mirrors 5a.2 OpenCL shape validates code1 callname md5 resolves outer primitive via hx_callname_for_entry FATAL on unknown 5b-deferred or non-SHA1 emits shared helpers md5_buf_global_metal and state_to_hex32_bytes_metal reused verbatim from e347 Metal twin emits per-primitive outer body emit_outer_sha1_concat_then_hash_metal includes BE-to-LE state byte-swap on both single-block fast path and multi-block tail per feedback memo so h0 to h4 LE uint frame matches harness compact_fp probe and EMIT_HIT_4 contract; kernel signature mirrors 2a.6 e347 Metal twin 18 args at same buffer indices salt-table args 3 4 5 IGNORED by family body ovr_set and ovr_gid explicit args 16 17 atomic_uint Metal cannot cast non-atomic pointer to atomic uniform; per-thread topology no SALT_BATCH outer loop; no scalar bitselect; only SHA1 reaches emit_family_md5pass_kernel_metal other primitives FATAL with deferred-to-5a4 diagnostic includes hx_patterns.h hx_spec_entry.h hx_emit_primitives.h
 *
 * Revision 1.4  2026/05/22 03:59:30  dlr
 * sub-phase 2a6 Metal twin byte-exact fix ports the two 2a5 OpenCL emitter fixes chain bug renames state_to_le_bytes16_metal to state_to_hex32_bytes_metal and md5_buf_private16_metal to md5_buf_private32_metal so kernel body feeds inner_hex 32 bytes into second inner MD5 padding bug adds first_has_pad flag in md5_outer_hex_combine_metal multi block branch mechanical port of OpenCL 1.3 to 1.4 diff with Metal token translations validated byte exact dev3 across smoke medium large edge_minlen edge_maxlen edge_empty all matched zero diff
 *
 * Revision 1.4  2026/05/22 04:30:00  dlr
 * sub-phase 2a.6 Metal twin byte-exact correctness fix ports the two 2a.5 OpenCL emitter fixes; chain bug replace state_to_le_bytes16_metal helper with state_to_hex32_bytes_metal emitting 32 lowercase hex chars and replace md5_buf_private16_metal with md5_buf_private32_metal processing 32 hex char input so kernel body feeds inner_hex 32 bytes into second inner MD5 producing MD5 hex32 MD5 hex32 MD5 pass concat salt matching mdxfind CPU JOB_MD5MD5SALT at mdxfind.c line 23174 and OpenCL twin rev 1.4; padding bug add first_has_pad flag in md5_outer_hex_combine_metal multi block branch so 0x80 EOM byte lands in first block at position 32 plus salt_in_first when salt fits but pad does not previously omitted broke all slen in 24 through 31; mechanical port of OpenCL 1.3 to 1.4 diff with Metal token translations device const uchar instead of global thread uint instead of private; validated byte-exact on Apple M Metal dev3 with smoke 32 medium 1024 large 1048576 edge_minlen edge_maxlen edge_empty all matched zero diff
 *
 * Revision 1.3  2026/05/22 02:53:08  dlr
 * sub-phase 2a.4 Metal twin of e347 OpenCL emitter plus per-opcode helpers; structural mirror of OpenCL emitter with token-level translations; emits 12781 bytes JIT clean on Apple M2 Max dev3.local; computes correct chain md5(hex32(md5(md5(pass))) . salt) matching hashpipe CPU oracle not the broken hand-written kernel; FATAL on JIT failure per external-failures-are-fatal
 *
 * Revision 1.3  2026/05/21 23:23:29  dlr
 * sub-phase 2a.4 real Metal per-opcode helpers plus e347 MD5MD5SALT tp0 pattern Metal twin emitter. Mirrors 2a.3 OpenCL emitter shape with token-level translations: kernel void, device, threadgroup, kernel arg uint gid thread_position_in_grid, atomic_fetch_add_explicit memory_order_relaxed, no noinline (Apple inlines + benefits), no scalar bitselect (none needed; MD5 only). Emitted source JITs via gpu_metal_jit_compile_source_with_common which prepends metal_common_str at newLibraryWithSource. Computes the correct e347 chain md5(md5_hex(md5_hex(pass)) . salt) matching the OpenCL twin, NOT the broken hand-written kernel B chain.
 *
 */

#include <stddef.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "hx_spec.h"
#include "hx_walker.h"
#include "hx_emit.h"
#include "hx_patterns.h"
#include "hx_spec_entry.h"
#include "hx_emit_primitives.h"

/* ---- skeleton helpers (2a.1; real implementations as of 2a.4) ---- */

int hx_emit_kernel_attribute_metal(char **out, size_t *out_cap,
                                   size_t *out_len)
{
    /* OpenCL emits "__kernel "; Metal emits "kernel " (no underscore
     * prefix; an explicit space terminator so the caller can append the
     * return type next). */
    return hx_appendf(out, out_cap, out_len, "kernel ");
}

int hx_emit_address_space_global_metal(char **out, size_t *out_cap,
                                       size_t *out_len)
{
    /* OpenCL __global -> Metal device. */
    return hx_appendf(out, out_cap, out_len, "device ");
}

int hx_emit_thread_id_load_metal(char **out, size_t *out_cap,
                                 size_t *out_len,
                                 const char *var_name)
{
    /* Metal does NOT have a get_global_id() builtin -- the gid is bound
     * via a kernel-argument attribute [[thread_position_in_grid]]. The
     * walker's EMIT_HEADER state already declared `gid` in the kernel
     * signature; here we just emit a no-op annotation comment so the
     * walker's per-state // hx: state marker is preserved and downstream
     * code can reference `gid` directly.
     *
     * Note: the variable name argument is honored only for OpenCL where
     * the walker emits `const uint <name> = get_global_id(0)`. On Metal
     * the binding is in the kernel signature attribute and the walker
     * always names it `gid`. If a caller asks for a different name we
     * emit an alias declaration so the rest of the emitted body still
     * compiles. */
    if (var_name && strcmp(var_name, "gid") != 0) {
        return hx_appendf(out, out_cap, out_len,
                          "  // hx: gid bound via [[thread_position_in_grid]] kernel arg\n"
                          "  uint %s = gid;\n", var_name);
    }
    return hx_appendf(out, out_cap, out_len,
                      "  // hx: gid bound via [[thread_position_in_grid]] kernel arg\n");
}

int hx_emit_atomic_inc_metal(char **out, size_t *out_cap, size_t *out_len,
                             const char *counter_expr)
{
    /* OpenCL atomic_inc(&x) -> Metal atomic_fetch_add_explicit(&x, 1u,
     * memory_order_relaxed). Counter MUST be typed `device atomic_uint *`
     * by the caller per metal_common.metal Pattern 1. */
    const char *expr = counter_expr ? counter_expr : "&counter[0]";
    return hx_appendf(out, out_cap, out_len,
                      "  atomic_fetch_add_explicit(%s, 1u, memory_order_relaxed);\n",
                      expr);
}

int hx_emit_payload_load_metal(char **out, size_t *out_cap, size_t *out_len)
{
    /* Skeleton placeholder analogous to the OpenCL twin. Sub-phase 2a.5
     * may expand this to a real payload-load if the generic walker is
     * used outside the e347 fast-path. */
    return hx_appendf(out, out_cap, out_len,
                      "  // hx: payload-load stub (Metal; deferred to 2a.5+)\n");
}

/* ---- per-opcode helpers (2a.2; real implementations as of 2a.4) ---- */

/*
 * Mirrors hx_emit_push_var_opencl. Metal-side address-space qualifier is
 * `thread` (default for stack-resident locals; we omit it to match the
 * OpenCL emitter's `uint _v_N` shape). The "_len" suffix disambiguates
 * multi-PUSH-VAR programs (e31, e347).
 */
int hx_emit_push_var_metal(char **out, size_t *cap, size_t *len,
                           int slot, const char *varname)
{
    return hx_appendf(out, cap, len,
        "  // hx: OP_PUSH_VAR slot=%d name=\"%s\"\n"
        "  // hx: (Metal placeholder -- actual variable load deferred to 2a.5)\n"
        "  uint _v_%zu = (uint)gid;\n",
        slot, varname ? varname : "?", *len);
}

/*
 * OP_PUSH_STR: emit a constant array initializer. OpenCL __constant ->
 * Metal `constant` (Metal's `constant` address space is the device-side
 * read-only equivalent). Symbol is suffixed with *len for uniqueness.
 */
int hx_emit_push_str_metal(char **out, size_t *cap, size_t *len,
                           int stridx, const char *literal, int literal_len)
{
    int rc = hx_appendf(out, cap, len,
        "  // hx: OP_PUSH_STR stridx=%d len=%d\n",
        stridx, literal_len);
    if (rc < 0) return rc;
    if (literal && literal_len > 0) {
        /* Metal `constant` storage requires a function-scope locals can
         * be `thread`; we use `thread const` for the lane-local copy
         * to mirror OpenCL's __constant scope without escalating to
         * Metal's device-wide constant address space. */
        rc = hx_appendf(out, cap, len,
                        "  thread const uchar _s_%d[%d] = {",
                        stridx, literal_len);
        if (rc < 0) return rc;
        for (int i = 0; i < literal_len; i++) {
            rc = hx_appendf(out, cap, len, "%s0x%02x",
                            i ? "," : "",
                            (unsigned)(unsigned char)literal[i]);
            if (rc < 0) return rc;
        }
        rc = hx_appendf(out, cap, len, "};\n");
    }
    return rc;
}

/* OP_PUSH_INT: declare a long literal (port of OpenCL form verbatim). */
int hx_emit_push_int_metal(char **out, size_t *cap, size_t *len,
                           int64_t ival)
{
    return hx_appendf(out, cap, len,
        "  // hx: OP_PUSH_INT %lld\n"
        "  long _i_%zu = (long)%lldL;\n",
        (long long)ival, *len, (long long)ival);
}

/* OP_STORE: comment-only annotation; storage handled by walker. */
int hx_emit_store_metal(char **out, size_t *cap, size_t *len,
                        int slot, const char *varname)
{
    return hx_appendf(out, cap, len,
        "  // hx: OP_STORE slot=%d name=\"%s\"\n",
        slot, varname ? varname : "?");
}

/*
 * OP_CALL: recognise md5 only (mirror of OpenCL twin). Other CALLs emit
 * an annotation comment and return -1 -- caller treats as fatal per
 * feedback_external_failures_are_fatal.md.
 *
 * The placeholder uint4 form mirrors the OpenCL twin: gid-derived
 * register declaration that the JIT compiler will accept. e347 lands at
 * a higher level via the pattern-recognised fast path, so this generic
 * CALL stub is exercised only for programs the pattern detector does
 * NOT match.
 */
int hx_emit_call_metal(char **out, size_t *cap, size_t *len,
                       const char *fn_name, int nargs, uint8_t role)
{
    int rc;
    if (!fn_name) fn_name = "?";

    rc = hx_appendf(out, cap, len,
        "  // hx: OP_CALL fn=\"%s\" nargs=%d role=%u\n",
        fn_name, nargs, (unsigned)role);
    if (rc < 0) return rc;

    if (strcmp(fn_name, "md5") == 0) {
        return hx_appendf(out, cap, len,
            "  uint4 _md5_state_%zu = uint4((uint)gid, (uint)gid ^ 0x67452301u, "
            "(uint)gid ^ 0xefcdab89u, (uint)gid ^ 0x98badcfeu);\n", *len);
    }

    rc = hx_appendf(out, cap, len,
        "  // hx: function '%s' not yet supported on Metal generic path "
        "(use pattern-recognised fast path)\n",
        fn_name);
    if (rc < 0) return rc;
    fprintf(stderr,
            "hx codegen: Metal CALL '%s' not implemented in sub-phase 2a.4 "
            "generic dispatch (use pattern fast path)\n", fn_name);
    return -1;
}

/* OP_CONCAT: comment-only annotation; semantics deferred. */
int hx_emit_concat_metal(char **out, size_t *cap, size_t *len)
{
    return hx_appendf(out, cap, len,
        "  // hx: OP_CONCAT (Metal placeholder; semantics deferred to 2a.5+)\n");
}

/*
 * OP_HALT: terminate. Mirrors OpenCL twin's never-taken atomic_inc dead-
 * write so the JIT can't DCE the body. Counter passed via the trivial
 * kernel signature must be typed `device atomic_uint *` -- the walker's
 * EMIT_HEADER state declares it that way for the Metal backend.
 */
int hx_emit_halt_metal(char **out, size_t *cap, size_t *len)
{
    return hx_appendf(out, cap, len,
        "  // hx: OP_HALT (terminate program; result on stack top)\n"
        "  if (gid == 0xffffffffu) {\n"
        "    atomic_fetch_add_explicit(&counter[0], 1u, memory_order_relaxed);\n"
        "  }\n");
}

/* ====================================================================
 * Sub-phase 2a.4 (2026-05-21) -- e347 (MD5MD5SALT) tp0-pattern Metal
 * emitter. Structural mirror of hx_emit_e347_md5md5md5salt_opencl().
 *
 * Walks the recognized 7-op shape from hx_detect_pattern() and emits a
 * self-contained Metal kernel B that JIT-compiles cleanly on Apple M
 * (M1/M2/M3 verified at the dispatch-smoke level on dev3.local).
 *
 * The emitted source is intended to be JIT-compiled via
 * gpu_metal_jit_compile_source_with_common(), which prepends
 * metal_common_str at newLibraryWithSource time. Helpers consumed from
 * metal_common.metal:
 *
 *   md5_block  (thread uint& h0..h3, thread const uint *M)
 *   MetalParams / OCLParams typedef bridge
 *   HIT_STRIDE
 *   EMIT_HIT_4_DEDUP_OR_OVERFLOW
 *   probe_compact_idx
 *
 * The emitter defines five Metal-side helpers inline (2a.6 names):
 *
 *   md5_buf_global_metal       -- MD5 of variable-length `device const
 *                                 uchar *` candidate. Twin of
 *                                 md5_buf_global.
 *   md5_buf_private32_metal    -- MD5 of a fixed 32-byte private buffer
 *                                 (the hex32-encoded intermediate between
 *                                 the two inner MD5 calls). Replaces 2a.4's
 *                                 md5_buf_private16_metal per 2a.6 chain
 *                                 fix.
 *   state_to_hex32_bytes_metal -- Stash 4-uint state -> 32 lowercase hex
 *                                 chars. Replaces 2a.4's state_to_le_bytes16
 *                                 _metal per 2a.6 chain fix.
 *   hex32_into_M_metal         -- Pack 32 hex chars of 4-uint state into
 *                                 M[0..7] of a private uint[16] block.
 *   md5_outer_hex_combine_metal-- Outer MD5 over hex32(inner) || salt
 *                                 (`device const uchar *salt`). 2a.6
 *                                 padding fix: first_has_pad flag for
 *                                 slen in [24..31].
 *
 * Algorithm semantics (matches OpenCL twin and CPU oracle hashpipe):
 *   digest = MD5( hex32( MD5( MD5(pass) ) ) || salt )
 *
 * SALT_BATCH=64 outer loop is per-thread serial (tp0 pattern). Each
 * thread processes 64 (candidate, salt) tuples; the inner MD5(MD5(pass))
 * pre-state is computed ONCE per thread and held in 4 registers across
 * the entire SALT_BATCH=64 loop. Apple Metal compiler inlines + benefits
 * (opposite of Pascal NVIDIA noinline discipline) -- helpers are NOT
 * decorated with noinline. Per feedback_md5_block_noinline_pascal.md.
 *
 * Address-space discipline:
 *   - All read-only data buffers (payload, b_packed_buf, salts, etc.):
 *     `device const`.
 *   - Atomic-write counters (hit_count, hashes_shown, ovr_*):
 *     `device atomic_uint *` per metal_common.metal Pattern 1.
 *   - Lane-local private memory (M[16], hexbuf, inner_hex):
 *     no qualifier (Metal default thread address space).
 *
 * Buffer indices follow the OpenCL twin's argument order via
 * sequential [[buffer(N)]] attributes for symmetry; the Phase-4
 * dispatcher will bind them at the same indices for swap-in compat.
 *
 * ==================================================================== */

/* Append the e347 Metal kernel preamble + the inline helper definitions
 * (md5_buf_global_metal, md5_buf_private32_metal, state_to_hex32_bytes_metal,
 * hex32_into_M_metal, md5_outer_hex_combine_metal). 2a.6 renamed two of the
 * five helpers per byte-exact chain fix; see file header $Log: hx_emit_metal.c,v $
 * five helpers per byte-exact chain fix; see file header Revision 1.6  2026/05/23 05:23:35  dlr
 * five helpers per byte-exact chain fix; see file header sub-phase 5a.4 Metal twin fan out 6 remaining MAKE_MD5PASS family primitives md4 rmd160 sha224 sha256 sha384 sha512 outer body emit helpers each structural mirror of the OpenCL twin in hx_emit_opencl.c 1.7 Metal-specific idioms device const uchar pass thread uint state pointers state_to_hex32_bytes_metal helper md5_buf_global_metal helper md4_block from metal_common.metal 1.24 sha256_block sha512_block rmd160_block from existing metal_common.metal byte-swap idiom pure shifts and masks no scalar bitselect per feedback_metal_xcode26_bitselect_scalar emit_family_md5pass_kernel_metal switch on outer_id selects helper name per primitive declaration line plus call line built inline 7-arm dispatch hx_emit_family_md5pass_metal widened the per-primitive switch dispatch HX_PRIM_MD5 e123 outlier remains deferred cross-arch validated PASS 8 of 8 on Apple M2 Max dev3 for e122 e159 e163 e165 e167 e169 plus e161 5a.3 regression e347 production regression PASS
 * five helpers per byte-exact chain fix; see file header
 * five helpers per byte-exact chain fix; see file header Revision 1.5  2026/05/23 03:21:35  dlr
 * five helpers per byte-exact chain fix; see file header sub-phase 5a.3 Metal twin of OpenCL family emitter add hx_emit_family_md5pass_metal entry point mirrors 5a.2 OpenCL shape validates code1 callname md5 resolves outer primitive via hx_callname_for_entry FATAL on unknown 5b-deferred or non-SHA1 emits shared helpers md5_buf_global_metal and state_to_hex32_bytes_metal reused verbatim from e347 Metal twin emits per-primitive outer body emit_outer_sha1_concat_then_hash_metal includes BE-to-LE state byte-swap on both single-block fast path and multi-block tail per feedback memo so h0 to h4 LE uint frame matches harness compact_fp probe and EMIT_HIT_4 contract; kernel signature mirrors 2a.6 e347 Metal twin 18 args at same buffer indices salt-table args 3 4 5 IGNORED by family body ovr_set and ovr_gid explicit args 16 17 atomic_uint Metal cannot cast non-atomic pointer to atomic uniform; per-thread topology no SALT_BATCH outer loop; no scalar bitselect; only SHA1 reaches emit_family_md5pass_kernel_metal other primitives FATAL with deferred-to-5a4 diagnostic includes hx_patterns.h hx_spec_entry.h hx_emit_primitives.h
 * five helpers per byte-exact chain fix; see file header
 * five helpers per byte-exact chain fix; see file header Revision 1.4  2026/05/22 03:59:30  dlr
 * five helpers per byte-exact chain fix; see file header sub-phase 2a6 Metal twin byte-exact fix ports the two 2a5 OpenCL emitter fixes chain bug renames state_to_le_bytes16_metal to state_to_hex32_bytes_metal and md5_buf_private16_metal to md5_buf_private32_metal so kernel body feeds inner_hex 32 bytes into second inner MD5 padding bug adds first_has_pad flag in md5_outer_hex_combine_metal multi block branch mechanical port of OpenCL 1.3 to 1.4 diff with Metal token translations validated byte exact dev3 across smoke medium large edge_minlen edge_maxlen edge_empty all matched zero diff
 * five helpers per byte-exact chain fix; see file header. */
static int emit_e347_helpers_metal(char **out, size_t *cap, size_t *len)
{
    int rc;

    rc = hx_appendf(out, cap, len,
        "// ====================================================================\n"
        "// hx codegen sub-phase 2a.4 (2026-05-21): e347 MD5MD5SALT tp0 Metal twin\n"
        "// Emitted by hx_emit_e347_md5md5md5salt_metal()\n"
        "// Pattern matched: HX_PATTERN_E347_MD5MD5MD5SALT\n"
        "// Algorithm: MD5( hex32( MD5( MD5(pass) ) ) || salt )\n"
        "// Structural reference: hx_emit_e347_md5md5md5salt_opencl (Pascal twin)\n"
        "// Helpers from metal_common.metal (prepended at JIT time):\n"
        "//   md5_block (thread uint& refs), MetalParams/OCLParams,\n"
        "//   HIT_STRIDE, EMIT_HIT_4_DEDUP_OR_OVERFLOW, probe_compact_idx\n"
        "// ====================================================================\n"
        "\n"
        "#ifndef SALT_BATCH\n"
        "#define SALT_BATCH 64\n"
        "#endif\n"
        "\n"
        "#ifndef HX_E347_MAX_SALT\n"
        "#define HX_E347_MAX_SALT 240\n"
        "#endif\n"
        "\n");
    if (rc < 0) return rc;

    /* md5_buf_global_metal: MD5 over a `device const uchar *` candidate.
     *
     * Body is a structural mirror of md5_buf_global (OpenCL) with the
     * following Metal-specific differences:
     *   - `device const uchar *data` instead of `__global const uchar *`
     *   - md5_block signature takes `thread uint &h0..h3` and
     *     `thread const uint *M` (per metal_common.metal line 246).
     *     We supply local uint vars and pass them by reference; the
     *     state is then copied back to the output pointers.
     *   - No noinline -- Apple Metal inlines + benefits.
     */
    rc = hx_appendf(out, cap, len,
        "// hx: helper md5_buf_global_metal -- MD5 of variable-length device candidate.\n"
        "// Structural twin of md5_buf_global (OpenCL) at hx_emit_opencl.c rev 1.3.\n"
        "static inline void md5_buf_global_metal(device const uchar *data, int len,\n"
        "                                        thread uint *hx, thread uint *hy,\n"
        "                                        thread uint *hz, thread uint *hw)\n"
        "{\n"
        "    uint M[16];\n"
        "    int pos = 0;\n"
        "    uint h0 = 0x67452301u;\n"
        "    uint h1 = 0xEFCDAB89u;\n"
        "    uint h2 = 0x98BADCFEu;\n"
        "    uint h3 = 0x10325476u;\n"
        "    while (len - pos >= 64) {\n"
        "        for (int j = 0; j < 16; j++) {\n"
        "            int b = pos + j * 4;\n"
        "            M[j] = (uint)data[b]\n"
        "                 | ((uint)data[b + 1] << 8)\n"
        "                 | ((uint)data[b + 2] << 16)\n"
        "                 | ((uint)data[b + 3] << 24);\n"
        "        }\n"
        "        md5_block(h0, h1, h2, h3, M);\n"
        "        pos += 64;\n"
        "    }\n"
        "    int rem = len - pos;\n"
        "    for (int j = 0; j < 16; j++) M[j] = 0;\n"
        "    for (int i = 0; i < rem; i++) {\n"
        "        uint v = (uint)data[pos + i];\n"
        "        M[i >> 2] |= v << ((i & 3) * 8);\n"
        "    }\n"
        "    M[rem >> 2] |= (uint)0x80 << ((rem & 3) * 8);\n"
        "    if (rem < 56) {\n"
        "        M[14] = (uint)(len * 8);\n"
        "        M[15] = 0;\n"
        "        md5_block(h0, h1, h2, h3, M);\n"
        "    } else {\n"
        "        md5_block(h0, h1, h2, h3, M);\n"
        "        for (int j = 0; j < 16; j++) M[j] = 0;\n"
        "        M[14] = (uint)(len * 8);\n"
        "        M[15] = 0;\n"
        "        md5_block(h0, h1, h2, h3, M);\n"
        "    }\n"
        "    *hx = h0; *hy = h1; *hz = h2; *hw = h3;\n"
        "}\n"
        "\n");
    if (rc < 0) return rc;

    /* md5_buf_private32_metal: MD5 of a fixed 32-byte private buffer (the
     * HEX32-encoded intermediate between inner CALL #1 and inner CALL #2).
     * Pure private memory -- the in-arg `data` lives in thread address
     * space (Metal default for stack arrays); no qualifier on the
     * pointer needed.  Single block (32 < 56).
     *
     * Sub-phase 2a.6 (2026-05-22) byte-exact correctness fix: ported from
     * OpenCL emitter rev 1.4. The 2a.4 Metal twin shipped md5_buf_private16_-
     * metal which MD5d the 16 binary bytes of the inner state and produced
     * the WRONG chain (same drift as the broken hand-written kernel B).
     * The correct chain hex32-expands the first inner state into a 32-byte
     * private buffer, then MD5s those 32 hex chars to produce the second
     * inner state. See feedback_handwritten_kernel_b_drift_md5md5salt.md +
     * hx_emit_opencl.c rev 1.4 for full context. */
    rc = hx_appendf(out, cap, len,
        "// hx: helper md5_buf_private32_metal -- MD5 of a 32-byte private buffer.\n"
        "// Used between the two inner CALL md5 ops to compute\n"
        "// MD5(hex32(MD5(pass))).\n"
        "static inline void md5_buf_private32_metal(thread const uchar *data,\n"
        "                                           thread uint *hx, thread uint *hy,\n"
        "                                           thread uint *hz, thread uint *hw)\n"
        "{\n"
        "    uint M[16];\n"
        "    uint h0 = 0x67452301u;\n"
        "    uint h1 = 0xEFCDAB89u;\n"
        "    uint h2 = 0x98BADCFEu;\n"
        "    uint h3 = 0x10325476u;\n"
        "    for (int j = 0; j < 16; j++) M[j] = 0;\n"
        "    for (int i = 0; i < 32; i++) {\n"
        "        uint v = (uint)data[i];\n"
        "        M[i >> 2] |= v << ((i & 3) * 8);\n"
        "    }\n"
        "    M[32 >> 2] |= (uint)0x80 << ((32 & 3) * 8);\n"
        "    M[14] = (uint)(32 * 8);\n"
        "    M[15] = 0;\n"
        "    md5_block(h0, h1, h2, h3, M);\n"
        "    *hx = h0; *hy = h1; *hz = h2; *hw = h3;\n"
        "}\n"
        "\n");
    if (rc < 0) return rc;

    /* state_to_hex32_bytes_metal: stash 4-uint state into 32 lowercase
     * hex characters in a private byte buffer. Used between the two
     * inner MD5s to produce the hex-encoded input to the second inner
     * MD5 (matches mdxfind CPU JOB_MD5MD5SALT semantics).
     *
     * Sub-phase 2a.6 (2026-05-22): replaces state_to_le_bytes16_metal;
     * the 16-byte LE-binary form fed the wrong chain (one hex32 expansion
     * instead of two). Mirror of hx_emit_opencl.c rev 1.4 state_to_hex32_
     * bytes helper; pointer is `thread uchar *`. */
    rc = hx_appendf(out, cap, len,
        "// hx: helper state_to_hex32_bytes_metal -- write 4-uint state as 32 lowercase hex chars.\n"
        "static inline void state_to_hex32_bytes_metal(uint a, uint b, uint c, uint d,\n"
        "                                              thread uchar *buf)\n"
        "{\n"
        "    uint state[4]; state[0]=a; state[1]=b; state[2]=c; state[3]=d;\n"
        "    for (int s = 0; s < 4; s++) {\n"
        "        uint v = state[s];\n"
        "        for (int by = 0; by < 4; by++) {\n"
        "            uint byteval = (v >> (by * 8)) & 0xffu;\n"
        "            uint hi = byteval >> 4;\n"
        "            uint lo = byteval & 0xfu;\n"
        "            int outpos = s * 8 + by * 2;\n"
        "            buf[outpos]     = (uchar)(hi < 10u ? ('0' + hi) : ('a' + hi - 10u));\n"
        "            buf[outpos + 1] = (uchar)(lo < 10u ? ('0' + lo) : ('a' + lo - 10u));\n"
        "        }\n"
        "    }\n"
        "}\n"
        "\n");
    if (rc < 0) return rc;

    /* hex32_into_M_metal: same body as OpenCL twin. Pure scalar +
     * private-memory writes. Output pointer is `thread uint *`. */
    rc = hx_appendf(out, cap, len,
        "// hx: helper hex32_into_M_metal -- pack 32 hex chars of a 4-uint state\n"
        "// into M[0..7] (8 uints, 32 bytes). Lowercase little-endian.\n"
        "static inline void hex32_into_M_metal(uint a, uint b, uint c, uint d,\n"
        "                                      thread uint *M)\n"
        "{\n"
        "    uint state[4]; state[0]=a; state[1]=b; state[2]=c; state[3]=d;\n"
        "    uchar hexbuf[32];\n"
        "    for (int s = 0; s < 4; s++) {\n"
        "        uint v = state[s];\n"
        "        for (int by = 0; by < 4; by++) {\n"
        "            uint byteval = (v >> (by * 8)) & 0xffu;\n"
        "            uint hi = byteval >> 4;\n"
        "            uint lo = byteval & 0xfu;\n"
        "            uchar hc_hi = (uchar)(hi < 10u ? ('0' + hi) : ('a' + hi - 10u));\n"
        "            uchar hc_lo = (uchar)(lo < 10u ? ('0' + lo) : ('a' + lo - 10u));\n"
        "            int outpos = s * 8 + by * 2;\n"
        "            hexbuf[outpos]     = hc_hi;\n"
        "            hexbuf[outpos + 1] = hc_lo;\n"
        "        }\n"
        "    }\n"
        "    for (int j = 0; j < 8; j++) {\n"
        "        int b0 = j * 4;\n"
        "        M[j] = (uint)hexbuf[b0]\n"
        "             | ((uint)hexbuf[b0 + 1] << 8)\n"
        "             | ((uint)hexbuf[b0 + 2] << 16)\n"
        "             | ((uint)hexbuf[b0 + 3] << 24);\n"
        "    }\n"
        "}\n"
        "\n");
    if (rc < 0) return rc;

    /* md5_outer_hex_combine_metal: outer MD5 over (hex32(inner) || salt).
     * Same body as OpenCL twin; salt pointer is `device const uchar *`
     * instead of `__global const uchar *`. md5_block calls pass scalar
     * refs via local uint vars; final state copied to output pointers.
     *
     * As with the OpenCL twin we use the straightforward full-md5_block
     * variant (not md5_block_from8). The pre-state register-hold across
     * the SALT_BATCH=64 loop is the tp0 register-held invariant; round
     * 1-8 reuse is a later micro-opt. */
    rc = hx_appendf(out, cap, len,
        "// hx: helper md5_outer_hex_combine_metal -- outer MD5 over\n"
        "// (hex32(inner_state) || salt). tp0 pattern; structural mirror\n"
        "// of md5_outer_hex_combine (OpenCL).\n"
        "static inline void md5_outer_hex_combine_metal(uint ihx, uint ihy,\n"
        "                                               uint ihz, uint ihw,\n"
        "                                               device const uchar *salt,\n"
        "                                               int slen,\n"
        "                                               thread uint *ohx,\n"
        "                                               thread uint *ohy,\n"
        "                                               thread uint *ohz,\n"
        "                                               thread uint *ohw)\n"
        "{\n"
        "    if (slen < 0) slen = 0;\n"
        "    if (slen > HX_E347_MAX_SALT) slen = HX_E347_MAX_SALT;\n"
        "    int total_len = 32 + slen;\n"
        "\n"
        "    uint M[16];\n"
        "    uint h0 = 0x67452301u;\n"
        "    uint h1 = 0xEFCDAB89u;\n"
        "    uint h2 = 0x98BADCFEu;\n"
        "    uint h3 = 0x10325476u;\n"
        "\n"
        "    int pos = 0;\n"
        "    // First block: pack hex32(state) into M[0..7] then begin\n"
        "    // salt packing into M[8..15] (up to 32 bytes of salt fit).\n"
        "    hex32_into_M_metal(ihx, ihy, ihz, ihw, M);\n"
        "    int salt_in_first = slen;\n"
        "    if (salt_in_first > 32) salt_in_first = 32;\n"
        "    for (int j = 8; j < 16; j++) M[j] = 0;\n"
        "    for (int i = 0; i < salt_in_first; i++) {\n"
        "        uint v = (uint)salt[i];\n"
        "        int dst = 32 + i;\n"
        "        M[dst >> 2] |= v << ((dst & 3) * 8);\n"
        "    }\n"
        "    if (32 + salt_in_first < 64) {\n"
        "        if (32 + slen + 1 <= 56) {\n"
        "            int padpos = 32 + slen;\n"
        "            M[padpos >> 2] |= (uint)0x80 << ((padpos & 3) * 8);\n"
        "            M[14] = (uint)(total_len * 8);\n"
        "            M[15] = 0;\n"
        "            md5_block(h0, h1, h2, h3, M);\n"
        "            *ohx = h0; *ohy = h1; *ohz = h2; *ohw = h3;\n"
        "            return;\n"
        "        }\n"
        "    }\n"
        "    // Multi-block path. Process first 64 bytes (32 hex + up to\n"
        "    // 32 salt). Sub-phase 2a.6 byte-exact fix (ported from OpenCL\n"
        "    // rev 1.4): when salt fits ENTIRELY within the first block but\n"
        "    // the 0x80 padding+length does NOT (i.e. 24 <= slen <= 31), the\n"
        "    // 0x80 byte MUST be emitted in the first block at position\n"
        "    // 32+slen and the tail block must contain ONLY the length (no\n"
        "    // extra 0x80). Previously omitted; broke all slen in [24..31].\n"
        "    int first_has_pad = 0;\n"
        "    if (salt_in_first < 32) {\n"
        "        int padpos = 32 + salt_in_first;\n"
        "        M[padpos >> 2] |= (uint)0x80 << ((padpos & 3) * 8);\n"
        "        first_has_pad = 1;\n"
        "    }\n"
        "    md5_block(h0, h1, h2, h3, M);\n"
        "    pos = 32;\n"
        "    int sleft = slen - salt_in_first;\n"
        "\n"
        "    while (sleft >= 64) {\n"
        "        for (int j = 0; j < 16; j++) {\n"
        "            int b = pos + j * 4;\n"
        "            M[j] = (uint)salt[b]\n"
        "                 | ((uint)salt[b + 1] << 8)\n"
        "                 | ((uint)salt[b + 2] << 16)\n"
        "                 | ((uint)salt[b + 3] << 24);\n"
        "        }\n"
        "        md5_block(h0, h1, h2, h3, M);\n"
        "        pos += 64;\n"
        "        sleft -= 64;\n"
        "    }\n"
        "    // Tail block. When first_has_pad==1 the 0x80 is already in the\n"
        "    // first block, so the tail block is length-only (no second\n"
        "    // 0x80). sleft is guaranteed 0 in that path.\n"
        "    for (int j = 0; j < 16; j++) M[j] = 0;\n"
        "    for (int i = 0; i < sleft; i++) {\n"
        "        uint v = (uint)salt[pos + i];\n"
        "        M[i >> 2] |= v << ((i & 3) * 8);\n"
        "    }\n"
        "    if (!first_has_pad) {\n"
        "        M[sleft >> 2] |= (uint)0x80 << ((sleft & 3) * 8);\n"
        "    }\n"
        "    if (sleft < 56) {\n"
        "        M[14] = (uint)(total_len * 8);\n"
        "        M[15] = 0;\n"
        "        md5_block(h0, h1, h2, h3, M);\n"
        "    } else {\n"
        "        md5_block(h0, h1, h2, h3, M);\n"
        "        for (int j = 0; j < 16; j++) M[j] = 0;\n"
        "        M[14] = (uint)(total_len * 8);\n"
        "        M[15] = 0;\n"
        "        md5_block(h0, h1, h2, h3, M);\n"
        "    }\n"
        "    *ohx = h0; *ohy = h1; *ohz = h2; *ohw = h3;\n"
        "}\n"
        "\n");
    if (rc < 0) return rc;

    return 0;
}

/* Emit the kernel entry. Per-thread serial SALT_BATCH=64 loop is the tp0
 * shape; the inner MD5(MD5(pass)) pre-state is held in 4 registers across
 * all 64 salt iterations. Kernel signature mirrors the OpenCL twin
 * argument order via sequential [[buffer(N)]] attributes so a future
 * Phase-4 dispatcher binds them at matching indices.
 *
 * Atomic-write counters (hit_count, hashes_shown) are typed `device
 * atomic_uint *` per metal_common.metal Pattern 1. The B3 overflow
 * ledger atomic pointers (ovr_set, ovr_gid) are derived from a base
 * `device atomic_uint *` (provided as a separate kernel arg).
 *
 * Note on the B3 overflow ledger pointers: the OpenCL twin reads them
 * from `payload + 100` and `payload + 104` via volatile-uint-pointer
 * casts. Metal forbids casting `device const uchar *` to
 * `device atomic_uint *` (atomic types cannot be aliased through a
 * non-atomic pointer cast). We therefore add explicit
 * `device atomic_uint *ovr_set` and `device atomic_uint *ovr_gid`
 * kernel arguments. The host (Phase 4 dispatcher) binds them to the
 * same payload offsets the OpenCL twin uses. */
static int emit_e347_kernel_metal(char **out, size_t *cap, size_t *len)
{
    int rc;

    rc = hx_appendf(out, cap, len,
        "// hx: Metal kernel signature for e347 tp0 pattern.\n"
        "// Buffer indices [[buffer(N)]] mirror the OpenCL twin's argument\n"
        "// order so the Phase-4 dispatcher binds symmetrically.\n"
        "kernel void kernelb_hx_e347_phase0(\n"
        "    device const uchar         *payload          [[buffer(0)]],\n"
        "    device const uchar         *b_packed_buf     [[buffer(1)]],\n"
        "    device const uint          *b_chunk_index    [[buffer(2)]],\n"
        "    device const uchar         *salts            [[buffer(3)]],\n"
        "    device const uint          *salt_offsets     [[buffer(4)]],\n"
        "    device const ushort        *salt_lens        [[buffer(5)]],\n"
        "    device const uint          *compact_fp       [[buffer(6)]],\n"
        "    device const uint          *compact_idx      [[buffer(7)]],\n"
        "    device const uchar         *hash_data_buf    [[buffer(8)]],\n"
        "    device const ulong         *hash_data_off    [[buffer(9)]],\n"
        "    device uint                *hits             [[buffer(10)]],\n"
        "    device atomic_uint         *hit_count        [[buffer(11)]],\n"
        "    device const ulong         *overflow_keys    [[buffer(12)]],\n"
        "    device const uchar         *overflow_hashes  [[buffer(13)]],\n"
        "    device const uint          *overflow_offsets [[buffer(14)]],\n"
        "    device atomic_uint         *hashes_shown     [[buffer(15)]],\n"
        "    device atomic_uint         *ovr_set          [[buffer(16)]],\n"
        "    device atomic_uint         *ovr_gid          [[buffer(17)]],\n"
        "    uint                        gid              [[thread_position_in_grid]])\n"
        "{\n"
        "    // hx: state EMIT_KERNEL_PREAMBLE (e347 tp0; Metal twin)\n"
        "    device const OCLParams *params_buf =\n"
        "        (device const OCLParams *)payload;\n"
        "    OCLParams params = *params_buf;\n"
        "\n"
        "    uint num_salts_total = params.num_salts;\n"
        "    uint num_salt_chunks =\n"
        "        (num_salts_total + SALT_BATCH - 1u) / SALT_BATCH;\n"
        "    if (num_salt_chunks == 0u) num_salt_chunks = 1u;\n"
        "\n"
        "    uint word_idx       = gid / num_salt_chunks;\n"
        "    uint salt_chunk_idx = gid - word_idx * num_salt_chunks;\n"
        "    uint salt_base      = salt_chunk_idx * SALT_BATCH;\n"
        "\n"
        "    if (word_idx >= params.num_words) return;\n"
        "\n"
        "    // hx: state EMIT_PRE_SALT_INVARIANT (template_pre_salt equivalent)\n"
        "    // Compute MD5(MD5(pass)) ONCE per thread; hold in 4 registers\n"
        "    // across the SALT_BATCH=64 loop.\n"
        "    uint wpos = b_chunk_index[word_idx];\n"
        "    if (wpos >= params.packed_size) return;\n"
        "    uint plen = (uint)b_packed_buf[wpos];\n"
        "    device const uchar *pass_bytes = b_packed_buf + wpos + 1u;\n"
        "\n"
        "    // OP_CALL md5 #1: inner MD5(pass) -> 4 uints in registers.\n"
        "    uint ia, ib, ic, id;\n"
        "    md5_buf_global_metal(pass_bytes, (int)plen, &ia, &ib, &ic, &id);\n"
        "\n"
        "    // OP_CALL md5 #2: hex32-encode (ia,ib,ic,id) and MD5 the 32\n"
        "    // hex chars. Sub-phase 2a.6 byte-exact correctness fix:\n"
        "    // the original 2a.4 emission MD5d the 16 binary bytes which\n"
        "    // produced the same drift documented in feedback_handwritten_-\n"
        "    // kernel_b_drift_md5md5salt.md. hx CONCAT-of-binary-and-string\n"
        "    // forces hex32 expansion at every digest-to-md5 boundary.\n"
        "    uchar inner_hex[32];\n"
        "    state_to_hex32_bytes_metal(ia, ib, ic, id, inner_hex);\n"
        "    uint mma, mmb, mmc, mmd;\n"
        "    md5_buf_private32_metal(inner_hex, &mma, &mmb, &mmc, &mmd);\n"
        "\n"
        "    // (mma, mmb, mmc, mmd) is the pre-salt invariant; hex32(this)\n"
        "    // forms M[0..7] of the outer block on every SALT_BATCH iter.\n"
        "\n"
        "    // hx: state EMIT_SALT_BATCH_LOOP (tp0 outer; SALT_BATCH=64).\n"
        "    for (uint sbi = 0; sbi < SALT_BATCH; sbi++) {\n"
        "        uint salt_local = salt_base + sbi;\n"
        "        if (salt_local >= num_salts_total) break;\n"
        "        uint salt_idx_global = params.salt_start + salt_local;\n"
        "\n"
        "        uint soff = salt_offsets[salt_idx_global];\n"
        "        int  slen = (int)salt_lens[salt_idx_global];\n"
        "        device const uchar *salt = salts + soff;\n"
        "\n"
        "        // OP_CALL md5 #3 (outer): MD5(hex32(inner) || salt)\n"
        "        uint hx, hy, hz, hw;\n"
        "        md5_outer_hex_combine_metal(mma, mmb, mmc, mmd,\n"
        "                                    salt, slen,\n"
        "                                    &hx, &hy, &hz, &hw);\n"
        "\n"
        "        // hx: state EMIT_PROBE_AND_HIT (compact_fp probe + emit)\n"
        "        uint matched_idx = 0u;\n"
        "        if (probe_compact_idx(hx, hy, hz, hw,\n"
        "                              compact_fp, compact_idx,\n"
        "                              params.compact_mask, params.max_probe,\n"
        "                              params.hash_data_count,\n"
        "                              hash_data_buf, hash_data_off,\n"
        "                              overflow_keys, overflow_hashes,\n"
        "                              overflow_offsets, params.overflow_count,\n"
        "                              &matched_idx))\n"
        "        {\n"
        "            uint widx = params.base_word_idx + word_idx;\n"
        "            uint mask = 1u;\n"
        "            EMIT_HIT_4_DEDUP_OR_OVERFLOW(hits, hit_count, params.max_hits,\n"
        "                       widx, salt_idx_global, 1u, hx, hy, hz, hw,\n"
        "                       hashes_shown, matched_idx, mask,\n"
        "                       ovr_set, ovr_gid, gid);\n"
        "        }\n"
        "    }\n"
        "    // hx: state EMIT_KERNEL_FOOTER\n"
        "}\n");

    return rc;
}

int hx_emit_e347_md5md5md5salt_metal(
    char **out, size_t *out_cap,
    const hx_program *prog,
    const struct hx_specialization *spec)
{
    if (!out || !out_cap || !prog || !spec) return -1;

    size_t cur_len = 0;
    if (*out == NULL) *out_cap = 0;

    int rc;

    /* Banner with structural details for dump-file readability. Mirrors
     * the OpenCL twin's banner format so dumps from both backends are
     * trivially diffable. */
    rc = hx_appendf(out, out_cap, &cur_len,
        "// hx codegen: PATTERN E347_MD5MD5MD5SALT matched (Metal backend)\n"
        "// hx: program ncode=%d nvars=%d max_stack=%d has_emit=%d\n"
        "// hx: specialization iter=%u rules=%u masks=%u bf=%u "
        "salt_minlen=%u salt_maxlen=%u regime=%d width=%u\n"
        "// hx: this kernel will be JIT-compiled with metal_common_str\n"
        "// hx: prepended (gpu_metal_jit_compile_source_with_common)\n"
        "\n",
        prog->ncode, prog->nvars, prog->max_stack, prog->has_emit,
        spec->iter_count_if_fixed,
        (unsigned)spec->has_rules,
        (unsigned)spec->has_masks,
        (unsigned)spec->has_bf,
        spec->salt_minlen,
        spec->salt_maxlen,
        (int)spec->salt_count_regime,
        spec->emit_width);
    if (rc < 0) return rc;

    rc = emit_e347_helpers_metal(out, out_cap, &cur_len);
    if (rc < 0) return rc;

    rc = emit_e347_kernel_metal(out, out_cap, &cur_len);
    if (rc < 0) return rc;

    /* Defensive NUL terminator. */
    if (cur_len + 1 > *out_cap) {
        char *np = (char *)realloc(*out, cur_len + 1);
        if (!np) return -1;
        *out = np;
        *out_cap = cur_len + 1;
    }
    (*out)[cur_len] = '\0';
    return 0;
}

/* ====================================================================
 * Sub-phase 5a.3 (2026-05-22): MAKE_MD5PASS family Metal emitter.
 *
 * Metal twin of hx_emit_family_md5pass_opencl (hx_emit_opencl.c rev 1.6
 * lines 880-1483). Structural mirror; only token translation differs.
 * Token deltas:
 *   __global const uchar *      ->  device const uchar *
 *   __global volatile uint *    ->  device atomic_uint *  (per Metal Pattern 1)
 *   uint *out_state             ->  thread uint *out_state
 *   __kernel void NAME(...)     ->  kernel void NAME(
 *                                       device const uchar *foo [[buffer(0)]],
 *                                       ...
 *                                       uint gid [[thread_position_in_grid]])
 *   __attribute__((reqd_work_group_size(64,1,1)))  -- not used on Metal;
 *                                                     the threadgroup size
 *                                                     is set host-side via
 *                                                     dispatchThreadgroups
 *                                                     + threadsPerThreadgroup.
 *   get_global_id(0)            ->  gid (kernel arg attribute)
 *   B3 overflow ledger ptrs (payload+100/+104 alias) -- Metal cannot cast
 *                                 a non-atomic pointer to atomic_uint, so
 *                                 ovr_set and ovr_gid are explicit kernel
 *                                 args 16/17 (same pattern as the 2a.6
 *                                 e347 Metal twin).
 *
 * Helpers reused from metal_common.metal (prepended at JIT time):
 *   md5_block       (thread uint *state -- pointer-state convention)
 *   sha1_block      (thread uint *state, thread const uint *M)
 *   OCLParams / MetalParams typedef bridge
 *   HIT_STRIDE
 *   EMIT_HIT_4_DEDUP_OR_OVERFLOW
 *   probe_compact_idx
 *
 * Per feedback_be_state_primitives_need_byteswap_in_codegen.md the SHA1
 * outer body MUST byte-swap state words before writing them into the
 * uint h0..h4 outputs that flow into probe_compact_idx + EMIT_HIT_4_*.
 * The probe + hit storage convention is LE-uint reinterpretation of the
 * CPU oracle's byte storage; SHA1 state words are BE so without the swap
 * every dispatch would miss every match.
 *
 * Per feedback_metal_xcode26_bitselect_scalar.md the SHA1 body MUST NOT
 * use scalar bitselect(); the byte-swap idiom uses arithmetic shifts +
 * masks which are scalar-safe on every Metal toolchain.
 *
 * Per feedback_no_nested_block_comments_in_cl.md the emitted source uses
 * // comments only.
 *
 * Future sub-phases:
 *   5a.4 -- md4 / md5 / sha224 / sha256 / sha384 / sha512 / rmd160 per-
 *           primitive emitters on BOTH backends (SHA1 byte-swap pattern
 *           is the template for the SHA-2 family + RMD160).
 *   5b   -- 22 deferred primitives once metal_common.metal gains the
 *           corresponding *_block functions.
 * ==================================================================== */

/* Emit shared family helpers for Metal: md5_buf_global_metal +
 * state_to_hex32_bytes_metal. Mirror of emit_e347_helpers_metal (subset;
 * the family has no second inner MD5 and no salt-concat outer, so
 * md5_buf_private32_metal, hex32_into_M_metal, and md5_outer_hex_combine_
 * metal are NOT emitted). */
static int emit_family_md5pass_helpers_metal(char **out, size_t *cap,
                                             size_t *len)
{
    int rc;

    rc = hx_appendf(out, cap, len,
        "// ====================================================================\n"
        "// hx codegen sub-phase 5a.3 (2026-05-22): MAKE_MD5PASS family (Metal)\n"
        "// Emitted by hx_emit_family_md5pass_metal()\n"
        "// Pattern matched: HX_PATTERN_FAMILY_MD5PASS\n"
        "// Algorithm: outer_hash( hex32( MD5(pass) ) || pass )\n"
        "//   (matches mdxfind CPU MAKE_MD5PASS chain at JOB_*MD5PASS:\n"
        "//    mymd5(pass) -> prmd5(.., linebuf, 32) -> strncpy(&linebuf[32],\n"
        "//    cur, len) -> outer(linebuf, 32+len). Per-primitive cases at\n"
        "//    mdxfind.c:25023 (e123 MD5MD5PASS) and 27272 (e161 SHA1MD5PASS).)\n"
        "// Helpers from metal_common.metal (prepended at JIT time):\n"
        "//   md5_block (thread uint *state), sha1_block (thread uint *state),\n"
        "//   OCLParams/MetalParams, HIT_STRIDE,\n"
        "//   EMIT_HIT_4_DEDUP_OR_OVERFLOW, probe_compact_idx\n"
        "// Structural reference: hx_emit_opencl.c rev 1.6 (Pascal twin).\n"
        "// ====================================================================\n"
        "\n"
        "#ifndef HX_FAMILY_MAX_PASS\n"
        "#define HX_FAMILY_MAX_PASS 240\n"
        "#endif\n"
        "\n");
    if (rc < 0) return rc;

    /* md5_buf_global_metal: MD5 over a `device const uchar *` candidate.
     * Body is the structural mirror of md5_buf_global (OpenCL) with Metal
     * pointer-state md5_block call convention. Reused verbatim from the
     * e347 Metal twin (emit_e347_helpers_metal). */
    rc = hx_appendf(out, cap, len,
        "// hx: helper md5_buf_global_metal -- MD5 of variable-length device candidate.\n"
        "// Structural twin of md5_buf_global (OpenCL) at hx_emit_opencl.c rev 1.6.\n"
        "// Reused verbatim from e347 Metal emitter (shared family helper).\n"
        "static inline void md5_buf_global_metal(device const uchar *data, int len,\n"
        "                                        thread uint *hx, thread uint *hy,\n"
        "                                        thread uint *hz, thread uint *hw)\n"
        "{\n"
        "    uint M[16];\n"
        "    int pos = 0;\n"
        "    uint h0 = 0x67452301u;\n"
        "    uint h1 = 0xEFCDAB89u;\n"
        "    uint h2 = 0x98BADCFEu;\n"
        "    uint h3 = 0x10325476u;\n"
        "    while (len - pos >= 64) {\n"
        "        for (int j = 0; j < 16; j++) {\n"
        "            int b = pos + j * 4;\n"
        "            M[j] = (uint)data[b]\n"
        "                 | ((uint)data[b + 1] << 8)\n"
        "                 | ((uint)data[b + 2] << 16)\n"
        "                 | ((uint)data[b + 3] << 24);\n"
        "        }\n"
        "        md5_block(h0, h1, h2, h3, M);\n"
        "        pos += 64;\n"
        "    }\n"
        "    int rem = len - pos;\n"
        "    for (int j = 0; j < 16; j++) M[j] = 0;\n"
        "    for (int i = 0; i < rem; i++) {\n"
        "        uint v = (uint)data[pos + i];\n"
        "        M[i >> 2] |= v << ((i & 3) * 8);\n"
        "    }\n"
        "    M[rem >> 2] |= (uint)0x80 << ((rem & 3) * 8);\n"
        "    if (rem < 56) {\n"
        "        M[14] = (uint)(len * 8);\n"
        "        M[15] = 0;\n"
        "        md5_block(h0, h1, h2, h3, M);\n"
        "    } else {\n"
        "        md5_block(h0, h1, h2, h3, M);\n"
        "        for (int j = 0; j < 16; j++) M[j] = 0;\n"
        "        M[14] = (uint)(len * 8);\n"
        "        M[15] = 0;\n"
        "        md5_block(h0, h1, h2, h3, M);\n"
        "    }\n"
        "    *hx = h0; *hy = h1; *hz = h2; *hw = h3;\n"
        "}\n"
        "\n");
    if (rc < 0) return rc;

    /* state_to_hex32_bytes_metal: writes 4-uint state as 32 lowercase
     * hex characters into a private thread byte buffer. Pure scalar
     * arithmetic; no scalar bitselect (per
     * feedback_metal_xcode26_bitselect_scalar.md). Reused verbatim from
     * the e347 Metal twin. */
    rc = hx_appendf(out, cap, len,
        "// hx: helper state_to_hex32_bytes_metal -- write 4-uint state as 32 lowercase hex chars.\n"
        "// Reused verbatim from e347 Metal emitter (shared family helper).\n"
        "static inline void state_to_hex32_bytes_metal(uint a, uint b, uint c, uint d,\n"
        "                                              thread uchar *buf)\n"
        "{\n"
        "    uint state[4]; state[0]=a; state[1]=b; state[2]=c; state[3]=d;\n"
        "    for (int s = 0; s < 4; s++) {\n"
        "        uint v = state[s];\n"
        "        for (int by = 0; by < 4; by++) {\n"
        "            uint byteval = (v >> (by * 8)) & 0xffu;\n"
        "            uint hi = byteval >> 4;\n"
        "            uint lo = byteval & 0xfu;\n"
        "            int outpos = s * 8 + by * 2;\n"
        "            buf[outpos]     = (uchar)(hi < 10u ? ('0' + hi) : ('a' + hi - 10u));\n"
        "            buf[outpos + 1] = (uchar)(lo < 10u ? ('0' + lo) : ('a' + lo - 10u));\n"
        "        }\n"
        "    }\n"
        "}\n"
        "\n");
    if (rc < 0) return rc;

    return 0;
}

/* Per-primitive outer body emit (Metal): SHA1.
 *
 * Metal twin of emit_outer_sha1_concat_then_hash (OpenCL). Structural
 * mirror; the only differences are:
 *   - device const uchar *pass instead of __global const uchar *pass
 *   - thread uint *h0..h4 instead of bare uint *
 *   - sha1_block (metal_common.metal line 322) signature
 *       void sha1_block(thread uint *state, thread const uint *M)
 *     matches OpenCL's pointer-state convention; we pass &state[0]
 *     directly to the helper.
 *   - Per feedback_be_state_primitives_need_byteswap_in_codegen.md, BE
 *     state words are byte-swapped to LE uints BOTH in the single-block
 *     fast path AND in the multi-block tail. This places h0..h4 in the
 *     same LE-uint frame the harness compact_fp probe + EMIT_HIT_4_*
 *     macro expect.
 *   - Per feedback_metal_xcode26_bitselect_scalar.md no scalar bitselect()
 *     is introduced; the byte-swap idiom is pure shifts + masks. */
static int emit_outer_sha1_concat_then_hash_metal(char **out,
                                                  size_t *cap,
                                                  size_t *len)
{
    int rc;

    rc = hx_appendf(out, cap, len,
        "// hx: helper outer_sha1_concat_then_hash_metal -- SHA1 over\n"
        "// (hex32(md5(pass)) || pass). Total input length = 32 + plen.\n"
        "// Output: 5 uints (h0..h4). probe_compact_idx uses h0..h3 only\n"
        "// (compact_fp is 64-bit key from first 8 bytes); h4 is unused\n"
        "// for probe but EMIT_HIT_4_DEDUP_OR_OVERFLOW takes h0..h3 and\n"
        "// matches the hand-tuned EMIT_HIT_5 macros' first-4 contract\n"
        "// when the production dispatcher routes round-trip readback.\n"
        "//\n"
        "// Structural twin of emit_outer_sha1_concat_then_hash (OpenCL)\n"
        "// at hx_emit_opencl.c rev 1.6. BE-to-LE state byte-swap on the\n"
        "// final state[] is REQUIRED -- the harness compact_fp probe and\n"
        "// EMIT_HIT_4_DEDUP_OR_OVERFLOW macro expect h0..h4 as LE uints\n"
        "// (CPU oracle stores BE bytes; harness reinterprets as LE uints).\n"
        "// Single-instruction-per-word byte-swap (4 shifts + masks + OR);\n"
        "// no scalar bitselect (Metal forbids ulong/uint scalar overload).\n"
        "static inline void outer_sha1_concat_then_hash_metal(\n"
        "    uint mma, uint mmb, uint mmc, uint mmd,\n"
        "    device const uchar *pass, int plen,\n"
        "    thread uint *h0, thread uint *h1, thread uint *h2,\n"
        "    thread uint *h3, thread uint *h4)\n"
        "{\n"
        "    if (plen < 0) plen = 0;\n"
        "    if (plen > HX_FAMILY_MAX_PASS) plen = HX_FAMILY_MAX_PASS;\n"
        "    int total_len = 32 + plen;\n"
        "\n"
        "    // SHA1 initial state.\n"
        "    uint state[5];\n"
        "    state[0] = 0x67452301u;\n"
        "    state[1] = 0xEFCDAB89u;\n"
        "    state[2] = 0x98BADCFEu;\n"
        "    state[3] = 0x10325476u;\n"
        "    state[4] = 0xC3D2E1F0u;\n"
        "\n"
        "    // Build hex32 prefix into a private byte buffer for cheap\n"
        "    // byte-indexed concatenation with pass.\n"
        "    uchar inner_hex[32];\n"
        "    state_to_hex32_bytes_metal(mma, mmb, mmc, mmd, inner_hex);\n"
        "\n"
        "    // SHA1 schedule words are BIG-ENDIAN: M[w] = (b0<<24)|(b1<<16)|\n"
        "    // (b2<<8)|b3. Build first 64-byte block from inner_hex (32 B)\n"
        "    // then as much of pass as fits.\n"
        "    uint M[16];\n"
        "    int byte_pos = 0;   // bytes consumed from logical stream\n"
        "    int pass_consumed = 0;\n"
        "    int first_has_pad = 0;\n"
        "\n"
        "    // First block, bytes [0..63] = hex32 (32) + first min(plen, 32)\n"
        "    // bytes of pass.\n"
        "    int p_in_first = plen;\n"
        "    if (p_in_first > 32) p_in_first = 32;\n"
        "    {\n"
        "        // M[0..7]: hex32 (32 B).\n"
        "        for (int w = 0; w < 8; w++) {\n"
        "            int b = w * 4;\n"
        "            M[w] = ((uint)inner_hex[b]     << 24)\n"
        "                 | ((uint)inner_hex[b + 1] << 16)\n"
        "                 | ((uint)inner_hex[b + 2] <<  8)\n"
        "                 |  (uint)inner_hex[b + 3];\n"
        "        }\n"
        "        // M[8..15]: zeros then pass bytes.\n"
        "        for (int w = 8; w < 16; w++) M[w] = 0u;\n"
        "        for (int i = 0; i < p_in_first; i++) {\n"
        "            int abs_pos = 32 + i;\n"
        "            uint v = (uint)pass[i];\n"
        "            int wi = abs_pos >> 2;\n"
        "            int sh = (3 - (abs_pos & 3)) * 8;  // BE\n"
        "            M[wi] |= v << sh;\n"
        "        }\n"
        "        pass_consumed = p_in_first;\n"
        "        byte_pos = 32 + p_in_first;\n"
        "    }\n"
        "\n"
        "    // Single-block fast path: if 32+plen+1+8 <= 64, pad+len fit\n"
        "    // here. Otherwise need multi-block tail.\n"
        "    if (total_len + 1 + 8 <= 64) {\n"
        "        // 32+plen <= 55 => plen <= 23. Pad at byte_pos, length\n"
        "        // at M[14]/M[15] BE.\n"
        "        int pad_pos = byte_pos;\n"
        "        int wi = pad_pos >> 2;\n"
        "        int sh = (3 - (pad_pos & 3)) * 8;\n"
        "        M[wi] |= 0x80u << sh;\n"
        "        ulong bitlen = (ulong)total_len * 8u;\n"
        "        M[14] = (uint)(bitlen >> 32);\n"
        "        M[15] = (uint)(bitlen & 0xffffffffu);\n"
        "        sha1_block(state, M);\n"
        "        // BE-to-LE swap (see comments at the tail-block path below).\n"
        "        *h0 = ((state[0] & 0x000000ffu) << 24) |\n"
        "              ((state[0] & 0x0000ff00u) <<  8) |\n"
        "              ((state[0] & 0x00ff0000u) >>  8) |\n"
        "              ((state[0] & 0xff000000u) >> 24);\n"
        "        *h1 = ((state[1] & 0x000000ffu) << 24) |\n"
        "              ((state[1] & 0x0000ff00u) <<  8) |\n"
        "              ((state[1] & 0x00ff0000u) >>  8) |\n"
        "              ((state[1] & 0xff000000u) >> 24);\n"
        "        *h2 = ((state[2] & 0x000000ffu) << 24) |\n"
        "              ((state[2] & 0x0000ff00u) <<  8) |\n"
        "              ((state[2] & 0x00ff0000u) >>  8) |\n"
        "              ((state[2] & 0xff000000u) >> 24);\n"
        "        *h3 = ((state[3] & 0x000000ffu) << 24) |\n"
        "              ((state[3] & 0x0000ff00u) <<  8) |\n"
        "              ((state[3] & 0x00ff0000u) >>  8) |\n"
        "              ((state[3] & 0xff000000u) >> 24);\n"
        "        *h4 = ((state[4] & 0x000000ffu) << 24) |\n"
        "              ((state[4] & 0x0000ff00u) <<  8) |\n"
        "              ((state[4] & 0x00ff0000u) >>  8) |\n"
        "              ((state[4] & 0xff000000u) >> 24);\n"
        "        return;\n"
        "    }\n"
        "\n"
        "    // Multi-block path. If pass fits entirely in first block but\n"
        "    // pad+len does not (24 <= plen <= 32), place the 0x80 in the\n"
        "    // first block now and the tail block carries only length.\n"
        "    // Mirrors the e347 first_has_pad pattern.\n"
        "    if (p_in_first == plen && byte_pos < 64) {\n"
        "        int pad_pos = byte_pos;\n"
        "        int wi = pad_pos >> 2;\n"
        "        int sh = (3 - (pad_pos & 3)) * 8;\n"
        "        M[wi] |= 0x80u << sh;\n"
        "        first_has_pad = 1;\n"
        "    }\n"
        "    sha1_block(state, M);\n"
        "\n"
        "    // Walk remaining pass bytes.\n"
        "    int pleft = plen - pass_consumed;\n"
        "    while (pleft >= 64) {\n"
        "        for (int w = 0; w < 16; w++) {\n"
        "            int b = pass_consumed + w * 4;\n"
        "            M[w] = ((uint)pass[b]     << 24)\n"
        "                 | ((uint)pass[b + 1] << 16)\n"
        "                 | ((uint)pass[b + 2] <<  8)\n"
        "                 |  (uint)pass[b + 3];\n"
        "        }\n"
        "        sha1_block(state, M);\n"
        "        pass_consumed += 64;\n"
        "        pleft -= 64;\n"
        "    }\n"
        "\n"
        "    // Tail block(s).\n"
        "    for (int w = 0; w < 16; w++) M[w] = 0u;\n"
        "    for (int i = 0; i < pleft; i++) {\n"
        "        uint v = (uint)pass[pass_consumed + i];\n"
        "        int wi = i >> 2;\n"
        "        int sh = (3 - (i & 3)) * 8;\n"
        "        M[wi] |= v << sh;\n"
        "    }\n"
        "    if (!first_has_pad) {\n"
        "        int pad_pos = pleft;\n"
        "        int wi = pad_pos >> 2;\n"
        "        int sh = (3 - (pad_pos & 3)) * 8;\n"
        "        M[wi] |= 0x80u << sh;\n"
        "    }\n"
        "    if (pleft + 1 + 8 <= 64 || (first_has_pad && pleft + 8 <= 64)) {\n"
        "        ulong bitlen = (ulong)total_len * 8u;\n"
        "        M[14] = (uint)(bitlen >> 32);\n"
        "        M[15] = (uint)(bitlen & 0xffffffffu);\n"
        "        sha1_block(state, M);\n"
        "    } else {\n"
        "        sha1_block(state, M);\n"
        "        for (int w = 0; w < 16; w++) M[w] = 0u;\n"
        "        ulong bitlen = (ulong)total_len * 8u;\n"
        "        M[14] = (uint)(bitlen >> 32);\n"
        "        M[15] = (uint)(bitlen & 0xffffffffu);\n"
        "        sha1_block(state, M);\n"
        "    }\n"
        "    // SHA1 state words are BIG-ENDIAN 32-bit values; the CPU\n"
        "    // oracle stores the digest as raw bytes and the harness\n"
        "    // probe reinterprets those bytes as LITTLE-ENDIAN uints to\n"
        "    // build the compact_fp key. Byte-swap each state word so the\n"
        "    // uint h0..h4 we hand back match what the CPU stores at\n"
        "    // oracle[0..4]. (MD5 doesn't need this; its schedule is LE so\n"
        "    // md5_block returns LE uints natively.)\n"
        "    *h0 = ((state[0] & 0x000000ffu) << 24) |\n"
        "          ((state[0] & 0x0000ff00u) <<  8) |\n"
        "          ((state[0] & 0x00ff0000u) >>  8) |\n"
        "          ((state[0] & 0xff000000u) >> 24);\n"
        "    *h1 = ((state[1] & 0x000000ffu) << 24) |\n"
        "          ((state[1] & 0x0000ff00u) <<  8) |\n"
        "          ((state[1] & 0x00ff0000u) >>  8) |\n"
        "          ((state[1] & 0xff000000u) >> 24);\n"
        "    *h2 = ((state[2] & 0x000000ffu) << 24) |\n"
        "          ((state[2] & 0x0000ff00u) <<  8) |\n"
        "          ((state[2] & 0x00ff0000u) >>  8) |\n"
        "          ((state[2] & 0xff000000u) >> 24);\n"
        "    *h3 = ((state[3] & 0x000000ffu) << 24) |\n"
        "          ((state[3] & 0x0000ff00u) <<  8) |\n"
        "          ((state[3] & 0x00ff0000u) >>  8) |\n"
        "          ((state[3] & 0xff000000u) >> 24);\n"
        "    *h4 = ((state[4] & 0x000000ffu) << 24) |\n"
        "          ((state[4] & 0x0000ff00u) <<  8) |\n"
        "          ((state[4] & 0x00ff0000u) >>  8) |\n"
        "          ((state[4] & 0xff000000u) >> 24);\n"
        "}\n"
        "\n");
    return rc;
}

/* ====================================================================
 * Per-primitive emit bodies for sub-phase 5a.4 (2026-05-23, Metal twins).
 * Six additional primitives mirroring the OpenCL helpers in
 * hx_emit_opencl.c. Structural mirror of those bodies with Metal
 * pointer-state idioms (`device const uchar *pass`, `thread uint *h*`)
 * and Metal-specific block-fn signatures from metal_common.metal.
 *
 * Per [[feedback-be-state-primitives-need-byteswap-in-codegen]]:
 *   md4, rmd160       -- LE-schedule, NO state byte-swap
 *   sha224, sha256    -- BE-schedule, swap each of first 4 uints
 *   sha384, sha512    -- BE-schedule, ULONG state (8 ulongs), swap
 *                        each of first 2 ulongs THEN split.
 *
 * Per [[feedback-metal-xcode26-bitselect-scalar]]: byte-swap idiom is
 * pure shifts + masks; no scalar bitselect.
 * ==================================================================== */

/* Per-primitive outer body emit (Metal): MD4. */
static int emit_outer_md4_concat_then_hash_metal(char **out,
                                                 size_t *cap, size_t *len)
{
    int rc;

    rc = hx_appendf(out, cap, len,
        "// hx: helper outer_md4_concat_then_hash_metal -- MD4 over\n"
        "// (hex32(md5(pass)) || pass). LE-schedule; NO state byte-swap.\n"
        "// md4_block lifted to metal_common.metal (rev) for 5a.4.\n"
        "static inline void outer_md4_concat_then_hash_metal(\n"
        "    uint mma, uint mmb, uint mmc, uint mmd,\n"
        "    device const uchar *pass, int plen,\n"
        "    thread uint *h0, thread uint *h1,\n"
        "    thread uint *h2, thread uint *h3)\n"
        "{\n"
        "    if (plen < 0) plen = 0;\n"
        "    if (plen > HX_FAMILY_MAX_PASS) plen = HX_FAMILY_MAX_PASS;\n"
        "    int total_len = 32 + plen;\n"
        "\n"
        "    uint state[4];\n"
        "    state[0] = 0x67452301u; state[1] = 0xEFCDAB89u;\n"
        "    state[2] = 0x98BADCFEu; state[3] = 0x10325476u;\n"
        "\n"
        "    uchar inner_hex[32];\n"
        "    state_to_hex32_bytes_metal(mma, mmb, mmc, mmd, inner_hex);\n"
        "\n"
        "    uint M[16];\n"
        "    int byte_pos = 0;\n"
        "    int pass_consumed = 0;\n"
        "    int first_has_pad = 0;\n"
        "\n"
        "    int p_in_first = plen;\n"
        "    if (p_in_first > 32) p_in_first = 32;\n"
        "    {\n"
        "        for (int w = 0; w < 8; w++) {\n"
        "            int bo = w * 4;\n"
        "            M[w] = (uint)inner_hex[bo]\n"
        "                 | ((uint)inner_hex[bo + 1] << 8)\n"
        "                 | ((uint)inner_hex[bo + 2] << 16)\n"
        "                 | ((uint)inner_hex[bo + 3] << 24);\n"
        "        }\n"
        "        for (int w = 8; w < 16; w++) M[w] = 0u;\n"
        "        for (int i = 0; i < p_in_first; i++) {\n"
        "            int abs_pos = 32 + i;\n"
        "            uint v = (uint)pass[i];\n"
        "            int wi = abs_pos >> 2;\n"
        "            int sh = (abs_pos & 3) * 8;\n"
        "            M[wi] |= v << sh;\n"
        "        }\n"
        "        pass_consumed = p_in_first;\n"
        "        byte_pos = 32 + p_in_first;\n"
        "    }\n"
        "\n"
        "    if (total_len + 1 + 8 <= 64) {\n"
        "        int pad_pos = byte_pos;\n"
        "        int wi = pad_pos >> 2;\n"
        "        int sh = (pad_pos & 3) * 8;\n"
        "        M[wi] |= 0x80u << sh;\n"
        "        ulong bitlen = (ulong)total_len * 8u;\n"
        "        M[14] = (uint)(bitlen & 0xffffffffu);\n"
        "        M[15] = (uint)(bitlen >> 32);\n"
        "        md4_block(state, M);\n"
        "        *h0 = state[0]; *h1 = state[1]; *h2 = state[2]; *h3 = state[3];\n"
        "        return;\n"
        "    }\n"
        "\n"
        "    if (p_in_first == plen && byte_pos < 64) {\n"
        "        int pad_pos = byte_pos;\n"
        "        int wi = pad_pos >> 2;\n"
        "        int sh = (pad_pos & 3) * 8;\n"
        "        M[wi] |= 0x80u << sh;\n"
        "        first_has_pad = 1;\n"
        "    }\n"
        "    md4_block(state, M);\n"
        "\n"
        "    int pleft = plen - pass_consumed;\n"
        "    while (pleft >= 64) {\n"
        "        for (int w = 0; w < 16; w++) {\n"
        "            int bo = pass_consumed + w * 4;\n"
        "            M[w] = (uint)pass[bo]\n"
        "                 | ((uint)pass[bo + 1] << 8)\n"
        "                 | ((uint)pass[bo + 2] << 16)\n"
        "                 | ((uint)pass[bo + 3] << 24);\n"
        "        }\n"
        "        md4_block(state, M);\n"
        "        pass_consumed += 64;\n"
        "        pleft -= 64;\n"
        "    }\n"
        "\n"
        "    for (int w = 0; w < 16; w++) M[w] = 0u;\n"
        "    for (int i = 0; i < pleft; i++) {\n"
        "        uint v = (uint)pass[pass_consumed + i];\n"
        "        int wi = i >> 2;\n"
        "        int sh = (i & 3) * 8;\n"
        "        M[wi] |= v << sh;\n"
        "    }\n"
        "    if (!first_has_pad) {\n"
        "        int pad_pos = pleft;\n"
        "        int wi = pad_pos >> 2;\n"
        "        int sh = (pad_pos & 3) * 8;\n"
        "        M[wi] |= 0x80u << sh;\n"
        "    }\n"
        "    if (pleft + 1 + 8 <= 64 || (first_has_pad && pleft + 8 <= 64)) {\n"
        "        ulong bitlen = (ulong)total_len * 8u;\n"
        "        M[14] = (uint)(bitlen & 0xffffffffu);\n"
        "        M[15] = (uint)(bitlen >> 32);\n"
        "        md4_block(state, M);\n"
        "    } else {\n"
        "        md4_block(state, M);\n"
        "        for (int w = 0; w < 16; w++) M[w] = 0u;\n"
        "        ulong bitlen = (ulong)total_len * 8u;\n"
        "        M[14] = (uint)(bitlen & 0xffffffffu);\n"
        "        M[15] = (uint)(bitlen >> 32);\n"
        "        md4_block(state, M);\n"
        "    }\n"
        "    *h0 = state[0]; *h1 = state[1]; *h2 = state[2]; *h3 = state[3];\n"
        "}\n"
        "\n");
    return rc;
}

/* Per-primitive outer body emit (Metal): RIPEMD-160. */
static int emit_outer_rmd160_concat_then_hash_metal(char **out,
                                                    size_t *cap, size_t *len)
{
    int rc;

    rc = hx_appendf(out, cap, len,
        "// hx: helper outer_rmd160_concat_then_hash_metal -- RIPEMD-160\n"
        "// over (hex32(md5(pass)) || pass). LE-schedule; NO state\n"
        "// byte-swap. rmd160_block from metal_common.metal (pointer-state).\n"
        "static inline void outer_rmd160_concat_then_hash_metal(\n"
        "    uint mma, uint mmb, uint mmc, uint mmd,\n"
        "    device const uchar *pass, int plen,\n"
        "    thread uint *h0, thread uint *h1, thread uint *h2,\n"
        "    thread uint *h3, thread uint *h4)\n"
        "{\n"
        "    if (plen < 0) plen = 0;\n"
        "    if (plen > HX_FAMILY_MAX_PASS) plen = HX_FAMILY_MAX_PASS;\n"
        "    int total_len = 32 + plen;\n"
        "\n"
        "    uint state[5];\n"
        "    state[0] = 0x67452301u;\n"
        "    state[1] = 0xEFCDAB89u;\n"
        "    state[2] = 0x98BADCFEu;\n"
        "    state[3] = 0x10325476u;\n"
        "    state[4] = 0xC3D2E1F0u;\n"
        "\n"
        "    uchar inner_hex[32];\n"
        "    state_to_hex32_bytes_metal(mma, mmb, mmc, mmd, inner_hex);\n"
        "\n"
        "    uint M[16];\n"
        "    int byte_pos = 0;\n"
        "    int pass_consumed = 0;\n"
        "    int first_has_pad = 0;\n"
        "\n"
        "    int p_in_first = plen;\n"
        "    if (p_in_first > 32) p_in_first = 32;\n"
        "    {\n"
        "        for (int w = 0; w < 8; w++) {\n"
        "            int bo = w * 4;\n"
        "            M[w] = (uint)inner_hex[bo]\n"
        "                 | ((uint)inner_hex[bo + 1] << 8)\n"
        "                 | ((uint)inner_hex[bo + 2] << 16)\n"
        "                 | ((uint)inner_hex[bo + 3] << 24);\n"
        "        }\n"
        "        for (int w = 8; w < 16; w++) M[w] = 0u;\n"
        "        for (int i = 0; i < p_in_first; i++) {\n"
        "            int abs_pos = 32 + i;\n"
        "            uint v = (uint)pass[i];\n"
        "            int wi = abs_pos >> 2;\n"
        "            int sh = (abs_pos & 3) * 8;\n"
        "            M[wi] |= v << sh;\n"
        "        }\n"
        "        pass_consumed = p_in_first;\n"
        "        byte_pos = 32 + p_in_first;\n"
        "    }\n"
        "\n"
        "    if (total_len + 1 + 8 <= 64) {\n"
        "        int pad_pos = byte_pos;\n"
        "        int wi = pad_pos >> 2;\n"
        "        int sh = (pad_pos & 3) * 8;\n"
        "        M[wi] |= 0x80u << sh;\n"
        "        ulong bitlen = (ulong)total_len * 8u;\n"
        "        M[14] = (uint)(bitlen & 0xffffffffu);\n"
        "        M[15] = (uint)(bitlen >> 32);\n"
        "        rmd160_block(state, M);\n"
        "        *h0 = state[0]; *h1 = state[1]; *h2 = state[2];\n"
        "        *h3 = state[3]; *h4 = state[4];\n"
        "        return;\n"
        "    }\n"
        "\n"
        "    if (p_in_first == plen && byte_pos < 64) {\n"
        "        int pad_pos = byte_pos;\n"
        "        int wi = pad_pos >> 2;\n"
        "        int sh = (pad_pos & 3) * 8;\n"
        "        M[wi] |= 0x80u << sh;\n"
        "        first_has_pad = 1;\n"
        "    }\n"
        "    rmd160_block(state, M);\n"
        "\n"
        "    int pleft = plen - pass_consumed;\n"
        "    while (pleft >= 64) {\n"
        "        for (int w = 0; w < 16; w++) {\n"
        "            int bo = pass_consumed + w * 4;\n"
        "            M[w] = (uint)pass[bo]\n"
        "                 | ((uint)pass[bo + 1] << 8)\n"
        "                 | ((uint)pass[bo + 2] << 16)\n"
        "                 | ((uint)pass[bo + 3] << 24);\n"
        "        }\n"
        "        rmd160_block(state, M);\n"
        "        pass_consumed += 64;\n"
        "        pleft -= 64;\n"
        "    }\n"
        "\n"
        "    for (int w = 0; w < 16; w++) M[w] = 0u;\n"
        "    for (int i = 0; i < pleft; i++) {\n"
        "        uint v = (uint)pass[pass_consumed + i];\n"
        "        int wi = i >> 2;\n"
        "        int sh = (i & 3) * 8;\n"
        "        M[wi] |= v << sh;\n"
        "    }\n"
        "    if (!first_has_pad) {\n"
        "        int pad_pos = pleft;\n"
        "        int wi = pad_pos >> 2;\n"
        "        int sh = (pad_pos & 3) * 8;\n"
        "        M[wi] |= 0x80u << sh;\n"
        "    }\n"
        "    if (pleft + 1 + 8 <= 64 || (first_has_pad && pleft + 8 <= 64)) {\n"
        "        ulong bitlen = (ulong)total_len * 8u;\n"
        "        M[14] = (uint)(bitlen & 0xffffffffu);\n"
        "        M[15] = (uint)(bitlen >> 32);\n"
        "        rmd160_block(state, M);\n"
        "    } else {\n"
        "        rmd160_block(state, M);\n"
        "        for (int w = 0; w < 16; w++) M[w] = 0u;\n"
        "        ulong bitlen = (ulong)total_len * 8u;\n"
        "        M[14] = (uint)(bitlen & 0xffffffffu);\n"
        "        M[15] = (uint)(bitlen >> 32);\n"
        "        rmd160_block(state, M);\n"
        "    }\n"
        "    *h0 = state[0]; *h1 = state[1]; *h2 = state[2];\n"
        "    *h3 = state[3]; *h4 = state[4];\n"
        "}\n"
        "\n");
    return rc;
}

/* Shared SHA-224/SHA-256 emit body (Metal), parametrized by IV. */
static int emit_outer_sha2_32_concat_then_hash_metal(char **out,
                                                     size_t *cap, size_t *len,
                                                     const char *fn_name,
                                                     const char *iv_init_str,
                                                     const char *primitive_label)
{
    int rc;

    rc = hx_appendf(out, cap, len,
        "// hx: helper %s -- %s over\n"
        "// (hex32(md5(pass)) || pass). BE-schedule; state byte-swap REQUIRED.\n"
        "static inline void %s(\n"
        "    uint mma, uint mmb, uint mmc, uint mmd,\n"
        "    device const uchar *pass, int plen,\n"
        "    thread uint *h0, thread uint *h1,\n"
        "    thread uint *h2, thread uint *h3)\n"
        "{\n"
        "    if (plen < 0) plen = 0;\n"
        "    if (plen > HX_FAMILY_MAX_PASS) plen = HX_FAMILY_MAX_PASS;\n"
        "    int total_len = 32 + plen;\n"
        "\n"
        "    uint state[8];\n"
        "%s"
        "\n"
        "    uchar inner_hex[32];\n"
        "    state_to_hex32_bytes_metal(mma, mmb, mmc, mmd, inner_hex);\n"
        "\n"
        "    uint M[16];\n"
        "    int byte_pos = 0;\n"
        "    int pass_consumed = 0;\n"
        "    int first_has_pad = 0;\n"
        "\n"
        "    int p_in_first = plen;\n"
        "    if (p_in_first > 32) p_in_first = 32;\n"
        "    {\n"
        "        for (int w = 0; w < 8; w++) {\n"
        "            int bo = w * 4;\n"
        "            M[w] = ((uint)inner_hex[bo]     << 24)\n"
        "                 | ((uint)inner_hex[bo + 1] << 16)\n"
        "                 | ((uint)inner_hex[bo + 2] <<  8)\n"
        "                 |  (uint)inner_hex[bo + 3];\n"
        "        }\n"
        "        for (int w = 8; w < 16; w++) M[w] = 0u;\n"
        "        for (int i = 0; i < p_in_first; i++) {\n"
        "            int abs_pos = 32 + i;\n"
        "            uint v = (uint)pass[i];\n"
        "            int wi = abs_pos >> 2;\n"
        "            int sh = (3 - (abs_pos & 3)) * 8;\n"
        "            M[wi] |= v << sh;\n"
        "        }\n"
        "        pass_consumed = p_in_first;\n"
        "        byte_pos = 32 + p_in_first;\n"
        "    }\n"
        "\n"
        "    if (total_len + 1 + 8 <= 64) {\n"
        "        int pad_pos = byte_pos;\n"
        "        int wi = pad_pos >> 2;\n"
        "        int sh = (3 - (pad_pos & 3)) * 8;\n"
        "        M[wi] |= 0x80u << sh;\n"
        "        ulong bitlen = (ulong)total_len * 8u;\n"
        "        M[14] = (uint)(bitlen >> 32);\n"
        "        M[15] = (uint)(bitlen & 0xffffffffu);\n"
        "        sha256_block(state, M);\n"
        "        *h0 = ((state[0] & 0x000000ffu) << 24) | ((state[0] & 0x0000ff00u) << 8) | ((state[0] & 0x00ff0000u) >> 8) | ((state[0] & 0xff000000u) >> 24);\n"
        "        *h1 = ((state[1] & 0x000000ffu) << 24) | ((state[1] & 0x0000ff00u) << 8) | ((state[1] & 0x00ff0000u) >> 8) | ((state[1] & 0xff000000u) >> 24);\n"
        "        *h2 = ((state[2] & 0x000000ffu) << 24) | ((state[2] & 0x0000ff00u) << 8) | ((state[2] & 0x00ff0000u) >> 8) | ((state[2] & 0xff000000u) >> 24);\n"
        "        *h3 = ((state[3] & 0x000000ffu) << 24) | ((state[3] & 0x0000ff00u) << 8) | ((state[3] & 0x00ff0000u) >> 8) | ((state[3] & 0xff000000u) >> 24);\n"
        "        return;\n"
        "    }\n"
        "\n"
        "    if (p_in_first == plen && byte_pos < 64) {\n"
        "        int pad_pos = byte_pos;\n"
        "        int wi = pad_pos >> 2;\n"
        "        int sh = (3 - (pad_pos & 3)) * 8;\n"
        "        M[wi] |= 0x80u << sh;\n"
        "        first_has_pad = 1;\n"
        "    }\n"
        "    sha256_block(state, M);\n"
        "\n"
        "    int pleft = plen - pass_consumed;\n"
        "    while (pleft >= 64) {\n"
        "        for (int w = 0; w < 16; w++) {\n"
        "            int bo = pass_consumed + w * 4;\n"
        "            M[w] = ((uint)pass[bo]     << 24)\n"
        "                 | ((uint)pass[bo + 1] << 16)\n"
        "                 | ((uint)pass[bo + 2] <<  8)\n"
        "                 |  (uint)pass[bo + 3];\n"
        "        }\n"
        "        sha256_block(state, M);\n"
        "        pass_consumed += 64;\n"
        "        pleft -= 64;\n"
        "    }\n"
        "\n"
        "    for (int w = 0; w < 16; w++) M[w] = 0u;\n"
        "    for (int i = 0; i < pleft; i++) {\n"
        "        uint v = (uint)pass[pass_consumed + i];\n"
        "        int wi = i >> 2;\n"
        "        int sh = (3 - (i & 3)) * 8;\n"
        "        M[wi] |= v << sh;\n"
        "    }\n"
        "    if (!first_has_pad) {\n"
        "        int pad_pos = pleft;\n"
        "        int wi = pad_pos >> 2;\n"
        "        int sh = (3 - (pad_pos & 3)) * 8;\n"
        "        M[wi] |= 0x80u << sh;\n"
        "    }\n"
        "    if (pleft + 1 + 8 <= 64 || (first_has_pad && pleft + 8 <= 64)) {\n"
        "        ulong bitlen = (ulong)total_len * 8u;\n"
        "        M[14] = (uint)(bitlen >> 32);\n"
        "        M[15] = (uint)(bitlen & 0xffffffffu);\n"
        "        sha256_block(state, M);\n"
        "    } else {\n"
        "        sha256_block(state, M);\n"
        "        for (int w = 0; w < 16; w++) M[w] = 0u;\n"
        "        ulong bitlen = (ulong)total_len * 8u;\n"
        "        M[14] = (uint)(bitlen >> 32);\n"
        "        M[15] = (uint)(bitlen & 0xffffffffu);\n"
        "        sha256_block(state, M);\n"
        "    }\n"
        "    *h0 = ((state[0] & 0x000000ffu) << 24) | ((state[0] & 0x0000ff00u) << 8) | ((state[0] & 0x00ff0000u) >> 8) | ((state[0] & 0xff000000u) >> 24);\n"
        "    *h1 = ((state[1] & 0x000000ffu) << 24) | ((state[1] & 0x0000ff00u) << 8) | ((state[1] & 0x00ff0000u) >> 8) | ((state[1] & 0xff000000u) >> 24);\n"
        "    *h2 = ((state[2] & 0x000000ffu) << 24) | ((state[2] & 0x0000ff00u) << 8) | ((state[2] & 0x00ff0000u) >> 8) | ((state[2] & 0xff000000u) >> 24);\n"
        "    *h3 = ((state[3] & 0x000000ffu) << 24) | ((state[3] & 0x0000ff00u) << 8) | ((state[3] & 0x00ff0000u) >> 8) | ((state[3] & 0xff000000u) >> 24);\n"
        "}\n"
        "\n",
        fn_name, primitive_label, fn_name, iv_init_str);
    return rc;
}

static int emit_outer_sha224_concat_then_hash_metal(char **out,
                                                    size_t *cap, size_t *len)
{
    static const char *iv =
        "    state[0] = 0xc1059ed8u; state[1] = 0x367cd507u;\n"
        "    state[2] = 0x3070dd17u; state[3] = 0xf70e5939u;\n"
        "    state[4] = 0xffc00b31u; state[5] = 0x68581511u;\n"
        "    state[6] = 0x64f98fa7u; state[7] = 0xbefa4fa4u;\n";
    return emit_outer_sha2_32_concat_then_hash_metal(out, cap, len,
        "outer_sha224_concat_then_hash_metal", iv, "SHA-224");
}

static int emit_outer_sha256_concat_then_hash_metal(char **out,
                                                    size_t *cap, size_t *len)
{
    static const char *iv =
        "    state[0] = 0x6a09e667u; state[1] = 0xbb67ae85u;\n"
        "    state[2] = 0x3c6ef372u; state[3] = 0xa54ff53au;\n"
        "    state[4] = 0x510e527fu; state[5] = 0x9b05688cu;\n"
        "    state[6] = 0x1f83d9abu; state[7] = 0x5be0cd19u;\n";
    return emit_outer_sha2_32_concat_then_hash_metal(out, cap, len,
        "outer_sha256_concat_then_hash_metal", iv, "SHA-256");
}

/* Shared SHA-384/SHA-512 emit body (Metal), parametrized by IV.
 * Block size 128 bytes; state is 8 ulongs. */
static int emit_outer_sha2_64_concat_then_hash_metal(char **out,
                                                     size_t *cap, size_t *len,
                                                     const char *fn_name,
                                                     const char *iv_init_str,
                                                     const char *primitive_label)
{
    int rc;

    rc = hx_appendf(out, cap, len,
        "// hx: helper %s -- %s over\n"
        "// (hex32(md5(pass)) || pass). 128-byte block; ulong state;\n"
        "// BE-schedule; swap-as-ulong + split into LE-uint pair.\n"
        "static inline void %s(\n"
        "    uint mma, uint mmb, uint mmc, uint mmd,\n"
        "    device const uchar *pass, int plen,\n"
        "    thread uint *h0, thread uint *h1,\n"
        "    thread uint *h2, thread uint *h3)\n"
        "{\n"
        "    if (plen < 0) plen = 0;\n"
        "    if (plen > HX_FAMILY_MAX_PASS) plen = HX_FAMILY_MAX_PASS;\n"
        "    int total_len = 32 + plen;\n"
        "\n"
        "    ulong state[8];\n"
        "%s"
        "\n"
        "    uchar inner_hex[32];\n"
        "    state_to_hex32_bytes_metal(mma, mmb, mmc, mmd, inner_hex);\n"
        "\n"
        "    ulong M[16];\n"
        "    int byte_pos = 0;\n"
        "    int pass_consumed = 0;\n"
        "    int first_has_pad = 0;\n"
        "\n"
        "    int p_in_first = plen;\n"
        "    if (p_in_first > 96) p_in_first = 96;\n"
        "    {\n"
        "        for (int w = 0; w < 4; w++) {\n"
        "            int bo = w * 8;\n"
        "            M[w] = ((ulong)inner_hex[bo]     << 56)\n"
        "                 | ((ulong)inner_hex[bo + 1] << 48)\n"
        "                 | ((ulong)inner_hex[bo + 2] << 40)\n"
        "                 | ((ulong)inner_hex[bo + 3] << 32)\n"
        "                 | ((ulong)inner_hex[bo + 4] << 24)\n"
        "                 | ((ulong)inner_hex[bo + 5] << 16)\n"
        "                 | ((ulong)inner_hex[bo + 6] <<  8)\n"
        "                 |  (ulong)inner_hex[bo + 7];\n"
        "        }\n"
        "        for (int w = 4; w < 16; w++) M[w] = 0ul;\n"
        "        for (int i = 0; i < p_in_first; i++) {\n"
        "            int abs_pos = 32 + i;\n"
        "            ulong v = (ulong)pass[i];\n"
        "            int wi = abs_pos >> 3;\n"
        "            int sh = (7 - (abs_pos & 7)) * 8;\n"
        "            M[wi] |= v << sh;\n"
        "        }\n"
        "        pass_consumed = p_in_first;\n"
        "        byte_pos = 32 + p_in_first;\n"
        "    }\n"
        "\n"
        "    if (total_len + 1 + 16 <= 128) {\n"
        "        int pad_pos = byte_pos;\n"
        "        int wi = pad_pos >> 3;\n"
        "        int sh = (7 - (pad_pos & 7)) * 8;\n"
        "        M[wi] |= ((ulong)0x80u) << sh;\n"
        "        ulong bitlen = (ulong)total_len * 8ul;\n"
        "        M[14] = 0ul;\n"
        "        M[15] = bitlen;\n"
        "        sha512_block(state, M);\n"
        "        ulong s0 = state[0], s1 = state[1];\n"
        "        ulong sw0 = ((s0 & 0x00000000000000ffUL) << 56) | ((s0 & 0x000000000000ff00UL) << 40) | ((s0 & 0x0000000000ff0000UL) << 24) | ((s0 & 0x00000000ff000000UL) << 8) | ((s0 & 0x000000ff00000000UL) >> 8) | ((s0 & 0x0000ff0000000000UL) >> 24) | ((s0 & 0x00ff000000000000UL) >> 40) | ((s0 & 0xff00000000000000UL) >> 56);\n"
        "        ulong sw1 = ((s1 & 0x00000000000000ffUL) << 56) | ((s1 & 0x000000000000ff00UL) << 40) | ((s1 & 0x0000000000ff0000UL) << 24) | ((s1 & 0x00000000ff000000UL) << 8) | ((s1 & 0x000000ff00000000UL) >> 8) | ((s1 & 0x0000ff0000000000UL) >> 24) | ((s1 & 0x00ff000000000000UL) >> 40) | ((s1 & 0xff00000000000000UL) >> 56);\n"
        "        *h0 = (uint)(sw0 & 0xffffffffUL);\n"
        "        *h1 = (uint)(sw0 >> 32);\n"
        "        *h2 = (uint)(sw1 & 0xffffffffUL);\n"
        "        *h3 = (uint)(sw1 >> 32);\n"
        "        return;\n"
        "    }\n"
        "\n"
        "    if (p_in_first == plen && byte_pos < 128) {\n"
        "        int pad_pos = byte_pos;\n"
        "        int wi = pad_pos >> 3;\n"
        "        int sh = (7 - (pad_pos & 7)) * 8;\n"
        "        M[wi] |= ((ulong)0x80u) << sh;\n"
        "        first_has_pad = 1;\n"
        "    }\n"
        "    sha512_block(state, M);\n"
        "\n"
        "    int pleft = plen - pass_consumed;\n"
        "    while (pleft >= 128) {\n"
        "        for (int w = 0; w < 16; w++) {\n"
        "            int bo = pass_consumed + w * 8;\n"
        "            M[w] = ((ulong)pass[bo]     << 56)\n"
        "                 | ((ulong)pass[bo + 1] << 48)\n"
        "                 | ((ulong)pass[bo + 2] << 40)\n"
        "                 | ((ulong)pass[bo + 3] << 32)\n"
        "                 | ((ulong)pass[bo + 4] << 24)\n"
        "                 | ((ulong)pass[bo + 5] << 16)\n"
        "                 | ((ulong)pass[bo + 6] <<  8)\n"
        "                 |  (ulong)pass[bo + 7];\n"
        "        }\n"
        "        sha512_block(state, M);\n"
        "        pass_consumed += 128;\n"
        "        pleft -= 128;\n"
        "    }\n"
        "\n"
        "    for (int w = 0; w < 16; w++) M[w] = 0ul;\n"
        "    for (int i = 0; i < pleft; i++) {\n"
        "        ulong v = (ulong)pass[pass_consumed + i];\n"
        "        int wi = i >> 3;\n"
        "        int sh = (7 - (i & 7)) * 8;\n"
        "        M[wi] |= v << sh;\n"
        "    }\n"
        "    if (!first_has_pad) {\n"
        "        int pad_pos = pleft;\n"
        "        int wi = pad_pos >> 3;\n"
        "        int sh = (7 - (pad_pos & 7)) * 8;\n"
        "        M[wi] |= ((ulong)0x80u) << sh;\n"
        "    }\n"
        "    if (pleft + 1 + 16 <= 128 || (first_has_pad && pleft + 16 <= 128)) {\n"
        "        ulong bitlen = (ulong)total_len * 8ul;\n"
        "        M[14] = 0ul;\n"
        "        M[15] = bitlen;\n"
        "        sha512_block(state, M);\n"
        "    } else {\n"
        "        sha512_block(state, M);\n"
        "        for (int w = 0; w < 16; w++) M[w] = 0ul;\n"
        "        ulong bitlen = (ulong)total_len * 8ul;\n"
        "        M[14] = 0ul;\n"
        "        M[15] = bitlen;\n"
        "        sha512_block(state, M);\n"
        "    }\n"
        "    {\n"
        "        ulong s0 = state[0], s1 = state[1];\n"
        "        ulong sw0 = ((s0 & 0x00000000000000ffUL) << 56) | ((s0 & 0x000000000000ff00UL) << 40) | ((s0 & 0x0000000000ff0000UL) << 24) | ((s0 & 0x00000000ff000000UL) << 8) | ((s0 & 0x000000ff00000000UL) >> 8) | ((s0 & 0x0000ff0000000000UL) >> 24) | ((s0 & 0x00ff000000000000UL) >> 40) | ((s0 & 0xff00000000000000UL) >> 56);\n"
        "        ulong sw1 = ((s1 & 0x00000000000000ffUL) << 56) | ((s1 & 0x000000000000ff00UL) << 40) | ((s1 & 0x0000000000ff0000UL) << 24) | ((s1 & 0x00000000ff000000UL) << 8) | ((s1 & 0x000000ff00000000UL) >> 8) | ((s1 & 0x0000ff0000000000UL) >> 24) | ((s1 & 0x00ff000000000000UL) >> 40) | ((s1 & 0xff00000000000000UL) >> 56);\n"
        "        *h0 = (uint)(sw0 & 0xffffffffUL);\n"
        "        *h1 = (uint)(sw0 >> 32);\n"
        "        *h2 = (uint)(sw1 & 0xffffffffUL);\n"
        "        *h3 = (uint)(sw1 >> 32);\n"
        "    }\n"
        "}\n"
        "\n",
        fn_name, primitive_label, fn_name, iv_init_str);
    return rc;
}

static int emit_outer_sha384_concat_then_hash_metal(char **out,
                                                    size_t *cap, size_t *len)
{
    static const char *iv =
        "    state[0] = 0xcbbb9d5dc1059ed8UL; state[1] = 0x629a292a367cd507UL;\n"
        "    state[2] = 0x9159015a3070dd17UL; state[3] = 0x152fecd8f70e5939UL;\n"
        "    state[4] = 0x67332667ffc00b31UL; state[5] = 0x8eb44a8768581511UL;\n"
        "    state[6] = 0xdb0c2e0d64f98fa7UL; state[7] = 0x47b5481dbefa4fa4UL;\n";
    return emit_outer_sha2_64_concat_then_hash_metal(out, cap, len,
        "outer_sha384_concat_then_hash_metal", iv, "SHA-384");
}

static int emit_outer_sha512_concat_then_hash_metal(char **out,
                                                    size_t *cap, size_t *len)
{
    static const char *iv =
        "    state[0] = 0x6a09e667f3bcc908UL; state[1] = 0xbb67ae8584caa73bUL;\n"
        "    state[2] = 0x3c6ef372fe94f82bUL; state[3] = 0xa54ff53a5f1d36f1UL;\n"
        "    state[4] = 0x510e527fade682d1UL; state[5] = 0x9b05688c2b3e6c1fUL;\n"
        "    state[6] = 0x1f83d9abfb41bd6bUL; state[7] = 0x5be0cd19137e2179UL;\n";
    return emit_outer_sha2_64_concat_then_hash_metal(out, cap, len,
        "outer_sha512_concat_then_hash_metal", iv, "SHA-512");
}

/* Emit the family kernel body (Metal). Per-thread (no SALT_BATCH loop).
 *
 * Kernel signature mirrors the 2a.6 e347 Metal twin's 18-arg [[buffer(N)]]
 * layout EXACTLY so the production dispatcher (5a.5) binds the same
 * arguments via setBuffer:offset:atIndex: regardless of family/non-family
 * routing.
 *
 * Sub-phase 5a.4 (2026-05-23): switch on outer_id to select the matching
 * per-primitive Metal helper. */
static int emit_family_md5pass_kernel_metal(char **out, size_t *cap,
                                            size_t *len,
                                            enum hx_primitive_id outer_id,
                                            const char *outer_name,
                                            int outer_digest_bytes,
                                            int job_enum)
{
    int rc;

    /* Sub-phase 5a.4 (2026-05-23): per-primitive dispatch for the Metal
     * twin. 7 of 8 5a-supported primitives wired (md4, sha1, sha224,
     * sha256, sha384, sha512, rmd160). HX_PRIM_MD5 is outlier (multi-emit
     * deferred). Other family members filtered upstream. */
    int helper_has_h4 = 0;
    switch (outer_id) {
        case HX_PRIM_SHA1:   helper_has_h4 = 1; break;
        case HX_PRIM_RMD160: helper_has_h4 = 1; break;
        case HX_PRIM_MD4:
        case HX_PRIM_SHA224:
        case HX_PRIM_SHA256:
        case HX_PRIM_SHA384:
        case HX_PRIM_SHA512:
            helper_has_h4 = 0;
            break;
        default:
            fprintf(stderr,
                "FATAL: %s:%d hx codegen family Metal emit kernel: outer "
                "primitive '%s' (id=%d) is not wired in 5a.4 (job=e%d).\n",
                __FILE__, __LINE__,
                outer_name ? outer_name : "(null)",
                (int)outer_id, job_enum);
            exit(1);
    }

    rc = hx_appendf(out, cap, len,
        "// hx: family kernel for e%d outer=%s (digest=%d bytes); probe\n"
        "// uses first 4 uints (h0..h3) per compact_fp/compact_idx contract.\n"
        "//\n"
        "// Kernel signature mirrors kernelb_hx_e347_phase0 (Metal) at\n"
        "// hx_emit_metal.c rev 1.4 so the production dispatcher binds the\n"
        "// same 18 args. Salt-table args (3,4,5 + payload->num_salts) are\n"
        "// IGNORED by the family body (family is unsalted; binding them\n"
        "// to existing device salt buffers is harmless).\n"
        "//\n"
        "// Atomics: hit_count + hashes_shown + ovr_set + ovr_gid typed\n"
        "// `device atomic_uint *` per metal_common.metal Pattern 1. The\n"
        "// OpenCL twin aliases ovr_set/ovr_gid off payload+100/+104; Metal\n"
        "// cannot cast a non-atomic pointer to atomic_uint so they are\n"
        "// explicit args (same convention as the 2a.6 e347 Metal twin).\n"
        "kernel void kernelb_hx_codegen_phase0(\n"
        "    device const uchar         *payload          [[buffer(0)]],\n"
        "    device const uchar         *b_packed_buf     [[buffer(1)]],\n"
        "    device const uint          *b_chunk_index    [[buffer(2)]],\n"
        "    device const uchar         *salts            [[buffer(3)]],\n"
        "    device const uint          *salt_offsets     [[buffer(4)]],\n"
        "    device const ushort        *salt_lens        [[buffer(5)]],\n"
        "    device const uint          *compact_fp       [[buffer(6)]],\n"
        "    device const uint          *compact_idx      [[buffer(7)]],\n"
        "    device const uchar         *hash_data_buf    [[buffer(8)]],\n"
        "    device const ulong         *hash_data_off    [[buffer(9)]],\n"
        "    device uint                *hits             [[buffer(10)]],\n"
        "    device atomic_uint         *hit_count        [[buffer(11)]],\n"
        "    device const ulong         *overflow_keys    [[buffer(12)]],\n"
        "    device const uchar         *overflow_hashes  [[buffer(13)]],\n"
        "    device const uint          *overflow_offsets [[buffer(14)]],\n"
        "    device atomic_uint         *hashes_shown     [[buffer(15)]],\n"
        "    device atomic_uint         *ovr_set          [[buffer(16)]],\n"
        "    device atomic_uint         *ovr_gid          [[buffer(17)]],\n"
        "    uint                        gid              [[thread_position_in_grid]])\n"
        "{\n"
        "    // hx: state EMIT_KERNEL_PREAMBLE (family MD5PASS; Metal)\n"
        "    device const OCLParams *params_buf =\n"
        "        (device const OCLParams *)payload;\n"
        "    OCLParams params = *params_buf;\n"
        "\n"
        "    // Per-thread topology (no SALT_BATCH loop; family is unsalted).\n"
        "    uint word_idx = gid;\n"
        "    if (word_idx >= params.num_words) return;\n"
        "\n"
        "    // Unused-arg suppression (silence compiler warnings about\n"
        "    // salts/salt_offsets/salt_lens read by no body code).\n"
        "    (void)salts; (void)salt_offsets; (void)salt_lens;\n"
        "\n"
        "    // hx: state EMIT_PRE_INVARIANT (compute MD5(pass) once)\n"
        "    uint wpos = b_chunk_index[word_idx];\n"
        "    if (wpos >= params.packed_size) return;  // defensive\n"
        "    uint plen = (uint)b_packed_buf[wpos];\n"
        "    device const uchar *pass_bytes = b_packed_buf + wpos + 1u;\n"
        "\n"
        "    // OP_CALL md5 #1 (inner): MD5(pass) -> (ia,ib,ic,id)\n"
        "    uint ia, ib, ic, id;\n"
        "    md5_buf_global_metal(pass_bytes, (int)plen, &ia, &ib, &ic, &id);\n"
        "\n"
        "    // OP_CALL outer (e%d %s): outer( hex32(MD5(pass)) || pass )\n"
        "%s"
        "\n"
        "    // hx: state EMIT_PROBE_AND_HIT (compact_fp probe + emit)\n"
        "    uint matched_idx = 0u;\n"
        "    if (probe_compact_idx(h0, h1, h2, h3,\n"
        "                          compact_fp, compact_idx,\n"
        "                          params.compact_mask, params.max_probe,\n"
        "                          params.hash_data_count,\n"
        "                          hash_data_buf, hash_data_off,\n"
        "                          overflow_keys, overflow_hashes,\n"
        "                          overflow_offsets, params.overflow_count,\n"
        "                          &matched_idx))\n"
        "    {\n"
        "        uint widx = params.base_word_idx + word_idx;\n"
        "        uint mask = 1u;  // iter==1; dedup slot 0\n"
        "        // Unsalted family: sidx is always 0 in the emitted hit.\n"
        "        EMIT_HIT_4_DEDUP_OR_OVERFLOW(hits, hit_count, params.max_hits,\n"
        "                   widx, 0u, 1u, h0, h1, h2, h3,\n"
        "                   hashes_shown, matched_idx, mask,\n"
        "                   ovr_set, ovr_gid, gid);\n"
        "    }\n"
        "%s"
        "    // hx: state EMIT_KERNEL_FOOTER\n"
        "}\n",
        job_enum, outer_name, outer_digest_bytes,
        job_enum, outer_name,
        /* Declaration + Metal helper call, per primitive. */
        (outer_id == HX_PRIM_SHA1) ?
            "    uint h0, h1, h2, h3, h4;\n"
            "    outer_sha1_concat_then_hash_metal(ia, ib, ic, id,\n"
            "                                      pass_bytes, (int)plen,\n"
            "                                      &h0, &h1, &h2, &h3, &h4);\n"
        : (outer_id == HX_PRIM_RMD160) ?
            "    uint h0, h1, h2, h3, h4;\n"
            "    outer_rmd160_concat_then_hash_metal(ia, ib, ic, id,\n"
            "                                        pass_bytes, (int)plen,\n"
            "                                        &h0, &h1, &h2, &h3, &h4);\n"
        : (outer_id == HX_PRIM_MD4) ?
            "    uint h0, h1, h2, h3;\n"
            "    outer_md4_concat_then_hash_metal(ia, ib, ic, id,\n"
            "                                     pass_bytes, (int)plen,\n"
            "                                     &h0, &h1, &h2, &h3);\n"
        : (outer_id == HX_PRIM_SHA224) ?
            "    uint h0, h1, h2, h3;\n"
            "    outer_sha224_concat_then_hash_metal(ia, ib, ic, id,\n"
            "                                        pass_bytes, (int)plen,\n"
            "                                        &h0, &h1, &h2, &h3);\n"
        : (outer_id == HX_PRIM_SHA256) ?
            "    uint h0, h1, h2, h3;\n"
            "    outer_sha256_concat_then_hash_metal(ia, ib, ic, id,\n"
            "                                        pass_bytes, (int)plen,\n"
            "                                        &h0, &h1, &h2, &h3);\n"
        : (outer_id == HX_PRIM_SHA384) ?
            "    uint h0, h1, h2, h3;\n"
            "    outer_sha384_concat_then_hash_metal(ia, ib, ic, id,\n"
            "                                        pass_bytes, (int)plen,\n"
            "                                        &h0, &h1, &h2, &h3);\n"
        : /* SHA-512 */
            "    uint h0, h1, h2, h3;\n"
            "    outer_sha512_concat_then_hash_metal(ia, ib, ic, id,\n"
            "                                        pass_bytes, (int)plen,\n"
            "                                        &h0, &h1, &h2, &h3);\n",
        helper_has_h4
            ? "    (void)h4;  // 5th word reserved for round-trip readback.\n"
            : "");

    return rc;
}

int hx_emit_family_md5pass_metal(
    char **out, size_t *out_cap,
    const hx_program *prog,
    const struct hx_specialization *spec,
    const struct hx_spec_entry *entry)
{
    if (!out || !out_cap || !prog || !spec || !entry) {
        fprintf(stderr,
            "FATAL: %s:%d hx_emit_family_md5pass_metal: NULL argument "
            "(out=%p cap=%p prog=%p spec=%p entry=%p)\n",
            __FILE__, __LINE__,
            (void*)out, (void*)out_cap, (void*)prog,
            (void*)spec, (void*)entry);
        return -1;
    }

    /* Validate code[1] callname == "md5" (the inner CALL). Family
     * detector is structural-only; emitter is the name-validator. */
    const char *inner_name = hx_callname_for_entry(entry, 1);
    if (!inner_name || strcmp(inner_name, "md5") != 0) {
        fprintf(stderr,
            "FATAL: %s:%d hx_emit_family_md5pass_metal: e%d %s code[1] "
            "callname is '%s' but FAMILY_MD5PASS requires 'md5'. "
            "Detector / sidecar drift?\n",
            __FILE__, __LINE__, entry->job_enum,
            entry->name ? entry->name : "(noname)",
            inner_name ? inner_name : "(null)");
        return -1;
    }

    /* Resolve outer-CALL primitive (code[4]). */
    const char *outer_name = hx_callname_for_entry(entry, 4);
    if (!outer_name) {
        fprintf(stderr,
            "FATAL: %s:%d hx_emit_family_md5pass_metal: e%d %s code[4] "
            "callname is NULL (sidecar missing).\n",
            __FILE__, __LINE__, entry->job_enum,
            entry->name ? entry->name : "(noname)");
        return -1;
    }
    enum hx_primitive_id outer_id = hx_primitive_id_for_name(outer_name);
    if (outer_id == HX_PRIM_UNKNOWN) {
        fprintf(stderr,
            "FATAL: %s:%d hx_emit_family_md5pass_metal: e%d %s outer "
            "callname '%s' not recognized in hx_emit_primitives.c table. "
            "Either it is a new primitive (add to prim_table) or the "
            "sidecar is corrupt.\n",
            __FILE__, __LINE__, entry->job_enum,
            entry->name ? entry->name : "(noname)", outer_name);
        return -1;
    }
    if (!hx_primitive_is_supported_5a(outer_id)) {
        fprintf(stderr,
            "FATAL: %s:%d hx_emit_family_md5pass_metal: e%d %s outer "
            "primitive '%s' is in the 5b-deferred set (not in "
            "metal_common.metal yet). Phase 5b lifts the missing *_block "
            "function; until then this algorithm routes to CPU only.\n",
            __FILE__, __LINE__, entry->job_enum,
            entry->name ? entry->name : "(noname)", outer_name);
        return -1;
    }
    /* Sub-phase 5a.4 (2026-05-23): 7 of 8 5a-supported primitives wired
     * on the Metal twin. e123 MD5MD5PASS (HX_PRIM_MD5) stays outlier
     * (multi-emit deferred). */
    if (outer_id == HX_PRIM_MD5) {
        fprintf(stderr,
            "FATAL: %s:%d hx_emit_family_md5pass_metal: e%d %s outer "
            "primitive 'md5' (e123 MD5MD5PASS) is an outlier in 5a -- "
            "multi-emit family deferred to a separate sub-phase. CPU "
            "continues to handle e123 in the interim.\n",
            __FILE__, __LINE__, entry->job_enum,
            entry->name ? entry->name : "(noname)");
        return -1;
    }
    if (outer_id != HX_PRIM_SHA1 && outer_id != HX_PRIM_MD4 &&
        outer_id != HX_PRIM_RMD160 && outer_id != HX_PRIM_SHA224 &&
        outer_id != HX_PRIM_SHA256 && outer_id != HX_PRIM_SHA384 &&
        outer_id != HX_PRIM_SHA512)
    {
        fprintf(stderr,
            "FATAL: %s:%d hx_emit_family_md5pass_metal: e%d %s outer "
            "primitive '%s' is in supported_5a but not in the 5a.4 "
            "wired Metal subset (md4 sha1 sha224 sha256 sha384 sha512 "
            "rmd160).\n",
            __FILE__, __LINE__, entry->job_enum,
            entry->name ? entry->name : "(noname)", outer_name);
        return -1;
    }

    int outer_digest_bytes = hx_primitive_digest_bytes(outer_id);

    size_t cur_len = 0;
    if (*out == NULL) *out_cap = 0;

    int rc;

    /* Banner with structural details for dump-file readability. Mirrors
     * the OpenCL twin's banner format so dumps from both backends are
     * trivially diffable. */
    rc = hx_appendf(out, out_cap, &cur_len,
        "// hx codegen: PATTERN FAMILY_MD5PASS matched (Metal backend; e%d %s outer=%s)\n"
        "// hx: program ncode=%d nvars=%d max_stack=%d has_emit=%d\n"
        "// hx: specialization iter=%u rules=%u masks=%u bf=%u "
        "salt_minlen=%u salt_maxlen=%u regime=%d width=%u\n"
        "// hx: this kernel will be JIT-compiled with metal_common_str\n"
        "// hx: prepended (gpu_metal_jit_compile_source_with_common_keep)\n"
        "\n",
        entry->job_enum, entry->name ? entry->name : "(noname)", outer_name,
        prog->ncode, prog->nvars, prog->max_stack, prog->has_emit,
        spec->iter_count_if_fixed,
        (unsigned)spec->has_rules,
        (unsigned)spec->has_masks,
        (unsigned)spec->has_bf,
        spec->salt_minlen,
        spec->salt_maxlen,
        (int)spec->salt_count_regime,
        spec->emit_width);
    if (rc < 0) return rc;

    rc = emit_family_md5pass_helpers_metal(out, out_cap, &cur_len);
    if (rc < 0) return rc;

    switch (outer_id) {
        case HX_PRIM_SHA1:
            rc = emit_outer_sha1_concat_then_hash_metal(out, out_cap, &cur_len); break;
        case HX_PRIM_MD4:
            rc = emit_outer_md4_concat_then_hash_metal(out, out_cap, &cur_len); break;
        case HX_PRIM_RMD160:
            rc = emit_outer_rmd160_concat_then_hash_metal(out, out_cap, &cur_len); break;
        case HX_PRIM_SHA224:
            rc = emit_outer_sha224_concat_then_hash_metal(out, out_cap, &cur_len); break;
        case HX_PRIM_SHA256:
            rc = emit_outer_sha256_concat_then_hash_metal(out, out_cap, &cur_len); break;
        case HX_PRIM_SHA384:
            rc = emit_outer_sha384_concat_then_hash_metal(out, out_cap, &cur_len); break;
        case HX_PRIM_SHA512:
            rc = emit_outer_sha512_concat_then_hash_metal(out, out_cap, &cur_len); break;
        default:
            fprintf(stderr,
                "FATAL: %s:%d hx_emit_family_md5pass_metal: unreachable "
                "(outer_id=%d not in 5a.4 wired set)\n",
                __FILE__, __LINE__, (int)outer_id);
            return -1;
    }
    if (rc < 0) return rc;

    rc = emit_family_md5pass_kernel_metal(out, out_cap, &cur_len,
                                          outer_id, outer_name,
                                          outer_digest_bytes,
                                          entry->job_enum);
    if (rc < 0) return rc;

    /* Defensive NUL terminator. */
    if (cur_len + 1 > *out_cap) {
        char *np = (char *)realloc(*out, cur_len + 1);
        if (!np) return -1;
        *out = np;
        *out_cap = cur_len + 1;
    }
    (*out)[cur_len] = '\0';
    return 0;
}
