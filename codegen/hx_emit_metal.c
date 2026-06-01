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
 * $Revision: 1.20 $
 * $Log: hx_emit_metal.c,v $
 * Revision 1.20  2026/05/31 14:08:22  dlr
 * Codegen kernel B iteration v1 (-i N>1) paired OpenCL + Metal per spec D-defaults all .a. (1) Runtime iter via existing OCLParams.max_iter (offset 60, zero ABI change). (2) Per-primitive iter-feed helpers for MD5+MD4+SHA1+SHA256 in codegen/hx_emit_opencl.c:4191-4480 + Metal twin codegen/hx_emit_metal.c:3829-4140. (3) Iteration loop wraps kernel B body with per-iter mask 1u << (iter & 31u). (4) Hex-encoded digest feedback mirrors legacy md5_rules_phase0 at gpu/gpu_md5_rules.cl:1158-1193 byte-exact. (5) OpenCL host drops iter==1 clause at gpu/gpujob_opencl.c:1172-1195; unhardcodes params.max_iter=1 at gpu/gpu_opencl.c:14071 + :13212. (6) Metal NEW capability: JOB_MD5 admission added to Metal codegen at gpu_metal.m:4814 (was salted-only); new accessor at :4077-4108; route gate at gpu/gpujob_metal.m:1248-1313. Validated 20-cell crack-parity Pascal+Maxwell+M1 byte-identical at iter in {1,2,5,10,100}; R1 hex-feedback verified via C-oracle harness BEFORE crack-parity. Production safety env UNSET unchanged. Gate C 99K rules x -i 10 x rockyou-1m x Pascal: legacy 305.42s vs codegen 425.56s = 1.39x slower (vs 1.46x at -i 1 — gap closes at -i 10). Gate D NEW: Metal -m e1 -i N>1 works correctly via codegen for first time (legacy template_iterate gap remained; codegen sidesteps). v1.1 follow-on: widen route gate for MD4/SHA1RAW/SHA256RAW admission. Spec project_codegen_iteration_v1_spec_2026-05-31.md.
 *
 * Revision 1.19  2026/05/28 14:32:03  dlr
 * Phase 1b Batch 1: add hx_emit_unsalted_single_opencl + hx_emit_unsalted_single_metal one-shot hash of pass emitters for HX_PATTERN_UNSALTED_SINGLE; reuse md5 md4 sha1 sha256 block from gpu_common.cl and metal_common.metal; strictly simpler than family no inner md5 no hex32 no concat; per-primitive usp buf-global helpers reproduce the family MD SHA padding applied to raw pass; SHA1 SHA256 BE to LE state byte-swap for the compact_fp probe; kernel signature mirrors kernelb_hx_codegen_phase0 salt args ignored; reqd work group size 64; C-mirror validated 80 of 80 byte-exact before GPU JIT; FATAL on callname not in wired set md5 md4 sha1 sha256
 *
 * Revision 1.18  2026/05/28 06:31:13  dlr
 * sub-phase 5c.3 Metal twin add emit_outer_md5_concat_then_hash_metal MD5-as-outer multi-emit helper for e123 MD5MD5PASS the FIRST multi-emit member mirror of OpenCL twin hx_emit_opencl.c rev 1.19 byte-for-byte modulo Metal idioms device const uchar pass thread uint h0 to h3 static inline Pattern 3 sep parameter sep 0 canonical hex32 then pass sep 1 colon hex32 then colon-byte then pass shifts pass to logical position 33 total_len 33 plus plen md5_block from metal_common.metal takes thread uint reference args accumulates into a b c d pre-seeded with MD5 IV LE-schedule NO state byte-swap single-block fast path plus multi-block first_has_pad tail R11 MD5 uses XOR add rotate only no scalar bitselect; add emit_family_md5pass_kernel_metal_multiemit computes md5 of pass once then N equals 2 unrolled probe-and-emit blocks one per variant each calling outer helper with its sep then probe_compact_idx then the EXISTING EMIT_HIT_4_DEDUP_OR_OVERFLOW macro unchanged dedup keys on per-variant matched_idx 16-byte fingerprint self-identifies no variant tag; emit_class threaded through emit_family_md5pass_kernel_metal plus hx_emit_family_md5pass_metal single-emit path untouched G2 no-op; replaced HX_PRIM_MD5 FATAL with emit_class gate MD5-outer admitted only when HX_EMIT_MULTI; wired MD5 into per-primitive emit dispatch plus FATAL filter; gpujob_metal.m caller untouched 4-arg oracle signature preserved metal_gpu_hash_words e123 4 words via default arm correct; built on dev1 Apple Silicon Metal validated on dev3 Apple M2 Max G1b dual-hash canary vn_hits 8 of 8 byte-exact e123 5-fixture matrix PASS half-large 1048576 rows G2 29 single-emit members n_variants 1 PASS G3 e347 smoke plus medium PASS family now 30 of 30 BOTH backends first multi-emit shipped
 *
 * Revision 1.17  2026/05/28 04:49:21  dlr
 * 5b.4b.3-metal twin: add bespoke emit_outer_gost_concat_then_hash_metal mirror of OpenCL twin modulo Metal idioms device const uchar pass thread uint h0..h3 static inline gost_block from metal_common.metal rev 1.33 MTL_GOST_SBOX_1..4 TEST set; sum8 carry + dual finalization + LE output; wired 4 Metal sites helper_has_h4 0 call-line tree FATAL filter dispatch switch
 *
 * Revision 1.16  2026/05/28 04:32:05  dlr
 * sub-phase 5b4a3-metal twin add emit_outer_snefru_concat_then_hash_metal mirror of OpenCL twin byte-for-byte modulo Metal idioms device const uchar pass plus thread uint h0 to h3 signature static inline Pattern 3 snefru_block thread uint state thread const uchar block int is256 from metal_common.metal rev 1.32 parameterised over is256 plus digest_bytes same as OpenCL twin block-size asymmetry SNE128 48-byte SNE256 32-byte DBLK plus length-field byte offsets baked per-width Snefru IV all-zero 8 rounds fixed BE schedule plus BE state output bswap32 into LE-uint probe frame CPU recompute fills SNE256 remaining 16 bytes wired 4 Metal sites helper_has_h4 0 set SNE128 SNE256 call-line tree 2 SNE branches FATAL gating filter widened sne128 sne256 emit dispatch switch routes SNE128 is256 0 SNE256 is256 1 to emit_outer_snefru_concat_then_hash_metal no cl2metal py translator Metal helpers hand-written mirrors per 5a.4 convention
 *
 * Revision 1.15  2026/05/28 03:52:49  dlr
 * sub-phase 5b3c3-metal wire 5 HAV*_5 enums into emit_outer_haval_concat_then_hash_metal dispatch added HX_PRIM_HAV128_5 through HAV256_5 to helper-name switch fall-through call-line tree terminal else comment FATAL gating filter widened to 26 wired Metal primitives new dispatch switch group routes HAV*_5 to emit_outer_haval_metal passes 5 mirror of OpenCL twin
 *
 * Revision 1.14  2026/05/28 03:19:46  dlr
 * sub-phase 5b3b3-metal wire 5 HAV*_4 enums into emit_outer_haval_concat_then_hash_metal dispatch added HX_PRIM_HAV128_4 through HAV256_4 to helper-name switch fall-through call-line tree terminal else comment FATAL gating filter widened to 21 wired Metal primitives new dispatch switch group routes HAV*_4 to emit_outer_haval_metal passes 4 mirror of OpenCL twin
 *
 * Revision 1.13  2026/05/28 02:25:01  dlr
 * sub-phase 5b3a3-metal twin add parameterised emit_outer_haval_concat_then_hash_metal helper to hx_emit_metal c mirror of OpenCL twin byte-for-byte modulo Metal idioms device const uchar pass thread uint h0 to h3 static inline Pattern 3 haval3_block thread uint state thread const uint M from metal_common metal rev 1.29 MTL_HAVAL_IV MTL_HAVAL_ROTR32 state_to_hex32_bytes_metal helper parameterised over passes plus digest_bytes same as OpenCL twin 5b3a ships 3-pass 128-byte block 32 LE-packed words PAD-TOGGLE 0x01 NOT 0x80 cited block 118 119 parameter encoding baked per-width digest fold JIT-specialised donor havalFinal 816-911 HAVAL state LE-native h0 to h3 state 0 to 3 direct wired 4 Metal sites helper-name switch 5 HAV arms call-line tree HAVAL branch FATAL filter widened 16 primitives emit dispatch switch 5 HAV arms route to emit_outer_haval_metal passes 3 digest_bytes outer_digest_bytes no cl2metal py translator Metal helpers hand-written mirrors per 5a.4 convention
 *
 * Revision 1.12  2026/05/27 23:09:06  dlr
 * sub-phase 5b2b3-metal twin add emit_outer_tiger_concat_then_hash_metal mirror of OpenCL twin hx_emit_opencl.c rev 1.13 byte-for-byte Metal-specific idioms device const uchar pass plus thread uint pointer signature plus state_to_hex32_bytes_metal helper plus tiger_block from metal_common.metal rev 1.28 same LE message-schedule packing same 8-byte LE length suffix at M7 same 0x01 padding byte legacy Tiger NOT Tiger2 0x80 same single-block fast path for plen le 23 same Tiger IV initialization same LE state output direct extract no byte-swap epilogue added HX_PRIM_TIGER to helper_has_h4 0 set added TIGER branch to call-line tree TIGER to FATAL gating filter widened to md2 md4 rmd128 sha1 sha224 sha256 sha384 sha512 rmd160 wrl tiger 11 of 11 wired Metal subset via 5a.4 plus 5b.1a plus 5b.1b plus 5b.2a plus 5b.2b TIGER case to emit dispatch switch routes outer_id TIGER to new emit_outer_tiger_concat_then_hash_metal Metal twin no cl2metal.py translator involved Metal helpers are hand-written mirrors per existing 5a.4 convention
 *
 * Revision 1.11  2026/05/27 22:26:09  dlr
 * sub-phase 5b2a3-metal twin add emit_outer_wrl_concat_then_hash_metal mirror of OpenCL twin in hx_emit_opencl.c rev 1.12 byte-for-byte Metal-specific idioms device const uchar pass plus thread uint pointer signature plus state_to_hex32_bytes_metal helper plus wrl_block from metal_common.metal rev 1.27 same BE message-schedule packing same BE 256-bit length suffix at M4 to M7 with high 24 bytes zero same first_has_pad logic bespoke per D16.3.a ALWAYS multi-block single-block fast path elided per Tier 2 spec finding 32 plus plen plus 1 plus 32 le 64 never holds added HX_PRIM_WRL to helper_has_h4 0 set added WRL branch to call-line tree WRL to FATAL gating filter widened to md2 md4 rmd128 sha1 sha224 sha256 sha384 sha512 rmd160 wrl 10 of 10 wired Metal subset via 5a.4 plus 5b.1a plus 5b.1b plus 5b.2a WRL case to emit dispatch switch routes outer_id WRL to new emit_outer_wrl_concat_then_hash_metal Metal twin no cl2metal.py translator involved Metal helpers are hand-written mirrors per existing 5a.4 convention
 *
 * Revision 1.10  2026/05/27 18:40:55  dlr
 * sub-phase 5b1b7 Metal twin revert RIPEMD-128 length-field bug-compat workaround in emit_outer_rmd128_concat_then_hash_metal now that the in-tree rmd128.c MDfinish length-encoding bug is fixed at rmd128.c rev 1.1 mirror of OpenCL twin commit at hx_emit_opencl.c rev 1.11. Removes bug_lswlen first_has_pad branch from both single-block tail branch and 2-block else branch. Both branches now use bitlen equals total_len times 8 unconditionally per Bosselaers 1996 reference. CPU and GPU now both standard-conformant. User-confirmed safe no production solved-hash records affected.
 *
 * Revision 1.9  2026/05/27 18:15:03  dlr
 * sub-phase 5b1b6 Metal twin parallel of OpenCL bug-fix add RIPEMD-128 length-field bug-compatibility in emit_outer_rmd128_concat_then_hash_metal mirror of OpenCL twin fix introduce bug_lswlen first_has_pad branch total_len pleft branch use bug_lswlen 8 instead of total_len 8 at final compress bitlen in both single-block tail branch and 2-block else branch see OpenCL twin commit for full rationale CPU oracle in-tree rmd128.c MDfinish has long-standing length-encoding bug GPU mirrors it for cross-arch byte-exact with CPU oracle
 *
 * Revision 1.8  2026/05/27 17:49:01  dlr
 * sub-phase 5b1b3 Metal twin add emit_outer_rmd128_concat_then_hash_metal mirror of OpenCL twin in hx_emit_opencl.c rev 1.9 byte-for-byte Metal-specific idioms device const uchar pass plus thread uint pointer signature plus state_to_hex32_bytes_metal helper plus rmd128_block from metal_common.metal rev 1.26 same LE message-schedule packing same LE 64-bit length suffix same first_has_pad logic for boundary cases same fast path total_len plus 1 plus 8 le 64 single block then multi-block tail RMD-128 right-pipeline F4 F3 F2 F1 ordering is in rmd128_block primitive itself not in this emit helper added HX_PRIM_RMD128 to helper_has_h4 0 set added RMD128 branch to call-line tree RMD128 to FATAL gating filter wired subset md2 md4 rmd128 sha1 sha224 sha256 sha384 sha512 rmd160 RMD128 case to emit dispatch switch routes outer_id RMD128 to new emit_outer_rmd128_concat_then_hash_metal 9 of 9 5a-supported primitives now wired plus MD2 RMD128 11 of 11 supported primitives via 5a.4 plus 5b.1a plus 5b.1b Metal twin no cl2metal.py translator involved Metal helpers are hand-written mirrors per existing 5a.4 convention
 *
 * Revision 1.7  2026/05/27 17:02:49  dlr
 * sub-phase 5b1a3 Metal twin add emit_outer_md2_concat_then_hash_metal mirror of OpenCL twin in hx_emit_opencl.c rev 1.8 byte-for-byte Metal-specific idioms device const uchar pass plus thread uint pointer signature plus state_to_hex32_bytes_metal helper plus md5_buf_global_metal helper plus md2_block from metal_common.metal rev 1.25 same 16-byte block plus PKCS pad plus checksum-as-final-block structure update_checksum 0 on final per RFC errata digest LE pack of state 0 to 15 added HX_PRIM_MD2 to helper_has_h4 0 set added MD2 branch to call-line tree added MD2 to FATAL gating filter in hx_emit_family_md5pass_metal added HX_PRIM_MD2 case to emit dispatch switch routes outer_id MD2 to new emit_outer_md2_concat_then_hash_metal 8 of 8 5a-supported primitives now wired plus MD2 9 of 9 supported primitives via 5a.4 plus 5b.1a Metal twin no cl2metal.py translator involved Metal helpers are hand-written mirrors per existing 5a.4 convention
 *
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
 * five helpers per byte-exact chain fix; see file header Revision 1.20  2026/05/31 14:08:22  dlr
 * five helpers per byte-exact chain fix; see file header Codegen kernel B iteration v1 (-i N>1) paired OpenCL + Metal per spec D-defaults all .a. (1) Runtime iter via existing OCLParams.max_iter (offset 60, zero ABI change). (2) Per-primitive iter-feed helpers for MD5+MD4+SHA1+SHA256 in codegen/hx_emit_opencl.c:4191-4480 + Metal twin codegen/hx_emit_metal.c:3829-4140. (3) Iteration loop wraps kernel B body with per-iter mask 1u << (iter & 31u). (4) Hex-encoded digest feedback mirrors legacy md5_rules_phase0 at gpu/gpu_md5_rules.cl:1158-1193 byte-exact. (5) OpenCL host drops iter==1 clause at gpu/gpujob_opencl.c:1172-1195; unhardcodes params.max_iter=1 at gpu/gpu_opencl.c:14071 + :13212. (6) Metal NEW capability: JOB_MD5 admission added to Metal codegen at gpu_metal.m:4814 (was salted-only); new accessor at :4077-4108; route gate at gpu/gpujob_metal.m:1248-1313. Validated 20-cell crack-parity Pascal+Maxwell+M1 byte-identical at iter in {1,2,5,10,100}; R1 hex-feedback verified via C-oracle harness BEFORE crack-parity. Production safety env UNSET unchanged. Gate C 99K rules x -i 10 x rockyou-1m x Pascal: legacy 305.42s vs codegen 425.56s = 1.39x slower (vs 1.46x at -i 1 — gap closes at -i 10). Gate D NEW: Metal -m e1 -i N>1 works correctly via codegen for first time (legacy template_iterate gap remained; codegen sidesteps). v1.1 follow-on: widen route gate for MD4/SHA1RAW/SHA256RAW admission. Spec project_codegen_iteration_v1_spec_2026-05-31.md.
 * five helpers per byte-exact chain fix; see file header
 * five helpers per byte-exact chain fix; see file header Revision 1.19  2026/05/28 14:32:03  dlr
 * five helpers per byte-exact chain fix; see file header Phase 1b Batch 1: add hx_emit_unsalted_single_opencl + hx_emit_unsalted_single_metal one-shot hash of pass emitters for HX_PATTERN_UNSALTED_SINGLE; reuse md5 md4 sha1 sha256 block from gpu_common.cl and metal_common.metal; strictly simpler than family no inner md5 no hex32 no concat; per-primitive usp buf-global helpers reproduce the family MD SHA padding applied to raw pass; SHA1 SHA256 BE to LE state byte-swap for the compact_fp probe; kernel signature mirrors kernelb_hx_codegen_phase0 salt args ignored; reqd work group size 64; C-mirror validated 80 of 80 byte-exact before GPU JIT; FATAL on callname not in wired set md5 md4 sha1 sha256
 * five helpers per byte-exact chain fix; see file header
 * five helpers per byte-exact chain fix; see file header Revision 1.18  2026/05/28 06:31:13  dlr
 * five helpers per byte-exact chain fix; see file header sub-phase 5c.3 Metal twin add emit_outer_md5_concat_then_hash_metal MD5-as-outer multi-emit helper for e123 MD5MD5PASS the FIRST multi-emit member mirror of OpenCL twin hx_emit_opencl.c rev 1.19 byte-for-byte modulo Metal idioms device const uchar pass thread uint h0 to h3 static inline Pattern 3 sep parameter sep 0 canonical hex32 then pass sep 1 colon hex32 then colon-byte then pass shifts pass to logical position 33 total_len 33 plus plen md5_block from metal_common.metal takes thread uint reference args accumulates into a b c d pre-seeded with MD5 IV LE-schedule NO state byte-swap single-block fast path plus multi-block first_has_pad tail R11 MD5 uses XOR add rotate only no scalar bitselect; add emit_family_md5pass_kernel_metal_multiemit computes md5 of pass once then N equals 2 unrolled probe-and-emit blocks one per variant each calling outer helper with its sep then probe_compact_idx then the EXISTING EMIT_HIT_4_DEDUP_OR_OVERFLOW macro unchanged dedup keys on per-variant matched_idx 16-byte fingerprint self-identifies no variant tag; emit_class threaded through emit_family_md5pass_kernel_metal plus hx_emit_family_md5pass_metal single-emit path untouched G2 no-op; replaced HX_PRIM_MD5 FATAL with emit_class gate MD5-outer admitted only when HX_EMIT_MULTI; wired MD5 into per-primitive emit dispatch plus FATAL filter; gpujob_metal.m caller untouched 4-arg oracle signature preserved metal_gpu_hash_words e123 4 words via default arm correct; built on dev1 Apple Silicon Metal validated on dev3 Apple M2 Max G1b dual-hash canary vn_hits 8 of 8 byte-exact e123 5-fixture matrix PASS half-large 1048576 rows G2 29 single-emit members n_variants 1 PASS G3 e347 smoke plus medium PASS family now 30 of 30 BOTH backends first multi-emit shipped
 * five helpers per byte-exact chain fix; see file header
 * five helpers per byte-exact chain fix; see file header Revision 1.17  2026/05/28 04:49:21  dlr
 * five helpers per byte-exact chain fix; see file header 5b.4b.3-metal twin: add bespoke emit_outer_gost_concat_then_hash_metal mirror of OpenCL twin modulo Metal idioms device const uchar pass thread uint h0..h3 static inline gost_block from metal_common.metal rev 1.33 MTL_GOST_SBOX_1..4 TEST set; sum8 carry + dual finalization + LE output; wired 4 Metal sites helper_has_h4 0 call-line tree FATAL filter dispatch switch
 * five helpers per byte-exact chain fix; see file header
 * five helpers per byte-exact chain fix; see file header Revision 1.16  2026/05/28 04:32:05  dlr
 * five helpers per byte-exact chain fix; see file header sub-phase 5b4a3-metal twin add emit_outer_snefru_concat_then_hash_metal mirror of OpenCL twin byte-for-byte modulo Metal idioms device const uchar pass plus thread uint h0 to h3 signature static inline Pattern 3 snefru_block thread uint state thread const uchar block int is256 from metal_common.metal rev 1.32 parameterised over is256 plus digest_bytes same as OpenCL twin block-size asymmetry SNE128 48-byte SNE256 32-byte DBLK plus length-field byte offsets baked per-width Snefru IV all-zero 8 rounds fixed BE schedule plus BE state output bswap32 into LE-uint probe frame CPU recompute fills SNE256 remaining 16 bytes wired 4 Metal sites helper_has_h4 0 set SNE128 SNE256 call-line tree 2 SNE branches FATAL gating filter widened sne128 sne256 emit dispatch switch routes SNE128 is256 0 SNE256 is256 1 to emit_outer_snefru_concat_then_hash_metal no cl2metal py translator Metal helpers hand-written mirrors per 5a.4 convention
 * five helpers per byte-exact chain fix; see file header
 * five helpers per byte-exact chain fix; see file header Revision 1.15  2026/05/28 03:52:49  dlr
 * five helpers per byte-exact chain fix; see file header sub-phase 5b3c3-metal wire 5 HAV*_5 enums into emit_outer_haval_concat_then_hash_metal dispatch added HX_PRIM_HAV128_5 through HAV256_5 to helper-name switch fall-through call-line tree terminal else comment FATAL gating filter widened to 26 wired Metal primitives new dispatch switch group routes HAV*_5 to emit_outer_haval_metal passes 5 mirror of OpenCL twin
 * five helpers per byte-exact chain fix; see file header
 * five helpers per byte-exact chain fix; see file header Revision 1.14  2026/05/28 03:19:46  dlr
 * five helpers per byte-exact chain fix; see file header sub-phase 5b3b3-metal wire 5 HAV*_4 enums into emit_outer_haval_concat_then_hash_metal dispatch added HX_PRIM_HAV128_4 through HAV256_4 to helper-name switch fall-through call-line tree terminal else comment FATAL gating filter widened to 21 wired Metal primitives new dispatch switch group routes HAV*_4 to emit_outer_haval_metal passes 4 mirror of OpenCL twin
 * five helpers per byte-exact chain fix; see file header
 * five helpers per byte-exact chain fix; see file header Revision 1.13  2026/05/28 02:25:01  dlr
 * five helpers per byte-exact chain fix; see file header sub-phase 5b3a3-metal twin add parameterised emit_outer_haval_concat_then_hash_metal helper to hx_emit_metal c mirror of OpenCL twin byte-for-byte modulo Metal idioms device const uchar pass thread uint h0 to h3 static inline Pattern 3 haval3_block thread uint state thread const uint M from metal_common metal rev 1.29 MTL_HAVAL_IV MTL_HAVAL_ROTR32 state_to_hex32_bytes_metal helper parameterised over passes plus digest_bytes same as OpenCL twin 5b3a ships 3-pass 128-byte block 32 LE-packed words PAD-TOGGLE 0x01 NOT 0x80 cited block 118 119 parameter encoding baked per-width digest fold JIT-specialised donor havalFinal 816-911 HAVAL state LE-native h0 to h3 state 0 to 3 direct wired 4 Metal sites helper-name switch 5 HAV arms call-line tree HAVAL branch FATAL filter widened 16 primitives emit dispatch switch 5 HAV arms route to emit_outer_haval_metal passes 3 digest_bytes outer_digest_bytes no cl2metal py translator Metal helpers hand-written mirrors per 5a.4 convention
 * five helpers per byte-exact chain fix; see file header
 * five helpers per byte-exact chain fix; see file header Revision 1.12  2026/05/27 23:09:06  dlr
 * five helpers per byte-exact chain fix; see file header sub-phase 5b2b3-metal twin add emit_outer_tiger_concat_then_hash_metal mirror of OpenCL twin hx_emit_opencl.c rev 1.13 byte-for-byte Metal-specific idioms device const uchar pass plus thread uint pointer signature plus state_to_hex32_bytes_metal helper plus tiger_block from metal_common.metal rev 1.28 same LE message-schedule packing same 8-byte LE length suffix at M7 same 0x01 padding byte legacy Tiger NOT Tiger2 0x80 same single-block fast path for plen le 23 same Tiger IV initialization same LE state output direct extract no byte-swap epilogue added HX_PRIM_TIGER to helper_has_h4 0 set added TIGER branch to call-line tree TIGER to FATAL gating filter widened to md2 md4 rmd128 sha1 sha224 sha256 sha384 sha512 rmd160 wrl tiger 11 of 11 wired Metal subset via 5a.4 plus 5b.1a plus 5b.1b plus 5b.2a plus 5b.2b TIGER case to emit dispatch switch routes outer_id TIGER to new emit_outer_tiger_concat_then_hash_metal Metal twin no cl2metal.py translator involved Metal helpers are hand-written mirrors per existing 5a.4 convention
 * five helpers per byte-exact chain fix; see file header
 * five helpers per byte-exact chain fix; see file header Revision 1.11  2026/05/27 22:26:09  dlr
 * five helpers per byte-exact chain fix; see file header sub-phase 5b2a3-metal twin add emit_outer_wrl_concat_then_hash_metal mirror of OpenCL twin in hx_emit_opencl.c rev 1.12 byte-for-byte Metal-specific idioms device const uchar pass plus thread uint pointer signature plus state_to_hex32_bytes_metal helper plus wrl_block from metal_common.metal rev 1.27 same BE message-schedule packing same BE 256-bit length suffix at M4 to M7 with high 24 bytes zero same first_has_pad logic bespoke per D16.3.a ALWAYS multi-block single-block fast path elided per Tier 2 spec finding 32 plus plen plus 1 plus 32 le 64 never holds added HX_PRIM_WRL to helper_has_h4 0 set added WRL branch to call-line tree WRL to FATAL gating filter widened to md2 md4 rmd128 sha1 sha224 sha256 sha384 sha512 rmd160 wrl 10 of 10 wired Metal subset via 5a.4 plus 5b.1a plus 5b.1b plus 5b.2a WRL case to emit dispatch switch routes outer_id WRL to new emit_outer_wrl_concat_then_hash_metal Metal twin no cl2metal.py translator involved Metal helpers are hand-written mirrors per existing 5a.4 convention
 * five helpers per byte-exact chain fix; see file header
 * five helpers per byte-exact chain fix; see file header Revision 1.10  2026/05/27 18:40:55  dlr
 * five helpers per byte-exact chain fix; see file header sub-phase 5b1b7 Metal twin revert RIPEMD-128 length-field bug-compat workaround in emit_outer_rmd128_concat_then_hash_metal now that the in-tree rmd128.c MDfinish length-encoding bug is fixed at rmd128.c rev 1.1 mirror of OpenCL twin commit at hx_emit_opencl.c rev 1.11. Removes bug_lswlen first_has_pad branch from both single-block tail branch and 2-block else branch. Both branches now use bitlen equals total_len times 8 unconditionally per Bosselaers 1996 reference. CPU and GPU now both standard-conformant. User-confirmed safe no production solved-hash records affected.
 * five helpers per byte-exact chain fix; see file header
 * five helpers per byte-exact chain fix; see file header Revision 1.9  2026/05/27 18:15:03  dlr
 * five helpers per byte-exact chain fix; see file header sub-phase 5b1b6 Metal twin parallel of OpenCL bug-fix add RIPEMD-128 length-field bug-compatibility in emit_outer_rmd128_concat_then_hash_metal mirror of OpenCL twin fix introduce bug_lswlen first_has_pad branch total_len pleft branch use bug_lswlen 8 instead of total_len 8 at final compress bitlen in both single-block tail branch and 2-block else branch see OpenCL twin commit for full rationale CPU oracle in-tree rmd128.c MDfinish has long-standing length-encoding bug GPU mirrors it for cross-arch byte-exact with CPU oracle
 * five helpers per byte-exact chain fix; see file header
 * five helpers per byte-exact chain fix; see file header Revision 1.8  2026/05/27 17:49:01  dlr
 * five helpers per byte-exact chain fix; see file header sub-phase 5b1b3 Metal twin add emit_outer_rmd128_concat_then_hash_metal mirror of OpenCL twin in hx_emit_opencl.c rev 1.9 byte-for-byte Metal-specific idioms device const uchar pass plus thread uint pointer signature plus state_to_hex32_bytes_metal helper plus rmd128_block from metal_common.metal rev 1.26 same LE message-schedule packing same LE 64-bit length suffix same first_has_pad logic for boundary cases same fast path total_len plus 1 plus 8 le 64 single block then multi-block tail RMD-128 right-pipeline F4 F3 F2 F1 ordering is in rmd128_block primitive itself not in this emit helper added HX_PRIM_RMD128 to helper_has_h4 0 set added RMD128 branch to call-line tree RMD128 to FATAL gating filter wired subset md2 md4 rmd128 sha1 sha224 sha256 sha384 sha512 rmd160 RMD128 case to emit dispatch switch routes outer_id RMD128 to new emit_outer_rmd128_concat_then_hash_metal 9 of 9 5a-supported primitives now wired plus MD2 RMD128 11 of 11 supported primitives via 5a.4 plus 5b.1a plus 5b.1b Metal twin no cl2metal.py translator involved Metal helpers are hand-written mirrors per existing 5a.4 convention
 * five helpers per byte-exact chain fix; see file header
 * five helpers per byte-exact chain fix; see file header Revision 1.7  2026/05/27 17:02:49  dlr
 * five helpers per byte-exact chain fix; see file header sub-phase 5b1a3 Metal twin add emit_outer_md2_concat_then_hash_metal mirror of OpenCL twin in hx_emit_opencl.c rev 1.8 byte-for-byte Metal-specific idioms device const uchar pass plus thread uint pointer signature plus state_to_hex32_bytes_metal helper plus md5_buf_global_metal helper plus md2_block from metal_common.metal rev 1.25 same 16-byte block plus PKCS pad plus checksum-as-final-block structure update_checksum 0 on final per RFC errata digest LE pack of state 0 to 15 added HX_PRIM_MD2 to helper_has_h4 0 set added MD2 branch to call-line tree added MD2 to FATAL gating filter in hx_emit_family_md5pass_metal added HX_PRIM_MD2 case to emit dispatch switch routes outer_id MD2 to new emit_outer_md2_concat_then_hash_metal 8 of 8 5a-supported primitives now wired plus MD2 9 of 9 supported primitives via 5a.4 plus 5b.1a Metal twin no cl2metal.py translator involved Metal helpers are hand-written mirrors per existing 5a.4 convention
 * five helpers per byte-exact chain fix; see file header
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
/* Per-primitive outer body emit (Metal): MD2. Sub-phase 5b.1a (2026-05-27).
 * Bespoke per D15.3.a -- MD2 structurally diverges from MD4/MD5 family
 * (16-byte block, PKCS pad, checksum-block-as-final). Mirrors the OpenCL
 * twin in hx_emit_opencl.c rev 1.8 byte-for-byte; Metal-specific idioms:
 * `device const uchar *pass` (Pattern 1) and `thread uint *h0..h3`
 * pointer signature; md2_block from metal_common.metal rev 1.25 takes
 * `thread uchar *state, thread uchar *checksum, thread const uchar
 * *data, int update_checksum`. */
static int emit_outer_md2_concat_then_hash_metal(char **out,
                                                 size_t *cap, size_t *len)
{
    int rc;

    rc = hx_appendf(out, cap, len,
        "// hx: helper outer_md2_concat_then_hash_metal -- MD2 over\n"
        "// (hex32(md5(pass)) || pass). Output: 4 uints (h0..h3, LE\n"
        "// pack of state[0..15]). md2_block lifted to metal_common.metal\n"
        "// rev 1.25 for sub-phase 5b.1a.\n"
        "static inline void outer_md2_concat_then_hash_metal(\n"
        "    uint mma, uint mmb, uint mmc, uint mmd,\n"
        "    device const uchar *pass, int plen,\n"
        "    thread uint *h0, thread uint *h1,\n"
        "    thread uint *h2, thread uint *h3)\n"
        "{\n"
        "    if (plen < 0) plen = 0;\n"
        "    if (plen > HX_FAMILY_MAX_PASS) plen = HX_FAMILY_MAX_PASS;\n"
        "    int total_len = 32 + plen;\n"
        "    (void)total_len;\n"
        "\n"
        "    uchar state[48];\n"
        "    uchar checksum[16];\n"
        "    for (int i = 0; i < 48; i++) state[i] = (uchar)0;\n"
        "    for (int i = 0; i < 16; i++) checksum[i] = (uchar)0;\n"
        "\n"
        "    uchar inner_hex[32];\n"
        "    state_to_hex32_bytes_metal(mma, mmb, mmc, mmd, inner_hex);\n"
        "\n"
        "    uchar block[16];\n"
        "\n"
        "    // Process the 2 full 16-byte blocks of inner_hex.\n"
        "    for (int b = 0; b < 2; b++) {\n"
        "        for (int j = 0; j < 16; j++) {\n"
        "            block[j] = inner_hex[b * 16 + j];\n"
        "        }\n"
        "        md2_block(state, checksum, block, 1);\n"
        "    }\n"
        "\n"
        "    // Process pass[] in 16-byte chunks while at least 16 bytes left.\n"
        "    int pass_off = 0;\n"
        "    while ((plen - pass_off) >= 16) {\n"
        "        for (int j = 0; j < 16; j++) {\n"
        "            block[j] = pass[pass_off + j];\n"
        "        }\n"
        "        md2_block(state, checksum, block, 1);\n"
        "        pass_off += 16;\n"
        "    }\n"
        "\n"
        "    // PKCS-pad tail to 16-byte boundary (pad_len in [1..16]).\n"
        "    int tail_len = plen - pass_off;\n"
        "    int pad_len = 16 - tail_len;\n"
        "    for (int j = 0; j < tail_len; j++) {\n"
        "        block[j] = pass[pass_off + j];\n"
        "    }\n"
        "    for (int j = tail_len; j < 16; j++) {\n"
        "        block[j] = (uchar)pad_len;\n"
        "    }\n"
        "    md2_block(state, checksum, block, 1);\n"
        "\n"
        "    // Final: checksum block (no checksum update per RFC errata).\n"
        "    md2_block(state, checksum, checksum, 0);\n"
        "\n"
        "    *h0 = (uint)state[ 0]\n"
        "        | ((uint)state[ 1] <<  8)\n"
        "        | ((uint)state[ 2] << 16)\n"
        "        | ((uint)state[ 3] << 24);\n"
        "    *h1 = (uint)state[ 4]\n"
        "        | ((uint)state[ 5] <<  8)\n"
        "        | ((uint)state[ 6] << 16)\n"
        "        | ((uint)state[ 7] << 24);\n"
        "    *h2 = (uint)state[ 8]\n"
        "        | ((uint)state[ 9] <<  8)\n"
        "        | ((uint)state[10] << 16)\n"
        "        | ((uint)state[11] << 24);\n"
        "    *h3 = (uint)state[12]\n"
        "        | ((uint)state[13] <<  8)\n"
        "        | ((uint)state[14] << 16)\n"
        "        | ((uint)state[15] << 24);\n"
        "}\n"
        "\n");
    return rc;
}

/* Per-primitive outer body emit (Metal): MD5-as-OUTER multi-emit helper
 * for e123 MD5MD5PASS -- the FIRST multi-emit member. Sub-phase 5c.3
 * (2026-05-27) Metal twin. Structural mirror of the OpenCL twin
 * emit_outer_md5_concat_then_hash (hx_emit_opencl.c rev 1.19) byte-for-byte
 * modulo Metal idioms: `device const uchar *pass` + `thread uint *` pointer
 * args; md5_block(a,b,c,d,M) from metal_common.metal takes `thread uint &`
 * references and ADDS the working vars back into a..d, so a..d are
 * pre-seeded with the MD5 IV (same accumulate convention as the OpenCL
 * &a,&b,&c,&d pointer form). LE-schedule; NO state byte-swap (MD5 state is
 * LE-native; CPU oracle mymd5() stores LE bytes; harness reinterprets as
 * LE uints -> direct byte-exact match).
 *
 * The `sep` parameter encodes the multi-emit variant (matching the CPU
 * oracle at mdxfind.c:25181-25204 which builds linebuf + linebuf2):
 *   sep == 0 -> canonical: outer message = hex32(md5(pass)) || pass
 *               (total_len = 32 + plen)
 *   sep == 1 -> colon:     outer message = hex32 || ':' || pass
 *               (total_len = 33 + plen; one ':' byte injected at logical
 *                position 32, shifting pass to start at position 33)
 *
 * R11 check: MD5 uses XOR/add/rotate (MTL_MD5_FF/GG/HH/II) only -- no
 * scalar bitselect() is in play (per feedback_metal_xcode26_bitselect_-
 * scalar.md). The family kernel body calls this helper TWICE (sep=0 then
 * sep=1) for emit_class == HX_EMIT_MULTI; each variant probes + emits
 * independently against its own matched loaded-hash slot. */
static int emit_outer_md5_concat_then_hash_metal(char **out,
                                                 size_t *cap, size_t *len)
{
    int rc;

    rc = hx_appendf(out, cap, len,
        "// hx: helper outer_md5_concat_then_hash_metal -- MD5 over the\n"
        "// multi-emit outer message. sep selects the variant:\n"
        "//   sep==0 : hex32(md5(pass)) || pass            (canonical)\n"
        "//   sep==1 : hex32(md5(pass)) || ':' || pass     (colon)\n"
        "// Output: 4 uints (h0..h3). MD5 schedule is LITTLE-ENDIAN;\n"
        "// md5_block accumulates into a..d natively; NO state byte-swap.\n"
        "// The hex32 prefix occupies logical bytes [0..31]; when sep==1\n"
        "// a single ':' byte sits at logical position 32 and pass starts\n"
        "// at position 33. base = 32 + sep is the logical start of pass.\n"
        "static inline void outer_md5_concat_then_hash_metal(\n"
        "    uint mma, uint mmb, uint mmc, uint mmd,\n"
        "    device const uchar *pass, int plen, int sep,\n"
        "    thread uint *h0, thread uint *h1,\n"
        "    thread uint *h2, thread uint *h3)\n"
        "{\n"
        "    if (plen < 0) plen = 0;\n"
        "    if (plen > HX_FAMILY_MAX_PASS) plen = HX_FAMILY_MAX_PASS;\n"
        "    int base = 32 + sep;               // logical start of pass\n"
        "    int total_len = base + plen;\n"
        "\n"
        "    // MD5 initial state. md5_block ADDS working vars back into\n"
        "    // a..d (thread uint& refs), so seed with the IV here.\n"
        "    uint a = 0x67452301u, b = 0xEFCDAB89u, c = 0x98BADCFEu, d = 0x10325476u;\n"
        "\n"
        "    uchar inner_hex[32];\n"
        "    state_to_hex32_bytes_metal(mma, mmb, mmc, mmd, inner_hex);\n"
        "\n"
        "    // MD5 schedule words are LITTLE-ENDIAN (b0|b1<<8|b2<<16|b3<<24).\n"
        "    uint M[16];\n"
        "    int byte_pos = 0;\n"
        "    int pass_consumed = 0;\n"
        "    int first_has_pad = 0;\n"
        "\n"
        "    // First block, logical bytes [0..63]:\n"
        "    //   [0..31]  hex32\n"
        "    //   [32]     ':' if sep==1\n"
        "    //   [base..] as much of pass as fits.\n"
        "    int p_in_first = plen;\n"
        "    if (p_in_first > 64 - base) p_in_first = 64 - base;\n"
        "    if (p_in_first < 0) p_in_first = 0;\n"
        "    {\n"
        "        for (int w = 0; w < 8; w++) {\n"
        "            int bo = w * 4;\n"
        "            M[w] = (uint)inner_hex[bo]\n"
        "                 | ((uint)inner_hex[bo + 1] << 8)\n"
        "                 | ((uint)inner_hex[bo + 2] << 16)\n"
        "                 | ((uint)inner_hex[bo + 3] << 24);\n"
        "        }\n"
        "        for (int w = 8; w < 16; w++) M[w] = 0u;\n"
        "        // Inject the ':' separator at logical position 32 (sep==1).\n"
        "        if (sep) {\n"
        "            int abs_pos = 32;\n"
        "            int wi = abs_pos >> 2;\n"
        "            int sh = (abs_pos & 3) * 8;  // LE\n"
        "            M[wi] |= (uint)':' << sh;\n"
        "        }\n"
        "        for (int i = 0; i < p_in_first; i++) {\n"
        "            int abs_pos = base + i;\n"
        "            uint v = (uint)pass[i];\n"
        "            int wi = abs_pos >> 2;\n"
        "            int sh = (abs_pos & 3) * 8;  // LE\n"
        "            M[wi] |= v << sh;\n"
        "        }\n"
        "        pass_consumed = p_in_first;\n"
        "        byte_pos = base + p_in_first;\n"
        "    }\n"
        "\n"
        "    // Single-block fast path: pad (1 byte) + 8-byte length fit in\n"
        "    // the first block iff total_len + 1 + 8 <= 64 (total_len <= 55).\n"
        "    if (total_len + 1 + 8 <= 64) {\n"
        "        int pad_pos = byte_pos;\n"
        "        int wi = pad_pos >> 2;\n"
        "        int sh = (pad_pos & 3) * 8;\n"
        "        M[wi] |= 0x80u << sh;\n"
        "        ulong bitlen = (ulong)total_len * 8u;\n"
        "        // MD5 length is LITTLE-ENDIAN 64-bit at M[14]/M[15].\n"
        "        M[14] = (uint)(bitlen & 0xffffffffu);\n"
        "        M[15] = (uint)(bitlen >> 32);\n"
        "        md5_block(a, b, c, d, M);\n"
        "        *h0 = a; *h1 = b; *h2 = c; *h3 = d;\n"
        "        return;\n"
        "    }\n"
        "\n"
        "    // Multi-block path. If ALL pass bytes fit in the first block\n"
        "    // (p_in_first == plen) but pad+len do not, place the 0x80 in\n"
        "    // the first block now (first_has_pad) and the tail carries\n"
        "    // only the length. Mirrors the MD4 / e347 first_has_pad fix.\n"
        "    if (p_in_first == plen && byte_pos < 64) {\n"
        "        int pad_pos = byte_pos;\n"
        "        int wi = pad_pos >> 2;\n"
        "        int sh = (pad_pos & 3) * 8;\n"
        "        M[wi] |= 0x80u << sh;\n"
        "        first_has_pad = 1;\n"
        "    }\n"
        "    md5_block(a, b, c, d, M);\n"
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
        "        md5_block(a, b, c, d, M);\n"
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
        "        md5_block(a, b, c, d, M);\n"
        "    } else {\n"
        "        md5_block(a, b, c, d, M);\n"
        "        for (int w = 0; w < 16; w++) M[w] = 0u;\n"
        "        ulong bitlen = (ulong)total_len * 8u;\n"
        "        M[14] = (uint)(bitlen & 0xffffffffu);\n"
        "        M[15] = (uint)(bitlen >> 32);\n"
        "        md5_block(a, b, c, d, M);\n"
        "    }\n"
        "    // MD5 state is LE; direct copy.\n"
        "    *h0 = a; *h1 = b; *h2 = c; *h3 = d;\n"
        "}\n"
        "\n");
    return rc;
}

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
/* Per-primitive outer body emit: RMD128 Metal twin (LE-schedule, 4-uint
 * state). Sub-phase 5b.1b (2026-05-27) Tier 1. Mirror of OpenCL twin
 * in hx_emit_opencl.c rev 1.9 byte-for-byte with Metal-specific
 * idioms: `device const uchar *pass` + `thread uint *` pointer args
 * via state_to_hex32_bytes_metal helper. rmd128_block resident in
 * metal_common.metal rev 1.26 carries the F4->F3->F2->F1 right-line
 * ordering correctly per Bosselaers Table 4 -- emit helper only needs
 * to drive the standard LE message schedule + padding. */
static int emit_outer_rmd128_concat_then_hash_metal(char **out,
                                                    size_t *cap, size_t *len)
{
    int rc;

    rc = hx_appendf(out, cap, len,
        "// hx: helper outer_rmd128_concat_then_hash_metal -- RIPEMD-128\n"
        "// over (hex32(md5(pass)) || pass). LE-schedule; NO state\n"
        "// byte-swap. rmd128_block from metal_common.metal (pointer-state).\n"
        "static inline void outer_rmd128_concat_then_hash_metal(\n"
        "    uint mma, uint mmb, uint mmc, uint mmd,\n"
        "    device const uchar *pass, int plen,\n"
        "    thread uint *h0, thread uint *h1, thread uint *h2,\n"
        "    thread uint *h3)\n"
        "{\n"
        "    if (plen < 0) plen = 0;\n"
        "    if (plen > HX_FAMILY_MAX_PASS) plen = HX_FAMILY_MAX_PASS;\n"
        "    int total_len = 32 + plen;\n"
        "\n"
        "    uint state[4];\n"
        "    state[0] = 0x67452301u;\n"
        "    state[1] = 0xEFCDAB89u;\n"
        "    state[2] = 0x98BADCFEu;\n"
        "    state[3] = 0x10325476u;\n"
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
        "        rmd128_block(state, M);\n"
        "        *h0 = state[0]; *h1 = state[1]; *h2 = state[2];\n"
        "        *h3 = state[3];\n"
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
        "    rmd128_block(state, M);\n"
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
        "        rmd128_block(state, M);\n"
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
        "    // RIPEMD-128 length suffix: TOTAL message bit-length\n"
        "    // (standard-conformant). The legacy CPU oracle bug in\n"
        "    // rmd128.c was fixed on 2026-05-27 -- the CPU now also\n"
        "    // encodes the total bit-length, matching Bosselaers's\n"
        "    // 1996 reference and sph_ripemd128. GPU emit therefore\n"
        "    // uses total_len*8 (no more bug-compat workaround).\n"
        "    ulong bitlen = (ulong)total_len * 8u;\n"
        "    if (pleft + 1 + 8 <= 64 || (first_has_pad && pleft + 8 <= 64)) {\n"
        "        M[14] = (uint)(bitlen & 0xffffffffu);\n"
        "        M[15] = (uint)(bitlen >> 32);\n"
        "        rmd128_block(state, M);\n"
        "    } else {\n"
        "        rmd128_block(state, M);\n"
        "        for (int w = 0; w < 16; w++) M[w] = 0u;\n"
        "        M[14] = (uint)(bitlen & 0xffffffffu);\n"
        "        M[15] = (uint)(bitlen >> 32);\n"
        "        rmd128_block(state, M);\n"
        "    }\n"
        "    *h0 = state[0]; *h1 = state[1]; *h2 = state[2];\n"
        "    *h3 = state[3];\n"
        "}\n"
        "\n");
    return rc;
}

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

/* Whirlpool Metal twin. Phase 5b Tier 2 sub-phase 5b.2a.3 (2026-05-27).
 *
 * Mirrors hx_emit_opencl.c rev 1.12 emit_outer_wrl_concat_then_hash
 * byte-for-byte with Metal-specific signatures (device-const pass,
 * thread* h0..h3) and the Metal wrl_block declared in metal_common.metal
 * rev 1.27. Bespoke per D16.3.a; ALWAYS multi-block path (single-block
 * fast path elided -- threshold 32+plen+1+32 <= 64 never holds for the
 * MAKE_MD5PASS family use case).
 *
 * BE-schedule + BE state output; epilogue byte-swap mirrors sha2_64
 * Metal helper. */
static int emit_outer_wrl_concat_then_hash_metal(char **out,
                                                 size_t *cap, size_t *len)
{
    int rc;

    rc = hx_appendf(out, cap, len,
        "// hx: helper outer_wrl_concat_then_hash_metal -- Whirlpool over\n"
        "// (hex32(md5(pass)) || pass). 64-byte block; 8-ulong state;\n"
        "// BE schedule; 32-byte BE length suffix at M[4..7] (M[4..6]=0);\n"
        "// ALWAYS multi-block (single-block fast path elided).\n"
        "static inline void outer_wrl_concat_then_hash_metal(\n"
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
        "    for (int i = 0; i < 8; i++) state[i] = 0ul;\n"
        "\n"
        "    uchar inner_hex[32];\n"
        "    state_to_hex32_bytes_metal(mma, mmb, mmc, mmd, inner_hex);\n"
        "\n"
        "    ulong M[8];\n"
        "    int byte_pos = 0;\n"
        "    int pass_consumed = 0;\n"
        "    int first_has_pad = 0;\n"
        "\n"
        "    int p_in_first = plen;\n"
        "    if (p_in_first > 32) p_in_first = 32;\n"
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
        "        for (int w = 4; w < 8; w++) M[w] = 0ul;\n"
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
        "    if (p_in_first == plen && byte_pos < 64) {\n"
        "        int pad_pos = byte_pos;\n"
        "        int wi = pad_pos >> 3;\n"
        "        int sh = (7 - (pad_pos & 7)) * 8;\n"
        "        M[wi] |= ((ulong)0x80u) << sh;\n"
        "        first_has_pad = 1;\n"
        "    }\n"
        "    wrl_block(state, M);\n"
        "\n"
        "    int pleft = plen - pass_consumed;\n"
        "    while (pleft >= 64) {\n"
        "        for (int w = 0; w < 8; w++) {\n"
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
        "        wrl_block(state, M);\n"
        "        pass_consumed += 64;\n"
        "        pleft -= 64;\n"
        "    }\n"
        "\n"
        "    for (int w = 0; w < 8; w++) M[w] = 0ul;\n"
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
        "    if ((!first_has_pad && pleft + 1 + 32 <= 64) ||\n"
        "        ( first_has_pad && pleft     + 32 <= 64)) {\n"
        "        ulong bitlen = (ulong)total_len * 8ul;\n"
        "        M[4] = 0ul;\n"
        "        M[5] = 0ul;\n"
        "        M[6] = 0ul;\n"
        "        M[7] = bitlen;\n"
        "        wrl_block(state, M);\n"
        "    } else {\n"
        "        wrl_block(state, M);\n"
        "        for (int w = 0; w < 8; w++) M[w] = 0ul;\n"
        "        ulong bitlen = (ulong)total_len * 8ul;\n"
        "        M[4] = 0ul;\n"
        "        M[5] = 0ul;\n"
        "        M[6] = 0ul;\n"
        "        M[7] = bitlen;\n"
        "        wrl_block(state, M);\n"
        "    }\n"
        "\n"
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
        "\n");
    return rc;
}

/* Tiger Metal twin. Phase 5b Tier 2 sub-phase 5b.2b.3 (2026-05-27).
 *
 * Mirrors hx_emit_opencl.c rev 1.13 emit_outer_tiger_concat_then_hash
 * byte-for-byte with Metal-specific signatures (device-const pass,
 * thread* h0..h3) and the Metal tiger_block declared in metal_common.metal
 * rev 1.28. Bespoke per D16.3.a; single-block fast path APPLICABLE for
 * plen <= 23.
 *
 * LE-schedule + LE state output direct extract; NO byte-swap epilogue
 * (unlike sha2_64/wrl). 0x01 padding byte (legacy Tiger, NOT Tiger2 0x80). */
static int emit_outer_tiger_concat_then_hash_metal(char **out,
                                                   size_t *cap, size_t *len)
{
    int rc;

    rc = hx_appendf(out, cap, len,
        "// hx: helper outer_tiger_concat_then_hash_metal -- Tiger over\n"
        "// (hex32(md5(pass)) || pass). 64-byte block; 3-ulong state;\n"
        "// LE schedule; 8-byte LE length suffix at M[7]; 0x01 pad byte.\n"
        "// Single-block fast path for plen <= 23.\n"
        "static inline void outer_tiger_concat_then_hash_metal(\n"
        "    uint mma, uint mmb, uint mmc, uint mmd,\n"
        "    device const uchar *pass, int plen,\n"
        "    thread uint *h0, thread uint *h1,\n"
        "    thread uint *h2, thread uint *h3)\n"
        "{\n"
        "    if (plen < 0) plen = 0;\n"
        "    if (plen > HX_FAMILY_MAX_PASS) plen = HX_FAMILY_MAX_PASS;\n"
        "    int total_len = 32 + plen;\n"
        "\n"
        "    ulong state[3];\n"
        "    state[0] = 0x0123456789abcdefUL;\n"
        "    state[1] = 0xfedcba9876543210UL;\n"
        "    state[2] = 0xf096a5b4c3b2e187UL;\n"
        "\n"
        "    uchar inner_hex[32];\n"
        "    state_to_hex32_bytes_metal(mma, mmb, mmc, mmd, inner_hex);\n"
        "\n"
        "    ulong M[8];\n"
        "    int byte_pos = 0;\n"
        "    int pass_consumed = 0;\n"
        "    int first_has_pad = 0;\n"
        "\n"
        "    int p_in_first = plen;\n"
        "    if (p_in_first > 32) p_in_first = 32;\n"
        "    {\n"
        "        for (int w = 0; w < 4; w++) {\n"
        "            int bo = w * 8;\n"
        "            M[w] =  (ulong)inner_hex[bo]\n"
        "                 | ((ulong)inner_hex[bo + 1] <<  8)\n"
        "                 | ((ulong)inner_hex[bo + 2] << 16)\n"
        "                 | ((ulong)inner_hex[bo + 3] << 24)\n"
        "                 | ((ulong)inner_hex[bo + 4] << 32)\n"
        "                 | ((ulong)inner_hex[bo + 5] << 40)\n"
        "                 | ((ulong)inner_hex[bo + 6] << 48)\n"
        "                 | ((ulong)inner_hex[bo + 7] << 56);\n"
        "        }\n"
        "        for (int w = 4; w < 8; w++) M[w] = 0ul;\n"
        "        for (int i = 0; i < p_in_first; i++) {\n"
        "            int abs_pos = 32 + i;\n"
        "            ulong v = (ulong)pass[i];\n"
        "            int wi = abs_pos >> 3;\n"
        "            int sh = (abs_pos & 7) * 8;\n"
        "            M[wi] |= v << sh;\n"
        "        }\n"
        "        pass_consumed = p_in_first;\n"
        "        byte_pos = 32 + p_in_first;\n"
        "    }\n"
        "\n"
        "    if (p_in_first == plen && byte_pos + 1 + 8 <= 64) {\n"
        "        int pad_pos = byte_pos;\n"
        "        int wi = pad_pos >> 3;\n"
        "        int sh = (pad_pos & 7) * 8;\n"
        "        M[wi] |= ((ulong)0x01u) << sh;\n"
        "        ulong bitlen = (ulong)total_len * 8ul;\n"
        "        M[7] = bitlen;\n"
        "        tiger_block(state, M);\n"
        "    } else {\n"
        "        if (p_in_first == plen && byte_pos < 64) {\n"
        "            int pad_pos = byte_pos;\n"
        "            int wi = pad_pos >> 3;\n"
        "            int sh = (pad_pos & 7) * 8;\n"
        "            M[wi] |= ((ulong)0x01u) << sh;\n"
        "            first_has_pad = 1;\n"
        "        }\n"
        "        tiger_block(state, M);\n"
        "\n"
        "        int pleft = plen - pass_consumed;\n"
        "        while (pleft >= 64) {\n"
        "            for (int w = 0; w < 8; w++) {\n"
        "                int bo = pass_consumed + w * 8;\n"
        "                M[w] =  (ulong)pass[bo]\n"
        "                     | ((ulong)pass[bo + 1] <<  8)\n"
        "                     | ((ulong)pass[bo + 2] << 16)\n"
        "                     | ((ulong)pass[bo + 3] << 24)\n"
        "                     | ((ulong)pass[bo + 4] << 32)\n"
        "                     | ((ulong)pass[bo + 5] << 40)\n"
        "                     | ((ulong)pass[bo + 6] << 48)\n"
        "                     | ((ulong)pass[bo + 7] << 56);\n"
        "            }\n"
        "            tiger_block(state, M);\n"
        "            pass_consumed += 64;\n"
        "            pleft -= 64;\n"
        "        }\n"
        "\n"
        "        for (int w = 0; w < 8; w++) M[w] = 0ul;\n"
        "        for (int i = 0; i < pleft; i++) {\n"
        "            ulong v = (ulong)pass[pass_consumed + i];\n"
        "            int wi = i >> 3;\n"
        "            int sh = (i & 7) * 8;\n"
        "            M[wi] |= v << sh;\n"
        "        }\n"
        "        if (!first_has_pad) {\n"
        "            int pad_pos = pleft;\n"
        "            int wi = pad_pos >> 3;\n"
        "            int sh = (pad_pos & 7) * 8;\n"
        "            M[wi] |= ((ulong)0x01u) << sh;\n"
        "        }\n"
        "        if ((!first_has_pad && pleft + 1 + 8 <= 64) ||\n"
        "            ( first_has_pad && pleft     + 8 <= 64)) {\n"
        "            ulong bitlen = (ulong)total_len * 8ul;\n"
        "            M[7] = bitlen;\n"
        "            tiger_block(state, M);\n"
        "        } else {\n"
        "            tiger_block(state, M);\n"
        "            for (int w = 0; w < 8; w++) M[w] = 0ul;\n"
        "            ulong bitlen = (ulong)total_len * 8ul;\n"
        "            M[7] = bitlen;\n"
        "            tiger_block(state, M);\n"
        "        }\n"
        "    }\n"
        "\n"
        "    *h0 = (uint)(state[0] & 0xffffffffUL);\n"
        "    *h1 = (uint)(state[0] >> 32);\n"
        "    *h2 = (uint)(state[1] & 0xffffffffUL);\n"
        "    *h3 = (uint)(state[1] >> 32);\n"
        "}\n"
        "\n");
    return rc;
}

/* Snefru Metal emit helper. Phase 5b Tier 4 sub-phase 5b.4a.3-metal twin
 * (2026-05-27). Mirror of OpenCL twin emit_outer_snefru_concat_then_hash
 * in hx_emit_opencl.c byte-for-byte modulo Metal idioms:
 *   - `device const uchar *pass` + `thread uint *h0..h3` signature.
 *   - `static inline` (Pattern 3; Metal inlines, opposite Pascal noinline).
 *   - snefru_block(thread uint*, thread const uchar*, int) from
 *     metal_common.metal rev 1.32. state_to_hex32_bytes_metal helper.
 *
 * PARAMETERISED per D18.1.a/D18.3.a over (is256, digest_bytes) -- same as
 * the OpenCL twin. Block-size asymmetry (R-Tier4-snefru-blocksize): SNE128
 * 48-byte data blocks / SNE256 32-byte; the per-width DBLK + length-field
 * byte offsets are baked into each emitted body. Snefru IV all-zero;
 * 8 rounds fixed. BE schedule + BE state output -> bswap32 into the
 * LE-uint probe frame (h0..h3). CPU recompute fills SNE256's remaining
 * 16 bytes on hit. */
static int emit_outer_snefru_concat_then_hash_metal(char **out,
                                                    size_t *cap, size_t *len,
                                                    int is256, int digest_bytes)
{
    int rc;
    int dblk = 64 - digest_bytes;   /* 48 (SNE128) or 32 (SNE256) */
    int off1 = dblk - 8;
    int off2 = dblk - 4;

    rc = hx_appendf(out, cap, len,
        "// hx: helper outer_snefru%d_concat_then_hash_metal -- Snefru-%d\n"
        "// over (hex32(md5(pass)) || pass). data_block_size = %d bytes\n"
        "// (is256=%d). IV all-zero; 8 rounds fixed. BE schedule + BE state\n"
        "// output -> bswap32 into LE-uint probe frame. Length field:\n"
        "// be2me_32(len>>29) at block[%d], be2me_32(len<<3) at block[%d].\n"
        "static inline void outer_snefru%d_concat_then_hash_metal(\n"
        "    uint mma, uint mmb, uint mmc, uint mmd,\n"
        "    device const uchar *pass, int plen,\n"
        "    thread uint *h0, thread uint *h1,\n"
        "    thread uint *h2, thread uint *h3)\n"
        "{\n"
        "    if (plen < 0) plen = 0;\n"
        "    if (plen > HX_FAMILY_MAX_PASS) plen = HX_FAMILY_MAX_PASS;\n"
        "    int total_len = 32 + plen;\n"
        "    const int DBLK = %d;\n"
        "\n"
        "    uint state[8];\n"
        "    for (int i = 0; i < 8; i++) state[i] = 0u;\n"
        "\n"
        "    uchar inner_hex[32];\n"
        "    state_to_hex32_bytes_metal(mma, mmb, mmc, mmd, inner_hex);\n"
        "\n"
        "    uchar block[48];\n"
        "    int consumed = 0;\n"
        "\n"
        "    while (total_len - consumed >= DBLK) {\n"
        "        for (int i = 0; i < DBLK; i++) {\n"
        "            int abs_pos = consumed + i;\n"
        "            block[i] = (abs_pos < 32) ? inner_hex[abs_pos]\n"
        "                                      : pass[abs_pos - 32];\n"
        "        }\n"
        "        snefru_block(state, block, %d);\n"
        "        consumed += DBLK;\n"
        "    }\n"
        "\n"
        "    int rem = total_len - consumed;\n"
        "    if (rem) {\n"
        "        for (int i = 0; i < DBLK; i++) block[i] = 0;\n"
        "        for (int i = 0; i < rem; i++) {\n"
        "            int abs_pos = consumed + i;\n"
        "            block[i] = (abs_pos < 32) ? inner_hex[abs_pos]\n"
        "                                      : pass[abs_pos - 32];\n"
        "        }\n"
        "        snefru_block(state, block, %d);\n"
        "    }\n"
        "\n"
        "    for (int i = 0; i < DBLK; i++) block[i] = 0;\n"
        "    ulong msglen = (ulong)total_len;\n"
        "    uint hi = (uint)(msglen >> 29);\n"
        "    uint lo = (uint)(msglen << 3);\n"
        "    block[%d + 0] = (uchar)(hi >> 24); block[%d + 1] = (uchar)(hi >> 16);\n"
        "    block[%d + 2] = (uchar)(hi >>  8); block[%d + 3] = (uchar)(hi);\n"
        "    block[%d + 0] = (uchar)(lo >> 24); block[%d + 1] = (uchar)(lo >> 16);\n"
        "    block[%d + 2] = (uchar)(lo >>  8); block[%d + 3] = (uchar)(lo);\n"
        "    snefru_block(state, block, %d);\n"
        "\n"
        "    *h0 = (state[0] >> 24) | ((state[0] >> 8) & 0xff00u)\n"
        "        | ((state[0] << 8) & 0xff0000u) | (state[0] << 24);\n"
        "    *h1 = (state[1] >> 24) | ((state[1] >> 8) & 0xff00u)\n"
        "        | ((state[1] << 8) & 0xff0000u) | (state[1] << 24);\n"
        "    *h2 = (state[2] >> 24) | ((state[2] >> 8) & 0xff00u)\n"
        "        | ((state[2] << 8) & 0xff0000u) | (state[2] << 24);\n"
        "    *h3 = (state[3] >> 24) | ((state[3] >> 8) & 0xff00u)\n"
        "        | ((state[3] << 8) & 0xff0000u) | (state[3] << 24);\n"
        "}\n"
        "\n",
        digest_bytes * 8, digest_bytes * 8,
        dblk, is256,
        off1, off2,
        digest_bytes * 8,
        dblk,
        is256,
        is256,
        off1, off1, off1, off1,
        off2, off2, off2, off2,
        is256);
    return rc;
}

/* GOST R 34.11-94 Metal emit helper. Phase 5b Tier 4 sub-phase 5b.4b.3-metal
 * twin (2026-05-27). Mirror of the OpenCL twin emit_outer_gost_concat_then_-
 * hash in hx_emit_opencl.c byte-for-byte modulo Metal idioms:
 *   - `device const uchar *pass` + `thread uint *h0..h3` signature.
 *   - `static inline` (Pattern 3; Metal inlines, opposite Pascal).
 *   - gost_block(thread uint*, thread const uint*) from metal_common.metal
 *     rev 1.33; MTL_GOST_SBOX_1..4 (TEST S-box set, NOT CryptoPro).
 *   - state_to_hex32_bytes_metal helper.
 *
 * Bespoke (D18.3.a) -- GOST is the only block-cipher-based family primitive,
 * carrying a running mod-2^256 checksum sum[8] across blocks + a dual
 * finalization (compress bit-length then checksum). 256-bit state[8]; 32-byte
 * (8 LE word) blocks. State output is LE; h0..h3 = hash[0..3] directly (no
 * byte-swap). CPU recompute fills the remaining 16 bytes on hit (digest = 32
 * bytes). Validated byte-exact in the C-mirror BEFORE shipping (the highest-
 * transcription-risk primitive in Phase 5b). */
static int emit_outer_gost_concat_then_hash_metal(char **out,
                                                  size_t *cap, size_t *len)
{
    int rc;

    rc = hx_appendf(out, cap, len,
        "// hx: helper outer_gost_concat_then_hash_metal -- GOST R 34.11-94\n"
        "// (TEST S-box set) over (hex32(md5(pass)) || pass). 256-bit\n"
        "// state[8]; 32-byte blocks (8 LE words). Running mod-2^256 checksum\n"
        "// sum[8] carried across blocks; dual finalization compresses the\n"
        "// bit-length block then the checksum block. State output LE;\n"
        "// h0..h3 = hash[0..3] directly (no byte-swap). CPU recompute fills\n"
        "// the remaining 16 bytes on hit (digest = 32 bytes).\n"
        "static inline void outer_gost_concat_then_hash_metal(\n"
        "    uint mma, uint mmb, uint mmc, uint mmd,\n"
        "    device const uchar *pass, int plen,\n"
        "    thread uint *h0, thread uint *h1,\n"
        "    thread uint *h2, thread uint *h3)\n"
        "{\n"
        "    if (plen < 0) plen = 0;\n"
        "    if (plen > HX_FAMILY_MAX_PASS) plen = HX_FAMILY_MAX_PASS;\n"
        "    int total_len = 32 + plen;\n"
        "\n"
        "    uint hash[8]; uint sum[8]; uint glen[8];\n"
        "    for (int i = 0; i < 8; i++) { hash[i] = 0u; sum[i] = 0u; glen[i] = 0u; }\n"
        "\n"
        "    uchar inner_hex[32];\n"
        "    state_to_hex32_bytes_metal(mma, mmb, mmc, mmd, inner_hex);\n"
        "\n"
        "    uchar block[32];\n"
        "    int consumed = 0;\n"
        "\n"
        "    while (consumed < total_len) {\n"
        "        int rem = total_len - consumed;\n"
        "        int blk = (rem >= 32) ? 32 : rem;\n"
        "        for (int i = 0; i < 32; i++) {\n"
        "            if (i < blk) {\n"
        "                int abs_pos = consumed + i;\n"
        "                block[i] = (abs_pos < 32) ? inner_hex[abs_pos]\n"
        "                                          : pass[abs_pos - 32];\n"
        "            } else {\n"
        "                block[i] = 0;\n"
        "            }\n"
        "        }\n"
        "        uint m[8];\n"
        "        uint c = 0u;\n"
        "        for (int i = 0; i < 8; i++) {\n"
        "            int j = i * 4;\n"
        "            uint a = ((uint)block[j]) | (((uint)block[j+1]) << 8)\n"
        "                   | (((uint)block[j+2]) << 16) | (((uint)block[j+3]) << 24);\n"
        "            m[i] = a;\n"
        "            uint b = sum[i];\n"
        "            uint cc = a + c + sum[i];\n"
        "            sum[i] = cc;\n"
        "            c = ((cc < a) || (cc < b)) ? 1u : 0u;\n"
        "        }\n"
        "        gost_block(hash, m);\n"
        "        uint bits = (uint)(blk << 3);\n"
        "        glen[0] = glen[0] + bits;\n"
        "        if (glen[0] < bits) glen[1] += 1u;\n"
        "        consumed += blk;\n"
        "    }\n"
        "\n"
        "    gost_block(hash, glen);\n"
        "    gost_block(hash, sum);\n"
        "\n"
        "    *h0 = hash[0];\n"
        "    *h1 = hash[1];\n"
        "    *h2 = hash[2];\n"
        "    *h3 = hash[3];\n"
        "}\n"
        "\n");
    return rc;
}

/* HAVAL Metal emit helper. Phase 5b Tier 3 sub-phase 5b.3a.3-metal twin
 * (2026-05-27). Mirror of OpenCL twin emit_outer_haval_concat_then_hash
 * in hx_emit_opencl.c rev 1.15+ byte-for-byte modulo Metal idioms:
 *   - `device const uchar *pass` + `thread uint *h0..h3` signature.
 *   - `static inline` (Pattern 3; Metal inlines, opposite Pascal).
 *   - haval3_block(thread uint*, thread const uint*) from metal_common
 *     .metal rev 1.29; MTL_HAVAL_IV + MTL_HAVAL_ROTR32.
 *   - state_to_hex32_bytes_metal helper.
 *
 * PARAMETERISED per D17.1.a over (passes, digest_bytes) -- same as the
 * OpenCL twin. 5b.3a ships 3-pass; 5b.3b/c extend to passes=4/5.
 *
 * CRITICAL HAVAL specifics IDENTICAL to OpenCL twin:
 *   - 128-byte block, 32 LE-packed uint32 words.
 *   - PAD-TOGGLE is 0x01 NOT 0x80 (donor mhash haval.c:760).
 *   - block[118..119] parameter encoding baked at C-emit time.
 *   - per-width digest fold JIT-specialised (donor havalFinal:816-911).
 *   - HAVAL state is LE-native; h0..h3 = state[0..3] direct (no swap).
 */
static int emit_outer_haval_concat_then_hash_metal(char **out, size_t *cap,
                                                   size_t *len,
                                                   int passes,
                                                   int digest_bytes)
{
    int rc;
    int hashbits = digest_bytes * 8;
    int byte118 = ((hashbits & 0x03) << 6) | ((passes & 0x07) << 3) | (1 & 0x07);
    int byte119 = (hashbits >> 2) & 0xff;

    rc = hx_appendf(out, cap, len,
        "// hx: helper outer_haval_concat_then_hash_metal -- HAVAL-%d/%d\n"
        "// over (hex32(md5(pass)) || pass). 128-byte HAVAL block; 32\n"
        "// LE-packed uint32 words. PAD-TOGGLE 0x01 NOT 0x80 (donor\n"
        "// mhash haval.c:760). block[118..119] = 0x%02x 0x%02x for this\n"
        "// variant. Output: first 16 bytes (h0..h3) of folded state.\n"
        "static inline void outer_haval_concat_then_hash_metal(\n"
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
        "    for (int i = 0; i < 8; i++) state[i] = MTL_HAVAL_IV[i];\n"
        "\n"
        "    uchar inner_hex[32];\n"
        "    state_to_hex32_bytes_metal(mma, mmb, mmc, mmd, inner_hex);\n"
        "\n"
        "    uint M[32];\n"
        "    uchar block[128];\n"
        "    int consumed = 0;\n"
        "\n"
        "    while (total_len - consumed >= 128) {\n"
        "        for (int i = 0; i < 128; i++) {\n"
        "            int abs_pos = consumed + i;\n"
        "            block[i] = (abs_pos < 32) ? inner_hex[abs_pos]\n"
        "                                      : pass[abs_pos - 32];\n"
        "        }\n"
        "        for (int w = 0; w < 32; w++) {\n"
        "            int bo = w * 4;\n"
        "            M[w] = (uint)block[bo] | ((uint)block[bo+1] << 8)\n"
        "                 | ((uint)block[bo+2] << 16) | ((uint)block[bo+3] << 24);\n"
        "        }\n"
        "        haval%d_block(state, M);\n"
        "        consumed += 128;\n"
        "    }\n"
        "\n"
        "    int occupied = total_len - consumed;\n"
        "    for (int i = 0; i < 128; i++) block[i] = 0;\n"
        "    for (int i = 0; i < occupied; i++) {\n"
        "        int abs_pos = consumed + i;\n"
        "        block[i] = (abs_pos < 32) ? inner_hex[abs_pos]\n"
        "                                  : pass[abs_pos - 32];\n"
        "    }\n"
        "    // HAVAL pad toggle is 0x01 NOT 0x80 (donor havalFinal:760).\n"
        "    block[occupied] = 0x01;\n"
        "    if (occupied + 1 > 118) {\n"
        "        for (int w = 0; w < 32; w++) {\n"
        "            int bo = w * 4;\n"
        "            M[w] = (uint)block[bo] | ((uint)block[bo+1] << 8)\n"
        "                 | ((uint)block[bo+2] << 16) | ((uint)block[bo+3] << 24);\n"
        "        }\n"
        "        haval%d_block(state, M);\n"
        "        for (int i = 0; i < 128; i++) block[i] = 0;\n"
        "    }\n"
        "    block[118] = (uchar)0x%02x;\n"
        "    block[119] = (uchar)0x%02x;\n"
        "    ulong bitlen = (ulong)total_len * 8ul;\n"
        "    block[120] = (uchar)(bitlen);\n"
        "    block[121] = (uchar)(bitlen >> 8);\n"
        "    block[122] = (uchar)(bitlen >> 16);\n"
        "    block[123] = (uchar)(bitlen >> 24);\n"
        "    block[124] = (uchar)(bitlen >> 32);\n"
        "    block[125] = (uchar)(bitlen >> 40);\n"
        "    block[126] = (uchar)(bitlen >> 48);\n"
        "    block[127] = (uchar)(bitlen >> 56);\n"
        "    for (int w = 0; w < 32; w++) {\n"
        "        int bo = w * 4;\n"
        "        M[w] = (uint)block[bo] | ((uint)block[bo+1] << 8)\n"
        "             | ((uint)block[bo+2] << 16) | ((uint)block[bo+3] << 24);\n"
        "    }\n"
        "    haval%d_block(state, M);\n"
        "\n",
        hashbits, passes,
        byte118, byte119,
        passes,         /* full-block compress */
        passes,         /* spill-block compress */
        byte118, byte119,
        passes);        /* final compress */
    if (rc < 0) return rc;

    if (digest_bytes == 16) {
        rc = hx_appendf(out, cap, len,
        "    // 128-bit digest fold (donor havalFinal:819-841).\n"
        "    state[3] += (state[7] & 0xFF000000u) | (state[6] & 0x00FF0000u)\n"
        "              | (state[5] & 0x0000FF00u) | (state[4] & 0x000000FFu);\n"
        "    state[2] += (((state[7] & 0x00FF0000u) | (state[6] & 0x0000FF00u)\n"
        "               | (state[5] & 0x000000FFu)) << 8)\n"
        "               | ((state[4] & 0xFF000000u) >> 24);\n"
        "    state[1] += (((state[7] & 0x0000FF00u) | (state[6] & 0x000000FFu)) << 16)\n"
        "               | (((state[5] & 0xFF000000u) | (state[4] & 0x00FF0000u)) >> 16);\n"
        "    state[0] += (((state[6] & 0xFF000000u) | (state[5] & 0x00FF0000u)\n"
        "               | (state[4] & 0x0000FF00u)) >> 8)\n"
        "               | ((state[7] & 0x000000FFu) << 24);\n");
    } else if (digest_bytes == 20) {
        rc = hx_appendf(out, cap, len,
        "    // 160-bit digest fold (donor havalFinal:848-859).\n"
        "    state[4] += ((state[7] & 0xFE000000u) | (state[6] & 0x01F80000u)\n"
        "               | (state[5] & 0x0007F000u)) >> 12;\n"
        "    state[3] += ((state[7] & 0x01F80000u) | (state[6] & 0x0007F000u)\n"
        "               | (state[5] & 0x00000FC0u)) >> 6;\n"
        "    state[2] += ((state[7] & 0x0007F000u) | (state[6] & 0x00000FC0u)\n"
        "               | (state[5] & 0x0000003Fu));\n"
        "    state[1] += MTL_HAVAL_ROTR32((state[7] & 0x00000FC0u)\n"
        "               | (state[6] & 0x0000003Fu) | (state[5] & 0xFE000000u), 25);\n"
        "    state[0] += MTL_HAVAL_ROTR32((state[7] & 0x0000003Fu)\n"
        "               | (state[6] & 0xFE000000u) | (state[5] & 0x01F80000u), 19);\n");
    } else if (digest_bytes == 24) {
        rc = hx_appendf(out, cap, len,
        "    // 192-bit digest fold (donor havalFinal:868-880).\n"
        "    state[5] += ((state[7] & 0xFC000000u) | (state[6] & 0x03E00000u)) >> 21;\n"
        "    state[4] += ((state[7] & 0x03E00000u) | (state[6] & 0x001F0000u)) >> 16;\n"
        "    state[3] += ((state[7] & 0x001F0000u) | (state[6] & 0x0000FC00u)) >> 10;\n"
        "    state[2] += ((state[7] & 0x0000FC00u) | (state[6] & 0x000003E0u)) >> 5;\n"
        "    state[1] += ((state[7] & 0x000003E0u) | (state[6] & 0x0000001Fu));\n"
        "    state[0] += MTL_HAVAL_ROTR32((state[7] & 0x0000001Fu)\n"
        "               | (state[6] & 0xFC000000u), 26);\n");
    } else if (digest_bytes == 28) {
        rc = hx_appendf(out, cap, len,
        "    // 224-bit digest fold (donor havalFinal:889-895).\n"
        "    state[6] += (state[7]      ) & 0x0000000Fu;\n"
        "    state[5] += (state[7] >>  4) & 0x0000001Fu;\n"
        "    state[4] += (state[7] >>  9) & 0x0000000Fu;\n"
        "    state[3] += (state[7] >> 13) & 0x0000001Fu;\n"
        "    state[2] += (state[7] >> 18) & 0x0000000Fu;\n"
        "    state[1] += (state[7] >> 22) & 0x0000001Fu;\n"
        "    state[0] += (state[7] >> 27) & 0x0000001Fu;\n");
    } else {
        rc = hx_appendf(out, cap, len,
        "    // 256-bit: NO fold (donor havalFinal:903-908 direct output).\n");
    }
    if (rc < 0) return rc;

    rc = hx_appendf(out, cap, len,
        "\n"
        "    // HAVAL state is LE-native; h0..h3 = state[0..3] direct.\n"
        "    *h0 = state[0];\n"
        "    *h1 = state[1];\n"
        "    *h2 = state[2];\n"
        "    *h3 = state[3];\n"
        "}\n"
        "\n");
    return rc;
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
/* Forward decl: Metal multi-emit kernel body (e123 MD5MD5PASS). Defined
 * below emit_family_md5pass_kernel_metal. */
static int emit_family_md5pass_kernel_metal_multiemit(
    char **out, size_t *cap, size_t *len, int job_enum);

static int emit_family_md5pass_kernel_metal(char **out, size_t *cap,
                                            size_t *len,
                                            enum hx_primitive_id outer_id,
                                            const char *outer_name,
                                            int outer_digest_bytes,
                                            int job_enum,
                                            int emit_class)
{
    int rc;

    /* Sub-phase 5c.3 (2026-05-27): multi-emit members (e123 MD5MD5PASS)
     * take a dedicated Metal kernel body that runs the probe + EMIT_HIT_4
     * block ONCE PER VARIANT (N=2: sep=0 canonical, sep=1 colon). Single-
     * emit members fall through to the existing body UNCHANGED (G2
     * regression safety: the per-variant logic is fully isolated; mirror
     * of the OpenCL twin at hx_emit_opencl.c rev 1.19). */
    if (emit_class == HX_EMIT_MULTI) {
        return emit_family_md5pass_kernel_metal_multiemit(out, cap, len,
                                                          job_enum);
    }

    /* Sub-phase 5a.4 (2026-05-23): per-primitive dispatch for the Metal
     * twin. 7 of 8 5a-supported primitives wired (md4, sha1, sha224,
     * sha256, sha384, sha512, rmd160). HX_PRIM_MD5 multi-emit handled by
     * the branch above. Other family members filtered upstream. */
    int helper_has_h4 = 0;
    switch (outer_id) {
        case HX_PRIM_SHA1:   helper_has_h4 = 1; break;
        case HX_PRIM_RMD160: helper_has_h4 = 1; break;
        case HX_PRIM_MD2:
        case HX_PRIM_MD4:
        case HX_PRIM_RMD128:
        case HX_PRIM_SHA224:
        case HX_PRIM_SHA256:
        case HX_PRIM_SHA384:
        case HX_PRIM_SHA512:
        case HX_PRIM_WRL:
        case HX_PRIM_TIGER:
        /* Phase 5b Tier 4 sub-phase 5b.4a-metal (2026-05-27): the 2 Snefru
         * widths. 4-uint probe (first 16 bytes); CPU recompute fills
         * SNE256's remaining 16 bytes on hit. SNE128 is exactly 16 bytes. */
        case HX_PRIM_SNE128:
        case HX_PRIM_SNE256:
        /* Phase 5b Tier 4 sub-phase 5b.4b-metal (2026-05-27): GOST R
         * 34.11-94 (e125). 4-uint probe (first 16 bytes); CPU recompute
         * fills the remaining 16 bytes on hit. */
        case HX_PRIM_GOST:
        /* Phase 5b Tier 3 sub-phase 5b.3a-metal (2026-05-27): 5 3-pass
         * HAVAL variants. 4-uint probe (first 16 bytes); CPU recompute
         * fills wider digests on hit. */
        case HX_PRIM_HAV128_3:
        case HX_PRIM_HAV160_3:
        case HX_PRIM_HAV192_3:
        case HX_PRIM_HAV224_3:
        case HX_PRIM_HAV256_3:
        /* Phase 5b Tier 3 sub-phase 5b.3b-metal (2026-05-27): 4-pass
         * HAVAL variants share the same emitted Metal function name. */
        case HX_PRIM_HAV128_4:
        case HX_PRIM_HAV160_4:
        case HX_PRIM_HAV192_4:
        case HX_PRIM_HAV224_4:
        case HX_PRIM_HAV256_4:
        /* Phase 5b Tier 3 sub-phase 5b.3c-metal (2026-05-27): 5-pass
         * HAVAL variants share the same emitted Metal function name. */
        case HX_PRIM_HAV128_5:
        case HX_PRIM_HAV160_5:
        case HX_PRIM_HAV192_5:
        case HX_PRIM_HAV224_5:
        case HX_PRIM_HAV256_5:
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
        : (outer_id == HX_PRIM_MD2) ?
            "    uint h0, h1, h2, h3;\n"
            "    outer_md2_concat_then_hash_metal(ia, ib, ic, id,\n"
            "                                     pass_bytes, (int)plen,\n"
            "                                     &h0, &h1, &h2, &h3);\n"
        : (outer_id == HX_PRIM_RMD128) ?
            "    uint h0, h1, h2, h3;\n"
            "    outer_rmd128_concat_then_hash_metal(ia, ib, ic, id,\n"
            "                                        pass_bytes, (int)plen,\n"
            "                                        &h0, &h1, &h2, &h3);\n"
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
        : (outer_id == HX_PRIM_SHA512) ?
            "    uint h0, h1, h2, h3;\n"
            "    outer_sha512_concat_then_hash_metal(ia, ib, ic, id,\n"
            "                                        pass_bytes, (int)plen,\n"
            "                                        &h0, &h1, &h2, &h3);\n"
        : (outer_id == HX_PRIM_WRL) ?
            "    uint h0, h1, h2, h3;\n"
            "    outer_wrl_concat_then_hash_metal(ia, ib, ic, id,\n"
            "                                     pass_bytes, (int)plen,\n"
            "                                     &h0, &h1, &h2, &h3);\n"
        : (outer_id == HX_PRIM_TIGER) ?
            "    uint h0, h1, h2, h3;\n"
            "    outer_tiger_concat_then_hash_metal(ia, ib, ic, id,\n"
            "                                       pass_bytes, (int)plen,\n"
            "                                       &h0, &h1, &h2, &h3);\n"
        : (outer_id == HX_PRIM_SNE128) ?
            "    uint h0, h1, h2, h3;\n"
            "    outer_snefru128_concat_then_hash_metal(ia, ib, ic, id,\n"
            "                                           pass_bytes, (int)plen,\n"
            "                                           &h0, &h1, &h2, &h3);\n"
        : (outer_id == HX_PRIM_SNE256) ?
            "    uint h0, h1, h2, h3;\n"
            "    outer_snefru256_concat_then_hash_metal(ia, ib, ic, id,\n"
            "                                           pass_bytes, (int)plen,\n"
            "                                           &h0, &h1, &h2, &h3);\n"
        : (outer_id == HX_PRIM_GOST) ?
            "    uint h0, h1, h2, h3;\n"
            "    outer_gost_concat_then_hash_metal(ia, ib, ic, id,\n"
            "                                      pass_bytes, (int)plen,\n"
            "                                      &h0, &h1, &h2, &h3);\n"
        : /* HAVAL (any 3-pass, 4-pass, or 5-pass variant; parameterised
           * helper -- emitted Metal function name is identical, passes/
           * width baked into the body). */
            "    uint h0, h1, h2, h3;\n"
            "    outer_haval_concat_then_hash_metal(ia, ib, ic, id,\n"
            "                                       pass_bytes, (int)plen,\n"
            "                                       &h0, &h1, &h2, &h3);\n",
        helper_has_h4
            ? "    (void)h4;  // 5th word reserved for round-trip readback.\n"
            : "");

    return rc;
}

/* Sub-phase 5c.3 (2026-05-27) Metal twin: multi-emit kernel body for
 * e123 MD5MD5PASS -- the FIRST multi-emit member. Structural mirror of the
 * OpenCL twin emit_family_md5pass_kernel_multiemit (hx_emit_opencl.c rev
 * 1.19). Computes MD5(pass) ONCE (shared inner; natural hoist), then runs
 * a compile-time-N=2 unrolled loop where each iteration builds its variant
 * outer message (sep=0 canonical / sep=1 colon), MD5s it via
 * outer_md5_concat_then_hash_metal, probes (-> its own matched_idx), and
 * calls the EXISTING EMIT_HIT_4_DEDUP_OR_OVERFLOW macro UNCHANGED. Dedup
 * keys on per-variant matched_idx (the matched loaded-hash slot), so two
 * distinct loaded targets emit two independent cracks; a same-target
 * collision (unreachable by MD5 construction) would correctly dedup --
 * matching CPU semantics (mdxfind.c:25181-25204). The 16-byte fingerprint
 * self-identifies the matched hash -> NO variant tag in the hit record. */
static int emit_family_md5pass_kernel_metal_multiemit(
    char **out, size_t *cap, size_t *len, int job_enum)
{
    int rc;

    rc = hx_appendf(out, cap, len,
        "// hx: family kernel for e%d MD5MD5PASS (MULTI-EMIT, N=2 variants;\n"
        "// Metal); digest=16 bytes; probe uses h0..h3 per compact_fp/idx.\n"
        "// variant 0 = md5(hex32(md5(pass)) . pass); variant 1 = md5(hex32 .\n"
        "// ':' . pass). Each variant probes + emits independently against\n"
        "// its own matched loaded-hash slot (dedup keyed on matched_idx,\n"
        "// unchanged). kernel signature mirrors the single-emit family body\n"
        "// (18 sequential [[buffer(N)]] args; salt args 3/4/5 IGNORED).\n"
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
        "    // hx: state EMIT_KERNEL_PREAMBLE (family MD5PASS, multi-emit; Metal)\n"
        "    device const OCLParams *params_buf =\n"
        "        (device const OCLParams *)payload;\n"
        "    OCLParams params = *params_buf;\n"
        "\n"
        "    uint word_idx = gid;\n"
        "    if (word_idx >= params.num_words) return;\n"
        "\n"
        "    (void)salts; (void)salt_offsets; (void)salt_lens;\n"
        "\n"
        "    // hx: state EMIT_PRE_INVARIANT (compute MD5(pass) ONCE)\n"
        "    uint wpos = b_chunk_index[word_idx];\n"
        "    if (wpos >= params.packed_size) return;  // defensive\n"
        "    uint plen = (uint)b_packed_buf[wpos];\n"
        "    device const uchar *pass_bytes = b_packed_buf + wpos + 1u;\n"
        "\n"
        "    // OP_CALL md5 #1 (inner): MD5(pass) -> (ia,ib,ic,id). Shared\n"
        "    // across BOTH variants (natural hoist).\n"
        "    uint ia, ib, ic, id;\n"
        "    md5_buf_global_metal(pass_bytes, (int)plen, &ia, &ib, &ic, &id);\n"
        "\n"
        "    uint widx = params.base_word_idx + word_idx;\n"
        "\n",
        job_enum);
    if (rc < 0) return rc;

    /* Emit N=2 unrolled probe + EMIT_HIT_4 blocks (sep=0, sep=1). The
     * EMIT_HIT_4_DEDUP_OR_OVERFLOW macro is reused VERBATIM from the
     * single-emit body; the only change is each block computes its own
     * variant digest (via outer_md5_concat_then_hash_metal(sep)) and
     * resolves its own matched_idx. Mirror of the OpenCL twin. */
    for (int sep = 0; sep <= 1; sep++) {
        rc = hx_appendf(out, cap, len,
            "    // hx: state EMIT_PROBE_AND_HIT variant %d (sep=%d)\n"
            "    {\n"
            "        uint h0, h1, h2, h3;\n"
            "        outer_md5_concat_then_hash_metal(ia, ib, ic, id,\n"
            "                                         pass_bytes, (int)plen, %d,\n"
            "                                         &h0, &h1, &h2, &h3);\n"
            "        uint matched_idx = 0u;\n"
            "        if (probe_compact_idx(h0, h1, h2, h3,\n"
            "                              compact_fp, compact_idx,\n"
            "                              params.compact_mask, params.max_probe,\n"
            "                              params.hash_data_count,\n"
            "                              hash_data_buf, hash_data_off,\n"
            "                              overflow_keys, overflow_hashes,\n"
            "                              overflow_offsets, params.overflow_count,\n"
            "                              &matched_idx))\n"
            "        {\n"
            "            uint mask = 1u;  // iter==1; dedup slot 0\n"
            "            // Unsalted family: sidx is always 0 in the emitted hit.\n"
            "            EMIT_HIT_4_DEDUP_OR_OVERFLOW(hits, hit_count, params.max_hits,\n"
            "                       widx, 0u, 1u, h0, h1, h2, h3,\n"
            "                       hashes_shown, matched_idx, mask,\n"
            "                       ovr_set, ovr_gid, gid);\n"
            "        }\n"
            "    }\n"
            "\n",
            sep, sep, sep);
        if (rc < 0) return rc;
    }

    rc = hx_appendf(out, cap, len,
        "    // hx: state EMIT_KERNEL_FOOTER (multi-emit; Metal)\n"
        "}\n");
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
    /* Sub-phase 5c.3 (2026-05-27) Metal twin: MD5-as-OUTER is now supported
     * for the e123 MD5MD5PASS multi-emit member (mirror of OpenCL twin at
     * hx_emit_opencl.c rev 1.19). It is admitted ONLY when the spec entry
     * is flagged emit_class == HX_EMIT_MULTI (the generator sets this for
     * e123 via the Note-[24] markup-strip). An MD5 outer with emit_class
     * SINGLE would be an unexpected non-multi-emit MD5 member; FATAL
     * because the single-emit MD5 path is not the intended shape (e123 is
     * the only MD5-outer family member, and it is multi-emit by
     * construction). */
    if (outer_id == HX_PRIM_MD5 && entry->emit_class != HX_EMIT_MULTI) {
        fprintf(stderr,
            "FATAL: %s:%d hx_emit_family_md5pass_metal: e%d %s outer "
            "primitive 'md5' with emit_class=%d (expected HX_EMIT_MULTI=%d). "
            "MD5-as-outer is only wired for the e123 multi-emit member; an "
            "MD5-outer single-emit shape is unexpected. Generator/markup "
            "drift?\n",
            __FILE__, __LINE__, entry->job_enum,
            entry->name ? entry->name : "(noname)",
            entry->emit_class, (int)HX_EMIT_MULTI);
        return -1;
    }
    if (outer_id != HX_PRIM_MD5 &&
        outer_id != HX_PRIM_SHA1 && outer_id != HX_PRIM_MD4 &&
        outer_id != HX_PRIM_MD2 && outer_id != HX_PRIM_RMD128 &&
        outer_id != HX_PRIM_RMD160 && outer_id != HX_PRIM_SHA224 &&
        outer_id != HX_PRIM_SHA256 && outer_id != HX_PRIM_SHA384 &&
        outer_id != HX_PRIM_SHA512 && outer_id != HX_PRIM_WRL &&
        outer_id != HX_PRIM_TIGER &&
        outer_id != HX_PRIM_SNE128 && outer_id != HX_PRIM_SNE256 &&
        outer_id != HX_PRIM_GOST &&
        outer_id != HX_PRIM_HAV128_3 && outer_id != HX_PRIM_HAV160_3 &&
        outer_id != HX_PRIM_HAV192_3 && outer_id != HX_PRIM_HAV224_3 &&
        outer_id != HX_PRIM_HAV256_3 &&
        outer_id != HX_PRIM_HAV128_4 && outer_id != HX_PRIM_HAV160_4 &&
        outer_id != HX_PRIM_HAV192_4 && outer_id != HX_PRIM_HAV224_4 &&
        outer_id != HX_PRIM_HAV256_4 &&
        outer_id != HX_PRIM_HAV128_5 && outer_id != HX_PRIM_HAV160_5 &&
        outer_id != HX_PRIM_HAV192_5 && outer_id != HX_PRIM_HAV224_5 &&
        outer_id != HX_PRIM_HAV256_5)
    {
        fprintf(stderr,
            "FATAL: %s:%d hx_emit_family_md5pass_metal: e%d %s outer "
            "primitive '%s' is in supported_5a but not in the 5a.4 + "
            "5b.1a + 5b.1b + 5b.2a + 5b.2b + 5b.3a + 5b.3b + 5b.3c + 5b.4a "
            "+ 5b.4b + 5c.3 wired Metal subset (md5(multi-emit) md2 md4 "
            "rmd128 sha1 sha224 sha256 "
            "sha384 sha512 rmd160 wrl tiger sne128 sne256 gost hav128_3 "
            "hav160_3 hav192_3 hav224_3 hav256_3 hav128_4 hav160_4 hav192_4 "
            "hav224_4 hav256_4 hav128_5 hav160_5 hav192_5 hav224_5 "
            "hav256_5).\n",
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
        /* Sub-phase 5c.3 (2026-05-27) Metal twin: MD5-as-outer multi-emit
         * helper (e123 MD5MD5PASS). The helper emits ONE function with a
         * `sep` parameter; the multi-emit kernel body calls it twice
         * (sep=0 canonical, sep=1 colon). Only reached for
         * emit_class==HX_EMIT_MULTI (gated above). */
        case HX_PRIM_MD5:
            rc = emit_outer_md5_concat_then_hash_metal(out, out_cap, &cur_len); break;
        case HX_PRIM_SHA1:
            rc = emit_outer_sha1_concat_then_hash_metal(out, out_cap, &cur_len); break;
        case HX_PRIM_MD2:
            rc = emit_outer_md2_concat_then_hash_metal(out, out_cap, &cur_len); break;
        case HX_PRIM_MD4:
            rc = emit_outer_md4_concat_then_hash_metal(out, out_cap, &cur_len); break;
        case HX_PRIM_RMD128:
            rc = emit_outer_rmd128_concat_then_hash_metal(out, out_cap, &cur_len); break;
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
        case HX_PRIM_WRL:
            rc = emit_outer_wrl_concat_then_hash_metal(out, out_cap, &cur_len); break;
        case HX_PRIM_TIGER:
            rc = emit_outer_tiger_concat_then_hash_metal(out, out_cap, &cur_len); break;
        /* Phase 5b Tier 4 sub-phase 5b.4a-metal (2026-05-27): the 2 Snefru
         * widths -> ONE parameterised Metal helper specialised on (is256,
         * digest_bytes). SNE128 is256=0 / SNE256 is256=1. gost (e125)
         * ships in 5b.4b. */
        case HX_PRIM_SNE128:
            rc = emit_outer_snefru_concat_then_hash_metal(out, out_cap, &cur_len,
                                                          0, outer_digest_bytes);
            break;
        case HX_PRIM_SNE256:
            rc = emit_outer_snefru_concat_then_hash_metal(out, out_cap, &cur_len,
                                                          1, outer_digest_bytes);
            break;
        /* Phase 5b Tier 4 sub-phase 5b.4b-metal (2026-05-27): GOST R
         * 34.11-94 (e125) -- bespoke Metal helper. Block-cipher core +
         * mod-2^256 checksum carry + dual finalization. After this ship the
         * MAKE_MD5PASS family reaches 29/30 GPU-eligible. */
        case HX_PRIM_GOST:
            rc = emit_outer_gost_concat_then_hash_metal(out, out_cap, &cur_len);
            break;
        /* Phase 5b Tier 3 sub-phase 5b.3a-metal (2026-05-27): 5 3-pass
         * HAVAL variants -> ONE parameterised Metal helper. passes=3;
         * digest_bytes from outer_digest_bytes. */
        case HX_PRIM_HAV128_3:
        case HX_PRIM_HAV160_3:
        case HX_PRIM_HAV192_3:
        case HX_PRIM_HAV224_3:
        case HX_PRIM_HAV256_3:
            rc = emit_outer_haval_concat_then_hash_metal(out, out_cap,
                                                         &cur_len, 3,
                                                         outer_digest_bytes);
            break;
        /* Phase 5b Tier 3 sub-phase 5b.3b-metal (2026-05-27): 5 4-pass
         * HAVAL variants -> SAME parameterised Metal helper with passes=4
         * (emits haval4_block call + block[118] passes=4 encoding). */
        case HX_PRIM_HAV128_4:
        case HX_PRIM_HAV160_4:
        case HX_PRIM_HAV192_4:
        case HX_PRIM_HAV224_4:
        case HX_PRIM_HAV256_4:
            rc = emit_outer_haval_concat_then_hash_metal(out, out_cap,
                                                         &cur_len, 4,
                                                         outer_digest_bytes);
            break;
        /* Phase 5b Tier 3 sub-phase 5b.3c-metal (2026-05-27): 5 5-pass
         * HAVAL variants -> SAME parameterised Metal helper with passes=5
         * (emits haval5_block call + block[118] passes=5 encoding). */
        case HX_PRIM_HAV128_5:
        case HX_PRIM_HAV160_5:
        case HX_PRIM_HAV192_5:
        case HX_PRIM_HAV224_5:
        case HX_PRIM_HAV256_5:
            rc = emit_outer_haval_concat_then_hash_metal(out, out_cap,
                                                         &cur_len, 5,
                                                         outer_digest_bytes);
            break;
        default:
            fprintf(stderr,
                "FATAL: %s:%d hx_emit_family_md5pass_metal: unreachable "
                "(outer_id=%d not in 5a.4 + 5b.1 + 5b.2a + 5b.2b + 5b.3a "
                "+ 5b.3b + 5b.3c wired set)\n",
                __FILE__, __LINE__, (int)outer_id);
            return -1;
    }
    if (rc < 0) return rc;

    rc = emit_family_md5pass_kernel_metal(out, out_cap, &cur_len,
                                          outer_id, outer_name,
                                          outer_digest_bytes,
                                          entry->job_enum,
                                          entry->emit_class);
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
 * Phase 1b Batch 1 (2026-05-28): unsalted single-hash emitter (Metal).
 *
 * Metal twin of hx_emit_unsalted_single_opencl. Same one-shot hash(pass)
 * shape (HX_PATTERN_UNSALTED_SINGLE). Token translations vs OpenCL twin:
 * device const uchar* candidate, thread uint* outputs, atomic_uint hit
 * args, sequential [[buffer(N)]] binding, gid via
 * [[thread_position_in_grid]]. Per metal_common.metal block signatures:
 *   md5_block(thread uint& h0..h3, thread const uint *M)   -- REFERENCES
 *   md4_block / sha1_block / sha256_block(thread uint *state, M) -- PTR
 * SHA1/SHA256 BE state byte-swapped to LE for the compact_fp probe (per
 * feedback_be_state_primitives_need_byteswap_in_codegen.md). No scalar
 * bitselect (feedback_metal_xcode26_bitselect_scalar.md) -- shift/mask/
 * add only. Validated byte-exact in plain C (80/80) before GPU JIT.
 * ==================================================================== */

static int emit_unsalted_single_helpers_metal(char **out, size_t *cap, size_t *len)
{
    int rc;
    rc = hx_appendf(out, cap, len,
        "// ====================================================================\n"
        "// hx codegen Phase 1b Batch 1 (2026-05-28): unsalted single-hash (Metal)\n"
        "// Emitted by hx_emit_unsalted_single_metal()\n"
        "// Pattern matched: HX_PATTERN_UNSALTED_SINGLE\n"
        "// Algorithm: hash(pass)  (no inner md5, no hex32, no concat)\n"
        "// Helpers from metal_common.metal: md5_block (refs), md4_block,\n"
        "//   sha1_block, sha256_block (ptrs), probe_compact_idx,\n"
        "//   EMIT_HIT_4_DEDUP_OR_OVERFLOW\n"
        "// ====================================================================\n"
        "\n");
    if (rc < 0) return rc;

    rc = hx_appendf(out, cap, len,
        "// hx: helper usp_md5_buf_global_metal -- MD5 of device candidate.\n"
        "static inline void usp_md5_buf_global_metal(device const uchar *data, int len,\n"
        "                                            thread uint *h0, thread uint *h1,\n"
        "                                            thread uint *h2, thread uint *h3)\n"
        "{\n"
        "    uint M[16];\n"
        "    int pos = 0;\n"
        "    uint a = 0x67452301u, b = 0xEFCDAB89u, c = 0x98BADCFEu, d = 0x10325476u;\n"
        "    while (len - pos >= 64) {\n"
        "        for (int j = 0; j < 16; j++) {\n"
        "            int q = pos + j * 4;\n"
        "            M[j] = (uint)data[q] | ((uint)data[q+1] << 8)\n"
        "                 | ((uint)data[q+2] << 16) | ((uint)data[q+3] << 24);\n"
        "        }\n"
        "        md5_block(a, b, c, d, M); pos += 64;\n"
        "    }\n"
        "    int rem = len - pos;\n"
        "    for (int j = 0; j < 16; j++) M[j] = 0;\n"
        "    for (int i = 0; i < rem; i++) M[i >> 2] |= (uint)data[pos+i] << ((i & 3) * 8);\n"
        "    M[rem >> 2] |= (uint)0x80 << ((rem & 3) * 8);\n"
        "    ulong bits = (ulong)len * 8ul;\n"
        "    if (rem < 56) {\n"
        "        M[14] = (uint)(bits & 0xfffffffful); M[15] = (uint)(bits >> 32);\n"
        "        md5_block(a, b, c, d, M);\n"
        "    } else {\n"
        "        md5_block(a, b, c, d, M);\n"
        "        for (int j = 0; j < 16; j++) M[j] = 0;\n"
        "        M[14] = (uint)(bits & 0xfffffffful); M[15] = (uint)(bits >> 32);\n"
        "        md5_block(a, b, c, d, M);\n"
        "    }\n"
        "    *h0 = a; *h1 = b; *h2 = c; *h3 = d;\n"
        "}\n"
        "\n");
    if (rc < 0) return rc;

    rc = hx_appendf(out, cap, len,
        "// hx: helper usp_md4_buf_global_metal -- MD4 of device candidate.\n"
        "static inline void usp_md4_buf_global_metal(device const uchar *data, int len,\n"
        "                                            thread uint *h0, thread uint *h1,\n"
        "                                            thread uint *h2, thread uint *h3)\n"
        "{\n"
        "    uint M[16];\n"
        "    int pos = 0;\n"
        "    uint st[4] = { 0x67452301u, 0xEFCDAB89u, 0x98BADCFEu, 0x10325476u };\n"
        "    while (len - pos >= 64) {\n"
        "        for (int j = 0; j < 16; j++) {\n"
        "            int q = pos + j * 4;\n"
        "            M[j] = (uint)data[q] | ((uint)data[q+1] << 8)\n"
        "                 | ((uint)data[q+2] << 16) | ((uint)data[q+3] << 24);\n"
        "        }\n"
        "        md4_block(st, M); pos += 64;\n"
        "    }\n"
        "    int rem = len - pos;\n"
        "    for (int j = 0; j < 16; j++) M[j] = 0;\n"
        "    for (int i = 0; i < rem; i++) M[i >> 2] |= (uint)data[pos+i] << ((i & 3) * 8);\n"
        "    M[rem >> 2] |= (uint)0x80 << ((rem & 3) * 8);\n"
        "    ulong bits = (ulong)len * 8ul;\n"
        "    if (rem < 56) {\n"
        "        M[14] = (uint)(bits & 0xfffffffful); M[15] = (uint)(bits >> 32);\n"
        "        md4_block(st, M);\n"
        "    } else {\n"
        "        md4_block(st, M);\n"
        "        for (int j = 0; j < 16; j++) M[j] = 0;\n"
        "        M[14] = (uint)(bits & 0xfffffffful); M[15] = (uint)(bits >> 32);\n"
        "        md4_block(st, M);\n"
        "    }\n"
        "    *h0 = st[0]; *h1 = st[1]; *h2 = st[2]; *h3 = st[3];\n"
        "}\n"
        "\n");
    if (rc < 0) return rc;

    rc = hx_appendf(out, cap, len,
        "// hx: helper usp_sha1_buf_global_metal -- SHA1 of device candidate.\n"
        "// BE schedule; first 4 state words byte-swapped to LE for probe.\n"
        "static inline void usp_sha1_buf_global_metal(device const uchar *data, int len,\n"
        "                                             thread uint *h0, thread uint *h1,\n"
        "                                             thread uint *h2, thread uint *h3)\n"
        "{\n"
        "    uint M[16];\n"
        "    int pos = 0;\n"
        "    uint st[5] = { 0x67452301u, 0xEFCDAB89u, 0x98BADCFEu,\n"
        "                   0x10325476u, 0xC3D2E1F0u };\n"
        "    while (len - pos >= 64) {\n"
        "        for (int j = 0; j < 16; j++) {\n"
        "            int q = pos + j * 4;\n"
        "            M[j] = ((uint)data[q] << 24) | ((uint)data[q+1] << 16)\n"
        "                 | ((uint)data[q+2] << 8) | (uint)data[q+3];\n"
        "        }\n"
        "        sha1_block(st, M); pos += 64;\n"
        "    }\n"
        "    int rem = len - pos;\n"
        "    uchar blk[64];\n"
        "    for (int i = 0; i < 64; i++) blk[i] = 0;\n"
        "    for (int i = 0; i < rem; i++) blk[i] = data[pos+i];\n"
        "    blk[rem] = 0x80;\n"
        "    ulong bits = (ulong)len * 8ul;\n"
        "    if (rem >= 56) {\n"
        "        for (int j = 0; j < 16; j++)\n"
        "            M[j] = ((uint)blk[j*4] << 24) | ((uint)blk[j*4+1] << 16)\n"
        "                 | ((uint)blk[j*4+2] << 8) | (uint)blk[j*4+3];\n"
        "        sha1_block(st, M);\n"
        "        for (int i = 0; i < 64; i++) blk[i] = 0;\n"
        "    }\n"
        "    for (int i = 0; i < 8; i++) blk[56+i] = (uchar)((bits >> (56 - i*8)) & 0xffu);\n"
        "    for (int j = 0; j < 16; j++)\n"
        "        M[j] = ((uint)blk[j*4] << 24) | ((uint)blk[j*4+1] << 16)\n"
        "             | ((uint)blk[j*4+2] << 8) | (uint)blk[j*4+3];\n"
        "    sha1_block(st, M);\n"
        "    uint sw[4];\n"
        "    for (int s = 0; s < 4; s++) {\n"
        "        uint v = st[s];\n"
        "        sw[s] = ((v & 0x000000ffu) << 24) | ((v & 0x0000ff00u) << 8)\n"
        "              | ((v & 0x00ff0000u) >> 8) | ((v & 0xff000000u) >> 24);\n"
        "    }\n"
        "    *h0 = sw[0]; *h1 = sw[1]; *h2 = sw[2]; *h3 = sw[3];\n"
        "}\n"
        "\n");
    if (rc < 0) return rc;

    rc = hx_appendf(out, cap, len,
        "// hx: helper usp_sha256_buf_global_metal -- SHA256 of device candidate.\n"
        "static inline void usp_sha256_buf_global_metal(device const uchar *data, int len,\n"
        "                                               thread uint *h0, thread uint *h1,\n"
        "                                               thread uint *h2, thread uint *h3)\n"
        "{\n"
        "    uint M[16];\n"
        "    int pos = 0;\n"
        "    uint st[8] = { 0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,\n"
        "                   0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u };\n"
        "    while (len - pos >= 64) {\n"
        "        for (int j = 0; j < 16; j++) {\n"
        "            int q = pos + j * 4;\n"
        "            M[j] = ((uint)data[q] << 24) | ((uint)data[q+1] << 16)\n"
        "                 | ((uint)data[q+2] << 8) | (uint)data[q+3];\n"
        "        }\n"
        "        sha256_block(st, M); pos += 64;\n"
        "    }\n"
        "    int rem = len - pos;\n"
        "    uchar blk[64];\n"
        "    for (int i = 0; i < 64; i++) blk[i] = 0;\n"
        "    for (int i = 0; i < rem; i++) blk[i] = data[pos+i];\n"
        "    blk[rem] = 0x80;\n"
        "    ulong bits = (ulong)len * 8ul;\n"
        "    if (rem >= 56) {\n"
        "        for (int j = 0; j < 16; j++)\n"
        "            M[j] = ((uint)blk[j*4] << 24) | ((uint)blk[j*4+1] << 16)\n"
        "                 | ((uint)blk[j*4+2] << 8) | (uint)blk[j*4+3];\n"
        "        sha256_block(st, M);\n"
        "        for (int i = 0; i < 64; i++) blk[i] = 0;\n"
        "    }\n"
        "    for (int i = 0; i < 8; i++) blk[56+i] = (uchar)((bits >> (56 - i*8)) & 0xffu);\n"
        "    for (int j = 0; j < 16; j++)\n"
        "        M[j] = ((uint)blk[j*4] << 24) | ((uint)blk[j*4+1] << 16)\n"
        "             | ((uint)blk[j*4+2] << 8) | (uint)blk[j*4+3];\n"
        "    sha256_block(st, M);\n"
        "    uint sw[4];\n"
        "    for (int s = 0; s < 4; s++) {\n"
        "        uint v = st[s];\n"
        "        sw[s] = ((v & 0x000000ffu) << 24) | ((v & 0x0000ff00u) << 8)\n"
        "              | ((v & 0x00ff0000u) >> 8) | ((v & 0xff000000u) >> 24);\n"
        "    }\n"
        "    *h0 = sw[0]; *h1 = sw[1]; *h2 = sw[2]; *h3 = sw[3];\n"
        "}\n"
        "\n");
    if (rc < 0) return rc;

    /* ============================================================
     * Iter v1 (2026-05-31). Metal twin of OpenCL iter-feed helpers.
     * See codegen/hx_emit_opencl.c for spec rationale. Token deltas:
     *   __global   ->  device
     *   uchar      ->  uchar  (Metal also uses uchar)
     *   md5_block  ->  md5_block  (Metal uses thread uint &h0..h3
     *                              by reference; we pass with addr
     *                              taken to match Pattern 1)
     *
     * Apple Metal benefits from `inline` (opposite Pascal noinline);
     * helpers are `static inline`.
     * ============================================================ */
    rc = hx_appendf(out, cap, len,
        "// hx iter v1 [Metal]: MD5 hex32-feed (LE schedule, fresh IV).\n"
        "static inline void usp_md5_iter_hex32_feed_metal(thread uint *h0,\n"
        "                                                 thread uint *h1,\n"
        "                                                 thread uint *h2,\n"
        "                                                 thread uint *h3)\n"
        "{\n"
        "    uint M[16];\n"
        "    md5_to_hex_lc(*h0, *h1, *h2, *h3, M);\n"
        "    M[8] = 0x80u;\n"
        "    for (int j = 9; j < 14; j++) M[j] = 0u;\n"
        "    M[14] = 32u * 8u;\n"
        "    M[15] = 0u;\n"
        "    *h0 = 0x67452301u; *h1 = 0xEFCDAB89u;\n"
        "    *h2 = 0x98BADCFEu; *h3 = 0x10325476u;\n"
        "    md5_block(*h0, *h1, *h2, *h3, M);\n"
        "}\n"
        "\n");
    if (rc < 0) return rc;

    rc = hx_appendf(out, cap, len,
        "// hx iter v1 [Metal]: MD4 hex32-feed. md4_block uses pointer-state\n"
        "// (thread uint *state, thread const uint *M) per metal_common.metal.\n"
        "static inline void usp_md4_iter_hex32_feed_metal(thread uint *h0,\n"
        "                                                 thread uint *h1,\n"
        "                                                 thread uint *h2,\n"
        "                                                 thread uint *h3)\n"
        "{\n"
        "    uint M[16];\n"
        "    md5_to_hex_lc(*h0, *h1, *h2, *h3, M);\n"
        "    M[8] = 0x80u;\n"
        "    for (int j = 9; j < 14; j++) M[j] = 0u;\n"
        "    M[14] = 32u * 8u;\n"
        "    M[15] = 0u;\n"
        "    uint st[4];\n"
        "    st[0] = 0x67452301u; st[1] = 0xEFCDAB89u;\n"
        "    st[2] = 0x98BADCFEu; st[3] = 0x10325476u;\n"
        "    md4_block(st, M);\n"
        "    *h0 = st[0]; *h1 = st[1]; *h2 = st[2]; *h3 = st[3];\n"
        "}\n"
        "\n");
    if (rc < 0) return rc;

    rc = hx_appendf(out, cap, len,
        "// hx iter v1 [Metal]: SHA1 hex40-feed (BE schedule, fresh IV).\n"
        "static inline void usp_sha1_iter_hex40_feed_metal(thread uint *st)\n"
        "{\n"
        "    uchar hex[40];\n"
        "    for (int s = 0; s < 5; s++) {\n"
        "        uint v = st[s];\n"
        "        uchar bs[4];\n"
        "        bs[0] = (uchar)((v >> 24) & 0xffu);\n"
        "        bs[1] = (uchar)((v >> 16) & 0xffu);\n"
        "        bs[2] = (uchar)((v >>  8) & 0xffu);\n"
        "        bs[3] = (uchar)( v        & 0xffu);\n"
        "        for (int k = 0; k < 4; k++) {\n"
        "            uchar b = bs[k];\n"
        "            uchar hi = (b >> 4) & 0xfu;\n"
        "            uchar lo = b & 0xfu;\n"
        "            hex[s*8 + k*2 + 0] = (uchar)(hi + ((hi < 10u) ? '0' : ('a' - 10)));\n"
        "            hex[s*8 + k*2 + 1] = (uchar)(lo + ((lo < 10u) ? '0' : ('a' - 10)));\n"
        "        }\n"
        "    }\n"
        "    uint M[16];\n"
        "    for (int j = 0; j < 10; j++) {\n"
        "        M[j] = ((uint)hex[j*4] << 24) | ((uint)hex[j*4+1] << 16)\n"
        "             | ((uint)hex[j*4+2] << 8) | (uint)hex[j*4+3];\n"
        "    }\n"
        "    M[10] = 0x80000000u;\n"
        "    for (int j = 11; j < 14; j++) M[j] = 0u;\n"
        "    M[14] = 0u;\n"
        "    M[15] = 320u;\n"
        "    st[0] = 0x67452301u; st[1] = 0xEFCDAB89u; st[2] = 0x98BADCFEu;\n"
        "    st[3] = 0x10325476u; st[4] = 0xC3D2E1F0u;\n"
        "    sha1_block(st, M);\n"
        "}\n"
        "\n");
    if (rc < 0) return rc;

    rc = hx_appendf(out, cap, len,
        "// hx iter v1 [Metal]: SHA256 hex64-feed. Two blocks (64+1+8 > 64).\n"
        "static inline void usp_sha256_iter_hex64_feed_metal(thread uint *st)\n"
        "{\n"
        "    uchar hex[64];\n"
        "    for (int s = 0; s < 8; s++) {\n"
        "        uint v = st[s];\n"
        "        uchar bs[4];\n"
        "        bs[0] = (uchar)((v >> 24) & 0xffu);\n"
        "        bs[1] = (uchar)((v >> 16) & 0xffu);\n"
        "        bs[2] = (uchar)((v >>  8) & 0xffu);\n"
        "        bs[3] = (uchar)( v        & 0xffu);\n"
        "        for (int k = 0; k < 4; k++) {\n"
        "            uchar b = bs[k];\n"
        "            uchar hi = (b >> 4) & 0xfu;\n"
        "            uchar lo = b & 0xfu;\n"
        "            hex[s*8 + k*2 + 0] = (uchar)(hi + ((hi < 10u) ? '0' : ('a' - 10)));\n"
        "            hex[s*8 + k*2 + 1] = (uchar)(lo + ((lo < 10u) ? '0' : ('a' - 10)));\n"
        "        }\n"
        "    }\n"
        "    uint M[16];\n"
        "    for (int j = 0; j < 16; j++) {\n"
        "        M[j] = ((uint)hex[j*4] << 24) | ((uint)hex[j*4+1] << 16)\n"
        "             | ((uint)hex[j*4+2] << 8) | (uint)hex[j*4+3];\n"
        "    }\n"
        "    st[0] = 0x6a09e667u; st[1] = 0xbb67ae85u; st[2] = 0x3c6ef372u; st[3] = 0xa54ff53au;\n"
        "    st[4] = 0x510e527fu; st[5] = 0x9b05688cu; st[6] = 0x1f83d9abu; st[7] = 0x5be0cd19u;\n"
        "    sha256_block(st, M);\n"
        "    M[0] = 0x80000000u;\n"
        "    for (int j = 1; j < 14; j++) M[j] = 0u;\n"
        "    M[14] = 0u;\n"
        "    M[15] = 512u;\n"
        "    sha256_block(st, M);\n"
        "}\n"
        "\n");
    return rc;
}

static int emit_unsalted_single_kernel_metal(char **out, size_t *cap, size_t *len,
                                              enum hx_primitive_id pid,
                                              const char *prim_name, int job_enum)
{
    /* Iter v1 (2026-05-31). Mirrors OpenCL twin. SHA1/SHA256 keep full
     * state alongside the LE-swapped probe key for iter feedback. */
    const char *seed_line;
    const char *probe_load;
    const char *feed_line;
    switch (pid) {
        case HX_PRIM_MD5:
            seed_line =
                "    uint h0, h1, h2, h3;\n"
                "    usp_md5_buf_global_metal(pass_bytes, (int)plen, &h0, &h1, &h2, &h3);\n";
            probe_load = "    /* h0..h3 are already LE probe key */\n";
            feed_line =
                "            usp_md5_iter_hex32_feed_metal(&h0, &h1, &h2, &h3);\n";
            break;
        case HX_PRIM_MD4:
            seed_line =
                "    uint h0, h1, h2, h3;\n"
                "    usp_md4_buf_global_metal(pass_bytes, (int)plen, &h0, &h1, &h2, &h3);\n";
            probe_load = "    /* h0..h3 are already LE probe key */\n";
            feed_line =
                "            usp_md4_iter_hex32_feed_metal(&h0, &h1, &h2, &h3);\n";
            break;
        case HX_PRIM_SHA1:
            seed_line =
                "    uint st[5];\n"
                "    uint h0, h1, h2, h3;\n"
                "    {\n"
                "        uint M[16];\n"
                "        int pos = 0;\n"
                "        st[0] = 0x67452301u; st[1] = 0xEFCDAB89u; st[2] = 0x98BADCFEu;\n"
                "        st[3] = 0x10325476u; st[4] = 0xC3D2E1F0u;\n"
                "        while ((int)plen - pos >= 64) {\n"
                "            for (int j = 0; j < 16; j++) {\n"
                "                int b = pos + j * 4;\n"
                "                M[j] = ((uint)pass_bytes[b] << 24) | ((uint)pass_bytes[b+1] << 16)\n"
                "                     | ((uint)pass_bytes[b+2] << 8) | (uint)pass_bytes[b+3];\n"
                "            }\n"
                "            sha1_block(st, M); pos += 64;\n"
                "        }\n"
                "        int rem = (int)plen - pos;\n"
                "        uchar blk[64]; for (int i = 0; i < 64; i++) blk[i] = 0;\n"
                "        for (int i = 0; i < rem; i++) blk[i] = pass_bytes[pos + i];\n"
                "        blk[rem] = 0x80;\n"
                "        ulong bits = (ulong)plen * 8ul;\n"
                "        if (rem >= 56) {\n"
                "            for (int j = 0; j < 16; j++)\n"
                "                M[j] = ((uint)blk[j*4] << 24) | ((uint)blk[j*4+1] << 16)\n"
                "                     | ((uint)blk[j*4+2] << 8) | (uint)blk[j*4+3];\n"
                "            sha1_block(st, M);\n"
                "            for (int i = 0; i < 64; i++) blk[i] = 0;\n"
                "        }\n"
                "        for (int i = 0; i < 8; i++) blk[56 + i] = (uchar)((bits >> (56 - i*8)) & 0xffu);\n"
                "        for (int j = 0; j < 16; j++)\n"
                "            M[j] = ((uint)blk[j*4] << 24) | ((uint)blk[j*4+1] << 16)\n"
                "                 | ((uint)blk[j*4+2] << 8) | (uint)blk[j*4+3];\n"
                "        sha1_block(st, M);\n"
                "    }\n";
            probe_load =
                "        h0 = ((st[0] & 0x000000ffu) << 24) | ((st[0] & 0x0000ff00u) << 8)\n"
                "           | ((st[0] & 0x00ff0000u) >> 8) | ((st[0] & 0xff000000u) >> 24);\n"
                "        h1 = ((st[1] & 0x000000ffu) << 24) | ((st[1] & 0x0000ff00u) << 8)\n"
                "           | ((st[1] & 0x00ff0000u) >> 8) | ((st[1] & 0xff000000u) >> 24);\n"
                "        h2 = ((st[2] & 0x000000ffu) << 24) | ((st[2] & 0x0000ff00u) << 8)\n"
                "           | ((st[2] & 0x00ff0000u) >> 8) | ((st[2] & 0xff000000u) >> 24);\n"
                "        h3 = ((st[3] & 0x000000ffu) << 24) | ((st[3] & 0x0000ff00u) << 8)\n"
                "           | ((st[3] & 0x00ff0000u) >> 8) | ((st[3] & 0xff000000u) >> 24);\n";
            feed_line =
                "            usp_sha1_iter_hex40_feed_metal(st);\n";
            break;
        case HX_PRIM_SHA256:
            seed_line =
                "    uint st[8];\n"
                "    uint h0, h1, h2, h3;\n"
                "    {\n"
                "        uint M[16];\n"
                "        int pos = 0;\n"
                "        st[0] = 0x6a09e667u; st[1] = 0xbb67ae85u; st[2] = 0x3c6ef372u; st[3] = 0xa54ff53au;\n"
                "        st[4] = 0x510e527fu; st[5] = 0x9b05688cu; st[6] = 0x1f83d9abu; st[7] = 0x5be0cd19u;\n"
                "        while ((int)plen - pos >= 64) {\n"
                "            for (int j = 0; j < 16; j++) {\n"
                "                int b = pos + j * 4;\n"
                "                M[j] = ((uint)pass_bytes[b] << 24) | ((uint)pass_bytes[b+1] << 16)\n"
                "                     | ((uint)pass_bytes[b+2] << 8) | (uint)pass_bytes[b+3];\n"
                "            }\n"
                "            sha256_block(st, M); pos += 64;\n"
                "        }\n"
                "        int rem = (int)plen - pos;\n"
                "        uchar blk[64]; for (int i = 0; i < 64; i++) blk[i] = 0;\n"
                "        for (int i = 0; i < rem; i++) blk[i] = pass_bytes[pos + i];\n"
                "        blk[rem] = 0x80;\n"
                "        ulong bits = (ulong)plen * 8ul;\n"
                "        if (rem >= 56) {\n"
                "            for (int j = 0; j < 16; j++)\n"
                "                M[j] = ((uint)blk[j*4] << 24) | ((uint)blk[j*4+1] << 16)\n"
                "                     | ((uint)blk[j*4+2] << 8) | (uint)blk[j*4+3];\n"
                "            sha256_block(st, M);\n"
                "            for (int i = 0; i < 64; i++) blk[i] = 0;\n"
                "        }\n"
                "        for (int i = 0; i < 8; i++) blk[56 + i] = (uchar)((bits >> (56 - i*8)) & 0xffu);\n"
                "        for (int j = 0; j < 16; j++)\n"
                "            M[j] = ((uint)blk[j*4] << 24) | ((uint)blk[j*4+1] << 16)\n"
                "                 | ((uint)blk[j*4+2] << 8) | (uint)blk[j*4+3];\n"
                "        sha256_block(st, M);\n"
                "    }\n";
            probe_load =
                "        h0 = ((st[0] & 0x000000ffu) << 24) | ((st[0] & 0x0000ff00u) << 8)\n"
                "           | ((st[0] & 0x00ff0000u) >> 8) | ((st[0] & 0xff000000u) >> 24);\n"
                "        h1 = ((st[1] & 0x000000ffu) << 24) | ((st[1] & 0x0000ff00u) << 8)\n"
                "           | ((st[1] & 0x00ff0000u) >> 8) | ((st[1] & 0xff000000u) >> 24);\n"
                "        h2 = ((st[2] & 0x000000ffu) << 24) | ((st[2] & 0x0000ff00u) << 8)\n"
                "           | ((st[2] & 0x00ff0000u) >> 8) | ((st[2] & 0xff000000u) >> 24);\n"
                "        h3 = ((st[3] & 0x000000ffu) << 24) | ((st[3] & 0x0000ff00u) << 8)\n"
                "           | ((st[3] & 0x00ff0000u) >> 8) | ((st[3] & 0xff000000u) >> 24);\n";
            feed_line =
                "            usp_sha256_iter_hex64_feed_metal(st);\n";
            break;
        default:
            fprintf(stderr,
                "FATAL: %s:%d hx unsalted-single Metal emit kernel: primitive "
                "'%s' (id=%d) not wired in Phase 1b Batch 1 (job=e%d).\n",
                __FILE__, __LINE__, prim_name ? prim_name : "(null)",
                (int)pid, job_enum);
            exit(1);
    }

    return hx_appendf(out, cap, len,
        "// hx: unsalted-single Metal kernel for e%d prim=%s; probe uses 4 LE uints.\n"
        "// Signature mirrors kernelb_hx_codegen_phase0 (family Metal); salt\n"
        "// args ignored. ovr_set/ovr_gid explicit atomic_uint buffers 16/17.\n"
        "//\n"
        "// Iter v1 (2026-05-31): runtime for-loop reading params.max_iter\n"
        "// (OCLParams offset 60). Same shape as the OpenCL twin. Mirrors\n"
        "// legacy md5_rules_phase0 iter pattern byte-for-byte.\n"
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
        "    device const OCLParams *params_buf =\n"
        "        (device const OCLParams *)payload;\n"
        "    OCLParams params = *params_buf;\n"
        "\n"
        "    uint word_idx = gid;\n"
        "    if (word_idx >= params.num_words) return;\n"
        "\n"
        "    (void)salts; (void)salt_offsets; (void)salt_lens;\n"
        "\n"
        "    uint wpos = b_chunk_index[word_idx];\n"
        "    if (wpos >= params.packed_size) return;  // defensive\n"
        "    uint plen = (uint)b_packed_buf[wpos];\n"
        "    device const uchar *pass_bytes = b_packed_buf + wpos + 1u;\n"
        "\n"
        "    // OP_CALL %s (seed: single hash of the unsalted pass; iter==1)\n"
        "%s"
        "\n"
        "    uint widx = params.base_word_idx + word_idx;\n"
        "    uint mi = params.max_iter; if (mi < 1u) mi = 1u;\n"
        "    for (uint iter = 1u; iter <= mi; iter++) {\n"
        "%s"
        "        uint matched_idx = 0u;\n"
        "        if (probe_compact_idx(h0, h1, h2, h3,\n"
        "                              compact_fp, compact_idx,\n"
        "                              params.compact_mask, params.max_probe,\n"
        "                              params.hash_data_count,\n"
        "                              hash_data_buf, hash_data_off,\n"
        "                              overflow_keys, overflow_hashes,\n"
        "                              overflow_offsets, params.overflow_count,\n"
        "                              &matched_idx))\n"
        "        {\n"
        "            uint mask = 1u << (iter & 31u);\n"
        "            EMIT_HIT_4_DEDUP_OR_OVERFLOW(hits, hit_count, params.max_hits,\n"
        "                       widx, 0u, iter, h0, h1, h2, h3,\n"
        "                       hashes_shown, matched_idx, mask,\n"
        "                       ovr_set, ovr_gid, gid);\n"
        "        }\n"
        "        if (iter < mi) {\n"
        "%s"
        "        }\n"
        "    }\n"
        "}\n",
        job_enum, prim_name, prim_name, seed_line,
        probe_load, feed_line);
}

int hx_emit_unsalted_single_metal(
    char **out, size_t *out_cap,
    const hx_program *prog,
    const struct hx_specialization *spec,
    const struct hx_spec_entry *entry)
{
    if (!out || !out_cap || !prog || !spec || !entry) {
        fprintf(stderr,
            "FATAL: %s:%d hx_emit_unsalted_single_metal: NULL argument "
            "(out=%p cap=%p prog=%p spec=%p entry=%p)\n",
            __FILE__, __LINE__, (void*)out, (void*)out_cap,
            (void*)prog, (void*)spec, (void*)entry);
        return -1;
    }

    const char *prim_name = hx_callname_for_entry(entry, 1);
    if (!prim_name) {
        fprintf(stderr,
            "FATAL: %s:%d hx_emit_unsalted_single_metal: e%d %s code[1] "
            "callname is NULL (sidecar missing).\n",
            __FILE__, __LINE__, entry->job_enum,
            entry->name ? entry->name : "(noname)");
        return -1;
    }
    enum hx_primitive_id pid = hx_primitive_id_for_name(prim_name);
    if (pid == HX_PRIM_UNKNOWN) {
        fprintf(stderr,
            "FATAL: %s:%d hx_emit_unsalted_single_metal: e%d %s callname "
            "'%s' not recognized in hx_emit_primitives.c table.\n",
            __FILE__, __LINE__, entry->job_enum,
            entry->name ? entry->name : "(noname)", prim_name);
        return -1;
    }
    if (pid != HX_PRIM_MD5 && pid != HX_PRIM_MD4 &&
        pid != HX_PRIM_SHA1 && pid != HX_PRIM_SHA256) {
        fprintf(stderr,
            "FATAL: %s:%d hx_emit_unsalted_single_metal: e%d %s primitive "
            "'%s' not in Phase 1b Batch-1 wired set (md5/md4/sha1/sha256).\n",
            __FILE__, __LINE__, entry->job_enum,
            entry->name ? entry->name : "(noname)", prim_name);
        return -1;
    }

    size_t cur_len = 0;
    if (*out == NULL) *out_cap = 0;
    int rc;

    rc = hx_appendf(out, out_cap, &cur_len,
        "// hx codegen: PATTERN UNSALTED_SINGLE matched (e%d %s prim=%s) [Metal]\n"
        "// hx: program ncode=%d nvars=%d code[1] role=%d (0=hex,1=raw; same digest)\n"
        "// hx: JIT-compiled with metal_common_str prepended\n"
        "\n",
        entry->job_enum, entry->name ? entry->name : "(noname)", prim_name,
        prog->ncode, prog->nvars, (int)prog->code[1].u.call.role);
    if (rc < 0) return rc;

    rc = emit_unsalted_single_helpers_metal(out, out_cap, &cur_len);
    if (rc < 0) return rc;

    rc = emit_unsalted_single_kernel_metal(out, out_cap, &cur_len,
                                           pid, prim_name, entry->job_enum);
    if (rc < 0) return rc;

    if (cur_len + 1 > *out_cap) {
        char *np = (char *)realloc(*out, cur_len + 1);
        if (!np) return -1;
        *out = np;
        *out_cap = cur_len + 1;
    }
    (*out)[cur_len] = '\0';
    return 0;
}
