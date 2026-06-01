/*
 * hx_emit_opencl.c -- OpenCL per-backend emit helpers for the hx P4
 *                     state-machine walker.
 *
 * Sub-phase 2a.2 adds per-opcode emit helpers for the minimum-viable
 * opcode set (PUSH_VAR/PUSH_STR/PUSH_INT/STORE/CALL[md5]/CONCAT/HALT).
 * Bodies are intentionally minimal -- they emit a placeholder C
 * declaration sufficient for the JIT compiler to accept the file plus
 * an annotation comment recording the opcode and operand. Real
 * semantics (tp0-pattern emission for e347) arrive in 2a.3.
 *
 * All comments emitted into the kernel source use "//" only; never
 * the slash-star form -- per feedback_no_nested_block_comments_in_cl.md.
 *
 * Sub-phase 2a.3 (2026-05-21): adds hx_emit_e347_md5md5md5salt_opencl
 * which emits a full self-contained tp0-pattern OpenCL kernel B for
 * the e347 MD5MD5SALT bytecode shape. Per-thread serial SALT_BATCH=64
 * outer loop; pre-state held in 4 registers across the loop; hex32
 * expansion between inner MD5s; helper functions (md5_buf_global,
 * md5_buf_private16, state_to_le_bytes16, hex32_into_M, md5_outer_hex_-
 * combine) inline in the emitted source. Calls md5_block, OCLParams,
 * EMIT_HIT_4_DEDUP_OR_OVERFLOW, probe_compact_idx from gpu_common.cl
 * which is prepended at JIT time via gpu_opencl_jit_compile_source_-
 * with_common.
 *
 * Sub-phase 4a.1 (2026-05-21) Part A: emit reqd_work_group_size(64,1,1)
 * attribute on the e347 kernel declaration. The production dispatcher in
 * gpu_opencl_kernelb_dispatch_proto passes lsize=64; without the attribute
 * Pascal/NVIDIA driver may select a different WG size and the salt-chunk
 * topology breaks. Risk R8 in project_hx_codegen_phase4_dispatcher_spec_-
 * 2026-05-21.md. Hand-written kernel B (gpu_kernelb_md5md5salt_nocache.cl
 * rev 1.4) already carries this attribute; codegen now matches.
 *
 * Sub-phase 5a.2 (2026-05-22): MAKE_MD5PASS family emitter
 * `hx_emit_family_md5pass_opencl()` lands. Composes:
 *   - md5_buf_global helper (reused verbatim from e347 emitter)
 *   - state_to_hex32_bytes helper (reused verbatim from e347 emitter)
 *   - per-primitive outer body (5a.2 ships SHA1 only via
 *     emit_outer_sha1_concat_then_hash; md4/md5/sha224/sha256/sha384/
 *     sha512/rmd160 are 5a.4 scope and FATAL with a "deferred to 5a.4"
 *     diagnostic).
 *
 * Algorithm semantics for e161 SHA1MD5PASS (per mdxfind CPU
 * JOB_SHA1MD5PASS at mdxfind.c:27272 + SHA1_start label):
 *   digest = SHA1( hex32( MD5(pass) ) || pass )
 * Input length to SHA1 is 32 + plen bytes. 20-byte digest output;
 * compact_fp probe still uses the first 4 uints (h0..h3) which is the
 * existing EMIT_HIT_4_DEDUP_OR_OVERFLOW contract -- SHA1's h4 is unused
 * for the per-thread probe; round-trip validation reads h0..h3 only.
 * This matches mdxfind's existing behavior (CompactFP+CompactIdx is
 * 64-bit fingerprint regardless of digest width).
 *
 * Kernel signature is byte-identical to e347's 16-arg layout so the
 * production dispatcher (5a.5) can bind the same way. The 4 salt-table
 * args (salts/salt_offsets/salt_lens, plus payload->num_salts) are
 * IGNORED by the family kernel (unsalted family) -- harness binds them
 * to small zero buffers; production dispatcher binds the device's
 * existing salt buffers since they're already provisioned.
 *
 * $Revision: 1.21 $
 * $Log: hx_emit_opencl.c,v $
 * Revision 1.21  2026/05/31 14:08:21  dlr
 * Codegen kernel B iteration v1 (-i N>1) paired OpenCL + Metal per spec D-defaults all .a. (1) Runtime iter via existing OCLParams.max_iter (offset 60, zero ABI change). (2) Per-primitive iter-feed helpers for MD5+MD4+SHA1+SHA256 in codegen/hx_emit_opencl.c:4191-4480 + Metal twin codegen/hx_emit_metal.c:3829-4140. (3) Iteration loop wraps kernel B body with per-iter mask 1u << (iter & 31u). (4) Hex-encoded digest feedback mirrors legacy md5_rules_phase0 at gpu/gpu_md5_rules.cl:1158-1193 byte-exact. (5) OpenCL host drops iter==1 clause at gpu/gpujob_opencl.c:1172-1195; unhardcodes params.max_iter=1 at gpu/gpu_opencl.c:14071 + :13212. (6) Metal NEW capability: JOB_MD5 admission added to Metal codegen at gpu_metal.m:4814 (was salted-only); new accessor at :4077-4108; route gate at gpu/gpujob_metal.m:1248-1313. Validated 20-cell crack-parity Pascal+Maxwell+M1 byte-identical at iter in {1,2,5,10,100}; R1 hex-feedback verified via C-oracle harness BEFORE crack-parity. Production safety env UNSET unchanged. Gate C 99K rules x -i 10 x rockyou-1m x Pascal: legacy 305.42s vs codegen 425.56s = 1.39x slower (vs 1.46x at -i 1 — gap closes at -i 10). Gate D NEW: Metal -m e1 -i N>1 works correctly via codegen for first time (legacy template_iterate gap remained; codegen sidesteps). v1.1 follow-on: widen route gate for MD4/SHA1RAW/SHA256RAW admission. Spec project_codegen_iteration_v1_spec_2026-05-31.md.
 *
 * Revision 1.20  2026/05/28 14:32:03  dlr
 * Phase 1b Batch 1: add hx_emit_unsalted_single_opencl + hx_emit_unsalted_single_metal one-shot hash of pass emitters for HX_PATTERN_UNSALTED_SINGLE; reuse md5 md4 sha1 sha256 block from gpu_common.cl and metal_common.metal; strictly simpler than family no inner md5 no hex32 no concat; per-primitive usp buf-global helpers reproduce the family MD SHA padding applied to raw pass; SHA1 SHA256 BE to LE state byte-swap for the compact_fp probe; kernel signature mirrors kernelb_hx_codegen_phase0 salt args ignored; reqd work group size 64; C-mirror validated 80 of 80 byte-exact before GPU JIT; FATAL on callname not in wired set md5 md4 sha1 sha256
 *
 * Revision 1.19  2026/05/28 06:12:15  dlr
 * sub-phase 5c.2.1 OpenCL multi-emit family kernel body plus MD5-as-outer helper for e123 MD5MD5PASS the FIRST multi-emit member; new emit_outer_md5_concat_then_hash mirrors the MD4 helper LE schedule 4-uint state 16-byte digest plus a sep parameter; sep 0 canonical hex32 then pass sep 1 colon hex32 then colon-byte then pass shifts pass to logical position 33 total_len 33 plus plen; new emit_family_md5pass_kernel_multiemit emits compute md5 of pass once then N equals 2 unrolled probe-and-emit blocks one per variant each calling outer helper with its sep then probe_compact_idx then the EXISTING EMIT_HIT_4_DEDUP_OR_OVERFLOW macro unchanged dedup keys on per-variant matched_idx; emit_class threaded through hx_emit_family_md5pass_opencl plus emit_family_md5pass_kernel single-emit path untouched; replaced HX_PRIM_MD5 FATAL with emit_class gate MD5-outer admitted only when HX_EMIT_MULTI; wired MD5 into per-primitive emit dispatch plus FATAL filter; dumped kernel on fpga Pascal shows both sep 0 and sep 1 blocks JIT-compiles clean
 *
 * Revision 1.18  2026/05/28 04:49:21  dlr
 * 5b.4b.3: add bespoke emit_outer_gost_concat_then_hash OpenCL helper for GOST R 34.11-94 e125; walks hex32-bar-pass in 32-byte LE blocks accumulating sum8 mod-2^256 checksum + bit-length glen, dual finalization compress glen then sum, LE state output h0..h3 equals hash 0..3 no byteswap; wired 4 sites helper-name switch call-line tree FATAL filter widened to 29 dispatch switch; unified-loop control flow validated byte-exact vs gosthash donor 20 of 20 lengths
 *
 * Revision 1.17  2026/05/28 04:32:05  dlr
 * sub-phase 5b4a3 add parameterised emit_outer_snefru_concat_then_hash helper to hx_emit_opencl.c per D18.1.a D18.3.a ONE C-side helper emits a GPU function specialised on is256 plus digest_bytes covers both Snefru widths e175 SNE128 16-byte is256 0 e177 SNE256 32-byte is256 1 block-size asymmetry R-Tier4-snefru-blocksize SNE128 48-byte data blocks SNE256 32-byte data blocks DBLK equals 64 minus digest_bytes padding plus length-field byte offsets baked per-width length be2me_32 len shifted 29 at block DBLK minus 8 be2me_32 len shifted left 3 at block DBLK minus 4 len in BYTES Snefru IV all-zero 8 rounds fixed BE schedule plus BE state output bswap32 into LE-uint probe frame per feedback_be_state_primitives_need_byteswap_in_codegen CPU recompute fills SNE256 remaining 16 bytes on hit SNE128 exactly 16 bytes wired 4 OpenCL sites helper-name switch outer_snefru128 outer_snefru256 distinct names call-line tree 2 SNE branches FATAL gating filter widened to add sne128 sne256 wired subset emit dispatch switch routes SNE128 is256 0 SNE256 is256 1 to emit_outer_snefru_concat_then_hash C-mirror test_snefru_port 56 of 56 cells PASS vs librhash both widths byte-exact
 *
 * Revision 1.16  2026/05/28 03:52:49  dlr
 * sub-phase 5b3c3 wire 5 HAV*_5 enums into emit_outer_haval_concat_then_hash OpenCL dispatch added HX_PRIM_HAV128_5 HAV160_5 HAV192_5 HAV224_5 HAV256_5 to 4 sites helper-name switch fall-through group call-line tree terminal else comment updated to any 3-pass 4-pass or 5-pass FATAL gating filter widened to 26 wired primitives new dispatch switch group routes HAV*_5 to emit_outer_haval with passes 5 the parameterised helper bakes haval5_block call plus block 118 0x29 5-pass encoding automatically R13 verified in dumped e131 e155 kernels
 *
 * Revision 1.15  2026/05/28 03:19:46  dlr
 * sub-phase 5b3b3 wire 5 HAV*_4 enums into emit_outer_haval_concat_then_hash OpenCL dispatch added HX_PRIM_HAV128_4 HAV160_4 HAV192_4 HAV224_4 HAV256_4 to 3 sites helper-name switch fall-through group call-line tree terminal else comment updated to any 3-pass or 4-pass FATAL gating filter widened to 21 wired primitives new dispatch switch group routes HAV*_4 to emit_outer_haval with passes 4 the parameterised helper bakes haval4_block call plus block 118 0x21 4-pass encoding automatically R13 verified in dumped e129 e153 kernels
 *
 * Revision 1.14  2026/05/28 02:24:53  dlr
 * sub-phase 5b3a3 add parameterised emit_outer_haval_concat_then_hash helper to hx_emit_opencl c per D17.1.a ONE C-side helper emits a GPU function specialised on passes plus digest_bytes covers all 15 HAVAL variants 5b3a ships 5 3-pass via passes 3 fixed digest_bytes 16 20 24 28 32 emitted GPU function outer_haval_concat_then_hash 128-byte block 32 LE-packed uint32 words PAD-TOGGLE 0x01 NOT 0x80 cited donor mhash haval c 760 block 118 119 parameter encoding computed at C-emit time baked as literal constants R1 mitigation 64-bit bitlen LE block 120 127 post-compression digest fold JIT-specialised per width exactly ONE branch emitted no runtime conditional donor havalFinal 816-911 128-bit heavy byte-redistribution 160-bit ROTR fold 192-bit 5-bit-slice 224-bit byte-slot-shift 256-bit no fold direct output R3 mitigation HAVAL state LE-native h0 to h3 state 0 to 3 direct no byte-swap wired 4 OpenCL sites helper-name switch 5 HAV arms call-line tree HAVAL branch FATAL filter widened to 16 primitives emit dispatch switch 5 HAV arms route to emit_outer_haval with passes 3 digest_bytes from outer_digest_bytes C-mirror validated 60 of 60 cells PASS pre-port
 *
 * Revision 1.13  2026/05/27 23:07:04  dlr
 * sub-phase 5b2b3 add emit_outer_tiger_concat_then_hash bespoke per-primitive emit helper for Tiger outer hash structurally divergent from sha2_64 and wrl in 5 ways LE schedule M packed lo-byte-first 8-byte LE length suffix at M7 padding byte 0x01 legacy Tiger NOT Tiger2 0x80 state IV Tiger initial chaining value 0x0123456789abcdef 0xfedcba9876543210 0xf096a5b4c3b2e187 3-ulong state single-block fast path APPLICABLE for plen le 23 threshold 32 plus plen plus 1 plus 8 le 64 unlike Whirlpool which ALWAYS multi-blocks calls tiger_block from gpu_common.cl rev 1.29 LE state output direct extract no byte-swap epilogue h0..h3 from state 0..1 directly added HX_PRIM_TIGER case to helper-name switch outer_tiger_concat_then_hash TIGER branch to call-line tree helper_has_h4 0 TIGER to FATAL gating filter widened to md2 md4 rmd128 sha1 sha224 sha256 sha384 sha512 rmd160 wrl tiger 11 of 11 wired subset via 5a.4 plus 5b.1a plus 5b.1b plus 5b.2a plus 5b.2b TIGER case to emit dispatch switch routes outer_id TIGER to new emit_outer_tiger_concat_then_hash CPU translation test 7 NESSIE vectors plus 1M-a stress PASS byte-exact vs rhash_tiger and sph_tiger R12 pre-flight 16 of 16 cells PASS 2026-05-27 OpenCL twin only Metal twin in 5b2b3-metal
 *
 * Revision 1.12  2026/05/27 22:23:47  dlr
 * sub-phase 5b2a3 add emit_outer_wrl_concat_then_hash bespoke per-primitive emit helper for Whirlpool outer hash structurally divergent from sha2_64 in 4 ways block size 64 not 128 length suffix 32 bytes BE at M4 to M7 high 24 bytes always zero state IV all zero per Whirlpool spec ALWAYS multi-block single-block fast path elided 32 plus plen plus 1 plus 32 le 64 never holds calls wrl_block from gpu_common.cl rev 1.28 BE state byte-swap epilogue identical to emit_outer_sha2_64 first_has_pad logic mirrors sha512 helper added HX_PRIM_WRL case to helper-name switch outer_wrl_concat_then_hash WRL branch to call-line tree helper_has_h4 0 WRL to FATAL gating filter widened to md2 md4 rmd128 sha1 sha224 sha256 sha384 sha512 rmd160 wrl 10 of 10 wired subset via 5a.4 plus 5b.1a plus 5b.1b plus 5b.2a WRL case to emit dispatch switch routes outer_id WRL to new emit_outer_wrl_concat_then_hash CPU translation test 8 NESSIE vectors PASS byte-exact vs librhash and OpenSSL R12 pre-flight confirmed 2026-05-27 OpenCL twin only Metal twin in 5b2a3-metal
 *
 * Revision 1.11  2026/05/27 18:40:48  dlr
 * sub-phase 5b1b7 revert RIPEMD-128 length-field bug-compat workaround in emit_outer_rmd128_concat_then_hash now that the in-tree rmd128.c MDfinish length-encoding bug is fixed at rmd128.c rev 1.1. Removes bug_lswlen first_has_pad branch from both single-block tail branch and 2-block else branch. Both branches now use bitlen equals total_len times 8 unconditionally per Bosselaers 1996 reference and sph_ripemd128. After the fix mdxfind CPU oracle and GPU emit both produce standard-conformant RIPEMD-128 digests byte-exact across the full plen range. User confirmed 2026-05-27 no production solved-hash records affected by the standard-conformance flip because plen greater than 60 inputs were never exercised in archives for RIPEMD128-using catalog entries e132 e157 e162 e211 e231.
 *
 * Revision 1.10  2026/05/27 18:15:00  dlr
 * sub-phase 5b1b6 fix RIPEMD-128 length-field bug-compatibility in emit_outer_rmd128_concat_then_hash discovered during validation matrix at family_edge_maxlen fixture maxlen plens 59 to 131 OpenCL FAIL with vn_hits 0 missing 128 root cause in-tree rmd128.c MDfinish encodes leftover bytes after full-block processing as length field NOT total message length verified bug against sph_ripemd128 plus Bosselaers official test vectors for 80-char standard buggy CPU returns 1959258 standard correct is 3f45ef19 long-standing bug in donor implementation since 1996 mdxfind CPU oracle JOB_RMD128MD5PASS calls this buggy RIPEMD128 customer existing rmd128 hashes were computed with buggy code GPU MUST reproduce bug-compatible digest to stay in sync introduce bug_lswlen first_has_pad branch total_len pleft branch use bug_lswlen 8 instead of total_len 8 at final compress bitlen in both single-block tail branch and 2-block else branch fixes maxlen and broader plen 32 plus tail paths preserves correctness for fast path total_len le 55 and first_has_pad 1 path total_len le 63 because in those cases CPU lswlen equals total_len no behavioral change for already-passing fixtures
 *
 * Revision 1.9  2026/05/27 17:47:42  dlr
 * sub-phase 5b1b3 add emit_outer_rmd128_concat_then_hash bespoke clone of emit_outer_rmd160_concat_then_hash with 4-uint state width drop h4 drop state 4 drop e1 calls rmd128_block from gpu_common.cl rev 1.27 mirror of the rmd160 multi-block pattern same LE message-schedule packing same LE 64-bit length suffix same first_has_pad logic for boundary cases pl 23 fast path total_len plus 1 plus 8 le 64 single-block then multi-block tail with first_has_pad flag CPU oracle RIPEMD128 stores LE bytes harness reinterprets as 4 LE uints direct byte-exact match no byte-swap RMD-128 right-pipeline F4 F3 F2 F1 ordering is in rmd128_block primitive itself not in this emit helper added HX_PRIM_RMD128 case to helper-name switch RMD128 branch to call-line tree RMD128 to FATAL gating filter wired subset md2 md4 rmd128 sha1 sha224 sha256 sha384 sha512 rmd160 RMD128 case to emit dispatch switch routes outer_id RMD128 to new emit_outer_rmd128_concat_then_hash 9 of 9 5a-supported primitives now wired plus MD2 RMD128 11 of 11 supported primitives via 5a.4 plus 5b.1a plus 5b.1b OpenCL twin only Metal twin in 5b1b3-metal twin commit
 *
 * Revision 1.8  2026/05/27 17:01:02  dlr
 * sub-phase 5b1a3 add emit_outer_md2_concat_then_hash bespoke per-primitive emit helper for MD2 outer hash unlike MD4 family helpers MD2 has 16-byte block plus PKCS padding plus checksum-as-final-block structurally distinct calls md2_block from gpu_common.cl rev 1.26 per 16-byte data block processes 32-byte hex32 md5 inner as 2 full blocks then chunks pass in 16-byte blocks then PKCS-pads tail to 16-byte boundary then runs final checksum block with update_checksum equals 0 per RFC errata digest reads state 0 to 15 LE-packed into h0 h1 h2 h3 added MD2 arm to per-primitive switch in emit_family_md5pass_kernel helper_has_h4 0 added MD2 branch to call-line tree added MD2 to FATAL gating filter in hx_emit_family_md5pass_opencl added HX_PRIM_MD2 case to emit dispatch switch routes outer_id MD2 to new emit_outer_md2_concat_then_hash 8 of 8 5a-supported primitives now wired plus MD2 9 of 9 supported primitives via 5a.4 plus 5b.1a 5b1a3 OpenCL twin only Metal twin lifts in 5b1a3-metal twin commit
 *
 * Revision 1.7  2026/05/23 05:23:25  dlr
 * sub-phase 5a.4 fan out the 6 remaining 5a-supported MAKE_MD5PASS family primitives on OpenCL md4 rmd160 sha224 sha256 sha384 sha512 outer body emit helpers each mirrors emit_outer_sha1_concat_then_hash structure single-block fast path multi-block with first_has_pad tail length encoding per-primitive endianness handling md4 LE schedule no state byte-swap rmd160 LE schedule no swap sha224 sha256 BE schedule swap each of first 4 state uints sha384 sha512 BE schedule ulong state swap each of first 2 ulongs then split into LE uint pair shared SHA-2 32-bit body parametrized by IV emit_outer_sha2_32_concat_then_hash with SHA-224 IV vs SHA-256 IV shared SHA-2 64-bit body emit_outer_sha2_64_concat_then_hash with SHA-384 IV vs SHA-512 IV 128-byte block 16-byte length suffix BE first_has_pad logic widened from total_len plus 1 plus 8 to total_len plus 1 plus 16 emit_family_md5pass_kernel switch table replaces SHA1-only FATAL with 7-arm dispatch on outer_id naming the helper per primitive 5-uint state primitives sha1 rmd160 take h4 4-uint state primitives md4 sha224 sha256 sha384 sha512 take only h0 through h3 hx_emit_family_md5pass_opencl widened the per-primitive emit body switch dispatch to 7 wired primitives with FATAL on HX_PRIM_MD5 e123 outlier deferred for multi-emit cross-arch validated PASS 8 of 8 on NVIDIA GTX 1080 Pascal for e122 e159 e163 e165 e167 e169 e161 still PASS as 5a.3 regression e347 production regression still PASS as well
 *
 * Revision 1.6  2026/05/23 02:02:46  dlr
 * sub-phase 5a.2 add hx_emit_family_md5pass_opencl entry point for MAKE_MD5PASS family emitter with SHA1 outer (e161) per-primitive body validates inner CALL is md5 and outer is in 5a-supported set FATAL on UNKNOWN or 5b-deferred; emits shared helpers md5_buf_global state_to_hex32_bytes plus per-primitive outer_sha1_concat_then_hash body single-block fast path for plen le 23 and multi-block tail with first_has_pad pattern mirroring e347 padding fix; SHA1 BE-to-LE byte-swap on state output so emitted h0 to h4 uints match CPU bytewise oracle reinterpreted as LE uints byte-exact validated 8 of 8 on Pascal GTX 1080
 *
 * Revision 1.5  2026/05/22 05:56:20  dlr
 * *** empty log message ***
 *
 * Revision 1.4  2026/05/22 03:34:17  dlr
 * sub-phase 2a.5 byte-exact correctness fix for e347 codegen kernel B; the 2a.3 emission computed MD5 hex32 MD5 MD5 pass concat salt with only one hex32 inner expansion mirroring the drift documented in feedback handwritten kernel b drift md5md5salt; the correct hx CONCAT semantics for digest-fed input requires hex32 at every digest-to-md5 boundary so the chain is MD5 hex32 MD5 hex32 MD5 pass concat salt with two hex32 inner expansions matching mdxfind CPU JOB_MD5MD5SALT at mdxfind.c line 23174; replaced state_to_le_bytes16 helper with state_to_hex32_bytes and md5_buf_private16 with md5_buf_private32 and updated kernel body to feed inner_hex 32 bytes into the second inner MD5; also fixed multi-block padding bug in md5_outer_hex_combine where salt lengths 24 through 31 produced a first block without the 0x80 padding byte missing 25 of 256 salts on the large fixture; added first_has_pad flag so the 0x80 lands in the first block at position 32+slen for slen in 24 through 31 and the tail block becomes length only; validated byte-exact on Pascal GTX 1080 fpga with smoke 32 medium 1024 and large 1048576 fixtures all 100 percent matched zero diff
 *
 * Revision 1.3  2026/05/22 02:09:44  dlr
 * sub-phase 2a.3 add hx_emit_e347_md5md5md5salt_opencl emitting self-contained tp0 pattern kernel B for the e347 MD5MD5SALT seven-op shape per-thread serial SALT_BATCH=64 outer loop pre-state in 4 registers hex32 expansion between inner MD5s helpers inline in emitted source kernel calls md5_block OCLParams EMIT_HIT_4_DEDUP_OR_OVERFLOW probe_compact_idx from gpu_common.cl which is prepended at JIT time via gpu_opencl_jit_compile_source_with_common
 *
 */

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "hx_spec.h"
#include "hx_walker.h"
#include "hx_emit.h"
#include "hx_patterns.h"
#include "hx_spec_entry.h"
#include "hx_emit_primitives.h"

/* ---- skeleton helpers (2a.1) ---- */

int hx_emit_kernel_attribute_opencl(char **out, size_t *out_cap,
                                    size_t *out_len)
{
    return hx_appendf(out, out_cap, out_len, "__kernel ");
}

int hx_emit_address_space_global_opencl(char **out, size_t *out_cap,
                                        size_t *out_len)
{
    return hx_appendf(out, out_cap, out_len, "__global ");
}

int hx_emit_thread_id_load_opencl(char **out, size_t *out_cap,
                                  size_t *out_len,
                                  const char *var_name)
{
    return hx_appendf(out, out_cap, out_len,
                      "  const uint %s = get_global_id(0);\n",
                      var_name ? var_name : "gid");
}

int hx_emit_atomic_inc_opencl(char **out, size_t *out_cap, size_t *out_len,
                              const char *counter_expr)
{
    return hx_appendf(out, out_cap, out_len,
                      "  atomic_inc(%s);\n",
                      counter_expr ? counter_expr : "&counter[0]");
}

/*
 * Payload-load skeleton. 2a.1/2a.2 emit only an annotation comment;
 * sub-phase 2a.3 expands this to emit the actual field reads
 * (num_words, words_off[], salt_lens[], etc.) needed by tp0 pattern.
 */
int hx_emit_payload_load_opencl(char **out, size_t *out_cap, size_t *out_len)
{
    return hx_appendf(out, out_cap, out_len,
                      "  // hx: payload-load stub (2a.2 -- expanded in 2a.3)\n");
}

/* ---- per-opcode helpers (2a.2) ----------------------------------- */

/*
 * OP_PUSH_VAR: declare a placeholder uint sized cell named after the
 * variable slot. Sub-phase 2a.3 replaces this with actual buffer
 * fetches into a candidate / salt buffer per the tp0 pattern.
 */
int hx_emit_push_var_opencl(char **out, size_t *cap, size_t *len,
                            int slot, const char *varname)
{
    /* Symbol is suffixed with *len so multi-PUSH-VAR programs (e31:
     * PUSH_VAR pass + PUSH_VAR salt) don't collide. *len is monotone-
     * growing across the walk. */
    return hx_appendf(out, cap, len,
        "  // hx: OP_PUSH_VAR slot=%d name=\"%s\"\n"
        "  // hx: (2a.2 placeholder -- actual variable load deferred)\n"
        "  uint _v_%zu = (uint)gid;\n",
        slot, varname ? varname : "?", *len);
}

/*
 * OP_PUSH_STR: emit a static const array initializer of the bytes
 * and a marker comment with the index. Length included in name so
 * each PUSH_STR in a program gets its own symbol.
 */
int hx_emit_push_str_opencl(char **out, size_t *cap, size_t *len,
                            int stridx, const char *literal, int literal_len)
{
    int rc = hx_appendf(out, cap, len,
        "  // hx: OP_PUSH_STR stridx=%d len=%d\n",
        stridx, literal_len);
    if (rc < 0) return rc;
    if (literal && literal_len > 0) {
        rc = hx_appendf(out, cap, len, "  __constant uchar _s_%d[%d] = {",
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

/*
 * OP_PUSH_INT: declare a long literal in the source. (Stack tracking
 * is purely comment-level until 2a.3; OpenCL doesn't see a "stack".)
 */
int hx_emit_push_int_opencl(char **out, size_t *cap, size_t *len,
                            int64_t ival)
{
    return hx_appendf(out, cap, len,
        "  // hx: OP_PUSH_INT %lld\n"
        "  long _i_%zu = (long)%lldL;\n",
        (long long)ival, *len, (long long)ival);
}

/* OP_STORE: comment-only annotation; storage handled by walker. */
int hx_emit_store_opencl(char **out, size_t *cap, size_t *len,
                         int slot, const char *varname)
{
    return hx_appendf(out, cap, len,
        "  // hx: OP_STORE slot=%d name=\"%s\"\n",
        slot, varname ? varname : "?");
}

/*
 * OP_CALL: for 2a.2 we recognize the `md5` function only and emit a
 * placeholder declaration that mimics md5_block call without actually
 * computing anything (we want JIT success). Other CALL targets emit
 * a `// hx: function 'NAME' not yet supported in 2a.2` comment and
 * return -1. Real implementation in 2a.3 emits inline md5_block /
 * sha-block calls per the tp0 pattern.
 */
int hx_emit_call_opencl(char **out, size_t *cap, size_t *len,
                        const char *fn_name, int nargs, uint8_t role)
{
    int rc;
    if (!fn_name) fn_name = "?";

    rc = hx_appendf(out, cap, len,
        "  // hx: OP_CALL fn=\"%s\" nargs=%d role=%u\n",
        fn_name, nargs, (unsigned)role);
    if (rc < 0) return rc;

    if (strcmp(fn_name, "md5") == 0) {
        /* Placeholder md5 call: declare a uint4 holding gid; subsequent
         * walker phases (2a.3) will replace this with real md5_block.
         * Symbol is suffixed with *len so multi-CALL programs (e31,
         * e347) don't collide at JIT time -- *len is monotone-growing
         * across the walk and uniquely identifies each emit position. */
        return hx_appendf(out, cap, len,
            "  uint4 _md5_state_%zu = (uint4)(gid, gid ^ 0x67452301u, "
            "gid ^ 0xefcdab89u, gid ^ 0x98badcfeu);\n", *len);
    }

    /* Unsupported function for 2a.2 -- annotate and fail. Walker
     * caller treats negative as fatal per
     * feedback_external_failures_are_fatal.md. */
    rc = hx_appendf(out, cap, len,
        "  // hx: function '%s' not yet supported in 2a.2 "
        "(deferred to 2a.3+)\n",
        fn_name);
    if (rc < 0) return rc;
    fprintf(stderr,
            "hx codegen: CALL '%s' not implemented in sub-phase 2a.2 "
            "(deferred to 2a.3+)\n", fn_name);
    return -1;
}

/* OP_CONCAT: comment-only annotation; semantics deferred to 2a.3. */
int hx_emit_concat_opencl(char **out, size_t *cap, size_t *len)
{
    return hx_appendf(out, cap, len,
        "  // hx: OP_CONCAT (2a.2 placeholder; semantics deferred)\n");
}

/*
 * OP_HALT: terminate the kernel body. For 2a.2 the trivial pattern is
 * to dead-write the md5 state into the counter via a never-taken
 * branch so the JIT can't DCE the rest of the body.
 */
int hx_emit_halt_opencl(char **out, size_t *cap, size_t *len)
{
    return hx_appendf(out, cap, len,
        "  // hx: OP_HALT (terminate program; result on stack top)\n"
        "  if (gid == 0xffffffffu) {\n"
        "    atomic_inc(&counter[0]);\n"
        "  }\n");
}

/* ====================================================================
 * Sub-phase 2a.3 (2026-05-21) -- e347 (MD5MD5SALT) tp0-pattern emitter
 *
 * Walks the recognized 7-op shape from hx_detect_pattern() and emits a
 * self-contained OpenCL kernel B that JIT-compiles on Pascal (GTX 1080
 * verified) AND structurally mirrors gpu/gpu_kernelb_md5md5salt_nocache
 * .cl per feedback_tp0_pattern_is_correct_for_pascal_salted_md5.md.
 *
 * The emitted source is intended to be JIT-compiled via
 * gpu_opencl_jit_compile_source_with_common(), which PREPENDS
 * gpu_common_str at clCreateProgramWithSource time. That gives the
 * emitted source access to:
 *
 *   md5_block (noinline)        -- gpu_common.cl
 *   OCLParams                    -- gpu_common.cl
 *   HIT_STRIDE                   -- gpu_common.cl
 *   EMIT_HIT_4_DEDUP_OR_OVERFLOW -- gpu_common.cl
 *   probe_compact_idx            -- gpu_common.cl
 *
 * The emitted source defines its own:
 *
 *   md5_buf_global               -- runs MD5 over a __global uchar*
 *                                   (mirrors gpu_kernelb_md5md5salt_-
 *                                   nocache.cl 112-157; couldn't be
 *                                   hoisted to gpu_common.cl because the
 *                                   template_phase0 kernels use a
 *                                   __private variant)
 *   md5_outer_hex_combine        -- runs MD5 over (hex32(inner_state) ||
 *                                   salt), the e347-specific outer chain
 *   hex32_into_M                 -- packs 32 hex chars of a 4-uint state
 *                                   into M[0..7] (8 uints) for use as
 *                                   the first 32 bytes of the outer
 *                                   message before salt_pack_uint-style
 *                                   tail packing
 *   kernelb_hx_e347_phase0       -- the tp0 entry point
 *
 * Algorithm semantics (per hx.8 line 357):
 *   digest = MD5( hex32( MD5( MD5(pass) ) ) || salt )
 *
 * Note that this is NOT the same chain as the existing hand-written
 * gpu_kernelb_md5md5salt_nocache.cl, which computes
 *   digest = MD5( MD5_bin(pass) || salt )
 * with no hex32 expansion. The 2a.5 validation harness will diff against
 * a CPU oracle that mirrors the e347 chain; the hand-written kernel B is
 * a STRUCTURAL reference for tp0 shape, not a byte-exact oracle.
 *
 * Per feedback_no_nested_block_comments_in_cl.md the emitted source
 * uses // comments only.
 *
 * Per feedback_md5_block_noinline_pascal.md we reuse the noinline
 * md5_block from gpu_common.cl rather than declaring a local copy.
 *
 * SALT_BATCH=64 outer loop is per-thread serial (tp0 pattern). Each
 * thread processes 64 (candidate, salt) tuples; the inner MD5(MD5(pass))
 * pre-state is computed ONCE per thread and held in 4 registers across
 * the entire SALT_BATCH=64 loop.
 *
 * ==================================================================== */

/* Append the e347 kernel preamble: comments + the two inline helper
 * function definitions (md5_buf_global, md5_outer_hex_combine,
 * hex32_into_M). Caller appends the kernel itself afterwards. */
static int emit_e347_helpers(char **out, size_t *cap, size_t *len)
{
    int rc;

    rc = hx_appendf(out, cap, len,
        "// ====================================================================\n"
        "// hx codegen sub-phase 2a.3+2a.5 (2026-05-21): e347 MD5MD5SALT tp0 pattern\n"
        "// Emitted by hx_emit_e347_md5md5md5salt_opencl()\n"
        "// Pattern matched: HX_PATTERN_E347_MD5MD5MD5SALT\n"
        "// Algorithm: MD5( hex32( MD5( hex32( MD5(pass) ) ) ) || salt )\n"
        "//   (matches mdxfind CPU JOB_MD5MD5SALT at mdxfind.c:23174 and\n"
        "//    hashpipe HT(MD5MD5SALT); fixed in 2a.5 from the broken 2a.3\n"
        "//    one-hex32-inner form that mirrored the hand-written kernel's drift)\n"
        "// Structural reference: gpu/gpu_kernelb_md5md5salt_nocache.cl\n"
        "// Helpers from gpu_common.cl (prepended at JIT time):\n"
        "//   md5_block, OCLParams, HIT_STRIDE, EMIT_HIT_4_DEDUP_OR_OVERFLOW,\n"
        "//   probe_compact_idx\n"
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

    /* md5_buf_global: MD5 over a __global const uchar* candidate. */
    rc = hx_appendf(out, cap, len,
        "// hx: helper md5_buf_global -- MD5 of variable-length __global candidate.\n"
        "// Mirrors gpu_kernelb_md5md5salt_nocache.cl rev 1.4 lines 112-157.\n"
        "static void md5_buf_global(__global const uchar *data, int len,\n"
        "                           uint *hx, uint *hy, uint *hz, uint *hw)\n"
        "{\n"
        "    uint M[16];\n"
        "    int pos = 0;\n"
        "    *hx = 0x67452301u;\n"
        "    *hy = 0xEFCDAB89u;\n"
        "    *hz = 0x98BADCFEu;\n"
        "    *hw = 0x10325476u;\n"
        "    while (len - pos >= 64) {\n"
        "        for (int j = 0; j < 16; j++) {\n"
        "            int b = pos + j * 4;\n"
        "            M[j] = (uint)data[b]\n"
        "                 | ((uint)data[b + 1] << 8)\n"
        "                 | ((uint)data[b + 2] << 16)\n"
        "                 | ((uint)data[b + 3] << 24);\n"
        "        }\n"
        "        md5_block(hx, hy, hz, hw, M);\n"
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
        "        md5_block(hx, hy, hz, hw, M);\n"
        "    } else {\n"
        "        md5_block(hx, hy, hz, hw, M);\n"
        "        for (int j = 0; j < 16; j++) M[j] = 0;\n"
        "        M[14] = (uint)(len * 8);\n"
        "        M[15] = 0;\n"
        "        md5_block(hx, hy, hz, hw, M);\n"
        "    }\n"
        "}\n"
        "\n");
    if (rc < 0) return rc;

    /* md5_state_to_bin16: writes 16 LE bytes of a 4-uint state into a
     * private buffer. Used by md5_buf_private (which takes the binary
     * inner digest and runs MD5 over it -- the SECOND inner MD5) and
     * by md5_outer_hex_combine (writes 32 hex chars of the SECOND
     * inner digest into M[0..7]).
     *
     * NOTE on e347 chain semantics: hx CONCAT pops two stack values
     * (top=salt-string, second=inner-binary-digest); the inner-binary
     * value stringifies to hex32 when concatenated with a string. That
     * is why the emitted code feeds hex32(inner_digest) into the outer
     * MD5 -- to match the hx VM's CONCAT-of-binary-and-string
     * semantics. */

    /* md5_buf_private32: MD5 of a fixed 32-byte private buffer (the
     * HEX32-encoded intermediate between inner CALL #1 and inner CALL #2).
     *
     * NOTE 2026-05-21 (sub-phase 2a.5 byte-exact correctness fix):
     * The original 2a.3 emitter declared md5_buf_private16 here and used
     * it on the binary MD5(pass) state. That produced the WRONG chain --
     * MD5( hex32(MD5(MD5_bin(pass))) || salt ) -- which is exactly the
     * same drift documented in feedback_handwritten_kernel_b_drift_-
     * md5md5salt.md for the hand-written gpu_kernelb_md5md5salt_nocache.cl.
     *
     * The hx VM's CONCAT-of-binary-and-string semantics ALSO apply to
     * a digest fed as INPUT to a subsequent md5() call: per hx.8 e347
     * `md5(md5(md5(pass)) . salt)` with the default role expansion is
     *   MD5( hex32(MD5(hex32(MD5(pass)))) || salt )
     * -- two hex32 expansions, not one. Matches mdxfind CPU JOB_MD5MD5SALT
     * at mdxfind.c:23174 (hex32 → JOB_MD5SALT input, which itself hex32s
     * before salt-concat).
     *
     * Fix: hex32-expand the first inner state into a 32-byte private
     * buffer, then MD5 those 32 hex chars to produce the second inner
     * state. Block padding: 32 < 56 so one md5_block call suffices. */
    rc = hx_appendf(out, cap, len,
        "// hx: helper md5_buf_private32 -- MD5 of a 32-byte private buffer.\n"
        "// Used between the two inner CALL md5 ops to compute\n"
        "// MD5(hex32(MD5(pass))).\n"
        "static void md5_buf_private32(const uchar *data,\n"
        "                              uint *hx, uint *hy, uint *hz, uint *hw)\n"
        "{\n"
        "    uint M[16];\n"
        "    *hx = 0x67452301u;\n"
        "    *hy = 0xEFCDAB89u;\n"
        "    *hz = 0x98BADCFEu;\n"
        "    *hw = 0x10325476u;\n"
        "    for (int j = 0; j < 16; j++) M[j] = 0;\n"
        "    for (int i = 0; i < 32; i++) {\n"
        "        uint v = (uint)data[i];\n"
        "        M[i >> 2] |= v << ((i & 3) * 8);\n"
        "    }\n"
        "    M[32 >> 2] |= (uint)0x80 << ((32 & 3) * 8);\n"
        "    M[14] = (uint)(32 * 8);\n"
        "    M[15] = 0;\n"
        "    md5_block(hx, hy, hz, hw, M);\n"
        "}\n"
        "\n");
    if (rc < 0) return rc;

    /* state_to_hex32_bytes: stash 4-uint state into 32 lowercase hex
     * characters in a private byte buffer. Used between the two inner
     * MD5s to produce the hex-encoded input to the second inner MD5
     * (matches mdxfind CPU JOB_MD5MD5SALT semantics).
     *
     * Sub-phase 2a.5 (2026-05-21): replaces state_to_le_bytes16; the
     * 16-byte LE-binary form fed the wrong chain (one hex32 expansion
     * instead of two). See md5_buf_private32 comment for context. */
    rc = hx_appendf(out, cap, len,
        "// hx: helper state_to_hex32_bytes -- write 4-uint state as 32 lowercase hex chars.\n"
        "static void state_to_hex32_bytes(uint a, uint b, uint c, uint d,\n"
        "                                 uchar *buf)\n"
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

    /* hex32_into_M: pack the 32 hex characters of a 4-uint MD5 state
     * directly into M[0..7] (8 uints = 32 bytes), in the canonical
     * lowercase little-endian hex order matching mdxfind's hex32
     * convention. Saves 32 byte-store instructions vs writing into a
     * private byte buffer and then re-packing into M. */
    rc = hx_appendf(out, cap, len,
        "// hx: helper hex32_into_M -- pack 32 hex chars of a 4-uint MD5 state\n"
        "// into M[0..7] (8 uints, 32 bytes). Lowercase little-endian per\n"
        "// mdxfind hex32 convention.\n"
        "static void hex32_into_M(uint a, uint b, uint c, uint d, uint *M)\n"
        "{\n"
        "    // Each uint expands to 8 hex chars. The hex of byte v is\n"
        "    // ('0'+v) for v<10, ('a'+v-10) for v>=10.\n"
        "    // Iterate byte-by-byte across (a,b,c,d) LE, producing the 32\n"
        "    // hex chars in order; then pack 4 chars per M[] uint LE.\n"
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

    /* md5_outer_hex_combine: outer MD5 over (hex32(inner_state) || salt)
     * Total message length = 32 + slen. salt comes from __global memory.
     *
     * The tp0 pattern uses md5_block_from8 here for a per-salt 12.5%
     * savings (pre-roll rounds 1-8 on M[0..7] which is the fixed hex32
     * prefix and reuse across the SALT_BATCH=64 loop). For 2a.3 we
     * implement the straightforward full-md5_block variant; the
     * pre-roll optimization is a later refinement (Phase 2a's
     * performance pass, post-correctness). The structural-mirror
     * objective is met as long as the SALT_BATCH=64 loop, hex32
     * pre-state, salt-packing, and probe-then-emit shape are present.
     * Per the brief Part B step 5 the md5_block_from8 call IS the
     * production tp0 form; including a comment marker here documents
     * the intentional simplification. */
    rc = hx_appendf(out, cap, len,
        "// hx: helper md5_outer_hex_combine -- outer MD5 over\n"
        "// (hex32(inner_state) || salt). tp0 pattern would use\n"
        "// md5_block_from8 here for per-salt round 1-8 reuse; the 2a.3\n"
        "// emission uses full md5_block for simplicity and structural\n"
        "// clarity. The pre-state across the SALT_BATCH=64 loop is the\n"
        "// inner-MD5-MD5 4-uint state; that IS the tp0 register-held\n"
        "// invariant. Round 1-8 reuse is a future micro-opt.\n"
        "static void md5_outer_hex_combine(uint ihx, uint ihy, uint ihz, uint ihw,\n"
        "                                  __global const uchar *salt, int slen,\n"
        "                                  uint *ohx, uint *ohy, uint *ohz, uint *ohw)\n"
        "{\n"
        "    if (slen < 0) slen = 0;\n"
        "    if (slen > HX_E347_MAX_SALT) slen = HX_E347_MAX_SALT;\n"
        "    int total_len = 32 + slen;\n"
        "\n"
        "    uint M[16];\n"
        "    *ohx = 0x67452301u;\n"
        "    *ohy = 0xEFCDAB89u;\n"
        "    *ohz = 0x98BADCFEu;\n"
        "    *ohw = 0x10325476u;\n"
        "\n"
        "    int pos = 0;\n"
        "    // First block: pack hex32(state) into M[0..7] then begin\n"
        "    // salt packing into M[8..15] (up to 32 bytes = 8 uints in\n"
        "    // this first block).\n"
        "    hex32_into_M(ihx, ihy, ihz, ihw, M);\n"
        "    int salt_in_first = slen;\n"
        "    if (salt_in_first > 32) salt_in_first = 32;\n"
        "    for (int j = 8; j < 16; j++) M[j] = 0;\n"
        "    for (int i = 0; i < salt_in_first; i++) {\n"
        "        uint v = (uint)salt[i];\n"
        "        int dst = 32 + i;\n"
        "        M[dst >> 2] |= v << ((dst & 3) * 8);\n"
        "    }\n"
        "    if (32 + salt_in_first < 64) {\n"
        "        // Padding fits in this block if (32 + slen + 1 + 8) <= 64\n"
        "        // i.e. slen <= 23. Otherwise need extra block(s).\n"
        "        if (32 + slen + 1 <= 56) {\n"
        "            int padpos = 32 + slen;\n"
        "            M[padpos >> 2] |= (uint)0x80 << ((padpos & 3) * 8);\n"
        "            M[14] = (uint)(total_len * 8);\n"
        "            M[15] = 0;\n"
        "            md5_block(ohx, ohy, ohz, ohw, M);\n"
        "            return;\n"
        "        }\n"
        "    }\n"
        "    // Multi-block path. Process first 64 bytes (32 hex + up to\n"
        "    // 32 salt). Sub-phase 2a.5 byte-exact fix: when salt fits\n"
        "    // ENTIRELY within the first block but the 0x80 padding+length\n"
        "    // does NOT (i.e. 24 <= slen <= 31), the 0x80 byte MUST be\n"
        "    // emitted in the first block at position 32+slen and the\n"
        "    // tail block must contain ONLY the length (no extra 0x80).\n"
        "    // Previously omitted; broke all slen in [24..31].\n"
        "    int first_has_pad = 0;\n"
        "    if (salt_in_first < 32) {\n"
        "        int padpos = 32 + salt_in_first;\n"
        "        M[padpos >> 2] |= (uint)0x80 << ((padpos & 3) * 8);\n"
        "        first_has_pad = 1;\n"
        "    }\n"
        "    md5_block(ohx, ohy, ohz, ohw, M);\n"
        "    pos = 32;  // bytes of salt consumed so far\n"
        "    int sleft = slen - salt_in_first;\n"
        "\n"
        "    // Process remaining salt 64 bytes at a time. salt[pos..pos+64)\n"
        "    while (sleft >= 64) {\n"
        "        for (int j = 0; j < 16; j++) {\n"
        "            int b = pos + j * 4;\n"
        "            M[j] = (uint)salt[b]\n"
        "                 | ((uint)salt[b + 1] << 8)\n"
        "                 | ((uint)salt[b + 2] << 16)\n"
        "                 | ((uint)salt[b + 3] << 24);\n"
        "        }\n"
        "        md5_block(ohx, ohy, ohz, ohw, M);\n"
        "        pos += 64;\n"
        "        sleft -= 64;\n"
        "    }\n"
        "    // Tail block. Remaining sleft bytes of salt + padding +\n"
        "    // length-in-bits. If padding fits (sleft < 56) one block,\n"
        "    // else two. When first_has_pad==1 the 0x80 is already in the\n"
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
        "        md5_block(ohx, ohy, ohz, ohw, M);\n"
        "    } else {\n"
        "        md5_block(ohx, ohy, ohz, ohw, M);\n"
        "        for (int j = 0; j < 16; j++) M[j] = 0;\n"
        "        M[14] = (uint)(total_len * 8);\n"
        "        M[15] = 0;\n"
        "        md5_block(ohx, ohy, ohz, ohw, M);\n"
        "    }\n"
        "}\n"
        "\n");
    if (rc < 0) return rc;

    return 0;
}

/* Emit the kernel entry. Per-thread serial SALT_BATCH=64 loop is the
 * tp0 shape from feedback_tp0_pattern_is_correct_for_pascal_salted_md5;
 * the inner MD5(MD5(pass)) pre-state is the register-held invariant
 * across all 64 salt iterations.
 *
 * Kernel signature matches gpu/gpu_kernelb_md5md5salt_nocache.cl rev 1.4
 * argument list so a future dispatcher can swap them. The buffer
 * quadruple (b_packed_buf, b_chunk_index, b_kernelA_state, b_params)
 * is consumed identically.
 */
static int emit_e347_kernel(char **out, size_t *cap, size_t *len)
{
    int rc;

    rc = hx_appendf(out, cap, len,
        "// hx: kernel signature mirrors kernelb_md5md5salt_nocache_phase0\n"
        "// (gpu/gpu_kernelb_md5md5salt_nocache.cl rev 1.4 line 281) so a\n"
        "// future Phase-4 dispatcher can swap them. reqd_work_group_size\n"
        "// (64,1,1) attribute pins the WG size to match the production\n"
        "// dispatcher's lsize=64; without it Pascal/NVIDIA may select a\n"
        "// different WG size and the salt-chunk topology breaks. Phase 4\n"
        "// sub-phase 4a.1 (2026-05-21) Part A: R8 fix.\n"
        "__attribute__((reqd_work_group_size(64,1,1)))\n"
        "__kernel void kernelb_hx_e347_phase0(\n"
        "    __global const uchar         *payload,\n"
        "    __global const uchar         *b_packed_buf,\n"
        "    __global const uint          *b_chunk_index,\n"
        "    __global const uchar         *salts,\n"
        "    __global const uint          *salt_offsets,\n"
        "    __global const ushort        *salt_lens,\n"
        "    __global const uint          *compact_fp,\n"
        "    __global const uint          *compact_idx,\n"
        "    __global const uchar         *hash_data_buf,\n"
        "    __global const ulong         *hash_data_off,\n"
        "    __global uint                *hits,\n"
        "    __global volatile uint       *hit_count,\n"
        "    __global const ulong         *overflow_keys,\n"
        "    __global const uchar         *overflow_hashes,\n"
        "    __global const uint          *overflow_offsets,\n"
        "    __global volatile uint       *hashes_shown\n"
        "    )\n"
        "{\n"
        "    // hx: state EMIT_KERNEL_PREAMBLE (e347 tp0)\n"
        "    __global const OCLParams *params_buf =\n"
        "        (__global const OCLParams *)payload;\n"
        "    OCLParams params = *params_buf;\n"
        "\n"
        "    // tp0 topology: 1 thread = 1 (word, salt_chunk_of_SALT_BATCH).\n"
        "    // Each thread iterates SALT_BATCH=64 salts serially with the\n"
        "    // inner-MD5-MD5 state held in registers across the loop.\n"
        "    uint num_salts_total = params.num_salts;\n"
        "    uint num_salt_chunks =\n"
        "        (num_salts_total + SALT_BATCH - 1u) / SALT_BATCH;\n"
        "    if (num_salt_chunks == 0u) num_salt_chunks = 1u;\n"
        "\n"
        "    uint gid              = get_global_id(0);\n"
        "    uint word_idx         = gid / num_salt_chunks;\n"
        "    uint salt_chunk_idx   = gid - word_idx * num_salt_chunks;\n"
        "    uint salt_base        = salt_chunk_idx * SALT_BATCH;\n"
        "\n"
        "    if (word_idx >= params.num_words) return;\n"
        "\n"
        "    // hx: state EMIT_PRE_SALT_INVARIANT (template_pre_salt equivalent)\n"
        "    // Compute MD5(MD5(pass)) ONCE per thread and hold across the\n"
        "    // SALT_BATCH=64 loop in 4 registers.\n"
        "    uint wpos = b_chunk_index[word_idx];\n"
        "    if (wpos >= params.packed_size) return;  // defensive guard\n"
        "    uint plen = (uint)b_packed_buf[wpos];\n"
        "    __global const uchar *pass_bytes = b_packed_buf + wpos + 1u;\n"
        "\n"
        "    // OP_CALL md5 #1: inner MD5(pass) -> 4 uints in registers.\n"
        "    uint ia, ib, ic, id;\n"
        "    md5_buf_global(pass_bytes, (int)plen, &ia, &ib, &ic, &id);\n"
        "\n"
        "    // OP_CALL md5 #2: hex32-encode (ia,ib,ic,id) and MD5 the 32\n"
        "    // hex chars. Sub-phase 2a.5 byte-exact correctness fix:\n"
        "    // the original 2a.3 emission MD5d the 16 binary bytes which\n"
        "    // produced the same drift documented in feedback_handwritten_-\n"
        "    // kernel_b_drift_md5md5salt.md. hx CONCAT-of-binary-and-string\n"
        "    // forces hex32 expansion at every digest-to-md5 boundary.\n"
        "    uchar inner_hex[32];\n"
        "    state_to_hex32_bytes(ia, ib, ic, id, inner_hex);\n"
        "    uint mma, mmb, mmc, mmd;\n"
        "    md5_buf_private32(inner_hex, &mma, &mmb, &mmc, &mmd);\n"
        "\n"
        "    // (mma,mmb,mmc,mmd) is the pre-salt invariant; hex32(this)\n"
        "    // forms M[0..7] of the outer block on every SALT_BATCH iter.\n"
        "\n"
        "    // B3 overflow ledger pointers (mirrors gpu_kernelb_md5md5salt_\n"
        "    // nocache.cl rev 1.4 line 380-383).\n"
        "    __global volatile uint *ovr_set =\n"
        "        (__global volatile uint *)(payload + 100);\n"
        "    __global volatile uint *ovr_gid =\n"
        "        (__global volatile uint *)(payload + 104);\n"
        "\n"
        "    // hx: state EMIT_SALT_BATCH_LOOP (tp0 outer; SALT_BATCH=64).\n"
        "    for (uint sbi = 0; sbi < SALT_BATCH; sbi++) {\n"
        "        uint salt_local = salt_base + sbi;\n"
        "        if (salt_local >= num_salts_total) break;\n"
        "        uint salt_idx_global = params.salt_start + salt_local;\n"
        "\n"
        "        uint soff = salt_offsets[salt_idx_global];\n"
        "        int  slen = (int)salt_lens[salt_idx_global];\n"
        "        __global const uchar *salt = salts + soff;\n"
        "\n"
        "        // OP_CALL md5 #3 (outer): MD5(hex32(inner) || salt)\n"
        "        uint hx, hy, hz, hw;\n"
        "        md5_outer_hex_combine(mma, mmb, mmc, mmd,\n"
        "                              salt, slen,\n"
        "                              &hx, &hy, &hz, &hw);\n"
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
        "            uint mask = 1u;  // iter == 1; dedup slot 0\n"
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

int hx_emit_e347_md5md5md5salt_opencl(
    char **out, size_t *out_cap,
    const hx_program *prog,
    const struct hx_specialization *spec)
{
    if (!out || !out_cap || !prog || !spec) return -1;

    size_t cur_len = 0;
    if (*out == NULL) *out_cap = 0;

    int rc;

    // Banner with structural details for dump-file readability.
    rc = hx_appendf(out, out_cap, &cur_len,
        "// hx codegen: PATTERN E347_MD5MD5MD5SALT matched\n"
        "// hx: program ncode=%d nvars=%d max_stack=%d has_emit=%d\n"
        "// hx: specialization iter=%u rules=%u masks=%u bf=%u "
        "salt_minlen=%u salt_maxlen=%u regime=%d width=%u\n"
        "// hx: this kernel will be JIT-compiled with gpu_common_str\n"
        "// hx: prepended (gpu_opencl_jit_compile_source_with_common)\n"
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

    rc = emit_e347_helpers(out, out_cap, &cur_len);
    if (rc < 0) return rc;

    rc = emit_e347_kernel(out, out_cap, &cur_len);
    if (rc < 0) return rc;

    // Defensive NUL terminator.
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
 * Sub-phase 5a.2 (2026-05-22) -- MAKE_MD5PASS family emitter (OpenCL)
 *
 * The family kernel computes  outer_hash( hex32( md5(pass) ) || pass )
 * for a per-thread (no salt) outer dispatch. Topology:
 *
 *   1 thread = 1 candidate word
 *   each thread:
 *     1. Compute MD5(pass) -> 16-byte digest -> hex32 expansion
 *     2. Build private buffer hex32 || pass (length 32 + plen)
 *     3. Run outer_hash over that buffer
 *     4. probe_compact_idx + EMIT_HIT_4_DEDUP_OR_OVERFLOW
 *
 * No SALT_BATCH loop (family is unsalted). num_salts is IGNORED by the
 * kernel body; binding it to 0 or N is harmless. The 4 salt-table args
 * (salts/salt_offsets/salt_lens/payload.num_salts) are kept in the
 * kernel signature so production dispatcher binds the same 16-arg
 * layout as e347 -- the kernel just doesn't read them.
 *
 * Helpers are reused VERBATIM from the e347 emitter:
 *   md5_buf_global             -- MD5 over a __global candidate buffer
 *   state_to_hex32_bytes       -- write 4-uint state as 32 hex bytes
 *
 * Per-primitive outer body is emitted via dispatch on the OP_CALL[4]
 * primitive name resolved from entry->call_names. 5a.2 implements
 * SHA1 only; the other 7 supported primitives FATAL.
 *
 * Per feedback_no_nested_block_comments_in_cl.md the emitted source
 * uses // comments only.
 * Per feedback_md5_block_noinline_pascal.md md5_block and sha1_block
 * keep their noinline status (from gpu_common.cl).
 * Per feedback_external_failures_are_fatal.md any unknown / unsupported
 * outer primitive FATALs at HOST emit time.
 *
 * Future sub-phases:
 *   5a.3 -- Metal twin (hx_emit_family_md5pass_metal)
 *   5a.4 -- md4 / md5 / sha224 / sha256 / sha384 / sha512 / rmd160 per-
 *           primitive emitters
 *   5b   -- 22 deferred primitives (md2/gost/haval/tiger/wrl/sne128/256/
 *           rmd128) once gpu_common.cl gains the corresponding *_block
 *           functions
 * ==================================================================== */

/* Emit shared family helpers: md5_buf_global + state_to_hex32_bytes.
 * Mirror of emit_e347_helpers (subset; the family has no second inner
 * MD5 and no salt-concat outer, so md5_buf_private32 and
 * md5_outer_hex_combine are NOT emitted). */
static int emit_family_md5pass_helpers(char **out, size_t *cap, size_t *len)
{
    int rc;

    rc = hx_appendf(out, cap, len,
        "// ====================================================================\n"
        "// hx codegen sub-phase 5a.2 (2026-05-22): MAKE_MD5PASS family\n"
        "// Emitted by hx_emit_family_md5pass_opencl()\n"
        "// Pattern matched: HX_PATTERN_FAMILY_MD5PASS\n"
        "// Algorithm: outer_hash( hex32( MD5(pass) ) || pass )\n"
        "//   (matches mdxfind CPU MAKE_MD5PASS chain at JOB_*MD5PASS:\n"
        "//    mymd5(pass) -> prmd5(.., linebuf, 32) -> strncpy(&linebuf[32],\n"
        "//    cur, len) -> outer(linebuf, 32+len). Per-primitive cases at\n"
        "//    mdxfind.c:25023 (e123 MD5MD5PASS) and 27272 (e161 SHA1MD5PASS).)\n"
        "// Helpers from gpu_common.cl (prepended at JIT time):\n"
        "//   md5_block, sha1_block, OCLParams, HIT_STRIDE,\n"
        "//   EMIT_HIT_4_DEDUP_OR_OVERFLOW, probe_compact_idx\n"
        "// ====================================================================\n"
        "\n"
        "#ifndef HX_FAMILY_MAX_PASS\n"
        "#define HX_FAMILY_MAX_PASS 240\n"
        "#endif\n"
        "\n");
    if (rc < 0) return rc;

    /* md5_buf_global: reused verbatim from e347. Computes MD5 of a
     * __global const uchar* candidate, returning the 4-uint state. */
    rc = hx_appendf(out, cap, len,
        "// hx: helper md5_buf_global -- MD5 of variable-length __global candidate.\n"
        "// Mirrors gpu_kernelb_md5md5salt_nocache.cl rev 1.4 lines 112-157 and the\n"
        "// e347 emitter's md5_buf_global verbatim (shared family helper).\n"
        "static void md5_buf_global(__global const uchar *data, int len,\n"
        "                           uint *hx, uint *hy, uint *hz, uint *hw)\n"
        "{\n"
        "    uint M[16];\n"
        "    int pos = 0;\n"
        "    *hx = 0x67452301u;\n"
        "    *hy = 0xEFCDAB89u;\n"
        "    *hz = 0x98BADCFEu;\n"
        "    *hw = 0x10325476u;\n"
        "    while (len - pos >= 64) {\n"
        "        for (int j = 0; j < 16; j++) {\n"
        "            int b = pos + j * 4;\n"
        "            M[j] = (uint)data[b]\n"
        "                 | ((uint)data[b + 1] << 8)\n"
        "                 | ((uint)data[b + 2] << 16)\n"
        "                 | ((uint)data[b + 3] << 24);\n"
        "        }\n"
        "        md5_block(hx, hy, hz, hw, M);\n"
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
        "        md5_block(hx, hy, hz, hw, M);\n"
        "    } else {\n"
        "        md5_block(hx, hy, hz, hw, M);\n"
        "        for (int j = 0; j < 16; j++) M[j] = 0;\n"
        "        M[14] = (uint)(len * 8);\n"
        "        M[15] = 0;\n"
        "        md5_block(hx, hy, hz, hw, M);\n"
        "    }\n"
        "}\n"
        "\n");
    if (rc < 0) return rc;

    /* state_to_hex32_bytes: reused verbatim from e347. Writes 4-uint
     * state as 32 lowercase hex characters into a private byte buffer. */
    rc = hx_appendf(out, cap, len,
        "// hx: helper state_to_hex32_bytes -- write 4-uint state as 32 lowercase hex chars.\n"
        "// Reused verbatim from e347 emitter (shared family helper).\n"
        "static void state_to_hex32_bytes(uint a, uint b, uint c, uint d,\n"
        "                                 uchar *buf)\n"
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

/* Per-primitive outer body emit: SHA1.
 *
 * The outer body is a static helper function in the emitted kernel
 * source. Signature:
 *
 *   static void outer_sha1_concat_then_hash(
 *       uint mma, uint mmb, uint mmc, uint mmd,   // pre-MD5(pass) state
 *       __global const uchar *pass, int plen,      // original pass
 *       uint *ha, uint *hb, uint *hc, uint *hd, uint *he);
 *
 * Body: write hex32(mma..mmd) into a 32-byte private buffer; append
 * `pass` from __global memory; total length = 32 + plen; pad per SHA1
 * convention (BIG-ENDIAN length encoding, vs MD5's little-endian);
 * iterate sha1_block over 64-byte chunks. Output is 5 uints (h0..h4).
 *
 * SHA1 padding details:
 *   - Append 0x80 byte at position L.
 *   - Pad with zeros up to position 56 (mod 64).
 *   - Append 64-bit BIG-ENDIAN length-in-bits (W[14]=upper, W[15]=lower
 *     in BE order, which means M[14] high 32 bits and M[15] low 32 bits
 *     when M is the BE-packed schedule words).
 *   - SHA1 block schedule is BE 32-bit words from the byte stream.
 *
 * For e161 the input length is 32 + plen (plen 1..240).
 *   plen ≤ 23  => 32+plen ≤ 55  => single 64B block (pad+len fit in
 *                  same block as data).
 *   plen ≤ 31  => 32+plen ≤ 63 but 32+plen+1+8 > 64 => two blocks
 *                  (data+pad fits; length spills).  Handled by the
 *                  "else" branch.
 *   plen ≥ 32  => 32+plen ≥ 64 => multi-block walk.
 *
 * Per gpu_common.cl line 942: signature is
 *   void sha1_block(uint *state, uint *M)
 * where state is uint[5] (a..e) and M is uint[16] BE-packed.
 */
static int emit_outer_sha1_concat_then_hash(char **out,
                                            size_t *cap, size_t *len)
{
    int rc;

    rc = hx_appendf(out, cap, len,
        "// hx: helper outer_sha1_concat_then_hash -- SHA1 over\n"
        "// (hex32(md5(pass)) || pass). Total input length = 32 + plen.\n"
        "// Output: 5 uints (h0..h4). probe_compact_idx uses h0..h3 only\n"
        "// (compact_fp is 64-bit key from first 8 bytes); h4 is unused\n"
        "// for probe but EMIT_HIT_4_DEDUP_OR_OVERFLOW takes h0..h3 and\n"
        "// matches the hand-tuned EMIT_HIT_5 macros' first-4 contract\n"
        "// when the production dispatcher routes round-trip readback.\n"
        "static void outer_sha1_concat_then_hash(\n"
        "    uint mma, uint mmb, uint mmc, uint mmd,\n"
        "    __global const uchar *pass, int plen,\n"
        "    uint *h0, uint *h1, uint *h2, uint *h3, uint *h4)\n"
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
        "    state_to_hex32_bytes(mma, mmb, mmc, mmd, inner_hex);\n"
        "\n"
        "    // SHA1 schedule words are BIG-ENDIAN: M[w] = (b0<<24)|(b1<<16)|\n"
        "    // (b2<<8)|b3. Build first 64-byte block from inner_hex (32 B)\n"
        "    // then as much of pass as fits.\n"
        "    uint M[16];\n"
        "    int byte_pos = 0;   // bytes consumed from logical stream\n"
        "    int pass_consumed = 0;\n"
        "    int first_has_pad = 0;\n"
        "\n"
        "    // Helper macro: pack 4 bytes at logical positions [s..s+3] into\n"
        "    // M[s>>2] big-endian. Caller is responsible for setting all of\n"
        "    // M[] -- the macro overwrites the target word.\n"
        "    // (Inlined per word to keep things simple; no loop overhead.)\n"
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
        "    // Mirrors the e347 first_has_pad pattern (md5_outer_hex_combine\n"
        "    // rev 1.5 lines 565-569). Without this, the tail block re-emits\n"
        "    // a 0x80 and the message-length encoding is off by one block.\n"
        "    if (p_in_first == plen && byte_pos < 64) {\n"
        "        int pad_pos = byte_pos;\n"
        "        int wi = pad_pos >> 2;\n"
        "        int sh = (3 - (pad_pos & 3)) * 8;\n"
        "        M[wi] |= 0x80u << sh;\n"
        "        first_has_pad = 1;\n"
        "    }\n"
        "    sha1_block(state, M);\n"
        "\n"
        "    // Walk remaining pass bytes. byte_pos for the stream stays at\n"
        "    // logical (32 + pass_consumed); pass[pass_consumed..plen) is\n"
        "    // what's left to process.\n"
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
        "    // build the compact_fp key. We must byte-swap each state\n"
        "    // word so the uint h0..h4 we hand back matches what the\n"
        "    // CPU stores at oracle[0..4]. (MD5 doesn't need this; its\n"
        "    // schedule is LE so md5_block returns LE uints natively.)\n"
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
 * Per-primitive emit bodies for sub-phase 5a.4 (2026-05-23).
 *
 * Six additional primitives wired here. Each helper emits a static
 * function named outer_<primitive>_concat_then_hash with signature
 * (uint mma, uint mmb, uint mmc, uint mmd, __global const uchar *pass,
 * int plen, uint *h0, uint *h1, uint *h2, uint *h3) -- and for the
 * 5-uint state primitives an additional *h4 (rmd160). Per
 * [[feedback-be-state-primitives-need-byteswap-in-codegen]]:
 *   md4, rmd160       -- LE-schedule, NO state byte-swap
 *   sha224, sha256    -- BE-schedule, swap each of first 4 uints
 *   sha384, sha512    -- BE-schedule, ULONG state (8 ulongs), swap
 *                        each of first 2 ulongs THEN split each into
 *                        2 LE uints (high 32 bits to h_odd, low 32 to
 *                        h_even).
 *
 * Compact_fp probe uses 4 uints from offset 0 of the digest (first 16
 * bytes for 20/28/32/48/64-byte digests). All six bodies write the
 * first 4 LE uints to *h0..*h3 regardless of full digest width.
 *
 * Block walk pattern mirrors the SHA1 reference (single-block fast
 * path, first_has_pad multi-block path, tail length encoding). The
 * only differences across primitives:
 *   - LE vs BE schedule packing (M[wi] word build, length suffix
 *     placement at end of block)
 *   - block size (64 for md4/rmd160/sha224/sha256; 128 for sha384/512)
 *   - length suffix width (8 bytes for 64B-block; 16 bytes for
 *     128B-block, all-zero high half since total_len < 2^32 always)
 *   - state width (4 uints for md4; 5 uints for rmd160; 8 uints for
 *     sha224/256; 8 ulongs for sha384/512)
 * ==================================================================== */

/* Per-primitive outer body emit: MD2 (16-byte block, PKCS pad, checksum-
 * block-as-final). Sub-phase 5b.1a (2026-05-27). Bespoke per D15.3.a --
 * MD2 structurally diverges from MD4/MD5 family (different block size,
 * different padding, distinct checksum-as-final-block step). Mirrors
 * md2/md2.c md2_update + md2_final byte-for-byte; calls md2_block (from
 * gpu_common.cl rev 1.26) per 16-byte data block. */
static int emit_outer_md2_concat_then_hash(char **out,
                                           size_t *cap, size_t *len)
{
    int rc;

    rc = hx_appendf(out, cap, len,
        "// hx: helper outer_md2_concat_then_hash -- MD2 over\n"
        "// (hex32(md5(pass)) || pass). Total input length = 32 + plen.\n"
        "// Output: 4 uints (h0..h3, LE pack of state[0..15]).\n"
        "//\n"
        "// MD2 is byte-oriented (no endianness). 16-byte data blocks;\n"
        "// PKCS padding (1-16 copies of byte N where N = bytes needed\n"
        "// to reach the next 16-byte boundary); one extra block of the\n"
        "// running checksum as final input.\n"
        "//\n"
        "// For the family use case the input length is exactly 32+plen\n"
        "// (deterministic; plen <= HX_FAMILY_MAX_PASS). Total processed\n"
        "// blocks = ((total_len) / 16) data blocks + 1 padded final +\n"
        "// 1 checksum block. With HX_FAMILY_MAX_PASS = 64 the max\n"
        "// processed-block count = ((32+64+15)/16) + 1 = 7+1 = 8.\n"
        "static void outer_md2_concat_then_hash(\n"
        "    uint mma, uint mmb, uint mmc, uint mmd,\n"
        "    __global const uchar *pass, int plen,\n"
        "    uint *h0, uint *h1, uint *h2, uint *h3)\n"
        "{\n"
        "    if (plen < 0) plen = 0;\n"
        "    if (plen > HX_FAMILY_MAX_PASS) plen = HX_FAMILY_MAX_PASS;\n"
        "    int total_len = 32 + plen;\n"
        "\n"
        "    // MD2 state: 48 zero bytes. Checksum: 16 zero bytes.\n"
        "    uchar state[48];\n"
        "    uchar checksum[16];\n"
        "    for (int i = 0; i < 48; i++) state[i] = (uchar)0;\n"
        "    for (int i = 0; i < 16; i++) checksum[i] = (uchar)0;\n"
        "\n"
        "    // Materialize hex32(md5(pass)) into a 32-byte buffer.\n"
        "    uchar inner_hex[32];\n"
        "    state_to_hex32_bytes(mma, mmb, mmc, mmd, inner_hex);\n"
        "\n"
        "    // Compose the full input (32 hex chars || pass) into a\n"
        "    // local buffer sized to HX_FAMILY_MAX_PASS + 32 bytes.\n"
        "    // PKCS pad adds up to 16 more; checksum step does not\n"
        "    // need a separate buffer (passed via &checksum[0]).\n"
        "    uchar block[16];\n"
        "    int consumed = 0;\n"
        "\n"
        "    // Process complete 16-byte blocks from inner_hex (32 bytes).\n"
        "    // 32 / 16 = 2 full blocks always.\n"
        "    for (int b = 0; b < 2; b++) {\n"
        "        for (int j = 0; j < 16; j++) {\n"
        "            block[j] = inner_hex[b * 16 + j];\n"
        "        }\n"
        "        md2_block(state, checksum, block, 1);\n"
        "        consumed += 16;\n"
        "    }\n"
        "\n"
        "    // Process pass[] in 16-byte chunks while at least 16 bytes\n"
        "    // remain. consumed - 32 = pass bytes processed so far.\n"
        "    int pass_off = 0;\n"
        "    while ((plen - pass_off) >= 16) {\n"
        "        for (int j = 0; j < 16; j++) {\n"
        "            block[j] = pass[pass_off + j];\n"
        "        }\n"
        "        md2_block(state, checksum, block, 1);\n"
        "        pass_off += 16;\n"
        "        consumed += 16;\n"
        "    }\n"
        "\n"
        "    // Tail: 0..15 leftover pass bytes; PKCS pad with N copies\n"
        "    // of (16 - tail_len). When tail_len == 0, pad is 16 copies\n"
        "    // of 16 -- a full padding block.\n"
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
        "    // Final: process the checksum as one more block. Per RFC\n"
        "    // errata the checksum is NOT updated on this call; passing\n"
        "    // update_checksum=0 keeps that semantics explicit. (The\n"
        "    // digest reads only state[0..15], so the checksum's final\n"
        "    // value is unread either way; matching B-Con donor with\n"
        "    // update_checksum=1 produces the same output.)\n"
        "    md2_block(state, checksum, checksum, 0);\n"
        "\n"
        "    // Digest: state[0..15] packed LE into h0..h3.\n"
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

/* Per-primitive outer body emit: MD4 (LE-schedule, 4-uint state). */
static int emit_outer_md4_concat_then_hash(char **out,
                                           size_t *cap, size_t *len)
{
    int rc;

    rc = hx_appendf(out, cap, len,
        "// hx: helper outer_md4_concat_then_hash -- MD4 over\n"
        "// (hex32(md5(pass)) || pass). Total input length = 32 + plen.\n"
        "// Output: 4 uints (h0..h3). MD4 schedule is LITTLE-ENDIAN\n"
        "// (same as MD5); NO state byte-swap before h0..h3 write. The\n"
        "// CPU oracle (MD4()) stores LE bytes; harness reinterprets\n"
        "// as LE uints; kernel state[i] is already LE so direct copy\n"
        "// matches.\n"
        "static void outer_md4_concat_then_hash(\n"
        "    uint mma, uint mmb, uint mmc, uint mmd,\n"
        "    __global const uchar *pass, int plen,\n"
        "    uint *h0, uint *h1, uint *h2, uint *h3)\n"
        "{\n"
        "    if (plen < 0) plen = 0;\n"
        "    if (plen > HX_FAMILY_MAX_PASS) plen = HX_FAMILY_MAX_PASS;\n"
        "    int total_len = 32 + plen;\n"
        "\n"
        "    // MD4 initial state (same as MD5).\n"
        "    uint a = 0x67452301u, b = 0xEFCDAB89u, c = 0x98BADCFEu, d = 0x10325476u;\n"
        "\n"
        "    uchar inner_hex[32];\n"
        "    state_to_hex32_bytes(mma, mmb, mmc, mmd, inner_hex);\n"
        "\n"
        "    // MD4 schedule words are LITTLE-ENDIAN (b0|b1<<8|b2<<16|b3<<24).\n"
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
        "            int sh = (abs_pos & 3) * 8;  // LE\n"
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
        "        // MD4 length is LITTLE-ENDIAN 64-bit at M[14]/M[15].\n"
        "        M[14] = (uint)(bitlen & 0xffffffffu);\n"
        "        M[15] = (uint)(bitlen >> 32);\n"
        "        md4_block(&a, &b, &c, &d, M);\n"
        "        *h0 = a; *h1 = b; *h2 = c; *h3 = d;\n"
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
        "    md4_block(&a, &b, &c, &d, M);\n"
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
        "        md4_block(&a, &b, &c, &d, M);\n"
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
        "        md4_block(&a, &b, &c, &d, M);\n"
        "    } else {\n"
        "        md4_block(&a, &b, &c, &d, M);\n"
        "        for (int w = 0; w < 16; w++) M[w] = 0u;\n"
        "        ulong bitlen = (ulong)total_len * 8u;\n"
        "        M[14] = (uint)(bitlen & 0xffffffffu);\n"
        "        M[15] = (uint)(bitlen >> 32);\n"
        "        md4_block(&a, &b, &c, &d, M);\n"
        "    }\n"
        "    // MD4 state is LE; direct copy.\n"
        "    *h0 = a; *h1 = b; *h2 = c; *h3 = d;\n"
        "}\n"
        "\n");
    return rc;
}

/* Sub-phase 5c.2 (2026-05-27): MD5-as-OUTER multi-emit helper for e123
 * MD5MD5PASS -- the FIRST multi-emit family member.
 *
 * MD5 was always the INNER hash in the shipped family; e123 needs MD5 as
 * the OUTER too. This helper is structurally identical to the MD4 helper
 * (LE schedule, 4-uint state, 16-byte digest, single-block fast path +
 * multi-block first_has_pad tail) with two differences:
 *   1. it calls md5_block (not md4_block); and
 *   2. it takes a `sep` parameter encoding the multi-emit variant:
 *        sep == 0 -> canonical: outer message = hex32(md5(pass)) || pass
 *                    (total_len = 32 + plen)
 *        sep == 1 -> colon:     outer message = hex32 || ':' || pass
 *                    (total_len = 33 + plen; one ':' byte injected at
 *                     logical position 32, shifting pass to start at 33)
 *
 * Both variants share the SAME md5(pass) inner state (mma..mmd); only the
 * outer concatenation + final MD5 differ. The family kernel body calls
 * this helper TWICE (sep=0 then sep=1) for emit_class==HX_EMIT_MULTI,
 * matching the CPU oracle (mdxfind.c:25181-25204) which builds linebuf
 * (canonical) + linebuf2 (colon) and checkhash()es each independently.
 *
 * MD5 schedule is LITTLE-ENDIAN; md5_block returns LE uints; CPU oracle
 * mymd5() stores LE bytes; harness reinterprets as LE uints -> direct
 * byte-exact match. NO state byte-swap. */
static int emit_outer_md5_concat_then_hash(char **out,
                                           size_t *cap, size_t *len)
{
    int rc;

    rc = hx_appendf(out, cap, len,
        "// hx: helper outer_md5_concat_then_hash -- MD5 over the\n"
        "// multi-emit outer message. sep selects the variant:\n"
        "//   sep==0 : hex32(md5(pass)) || pass            (canonical)\n"
        "//   sep==1 : hex32(md5(pass)) || ':' || pass     (colon)\n"
        "// Output: 4 uints (h0..h3). MD5 schedule is LITTLE-ENDIAN;\n"
        "// md5_block returns LE uints natively; NO state byte-swap.\n"
        "// The hex32 prefix occupies logical bytes [0..31]; when sep==1\n"
        "// a single ':' byte sits at logical position 32 and pass starts\n"
        "// at position 33. base = 32 + sep is the logical start of pass.\n"
        "static void outer_md5_concat_then_hash(\n"
        "    uint mma, uint mmb, uint mmc, uint mmd,\n"
        "    __global const uchar *pass, int plen, int sep,\n"
        "    uint *h0, uint *h1, uint *h2, uint *h3)\n"
        "{\n"
        "    if (plen < 0) plen = 0;\n"
        "    if (plen > HX_FAMILY_MAX_PASS) plen = HX_FAMILY_MAX_PASS;\n"
        "    int base = 32 + sep;               // logical start of pass\n"
        "    int total_len = base + plen;\n"
        "\n"
        "    // MD5 initial state.\n"
        "    uint a = 0x67452301u, b = 0xEFCDAB89u, c = 0x98BADCFEu, d = 0x10325476u;\n"
        "\n"
        "    uchar inner_hex[32];\n"
        "    state_to_hex32_bytes(mma, mmb, mmc, mmd, inner_hex);\n"
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
        "    // p_in_first = how many pass bytes land in the first block.\n"
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
        "        md5_block(&a, &b, &c, &d, M);\n"
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
        "    md5_block(&a, &b, &c, &d, M);\n"
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
        "        md5_block(&a, &b, &c, &d, M);\n"
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
        "        md5_block(&a, &b, &c, &d, M);\n"
        "    } else {\n"
        "        md5_block(&a, &b, &c, &d, M);\n"
        "        for (int w = 0; w < 16; w++) M[w] = 0u;\n"
        "        ulong bitlen = (ulong)total_len * 8u;\n"
        "        M[14] = (uint)(bitlen & 0xffffffffu);\n"
        "        M[15] = (uint)(bitlen >> 32);\n"
        "        md5_block(&a, &b, &c, &d, M);\n"
        "    }\n"
        "    // MD5 state is LE; direct copy.\n"
        "    *h0 = a; *h1 = b; *h2 = c; *h3 = d;\n"
        "}\n"
        "\n");
    return rc;
}

/* Per-primitive outer body emit: RIPEMD-160 (LE-schedule, 5-uint state). */
/* Per-primitive outer body emit: RMD128 (LE-schedule, 4-uint state).
 * Sub-phase 5b.1b (2026-05-27) Tier 1. Clone of the RMD-160 helper
 * adjusted for 4-uint state (drop h4 / e1 / state[4]); the underlying
 * rmd128_block primitive resident in gpu_common.cl rev 1.27 carries
 * the F4->F3->F2->F1 right-pipeline ordering correctly (Bosselaers
 * Table 4) so the emit helper does not need its own ordering knowledge
 * -- it only needs to drive the standard 64-byte LE message schedule
 * + length suffix + padding. CPU oracle RIPEMD128 returns 16 LE bytes;
 * harness reinterprets as 4 LE uints -> direct byte-exact match. */
static int emit_outer_rmd128_concat_then_hash(char **out,
                                              size_t *cap, size_t *len)
{
    int rc;

    rc = hx_appendf(out, cap, len,
        "// hx: helper outer_rmd128_concat_then_hash -- RIPEMD-128 over\n"
        "// (hex32(md5(pass)) || pass). Total input length = 32 + plen.\n"
        "// Output: 4 uints (h0..h3). RIPEMD-128 schedule is LITTLE-\n"
        "// ENDIAN per spec; rmd128_block returns LE uints; CPU oracle\n"
        "// RIPEMD128() stores LE bytes -> harness reinterprets as LE\n"
        "// uints -> kernel state matches directly. NO byte-swap.\n"
        "static void outer_rmd128_concat_then_hash(\n"
        "    uint mma, uint mmb, uint mmc, uint mmd,\n"
        "    __global const uchar *pass, int plen,\n"
        "    uint *h0, uint *h1, uint *h2, uint *h3)\n"
        "{\n"
        "    if (plen < 0) plen = 0;\n"
        "    if (plen > HX_FAMILY_MAX_PASS) plen = HX_FAMILY_MAX_PASS;\n"
        "    int total_len = 32 + plen;\n"
        "\n"
        "    // RIPEMD-128 initial state -- same first 4 as MD5 / MD4.\n"
        "    uint state[4];\n"
        "    state[0] = 0x67452301u;\n"
        "    state[1] = 0xEFCDAB89u;\n"
        "    state[2] = 0x98BADCFEu;\n"
        "    state[3] = 0x10325476u;\n"
        "\n"
        "    uchar inner_hex[32];\n"
        "    state_to_hex32_bytes(mma, mmb, mmc, mmd, inner_hex);\n"
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
        "        // RIPEMD-128 length is LITTLE-ENDIAN 64-bit.\n"
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
        "    // rmd128.c that previously required a bug-compatible\n"
        "    // encoding (bug_lswlen = leftover bytes) was fixed in\n"
        "    // rmd128.c on 2026-05-27 -- the CPU now also encodes the\n"
        "    // total bit-length, matching Bosselaers's 1996 reference\n"
        "    // and sph_ripemd128. GPU emit therefore uses total_len*8.\n"
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

static int emit_outer_rmd160_concat_then_hash(char **out,
                                              size_t *cap, size_t *len)
{
    int rc;

    rc = hx_appendf(out, cap, len,
        "// hx: helper outer_rmd160_concat_then_hash -- RIPEMD-160 over\n"
        "// (hex32(md5(pass)) || pass). Total input length = 32 + plen.\n"
        "// Output: 5 uints (h0..h4). RIPEMD-160 schedule is LITTLE-\n"
        "// ENDIAN per spec; rmd160_block returns LE uints; CPU oracle\n"
        "// RIPEMD160() stores LE bytes -> harness reinterprets as LE\n"
        "// uints -> kernel state matches directly. NO byte-swap.\n"
        "static void outer_rmd160_concat_then_hash(\n"
        "    uint mma, uint mmb, uint mmc, uint mmd,\n"
        "    __global const uchar *pass, int plen,\n"
        "    uint *h0, uint *h1, uint *h2, uint *h3, uint *h4)\n"
        "{\n"
        "    if (plen < 0) plen = 0;\n"
        "    if (plen > HX_FAMILY_MAX_PASS) plen = HX_FAMILY_MAX_PASS;\n"
        "    int total_len = 32 + plen;\n"
        "\n"
        "    // RIPEMD-160 initial state (RFC: same first 4 as MD5; 5th\n"
        "    // is 0xC3D2E1F0u, same as SHA1).\n"
        "    uint state[5];\n"
        "    state[0] = 0x67452301u;\n"
        "    state[1] = 0xEFCDAB89u;\n"
        "    state[2] = 0x98BADCFEu;\n"
        "    state[3] = 0x10325476u;\n"
        "    state[4] = 0xC3D2E1F0u;\n"
        "\n"
        "    uchar inner_hex[32];\n"
        "    state_to_hex32_bytes(mma, mmb, mmc, mmd, inner_hex);\n"
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
        "        // RIPEMD-160 length is LITTLE-ENDIAN 64-bit.\n"
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

/* Shared SHA-256/SHA-224 emit body (parametrized by IV).
 *
 * sha224_block does not exist; we use sha256_block with a different
 * starting state. Output is 8 uints; compact_fp probe uses first 4.
 * Both algorithms write the first 4 BE-state-bytes-as-LE-uint words.
 * Block size 64; length suffix BE 8 bytes at M[14]/M[15].
 *
 * Caller supplies fn_name (outer_sha224_concat_then_hash or
 * outer_sha256_concat_then_hash) + the 8 IV constants as a string. */
static int emit_outer_sha2_32_concat_then_hash(char **out,
                                               size_t *cap, size_t *len,
                                               const char *fn_name,
                                               const char *iv_init_str,
                                               const char *primitive_label)
{
    int rc;

    rc = hx_appendf(out, cap, len,
        "// hx: helper %s -- %s over\n"
        "// (hex32(md5(pass)) || pass). Total input length = 32 + plen.\n"
        "// Output: 8 uints internally; we write first 4 LE-swapped to\n"
        "// h0..h3 for compact_fp probe. BE-schedule (same word-pack as\n"
        "// SHA1). State byte-swap REQUIRED per\n"
        "// [[feedback-be-state-primitives-need-byteswap-in-codegen]].\n"
        "static void %s(\n"
        "    uint mma, uint mmb, uint mmc, uint mmd,\n"
        "    __global const uchar *pass, int plen,\n"
        "    uint *h0, uint *h1, uint *h2, uint *h3)\n"
        "{\n"
        "    if (plen < 0) plen = 0;\n"
        "    if (plen > HX_FAMILY_MAX_PASS) plen = HX_FAMILY_MAX_PASS;\n"
        "    int total_len = 32 + plen;\n"
        "\n"
        "    uint state[8];\n"
        "%s"
        "\n"
        "    uchar inner_hex[32];\n"
        "    state_to_hex32_bytes(mma, mmb, mmc, mmd, inner_hex);\n"
        "\n"
        "    uint M[16];\n"
        "    int byte_pos = 0;\n"
        "    int pass_consumed = 0;\n"
        "    int first_has_pad = 0;\n"
        "\n"
        "    int p_in_first = plen;\n"
        "    if (p_in_first > 32) p_in_first = 32;\n"
        "    {\n"
        "        // BE message-word pack (b0<<24 | b1<<16 | b2<<8 | b3).\n"
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
        "            int sh = (3 - (abs_pos & 3)) * 8;  // BE\n"
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
        "        // BE state -> LE uints (first 4 only; probe uses h0..h3).\n"
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
        "}\n"
        "\n",
        fn_name, primitive_label, fn_name, iv_init_str);
    return rc;
}

static int emit_outer_sha224_concat_then_hash(char **out,
                                              size_t *cap, size_t *len)
{
    /* SHA-224 IV (FIPS 180-4 §6.3.1). */
    static const char *iv =
        "    state[0] = 0xc1059ed8u; state[1] = 0x367cd507u;\n"
        "    state[2] = 0x3070dd17u; state[3] = 0xf70e5939u;\n"
        "    state[4] = 0xffc00b31u; state[5] = 0x68581511u;\n"
        "    state[6] = 0x64f98fa7u; state[7] = 0xbefa4fa4u;\n";
    return emit_outer_sha2_32_concat_then_hash(out, cap, len,
        "outer_sha224_concat_then_hash", iv, "SHA-224");
}

static int emit_outer_sha256_concat_then_hash(char **out,
                                              size_t *cap, size_t *len)
{
    /* SHA-256 IV (FIPS 180-4 §5.3.3). */
    static const char *iv =
        "    state[0] = 0x6a09e667u; state[1] = 0xbb67ae85u;\n"
        "    state[2] = 0x3c6ef372u; state[3] = 0xa54ff53au;\n"
        "    state[4] = 0x510e527fu; state[5] = 0x9b05688cu;\n"
        "    state[6] = 0x1f83d9abu; state[7] = 0x5be0cd19u;\n";
    return emit_outer_sha2_32_concat_then_hash(out, cap, len,
        "outer_sha256_concat_then_hash", iv, "SHA-256");
}

/* Shared SHA-512/SHA-384 emit body (parametrized by IV).
 *
 * Block size 128 bytes; state is 8 ulongs (64-bit each). Compact_fp
 * probe uses first 4 LE uints = first 16 bytes = first 2 ulongs of
 * BE state. We byte-swap each ulong, then split each into 2 LE uints:
 *   ulong s = state[i] (BE 64-bit)
 *   swapped = bswap64(s) - puts byte 0 of digest in low 8 bits
 *   h_even = (uint)(swapped & 0xffffffff)
 *   h_odd  = (uint)(swapped >> 32)
 * For probe order [h0,h1,h2,h3] = [low(s[0]), high(s[0]), low(s[1]), high(s[1])].
 *
 * Length suffix: 16 bytes at M[14]/M[15] BE-packed (M[14] holds high
 * 64 bits, always 0 for plen < 2^29). Padding fits in single block
 * iff total_len + 1 + 16 <= 128 -> plen <= 79.
 *
 * Hex32 (32 B inner) occupies M[0..3] (each M is 8 bytes = 1 ulong);
 * the rest of the first 128B block holds zero-padded pass bytes.  */
static int emit_outer_sha2_64_concat_then_hash(char **out,
                                               size_t *cap, size_t *len,
                                               const char *fn_name,
                                               const char *iv_init_str,
                                               const char *primitive_label)
{
    int rc;

    rc = hx_appendf(out, cap, len,
        "// hx: helper %s -- %s over\n"
        "// (hex32(md5(pass)) || pass). Total input length = 32 + plen.\n"
        "// Output: 8 ulongs internally; we write first 4 LE-uints to\n"
        "// h0..h3 for compact_fp probe (= first 16 bytes of digest =\n"
        "// first 2 ulongs of BE state, byte-swap-as-ulong then split).\n"
        "// Block size 128; length suffix 16 bytes BE at M[14]/M[15].\n"
        "// BE-schedule state byte-swap REQUIRED.\n"
        "static void %s(\n"
        "    uint mma, uint mmb, uint mmc, uint mmd,\n"
        "    __global const uchar *pass, int plen,\n"
        "    uint *h0, uint *h1, uint *h2, uint *h3)\n"
        "{\n"
        "    if (plen < 0) plen = 0;\n"
        "    if (plen > HX_FAMILY_MAX_PASS) plen = HX_FAMILY_MAX_PASS;\n"
        "    int total_len = 32 + plen;\n"
        "\n"
        "    ulong state[8];\n"
        "%s"
        "\n"
        "    uchar inner_hex[32];\n"
        "    state_to_hex32_bytes(mma, mmb, mmc, mmd, inner_hex);\n"
        "\n"
        "    // 128-byte SHA-512 block, 16 ulong message words. M[w]\n"
        "    // packs 8 bytes BE (b0<<56 | ... | b7).\n"
        "    ulong M[16];\n"
        "    int byte_pos = 0;\n"
        "    int pass_consumed = 0;\n"
        "    int first_has_pad = 0;\n"
        "\n"
        "    int p_in_first = plen;\n"
        "    if (p_in_first > 96) p_in_first = 96;  // 128 - 32 = 96\n"
        "    {\n"
        "        // M[0..3]: hex32 (32 B) packed BE per ulong.\n"
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
        "            int sh = (7 - (abs_pos & 7)) * 8;  // BE\n"
        "            M[wi] |= v << sh;\n"
        "        }\n"
        "        pass_consumed = p_in_first;\n"
        "        byte_pos = 32 + p_in_first;\n"
        "    }\n"
        "\n"
        "    // Single-block fast path: total_len + 1 + 16 <= 128 -> plen <= 79.\n"
        "    if (total_len + 1 + 16 <= 128) {\n"
        "        int pad_pos = byte_pos;\n"
        "        int wi = pad_pos >> 3;\n"
        "        int sh = (7 - (pad_pos & 7)) * 8;\n"
        "        M[wi] |= ((ulong)0x80u) << sh;\n"
        "        ulong bitlen = (ulong)total_len * 8ul;\n"
        "        M[14] = 0ul;       // high 64 bits, plen always fits in low\n"
        "        M[15] = bitlen;    // low 64 bits, BE-packed already (ulong)\n"
        "        sha512_block(state, M);\n"
        "        // BE state -> LE uint pairs. swap-as-ulong then split.\n"
        "        ulong s0 = state[0], s1 = state[1];\n"
        "        ulong sw0 = ((s0 & 0x00000000000000ffUL) << 56) |\n"
        "                    ((s0 & 0x000000000000ff00UL) << 40) |\n"
        "                    ((s0 & 0x0000000000ff0000UL) << 24) |\n"
        "                    ((s0 & 0x00000000ff000000UL) <<  8) |\n"
        "                    ((s0 & 0x000000ff00000000UL) >>  8) |\n"
        "                    ((s0 & 0x0000ff0000000000UL) >> 24) |\n"
        "                    ((s0 & 0x00ff000000000000UL) >> 40) |\n"
        "                    ((s0 & 0xff00000000000000UL) >> 56);\n"
        "        ulong sw1 = ((s1 & 0x00000000000000ffUL) << 56) |\n"
        "                    ((s1 & 0x000000000000ff00UL) << 40) |\n"
        "                    ((s1 & 0x0000000000ff0000UL) << 24) |\n"
        "                    ((s1 & 0x00000000ff000000UL) <<  8) |\n"
        "                    ((s1 & 0x000000ff00000000UL) >>  8) |\n"
        "                    ((s1 & 0x0000ff0000000000UL) >> 24) |\n"
        "                    ((s1 & 0x00ff000000000000UL) >> 40) |\n"
        "                    ((s1 & 0xff00000000000000UL) >> 56);\n"
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
        "        ulong sw0 = ((s0 & 0x00000000000000ffUL) << 56) |\n"
        "                    ((s0 & 0x000000000000ff00UL) << 40) |\n"
        "                    ((s0 & 0x0000000000ff0000UL) << 24) |\n"
        "                    ((s0 & 0x00000000ff000000UL) <<  8) |\n"
        "                    ((s0 & 0x000000ff00000000UL) >>  8) |\n"
        "                    ((s0 & 0x0000ff0000000000UL) >> 24) |\n"
        "                    ((s0 & 0x00ff000000000000UL) >> 40) |\n"
        "                    ((s0 & 0xff00000000000000UL) >> 56);\n"
        "        ulong sw1 = ((s1 & 0x00000000000000ffUL) << 56) |\n"
        "                    ((s1 & 0x000000000000ff00UL) << 40) |\n"
        "                    ((s1 & 0x0000000000ff0000UL) << 24) |\n"
        "                    ((s1 & 0x00000000ff000000UL) <<  8) |\n"
        "                    ((s1 & 0x000000ff00000000UL) >>  8) |\n"
        "                    ((s1 & 0x0000ff0000000000UL) >> 24) |\n"
        "                    ((s1 & 0x00ff000000000000UL) >> 40) |\n"
        "                    ((s1 & 0xff00000000000000UL) >> 56);\n"
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

static int emit_outer_sha384_concat_then_hash(char **out,
                                              size_t *cap, size_t *len)
{
    /* SHA-384 IV (FIPS 180-4 §5.3.4). */
    static const char *iv =
        "    state[0] = 0xcbbb9d5dc1059ed8UL; state[1] = 0x629a292a367cd507UL;\n"
        "    state[2] = 0x9159015a3070dd17UL; state[3] = 0x152fecd8f70e5939UL;\n"
        "    state[4] = 0x67332667ffc00b31UL; state[5] = 0x8eb44a8768581511UL;\n"
        "    state[6] = 0xdb0c2e0d64f98fa7UL; state[7] = 0x47b5481dbefa4fa4UL;\n";
    return emit_outer_sha2_64_concat_then_hash(out, cap, len,
        "outer_sha384_concat_then_hash", iv, "SHA-384");
}

static int emit_outer_sha512_concat_then_hash(char **out,
                                              size_t *cap, size_t *len)
{
    /* SHA-512 IV (FIPS 180-4 §5.3.5). */
    static const char *iv =
        "    state[0] = 0x6a09e667f3bcc908UL; state[1] = 0xbb67ae8584caa73bUL;\n"
        "    state[2] = 0x3c6ef372fe94f82bUL; state[3] = 0xa54ff53a5f1d36f1UL;\n"
        "    state[4] = 0x510e527fade682d1UL; state[5] = 0x9b05688c2b3e6c1fUL;\n"
        "    state[6] = 0x1f83d9abfb41bd6bUL; state[7] = 0x5be0cd19137e2179UL;\n";
    return emit_outer_sha2_64_concat_then_hash(out, cap, len,
        "outer_sha512_concat_then_hash", iv, "SHA-512");
}

/* Whirlpool emit helper. Phase 5b Tier 2 sub-phase 5b.2a.3 (2026-05-27).
 *
 * Bespoke per D16.3.a. Differs from emit_outer_sha512 in 4 key ways:
 *   1. Block size 64 (not 128); 8 ulong words per block.
 *   2. Length suffix is 32 bytes BE at M[4..7]; in practice the
 *      family use case never exceeds 2^64 bits so M[4..6] = 0 and
 *      M[7] = bitlen.
 *   3. State IV is all zero (Whirlpool spec); not the SHA-2 IVs.
 *   4. ALWAYS multi-block: single-block fast path threshold
 *      32 + plen + 1 + 32 <= 64 -> plen <= -1 never holds. The fast
 *      path branch is elided entirely. See Tier 2 spec §3 D16.3.
 *
 * State output is BE bytes of state[0..7]; first 16 bytes for the
 * compact_fp probe come from state[0..1] via byte-swap-as-ulong then
 * LE-uint split (identical epilogue to the sha2_64 helper). */
static int emit_outer_wrl_concat_then_hash(char **out,
                                           size_t *cap, size_t *len)
{
    int rc;

    rc = hx_appendf(out, cap, len,
        "// hx: helper outer_wrl_concat_then_hash -- Whirlpool over\n"
        "// (hex32(md5(pass)) || pass). Total input length = 32 + plen.\n"
        "// Output: 8 ulongs internally (Whirlpool full state). First 4\n"
        "// LE-uints written to h0..h3 for compact_fp probe (= first 16\n"
        "// bytes of digest = first 2 ulongs of BE state, byte-swap as\n"
        "// ulong then split). Block size 64; length suffix 32 bytes BE\n"
        "// at M[4..7] (M[4..6] always zero for family use case).\n"
        "// ALWAYS multi-block (single-block fast path elided per Tier 2\n"
        "// spec finding: 32+plen+1+32 <= 64 never holds).\n"
        "static void outer_wrl_concat_then_hash(\n"
        "    uint mma, uint mmb, uint mmc, uint mmd,\n"
        "    __global const uchar *pass, int plen,\n"
        "    uint *h0, uint *h1, uint *h2, uint *h3)\n"
        "{\n"
        "    if (plen < 0) plen = 0;\n"
        "    if (plen > HX_FAMILY_MAX_PASS) plen = HX_FAMILY_MAX_PASS;\n"
        "    int total_len = 32 + plen;\n"
        "\n"
        "    // Whirlpool IV: all-zero.\n"
        "    ulong state[8];\n"
        "    for (int i = 0; i < 8; i++) state[i] = 0ul;\n"
        "\n"
        "    uchar inner_hex[32];\n"
        "    state_to_hex32_bytes(mma, mmb, mmc, mmd, inner_hex);\n"
        "\n"
        "    // 64-byte Whirlpool block, 8 ulong message words. M[w]\n"
        "    // packs 8 bytes BE (b0<<56 | ... | b7).\n"
        "    ulong M[8];\n"
        "    int byte_pos = 0;\n"
        "    int pass_consumed = 0;\n"
        "    int first_has_pad = 0;\n"
        "\n"
        "    // p_in_first = how many pass bytes fit in the first block\n"
        "    // alongside the 32-byte hex32 prefix. Block-cap = 64 - 32 = 32.\n"
        "    int p_in_first = plen;\n"
        "    if (p_in_first > 32) p_in_first = 32;\n"
        "    {\n"
        "        // M[0..3]: hex32 (32 B) packed BE per ulong.\n"
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
        "            int sh = (7 - (abs_pos & 7)) * 8;  // BE\n"
        "            M[wi] |= v << sh;\n"
        "        }\n"
        "        pass_consumed = p_in_first;\n"
        "        byte_pos = 32 + p_in_first;\n"
        "    }\n"
        "\n"
        "    // ALWAYS multi-block (per Tier 2 spec elision finding).\n"
        "    // If pass fully fit in first block AND there's room for 0x80,\n"
        "    // mark first_has_pad so the tail loop doesn't re-emit it.\n"
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
        "    // Tail block(s): pack remaining pass bytes, set 0x80 pad if\n"
        "    // not already in first block, append 32-byte BE length suffix\n"
        "    // at M[4..7]. If pad+length don't fit in one block, emit two.\n"
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
        "    // Length-pad fit: need pleft + 1 (pad) + 32 (length) <= 64,\n"
        "    // i.e. pleft <= 31. If first_has_pad we already paid the +1\n"
        "    // upstream so the threshold is pleft + 32 <= 64 -> pleft <= 32.\n"
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
        "    // BE state -> LE uint pairs. Byte-swap state[0..1] as ulong,\n"
        "    // then split into pair of LE uints (identical epilogue to\n"
        "    // emit_outer_sha2_64_concat_then_hash).\n"
        "    {\n"
        "        ulong s0 = state[0], s1 = state[1];\n"
        "        ulong sw0 = ((s0 & 0x00000000000000ffUL) << 56) |\n"
        "                    ((s0 & 0x000000000000ff00UL) << 40) |\n"
        "                    ((s0 & 0x0000000000ff0000UL) << 24) |\n"
        "                    ((s0 & 0x00000000ff000000UL) <<  8) |\n"
        "                    ((s0 & 0x000000ff00000000UL) >>  8) |\n"
        "                    ((s0 & 0x0000ff0000000000UL) >> 24) |\n"
        "                    ((s0 & 0x00ff000000000000UL) >> 40) |\n"
        "                    ((s0 & 0xff00000000000000UL) >> 56);\n"
        "        ulong sw1 = ((s1 & 0x00000000000000ffUL) << 56) |\n"
        "                    ((s1 & 0x000000000000ff00UL) << 40) |\n"
        "                    ((s1 & 0x0000000000ff0000UL) << 24) |\n"
        "                    ((s1 & 0x00000000ff000000UL) <<  8) |\n"
        "                    ((s1 & 0x000000ff00000000UL) >>  8) |\n"
        "                    ((s1 & 0x0000ff0000000000UL) >> 24) |\n"
        "                    ((s1 & 0x00ff000000000000UL) >> 40) |\n"
        "                    ((s1 & 0xff00000000000000UL) >> 56);\n"
        "        *h0 = (uint)(sw0 & 0xffffffffUL);\n"
        "        *h1 = (uint)(sw0 >> 32);\n"
        "        *h2 = (uint)(sw1 & 0xffffffffUL);\n"
        "        *h3 = (uint)(sw1 >> 32);\n"
        "    }\n"
        "}\n"
        "\n");
    return rc;
}

/* Tiger emit helper. Phase 5b Tier 2 sub-phase 5b.2b.3 (2026-05-27).
 *
 * Bespoke per D16.3.a. Differs from emit_outer_sha512 / emit_outer_wrl in
 * 5 key ways:
 *   1. LE schedule (M packed LE, lowest-byte-first). Tiger spec is LE,
 *      matching MD-family convention; OPPOSITE to Whirlpool/SHA-2 BE.
 *   2. Length suffix is 8 bytes LE at M[7]. M[7] = bitlen as LE ulong.
 *   3. Padding byte is 0x01 (legacy Tiger, NOT Tiger2's 0x80). This is
 *      a critical distinction -- the e171 catalog entry uses Tiger (not
 *      Tiger2); mdxfind's CPU oracle calls sph_tiger_close (the Tiger
 *      variant), and sph_tiger_close uses 0x01 padding (sph_tiger.c).
 *   4. State IV is the Tiger initial chaining value
 *      (0x0123456789abcdefUL, 0xfedcba9876543210UL, 0xf096a5b4c3b2e187UL);
 *      3-ulong state (not 8 like Whirlpool).
 *   5. Single-block fast path APPLICABLE for plen <= 23 (threshold
 *      32 + plen + 1 + 8 <= 64 -> plen <= 23). Unlike Whirlpool which
 *      ALWAYS multi-blocks. Common case (short passwords) takes the
 *      fast path.
 *
 * State output is LE bytes of state[0..2]; first 16 bytes for the
 * compact_fp probe come from state[0..1] DIRECTLY (no byte-swap) as
 * (state[0] lo32, state[0] hi32, state[1] lo32, state[1] hi32). State[2]
 * holds bytes 16..23 which the GPU dispatch discards; the CPU recompute
 * path (5a.5 _proto_hexlen = 48 wiring) fills the remaining 8 bytes on
 * hit for byte-exact full-width verification. */
static int emit_outer_tiger_concat_then_hash(char **out,
                                             size_t *cap, size_t *len)
{
    int rc;

    rc = hx_appendf(out, cap, len,
        "// hx: helper outer_tiger_concat_then_hash -- Tiger over\n"
        "// (hex32(md5(pass)) || pass). Total input length = 32 + plen.\n"
        "// Output: 3 ulongs internally (Tiger full state). First 4\n"
        "// LE-uints written to h0..h3 for compact_fp probe (= first 16\n"
        "// bytes of digest = first 2 ulongs of LE state DIRECTLY). LE\n"
        "// schedule; 0x01 pad byte (legacy Tiger, not Tiger2 0x80);\n"
        "// 8-byte LE length suffix at M[7]. Single-block fast path for\n"
        "// plen <= 23 (threshold 32+plen+1+8 <= 64).\n"
        "static void outer_tiger_concat_then_hash(\n"
        "    uint mma, uint mmb, uint mmc, uint mmd,\n"
        "    __global const uchar *pass, int plen,\n"
        "    uint *h0, uint *h1, uint *h2, uint *h3)\n"
        "{\n"
        "    if (plen < 0) plen = 0;\n"
        "    if (plen > HX_FAMILY_MAX_PASS) plen = HX_FAMILY_MAX_PASS;\n"
        "    int total_len = 32 + plen;\n"
        "\n"
        "    // Tiger IV (Anderson + Biham 1996; matches sph_tiger_init\n"
        "    // and rhash_tiger_init).\n"
        "    ulong state[3];\n"
        "    state[0] = 0x0123456789abcdefUL;\n"
        "    state[1] = 0xfedcba9876543210UL;\n"
        "    state[2] = 0xf096a5b4c3b2e187UL;\n"
        "\n"
        "    uchar inner_hex[32];\n"
        "    state_to_hex32_bytes(mma, mmb, mmc, mmd, inner_hex);\n"
        "\n"
        "    // 64-byte Tiger block, 8 ulong message words. M[w]\n"
        "    // packs 8 bytes LE (b0 | b1<<8 | ... | b7<<56).\n"
        "    ulong M[8];\n"
        "    int byte_pos = 0;\n"
        "    int pass_consumed = 0;\n"
        "    int first_has_pad = 0;\n"
        "\n"
        "    int p_in_first = plen;\n"
        "    if (p_in_first > 32) p_in_first = 32;\n"
        "    {\n"
        "        // M[0..3]: hex32 (32 B) packed LE per ulong.\n"
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
        "            int sh = (abs_pos & 7) * 8;  // LE\n"
        "            M[wi] |= v << sh;\n"
        "        }\n"
        "        pass_consumed = p_in_first;\n"
        "        byte_pos = 32 + p_in_first;\n"
        "    }\n"
        "\n"
        "    // Single-block fast path: plen <= 23 lets pad + 8-byte LE\n"
        "    // length suffix fit in the same block as the (32 + plen)\n"
        "    // bytes of data. Threshold: byte_pos + 1 + 8 <= 64.\n"
        "    if (p_in_first == plen && byte_pos + 1 + 8 <= 64) {\n"
        "        int pad_pos = byte_pos;\n"
        "        int wi = pad_pos >> 3;\n"
        "        int sh = (pad_pos & 7) * 8;\n"
        "        // 0x01 padding byte (legacy Tiger, NOT Tiger2 0x80).\n"
        "        M[wi] |= ((ulong)0x01u) << sh;\n"
        "        // LE bitlen at M[7].\n"
        "        ulong bitlen = (ulong)total_len * 8ul;\n"
        "        M[7] = bitlen;\n"
        "        tiger_block(state, M);\n"
        "    } else {\n"
        "        // Multi-block path: emit first block with pad bit if room,\n"
        "        // then consume remaining pass bytes in 64-byte chunks,\n"
        "        // then a tail block with pad (if not already) + 8-byte LE\n"
        "        // length suffix at M[7].\n"
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
        "        // Tail block(s): pack remaining pass bytes, set 0x01 pad if\n"
        "        // not already in first block, append 8-byte LE length\n"
        "        // suffix at M[7]. If pad+length don't fit in one block,\n"
        "        // emit two.\n"
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
        "    // LE state output direct extract -- no byte swap epilogue\n"
        "    // (unlike sha512/wrl). h0..h3 = first 16 bytes = state[0..1].\n"
        "    *h0 = (uint)(state[0] & 0xffffffffUL);\n"
        "    *h1 = (uint)(state[0] >> 32);\n"
        "    *h2 = (uint)(state[1] & 0xffffffffUL);\n"
        "    *h3 = (uint)(state[1] >> 32);\n"
        "}\n"
        "\n");
    return rc;
}

/* HAVAL emit helper. Phase 5b Tier 3 sub-phase 5b.3a.3 (2026-05-27).
 *
 * PARAMETERISED per D17.1.a: ONE C-side helper emits a GPU function
 * specialised on (passes, digest_bytes). The 15 HAVAL variants (5 widths
 * x 3 pass counts) all route through this single helper; each emit call
 * produces a distinct GPU function body. Sub-phase 5b.3a ships the 5
 * 3-pass variants (passes==3); 5b.3b + 5b.3c extend the haval<P>_block
 * dispatch to passes==4 and passes==5.
 *
 * The emitted GPU function is named `outer_haval_concat_then_hash`
 * (single fixed name -- only ONE HAV primitive is emitted per kernel
 * since each codegen JIT specialises to one JOB enum). It computes:
 *
 *     HAVAL-<W*8>/<P> ( hex32(md5(pass)) || pass )
 *
 * over the 128-byte HAVAL block (twice the 64-byte MD-family block).
 *
 * CRITICAL HAVAL specifics (per Tier 3 spec §3 + donor mhash haval.c):
 *
 *   - 128-byte block, 32 LE-packed uint32 message words M[0..31].
 *   - PAD-TOGGLE byte is 0x01 NOT 0x80 (donor havalFinal:760
 *     "corrected from 0x80"). EVERY other primitive in gpu_common.cl
 *     uses 0x80; HAVAL is the exception. Wrong toggle = silently wrong
 *     digest for ALL inputs.
 *   - block[118..119] PARAMETER ENCODING (donor havalFinal:786-790):
 *       block[118] = ((hashLength & 0x03) << 6) | ((passes & 0x07) << 3)
 *                    | (HAVAL_VERSION=1 & 0x07)
 *       block[119] = hashLength >> 2
 *     where hashLength = digest_bytes * 8 (in bits). Each (W, P) tuple
 *     produces a DIFFERENT 2-byte encoding -- the most likely Tier 3
 *     bug class (R1). Computed at C-emit time and baked into the
 *     emitted source as literal constants.
 *   - 64-bit message bitlen LE at block[120..127].
 *   - POST-COMPRESSION DIGEST FOLD per width (donor havalFinal:816-911):
 *       128-bit: heavy byte-redistribution fold of state[4..7] into [0..3]
 *       160-bit: ROTR-using fold
 *       192-bit: 5-bit-slice fold
 *       224-bit: byte-slot-shift fold
 *       256-bit: NO fold (direct state output)
 *     JIT-specialised per digest_bytes -- each emitted kernel has exactly
 *     ONE fold branch (no runtime conditional).
 *   - Output: first 16 bytes (h0..h3) of the FOLDED state go to the hit
 *     record; for widths > 16 bytes the CPU recompute fills the rest.
 *
 * This helper's C-mirror was validated 60/60 cells PASS vs sph_haval
 * (5 widths x 12 inputs incl multi-block boundary cases) in 5b.3a.1.
 */
static int emit_outer_haval_concat_then_hash(char **out, size_t *cap,
                                             size_t *len,
                                             int passes, int digest_bytes)
{
    int rc;
    int hashbits = digest_bytes * 8;

    /* R1 mitigation: compute the block[118..119] parameter bytes at
     * C-emit time. Each (W, P) variant produces a distinct pair. */
    int byte118 = ((hashbits & 0x03) << 6) | ((passes & 0x07) << 3) | (1 & 0x07);
    int byte119 = (hashbits >> 2) & 0xff;

    /* Banner + signature + prologue. Block packing identical for all
     * widths/passes; only the compression-function call + fold differ. */
    rc = hx_appendf(out, cap, len,
        "// hx: helper outer_haval_concat_then_hash -- HAVAL-%d/%d over\n"
        "// (hex32(md5(pass)) || pass). Total input length = 32 + plen.\n"
        "// 128-byte HAVAL block; 32 LE-packed uint32 message words.\n"
        "// PAD-TOGGLE is 0x01 NOT 0x80 (see donor mhash haval.c:760).\n"
        "// block[118..119] parameter encoding for THIS variant:\n"
        "//   byte118 = 0x%02x  (hashLength=%d passes=%d version=1)\n"
        "//   byte119 = 0x%02x  (hashLength>>2)\n"
        "// Output: first 16 bytes (h0..h3) of folded state for probe;\n"
        "// CPU recompute fills the remaining %d bytes on hit.\n"
        "static void outer_haval_concat_then_hash(\n"
        "    uint mma, uint mmb, uint mmc, uint mmd,\n"
        "    __global const uchar *pass, int plen,\n"
        "    uint *h0, uint *h1, uint *h2, uint *h3)\n"
        "{\n"
        "    if (plen < 0) plen = 0;\n"
        "    if (plen > HX_FAMILY_MAX_PASS) plen = HX_FAMILY_MAX_PASS;\n"
        "    int total_len = 32 + plen;\n"
        "\n"
        "    // HAVAL IV (8 uints; Pi-fractional constants).\n"
        "    uint state[8];\n"
        "    for (int i = 0; i < 8; i++) state[i] = HAVAL_IV[i];\n"
        "\n"
        "    uchar inner_hex[32];\n"
        "    state_to_hex32_bytes(mma, mmb, mmc, mmd, inner_hex);\n"
        "\n"
        "    // 128-byte HAVAL block as 32 LE-packed uint32 words.\n"
        "    // M[w] = b0 | b1<<8 | b2<<16 | b3<<24 (LITTLE-endian).\n"
        "    uint M[32];\n"
        "    uchar block[128];\n"
        "    int consumed = 0;\n"
        "\n"
        "    // Process all full 128-byte blocks of (hex32 || pass).\n"
        "    // The combined message is inner_hex[0..31] then pass[0..plen).\n"
        "    // total_len = 32 + plen. Walk it in 128-byte chunks; the\n"
        "    // first 32 bytes come from inner_hex, the rest from pass.\n"
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
        "    // Tail block: remaining (total_len - consumed) bytes, then\n"
        "    // the 0x01 pad toggle, zero-fill, parameter bytes, bitlen.\n"
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
        "        // No room for parameter+length bytes; compress this\n"
        "        // block then start a fresh zeroed one (donor :763-780).\n"
        "        for (int w = 0; w < 32; w++) {\n"
        "            int bo = w * 4;\n"
        "            M[w] = (uint)block[bo] | ((uint)block[bo+1] << 8)\n"
        "                 | ((uint)block[bo+2] << 16) | ((uint)block[bo+3] << 24);\n"
        "        }\n"
        "        haval%d_block(state, M);\n"
        "        for (int i = 0; i < 128; i++) block[i] = 0;\n"
        "    }\n"
        "    // Parameter bytes at block[118..119] for HAVAL-%d/%d.\n"
        "    block[118] = (uchar)0x%02x;\n"
        "    block[119] = (uchar)0x%02x;\n"
        "    // 64-bit message bitlen LE at block[120..127].\n"
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
        byte118, hashbits, passes, byte119,
        digest_bytes,
        passes,         /* full-block compress */
        passes,         /* spill-block compress */
        hashbits, passes,
        byte118, byte119,
        passes);        /* final compress */
    if (rc < 0) return rc;

    /* Post-compression digest fold -- JIT-specialised per digest width.
     * Exactly ONE branch emitted (no runtime conditional). Donor
     * havalFinal:816-911 transcribed. R3 mitigation: copy-paste-no-retype
     * with explicit donor line citation; validated 60/60 in C-mirror. */
    if (digest_bytes == 16) {
        /* 128-bit fold (donor :819-841). */
        rc = hx_appendf(out, cap, len,
        "    // 128-bit digest fold (donor havalFinal:819-841): heavy\n"
        "    // byte-redistribution of state[4..7] into state[0..3].\n"
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
        /* 160-bit fold (donor :848-859). Uses HAVAL_ROTR32. */
        rc = hx_appendf(out, cap, len,
        "    // 160-bit digest fold (donor havalFinal:848-859).\n"
        "    state[4] += ((state[7] & 0xFE000000u) | (state[6] & 0x01F80000u)\n"
        "               | (state[5] & 0x0007F000u)) >> 12;\n"
        "    state[3] += ((state[7] & 0x01F80000u) | (state[6] & 0x0007F000u)\n"
        "               | (state[5] & 0x00000FC0u)) >> 6;\n"
        "    state[2] += ((state[7] & 0x0007F000u) | (state[6] & 0x00000FC0u)\n"
        "               | (state[5] & 0x0000003Fu));\n"
        "    state[1] += HAVAL_ROTR32((state[7] & 0x00000FC0u)\n"
        "               | (state[6] & 0x0000003Fu) | (state[5] & 0xFE000000u), 25);\n"
        "    state[0] += HAVAL_ROTR32((state[7] & 0x0000003Fu)\n"
        "               | (state[6] & 0xFE000000u) | (state[5] & 0x01F80000u), 19);\n");
    } else if (digest_bytes == 24) {
        /* 192-bit fold (donor :868-880). */
        rc = hx_appendf(out, cap, len,
        "    // 192-bit digest fold (donor havalFinal:868-880).\n"
        "    state[5] += ((state[7] & 0xFC000000u) | (state[6] & 0x03E00000u)) >> 21;\n"
        "    state[4] += ((state[7] & 0x03E00000u) | (state[6] & 0x001F0000u)) >> 16;\n"
        "    state[3] += ((state[7] & 0x001F0000u) | (state[6] & 0x0000FC00u)) >> 10;\n"
        "    state[2] += ((state[7] & 0x0000FC00u) | (state[6] & 0x000003E0u)) >> 5;\n"
        "    state[1] += ((state[7] & 0x000003E0u) | (state[6] & 0x0000001Fu));\n"
        "    state[0] += HAVAL_ROTR32((state[7] & 0x0000001Fu)\n"
        "               | (state[6] & 0xFC000000u), 26);\n");
    } else if (digest_bytes == 28) {
        /* 224-bit fold (donor :889-895). */
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
        /* 256-bit: NO fold (donor :903-908 direct output). */
        rc = hx_appendf(out, cap, len,
        "    // 256-bit: NO fold (donor havalFinal:903-908 direct output).\n");
    }
    if (rc < 0) return rc;

    /* Emit the first 16 bytes (h0..h3) of folded state as LE uints.
     * state[0..7] are LE-native (matches donor + test vectors); h0..h3 =
     * state[0..3] DIRECTLY (no byte-swap). The probe uses first 16 bytes;
     * the CPU recompute supplies the full digest for widths > 16. */
    rc = hx_appendf(out, cap, len,
        "\n"
        "    // LE state output direct extract -- HAVAL state is LE-native.\n"
        "    // h0..h3 = state[0..3] (first 16 bytes of folded digest).\n"
        "    *h0 = state[0];\n"
        "    *h1 = state[1];\n"
        "    *h2 = state[2];\n"
        "    *h3 = state[3];\n"
        "}\n"
        "\n");
    return rc;
}

/* Snefru emit helper. Phase 5b Tier 4 sub-phase 5b.4a.3 (2026-05-27).
 *
 * PARAMETERISED per D18.1.a/D18.3.a: ONE C-side helper emits a GPU
 * function specialised on (is256, digest_bytes). Both Snefru widths
 * (e175 SNE128 16-byte, e177 SNE256 32-byte) route through this helper;
 * is256 + the per-width data-block size (48 vs 32) + the length-field
 * byte offsets are baked as compile-time literals into the emitted body.
 *
 * Block-size asymmetry (R-Tier4-snefru-blocksize): SNE128 processes
 * 48-byte data blocks; SNE256 processes 32-byte data blocks. The
 * padding + length-field placement differs per width:
 *   - data_block_size dblk = 64 - digest_bytes (48 for SNE128, 32 SNE256).
 *   - rhash_snefru_final zero-pads the last partial block, compresses it,
 *     then builds a length block: be2me_32(length >> 29) at byte offset
 *     dblk-8 and be2me_32(length << 3) at dblk-4. `length` is the message
 *     length in BYTES (donor ctx->length). This is verified byte-exact
 *     in the C-mirror (/tmp/test_snefru_port.c 56/56 cells, both widths,
 *     28 lengths incl. block boundaries 31/32/33/47/48/49/63/64/65/...).
 *
 * Snefru IV is all-zero (donor rhash_snefru128/256_init memset). Schedule
 * is BIG-ENDIAN; state output is BE (donor be32_copy). The CPU oracle
 * stores those BE bytes; the harness reinterprets them as LE uints, so
 * the kernel byte-swaps each state word into the LE-uint frame here
 * (h0..h3 = bswap32(state[0..3]) = first 16 bytes of the digest) per
 * feedback_be_state_primitives_need_byteswap_in_codegen.md. SNE256's
 * remaining 16 bytes are filled by the CPU recompute on hit (digest > 16
 * via _proto_hexlen=64); SNE128 is exactly 16 bytes, no recompute. */
static int emit_outer_snefru_concat_then_hash(char **out,
                                             size_t *cap, size_t *len,
                                             int is256, int digest_bytes)
{
    int rc;
    int dblk = 64 - digest_bytes;   /* 48 (SNE128) or 32 (SNE256) */
    int off1 = dblk - 8;            /* be2me_32(len>>29) byte offset */
    int off2 = dblk - 4;            /* be2me_32(len<<3)  byte offset */
    int state_words = is256 ? 8 : 4;

    rc = hx_appendf(out, cap, len,
        "// hx: helper outer_snefru%d_concat_then_hash -- Snefru-%d over\n"
        "// (hex32(md5(pass)) || pass). Total input length = 32 + plen.\n"
        "// data_block_size = 64 - %d = %d bytes (is256=%d). Snefru IV is\n"
        "// all-zero; 8 rounds fixed. BE schedule + BE state output; h0..h3\n"
        "// are bswap32(state[0..3]) = first 16 bytes of the digest. Length\n"
        "// field: be2me_32(len>>29) at block[%d], be2me_32(len<<3) at\n"
        "// block[%d] (len in BYTES). CPU recompute fills the remaining\n"
        "// %d bytes on hit for SNE256.\n"
        "static void outer_snefru%d_concat_then_hash(\n"
        "    uint mma, uint mmb, uint mmc, uint mmd,\n"
        "    __global const uchar *pass, int plen,\n"
        "    uint *h0, uint *h1, uint *h2, uint *h3)\n"
        "{\n"
        "    if (plen < 0) plen = 0;\n"
        "    if (plen > HX_FAMILY_MAX_PASS) plen = HX_FAMILY_MAX_PASS;\n"
        "    int total_len = 32 + plen;\n"
        "    const int DBLK = %d;   // data_block_size for this width\n"
        "\n"
        "    // Snefru 512-bit state[8]; IV all-zero.\n"
        "    uint state[8];\n"
        "    for (int i = 0; i < 8; i++) state[i] = 0u;\n"
        "\n"
        "    uchar inner_hex[32];\n"
        "    state_to_hex32_bytes(mma, mmb, mmc, mmd, inner_hex);\n"
        "\n"
        "    uchar block[48];   // max data-block size (SNE128)\n"
        "    int consumed = 0;\n"
        "\n"
        "    // Process all full DBLK-byte data blocks of (hex32 || pass).\n"
        "    // The combined message is inner_hex[0..31] then pass[0..plen).\n"
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
        "    // Final padding (donor rhash_snefru_final). If a partial block\n"
        "    // remains, zero-pad it to DBLK and compress.\n"
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
        "    // Length block: be2me_32(len>>29) at block[%d],\n"
        "    // be2me_32(len<<3) at block[%d]. len is the message length in\n"
        "    // BYTES (total_len). Stored big-endian.\n"
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
        "    // BE state output -> LE-uint probe frame (bswap32 each word).\n"
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
        digest_bytes, dblk, is256,
        off1, off2, (digest_bytes > 16) ? (digest_bytes - 16) : 0,
        digest_bytes * 8,
        dblk,
        is256,
        is256,
        off1, off2,
        off1, off1, off1, off1,
        off2, off2, off2, off2,
        is256);
    (void)state_words;
    return rc;
}

/* GOST R 34.11-94 emit helper. Phase 5b Tier 4 sub-phase 5b.4b.3 (2026-05-27).
 *
 * Bespoke per-primitive helper (D18.3.a) -- GOST has no structural overlap
 * with any other family primitive. It is the ONLY block-cipher-based
 * primitive (GOST 28147-89 with a 32-round Feistel key schedule) and the
 * only one carrying a running mod-2^256 checksum sum[8] across blocks plus a
 * dual finalization. Donor: in-tree gosthash/gosthash.c gosthash_compress /
 * gosthash_bytes / gosthash_final (the LIVE CPU oracle for e125 via
 * gosthash.o; gosthash() at mdxfind.c:29076). TEST S-box set (NOT CryptoPro)
 * -- the gost_block primitive in gpu_common.cl bakes the 4 derived
 * GOST_SBOX_1..4 tables (R-Tier4-gost-sbox HIGH; verified via
 * test_gost_vectors.c: 4 published TEST-set vectors + 22-len cross-check vs
 * rhash RHASH_GOST, zero CryptoPro collisions).
 *
 * Message layout: (hex32(md5(pass)) || pass), total = 32 + plen. GOST
 * processes the message in 32-byte blocks (256-bit). For each block the
 * 32 bytes are converted to 8 LE uint32 words (gosthash_bytes:285-297),
 * accumulated into the running checksum sum[8] (mod 2^256 add with carry
 * propagation `c = (c<a)||(c<b)`), the bit-length counter len[0..1] is
 * advanced, and gost_block(hash, m) compresses. A trailing partial block is
 * zero-padded to 32 bytes and compressed over its partial bit-length. Then
 * the DUAL finalization (gosthash_final:358-359): gost_block(hash, len)
 * compresses the 256-bit bit-length block, then gost_block(hash, sum)
 * compresses the accumulated checksum. State output is LE byte-order
 * (gosthash_final:364-372); the harness reinterprets those LE bytes as LE
 * uints, so h0..h3 = state[0..3] DIRECTLY (no byte-swap). digest = 32 bytes;
 * the GPU probe carries h0..h3 (first 16 bytes); the CPU recompute fills the
 * remaining 16 bytes on hit (digest > 16 via _proto_hexlen=64).
 *
 * Validated byte-exact in the C-mirror (/tmp/test_gost_port.c 27/27 cells,
 * lengths straddling the 32-byte block boundary incl 31/32/33/63/64/65) vs
 * gosthash() BEFORE this GPU code shipped (the gost_block + sum[8] carry +
 * dual finalization are the highest-transcription-risk primitive in Phase
 * 5b; R-Tier4-gost-blockcipher / -checksum-carry HIGH/MED). */
static int emit_outer_gost_concat_then_hash(char **out,
                                            size_t *cap, size_t *len)
{
    int rc;

    rc = hx_appendf(out, cap, len,
        "// hx: helper outer_gost_concat_then_hash -- GOST R 34.11-94 (TEST\n"
        "// S-box set) over (hex32(md5(pass)) || pass). Total input length =\n"
        "// 32 + plen. 256-bit state[8]; 256-bit blocks (32 bytes, 8 LE words).\n"
        "// Running mod-2^256 checksum sum[8] carried across blocks; dual\n"
        "// finalization compresses the bit-length block then the checksum\n"
        "// block (gosthash_final). State output is LE; h0..h3 = state[0..3]\n"
        "// directly (no byte-swap). CPU recompute fills the remaining 16\n"
        "// bytes on hit (digest = 32 bytes).\n"
        "static void outer_gost_concat_then_hash(\n"
        "    uint mma, uint mmb, uint mmc, uint mmd,\n"
        "    __global const uchar *pass, int plen,\n"
        "    uint *h0, uint *h1, uint *h2, uint *h3)\n"
        "{\n"
        "    if (plen < 0) plen = 0;\n"
        "    if (plen > HX_FAMILY_MAX_PASS) plen = HX_FAMILY_MAX_PASS;\n"
        "    int total_len = 32 + plen;\n"
        "\n"
        "    // GOST 256-bit state hash[8]; checksum sum[8]; bit-length\n"
        "    // counter len[8] (only len[0..1] used for family-size inputs).\n"
        "    // All zero-initialised (gosthash_reset).\n"
        "    uint hash[8]; uint sum[8]; uint glen[8];\n"
        "    for (int i = 0; i < 8; i++) { hash[i] = 0u; sum[i] = 0u; glen[i] = 0u; }\n"
        "\n"
        "    uchar inner_hex[32];\n"
        "    state_to_hex32_bytes(mma, mmb, mmc, mmd, inner_hex);\n"
        "\n"
        "    uchar block[32];   // one GOST data block\n"
        "    int consumed = 0;\n"
        "\n"
        "    // Process all full 32-byte blocks of (hex32 || pass), then the\n"
        "    // trailing partial block (if any). Each block: convert 32 bytes\n"
        "    // to 8 LE uint32 words, accumulate sum[8] (mod 2^256), advance\n"
        "    // the bit-length counter, then gost_block.\n"
        "    while (consumed < total_len) {\n"
        "        int rem = total_len - consumed;\n"
        "        int blk = (rem >= 32) ? 32 : rem;   // bytes from the message\n"
        "        for (int i = 0; i < 32; i++) {\n"
        "            if (i < blk) {\n"
        "                int abs_pos = consumed + i;\n"
        "                block[i] = (abs_pos < 32) ? inner_hex[abs_pos]\n"
        "                                          : pass[abs_pos - 32];\n"
        "            } else {\n"
        "                block[i] = 0;   // zero-pad the partial final block\n"
        "            }\n"
        "        }\n"
        "        // bytes -> 8 LE words + checksum accumulate (gosthash_bytes).\n"
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
        "        // 64-bit bit-length counter spread over glen[0],glen[1].\n"
        "        uint bits = (uint)(blk << 3);\n"
        "        uint prev = glen[0];\n"
        "        glen[0] = prev + bits;\n"
        "        if (glen[0] < bits) glen[1] += 1u;\n"
        "        consumed += blk;\n"
        "        (void)prev;\n"
        "    }\n"
        "\n"
        "    // DUAL finalization (gosthash_final): compress the bit-length\n"
        "    // block, then the accumulated checksum block.\n"
        "    gost_block(hash, glen);\n"
        "    gost_block(hash, sum);\n"
        "\n"
        "    // State output is LE byte-order; LE-uint reinterpretation of the\n"
        "    // first 16 output bytes == hash[0..3] directly (no byte-swap).\n"
        "    *h0 = hash[0];\n"
        "    *h1 = hash[1];\n"
        "    *h2 = hash[2];\n"
        "    *h3 = hash[3];\n"
        "}\n"
        "\n");
    return rc;
}

/* Emit the family kernel body. Per-thread (no SALT_BATCH loop).
 * Each thread processes one word: gid -> word_idx; if word_idx >=
 * params.num_words return.
 *
 * Kernel signature is byte-identical to e347's so the production
 * dispatcher (5a.5) binds the same 16 args. The 4 salt-table args
 * (idx 3,4,5 + payload.num_salts) are IGNORED.
 *
 * Sub-phase 5a.4 (2026-05-23): switch on outer_id to select the
 * per-primitive helper. 7 of 8 5a-supported primitives wired here;
 * e123 MD5MD5PASS (HX_PRIM_MD5) stays outlier (multi-emit deferred).
 */
/* Forward decl: multi-emit kernel body (e123 MD5MD5PASS). Defined below
 * emit_family_md5pass_kernel. */
static int emit_family_md5pass_kernel_multiemit(
    char **out, size_t *cap, size_t *len, int job_enum);

static int emit_family_md5pass_kernel(char **out, size_t *cap, size_t *len,
                                      enum hx_primitive_id outer_id,
                                      const char *outer_name,
                                      int outer_digest_bytes,
                                      int job_enum,
                                      int emit_class)
{
    int rc;

    /* Sub-phase 5c.2 (2026-05-27): multi-emit members (e123 MD5MD5PASS)
     * take a dedicated kernel body that runs the probe + EMIT_HIT_4 block
     * ONCE PER VARIANT (N=2: sep=0 canonical, sep=1 colon). Single-emit
     * members fall through to the existing body UNCHANGED (G2 regression
     * safety: the per-variant logic is fully isolated). */
    if (emit_class == HX_EMIT_MULTI) {
        return emit_family_md5pass_kernel_multiemit(out, cap, len, job_enum);
    }

    /* Sub-phase 5a.4 (2026-05-23): per-primitive dispatch table for the
     * outer-CALL hash. 7 primitives wired (md4, sha1, sha224, sha256,
     * sha384, sha512, rmd160). e123 MD5MD5PASS (HX_PRIM_MD5) stays
     * outlier in 5a (multi-emit deferred sub-phase). Other family
     * members route to CPU via the per-primitive emit gate above
     * (hx_emit_family_md5pass_opencl) which restricts allowable
     * outer_ids to the 5a-supported subset.
     *
     * Pick the helper name + slot signature (4-uint state vs 5-uint
     * state writes h4 in addition to h0..h3). The probe uses h0..h3
     * only (compact_fp); h4 is computed for round-trip readback
     * regardless. */
    const char *helper_name = NULL;
    int helper_has_h4 = 0;
    switch (outer_id) {
        case HX_PRIM_SHA1:
            helper_name = "outer_sha1_concat_then_hash";
            helper_has_h4 = 1;
            break;
        case HX_PRIM_MD2:
            helper_name = "outer_md2_concat_then_hash";
            helper_has_h4 = 0;
            break;
        case HX_PRIM_MD4:
            helper_name = "outer_md4_concat_then_hash";
            helper_has_h4 = 0;
            break;
        case HX_PRIM_RMD128:
            helper_name = "outer_rmd128_concat_then_hash";
            helper_has_h4 = 0;
            break;
        case HX_PRIM_RMD160:
            helper_name = "outer_rmd160_concat_then_hash";
            helper_has_h4 = 1;
            break;
        case HX_PRIM_SHA224:
            helper_name = "outer_sha224_concat_then_hash";
            helper_has_h4 = 0;
            break;
        case HX_PRIM_SHA256:
            helper_name = "outer_sha256_concat_then_hash";
            helper_has_h4 = 0;
            break;
        case HX_PRIM_SHA384:
            helper_name = "outer_sha384_concat_then_hash";
            helper_has_h4 = 0;
            break;
        case HX_PRIM_SHA512:
            helper_name = "outer_sha512_concat_then_hash";
            helper_has_h4 = 0;
            break;
        case HX_PRIM_WRL:
            helper_name = "outer_wrl_concat_then_hash";
            helper_has_h4 = 0;
            break;
        case HX_PRIM_TIGER:
            helper_name = "outer_tiger_concat_then_hash";
            helper_has_h4 = 0;
            break;
        /* Phase 5b Tier 4 sub-phase 5b.4a (2026-05-27): the 2 Snefru
         * widths route through distinct emitted function names
         * (outer_snefru128_/outer_snefru256_concat_then_hash) since the
         * data-block size + length placement differ per width; the C-side
         * helper bakes is256 + DBLK + offsets into each body. 4-uint
         * probe; CPU recompute fills SNE256's remaining 16 bytes. */
        case HX_PRIM_SNE128:
            helper_name = "outer_snefru128_concat_then_hash";
            helper_has_h4 = 0;
            break;
        case HX_PRIM_SNE256:
            helper_name = "outer_snefru256_concat_then_hash";
            helper_has_h4 = 0;
            break;
        /* Phase 5b Tier 4 sub-phase 5b.4b (2026-05-27): GOST R 34.11-94
         * (e125, the final GPU-eligible MAKE_MD5PASS member) routes to its
         * own bespoke helper. 4-uint probe (first 16 bytes); CPU recompute
         * fills the remaining 16 bytes on hit. */
        case HX_PRIM_GOST:
            helper_name = "outer_gost_concat_then_hash";
            helper_has_h4 = 0;
            break;
        /* Phase 5b Tier 3 sub-phase 5b.3a (2026-05-27): all 5 3-pass
         * HAVAL variants route through the single parameterised helper
         * named outer_haval_concat_then_hash (the C-side helper bakes
         * the per-variant passes + digest_bytes into one emitted GPU
         * function). 4-uint probe (first 16 bytes); CPU recompute fills
         * the rest for widths > 16. */
        case HX_PRIM_HAV128_3:
        case HX_PRIM_HAV160_3:
        case HX_PRIM_HAV192_3:
        case HX_PRIM_HAV224_3:
        case HX_PRIM_HAV256_3:
        /* Phase 5b Tier 3 sub-phase 5b.3b (2026-05-27): 4-pass HAVAL
         * variants share the same emitted GPU function name (the helper
         * differs only in the haval<P>_block call + block[118] passes
         * field, both baked at C-emit time via the passes parameter). */
        case HX_PRIM_HAV128_4:
        case HX_PRIM_HAV160_4:
        case HX_PRIM_HAV192_4:
        case HX_PRIM_HAV224_4:
        case HX_PRIM_HAV256_4:
        /* Phase 5b Tier 3 sub-phase 5b.3c (2026-05-27): 5-pass HAVAL
         * variants share the same emitted GPU function name (the helper
         * differs only in the haval<P>_block call + block[118] passes
         * field, both baked at C-emit time via the passes parameter). */
        case HX_PRIM_HAV128_5:
        case HX_PRIM_HAV160_5:
        case HX_PRIM_HAV192_5:
        case HX_PRIM_HAV224_5:
        case HX_PRIM_HAV256_5:
            helper_name = "outer_haval_concat_then_hash";
            helper_has_h4 = 0;
            break;
        default:
            fprintf(stderr,
                "FATAL: %s:%d hx codegen family emit kernel: outer "
                "primitive '%s' (id=%d) is not wired in sub-phase 5a.4 "
                "(job=e%d). e123 MD5 is deferred to a future multi-emit "
                "sub-phase. 22 other primitives are 5b.\n",
                __FILE__, __LINE__,
                outer_name ? outer_name : "(null)",
                (int)outer_id, job_enum);
            exit(1);
    }

    /* Probe-result digest width annotation comment; compact_fp probe
     * always uses 4 uints from offset 0 of the digest (first 16 bytes
     * for 20/32/48/64-byte digests). The hit record (EMIT_HIT_4_*)
     * carries h0..h3 only; round-trip-readback uses HashDataBuf to
     * verify the full digest matches when the host inspects the slot. */
    rc = hx_appendf(out, cap, len,
        "// hx: family kernel for e%d outer=%s (digest=%d bytes); probe\n"
        "// uses first 4 uints (h0..h3) per compact_fp/compact_idx contract.\n"
        "//\n"
        "// kernel signature mirrors kernelb_hx_e347_phase0 so the\n"
        "// production dispatcher (5a.5) binds the same 16 args. The 4\n"
        "// salt-table args (3,4,5 + payload->num_salts) are IGNORED by\n"
        "// the family body since the family is unsalted; binding them\n"
        "// to existing device salt buffers is harmless.\n"
        "//\n"
        "// reqd_work_group_size(64,1,1) attribute pins the WG size to\n"
        "// match the production dispatcher's lsize=64; same pattern as\n"
        "// the e347 emitter R8 fix.\n"
        "__attribute__((reqd_work_group_size(64,1,1)))\n"
        "__kernel void kernelb_hx_codegen_phase0(\n"
        "    __global const uchar         *payload,\n"
        "    __global const uchar         *b_packed_buf,\n"
        "    __global const uint          *b_chunk_index,\n"
        "    __global const uchar         *salts,\n"
        "    __global const uint          *salt_offsets,\n"
        "    __global const ushort        *salt_lens,\n"
        "    __global const uint          *compact_fp,\n"
        "    __global const uint          *compact_idx,\n"
        "    __global const uchar         *hash_data_buf,\n"
        "    __global const ulong         *hash_data_off,\n"
        "    __global uint                *hits,\n"
        "    __global volatile uint       *hit_count,\n"
        "    __global const ulong         *overflow_keys,\n"
        "    __global const uchar         *overflow_hashes,\n"
        "    __global const uint          *overflow_offsets,\n"
        "    __global volatile uint       *hashes_shown\n"
        "    )\n"
        "{\n"
        "    // hx: state EMIT_KERNEL_PREAMBLE (family MD5PASS)\n"
        "    __global const OCLParams *params_buf =\n"
        "        (__global const OCLParams *)payload;\n"
        "    OCLParams params = *params_buf;\n"
        "\n"
        "    // Per-thread topology (no SALT_BATCH loop; family is unsalted)\n"
        "    uint gid = get_global_id(0);\n"
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
        "    __global const uchar *pass_bytes = b_packed_buf + wpos + 1u;\n"
        "\n"
        "    // OP_CALL md5 #1 (inner): MD5(pass) -> (ia,ib,ic,id)\n"
        "    uint ia, ib, ic, id;\n"
        "    md5_buf_global(pass_bytes, (int)plen, &ia, &ib, &ic, &id);\n"
        "\n"
        "    // OP_CALL outer (e%d %s): outer( hex32(MD5(pass)) || pass )\n"
        "%s"  /* declaration + helper call line, built per-primitive */
        "\n"
        "    // B3 overflow ledger pointers (mirrors e347 emitter rev 1.5\n"
        "    // line 708-713 -- same payload offsets).\n"
        "    __global volatile uint *ovr_set =\n"
        "        (__global volatile uint *)(payload + 100);\n"
        "    __global volatile uint *ovr_gid =\n"
        "        (__global volatile uint *)(payload + 104);\n"
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
        "%s"  /* (void)h4; suppression for 5-uint helpers or empty */
        "    // hx: state EMIT_KERNEL_FOOTER\n"
        "}\n",
        job_enum, outer_name, outer_digest_bytes,
        job_enum, outer_name,
        /* Declaration + call line. 5-uint helpers (sha1, rmd160) take
         * a 5th &h4 arg; 4-uint helpers (md4, sha224, sha256, sha384,
         * sha512) take only 4. */
        helper_has_h4
            ? ((outer_id == HX_PRIM_SHA1) ?
                "    uint h0, h1, h2, h3, h4;\n"
                "    outer_sha1_concat_then_hash(ia, ib, ic, id,\n"
                "                                pass_bytes, (int)plen,\n"
                "                                &h0, &h1, &h2, &h3, &h4);\n"
              : /* RMD160 */
                "    uint h0, h1, h2, h3, h4;\n"
                "    outer_rmd160_concat_then_hash(ia, ib, ic, id,\n"
                "                                  pass_bytes, (int)plen,\n"
                "                                  &h0, &h1, &h2, &h3, &h4);\n")
            : /* 4-uint state helpers: helper_name selects branch */
              ((outer_id == HX_PRIM_MD2) ?
                "    uint h0, h1, h2, h3;\n"
                "    outer_md2_concat_then_hash(ia, ib, ic, id,\n"
                "                               pass_bytes, (int)plen,\n"
                "                               &h0, &h1, &h2, &h3);\n"
              : (outer_id == HX_PRIM_RMD128) ?
                "    uint h0, h1, h2, h3;\n"
                "    outer_rmd128_concat_then_hash(ia, ib, ic, id,\n"
                "                                  pass_bytes, (int)plen,\n"
                "                                  &h0, &h1, &h2, &h3);\n"
              : (outer_id == HX_PRIM_MD4) ?
                "    uint h0, h1, h2, h3;\n"
                "    outer_md4_concat_then_hash(ia, ib, ic, id,\n"
                "                               pass_bytes, (int)plen,\n"
                "                               &h0, &h1, &h2, &h3);\n"
              : (outer_id == HX_PRIM_SHA224) ?
                "    uint h0, h1, h2, h3;\n"
                "    outer_sha224_concat_then_hash(ia, ib, ic, id,\n"
                "                                  pass_bytes, (int)plen,\n"
                "                                  &h0, &h1, &h2, &h3);\n"
              : (outer_id == HX_PRIM_SHA256) ?
                "    uint h0, h1, h2, h3;\n"
                "    outer_sha256_concat_then_hash(ia, ib, ic, id,\n"
                "                                  pass_bytes, (int)plen,\n"
                "                                  &h0, &h1, &h2, &h3);\n"
              : (outer_id == HX_PRIM_SHA384) ?
                "    uint h0, h1, h2, h3;\n"
                "    outer_sha384_concat_then_hash(ia, ib, ic, id,\n"
                "                                  pass_bytes, (int)plen,\n"
                "                                  &h0, &h1, &h2, &h3);\n"
              : (outer_id == HX_PRIM_SHA512) ?
                "    uint h0, h1, h2, h3;\n"
                "    outer_sha512_concat_then_hash(ia, ib, ic, id,\n"
                "                                  pass_bytes, (int)plen,\n"
                "                                  &h0, &h1, &h2, &h3);\n"
              : (outer_id == HX_PRIM_WRL) ?
                "    uint h0, h1, h2, h3;\n"
                "    outer_wrl_concat_then_hash(ia, ib, ic, id,\n"
                "                               pass_bytes, (int)plen,\n"
                "                               &h0, &h1, &h2, &h3);\n"
              : (outer_id == HX_PRIM_TIGER) ?
                "    uint h0, h1, h2, h3;\n"
                "    outer_tiger_concat_then_hash(ia, ib, ic, id,\n"
                "                                 pass_bytes, (int)plen,\n"
                "                                 &h0, &h1, &h2, &h3);\n"
              : (outer_id == HX_PRIM_SNE128) ?
                "    uint h0, h1, h2, h3;\n"
                "    outer_snefru128_concat_then_hash(ia, ib, ic, id,\n"
                "                                     pass_bytes, (int)plen,\n"
                "                                     &h0, &h1, &h2, &h3);\n"
              : (outer_id == HX_PRIM_SNE256) ?
                "    uint h0, h1, h2, h3;\n"
                "    outer_snefru256_concat_then_hash(ia, ib, ic, id,\n"
                "                                     pass_bytes, (int)plen,\n"
                "                                     &h0, &h1, &h2, &h3);\n"
              : (outer_id == HX_PRIM_GOST) ?
                "    uint h0, h1, h2, h3;\n"
                "    outer_gost_concat_then_hash(ia, ib, ic, id,\n"
                "                                pass_bytes, (int)plen,\n"
                "                                &h0, &h1, &h2, &h3);\n"
              : /* HAVAL (any 3-pass, 4-pass, or 5-pass variant;
                 * parameterised helper -- emitted GPU function name is
                 * identical, the passes/width differences are baked into
                 * the body). */
                "    uint h0, h1, h2, h3;\n"
                "    outer_haval_concat_then_hash(ia, ib, ic, id,\n"
                "                                 pass_bytes, (int)plen,\n"
                "                                 &h0, &h1, &h2, &h3);\n"),
        helper_has_h4
            ? "    (void)h4;  // 5th word reserved for round-trip readback.\n"
            : "");

    (void)helper_name;  /* selected by outer_id in the literal above */
    return rc;
}

/* Sub-phase 5c.2 (2026-05-27): multi-emit family kernel body (e123
 * MD5MD5PASS) -- the FIRST multi-emit algorithm.
 *
 * ONE password produces TWO outer-hash digests, each probed against the
 * loaded hash table as an INDEPENDENT found-hash candidate (byte-exact
 * match for the CPU oracle at mdxfind.c:25181-25204, which calls
 * checkhash() once per variant):
 *   variant 0 (sep=0, canonical): md5( hex32(md5(pass)) . pass )
 *   variant 1 (sep=1, colon):     md5( hex32(md5(pass)) . ':' . pass )
 *
 * Structure: compute md5(pass) ONCE (natural hoist; both variants reuse
 * the inner state ia..id), then a compile-time-N=2 UNROLLED set of
 * probe + EMIT_HIT_4_DEDUP_OR_OVERFLOW blocks, one per variant, each
 * with its OWN digest -> its OWN matched_idx. The dedup macro is the
 * EXISTING EMIT_HIT_4_DEDUP_OR_OVERFLOW (gpu_common.cl) UNCHANGED -- it
 * keys on hashes_shown[matched_idx] (the matched loaded-hash slot), which
 * is ALREADY the correct multi-emit key: two variants hitting two
 * DIFFERENT loaded hashes land in two different dedup cells and BOTH
 * emit. No new field, no buffer resize, no key widening.
 *
 * Recompute-per-variant (NOT hoisted beyond the shared inner): each
 * variant builds its own outer message + final MD5 in
 * outer_md5_concat_then_hash(sep). The hit record stays 16 bytes with NO
 * variant tag -- the emitted fingerprint self-identifies the matched
 * loaded hash on hit-replay (matching CPU semantics: mdxfind prints the
 * matched hash FORM, not the variant). */
static int emit_family_md5pass_kernel_multiemit(
    char **out, size_t *cap, size_t *len, int job_enum)
{
    int rc;

    rc = hx_appendf(out, cap, len,
        "// hx: family kernel for e%d MD5MD5PASS (MULTI-EMIT, N=2 variants);\n"
        "// digest=16 bytes; probe uses h0..h3 per compact_fp/compact_idx.\n"
        "// variant 0 = md5(hex32(md5(pass)) . pass); variant 1 = md5(hex32 .\n"
        "// ':' . pass). Each variant probes + emits independently against\n"
        "// its own matched loaded-hash slot (dedup keyed on matched_idx,\n"
        "// unchanged). kernel signature mirrors the single-emit family body.\n"
        "__attribute__((reqd_work_group_size(64,1,1)))\n"
        "__kernel void kernelb_hx_codegen_phase0(\n"
        "    __global const uchar         *payload,\n"
        "    __global const uchar         *b_packed_buf,\n"
        "    __global const uint          *b_chunk_index,\n"
        "    __global const uchar         *salts,\n"
        "    __global const uint          *salt_offsets,\n"
        "    __global const ushort        *salt_lens,\n"
        "    __global const uint          *compact_fp,\n"
        "    __global const uint          *compact_idx,\n"
        "    __global const uchar         *hash_data_buf,\n"
        "    __global const ulong         *hash_data_off,\n"
        "    __global uint                *hits,\n"
        "    __global volatile uint       *hit_count,\n"
        "    __global const ulong         *overflow_keys,\n"
        "    __global const uchar         *overflow_hashes,\n"
        "    __global const uint          *overflow_offsets,\n"
        "    __global volatile uint       *hashes_shown\n"
        "    )\n"
        "{\n"
        "    // hx: state EMIT_KERNEL_PREAMBLE (family MD5PASS, multi-emit)\n"
        "    __global const OCLParams *params_buf =\n"
        "        (__global const OCLParams *)payload;\n"
        "    OCLParams params = *params_buf;\n"
        "\n"
        "    uint gid = get_global_id(0);\n"
        "    uint word_idx = gid;\n"
        "    if (word_idx >= params.num_words) return;\n"
        "\n"
        "    (void)salts; (void)salt_offsets; (void)salt_lens;\n"
        "\n"
        "    // hx: state EMIT_PRE_INVARIANT (compute MD5(pass) ONCE)\n"
        "    uint wpos = b_chunk_index[word_idx];\n"
        "    if (wpos >= params.packed_size) return;  // defensive\n"
        "    uint plen = (uint)b_packed_buf[wpos];\n"
        "    __global const uchar *pass_bytes = b_packed_buf + wpos + 1u;\n"
        "\n"
        "    // OP_CALL md5 #1 (inner): MD5(pass) -> (ia,ib,ic,id). Shared\n"
        "    // across BOTH variants (natural hoist).\n"
        "    uint ia, ib, ic, id;\n"
        "    md5_buf_global(pass_bytes, (int)plen, &ia, &ib, &ic, &id);\n"
        "\n"
        "    // B3 overflow ledger pointers (shared by both variant emits).\n"
        "    __global volatile uint *ovr_set =\n"
        "        (__global volatile uint *)(payload + 100);\n"
        "    __global volatile uint *ovr_gid =\n"
        "        (__global volatile uint *)(payload + 104);\n"
        "\n"
        "    uint widx = params.base_word_idx + word_idx;\n"
        "\n",
        job_enum);
    if (rc < 0) return rc;

    /* Emit N=2 unrolled probe + EMIT_HIT_4 blocks (sep=0, sep=1). The
     * EMIT_HIT_4_DEDUP_OR_OVERFLOW macro is reused VERBATIM from the
     * single-emit body; the only change is each block computes its own
     * variant digest (via outer_md5_concat_then_hash(sep)) and resolves
     * its own matched_idx. */
    for (int sep = 0; sep <= 1; sep++) {
        rc = hx_appendf(out, cap, len,
            "    // hx: state EMIT_PROBE_AND_HIT variant %d (sep=%d)\n"
            "    {\n"
            "        uint h0, h1, h2, h3;\n"
            "        outer_md5_concat_then_hash(ia, ib, ic, id,\n"
            "                                   pass_bytes, (int)plen, %d,\n"
            "                                   &h0, &h1, &h2, &h3);\n"
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
        "    // hx: state EMIT_KERNEL_FOOTER (multi-emit)\n"
        "}\n");
    return rc;
}

int hx_emit_family_md5pass_opencl(
    char **out, size_t *out_cap,
    const hx_program *prog,
    const struct hx_specialization *spec,
    const struct hx_spec_entry *entry)
{
    if (!out || !out_cap || !prog || !spec || !entry) {
        fprintf(stderr,
            "FATAL: %s:%d hx_emit_family_md5pass_opencl: NULL argument "
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
            "FATAL: %s:%d hx_emit_family_md5pass_opencl: e%d %s code[1] "
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
            "FATAL: %s:%d hx_emit_family_md5pass_opencl: e%d %s code[4] "
            "callname is NULL (sidecar missing).\n",
            __FILE__, __LINE__, entry->job_enum,
            entry->name ? entry->name : "(noname)");
        return -1;
    }
    enum hx_primitive_id outer_id = hx_primitive_id_for_name(outer_name);
    if (outer_id == HX_PRIM_UNKNOWN) {
        fprintf(stderr,
            "FATAL: %s:%d hx_emit_family_md5pass_opencl: e%d %s outer "
            "callname '%s' not recognized in hx_emit_primitives.c table. "
            "Either it is a new primitive (add to prim_table) or the "
            "sidecar is corrupt.\n",
            __FILE__, __LINE__, entry->job_enum,
            entry->name ? entry->name : "(noname)", outer_name);
        return -1;
    }
    if (!hx_primitive_is_supported_5a(outer_id)) {
        fprintf(stderr,
            "FATAL: %s:%d hx_emit_family_md5pass_opencl: e%d %s outer "
            "primitive '%s' is in the 5b-deferred set (not in "
            "gpu_common.cl yet). Phase 5b lifts the missing *_block "
            "function; until then this algorithm routes to CPU only.\n",
            __FILE__, __LINE__, entry->job_enum,
            entry->name ? entry->name : "(noname)", outer_name);
        return -1;
    }
    /* Sub-phase 5c.2 (2026-05-27): MD5-as-OUTER is now supported for the
     * e123 MD5MD5PASS multi-emit member. It is admitted ONLY when the
     * spec entry is flagged emit_class == HX_EMIT_MULTI (the generator
     * sets this for e123 via the Note-[24] markup-strip). An MD5 outer
     * with emit_class SINGLE would be an unexpected non-multi-emit MD5
     * member; FATAL because the single-emit MD5 path is not the intended
     * shape (e123 is the only MD5-outer family member, and it is
     * multi-emit by construction). */
    if (outer_id == HX_PRIM_MD5 && entry->emit_class != HX_EMIT_MULTI) {
        fprintf(stderr,
            "FATAL: %s:%d hx_emit_family_md5pass_opencl: e%d %s outer "
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
            "FATAL: %s:%d hx_emit_family_md5pass_opencl: e%d %s outer "
            "primitive '%s' is in supported_5a set but not in the "
            "5a.4 + 5b.1a + 5b.1b + 5b.2a + 5b.2b + 5b.3a + 5b.3b + 5b.3c "
            "+ 5b.4a + 5b.4b + 5c.2 wired subset (md5(multi-emit) md2 md4 "
            "rmd128 sha1 sha224 sha256 "
            "sha384 sha512 rmd160 wrl tiger sne128 sne256 gost hav128_3 "
            "hav160_3 hav192_3 hav224_3 hav256_3 hav128_4 hav160_4 hav192_4 "
            "hav224_4 hav256_4 hav128_5 hav160_5 hav192_5 hav224_5 "
            "hav256_5). "
            "Either add to dispatch above or this is a logic bug.\n",
            __FILE__, __LINE__, entry->job_enum,
            entry->name ? entry->name : "(noname)", outer_name);
        return -1;
    }

    int outer_digest_bytes = hx_primitive_digest_bytes(outer_id);

    size_t cur_len = 0;
    if (*out == NULL) *out_cap = 0;

    int rc;

    /* Banner with structural details for dump-file readability. */
    rc = hx_appendf(out, out_cap, &cur_len,
        "// hx codegen: PATTERN FAMILY_MD5PASS matched (e%d %s outer=%s)\n"
        "// hx: program ncode=%d nvars=%d max_stack=%d has_emit=%d\n"
        "// hx: specialization iter=%u rules=%u masks=%u bf=%u "
        "salt_minlen=%u salt_maxlen=%u regime=%d width=%u\n"
        "// hx: this kernel will be JIT-compiled with gpu_common_str\n"
        "// hx: prepended (gpu_opencl_jit_compile_source_with_common)\n"
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

    rc = emit_family_md5pass_helpers(out, out_cap, &cur_len);
    if (rc < 0) return rc;

    /* Emit ONE per-primitive helper body matching outer_id. The
     * kernel body (next emit) calls the matching helper by name. */
    switch (outer_id) {
        /* Sub-phase 5c.2 (2026-05-27): MD5-as-outer multi-emit helper
         * (e123 MD5MD5PASS). The helper emits ONE function with a `sep`
         * parameter; the kernel body calls it twice (sep=0 canonical,
         * sep=1 colon). Only reached for emit_class==HX_EMIT_MULTI
         * (gated above). */
        case HX_PRIM_MD5:
            rc = emit_outer_md5_concat_then_hash(out, out_cap, &cur_len); break;
        case HX_PRIM_SHA1:
            rc = emit_outer_sha1_concat_then_hash(out, out_cap, &cur_len); break;
        case HX_PRIM_MD2:
            rc = emit_outer_md2_concat_then_hash(out, out_cap, &cur_len); break;
        case HX_PRIM_MD4:
            rc = emit_outer_md4_concat_then_hash(out, out_cap, &cur_len); break;
        case HX_PRIM_RMD128:
            rc = emit_outer_rmd128_concat_then_hash(out, out_cap, &cur_len); break;
        case HX_PRIM_RMD160:
            rc = emit_outer_rmd160_concat_then_hash(out, out_cap, &cur_len); break;
        case HX_PRIM_SHA224:
            rc = emit_outer_sha224_concat_then_hash(out, out_cap, &cur_len); break;
        case HX_PRIM_SHA256:
            rc = emit_outer_sha256_concat_then_hash(out, out_cap, &cur_len); break;
        case HX_PRIM_SHA384:
            rc = emit_outer_sha384_concat_then_hash(out, out_cap, &cur_len); break;
        case HX_PRIM_SHA512:
            rc = emit_outer_sha512_concat_then_hash(out, out_cap, &cur_len); break;
        case HX_PRIM_WRL:
            rc = emit_outer_wrl_concat_then_hash(out, out_cap, &cur_len); break;
        case HX_PRIM_TIGER:
            rc = emit_outer_tiger_concat_then_hash(out, out_cap, &cur_len); break;
        /* Phase 5b Tier 4 sub-phase 5b.4a (2026-05-27): the 2 Snefru
         * widths route to ONE parameterised helper specialised on
         * (is256, digest_bytes). SNE128 is256=0 16-byte / SNE256 is256=1
         * 32-byte. The helper bakes the per-width data-block size (48 vs
         * 32) + length-field byte offsets into distinct emitted
         * functions. gost (e125) ships in 5b.4b. */
        case HX_PRIM_SNE128:
            rc = emit_outer_snefru_concat_then_hash(out, out_cap, &cur_len,
                                                    0, outer_digest_bytes);
            break;
        case HX_PRIM_SNE256:
            rc = emit_outer_snefru_concat_then_hash(out, out_cap, &cur_len,
                                                    1, outer_digest_bytes);
            break;
        /* Phase 5b Tier 4 sub-phase 5b.4b (2026-05-27): GOST R 34.11-94
         * (e125) -- bespoke helper. Block-cipher core + mod-2^256 checksum
         * carry + dual finalization (the highest-transcription-risk
         * primitive in Phase 5b). After this ship, the MAKE_MD5PASS family
         * reaches 29/30 GPU-eligible (only e123 multi-emit remains). */
        case HX_PRIM_GOST:
            rc = emit_outer_gost_concat_then_hash(out, out_cap, &cur_len);
            break;
        /* Phase 5b Tier 3 sub-phase 5b.3a (2026-05-27): 5 3-pass HAVAL
         * variants route to ONE parameterised helper. passes=3 fixed
         * (5b.3a); digest_bytes from outer_digest_bytes (16/20/24/28/32
         * for hav128/160/192/224/256). The helper bakes the per-variant
         * block[118..119] encoding + per-width fold into one GPU
         * function. 4-pass + 5-pass add passes=4/5 in 5b.3b + 5b.3c. */
        case HX_PRIM_HAV128_3:
        case HX_PRIM_HAV160_3:
        case HX_PRIM_HAV192_3:
        case HX_PRIM_HAV224_3:
        case HX_PRIM_HAV256_3:
            rc = emit_outer_haval_concat_then_hash(out, out_cap, &cur_len,
                                                   3, outer_digest_bytes);
            break;
        /* Phase 5b Tier 3 sub-phase 5b.3b (2026-05-27): 5 4-pass HAVAL
         * variants route to the SAME parameterised helper with passes=4
         * (emits haval4_block call + block[118] passes=4 encoding). */
        case HX_PRIM_HAV128_4:
        case HX_PRIM_HAV160_4:
        case HX_PRIM_HAV192_4:
        case HX_PRIM_HAV224_4:
        case HX_PRIM_HAV256_4:
            rc = emit_outer_haval_concat_then_hash(out, out_cap, &cur_len,
                                                   4, outer_digest_bytes);
            break;
        /* Phase 5b Tier 3 sub-phase 5b.3c (2026-05-27): 5 5-pass HAVAL
         * variants route to the SAME parameterised helper with passes=5
         * (emits haval5_block call + block[118] passes=5 encoding => the
         * byte118 nibble is (W&3)<<6 | (5<<3) | 1 = 0x29 for 128-bit). */
        case HX_PRIM_HAV128_5:
        case HX_PRIM_HAV160_5:
        case HX_PRIM_HAV192_5:
        case HX_PRIM_HAV224_5:
        case HX_PRIM_HAV256_5:
            rc = emit_outer_haval_concat_then_hash(out, out_cap, &cur_len,
                                                   5, outer_digest_bytes);
            break;
        default:
            fprintf(stderr,
                "FATAL: %s:%d hx_emit_family_md5pass_opencl: unreachable "
                "(outer_id=%d not in 5a.4 + 5b.1 + 5b.2a + 5b.2b + 5b.3a "
                "+ 5b.3b + 5b.3c wired set)\n",
                __FILE__, __LINE__, (int)outer_id);
            return -1;
    }
    if (rc < 0) return rc;

    rc = emit_family_md5pass_kernel(out, out_cap, &cur_len,
                                    outer_id, outer_name,
                                    outer_digest_bytes, entry->job_enum,
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
 * Phase 1b Batch 1 (2026-05-28): unsalted single-hash emitter.
 *
 * Emits a one-shot `hash(pass)` kernel for the category-(a) MD/SHA
 * family (HX_PATTERN_UNSALTED_SINGLE). Validated byte-exact in plain C
 * (/tmp/test_unsalted_single_port.c, 80/80 across md5/md4/sha1/sha256 x
 * block-boundary-straddling lengths 0..240) BEFORE GPU JIT, per
 * feedback_c_mirror_before_gpu_port.md.
 *
 * STRICTLY SIMPLER than the family emitter: the hash input IS the pass
 * (length plen), with NO inner md5, NO hex32 prefix, NO concat. The
 * per-primitive usp_*_buf_global helpers below reproduce the SAME MD/SHA
 * padding the family emitter uses, applied to the raw pass. Calls
 * md5_block/md4_block/sha1_block/sha256_block (gpu_common.cl).
 *
 * probe uses the first 4 LE uints of the digest (compact_fp contract,
 * identical to the family path). For BE-state primitives (sha1/sha256)
 * each of the first 4 state words is byte-swapped to LE before probing,
 * per feedback_be_state_primitives_need_byteswap_in_codegen.md.
 * ==================================================================== */

static int emit_unsalted_single_helpers(char **out, size_t *cap, size_t *len)
{
    int rc;
    rc = hx_appendf(out, cap, len,
        "// ====================================================================\n"
        "// hx codegen Phase 1b Batch 1 (2026-05-28): unsalted single-hash\n"
        "// Emitted by hx_emit_unsalted_single_opencl()\n"
        "// Pattern matched: HX_PATTERN_UNSALTED_SINGLE\n"
        "// Algorithm: hash(pass)  (no inner md5, no hex32, no concat)\n"
        "// Helpers from gpu_common.cl (prepended at JIT time):\n"
        "//   md5_block, md4_block, sha1_block, sha256_block, OCLParams,\n"
        "//   EMIT_HIT_4_DEDUP_OR_OVERFLOW, probe_compact_idx\n"
        "// ====================================================================\n"
        "\n"
        "#ifndef HX_USP_MAX_PASS\n"
        "#define HX_USP_MAX_PASS 256\n"
        "#endif\n"
        "\n");
    if (rc < 0) return rc;

    rc = hx_appendf(out, cap, len,
        "// hx: helper usp_md_buf_global -- MD5/MD4 of variable-length __global\n"
        "// candidate. is_md4 selects md4_block vs md5_block. LE schedule, LE\n"
        "// 64-bit bit-length split M[14] (low) / M[15] (high).\n"
        "static void usp_md_buf_global(__global const uchar *data, int len,\n"
        "                              int is_md4,\n"
        "                              uint *h0, uint *h1, uint *h2, uint *h3)\n"
        "{\n"
        "    uint M[16];\n"
        "    int pos = 0;\n"
        "    *h0 = 0x67452301u; *h1 = 0xEFCDAB89u;\n"
        "    *h2 = 0x98BADCFEu; *h3 = 0x10325476u;\n"
        "    while (len - pos >= 64) {\n"
        "        for (int j = 0; j < 16; j++) {\n"
        "            int b = pos + j * 4;\n"
        "            M[j] = (uint)data[b] | ((uint)data[b+1] << 8)\n"
        "                 | ((uint)data[b+2] << 16) | ((uint)data[b+3] << 24);\n"
        "        }\n"
        "        if (is_md4) md4_block(h0, h1, h2, h3, M);\n"
        "        else        md5_block(h0, h1, h2, h3, M);\n"
        "        pos += 64;\n"
        "    }\n"
        "    int rem = len - pos;\n"
        "    for (int j = 0; j < 16; j++) M[j] = 0;\n"
        "    for (int i = 0; i < rem; i++) {\n"
        "        uint v = (uint)data[pos + i];\n"
        "        M[i >> 2] |= v << ((i & 3) * 8);\n"
        "    }\n"
        "    M[rem >> 2] |= (uint)0x80 << ((rem & 3) * 8);\n"
        "    ulong bits = (ulong)len * 8ul;\n"
        "    if (rem < 56) {\n"
        "        M[14] = (uint)(bits & 0xfffffffful);\n"
        "        M[15] = (uint)(bits >> 32);\n"
        "        if (is_md4) md4_block(h0, h1, h2, h3, M);\n"
        "        else        md5_block(h0, h1, h2, h3, M);\n"
        "    } else {\n"
        "        if (is_md4) md4_block(h0, h1, h2, h3, M);\n"
        "        else        md5_block(h0, h1, h2, h3, M);\n"
        "        for (int j = 0; j < 16; j++) M[j] = 0;\n"
        "        M[14] = (uint)(bits & 0xfffffffful);\n"
        "        M[15] = (uint)(bits >> 32);\n"
        "        if (is_md4) md4_block(h0, h1, h2, h3, M);\n"
        "        else        md5_block(h0, h1, h2, h3, M);\n"
        "    }\n"
        "}\n"
        "\n");
    if (rc < 0) return rc;

    rc = hx_appendf(out, cap, len,
        "// hx: helper usp_sha1_buf_global -- SHA1 of variable-length __global\n"
        "// candidate. BE schedule, BE 64-bit bit-length at byte offset 56.\n"
        "// First 4 state words byte-swapped to LE for probe (compact_fp).\n"
        "static void usp_sha1_buf_global(__global const uchar *data, int len,\n"
        "                                uint *h0, uint *h1, uint *h2, uint *h3)\n"
        "{\n"
        "    uint M[16];\n"
        "    int pos = 0;\n"
        "    uint st[5] = { 0x67452301u, 0xEFCDAB89u, 0x98BADCFEu,\n"
        "                   0x10325476u, 0xC3D2E1F0u };\n"
        "    while (len - pos >= 64) {\n"
        "        for (int j = 0; j < 16; j++) {\n"
        "            int b = pos + j * 4;\n"
        "            M[j] = ((uint)data[b] << 24) | ((uint)data[b+1] << 16)\n"
        "                 | ((uint)data[b+2] << 8) | (uint)data[b+3];\n"
        "        }\n"
        "        sha1_block(st, M); pos += 64;\n"
        "    }\n"
        "    int rem = len - pos;\n"
        "    uchar blk[64];\n"
        "    for (int i = 0; i < 64; i++) blk[i] = 0;\n"
        "    for (int i = 0; i < rem; i++) blk[i] = data[pos + i];\n"
        "    blk[rem] = 0x80;\n"
        "    ulong bits = (ulong)len * 8ul;\n"
        "    if (rem >= 56) {\n"
        "        for (int j = 0; j < 16; j++)\n"
        "            M[j] = ((uint)blk[j*4] << 24) | ((uint)blk[j*4+1] << 16)\n"
        "                 | ((uint)blk[j*4+2] << 8) | (uint)blk[j*4+3];\n"
        "        sha1_block(st, M);\n"
        "        for (int i = 0; i < 64; i++) blk[i] = 0;\n"
        "    }\n"
        "    for (int i = 0; i < 8; i++) blk[56 + i] = (uchar)((bits >> (56 - i*8)) & 0xffu);\n"
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
        "// hx: helper usp_sha256_buf_global -- SHA256 of variable-length\n"
        "// __global candidate. BE schedule, BE 64-bit length. First 4 state\n"
        "// words byte-swapped to LE for probe.\n"
        "static void usp_sha256_buf_global(__global const uchar *data, int len,\n"
        "                                  uint *h0, uint *h1, uint *h2, uint *h3)\n"
        "{\n"
        "    uint M[16];\n"
        "    int pos = 0;\n"
        "    uint st[8] = { 0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,\n"
        "                   0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u };\n"
        "    while (len - pos >= 64) {\n"
        "        for (int j = 0; j < 16; j++) {\n"
        "            int b = pos + j * 4;\n"
        "            M[j] = ((uint)data[b] << 24) | ((uint)data[b+1] << 16)\n"
        "                 | ((uint)data[b+2] << 8) | (uint)data[b+3];\n"
        "        }\n"
        "        sha256_block(st, M); pos += 64;\n"
        "    }\n"
        "    int rem = len - pos;\n"
        "    uchar blk[64];\n"
        "    for (int i = 0; i < 64; i++) blk[i] = 0;\n"
        "    for (int i = 0; i < rem; i++) blk[i] = data[pos + i];\n"
        "    blk[rem] = 0x80;\n"
        "    ulong bits = (ulong)len * 8ul;\n"
        "    if (rem >= 56) {\n"
        "        for (int j = 0; j < 16; j++)\n"
        "            M[j] = ((uint)blk[j*4] << 24) | ((uint)blk[j*4+1] << 16)\n"
        "                 | ((uint)blk[j*4+2] << 8) | (uint)blk[j*4+3];\n"
        "        sha256_block(st, M);\n"
        "        for (int i = 0; i < 64; i++) blk[i] = 0;\n"
        "    }\n"
        "    for (int i = 0; i < 8; i++) blk[56 + i] = (uchar)((bits >> (56 - i*8)) & 0xffu);\n"
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
     * Iter v1 (2026-05-31, project_codegen_iteration_v1_spec_-
     * 2026-05-31.md): per-primitive iter-feed helpers. Mirror the
     * legacy md5_rules_phase0 iter loop (gpu/gpu_md5_rules.cl:1158-
     * 1193) byte-for-byte: digest is hex-encoded (lower-case), then
     * re-hashed from a FRESH IV. R1 mitigation per spec.
     *
     *   MD5/MD4:  32 hex chars + 0x80 + length=256  -> single block
     *   SHA1:     40 hex chars + 0x80 + length=320  -> single block
     *   SHA256:   64 hex chars + 0x80 + length=512  -> two blocks
     *
     * In/out: pointer-to-state (h0..h3 for MD5/MD4 single uints;
     * st[5]/st[8] for SHA1/SHA256 which keep full state internally
     * and swap-back the first 4 state words to LE for compact probe).
     * The kernel emit calls these BEFORE each subsequent probe at
     * iter >= 2. R7 per-iter probe mask: `1u << (iter & 31u)`.
     * ============================================================ */
    rc = hx_appendf(out, cap, len,
        "// hx iter v1: MD5 hex32-feed (LE schedule, fresh IV). Used by the\n"
        "// codegen kernel B body between iter levels when max_iter > 1.\n"
        "static void usp_md5_iter_hex32_feed(uint *h0, uint *h1, uint *h2, uint *h3)\n"
        "{\n"
        "    uint M[16];\n"
        "    md5_to_hex_lc(*h0, *h1, *h2, *h3, M);\n"
        "    M[8] = 0x80u;\n"
        "    for (int j = 9; j < 14; j++) M[j] = 0u;\n"
        "    M[14] = 32u * 8u;\n"
        "    M[15] = 0u;\n"
        "    *h0 = 0x67452301u; *h1 = 0xEFCDAB89u;\n"
        "    *h2 = 0x98BADCFEu; *h3 = 0x10325476u;\n"
        "    md5_block(h0, h1, h2, h3, M);\n"
        "}\n"
        "\n");
    if (rc < 0) return rc;

    rc = hx_appendf(out, cap, len,
        "// hx iter v1: MD4 hex32-feed (LE schedule, fresh IV). Same shape\n"
        "// as MD5 since MD4 IV/schedule are identical to MD5 at this layer.\n"
        "static void usp_md4_iter_hex32_feed(uint *h0, uint *h1, uint *h2, uint *h3)\n"
        "{\n"
        "    uint M[16];\n"
        "    md5_to_hex_lc(*h0, *h1, *h2, *h3, M);\n"
        "    M[8] = 0x80u;\n"
        "    for (int j = 9; j < 14; j++) M[j] = 0u;\n"
        "    M[14] = 32u * 8u;\n"
        "    M[15] = 0u;\n"
        "    *h0 = 0x67452301u; *h1 = 0xEFCDAB89u;\n"
        "    *h2 = 0x98BADCFEu; *h3 = 0x10325476u;\n"
        "    md4_block(h0, h1, h2, h3, M);\n"
        "}\n"
        "\n");
    if (rc < 0) return rc;

    /* SHA1: between iters, we need to re-hash the 40-char ASCII hex
     * representation of the previous digest. Caller passes in the LE-
     * swapped probe key (h0..h3 are the first 4 LE state words). We
     * must reconstruct the BE state values, hex-encode, then rebuild
     * SHA1's 5-word state from cold IV (the 5th state word is not in
     * the probe key — we need to thread it through). Simpler path:
     * recompute from h0..h4 array carried alongside. See kernel emit
     * for the storage shape. */
    rc = hx_appendf(out, cap, len,
        "// hx iter v1: SHA1 hex40-feed (BE schedule, fresh IV). The full\n"
        "// 5-word state (st[5]) is the input; the kernel emit keeps it\n"
        "// alongside the LE-swapped 4-word probe key. After feed, st[5]\n"
        "// holds the new digest; first 4 state words are byte-swapped to\n"
        "// LE for the next probe.\n"
        "static void usp_sha1_iter_hex40_feed(uint *st)\n"
        "{\n"
        "    /* Build 40 ASCII hex bytes from st[0..4] (BE digest -> hex). */\n"
        "    uchar hex[40];\n"
        "    for (int s = 0; s < 5; s++) {\n"
        "        uint v = st[s];\n"
        "        uchar b0 = (uchar)((v >> 24) & 0xffu);\n"
        "        uchar b1 = (uchar)((v >> 16) & 0xffu);\n"
        "        uchar b2 = (uchar)((v >>  8) & 0xffu);\n"
        "        uchar b3 = (uchar)( v        & 0xffu);\n"
        "        uchar bs[4]; bs[0]=b0; bs[1]=b1; bs[2]=b2; bs[3]=b3;\n"
        "        for (int k = 0; k < 4; k++) {\n"
        "            uchar b = bs[k];\n"
        "            uchar hi = (b >> 4) & 0xfu;\n"
        "            uchar lo = b & 0xfu;\n"
        "            hex[s*8 + k*2 + 0] = (uchar)(hi + ((hi < 10u) ? '0' : ('a' - 10)));\n"
        "            hex[s*8 + k*2 + 1] = (uchar)(lo + ((lo < 10u) ? '0' : ('a' - 10)));\n"
        "        }\n"
        "    }\n"
        "    /* Single block (40 + 1 + 8 = 49 <= 64). BE schedule. */\n"
        "    uint M[16];\n"
        "    for (int j = 0; j < 10; j++) {\n"
        "        M[j] = ((uint)hex[j*4] << 24) | ((uint)hex[j*4+1] << 16)\n"
        "             | ((uint)hex[j*4+2] << 8) | (uint)hex[j*4+3];\n"
        "    }\n"
        "    M[10] = 0x80000000u;   /* sentinel byte at offset 40 (BE word 10, top byte) */\n"
        "    for (int j = 11; j < 14; j++) M[j] = 0u;\n"
        "    M[14] = 0u;\n"
        "    M[15] = 320u;          /* 40 bytes * 8 bits */\n"
        "    st[0] = 0x67452301u; st[1] = 0xEFCDAB89u; st[2] = 0x98BADCFEu;\n"
        "    st[3] = 0x10325476u; st[4] = 0xC3D2E1F0u;\n"
        "    sha1_block(st, M);\n"
        "}\n"
        "\n");
    if (rc < 0) return rc;

    rc = hx_appendf(out, cap, len,
        "// hx iter v1: SHA256 hex64-feed (BE schedule, fresh IV). 64 hex\n"
        "// bytes + 1 sentinel + 8 length = 73 > 64; TWO blocks required.\n"
        "// st[8] is the full state; kernel emit threads it through.\n"
        "static void usp_sha256_iter_hex64_feed(uint *st)\n"
        "{\n"
        "    /* Build 64 ASCII hex bytes from st[0..7] (BE digest -> hex). */\n"
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
        "    /* Block 1: all 64 hex bytes, no sentinel. BE schedule. */\n"
        "    uint M[16];\n"
        "    for (int j = 0; j < 16; j++) {\n"
        "        M[j] = ((uint)hex[j*4] << 24) | ((uint)hex[j*4+1] << 16)\n"
        "             | ((uint)hex[j*4+2] << 8) | (uint)hex[j*4+3];\n"
        "    }\n"
        "    st[0] = 0x6a09e667u; st[1] = 0xbb67ae85u; st[2] = 0x3c6ef372u; st[3] = 0xa54ff53au;\n"
        "    st[4] = 0x510e527fu; st[5] = 0x9b05688cu; st[6] = 0x1f83d9abu; st[7] = 0x5be0cd19u;\n"
        "    sha256_block(st, M);\n"
        "    /* Block 2: 0x80 sentinel at byte 0, zero pad, BE 64-bit length at end. */\n"
        "    M[0] = 0x80000000u;\n"
        "    for (int j = 1; j < 14; j++) M[j] = 0u;\n"
        "    M[14] = 0u;\n"
        "    M[15] = 512u;     /* 64 bytes * 8 bits */\n"
        "    sha256_block(st, M);\n"
        "}\n"
        "\n");
    return rc;
}

static int emit_unsalted_single_kernel(char **out, size_t *cap, size_t *len,
                                       enum hx_primitive_id pid,
                                       const char *prim_name, int job_enum)
{
    /* Iter v1 (2026-05-31): unlike Batch 1's one-shot probe, the kernel
     * body now wraps the hash + probe with a runtime for-loop reading
     * `params.max_iter` (offset 60). For SHA1/SHA256 the iter feed needs
     * the full 5/8-word BE state (NOT just the 4-word LE probe key); we
     * thread the state through and byte-swap-to-LE per iter for probe. */
    const char *seed_line;    /* iter==1: produce initial digest */
    const char *probe_load;   /* per-iter: load probe key (LE 4 uints) */
    const char *feed_line;    /* iter<max: hex-feed to next iter */
    switch (pid) {
        case HX_PRIM_MD5:
            seed_line =
                "    uint h0, h1, h2, h3;\n"
                "    usp_md_buf_global(pass_bytes, (int)plen, 0, &h0, &h1, &h2, &h3);\n";
            probe_load = "    /* h0..h3 are already LE probe key */\n";
            feed_line =
                "            usp_md5_iter_hex32_feed(&h0, &h1, &h2, &h3);\n";
            break;
        case HX_PRIM_MD4:
            seed_line =
                "    uint h0, h1, h2, h3;\n"
                "    usp_md_buf_global(pass_bytes, (int)plen, 1, &h0, &h1, &h2, &h3);\n";
            probe_load = "    /* h0..h3 are already LE probe key */\n";
            feed_line =
                "            usp_md4_iter_hex32_feed(&h0, &h1, &h2, &h3);\n";
            break;
        case HX_PRIM_SHA1:
            /* The existing helper byte-swaps st[]->h*; we need to thread
             * the full BE state through iter, so re-implement seed inline
             * to keep st[5] (then derive h0..h3 = bswap32(st[0..3])). */
            seed_line =
                "    uint st[5];\n"
                "    uint h0, h1, h2, h3;\n"
                "    /* seed: compute SHA1(pass) -> st[5], then derive LE probe */\n"
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
                "        /* SHA1: BE state -> LE probe key (mirror usp_sha1_buf_global) */\n"
                "        h0 = ((st[0] & 0x000000ffu) << 24) | ((st[0] & 0x0000ff00u) << 8)\n"
                "           | ((st[0] & 0x00ff0000u) >> 8) | ((st[0] & 0xff000000u) >> 24);\n"
                "        h1 = ((st[1] & 0x000000ffu) << 24) | ((st[1] & 0x0000ff00u) << 8)\n"
                "           | ((st[1] & 0x00ff0000u) >> 8) | ((st[1] & 0xff000000u) >> 24);\n"
                "        h2 = ((st[2] & 0x000000ffu) << 24) | ((st[2] & 0x0000ff00u) << 8)\n"
                "           | ((st[2] & 0x00ff0000u) >> 8) | ((st[2] & 0xff000000u) >> 24);\n"
                "        h3 = ((st[3] & 0x000000ffu) << 24) | ((st[3] & 0x0000ff00u) << 8)\n"
                "           | ((st[3] & 0x00ff0000u) >> 8) | ((st[3] & 0xff000000u) >> 24);\n";
            feed_line =
                "            usp_sha1_iter_hex40_feed(st);\n";
            break;
        case HX_PRIM_SHA256:
            seed_line =
                "    uint st[8];\n"
                "    uint h0, h1, h2, h3;\n"
                "    /* seed: compute SHA256(pass) -> st[8], then derive LE probe */\n"
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
                "        /* SHA256: BE state -> LE probe key (mirror usp_sha256_buf_global) */\n"
                "        h0 = ((st[0] & 0x000000ffu) << 24) | ((st[0] & 0x0000ff00u) << 8)\n"
                "           | ((st[0] & 0x00ff0000u) >> 8) | ((st[0] & 0xff000000u) >> 24);\n"
                "        h1 = ((st[1] & 0x000000ffu) << 24) | ((st[1] & 0x0000ff00u) << 8)\n"
                "           | ((st[1] & 0x00ff0000u) >> 8) | ((st[1] & 0xff000000u) >> 24);\n"
                "        h2 = ((st[2] & 0x000000ffu) << 24) | ((st[2] & 0x0000ff00u) << 8)\n"
                "           | ((st[2] & 0x00ff0000u) >> 8) | ((st[2] & 0xff000000u) >> 24);\n"
                "        h3 = ((st[3] & 0x000000ffu) << 24) | ((st[3] & 0x0000ff00u) << 8)\n"
                "           | ((st[3] & 0x00ff0000u) >> 8) | ((st[3] & 0xff000000u) >> 24);\n";
            feed_line =
                "            usp_sha256_iter_hex64_feed(st);\n";
            break;
        default:
            fprintf(stderr,
                "FATAL: %s:%d hx unsalted-single emit kernel: primitive "
                "'%s' (id=%d) not wired in Phase 1b Batch 1 (job=e%d). "
                "Batch-1 wired set is md5/md4/sha1/sha256.\n",
                __FILE__, __LINE__, prim_name ? prim_name : "(null)",
                (int)pid, job_enum);
            exit(1);
    }

    return hx_appendf(out, cap, len,
        "// hx: unsalted-single kernel for e%d prim=%s; probe uses 4 LE uints.\n"
        "// kernel signature mirrors kernelb_hx_codegen_phase0 (family) so\n"
        "// the dispatcher binds the same 16 args. The 4 salt args are\n"
        "// IGNORED (this shape is unsalted). reqd_work_group_size(64) pins\n"
        "// WG size to the dispatcher lsize=64 (same R8 fix as e347/family).\n"
        "//\n"
        "// Iter v1 (2026-05-31): runtime for-loop reading params.max_iter\n"
        "// (OCLParams offset 60). At iter==1 the body is byte-equivalent\n"
        "// to the Batch-1 single-probe kernel (no feed runs). At iter>1\n"
        "// each iter probes then hex-feeds the digest to the next iter,\n"
        "// mirroring legacy md5_rules_phase0 (gpu_md5_rules.cl:1158-1193)\n"
        "// byte-for-byte. R7 per-iter mask = (1u << (iter & 31u)).\n"
        "__attribute__((reqd_work_group_size(64,1,1)))\n"
        "__kernel void kernelb_hx_codegen_phase0(\n"
        "    __global const uchar         *payload,\n"
        "    __global const uchar         *b_packed_buf,\n"
        "    __global const uint          *b_chunk_index,\n"
        "    __global const uchar         *salts,\n"
        "    __global const uint          *salt_offsets,\n"
        "    __global const ushort        *salt_lens,\n"
        "    __global const uint          *compact_fp,\n"
        "    __global const uint          *compact_idx,\n"
        "    __global const uchar         *hash_data_buf,\n"
        "    __global const ulong         *hash_data_off,\n"
        "    __global uint                *hits,\n"
        "    __global volatile uint       *hit_count,\n"
        "    __global const ulong         *overflow_keys,\n"
        "    __global const uchar         *overflow_hashes,\n"
        "    __global const uint          *overflow_offsets,\n"
        "    __global volatile uint       *hashes_shown\n"
        "    )\n"
        "{\n"
        "    __global const OCLParams *params_buf =\n"
        "        (__global const OCLParams *)payload;\n"
        "    OCLParams params = *params_buf;\n"
        "\n"
        "    uint gid = get_global_id(0);\n"
        "    uint word_idx = gid;\n"
        "    if (word_idx >= params.num_words) return;\n"
        "\n"
        "    (void)salts; (void)salt_offsets; (void)salt_lens;\n"
        "\n"
        "    uint wpos = b_chunk_index[word_idx];\n"
        "    if (wpos >= params.packed_size) return;  // defensive\n"
        "    uint plen = (uint)b_packed_buf[wpos];\n"
        "    __global const uchar *pass_bytes = b_packed_buf + wpos + 1u;\n"
        "\n"
        "    // OP_CALL %s (seed: single hash of the unsalted pass; iter==1)\n"
        "%s"
        "\n"
        "    __global volatile uint *ovr_set =\n"
        "        (__global volatile uint *)(payload + 100);\n"
        "    __global volatile uint *ovr_gid =\n"
        "        (__global volatile uint *)(payload + 104);\n"
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

int hx_emit_unsalted_single_opencl(
    char **out, size_t *out_cap,
    const hx_program *prog,
    const struct hx_specialization *spec,
    const struct hx_spec_entry *entry)
{
    if (!out || !out_cap || !prog || !spec || !entry) {
        fprintf(stderr,
            "FATAL: %s:%d hx_emit_unsalted_single_opencl: NULL argument "
            "(out=%p cap=%p prog=%p spec=%p entry=%p)\n",
            __FILE__, __LINE__, (void*)out, (void*)out_cap,
            (void*)prog, (void*)spec, (void*)entry);
        return -1;
    }

    const char *prim_name = hx_callname_for_entry(entry, 1);
    if (!prim_name) {
        fprintf(stderr,
            "FATAL: %s:%d hx_emit_unsalted_single_opencl: e%d %s code[1] "
            "callname is NULL (sidecar missing).\n",
            __FILE__, __LINE__, entry->job_enum,
            entry->name ? entry->name : "(noname)");
        return -1;
    }
    enum hx_primitive_id pid = hx_primitive_id_for_name(prim_name);
    if (pid == HX_PRIM_UNKNOWN) {
        fprintf(stderr,
            "FATAL: %s:%d hx_emit_unsalted_single_opencl: e%d %s callname "
            "'%s' not recognized in hx_emit_primitives.c table.\n",
            __FILE__, __LINE__, entry->job_enum,
            entry->name ? entry->name : "(noname)", prim_name);
        return -1;
    }
    if (pid != HX_PRIM_MD5 && pid != HX_PRIM_MD4 &&
        pid != HX_PRIM_SHA1 && pid != HX_PRIM_SHA256) {
        fprintf(stderr,
            "FATAL: %s:%d hx_emit_unsalted_single_opencl: e%d %s primitive "
            "'%s' is not in the Phase 1b Batch-1 wired set "
            "(md5/md4/sha1/sha256). Batch 2/3 widen this gate.\n",
            __FILE__, __LINE__, entry->job_enum,
            entry->name ? entry->name : "(noname)", prim_name);
        return -1;
    }

    size_t cur_len = 0;
    if (*out == NULL) *out_cap = 0;
    int rc;

    rc = hx_appendf(out, out_cap, &cur_len,
        "// hx codegen: PATTERN UNSALTED_SINGLE matched (e%d %s prim=%s)\n"
        "// hx: program ncode=%d nvars=%d max_stack=%d has_emit=%d\n"
        "// hx: code[1] role=%d (0=hex/default, 1=raw-bin; identical digest)\n"
        "// hx: this kernel JIT-compiled with gpu_common_str prepended\n"
        "\n",
        entry->job_enum, entry->name ? entry->name : "(noname)", prim_name,
        prog->ncode, prog->nvars, prog->max_stack, prog->has_emit,
        (int)prog->code[1].u.call.role);
    if (rc < 0) return rc;

    rc = emit_unsalted_single_helpers(out, out_cap, &cur_len);
    if (rc < 0) return rc;

    rc = emit_unsalted_single_kernel(out, out_cap, &cur_len,
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
