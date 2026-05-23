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
 * $Revision: 1.7 $
 * $Log: hx_emit_opencl.c,v $
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

/* Per-primitive outer body emit: RIPEMD-160 (LE-schedule, 5-uint state). */
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
static int emit_family_md5pass_kernel(char **out, size_t *cap, size_t *len,
                                      enum hx_primitive_id outer_id,
                                      const char *outer_name,
                                      int outer_digest_bytes,
                                      int job_enum)
{
    int rc;

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
        case HX_PRIM_MD4:
            helper_name = "outer_md4_concat_then_hash";
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
              ((outer_id == HX_PRIM_MD4) ?
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
              : /* SHA512 */
                "    uint h0, h1, h2, h3;\n"
                "    outer_sha512_concat_then_hash(ia, ib, ic, id,\n"
                "                                  pass_bytes, (int)plen,\n"
                "                                  &h0, &h1, &h2, &h3);\n"),
        helper_has_h4
            ? "    (void)h4;  // 5th word reserved for round-trip readback.\n"
            : "");

    (void)helper_name;  /* selected by outer_id in the literal above */
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
    /* Sub-phase 5a.4 (2026-05-23): 7 of 8 5a-supported primitives wired.
     * e123 MD5MD5PASS (HX_PRIM_MD5) is outlier (multi-emit variant
     * deferred). 22 5b-deferred primitives are filtered upstream by
     * supported_5a check. */
    if (outer_id == HX_PRIM_MD5) {
        fprintf(stderr,
            "FATAL: %s:%d hx_emit_family_md5pass_opencl: e%d %s outer "
            "primitive 'md5' (e123 MD5MD5PASS) is an outlier in 5a -- "
            "the family is multi-emit (canonical + colon variant) and "
            "ships in a separate multi-emit sub-phase. CPU continues "
            "to handle e123 in the interim.\n",
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
            "FATAL: %s:%d hx_emit_family_md5pass_opencl: e%d %s outer "
            "primitive '%s' is in supported_5a set but not in the "
            "5a.4 wired subset (md4 sha1 sha224 sha256 sha384 sha512 "
            "rmd160). Either add to dispatch above or this is a logic "
            "bug.\n",
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
        case HX_PRIM_SHA1:
            rc = emit_outer_sha1_concat_then_hash(out, out_cap, &cur_len); break;
        case HX_PRIM_MD4:
            rc = emit_outer_md4_concat_then_hash(out, out_cap, &cur_len); break;
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
        default:
            fprintf(stderr,
                "FATAL: %s:%d hx_emit_family_md5pass_opencl: unreachable "
                "(outer_id=%d not in 5a.4 wired set)\n",
                __FILE__, __LINE__, (int)outer_id);
            return -1;
    }
    if (rc < 0) return rc;

    rc = emit_family_md5pass_kernel(out, out_cap, &cur_len,
                                    outer_id, outer_name,
                                    outer_digest_bytes, entry->job_enum);
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
