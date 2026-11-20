/*
 * hx_gen.h -- emit a template core (the gpu_template hooks) from an hx plan.
 *
 * The emitted text is one source for every backend. It is written in a small
 * macro dialect -- HX_FINALIZE_SIG, HX_ITERATE_SIG, ST(i), the HXB_* block
 * calls, the HX_ABS* absorbers -- which hx_gen_dialect() defines per backend:
 * OpenCL, Metal, and C. The C dialect exists so the very text a GPU compiles
 * can be compiled and run on the host and compared with the hx VM
 * (codegen/tests/hx_gen_check.c).
 *
 * template_state.h holds the STORED FORM's binary as little-endian words, so
 * the probe and the hit record are fixed per backend and need no byte swap.
 *
 * $Revision: 1.4 $
 * $Log: hx_gen.h,v $
 * Revision 1.4  2026/11/20 05:22:13  dlr
 * Generic hx walker stage 1c: user-defined types run on the GPU through the generated walker. A user type is served when it takes no salt or user, declares no stored form, and its result is the hex of its final digest; rounds past x01 re-run the whole program on the previous round's hex text, keeping its case, as the userdef CPU arm does. The selected user type's status line says whether it runs on the GPU, or why not, in place of the former not-yet-enabled line. Verified: the C mirror agrees for 529 of 529 cores, including 10 user cores whose later rounds are checked against the hx VM re-running the program, and a planted case defect is caught in exactly the 8 case-sensitive ones; the 519 catalog cores are byte-identical; on dev3 (M2 Max, Metal) and gp1 (Tahiti, OpenCL) 10 user types match -G none at x01 and x02 and deeper with rules, a -n mask and a mixed run with a catalog type, except one type whose kernel the Tahiti compiler crashes on, which is FATAL there as for any such kernel.
 *
 * Revision 1.3  2026/11/20 02:52:03  dlr
 * Generic hx walker stage 2: the plan and emitter take a salt or user channel, HMAC, round-1-only types and peppers. A salt or user variable reads the 6-argument salted template_finalize; HMAC is generated over any existing body following mhash: block from pblock with 0 as 64, MD2 16, HAVAL and SHA-384/512 128, a key longer than the block replaced by its digest, the inner digest entering the outer hash at the body width, Tiger in mhash byte order; HX_NO_ITERATE marks a type compared at round 1 only. A pepper rides the salt channel: the host packs salt . pepper . the pepper length in 2 bytes, and the core splits it. Verified: the C mirror agrees for 519 of 519 cores over 67548 finalize and 64194 iterate vectors, including the 14 pepper cores over 8 salts by 6 peppers up to 300 bytes; planted defects in the split and in the length high byte are caught in exactly those 14; the 505 earlier cores are byte-identical.
 *
 * Revision 1.2  2026/11/19 02:29:18  dlr
 * Generic hx walker stage 1b: RIPEMD-128, RIPEMD-160, Tiger, Whirlpool and HAVAL (3, 4 and 5 pass, 128 to 256 bit) bodies for the OpenCL and Metal dialects. Body rows carry block size, word mode, absorber, finaliser and fold; existing bodies emit identical text. hx_gen_c_mirror reports whether the C dialect can check a plan on the host.
 *
 * Revision 1.1  2026/11/18 04:08:38  dlr
 * Initial revision.
 *
 */

#ifndef HX_GEN_H
#define HX_GEN_H

#include <stddef.h>
#include "hx_plan.h"

#ifdef __cplusplus
extern "C" {
#endif

enum hx_gen_backend { HXG_OPENCL, HXG_METAL, HXG_C };

/*
 * Emit the type-specific hooks (template_finalize, template_iterate, the
 * HASH_WORDS and HASH_BLOCK_BYTES defines) for plan into a malloc'd string.
 * x01_silent: the type reports nothing at x01, so the kernel compares only
 * from iteration 2 (HX_X01_SILENT). Returns NULL with a reason in why when the
 * plan has a shape this generator does not produce yet.
 */
char *hx_gen_core(const struct hx_plan *plan, int x01_silent, char *why, int whycap);

/* As hx_gen_core, for a type whose CPU code reports round 1 only whatever -i
 * says (hx_spec_entry drift_ok 3, the ledger's no-iter): the kernel compares
 * round 1 and stops (HX_NO_ITERATE). */
char *hx_gen_core_x01(const struct hx_plan *plan, char *why, int whycap);
/* A user-defined type's core: x02 and deeper re-run the whole program with
 * the previous round's hex text as the password, as the userdef CPU arm does.
 * The result must be the hex of the final digest, unsalted. */
char *hx_gen_core_whole(const struct hx_plan *plan, char *why, int whycap);

/* The dialect and the fixed hooks for backend, to precede hx_gen_core text. */
const char *hx_gen_dialect(enum hx_gen_backend backend);

/* Words of the stored form: what HASH_WORDS will be. */
int hx_gen_words(const struct hx_plan *plan);

/* Bytes of the digest the stored form shows (its binary width). */
int hx_gen_digest_bytes(const struct hx_plan *plan);

/* 1 when every body in plan has a block function in the C dialect, so the
 * host check can run its core; the stage-1b bodies do not. */
int hx_gen_c_mirror(const struct hx_plan *plan);

#ifdef __cplusplus
}
#endif

#endif /* HX_GEN_H */
