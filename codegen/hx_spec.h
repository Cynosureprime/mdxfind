/*
 * hx_spec.h -- codegen specialization-context types for the in-process
 *              hx P4 state-machine walker.
 *
 * Sub-phase 2a.2 (per project_hx_codegen_phase2_3_spec_2026-05-21.md
 * D12.2.c REVISED-AGAIN -- Path A): the invented bytecode + struct
 * hx_spec from 2a.1 are DISCARDED. The walker consumes the production
 * hx VM `hx_program` directly via the 16-opcode `hx_opcode` enum in
 * hx_vm.h. This header now only carries codegen-context types --
 * backend selector, specialization tuple, salt regime, iteration shape --
 * which are NOT part of hx the language; they describe the
 * codegen-time invocation envelope around an hx program.
 *
 * $Revision: 1.2 $
 * $Log: hx_spec.h,v $
 * Revision 1.2  2026/05/21 23:23:29  dlr
 * sub-phase 2a.2: replace invented bytecode enum with the production hx_vm.h types. The walker now consumes hx_program directly via the 16-opcode hx_opcode enum. Header now carries only codegen-context (backend, salt regime, iter shape, digest endianness, specialization) which are NOT part of hx the language. struct hx_spec is gone; struct hx_specialization extended with iter_shape, digest_endianness, emit_width fields previously on hx_spec.
 *
 */

#ifndef HX_SPEC_H
#define HX_SPEC_H

#include <stddef.h>
#include <stdint.h>

/* The production VM types -- hx_inst, hx_program, hx_opcode, hx_role,
 * hx_func_entry -- live in hx_vm.h. The walker dispatches on the
 * 16-opcode enum defined there. */
#include "../hx_vm.h"
#include "hx_spec_entry.h"

/*
 * Backend selector. The walker is a single function; per-backend emit
 * helpers branch on this enum. See spec D12.3.a (single walker, backend
 * parameter, per-backend emit helper files).
 */
enum hx_backend {
    HX_BACKEND_OPENCL = 0,
    HX_BACKEND_METAL  = 1
};

/*
 * Salt-count regime drives the outer-loop topology of emitted kernel B.
 * tp0 pattern (see feedback_tp0_pattern_is_correct_for_pascal_salted_md5)
 * uses BATCH_64 for fast-hash salted MD5 family. Walker bakes this as a
 * compile-time constant into the emitted source.
 */
enum hx_salt_count_regime {
    HX_SALT_SINGLE        = 0,
    HX_SALT_BATCH_64      = 1,
    HX_SALT_LARGE_FANOUT  = 2
};

/*
 * Digest byte order of the EMITTED final digest. The intermediate digest
 * after each digest-bin call is always little-endian on-device (the way
 * OpenCL/Metal md5/sha primitives work); this field controls the
 * byteswap the walker emits before the final hit-emit.
 */
enum hx_digest_endianness {
    HX_DIGEST_LE = 0,
    HX_DIGEST_BE = 1
};

/*
 * Iteration shape. e347 is ITER_NONE (iter_count=1). HEX_FEEDBACK is
 * the SHA512CRYPT/MD5CRYPT family (feed prior digest back as hex32/64
 * string). BIN_FEEDBACK is the raw-binary feedback family.
 */
enum hx_iter_shape {
    HX_ITER_NONE         = 0,
    HX_ITER_HEX_FEEDBACK = 1,
    HX_ITER_BIN_FEEDBACK = 2
};

/*
 * Specialization tuple -- runtime invocation context baked into emitted
 * source. (iter, mask, rules, salt regime) drive structural variants.
 * e347 fills: iter_count_if_fixed=1, has_rules=1, has_masks=0,
 * has_bf=0, salt_count_regime=BATCH_64.
 */
struct hx_specialization {
    uint32_t                    iter_count_if_fixed; /* 1 = no iter loop */
    uint8_t                     has_rules;
    uint8_t                     has_masks;
    uint8_t                     has_bf;
    uint32_t                    salt_minlen;
    uint32_t                    salt_maxlen;
    enum hx_salt_count_regime   salt_count_regime;
    enum hx_iter_shape          iter_shape;
    enum hx_digest_endianness   digest_endianness;
    uint32_t                    emit_width;       /* bytes of final digest */
};

#endif /* HX_SPEC_H */
