/*
 * hx_plan.h -- the generic GPU walker's plan: an hx program read once, at
 * load time, into a tree of values, with the reason it is or is not
 * eligible for the GPU.
 *
 * A value is a SOURCE (pass, salt, user, a constant, a digest held in
 * registers) or a transform of other values. Nothing is materialized: a hash
 * call absorbs its argument segments byte by byte, in order, and every
 * transform here can produce its bytes in order. rev additionally needs the
 * byte at an arbitrary index, which is index arithmetic over its source when
 * that source is ADDRESSABLE (see hx_plan_node.addressable).
 *
 * Stage 1a scope: bodies md4, md5, sha1, sha224, sha256, sha384, sha512;
 * transforms hex, upper, lower, cap, cut, trunc, pad, rev, rot13, utf16le,
 * utf16be, base64, fromhex; channels pass plus at most one of salt or user.
 * Anything else is ineligible, with the reason.
 *
 * $Revision: 1.5 $
 * $Log: hx_plan.h,v $
 * Revision 1.5  2026/11/20 15:45:12  dlr
 * Generic hx walker stages 3b and 4a. New walker bodies: MySQL 3 OLD_PASSWORD as a byte-wise accumulator in the dialect, BLAKE2b-512 and -256 and BLAKE2s-256 over the common b2b and b2s compress functions, and SHA3 and Keccak at 224, 256, 384 and 512 bits over keccakf1600; a final digest shorter than 16 bytes is compared as four words with the upper ones zero. User-defined types may now name mdxfind own hash functions: sha224, sha384, rmd128, rmd160, tiger, wrl, md2, gost, sne128, sne256, the fifteen HAVAL forms, mysql3, blake2b512, blake2b256, blake2s256, sha3_224 to sha3_512 and keccak224 to keccak512, each the call its catalog type makes. hashpipe registers its BLAKE2 digests as hx functions, which the verify registration had hidden. The drift check reads a NAME-prefixed hex digest and the bare digest as one value. 11 more catalog types run on the GPU: e410 e416 e449 e474 e478 e548 e711 e842 e843 e846 e847. Verified: each new user function agrees with its catalog type at x01 to x03 on the CPU and with hashpipe -X; the C mirror agrees for 567 of 567 cores and every earlier core changed only by the added dialect text; on dev3 (M2 Max, Metal) all new catalog types and 42 user types match -G none, and on gp1 (Tahiti, OpenCL) all do except the md2 user type, whose kernel the Tahiti compiler crashes on, which is FATAL there as for any such kernel.
 *
 * Revision 1.4  2026/11/20 02:52:03  dlr
 * Generic hx walker stage 2: the plan and emitter take a salt or user channel, HMAC, round-1-only types and peppers. A salt or user variable reads the 6-argument salted template_finalize; HMAC is generated over any existing body following mhash: block from pblock with 0 as 64, MD2 16, HAVAL and SHA-384/512 128, a key longer than the block replaced by its digest, the inner digest entering the outer hash at the body width, Tiger in mhash byte order; HX_NO_ITERATE marks a type compared at round 1 only. A pepper rides the salt channel: the host packs salt . pepper . the pepper length in 2 bytes, and the core splits it. Verified: the C mirror agrees for 519 of 519 cores over 67548 finalize and 64194 iterate vectors, including the 14 pepper cores over 8 salts by 6 peppers up to 300 bytes; planted defects in the split and in the length high byte are caught in exactly those 14; the 505 earlier cores are byte-identical.
 *
 * Revision 1.3  2026/11/19 04:28:21  dlr
 * Generic hx walker stage 1b, second batch: MD2, GOST R 34.11-94 (test S-box), Snefru-128 and Snefru-256 bodies for the OpenCL and Metal dialects. MD2 keeps its 48-byte state and 16-byte checksum in a 16-word state array and ends with the checksum block; GOST keeps the 256-bit block sum in message words 8 to 15 and ends with the length block and then the sum block; Snefru zero-pads and ends with the big-endian bit length. Existing bodies emit identical core text.
 *
 * Revision 1.2  2026/11/19 02:29:18  dlr
 * Generic hx walker stage 1b: RIPEMD-128, RIPEMD-160, Tiger, Whirlpool and HAVAL (3, 4 and 5 pass, 128 to 256 bit) bodies for the OpenCL and Metal dialects. Body rows carry block size, word mode, absorber, finaliser and fold; existing bodies emit identical text. hx_gen_c_mirror reports whether the C dialect can check a plan on the host.
 *
 * Revision 1.1  2026/11/18 04:08:38  dlr
 * Initial revision.
 *
 */

#ifndef HX_PLAN_H
#define HX_PLAN_H

#include <stdint.h>
#include "../hx_vm.h"

#ifdef __cplusplus
extern "C" {
#endif

enum hx_plan_kind {
    HXP_VAR,        /* pass, salt or user */
    HXP_CONST,      /* string literal */
    HXP_CAT,        /* a . b */
    HXP_BODY,       /* a hash over its one argument, output in a role */
    HXP_XFORM,      /* a byte transform of its first argument */
    HXP_HMAC        /* hmac_X(message a, key b) over body op, mhash's rules */
};

enum hx_plan_body {
    HXB_MD4, HXB_MD5, HXB_SHA1, HXB_SHA224, HXB_SHA256, HXB_SHA384,
    HXB_SHA512,
    /* stage 1b: no host C mirror (hx_gen_c_mirror) */
    HXB_RMD128, HXB_RMD160, HXB_TIGER, HXB_WRL,
    HXB_HAV128_3, HXB_HAV128_4, HXB_HAV128_5, HXB_HAV160_3, HXB_HAV160_4,
    HXB_HAV160_5, HXB_HAV192_3, HXB_HAV192_4, HXB_HAV192_5, HXB_HAV224_3,
    HXB_HAV224_4, HXB_HAV224_5, HXB_HAV256_3, HXB_HAV256_4, HXB_HAV256_5,
    HXB_MD2, HXB_GOST, HXB_SNE128, HXB_SNE256,
    /* stage 2, HMAC only: Tiger as mhash writes it (each 64-bit word
     * most-significant byte first, the reverse of reference Tiger), whole and
     * cut to 128 and 160 bits -- mhash's TIGER, TIGER128 and TIGER160, which
     * HMAC-TIGER uses at every step */
    HXB_TIGER128, HXB_TIGER160, HXB_TIGERMH,
    /* stage 3b: MySQL 3 OLD_PASSWORD, a byte-wise accumulator, no blocks */
    HXB_MYSQL3,
    /* stage 3b: over gpu_common's b2b_compress, b2s_compress, keccakf1600 */
    HXB_B2B512, HXB_B2B256, HXB_SHA3_256,
    /* stage 4: reached by user types (mdx_hx_register) */
    HXB_B2S256, HXB_SHA3_224, HXB_SHA3_384, HXB_SHA3_512,
    HXB_KEC224, HXB_KEC256, HXB_KEC384, HXB_KEC512,
    HXB_COUNT
};

enum hx_plan_xform {
    HXX_HEX, HXX_UPPER, HXX_LOWER, HXX_CAP_FIRST, HXX_CAP_AT, HXX_CUT,
    HXX_PAD, HXX_REV, HXX_ROT13, HXX_UTF16LE, HXX_UTF16BE, HXX_BASE64,
    HXX_FROMHEX, HXX_COUNT
};

/* Output view of a body: what bytes the call yields. */
enum hx_plan_view { HXV_BIN, HXV_HEX, HXV_HEXUC, HXV_B64 };

#define HXP_MAX_NODES 64
#define HXP_LEN_RUNTIME (-1)

struct hx_plan_node {
    uint8_t kind;           /* enum hx_plan_kind */
    uint8_t op;             /* enum hx_plan_body or enum hx_plan_xform */
    uint8_t view;           /* HXP_BODY: enum hx_plan_view */
    uint8_t var;            /* HXP_VAR: HX_SLOT_PASS / _SALT / _USERID */
    uint8_t addressable;    /* byte i computable without the bytes before it */
    int     a, b;           /* children (node index), -1 if none */
    int     ia, ib;         /* integer arguments: cut start,len; pad len; cap pos */
    int     nia;            /* how many integer arguments were given */
    int     len;            /* static length in bytes, or HXP_LEN_RUNTIME */
    int     maxlen;         /* upper bound when len is runtime; -1 unbounded */
    const char *str;        /* HXP_CONST bytes */
};

enum hx_plan_channel { HXC_NONE, HXC_SALT, HXC_USER };

/* How mdxfind's -i iterates the type: round N+1 is the outer hash, with the
 * case transforms around it, over round N's text. iter_body is that outer
 * body node; iter_wrap lists the transforms around it, outermost first. */
#define HXP_MAX_WRAP 4

struct hx_plan {
    int     eligible;
    char    reason[160];
    int     n;
    struct hx_plan_node node[HXP_MAX_NODES];
    int     root;
    int     channel;        /* enum hx_plan_channel */
    int     pepper;         /* 1: also reads pepper, packed after the salt */
    int     out_len;        /* static length of the final value, or -1 */
    int     final_body;     /* the digest the stored form shows */
    int     final_prefix;   /* HXP_CONST node in front of it, or -1 */
    int     final_cut;      /* HXX_CUT node applied to it, or -1 */
    int     final_b64;      /* 1 when the stored form is base64 of the binary */
    int     iter_body;      /* node index, or -1 when the type cannot iterate */
    int     iter_nwrap;
    int     iter_wrap[HXP_MAX_WRAP];
};

/*
 * Build the plan for prog. call_names[i] is the function name of the OP_CALL
 * at code index i (the hx_specs_data sidecar); when NULL, names come from
 * each call's registry entry. pass_max bounds the candidate length (the
 * carrier's input limit), used for maxlen. Returns plan->eligible.
 */
int hx_plan_build(const hx_program *prog, const char *const *call_names,
                  int pass_max, struct hx_plan *plan);

/* One-line rendering of the plan tree, for diagnostics and tests. */
void hx_plan_render(const struct hx_plan *plan, int node, char *out, int cap);

#ifdef __cplusplus
}
#endif

#endif /* HX_PLAN_H */
