/*
 * hx_plan.c -- build the generic GPU walker's plan from an hx program.
 * See hx_plan.h.
 *
 * $Revision: 1.6 $
 * $Log: hx_plan.c,v $
 * Revision 1.6  2026/11/20 15:45:12  dlr
 * Generic hx walker stages 3b and 4a. New walker bodies: MySQL 3 OLD_PASSWORD as a byte-wise accumulator in the dialect, BLAKE2b-512 and -256 and BLAKE2s-256 over the common b2b and b2s compress functions, and SHA3 and Keccak at 224, 256, 384 and 512 bits over keccakf1600; a final digest shorter than 16 bytes is compared as four words with the upper ones zero. User-defined types may now name mdxfind own hash functions: sha224, sha384, rmd128, rmd160, tiger, wrl, md2, gost, sne128, sne256, the fifteen HAVAL forms, mysql3, blake2b512, blake2b256, blake2s256, sha3_224 to sha3_512 and keccak224 to keccak512, each the call its catalog type makes. hashpipe registers its BLAKE2 digests as hx functions, which the verify registration had hidden. The drift check reads a NAME-prefixed hex digest and the bare digest as one value. 11 more catalog types run on the GPU: e410 e416 e449 e474 e478 e548 e711 e842 e843 e846 e847. Verified: each new user function agrees with its catalog type at x01 to x03 on the CPU and with hashpipe -X; the C mirror agrees for 567 of 567 cores and every earlier core changed only by the added dialect text; on dev3 (M2 Max, Metal) all new catalog types and 42 user types match -G none, and on gp1 (Tahiti, OpenCL) all do except the md2 user type, whose kernel the Tahiti compiler crashes on, which is FATAL there as for any such kernel.
 *
 * Revision 1.5  2026/11/20 12:41:38  dlr
 * Generic hx walker stage 3a: the plan builder accepts named bindings, where a later use of a bound name refers to the node it holds, and unrolls the counted repeat f^k into k calls; any other control flow is still refused, as is a repeat too long to unroll. The drift check reads bindings and f^k calls and applies its iteration check to the final statement. 28 more catalog types agree with their CPU arms and run on the GPU. Verified: the C mirror agrees for 567 of 567 cores and a planted repeat-count defect is caught; the 529 earlier cores are byte-identical; on dev3 (M2 Max, Metal) and gp1 (Tahiti, OpenCL) all 28 match -G none at x01 and at x02 and deeper with rules.
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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "hx_plan.h"

static const struct { const char *name; int digest; } bodies[HXB_COUNT] = {
    [HXB_MD4]    = { "md4",    16 },
    [HXB_MD5]    = { "md5",    16 },
    [HXB_SHA1]   = { "sha1",   20 },
    [HXB_SHA224] = { "sha224", 28 },
    [HXB_SHA256] = { "sha256", 32 },
    [HXB_SHA384] = { "sha384", 48 },
    [HXB_SHA512] = { "sha512", 64 },
    [HXB_RMD128] = { "rmd128", 16 },
    [HXB_RMD160] = { "rmd160", 20 },
    [HXB_TIGER]  = { "tiger",  24 },
    [HXB_WRL]    = { "wrl",    64 },
    /* HAVAL: the catalog's names; hav128 and hav256 are the 3-pass forms */
    [HXB_HAV128_3] = { "hav128",   16 }, [HXB_HAV128_4] = { "hav128_4", 16 },
    [HXB_HAV128_5] = { "hav128_5", 16 }, [HXB_HAV160_3] = { "hav160_3", 20 },
    [HXB_HAV160_4] = { "hav160_4", 20 }, [HXB_HAV160_5] = { "hav160_5", 20 },
    [HXB_HAV192_3] = { "hav192_3", 24 }, [HXB_HAV192_4] = { "hav192_4", 24 },
    [HXB_HAV192_5] = { "hav192_5", 24 }, [HXB_HAV224_3] = { "hav224_3", 28 },
    [HXB_HAV224_4] = { "hav224_4", 28 }, [HXB_HAV224_5] = { "hav224_5", 28 },
    [HXB_HAV256_3] = { "hav256",   32 }, [HXB_HAV256_4] = { "hav256_4", 32 },
    [HXB_HAV256_5] = { "hav256_5", 32 },
    [HXB_MD2]    = { "md2",    16 },
    [HXB_GOST]   = { "gost",   32 },    /* GOST R 34.11-94, test S-box */
    [HXB_SNE128] = { "sne128", 16 },
    [HXB_SNE256] = { "sne256", 32 },
    [HXB_TIGER128] = { "tiger128", 16 },
    [HXB_TIGER160] = { "tiger160", 20 },
    [HXB_TIGERMH]  = { "tiger_mhash", 24 },
    [HXB_MYSQL3]   = { "mysql3", 8 },
    [HXB_B2B512]   = { "blake2b512", 64 },
    [HXB_B2B256]   = { "blake2b256", 32 },
    [HXB_SHA3_256] = { "sha3_256", 32 },
    [HXB_B2S256]   = { "blake2s256", 32 },
    [HXB_SHA3_224] = { "sha3_224", 28 },
    [HXB_SHA3_384] = { "sha3_384", 48 }, [HXB_SHA3_512] = { "sha3_512", 64 },
    [HXB_KEC224]   = { "keccak224", 28 }, [HXB_KEC256] = { "keccak256", 32 },
    [HXB_KEC384]   = { "keccak384", 48 }, [HXB_KEC512] = { "keccak512", 64 },
};

/* hmac_X: the body each mhash HMAC name runs over (the CPU and hashpipe use
 * mhash_hmac_init with these hashids). MHASH_HAVALnnn is the 3-pass HAVAL;
 * MHASH_TIGER192 is plain Tiger. Every other name is the body's own. */
static const struct { const char *name; int body; } hmac_alias[] = {
    { "hav128", HXB_HAV128_3 }, { "hav160", HXB_HAV160_3 }, { "hav192", HXB_HAV192_3 },
    { "hav224", HXB_HAV224_3 }, { "hav256", HXB_HAV256_3 },
    /* mhash's Tiger byte order (hx_plan.h) */
    { "tiger192", HXB_TIGERMH }, { "tiger", HXB_TIGERMH },
};

static const char *xnames[HXX_COUNT] = {
    [HXX_HEX] = "hex", [HXX_UPPER] = "upper", [HXX_LOWER] = "lower",
    [HXX_CAP_FIRST] = "cap", [HXX_CAP_AT] = "cap", [HXX_CUT] = "cut",
    [HXX_PAD] = "pad", [HXX_REV] = "rev", [HXX_ROT13] = "rot13",
    [HXX_UTF16LE] = "utf16le", [HXX_UTF16BE] = "utf16be",
    [HXX_BASE64] = "base64", [HXX_FROMHEX] = "fromhex",
};

/* A stack slot: a node, or an integer literal (only ever a call argument). */
struct slot { int node; int is_int; long long ival; };

/* Per-node facts used while building, beyond what the plan keeps. */
struct facts { uint8_t ascii, utf8; };   /* every byte < 0x80; valid UTF-8 */

static int fail(struct hx_plan *p, const char *fmt, const char *arg)
{
    p->eligible = 0;
    snprintf(p->reason, sizeof p->reason, fmt, arg ? arg : "");
    return 0;
}

static int add(struct hx_plan *p, struct hx_plan_node nd)
{
    if (p->n >= HXP_MAX_NODES) return -1;
    p->node[p->n] = nd;
    return p->n++;
}

static int sum_max(int a, int b) { return (a < 0 || b < 0) ? -1 : a + b; }

static int body_view_len(int digest, int view)
{
    switch (view) {
    case HXV_BIN: return digest;
    case HXV_B64: return ((digest + 2) / 3) * 4;
    default:      return digest * 2;
    }
}

/* Length and bound of node from its children. */
static void size_node(struct hx_plan *p, struct hx_plan_node *nd)
{
    struct hx_plan_node *a = nd->a >= 0 ? &p->node[nd->a] : NULL;
    int al = a ? a->len : 0, am = a ? (a->len >= 0 ? a->len : a->maxlen) : 0;
    nd->len = HXP_LEN_RUNTIME;
    nd->maxlen = -1;
    switch (nd->kind) {
    case HXP_CAT: {
        struct hx_plan_node *b = &p->node[nd->b];
        int bm = b->len >= 0 ? b->len : b->maxlen;
        if (al >= 0 && b->len >= 0) nd->len = al + b->len;
        else nd->maxlen = sum_max(am, bm);
        return;
    }
    case HXP_BODY: case HXP_HMAC:
        nd->len = body_view_len(bodies[nd->op].digest, nd->view);
        return;
    default:
        break;
    }
    switch (nd->op) {
    case HXX_HEX:
        if (al >= 0) nd->len = 2 * al; else nd->maxlen = am < 0 ? -1 : 2 * am;
        return;
    case HXX_UTF16LE: case HXX_UTF16BE:
        /* UTF-8 to UTF-16 never more than doubles the byte count */
        if (al >= 0 && nd->addressable) nd->len = 2 * al;  /* ASCII source */
        else nd->maxlen = am < 0 ? -1 : 2 * am;
        return;
    case HXX_BASE64:
        if (al >= 0) nd->len = ((al + 2) / 3) * 4;
        else nd->maxlen = am < 0 ? -1 : ((am + 2) / 3) * 4;
        return;
    case HXX_FROMHEX:
        if (al >= 0) nd->len = al / 2; else nd->maxlen = am < 0 ? -1 : am / 2;
        return;
    case HXX_PAD:
        nd->len = nd->ia;
        return;
    case HXX_CUT: {
        int start = nd->ia, len = nd->nia >= 2 ? nd->ib : -1;
        if (al >= 0) {
            /* exactly fn_cut's arithmetic */
            if (start < 0) { start = al + start; if (start < 0) start = 0; }
            if (start > al) start = al;
            if (len < 0) len = al - start;
            if (start + len > al) len = al - start;
            nd->len = len;
        } else {
            nd->maxlen = len >= 0 ? (am >= 0 && am < len ? am : len) : am;
        }
        return;
    }
    default:   /* upper, lower, cap, rev, rot13 keep the length */
        nd->len = al;
        nd->maxlen = a->maxlen;
        return;
    }
}

static int xform_for(const char *name, int nargs, int role, int *op)
{
    int i;
    if (strcmp(name, "trunc") == 0) { *op = HXX_CUT; return 1; }
    if (strcmp(name, "cap") == 0) { *op = nargs >= 2 ? HXX_CAP_AT : HXX_CAP_FIRST; return 1; }
    (void)role;
    for (i = 0; i < HXX_COUNT; i++)
        if (xnames[i] && strcmp(xnames[i], name) == 0) { *op = i; return 1; }
    return 0;
}

static int body_for(const char *name)
{
    int i;
    for (i = 0; i < HXB_COUNT; i++)
        if (strcmp(bodies[i].name, name) == 0) return i;
    return -1;
}

/* The body under hmac_X, or -1 when name is not an HMAC of a body here. */
static int hmac_body_for(const char *name)
{
    int i;
    if (strncmp(name, "hmac_", 5) != 0) return -1;
    name += 5;
    for (i = 0; i < (int)(sizeof hmac_alias / sizeof hmac_alias[0]); i++)
        if (strcmp(hmac_alias[i].name, name) == 0) return hmac_alias[i].body;
    i = body_for(name);
    /* tiger128/160 exist only as HMAC bodies */
    return i;
}

/* f^k compiles to one call of f and then a fixed do-while on a counter slot:
 *   PUSH_INT c; STORE s; L: CALL f/1; INC s; PUSH_VAR s; PUSH_INT lim; JUMP_LE L
 * which runs the call lim - c + 1 more times. Expand it into those calls, so
 * the walk below sees straight-line code; *ncode -1 when the repeat is longer
 * than the plan could hold anyway. Names follow their calls. */
#define HXP_MAX_CODE 512
static int unroll(const hx_program *prog, const char *const *call_names,
                  hx_inst *code, const char **names)
{
    int i, n = 0, r;
    for (i = 0; i < prog->ncode; i++) {
        const hx_inst *c = &prog->code[i];
        if (i + 6 < prog->ncode && c[0].op == OP_PUSH_INT && c[1].op == OP_STORE &&
            c[2].op == OP_CALL && c[2].u.call.nargs == 1 &&
            c[3].op == OP_INC && c[3].u.slot == c[1].u.slot &&
            c[4].op == OP_PUSH_VAR && c[4].u.slot == c[1].u.slot &&
            c[5].op == OP_PUSH_INT && c[6].op == OP_JUMP_LE && c[6].u.addr == i + 2) {
            long long times = c[5].u.ival - c[0].u.ival + 1;
            if (times < 1) times = 1;
            if (times > HXP_MAX_NODES) return -1;
            for (r = 0; r < times; r++) {
                if (n >= HXP_MAX_CODE) return -1;
                names[n] = call_names ? call_names[i + 2] : NULL;
                code[n++] = c[2];
            }
            i += 6;
            continue;
        }
        if (n >= HXP_MAX_CODE) return -1;
        names[n] = call_names ? call_names[i] : NULL;
        code[n++] = *c;
    }
    return n;
}

int hx_plan_build(const hx_program *prog, const char *const *call_names,
                  int pass_max, struct hx_plan *p)
{
    struct slot st[HXP_MAX_NODES];
    struct facts fx[HXP_MAX_NODES];
    int sp = 0, i, used_salt = 0, used_user = 0, used_pepper = 0, ncode;
    char num[24];
    static hx_inst code[HXP_MAX_CODE];     /* single-threaded: built on one thread */
    static const char *names[HXP_MAX_CODE];
    /* h = ...; values bound to the expression's own variables (slot 5 up) */
    int bound[64];

    memset(p, 0, sizeof *p);
    p->root = p->iter_body = -1;
    p->eligible = 1;
    if (!prog) return fail(p, "the expression did not compile", NULL);
    for (i = 0; i < 64; i++) bound[i] = -1;
    ncode = unroll(prog, call_names, code, names);
    if (ncode < 0) return fail(p, "a repeat too long to unroll", NULL);
    if (prog->has_emit) return fail(p, "emit(): several digests per input (stage 6)", NULL);

    for (i = 0; i < ncode; i++) {
        const hx_inst *in = &code[i];
        struct hx_plan_node nd;
        int k;
        memset(&nd, 0, sizeof nd);
        nd.a = nd.b = -1;
        switch (in->op) {
        case OP_STORE:
            /* h = value: later uses of h are that node (the plan is a DAG) */
            if (sp < 1 || st[sp - 1].is_int || in->u.slot < HX_SLOT_USERVARS || in->u.slot >= 64)
                return fail(p, "control flow or variables (opcode %s)", "3");
            bound[in->u.slot] = st[--sp].node;
            break;
        case OP_PUSH_VAR:
            if (in->u.slot >= HX_SLOT_USERVARS && in->u.slot < 64 && bound[in->u.slot] >= 0) {
                if (sp >= HXP_MAX_NODES) return fail(p, "too many values", NULL);
                st[sp].node = bound[in->u.slot]; st[sp].is_int = 0; sp++;
                break;
            }
            if (in->u.slot == HX_SLOT_SALT) used_salt = 1;
            else if (in->u.slot == HX_SLOT_USERID) used_user = 1;
            else if (in->u.slot == HX_SLOT_PEPPER) used_pepper = 1;
            else if (in->u.slot != HX_SLOT_PASS)
                return fail(p, "uses %s (stage 2 or later)",
                            prog->varnames && in->u.slot < prog->nvars
                                ? prog->varnames[in->u.slot] : "a variable");
            nd.kind = HXP_VAR;
            nd.var = (uint8_t)in->u.slot;
            nd.addressable = 1;
            nd.len = HXP_LEN_RUNTIME;
            nd.maxlen = in->u.slot == HX_SLOT_PASS ? pass_max : 65535;
            if ((k = add(p, nd)) < 0) return fail(p, "too many values", NULL);
            fx[k].ascii = 0;
            fx[k].utf8 = 1;   /* a password or name is taken as UTF-8 text */
            st[sp].node = k; st[sp].is_int = 0; sp++;
            break;
        case OP_PUSH_STR: {
            int j, asc = 1;
            nd.kind = HXP_CONST;
            nd.str = prog->strings[in->u.stridx];
            nd.len = prog->strlens[in->u.stridx];
            nd.addressable = 1;
            for (j = 0; j < nd.len; j++) if ((unsigned char)nd.str[j] >= 0x80) asc = 0;
            if ((k = add(p, nd)) < 0) return fail(p, "too many values", NULL);
            fx[k].ascii = (uint8_t)asc;
            fx[k].utf8 = (uint8_t)asc;   /* a non-ASCII literal is not modelled */
            st[sp].node = k; st[sp].is_int = 0; sp++;
            break;
        }
        case OP_PUSH_INT:
            st[sp].node = -1; st[sp].is_int = 1; st[sp].ival = in->u.ival; sp++;
            break;
        case OP_CONCAT: {
            struct slot b = st[--sp], a = st[--sp];
            if (a.is_int || b.is_int) return fail(p, "concatenates an integer", NULL);
            nd.kind = HXP_CAT;
            nd.a = a.node; nd.b = b.node;
            nd.addressable = p->node[a.node].addressable && p->node[b.node].addressable;
            size_node(p, &nd);
            if ((k = add(p, nd)) < 0) return fail(p, "too many values", NULL);
            fx[k].ascii = fx[a.node].ascii && fx[b.node].ascii;
            fx[k].utf8 = fx[a.node].utf8 && fx[b.node].utf8;
            st[sp].node = k; st[sp].is_int = 0; sp++;
            break;
        }
        case OP_CALL: {
            const char *name = call_names ? names[i]
                             : (in->u.call.entry ? in->u.call.entry->name : NULL);
            int nargs = in->u.call.nargs, role = in->u.call.role, bi, xo, j;
            struct slot args[8];
            if (!name) return fail(p, "a call has no name", NULL);
            if (nargs < 1 || nargs > 8 || nargs > sp)
                return fail(p, "%s: unexpected argument count", name);
            sp -= nargs;
            for (j = 0; j < nargs; j++) args[j] = st[sp + j];
            if ((bi = hmac_body_for(name)) >= 0) {
                /* hmac_X(message, key): both are values */
                if (nargs != 2 || args[0].is_int || args[1].is_int)
                    return fail(p, "%s: expects a message and a key", name);
                nd.kind = HXP_HMAC;
                nd.op = (uint8_t)bi;
                switch (role) {
                case ROLE_BIN: nd.view = HXV_BIN; break;
                case ROLE_UC:  nd.view = HXV_HEXUC; break;
                case ROLE_B64: nd.view = HXV_B64; break;
                case ROLE_MCF: return fail(p, "%s_mcf: MCF output (stage 7)", name);
                default:       nd.view = HXV_HEX; break;
                }
                nd.a = args[0].node;
                nd.b = args[1].node;
                nd.addressable = 1;
                size_node(p, &nd);
                if ((k = add(p, nd)) < 0) return fail(p, "too many values", NULL);
                fx[k].ascii = fx[k].utf8 = nd.view != HXV_BIN;
                st[sp].node = k; st[sp].is_int = 0; sp++;
                break;
            }
            bi = body_for(name);
            if (bi == HXB_TIGER128 || bi == HXB_TIGER160 || bi == HXB_TIGERMH)
                bi = -1;   /* HMAC only */
            if (bi < 0 && !xform_for(name, nargs, role, &xo))
                return fail(p, "no GPU body for %s", name);
            if (args[0].is_int) return fail(p, "%s: integer first argument", name);
            for (j = 1; j < nargs; j++)
                if (!args[j].is_int) return fail(p, "%s: argument is not an integer literal", name);
            if (bi >= 0) {
                if (nargs != 1) return fail(p, "%s: more than one argument", name);
                nd.kind = HXP_BODY;
                nd.op = (uint8_t)bi;
                switch (role) {
                case ROLE_BIN: nd.view = HXV_BIN; break;
                case ROLE_UC:  nd.view = HXV_HEXUC; break;
                case ROLE_B64: nd.view = HXV_B64; break;
                case ROLE_MCF: return fail(p, "%s_mcf: MCF output (stage 7)", name);
                default:       nd.view = HXV_HEX; break;
                }
                nd.a = args[0].node;
                nd.addressable = 1;
                size_node(p, &nd);
                if ((k = add(p, nd)) < 0) return fail(p, "too many values", NULL);
                fx[k].ascii = fx[k].utf8 = nd.view != HXV_BIN;
            } else {
                struct hx_plan_node *a = &p->node[args[0].node];
                if (xo == HXX_HEX && role == ROLE_BIN) {   /* hex_bin(x) is x */
                    st[sp].node = args[0].node; st[sp].is_int = 0; sp++;
                    break;
                }
                nd.kind = HXP_XFORM;
                nd.op = (uint8_t)xo;
                nd.a = args[0].node;
                nd.nia = nargs - 1;
                if (nargs >= 2) nd.ia = (int)args[1].ival;
                if (nargs >= 3) nd.ib = (int)args[2].ival;
                if (strcmp(name, "trunc") == 0) { nd.nia = 2; nd.ib = nd.ia; nd.ia = 0; }
                if ((xo == HXX_PAD && nargs != 2) || (xo == HXX_CUT && nargs > 3) ||
                    (xo != HXX_PAD && xo != HXX_CUT && xo != HXX_CAP_AT && nargs != 1))
                    return fail(p, "%s: unexpected argument count", name);
                switch (xo) {
                case HXX_UTF16LE: case HXX_UTF16BE:
                    /* An ASCII source widens byte by byte and stays addressable.
                     * pass is decoded from UTF-8 in order. A binary value is not
                     * UTF-8, and what the CPU does with it is not modelled here. */
                    if (!fx[args[0].node].utf8)
                        return fail(p, "%s of a value that may not be UTF-8", name);
                    nd.addressable = fx[args[0].node].ascii && a->addressable;
                    break;
                case HXX_CAP_FIRST:
                    nd.addressable = 0;          /* needs the first letter's position */
                    break;
                case HXX_REV:
                    if (!a->addressable)
                        return fail(p, "rev of a value that can only be streamed", NULL);
                    nd.addressable = 1;
                    break;
                case HXX_CUT:
                    if (!a->addressable && (nd.ia < 0 || nd.nia < 2))
                        return fail(p, "cut from the end of a value that can only be streamed", NULL);
                    nd.addressable = a->addressable;
                    break;
                default:
                    nd.addressable = a->addressable;
                    break;
                }
                size_node(p, &nd);
                if ((k = add(p, nd)) < 0) return fail(p, "too many values", NULL);
                fx[k].ascii = (xo == HXX_FROMHEX || xo == HXX_UTF16LE || xo == HXX_UTF16BE)
                              ? 0 : (xo == HXX_HEX || xo == HXX_BASE64) ? 1 : fx[args[0].node].ascii;
                /* the case maps and rot13 touch ASCII letters only, so UTF-8
                 * stays UTF-8; cut, pad and rev may split a character */
                fx[k].utf8 = fx[k].ascii ||
                             ((xo == HXX_UPPER || xo == HXX_LOWER || xo == HXX_ROT13 ||
                               xo == HXX_CAP_FIRST || xo == HXX_CAP_AT) && fx[args[0].node].utf8);
            }
            st[sp].node = k; st[sp].is_int = 0; sp++;
            break;
        }
        case OP_HALT:
            i = ncode;
            break;
        default:
            snprintf(num, sizeof num, "%d", in->op);
            return fail(p, "control flow or variables (opcode %s)", num);
        }
    }
    if (sp != 1 || st[0].is_int) return fail(p, "does not end in one value", NULL);
    if (used_salt && used_user) return fail(p, "uses both salt and user (stage 6)", NULL);
    /* A pepper rides the salt channel: the host packs salt . pepper . plen (2 bytes LE)
     * per salt entry, one entry per pepper (mdxfind type_pepper_mode). */
    if (used_pepper && !used_salt) return fail(p, "uses pepper without salt", NULL);
    p->channel = used_salt ? HXC_SALT : used_user ? HXC_USER : HXC_NONE;
    p->pepper = used_pepper;
    p->root = st[0].node;
    p->out_len = p->node[p->root].len;

    /* The final value is the stored form: [a constant prefix .] a view of
     * one digest, possibly cased, cut, or base64 of its binary. Whether that
     * matches the type's TypeDigest is the host's call. */
    {
        int r = p->root, nw = 0, wrap[HXP_MAX_WRAP];
        p->final_prefix = -1;
        p->final_cut = -1;
        p->final_b64 = 0;
        if (p->node[r].kind == HXP_CAT && p->node[p->node[r].a].kind == HXP_CONST) {
            p->final_prefix = p->node[r].a;
            r = p->node[r].b;
        }
        for (;;) {
            struct hx_plan_node *nd = &p->node[r];
            if (nd->kind != HXP_XFORM) break;
            if (nd->op == HXX_UPPER || nd->op == HXX_LOWER ||
                nd->op == HXX_CAP_FIRST || nd->op == HXX_CAP_AT) {
                if (nw < HXP_MAX_WRAP) wrap[nw] = r;
                nw++;
            } else if (nd->op == HXX_CUT && p->final_cut < 0) {
                p->final_cut = r;
            } else if (nd->op == HXX_BASE64 && !p->final_b64) {
                p->final_b64 = 1;
            } else {
                break;
            }
            r = nd->a;
        }
        if (p->node[r].kind != HXP_BODY && p->node[r].kind != HXP_HMAC)
            return fail(p, "the final value is not a view of one digest", NULL);
        if (p->final_b64 && p->node[r].view != HXV_BIN)
            return fail(p, "the final value is base64 of a text digest", NULL);
        p->final_body = r;
        /* -i iterates the outer hash with the case maps around it; a stored
         * form with a prefix, a cut or base64 is not something mdxfind
         * re-hashes as text. */
        /* an HMAC is not something -i re-hashes either: its CPU code does
         * not iterate (the host requires the ledger's no-iter for it) */
        if (p->final_prefix < 0 && p->final_cut < 0 && !p->final_b64 && nw <= HXP_MAX_WRAP &&
            p->node[r].kind == HXP_BODY) {
            p->iter_body = r;
            p->iter_nwrap = nw;
            memcpy(p->iter_wrap, wrap, sizeof wrap);
        }
    }
    if (p->out_len < 0) return fail(p, "the final value has no fixed length", NULL);
    return 1;
}

static void cat(char *out, int cap, const char *s)
{
    int l = (int)strlen(out);
    if (l < cap - 1) snprintf(out + l, cap - l, "%s", s);
}

void hx_plan_render(const struct hx_plan *p, int i, char *out, int cap)
{
    const struct hx_plan_node *nd = &p->node[i];
    static const char *vn[] = { "pass", "salt", "salt2", "pepper", "user" };
    static const char *views[] = { "_bin", "", "_uc", "_b64" };
    char buf[32];
    if (i == p->root) out[0] = 0;
    switch (nd->kind) {
    case HXP_VAR: cat(out, cap, nd->var < 5 ? vn[nd->var] : "?"); break;
    case HXP_CONST: cat(out, cap, "\""); cat(out, cap, nd->str); cat(out, cap, "\""); break;
    case HXP_CAT:
        hx_plan_render(p, nd->a, out, cap); cat(out, cap, " . ");
        hx_plan_render(p, nd->b, out, cap); break;
    case HXP_BODY:
        cat(out, cap, bodies[nd->op].name); cat(out, cap, views[nd->view]);
        cat(out, cap, "("); hx_plan_render(p, nd->a, out, cap); cat(out, cap, ")"); break;
    case HXP_HMAC:
        cat(out, cap, "hmac_"); cat(out, cap, bodies[nd->op].name); cat(out, cap, views[nd->view]);
        cat(out, cap, "("); hx_plan_render(p, nd->a, out, cap); cat(out, cap, ", ");
        hx_plan_render(p, nd->b, out, cap); cat(out, cap, ")"); break;
    default:
        cat(out, cap, xnames[nd->op]); cat(out, cap, "(");
        hx_plan_render(p, nd->a, out, cap);
        if (nd->nia >= 1) { snprintf(buf, sizeof buf, ", %d", nd->ia); cat(out, cap, buf); }
        if (nd->nia >= 2) { snprintf(buf, sizeof buf, ", %d", nd->ib); cat(out, cap, buf); }
        cat(out, cap, ")"); break;
    }
}
