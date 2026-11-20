/*
 * hx_gen.c -- emit a template core from an hx plan. See hx_gen.h.
 *
 * Every value is produced byte by byte, in order, into the absorber of the
 * hash that consumes it (a "sink"). A transform is a sink in front of another
 * sink: upper maps the byte, hex turns one byte into two, base64 groups three
 * into four, pad fills to its length afterwards, cut drops what is outside its
 * window. Nothing is materialized. rev, and a cut or cap counted from the end,
 * need the byte at an index, which gen_byte() computes by index arithmetic
 * over an addressable source.
 *
 * Digests are held in their native state words (md4/md5 little-endian, the
 * SHA family big-endian) and read out a byte at a time by shifts.
 *
 * $Revision: 1.6 $
 * $Log: hx_gen.c,v $
 * Revision 1.6  2026/11/20 15:45:12  dlr
 * Generic hx walker stages 3b and 4a. New walker bodies: MySQL 3 OLD_PASSWORD as a byte-wise accumulator in the dialect, BLAKE2b-512 and -256 and BLAKE2s-256 over the common b2b and b2s compress functions, and SHA3 and Keccak at 224, 256, 384 and 512 bits over keccakf1600; a final digest shorter than 16 bytes is compared as four words with the upper ones zero. User-defined types may now name mdxfind own hash functions: sha224, sha384, rmd128, rmd160, tiger, wrl, md2, gost, sne128, sne256, the fifteen HAVAL forms, mysql3, blake2b512, blake2b256, blake2s256, sha3_224 to sha3_512 and keccak224 to keccak512, each the call its catalog type makes. hashpipe registers its BLAKE2 digests as hx functions, which the verify registration had hidden. The drift check reads a NAME-prefixed hex digest and the bare digest as one value. 11 more catalog types run on the GPU: e410 e416 e449 e474 e478 e548 e711 e842 e843 e846 e847. Verified: each new user function agrees with its catalog type at x01 to x03 on the CPU and with hashpipe -X; the C mirror agrees for 567 of 567 cores and every earlier core changed only by the added dialect text; on dev3 (M2 Max, Metal) all new catalog types and 42 user types match -G none, and on gp1 (Tahiti, OpenCL) all do except the md2 user type, whose kernel the Tahiti compiler crashes on, which is FATAL there as for any such kernel.
 *
 * Revision 1.5  2026/11/20 05:22:13  dlr
 * Generic hx walker stage 1c: user-defined types run on the GPU through the generated walker. A user type is served when it takes no salt or user, declares no stored form, and its result is the hex of its final digest; rounds past x01 re-run the whole program on the previous round's hex text, keeping its case, as the userdef CPU arm does. The selected user type's status line says whether it runs on the GPU, or why not, in place of the former not-yet-enabled line. Verified: the C mirror agrees for 529 of 529 cores, including 10 user cores whose later rounds are checked against the hx VM re-running the program, and a planted case defect is caught in exactly the 8 case-sensitive ones; the 519 catalog cores are byte-identical; on dev3 (M2 Max, Metal) and gp1 (Tahiti, OpenCL) 10 user types match -G none at x01 and x02 and deeper with rules, a -n mask and a mixed run with a catalog type, except one type whose kernel the Tahiti compiler crashes on, which is FATAL there as for any such kernel.
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
#include <stdarg.h>
#include "hx_gen.h"

struct body_info {
    int state_words;    /* chaining words */
    int wide;           /* 1: 64-bit words */
    int be;             /* 1: big-endian words */
    int out_bytes;      /* digest bytes */
    const char *blk;    /* dialect block call */
    const char *iv;     /* initial state */
    int blk_bytes;      /* message block */
    int mw;             /* message words */
    const char *abs;    /* absorb one byte */
    const char *fin;    /* padding and length, then the last compress */
    const char *fin_arg;/* extra arguments to fin, or "" */
    const char *fold;   /* applied to the state after fin, or NULL */
    int cmirror;        /* the C dialect has this block function */
    int dwords;         /* words in the state array, 0 for 8 */
};

#define MD32LE  64, 16, "HX_ABS32LE", "HX_FIN32LE", "", NULL
#define MD32BE  64, 16, "HX_ABS32BE", "HX_FIN32BE", "", NULL
#define MD64BE 128, 16, "HX_ABS64BE", "HX_FIN64BE", "", NULL
#define HAVAL_IV_TXT "0x243f6a88u, 0x85a308d3u, 0x13198a2eu, 0x03707344u, " \
                     "0xa4093822u, 0x299f31d0u, 0x082efa98u, 0xec4e6c89u"
/* HAVAL: 128-byte block, pad 0x01, then bytes 118-119 = (passes << 3 | 1)
 * and (bits >> 2), then the bit length; widths below 256 fold the state. */
#define HAV(P, BYTES, PARAM, FOLD) \
    { 8, 0, 0, BYTES, "HXB_HAVAL" #P, HAVAL_IV_TXT, 128, 32, "HX_ABS32LE128", \
      "HX_FINHAVAL", ", " PARAM, FOLD, 0 }

static const struct body_info BI[HXB_COUNT] = {
    [HXB_MD4]    = { 4, 0, 0, 16, "HXB_MD4",
                     "0x67452301u, 0xefcdab89u, 0x98badcfeu, 0x10325476u", MD32LE, 1 },
    [HXB_MD5]    = { 4, 0, 0, 16, "HXB_MD5",
                     "0x67452301u, 0xefcdab89u, 0x98badcfeu, 0x10325476u", MD32LE, 1 },
    [HXB_SHA1]   = { 5, 0, 1, 20, "HXB_SHA1",
                     "0x67452301u, 0xefcdab89u, 0x98badcfeu, 0x10325476u, 0xc3d2e1f0u", MD32BE, 1 },
    [HXB_SHA224] = { 8, 0, 1, 28, "HXB_SHA256",
                     "0xc1059ed8u, 0x367cd507u, 0x3070dd17u, 0xf70e5939u, "
                     "0xffc00b31u, 0x68581511u, 0x64f98fa7u, 0xbefa4fa4u", MD32BE, 1 },
    [HXB_SHA256] = { 8, 0, 1, 32, "HXB_SHA256",
                     "0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au, "
                     "0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u", MD32BE, 1 },
    [HXB_SHA384] = { 8, 1, 1, 48, "HXB_SHA512",
                     "0xcbbb9d5dc1059ed8ul, 0x629a292a367cd507ul, 0x9159015a3070dd17ul, "
                     "0x152fecd8f70e5939ul, 0x67332667ffc00b31ul, 0x8eb44a8768581511ul, "
                     "0xdb0c2e0d64f98fa7ul, 0x47b5481dbefa4fa4ul", MD64BE, 1 },
    [HXB_SHA512] = { 8, 1, 1, 64, "HXB_SHA512",
                     "0x6a09e667f3bcc908ul, 0xbb67ae8584caa73bul, 0x3c6ef372fe94f82bul, "
                     "0xa54ff53a5f1d36f1ul, 0x510e527fade682d1ul, 0x9b05688c2b3e6c1ful, "
                     "0x1f83d9abfb41bd6bul, 0x5be0cd19137e2179ul", MD64BE, 1 },
    /* stage 1b */
    [HXB_RMD128] = { 4, 0, 0, 16, "HXB_RMD128",
                     "0x67452301u, 0xefcdab89u, 0x98badcfeu, 0x10325476u", MD32LE, 0 },
    [HXB_RMD160] = { 5, 0, 0, 20, "HXB_RMD160",
                     "0x67452301u, 0xefcdab89u, 0x98badcfeu, 0x10325476u, 0xc3d2e1f0u", MD32LE, 0 },
    /* Tiger: little-endian 64-bit words, 64-byte block, pad 0x01 (not
     * Tiger2's 0x80), 64-bit length in M[7] */
    [HXB_TIGER]  = { 3, 1, 0, 24, "HXB_TIGER",
                     "0x0123456789abcdeful, 0xfedcba9876543210ul, 0xf096a5b4c3b2e187ul",
                     64, 8, "HX_ABS64LE", "HX_FIN64LE01", "", NULL, 0 },
    /* Whirlpool: big-endian 64-bit words, 64-byte block, 256-bit length
     * field (the low 64 bits in M[7]), IV zero */
    [HXB_WRL]    = { 8, 1, 1, 64, "HXB_WRL", "0ul, 0ul, 0ul, 0ul, 0ul, 0ul, 0ul, 0ul",
                     64, 8, "HX_ABS64BE64", "HX_FIN64BE256", "", NULL, 0 },
    [HXB_HAV128_3] = HAV(3, 16, "0x2019u", "HX_HAVFOLD16"),
    [HXB_HAV128_4] = HAV(4, 16, "0x2021u", "HX_HAVFOLD16"),
    [HXB_HAV128_5] = HAV(5, 16, "0x2029u", "HX_HAVFOLD16"),
    [HXB_HAV160_3] = HAV(3, 20, "0x2819u", "HX_HAVFOLD20"),
    [HXB_HAV160_4] = HAV(4, 20, "0x2821u", "HX_HAVFOLD20"),
    [HXB_HAV160_5] = HAV(5, 20, "0x2829u", "HX_HAVFOLD20"),
    [HXB_HAV192_3] = HAV(3, 24, "0x3019u", "HX_HAVFOLD24"),
    [HXB_HAV192_4] = HAV(4, 24, "0x3021u", "HX_HAVFOLD24"),
    [HXB_HAV192_5] = HAV(5, 24, "0x3029u", "HX_HAVFOLD24"),
    [HXB_HAV224_3] = HAV(3, 28, "0x3819u", "HX_HAVFOLD28"),
    [HXB_HAV224_4] = HAV(4, 28, "0x3821u", "HX_HAVFOLD28"),
    [HXB_HAV224_5] = HAV(5, 28, "0x3829u", "HX_HAVFOLD28"),
    [HXB_HAV256_3] = HAV(3, 32, "0x4019u", NULL),
    [HXB_HAV256_4] = HAV(4, 32, "0x4021u", NULL),
    [HXB_HAV256_5] = HAV(5, 32, "0x4029u", NULL),
    /* MD2: the state array holds the 48-byte state then the 16-byte
     * checksum, as bytes; 16-byte blocks; IV zero. */
    [HXB_MD2]    = { 16, 0, 0, 16, "HXB_MD2", "0u",
                     16, 4, "HX_ABS32LE16", "HX_FINMD2", "", NULL, 0, 16 },
    /* GOST R 34.11-94: 32-byte blocks of little-endian words; m[8..15]
     * carries the 256-bit sum of the blocks. IV zero. */
    [HXB_GOST]   = { 8, 0, 0, 32, "HXB_GOST", "0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u",
                     32, 16, "HX_ABSGOST", "HX_FINGOST", "", NULL, 0, 0 },
    /* Snefru: the message is held as bytes in order (little-endian words),
     * 48-byte blocks for 128 and 32-byte for 256; big-endian state, IV zero. */
    [HXB_SNE128] = { 8, 0, 1, 16, "HXB_SNE128", "0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u",
                     48, 12, "HX_ABS32LE48", "HX_FINSNE", ", 48u", NULL, 0, 0 },
    [HXB_SNE256] = { 8, 0, 1, 32, "HXB_SNE256", "0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u",
                     32, 8, "HX_ABS32LE32", "HX_FINSNE", ", 32u", NULL, 0, 0 },
    /* stage 2, HMAC only: mhash's Tiger -- the same compression and the
     * same little-endian message, but the digest read out of each state
     * word most-significant byte first (be = 1); whole, and its first 16
     * and 20 bytes */
    [HXB_TIGER128] = { 3, 1, 1, 16, "HXB_TIGER",
                     "0x0123456789abcdeful, 0xfedcba9876543210ul, 0xf096a5b4c3b2e187ul",
                     64, 8, "HX_ABS64LE", "HX_FIN64LE01", "", NULL, 0 },
    [HXB_TIGER160] = { 3, 1, 1, 20, "HXB_TIGER",
                     "0x0123456789abcdeful, 0xfedcba9876543210ul, 0xf096a5b4c3b2e187ul",
                     64, 8, "HX_ABS64LE", "HX_FIN64LE01", "", NULL, 0 },
    [HXB_TIGERMH]  = { 3, 1, 1, 24, "HXB_TIGER",
                     "0x0123456789abcdeful, 0xfedcba9876543210ul, 0xf096a5b4c3b2e187ul",
                     64, 8, "HX_ABS64LE", "HX_FIN64LE01", "", NULL, 0 },
    /* stage 3b: MySQL 3 (mdxfind mysql3()): S = nr, nr2, add, updated by
     * each byte but space and tab; no message block. The digest is nr then
     * nr2, each masked to 31 bits, big-endian -- the "%08x%08x" text. */
    [HXB_MYSQL3]   = { 3, 0, 1, 8, "HXB_NONE", "1345345333u, 0x12345671u, 7u",
                     64, 1, "HX_ABSMYSQL3", "HX_FINMYSQL3", "", NULL, 0, 0 },
    /* BLAKE2b (RFC 7693), unkeyed: h0 carries the parameter block (digest
     * length, fanout 1, depth 1). The last block is compressed only at
     * fin, with the final flag, so a full block waits for the next byte. */
    [HXB_B2B512] = { 8, 1, 0, 64, "HXB_NONE",
                     "0x6a09e667f2bdc948ul, 0xbb67ae8584caa73bul, 0x3c6ef372fe94f82bul, "
                     "0xa54ff53a5f1d36f1ul, 0x510e527fade682d1ul, 0x9b05688c2b3e6c1ful, "
                     "0x1f83d9abfb41bd6bul, 0x5be0cd19137e2179ul",
                     128, 16, "HX_ABSB2B", "HX_FINB2B", "", NULL, 0, 0 },
    [HXB_B2B256] = { 8, 1, 0, 32, "HXB_NONE",
                     "0x6a09e667f2bdc928ul, 0xbb67ae8584caa73bul, 0x3c6ef372fe94f82bul, "
                     "0xa54ff53a5f1d36f1ul, 0x510e527fade682d1ul, 0x9b05688c2b3e6c1ful, "
                     "0x1f83d9abfb41bd6bul, 0x5be0cd19137e2179ul",
                     128, 16, "HX_ABSB2B", "HX_FINB2B", "", NULL, 0, 0 },
    /* Keccak sponge, 25-word state, rate R bytes (R/8 message words);
     * SHA-3 pads with 0x06, Keccak with 0x01, both end the rate with 0x80. */
#define KEC(R, OUT, PAD) { 25, 1, 0, OUT, "HXB_NONE", "0ul", R, (R) / 8, \
                           "HX_ABSKEC" #R, "HX_FINKEC", ", " PAD ", " #R "u", NULL, 0, 25 }
    [HXB_SHA3_256] = KEC(136, 32, "0x06u"),
    [HXB_SHA3_224] = KEC(144, 28, "0x06u"),
    [HXB_SHA3_384] = KEC(104, 48, "0x06u"), [HXB_SHA3_512] = KEC(72, 64, "0x06u"),
    [HXB_KEC224]   = KEC(144, 28, "0x01u"), [HXB_KEC256]   = KEC(136, 32, "0x01u"),
    [HXB_KEC384]   = KEC(104, 48, "0x01u"), [HXB_KEC512]   = KEC(72, 64, "0x01u"),
    [HXB_B2S256] = { 8, 0, 0, 32, "HXB_NONE",
                     "0x6b08e647u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au, "
                     "0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u",
                     64, 16, "HX_ABSB2S", "HX_FINB2S", "", NULL, 0, 0 },
#undef KEC
};

/* The HMAC block of each body: what mhash_get_hash_pblock gives, with its
 * 0 (GOST) read as 64 the way mhash_hmac_init reads it. Checked against
 * mhash 0.9.9.9, which the CPU code and hashpipe use for these. */
static int hmac_block(int op)
{
    switch (op) {
    case HXB_MD2: return 16;
    case HXB_SHA384: case HXB_SHA512:
    case HXB_HAV128_3: case HXB_HAV128_4: case HXB_HAV128_5:
    case HXB_HAV160_3: case HXB_HAV160_4: case HXB_HAV160_5:
    case HXB_HAV192_3: case HXB_HAV192_4: case HXB_HAV192_5:
    case HXB_HAV224_3: case HXB_HAV224_4: case HXB_HAV224_5:
    case HXB_HAV256_3: case HXB_HAV256_4: case HXB_HAV256_5:
        return 128;
    default: return 64;
    }
}

struct gen {
    char *b;
    size_t n, cap;
    int tmp;
    int depth;
    const struct hx_plan *p;
    int from_st;        /* final_body's digest is read from ST (in iterate) */
    char why[160];
};

static void out(struct gen *g, const char *fmt, ...)
{
    va_list ap;
    int k;
    for (;;) {
        va_start(ap, fmt);
        k = vsnprintf(g->b + g->n, g->cap - g->n, fmt, ap);
        va_end(ap);
        if (k < 0) { fprintf(stderr, "FATAL %s:%d: vsnprintf\n", __FILE__, __LINE__); exit(1); }
        if (g->n + (size_t)k < g->cap) { g->n += k; return; }
        g->cap = g->cap * 2 + k + 256;
        g->b = realloc(g->b, g->cap);
        if (!g->b) { fprintf(stderr, "FATAL %s:%d: out of memory\n", __FILE__, __LINE__); exit(1); }
    }
}

static void ind(struct gen *g) { int i; for (i = 0; i < g->depth; i++) out(g, "    "); }

static int refuse(struct gen *g, const char *msg)
{
    if (!g->why[0]) snprintf(g->why, sizeof g->why, "%s", msg);
    return -1;
}

static const struct hx_plan_node *N(struct gen *g, int i) { return &g->p->node[i]; }

/* ---- digests ---- */

/* Expression for byte idx of the binary digest of a body op held in state
 * array st (a variable name). */
static void state_byte(int op, const char *st, const char *idx, char *e, int cap)
{
    const struct body_info *bi = &BI[op];
    if (bi->wide && !bi->be)
        snprintf(e, cap, "((uint)(%s[(%s) >> 3] >> (((%s) & 7) * 8)) & 0xffu)", st, idx, idx);
    else if (bi->wide)
        snprintf(e, cap, "((uint)(%s[(%s) >> 3] >> ((7 - ((%s) & 7)) * 8)) & 0xffu)", st, idx, idx);
    else if (bi->be)
        snprintf(e, cap, "((%s[(%s) >> 2] >> ((3 - ((%s) & 3)) * 8)) & 0xffu)", st, idx, idx);
    else
        snprintf(e, cap, "((%s[(%s) >> 2] >> (((%s) & 3) * 8)) & 0xffu)", st, idx, idx);
}

/* Expression for byte idx (a variable name) of body k's binary digest. */
static void digest_byte(struct gen *g, int k, const char *idx, char *e, int cap)
{
    char st[16];
    if (g->from_st && k == g->p->final_body) {
        snprintf(e, cap, "((ST((%s) >> 2) >> (((%s) & 3) * 8)) & 0xffu)", idx, idx);
        return;
    }
    snprintf(st, sizeof st, "d%d", k);
    state_byte(N(g, k)->op, st, idx, e, cap);
}

/* ---- lengths, for addressable values ---- */

static void gen_len(struct gen *g, int i, char *e, int cap)
{
    const struct hx_plan_node *nd = N(g, i);
    char a[512], b[512];
    if (nd->len >= 0) { snprintf(e, cap, "%d", nd->len); return; }
    switch (nd->kind) {
    case HXP_VAR:
        snprintf(e, cap, nd->var == HX_SLOT_PASS ? "len" : nd->var == HX_SLOT_PEPPER ? "plen" : "slen");
        return;
    case HXP_CAT:
        gen_len(g, nd->a, a, sizeof a); gen_len(g, nd->b, b, sizeof b);
        snprintf(e, cap, "(%s + %s)", a, b); return;
    default: break;
    }
    gen_len(g, nd->a, a, sizeof a);
    switch (nd->op) {
    case HXX_HEX: case HXX_UTF16LE: case HXX_UTF16BE: snprintf(e, cap, "(2 * %s)", a); return;
    case HXX_BASE64: snprintf(e, cap, "(((%s) + 2) / 3 * 4)", a); return;
    case HXX_FROMHEX: snprintf(e, cap, "((%s) / 2)", a); return;
    case HXX_PAD: snprintf(e, cap, "%d", nd->ia); return;
    case HXX_CUT:
        snprintf(e, cap, "HX_CUT_LEN(%s, %d, %d, %d)", a, nd->ia, nd->nia >= 2 ? nd->ib : -1,
                 nd->nia >= 2);
        return;
    default: snprintf(e, cap, "%s", a); return;
    }
}

/* ---- the byte at an index ---- */

/* Emit statements leaving byte idx of node i in a new uint variable; return
 * its number (vN). idx must be a plain variable name. */
static int gen_byte(struct gen *g, int i, const char *idx)
{
    const struct hx_plan_node *nd = N(g, i);
    int v = g->tmp++, c, t;
    char e[512], L[512], ix[32];
    switch (nd->kind) {
    case HXP_VAR:
        if (nd->var == HX_SLOT_PEPPER) {
            ind(g); out(g, "uint v%d = (uint)salt[slen + %s];\n", v, idx);
        } else {
            ind(g); out(g, "uint v%d = (uint)%s[%s];\n", v,
                        nd->var == HX_SLOT_PASS ? "data" : "salt", idx);
        }
        return v;
    case HXP_CONST: {
        int j;
        /* an initialised array, not a string literal: Metal has no
         * indexable string literals */
        ind(g); out(g, "const uchar k%d[%d] = {", v, nd->len > 0 ? nd->len : 1);
        for (j = 0; j < nd->len; j++) out(g, "%s0x%02x", j ? ", " : " ", (unsigned char)nd->str[j]);
        out(g, "%s };\n", nd->len > 0 ? "" : " 0");
        ind(g); out(g, "uint v%d = (uint)k%d[%s];\n", v, v, idx);
        return v;
    }
    case HXP_BODY: case HXP_HMAC:
        switch (nd->view) {
        case HXV_BIN:
            digest_byte(g, i, idx, e, sizeof e);
            ind(g); out(g, "uint v%d = %s;\n", v, e);
            return v;
        case HXV_HEX: case HXV_HEXUC:
            t = g->tmp++;
            snprintf(ix, sizeof ix, "i%d", t);
            ind(g); out(g, "int i%d = (%s) >> 1;\n", t, idx);
            digest_byte(g, i, ix, e, sizeof e);
            ind(g); out(g, "uint v%d = %s(((%s) >> (((%s) & 1) ? 0 : 4)) & 15u);\n", v,
                        nd->view == HXV_HEX ? "HX_HEXC" : "HX_HEXUC", e, idx);
            return v;
        default:
            return refuse(g, "base64 of a digest read by index");
        }
    case HXP_CAT:
        gen_len(g, nd->a, L, sizeof L);
        t = g->tmp++;
        ind(g); out(g, "uint v%d;\n", v);
        ind(g); out(g, "if ((%s) < (%s)) {\n", idx, L);
        g->depth++;
        if ((c = gen_byte(g, nd->a, idx)) < 0) return -1;
        ind(g); out(g, "v%d = v%d;\n", v, c);
        g->depth--;
        ind(g); out(g, "} else {\n");
        g->depth++;
        ind(g); out(g, "int i%d = (%s) - (%s);\n", t, idx, L);
        snprintf(ix, sizeof ix, "i%d", t);
        if ((c = gen_byte(g, nd->b, ix)) < 0) return -1;
        ind(g); out(g, "v%d = v%d;\n", v, c);
        g->depth--;
        ind(g); out(g, "}\n");
        return v;
    default:
        break;
    }
    /* transforms */
    t = g->tmp++;
    snprintf(ix, sizeof ix, "i%d", t);
    gen_len(g, nd->a, L, sizeof L);
    switch (nd->op) {
    case HXX_UPPER: case HXX_LOWER: case HXX_ROT13:
        if ((c = gen_byte(g, nd->a, idx)) < 0) return -1;
        ind(g); out(g, "uint v%d = %s(v%d);\n", v,
                    nd->op == HXX_UPPER ? "HX_UP" : nd->op == HXX_LOWER ? "HX_LO" : "HX_ROT13", c);
        return v;
    case HXX_CAP_AT:
        if ((c = gen_byte(g, nd->a, idx)) < 0) return -1;
        ind(g); out(g, "uint v%d = ((%s) == HX_CAP_POS(%s, %d)) ? HX_UP(v%d) : v%d;\n",
                    v, idx, L, nd->ia, c, c);
        return v;
    case HXX_CUT:
        ind(g); out(g, "int i%d = (%s) + HX_CUT_START(%s, %d);\n", t, idx, L, nd->ia);
        if ((c = gen_byte(g, nd->a, ix)) < 0) return -1;
        ind(g); out(g, "uint v%d = v%d;\n", v, c);
        return v;
    case HXX_REV:
        ind(g); out(g, "int i%d = (%s) - 1 - (%s);\n", t, L, idx);
        if ((c = gen_byte(g, nd->a, ix)) < 0) return -1;
        ind(g); out(g, "uint v%d = v%d;\n", v, c);
        return v;
    case HXX_PAD:
        ind(g); out(g, "uint v%d = 0u;\n", v);
        ind(g); out(g, "if ((%s) < (%s)) {\n", idx, L);
        g->depth++;
        if ((c = gen_byte(g, nd->a, idx)) < 0) return -1;
        ind(g); out(g, "v%d = v%d;\n", v, c);
        g->depth--;
        ind(g); out(g, "}\n");
        return v;
    case HXX_HEX:
        ind(g); out(g, "int i%d = (%s) >> 1;\n", t, idx);
        if ((c = gen_byte(g, nd->a, ix)) < 0) return -1;
        ind(g); out(g, "uint v%d = HX_HEXC((v%d >> (((%s) & 1) ? 0 : 4)) & 15u);\n", v, c, idx);
        return v;
    case HXX_UTF16LE: case HXX_UTF16BE:
        ind(g); out(g, "int i%d = (%s) >> 1;\n", t, idx);
        if ((c = gen_byte(g, nd->a, ix)) < 0) return -1;
        ind(g); out(g, "uint v%d = (((%s) & 1) == %d) ? v%d : 0u;\n", v, idx,
                    nd->op == HXX_UTF16LE ? 0 : 1, c);
        return v;
    case HXX_FROMHEX: {
        int c2, t2 = g->tmp++;
        ind(g); out(g, "int i%d = (%s) * 2;\n", t, idx);
        if ((c = gen_byte(g, nd->a, ix)) < 0) return -1;
        ind(g); out(g, "int i%d = (%s) * 2 + 1;\n", t2, idx);
        snprintf(ix, sizeof ix, "i%d", t2);
        if ((c2 = gen_byte(g, nd->a, ix)) < 0) return -1;
        ind(g); out(g, "uint v%d = (HX_HEXV(v%d) << 4) | HX_HEXV(v%d);\n", v, c, c2);
        return v;
    }
    case HXX_BASE64: {
        /* group idx/4 of three source bytes; '=' where they run out */
        int a, b2, c3, tg = g->tmp++, t1 = g->tmp++, t2 = g->tmp++;
        ind(g); out(g, "int i%d = ((%s) >> 2) * 3;\n", tg, idx);
        snprintf(ix, sizeof ix, "i%d", tg);
        if ((a = gen_byte(g, nd->a, ix)) < 0) return -1;
        ind(g); out(g, "uint v%d = 0u, v%d = 0u;\n", t1, t2);
        ind(g); out(g, "int i%d = i%d + 1;\n", t, tg);
        ind(g); out(g, "if (i%d < (%s)) {\n", t, L);
        g->depth++;
        snprintf(ix, sizeof ix, "i%d", t);
        if ((b2 = gen_byte(g, nd->a, ix)) < 0) return -1;
        ind(g); out(g, "v%d = v%d;\n", t1, b2);
        g->depth--;
        ind(g); out(g, "}\n");
        ind(g); out(g, "if (i%d + 1 < (%s)) {\n", t, L);
        g->depth++;
        ind(g); out(g, "int i%d_2 = i%d + 1;\n", t, t);
        snprintf(ix, sizeof ix, "i%d_2", t);
        if ((c3 = gen_byte(g, nd->a, ix)) < 0) return -1;
        ind(g); out(g, "v%d = v%d;\n", t2, c3);
        g->depth--;
        ind(g); out(g, "}\n");
        ind(g); out(g, "uint v%d = HX_B64CHAR(v%d, v%d, v%d, (%s) & 3, i%d < (%s), i%d + 1 < (%s));\n",
                    v, a, t1, t2, idx, t, L, t, L);
        return v;
    }
    default:
        return refuse(g, "this transform has no index form");
    }
}

/* ---- sinks: where a streamed byte goes ---- */

enum sink_kind { SK_ABSORB, SK_MAP, SK_HEX, SK_UTF16A, SK_CAPFIRST, SK_CAPAT, SK_CUT,
                 SK_PAD, SK_B64, SK_FROMHEX, SK_U8VALID, SK_U8DECODE, SK_HMACKEY };

struct sink {
    int kind, id, op, be, n;
    const char *map;
    char tag[16];           /* SK_ABSORB: the state's suffix (d<tag>, m<tag>, ...) */
    struct sink *next;
};

static void put(struct gen *g, struct sink *s, const char *x);

static void putv(struct gen *g, struct sink *s, int v)
{
    char e[32];
    snprintf(e, sizeof e, "v%d", v);
    put(g, s, e);
}

static void put(struct gen *g, struct sink *s, const char *x)
{
    int v = g->tmp++;
    char e[64];
    switch (s->kind) {
    case SK_ABSORB: {
        const struct body_info *bi = &BI[s->op];
        ind(g); out(g, "%s(d%s, m%s, b%s, t%s, %s, %s);\n",
                    bi->abs, s->tag, s->tag, s->tag, s->tag, x, bi->blk);
        return;
    }
    case SK_HMACKEY: {
        /* the key: into the block buffer while it fits, and into a hash of
         * it in case it does not (mhash: a key longer than the block is
         * replaced by its digest) */
        const struct body_info *bi = &BI[s->op];
        ind(g); out(g, "{ uint v%d = %s;\n", v, x);
        g->depth++;
        ind(g); out(g, "if (kl%d < %d) kp%d[kl%d] = (uchar)v%d;\n", s->id, s->n, s->id, s->id, v);
        ind(g); out(g, "kl%d++;\n", s->id);
        ind(g); out(g, "%s(d%dk, m%dk, b%dk, t%dk, v%d, %s);\n",
                    bi->abs, s->id, s->id, s->id, s->id, v, bi->blk);
        g->depth--;
        ind(g); out(g, "}\n");
        return;
    }
    case SK_MAP:
        snprintf(e, sizeof e, "%s(%s)", s->map, x);
        if (strlen(x) > 40) {   /* keep the text flat */
            ind(g); out(g, "uint v%d = %s;\n", v, x);
            snprintf(e, sizeof e, "%s(v%d)", s->map, v);
        }
        put(g, s->next, e);
        return;
    case SK_HEX:
        ind(g); out(g, "{ uint v%d = %s;\n", v, x);
        g->depth++;
        snprintf(e, sizeof e, "HX_%s(v%d >> 4)", s->n ? "HEXUC" : "HEXC", v);
        put(g, s->next, e);
        snprintf(e, sizeof e, "HX_%s(v%d & 15u)", s->n ? "HEXUC" : "HEXC", v);
        put(g, s->next, e);
        g->depth--;
        ind(g); out(g, "}\n");
        return;
    case SK_UTF16A:
        ind(g); out(g, "{ uint v%d = %s;\n", v, x);
        g->depth++;
        if (s->be) { put(g, s->next, "0u"); putv(g, s->next, v); }
        else       { putv(g, s->next, v); put(g, s->next, "0u"); }
        g->depth--;
        ind(g); out(g, "}\n");
        return;
    case SK_CAPFIRST:
        ind(g); out(g, "{ uint v%d = %s;\n", v, x);
        g->depth++;
        ind(g); out(g, "if (!c%d && v%d >= 97u && v%d <= 122u) { v%d -= 32u; c%d = 1; }\n",
                    s->id, v, v, v, s->id);
        putv(g, s->next, v);
        g->depth--;
        ind(g); out(g, "}\n");
        return;
    case SK_CAPAT:
        ind(g); out(g, "{ uint v%d = %s;\n", v, x);
        g->depth++;
        ind(g); out(g, "if (n%d == p%d) v%d = HX_UP(v%d);\n", s->id, s->id, v, v);
        ind(g); out(g, "n%d++;\n", s->id);
        putv(g, s->next, v);
        g->depth--;
        ind(g); out(g, "}\n");
        return;
    case SK_CUT:
        ind(g); out(g, "{ uint v%d = %s;\n", v, x);
        g->depth++;
        ind(g); out(g, "if (n%d >= cs%d && n%d - cs%d < cl%d) {\n", s->id, s->id, s->id, s->id, s->id);
        g->depth++;
        putv(g, s->next, v);
        g->depth--;
        ind(g); out(g, "}\n");
        ind(g); out(g, "n%d++;\n", s->id);
        g->depth--;
        ind(g); out(g, "}\n");
        return;
    case SK_PAD:
        ind(g); out(g, "{ uint v%d = %s;\n", v, x);
        g->depth++;
        ind(g); out(g, "if (n%d < %d) {\n", s->id, s->n);
        g->depth++;
        putv(g, s->next, v);
        g->depth--;
        ind(g); out(g, "}\n");
        ind(g); out(g, "n%d++;\n", s->id);
        g->depth--;
        ind(g); out(g, "}\n");
        return;
    case SK_B64:
        ind(g); out(g, "a%d = (a%d << 8) | (%s);\n", s->id, s->id, x);
        ind(g); out(g, "if (++q%d == 3) {\n", s->id);
        g->depth++;
        snprintf(e, sizeof e, "HX_B64C(a%d >> 18)", s->id); put(g, s->next, e);
        snprintf(e, sizeof e, "HX_B64C(a%d >> 12)", s->id); put(g, s->next, e);
        snprintf(e, sizeof e, "HX_B64C(a%d >> 6)", s->id);  put(g, s->next, e);
        snprintf(e, sizeof e, "HX_B64C(a%d)", s->id);       put(g, s->next, e);
        ind(g); out(g, "a%d = 0u; q%d = 0;\n", s->id, s->id);
        g->depth--;
        ind(g); out(g, "}\n");
        return;
    case SK_FROMHEX:
        ind(g); out(g, "{ uint v%d = HX_HEXV(%s);\n", v, x);
        g->depth++;
        ind(g); out(g, "if (q%d) {\n", s->id);
        g->depth++;
        snprintf(e, sizeof e, "((a%d << 4) | v%d)", s->id, v);
        put(g, s->next, e);
        ind(g); out(g, "q%d = 0;\n", s->id);
        g->depth--;
        ind(g); out(g, "} else { a%d = v%d; q%d = 1; }\n", s->id, v, s->id);
        g->depth--;
        ind(g); out(g, "}\n");
        return;
    case SK_U8VALID:
        ind(g); out(g, "HX_U8_STEP(%s, u%d, r%d, ok%d);\n", x, s->id, s->id, s->id);
        return;
    case SK_U8DECODE:
        ind(g); out(g, "HX_U8_STEP(%s, u%d, r%d, ok%d);\n", x, s->id, s->id, s->id);
        ind(g); out(g, "if (r%d == 0) {\n", s->id);
        g->depth++;
        ind(g); out(g, "if (u%d >= 0x10000u) {\n", s->id);
        g->depth++;
        ind(g); out(g, "uint w%d = u%d - 0x10000u;\n", v, s->id);
        snprintf(e, sizeof e, "(0xd800u | (w%d >> 10))", v);
        if (s->be) {
            char e2[64];
            snprintf(e2, sizeof e2, "(%s >> 8)", e);  put(g, s->next, e2);
            snprintf(e2, sizeof e2, "(%s & 0xffu)", e); put(g, s->next, e2);
            snprintf(e, sizeof e, "(0xdc00u | (w%d & 0x3ffu))", v);
            snprintf(e2, sizeof e2, "(%s >> 8)", e);  put(g, s->next, e2);
            snprintf(e2, sizeof e2, "(%s & 0xffu)", e); put(g, s->next, e2);
        } else {
            char e2[64];
            snprintf(e2, sizeof e2, "(%s & 0xffu)", e); put(g, s->next, e2);
            snprintf(e2, sizeof e2, "(%s >> 8)", e);  put(g, s->next, e2);
            snprintf(e, sizeof e, "(0xdc00u | (w%d & 0x3ffu))", v);
            snprintf(e2, sizeof e2, "(%s & 0xffu)", e); put(g, s->next, e2);
            snprintf(e2, sizeof e2, "(%s >> 8)", e);  put(g, s->next, e2);
        }
        g->depth--;
        ind(g); out(g, "} else {\n");
        g->depth++;
        snprintf(e, sizeof e, "(u%d & 0xffu)", s->id);
        if (s->be) {
            char e2[64];
            snprintf(e2, sizeof e2, "(u%d >> 8)", s->id); put(g, s->next, e2);
            put(g, s->next, e);
        } else {
            char e2[64];
            put(g, s->next, e);
            snprintf(e2, sizeof e2, "(u%d >> 8)", s->id); put(g, s->next, e2);
        }
        g->depth--;
        ind(g); out(g, "}\n");
        ind(g); out(g, "u%d = 0u;\n", s->id);
        g->depth--;
        ind(g); out(g, "}\n");
        return;
    }
}

/* ---- streaming a value into a sink ---- */

static int gen_stream(struct gen *g, int i, struct sink *s);

/* Stream an addressable value by index. */
static int stream_indexed(struct gen *g, int i, struct sink *s)
{
    char L[512];
    int t = g->tmp++, c;
    char ix[32];
    gen_len(g, i, L, sizeof L);
    snprintf(ix, sizeof ix, "i%d", t);
    ind(g); out(g, "for (int i%d = 0; i%d < (%s); i%d++) {\n", t, t, L, t);
    g->depth++;
    if ((c = gen_byte(g, i, ix)) < 0) return -1;
    putv(g, s, c);
    g->depth--;
    ind(g); out(g, "}\n");
    return 0;
}

static int gen_stream(struct gen *g, int i, struct sink *s)
{
    const struct hx_plan_node *nd = N(g, i);
    struct sink k;
    int id;
    char L[512];
    memset(&k, 0, sizeof k);
    k.next = s;
    switch (nd->kind) {
    case HXP_VAR: {
        int t = g->tmp++;
        char e[32];
        {
            int pw = nd->var == HX_SLOT_PASS, pp = nd->var == HX_SLOT_PEPPER;
            ind(g); out(g, "for (int i%d = 0; i%d < %s; i%d++) {\n", t, t,
                        pw ? "len" : pp ? "plen" : "slen", t);
            g->depth++;
            snprintf(e, sizeof e, "(uint)salt[%si%d]", pp ? "slen + " : "", t);
            if (pw) snprintf(e, sizeof e, "(uint)data[i%d]", t);
        }
        put(g, s, e);
        g->depth--;
        ind(g); out(g, "}\n");
        return 0;
    }
    case HXP_CONST: {
        int j;
        char e[16];
        for (j = 0; j < nd->len; j++) {
            snprintf(e, sizeof e, "0x%02xu", (unsigned char)nd->str[j]);
            put(g, s, e);
        }
        return 0;
    }
    case HXP_BODY: case HXP_HMAC:
        if (nd->view == HXV_B64) {
            /* base64 of the binary digest: group the digest bytes */
            int t = g->tmp++, j;
            char e[64];
            k.kind = SK_B64; k.id = t;
            ind(g); out(g, "uint a%d = 0u; int q%d = 0;\n", t, t);
            for (j = 0; j < BI[nd->op].out_bytes; j++) {
                char ix[16];
                snprintf(ix, sizeof ix, "%d", j);
                digest_byte(g, i, ix, e, sizeof e);
                put(g, &k, e);
            }
            goto b64_tail;
        }
        return stream_indexed(g, i, s);
    case HXP_CAT:
        if (gen_stream(g, nd->a, s) < 0) return -1;
        return gen_stream(g, nd->b, s);
    default:
        break;
    }
    id = g->tmp++;
    k.id = id;
    switch (nd->op) {
    case HXX_UPPER: case HXX_LOWER: case HXX_ROT13:
        k.kind = SK_MAP;
        k.map = nd->op == HXX_UPPER ? "HX_UP" : nd->op == HXX_LOWER ? "HX_LO" : "HX_ROT13";
        return gen_stream(g, nd->a, &k);
    case HXX_HEX:
        k.kind = SK_HEX;
        return gen_stream(g, nd->a, &k);
    case HXX_UTF16LE: case HXX_UTF16BE:
        k.be = nd->op == HXX_UTF16BE;
        if (nd->addressable) {      /* an ASCII source widens byte by byte */
            k.kind = SK_UTF16A;
            return gen_stream(g, nd->a, &k);
        }
        /* UTF-8 decode, after a first pass proves the whole value well formed:
         * an ill-formed value converts to nothing at all, as utf8_to_utf16 does */
        ind(g); out(g, "uint u%d = 0u; int r%d = 0, ok%d = 1;\n", id, id, id);
        k.kind = SK_U8VALID;
        if (gen_stream(g, nd->a, &k) < 0) return -1;
        ind(g); out(g, "if (r%d != 0) ok%d = 0;\n", id, id);
        ind(g); out(g, "if (ok%d) {\n", id);
        g->depth++;
        ind(g); out(g, "u%d = 0u; r%d = 0;\n", id, id);
        k.kind = SK_U8DECODE;
        if (gen_stream(g, nd->a, &k) < 0) return -1;
        g->depth--;
        ind(g); out(g, "}\n");
        return 0;
    case HXX_CAP_FIRST:
        k.kind = SK_CAPFIRST;
        ind(g); out(g, "int c%d = 0;\n", id);
        return gen_stream(g, nd->a, &k);
    case HXX_CAP_AT:
        k.kind = SK_CAPAT;
        if (nd->ia < 0 && !N(g, nd->a)->addressable)
            return refuse(g, "cap counted from the end of a streamed value");
        gen_len(g, nd->a, L, sizeof L);
        ind(g); out(g, "int n%d = 0, p%d = HX_CAP_POS(%s, %d);\n", id, id,
                    nd->ia >= 0 ? "0" : L, nd->ia);
        return gen_stream(g, nd->a, &k);
    case HXX_CUT:
        k.kind = SK_CUT;
        if ((nd->ia < 0 || nd->nia < 2) && !N(g, nd->a)->addressable)
            return refuse(g, "cut from the end of a streamed value");
        if (nd->ia >= 0 && nd->nia >= 2) {
            ind(g); out(g, "int n%d = 0, cs%d = %d, cl%d = %d;\n", id, id, nd->ia, id,
                        nd->ib < 0 ? 0 : nd->ib);
        } else {
            gen_len(g, nd->a, L, sizeof L);
            ind(g); out(g, "int n%d = 0, cs%d = HX_CUT_START(%s, %d), cl%d = HX_CUT_LEN(%s, %d, %d, %d);\n",
                        id, id, L, nd->ia, id, L, nd->ia, nd->nia >= 2 ? nd->ib : -1, nd->nia >= 2);
        }
        return gen_stream(g, nd->a, &k);
    case HXX_PAD:
        k.kind = SK_PAD; k.n = nd->ia;
        ind(g); out(g, "int n%d = 0;\n", id);
        if (gen_stream(g, nd->a, &k) < 0) return -1;
        ind(g); out(g, "while (n%d < %d) {\n", id, nd->ia);
        g->depth++;
        put(g, s, "0u");
        ind(g); out(g, "n%d++;\n", id);
        g->depth--;
        ind(g); out(g, "}\n");
        return 0;
    case HXX_REV:
        if (!N(g, nd->a)->addressable) return refuse(g, "rev of a streamed value");
        return stream_indexed(g, i, s);
    case HXX_FROMHEX:
        k.kind = SK_FROMHEX;
        ind(g); out(g, "uint a%d = 0u; int q%d = 0;\n", id, id);
        return gen_stream(g, nd->a, &k);
    case HXX_BASE64: {
        int t = id;
        k.kind = SK_B64;
        ind(g); out(g, "uint a%d = 0u; int q%d = 0;\n", t, t);
        if (gen_stream(g, nd->a, &k) < 0) return -1;
        i = -1;
        goto b64_tail;
    }
    default:
        return refuse(g, "transform not generated");
    }

b64_tail:
    /* the last one or two bytes of a base64 run, with their '=' */
    {
        int t = k.id;
        char e[64];
        ind(g); out(g, "if (q%d == 1) {\n", t);
        g->depth++;
        ind(g); out(g, "a%d <<= 16;\n", t);
        snprintf(e, sizeof e, "HX_B64C(a%d >> 18)", t); put(g, s, e);
        snprintf(e, sizeof e, "HX_B64C(a%d >> 12)", t); put(g, s, e);
        put(g, s, "61u"); put(g, s, "61u");
        g->depth--;
        ind(g); out(g, "} else if (q%d == 2) {\n", t);
        g->depth++;
        ind(g); out(g, "a%d <<= 8;\n", t);
        snprintf(e, sizeof e, "HX_B64C(a%d >> 18)", t); put(g, s, e);
        snprintf(e, sizeof e, "HX_B64C(a%d >> 12)", t); put(g, s, e);
        snprintf(e, sizeof e, "HX_B64C(a%d >> 6)", t);  put(g, s, e);
        put(g, s, "61u");
        g->depth--;
        ind(g); out(g, "}\n");
        (void)i;
        return 0;
    }
}

/* ---- a hash call ---- */

/* Declare and initialise the state of body op under suffix tag. */
static void state_init(struct gen *g, int op, const char *tag)
{
    const struct body_info *bi = &BI[op];
    const char *w = bi->wide ? "ulong" : "uint";
    ind(g); out(g, "%s d%s[%d] = { %s };\n", w, tag, bi->dwords ? bi->dwords : 8, bi->iv);
    ind(g); out(g, "%s m%s[%d];\n", w, tag, bi->mw);
    ind(g); out(g, "for (int z%s = 0; z%s < %d; z%s++) m%s[z%s] = 0;\n", tag, tag, bi->mw, tag, tag, tag);
    ind(g); out(g, "uint b%s = 0, t%s = 0;\n", tag, tag);
}

/* Pad, last compress and fold of the state under tag. */
static void state_fin(struct gen *g, int op, const char *tag)
{
    const struct body_info *bi = &BI[op];
    ind(g); out(g, "%s(d%s, m%s, b%s, t%s, %s%s);\n",
                bi->fin, tag, tag, tag, tag, bi->blk, bi->fin_arg);
    if (bi->fold) { ind(g); out(g, "%s(d%s);\n", bi->fold, tag); }
}

static int gen_body_into(struct gen *g, int op, int arg, int k)
{
    struct sink s;
    memset(&s, 0, sizeof s);
    s.kind = SK_ABSORB; s.id = k; s.op = op;
    snprintf(s.tag, sizeof s.tag, "%d", k);
    state_init(g, op, s.tag);
    if (gen_stream(g, arg, &s) < 0) return -1;
    state_fin(g, op, s.tag);
    return 0;
}

/* HMAC as mhash computes it (mhash_hmac_init / mhash_hmac_deinit): with
 * block B = hmac_block(op) and the body's digest width D (TIGER128 is 16 at
 * every step), a key longer than B is replaced by its D-byte digest; inner =
 * H((key ^ 0x36) padded with 0x36 to B, then the message); outer =
 * H((key ^ 0x5c) padded likewise, then the D-byte inner digest). The result
 * is left in d<k>, where every reader of a body's digest finds it. */
static int gen_hmac_into(struct gen *g, int k)
{
    const struct hx_plan_node *nd = N(g, k);
    int op = nd->op, B = hmac_block(op), D = BI[op].out_bytes, j;
    struct sink s;
    char tag[16], e[160], ix[16];
    /* the key */
    ind(g); out(g, "uchar kp%d[%d];\n", k, B);
    ind(g); out(g, "for (int z%dp = 0; z%dp < %d; z%dp++) kp%d[z%dp] = 0;\n", k, k, B, k, k, k);
    ind(g); out(g, "int kl%d = 0;\n", k);
    snprintf(tag, sizeof tag, "%dk", k);
    state_init(g, op, tag);
    memset(&s, 0, sizeof s);
    s.kind = SK_HMACKEY; s.id = k; s.op = op; s.n = B;
    /* ponytail: the key is hashed on every candidate in case it is longer
     * than the block; branch on its length first if HMAC throughput matters */
    if (gen_stream(g, nd->b, &s) < 0) return -1;
    ind(g); out(g, "if (kl%d > %d) {\n", k, B);
    g->depth++;
    state_fin(g, op, tag);
    snprintf(tag, sizeof tag, "d%dk", k);
    for (j = 0; j < B; j++) {
        if (j < D) {
            snprintf(ix, sizeof ix, "%d", j);
            state_byte(op, tag, ix, e, sizeof e);
            ind(g); out(g, "kp%d[%d] = (uchar)%s;\n", k, j, e);
        } else {
            ind(g); out(g, "kp%d[%d] = 0;\n", k, j);
        }
    }
    g->depth--;
    ind(g); out(g, "}\n");
    /* inner */
    snprintf(tag, sizeof tag, "%di", k);
    state_init(g, op, tag);
    memset(&s, 0, sizeof s);
    s.kind = SK_ABSORB; s.op = op;
    snprintf(s.tag, sizeof s.tag, "%s", tag);
    ind(g); out(g, "for (int z%dq = 0; z%dq < %d; z%dq++) {\n", k, k, B, k);
    g->depth++;
    snprintf(e, sizeof e, "((uint)kp%d[z%dq] ^ 0x36u)", k, k);
    put(g, &s, e);
    g->depth--;
    ind(g); out(g, "}\n");
    if (gen_stream(g, nd->a, &s) < 0) return -1;
    state_fin(g, op, tag);
    /* outer, into d<k> */
    snprintf(tag, sizeof tag, "%d", k);
    state_init(g, op, tag);
    memset(&s, 0, sizeof s);
    s.kind = SK_ABSORB; s.op = op;
    snprintf(s.tag, sizeof s.tag, "%s", tag);
    ind(g); out(g, "for (int z%dr = 0; z%dr < %d; z%dr++) {\n", k, k, B, k);
    g->depth++;
    snprintf(e, sizeof e, "((uint)kp%d[z%dr] ^ 0x5cu)", k, k);
    put(g, &s, e);
    g->depth--;
    ind(g); out(g, "}\n");
    snprintf(tag, sizeof tag, "d%di", k);
    for (j = 0; j < D; j++) {
        snprintf(ix, sizeof ix, "%d", j);
        state_byte(op, tag, ix, e, sizeof e);
        put(g, &s, e);
    }
    snprintf(tag, sizeof tag, "%d", k);
    state_fin(g, op, tag);
    return 0;
}

/* Write body k's digest into ST as little-endian words of its binary. */
static void store_st(struct gen *g, int k)
{
    int words = (BI[N(g, k)->op].out_bytes + 3) / 4, w, j;
    char e[160], ix[8];
    for (w = 0; w < words; w++) {
        ind(g); out(g, "ST(%d) = ", w);
        for (j = 0; j < 4; j++) {
            snprintf(ix, sizeof ix, "%d", 4 * w + j);
            digest_byte(g, k, ix, e, sizeof e);
            out(g, "%s(%s << %d)", j ? " | " : "", e, 8 * j);
        }
        out(g, ";\n");
    }
    for (; w < 4; w++) { ind(g); out(g, "ST(%d) = 0u;\n", w); }
}

int hx_gen_c_mirror(const struct hx_plan *p)
{
    int i;
    for (i = 0; i < p->n; i++)
        if ((p->node[i].kind == HXP_BODY || p->node[i].kind == HXP_HMAC) &&
            !BI[p->node[i].op].cmirror) return 0;
    return 1;
}

int hx_gen_digest_bytes(const struct hx_plan *p)
{
    return BI[p->node[p->final_body].op].out_bytes;
}

/* At least 4: the probe reads ST(0..3), and an 8-byte digest (mysql3) is
 * compared with its upper words zero, as the hand MYSQL3 kernel stores it. */
int hx_gen_words(const struct hx_plan *p)
{
    int w = (BI[p->node[p->final_body].op].out_bytes + 3) / 4;
    return w < 4 ? 4 : w;
}

static char *gen_core(const struct hx_plan *p, int x01_silent, int x01_only, int whole,
                      char *why, int whycap);

char *hx_gen_core(const struct hx_plan *p, int x01_silent, char *why, int whycap)
{
    return gen_core(p, x01_silent, 0, 0, why, whycap);
}

char *hx_gen_core_x01(const struct hx_plan *p, char *why, int whycap)
{
    return gen_core(p, 0, 1, 0, why, whycap);
}

char *hx_gen_core_whole(const struct hx_plan *p, char *why, int whycap)
{
    return gen_core(p, 0, 0, 1, why, whycap);
}

/* The case of the text a whole-program result reads as: 1 lower-case hex,
 * 2 upper-case, 0 when the result is not the hex of its final digest (the
 * userdef CPU arm decodes that text, and feeds it back as the next password). */
static int whole_hex_case(const struct hx_plan *p)
{
    const struct hx_plan_node *r = &p->node[p->root];
    int ucase = 0;
    if (r->kind == HXP_XFORM && (r->op == HXX_UPPER || r->op == HXX_LOWER)) {
        ucase = r->op == HXX_UPPER;
        r = &p->node[r->a];
        if (r->kind != HXP_BODY || (r->view != HXV_HEX && r->view != HXV_HEXUC)) return 0;
        return ucase ? 2 : 1;
    }
    if (r->kind != HXP_BODY) return 0;
    return r->view == HXV_HEX ? 1 : r->view == HXV_HEXUC ? 2 : 0;
}

/* Every body and HMAC node, into ST: the finalize, and a whole-program
 * iterate, share it. */
static int gen_bodies(struct gen *g)
{
    const struct hx_plan *p = g->p;
    int i;
    for (i = 0; i < p->n; i++) {
        if (p->node[i].kind == HXP_HMAC) {
            if (gen_hmac_into(g, i) < 0) return -1;
            continue;
        }
        if (p->node[i].kind != HXP_BODY) continue;
        if (gen_body_into(g, p->node[i].op, p->node[i].a, i) < 0) return -1;
    }
    store_st(g, p->final_body);
    return 0;
}

static char *gen_core(const struct hx_plan *p, int x01_silent, int x01_only, int whole,
                      char *why, int whycap)
{
    struct gen g;
    int i, fb, op;
    memset(&g, 0, sizeof g);
    g.p = p;
    g.cap = 8192;
    g.b = malloc(g.cap);
    if (!g.b) { fprintf(stderr, "FATAL %s:%d: out of memory\n", __FILE__, __LINE__); exit(1); }
    g.b[0] = 0;
#define FAIL(msg) do { snprintf(why, whycap, "%s", (msg)); free(g.b); return NULL; } while (0)
    if (!p->eligible) FAIL(p->reason);
    if (p->final_prefix >= 0 || p->final_cut >= 0 || p->final_b64)
        FAIL("a stored form with a prefix, a cut or base64 is not generated yet");
    fb = p->final_body;
    op = p->node[fb].op;
    if (whole && p->channel != HXC_NONE)
        FAIL("a salted or user-keyed user type is not generated yet");
    if (whole && !whole_hex_case(p))
        FAIL("the result is not the hex of its final digest");

    out(&g, "#undef HASH_WORDS\n#define HASH_WORDS %d\n", hx_gen_words(p));
    out(&g, "#undef HASH_BLOCK_BYTES\n#define HASH_BLOCK_BYTES %d\n", BI[op].blk_bytes);
    if (x01_silent) out(&g, "#define HX_X01_SILENT 1\n");
    /* the carrier compares round 1 and stops */
    if (!whole && (p->iter_body < 0 || x01_only)) out(&g, "#define HX_NO_ITERATE 1\n");
    if (p->channel != HXC_NONE)   /* the carrier's salted signature and salt axis */
        out(&g, "#ifndef GPU_TEMPLATE_HAS_SALT\n#define GPU_TEMPLATE_HAS_SALT 1\n#endif\n");
    out(&g, "HX_STATE_DECL\n\n");

    out(&g, "%s\n{\n", p->channel != HXC_NONE ? "HX_FINALIZE_SIG_SALT" : "HX_FINALIZE_SIG");
    g.depth = 1;
    /* salt or user bytes; the carrier passes their length unsigned */
    /* a pepper entry is salt . pepper . the pepper length, 2 bytes LE */
    if (p->channel != HXC_NONE && p->pepper)
        out(&g, "    int plen = (int)salt[slen_u - 2u] | ((int)salt[slen_u - 1u] << 8);\n"
                "    int slen = (int)slen_u - 2 - plen;\n    (void)algo_mode;\n");
    else if (p->channel != HXC_NONE) out(&g, "    int slen = (int)slen_u;\n    (void)algo_mode;\n");
    if (gen_bodies(&g) < 0) FAIL(g.why);
    g.depth = 0;
    out(&g, "}\n\n");

    /* A user type (decision D5): round N+1 runs the WHOLE program again with
     * round N's text, the hex of ST, as the password -- what the userdef CPU
     * arm does. The body is emitted again over a private copy of that text. */
    if (whole) {
        int d = hx_gen_digest_bytes(p);
        out(&g, "HX_ITERATE_SIG\n{\n");
        out(&g, "    uchar hxw[%d];\n", 2 * d);
        out(&g, "    for (int i = 0; i < %d; i++) {\n", d);
        out(&g, "        uint b = (ST(i >> 2) >> ((i & 3) * 8)) & 0xffu;\n");
        out(&g, "        hxw[2 * i] = (uchar)%s(b >> 4);\n", whole_hex_case(p) == 2 ? "HX_HEXUC" : "HX_HEXC");
        out(&g, "        hxw[2 * i + 1] = (uchar)%s(b);\n", whole_hex_case(p) == 2 ? "HX_HEXUC" : "HX_HEXC");
        out(&g, "    }\n");
        out(&g, "    HX_WHOLE_DATA(data, hxw);\n    int len = %d;\n", 2 * d);
        g.depth = 1;
        if (gen_bodies(&g) < 0) FAIL(g.why);
        g.depth = 0;
        out(&g, "}\n");
        return g.b;
    }

    /* -i: round N+1 is the outer hash over round N's text, which is the final
     * value (with its case maps) read back out of ST. */
    out(&g, "HX_ITERATE_SIG\n{\n");
    g.depth = 1;
    if (p->iter_body >= 0) {
        int k = p->n;       /* a fresh body number */
        g.from_st = 1;
        if (gen_body_into(&g, op, p->root, k) < 0) FAIL(g.why);
        {
            /* store_st reads body k by node index: point it at a copy */
            const struct body_info *bi = &BI[op];
            int words = (bi->out_bytes + 3) / 4, w, j;
            for (w = 0; w < words; w++) {
                ind(&g); out(&g, "ST(%d) = ", w);
                for (j = 0; j < 4; j++) {
                    int b = 4 * w + j;
                    if (bi->wide && !bi->be)
                        out(&g, "%s((uint)(d%d[%d] >> %d) & 0xffu) << %d", j ? " | " : "",
                            k, b >> 3, (b & 7) * 8, 8 * j);
                    else if (bi->wide)
                        out(&g, "%s((uint)(d%d[%d] >> %d) & 0xffu) << %d", j ? " | " : "",
                            k, b >> 3, (7 - (b & 7)) * 8, 8 * j);
                    else if (bi->be)
                        out(&g, "%s((d%d[%d] >> %d) & 0xffu) << %d", j ? " | " : "",
                            k, b >> 2, (3 - (b & 3)) * 8, 8 * j);
                    else
                        out(&g, "%s((d%d[%d] >> %d) & 0xffu) << %d", j ? " | " : "",
                            k, b >> 2, (b & 3) * 8, 8 * j);
                }
                out(&g, ";\n");
            }
        }
        g.from_st = 0;
    }
    g.depth = 0;
    out(&g, "}\n");
#undef FAIL
    return g.b;
}

/* ---- the dialects ---- */

static const char dialect_common[] =
"/* hx generic walker dialect: shared by every backend */\n"
"#define HX_HEXC(n)   ((uint)(((n) & 15u) < 10u ? 48u + ((n) & 15u) : 87u + ((n) & 15u)))\n"
"#define HX_HEXUC(n)  ((uint)(((n) & 15u) < 10u ? 48u + ((n) & 15u) : 55u + ((n) & 15u)))\n"
"#define HX_HEXV(c)   ((uint)((c) >= 48u && (c) <= 57u ? (c) - 48u : (c) >= 97u && (c) <= 102u ? (c) - 87u : (c) >= 65u && (c) <= 70u ? (c) - 55u : 0u))\n"
"#define HX_UP(c)     ((uint)((c) >= 97u && (c) <= 122u ? (c) - 32u : (c)))\n"
"#define HX_LO(c)     ((uint)((c) >= 65u && (c) <= 90u ? (c) + 32u : (c)))\n"
"#define HX_ROT13(c)  ((uint)((c) >= 65u && (c) <= 90u ? 65u + ((c) - 65u + 13u) % 26u : (c) >= 97u && (c) <= 122u ? 97u + ((c) - 97u + 13u) % 26u : (c)))\n"
"#define HX_B64C(x)   HX_B64V((uint)(x) & 63u)\n"
"#define HX_B64V(x)   ((x) < 26u ? 65u + (x) : (x) < 52u ? 71u + (x) : (x) < 62u ? (x) - 4u : (x) == 62u ? 43u : 47u)\n"
"/* output byte k (0..3) of the base64 group a b c, given which of b, c exist */\n"
"#define HX_B64CHAR(a, b, c, k, hb, hc) ((k) == 0 ? HX_B64C((a) >> 2) : (k) == 1 ? HX_B64C((((a) & 3u) << 4) | ((b) >> 4)) : (k) == 2 ? ((hb) ? HX_B64C((((b) & 15u) << 2) | ((c) >> 6)) : 61u) : ((hc) ? HX_B64C((c) & 63u) : 61u))\n"
"/* fn_cut: start and length of cut(x, s[, l]) of an L-byte value */\n"
"#define HX_CUT_S0(L, s)   ((s) < 0 ? ((L) + (s) < 0 ? 0 : (L) + (s)) : (s))\n"
"#define HX_CUT_START(L, s) (HX_CUT_S0(L, s) > (L) ? (L) : HX_CUT_S0(L, s))\n"
"#define HX_CUT_LEN(L, s, l, hl) (!(hl) ? (L) - HX_CUT_START(L, s) : ((l) < 0 ? 0 : (HX_CUT_START(L, s) + (l) > (L) ? (L) - HX_CUT_START(L, s) : (l))))\n"
"/* fn_cap: position N, counted from the end when negative */\n"
"#define HX_CAP_POS(L, p) ((p) < 0 ? (L) + (p) : (p))\n"
"/* One byte of strict UTF-8 (utf8_to_utf32): u accumulates the codepoint, r\n"
" * counts continuation bytes still owed, ok drops to 0 on any ill-formed\n"
" * sequence (no lead, 0xC0/0xC1/0xF5+, a non-continuation, an overlong, a\n"
" * surrogate, or past U+10FFFF). r == 0 after the byte: a codepoint is done. */\n"
"#define HX_U8_STEP(x, u, r, ok) do { uint _c = (x); \\\n"
"    if ((r) == 0) { \\\n"
"        if (_c < 0x80u) { (u) = _c; } \\\n"
"        else if (_c >= 0xc2u && _c <= 0xdfu) { (u) = _c & 0x1fu; (r) = 1; } \\\n"
"        else if (_c >= 0xe0u && _c <= 0xefu) { (u) = (_c & 0x0fu) | 0x80000000u; (r) = 2; } \\\n"
"        else if (_c >= 0xf0u && _c <= 0xf4u) { (u) = (_c & 0x07u) | 0x40000000u; (r) = 3; } \\\n"
"        else { (ok) = 0; } \\\n"
"    } else if ((_c & 0xc0u) != 0x80u) { (ok) = 0; (r) = 0; } \\\n"
"    else { uint _f = (u) & 0xc0000000u; (u) = ((((u) & 0x3fffffffu) << 6) | (_c & 0x3fu)) | _f; \\\n"
"        if (--(r) == 0) { uint _cp = (u) & 0x3fffffffu; \\\n"
"            if ((_f == 0x80000000u && _cp < 0x800u) || (_f == 0x40000000u && _cp < 0x10000u) || \\\n"
"                (_cp >= 0xd800u && _cp <= 0xdfffu) || _cp > 0x10ffffu) (ok) = 0; \\\n"
"            (u) = _cp; } } } while (0)\n"
"/* Absorb one byte into a block; compress when it fills. */\n"
"#define HX_ABS32LE(S, M, B, T, X, BLK) do { (M)[(B) >> 2] |= ((uint)(X)) << (((B) & 3u) * 8u); (T)++; \\\n"
"    if (++(B) == 64u) { BLK(S, M); (B) = 0; for (int _z = 0; _z < 16; _z++) (M)[_z] = 0; } } while (0)\n"
"#define HX_ABS32BE(S, M, B, T, X, BLK) do { (M)[(B) >> 2] |= ((uint)(X)) << ((3u - ((B) & 3u)) * 8u); (T)++; \\\n"
"    if (++(B) == 64u) { BLK(S, M); (B) = 0; for (int _z = 0; _z < 16; _z++) (M)[_z] = 0; } } while (0)\n"
"#define HX_ABS64BE(S, M, B, T, X, BLK) do { (M)[(B) >> 3] |= ((ulong)(X)) << ((7u - ((B) & 7u)) * 8u); (T)++; \\\n"
"    if (++(B) == 128u) { BLK(S, M); (B) = 0; for (int _z = 0; _z < 16; _z++) (M)[_z] = 0; } } while (0)\n"
"/* Merkle-Damgard padding: 0x80, zeros, the bit length. The block is not\n"
" * full on entry, so 0x80 always fits; the words past it are already zero.\n"
" * Written directly rather than through HX_ABS*: an absorb loop and an\n"
" * HX_ABS-built padding in one function crashed the AMD fglrx compiler\n"
" * (Tahiti, libamdocl64 segfault) for an outer sha256. */\n"
"#define HX_FIN32LE(S, M, B, T, BLK) do { uint _t = (T); \\\n"
"    (M)[(B) >> 2] |= 0x80u << (((B) & 3u) * 8u); \\\n"
"    if ((B) >= 56u) { BLK(S, M); for (int _z = 0; _z < 16; _z++) (M)[_z] = 0; } \\\n"
"    (M)[14] = _t << 3; (M)[15] = _t >> 29; BLK(S, M); } while (0)\n"
"#define HX_FIN32BE(S, M, B, T, BLK) do { uint _t = (T); \\\n"
"    (M)[(B) >> 2] |= 0x80u << ((3u - ((B) & 3u)) * 8u); \\\n"
"    if ((B) >= 56u) { BLK(S, M); for (int _z = 0; _z < 16; _z++) (M)[_z] = 0; } \\\n"
"    (M)[14] = _t >> 29; (M)[15] = _t << 3; BLK(S, M); } while (0)\n"
"#define HX_FIN64BE(S, M, B, T, BLK) do { uint _t = (T); \\\n"
"    (M)[(B) >> 3] |= ((ulong)0x80u) << ((7u - ((B) & 7u)) * 8u); \\\n"
"    if ((B) >= 112u) { BLK(S, M); for (int _z = 0; _z < 16; _z++) (M)[_z] = 0; } \\\n"
"    (M)[14] = 0; (M)[15] = ((ulong)_t) << 3; BLK(S, M); } while (0)\n"
"/* Stage 1b block shapes. Tiger: little-endian 64-bit words in a 64-byte\n"
" * block, pad 0x01, length in M[7]. Whirlpool: big-endian 64-bit words in a\n"
" * 64-byte block, a 256-bit length field whose low 64 bits are M[7]. HAVAL:\n"
" * a 128-byte block of 32 little-endian words, pad 0x01, then the two\n"
" * parameter bytes at 118-119 (P, in the upper half of M[29]) and the bit\n"
" * length at 120-127. */\n"
"#define HX_ABS64LE(S, M, B, T, X, BLK) do { (M)[(B) >> 3] |= ((ulong)(X)) << (((B) & 7u) * 8u); (T)++; \\\n"
"    if (++(B) == 64u) { BLK(S, M); (B) = 0; for (int _z = 0; _z < 8; _z++) (M)[_z] = 0; } } while (0)\n"
"#define HX_ABS64BE64(S, M, B, T, X, BLK) do { (M)[(B) >> 3] |= ((ulong)(X)) << ((7u - ((B) & 7u)) * 8u); (T)++; \\\n"
"    if (++(B) == 64u) { BLK(S, M); (B) = 0; for (int _z = 0; _z < 8; _z++) (M)[_z] = 0; } } while (0)\n"
"#define HX_ABS32LE128(S, M, B, T, X, BLK) do { (M)[(B) >> 2] |= ((uint)(X)) << (((B) & 3u) * 8u); (T)++; \\\n"
"    if (++(B) == 128u) { BLK(S, M); (B) = 0; for (int _z = 0; _z < 32; _z++) (M)[_z] = 0; } } while (0)\n"
"#define HX_FIN64LE01(S, M, B, T, BLK) do { uint _t = (T); \\\n"
"    (M)[(B) >> 3] |= ((ulong)0x01u) << (((B) & 7u) * 8u); \\\n"
"    if ((B) >= 56u) { BLK(S, M); for (int _z = 0; _z < 8; _z++) (M)[_z] = 0; } \\\n"
"    (M)[7] = ((ulong)_t) << 3; BLK(S, M); } while (0)\n"
"#define HX_FIN64BE256(S, M, B, T, BLK) do { uint _t = (T); \\\n"
"    (M)[(B) >> 3] |= ((ulong)0x80u) << ((7u - ((B) & 7u)) * 8u); \\\n"
"    if ((B) >= 32u) { BLK(S, M); for (int _z = 0; _z < 8; _z++) (M)[_z] = 0; } \\\n"
"    (M)[7] = ((ulong)_t) << 3; BLK(S, M); } while (0)\n"
"#define HX_FINHAVAL(S, M, B, T, BLK, P) do { uint _t = (T); \\\n"
"    (M)[(B) >> 2] |= 0x01u << (((B) & 3u) * 8u); \\\n"
"    if ((B) >= 118u) { BLK(S, M); for (int _z = 0; _z < 32; _z++) (M)[_z] = 0; } \\\n"
"    (M)[29] |= ((uint)(P)) << 16; (M)[30] = _t << 3; (M)[31] = _t >> 29; BLK(S, M); } while (0)\n"
"/* HAVAL output folds (mhash havalFinal), for widths below 256 bits */\n"
"#define HX_ROTR32(x, n) ((((uint)(x)) >> (n)) | (((uint)(x)) << (32 - (n))))\n"
"#define HX_HAVFOLD16(S) do { \\\n"
"    (S)[3] += ((S)[7] & 0xff000000u) | ((S)[6] & 0x00ff0000u) | ((S)[5] & 0x0000ff00u) | ((S)[4] & 0x000000ffu); \\\n"
"    (S)[2] += ((((S)[7] & 0x00ff0000u) | ((S)[6] & 0x0000ff00u) | ((S)[5] & 0x000000ffu)) << 8) | (((S)[4] & 0xff000000u) >> 24); \\\n"
"    (S)[1] += ((((S)[7] & 0x0000ff00u) | ((S)[6] & 0x000000ffu)) << 16) | ((((S)[5] & 0xff000000u) | ((S)[4] & 0x00ff0000u)) >> 16); \\\n"
"    (S)[0] += ((((S)[6] & 0xff000000u) | ((S)[5] & 0x00ff0000u) | ((S)[4] & 0x0000ff00u)) >> 8) | (((S)[7] & 0x000000ffu) << 24); \\\n"
"    } while (0)\n"
"#define HX_HAVFOLD20(S) do { \\\n"
"    (S)[4] += (((S)[7] & 0xfe000000u) | ((S)[6] & 0x01f80000u) | ((S)[5] & 0x0007f000u)) >> 12; \\\n"
"    (S)[3] += (((S)[7] & 0x01f80000u) | ((S)[6] & 0x0007f000u) | ((S)[5] & 0x00000fc0u)) >> 6; \\\n"
"    (S)[2] += ((S)[7] & 0x0007f000u) | ((S)[6] & 0x00000fc0u) | ((S)[5] & 0x0000003fu); \\\n"
"    (S)[1] += HX_ROTR32(((S)[7] & 0x00000fc0u) | ((S)[6] & 0x0000003fu) | ((S)[5] & 0xfe000000u), 25); \\\n"
"    (S)[0] += HX_ROTR32(((S)[7] & 0x0000003fu) | ((S)[6] & 0xfe000000u) | ((S)[5] & 0x01f80000u), 19); \\\n"
"    } while (0)\n"
"#define HX_HAVFOLD24(S) do { \\\n"
"    (S)[5] += (((S)[7] & 0xfc000000u) | ((S)[6] & 0x03e00000u)) >> 21; \\\n"
"    (S)[4] += (((S)[7] & 0x03e00000u) | ((S)[6] & 0x001f0000u)) >> 16; \\\n"
"    (S)[3] += (((S)[7] & 0x001f0000u) | ((S)[6] & 0x0000fc00u)) >> 10; \\\n"
"    (S)[2] += (((S)[7] & 0x0000fc00u) | ((S)[6] & 0x000003e0u)) >> 5; \\\n"
"    (S)[1] += ((S)[7] & 0x000003e0u) | ((S)[6] & 0x0000001fu); \\\n"
"    (S)[0] += HX_ROTR32(((S)[7] & 0x0000001fu) | ((S)[6] & 0xfc000000u), 26); \\\n"
"    } while (0)\n"
"#define HX_HAVFOLD28(S) do { \\\n"
"    (S)[6] += (S)[7] & 0x0000000fu; (S)[5] += ((S)[7] >> 4) & 0x0000001fu; \\\n"
"    (S)[4] += ((S)[7] >> 9) & 0x0000000fu; (S)[3] += ((S)[7] >> 13) & 0x0000001fu; \\\n"
"    (S)[2] += ((S)[7] >> 18) & 0x0000000fu; (S)[1] += ((S)[7] >> 22) & 0x0000001fu; \\\n"
"    (S)[0] += ((S)[7] >> 27) & 0x0000001fu; \\\n"
"    } while (0)\n"
"/* Stage 1b, non-MD finalisers. A message held as bytes in order, in\n"
" * little-endian words, N bytes to a block of W words. */\n"
"#define HX_ABS32LEN(S, M, B, T, X, BLK, N, W) do { (M)[(B) >> 2] |= ((uint)(X)) << (((B) & 3u) * 8u); (T)++; \\\n"
"    if (++(B) == (N)) { BLK(S, M); (B) = 0; for (int _z = 0; _z < (W); _z++) (M)[_z] = 0; } } while (0)\n"
"#define HX_ABS32LE16(S, M, B, T, X, BLK) HX_ABS32LEN(S, M, B, T, X, BLK, 16u, 4)\n"
"#define HX_ABS32LE32(S, M, B, T, X, BLK) HX_ABS32LEN(S, M, B, T, X, BLK, 32u, 8)\n"
"#define HX_ABS32LE48(S, M, B, T, X, BLK) HX_ABS32LEN(S, M, B, T, X, BLK, 48u, 12)\n"
"#define HX_BSWAP32(x) ((((uint)(x)) >> 24) | ((((uint)(x)) >> 8) & 0xff00u) | ((((uint)(x)) << 8) & 0xff0000u) | (((uint)(x)) << 24))\n"
"/* MD2 (RFC 1319): pad 1..16 bytes of value 16 - len % 16, then the\n"
" * checksum as a last block that does not update the checksum. */\n"
"#define HX_FINMD2(S, M, B, T, BLK) do { uint _p = 16u - (B); \\\n"
"    for (uint _i = (B); _i < 16u; _i++) (M)[_i >> 2] |= _p << ((_i & 3u) * 8u); \\\n"
"    BLK(S, M); for (int _z = 0; _z < 4; _z++) (M)[_z] = (S)[12 + _z]; HXB_MD2F(S, M); } while (0)\n"
"/* GOST R 34.11-94 (gosthash.c): each block, the last zero-padded, is\n"
" * compressed and added into the 256-bit sum in m[8..15]; then the bit\n"
" * length is compressed, then the sum. */\n"
"#define HX_GOSTSUM(M) do { uint _c = 0u; for (int _i = 0; _i < 8; _i++) { \\\n"
"    uint _a = (M)[_i], _b = (M)[8 + _i]; _c = _a + _c + _b; (M)[8 + _i] = _c; \\\n"
"    _c = ((_c < _a) || (_c < _b)) ? 1u : 0u; } } while (0)\n"
"#define HX_ABSGOST(S, M, B, T, X, BLK) do { (M)[(B) >> 2] |= ((uint)(X)) << (((B) & 3u) * 8u); (T)++; \\\n"
"    if (++(B) == 32u) { BLK(S, M); HX_GOSTSUM(M); (B) = 0; for (int _z = 0; _z < 8; _z++) (M)[_z] = 0; } } while (0)\n"
"#define HX_FINGOST(S, M, B, T, BLK) do { uint _t = (T); \\\n"
"    if ((B) > 0u) { BLK(S, M); HX_GOSTSUM(M); } \\\n"
"    for (int _z = 0; _z < 8; _z++) (M)[_z] = 0; (M)[0] = _t << 3; (M)[1] = _t >> 29; BLK(S, M); \\\n"
"    for (int _z = 0; _z < 8; _z++) (M)[_z] = (M)[8 + _z]; BLK(S, M); } while (0)\n"
"/* Snefru (rhash): a partial block zero-padded, then a block holding only\n"
" * the 64-bit big-endian bit length in its last 8 bytes. */\n"
"#define HX_FINSNE(S, M, B, T, BLK, N) do { uint _t = (T); \\\n"
"    if ((B) > 0u) { BLK(S, M); for (int _z = 0; _z < (int)((N) / 4u); _z++) (M)[_z] = 0; } \\\n"
"    (M)[(N) / 4u - 2u] = HX_BSWAP32(_t >> 29); (M)[(N) / 4u - 1u] = HX_BSWAP32(_t << 3); BLK(S, M); } while (0)\n"
"/* MySQL 3 OLD_PASSWORD: no blocks; BLK and M are unused. */\n"
"#define HX_ABSMYSQL3(S, M, B, T, X, BLK) do { uint _c = (uint)(X); (T)++; \\\n"
"    if (_c != 0x20u && _c != 0x09u) { (S)[0] ^= ((((S)[0] & 63u) + (S)[2]) * _c) + ((S)[0] << 8); \\\n"
"        (S)[1] += ((S)[1] << 8) ^ (S)[0]; (S)[2] += _c; } } while (0)\n"
"#define HX_FINMYSQL3(S, M, B, T, BLK) do { (S)[0] &= 0x7fffffffu; (S)[1] &= 0x7fffffffu; } while (0)\n"
"/* BLAKE2b and BLAKE2s: a full block is compressed when the next byte\n"
" * arrives, with the byte count so far; fin compresses the zero-padded\n"
" * last block (all zeros for an empty input) with the final flag. */\n"
"#define HX_ABSB2B(S, M, B, T, X, BLK) do { if ((B) == 128u) { HXB_B2BC(S, M, (T), 0); (B) = 0; \\\n"
"    for (int _z = 0; _z < 16; _z++) (M)[_z] = 0; } \\\n"
"    (M)[(B) >> 3] |= ((ulong)(X)) << (((B) & 7u) * 8u); (B)++; (T)++; } while (0)\n"
"#define HX_FINB2B(S, M, B, T, BLK) HXB_B2BC(S, M, (T), 1)\n"
"#define HX_ABSB2S(S, M, B, T, X, BLK) do { if ((B) == 64u) { HXB_B2SC(S, M, (T), 0); (B) = 0; \\\n"
"    for (int _z = 0; _z < 16; _z++) (M)[_z] = 0; } \\\n"
"    (M)[(B) >> 2] |= ((uint)(X)) << (((B) & 3u) * 8u); (B)++; (T)++; } while (0)\n"
"#define HX_FINB2S(S, M, B, T, BLK) HXB_B2SC(S, M, (T), 1)\n"
"/* Keccak sponge: the rate's message words are XORed into the state and\n"
" * permuted when they fill; fin adds the domain pad P and the final 0x80. */\n"
"#define HX_KECXOR(S, M, R) do { for (int _z = 0; _z < (int)((R) / 8u); _z++) { (S)[_z] ^= (M)[_z]; (M)[_z] = 0; } \\\n"
"    keccakf1600(S); } while (0)\n"
"#define HX_ABSKECN(S, M, B, T, X, R) do { (M)[(B) >> 3] |= ((ulong)(X)) << (((B) & 7u) * 8u); (T)++; \\\n"
"    if (++(B) == (R)) { HX_KECXOR(S, M, R); (B) = 0; } } while (0)\n"
"#define HX_ABSKEC144(S, M, B, T, X, BLK) HX_ABSKECN(S, M, B, T, X, 144u)\n"
"#define HX_ABSKEC136(S, M, B, T, X, BLK) HX_ABSKECN(S, M, B, T, X, 136u)\n"
"#define HX_ABSKEC104(S, M, B, T, X, BLK) HX_ABSKECN(S, M, B, T, X, 104u)\n"
"#define HX_ABSKEC72(S, M, B, T, X, BLK) HX_ABSKECN(S, M, B, T, X, 72u)\n"
"#define HX_FINKEC(S, M, B, T, BLK, P, R) do { (M)[(B) >> 3] ^= ((ulong)(P)) << (((B) & 7u) * 8u); \\\n"
"    (M)[((R) - 1u) >> 3] ^= 0x80ul << ((((R) - 1u) & 7u) * 8u); HX_KECXOR(S, M, R); } while (0)\n";

static const char dialect_c[] =
"/* hx generic walker dialect: C, for the host check */\n"
"#include <stdint.h>\n"
"typedef uint32_t uint; typedef uint8_t uchar; typedef uint64_t ulong;\n"
"typedef struct { uint h[16]; } template_state;\n"
"void hxc_md4(uint *s, uint *m); void hxc_md5(uint *s, uint *m);\n"
"void hxc_sha1(uint *s, uint *m); void hxc_sha256(uint *s, uint *m);\n"
"void hxc_sha512(ulong *s, ulong *m);\n"
"#define HXB_MD4(S, M)    hxc_md4(S, M)\n"
"#define HXB_MD5(S, M)    hxc_md5(S, M)\n"
"#define HXB_SHA1(S, M)   hxc_sha1(S, M)\n"
"#define HXB_SHA256(S, M) hxc_sha256(S, M)\n"
"#define HXB_SHA512(S, M) hxc_sha512(S, M)\n"
"#define ST(i) (st->h[i])\n"
"#define HX_STATE_DECL\n"
"#define HX_PASTE2(a, b) a##b\n"
"#define HX_PASTE(a, b) HX_PASTE2(a, b)\n"
"#define HX_FINALIZE_SIG static void HX_PASTE(hx_fin_, HXID)(template_state *st, const uchar *data, int len)\n"
"#define HX_FINALIZE_SIG_SALT static void HX_PASTE(hx_fin_, HXID)(template_state *st, const uchar *data, int len, const uchar *salt, uint slen_u, uint algo_mode)\n"
"#define HX_ITERATE_SIG static void HX_PASTE(hx_iter_, HXID)(template_state *st)\n"
"#define HX_WHOLE_DATA(d, b) const uchar *d = (b)\n";


/* Hit records, shared by the GPU dialects: EMIT_HIT_<HASH_WORDS>. ST already
 * holds little-endian words of the stored binary, which is what the probe and
 * the record want, so neither swaps. EMIT_HIT_4 takes four words, the rest an
 * array. */
#define GPU_EMIT \
"#define HX_P2(a, b) a##b\n" \
"#define HX_P(a, b) HX_P2(a, b)\n" \
"#define HX_EMIT_4(hits, hc, mh, w, s, it, h, hs, mi, dm, os, og, lg) EMIT_HIT_4_DEDUP_OR_OVERFLOW(hits, hc, mh, w, s, it, (h)[0], (h)[1], (h)[2], (h)[3], hs, mi, dm, os, og, lg)\n" \
"#define HX_EMIT_5 EMIT_HIT_5_DEDUP_OR_OVERFLOW\n" \
"#define HX_EMIT_6 EMIT_HIT_6_DEDUP_OR_OVERFLOW\n" \
"#define HX_EMIT_7 EMIT_HIT_7_DEDUP_OR_OVERFLOW\n" \
"#define HX_EMIT_8 EMIT_HIT_8_DEDUP_OR_OVERFLOW\n" \
"#define HX_EMIT_12 EMIT_HIT_12_DEDUP_OR_OVERFLOW\n" \
"#define HX_EMIT_16 EMIT_HIT_16_DEDUP_OR_OVERFLOW\n"

static const char dialect_opencl[] =
"/* hx generic walker dialect: OpenCL (gpu_template.cl hooks) */\n"
"#define HXB_MD4(S, M)    md4_block(&(S)[0], &(S)[1], &(S)[2], &(S)[3], M)\n"
"#define HXB_MD5(S, M)    md5_block(&(S)[0], &(S)[1], &(S)[2], &(S)[3], M)\n"
"#define HXB_SHA1(S, M)   sha1_block(S, M)\n"
"#define HXB_SHA256(S, M) sha256_block(S, M)\n"
"#define HXB_SHA512(S, M) sha512_block(S, M)\n"
"#define HXB_RMD128(S, M) rmd128_block(S, M)\n"
"#define HXB_RMD160(S, M) rmd160_block(S, M)\n"
"#define HXB_TIGER(S, M)  tiger_block(S, M)\n"
"#define HXB_WRL(S, M)    wrl_block(S, M)\n"
"#define HXB_HAVAL3(S, M) haval3_block(S, M)\n"
"#define HXB_HAVAL4(S, M) haval4_block(S, M)\n"
"#define HXB_HAVAL5(S, M) haval5_block(S, M)\n"
"#define HXB_MD2(S, M)    md2_block((uchar *)(S), (uchar *)(S) + 48, (const uchar *)(M), 1)\n"
"#define HXB_MD2F(S, M)   md2_block((uchar *)(S), (uchar *)(S) + 48, (const uchar *)(M), 0)\n"
"#define HXB_GOST(S, M)   gost_block(S, M)\n"
"#define HXB_SNE128(S, M) snefru_block(S, (const uchar *)(M), 0)\n"
"#define HXB_SNE256(S, M) snefru_block(S, (const uchar *)(M), 1)\n"
"#define HXB_B2BC(S, M, T, L) b2b_compress(S, (const uchar *)(M), (ulong)(T), 0ul, L)\n"
"#define HXB_B2SC(S, M, T, L) b2s_compress(S, (const uchar *)(M), (ulong)(T), L)\n"
"#define ST(i) (st->h[i])\n"
GPU_EMIT
"#define HX_STATE_DECL \\\n"
"typedef struct { uint h[HASH_WORDS]; } template_state; \\\n"
"static inline void template_init(template_state *st) { (void)st; } \\\n"
"static inline int template_digest_compare(const template_state *st, \\\n"
"    __global const uint *compact_fp, __global const uint *compact_idx, \\\n"
"    ulong compact_mask, uint max_probe, uint hash_data_count, \\\n"
"    __global const uchar *hash_data_buf, __global const ulong *hash_data_off, \\\n"
"    __global const ulong *overflow_keys, __global const uchar *overflow_hashes, \\\n"
"    __global const uint *overflow_offsets, uint overflow_count, uint *out_idx) \\\n"
"{ return probe_compact_idx(st->h[0], st->h[1], st->h[2], st->h[3], compact_fp, compact_idx, \\\n"
"    compact_mask, max_probe, hash_data_count, hash_data_buf, hash_data_off, \\\n"
"    overflow_keys, overflow_hashes, overflow_offsets, overflow_count, out_idx); }\n"
"#define template_emit_hit_or_overflow(hits, hit_count, max_hits, st, widx, sidx, iter, \\\n"
"        hashes_shown, matched_idx, dedup_mask, ovr_set, ovr_gid, lane_gid) \\\n"
"    HX_P(HX_EMIT_, HASH_WORDS)((hits), (hit_count), (max_hits), (widx), (sidx), (iter), (st)->h, \\\n"
"        (hashes_shown), (matched_idx), (dedup_mask), (ovr_set), (ovr_gid), (lane_gid))\n"
"#define HX_FINALIZE_SIG static inline void template_finalize(template_state *st, const uchar *data, int len)\n"
"#define HX_FINALIZE_SIG_SALT static inline void template_finalize(template_state *st, const uchar *data, int len, \\\n"
"    __global const uchar *salt, uint slen_u, uint algo_mode)\n"
"#define HX_ITERATE_SIG static inline void template_iterate(template_state *st)\n"
"#define HX_WHOLE_DATA(d, b) const uchar *d = (b)\n";

static const char dialect_metal[] =
"/* hx generic walker dialect: Metal (metal_template.metal hooks) */\n"
"#define HXB_MD4(S, M)    md4_block(S, M)\n"
"#define HXB_MD5(S, M)    md5_block((S)[0], (S)[1], (S)[2], (S)[3], M)\n"
"#define HXB_SHA1(S, M)   sha1_block(S, M)\n"
"#define HXB_SHA256(S, M) sha256_block(S, M)\n"
"#define HXB_SHA512(S, M) sha512_block(S, M)\n"
"#define HXB_RMD128(S, M) rmd128_block(S, M)\n"
"#define HXB_RMD160(S, M) rmd160_block(S, M)\n"
"#define HXB_TIGER(S, M)  tiger_block(S, M)\n"
"#define HXB_WRL(S, M)    wrl_block(S, M)\n"
"#define HXB_HAVAL3(S, M) haval3_block(S, M)\n"
"#define HXB_HAVAL4(S, M) haval4_block(S, M)\n"
"#define HXB_HAVAL5(S, M) haval5_block(S, M)\n"
"#define HXB_MD2(S, M)    md2_block((thread uchar *)(S), (thread uchar *)(S) + 48, (thread const uchar *)(M), 1)\n"
"#define HXB_MD2F(S, M)   md2_block((thread uchar *)(S), (thread uchar *)(S) + 48, (thread const uchar *)(M), 0)\n"
"#define HXB_GOST(S, M)   gost_block(S, M)\n"
"#define HXB_SNE128(S, M) snefru_block(S, (thread const uchar *)(M), 0)\n"
"#define HXB_SNE256(S, M) snefru_block(S, (thread const uchar *)(M), 1)\n"
"#define HXB_B2BC(S, M, T, L) b2b_compress(S, (thread const uchar *)(M), (ulong)(T), 0ul, L)\n"
"#define HXB_B2SC(S, M, T, L) b2s_compress(S, (thread const uchar *)(M), (ulong)(T), L)\n"
"#define ST(i) (st.h[i])\n"
GPU_EMIT
"#define HX_STATE_DECL \\\n"
"struct template_state { uint h[HASH_WORDS]; }; \\\n"
"static inline void template_init(thread template_state &st) { (void)st; } \\\n"
"static inline int template_digest_compare(thread const template_state &st, \\\n"
"    device const uint *compact_fp, device const uint *compact_idx, \\\n"
"    ulong compact_mask, uint max_probe, uint hash_data_count, \\\n"
"    device const uchar *hash_data_buf, device const ulong *hash_data_off, \\\n"
"    device const ulong *overflow_keys, device const uchar *overflow_hashes, \\\n"
"    device const uint *overflow_offsets, uint overflow_count, thread uint *out_idx) \\\n"
"{ return probe_compact_idx(st.h[0], st.h[1], st.h[2], st.h[3], compact_fp, compact_idx, \\\n"
"    compact_mask, max_probe, hash_data_count, hash_data_buf, hash_data_off, \\\n"
"    overflow_keys, overflow_hashes, overflow_offsets, overflow_count, out_idx); }\n"
"#define template_emit_hit_or_overflow(hits, hit_count, max_hits, st, widx, sidx, iter, \\\n"
"        hashes_shown, matched_idx, dedup_mask, ovr_set, ovr_gid, lane_gid) \\\n"
"    HX_P(HX_EMIT_, HASH_WORDS)((hits), (hit_count), (max_hits), (widx), (sidx), (iter), (st).h, \\\n"
"        (hashes_shown), (matched_idx), (dedup_mask), (ovr_set), (ovr_gid), (lane_gid))\n"
"#define HX_FINALIZE_SIG static inline void template_finalize(thread template_state &st, device const uchar *data, int len)\n"
"#define HX_FINALIZE_SIG_SALT static inline void template_finalize(thread template_state &st, device const uchar *data, int len, \\\n"
"    device const uchar *salt, uint slen_u, uint algo_mode)\n"
"#define HX_ITERATE_SIG static inline void template_iterate(thread template_state &st)\n"
"#define HX_WHOLE_DATA(d, b) thread const uchar *d = (b)\n";

static char *dialect_buf[3];

const char *hx_gen_dialect(enum hx_gen_backend backend)
{
    const char *own;
    size_t a, b;
    if (dialect_buf[backend]) return dialect_buf[backend];
    switch (backend) {
    case HXG_C: own = dialect_c; break;
    case HXG_OPENCL: own = dialect_opencl; break;
    case HXG_METAL: own = dialect_metal; break;
    default:
        fprintf(stderr, "FATAL %s:%d: unknown backend %d\n", __FILE__, __LINE__, (int)backend);
        exit(1);
    }
    a = strlen(own); b = strlen(dialect_common);
    dialect_buf[backend] = malloc(a + b + 1);
    if (!dialect_buf[backend]) { fprintf(stderr, "FATAL %s:%d: out of memory\n", __FILE__, __LINE__); exit(1); }
    memcpy(dialect_buf[backend], own, a);
    memcpy(dialect_buf[backend] + a, dialect_common, b + 1);
    return dialect_buf[backend];
}
