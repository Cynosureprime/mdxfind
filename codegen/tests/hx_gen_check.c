/*
 * hx_gen_check.c -- run every core hx_gen_emit wrote, compiled as C through
 * the C dialect, against the hx VM.
 *
 * For each type and each test password: template_finalize must leave in ST
 * the binary of the stored form the VM computes (hex is decoded, case
 * ignored, since the compact table holds the binary), and template_iterate
 * must then leave the binary of the outer hash over that stored text, which
 * is what mdxfind's x02 is.
 *
 *   hx_gen_check          summary
 *   hx_gen_check -v       also every failure
 *
 * Built by: hx_gen_emit cores.c, then cc ... hx_gen_check.c cores.c (see the
 * make target).
 *
 * $Revision: 1.3 $
 * $Log: hx_gen_check.c,v $
 * Revision 1.3  2026/11/20 05:22:13  dlr
 * Generic hx walker stage 1c: user-defined types run on the GPU through the generated walker. A user type is served when it takes no salt or user, declares no stored form, and its result is the hex of its final digest; rounds past x01 re-run the whole program on the previous round's hex text, keeping its case, as the userdef CPU arm does. The selected user type's status line says whether it runs on the GPU, or why not, in place of the former not-yet-enabled line. Verified: the C mirror agrees for 529 of 529 cores, including 10 user cores whose later rounds are checked against the hx VM re-running the program, and a planted case defect is caught in exactly the 8 case-sensitive ones; the 519 catalog cores are byte-identical; on dev3 (M2 Max, Metal) and gp1 (Tahiti, OpenCL) 10 user types match -G none at x01 and x02 and deeper with rules, a -n mask and a mixed run with a catalog type, except one type whose kernel the Tahiti compiler crashes on, which is FATAL there as for any such kernel.
 *
 * Revision 1.2  2026/11/20 02:52:03  dlr
 * Generic hx walker stage 2: the plan and emitter take a salt or user channel, HMAC, round-1-only types and peppers. A salt or user variable reads the 6-argument salted template_finalize; HMAC is generated over any existing body following mhash: block from pblock with 0 as 64, MD2 16, HAVAL and SHA-384/512 128, a key longer than the block replaced by its digest, the inner digest entering the outer hash at the body width, Tiger in mhash byte order; HX_NO_ITERATE marks a type compared at round 1 only. A pepper rides the salt channel: the host packs salt . pepper . the pepper length in 2 bytes, and the core splits it. Verified: the C mirror agrees for 519 of 519 cores over 67548 finalize and 64194 iterate vectors, including the 14 pepper cores over 8 salts by 6 peppers up to 300 bytes; planted defects in the split and in the length high byte are caught in exactly those 14; the 505 earlier cores are byte-identical.
 *
 * Revision 1.1  2026/11/18 04:08:38  dlr
 * Initial revision.
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <openssl/md4.h>
#include <openssl/md5.h>
#include <openssl/sha.h>
#include <openssl/hmac.h>
#include <openssl/evp.h>
#include "hx_vm.h"
#include "codegen/hx_spec_entry.h"
#include "codegen/hx_plan.h"

typedef struct { uint32_t h[16]; } template_state;
struct hx_gen_fn { int job; void (*fin)(template_state *, const uint8_t *, int);
    void (*fin_s)(template_state *, const uint8_t *, int, const uint8_t *, uint32_t, uint32_t);
    void (*iter)(template_state *); int has_iter; int chan; };
extern const struct hx_gen_fn hx_gen_table[];
extern const int hx_gen_count;
/* user-type expressions, jobs 1100 + index, has_iter 2 (whole program) */
extern const char *const hx_gen_user_expr[];

/* ---- block functions for the C dialect, over OpenSSL's ---- */

void hxc_md5(uint32_t *s, uint32_t *m)
{
    MD5_CTX c; unsigned char b[64]; int i;
    for (i = 0; i < 64; i++) b[i] = (unsigned char)(m[i >> 2] >> ((i & 3) * 8));
    c.A = s[0]; c.B = s[1]; c.C = s[2]; c.D = s[3];
    MD5_Transform(&c, b);
    s[0] = c.A; s[1] = c.B; s[2] = c.C; s[3] = c.D;
}
void hxc_md4(uint32_t *s, uint32_t *m)
{
    MD4_CTX c; unsigned char b[64]; int i;
    for (i = 0; i < 64; i++) b[i] = (unsigned char)(m[i >> 2] >> ((i & 3) * 8));
    c.A = s[0]; c.B = s[1]; c.C = s[2]; c.D = s[3];
    MD4_Transform(&c, b);
    s[0] = c.A; s[1] = c.B; s[2] = c.C; s[3] = c.D;
}
void hxc_sha1(uint32_t *s, uint32_t *m)
{
    SHA_CTX c; unsigned char b[64]; int i;
    for (i = 0; i < 64; i++) b[i] = (unsigned char)(m[i >> 2] >> ((3 - (i & 3)) * 8));
    c.h0 = s[0]; c.h1 = s[1]; c.h2 = s[2]; c.h3 = s[3]; c.h4 = s[4];
    SHA1_Transform(&c, b);
    s[0] = c.h0; s[1] = c.h1; s[2] = c.h2; s[3] = c.h3; s[4] = c.h4;
}
void hxc_sha256(uint32_t *s, uint32_t *m)
{
    SHA256_CTX c; unsigned char b[64]; int i;
    for (i = 0; i < 64; i++) b[i] = (unsigned char)(m[i >> 2] >> ((3 - (i & 3)) * 8));
    for (i = 0; i < 8; i++) c.h[i] = s[i];
    SHA256_Transform(&c, b);
    for (i = 0; i < 8; i++) s[i] = c.h[i];
}
void hxc_sha512(uint64_t *s, uint64_t *m)
{
    SHA512_CTX c; unsigned char b[128]; int i;
    for (i = 0; i < 128; i++) b[i] = (unsigned char)(m[i >> 3] >> ((7 - (i & 7)) * 8));
    for (i = 0; i < 8; i++) c.h[i] = s[i];
    SHA512_Transform(&c, b);
    for (i = 0; i < 8; i++) s[i] = c.h[i];
}

/* ---- the reference ---- */

/* The standalone hx has no sha224 or sha384 (hashpipe registers them), so the
 * check registers its own, with hx's four output forms. */
static void ref_emit(const unsigned char *d, int n, hx_val *r, hx_arena *a, uint8_t role)
{
    static const char b64[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    int i, o = 0;
    char *x = hx_arena_alloc(a, n * 2 + 8);
    if (role == ROLE_BIN) { memcpy(x, d, n); o = n; }
    else if (role == ROLE_B64) {
        for (i = 0; i < n; i += 3) {
            unsigned v = d[i] << 16 | (i + 1 < n ? d[i + 1] << 8 : 0) | (i + 2 < n ? d[i + 2] : 0);
            x[o++] = b64[v >> 18 & 63]; x[o++] = b64[v >> 12 & 63];
            x[o++] = i + 1 < n ? b64[v >> 6 & 63] : '=';
            x[o++] = i + 2 < n ? b64[v & 63] : '=';
        }
    } else {
        const char *h = role == ROLE_UC ? "0123456789ABCDEF" : "0123456789abcdef";
        for (i = 0; i < n; i++) { x[o++] = h[d[i] >> 4]; x[o++] = h[d[i] & 15]; }
    }
    x[o] = 0;
    r->data = x; r->len = o; r->is_int = 0;
}
static void ref_sha224(hx_func_entry *s, hx_val *a, int n, hx_val *r, hx_arena *ar, uint8_t role)
{ unsigned char d[28]; (void)s; (void)n; SHA224((unsigned char *)a[0].data, a[0].len, d); ref_emit(d, 28, r, ar, role); }
static void ref_sha384(hx_func_entry *s, hx_val *a, int n, hx_val *r, hx_arena *ar, uint8_t role)
{ unsigned char d[48]; (void)s; (void)n; SHA384((unsigned char *)a[0].data, a[0].len, d); ref_emit(d, 48, r, ar, role); }
#define REF_ROLES (ROLE_CAP_BIN | ROLE_CAP_HEX | ROLE_CAP_B64 | ROLE_CAP_UC)
/* hmac_X(message, key), as hx and mdxfind read it: the SECOND operand is
 * the key. The standalone hx has only sha1 and sha256. */
#define REF_HMAC(NAME, MD, N) \
static void ref_##NAME(hx_func_entry *s, hx_val *a, int n, hx_val *r, hx_arena *ar, uint8_t role) \
{ unsigned char d[64]; unsigned int dl = N; (void)s; (void)n; \
  HMAC(MD(), a[1].data, a[1].len, (unsigned char *)a[0].data, a[0].len, d, &dl); \
  ref_emit(d, N, r, ar, role); }
/* OpenSSL 3 keeps MD4 in the legacy provider, where HMAC(EVP_md4()) returns
 * NULL; build HMAC-MD4 on the low-level MD4() instead. */
static void ref_hmac_md4(hx_func_entry *s, hx_val *a, int n, hx_val *r, hx_arena *ar, uint8_t role)
{
    unsigned char k[64], buf[64 + 4096 + 16], in[16], d[16];
    int kl = a[1].len, i, ml = a[0].len < 4096 ? a[0].len : 4096;
    (void)s; (void)n;
    memset(k, 0, sizeof k);
    if (kl > 64) { MD4((unsigned char *)a[1].data, kl, k); } else memcpy(k, a[1].data, kl);
    for (i = 0; i < 64; i++) buf[i] = k[i] ^ 0x36;
    memcpy(buf + 64, a[0].data, ml);
    MD4(buf, 64 + ml, in);
    for (i = 0; i < 64; i++) buf[i] = k[i] ^ 0x5c;
    memcpy(buf + 64, in, 16);
    MD4(buf, 80, d);
    ref_emit(d, 16, r, ar, role);
}
REF_HMAC(hmac_md5, EVP_md5, 16)
REF_HMAC(hmac_sha224, EVP_sha224, 28)
REF_HMAC(hmac_sha384, EVP_sha384, 48)
REF_HMAC(hmac_sha512, EVP_sha512, 64)

static int hexv(int c)
{
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}

static int b64v(int c)
{
    if (c >= 'A' && c <= 'Z') return c - 'A';
    if (c >= 'a' && c <= 'z') return c - 'a' + 26;
    if (c >= '0' && c <= '9') return c - '0' + 52;
    if (c == '+') return 62;
    if (c == '/') return 63;
    return -1;
}

/* The stored form's binary from the VM's final value. */
static int stored_binary(const struct hx_plan *p, const char *v, int vl, unsigned char *out)
{
    int view = p->node[p->final_body].view, i;
    if (view == HXV_BIN) { memcpy(out, v, vl); return vl; }
    if (view == HXV_B64) {
        int o = 0, acc = 0, bits = 0;
        for (i = 0; i < vl && v[i] != '='; i++) {
            int x = b64v((unsigned char)v[i]);
            if (x < 0) return -1;
            acc = (acc << 6) | x; bits += 6;
            if (bits >= 8) { bits -= 8; out[o++] = (unsigned char)(acc >> bits); }
        }
        return o;
    }
    for (i = 0; i + 1 < vl; i += 2) {
        int a = hexv((unsigned char)v[i]), b = hexv((unsigned char)v[i + 1]);
        if (a < 0 || b < 0) return -1;
        out[i / 2] = (unsigned char)(a << 4 | b);
    }
    return vl / 2;
}

static const char *bodyname[HXB_COUNT] = { "md4_bin(pass)", "md5_bin(pass)", "sha1_bin(pass)",
    "sha224_bin(pass)", "sha256_bin(pass)", "sha384_bin(pass)", "sha512_bin(pass)" };

/* chan: 0 none, 1 salt, 2 user (enum hx_plan_channel), 3 salt and pepper;
 * k is that value, pp the pepper */
static hx_val vm(hx_program *prog, const char *pw, int pl, int chan, const char *k, int kl,
                 const char *pp, int ppl)
{
    hx_vm m;
    hx_vm_init(&m, prog);
    return hx_vm_run(&m, pw, pl, (chan == 1 || chan == 3) ? k : "", (chan == 1 || chan == 3) ? kl : 0,
                     "", 0, chan == 3 ? pp : "", chan == 3 ? ppl : 0,
                     chan == 2 ? k : "", chan == 2 ? kl : 0);
    /* the arena is kept: callers copy before the next run */
}

int main(int argc, char **argv)
{
    static const char *pw[] = { "", "a", "password", "PaSsWoRd", "p\xc3\xa4ssw\xc3\xb6rd",
        "\xf0\x9f\x98\x80 emoji", "\xff\xfe bad", "\xc0\x80 overlong", "\xed\xa0\x80 surrogate",
        "\xe2\x82 truncated", "\xe0\x80\xaf overlong3", "\xf0\x80\x80\xaf overlong4",
        "\xf4\x90\x80\x80 above10ffff", "\xe0\xa4\x85 ok3", "0123456789", "deadBEEF", NULL };
    static char longs[10][300];
    /* salts and users: empty, short, a block's worth, past a block, UTF-8,
     * ill-formed UTF-8 (the walker must decode a salt exactly as the VM) */
    static const char *keys[] = { "", "s", "salt1234", "Administrator",
        "0123456789abcdef0123456789abcdef01234567",
        "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789",
        "s\xc3\xa4lz", "\xff\xfe k", NULL };
    /* peppers: empty, the built-in defaults, UTF-8, past a block */
    static char longpep[301];
    static const char *peps[] = { "", ":", "@gmail.c", "p\xc3\xa4p",
        "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789",
        longpep, NULL };   /* longpep: over 255, so the length's high byte is used */
    int nk, kk, nkey = 0, npep = 0;
    unsigned char packed[1024];
    static const int ll[] = { 54, 55, 56, 63, 64, 111, 112, 119, 128, 200 };
    const char *all[64];
    int np = 0, i, j, verbose = argc > 1 && strcmp(argv[1], "-v") == 0;
    int types_ok = 0, types_bad = 0, vec = 0, bad = 0, iter_vec = 0;
    static struct hx_plan plan;
    hx_program *rprog[HXB_COUNT];

    if (!hx_func_lookup("sha224")) hx_func_register("sha224", ref_sha224, 28, NULL, 0, REF_ROLES, ROLE_HEX);
    if (!hx_func_lookup("sha384")) hx_func_register("sha384", ref_sha384, 48, NULL, 0, REF_ROLES, ROLE_HEX);
    if (!hx_func_lookup("hmac_md4")) hx_func_register("hmac_md4", ref_hmac_md4, 16, NULL, 0, REF_ROLES, ROLE_HEX);
    if (!hx_func_lookup("hmac_md5")) hx_func_register("hmac_md5", ref_hmac_md5, 16, NULL, 0, REF_ROLES, ROLE_HEX);
    if (!hx_func_lookup("hmac_sha224")) hx_func_register("hmac_sha224", ref_hmac_sha224, 28, NULL, 0, REF_ROLES, ROLE_HEX);
    if (!hx_func_lookup("hmac_sha384")) hx_func_register("hmac_sha384", ref_hmac_sha384, 48, NULL, 0, REF_ROLES, ROLE_HEX);
    if (!hx_func_lookup("hmac_sha512")) hx_func_register("hmac_sha512", ref_hmac_sha512, 64, NULL, 0, REF_ROLES, ROLE_HEX);
    for (i = 0; i < 300; i++) longpep[i] = (char)('A' + i % 26);
    for (i = 0; pw[i]; i++) all[np++] = pw[i];
    for (i = 0; i < 10; i++) {
        for (j = 0; j < ll[i]; j++) longs[i][j] = (char)('a' + (j * 7 + i) % 26);
        longs[i][ll[i]] = 0;
        all[np++] = longs[i];
    }
    for (i = 0; i < HXB_COUNT; i++) rprog[i] = bodyname[i] ? hx_compile_expr(bodyname[i], NULL) : NULL;

    for (i = 0; i < hx_gen_count; i++) {
        const struct hx_gen_fn *t = &hx_gen_table[i];
        int whole = t->has_iter == 2;
        const struct hx_spec_entry *e = whole ? NULL : hx_specs_lookup(t->job);
        const char *expr = whole ? hx_gen_user_expr[t->job - 1100] : e ? e->expression : NULL;
        const char *name = whole ? "user" : e ? e->name : "?";
        hx_program *prog = expr ? hx_compile_expr(expr, NULL) : NULL;
        int tbad = 0, k;
        if (!prog) { printf("e%d: does not compile in standalone hx\n", t->job); types_bad++; continue; }
        if (whole) hx_plan_build(prog, NULL, 256, &plan);
        else hx_plan_build(e->program, e->call_names, 256, &plan);
        while (keys[nkey]) nkey++;
        while (peps[npep]) npep++;
        nk = !t->chan ? 1 : t->chan == 3 ? nkey * npep : nkey;
        for (kk = 0; kk < nk; kk++)
        for (k = 0; k < np; k++) {
            const char *key = t->chan ? keys[kk % nkey] : "";
            const char *pep = t->chan == 3 ? peps[kk / nkey] : "";
            int kl = (int)strlen(key), pl = (int)strlen(pep);
            unsigned char want[128], got[128], want2[128];
            char text[512];
            int wl, tl, w2;
            template_state st;
            hx_val r = vm(prog, all[k], (int)strlen(all[k]), t->chan, key, kl, pep, pl);
            tl = r.len < (int)sizeof text ? r.len : (int)sizeof text;
            memcpy(text, r.data, tl);
            wl = stored_binary(&plan, text, tl, want);
            memset(&st, 0, sizeof st);
            if (t->chan == 3) {
                /* as the host packs it: salt . pepper . pepper length */
                memcpy(packed, key, kl);
                memcpy(packed + kl, pep, pl);
                packed[kl + pl] = (unsigned char)pl;
                packed[kl + pl + 1] = (unsigned char)(pl >> 8);
                t->fin_s(&st, (const uint8_t *)all[k], (int)strlen(all[k]),
                         packed, (uint32_t)(kl + pl + 2), 0);
            } else if (t->chan)
                t->fin_s(&st, (const uint8_t *)all[k], (int)strlen(all[k]),
                         (const uint8_t *)key, (uint32_t)kl, 0);
            else
                t->fin(&st, (const uint8_t *)all[k], (int)strlen(all[k]));
            for (j = 0; j < 64; j++) got[j] = (unsigned char)(st.h[j >> 2] >> ((j & 3) * 8));
            vec++;
            if (wl < 0 || memcmp(want, got, wl) != 0) {
                bad++; tbad++;
                if (verbose) {
                    printf("e%d %s: finalize differs for password %d (%d bytes) key %d: want ", t->job, name, k, (int)strlen(all[k]), kk);
                    for (j = 0; j < wl; j++) printf("%02x", want[j]);
                    printf(" got ");
                    for (j = 0; j < (wl > 0 ? wl : 16); j++) printf("%02x", got[j]);
                    printf("\n");
                }
                continue;
            }
            if (!t->has_iter) continue;
            {
                /* whole: the next round is the program over this round's text */
                hx_val r2 = vm(whole ? prog : rprog[plan.node[plan.final_body].op],
                               text, tl, 0, "", 0, "", 0);
                w2 = r2.len;
                memcpy(want2, r2.data, w2);
            }
            if (whole) {    /* the VM gives hex text: the state holds its binary */
                tl = w2 < (int)sizeof text ? w2 : (int)sizeof text;
                memcpy(text, want2, tl);
                w2 = stored_binary(&plan, text, tl, want2);
            }
            t->iter(&st);
            for (j = 0; j < 64; j++) got[j] = (unsigned char)(st.h[j >> 2] >> ((j & 3) * 8));
            iter_vec++;
            if (w2 < 0 || memcmp(want2, got, w2) != 0) {
                bad++; tbad++;
                if (verbose) printf("e%d %s: iterate differs for password %d\n", t->job, name, k);
            }
        }
        if (tbad) types_bad++; else types_ok++;
        hx_program_free(prog);
    }
    printf("%d types agree, %d differ; %d finalize and %d iterate vectors, %d differ\n",
           types_ok, types_bad, vec, iter_vec, bad);
    return types_bad != 0;
}
