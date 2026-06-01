/*
 * test_wrl_nessie.c -- R12 pre-flight donor verification for Phase 5b Tier 2
 * Whirlpool ship (sub-phase 5b.2a.0).
 *
 * Cross-checks two CPU Whirlpool donors against published NESSIE test vectors:
 *   - librhash rhash_whirlpool_init / _update / _final  (D16.2.b candidate)
 *   - OpenSSL  WHIRLPOOL()                              (D16.2.a candidate
 *                                                        + mdxfind's CPU oracle)
 *
 * Per Phase 5b Tier 2 architect spec R12: any donor that fails ANY published
 * NESSIE vector is escalated and the port is paused until resolved. Both
 * donors must agree byte-for-byte with the canonical NESSIE vectors below.
 *
 * Build (iMac):
 *   cc -O2 -I RHash-master/librhash \
 *      -o /tmp/test_wrl_nessie codegen/tests/test_wrl_nessie.c \
 *      RHash-master/librhash/whirlpool.c \
 *      RHash-master/librhash/whirlpool_sbox.c \
 *      RHash-master/librhash/byte_order.c \
 *      libcrypto.a
 *
 * Exit status 0 -> all vectors PASS on both donors.
 * Non-zero    -> at least one vector MISMATCH; STOP and escalate.
 *
 * $Revision: 1.1 $
 * $Log: test_wrl_nessie.c,v $
 * Revision 1.1  2026/05/27 22:15:01  dlr
 * Initial check-in: R12 pre-flight donor verification, sub-phase 5b.2a.0.
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include "whirlpool.h"
#include <openssl/whrlpool.h>

/* NESSIE-canonical Whirlpool test vectors (ISO/IEC 10118-3:2004).
 * Hex digests are 128 chars (64 bytes) each.
 */
struct vec {
    const char *label;
    const unsigned char *msg;     /* may be NULL for stress vector */
    size_t msg_len;
    const char *expected_hex;
};

static const unsigned char v_empty[]   = "";
static const unsigned char v_a[]       = "a";
static const unsigned char v_abc[]     = "abc";
static const unsigned char v_msgdig[]  = "message digest";
static const unsigned char v_az[]      = "abcdefghijklmnopqrstuvwxyz";
static const unsigned char v_alnum[]   =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
static const unsigned char v_8x1234[]  =
    "12345678901234567890123456789012345678901234567890"
    "1234567890123456789012345678901234567890";

static const struct vec vectors[] = {
    { "empty (0 bytes)",        v_empty,  0,
      "19fa61d75522a4669b44e39c1d2e1726c530232130d407f89afee0964997f7a7"
      "3e83be698b288febcf88e3e03c4f0757ea8964e59b63d93708b138cc42a66eb3" },
    { "\"a\"",                  v_a,      1,
      "8aca2602792aec6f11a67206531fb7d7f0dff59413145e6973c45001d0087b42"
      "d11bc645413aeff63a42391a39145a591a92200d560195e53b478584fdae231a" },
    { "\"abc\"",                v_abc,    3,
      "4e2448a4c6f486bb16b6562c73b4020bf3043e3a731bce721ae1b303d97e6d4c"
      "7181eebdb6c57e277d0e34957114cbd6c797fc9d95d8b582d225292076d4eef5" },
    { "\"message digest\"",     v_msgdig, 14,
      "378c84a4126e2dc6e56dcc7458377aac838d00032230f53ce1f5700c0ffb4d3b"
      "8421557659ef55c106b4b52ac5a4aaa692ed920052838f3362e86dbd37a8903e" },
    { "a..z (26 bytes)",        v_az,     26,
      "f1d754662636ffe92c82ebb9212a484a8d38631ead4238f5442ee13b8054e41b"
      "08bf2a9251c30b6a0b8aae86177ab4a6f68f673e7207865d5d9819a3dba4eb3b" },
    { "A..Z+a..z+0..9 (62)",    v_alnum,  62,
      "dc37e008cf9ee69bf11f00ed9aba26901dd7c28cdec066cc6af42e40f82f3a1e"
      "08eba26629129d8fb7cb57211b9281a65517cc879d7b962142c65f5a7af01467" },
    { "8x'1234567890' (80)",    v_8x1234, 80,
      "466ef18babb0154d25b9d38a6414f5c08784372bccb204d6549c4afadb601429"
      "4d5bd8df2a6c44e538cd047b2681a51a2c60481e88c5a20b2c2a80cf3a9a083b" },
    /* 1-million-'a' stress vector handled separately (allocated). */
    { NULL, NULL, 0, NULL }
};

static const char *million_a_expected =
    "0c99005beb57eff50a7cf005560ddf5d29057fd86b20bfd62deca0f1ccea4af5"
    "1fc15490eddc47af32bb2b66c34ff9ad8c6008ad677f77126953b226e4ed8b01";

static void hexify(const unsigned char *bin, size_t n, char *out) {
    static const char H[] = "0123456789abcdef";
    for (size_t i = 0; i < n; i++) {
        out[2*i]   = H[(bin[i] >> 4) & 0xF];
        out[2*i+1] = H[bin[i] & 0xF];
    }
    out[2*n] = '\0';
}

static int compare_hex(const char *label, const char *donor,
                       const char *got, const char *expected) {
    if (strcmp(got, expected) == 0) {
        printf("  %-10s %-32s PASS\n", donor, label);
        return 0;
    }
    printf("  %-10s %-32s FAIL\n", donor, label);
    printf("    got:      %s\n", got);
    printf("    expected: %s\n", expected);
    return 1;
}

static int run_librhash(const unsigned char *msg, size_t n,
                        const char *label, const char *expected) {
    whirlpool_ctx ctx;
    unsigned char digest[64];
    char hex[129];
    rhash_whirlpool_init(&ctx);
    rhash_whirlpool_update(&ctx, msg, n);
    rhash_whirlpool_final(&ctx, digest);
    hexify(digest, 64, hex);
    return compare_hex(label, "librhash", hex, expected);
}

static int run_openssl(const unsigned char *msg, size_t n,
                       const char *label, const char *expected) {
    unsigned char digest[64];
    char hex[129];
    WHIRLPOOL(msg, n, digest);
    hexify(digest, 64, hex);
    return compare_hex(label, "openssl", hex, expected);
}

int main(void) {
    int fails = 0;
    int total = 0;

    printf("Whirlpool NESSIE donor pre-flight (R12) — sub-phase 5b.2a.0\n");
    printf("================================================================\n");

    for (size_t i = 0; vectors[i].label != NULL; i++) {
        printf("%s\n", vectors[i].label);
        fails += run_librhash(vectors[i].msg, vectors[i].msg_len,
                              vectors[i].label, vectors[i].expected_hex);
        fails += run_openssl(vectors[i].msg, vectors[i].msg_len,
                             vectors[i].label, vectors[i].expected_hex);
        total += 2;
    }

    /* Stress vector: 1,000,000 'a' bytes. */
    {
        size_t n = 1000000;
        unsigned char *buf = (unsigned char *)malloc(n);
        if (!buf) {
            fprintf(stderr, "alloc failed for 1M-a stress vector\n");
            return 2;
        }
        memset(buf, 'a', n);
        printf("1,000,000 x 'a' (stress)\n");
        fails += run_librhash(buf, n, "1M-a", million_a_expected);
        fails += run_openssl(buf, n, "1M-a", million_a_expected);
        total += 2;
        free(buf);
    }

    printf("================================================================\n");
    printf("Total cells: %d   Failures: %d\n", total, fails);
    if (fails == 0) {
        printf("R12 PRE-FLIGHT PASS — both donors standard-conformant.\n");
        return 0;
    }
    printf("R12 PRE-FLIGHT FAIL — STOP and escalate.\n");
    return 1;
}
